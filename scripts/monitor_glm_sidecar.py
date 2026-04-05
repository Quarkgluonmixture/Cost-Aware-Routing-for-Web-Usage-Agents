#!/usr/bin/env python3
"""
Realtime sidecar monitor for VWA phase runs.

What it does:
1) Watches newly generated *_summary_v2.json episode files.
2) Computes rolling health metrics (success_rate, wrong_url, no_progress, max_steps, avg step latency).
3) Triggers alerts when thresholds are exceeded.
4) Optionally calls GLM (configured from a local config file) to generate a short diagnosis.
5) Optionally pushes alert text to ntfy.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

SUMMARY_RE = re.compile(r"^(?P<site>.+)_task_(?P<task_id>\d+)_summary_v2\.json$")


@dataclass
class EpisodeRecord:
    key: str
    condition_id: str
    site: str
    task_id: int
    success: bool
    steps: int
    step_latency_s: Optional[float]
    reason: str
    final_url: str
    reference_url: str
    page_unchanged_rate: float
    no_op_rate: float


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _load_glm_config(cfg_path: Path) -> Dict[str, str]:
    lines = []
    for raw in cfg_path.read_text(encoding="utf-8").splitlines():
        t = raw.strip()
        if not t or t.startswith("#"):
            continue
        lines.append(t)
    if len(lines) < 3:
        raise ValueError(f"GLM config invalid: need 3 lines (endpoint/model/api_key), got {len(lines)}")
    return {"endpoint": lines[0], "model": lines[1], "api_key": lines[2]}


def _candidate_glm_urls(endpoint: str) -> List[str]:
    ep = endpoint.rstrip("/")
    if ep.endswith("/chat/completions"):
        return [ep]
    return [f"{ep}/chat/completions", ep]


def _call_glm_chat(glmm: Dict[str, str], messages: Sequence[Dict[str, str]], timeout_s: int = 30) -> str:
    payload = {
        "model": glmm["model"],
        "messages": list(messages),
        "temperature": 0.1,
        "max_tokens": 220,
    }
    body = json.dumps(payload).encode("utf-8")
    last_err = None
    for url in _candidate_glm_urls(glmm["endpoint"]):
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {glmm['api_key']}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            choices = data.get("choices") or []
            if choices:
                msg = (choices[0].get("message") or {}).get("content")
                if isinstance(msg, str) and msg.strip():
                    return msg.strip()
            # fallback for non-openai style responses
            text = data.get("output_text") or data.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
            return json.dumps(data, ensure_ascii=False)[:500]
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"GLM request failed: {last_err}")


def _post_ntfy(topic: str, title: str, body: str, priority: str = "default", timeout_s: int = 15) -> None:
    url = f"https://ntfy.sh/{topic}"
    req = urllib.request.Request(
        url,
        data=body.encode("utf-8"),
        method="POST",
        headers={"Title": title, "Priority": priority},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s):
            return
    except urllib.error.URLError:
        return


def _normalize_ref_urls(ref_url: Any) -> List[str]:
    if not isinstance(ref_url, str):
        return []
    t = ref_url.strip()
    if not t:
        return []
    if "|OR|" in t:
        return [x.strip() for x in t.split("|OR|") if x.strip()]
    return [t]


def _extract_last_step(steps_path: Path) -> Optional[Dict[str, Any]]:
    rows = _read_jsonl(steps_path)
    if not rows:
        return None
    rows.sort(key=lambda x: int(x.get("step_idx", 0)))
    return rows[-1]


def _task_cfg(task_cfg_cache: Dict[Tuple[str, int], Dict[str, Any]], run_dir: Path, site: str, task_id: int) -> Dict[str, Any]:
    key = (site, task_id)
    if key in task_cfg_cache:
        return task_cfg_cache[key]
    p = run_dir / "task_configs" / f"{site}_task_{task_id}.json"
    data = _read_json(p) if p.exists() else {}
    task_cfg_cache[key] = data
    return data


def _classify_episode(
    summary: Dict[str, Any],
    last_step: Optional[Dict[str, Any]],
    task_meta: Dict[str, Any],
    max_steps: int,
    no_progress_unchanged_threshold: float,
    no_progress_noop_threshold: float,
) -> Tuple[str, str, str]:
    """
    Returns: (reason, final_url, reference_url)
    """
    success = bool(summary.get("success", False))
    if success:
        return "success", "", ""

    if summary.get("error"):
        return "runtime_error", "", ""

    final_url = ""
    if last_step:
        final_url = str(((last_step.get("state_digest") or {}).get("url_after")) or "")
    final_action_type = str(((last_step or {}).get("action") or {}).get("action_type", "")).lower()

    eval_cfg = task_meta.get("eval") or {}
    eval_types = [str(x) for x in (eval_cfg.get("eval_types") or [])]
    ref_url = str(eval_cfg.get("reference_url") or "")
    ref_urls = _normalize_ref_urls(ref_url)

    # url_match failures (core real-world signal for classifieds tasks)
    if "url_match" in eval_types and ref_urls:
        if final_url not in ref_urls:
            return "wrong_url", final_url, ref_url

    steps = int(summary.get("steps", 0) or 0)
    if steps >= max_steps:
        return "max_steps", final_url, ref_url

    page_unchanged_rate = float(summary.get("page_unchanged_rate", 0.0) or 0.0)
    no_op_rate = float(summary.get("no_op_rate", 0.0) or 0.0)
    retry_hint = int(((summary.get("state_change_reason_distribution") or {}).get("baseline_no_progress_retry_applied")) or 0)
    if (
        page_unchanged_rate >= no_progress_unchanged_threshold
        or no_op_rate >= no_progress_noop_threshold
        or retry_hint > 0
    ):
        return "no_progress", final_url, ref_url

    if final_action_type == "finish":
        return "finish_fail", final_url, ref_url
    return "other_fail", final_url, ref_url


def _scan_summaries(run_dir: Path, condition_filter: Optional[str]) -> List[Path]:
    if condition_filter:
        roots = [run_dir / condition_filter]
    else:
        roots = [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("phase")]
    files: List[Path] = []
    for root in roots:
        ep_dir = root / "episodes"
        if not ep_dir.exists():
            continue
        files.extend(ep_dir.glob("*_summary_v2.json"))
    return sorted(files)


def _episode_key(path: Path) -> str:
    return str(path.resolve())


def _compute_metrics(window_records: Sequence[EpisodeRecord]) -> Dict[str, float]:
    n = len(window_records)
    if n == 0:
        return {
            "count": 0.0,
            "success_rate": 0.0,
            "wrong_url_rate": 0.0,
            "no_progress_rate": 0.0,
            "max_steps_rate": 0.0,
            "avg_step_latency_s": 0.0,
        }
    c = Counter(r.reason for r in window_records)
    success = sum(1 for r in window_records if r.success)
    lat_vals = [r.step_latency_s for r in window_records if r.step_latency_s is not None]
    avg_step_latency_s = sum(lat_vals) / len(lat_vals) if lat_vals else 0.0
    return {
        "count": float(n),
        "success_rate": success / n,
        "wrong_url_rate": c["wrong_url"] / n,
        "no_progress_rate": c["no_progress"] / n,
        "max_steps_rate": c["max_steps"] / n,
        "avg_step_latency_s": avg_step_latency_s,
    }


def _triggered_rules(m: Dict[str, float], args: argparse.Namespace) -> List[str]:
    n = int(m["count"])
    if n < args.min_alert_samples:
        return []
    rules: List[str] = []
    if m["wrong_url_rate"] >= args.wrong_url_threshold:
        rules.append("wrong_url_high")
    if m["no_progress_rate"] >= args.no_progress_threshold:
        rules.append("no_progress_high")
    if m["max_steps_rate"] >= args.max_steps_threshold:
        rules.append("max_steps_high")
    if m["avg_step_latency_s"] >= args.avg_step_latency_threshold:
        rules.append("latency_high")
    if m["success_rate"] <= args.success_rate_floor:
        rules.append("success_low")
    return rules


def _ai_diagnosis(
    glmm: Optional[Dict[str, str]],
    run_id: str,
    condition_id: str,
    metrics: Dict[str, float],
    rules: Sequence[str],
    recent_failures: Sequence[EpisodeRecord],
) -> str:
    fallback = (
        f"触发规则: {', '.join(rules)}\n"
        f"观察: success={metrics['success_rate']:.1%}, wrong_url={metrics['wrong_url_rate']:.1%}, "
        f"no_progress={metrics['no_progress_rate']:.1%}, max_steps={metrics['max_steps_rate']:.1%}, "
        f"avg_step={metrics['avg_step_latency_s']:.1f}s\n"
        "建议: 先排查输入动作清空覆盖、元素定位稳定性、以及 url_match 任务中 finish 前是否进入目标 item URL。"
    )
    if not glmm:
        return fallback

    sample = []
    for r in recent_failures[:5]:
        sample.append(
            {
                "task_id": r.task_id,
                "reason": r.reason,
                "final_url": r.final_url,
                "reference_url": r.reference_url,
                "page_unchanged_rate": r.page_unchanged_rate,
            }
        )

    user_payload = {
        "run_id": run_id,
        "condition_id": condition_id,
        "window_metrics": metrics,
        "triggered_rules": list(rules),
        "recent_failures": sample,
    }
    messages = [
        {
            "role": "system",
            "content": (
                "你是实验监控诊断助手。请用中文输出：\n"
                "1) 三条简短诊断（每条一行）\n"
                "2) 一条可执行修复建议（单行）\n"
                "总计不超过120字。"
            ),
        },
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
    try:
        return _call_glm_chat(glmm, messages)
    except Exception:
        return fallback


def _load_state(path: Optional[Path]) -> set[str]:
    if not path or not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        seen = data.get("seen_keys") or []
        if isinstance(seen, list):
            return {str(x) for x in seen}
    except Exception:
        pass
    return set()


def _save_state(path: Optional[Path], seen_keys: set[str]) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seen_keys": sorted(seen_keys), "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Realtime GLM sidecar monitor for VWA runs")
    p.add_argument("--run-dir", required=True, help="Run directory, e.g. results/.../<run_id>")
    p.add_argument("--condition", default=None, help="Optional condition id, e.g. phase1_dom_router_0")
    p.add_argument("--poll-secs", type=int, default=30, help="Polling interval in seconds")
    p.add_argument("--window-size", type=int, default=20, help="Rolling window size (episodes)")
    p.add_argument("--min-alert-samples", type=int, default=12, help="Minimum episodes in window to alert")
    p.add_argument("--max-steps", type=int, default=30, help="Configured episode max steps")
    p.add_argument("--no-progress-unchanged-threshold", type=float, default=0.66)
    p.add_argument("--no-progress-noop-threshold", type=float, default=0.5)
    p.add_argument("--wrong-url-threshold", type=float, default=0.30)
    p.add_argument("--no-progress-threshold", type=float, default=0.25)
    p.add_argument("--max-steps-threshold", type=float, default=0.25)
    p.add_argument("--avg-step-latency-threshold", type=float, default=22.0, help="seconds")
    p.add_argument("--success-rate-floor", type=float, default=0.10)
    p.add_argument("--alert-cooldown-secs", type=int, default=600)
    p.add_argument("--glm-config", default="glm", help="Path to glm config file (endpoint/model/key)")
    p.add_argument("--disable-glm", action="store_true", help="Disable GLM calls (local heuristics only)")
    p.add_argument("--ntfy-topic", default=None, help="Optional ntfy topic for alert push")
    p.add_argument("--state-file", default=None, help="Optional state file to persist seen episodes")
    p.add_argument(
        "--alert-on-bootstrap",
        action="store_true",
        help="Also alert for episodes that already existed at monitor startup",
    )
    p.add_argument("--once", action="store_true", help="Scan once then exit")
    return p


def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"run_dir not found: {run_dir}")

    glmm: Optional[Dict[str, str]] = None
    if not args.disable_glm:
        cfg_path = Path(args.glm_config)
        if cfg_path.exists():
            try:
                glmm = _load_glm_config(cfg_path)
                print(f"[monitor] GLM enabled: model={glmm['model']} endpoint={glmm['endpoint']}")
            except Exception as e:  # noqa: BLE001
                print(f"[monitor] GLM config invalid ({cfg_path}): {e}. Fallback to heuristic diagnosis.")
        else:
            print(f"[monitor] GLM config not found ({cfg_path}). Fallback to heuristic diagnosis.")
    else:
        print("[monitor] GLM disabled.")

    state_file = Path(args.state_file).resolve() if args.state_file else None
    seen_keys = _load_state(state_file)
    window: Deque[EpisodeRecord] = deque(maxlen=max(1, int(args.window_size)))
    task_cfg_cache: Dict[Tuple[str, int], Dict[str, Any]] = {}

    last_alert_sig: Optional[str] = None
    last_alert_ts: float = 0.0
    run_id = run_dir.name
    bootstrap_keys = {_episode_key(p) for p in _scan_summaries(run_dir, args.condition)}

    print(
        f"[monitor] watching run_id={run_id} condition={args.condition or '*'} "
        f"poll={args.poll_secs}s window={args.window_size}"
    )

    while True:
        summaries = _scan_summaries(run_dir, args.condition)
        new_paths = [p for p in summaries if _episode_key(p) not in seen_keys]
        if new_paths:
            # process in deterministic order: task id first, then file path
            def _sort_key(path: Path) -> Tuple[str, int, str]:
                m = SUMMARY_RE.match(path.name)
                tid = int(m.group("task_id")) if m else 10**9
                return (path.parent.parent.name, tid, str(path))

            for summary_path in sorted(new_paths, key=_sort_key):
                key = _episode_key(summary_path)
                m = SUMMARY_RE.match(summary_path.name)
                if not m:
                    seen_keys.add(key)
                    continue
                site = m.group("site")
                task_id = int(m.group("task_id"))
                condition_id = summary_path.parent.parent.name
                try:
                    summary = _read_json(summary_path)
                except Exception:  # noqa: BLE001
                    # Summary file may still be writing; retry next poll.
                    continue
                steps_path = summary_path.with_name(summary_path.name.replace("_summary_v2.json", "_steps_v2.jsonl"))
                last_step = _extract_last_step(steps_path) if steps_path.exists() else None

                meta = _task_cfg(task_cfg_cache, run_dir, site, task_id)
                reason, final_url, ref_url = _classify_episode(
                    summary=summary,
                    last_step=last_step,
                    task_meta=meta,
                    max_steps=int(args.max_steps),
                    no_progress_unchanged_threshold=float(args.no_progress_unchanged_threshold),
                    no_progress_noop_threshold=float(args.no_progress_noop_threshold),
                )

                steps = int(summary.get("steps", 0) or 0)
                total_latency_ms = float(summary.get("total_latency_ms", 0.0) or 0.0)
                step_latency_s = (total_latency_ms / steps / 1000.0) if steps > 0 else None

                rec = EpisodeRecord(
                    key=key,
                    condition_id=condition_id,
                    site=site,
                    task_id=task_id,
                    success=bool(summary.get("success", False)),
                    steps=steps,
                    step_latency_s=step_latency_s,
                    reason=reason,
                    final_url=final_url,
                    reference_url=ref_url,
                    page_unchanged_rate=float(summary.get("page_unchanged_rate", 0.0) or 0.0),
                    no_op_rate=float(summary.get("no_op_rate", 0.0) or 0.0),
                )
                window.append(rec)
                seen_keys.add(key)

                metrics = _compute_metrics(list(window))
                print(
                    "[monitor] "
                    f"{condition_id} task={task_id:>3d} reason={reason:<12s} "
                    f"window={int(metrics['count'])}/{args.window_size} "
                    f"succ={metrics['success_rate']:.1%} wrong_url={metrics['wrong_url_rate']:.1%} "
                    f"no_prog={metrics['no_progress_rate']:.1%} max_steps={metrics['max_steps_rate']:.1%} "
                    f"avg_step={metrics['avg_step_latency_s']:.1f}s"
                )

                rules = _triggered_rules(metrics, args)
                is_bootstrap_episode = key in bootstrap_keys
                allow_alert = args.alert_on_bootstrap or (not is_bootstrap_episode)
                if rules and allow_alert:
                    sig = ",".join(sorted(rules))
                    now = time.time()
                    if sig != last_alert_sig or (now - last_alert_ts) >= int(args.alert_cooldown_secs):
                        fails = [x for x in reversed(window) if not x.success]
                        diagnosis = _ai_diagnosis(
                            glmm=glmm,
                            run_id=run_id,
                            condition_id=condition_id,
                            metrics=metrics,
                            rules=rules,
                            recent_failures=fails,
                        )
                        alert_title = f"P79 Monitor Alert [{condition_id}]"
                        alert_body = (
                            f"run_id={run_id}\n"
                            f"rules={sig}\n"
                            f"window={int(metrics['count'])}\n"
                            f"succ={metrics['success_rate']:.1%}, wrong_url={metrics['wrong_url_rate']:.1%}, "
                            f"no_progress={metrics['no_progress_rate']:.1%}, max_steps={metrics['max_steps_rate']:.1%}, "
                            f"avg_step={metrics['avg_step_latency_s']:.1f}s\n"
                            f"{diagnosis}"
                        )
                        print(f"[monitor][ALERT] {alert_body}")
                        if args.ntfy_topic:
                            _post_ntfy(args.ntfy_topic, alert_title, alert_body, priority="high")
                        last_alert_sig = sig
                        last_alert_ts = now

                _save_state(state_file, seen_keys)

        if args.once:
            break
        time.sleep(max(1, int(args.poll_secs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
