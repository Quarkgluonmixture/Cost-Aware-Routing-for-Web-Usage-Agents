#!/usr/bin/env python3
"""
Experiment health watchdog — lightweight monitoring and alerting.

What it does:
1) Watches newly generated *_summary_v2.json episode files.
2) Computes rolling health metrics (success_rate, wrong_url, no_progress, max_steps, avg step latency).
3) Triggers alerts when thresholds are exceeded.
4) Uses heuristic rules to generate a short diagnosis (no GLM dependency).
5) Optionally pushes alert text to ntfy.
6) Tracks idle time — alerts if no new episode appears within --watchdog-idle-alert-mins.
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
    observation_mode: str
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


def _get_observation_mode(condition_dir: Path, cache: Dict[str, str]) -> str:
    condition_id = condition_dir.name
    if condition_id in cache:
        return cache[condition_id]
    meta_path = condition_dir / "condition_meta.json"
    mode = "dom"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            mode = str(meta.get("observation_mode", "dom"))
        except Exception:
            pass
    else:
        cid = condition_id.lower()
        if "som" in cid:
            mode = "som"
        elif "vision" in cid:
            mode = "vision"
    cache[condition_id] = mode
    return mode


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


def _mode_summary_line(windows: Dict[str, "Deque[EpisodeRecord]"]) -> str:
    """Return a one-line per-mode aggregate for quick comparison."""
    from collections import defaultdict
    mode_recs: Dict[str, List[EpisodeRecord]] = defaultdict(list)
    for recs in windows.values():
        for r in recs:
            mode_recs[r.observation_mode].append(r)
    if not mode_recs:
        return ""
    parts = []
    for mode in sorted(mode_recs):
        recs = mode_recs[mode]
        n = len(recs)
        succ = sum(1 for r in recs if r.success)
        parts.append(f"{mode}: {succ}/{n} ({succ/n:.1%})")
    return "  |  ".join(parts)


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


def _heuristic_diagnosis(
    metrics: Dict[str, float],
    rules: Sequence[str],
) -> str:
    """Generate a short heuristic diagnosis without GLM."""
    m = metrics
    diagnosis = (
        f"触发规则: {', '.join(rules)}\n"
        f"success={m['success_rate']:.1%}, wrong_url={m['wrong_url_rate']:.1%}, "
        f"no_progress={m['no_progress_rate']:.1%}, max_steps={m['max_steps_rate']:.1%}, "
        f"avg_step={m['avg_step_latency_s']:.1f}s"
    )
    return diagnosis


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
    p = argparse.ArgumentParser(description="Experiment health watchdog for VWA runs")
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
    p.add_argument("--avg-step-latency-threshold", type=float, default=30.0, help="seconds")
    p.add_argument("--success-rate-floor", type=float, default=0.05)
    p.add_argument("--alert-cooldown-secs", type=int, default=600)
    p.add_argument("--ntfy-topic", default=None, help="Optional ntfy topic for alert push")
    p.add_argument("--state-file", default=None, help="Optional state file to persist seen episodes")
    p.add_argument(
        "--alert-on-bootstrap",
        action="store_true",
        help="Also alert for episodes that already existed at monitor startup",
    )
    p.add_argument(
        "--watchdog-idle-alert-mins",
        type=int,
        default=20,
        help="Alert if no new episode appears within this many minutes (default: 20)",
    )
    p.add_argument("--once", action="store_true", help="Scan once then exit")
    return p


def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"run_dir not found: {run_dir}")

    state_file = Path(args.state_file).resolve() if args.state_file else None
    seen_keys = _load_state(state_file)
    windows: Dict[str, Deque[EpisodeRecord]] = {}  # condition_id -> rolling window
    all_records: List[EpisodeRecord] = []  # all-time for final summary
    task_cfg_cache: Dict[Tuple[str, int], Dict[str, Any]] = {}
    condition_mode_cache: Dict[str, str] = {}

    # Per-condition cooldown: condition_id -> last alert timestamp
    last_alert_ts_by_cond: Dict[str, float] = {}
    run_id = run_dir.name
    bootstrap_keys = {_episode_key(p) for p in _scan_summaries(run_dir, args.condition)}

    # Idle tracking
    last_new_episode_ts: float = time.time()
    idle_alerted: bool = False
    idle_alert_secs = max(60, args.watchdog_idle_alert_mins * 60)

    print(
        f"[watchdog] watching run_id={run_id} condition={args.condition or '*'} "
        f"poll={args.poll_secs}s window={args.window_size} "
        f"idle_alert={args.watchdog_idle_alert_mins}min"
    )

    while True:
        summaries = _scan_summaries(run_dir, args.condition)
        new_paths = [p for p in summaries if _episode_key(p) not in seen_keys]
        if new_paths:
            last_new_episode_ts = time.time()
            idle_alerted = False

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

                condition_dir = summary_path.parent.parent
                obs_mode = _get_observation_mode(condition_dir, condition_mode_cache)

                steps = int(summary.get("steps", 0) or 0)
                total_latency_ms = float(summary.get("total_latency_ms", 0.0) or 0.0)
                step_latency_s = (total_latency_ms / steps / 1000.0) if steps > 0 else None

                rec = EpisodeRecord(
                    key=key,
                    condition_id=condition_id,
                    observation_mode=obs_mode,
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
                if condition_id not in windows:
                    windows[condition_id] = deque(maxlen=max(1, int(args.window_size)))
                windows[condition_id].append(rec)
                all_records.append(rec)
                seen_keys.add(key)

                cond_window = list(windows[condition_id])
                metrics = _compute_metrics(cond_window)
                mode_summary = _mode_summary_line(windows)
                print(
                    "[watchdog] "
                    f"[{obs_mode}] {condition_id} task={task_id:>3d} reason={reason:<12s} "
                    f"window={int(metrics['count'])}/{args.window_size} "
                    f"succ={metrics['success_rate']:.1%} wrong_url={metrics['wrong_url_rate']:.1%} "
                    f"no_prog={metrics['no_progress_rate']:.1%} max_steps={metrics['max_steps_rate']:.1%} "
                    f"avg_step={metrics['avg_step_latency_s']:.1f}s"
                )
                if mode_summary:
                    print(f"[watchdog][MODE]  {mode_summary}")

                rules = _triggered_rules(metrics, args)
                is_bootstrap_episode = key in bootstrap_keys
                allow_alert = args.alert_on_bootstrap or (not is_bootstrap_episode)
                if rules and allow_alert:
                    sig = ",".join(sorted(rules))
                    now = time.time()
                    cond_last_ts = last_alert_ts_by_cond.get(condition_id, 0.0)
                    if (now - cond_last_ts) >= int(args.alert_cooldown_secs):
                        diagnosis = _heuristic_diagnosis(
                            metrics=metrics,
                            rules=rules,
                        )
                        alert_title = f"P79 Watchdog Alert [{obs_mode}/{condition_id}]"
                        alert_body = (
                            f"run_id={run_id}\n"
                            f"mode={obs_mode} condition={condition_id}\n"
                            f"rules={sig}\n"
                            f"window={int(metrics['count'])}\n"
                            f"succ={metrics['success_rate']:.1%}, wrong_url={metrics['wrong_url_rate']:.1%}, "
                            f"no_progress={metrics['no_progress_rate']:.1%}, max_steps={metrics['max_steps_rate']:.1%}, "
                            f"avg_step={metrics['avg_step_latency_s']:.1f}s\n"
                            f"{diagnosis}"
                        )
                        print(f"[watchdog][ALERT] {alert_body}")
                        if args.ntfy_topic:
                            _post_ntfy(args.ntfy_topic, alert_title, alert_body, priority="high")
                        last_alert_ts_by_cond[condition_id] = now

                _save_state(state_file, seen_keys)
        else:
            # No new episodes — check idle timeout
            idle_elapsed = time.time() - last_new_episode_ts
            if idle_elapsed >= idle_alert_secs and not idle_alerted:
                idle_mins = int(idle_elapsed / 60)
                idle_msg = (
                    f"run_id={run_id}\n"
                    f"condition={args.condition or '*'}\n"
                    f"已 {idle_mins} 分钟无新 episode（阈值={args.watchdog_idle_alert_mins}min）"
                )
                print(f"[watchdog][IDLE] {idle_msg}")
                if args.ntfy_topic:
                    _post_ntfy(
                        args.ntfy_topic,
                        f"P79 Watchdog idle_{idle_mins}min",
                        idle_msg,
                        priority="high",
                    )
                idle_alerted = True

        if args.once:
            break
        time.sleep(max(1, int(args.poll_secs)))

    # Final summary table — use all_records (not rolling window) for accurate totals
    if all_records:
        print("\n[watchdog][FINAL SUMMARY]")
        from collections import defaultdict
        cond_totals: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "success": 0})
        mode_totals: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "success": 0})
        for r in all_records:
            cond_totals[r.condition_id]["total"] += 1
            cond_totals[r.condition_id]["success"] += int(r.success)
            mode_totals[r.observation_mode]["total"] += 1
            mode_totals[r.observation_mode]["success"] += int(r.success)
        for cid in sorted(cond_totals):
            t = cond_totals[cid]
            n, s = t["total"], t["success"]
            mode = condition_mode_cache.get(cid, "?")
            print(f"  {mode:6s}  {cid:<40s}  {s:>3d}/{n:<3d}  ({s/n:.1%})" if n else f"  {mode:6s}  {cid}")
        print("  ---")
        for mode in sorted(mode_totals):
            t = mode_totals[mode]
            n, s = t["total"], t["success"]
            print(f"  {mode:6s}  {'TOTAL':<40s}  {s:>3d}/{n:<3d}  ({s/n:.1%})" if n else f"  {mode:6s}  TOTAL  0/0")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
