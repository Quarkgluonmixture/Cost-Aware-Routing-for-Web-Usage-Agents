#!/usr/bin/env python3
"""
Experiment watchdog — lightweight monitoring with periodic status reports.

Notifications (push to ntfy):
1) REPORT:   periodic status every --report-interval-mins (success rate + counts)
2) IDLE:     no new episode for --idle-alert-mins → may need restart
3) COMPLETE: condition finished (condition_summary_v2.json appeared)
4) ANALYSIS: post-condition analysis script completed (output files detected)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

SUMMARY_RE = re.compile(r"^(?P<site>.+)_task_(?P<task_id>\d+)_summary_v2\.json$")


@dataclass
class EpisodeRecord:
    condition_id: str
    observation_mode: str
    site: str
    task_id: int
    success: bool
    steps: int
    reason: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _post_ntfy(topic: str, title: str, body: str, priority: str = "default") -> None:
    url = f"https://ntfy.sh/{topic}"
    req = urllib.request.Request(
        url, data=body.encode("utf-8"), method="POST",
        headers={"Title": title, "Priority": priority},
    )
    try:
        with urllib.request.urlopen(req, timeout=15):
            pass
    except urllib.error.URLError:
        pass


def _normalize_ref_urls(ref_url: Any) -> List[str]:
    if not isinstance(ref_url, str):
        return []
    t = ref_url.strip()
    if not t:
        return []
    if "|OR|" in t:
        return [x.strip() for x in t.split("|OR|") if x.strip()]
    return [t]


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
    task_meta: Dict[str, Any],
    max_steps: int,
) -> str:
    if bool(summary.get("success", False)):
        return "success"
    if summary.get("error"):
        return "error"
    steps = int(summary.get("steps", 0) or 0)
    if steps >= max_steps:
        return "max_steps"
    return "fail"


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


# ---------------------------------------------------------------------------
# Status report builder
# ---------------------------------------------------------------------------

def _build_status_report(
    all_records: List[EpisodeRecord],
    condition_mode_cache: Dict[str, str],
    completed_conditions: Set[str],
    run_id: str,
) -> Optional[str]:
    """Build status report for currently running (incomplete) conditions only.
    Returns None if nothing is running."""
    if not all_records:
        return None

    # Per-condition stats, only for incomplete conditions
    cond_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "success": 0})
    for r in all_records:
        if r.condition_id in completed_conditions:
            continue
        cond_stats[r.condition_id]["total"] += 1
        cond_stats[r.condition_id]["success"] += int(r.success)

    if not cond_stats:
        return None

    lines = [f"run_id={run_id}"]
    for cid in sorted(cond_stats):
        s = cond_stats[cid]
        n, succ = s["total"], s["success"]
        mode = condition_mode_cache.get(cid, "?")
        lines.append(f"[{mode}] {cid}: {succ}/{n} ({succ/n:.1%})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis output watchers
# ---------------------------------------------------------------------------

# Files to watch for analysis completion
_ANALYSIS_MARKERS = {
    "condition_analysis": "analysis/_overview/tables/condition_metrics.csv",
    "reason_diagnostics": "analysis/reason_diagnostics/reason_diagnostics_summary.json",
    "cross_representation": "analysis/cross_representation/cross_representation_summary.json",
}


def _check_analysis_outputs(
    run_dir: Path,
    seen_analysis: Dict[str, float],
) -> List[Tuple[str, Path]]:
    """Return list of (analysis_name, path) for newly appeared/updated analysis outputs."""
    new_outputs: List[Tuple[str, Path]] = []
    for name, rel_path in _ANALYSIS_MARKERS.items():
        p = run_dir / rel_path
        if not p.exists():
            continue
        mtime = p.stat().st_mtime
        prev_mtime = seen_analysis.get(name, 0.0)
        if mtime > prev_mtime:
            seen_analysis[name] = mtime
            if prev_mtime > 0:  # skip first detection (bootstrap)
                new_outputs.append((name, p))
            else:
                seen_analysis[name] = mtime  # record but don't alert
    return new_outputs


# ---------------------------------------------------------------------------
# Condition completion watcher
# ---------------------------------------------------------------------------

def _check_condition_completions(
    run_dir: Path,
    condition_filter: Optional[str],
    seen_completions: Set[str],
    condition_mode_cache: Dict[str, str],
) -> List[Tuple[str, str]]:
    """Return list of (condition_id, obs_mode) for newly completed conditions."""
    new_completions: List[Tuple[str, str]] = []
    if condition_filter:
        cond_dirs = [run_dir / condition_filter]
    else:
        cond_dirs = [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("phase")]

    for cond_dir in cond_dirs:
        cid = cond_dir.name
        if cid in seen_completions:
            continue
        summary_path = cond_dir / "condition_summary_v2.json"
        if summary_path.exists():
            seen_completions.add(cid)
            mode = _get_observation_mode(cond_dir, condition_mode_cache)
            new_completions.append((cid, mode))
    return new_completions


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def _load_state(path: Optional[Path]) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(
    path: Optional[Path],
    seen_keys: Set[str],
    seen_completions: Set[str],
    seen_analysis: Dict[str, float],
) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seen_keys": sorted(seen_keys),
        "seen_completions": sorted(seen_completions),
        "seen_analysis": seen_analysis,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Auto-digest: update reason CSV + run batch digest
# ---------------------------------------------------------------------------

def _run_auto_digest(run_dir: Path, glm_config: Path, digest_dir: Path) -> Optional[str]:
    """Run reason diagnostics → batch digest pipeline. Returns status string or None on skip."""
    scripts_dir = Path(__file__).parent
    python = sys.executable

    # 1. Update reason diagnostics CSV
    diag_script = scripts_dir / "analysis" / "analyze_reason_diagnostics.py"
    if not diag_script.exists():
        return None
    try:
        r = subprocess.run(
            [python, str(diag_script), "--run-dir", str(run_dir), "--skip-similarity"],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            print(f"[watchdog][DIGEST] reason diagnostics failed: {r.stderr[-300:]}")
            return "diagnostics_failed"
    except subprocess.TimeoutExpired:
        print("[watchdog][DIGEST] reason diagnostics timed out")
        return "diagnostics_timeout"

    # 2. Run batch digest (auto-resumes, writes to digest_dir/digest_{mode}.jsonl)
    digest_script = scripts_dir / "glm_batch_digest.py"
    if not digest_script.exists() or not glm_config.exists():
        return None
    try:
        r = subprocess.run(
            [python, str(digest_script),
             "--run-dir", str(run_dir),
             "--output", str(digest_dir),
             "--glm-config", str(glm_config),
             "--max-images", "3",
             "--delay-secs", "3.0",
             "--site", "classifieds"],
            capture_output=True, text=True, timeout=600,
        )
        # Extract last few lines for status
        out_lines = (r.stdout or "").strip().splitlines()
        tail = "\n".join(out_lines[-3:]) if out_lines else ""
        if r.returncode == 0:
            print(f"[watchdog][DIGEST] completed:\n  {tail}")
            return tail
        else:
            print(f"[watchdog][DIGEST] digest failed: {r.stderr[-300:]}")
            return "digest_failed"
    except subprocess.TimeoutExpired:
        print("[watchdog][DIGEST] digest timed out (10min)")
        return "digest_timeout"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Experiment watchdog — status reports & idle alerts")
    p.add_argument("--run-dir", required=True, help="Run directory")
    p.add_argument("--condition", default=None, help="Filter to specific condition_id")
    p.add_argument("--poll-secs", type=int, default=30, help="Polling interval (seconds)")
    p.add_argument("--max-steps", type=int, default=30, help="Configured episode max steps")
    p.add_argument("--report-interval-mins", type=int, default=30,
                    help="Push status report every N minutes (default: 30)")
    p.add_argument("--idle-alert-mins", type=int, default=20,
                    help="Alert if no new episode for N minutes (default: 20)")
    p.add_argument("--ntfy-topic", default=None, help="ntfy topic for push notifications")
    p.add_argument("--state-file", default=None, help="State file for persistence across restarts")
    p.add_argument("--glm-config", default=None, type=Path,
                    help="GLM config file for auto-digest (omit to disable)")
    p.add_argument("--digest-dir", default=None, type=Path,
                    help="Digest output directory (default: <run-dir>/analysis/digest/)")
    p.add_argument("--once", action="store_true", help="Scan once then exit")
    return p


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"run_dir not found: {run_dir}")

    run_id = run_dir.name
    state_file = Path(args.state_file).resolve() if args.state_file else None

    # Load persisted state
    saved = _load_state(state_file)
    seen_keys: Set[str] = set(saved.get("seen_keys", []))
    seen_completions: Set[str] = set(saved.get("seen_completions", []))
    seen_analysis: Dict[str, float] = saved.get("seen_analysis", {})

    all_records: List[EpisodeRecord] = []
    condition_mode_cache: Dict[str, str] = {}

    # Bootstrap: rebuild all_records from existing summaries (for accurate counts)
    if seen_keys:
        for summary_path in _scan_summaries(run_dir, args.condition):
            key = _episode_key(summary_path)
            if key not in seen_keys:
                continue
            m_re = SUMMARY_RE.match(summary_path.name)
            if not m_re:
                continue
            try:
                summary = _read_json(summary_path)
            except Exception:
                continue
            condition_dir = summary_path.parent.parent
            obs_mode = _get_observation_mode(condition_dir, condition_mode_cache)
            reason = _classify_episode(summary, {}, args.max_steps)
            all_records.append(EpisodeRecord(
                condition_id=condition_dir.name,
                observation_mode=obs_mode,
                site=m_re.group("site"),
                task_id=int(m_re.group("task_id")),
                success=bool(summary.get("success", False)),
                steps=int(summary.get("steps", 0) or 0),
                reason=reason,
            ))
        print(f"[watchdog] Restored {len(all_records)} episodes from state")

    # Timers
    last_new_episode_ts: float = time.time()
    last_report_ts: float = 0.0  # 0 → trigger initial report immediately after bootstrap
    idle_alerted: bool = False
    idle_alert_secs = max(60, args.idle_alert_mins * 60)
    report_interval_secs = max(60, args.report_interval_mins * 60)

    # Bootstrap: scan existing analysis files and completions without alerting
    _check_analysis_outputs(run_dir, seen_analysis)
    _check_condition_completions(run_dir, args.condition, seen_completions, condition_mode_cache)

    print(
        f"[watchdog] run_id={run_id} condition={args.condition or '*'} "
        f"poll={args.poll_secs}s report_every={args.report_interval_mins}min "
        f"idle_alert={args.idle_alert_mins}min"
    )

    while True:
        now = time.time()

        # --- 1. Scan new episodes ---
        summaries = _scan_summaries(run_dir, args.condition)
        new_paths = [p for p in summaries if _episode_key(p) not in seen_keys]

        if new_paths:
            last_new_episode_ts = now
            idle_alerted = False

            for summary_path in sorted(new_paths):
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
                except Exception:
                    continue

                condition_dir = summary_path.parent.parent
                obs_mode = _get_observation_mode(condition_dir, condition_mode_cache)
                reason = _classify_episode(summary, {}, args.max_steps)

                rec = EpisodeRecord(
                    condition_id=condition_id,
                    observation_mode=obs_mode,
                    site=site,
                    task_id=task_id,
                    success=bool(summary.get("success", False)),
                    steps=int(summary.get("steps", 0) or 0),
                    reason=reason,
                )
                all_records.append(rec)
                seen_keys.add(key)

                # Per-condition cumulative
                cond_all = [r for r in all_records if r.condition_id == condition_id]
                cond_total = len(cond_all)
                cond_succ = sum(1 for r in cond_all if r.success)

                print(
                    f"[watchdog] [{obs_mode}] {condition_id} task={task_id:>3d} "
                    f"{'OK' if rec.success else reason:<10s} "
                    f"succ={cond_succ}/{cond_total} ({cond_succ/cond_total:.1%})"
                )

            _save_state(state_file, seen_keys, seen_completions, seen_analysis)

        # --- 2. Periodic status report (only running conditions) ---
        if (now - last_report_ts) >= report_interval_secs and all_records:
            report = _build_status_report(all_records, condition_mode_cache, seen_completions, run_id)
            if report:
                print(f"[watchdog][REPORT]\n{report}")
                if args.ntfy_topic:
                    _post_ntfy(args.ntfy_topic, f"P79 Status", report)

            # Auto-digest: update reason CSV + run batch digest
            if args.glm_config:
                digest_dir = args.digest_dir or (run_dir / "analysis" / "digest")
                _run_auto_digest(run_dir, args.glm_config, digest_dir)

            last_report_ts = now

        # --- 3. Idle alert ---
        if not new_paths:
            idle_elapsed = now - last_new_episode_ts
            if idle_elapsed >= idle_alert_secs and not idle_alerted:
                idle_mins = int(idle_elapsed / 60)
                report = _build_status_report(all_records, condition_mode_cache, seen_completions, run_id) or ""
                idle_body = (
                    f"已 {idle_mins} 分钟无新 episode（阈值={args.idle_alert_mins}min）\n\n"
                    f"{report}"
                ).strip()
                print(f"[watchdog][IDLE] {idle_body}")
                if args.ntfy_topic:
                    _post_ntfy(args.ntfy_topic, f"P79 IDLE {idle_mins}min", idle_body, priority="high")
                idle_alerted = True

        # --- 4. Condition completion ---
        new_completions = _check_condition_completions(
            run_dir, args.condition, seen_completions, condition_mode_cache
        )
        for cid, mode in new_completions:
            # Read condition summary for final stats
            cond_all = [r for r in all_records if r.condition_id == cid]
            cond_total = len(cond_all)
            cond_succ = sum(1 for r in cond_all if r.success)
            body = (
                f"run_id={run_id}\n"
                f"[{mode}] {cid} 已完成\n"
                f"结果: {cond_succ}/{cond_total} ({cond_succ/cond_total:.1%})" if cond_total else
                f"run_id={run_id}\n[{mode}] {cid} 已完成"
            )
            print(f"[watchdog][COMPLETE] {body}")
            if args.ntfy_topic:
                _post_ntfy(args.ntfy_topic, f"P79 COMPLETE [{mode}/{cid}]", body, priority="high")
            _save_state(state_file, seen_keys, seen_completions, seen_analysis)

        # --- 5. Analysis script completion ---
        new_analysis = _check_analysis_outputs(run_dir, seen_analysis)
        for name, path in new_analysis:
            body = f"run_id={run_id}\n分析脚本完成: {name}\n输出: {path.relative_to(run_dir)}"
            print(f"[watchdog][ANALYSIS] {body}")
            if args.ntfy_topic:
                _post_ntfy(args.ntfy_topic, f"P79 Analysis [{name}]", body)
            _save_state(state_file, seen_keys, seen_completions, seen_analysis)

        if args.once:
            break
        time.sleep(max(1, args.poll_secs))

    # Final summary
    if all_records:
        report = _build_status_report(all_records, condition_mode_cache, seen_completions, run_id)
        print(f"\n[watchdog][FINAL]\n{report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
