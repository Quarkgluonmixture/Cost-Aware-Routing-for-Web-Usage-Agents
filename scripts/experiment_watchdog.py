#!/usr/bin/env python3
"""
Experiment watchdog — lightweight monitoring with periodic status reports.

Notifications (push to ntfy):
1) REPORT:   periodic status every --report-interval-mins (success rate + counts)
2) IDLE:     no new episode for --idle-alert-mins → may need restart
3) COMPLETE: condition finished (condition_summary_v2.json appeared)
4) ANALYSIS: post-condition analysis script completed (output files detected)
5) DIGEST:   GLM batch digest completed for a mode (all failed episodes processed)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
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

# Session-health heuristics: patterns in step_000 DOM that indicate login state
_LOGIN_ABSENT_RE = re.compile(r"link\s+'(?:Login|Sign In)'", re.IGNORECASE)
_LOGIN_PRESENT_RE = re.compile(r"link\s+'(?:Logout|My account|Sign Out|Log Out)'", re.IGNORECASE)
_SESSION_ALERT_THRESHOLD = 3  # consecutive tasks w/o login before alerting


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
        headers={"Title": title, "Priority": priority, "Markdown": "yes"},
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


def _check_session_health(condition_dir: Path, site: str, task_id: int) -> Optional[bool]:
    """Check step_000 DOM for login state. True=logged-in, False=not, None=unknown."""
    dom_path = condition_dir / "artifacts" / f"{site}_task_{task_id}" / "step_000" / "observation_dom.txt"
    if not dom_path.exists():
        return None
    try:
        text = dom_path.read_text(encoding="utf-8", errors="replace")[:5000]
    except Exception:
        return None
    has_login_link = bool(_LOGIN_ABSENT_RE.search(text))
    has_logout_link = bool(_LOGIN_PRESENT_RE.search(text))
    if has_logout_link:
        return True
    if has_login_link:
        return False
    return None


def _classify_episode(
    summary: Dict[str, Any],
    task_meta: Dict[str, Any],
    max_steps: int,
) -> str:
    if bool(summary.get("success", False)):
        return "success"
    if summary.get("error"):
        if summary.get("benchmark_noise"):
            cat = summary.get("benchmark_noise_category", "unknown")
            return f"error({cat})"
        return "error(code_bug)"
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

# Digest modes to track (matches digest_{mode}.jsonl naming)
_DIGEST_MODES = ("dom", "som", "vision")


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


def _count_failed_episodes_by_mode(
    all_records: List[EpisodeRecord],
    completed_conditions: Set[str],
) -> Dict[str, int]:
    """Count failed episodes per observation mode across completed conditions."""
    counts: Dict[str, int] = defaultdict(int)
    for r in all_records:
        if r.condition_id in completed_conditions and not r.success:
            counts[r.observation_mode] += 1
    return dict(counts)


def _check_digest_completions(
    digest_dir: Path,
    all_records: List[EpisodeRecord],
    completed_conditions: Set[str],
    seen_digest_completions: Set[str],
) -> List[Tuple[str, int, int]]:
    """Check if digest JSONL has covered all failed episodes for each mode.

    Returns list of (mode, digested_count, expected_count) for newly complete modes.
    """
    if not digest_dir.exists():
        return []
    expected_by_mode = _count_failed_episodes_by_mode(all_records, completed_conditions)
    if not expected_by_mode:
        return []

    newly_complete: List[Tuple[str, int, int]] = []
    for mode in _DIGEST_MODES:
        if mode in seen_digest_completions:
            continue
        expected = expected_by_mode.get(mode, 0)
        if expected == 0:
            continue
        digest_file = digest_dir / f"digest_{mode}.jsonl"
        if not digest_file.exists():
            continue
        try:
            # Count unique (condition_id, task_id) pairs to tolerate duplicate lines
            seen_pairs: set = set()
            for line in digest_file.open(encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    seen_pairs.add((str(obj.get("condition_id", "")), obj.get("task_id")))
                except Exception:
                    pass
            digested = len(seen_pairs)
        except Exception:
            continue
        if digested >= expected:
            seen_digest_completions.add(mode)
            newly_complete.append((mode, digested, expected))
    return newly_complete


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


def _annotate_screenshots(run_dir: Path, condition: Optional[str] = None) -> str:
    """Run screenshot annotation (best-effort, non-blocking). Returns status string."""
    try:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "annotate_screenshots.py"),
            "--run-dir", str(run_dir),
        ]
        if condition:
            cmd += ["--condition", condition]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            # Last line has the summary
            last_line = (r.stdout.strip().splitlines() or [""])[-1]
            print(f"[watchdog][ANNOTATE] {last_line}")
            return last_line
        else:
            msg = f"failed: {r.stderr[-200:]}"
            print(f"[watchdog][ANNOTATE] {msg}")
            return msg
    except subprocess.TimeoutExpired:
        print("[watchdog][ANNOTATE] timed out (300s)")
        return "timed out"
    except Exception as exc:
        print(f"[watchdog][ANNOTATE] error: {exc}")
        return f"error: {exc}"


def _regenerate_gallery(run_dir: Path, condition: Optional[str] = None) -> str:
    """Regenerate the gallery HTML (best-effort, non-blocking). Returns status string."""
    try:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "generate_gallery.py"),
            "--run-dir", str(run_dir),
        ]
        if condition:
            cmd += ["--condition", condition]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            print(f"[watchdog][GALLERY] regenerated: {run_dir / 'gallery.html'}")
            return "updated"
        else:
            msg = f"failed: {r.stderr[-200:]}"
            print(f"[watchdog][GALLERY] {msg}")
            return msg
    except subprocess.TimeoutExpired:
        print("[watchdog][GALLERY] timed out (120s)")
        return "timed out"
    except Exception as exc:
        print(f"[watchdog][GALLERY] error: {exc}")
        return f"error: {exc}"


def _save_state(
    path: Optional[Path],
    seen_keys: Set[str],
    seen_completions: Set[str],
    seen_analysis: Dict[str, float],
    seen_digest_completions: Optional[Set[str]] = None,
    reported_keys: Optional[Set[str]] = None,
) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seen_keys": sorted(seen_keys),
        "seen_completions": sorted(seen_completions),
        "seen_analysis": seen_analysis,
        "seen_digest_completions": sorted(seen_digest_completions or set()),
        "reported_keys": sorted(reported_keys or set()),
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
    seen_digest_completions: Set[str] = set(saved.get("seen_digest_completions", []))
    reported_keys: Set[str] = set(saved.get("reported_keys", []))

    all_records: List[EpisodeRecord] = []
    condition_mode_cache: Dict[str, str] = {}

    # Bootstrap: rebuild all_records from existing summaries (for accurate counts)
    if seen_keys:
        live_keys: Set[str] = set()
        for summary_path in _scan_summaries(run_dir, args.condition):
            key = _episode_key(summary_path)
            live_keys.add(key)
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
        # Prune stale keys whose files were deleted (e.g. cleared for retry)
        stale_keys = seen_keys - live_keys
        if stale_keys:
            seen_keys -= stale_keys
            reported_keys -= stale_keys
            print(f"[watchdog] Pruned {len(stale_keys)} stale keys (files deleted since last run)")
        print(f"[watchdog] Restored {len(all_records)} episodes from state")

    # Session-loss tracking: per-site streak counters
    session_loss_streak: Dict[str, int] = defaultdict(int)
    session_alerted: Dict[str, bool] = defaultdict(bool)

    # Timers
    last_new_episode_ts: float = time.time()
    last_report_ts: float = 0.0  # 0 → trigger initial report immediately after bootstrap
    idle_alerted: bool = False
    # (recent episodes computed from seen_keys - reported_keys at report time)
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

                # Auto-cleanup: delete benchmark noise errors so runner can retry.
                # Only when the condition is still running (no condition_summary yet);
                # once aggregated, deleting episodes would create inconsistency.
                condition_completed = (condition_dir / "condition_summary_v2.json").exists()
                if (
                    reason.startswith("error(")
                    and reason != "error(code_bug)"
                    and not condition_completed
                ):
                    # 1. Delete summary JSON
                    try:
                        summary_path.unlink()
                    except OSError:
                        pass
                    # 2. Delete steps JSONL
                    steps_file = summary_path.parent / f"{site}_task_{task_id}_steps_v2.jsonl"
                    try:
                        if steps_file.exists():
                            steps_file.unlink()
                    except OSError:
                        pass
                    # 3. Delete artifacts directory
                    artifacts_dir = condition_dir / "artifacts" / f"{site}_task_{task_id}"
                    try:
                        if artifacts_dir.exists():
                            shutil.rmtree(artifacts_dir)
                    except OSError:
                        pass
                    print(
                        f"[watchdog][AUTO-RETRY] deleted benchmark noise error: "
                        f"task {task_id} ({reason})"
                    )
                    if args.ntfy_topic:
                        _post_ntfy(
                            args.ntfy_topic,
                            f"P79 AUTO-RETRY task {task_id}",
                            f"run_id={run_id}\n{condition_id} task {task_id}\n"
                            f"deleted {reason} — waiting for runner retry",
                        )
                    # Do NOT add to seen_keys/all_records — treat as never happened
                    continue

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

                # --- Session health check ---
                session_ok = _check_session_health(condition_dir, site, task_id)
                if session_ok is False:
                    session_loss_streak[site] += 1
                    streak = session_loss_streak[site]
                    print(
                        f"[watchdog][SESSION] {site} task {task_id} "
                        f"NOT LOGGED IN (streak={streak})"
                    )
                    if (
                        streak >= _SESSION_ALERT_THRESHOLD
                        and not session_alerted[site]
                    ):
                        session_alerted[site] = True
                        body = (
                            f"run_id={run_id}\n"
                            f"{site}: {streak} consecutive tasks without login!\n"
                            f"Cookie/session 已过期，需刷新 .auth/{site}_state.json\n"
                            f"python auto_login.py --site {site}"
                        )
                        print(f"[watchdog][SESSION] ALERT: {body}")
                        if args.ntfy_topic:
                            _post_ntfy(
                                args.ntfy_topic,
                                f"P79 SESSION LOST [{site}]",
                                body,
                                priority="urgent",
                            )
                elif session_ok is True:
                    if session_loss_streak[site] > 0:
                        print(
                            f"[watchdog][SESSION] {site} login restored "
                            f"(was streak={session_loss_streak[site]})"
                        )
                    session_loss_streak[site] = 0
                    session_alerted[site] = False

                # Per-condition cumulative
                cond_all = [r for r in all_records if r.condition_id == condition_id]
                cond_total = len(cond_all)
                cond_succ = sum(1 for r in cond_all if r.success)

                print(
                    f"[watchdog] [{obs_mode}] {condition_id} task={task_id:>3d} "
                    f"{'OK' if rec.success else reason:<10s} "
                    f"succ={cond_succ}/{cond_total} ({cond_succ/cond_total:.1%})"
                )

            _save_state(state_file, seen_keys, seen_completions, seen_analysis, seen_digest_completions, reported_keys)

        # --- 2. Periodic status report (only running conditions) ---
        if (now - last_report_ts) >= report_interval_secs and all_records:
            report = _build_status_report(all_records, condition_mode_cache, seen_completions, run_id)
            if report:
                print(f"[watchdog][REPORT]\n{report}")

            # Auto-digest: update reason CSV + run batch digest
            digest_status = None
            digest_completions_info: List[str] = []
            if args.glm_config:
                digest_dir = args.digest_dir or (run_dir / "analysis" / "digest")
                digest_status = _run_auto_digest(run_dir, args.glm_config, digest_dir)

                # Check if digest is newly complete for any mode
                newly_done = _check_digest_completions(
                    digest_dir, all_records, seen_completions, seen_digest_completions,
                )
                for mode, digested, expected in newly_done:
                    info = f"[{mode}] digest 完成: {digested}/{expected}"
                    digest_completions_info.append(info)
                    print(f"[watchdog][DIGEST] {info}")
                    if args.ntfy_topic:
                        _post_ntfy(args.ntfy_topic, f"P79 Digest [{mode}]",
                                   f"run_id={run_id}\n{info}\n输出: analysis/digest/digest_{mode}.jsonl",
                                   priority="high")
                    _save_state(state_file, seen_keys, seen_completions, seen_analysis, seen_digest_completions, reported_keys)

            # Auto-annotate screenshots then regenerate gallery HTML
            annotate_status = _annotate_screenshots(run_dir, args.condition)
            gallery_status = _regenerate_gallery(run_dir, args.condition)

            # Run analysis scripts (results fed into consolidated notification)
            new_analysis = _check_analysis_outputs(run_dir, seen_analysis)
            analysis_names = []
            for name, path in new_analysis:
                print(f"[watchdog][ANALYSIS] run_id={run_id}\n分析脚本完成: {name}\n输出: {path.relative_to(run_dir)}")
                analysis_names.append(name)
                _save_state(state_file, seen_keys, seen_completions, seen_analysis, seen_digest_completions, reported_keys)

            # --- Build consolidated periodic notification ---
            if args.ntfy_topic and report:
                parts = [report]

                # Recent tasks: episodes in seen_keys but not yet reported
                unreported_keys = seen_keys - reported_keys
                if unreported_keys:
                    # Match records by condition-aware path, not just filename suffix.
                    # Filenames are identical across conditions (e.g. classifieds_task_0_summary_v2.json),
                    # so we must include the condition_id directory in the match.
                    recent = [
                        r for r in all_records
                        if any(
                            k.endswith(f"{r.condition_id}/episodes/{r.site}_task_{r.task_id}_summary_v2.json")
                            for k in unreported_keys
                        )
                    ]
                    if recent:
                        task_lines = []
                        for ep in sorted(recent, key=lambda e: (e.condition_id, e.task_id)):
                            status = "OK" if ep.success else ep.reason
                            task_lines.append(f"  [{ep.observation_mode}] task {ep.task_id}: {status} ({ep.steps} steps)")
                        parts.append(f"**新完成 ({len(recent)} tasks)**")
                        parts.extend(task_lines)

                # Pipeline status
                pipeline = []
                if digest_status is not None:
                    # Take last meaningful line only
                    digest_short = (digest_status.strip().splitlines() or [""])[-1][:80]
                    pipeline.append(f"digest: {digest_short}")
                pipeline.append(f"annotate: {annotate_status.strip().splitlines()[-1][:80]}")
                pipeline.append(f"gallery: {gallery_status}")
                if analysis_names:
                    pipeline.append(f"analysis: {', '.join(analysis_names)}")
                parts.append("**pipeline**")
                parts.extend(pipeline)

                _post_ntfy(args.ntfy_topic, "P79 Status", "\n".join(parts))

            reported_keys = seen_keys.copy()
            _save_state(state_file, seen_keys, seen_completions, seen_analysis, seen_digest_completions, reported_keys)
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
            _save_state(state_file, seen_keys, seen_completions, seen_analysis, seen_digest_completions, reported_keys)

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
