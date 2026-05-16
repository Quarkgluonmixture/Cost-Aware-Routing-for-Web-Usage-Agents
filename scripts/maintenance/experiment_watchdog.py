#!/usr/bin/env python3
"""
Experiment watchdog — lightweight monitoring with periodic status reports.

Notifications (push to ntfy):
1) REPORT:   periodic status every --report-interval-mins (success rate + counts)
2) IDLE:     no new episode for --idle-alert-mins → may need restart
3) COMPLETE: optional (off by default; enable with --notify-completion)
4) ANALYSIS: post-condition analysis summary (kept on by default)
5) DIGEST:   included as pipeline summary in periodic status (no standalone push)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
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
_LOGIN_ABSENT_RE = re.compile(r"link\s+'(?:Login|Log in|Sign In)'", re.IGNORECASE)
_LOGIN_PRESENT_RE = re.compile(r"link\s+'(?:Logout|Log out|Sign Out)'", re.IGNORECASE)
_SESSION_ALERT_THRESHOLD = 3  # consecutive tasks w/o login before alerting

# Directories inside run_dir that are NOT condition directories
_EXCLUDED_DIRS = {"analysis", ".git", "gallery_data", "task_configs"}


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



def _post_ntfy(topic: str, title: str, body: str, priority: str = "default") -> None:
    url = f"https://ntfy.sh/{topic}"
    req = urllib.request.Request(
        url, data=body.encode("utf-8"), method="POST",
        headers={"Title": title, "Priority": priority, "Markdown": "yes"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15):
            pass
    except urllib.error.HTTPError as e:
        print(f"[watchdog][NTFY] HTTP {e.code} {e.reason} — notification dropped: {title}")
    except urllib.error.URLError as e:
        print(f"[watchdog][NTFY] URLError: {e.reason} — notification dropped: {title}")


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


def _auto_refresh_auth(site: str, *, benchmark: str = "") -> bool:
    """Re-login to site and refresh .auth/{site}_state.json using Playwright.

    Thin wrapper that delegates to shared module ``p79.utils.auth_refresh``.

    A1.5 cleanup (2026-05-16, B-211 + B-225 per Item 8): the inline-fallback
    ``_ACCOUNTS`` + ``_BASE_URLS`` + ``_LOGIN_PATHS`` duplicate block (previously
    ImportError fallback) is DELETED. Two reasons:
      (a) double-leak of plaintext credentials in tracked code (B-211)
      (b) hardcoded ``100.95.81.103`` IP defeated ``VWA_REMOTE_HOST`` env override (B-225)
    If ``p79.utils.auth_refresh`` import fails, fail loud with a helpful
    diagnostic — silent fallback is the worse outcome (writes wrong-domain
    storage state, contaminates downstream episodes).
    """
    try:
        from p79.utils.auth_refresh import refresh_site_auth
    except ImportError as exc:
        print(
            f"[watchdog][SESSION][FATAL] cannot import p79.utils.auth_refresh: {exc}. "
            f"Watchdog requires the project venv; activate with `source .venv/bin/activate` "
            f"or run `pip install -e .` then retry. NOT silently falling back to inline "
            f"credentials — that path was deleted 2026-05-16 (A1.5 B-211 / B-225 cleanup)."
        )
        return False
    repo_dir = Path(__file__).resolve().parent.parent.parent
    auth_dir = repo_dir / ".auth"
    ok = refresh_site_auth(site, auth_dir, benchmark=benchmark)
    if ok:
        print(f"[watchdog][SESSION] {site} auth auto-refreshed")
    else:
        print(f"[watchdog][SESSION][warn] {site} auto-refresh failed")
    return ok


def _purge_digest_records(digest_dir: Path, condition_id: str, task_id: int, obs_mode: str) -> int:
    """Remove records matching (condition_id, task_id) from digest_{obs_mode}.jsonl.

    Returns number of records removed.
    """
    if not digest_dir.exists():
        return 0
    digest_file = digest_dir / f"digest_{obs_mode}.jsonl"
    if not digest_file.exists():
        return 0
    try:
        lines = digest_file.read_text(encoding="utf-8").splitlines()
        keep, removed = [], 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                cid = rec.get("condition_id", "")
                if rec.get("task_id") == task_id and (not cid or cid == condition_id):
                    removed += 1
                    continue
            except Exception:
                pass
            keep.append(line)
        if removed:
            tmp_file = digest_file.with_suffix(".jsonl.tmp")
            tmp_file.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
            tmp_file.replace(digest_file)  # atomic on same filesystem
        return removed
    except Exception as exc:
        print(f"[watchdog][warn] purge_digest task {task_id}: {exc}")
        return 0


def _check_session_health(condition_dir: Path, site: str, task_id: int) -> Optional[bool]:
    """Check step_000 DOM for login state. True=logged-in, False=not, None=unknown."""
    dom_path = condition_dir / "artifacts" / f"{site}_task_{task_id}" / "step_000" / "observation_dom.txt"
    if not dom_path.exists():
        return None
    try:
        text = dom_path.read_text(encoding="utf-8", errors="replace")[:5000]
    except Exception:
        return None
    # Cross-site tasks may start on a different site's tab.  Detect via the
    # Tab 0 header line and skip session check if the active page belongs to
    # another site (login/logout links would be from the wrong site).
    first_line = text.split("\n", 1)[0].strip().lower()
    # Extract only the active tab portion (before first " | " separator)
    # to avoid matching site keywords in non-active tab labels.
    # e.g. "Tab 0 (current): Shopping Page | Tab 1: Classifieds"
    #   → only check "Tab 0 (current): Shopping Page"
    active_tab_part = first_line.split(" | ")[0]
    # Map site names to tab-header keywords (VWA uses page title as tab label)
    _SITE_TAB_KW: Dict[str, List[str]] = {
        "classifieds": ["classifieds"],
        "reddit": ["reddit"],
        "shopping": ["shopping", "one stop market"],
        "shopping_admin": ["shopping", "magento", "admin"],
    }
    expected_kws = _SITE_TAB_KW.get(site, [])
    if expected_kws and active_tab_part.startswith("tab ") and not any(kw in active_tab_part for kw in expected_kws):
        return None  # active tab belongs to another site; skip
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
        err_str = str(summary.get("error", ""))
        if err_str.startswith("evaluator_error:"):
            return "error(evaluator)"
        return "error(code_bug)"
    steps = int(summary.get("steps", 0) or 0)
    if steps >= max_steps:
        return "max_steps"
    return "fail"


def _scan_summaries(run_dir: Path, condition_filter: Optional[str]) -> List[Path]:
    if condition_filter:
        roots = [run_dir / condition_filter]
    else:
        roots = [p for p in run_dir.iterdir() if p.is_dir() and p.name not in _EXCLUDED_DIRS]
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
    "condition_analysis": "analysis/results/_overview/tables/condition_metrics.csv",
    "reason_diagnostics": "analysis/reason_diagnostics/reason_diagnostics_summary.json",
    "cross_representation": "analysis/results/cross_representation/cross_representation_summary.json",
    "confidence_calibration": "analysis/signals/combined/confidence_summary.json",
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
    """Return list of (condition_id, obs_mode) for newly completed conditions.

    A condition is considered "newly completed" if:
    1. condition_summary_v2.json exists AND not in seen_completions, OR
    2. condition_summary_v2.json is newer than analysis outputs (post-analysis stale).

    Case 2 handles scenarios where tasks were cleared and re-run, or where
    a previous watchdog instance was killed before post-analysis completed.
    """
    new_completions: List[Tuple[str, str]] = []
    if condition_filter:
        cond_dirs = [run_dir / condition_filter]
    else:
        cond_dirs = [p for p in run_dir.iterdir() if p.is_dir() and p.name not in _EXCLUDED_DIRS]

    # Reference analysis output for freshness check
    cross_rep_summary = run_dir / "analysis" / "results" / "cross_representation" / "cross_representation_summary.json"
    analysis_summary = run_dir / "analysis" / "analysis_summary.json"
    analysis_mtime = 0.0
    for ref in (cross_rep_summary, analysis_summary):
        if ref.exists():
            analysis_mtime = max(analysis_mtime, ref.stat().st_mtime)

    for cond_dir in cond_dirs:
        cid = cond_dir.name
        summary_path = cond_dir / "condition_summary_v2.json"
        if not summary_path.exists():
            continue

        if cid not in seen_completions:
            # Case 1: brand new completion
            seen_completions.add(cid)
            mode = _get_observation_mode(cond_dir, condition_mode_cache)
            new_completions.append((cid, mode))
        elif analysis_mtime > 0 and summary_path.stat().st_mtime > analysis_mtime:
            # Case 2: condition was re-run after last analysis (cleared & re-run)
            mode = _get_observation_mode(cond_dir, condition_mode_cache)
            new_completions.append((cid, mode))
            print(f"[watchdog] {cid}: condition_summary newer than analysis outputs, re-triggering post-analysis")
        elif analysis_mtime == 0 and cid in seen_completions:
            # Case 3: seen_completions says done but no analysis outputs at all
            mode = _get_observation_mode(cond_dir, condition_mode_cache)
            new_completions.append((cid, mode))
            print(f"[watchdog] {cid}: no analysis outputs found, re-triggering post-analysis")

    return new_completions


def _prune_stale_condition_completions(
    run_dir: Path,
    condition_filter: Optional[str],
    seen_completions: Set[str],
) -> int:
    """Remove stale completion flags that no longer have condition_summary files."""
    if condition_filter:
        cond_dirs = [run_dir / condition_filter]
    else:
        cond_dirs = [p for p in run_dir.iterdir() if p.is_dir() and p.name not in _EXCLUDED_DIRS]

    completed_now: Set[str] = set()
    for cond_dir in cond_dirs:
        if (cond_dir / "condition_summary_v2.json").exists():
            completed_now.add(cond_dir.name)

    stale = seen_completions - completed_now
    if not stale:
        return 0
    seen_completions -= stale
    return len(stale)


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

# Watchdog state schema version. Bumped when state shape changes (audit §97).
# v1 = pre-§97 (no session_*, no retry_decay)
# v2 = §97 audit (session_* persisted, retry counts reset on condition completion,
#                 schema_version recorded)
_STATE_SCHEMA_VERSION = "v2"


def _load_state(path: Optional[Path]) -> Dict[str, Any]:
    """Load watchdog state. Returns {} if missing/corrupt.

    Forward-compatible: callers use `state.get(key, default)` so adding new
    fields in newer schema versions does not break loading old state files.
    """
    if not path or not path.exists():
        return {}
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        # Migration: pre-v2 state files have no _schema_version. Treat as v1
        # and let the in-memory restore logic fill defaults for new fields.
        if isinstance(d, dict):
            d.setdefault("_schema_version", "v1")
        return d if isinstance(d, dict) else {}
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


def _run_post_condition_analysis(run_dir: Path) -> str:
    """Run analysis pipeline after a condition completes (best-effort). Returns status."""
    # __file__ lives in scripts/maintenance/; analysis scripts moved to
    # scripts/analysis/ during §99 reorg, so go up one then into analysis/.
    scripts_dir = Path(__file__).resolve().parent.parent
    statuses = []

    # 1. Main experiment analysis (analyze_experiment.py)
    analyze_script = scripts_dir / "analysis" / "analyze_experiment.py"
    if analyze_script.exists():
        try:
            r = subprocess.run(
                [sys.executable, str(analyze_script), "--run_dir", str(run_dir)],
                capture_output=True, text=True, timeout=300,
            )
            if r.returncode == 0:
                print("[watchdog][AUTO-ANALYSIS] analyze_experiment completed")
                statuses.append("experiment:ok")
            else:
                msg = r.stderr[-200:] if r.stderr else "unknown"
                print(f"[watchdog][AUTO-ANALYSIS] analyze_experiment failed: {msg}")
                statuses.append(f"experiment:failed")
        except subprocess.TimeoutExpired:
            print("[watchdog][AUTO-ANALYSIS] analyze_experiment timed out (300s)")
            statuses.append("experiment:timeout")
        except Exception as exc:
            print(f"[watchdog][AUTO-ANALYSIS] analyze_experiment error: {exc}")
            statuses.append(f"experiment:error")

    # 2. Confidence calibration
    conf_script = scripts_dir / "analysis" / "analyze_confidence_calibration.py"
    if conf_script.exists():
        try:
            r = subprocess.run(
                [sys.executable, str(conf_script), "--run-dir", str(run_dir)],
                capture_output=True, text=True, timeout=300,
            )
            if r.returncode == 0:
                print("[watchdog][AUTO-ANALYSIS] confidence_calibration completed")
                statuses.append("confidence:ok")
            else:
                msg = r.stderr[-200:] if r.stderr else "unknown"
                print(f"[watchdog][AUTO-ANALYSIS] confidence_calibration failed: {msg}")
                statuses.append("confidence:failed")
        except subprocess.TimeoutExpired:
            print("[watchdog][AUTO-ANALYSIS] confidence_calibration timed out (300s)")
            statuses.append("confidence:timeout")
        except Exception as exc:
            print(f"[watchdog][AUTO-ANALYSIS] confidence_calibration error: {exc}")
            statuses.append(f"confidence:error")

    # 3. Cross-representation analysis (need >=2 condition dirs)
    cross_script = scripts_dir / "analysis" / "analyze_cross_representation.py"
    if cross_script.exists():
        cond_dirs = [d for d in run_dir.iterdir()
                     if d.is_dir() and (d / "condition_summary_v2.json").exists()]
        if len(cond_dirs) >= 2:
            try:
                r = subprocess.run(
                    [sys.executable, str(cross_script),
                     "--run-dir", str(run_dir), "--priority", "all"],
                    capture_output=True, text=True, timeout=300,
                )
                if r.returncode == 0:
                    print("[watchdog][AUTO-ANALYSIS] cross_representation completed")
                    statuses.append("cross_rep:ok")
                else:
                    msg = r.stderr[-200:] if r.stderr else "unknown"
                    print(f"[watchdog][AUTO-ANALYSIS] cross_representation failed: {msg}")
                    statuses.append("cross_rep:failed")
            except subprocess.TimeoutExpired:
                print("[watchdog][AUTO-ANALYSIS] cross_representation timed out (300s)")
                statuses.append("cross_rep:timeout")
            except Exception as exc:
                print(f"[watchdog][AUTO-ANALYSIS] cross_representation error: {exc}")
                statuses.append(f"cross_rep:error")
        else:
            statuses.append("cross_rep:skipped(<2 conditions)")

    return "; ".join(statuses) if statuses else "skipped (no scripts found)"


# ---------------------------------------------------------------------------
# Cross-run analysis (跨 site 聚合 / 跨 baseline 对比)
# ---------------------------------------------------------------------------
# Naming convention: results/{benchmark}/phase1/{B0|B1|B2}_(?:wa_)?3mode_{site}_{YYYYMMDD}
_RUN_ID_RE = re.compile(r"^(B[012])_(?:wa_)?3mode_(.+?)_(\d{8})$")


def _parse_run_id(run_dir: Path) -> Optional[Dict[str, str]]:
    """Parse run_id; returns {benchmark, baseline, site, date} or None."""
    parts = run_dir.parts
    try:
        benchmark = parts[parts.index("results") + 1]
    except (ValueError, IndexError):
        return None
    m = _RUN_ID_RE.match(run_dir.name)
    if not m:
        return None
    return {"benchmark": benchmark, "baseline": m.group(1),
            "site": m.group(2), "date": m.group(3)}


def _has_any_completion(run_dir: Path) -> bool:
    """True iff at least one condition has condition_summary_v2.json."""
    if not run_dir.exists():
        return False
    return any(
        (d / "condition_summary_v2.json").exists()
        for d in run_dir.iterdir() if d.is_dir()
    )


def _regenerate_paper_figures() -> str:
    """Regenerate paper figures (best-effort, non-blocking). Returns status string."""
    figures_dir = Path(__file__).resolve().parent.parent / "analysis" / "figures"
    if not figures_dir.is_dir():
        return "skipped: figures dir missing"
    scripts = sorted(figures_dir.glob("fig*.py"))
    if not scripts:
        return "skipped: no figure scripts"
    ok, failed = 0, 0
    for s in scripts:
        try:
            r = subprocess.run(
                [sys.executable, str(s)],
                capture_output=True, text=True, timeout=60,
            )
            if r.returncode == 0:
                ok += 1
            else:
                failed += 1
                print(f"[watchdog][FIGURES] {s.name} failed: {r.stderr[-200:]}")
        except Exception as exc:
            failed += 1
            print(f"[watchdog][FIGURES] {s.name} error: {exc}")
    msg = f"{ok}/{ok+failed} ok"
    print(f"[watchdog][FIGURES] regenerated: {msg}")
    return msg


def _find_sibling_runs(phase_dir: Path, baseline: str, *, site: Optional[str] = None,
                       exclude: Optional[Path] = None) -> Dict[str, Path]:
    """Return {site: latest run_dir} for given baseline (and optional site filter).
    Skips runs with no completed condition."""
    by_site: Dict[str, Tuple[str, Path]] = {}
    if not phase_dir.exists():
        return {}
    for d in phase_dir.iterdir():
        if not d.is_dir() or (exclude and d == exclude):
            continue
        info = _parse_run_id(d)
        if not info or info["baseline"] != baseline:
            continue
        if site and info["site"] != site:
            continue
        if not _has_any_completion(d):
            continue
        prev = by_site.get(info["site"])
        if prev is None or prev[0] < info["date"]:
            by_site[info["site"]] = (info["date"], d)
    return {s: d for s, (_, d) in by_site.items()}


def _run_cross_run_analysis(run_dir: Path) -> Optional[str]:
    """Trigger aggregate_cross_site + compare_b0_b1 when sibling runs make them
    possible. Returns short status or None if nothing was triggered."""
    info = _parse_run_id(run_dir)
    if not info:
        return None
    phase_dir = run_dir.parent
    # §99 reorg: analysis scripts at scripts/analysis/ (sibling of maintenance/)
    scripts_dir = Path(__file__).resolve().parent.parent / "analysis"
    statuses: List[str] = []

    # 1) Pairwise baseline compare (3-baseline aware: B0/B1/B2)
    # For each OTHER baseline with a sibling run on the same site, trigger
    # compare_b0_b1.py. Labels still say b0/b1 (script not yet renamed) but
    # data wiring picks the lexicographically-lower baseline as "b0" so the
    # output is deterministic. Future: rename to compare_baselines.py.
    ALL_BASELINES = ("B0", "B1", "B2")
    self_b = info["baseline"]
    cmp_script = scripts_dir / "compare_b0_b1.py"
    for other_b in ALL_BASELINES:
        if other_b == self_b or not cmp_script.exists():
            continue
        sib = _find_sibling_runs(phase_dir, other_b, site=info["site"], exclude=run_dir)
        sib_run = sib.get(info["site"])
        if not sib_run:
            continue
        # Deterministic ordering: alphabetically-lower baseline → b0 slot
        if self_b < other_b:
            b0_dir, b1_dir, b0_label, b1_label = run_dir, sib_run, self_b, other_b
        else:
            b0_dir, b1_dir, b0_label, b1_label = sib_run, run_dir, other_b, self_b
        try:
            r = subprocess.run(
                [sys.executable, str(cmp_script),
                 "--b0-run-dir", str(b0_dir),
                 "--b1-run-dir", str(b1_dir),
                 "--site", info["site"]],
                capture_output=True, text=True, timeout=300,
            )
            tag = f"compare_{b0_label}_{b1_label}[{info['site']}]"
            statuses.append(f"{tag}:{'ok' if r.returncode == 0 else 'failed'}")
            if r.returncode != 0:
                print(f"[watchdog][CROSS-RUN] {tag} failed: {r.stderr[-200:]}")
        except subprocess.TimeoutExpired:
            statuses.append(f"compare_{b0_label}_{b1_label}[{info['site']}]:timeout")
        except Exception as exc:
            statuses.append(f"compare_{b0_label}_{b1_label}[{info['site']}]:error")
            print(f"[watchdog][CROSS-RUN] compare {b0_label}/{b1_label} error: {exc}")

    # 2) aggregate_cross_site: same baseline, ≥2 distinct sites
    same_b = _find_sibling_runs(phase_dir, info["baseline"], exclude=None)
    same_b[info["site"]] = run_dir  # ensure self present
    agg_script = scripts_dir / "aggregate_cross_site.py"
    if len(same_b) >= 2 and agg_script.exists():
        try:
            r = subprocess.run(
                [sys.executable, str(agg_script),
                 "--run-dirs", *(str(d) for d in same_b.values()),
                 "--b1-label", info["baseline"]],
                capture_output=True, text=True, timeout=300,
            )
            sites_str = ",".join(sorted(same_b.keys()))
            tag = f"cross_site[{info['baseline']}/{sites_str}]"
            statuses.append(f"{tag}:{'ok' if r.returncode == 0 else 'failed'}")
            if r.returncode != 0:
                print(f"[watchdog][CROSS-RUN] {tag} failed: {r.stderr[-200:]}")
        except subprocess.TimeoutExpired:
            statuses.append(f"cross_site[{info['baseline']}]:timeout")
        except Exception as exc:
            statuses.append(f"cross_site[{info['baseline']}]:error")
            print(f"[watchdog][CROSS-RUN] aggregate_cross_site error: {exc}")

    return "; ".join(statuses) if statuses else None


def _regenerate_gallery(run_dir: Path, condition: Optional[str] = None,
                        aggregate_prefix: str = "B1_3mode") -> str:
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
            # Also refresh aggregate gallery (best-effort). Pass run_dir so the
            # combined/unified rebuild can include it as --extra-run-dir when
            # its name doesn't match the aggregate prefix (legacy naming).
            _regenerate_aggregate_gallery(run_dir.parent, aggregate_prefix, self_run_dir=run_dir)
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


def _regenerate_aggregate_gallery(
    phase_dir: Path,
    prefix: str = "B1_3mode",
    self_run_dir: Optional[Path] = None,
) -> None:
    """Refresh the aggregate gallery (best-effort, silent)."""
    try:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "generate_gallery.py"),
            "--phase-dir", str(phase_dir),
            "--prefix", prefix,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            print(f"[watchdog][GALLERY] aggregate refreshed: {phase_dir / prefix / 'gallery.html'}")
        else:
            print(f"[watchdog][GALLERY] aggregate failed: {r.stderr[-200:]}")
    except Exception as exc:
        print(f"[watchdog][GALLERY] aggregate error: {exc}")

    # Also refresh combined (cross-benchmark) gallery if both VWA and WA dirs exist
    _regenerate_combined_gallery(phase_dir, prefix, self_run_dir=self_run_dir)
    # And refresh the per-baseline unified gallery (3mode + phantom)
    _regenerate_unified_gallery(phase_dir, prefix, self_run_dir=self_run_dir)


def _build_extra_run_args(prefix: str, self_run_dir: Optional[Path]) -> List[str]:
    """If self_run_dir's name doesn't start with prefix_, return [--extra-run-dir, path]."""
    if not self_run_dir:
        return []
    if self_run_dir.name.startswith(f"{prefix}_"):
        return []
    return ["--extra-run-dir", str(self_run_dir)]


def _regenerate_combined_gallery(
    phase_dir: Path, prefix: str, self_run_dir: Optional[Path] = None,
) -> None:
    """Refresh the combined VWA+WA gallery (best-effort, silent).

    Infers the counterpart benchmark dir and generates a combined gallery
    under results/<prefix>_gallery/.
    """
    try:
        results_root = phase_dir.parent.parent  # results/
        phase_name = phase_dir.name  # phase1
        vwa_phase = results_root / "visualwebarena" / phase_name
        wa_phase = results_root / "webarena" / phase_name
        phase_dirs = [str(d) for d in (vwa_phase, wa_phase) if d.is_dir()]
        if not phase_dirs:
            return  # No benchmark dirs found

        # Derive combined prefix: B1_wa_3mode -> B1_3mode, B0_wa_3mode -> B0_3mode
        combined_prefix = prefix.replace("_wa_", "_") if "_wa_" in prefix else prefix
        output_dir = results_root / combined_prefix

        extra_args = _build_extra_run_args(combined_prefix, self_run_dir)

        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "generate_gallery.py"),
            "--phase-dirs", *phase_dirs,
            "--prefix", combined_prefix,
            "--output-dir", str(output_dir),
            *extra_args,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if r.returncode == 0:
            print(f"[watchdog][GALLERY] combined refreshed: {output_dir / 'gallery.html'}")
        else:
            print(f"[watchdog][GALLERY] combined failed: {r.stderr[-200:]}")
    except Exception as exc:
        print(f"[watchdog][GALLERY] combined error: {exc}")


def _regenerate_unified_gallery(
    phase_dir: Path, prefix: str, self_run_dir: Optional[Path] = None,
) -> None:
    """Refresh the per-baseline unified gallery (3mode + phantom, VWA+WA).

    Triggers when *prefix* belongs to a known unified family (B0_3mode,
    B0_phantom, B1_3mode, B1_phantom). Output: results/<baseline>_unified/.
    """
    # Map any known component prefix to its baseline letter (B0/B1/B2)
    baseline = None
    for b in ("B0", "B1", "B2"):
        if prefix in (f"{b}_3mode", f"{b}_wa_3mode", f"{b}_phantom", f"{b}_wa_phantom"):
            baseline = b
            break
    if baseline is None:
        return  # Not a unified-family prefix; skip

    try:
        results_root = phase_dir.parent.parent  # results/
        phase_name = phase_dir.name  # phase1
        vwa_phase = results_root / "visualwebarena" / phase_name
        wa_phase = results_root / "webarena" / phase_name
        phase_dirs = [str(d) for d in (vwa_phase, wa_phase) if d.is_dir()]
        if not phase_dirs:
            return

        unified_prefixes = [f"{baseline}_3mode", f"{baseline}_phantom"]
        output_dir = results_root / f"{baseline}_unified"

        # Forward self_run_dir if it doesn't match either unified component
        extra_args: List[str] = []
        if self_run_dir and not any(
            self_run_dir.name.startswith(f"{p}_") for p in unified_prefixes
        ):
            extra_args = ["--extra-run-dir", str(self_run_dir)]

        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "generate_gallery.py"),
            "--phase-dirs", *phase_dirs,
            "--prefix", *unified_prefixes,
            "--output-dir", str(output_dir),
            *extra_args,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
        if r.returncode == 0:
            print(f"[watchdog][GALLERY] unified refreshed: {output_dir / 'gallery.html'}")
        else:
            print(f"[watchdog][GALLERY] unified failed: {r.stderr[-200:]}")
    except Exception as exc:
        print(f"[watchdog][GALLERY] unified error: {exc}")


def _save_state(
    path: Optional[Path],
    seen_keys: Set[str],
    seen_completions: Set[str],
    seen_analysis: Dict[str, float],
    seen_digest_completions: Optional[Set[str]] = None,
    reported_keys: Optional[Set[str]] = None,
    error_retry_counts: Optional[Dict[str, int]] = None,
    *,
    session_loss_streak: Optional[Dict[str, int]] = None,
    session_alerted: Optional[Dict[str, bool]] = None,
    session_auto_refresh_attempted: Optional[Dict[str, bool]] = None,
    session_contaminated: Optional[Dict[str, list]] = None,
) -> None:
    """Persist watchdog state. §97 audit added session_* fields so they survive
    watchdog restarts (was: lost on restart → missed cleanup of contaminated
    NOT-LOGGED-IN episodes)."""
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # session_contaminated values are list of (cond_id, condition_dir, task_id,
    # site, key) where condition_dir is a Path — JSON can't serialize Path,
    # so stringify here and re-Path on load.
    session_contaminated_serializable: Dict[str, list] = {}
    for site_key, items in (session_contaminated or {}).items():
        session_contaminated_serializable[site_key] = [
            [str(cond_id), str(condition_dir), int(task_id), str(site), str(key)]
            for (cond_id, condition_dir, task_id, site, key) in items
        ]
    payload = {
        "_schema_version": _STATE_SCHEMA_VERSION,
        "seen_keys": sorted(seen_keys),
        "seen_completions": sorted(seen_completions),
        "seen_analysis": seen_analysis,
        "seen_digest_completions": sorted(seen_digest_completions or set()),
        "reported_keys": sorted(reported_keys or set()),
        "error_retry_counts": error_retry_counts or {},
        # §97 audit: session-loss tracking persisted.
        "session_loss_streak": dict(session_loss_streak or {}),
        "session_alerted": dict(session_alerted or {}),
        "session_auto_refresh_attempted": dict(session_auto_refresh_attempted or {}),
        "session_contaminated": session_contaminated_serializable,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    # B-223 (2026-05-16, A1.5 Item 7): atomic write via tmp + os.replace +
    # fsync_dir to prevent corrupt-mid-write JSON on watchdog crash.
    # Pre-fix: ``path.write_text(...)`` is non-atomic — crash between open()
    # and final flush leaves a truncated file, ``_load_state`` then returns
    # ``{}`` silently (line 548-564), losing ``error_retry_counts`` +
    # ``session_contaminated`` + ``seen_keys`` history. Matches the
    # LoggerV2._fsync_dir pattern per B-198.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as _f:
        json.dump(payload, _f, ensure_ascii=False, indent=2)
        _f.flush()
        os.fsync(_f.fileno())
    os.replace(tmp_path, path)
    # fsync directory entry so the rename hits stable storage
    try:
        _dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(_dir_fd)
        finally:
            os.close(_dir_fd)
    except OSError:
        pass  # platform doesn't support dir fsync; not a hard failure


# ---------------------------------------------------------------------------
# Auto-digest: update reason CSV + run batch digest
# ---------------------------------------------------------------------------

def _run_auto_digest(run_dir: Path, glm_config: Path, digest_dir: Path, site: Optional[str] = None) -> Optional[str]:
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
        cmd = [python, str(digest_script),
             "--run-dir", str(run_dir),
             "--output", str(digest_dir),
             "--glm-config", str(glm_config),
             "--max-images", "5",
             "--delay-secs", "3.0"]
        if site:
            cmd += ["--site", site]
        r = subprocess.run(
            cmd,
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
    p.add_argument(
        "--notify-completion",
        action="store_true",
        help="Enable standalone COMPLETE push (P79 COMPLETE [...]). "
             "Default off to reduce notification noise.",
    )
    p.add_argument("--once", action="store_true", help="Scan once then exit")
    p.add_argument("--runner-pid", type=int, default=None,
                   help="Runner PID. If set, watchdog auto-exits when runner "
                        "dies and the (single) condition has finalized "
                        "(condition_summary_v2.json present). Without this, "
                        "falls back to extended-idle exit after condition done.")
    p.add_argument("--aggregate-prefix", default="B1_3mode",
                    help="Prefix used for aggregate gallery regeneration (default: B1_3mode)")
    p.add_argument("--reset-state", action="store_true",
                   help="Clear state file before starting (full watchdog state reset)")
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

    if getattr(args, "reset_state", False) and state_file and state_file.exists():
        state_file.unlink()
        print(f"[watchdog] --reset-state: cleared {state_file}")

    # Load persisted state
    saved = _load_state(state_file)
    seen_keys: Set[str] = set(saved.get("seen_keys", []))
    seen_completions: Set[str] = set(saved.get("seen_completions", []))
    seen_analysis: Dict[str, float] = saved.get("seen_analysis", {})
    seen_digest_completions: Set[str] = set(saved.get("seen_digest_completions", []))
    reported_keys: Set[str] = set(saved.get("reported_keys", []))
    error_retry_counts: Dict[str, int] = saved.get("error_retry_counts", {})

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

    # B-222 fix (2026-05-16, A1.5 Item 6): orphan cleanup now requires BOTH
    # mtime > 10min AND no live-runner process. Pre-fix: 10min mtime alone is
    # not safe for long episodes (image render hang / browser stuck) — a
    # runner still actively writing artifacts could have its files nuked.
    # Live-runner detection via `pgrep run_experiment.*${run_dir basename}`:
    # if any active runner targets the same run_dir, skip all pruning for
    # this watchdog cycle. The mtime guard is kept as secondary defence.
    _orphan_count = 0
    _orphan_cutoff = time.time() - 10 * 60
    # Step 1: probe for live runner attached to this run_dir
    try:
        _live_runner = subprocess.run(
            ["pgrep", "-fa", f"run_experiment.*{run_dir.name}"],
            capture_output=True, text=True, timeout=5,
        )
        _has_live_runner = _live_runner.returncode == 0 and _live_runner.stdout.strip()
    except Exception:
        _has_live_runner = False  # pgrep unavailable → fall back to mtime-only
    if _has_live_runner:
        print(
            "[watchdog] live runner detected for run_dir="
            f"{run_dir.name} — skipping orphan cleanup this cycle "
            "(B-222 guard: prevents deletion of artifacts currently being written)"
        )
    else:
        _cond_dirs_to_scan = (
            [run_dir / args.condition] if args.condition
            else [p for p in run_dir.iterdir() if p.is_dir() and p.name not in _EXCLUDED_DIRS]
        )
        for _cdir in _cond_dirs_to_scan:
            _art_root = _cdir / "artifacts"
            _ep_root = _cdir / "episodes"
            # Orphan artifact directories
            if _art_root.exists():
                for _art in _art_root.iterdir():
                    if not _art.is_dir():
                        continue
                    if (_ep_root / f"{_art.name}_summary_v2.json").exists():
                        continue
                    if _art.stat().st_mtime > _orphan_cutoff:
                        continue
                    # B-222 secondary guard: check for per-episode .in_progress marker.
                    # Runner writes this when it starts an episode + removes on completion;
                    # if present, episode is mid-flight even if mtime suggests stale.
                    if (_art / ".in_progress").exists():
                        continue
                    shutil.rmtree(_art)
                    _orphan_count += 1
            # Orphan steps files (steps JSONL without summary)
            if _ep_root.exists():
                for _sf in _ep_root.glob("*_steps_v2.jsonl"):
                    _summary = _ep_root / _sf.name.replace("_steps_v2.jsonl", "_summary_v2.json")
                    if _summary.exists():
                        continue
                    if _sf.stat().st_mtime > _orphan_cutoff:
                        continue
                    # B-222: same per-episode marker check (marker lives in artifacts/<task_id>/)
                    _ep_stem = _sf.name.replace("_steps_v2.jsonl", "")
                    if (_cdir / "artifacts" / _ep_stem / ".in_progress").exists():
                        continue
                    _sf.unlink()
                    _orphan_count += 1
        if _orphan_count:
            print(f"[watchdog] Pruned {_orphan_count} orphan item(s) (artifact dirs / steps files without summary)")

    # Session-loss tracking: per-site streak counters.
    # §97 audit: restore from persisted state so watchdog restarts don't
    # lose contaminated-episode tracking (was: in-memory only → restart →
    # missed cleanup of NOT-LOGGED-IN episodes when login restored).
    session_loss_streak: Dict[str, int] = defaultdict(
        int, saved.get("session_loss_streak", {})
    )
    session_alerted: Dict[str, bool] = defaultdict(
        bool, saved.get("session_alerted", {})
    )
    session_auto_refresh_attempted: Dict[str, bool] = defaultdict(
        bool, saved.get("session_auto_refresh_attempted", {})
    )
    # Contaminated episodes to auto-clean when login is restored:
    # {site: [(condition_id, condition_dir, task_id, site, key), ...]}
    # Re-Path the condition_dir on load (was stringified for JSON).
    session_contaminated: Dict[str, List[Tuple[str, Path, int, str, str]]] = defaultdict(list)
    for _site_key, _items in (saved.get("session_contaminated") or {}).items():
        for _item in _items:
            if isinstance(_item, (list, tuple)) and len(_item) == 5:
                _cid, _cdir_str, _tid, _site, _key = _item
                session_contaminated[_site_key].append(
                    (str(_cid), Path(_cdir_str), int(_tid), str(_site), str(_key))
                )
    if session_loss_streak or session_contaminated:
        print(
            f"[watchdog] Restored session state: "
            f"loss_streak={dict(session_loss_streak)}, "
            f"contaminated_sites={list(session_contaminated.keys())}"
        )

    # Closure that captures current session state — every _save_state caller
    # in this function should use this instead so session_* fields persist.
    def _persist_state() -> None:
        _save_state(
            state_file, seen_keys, seen_completions, seen_analysis,
            seen_digest_completions, reported_keys, error_retry_counts,
            session_loss_streak=dict(session_loss_streak),
            session_alerted=dict(session_alerted),
            session_auto_refresh_attempted=dict(session_auto_refresh_attempted),
            session_contaminated=dict(session_contaminated),
        )

    # Timers
    last_new_episode_ts: float = time.time()
    last_report_ts: float = 0.0  # 0 → trigger initial report immediately after bootstrap
    idle_alerted: bool = False
    # (recent episodes computed from seen_keys - reported_keys at report time)
    idle_alert_secs = max(60, args.idle_alert_mins * 60)
    report_interval_secs = max(60, args.report_interval_mins * 60)

    # Bootstrap: scan existing analysis files and completions without alerting
    pruned_completions = _prune_stale_condition_completions(run_dir, args.condition, seen_completions)
    if pruned_completions > 0:
        print(f"[watchdog] Pruned {pruned_completions} stale completions (missing condition_summary_v2.json)")
        _persist_state()

    _check_analysis_outputs(run_dir, seen_analysis)
    _check_condition_completions(run_dir, args.condition, seen_completions, condition_mode_cache)

    print(
        f"[watchdog] run_id={run_id} condition={args.condition or '*'} "
        f"poll={args.poll_secs}s report_every={args.report_interval_mins}min "
        f"idle_alert={args.idle_alert_mins}min"
    )

    # Manual immediate-status trigger (signal):
    #   kill -USR1 <watchdog_pid>
    force_report_once = False

    def _on_force_report_signal(sig_num: int, _frame: Any) -> None:
        nonlocal force_report_once
        force_report_once = True
        try:
            sig_name = signal.Signals(sig_num).name
        except Exception:
            sig_name = str(sig_num)
        print(f"[watchdog][MANUAL] Received {sig_name}; scheduling immediate status cycle.")

    try:
        signal.signal(signal.SIGUSR1, _on_force_report_signal)
    except Exception:
        # Some environments may not support SIGUSR1 registration.
        pass

    while True:
        now = time.time()

        # --- 1. Scan new episodes ---
        summaries = _scan_summaries(run_dir, args.condition)
        new_paths = [p for p in summaries if _episode_key(p) not in seen_keys]

        if new_paths:
            last_new_episode_ts = now
            idle_alerted = False
            auto_retry_batch: List[str] = []

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

                # Auto-cleanup: delete error episodes so runner can retry.
                # Covers both benchmark noise errors AND code bugs.
                # All error types get max retries to avoid infinite loops
                # (some benchmark noise errors like ERR_ABORTED are persistent).
                # Only when the condition is still running (no condition_summary yet);
                # once aggregated, deleting episodes would create inconsistency.
                MAX_CODE_BUG_RETRIES = 2
                MAX_NOISE_RETRIES = 3
                condition_completed = (condition_dir / "condition_summary_v2.json").exists()
                retry_key = f"{condition_id}/{site}_task_{task_id}"
                retries_so_far = error_retry_counts.get(retry_key, 0)
                is_noise = reason.startswith("error(") and reason != "error(evaluator)" and reason != "error(code_bug)"
                can_retry = (
                    reason.startswith("error(")
                    and reason != "error(evaluator)"
                    and not condition_completed
                    and (
                        (reason == "error(code_bug)" and retries_so_far < MAX_CODE_BUG_RETRIES)
                        or (is_noise and retries_so_far < MAX_NOISE_RETRIES)
                    )
                )
                if reason.startswith("error(") and not condition_completed and not can_retry:
                    # Persistent code_bug — exhausted retries, notify and keep
                    print(
                        f"[watchdog][PERSISTENT-ERROR] task {task_id} ({reason}) "
                        f"failed {retries_so_far} retries, giving up"
                    )
                    if args.ntfy_topic:
                        _post_ntfy(
                            args.ntfy_topic,
                            f"P79 PERSISTENT ERROR task {task_id}",
                            f"run_id={run_id}\n{condition_id} task {task_id}\n"
                            f"{reason} — failed after {retries_so_far} retries, needs manual fix",
                            priority="high",
                        )
                if can_retry:
                    error_retry_counts[retry_key] = retries_so_far + 1
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
                    # 4. Clean digest records (keep data consistent with deleted episode)
                    _digest_dir = args.digest_dir or (run_dir / "analysis" / "digest")
                    purged = _purge_digest_records(_digest_dir, condition_id, task_id, obs_mode)
                    max_for_type = MAX_NOISE_RETRIES if is_noise else MAX_CODE_BUG_RETRIES
                    print(
                        f"[watchdog][AUTO-RETRY] deleted error episode: "
                        f"task {task_id} ({reason}) retry {retries_so_far + 1}/{max_for_type}"
                        + (f" (+{purged} digest records)" if purged else "")
                    )
                    auto_retry_batch.append(
                        f"task {task_id} ({reason}) retry {retries_so_far + 1}/{max_for_type}"
                    )
                    _persist_state()

                    # B-314 (A1.17 Option K Trajectory Event Log hook,
                    # 2026-05-16): log auto-clean event to trajectory_events.jsonl
                    # so analysis aggregator can build per-episode
                    # `had_auth_clear` / `prior_clear_count` covariates for
                    # paper §4 GLMM bias absorption (Tier 1 stack (1)-gemini).
                    # Generalizes P1-5-B reset event tracking to auth-loss /
                    # noise-clean class (user cross-talk insight 2026-05-16).
                    try:
                        from p79.experiment.logger_v2 import log_trajectory_event_external
                        log_trajectory_event_external(
                            condition_dir=condition_dir,
                            event_type="task_auto_cleared",
                            task_index=task_id,
                            metadata={
                                "reason": reason,
                                "retry_attempt": retries_so_far + 1,
                                "max_retries": max_for_type,
                                "is_noise": is_noise,
                                "is_auth_loss": bool(reason.startswith("error(session") or reason.startswith("error(auth")),
                                "purged_digest_records": purged,
                            },
                        )
                    except Exception as _trajectory_log_exc:
                        # Non-fatal — trajectory event log is a paper-§4 covariate
                        # enrichment, not a blocking step. Best-effort.
                        print(
                            f"[watchdog][trajectory-event][warn] failed to log "
                            f"task_auto_cleared event for task {task_id}: {_trajectory_log_exc}"
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
                    # Track this episode as contaminated (will be cleaned when login restored)
                    session_contaminated[site].append(
                        (condition_id, condition_dir, task_id, site, key)
                    )
                    print(
                        f"[watchdog][SESSION] {site} task {task_id} "
                        f"NOT LOGGED IN (streak={streak})"
                    )
                    if streak >= _SESSION_ALERT_THRESHOLD and not session_alerted[site]:
                        session_alerted[site] = True
                        body = (
                            f"run_id={run_id}\n"
                            f"{site}: {streak} consecutive tasks without login!\n"
                            f"正在尝试自动刷新 auth..."
                        )
                        print(f"[watchdog][SESSION] ALERT: {body}")
                        if args.ntfy_topic:
                            _post_ntfy(
                                args.ntfy_topic,
                                f"P79 SESSION LOST [{site}]",
                                body,
                                priority="urgent",
                            )
                    # Attempt auto-refresh once per loss wave
                    if streak >= _SESSION_ALERT_THRESHOLD and not session_auto_refresh_attempted[site]:
                        session_auto_refresh_attempted[site] = True
                        print(f"[watchdog][SESSION] attempting auto-refresh for {site}...")
                        _bm = "webarena" if any(p == "webarena" for p in run_dir.parts) else ""
                        if _auto_refresh_auth(site, benchmark=_bm):
                            print(f"[watchdog][SESSION] {site} auth refreshed — next tasks will use new cookies")
                        else:
                            print(f"[watchdog][SESSION][warn] {site} auto-refresh failed — manual intervention needed")
                elif session_ok is True:
                    was_streak = session_loss_streak[site]
                    if was_streak > 0:
                        print(
                            f"[watchdog][SESSION] {site} login restored "
                            f"(was streak={was_streak})"
                        )
                        # Auto-clean all contaminated episodes from this loss wave
                        contaminated = session_contaminated.pop(site, [])
                        if contaminated:
                            _ddir = run_dir / "analysis" / "digest"
                            cleaned = 0
                            for (cond_id, cond_dir, ctask_id, csite, ckey) in contaminated:
                                # Delete episode files
                                for p in [
                                    cond_dir / "episodes" / f"{csite}_task_{ctask_id}_summary_v2.json",
                                    cond_dir / "episodes" / f"{csite}_task_{ctask_id}_steps_v2.jsonl",
                                ]:
                                    try:
                                        if p.exists(): p.unlink()
                                    except OSError:
                                        pass
                                cart = cond_dir / "artifacts" / f"{csite}_task_{ctask_id}"
                                try:
                                    if cart.exists(): shutil.rmtree(cart)
                                except OSError:
                                    pass
                                # Clean digest
                                cmode = _get_observation_mode(cond_dir, condition_mode_cache)
                                _purge_digest_records(_ddir, cond_id, ctask_id, cmode)
                                # Remove from in-memory tracking
                                all_records[:] = [
                                    r for r in all_records
                                    if not (r.condition_id == cond_id and r.task_id == ctask_id)
                                ]
                                seen_keys.discard(ckey)
                                reported_keys.discard(ckey)
                                cleaned += 1
                            print(f"[watchdog][SESSION] {site} auto-cleaned {cleaned} NOT-LOGGED-IN episodes")
                            _persist_state()
                            if args.ntfy_topic:
                                _post_ntfy(
                                    args.ntfy_topic,
                                    f"P79 SESSION RESTORED [{site}]",
                                    f"run_id={run_id}\n{site}: login restored\n"
                                    f"auto-cleaned {cleaned} NOT-LOGGED-IN episodes",
                                    priority="default",
                                )
                    session_loss_streak[site] = 0
                    session_alerted[site] = False
                    session_auto_refresh_attempted[site] = False

                # Per-condition cumulative
                cond_all = [r for r in all_records if r.condition_id == condition_id]
                cond_total = len(cond_all)
                cond_succ = sum(1 for r in cond_all if r.success)
                cond_rate = (cond_succ / cond_total) if cond_total else 0.0

                print(
                    f"[watchdog] [{obs_mode}] {condition_id} task={task_id:>3d} "
                    f"{'OK' if rec.success else reason:<10s} "
                    f"succ={cond_succ}/{cond_total} ({cond_rate:.1%})"
                )

            # Batch-send AUTO-RETRY notifications (avoid per-task spam)
            if auto_retry_batch and args.ntfy_topic:
                _post_ntfy(
                    args.ntfy_topic,
                    f"P79 AUTO-RETRY ({len(auto_retry_batch)} tasks)",
                    f"run_id={run_id}\n" + "\n".join(auto_retry_batch[:20]),
                )

            _persist_state()

        # --- 1.5. Prune stale completions (queue may delete condition_summary) ---
        _prune_stale_condition_completions(run_dir, args.condition, seen_completions)

        # --- 2. Periodic/manual status report ---
        manual_report_now = force_report_once
        if manual_report_now:
            force_report_once = False

        if ((now - last_report_ts) >= report_interval_secs and all_records) or manual_report_now:
            report = _build_status_report(all_records, condition_mode_cache, seen_completions, run_id)
            if report:
                print(f"[watchdog][REPORT]\n{report}")
            elif manual_report_now:
                print(f"[watchdog][REPORT]\nrun_id={run_id}\n(manual) 当前无运行中 condition")

            # Auto-digest: update reason CSV + run batch digest
            digest_status = None
            if args.glm_config:
                digest_dir = args.digest_dir or (run_dir / "analysis" / "digest")
                # Infer site: pass --site only when all episodes belong to a single site
                sites = {r.site for r in all_records}
                digest_site = sites.pop() if len(sites) == 1 else None
                digest_status = _run_auto_digest(run_dir, args.glm_config, digest_dir, site=digest_site)

                # Check if digest is newly complete for any mode
                newly_done = _check_digest_completions(
                    digest_dir, all_records, seen_completions, seen_digest_completions,
                )
                for mode, digested, expected in newly_done:
                    info = f"[{mode}] digest 完成: {digested}/{expected}"
                    print(f"[watchdog][DIGEST] {info}")
                    _persist_state()

            # Auto-annotate screenshots then regenerate gallery HTML
            annotate_status = _annotate_screenshots(run_dir, args.condition)
            # Keep primary gallery as full run view; avoid overwriting with single-condition subset.
            gallery_status = _regenerate_gallery(run_dir, None, args.aggregate_prefix)

            # Run analysis scripts (results fed into consolidated notification)
            new_analysis = _check_analysis_outputs(run_dir, seen_analysis)
            analysis_names = []
            for name, path in new_analysis:
                print(f"[watchdog][ANALYSIS] run_id={run_id}\n分析脚本完成: {name}\n输出: {path.relative_to(run_dir)}")
                analysis_names.append(name)
                _persist_state()

            # --- Build consolidated periodic notification ---
            if args.ntfy_topic and (report or manual_report_now):
                if report:
                    parts = [report]
                else:
                    parts = [f"run_id={run_id}", "(manual) 当前无运行中 condition"]

                # Recent tasks: episodes in seen_keys but not yet reported
                unreported_keys = seen_keys - reported_keys
                if report and unreported_keys:
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
            _persist_state()
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
            # §97 audit: reset error_retry_counts for all tasks under this
            # condition. Was: counts persisted forever → re-running a task
            # in a new condition could exhaust retries from prior history.
            # Format: retry_key = "{condition_id}/{site}_task_{task_id}"
            _reset_keys = [k for k in error_retry_counts if k.startswith(f"{cid}/")]
            if _reset_keys:
                for _k in _reset_keys:
                    error_retry_counts.pop(_k, None)
                print(
                    f"[watchdog][RETRY-RESET] {cid} completed → cleared "
                    f"{len(_reset_keys)} retry counters"
                )
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
            if args.ntfy_topic and args.notify_completion:
                _post_ntfy(args.ntfy_topic, f"P79 COMPLETE [{mode}/{cid}]", body, priority="high")

            # Auto-run analysis pipeline after condition completion
            analysis_status = _run_post_condition_analysis(run_dir)
            annotate_status = _annotate_screenshots(run_dir, cid)
            # Keep primary gallery as full run view; avoid overwriting with single-condition subset.
            gallery_status = _regenerate_gallery(run_dir, None, args.aggregate_prefix)
            # Cross-run analysis: triggers compare_b0_b1 when sibling baseline run
            # exists for same site, and aggregate_cross_site when ≥2 sites under
            # same baseline have data. Returns None if nothing was triggered.
            cross_run_status = _run_cross_run_analysis(run_dir)
            # Regenerate paper figures (cls/red 4-mode oracle, drop-one bar, etc.)
            figures_status = _regenerate_paper_figures()
            if args.ntfy_topic:
                body = (
                    f"run_id={run_id}\nanalysis: {analysis_status}\n"
                    f"annotate: {annotate_status}\ngallery: {gallery_status}"
                )
                if cross_run_status:
                    body += f"\ncross_run: {cross_run_status}"
                body += f"\nfigures: {figures_status}"
                _post_ntfy(args.ntfy_topic, f"P79 POST-ANALYSIS [{cid}]", body)

            _persist_state()

        if args.once:
            break

        # ---- Self-exit when work is done (avoids init-orphan idle loops) ----
        # Only meaningful in single-condition mode — multi-condition watchdog
        # waits for all conditions, harder to bound generally.
        if args.condition:
            cond_done = (run_dir / args.condition / "condition_summary_v2.json").exists()
            if cond_done:
                if args.runner_pid is not None:
                    # Path A: explicit runner PID — exit as soon as runner dies.
                    try:
                        os.kill(args.runner_pid, 0)
                        runner_alive = True
                    except ProcessLookupError:
                        runner_alive = False
                    except PermissionError:
                        # Process exists but owned by another uid — still alive.
                        runner_alive = True
                    except OSError:
                        runner_alive = True  # err on safe side
                    if not runner_alive:
                        print(
                            f"[watchdog] condition {args.condition} complete + "
                            f"runner pid={args.runner_pid} dead → exiting"
                        )
                        _persist_state()
                        break
                else:
                    # Path B (legacy launchers without --runner-pid): exit if
                    # no new episodes for ≥ idle_alert_secs after summary written.
                    idle_secs = time.time() - last_new_episode_ts
                    if idle_secs >= idle_alert_secs:
                        print(
                            f"[watchdog] condition {args.condition} complete + "
                            f"idle {int(idle_secs/60)}min → exiting"
                        )
                        _persist_state()
                        break

        time.sleep(max(1, args.poll_secs))

    # Final summary
    if all_records:
        report = _build_status_report(all_records, condition_mode_cache, seen_completions, run_id)
        print(f"\n[watchdog][FINAL]\n{report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
