#!/usr/bin/env python3
"""B-389 (A1.15 C2 Aggregator, 2026-05-16) — Option K Tier 1 covariate emission.

Reads `trajectory_events.jsonl` + `condition_summary_v2.json` per condition_dir
and emits per-episode covariates for paper §4 GLMM fixed-effect adjustment.

Covariates emitted per (condition_id, site, task_id):
  - is_after_reset (bool): any `reset_post_interrupt` event ts < episode start ts
  - had_auth_clear (bool): any `task_auto_cleared` event for this task w/
    metadata.is_auth_loss=True (either retry path B-314 or session-wave B-384)
  - had_finalize_race_clear (bool): episode IS in condition_summary AND task_id
    has any `task_auto_cleared` event in trajectory_events (post-hoc race
    detection per B-385 — denominator counted the episode but auto-clean
    also touched it → finalize race race window did trigger)
  - prior_event_count (int): count of events with ts < episode start ts
  - cleared_in_session_wave (bool): any task_auto_cleared event for this
    task with metadata.cleared_in_session_wave=True (passed-through from B-384)
  - session_wave_size (Optional[int]): if cleared_in_session_wave, the wave_size
    metadata from the event (helpful for cluster-correlation analysis)

Output: `<condition_dir>/analysis/trajectory_covariates.jsonl` — one row per
episode in condition_summary's episode list. CSV format also written for
downstream pandas / R consumption.

Usage:
    python scripts/analysis/aggregate_trajectory_covariates.py --run-dir results/visualwebarena/phase1/B0_3mode_classifieds_20260413
    python scripts/analysis/aggregate_trajectory_covariates.py --run-dir <run> --condition <cid>

This is the stub form of Tier 1 (iii) — paper §4 GLMM consumer is post-data.
Aggregator output schema is locked here; GLMM script reads from this output.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Excluded directory names within run_dir (mirror experiment_watchdog.py)
_EXCLUDED_DIRS = {"analysis", ".git", "gallery_data", "task_configs"}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Best-effort JSONL read. Skip corrupt lines silently (B-196 pattern)."""
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return out


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _read_episode_summary(path: Path) -> Optional[Dict[str, Any]]:
    """Read per-episode summary_v2 JSON. Returns dict with `wallclock_start`
    if available (used for ordering events relative to episode)."""
    return _read_json(path)


def _episode_start_ts(summary: Dict[str, Any]) -> Optional[str]:
    """Extract episode wallclock-start from summary. Schema v2 uses
    `created_at` or `wallclock_start`; fall back to `start_ts`."""
    for key in ("wallclock_start", "started_at", "created_at", "start_ts"):
        if key in summary and summary[key]:
            return str(summary[key])
    return None


def compute_episode_covariates(
    condition_dir: Path,
) -> List[Dict[str, Any]]:
    """Compute per-episode covariates for one condition_dir. Returns list of
    dicts ready for JSONL/CSV emission."""
    trajectory_events = _read_jsonl(condition_dir / "trajectory_events.jsonl")
    cond_summary = _read_json(condition_dir / "condition_summary_v2.json")
    if cond_summary is None:
        return []

    condition_id = condition_dir.name

    # Identify all episodes in this condition_dir. Prefer condition_summary's
    # explicit episode list; fall back to scanning episodes/ for summary_v2 files.
    # Schema variants (B-302 + A1.8 audit):
    #   - "episode_summaries": list[dict]  (v2 paper-grade schema, full episode meta)
    #   - "episodes":          list[dict]  (alt v2 layout, full episode meta)
    #   - "episodes":          int         (LEGACY pre-B-302: count only)
    #     → fall back to filesystem scan in this case
    episodes: List[Dict[str, Any]] = []
    _es = cond_summary.get("episode_summaries")
    _ep = cond_summary.get("episodes")
    if isinstance(_es, list):
        ep_meta_list: List[Any] = _es
    elif isinstance(_ep, list):
        ep_meta_list = _ep
    else:
        ep_meta_list = []  # legacy int count, etc.
    if ep_meta_list:
        for ep in ep_meta_list:
            if not isinstance(ep, dict):
                continue
            site = ep.get("site") or ep.get("benchmark_site") or "unknown"
            task_id = ep.get("task_id")
            if task_id is None:
                continue
            episodes.append({
                "site": str(site),
                "task_id": int(task_id),
                "summary": ep,
            })
    else:
        # Fallback: scan episodes/<site>_task_<id>_summary_v2.json
        ep_dir = condition_dir / "episodes"
        if ep_dir.exists():
            for f in ep_dir.glob("*_summary_v2.json"):
                # Pattern: <site>_task_<task_id>_summary_v2.json
                stem = f.stem.replace("_summary_v2", "")
                parts = stem.rsplit("_task_", 1)
                if len(parts) != 2:
                    continue
                try:
                    task_id = int(parts[1])
                except ValueError:
                    continue
                ep_summary = _read_episode_summary(f) or {}
                episodes.append({
                    "site": parts[0],
                    "task_id": task_id,
                    "summary": ep_summary,
                })

    # Build event lookup tables keyed by task_index.
    events_by_task: Dict[int, List[Dict[str, Any]]] = {}
    cell_level_events: List[Dict[str, Any]] = []  # task_index is None
    for ev in trajectory_events:
        ti = ev.get("task_index")
        if ti is None:
            cell_level_events.append(ev)
        else:
            try:
                ti = int(ti)
            except (ValueError, TypeError):
                continue
            events_by_task.setdefault(ti, []).append(ev)

    # cell-level reset_post_interrupt events (used for is_after_reset)
    reset_events = [
        ev for ev in cell_level_events
        if ev.get("event_type") == "reset_post_interrupt"
    ]

    # Set of task_ids that appear in condition_summary (used for race detection)
    task_ids_in_summary: Set[int] = {ep["task_id"] for ep in episodes}

    rows: List[Dict[str, Any]] = []
    for ep in episodes:
        task_id = ep["task_id"]
        site = ep["site"]
        ep_summary = ep["summary"]
        ep_start = _episode_start_ts(ep_summary)

        task_events = events_by_task.get(task_id, [])

        # is_after_reset: any reset_post_interrupt happened before ep start ts.
        # If ep_start ts unavailable, conservatively True if ANY reset event present.
        if reset_events:
            if ep_start:
                is_after_reset = any(
                    str(rev.get("wallclock_ts", "")) < ep_start
                    for rev in reset_events
                )
            else:
                is_after_reset = True
        else:
            is_after_reset = False

        # had_auth_clear: any task_auto_cleared for this task w/ is_auth_loss=True
        had_auth_clear = any(
            ev.get("event_type") == "task_auto_cleared"
            and bool((ev.get("metadata") or {}).get("is_auth_loss"))
            for ev in task_events
        )

        # had_finalize_race_clear: per B-385 post-hoc detection — task IS in
        # condition_summary AND has task_auto_cleared event. Means the auto-clean
        # touched the same task that runner's final aggregation counted →
        # race-window denominator drift.
        any_auto_clear = any(
            ev.get("event_type") == "task_auto_cleared" for ev in task_events
        )
        had_finalize_race_clear = (
            any_auto_clear and (task_id in task_ids_in_summary)
        )

        # cleared_in_session_wave: passed through from B-384 metadata.
        cleared_in_session_wave = any(
            ev.get("event_type") == "task_auto_cleared"
            and bool((ev.get("metadata") or {}).get("cleared_in_session_wave"))
            for ev in task_events
        )
        # session_wave_size: from first matching session-wave event's metadata.
        session_wave_size: Optional[int] = None
        for ev in task_events:
            if (
                ev.get("event_type") == "task_auto_cleared"
                and bool((ev.get("metadata") or {}).get("cleared_in_session_wave"))
            ):
                ws = (ev.get("metadata") or {}).get("wave_size")
                if isinstance(ws, int):
                    session_wave_size = ws
                    break

        # prior_event_count: events occurring before this episode (by ts).
        if ep_start:
            prior_event_count = sum(
                1 for ev in trajectory_events
                if str(ev.get("wallclock_ts", "")) < ep_start
            )
        else:
            prior_event_count = 0

        rows.append({
            "condition_id": condition_id,
            "site": site,
            "task_id": task_id,
            "is_after_reset": is_after_reset,
            "had_auth_clear": had_auth_clear,
            "had_finalize_race_clear": had_finalize_race_clear,
            "cleared_in_session_wave": cleared_in_session_wave,
            "session_wave_size": session_wave_size,
            "prior_event_count": prior_event_count,
            "n_task_events": len(task_events),
            "ep_wallclock_start": ep_start,
        })

    return rows


def emit_covariates(condition_dir: Path) -> Tuple[int, Path, Path]:
    """Compute + write trajectory_covariates.{jsonl,csv} under
    condition_dir/analysis/. Returns (rows_written, jsonl_path, csv_path)."""
    rows = compute_episode_covariates(condition_dir)
    out_dir = condition_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "trajectory_covariates.jsonl"
    csv_path = out_dir / "trajectory_covariates.csv"
    # JSONL
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    # CSV (paper §4 GLMM friendly)
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = [
            "condition_id", "site", "task_id",
            "is_after_reset", "had_auth_clear", "had_finalize_race_clear",
            "cleared_in_session_wave", "session_wave_size",
            "prior_event_count", "n_task_events", "ep_wallclock_start",
        ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows), jsonl_path, csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Option K covariate aggregator (B-389)")
    parser.add_argument("--run-dir", required=True, help="Run directory (results/<bench>/<phase>/<run>)")
    parser.add_argument("--condition", default=None, help="Filter to specific condition_id")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"ERROR: run_dir not found: {run_dir}", file=sys.stderr)
        return 1

    if args.condition:
        cond_dirs = [run_dir / args.condition]
    else:
        cond_dirs = [
            p for p in run_dir.iterdir()
            if p.is_dir() and p.name not in _EXCLUDED_DIRS
        ]

    total_rows = 0
    for cdir in sorted(cond_dirs):
        if not (cdir / "condition_summary_v2.json").exists():
            print(f"[trajectory-covariates] skip {cdir.name}: no condition_summary_v2.json")
            continue
        n, jp, cp = emit_covariates(cdir)
        total_rows += n
        print(f"[trajectory-covariates] {cdir.name}: {n} episodes → {jp.relative_to(run_dir)}")

    print(f"[trajectory-covariates] DONE total_rows={total_rows} across {len(cond_dirs)} conditions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
