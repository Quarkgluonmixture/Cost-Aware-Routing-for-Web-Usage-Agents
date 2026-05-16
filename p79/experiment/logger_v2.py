from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def _fsync_dir(directory: Path) -> None:
    """B-198 (/stress A1.4b-ii Claude D6): fsync the directory entry after
    `os.replace` so the rename hits stable storage on the next journal
    cycle. Pre-fix: `os.replace` is atomic at the inode level, but the
    directory entry update sits in ext4 journal up to ~30s before flush.
    DGX crash between rename + flush → reboot sees pre-rename state, paper
    runs may silently lose just-written summaries. Best-effort: on platforms
    where dir fsync raises (NFS, FAT), swallow the error — atomicity is
    still better than nothing."""
    try:
        fd = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass  # platform doesn't support dir fsync; not a hard failure


class LoggerV2:
    def __init__(self, condition_dir: Path):
        self.condition_dir = condition_dir
        self.episodes_dir = self.condition_dir / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        # B-289 fix (2026-05-16, A1.8): fsync the parent of `episodes_dir` so
        # the directory entry hits stable storage at construction time. Pre-fix
        # the `mkdir(parents=True)` left the entry in ext4 journal up to ~30s;
        # a Spark crash between mkdir and the first write_step would leave the
        # parent dir entry unflushed → reboot sees no episodes/ → subsequent
        # writes go to a recreated dir but run_meta references the missing
        # original path. Same B-198 lineage as write_episode_summary.
        _fsync_dir(self.condition_dir)
        _fsync_dir(self.episodes_dir)

    def write_condition_meta(self, meta: Dict[str, Any]) -> None:
        self.condition_dir.mkdir(parents=True, exist_ok=True)
        path = self.condition_dir / "condition_meta.json"
        tmp_path = path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)  # atomic on same filesystem
        _fsync_dir(path.parent)  # B-198 flush dir entry to disk

    def step_log_path(self, site: str, task_id: int) -> Path:
        return self.episodes_dir / f"{site}_task_{task_id}_steps_v2.jsonl"

    def summary_path(self, site: str, task_id: int) -> Path:
        return self.episodes_dir / f"{site}_task_{task_id}_summary_v2.json"

    def write_step(self, site: str, task_id: int, record: Dict[str, Any]) -> None:
        path = self.step_log_path(site, task_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def write_episode_summary(self, site: str, task_id: int, summary: Dict[str, Any]) -> None:
        path = self.summary_path(site, task_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)  # atomic on same filesystem
        _fsync_dir(path.parent)  # B-198 flush dir entry to disk

    def condition_summary_path(self) -> Path:
        return self.condition_dir / "condition_summary_v2.json"

    def write_condition_summary(self, payload: Dict[str, Any]) -> None:
        self.condition_dir.mkdir(parents=True, exist_ok=True)
        path = self.condition_summary_path()
        tmp_path = path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)  # atomic on same filesystem
        _fsync_dir(path.parent)  # B-198 flush dir entry to disk

    # B-313 (A1.17 P1-5-B Tier 1 α' + auth-loss generalization, 2026-05-16):
    # Option K Trajectory Event Log — unified schema covering ALL mid-trajectory
    # state-perturbing events (reset interrupts, auth-loss + auto-clear, watchdog
    # auth-refresh, runner restart). User cross-talk insight 2026-05-16: P1-5-B
    # reset-discontinuity and auth-loss/auto-clear are isomorphic bug classes —
    # both cause JSONL ↔ site state inconsistency, just in opposite directions.
    # Tier 1 analysis-layer fixes ((1)-gemini GLMM / (4)-gemini Fisher / §3 reframe)
    # generalize to both at zero additional cost by absorbing `is_after_reset`
    # AND `had_auth_clear` covariates.
    #
    # Schema: append-only JSONL at condition_dir/trajectory_events.jsonl. Each
    # event = single line {event_type, task_index, wallclock_ts, metadata}.
    # event_type values: "reset_post_interrupt" / "auth_clear_task" /
    # "auth_refresh_no_clear" / "runner_restart" / "watchdog_intervention".
    # task_index = episode/task index at event time; None for cell-level events.
    # metadata = event-specific dict (e.g. reset rc, auth_refresh_method,
    # cleared_task_count). Aggregator emits per-episode `is_after_reset` /
    # `had_auth_clear` / `prior_event_count` columns for GLMM covariate
    # adjustment in paper §4.
    def log_trajectory_event(
        self,
        event_type: str,
        task_index: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append a trajectory perturbation event to trajectory_events.jsonl.

        See Option K Trajectory Event Log spec (B-313 A1.17). Used by reset gate,
        watchdog auth-clear, runner restart paths. Append-only, atomic append +
        fsync. No batching — each event is a separate line, recoverable from
        partial writes via the dedup primitive in `io_utils.read_jsonl_dedup`.
        """
        if metadata is None:
            metadata = {}
        event = {
            "event_type": event_type,
            "task_index": task_index,
            "wallclock_ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "metadata": metadata,
        }
        path = self.condition_dir / "trajectory_events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())


# B-313 module-level helper for shell-side / out-of-band callers (reset gate
# bash heredoc, watchdog Python script) that may not have a LoggerV2 instance.
# Constructs a transient logger pointing at the given condition_dir.
def log_trajectory_event_external(
    condition_dir: Path,
    event_type: str,
    task_index: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Out-of-band trajectory event logger for callers without a LoggerV2 instance.

    Used by:
      - scripts/queues/_lib_paper_grade_gates.sh:reset_and_auth_gate (reset event)
      - scripts/maintenance/experiment_watchdog.py (auth-clear / refresh)
      - phase1a_relaunch_missing.sh future hook (resume event)

    If condition_dir doesn't exist yet (e.g., reset event before runner creates
    dir), no-op silently — runner will create dir on first write_step and any
    later events will land. This degrades gracefully and never raises in the
    hot path.
    """
    if not condition_dir.exists():
        return
    LoggerV2(condition_dir).log_trajectory_event(event_type, task_index, metadata)


def write_run_summary_atomic(run_summary_path: Path, payload: Dict[str, Any]) -> None:
    """B-331 (/stress A1.9 Mode B F6 OOB, 2026-05-16): atomic + fsync write
    for run_summary_v2.json. Pre-fix runner/main.py:710 used plain
    `json.dump` → crash mid-write → run_summary truncated. condition_summary
    already uses the LoggerV2 atomic+fsync chain (line 86-91); this helper
    extends the same durability contract to the run-level summary.

    Standalone function (not a LoggerV2 method) because run_summary lives at
    output_root, not at any single condition_dir.
    """
    run_summary_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = run_summary_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, run_summary_path)
    _fsync_dir(run_summary_path.parent)
