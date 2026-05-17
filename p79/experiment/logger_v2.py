from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

# B-736 fix (/stress A1.8 cold-start P1-2-A* Claude OOB, 2026-05-17): POSIX
# advisory file lock for concurrent-append paths (`write_step` +
# `log_trajectory_event` + `merge_staging_trajectory_events`). Pre-fix two
# callers (in-process runner + out-of-band reset gate bash via
# `log_trajectory_event_external` + watchdog python) could write to the same
# JSONL concurrently — kernel-level append atomicity (`O_APPEND`) is only
# guaranteed for writes ≤ `PIPE_BUF` (4096 bytes on Linux). Serialized
# trajectory event JSON lines with `metadata` containing wave_task lists /
# purged_digest_records can easily exceed 4KB → torn writes → corrupt JSONL
# → dedup can't see it, aggregator `prior_event_count` miscounts → paper §4
# Option K covariate analysis silently wrong.
#
# Cross-platform: `fcntl` is POSIX-only (Linux + macOS). Windows would need
# `msvcrt.locking`. Phase 1a target is DGX Spark (Linux aarch64) +
# Condenser A100 VM (Linux) — fcntl is the right primitive. Imported lazily
# in a try/except so non-POSIX dev environments don't hard-fail at module
# import.
try:
    import fcntl  # POSIX advisory locks (DGX Spark + Condenser A100 paper-grade hosts)
    _HAS_FCNTL = True
except ImportError:  # pragma: no cover — non-POSIX
    fcntl = None  # type: ignore[assignment]
    _HAS_FCNTL = False


def _event_fingerprint(event: Dict[str, Any]) -> str:
    """B-491 (/stress A1.5b Phase 1 P1-3-B codex OOB, 2026-05-17): canonical
    fingerprint for trajectory event dedup. sha256[:16] of (event_type +
    wallclock_ts + sorted metadata JSON). Used by
    `merge_staging_trajectory_events` to skip already-merged events on
    resume+reset scenario.

    Excludes top-level transient fields injected at merge time
    (`merged_from_staging`, `staging_run_id`) so the fingerprint is stable
    across merge ↔ staging round-trips.
    """
    metadata = event.get("metadata") or {}
    if isinstance(metadata, dict):
        # Strip merge-time transient fields for canonical hash
        canonical_meta = {
            k: v for k, v in metadata.items()
            if k not in {"merged_from_staging", "staging_run_id"}
        }
        meta_str = json.dumps(canonical_meta, sort_keys=True, default=str)
    else:
        meta_str = str(metadata)
    components = (
        f"{event.get('event_type', '')}|"
        f"{event.get('wallclock_ts', '')}|"
        f"{event.get('task_index', '')}|"
        f"{meta_str}"
    )
    return hashlib.sha256(components.encode("utf-8")).hexdigest()[:16]


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
        # B-492 (/stress A1.5b Phase 1 P1-4-B codex OOB, 2026-05-17): fsync
        # parent dir entry on first-create. File-fd fsync alone does not flush
        # the dirent for a newly-created file — ext4 journal can hold the
        # dirent up to ~30s. DGX SIGKILL between first append + dirent flush
        # → reboot sees inode with data but no dirent → step JSONL "evaporates".
        # B-198/B-289 lineage (parent dir fsync after os.replace atomic write)
        # but for append-create code path. Best-effort: cheap (~10ms) only on
        # first-create; idempotent on subsequent appends.
        #
        # B-736 fix (/stress A1.8 cold-start P1-2-A* Claude OOB, 2026-05-17):
        # `fcntl.flock(LOCK_EX)` for the duration of the write. Long step
        # records with embedded screenshot/DOM artifact references can exceed
        # PIPE_BUF=4096 — POSIX append atomicity is NOT guaranteed beyond
        # that bound. Watchdog auto-clean retry path could race with the
        # primary runner on the same step file → torn writes → corrupt
        # JSONL line silently consumed by aggregators.
        _existed_pre_write = path.exists()
        with open(path, "a", encoding="utf-8") as f:
            if _HAS_FCNTL:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            finally:
                if _HAS_FCNTL:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        if not _existed_pre_write:
            _fsync_dir(path.parent)

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
    # event_type values (B-313 + B-384/B-385 update, 2026-05-16 A1.15 C1):
    #   - "reset_post_interrupt"   (cell-level, from reset_and_auth_gate post-reset)
    #   - "task_auto_cleared"      (per-task, from watchdog auto-clean retry path
    #                                AND session-cleanup wave path — distinguish via
    #                                metadata.cleared_in_session_wave bool)
    #   - "auth_refresh_no_clear"  (cell-level, watchdog auth refresh w/o cleanup)
    #   - "runner_restart"         (cell-level, planned for future runner audit)
    #   - "watchdog_intervention"  (cell-level, generic catch-all)
    # task_index = episode/task index at event time; None for cell-level events.
    # metadata = event-specific dict. Standard keys:
    #   reason / is_auth_loss / is_noise / cleared_in_session_wave / wave_size /
    #   wave_task_index / retry_attempt / max_retries / purged_digest_records /
    #   site / auth_refresh_method / reset_rc.
    #
    # B-385 (A1.15 C1 P0-4 reframe, 2026-05-16): condition_finalize race
    # (runner writes condition_summary AFTER watchdog reads `.exists()=False`
    # AND BEFORE watchdog destructive op completes) is detected post-hoc by
    # aggregator, NOT via a separate event_type. Aggregator intersection:
    # {task_id in condition_summary_v2.json["episode_ids"]} ∩ {task_id with
    # `task_auto_cleared` event in trajectory_events.jsonl} = race-cleared
    # episodes (denominator counted them, source files deleted). Emit derived
    # covariate `had_finalize_race_clear: bool` per episode in aggregator
    # output. No watchdog code change needed for race detection — best-effort
    # event emission already provides the data substrate (P0-3 hook C completes
    # the session path coverage at B-384).
    #
    # Aggregator emits per-episode `is_after_reset` / `had_auth_clear` /
    # `had_finalize_race_clear` / `prior_event_count` columns for GLMM
    # covariate adjustment in paper §4 (deferred (iii) -> C2 B-389).
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
        # B-492 (/stress A1.5b Phase 1 P1-4-B codex OOB, 2026-05-17): parent-
        # dir fsync on first-create — see write_step docstring for full
        # rationale (DGX SIGKILL between first append + dirent flush would
        # otherwise evaporate the trajectory event log entirely).
        #
        # B-736 fix (/stress A1.8 cold-start P1-2-A* Claude OOB, 2026-05-17):
        # `fcntl.flock(LOCK_EX)` — this is the highest-risk append path because
        # 2 callers concurrently write (in-process runner via this method,
        # AND out-of-band reset gate / watchdog via
        # `log_trajectory_event_external` which constructs a fresh LoggerV2 to
        # reach this same path). `metadata` payload (wave_task_index list,
        # purged_digest_records dict, etc.) routinely exceeds PIPE_BUF=4096.
        _existed_pre_write = path.exists()
        with open(path, "a", encoding="utf-8") as f:
            if _HAS_FCNTL:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            finally:
                if _HAS_FCNTL:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        if not _existed_pre_write:
            _fsync_dir(path.parent)


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


# B-388 (A1.15 C2 Merge (i), 2026-05-16): runner-side staging pickup.
# Pre-fix: B-314 Hook B wrote `reset_post_interrupt` events to
# `${repo_root}/logs/trajectory_events_staging/RUN_${RUN_ID}.jsonl` because
# condition_dir doesn't exist yet at reset gate time. Without pickup the
# staging file accumulated events but never made it into per-condition
# trajectory_events.jsonl → paper §4 aggregator (B-389) had no `is_after_reset`
# covariate data → Option K Tier 1 stack analysis layer received zero events
# from reset class. Runner calls this once per condition_dir at creation.
# Idempotent via "fresh dir only" guard: if condition_dir already contains
# trajectory_events.jsonl (resume case), pickup is skipped and existing events
# preserved.
def merge_staging_trajectory_events(
    condition_dir: Path,
    run_id: str,
    repo_root: Optional[Path] = None,
) -> int:
    """Pickup + merge per-RUN_ID staging file into condition_dir/trajectory_events.jsonl.

    Args:
      condition_dir: target condition_dir (must exist).
      run_id: the RUN_ID matching the staging file naming convention.
      repo_root: project root for staging dir location; auto-detected from
        condition_dir.parents if None (handles results/<bench>/<phase>/<run>/<cond>).

    Returns:
      Number of events merged. 0 if staging file absent, condition_dir already
      has events (resume case), or staging file empty.

    Idempotency: skipped silently if condition_dir/trajectory_events.jsonl
    already exists (treat as resume — prior pickup or runner-side events
    already in flight). Cell-level events (e.g. reset_post_interrupt) are
    duplicated across each condition_dir under the same RUN_ID by design —
    each condition's covariate-emission view sees its own copy.
    """
    if not condition_dir.exists():
        return 0
    target = condition_dir / "trajectory_events.jsonl"
    # B-491 (/stress A1.5b Phase 1 P1-3-B codex OOB, 2026-05-17): hash-based
    # idempotency. Pre-fix `if target.exists(): return 0` dropped staging
    # events on resume+reset scenario: T1 created target (empty or with reset
    # event); T2 same RUN_ID resume + RESET_BEFORE=1 → queue gate appends new
    # reset_post_interrupt to staging → runner sees target.exists() → skips
    # merge → new reset discontinuity false-negative in covariate trail
    # (paper §4 covariate analysis sees post-reset episodes as normal).
    # Now compute fingerprint of existing target events; on merge skip any
    # staging event whose fingerprint matches → idempotent against legitimate
    # re-merge, durable against new-event drops.
    _existing_fingerprints: set = set()
    if target.exists():
        try:
            with open(target, "r", encoding="utf-8") as _tf:
                for _line in _tf:
                    _line = _line.strip()
                    if not _line:
                        continue
                    try:
                        _ev = json.loads(_line)
                    except json.JSONDecodeError:
                        continue
                    _fp = _event_fingerprint(_ev)
                    _existing_fingerprints.add(_fp)
        except OSError:
            # Defensive: cannot read existing → treat as empty (allow merge).
            _existing_fingerprints = set()
    if repo_root is None:
        # Auto-detect: condition_dir = <repo_root>/results/<bench>/<phase>/<run>/<cond>
        # so repo_root = condition_dir.parents[4]
        try:
            repo_root = condition_dir.resolve().parents[4]
        except (IndexError, OSError):
            return 0
    staging_file = repo_root / "logs" / "trajectory_events_staging" / f"RUN_{run_id}.jsonl"
    if not staging_file.exists():
        return 0
    try:
        with open(staging_file, "r", encoding="utf-8") as src:
            lines = [ln.strip() for ln in src if ln.strip()]
    except OSError:
        return 0
    if not lines:
        return 0
    # Append each event line into condition_dir/trajectory_events.jsonl.
    # We re-emit via LoggerV2.log_trajectory_event so the wallclock_ts is
    # preserved from the staging file (parse, re-write); each event gets
    # added metadata `merged_from_staging=True` for aggregator awareness.
    logger = LoggerV2(condition_dir)
    merged = 0
    for line in lines:
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        meta = ev.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        meta = {**meta, "merged_from_staging": True, "staging_run_id": run_id}
        # Use the LoggerV2 helper but preserve the original wallclock_ts:
        # we write directly rather than via log_trajectory_event because the
        # helper sets wallclock_ts=now. Preserving the original ts is
        # important so paper §4 covariate `is_after_reset` correctly orders
        # events relative to per-episode wallclock.
        preserved_ts = ev.get("wallclock_ts") or datetime.datetime.now(datetime.timezone.utc).isoformat()
        event = {
            "event_type": ev.get("event_type", "unknown"),
            "task_index": ev.get("task_index"),
            "wallclock_ts": preserved_ts,
            "metadata": meta,
        }
        # B-491 (/stress A1.5b Phase 1 P1-3-B codex OOB, 2026-05-17): skip
        # if fingerprint matches an existing target event (canonical hash
        # excludes merged_from_staging + staging_run_id transients).
        _fp = _event_fingerprint(event)
        if _fp in _existing_fingerprints:
            continue
        _existing_fingerprints.add(_fp)
        # B-492 (/stress A1.5b Phase 1 P1-4-B codex OOB, 2026-05-17): first-
        # create parent-dir fsync — see write_step / log_trajectory_event for
        # full rationale.
        #
        # B-736 fix (/stress A1.8 cold-start P1-2-A* Claude OOB, 2026-05-17):
        # `fcntl.flock(LOCK_EX)` — merge races with concurrent
        # `log_trajectory_event` calls on the same target path (resume +
        # ongoing runner) are possible. Lock here is symmetric with the
        # in-process logger path.
        _existed_pre_write = target.exists()
        with open(target, "a", encoding="utf-8") as f:
            if _HAS_FCNTL:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            finally:
                if _HAS_FCNTL:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        if not _existed_pre_write:
            _fsync_dir(target.parent)
        merged += 1
    return merged
