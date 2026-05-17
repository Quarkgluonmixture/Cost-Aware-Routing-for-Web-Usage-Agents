"""B-862 (/stress A1.24 P0-7-C*, 2026-05-17): shared `clear_task_files` API.

Pre-fix: clear_tasks.py and experiment_watchdog.py independently called
`shutil.rmtree` + `Path.unlink` on the same file set (summary JSON, steps
JSONL, artifact dir). gemini Mode C audit framed it as "structural lie" of
CLAUDE.md "统一入口" claim — code-path divergence forces fix-one-forget-other
risk (same pattern as A1.4 B-451 `_shared_vl_utils` 4-consumer drift).

Solution: single canonical `clear_task_files()` function. Both clear_tasks.py
(operator-driven manual cleanup) AND experiment_watchdog.py auto-clean paths
import + call. Single source of truth for:
  - The exact file set deleted per task (summary + steps + artifact dir)
  - Idempotent deletion semantics (safe_unlink + safe_rmtree)
  - Option K event log emission (parity watchdog auto-clean + manual)

User directive 2026-05-17: "都应该用 Option K" — closes the symmetric audit
trail gap (manual + auto-clean both emit canonical events; reviewer/OSF
audit reads single event_type discriminator: `task_auto_cleared` vs
`manual_task_cleared`).
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def safe_unlink(path: Path) -> bool:
    """B-858 (/stress A1.24 P0-4-AB*): idempotent unlink that tolerates
    concurrent watchdog cleanup race. Returns True if file was deleted by
    THIS caller, False if path already gone (FileNotFoundError caught) OR
    didn't exist pre-call.

    Use over bare `Path.unlink()` whenever the deletion target could be
    simultaneously cleaned by another process (watchdog cron, sync rsync,
    manual operator).
    """
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


def safe_rmtree(path: Path) -> bool:
    """B-858 (/stress A1.24 P0-4-AB*): idempotent rmtree companion to
    `safe_unlink`. Returns True if this caller did the deletion, False if
    path already gone."""
    try:
        shutil.rmtree(path)
        return True
    except FileNotFoundError:
        return False


def _emit_option_k_event(
    condition_dir: Path,
    *,
    event_type: str,
    task_id: int,
    metadata: Dict[str, Any],
) -> None:
    """Best-effort wrapper around `log_trajectory_event_external`. Swallows
    import/IO failures so the calling deletion path stays robust. Parity
    with B-314 / B-384 watchdog auto-clean convention (T2'=a)."""
    try:
        from p79.experiment.logger_v2 import log_trajectory_event_external
        log_trajectory_event_external(
            condition_dir=condition_dir,
            event_type=event_type,
            task_index=task_id,
            metadata=metadata,
        )
    except Exception as _ev_exc:
        print(
            f"[cleanup][trajectory-event][warn] failed to log "
            f"{event_type} event for task {task_id}: {_ev_exc}",
            file=sys.stderr,
        )


def clear_task_files(
    condition_dir: Path,
    site: str,
    task_id: int,
    *,
    audit_event: bool = True,
    event_type: str = "manual_task_cleared",
    reason: str = "unspecified",
    force: bool = False,
    dry_run: bool = False,
    operator_pid: Optional[int] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Single-source-of-truth task file deletion for P79.

    Deletes summary JSON + steps JSONL + artifact dir for one (site, task_id)
    under `condition_dir`. Emits Option K trajectory event by default.

    Args:
        condition_dir: Path to condition dir (parent of `episodes/` + `artifacts/`)
        site: Site name (classifieds / reddit / shopping)
        task_id: Integer task ID
        audit_event: Emit Option K trajectory_events.jsonl entry (default True)
        event_type: Option K event type — "manual_task_cleared" (operator
            manual via clear_tasks.py) or "task_auto_cleared" (watchdog
            auto-clean retry / session-wave). Parity with B-314 / B-384.
        reason: Operator-supplied or context-derived reason annotation
        force: --force flag value (audit metadata)
        dry_run: True → no actual deletion, only event log + return False
        operator_pid: Process ID of caller (audit metadata)
        extra_metadata: Caller-specific metadata merged into the event payload
            (e.g. watchdog's `retry_attempt` / `cleared_in_session_wave`)

    Returns:
        True iff this caller actually deleted at least one file
        (False if all targets already absent OR dry_run).
    """
    if operator_pid is None:
        operator_pid = os.getpid()

    prefix = f"{site}_task_{task_id}"
    ep_dir = condition_dir / "episodes"
    art_dir = condition_dir / "artifacts"

    summary_file = ep_dir / f"{prefix}_summary_v2.json"
    steps_file = ep_dir / f"{prefix}_steps_v2.jsonl"
    artifact_dir = art_dir / prefix

    deleted_any = False
    if not dry_run:
        if summary_file.exists() and safe_unlink(summary_file):
            deleted_any = True
        if steps_file.exists() and safe_unlink(steps_file):
            deleted_any = True
        if artifact_dir.exists() and safe_rmtree(artifact_dir):
            deleted_any = True

    if audit_event and (deleted_any or dry_run):
        metadata: Dict[str, Any] = {
            "site": site,
            "reason": reason,
            "force": force,
            "dry_run": dry_run,
            "operator_pid": operator_pid,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        _emit_option_k_event(
            condition_dir=condition_dir,
            event_type=event_type,
            task_id=task_id,
            metadata=metadata,
        )

    return deleted_any
