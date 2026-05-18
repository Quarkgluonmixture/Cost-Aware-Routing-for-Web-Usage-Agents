#!/usr/bin/env python3
"""Delete result files for specific tasks so the runner retry pass can re-run them.

Deletes: summary JSON, steps JSONL, artifacts directory, and digest records for each task.

Examples:
    # Delete tasks 85-131 for classifieds (B1)
    python scripts/maintenance/clear_tasks.py --run-dir results/.../B1_run \
        --condition phase1_dom_router_0 --site classifieds --tasks 85-131

    # B0 / Gemma3-VL (B2) follow identical CLI shape — site is the discriminator
    python scripts/maintenance/clear_tasks.py --run-dir results/.../B0_run \
        --condition phase1_som_router_0 --site reddit --tasks 5,10,20
    python scripts/maintenance/clear_tasks.py --run-dir results/.../B2_run \
        --condition phase1_vision_router_0 --site shopping --tasks 100-150

    # Dry run (show what would be deleted)
    python scripts/maintenance/clear_tasks.py --run-dir results/.../B1_run \
        --condition phase1_dom_router_0 --site classifieds --tasks 85-131 --dry-run

    # Clean orphan artifact dirs (no summary file) across all conditions
    python scripts/maintenance/clear_tasks.py --run-dir results/.../B1_run --clean-orphan-artifacts

    # Clean orphan artifacts for a specific condition
    python scripts/maintenance/clear_tasks.py --run-dir results/.../B1_run \
        --condition phase1_som_router_0 --clean-orphan-artifacts

Safety guards (/stress A1.24 B-873~B-889, 2026-05-17):
    - `--force` requires `--confirm-run-id <run_id>` matching the target run; rejects
      when (a) any `.in_progress` marker present per-task, (b) live `pgrep -f
      run_experiment.*--config.*<site>` returns active PID, (c) `P79_PAPER_GRADE=1`
      env is set (paper-grade fire disallows --force entirely).
    - Orphan cleanup respects `.in_progress` markers + skips `.stale_<ts>`
      forensic archives (parity with `experiment_watchdog.py:1418,1427,1435,1444`).
    - All deletions are wrapped in `safe_unlink` / `safe_rmtree` (idempotent vs
      concurrent watchdog cleanup).
    - Per-run advisory lock (`/tmp/p79_clear_tasks_<run_hash>.lock`) prevents
      two concurrent clear_tasks invocations from leaving half-deleted state.
    - `--site` is validated against `_VALID_SITES` (whitelist enforces CLAUDE.md
      "VWA 只有 shopping/reddit/classifieds 三站" hard rule at the entry script).
    - `_parse_task_ids` rejects `lo > hi` ranges and `task_id > scored_task_count`.
    - Every deletion (manual mode) writes an Option K
      `event_type="manual_task_cleared"` event to the condition's
      trajectory_events.jsonl (parity with watchdog L1697/L1860
      `task_auto_cleared`). Set `CLEAR_REASON='<2-3 word reason>'` env var to
      annotate the event.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, IO, List, Optional, Set

try:
    import fcntl  # POSIX advisory locks — same primitive as logger_v2 (B-736)
    _HAS_FCNTL = True
except ImportError:  # pragma: no cover — non-POSIX dev env
    fcntl = None  # type: ignore[assignment]
    _HAS_FCNTL = False

_EXCLUDED_DIRS = {"analysis", "task_configs", "_vwa"}

_VALID_SITES = {"classifieds", "reddit", "shopping"}
"""B-888 (/stress A1.24 P1-6-A, 2026-05-17): CLAUDE.md hard rule
"VWA 只有 shopping/reddit/classifieds 三站" enforced at entry — typo
`--site shoping` previously silent no-op + "skipped 0" misled operator into
re-firing runner over stale episodes."""


# B-880 (/stress A1.24 P0-7-C*, 2026-05-17): import canonical safe_unlink /
# safe_rmtree + clear_task_files from `p79.experiment.cleanup` shared module.
# Pre-fix: clear_tasks defined these locally + watchdog L1830-1850 inlined
# bare shutil.rmtree/unlink → code-path divergence (gemini "structural lie"
# framing). Now: single canonical source. clear_tasks.py keeps local thin
# alias for downstream tests / external scripts that may still import here.
from p79.experiment.cleanup import (
    clear_task_files,
    safe_rmtree,
    safe_unlink,
)


def _run_lock_path(run_dir: Path) -> Path:
    """B-876 (/stress A1.24 P0-4-AB*): per-run advisory lock path.

    Uses sha256[:16] of resolved run_dir absolute path so the lock survives
    rename/symlink games. /tmp keeps it host-local (cross-host coordination
    handled separately at sync_a100 layer, see B-877/B-878)."""
    run_hash = hashlib.sha256(str(run_dir.resolve()).encode()).hexdigest()[:16]
    return Path(f"/tmp/p79_clear_tasks_{run_hash}.lock")


def _acquire_run_lock(run_dir: Path) -> Optional[IO[str]]:
    """B-876: acquire exclusive flock; return open fd handle (caller closes
    on exit, kernel releases lock). Returns None if fcntl unavailable
    (non-POSIX) — caller proceeds at-own-risk with explicit warning."""
    if not _HAS_FCNTL:
        print(
            f"[clear_tasks][WARN] fcntl unavailable on this platform; "
            f"per-run lock skipped — concurrent invocation may leave "
            f"half-deleted state",
            file=sys.stderr,
        )
        return None
    import time as _time
    lock_path = _run_lock_path(run_dir)
    fh = open(lock_path, "w", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        fh.close()
        print(
            f"[clear_tasks][BLOCKED] Another clear_tasks invocation already "
            f"holds {lock_path}. Wait for it to complete or kill stale lock "
            f"holder. Aborting to prevent half-deleted state.",
            file=sys.stderr,
        )
        sys.exit(2)
    fh.write(f"pid={os.getpid()}\nstart={int(_time.time())}\n")
    fh.flush()
    return fh


def _parse_task_ids(spec: str, *, max_task_id: Optional[int] = None) -> List[int]:
    """Parse '85-131' or '80,95,104' or '85-90,95,100-104' into sorted list.

    B-886 (/stress A1.24 P1-5-A, 2026-05-17): sanity guards added —
    rejects `lo > hi` (typo `100-99` previously silently produced empty
    range, reported as "skipped 0 (not found)", misleading operator),
    rejects negatives, and caps at `max_task_id` if provided (passed in
    by main() from `scored_task_count(site)`). Operator typo `--tasks
    100-99` now raises immediately instead of pretending success.
    """
    ids: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"--tasks empty segment in spec {spec!r}")
        # B-886 edge: leading minus sign (negative task_id) — reject explicitly
        # before split, since split('-',1) on '-5' yields ('','5') → confusing
        # 'int("")' ValueError trace. P79 task_ids are always non-negative.
        if part.startswith("-"):
            raise ValueError(
                f"--tasks {part!r}: leading '-' (negative task_id) — "
                f"P79 task_ids are non-negative integers"
            )
        if "-" in part:
            lo_str, hi_str = part.split("-", 1)
            try:
                lo, hi = int(lo_str.strip()), int(hi_str.strip())
            except ValueError as exc:
                raise ValueError(
                    f"--tasks range {part!r}: non-integer endpoint ({exc})"
                ) from exc
            if lo > hi:
                raise ValueError(
                    f"--tasks range {part!r}: lo > hi ({lo} > {hi}) — "
                    f"likely typo (use {hi}-{lo} for descending semantics)"
                )
            ids.update(range(lo, hi + 1))
        else:
            try:
                ids.add(int(part))
            except ValueError as exc:
                raise ValueError(f"--tasks non-integer {part!r}: {exc}") from exc
    if any(t < 0 for t in ids):
        raise ValueError(f"--tasks contains negative task_id: {sorted(t for t in ids if t < 0)}")
    if max_task_id is not None and any(t > max_task_id for t in ids):
        over = sorted(t for t in ids if t > max_task_id)
        raise ValueError(
            f"--tasks contains task_id > scored_task_count ({max_task_id}): {over[:5]}"
            f"{'...' if len(over) > 5 else ''} — likely typo or wrong site"
        )
    return sorted(ids)


def _has_active_runner(site: str) -> Optional[List[str]]:
    """B-873 (/stress A1.24 P0-1-ABC*, 2026-05-17): pgrep for active runner
    processes on the target site. Returns list of "PID CMDLINE" strings if
    runners active, None if none. Used by --force to reject deletion when
    a runner is mid-flight (pilot wave-1 destruction 2026-04-30 实证 —
    another Claude session ran --force during active fire and destroyed
    in-flight episodes)."""
    pattern = f"run_experiment.*--config.*{site}"
    try:
        result = subprocess.run(
            ["pgrep", "-af", pattern],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return [line.strip() for line in result.stdout.strip().split("\n")]
        return None
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        # pgrep unavailable (rare on Linux); fall through to other guards
        return None


def _has_in_progress_marker(cond_dir: Path, site: str, task_ids: List[int]) -> List[int]:
    """B-873 / P0-2 companion: scan per-task `.in_progress` markers under
    `artifacts/{site}_task_{tid}/.in_progress`. Returns list of task_ids
    that have an active marker (parity with watchdog L1427/L1444)."""
    art_dir = cond_dir / "artifacts"
    if not art_dir.exists():
        return []
    in_progress = []
    for tid in task_ids:
        marker = art_dir / f"{site}_task_{tid}" / ".in_progress"
        if marker.exists():
            in_progress.append(tid)
    return in_progress


def _ntfy_force_alert(run_dir: Path, condition: str, site: str, tasks: List[int]) -> None:
    """B-873 / P0-1: best-effort ntfy alert on --force invocation.
    Audit trail substrate (operator visibility for multi-session 误用).
    Topic `p79-claude` is the project default."""
    try:
        import urllib.request as _u
        msg = (
            f"[clear_tasks --force] pid={os.getpid()} on "
            f"{run_dir.name}/{condition}/{site} tasks={tasks[:10]}"
            f"{'...' if len(tasks) > 10 else ''} at {os.popen('date').read().strip()}"
        )
        req = _u.Request(
            "https://ntfy.sh/p79-claude",
            data=msg.encode("utf-8"),
            method="POST",
            headers={"Title": "P79 destructive op", "Priority": "high"},
        )
        _u.urlopen(req, timeout=3)
    except Exception:
        pass  # ntfy is best-effort, never blocks deletion


# B-880 / B-881: `_emit_manual_clear_event` removed — Option K event log is
# now centralized in `p79.experiment.cleanup.clear_task_files` (called below
# in per-task loop). Single emit-site eliminates fix-one-forget-other drift
# (lesson: A1.4 B-451 `_shared_vl_utils` 4-consumer drift).


def _clean_orphan_artifacts(
    run_dir: Path,
    condition: Optional[str],
    dry_run: bool,
    stale_mins: int = 10,
) -> int:
    """Delete artifact dirs and orphan steps files that have no corresponding summary.

    Items modified within the last `stale_mins` minutes are skipped — they may
    belong to an in-progress episode (runner creates artifacts/steps before writing
    the summary).

    B-874 (/stress A1.24 P0-2-AB*, 2026-05-17): mtime cutoff alone is
    insufficient for long-running B0 episodes (5-15 min wallclock + LLM
    eval). Watchdog has dual-guard (`.in_progress` marker + mtime); clear_tasks
    now matches via `_in_progress_marker_present` per-artifact + per-steps.

    B-875 (/stress A1.24 P0-3-ABC*, 2026-05-17): `.stale_<ts>` archive
    skip (parity with watchdog L1418 + L1435). Runner-side `_run_episode`
    archives prior-attempt forensic to `<task>.stale_<ts>` siblings when
    `.in_progress` marker present (runner-crash recovery path, B-488).
    Pre-fix clear_tasks orphan mode treated these as deletable orphans →
    forensic loss. Now: skip-by-name like watchdog.

    B-876 (/stress A1.24 P0-4-AB*, 2026-05-17): all deletions wrapped in
    `safe_rmtree` / `safe_unlink` — idempotent vs concurrent watchdog
    orphan-clean race (was: bare `shutil.rmtree` raised FileNotFoundError
    on 2nd cleaner, halted digest cleanup, left half-deleted state).
    """
    import time as _time
    cutoff = _time.time() - stale_mins * 60

    if condition:
        cond_dirs = [run_dir / condition]
    else:
        cond_dirs = sorted(
            p for p in run_dir.iterdir()
            if p.is_dir() and p.name not in _EXCLUDED_DIRS and not p.suffix
        )

    deleted = 0
    skipped_recent = 0
    skipped_in_progress = 0
    skipped_stale_archive = 0
    for cond_dir in cond_dirs:
        if not cond_dir.is_dir():
            continue
        art_dir = cond_dir / "artifacts"
        ep_dir = cond_dir / "episodes"

        # 1. Orphan artifact directories (no summary)
        if art_dir.exists():
            for artifact in sorted(art_dir.iterdir()):
                if not artifact.is_dir():
                    continue
                # B-875: skip B-488 forensic archives (no-summary by design)
                if ".stale_" in artifact.name:
                    skipped_stale_archive += 1
                    continue
                if (ep_dir / f"{artifact.name}_summary_v2.json").exists():
                    continue
                if artifact.stat().st_mtime > cutoff:
                    skipped_recent += 1
                    continue
                # B-874: respect runner .in_progress marker (dual-guard parity
                # with watchdog L1427-1428)
                if (artifact / ".in_progress").exists():
                    skipped_in_progress += 1
                    continue
                rel = artifact.relative_to(run_dir)
                if dry_run:
                    print(f"  [dry-run] rm -rf {rel}  (orphan artifact — no summary)")
                else:
                    if safe_rmtree(artifact):
                        print(f"  deleted orphan artifact: {rel}")
                deleted += 1

        # 2. Orphan steps files (steps JSONL without corresponding summary)
        if ep_dir.exists():
            for steps_file in sorted(ep_dir.glob("*_steps_v2.jsonl")):
                # B-875: skip <task>.stale_<ts>.jsonl archives (parity watchdog L1435)
                if ".stale_" in steps_file.name:
                    skipped_stale_archive += 1
                    continue
                # Derive the expected summary filename
                summary = ep_dir / steps_file.name.replace("_steps_v2.jsonl", "_summary_v2.json")
                if summary.exists():
                    continue
                if steps_file.stat().st_mtime > cutoff:
                    skipped_recent += 1
                    continue
                # B-874: per-episode .in_progress marker (parity watchdog L1444-1445)
                ep_stem = steps_file.name.replace("_steps_v2.jsonl", "")
                if (art_dir / ep_stem / ".in_progress").exists():
                    skipped_in_progress += 1
                    continue
                rel = steps_file.relative_to(run_dir)
                if dry_run:
                    print(f"  [dry-run] rm {rel}  (orphan steps — no summary)")
                else:
                    if safe_unlink(steps_file):
                        print(f"  deleted orphan steps: {rel}")
                deleted += 1

    if skipped_recent:
        print(f"  (skipped {skipped_recent} recently-modified item(s) — may be in-progress)")
    if skipped_in_progress:
        print(f"  (skipped {skipped_in_progress} item(s) with .in_progress marker — B-874 guard)")
    if skipped_stale_archive:
        print(f"  (skipped {skipped_stale_archive} .stale_<ts> forensic archive(s) — B-875 guard)")
    return deleted


def main() -> int:
    p = argparse.ArgumentParser(description="Delete task results for runner retry")
    p.add_argument("--run-dir", required=True, help="Run directory")
    p.add_argument("--condition", default=None, help="Condition ID (e.g. phase1_dom_router_0)")
    p.add_argument("--site", default=None,
                    help=f"Site name — must be one of {sorted(_VALID_SITES)}")
    p.add_argument("--tasks", default=None, help="Task IDs: '85-131' or '80,95,104' or '85-90,100-104'")
    p.add_argument("--clean-orphan-artifacts", action="store_true",
                    help="Delete artifact dirs that have no corresponding summary file")
    p.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")
    p.add_argument("--force", action="store_true",
                    help="Also delete tasks that may be in-progress (has steps but no summary). "
                         "REQUIRES --confirm-run-id <run_id>; rejects if .in_progress marker present, "
                         "active runner PID detected, OR P79_PAPER_GRADE=1 env set (B-873).")
    p.add_argument("--confirm-run-id", default=None,
                    help="When --force is used, must equal the run-dir basename (B-873 double-flag).")
    args = p.parse_args()

    # Validate: either --tasks or --clean-orphan-artifacts must be provided
    if not args.tasks and not args.clean_orphan_artifacts:
        p.error("one of --tasks or --clean-orphan-artifacts is required")
    if args.tasks and not args.condition:
        p.error("--condition is required when using --tasks")
    if args.tasks and not args.site:
        p.error("--site is required when using --tasks")

    # B-888 (/stress A1.24 P1-6-A): --site whitelist enforcement.
    # CLAUDE.md hard rule "VWA 只有 shopping/reddit/classifieds 三站"
    # previously not enforced at entry script — typo `--site shoping`
    # silently passed → no prefix match → "skipped 0" misled operator.
    if args.site is not None and args.site not in _VALID_SITES:
        p.error(
            f"--site {args.site!r} not in {sorted(_VALID_SITES)}. "
            f"CLAUDE.md hard rule: VWA only has shopping/reddit/classifieds."
        )

    run_dir = Path(args.run_dir).resolve()

    # B-876 (/stress A1.24 P0-4-AB*): acquire per-run advisory lock BEFORE any
    # deletion. Held until process exit (kernel releases on fd close). Prevents
    # two concurrent clear_tasks from leaving half-deleted state (digest cleanup
    # in 2nd caller halted by FileNotFoundError on already-removed file).
    _lock_handle = _acquire_run_lock(run_dir) if not args.dry_run else None
    # NOTE: dry-run skips lock — observer only, no state mutation.

    # --- Orphan artifact cleanup mode ---
    if args.clean_orphan_artifacts:
        # B-1414 (/stress A2.7 P2-15-A Claude Mode A, 2026-05-18): paper-grade
        # mid-fire defense — orphan cleanup is the watchdog auto-clean path's
        # job, NOT a manual operator action under P79_PAPER_GRADE=1. Pre-fix
        # `--clean-orphan-artifacts` could be invoked mid-fire and would delete
        # B-488 forensic substrate or pre-quarantine artifacts without paper-
        # grade context. Same symmetric guard as `--force` at L448 above
        # (paper-grade env disables destructive cleanup). Override via
        # explicit P79_PAPER_GRADE=0 env opt-out, NOT recommended during fire.
        if os.environ.get("P79_PAPER_GRADE", "") == "1" and not args.dry_run:
            print(
                "ERROR: --clean-orphan-artifacts REJECTED — P79_PAPER_GRADE=1 env "
                "set. Watchdog auto-clean is the paper-grade-correct orphan-cleanup "
                "path (it respects .in_progress markers + .stale_<ts> forensic "
                "archives + ntfy alerts on destructive ops). Manual orphan cleanup "
                "during fire risks destroying B-488 forensic substrate or pre-"
                "quarantine artifacts. Run with --dry-run to preview, or unset "
                "P79_PAPER_GRADE explicitly if you really intend to clean under "
                "paper-grade context. See B-1414 /stress A2.7 P2-15-A.",
                file=sys.stderr,
            )
            return 3
        orphans_deleted = _clean_orphan_artifacts(run_dir, args.condition, args.dry_run)
        action = "would delete" if args.dry_run else "deleted"
        print(f"\nDone: {action} {orphans_deleted} orphan artifact dir(s)")
        if not args.tasks:
            return 0

    # --- Task-level cleanup ---
    if not args.condition:
        p.error("--condition is required when using --tasks")
    cond_dir = run_dir / args.condition
    if not cond_dir.exists():
        print(f"ERROR: condition dir not found: {cond_dir}", file=sys.stderr)
        return 1

    ep_dir = cond_dir / "episodes"
    art_dir = cond_dir / "artifacts"
    site = args.site

    # B-886 (/stress A1.24 P1-5-A): pass max_task_id cap to _parse_task_ids.
    # Best-effort scored_task_count lookup; fall back to None if site config
    # missing (legacy archive runs).
    _max_tid: Optional[int] = None
    try:
        from p79.experiment.analysis import scored_task_count as _sct
        _max_tid = _sct(site, "visualwebarena", strict=False)
        if _max_tid <= 0:
            _max_tid = None  # config missing → don't enforce cap
    except Exception:
        pass  # legacy or analysis-deps missing — fall through without cap
    try:
        task_ids = _parse_task_ids(args.tasks, max_task_id=_max_tid)
    except ValueError as _parse_exc:
        # B-886: surface clean error (was bare ValueError traceback)
        print(f"ERROR: {_parse_exc}", file=sys.stderr)
        return 2

    # B-873 (/stress A1.24 P0-1-ABC*): --force hardening cluster.
    # Paper-grade env disables --force entirely.
    if args.force and os.environ.get("P79_PAPER_GRADE", "") == "1":
        print(
            "ERROR: --force REJECTED — P79_PAPER_GRADE=1 env set. "
            "Paper-grade fire disallows in-progress deletion. "
            "If you need to clear data, unset P79_PAPER_GRADE and re-run "
            "with explicit acknowledgement.",
            file=sys.stderr,
        )
        return 3
    # --confirm-run-id double-flag check
    if args.force:
        expected_run_id = run_dir.name
        if args.confirm_run_id != expected_run_id:
            print(
                f"ERROR: --force REQUIRES --confirm-run-id {expected_run_id!r} "
                f"(got {args.confirm_run_id!r}). This double-flag prevents "
                f"copy-paste destruction of unrelated runs (pilot wave-1 destruction "
                f"2026-04-30 实证: another Claude session ran --force, wiped live data).",
                file=sys.stderr,
            )
            return 3
        # PID-liveness check on this site
        active = _has_active_runner(site)
        if active and not args.dry_run:
            print(
                f"ERROR: --force REJECTED — active runner process(es) on site={site}:",
                file=sys.stderr,
            )
            for line in active:
                print(f"    {line}", file=sys.stderr)
            print(
                "Wait for runner to complete or kill before --force. "
                "(CLAUDE.md: same-site single-baseline hard rule)",
                file=sys.stderr,
            )
            return 3
        # Per-task .in_progress marker check
        in_progress = _has_in_progress_marker(cond_dir, site, task_ids)
        if in_progress and not args.dry_run:
            print(
                f"ERROR: --force REJECTED — .in_progress marker(s) present for tasks: "
                f"{in_progress[:10]}{'...' if len(in_progress) > 10 else ''}. "
                f"Runner is mid-flight on these tasks. Wait for completion or "
                f"clear .in_progress markers explicitly via watchdog kill path.",
                file=sys.stderr,
            )
            return 3
        # All --force guards passed — fire ntfy alert (audit trail)
        if not args.dry_run:
            _ntfy_force_alert(run_dir, args.condition, site, task_ids)

    deleted = 0
    skipped = 0
    in_progress_skipped = 0

    for tid in task_ids:
        prefix = f"{site}_task_{tid}"
        summary_file = ep_dir / f"{prefix}_summary_v2.json"
        steps_file = ep_dir / f"{prefix}_steps_v2.jsonl"
        artifact_dir = art_dir / prefix

        # Safety: skip tasks that may be currently running
        # (has steps JSONL or artifacts but no summary yet)
        if not summary_file.exists() and (steps_file.exists() or artifact_dir.exists()):
            if not args.force:
                print(f"  SKIP {prefix} — in-progress (has steps/artifacts but no summary). Use --force to override")
                in_progress_skipped += 1
                continue

        # B-880 (/stress A1.24 P0-7-C*): per-task deletion now via shared
        # `clear_task_files` API (safe_unlink/safe_rmtree + Option K event
        # emission rolled into single call). Dry-run output handled inline
        # because shared API's dry_run=True returns False and skips deletion;
        # we still need operator-visible "[dry-run] rm ..." lines here.
        if args.dry_run:
            files = [summary_file, steps_file]
            dirs = [artifact_dir]
            found_any = False
            for f in files:
                if f.exists():
                    found_any = True
                    print(f"  [dry-run] rm {f.relative_to(run_dir)}")
            for d in dirs:
                if d.exists():
                    found_any = True
                    print(f"  [dry-run] rm -rf {d.relative_to(run_dir)}")
            if found_any:
                deleted += 1
            else:
                skipped += 1
        else:
            did_delete = clear_task_files(
                condition_dir=cond_dir,
                site=site,
                task_id=tid,
                event_type="manual_task_cleared",
                reason=os.environ.get("CLEAR_REASON", "unspecified"),
                force=args.force,
                dry_run=False,
                operator_pid=os.getpid(),
            )
            if did_delete:
                deleted += 1
                print(f"  deleted {prefix}")
            else:
                skipped += 1

    # B-890 (/stress A1.24 P1-8-C, 2026-05-17): `.cleaning` flag wrapping
    # digest + cond_summary operations. Pre-fix: digest atomic write
    # (B-226) + cond_summary unlink were independently atomic but no joint
    # guarantee — Ctrl+C / crash between them left "digest cleaned ✓ +
    # cond_summary still Finalized" zombie state → runner skips condition,
    # aggregator pulls stale-but-task-gone records → inconsistent report.
    # Flag-file approach: touch `.cleaning` at start, remove at end;
    # forensic operator can detect interrupted state via `ls .cleaning`.
    # Future runner resume gate can be enhanced to detect+quarantine in C4
    # (preflight check_clear_tasks_recovery).
    _cleaning_flag: Optional[Path] = None
    if not args.dry_run and deleted > 0:
        _cleaning_flag = cond_dir / ".cleaning"
        try:
            _cleaning_flag.touch(exist_ok=True)
        except OSError:
            _cleaning_flag = None  # best-effort; never block cleanup

    # --- Clean digest records for deleted tasks ---
    #
    # B-882 (/stress A1.24 P1-1-A* disclose-only, 2026-05-17): digest layer
    # is fully RETIRED — empirical 2026-05-17 scan finds zero digest_*.jsonl
    # files on filesystem + zero `aggregate_*.py` consumers (B-743 retired
    # watchdog write path; aggregators followed). `validate_run.py:920-936`
    # remains as defensive "skip" check. This task-mode digest cleanup block
    # is therefore dead code in practice but harmless (early return at L218
    # if `digest_dir` absent). The corresponding orphan-mode gap (original
    # P1-1 finding "`--clean-orphan-artifacts` doesn't clean digest") is
    # MOOT — no digest to clean. Block retained for legacy archive runs
    # that may still have stale digest_*.jsonl from pre-B-743 era.
    digest_cleaned = 0
    task_id_set: Set[int] = set(task_ids)
    digest_dir = run_dir / "analysis" / "digest"
    if digest_dir.exists():
        for jsonl_file in sorted(digest_dir.glob("digest_*.jsonl")):
            try:
                lines = jsonl_file.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue
            keep = []
            removed_here = 0
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    keep.append(line)
                    continue
                # B-885 (/stress A1.24 P1-4-C*, 2026-05-17): int/string type
                # coercion for cross-baseline (WA) parity. WebArena (WA)
                # digest may store task_id as string ("123") while
                # _parse_task_ids returns int; `tid in task_id_set` then
                # silently false → digest record never removed → aggregator
                # later joins stale record with fresh rerun → silent SR
                # pollution. Coerce both sides to canonical int. Non-int
                # raw values (rare) fall to -1 sentinel and naturally skip.
                tid_raw = rec.get("task_id", -1)
                try:
                    tid = int(tid_raw)
                except (TypeError, ValueError):
                    tid = -1
                cid = rec.get("condition_id", "")
                if tid in task_id_set and (not cid or cid == args.condition):
                    removed_here += 1
                else:
                    keep.append(line)
            if removed_here:
                if args.dry_run:
                    print(f"  [dry-run] remove {removed_here} records from {jsonl_file.name}")
                else:
                    # B-226 fix (2026-05-16, A1.5 Item 10): atomic temp-write + fsync
                    # + os.replace to prevent half-written digest after partial crash.
                    # Pre-fix: jsonl_file.write_text(...) is non-atomic — crash mid-write
                    # leaves digest 半写或残留旧记录 (episode files 已 unlink at line 203).
                    # Post-fix: write to .tmp, fsync, atomic rename — followers see either
                    # old or new digest, never partial.
                    _new_content = "\n".join(keep) + ("\n" if keep else "")
                    _tmp_path = jsonl_file.with_suffix(jsonl_file.suffix + ".tmp")
                    with open(_tmp_path, "w", encoding="utf-8") as _f:
                        _f.write(_new_content)
                        _f.flush()
                        try:
                            os.fsync(_f.fileno())
                        except OSError:
                            pass
                    os.replace(_tmp_path, jsonl_file)
                    # fsync dir entry so rename hits stable storage
                    try:
                        _dir_fd = os.open(str(jsonl_file.parent), os.O_RDONLY)
                        try:
                            os.fsync(_dir_fd)
                        finally:
                            os.close(_dir_fd)
                    except OSError:
                        pass
                digest_cleaned += removed_here

    # --- Remove stale condition_summary if episodes are now incomplete ---
    # This ensures the watchdog and queue scripts detect the condition as
    # incomplete and re-run the missing tasks + post-analysis.
    #
    # B-889 (/stress A1.24 P1-7-A, 2026-05-17): glob `*.json` was fragile —
    # any future task_configs sibling (e.g. `_manifest.json`, `.skipped.json`)
    # would inflate `total` → `remaining < total` permanently true → infinite
    # re-aggregation loop. Switched to `*_task_*.json` precise pattern matching
    # P79 task config filename convention (`reddit_task_0.json` etc.).
    cond_summary_removed = False
    if deleted > 0:
        cond_summary_path = cond_dir / "condition_summary_v2.json"
        if cond_summary_path.exists():
            # Count remaining summaries vs total task configs
            remaining = len(list(ep_dir.glob("*_summary_v2.json"))) if ep_dir.exists() else 0
            tc_dir = run_dir / "task_configs"
            total = len(list(tc_dir.glob("*_task_*.json"))) if tc_dir.exists() else 0
            if total > 0 and remaining < total:
                if args.dry_run:
                    print(f"  [dry-run] rm {cond_summary_path.relative_to(run_dir)}  (stale: {remaining}/{total} episodes)")
                else:
                    # B-876: safe_unlink — concurrent watchdog cleanup race-safe.
                    if safe_unlink(cond_summary_path):
                        print(f"  deleted stale condition_summary ({remaining}/{total} episodes remaining)")
                cond_summary_removed = True

    # B-890 closure: remove .cleaning flag (all critical ops past this point)
    if _cleaning_flag is not None and _cleaning_flag.exists():
        try:
            _cleaning_flag.unlink()
        except OSError:
            pass

    action = "would delete" if args.dry_run else "deleted"
    parts = [f"{action} {deleted} tasks"]
    if cond_summary_removed:
        parts.append(f"{action} stale condition_summary")
    if digest_cleaned:
        parts.append(f"{action} {digest_cleaned} digest records")
    if skipped:
        parts.append(f"skipped {skipped} (not found)")
    if in_progress_skipped:
        parts.append(f"skipped {in_progress_skipped} (in-progress)")
    print(f"\nDone: {', '.join(parts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
