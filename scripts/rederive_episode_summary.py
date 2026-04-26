#!/usr/bin/env python3
"""Re-derive bug-affected fields in existing episode summaries from step JSONL.

Targets fields that were silently polluted by §97 audit findings:
  - page_unchanged_rate (RU-1): runner.py inflated this by counting finish/stop
    steps as "unchanged" — recompute from steps excluding finish/stop.
  - energy_partial (RU-5): new flag indicating that some steps lack an energy
    reading (NVML probe failed). Old summaries don't have the field.
  - energy_step_complete_count (RU-5): companion to above.
  - busy_wait_total_ms (RU-4): old data didn't record per-wait latency, so we
    can only set 0 with an audit warning. New runs (post-fix) record properly.

For each episode summary touched, the original is backed up to
`<condition>/episodes/.bak_pre_rederive/<orig_name>` (one-shot — never overwrite
an existing backup, so re-running this script is idempotent w.r.t. backups).

After all episodes in a condition are rewritten, recompute and re-write
`condition_summary_v2.json` using `aggregate_condition_metrics`.

Usage:
    # Dry-run on one run dir (prints diff, doesn't write):
    python scripts/rederive_episode_summary.py --run-dir results/visualwebarena/phase1/B0_3mode_classifieds_20260413 --dry-run

    # Apply to all B0 run dirs across both benchmarks:
    python scripts/rederive_episode_summary.py --all-b0

    # Apply to a specific run dir:
    python scripts/rederive_episode_summary.py --run-dir results/visualwebarena/phase1/B0_3mode_reddit_20260422
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make repo root importable so we can use p79.* utilities.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from p79.experiment.io_utils import read_jsonl_dedup  # noqa: E402
from p79.experiment.logger_v2 import LoggerV2  # noqa: E402
from p79.experiment.metrics import aggregate_condition_metrics  # noqa: E402

_FINISH_TYPES = ("finish", "stop")
_BACKUP_DIR_NAME = ".bak_pre_rederive"


@dataclass
class EpisodeRederiveResult:
    site: str
    task_id: int
    old_page_unchanged_rate: Optional[float]
    new_page_unchanged_rate: float
    energy_partial: bool
    energy_step_complete_count: int
    n_steps: int
    n_finish_steps_excluded: int
    has_effective_action: bool = False
    adjusted_success: Optional[bool] = None
    fp_reason: str = ""


def _action_type(step: Dict[str, Any]) -> str:
    """Return action_type, preferring nested action.action_type but falling
    back to top-level for older schema."""
    action = step.get("action") or {}
    at = action.get("action_type")
    if at:
        return str(at).lower()
    return str(step.get("action_type", "") or "").lower()


def _rederive_one(steps: List[Dict[str, Any]]) -> Tuple[float, bool, int, int, bool]:
    """Compute (page_unchanged_rate, energy_partial, energy_step_complete_count,
    n_finish_steps_excluded, has_effective_action) from raw step records."""
    if not steps:
        return 0.0, False, 0, 0, False
    n_total = len(steps)
    # page_unchanged_rate excludes finish/stop steps from numerator AND denominator
    # would be ideal, but to stay consistent with the runner's new fix (which only
    # excludes from numerator), we mirror that exact formula.
    unchanged = 0
    n_finish = 0
    has_effective_action = False
    for s in steps:
        at = _action_type(s)
        if at in _FINISH_TYPES:
            n_finish += 1
            continue
        if not bool(s.get("page_changed", False)):
            unchanged += 1
        if at in ("type", "select_option"):
            has_effective_action = True
    page_unchanged_rate = unchanged / n_total if n_total else 0.0

    # Energy completeness
    energy_complete = sum(
        1 for s in steps
        if isinstance(s.get("energy"), dict) and s["energy"].get("kwh") is not None
    )
    energy_partial = energy_complete < n_total
    return page_unchanged_rate, energy_partial, energy_complete, n_finish, has_effective_action


def _process_episode(
    summary_path: Path,
    steps_path: Path,
    *,
    dry_run: bool,
    rewrite_set: set,
) -> Optional[EpisodeRederiveResult]:
    if not steps_path.exists():
        print(f"  [SKIP] no step JSONL for {summary_path.name}", file=sys.stderr)
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception as exc:
        print(f"  [SKIP] cannot read {summary_path.name}: {exc}", file=sys.stderr)
        return None

    site = str(summary.get("benchmark_site") or "")
    task_id_raw = summary.get("task_id")
    if not site or task_id_raw is None:
        print(f"  [SKIP] missing site/task_id in {summary_path.name}", file=sys.stderr)
        return None
    try:
        task_id = int(task_id_raw)
    except (TypeError, ValueError):
        return None

    steps = read_jsonl_dedup(steps_path)
    new_pur, energy_partial, energy_complete, n_finish, has_eff = _rederive_one(steps)

    # §95 adjusted_success — re-derive for old data using runner's canonical
    # logic (Step 2 of plan). agent_finished is read from summary if present;
    # otherwise derived from the last step's action_type + fallback_finish.
    adj_success: Optional[bool] = None
    fp_reason = ""
    try:
        from p79.experiment.analysis import compute_adjusted_success, _load_na_task_ids
        # Determine benchmark from run_dir path.
        _bench = "webarena" if any(p == "webarena" for p in summary_path.parts) else "visualwebarena"
        _na_ids = _load_na_task_ids(site, _bench)
        # agent_finished: prefer summary field; fall back to step-derived.
        if "agent_finished" in summary and summary["agent_finished"] is not None:
            af = bool(summary["agent_finished"])
        elif steps:
            _last_at = str((steps[-1].get("action") or {}).get("action_type", "")).lower()
            _last_fb = bool(steps[-1].get("fallback_finish", False))
            af = (_last_at in ("finish", "stop")) and not _last_fb
        else:
            af = None
        # eval_type from summary or empty (older data may not have it).
        et = str(summary.get("eval_type", "") or "")
        adj_success_val, fp_val = compute_adjusted_success(
            task_id, site, bool(summary.get("success", False)),
            na_task_ids=_na_ids,
            agent_finished=af,
            eval_type=et,
            has_effective_action=has_eff,
        )
        adj_success = bool(adj_success_val)
        fp_reason = str(fp_val)
    except Exception as exc:
        print(f"  [WARN] adjusted_success derive failed for {site} task {task_id}: {exc}",
              file=sys.stderr)

    result = EpisodeRederiveResult(
        site=site,
        task_id=task_id,
        old_page_unchanged_rate=summary.get("page_unchanged_rate"),
        new_page_unchanged_rate=new_pur,
        energy_partial=energy_partial,
        energy_step_complete_count=energy_complete,
        n_steps=len(steps),
        n_finish_steps_excluded=n_finish,
        has_effective_action=has_eff,
        adjusted_success=adj_success,
        fp_reason=fp_reason,
    )

    if dry_run:
        return result

    # Backup once (never overwrite existing backup — keeps original truly safe).
    backup_dir = summary_path.parent / _BACKUP_DIR_NAME
    backup_dir.mkdir(exist_ok=True)
    backup_path = backup_dir / summary_path.name
    if not backup_path.exists():
        shutil.copy2(summary_path, backup_path)

    # Apply only the rewrite-eligible fields. Preserve everything else verbatim.
    if "page_unchanged_rate" in rewrite_set:
        summary["page_unchanged_rate"] = new_pur
    if "energy_partial" in rewrite_set:
        summary["energy_partial"] = energy_partial
    if "energy_step_complete_count" in rewrite_set:
        summary["energy_step_complete_count"] = energy_complete
    if "busy_wait_total_ms" in rewrite_set and "busy_wait_total_ms" not in summary:
        # Old data: per-wait latency wasn't recorded. Set 0 + audit marker so
        # it's clear this is "unknown / pre-fix" rather than "no busy waits".
        summary["busy_wait_total_ms"] = 0.0
        summary["busy_wait_total_ms_unknown_pre_fix"] = True
    # §95 adjusted_success fields (Step 2): always update if derivation succeeded.
    if "adjusted_success" in rewrite_set and adj_success is not None:
        summary["adjusted_success"] = adj_success
        summary["fp_reason"] = fp_reason
        summary["has_effective_action"] = has_eff

    condition_dir = summary_path.parent.parent
    logger = LoggerV2(condition_dir)
    logger.write_episode_summary(site, task_id, summary)
    return result


def _process_condition(
    condition_dir: Path,
    *,
    dry_run: bool,
    rewrite_set: set,
    rebuild_aggregate: bool,
) -> Tuple[int, int]:
    """Returns (n_processed, n_changed_rate)."""
    episodes_dir = condition_dir / "episodes"
    if not episodes_dir.is_dir():
        return 0, 0
    summaries = sorted(episodes_dir.glob("*_summary_v2.json"))
    if not summaries:
        return 0, 0

    print(f"\n=== condition: {condition_dir.name} ({len(summaries)} episodes) ===")
    n_processed = 0
    n_changed_rate = 0
    max_delta = 0.0
    for sp in summaries:
        steps_p = sp.with_name(sp.name.replace("_summary_v2.json", "_steps_v2.jsonl"))
        result = _process_episode(sp, steps_p, dry_run=dry_run, rewrite_set=rewrite_set)
        if result is None:
            continue
        n_processed += 1
        old_pur = result.old_page_unchanged_rate
        new_pur = result.new_page_unchanged_rate
        if old_pur is None:
            delta = None
        else:
            delta = new_pur - float(old_pur)
            if abs(delta) > 1e-6:
                n_changed_rate += 1
                if abs(delta) > max_delta:
                    max_delta = abs(delta)

    print(
        f"  rederived: {n_processed} episodes, "
        f"{n_changed_rate} had page_unchanged_rate change (max |Δ|={max_delta:.4f})"
    )

    # Re-aggregate condition_summary_v2.json from the freshly-written episodes.
    if rebuild_aggregate and not dry_run and n_processed > 0:
        ep_summaries: List[Dict[str, Any]] = []
        for sp in sorted(episodes_dir.glob("*_summary_v2.json")):
            try:
                with open(sp, "r", encoding="utf-8") as f:
                    ep_summaries.append(json.load(f))
            except Exception:
                continue
        if ep_summaries:
            aggregate = aggregate_condition_metrics(ep_summaries)
            # Preserve identifying fields from existing condition_summary if present
            existing_path = condition_dir / "condition_summary_v2.json"
            if existing_path.exists():
                try:
                    with open(existing_path, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                    for k in (
                        "condition_id", "seed", "phase", "backend_id",
                        "som_on", "observation_mode", "router_on", "module_flags",
                    ):
                        if k in existing and k not in aggregate:
                            aggregate[k] = existing[k]
                except Exception:
                    pass
            logger = LoggerV2(condition_dir)
            logger.write_condition_summary(aggregate)
            print(f"  rebuilt condition_summary_v2.json (n={len(ep_summaries)})")

    return n_processed, n_changed_rate


def _resolve_run_dirs(args: argparse.Namespace) -> List[Path]:
    if args.run_dir:
        rd = Path(args.run_dir).expanduser().resolve()
        if not rd.is_dir():
            print(f"[ERROR] --run-dir not found: {rd}", file=sys.stderr)
            sys.exit(1)
        return [rd]

    if args.all_b0:
        out: List[Path] = []
        for bench in ("visualwebarena", "webarena"):
            phase_root = _REPO_ROOT / "results" / bench / "phase1"
            if not phase_root.is_dir():
                continue
            for child in sorted(phase_root.iterdir()):
                if child.is_dir() and child.name.startswith("B0_"):
                    out.append(child.resolve())
        if not out:
            print("[ERROR] --all-b0 found no B0_* directories", file=sys.stderr)
            sys.exit(1)
        return out

    print("[ERROR] specify --run-dir <path> or --all-b0", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-derive page_unchanged_rate / energy_partial / "
                    "energy_step_complete_count / busy_wait_total_ms in "
                    "existing episode summaries from step JSONL."
    )
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Single run directory to process")
    parser.add_argument("--all-b0", action="store_true",
                        help="Process all B0_* run dirs under results/{visualwebarena,webarena}/phase1/")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and print stats but don't write files")
    parser.add_argument(
        "--rederive-fields",
        type=str,
        default="page_unchanged_rate,energy_partial,energy_step_complete_count,busy_wait_total_ms,adjusted_success",
        help="Comma-separated subset of fields to rewrite. Default: all five.",
    )
    parser.add_argument("--no-condition-aggregate", action="store_true",
                        help="Skip rebuilding condition_summary_v2.json after episode rewrites")
    args = parser.parse_args()

    rewrite_set = set(x.strip() for x in args.rederive_fields.split(",") if x.strip())
    valid_fields = {"page_unchanged_rate", "energy_partial",
                    "energy_step_complete_count", "busy_wait_total_ms",
                    "adjusted_success"}
    unknown = rewrite_set - valid_fields
    if unknown:
        print(f"[ERROR] unknown rederive-fields: {unknown}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Rederive: dry_run={args.dry_run} fields={sorted(rewrite_set)} "
        f"rebuild_aggregate={not args.no_condition_aggregate}"
    )

    run_dirs = _resolve_run_dirs(args)
    print(f"Processing {len(run_dirs)} run dir(s):")
    for rd in run_dirs:
        print(f"  - {rd}")

    grand_processed = 0
    grand_changed = 0
    for run_dir in run_dirs:
        condition_dirs = sorted(
            d for d in run_dir.iterdir()
            if d.is_dir() and not d.name.startswith(("_", "."))
            and not d.name == "analysis"
        )
        for cond_dir in condition_dirs:
            n_proc, n_changed = _process_condition(
                cond_dir,
                dry_run=args.dry_run,
                rewrite_set=rewrite_set,
                rebuild_aggregate=not args.no_condition_aggregate,
            )
            grand_processed += n_proc
            grand_changed += n_changed

    mode = "DRY-RUN" if args.dry_run else "APPLIED"
    print(
        f"\n[{mode}] total episodes processed: {grand_processed}, "
        f"page_unchanged_rate changed: {grand_changed}"
    )


if __name__ == "__main__":
    main()
