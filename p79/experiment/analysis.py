from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from p79.experiment.io_utils import read_jsonl_dedup
from p79.experiment.metrics import net_saving, net_saving_latency, net_saving_energy

logger = logging.getLogger(__name__)

_CONFIG_BASE = Path(__file__).resolve().parent.parent.parent / "external" / "visualwebarena" / "config_files"


def _get_config_dir(benchmark: str = "visualwebarena") -> Path:
    if benchmark == "webarena":
        return _CONFIG_BASE / "wa"
    return _CONFIG_BASE / "vwa"


def _resolve_site_config(site: str, benchmark: str) -> Optional[Path]:
    """Return the per-site VWA / WA test config path, or None if missing."""
    config_dir = _get_config_dir(benchmark)
    config_path = config_dir / f"test_{site}.json"
    if config_path.exists():
        return config_path
    config_path = config_dir / f"test_{site}.raw.json"
    if config_path.exists():
        return config_path
    return None


def _load_site_tasks(site: str, benchmark: str) -> Optional[list]:
    """Read the per-site VWA / WA test config JSON. Returns None on any error."""
    config_path = _resolve_site_config(site, benchmark)
    if config_path is None:
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning(
            "Failed to parse task config %s: %s",
            config_path, exc,
        )
        return None


def _load_na_task_ids(site: str, benchmark: str = "visualwebarena") -> set:
    """Return task_ids whose reference answer is N/A (unanswerable tasks).

    Post-§139.8 + /stress A1.6 (2026-05-16): N/A definition is single-sourced
    via `p79.experiment.tasks._is_na_task` to avoid DRY drift across the
    task-load exclusion path and the analysis-time fallback path.
    """
    from p79.experiment.tasks import _is_na_task  # local import to avoid cycle

    tasks = _load_site_tasks(site, benchmark)
    if tasks is None:
        logger.warning(
            "N/A task config not found / unreadable for site=%s benchmark=%s; "
            "scored_task_count will fall back to 0 unless strict=True",
            site, benchmark,
        )
        return set()
    return {int(t["task_id"]) for t in tasks if _is_na_task(t)}


def scored_task_count(
    site: str, benchmark: str = "visualwebarena", *, strict: bool = False,
) -> int:
    """Number of tasks in the SCORED set for (site, benchmark).

    §139.8 single source of truth for "EXPECTED_N": total tasks in the site
    config minus the N/A (unanswerable) tasks excluded at load time
    (`task.exclude_na_tasks`, see `tasks.py::load_tasks`). Replaces the
    hardcoded `EXPECTED_N = {classifieds: 234, ...}` dicts scattered across
    analysis + maintenance scripts — those pre-exclusion counts are stale
    once N/A tasks are excluded. Post-exclusion: cls=224, red=205, shop=435,
    wa-shop=173, wa-admin=176, wa-red=104.

    Post-§A1.6 (2026-05-16): `strict=True` raises FileNotFoundError when the
    site config cannot be read, replacing the silent 0-fallback that caused
    paper-grade completion checks to silently mark missing data as complete
    (`run_registry.is_complete`, `fig1ab_cascade_diamond` 200-or-expected
    threshold). Paper-grade paths should pass `strict=True`.
    """
    tasks = _load_site_tasks(site, benchmark)
    if tasks is None:
        if strict:
            raise FileNotFoundError(
                f"scored_task_count: config not found for site={site} "
                f"benchmark={benchmark}; strict mode refuses to fall back to 0"
            )
        logger.warning("scored_task_count: config not found for site=%s benchmark=%s", site, benchmark)
        return 0
    n_total = len(tasks)
    n_na = len(_load_na_task_ids(site, benchmark))
    return max(0, n_total - n_na)


# §139.8: `compute_adjusted_success` + `compute_adjusted_success_batch` were
# retired here. The post-hoc na_fp / eval_fp filter layer is replaced by
# source-level fixes — empty-pred guard in the VWA evaluator (master bug
# B-91) + N/A task exclusion at load time (`tasks.py::load_tasks`,
# `task.exclude_na_tasks`). The runner's `success` is now the canonical
# paper-grade outcome; nothing downstream re-derives an "adjusted" variant.
# See 实验笔记 §139.8 + master_bug_catalog.md §139.8 piece 4.


def _compute_pareto_front(points: List[Dict[str, float]], maximize: str, minimize: str) -> List[int]:
    """Return indices of Pareto-optimal points (maximize one axis, minimize another).

    Sweep order: by `maximize` desc, then `minimize` asc. A point joins the
    front only when its `minimize` is strictly less than the running best —
    `<=` would let dominated points (same minimize, lower maximize) sneak in.

    B-173 (/stress A1.4b-i Claude A8 + gemini C5): when two conditions have
    EXACTLY equal `maximize` AND equal `minimize`, the sort-order winner is
    kept and the tied loser is dropped (because `<` is strict). The standard
    Pareto definition would include both (neither dominates the other). For
    paper figures, this means: if 2 conditions share the same SR and the same
    cost (rare at observed N>=24 cells but possible after rounding), only one
    point is plotted on the front. Callers (paper figure captions) should
    disclose "ties broken by sort order".
    """
    indexed = list(enumerate(points))
    indexed.sort(key=lambda x: (-x[1].get(maximize, 0.0), x[1].get(minimize, 0.0)))
    pareto_indices: List[int] = []
    best_min = float("inf")
    for idx, pt in indexed:
        val = pt.get(minimize, float("inf"))
        if val < best_min:
            pareto_indices.append(idx)
            best_min = val
    pareto_indices.sort(key=lambda i: points[i].get(minimize, 0.0))
    return pareto_indices


def _collect_episode_summaries(run_dir: Path) -> List[Dict[str, Any]]:
    # B-192 (/stress A1.4b-ii Claude D2, P1): actively exercise the
    # schema_migrations framework by running every loaded episode summary
    # through `fill_defaults`. Pre-fix the framework was dead infrastructure
    # (only 1 test referenced its constants for alignment check, ZERO
    # production callers); v3 schema bump would have cold-started under
    # pressure. Now: legacy summaries written before B-166/B-167/B-168
    # telemetry was added receive default-typed values for the new fields,
    # so downstream `aggregate_condition_metrics` (B-193 emits trajectory
    # rates) never sees KeyError. User chose "历史归档" → defaults are
    # baseline-typed (False / empty dict / 0), legacy rows count as
    # "no trajectory_incomplete / no partial_recovery" by convention.
    from p79.experiment.schema_migrations.v2 import (
        EPISODE_SUMMARY_V2_DEFAULTS,
        fill_defaults,
    )
    # B-322 (/stress A1.9 Mode A F3 + Mode B F5 OOB, 2026-05-16): use strict
    # loader to enforce bool type on `success` field at load boundary.
    # Pre-fix `json.load(f)` returned whatever JSON literal was on disk
    # (`"false"` string → Python `bool("false") = True` → paper §1 SR
    # inflated when downstream `astype(bool).astype(int)` coerced).
    # Pairs with `aggregate_condition_metrics` entry strict-type-check
    # (B-322 metrics.py) for defense-in-depth — analysis.py 3-way coercion
    # drift (line 596 pd.to_numeric / line 715 astype(bool) / line 1148
    # pd.to_numeric) all now operate on already-validated bool source.
    from p79.experiment.io_utils import load_episode_summary_strict
    rows: List[Dict[str, Any]] = []
    # B-596 (/stress A1.6a P0-1-ABC* 3-AI overlap OOB, 2026-05-17):
    # strict-loader 三层 contract fully wired. Pre-fix used
    # `mode="lenient"` without `reject_needs_reevaluation=True` AND
    # fell back to raw `json.load(f)` on any Exception → B-322 bool-type
    # guard + B-542 quarantine reject both silently bypassed
    # (`"success": "false"` Python-truthy AND B-486 quarantined episodes
    # with `needs_reevaluation=True` entered analysis as `success=False`
    # — bookkeeping, not real evaluator outcome). The Exception fallback
    # had no defensible failure mode: lenient mode already returns None
    # for JSONDecodeError + type-mismatch; the remaining Exception
    # surface is race-condition FileNotFoundError, which now propagates
    # correctly (path deleted mid-iterate IS abnormal, fail-loud is
    # right). Pairs with `aggregate_condition_metrics` entry strict-type
    # check for defense-in-depth.
    for summary_path in run_dir.glob("*/episodes/*_summary_v2.json"):
        raw = load_episode_summary_strict(
            summary_path, mode="lenient", reject_needs_reevaluation=True,
        )
        if raw is None:
            continue
        rows.append(fill_defaults(raw, EPISODE_SUMMARY_V2_DEFAULTS))
    return rows


def _synthesize_condition_summary(
    cond_dir: Path, ep_summaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a condition summary from condition_meta + episode summaries.

    Used for in-progress conditions that haven't finished yet.

    B-179 (/stress A1.4b-i Claude A3 + codex B6 + gemini C4, P0 triple-cross):
    The pre-§A1.4b-i implementation hand-aggregated success_rate / avg_steps /
    p95_latency / cost / energy with `e.get(k, 0) or 0` (silent None→0 →
    B0 cost / energy systematic underestimation) AND was schema-incomplete vs
    `aggregate_condition_metrics` (`metrics.py:263-396`): missing
    `avg_total_latency_ms`, `avg_obs_prepare_cost_usd`, `avg_input_cost_usd`,
    `avg_output_cost_usd`, `avg_busy_wait_total_ms`, `energy_partial_*`. It
    also hard-zeroed `avg_router_overhead_cost_usd` / `wasted_*` /
    `benchmark_noise_rate` / `cost_efficiency_ratio` → partial-condition rows
    looked artificially clean. Headline `_plot_phase1` consumed mixed
    partial+complete rows with no visual distinction.

    Fix: delegate aggregation to the same `aggregate_condition_metrics` the
    runner uses on completion, then overlay condition_meta + the `_synthesized`
    flag (so downstream plotters can visually distinguish; see B-179 plot edit
    in `_plot_phase1`).
    """
    meta_path = cond_dir / "condition_meta.json"
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if not ep_summaries:
        return {}

    # Canonical aggregator path: identical schema to completed condition_summary_v2.
    from p79.experiment.metrics import aggregate_condition_metrics
    canonical = aggregate_condition_metrics(ep_summaries)

    payload = {
        "condition_id": meta.get("condition_id", cond_dir.name),
        "seed": meta.get("seed", 42),
        "phase": meta.get("phase", "phase1"),
        "backend_id": meta.get("backend_id", ""),
        "som_on": meta.get("som_on", False),
        "observation_mode": meta.get("observation_mode", "unknown"),
        "router_on": meta.get("router_on", False),
        "module_flags": meta.get("modules", {}),
        **canonical,
        "_synthesized": True,  # B-179: downstream plotters MUST honor this flag.
    }
    return payload


# B-601 (/stress A1.6a P1-3-BC codex+gemini overlap + user Q3=(B) gate,
# 2026-05-17): pre-§139.8 archives carry retired post-hoc FP fields
# (`adjusted_success` / `fp_reason` / `raw_success` / `is_na_reference`).
# Phase 1 rerun is canonical (user directive 2026-05-17 "archive 全不可信 →
# Phase 1 rerun"). Loading stale archive silently into the analysis
# pipeline = paper-grade contamination vector. Hard gate: refuse archive
# containing any retired field; user opt-in via env `P79_ANALYZE_ALLOW_STALE_ARCHIVE=1`
# for legacy-archive triage only.
_RETIRED_POST_HOC_FP_FIELDS = (
    "adjusted_success",
    "fp_reason",
    "raw_success",
    "is_na_reference",
    "adjusted_reason_bucket",
)


def _is_post_fp_retirement_archive(data: Dict[str, Any]) -> bool:
    """True if condition_summary_v2.json contains NO retired post-hoc FP fields.

    B-601 gate: §139.8 retired the post-hoc filtering layer (B-91 upstream
    LLM-judge guard + `exclude_na_tasks: true` load-time exclude replace
    `compute_adjusted_success`). Canonical post-§139.8 archive has only
    `success` (no override). Any presence of retired fields = pre-§139.8
    archive = REFUSE unless user explicitly opts in.
    """
    return not any(k in data for k in _RETIRED_POST_HOC_FP_FIELDS)


def _collect_condition_summaries(
    run_dir: Path, *, allow_stale_archive: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Collect per-condition rows. Gates stale archives by default (B-601).

    Args:
        run_dir: analysis target directory.
        allow_stale_archive: if True, bypass §139.8 retired-field gate
            (legacy archive triage only). If None, reads
            `P79_ANALYZE_ALLOW_STALE_ARCHIVE` env (default off).
    """
    import os

    if allow_stale_archive is None:
        allow_stale_archive = os.environ.get(
            "P79_ANALYZE_ALLOW_STALE_ARCHIVE", ""
        ) == "1"

    rows: List[Dict[str, Any]] = []
    seen_conds: set = set()
    refused_rows: List[Dict[str, Any]] = []  # B-601 audit
    # 1. Completed conditions (have condition_summary_v2.json)
    for summary_path in run_dir.glob("*/condition_summary_v2.json"):
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # B-601 schema-version gate (Q3=(B), 2026-05-17 user directive).
        if not _is_post_fp_retirement_archive(data):
            offending = [k for k in _RETIRED_POST_HOC_FP_FIELDS if k in data]
            if not allow_stale_archive:
                logger.warning(
                    "B-601 REFUSED stale archive (pre-§139.8 fields %s) at %s; "
                    "set P79_ANALYZE_ALLOW_STALE_ARCHIVE=1 to bypass (legacy "
                    "triage only — Phase 1 rerun is canonical).",
                    offending, summary_path,
                )
                refused_rows.append({
                    "summary_path": str(summary_path),
                    "retired_fields_found": ",".join(offending),
                    "schema_version": str(data.get("schema_version")),
                })
                seen_conds.add(summary_path.parent.name)  # don't synthesize fallback
                continue
            else:
                logger.warning(
                    "B-601 ALLOWED stale archive (P79_ANALYZE_ALLOW_STALE_ARCHIVE=1) "
                    "with retired fields %s at %s — paper-grade not safe.",
                    offending, summary_path,
                )
        rows.append(data)
        seen_conds.add(summary_path.parent.name)

    # B-601 emit refused-archive audit alongside the analysis dir.
    if refused_rows:
        try:
            import pandas as pd  # type: ignore
            (run_dir / "analysis").mkdir(parents=True, exist_ok=True)
            pd.DataFrame(refused_rows).to_csv(
                run_dir / "analysis" / "archive_refused_b582.csv", index=False,
            )
        except Exception as exc:
            logger.warning("B-601 failed to write archive_refused audit: %s", exc)

    # 2. In-progress conditions (have condition_meta.json but no summary)
    for meta_path in run_dir.glob("*/condition_meta.json"):
        cond_dir = meta_path.parent
        if cond_dir.name in seen_conds:
            continue
        if cond_dir.name.startswith("_"):
            continue
        # Load episode summaries for this condition
        ep_summaries = []
        for ep_path in cond_dir.glob("episodes/*_summary_v2.json"):
            with open(ep_path, "r", encoding="utf-8") as f:
                ep_summaries.append(json.load(f))
        if not ep_summaries:
            continue
        synth = _synthesize_condition_summary(cond_dir, ep_summaries)
        if synth:
            logger.info(
                "Synthesized condition summary for %s (%d episodes, in-progress)",
                cond_dir.name, len(ep_summaries),
            )
            rows.append(synth)

    return rows


def _collect_step_records(run_dir: Path) -> List[Dict[str, Any]]:
    # B-180 (codex B7): pass sibling summary_v2.json for identity check.
    # `*_steps_v2.jsonl` ↔ `*_summary_v2.json` naming convention is
    # established by `runner/main.py::_run_and_record_episode`; the
    # identity check warns (does not raise) on mismatch so analysis can
    # still proceed but audit can see restart-crash divergence.
    #
    # B-599 (/stress A1.6a P1-1-BC codex+gemini overlap, 2026-05-17):
    # `strict_identity=True` enforces B-571 (/stress A1.22) paper-grade
    # fail-loud contract. Pre-fix called `read_jsonl_dedup` with the
    # default `strict_identity=False` (io_utils.py:163), so restart-crash
    # tail mismatch was a warning only — step-level cost / latency /
    # trigger / checklist / state-change diagnostics could be computed
    # from a different attempt than the authoritative summary. Paper-
    # grade analysis call sites MUST fail-loud; opt out via env
    # `P79_STRICT_READ_JSONL=0` for legacy-archive triage only.
    rows: List[Dict[str, Any]] = []
    for step_path in run_dir.glob("*/episodes/*_steps_v2.jsonl"):
        summary_path = step_path.with_name(step_path.name.replace("_steps_v2.jsonl", "_summary_v2.json"))
        rows.extend(read_jsonl_dedup(
            step_path, summary_path=summary_path, strict_identity=True,
        ))
    return rows


_TO_MAPPING_PARSE_FAILURES: List[Dict[str, Any]] = []


def _to_mapping(value: Any, context: Optional[str] = None) -> Dict[str, Any]:
    # B-174 (/stress A1.4b-i Claude A9, OOB): pre-fix swallowed `json.loads`
    # exceptions silently → pivot tables for trigger_distribution /
    # state_change_reason_distribution showed zero counts when input was
    # malformed JSON-string, reader 误读为 "no trigger fired". Log + collect
    # so audit can see what got dropped; `analyze_run` emits parse_failures.csv.
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
            logger.warning(
                "_to_mapping: parsed JSON is not a dict (got %s); context=%s",
                type(parsed).__name__, context or "(unspecified)",
            )
            _TO_MAPPING_PARSE_FAILURES.append({
                "context": context or "(unspecified)",
                "reason": f"non_dict_{type(parsed).__name__}",
                "value_snippet": value[:120],
            })
            return {}
        except Exception as exc:
            logger.warning(
                "_to_mapping: JSON parse failed (%s); context=%s",
                exc, context or "(unspecified)",
            )
            _TO_MAPPING_PARSE_FAILURES.append({
                "context": context or "(unspecified)",
                "reason": f"json_parse_error:{type(exc).__name__}",
                "value_snippet": value[:120] if isinstance(value, str) else repr(value)[:120],
            })
            return {}
    return {}


def _flatten_state_change_reasons(cond_df) -> Any:
    import pandas as pd  # type: ignore

    rows: List[Dict[str, Any]] = []
    if cond_df.empty or "state_change_reason_distribution" not in cond_df.columns:
        return pd.DataFrame(columns=["condition_id", "reason", "count"])

    for _, row in cond_df.iterrows():
        cid = row.get("condition_id", "?")
        dist = _to_mapping(
            row.get("state_change_reason_distribution", {}),
            context=f"state_change_reason_distribution@cond={cid}",
        )
        for reason, count in dist.items():
            try:
                rows.append(
                    {
                        "condition_id": row.get("condition_id"),
                        "reason": str(reason),
                        "count": int(count),
                    }
                )
            except Exception:
                continue

    if not rows:
        return pd.DataFrame(columns=["condition_id", "reason", "count"])
    return pd.DataFrame(rows)


def _flatten_trigger_distribution(cond_df) -> Any:
    import pandas as pd  # type: ignore

    rows: List[Dict[str, Any]] = []
    if cond_df.empty or "trigger_distribution" not in cond_df.columns:
        return pd.DataFrame(columns=["condition_id", "trigger", "count"])

    for _, row in cond_df.iterrows():
        cid = row.get("condition_id", "?")
        dist = _to_mapping(
            row.get("trigger_distribution", {}),
            context=f"trigger_distribution@cond={cid}",
        )
        for trigger, count in dist.items():
            try:
                rows.append(
                    {
                        "condition_id": row.get("condition_id"),
                        "trigger": str(trigger),
                        "count": int(count),
                    }
                )
            except Exception:
                continue

    if not rows:
        return pd.DataFrame(columns=["condition_id", "trigger", "count"])
    return pd.DataFrame(rows)


def _analyze_checklist(step_df, ep_df, plots_dir: Path, tables_dir: Path) -> None:
    import matplotlib.pyplot as plt  # type: ignore
    import pandas as pd  # type: ignore

    curve_path = tables_dir / "checklist_progress_curve.csv"
    fail_path = tables_dir / "checklist_failure_distribution.csv"

    # Skip entirely if checklist column is absent or all-None (e.g. VWA doesn't log checklists)
    has_checklist_data = (
        not step_df.empty
        and "checklist" in step_df.columns
        and step_df["checklist"].notna().any()
    )
    if not has_checklist_data:
        return

    step_df = step_df.copy()
    step_df["checklist_status"] = step_df["checklist"].apply(lambda x: _to_mapping(x).get("status", {}))
    step_df["checklist_completion_rate"] = step_df["checklist_status"].apply(
        lambda x: _to_mapping(x).get("completion_rate")
    )
    step_df["checklist_failed"] = step_df["checklist_status"].apply(lambda x: _to_mapping(x).get("failed"))
    step_df["checklist_completion_rate"] = pd.to_numeric(step_df["checklist_completion_rate"], errors="coerce")
    step_df["checklist_failed"] = pd.to_numeric(step_df["checklist_failed"], errors="coerce")

    progress_df = step_df[step_df["checklist_completion_rate"].notna()].copy()
    if progress_df.empty:
        pd.DataFrame(columns=["condition_id", "step_idx", "avg_completion_rate"]).to_csv(curve_path, index=False)
    else:
        curve_df = (
            progress_df.groupby(["condition_id", "step_idx"], as_index=False)["checklist_completion_rate"]
            .mean()
            .rename(columns={"checklist_completion_rate": "avg_completion_rate"})
        )
        curve_df.to_csv(curve_path, index=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        for cond_id, grp in curve_df.groupby("condition_id"):
            grp = grp.sort_values("step_idx")
            ax.plot(grp["step_idx"], grp["avg_completion_rate"], marker="o", label=str(cond_id))
        ax.set_xlabel("Step Index")
        ax.set_ylabel("Checklist Completion Rate")
        ax.set_ylim(0, 1)
        ax.set_title("Checklist Progress Curve")
        ax.grid(alpha=0.3)
        if curve_df["condition_id"].nunique() <= 12:
            ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "checklist_progress_curve.png")
        plt.close(fig)

    if not ep_df.empty and {"condition_id", "checklist_failed_items", "checklist_completion_rate"}.issubset(ep_df.columns):
        summary = (
            ep_df.groupby("condition_id", as_index=False)
            .agg(
                episodes=("task_id", "count"),
                episodes_with_failed=("checklist_failed_items", lambda s: int((s.fillna(0) > 0).sum())),
                avg_completion_rate=("checklist_completion_rate", "mean"),
            )
            .fillna({"avg_completion_rate": 0.0})
        )
    else:
        # Fallback from step-level snapshots.
        tail = step_df.sort_values("step_idx").groupby(
            ["condition_id", "benchmark_site", "task_id"], as_index=False
        ).tail(1)
        summary = (
            tail.groupby("condition_id", as_index=False)
            .agg(
                episodes=("task_id", "count"),
                episodes_with_failed=("checklist_failed", lambda s: int((s.fillna(0) > 0).sum())),
                avg_completion_rate=("checklist_completion_rate", "mean"),
            )
            .fillna({"avg_completion_rate": 0.0})
        )

    if summary.empty:
        summary = pd.DataFrame(
            columns=["condition_id", "episodes", "episodes_with_failed", "failure_rate", "avg_completion_rate"]
        )
    else:
        summary["failure_rate"] = summary.apply(
            lambda r: (float(r["episodes_with_failed"]) / float(r["episodes"])) if float(r["episodes"]) > 0 else 0.0,
            axis=1,
        )
    summary.to_csv(fail_path, index=False)

    if not summary.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(summary["condition_id"], summary["failure_rate"])
        ax.set_ylabel("Checklist Failure Episode Rate")
        ax.set_xlabel("Condition")
        ax.set_ylim(0, 1)
        ax.set_title("Checklist Failure Distribution")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(plots_dir / "checklist_failure_distribution.png")
        plt.close(fig)


def _plot_state_change_reason_distribution(cond_df, plots_dir: Path, tables_dir: Path, phase: str) -> None:
    import matplotlib.pyplot as plt  # type: ignore
    import pandas as pd  # type: ignore

    flat_df = _flatten_state_change_reasons(cond_df)
    flat_df.to_csv(tables_dir / "state_change_reason_distribution.csv", index=False)
    if flat_df.empty:
        return

    pivot = flat_df.pivot_table(index="condition_id", columns="reason", values="count", aggfunc="sum", fill_value=0)
    pivot.to_csv(tables_dir / "state_change_reason_distribution_pivot.csv")

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Reason Count")
    ax.set_xlabel("Condition")
    ax.set_title("State-Change Reason Distribution")
    ax.legend(title="Reason", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(plots_dir / "state_change_reason_distribution.png")
    plt.close(fig)

    if phase == "phase2":
        fig, ax = plt.subplots(figsize=(9, 5))
        pivot.plot(kind="bar", stacked=True, ax=ax)
        ax.set_ylabel("Reason Count")
        ax.set_xlabel("Condition")
        ax.set_title("Phase2 State-Change Reason Distribution")
        ax.legend(title="Reason", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(plots_dir / "phase2_state_change_reason_distribution.png")
        plt.close(fig)
    elif phase == "phase3":
        fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(pivot.columns) + 4), 5))
        im = ax.imshow(pivot.values, cmap="YlOrRd")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns], rotation=30, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(i) for i in pivot.index])
        ax.set_xlabel("State-Change Reason")
        ax.set_ylabel("Condition")
        ax.set_title("Phase3 State-Change Reason Heatmap")
        fig.colorbar(im, ax=ax, label="Count")
        fig.tight_layout()
        fig.savefig(plots_dir / "phase3_state_change_reason_heatmap.png")
        plt.close(fig)


def _plot_trigger_distribution(cond_df, plots_dir: Path, tables_dir: Path, phase: str) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    flat_df = _flatten_trigger_distribution(cond_df)
    flat_df.to_csv(tables_dir / "trigger_distribution.csv", index=False)
    if flat_df.empty:
        return

    pivot = flat_df.pivot_table(index="condition_id", columns="trigger", values="count", aggfunc="sum", fill_value=0)
    pivot.to_csv(tables_dir / "trigger_distribution_pivot.csv")

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Trigger Count")
    ax.set_xlabel("Condition")
    ax.set_title("Trigger Distribution")
    ax.legend(title="Trigger", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(plots_dir / "trigger_distribution.png")
    plt.close(fig)

    if phase == "phase2":
        fig, ax = plt.subplots(figsize=(9, 5))
        pivot.plot(kind="bar", stacked=True, ax=ax)
        ax.set_ylabel("Trigger Count")
        ax.set_xlabel("Condition")
        ax.set_title("Phase2 Trigger Distribution")
        ax.legend(title="Trigger", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(plots_dir / "phase2_trigger_distribution.png")
        plt.close(fig)


def _emit_benchmark_noise_report(ep_df, tables_dir: Path) -> None:
    """Write `benchmark_noise_report.csv` with per-category counts.

    B-600 (/stress A1.6a P1-2-AC Claude+codex overlap, 2026-05-17):
    extracted from duplicated logic at `_analyze_condition` (per-condition
    path) + `analyze_run` (overview path) — pre-fix two copies could
    silently drift under maintenance. Single source. Always writes the
    CSV (empty-frame if no noise rows) so downstream readers can
    distinguish "no noise data column" from "noise data column present
    but empty". `benchmark_noise` is infra-noise (api_rate_limit /
    playwright_crash etc), NOT N/A FP / eval FP (those are §139.8
    upstream-fixed); see `metrics.py::clean_success_rate` docstring for
    the estimand framework — this report is paper §3 transparency
    appendix only.
    """
    import pandas as pd  # type: ignore

    if ep_df is None or ep_df.empty or "benchmark_noise" not in ep_df.columns:
        pd.DataFrame(columns=["benchmark_noise_category", "count"]).to_csv(
            tables_dir / "benchmark_noise_report.csv", index=False,
        )
        return
    noise_df = ep_df[ep_df["benchmark_noise"] == True]  # noqa: E712
    if noise_df.empty:
        pd.DataFrame(columns=["benchmark_noise_category", "count"]).to_csv(
            tables_dir / "benchmark_noise_report.csv", index=False,
        )
        return
    noise_counts = (
        noise_df.groupby("benchmark_noise_category").size().reset_index(name="count")
    )
    noise_counts.to_csv(tables_dir / "benchmark_noise_report.csv", index=False)


def _analyze_condition(
    cond_id: str,
    cond_dir: Path,
    ep_rows: List[Dict[str, Any]],
    step_rows: List[Dict[str, Any]],
    phase: str,
    run_dir: Optional[Path] = None,
) -> None:
    """Generate analysis for a single condition into analysis/<cond_id>/."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import pandas as pd  # type: ignore
    except Exception:
        return

    out_dir = cond_dir
    plots_dir = out_dir / "plots"
    tables_dir = out_dir / "tables"
    for d in (out_dir, plots_dir, tables_dir):
        d.mkdir(parents=True, exist_ok=True)
    # Remove stale plots/tables from previous runs so no stale files linger
    for _f in plots_dir.glob("*.png"):
        _f.unlink(missing_ok=True)
    for _f in tables_dir.glob("*.csv"):
        _f.unlink(missing_ok=True)

    ep_df = pd.DataFrame(ep_rows)
    step_df = pd.DataFrame(step_rows)

    if not ep_df.empty:
        ep_df.to_csv(tables_dir / "episode_metrics.csv", index=False)
    if not step_df.empty:
        step_df.to_csv(tables_dir / "step_metrics.csv", index=False)

    # B-600: noise report via single source helper (DRY).
    _emit_benchmark_noise_report(ep_df, tables_dir)

    if ep_df.empty:
        return

    # B-603 (/stress A1.6a P1-5-A* Claude OOB, Q4=Delete user directive
    # 2026-05-17): cumulative success rate plot deleted. Pre-fix:
    # `sort_values("task_id")` + `expanding().mean()` produced what looked
    # like a "learning curve" but was actually cumulative SR ordered by
    # VWA-assigned task_id (arbitrary source-file index, NOT
    # chronological / completion-time / difficulty order). Reader / advisor
    # / reviewer misread risk: "agent learns over the run". Plot was not
    # consumed by paper §3 figures. Deleted entirely rather than caption-
    # disclosure (which would still leave the misleading shape in
    # `cumulative_success_rate.png` outputs).

    # --- Step count distribution ---
    if "steps" in ep_df.columns:
        steps_series = pd.to_numeric(ep_df["steps"], errors="coerce").dropna()
        if not steps_series.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(steps_series, bins=20, edgecolor="black", alpha=0.75)
            ax.axvline(steps_series.mean(), color="red", linestyle="--", label=f"mean={steps_series.mean():.1f}")
            ax.set_xlabel("Steps per Episode")
            ax.set_ylabel("Count")
            ax.set_title(f"Step Count Distribution — {cond_id}")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(plots_dir / "step_count_distribution.png")
            plt.close(fig)

    # --- Cost distribution ---
    if "total_cost_usd" in ep_df.columns:
        cost_series = pd.to_numeric(ep_df["total_cost_usd"], errors="coerce").dropna()
        if not cost_series.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(cost_series, bins=20, edgecolor="black", alpha=0.75, color="#DD8452")
            ax.axvline(cost_series.mean(), color="red", linestyle="--", label=f"mean=${cost_series.mean():.4f}")
            ax.set_xlabel("Total Cost per Episode (USD)")
            ax.set_ylabel("Count")
            ax.set_title(f"Cost Distribution — {cond_id}")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(plots_dir / "cost_distribution.png")
            plt.close(fig)

    # --- Latency distribution ---
    if "p95_step_latency_ms" in ep_df.columns:
        lat_series = pd.to_numeric(ep_df["p95_step_latency_ms"], errors="coerce").dropna() / 1000.0
        if not lat_series.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(lat_series, bins=20, edgecolor="black", alpha=0.75, color="#55A868")
            ax.axvline(lat_series.mean(), color="red", linestyle="--", label=f"mean={lat_series.mean():.1f}s")
            # B-183 (/stress A1.4b-i Claude A7): each x-axis value is the
            # per-episode P95, NOT a per-step latency. The histogram shows
            # distribution OF episode-P95s, not all step latencies — Jensen-like
            # inequality means this is generally larger than overall p95 of all
            # steps. See `metrics.py:344-347` docstring for the rationale;
            # `avg_total_latency_ms` is the proper per-episode end-to-end
            # measure subtractable across conditions for net-saving claims.
            ax.set_xlabel("Per-episode P95 step latency (s)\n(NOT per-step distribution)")
            ax.set_ylabel("Episode count")
            ax.set_title(f"Per-episode P95 Latency Distribution — {cond_id}")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(plots_dir / "latency_distribution.png")
            plt.close(fig)

    # --- State-change reason distribution ---
    if "state_change_reason_distribution" in ep_df.columns:
        reason_rows: List[Dict[str, Any]] = []
        for _, row in ep_df.iterrows():
            dist = _to_mapping(row.get("state_change_reason_distribution", {}))
            for reason, count in dist.items():
                try:
                    reason_rows.append({"reason": str(reason), "count": int(count)})
                except Exception:
                    continue
        if reason_rows:
            r_df = pd.DataFrame(reason_rows).groupby("reason", as_index=False)["count"].sum()
            r_df = r_df.sort_values("count", ascending=True)
            r_df.to_csv(tables_dir / "state_change_reason_distribution.csv", index=False)
            fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(r_df))))
            ax.barh(r_df["reason"], r_df["count"], color="#4C72B0")
            ax.set_xlabel("Total Count")
            ax.set_title(f"State-Change Reasons — {cond_id}")
            fig.tight_layout()
            fig.savefig(plots_dir / "state_change_reason_distribution.png")
            plt.close(fig)

    # --- Trigger distribution ---
    if "trigger_distribution" in ep_df.columns:
        trig_rows: List[Dict[str, Any]] = []
        for _, row in ep_df.iterrows():
            dist = _to_mapping(row.get("trigger_distribution", {}))
            for trig, count in dist.items():
                try:
                    trig_rows.append({"trigger": str(trig), "count": int(count)})
                except Exception:
                    continue
        if trig_rows:
            t_df = pd.DataFrame(trig_rows).groupby("trigger", as_index=False)["count"].sum()
            t_df = t_df.sort_values("count", ascending=True)
            t_df.to_csv(tables_dir / "trigger_distribution.csv", index=False)
            fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(t_df))))
            ax.barh(t_df["trigger"], t_df["count"], color="#C44E52")
            ax.set_xlabel("Total Count")
            ax.set_title(f"Trigger Distribution — {cond_id}")
            fig.tight_layout()
            fig.savefig(plots_dir / "trigger_distribution.png")
            plt.close(fig)

    # --- Success vs steps (bar chart: success rate per step-count bucket) ---
    if {"success", "steps"}.issubset(ep_df.columns):
        import numpy as np
        sv_df = ep_df[["steps", "success"]].copy()
        sv_df["steps"] = pd.to_numeric(sv_df["steps"], errors="coerce")
        sv_df["success_int"] = sv_df["success"].astype(bool).astype(int)
        sv_df = sv_df.dropna(subset=["steps"])
        if not sv_df.empty:
            # Group by step bucket (bins of 5)
            max_steps = int(sv_df["steps"].max())
            bins = list(range(0, max_steps + 6, 5))
            sv_df["step_bucket"] = pd.cut(sv_df["steps"], bins=bins, right=True)
            bucket_stats = sv_df.groupby("step_bucket", observed=True).agg(
                episodes=("success_int", "count"),
                successes=("success_int", "sum"),
            )
            bucket_stats["success_rate"] = bucket_stats["successes"] / bucket_stats["episodes"].replace(0, float("nan"))
            bucket_labels = [str(b) for b in bucket_stats.index]

            fig, ax1 = plt.subplots(figsize=(9, 4))
            x = np.arange(len(bucket_stats))
            bars = ax1.bar(x, bucket_stats["episodes"], color="#4C72B0", alpha=0.6, label="Episodes")
            ax2 = ax1.twinx()
            ax2.plot(x, bucket_stats["success_rate"], color="#C44E52", marker="o", linewidth=2, label="Success Rate")
            ax2.set_ylim(0, 1)
            ax2.set_ylabel("Success Rate", color="#C44E52")
            ax1.set_ylabel("Episode Count")
            ax1.set_xticks(x)
            ax1.set_xticklabels(bucket_labels, rotation=30, ha="right")
            ax1.set_xlabel("Steps (bucket)")
            ax1.set_title(f"Outcome vs Steps — {cond_id}")
            fig.tight_layout()
            fig.savefig(plots_dir / "outcome_vs_steps.png")
            plt.close(fig)

    # --- Energy / CO₂ distribution ---
    has_energy = "total_energy_kwh" in ep_df.columns and pd.to_numeric(ep_df["total_energy_kwh"], errors="coerce").notna().any()
    has_co2 = "total_co2e_kg" in ep_df.columns and pd.to_numeric(ep_df["total_co2e_kg"], errors="coerce").notna().any()
    if has_energy or has_co2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        if has_energy:
            e_series = pd.to_numeric(ep_df["total_energy_kwh"], errors="coerce").dropna()
            axes[0].hist(e_series, bins=20, edgecolor="black", alpha=0.75, color="#55A868")
            axes[0].axvline(e_series.mean(), color="red", linestyle="--", label=f"mean={e_series.mean():.4f}")
            axes[0].set_xlabel("Total Energy per Episode (kWh)")
            axes[0].set_ylabel("Count")
            axes[0].set_title(f"Energy — {cond_id}")
            axes[0].legend()
            axes[0].grid(alpha=0.3)
        else:
            axes[0].set_visible(False)
        if has_co2:
            c_series = pd.to_numeric(ep_df["total_co2e_kg"], errors="coerce").dropna()
            axes[1].hist(c_series, bins=20, edgecolor="black", alpha=0.75, color="#8172B2")
            axes[1].axvline(c_series.mean(), color="red", linestyle="--", label=f"mean={c_series.mean():.6f}")
            axes[1].set_xlabel("Total CO₂e per Episode (kg)")
            axes[1].set_ylabel("Count")
            axes[1].set_title(f"CO₂e — {cond_id}")
            axes[1].legend()
            axes[1].grid(alpha=0.3)
        else:
            axes[1].set_visible(False)
        fig.tight_layout()
        fig.savefig(plots_dir / "energy_co2_distribution.png")
        plt.close(fig)

    # --- Token distribution ---
    if "total_tokens" in ep_df.columns:
        tok_series = pd.to_numeric(ep_df["total_tokens"], errors="coerce").dropna()
        if not tok_series.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(tok_series, bins=20, edgecolor="black", alpha=0.75, color="#4C72B0")
            ax.axvline(tok_series.mean(), color="red", linestyle="--", label=f"mean={tok_series.mean():.0f}")
            ax.set_xlabel("Total Tokens per Episode")
            ax.set_ylabel("Count")
            ax.set_title(f"Token Distribution — {cond_id}")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(plots_dir / "token_distribution.png")
            plt.close(fig)

    # --- No-op rate / retries distribution ---
    has_noop = "no_op_rate" in ep_df.columns and pd.to_numeric(ep_df["no_op_rate"], errors="coerce").notna().any()
    has_retries = "retries" in ep_df.columns and pd.to_numeric(ep_df["retries"], errors="coerce").notna().any()
    if has_noop or has_retries:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        if has_noop:
            n_series = pd.to_numeric(ep_df["no_op_rate"], errors="coerce").dropna()
            axes[0].hist(n_series, bins=20, edgecolor="black", alpha=0.75, color="#DD8452")
            axes[0].axvline(n_series.mean(), color="red", linestyle="--", label=f"mean={n_series.mean():.3f}")
            axes[0].set_xlabel("No-op Rate per Episode")
            axes[0].set_ylabel("Count")
            axes[0].set_title(f"No-op Rate — {cond_id}")
            axes[0].legend()
            axes[0].grid(alpha=0.3)
        else:
            axes[0].set_visible(False)
        if has_retries:
            r_series = pd.to_numeric(ep_df["retries"], errors="coerce").dropna().astype(int)
            retry_counts = r_series.value_counts().sort_index()
            axes[1].bar(retry_counts.index.astype(str), retry_counts.values, color="#C44E52", edgecolor="black", alpha=0.75)
            axes[1].set_xlabel("Retries per Episode")
            axes[1].set_ylabel("Count")
            axes[1].set_title(f"Retry Distribution — {cond_id}")
            axes[1].grid(alpha=0.3, axis="y")
        else:
            axes[1].set_visible(False)
        fig.tight_layout()
        fig.savefig(plots_dir / "noop_retry_distribution.png")
        plt.close(fig)

    # Condition-level summary JSON
    cond_summary_path = (run_dir / cond_id / "condition_summary_v2.json") if run_dir else Path("/nonexistent")
    summary: Dict[str, Any] = {}
    if cond_summary_path.exists():
        with open(cond_summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

    # Compute avg_total_tokens from episode data (not in condition_summary_v2)
    avg_total_tokens = None
    if not ep_df.empty and "total_tokens" in ep_df.columns:
        tok_vals = pd.to_numeric(ep_df["total_tokens"], errors="coerce").dropna()
        if not tok_vals.empty:
            avg_total_tokens = float(tok_vals.mean())

    with open(out_dir / "session_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "condition_id": cond_id,
                "phase": phase,
                "episode_count": int(len(ep_df)),
                "step_count": int(len(step_df)),
                "success_rate": summary.get("success_rate"),
                "avg_total_cost_usd": summary.get("avg_total_cost_usd"),
                "p95_step_latency_ms": summary.get("p95_step_latency_ms"),
                "avg_steps": summary.get("avg_steps"),
                "avg_no_op_rate": summary.get("avg_no_op_rate"),
                "avg_page_unchanged_rate": summary.get("avg_page_unchanged_rate"),
                "avg_retries": summary.get("avg_retries"),
                "avg_total_energy_kwh": summary.get("avg_total_energy_kwh"),
                "avg_total_co2e_kg": summary.get("avg_total_co2e_kg"),
                "avg_escalation_count": summary.get("avg_escalation_count"),
                "avg_total_tokens": avg_total_tokens,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


def _compute_statistical_tests(
    cond_df,
    ep_df,
    reports_dir: Path,
    tables_dir: Path,
) -> None:
    """Compute bootstrap CIs, McNemar's test, and Wilcoxon signed-rank tests."""
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
        from scipy import stats as scipy_stats  # type: ignore
    except ImportError:
        logger.warning("numpy/pandas/scipy not installed; skipping statistical tests")
        return

    if ep_df.empty or "condition_id" not in ep_df.columns:
        return

    cond_ids = cond_df["condition_id"].tolist() if not cond_df.empty and "condition_id" in cond_df.columns else ep_df["condition_id"].unique().tolist()
    results: Dict[str, Any] = {"bootstrap_ci": {}, "mcnemar": {}, "wilcoxon": {}, "notes": []}
    flat_rows: List[Dict[str, Any]] = []

    # B-176 (/stress A1.4b-i codex B9): bootstrap RNG seed pinned to 42 +
    # B=10_000 for run-to-run reproducibility. Paper §3.5 should disclose
    # ("All bootstrap CIs use task-level resampling with B=10000 and
    # analysis RNG seed 42; scripts/analysis/aggregate_phantom_lift.py +
    # aggregate_phantom_meta.py share the same default seed.").
    #
    # B-597 (/stress A1.6a P0-2-C* gemini OOB, Q1=A user picked
    # caption-disclosure 2026-05-17): bootstrap denominator is
    # `len(successes)` (observed-N), NOT the paper §1 Hero Table
    # scored-set denominator `scored_task_count(site)`. Phase 1 rerun
    # canonical (cells run to completion) so submission-final figures
    # have `n_episodes == scored_set_n` and the two estimands converge,
    # but in-progress / partial cells the bootstrap CI is conditional-
    # on-observed (narrower than scored-set pessimistic CI). Paper §3.5
    # prose must disclose: "Bootstrap CIs use task-level resampling
    # over observed episodes; Hero SR denominator is scored_task_count
    # (cls 224 / red 205 / shop 435 post-§139.8 N/A exclude). At
    # submission, observed-N == scored-set-N; in-progress CIs are
    # conditional-on-observed and should not be quoted as final."
    # `bootstrap_ci[cid].estimand` field exposes the basis for
    # downstream auditors.
    rng = np.random.default_rng(42)
    n_boot = 10_000

    # a) Bootstrap 95% CI per condition
    for cid in cond_ids:
        mask = ep_df["condition_id"] == cid
        sub = ep_df[mask]
        if sub.empty or "success" not in sub.columns:
            continue
        successes = sub["success"].astype(float).fillna(0).values
        boot_means = [rng.choice(successes, size=len(successes), replace=True).mean() for _ in range(n_boot)]
        ci_lo = float(np.percentile(boot_means, 2.5))
        ci_hi = float(np.percentile(boot_means, 97.5))
        # B-597: best-effort scored-set N lookup for auditor cross-check.
        scored_set_n: Optional[int] = None
        if "benchmark_site" in sub.columns and "benchmark" in sub.columns:
            try:
                site_val = str(sub["benchmark_site"].iloc[0])
                bench_val = str(sub["benchmark"].iloc[0])
                scored_set_n = scored_task_count(site_val, bench_val, strict=False)
            except Exception:
                scored_set_n = None
        results["bootstrap_ci"][cid] = {
            "success_rate": float(successes.mean()),
            "ci_lower_95": ci_lo,
            "ci_upper_95": ci_hi,
            "n_episodes": int(len(successes)),
            # B-597: paper §3.5 estimand disclosure.
            "estimand": "conditional_on_observed_n",
            "scored_set_n_hero_table": scored_set_n,
            "estimand_note": (
                "Bootstrap CI denominator = observed n_episodes. "
                "Paper §1 Hero SR denominator = scored_task_count (cls 224 / red 205 / "
                "shop 435 post-§139.8 N/A exclude, B-91 upstream guard). "
                "At Phase 1 rerun completion n_episodes == scored_set_n_hero_table and "
                "the two estimands converge; in-progress CIs are conditional-on-observed "
                "and MUST NOT be quoted as final paper §3.5 numbers."
            ),
        }
        flat_rows.append({
            "comparison": cid,
            "metric": "success_rate",
            "test": "bootstrap_ci",
            "statistic": float(successes.mean()),
            "p_value": None,
            "significant_05": None,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
        })

    # b) McNemar's exact test and c) Wilcoxon signed-rank (per condition-pair)
    # B-170 (/stress A1.4b-i Claude A1, OOB): cls/red/shop task_id ranges all
    # overlap [0, 209] empirically — `merge(on="task_id")` alone cross-pairs
    # tasks across sites, silently corrupting McNemar contingency. Always
    # include `benchmark_site` in the join key so a pair is unique to one site.
    pair_cols = ["task_id", "success", "total_cost_usd", "p95_step_latency_ms"]
    has_site = "benchmark_site" in ep_df.columns
    if has_site:
        pair_cols = ["benchmark_site"] + pair_cols
    join_on = ["benchmark_site", "task_id"] if has_site else ["task_id"]
    if len(cond_ids) >= 2 and "task_id" in ep_df.columns:
        for i in range(len(cond_ids)):
            for j in range(i + 1, len(cond_ids)):
                cid_a, cid_b = cond_ids[i], cond_ids[j]
                df_a = ep_df[ep_df["condition_id"] == cid_a][pair_cols].copy()
                df_b = ep_df[ep_df["condition_id"] == cid_b][pair_cols].copy()
                merged = df_a.merge(df_b, on=join_on, suffixes=("_a", "_b"))
                if merged.empty:
                    results["notes"].append(f"No paired tasks for {cid_a} vs {cid_b}")
                    continue

                pair_key = f"{cid_a}_vs_{cid_b}"

                # McNemar
                if {"success_a", "success_b"}.issubset(merged.columns):
                    try:
                        a_succ = merged["success_a"].astype(float).fillna(0).astype(bool)
                        b_succ = merged["success_b"].astype(float).fillna(0).astype(bool)
                        n00 = int((~a_succ & ~b_succ).sum())
                        n01 = int((~a_succ & b_succ).sum())
                        n10 = int((a_succ & ~b_succ).sum())
                        n11 = int((a_succ & b_succ).sum())
                        # McNemar exact test via binomial (scipy has no mcnemar)
                        n_discordant = n01 + n10
                        stat = float(min(n01, n10))
                        if n_discordant == 0:
                            p_val = 1.0
                        else:
                            from scipy.stats import binomtest
                            bres = binomtest(n10, n_discordant, 0.5, alternative='two-sided')
                            p_val = float(bres.pvalue)
                        results["mcnemar"][pair_key] = {
                            "statistic": stat,
                            "p_value": p_val,
                            "significant_05": p_val < 0.05,
                            "contingency": {"n11": n11, "n10": n10, "n01": n01, "n00": n00},
                        }
                        flat_rows.append({
                            "comparison": pair_key,
                            "metric": "success",
                            "test": "mcnemar_exact",
                            "statistic": stat,
                            "p_value": p_val,
                            "significant_05": p_val < 0.05,
                            "ci_lower": None,
                            "ci_upper": None,
                        })
                    except Exception as exc:
                        results["notes"].append(f"McNemar failed for {pair_key}: {exc}")

                # Wilcoxon for cost and latency
                for metric, col_a, col_b in [
                    ("total_cost_usd", "total_cost_usd_a", "total_cost_usd_b"),
                    ("p95_step_latency_ms", "p95_step_latency_ms_a", "p95_step_latency_ms_b"),
                ]:
                    if col_a not in merged.columns or col_b not in merged.columns:
                        continue
                    a_vals = pd.to_numeric(merged[col_a], errors="coerce").dropna()
                    b_vals = pd.to_numeric(merged[col_b], errors="coerce").dropna()
                    # Align by index
                    common_idx = a_vals.index.intersection(b_vals.index)
                    if len(common_idx) < 5:
                        # B-172 (/stress A1.4b-i Claude A6): silent skip writing only to
                        # `results["notes"]` made downstream readers of `statistical_tests.csv`
                        # see a missing row + assume "no diff". Emit a CSV row with
                        # p_value=None + reason so auditors can re-discover skipped pairs.
                        skip_reason = f"insufficient_paired_samples_n{len(common_idx)}"
                        results["notes"].append(f"Wilcoxon {metric} {pair_key}: too few paired samples ({len(common_idx)})")
                        flat_rows.append({
                            "comparison": pair_key,
                            "metric": metric,
                            "test": "wilcoxon_signed_rank",
                            "statistic": None,
                            "p_value": None,
                            "significant_05": None,
                            "ci_lower": None,
                            "ci_upper": None,
                            "skipped_reason": skip_reason,
                        })
                        continue
                    try:
                        wres = scipy_stats.wilcoxon(
                            a_vals.loc[common_idx].values,
                            b_vals.loc[common_idx].values,
                            alternative="two-sided",
                        )
                        p_val = float(wres.pvalue)
                        stat = float(wres.statistic)
                        wk = f"{pair_key}_{metric}"
                        results["wilcoxon"][wk] = {
                            "metric": metric,
                            "statistic": stat,
                            "p_value": p_val,
                            "significant_05": p_val < 0.05,
                            "n_pairs": len(common_idx),
                        }
                        flat_rows.append({
                            "comparison": pair_key,
                            "metric": metric,
                            "test": "wilcoxon_signed_rank",
                            "statistic": stat,
                            "p_value": p_val,
                            "significant_05": p_val < 0.05,
                            "ci_lower": None,
                            "ci_upper": None,
                        })
                    except Exception as exc:
                        results["notes"].append(f"Wilcoxon {metric} {pair_key}: {exc}")
    else:
        results["notes"].append("Fewer than 2 conditions — pairwise tests skipped")

    # B-178 (/stress A1.4b-i Claude A2 + gemini C2): apply Holm-Bonferroni
    # step-down per (test_family, metric) family to the raw pairwise p-values
    # produced above. The preregistration §3.6 declares Holm-corrected paired
    # tests for cross-condition comparisons; pre-fix `_compute_statistical_tests`
    # output ONLY raw p-values so any reader of `statistical_tests.json` /
    # `statistical_tests.csv` who paste those into the paper directly inflated
    # FWER for 36 conditions × 630 pairs. Holm is applied within each
    # (test, metric) sub-family (McNemar success / Wilcoxon cost / Wilcoxon
    # latency) so the family size is len(cond_ids)*(len(cond_ids)-1)/2 each.
    def _holm_correct(p_values: List[Optional[float]]) -> List[Optional[float]]:
        """Holm-Bonferroni step-down. Preserves None for skipped rows."""
        # Filter out None for the correction; reinject in-place at original idx.
        finite_idx = [i for i, p in enumerate(p_values) if p is not None]
        finite_p = [float(p_values[i]) for i in finite_idx]
        m = len(finite_p)
        if m == 0:
            return list(p_values)
        # Sort by p ascending; rank j (1-indexed) gets multiplier (m - j + 1)
        order = sorted(range(m), key=lambda j: finite_p[j])
        adj = [0.0] * m
        running_max = 0.0
        for rank, src_j in enumerate(order, start=1):
            scaled = min(1.0, finite_p[src_j] * (m - rank + 1))
            running_max = max(running_max, scaled)  # monotone non-decreasing
            adj[src_j] = running_max
        out: List[Optional[float]] = list(p_values)
        for slot, src_j in enumerate(finite_idx):
            out[src_j] = adj[slot]
        return out

    # Group flat_rows by (test, metric) and apply Holm within each family.
    if flat_rows:
        from collections import defaultdict as _dd
        families: Dict[Any, List[int]] = _dd(list)
        for idx, row in enumerate(flat_rows):
            test = row.get("test")
            metric = row.get("metric")
            if test in ("mcnemar_exact", "wilcoxon_signed_rank") and row.get("p_value") is not None:
                families[(test, metric)].append(idx)
            # rows with p_value=None (skipped) get holm_p=None automatically
        for family_key, idx_list in families.items():
            family_p = [flat_rows[i]["p_value"] for i in idx_list]
            family_holm = _holm_correct(family_p)
            for slot, i in enumerate(idx_list):
                flat_rows[i]["p_value_holm"] = family_holm[slot]
                flat_rows[i]["significant_05_holm"] = (
                    family_holm[slot] is not None and family_holm[slot] < 0.05
                )
                flat_rows[i]["holm_family"] = f"{family_key[0]}_{family_key[1]}"
                flat_rows[i]["holm_family_m"] = len(idx_list)
        # rows that didn't enter a family (bootstrap_ci / skipped) get explicit None
        for row in flat_rows:
            row.setdefault("p_value_holm", None)
            row.setdefault("significant_05_holm", None)
            row.setdefault("holm_family", None)
            row.setdefault("holm_family_m", None)

        # Also stamp on the JSON side under a separate `holm` key for symmetry.
        results["holm_corrected"] = {
            "families": {
                f"{k[0]}_{k[1]}": {
                    "m": len(v),
                    "method": "holm-bonferroni step-down (within-family)",
                }
                for k, v in families.items()
            },
            "note": "Raw p_values are preserved in mcnemar/wilcoxon blocks; "
                    "Holm-adjusted values are in `statistical_tests.csv` columns "
                    "p_value_holm + significant_05_holm. Paper §3.6 reads Holm.",
        }

    with open(reports_dir / "statistical_tests.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if flat_rows:
        import pandas as pd  # type: ignore  (already imported above, but safe)
        flat_df = pd.DataFrame(flat_rows)
        flat_df.to_csv(tables_dir / "statistical_tests.csv", index=False)


def _analyze_per_site(
    ep_df,
    plots_dir: Path,
    tables_dir: Path,
) -> None:
    """Per-site breakdown: success rate, steps, cost, energy per (condition, site)."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
    except ImportError:
        logger.warning("matplotlib/numpy/pandas not installed; skipping per-site analysis")
        return

    if ep_df.empty:
        return
    if "benchmark_site" not in ep_df.columns:
        return
    sites = ep_df["benchmark_site"].dropna().unique().tolist()
    if len(sites) <= 1:
        return

    if "condition_id" not in ep_df.columns:
        return

    agg_parts: List[Dict[str, Any]] = []
    for (cid, site), grp in ep_df.groupby(["condition_id", "benchmark_site"]):
        row: Dict[str, Any] = {"condition_id": cid, "benchmark_site": site, "n_episodes": len(grp)}
        if "success" in grp.columns:
            row["success_rate"] = float(pd.to_numeric(grp["success"], errors="coerce").fillna(0).mean())
        if "steps" in grp.columns:
            row["avg_steps"] = float(pd.to_numeric(grp["steps"], errors="coerce").mean())
        if "total_cost_usd" in grp.columns:
            row["avg_total_cost_usd"] = float(pd.to_numeric(grp["total_cost_usd"], errors="coerce").mean())
        if "total_energy_kwh" in grp.columns:
            row["avg_total_energy_kwh"] = float(pd.to_numeric(grp["total_energy_kwh"], errors="coerce").mean())
        agg_parts.append(row)

    if not agg_parts:
        return

    site_df = pd.DataFrame(agg_parts)
    site_df.to_csv(tables_dir / "per_site_metrics.csv", index=False)

    if "success_rate" not in site_df.columns:
        return

    conds = site_df["condition_id"].unique().tolist()
    x = np.arange(len(sites))
    width = 0.8 / max(len(conds), 1)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

    fig, ax = plt.subplots(figsize=(max(8, len(sites) * 1.5), 5))
    for ci, cid in enumerate(conds):
        sub = site_df[site_df["condition_id"] == cid].set_index("benchmark_site")
        vals = [float(sub.loc[s, "success_rate"]) if s in sub.index else float("nan") for s in sites]
        offset = (ci - len(conds) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width=width * 0.9, label=str(cid), color=colors[ci % len(colors)], alpha=0.85, edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(sites, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Success Rate")
    ax.set_xlabel("Benchmark Site")
    ax.set_title("Per-Site Success Rate by Condition")
    ax.legend(title="Condition", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plots_dir / "per_site_success_rate.png")
    plt.close(fig)


def analyze_run(run_dir: str) -> Path:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "analyze_run requires pandas and matplotlib to be installed in the runtime environment"
        ) from exc

    root = Path(run_dir)
    if not root.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    results_dir = analysis_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    # §139.8 + A1.6 (2026-05-16): `analysis/benchmark_noise/` dir retired —
    # used only to host `na_reference_tasks.csv`, which is now obsolete
    # (N/A tasks excluded at task-load time).

    # B-174: reset parse-failure collector at start of each run; emitted to
    # analysis/parse_failures.csv at the end so silent JSON drops become audit-visible.
    _TO_MAPPING_PARSE_FAILURES.clear()
    # B-196 (/stress A1.4b-ii codex B-ii-4): reset JSONL integrity counter
    # so corrupt-line + dedup-discarded + identity-mismatch counts can be
    # emitted to `analysis/jsonl_integrity_report.csv` at end of run.
    from p79.experiment.io_utils import _JSONL_INTEGRITY_LOG
    _JSONL_INTEGRITY_LOG.clear()

    run_summary_path = root / "run_summary_v2.json"
    if run_summary_path.exists():
        with open(run_summary_path, "r", encoding="utf-8") as f:
            run_summary = json.load(f)
        phase = run_summary.get("phase", "phase1")
    else:
        # Run still in progress — infer phase from condition directory names
        run_summary = {}
        cond_dir_names = [d.name for d in root.iterdir() if d.is_dir() and (d / "condition_summary_v2.json").exists()]
        if any("phase2" in n for n in cond_dir_names):
            phase = "phase2"
        elif any("phase3" in n for n in cond_dir_names):
            phase = "phase3"
        else:
            phase = "phase1"

    # Collect all data
    condition_rows = _collect_condition_summaries(root)
    episode_rows = _collect_episode_summaries(root)
    step_rows = _collect_step_records(root)

    cond_df = pd.DataFrame(condition_rows)
    ep_df = pd.DataFrame(episode_rows)
    step_df = pd.DataFrame(step_rows)

    # B-181 (/stress A1.4b-i codex B5, P1): fail-closed phase-consistency
    # validator. Pre-fix: re-launching Phase 2 into a Phase 1 run dir made
    # `analyze_run` silently consume mixed-phase rows (Phase 1 flat arms
    # leaking into Phase 2 Pareto plots, or vice versa). Now: every
    # condition_id must match the selected phase prefix (`phase{1,2,3}_*`);
    # mismatches → write diagnostic + skip the heterogeneous rows so plotters
    # see a clean single-phase cond_df.
    if not cond_df.empty and "condition_id" in cond_df.columns:
        cid_phases = []
        for cid in cond_df["condition_id"].dropna():
            cid_str = str(cid)
            if cid_str.startswith("phase1_"):
                cid_phases.append("phase1")
            elif cid_str.startswith("phase2_"):
                cid_phases.append("phase2")
            elif cid_str.startswith("phase3_"):
                cid_phases.append("phase3")
            else:
                cid_phases.append("unknown")
        unique_phases = set(cid_phases) - {"unknown"}
        if len(unique_phases) > 1:
            mismatch_msg = (
                f"B-181 phase mix detected in {root}: cond_df contains "
                f"{sorted(unique_phases)} but selected phase={phase}. "
                "Likely a re-launch into an existing run dir. Dropping rows "
                "outside the selected phase from cross-condition plots."
            )
            logger.warning(mismatch_msg)
            (analysis_dir / "phase_mix_warning.txt").write_text(
                mismatch_msg, encoding="utf-8"
            )
            cond_df["_inferred_phase"] = cid_phases
            cond_df = cond_df[cond_df["_inferred_phase"].isin([phase, "unknown"])].drop(
                columns=["_inferred_phase"]
            ).reset_index(drop=True)

    # Infer benchmark from episode data or run_dir path
    def _infer_benchmark(ep_df, run_dir_path: Path) -> str:
        if not ep_df.empty and "benchmark" in ep_df.columns:
            return str(ep_df["benchmark"].iloc[0])
        # Infer from run_dir path: results/webarena/... vs results/visualwebarena/...
        for part in run_dir_path.parts:
            if part == "webarena":
                return "webarena"
        return "visualwebarena"

    _benchmark = _infer_benchmark(ep_df, root)

    # §139.8 + /stress A1.6 (2026-05-16) hard-delete: `is_na_reference` flag
    # + `na_reference_tasks.csv` emission removed. N/A tasks are excluded at
    # task-load time (`tasks.py::load_tasks`, `task.exclude_na_tasks` default
    # True), so episodes never contain N/A rows. The post-hoc per-episode
    # marker layer is dead code.

    # §139.8 retire layer: `success` is canonical; `raw_success` /
    # `adjusted_success` aliases removed in /stress A1.6 hard-delete sweep
    # (2026-05-16). Selective-retain-for-schema-stability policy overruled —
    # downstream readers should reference `success` directly.

    # --- Per-condition (per-session) analysis ---
    cond_ids = [row.get("condition_id") for row in condition_rows if row.get("condition_id")]
    for cid in cond_ids:
        cond_analysis_dir = results_dir / cid
        cond_ep_rows = [r for r in episode_rows if r.get("condition_id") == cid]
        cond_step_rows = [r for r in step_rows if r.get("condition_id") == cid]
        _analyze_condition(cid, cond_analysis_dir, cond_ep_rows, cond_step_rows, phase, run_dir=root)

    # --- Cross-condition overview ---
    if len(cond_ids) <= 1 and cond_df.empty:
        # Nothing to compare yet
        with open(analysis_dir / "analysis_summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {"run_dir": str(root), "phase": phase, "condition_count": 0,
                 "episode_count": 0, "step_count": 0},
                f, indent=2, ensure_ascii=False,
            )
        return analysis_dir

    overview_dir = results_dir / "_overview"
    ov_plots = overview_dir / "plots"
    ov_tables = overview_dir / "tables"
    ov_reports = overview_dir / "reports"
    for d in (overview_dir, ov_plots, ov_tables, ov_reports):
        d.mkdir(parents=True, exist_ok=True)

    if not cond_df.empty:
        cond_df.to_csv(ov_tables / "condition_metrics.csv", index=False)
    if not ep_df.empty:
        ep_df.to_csv(ov_tables / "episode_metrics.csv", index=False)
    if not step_df.empty:
        step_df.to_csv(ov_tables / "step_metrics.csv", index=False)

    # B-600: noise report via single source helper (DRY).
    _emit_benchmark_noise_report(ep_df, ov_tables)

    # Enrich cond_df with avg_total_tokens from episode data (not stored in condition_summary_v2)
    if not cond_df.empty and not ep_df.empty and "total_tokens" in ep_df.columns and "condition_id" in ep_df.columns:
        tok_means = (
            ep_df.groupby("condition_id")["total_tokens"]
            .apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
            .reset_index()
            .rename(columns={"total_tokens": "avg_total_tokens"})
        )
        cond_df = cond_df.merge(tok_means, on="condition_id", how="left")

    if phase == "phase1" and not cond_df.empty:
        _plot_phase1(cond_df, ov_plots, ov_tables, ep_df=ep_df)
    elif phase == "phase2" and not cond_df.empty:
        _plot_phase2(cond_df, ov_plots, ov_tables, ov_reports)
    elif phase == "phase3" and not cond_df.empty:
        _plot_phase3(cond_df, ov_plots, ov_tables)

    if not cond_df.empty:
        _plot_state_change_reason_distribution(cond_df, ov_plots, ov_tables, phase)
        _plot_trigger_distribution(cond_df, ov_plots, ov_tables, phase)
    _analyze_checklist(step_df, ep_df, ov_plots, ov_tables)

    # Statistical tests (requires ≥2 conditions)
    if not ep_df.empty:
        _compute_statistical_tests(cond_df, ep_df, ov_reports, ov_tables)

    # Per-site breakdown
    if not ep_df.empty:
        _analyze_per_site(ep_df, ov_plots, ov_tables)

    # §139.8 + /stress A1.6 (2026-05-16) hard-delete: post-hoc adjusted layer
    # + `is_na_reference` diagnostic both retired. `ep_df["success"]` is the
    # canonical outcome; N/A tasks are excluded at task-load time.

    # B-174: emit collected JSON-parse failures so audit can see what got dropped.
    if _TO_MAPPING_PARSE_FAILURES:
        pf_df = pd.DataFrame(_TO_MAPPING_PARSE_FAILURES)
        pf_df.to_csv(analysis_dir / "parse_failures.csv", index=False)
        logger.warning(
            "analyze_run: %d _to_mapping parse failures recorded → %s",
            len(_TO_MAPPING_PARSE_FAILURES), analysis_dir / "parse_failures.csv",
        )

    # B-196: emit JSONL integrity report — paper §3 reviewers can verify
    # denominator transparency (how many lines / how many corrupt / how
    # many identity mismatches across the canonical analysis run).
    if _JSONL_INTEGRITY_LOG:
        ig_df = pd.DataFrame(_JSONL_INTEGRITY_LOG)
        ig_df.to_csv(analysis_dir / "jsonl_integrity_report.csv", index=False)
        total_corrupt = int(ig_df["corrupt_lines"].sum())
        total_mismatch = int(ig_df["summary_identity_mismatch"].sum())
        if total_corrupt > 0 or total_mismatch > 0:
            logger.warning(
                "analyze_run: JSONL integrity report — %d files scanned, "
                "%d total corrupt lines dropped, %d summary identity mismatches → %s",
                len(_JSONL_INTEGRITY_LOG), total_corrupt, total_mismatch,
                analysis_dir / "jsonl_integrity_report.csv",
            )

    with open(analysis_dir / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_dir": str(root),
                "phase": phase,
                "condition_count": int(len(cond_df)),
                "episode_count": int(len(ep_df)),
                "step_count": int(len(step_df)),
                "parse_failure_count": len(_TO_MAPPING_PARSE_FAILURES),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return analysis_dir


def _plot_phase1(cond_df, plots_dir: Path, tables_dir: Path, ep_df=None) -> None:
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    # Extended representation screening table
    extended_cols = [
        "condition_id", "observation_mode", "success_rate",
        "avg_total_cost_usd", "p95_step_latency_ms",
        "avg_steps", "avg_no_op_rate", "avg_retries",
        "avg_total_tokens", "avg_total_energy_kwh", "avg_total_co2e_kg",
    ]
    available = [c for c in extended_cols if c in cond_df.columns]
    table = cond_df[available]
    table.to_csv(tables_dir / "phase1_representation_screening.csv", index=False)

    if "observation_mode" not in cond_df.columns or "success_rate" not in cond_df.columns:
        return

    # B-87: 6-mode order (was hardcoded ["dom","som","vision"] — silently dropped
    # the 3 phantom arms from the headline plot). phantom_dom is the legacy alias
    # for phantom_text; accept both. Unknown modes are appended + warned, never dropped.
    _canonical_order = ["dom", "phantom_text", "phantom_dom", "phantom_prompt",
                        "phantom_som", "som", "vision"]
    _present = list(cond_df["observation_mode"].values)
    mode_order = [m for m in _canonical_order if m in _present]
    _unknown = sorted(set(_present) - set(_canonical_order))
    if _unknown:
        logger.warning("_plot_phase1: observation_mode(s) outside canonical order, "
                       "appended to plot rather than dropped: %s", _unknown)
        mode_order += _unknown

    success = [
        float(cond_df.loc[cond_df["observation_mode"] == m, "success_rate"].mean())
        for m in mode_order
    ]

    fig, ax = plt.subplots(figsize=(max(6, len(mode_order) * 1.1), 4))
    _palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "#DA8BC3"]
    # B-179 (Claude A3 + codex B6 + gemini C4): mark mode bars whose source
    # contains ≥1 partial/synthesized condition with a hatch pattern so paper
    # readers can visually distinguish "complete data" from "in-progress mix".
    # Pre-fix, partial conditions were silently averaged in with no visual cue.
    has_synth_col = "_synthesized" in cond_df.columns
    is_partial_per_mode = []
    for m in mode_order:
        if has_synth_col:
            sub = cond_df.loc[cond_df["observation_mode"] == m]
            any_synth = bool(sub.get("_synthesized", pd.Series(dtype=bool)).fillna(False).any())
        else:
            any_synth = False
        is_partial_per_mode.append(any_synth)
    bars = ax.bar(
        mode_order, success,
        color=[_palette[i % len(_palette)] for i in range(len(mode_order))],
        hatch=["//" if partial else None for partial in is_partial_per_mode],
        edgecolor="black",
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Success Rate")
    # B-171 (/stress A1.4b-i Claude A5 + gemini C1): "(adjusted)" prose
    # remnant from pre-§139.8 era retired post-hoc layer. `success` is canonical
    # now; only N/A tasks are excluded (at task-load, see `tasks.py::load_tasks`).
    _title_extra = " — partial conditions hatched (//)" if any(is_partial_per_mode) else ""
    ax.set_title(
        f"Phase 1 Representation Screening (N/A excluded at task-load){_title_extra}"
    )
    for bar, val in zip(bars, success):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(plots_dir / "phase1_representation_screening.png")
    plt.close(fig)

    # --- Per-task success heatmap (task_id × condition) ---
    if ep_df is not None and not ep_df.empty and len(mode_order) >= 2:
        pivot = ep_df.pivot_table(
            index="task_id", columns="condition_id",
            values="success", aggfunc="max",
        )
        # Sort by total successes for readability
        pivot = pivot.reindex(columns=[
            c for c in cond_df["condition_id"].tolist() if c in pivot.columns
        ])
        pivot_sorted = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
        # Limit to top 80 tasks for readability
        if len(pivot_sorted) > 80:
            pivot_sorted = pivot_sorted.head(80)
        fig_h, ax_h = plt.subplots(figsize=(max(4, len(pivot.columns) * 2), max(6, len(pivot_sorted) * 0.15)))
        heatmap_data = pivot_sorted.apply(pd.to_numeric, errors="coerce").fillna(-1).values.astype(float)
        im = ax_h.imshow(heatmap_data, aspect="auto",
                         cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
        ax_h.set_xticks(range(len(pivot_sorted.columns)))
        ax_h.set_xticklabels(pivot_sorted.columns, rotation=30, ha="right", fontsize=9)
        ax_h.set_ylabel(f"Task ID (top {len(pivot_sorted)})")
        ax_h.set_yticks(range(0, len(pivot_sorted), max(1, len(pivot_sorted) // 20)))
        ax_h.set_yticklabels(
            [pivot_sorted.index[i] for i in range(0, len(pivot_sorted), max(1, len(pivot_sorted) // 20))],
            fontsize=7,
        )
        ax_h.set_title("Phase 1: Per-Task Success Heatmap (N/A excluded at task-load)")
        fig_h.colorbar(im, ax=ax_h, label="Success (1=yes, 0=no, gray=N/A)")
        fig_h.tight_layout()
        fig_h.savefig(plots_dir / "phase1_success_heatmap.png", dpi=150)
        plt.close(fig_h)

    # --- 2×3 multi-metric comparison overview ---
    metric_specs = [
        ("success_rate", "Success Rate", None, (0.0, 1.0)),
        ("avg_steps", "Avg Steps", None, None),
        ("avg_total_cost_usd", "Avg Cost (USD)", None, None),
        ("p95_step_latency_ms", "P95 Latency (s)", lambda x: x / 1000.0, None),
        ("avg_total_energy_kwh", "Avg Energy (kWh)", None, None),
        ("avg_total_co2e_kg", "Avg CO₂e (kg)", None, None),
    ]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat = axes.flatten()
    x_labels = cond_df["condition_id"].tolist() if "condition_id" in cond_df.columns else list(range(len(cond_df)))

    for ax_idx, (col, ylabel, transform, ylim) in enumerate(metric_specs):
        ax = axes_flat[ax_idx]
        if col not in cond_df.columns:
            ax.set_visible(False)
            continue
        vals = []
        for v in cond_df[col]:
            try:
                fv = float(v) if v is not None else float("nan")
            except (TypeError, ValueError):
                fv = float("nan")
            if transform is not None:
                fv = transform(fv)
            vals.append(fv)
        x = np.arange(len(x_labels))
        bar_colors = [colors[i % len(colors)] for i in range(len(x_labels))]
        ax.bar(x, vals, color=bar_colors, edgecolor="black", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel, fontsize=10)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Phase 1 Multi-Metric Comparison (N/A excluded at task-load)", fontsize=13)
    fig.tight_layout()
    fig.savefig(plots_dir / "phase1_comparison_overview.png")
    plt.close(fig)


def _plot_phase2(cond_df, plots_dir: Path, tables_dir: Path, reports_dir: Path) -> None:
    import matplotlib.pyplot as plt  # type: ignore
    import pandas as pd  # type: ignore

    # B-601 (/stress A1.6a P1-3-BC gemini F6, 2026-05-17): exclude
    # `_synthesized=True` partial-data rows from Pareto front computation.
    # Pre-fix: in-progress conditions with 5/224 episodes occasionally hit
    # 100% SR and grabbed Pareto front vertices, hiding the real B0/B1
    # trend behind high-variance partial points. Now: opt-in via env
    # `P79_PARETO_ALLOW_PARTIAL=1` for live-monitoring dashboards; default
    # paper-grade plotting drops them.
    import os
    work_df = cond_df.copy()
    if (
        "_synthesized" in work_df.columns
        and os.environ.get("P79_PARETO_ALLOW_PARTIAL", "") != "1"
    ):
        n_before = len(work_df)
        work_df = work_df[work_df["_synthesized"] != True].copy()  # noqa: E712
        n_after = len(work_df)
        if n_after < n_before:
            logger.info(
                "B-601 Pareto excluded %d synthesized partial-condition row(s); "
                "set P79_PARETO_ALLOW_PARTIAL=1 to include for live monitoring.",
                n_before - n_after,
            )
    if "avg_total_model_cost_usd" not in work_df.columns:
        if {"avg_total_cost_usd", "avg_router_overhead_cost_usd"}.issubset(work_df.columns):
            work_df["avg_total_model_cost_usd"] = (
                work_df["avg_total_cost_usd"].astype(float) - work_df["avg_router_overhead_cost_usd"].astype(float)
            )
        else:
            work_df["avg_total_model_cost_usd"] = 0.0

    # B-177 carries `avg_obs_prepare_cost_usd` when present (legacy summaries
    # may lack it); downstream net-saving decomposition reads it from `routed`.
    pareto_cols = [
        "condition_id",
        "success_rate",
        "avg_total_model_cost_usd",
        "avg_router_overhead_cost_usd",
        "avg_total_cost_usd",
    ]
    if "avg_obs_prepare_cost_usd" in work_df.columns:
        pareto_cols.append("avg_obs_prepare_cost_usd")
    plot_df = work_df[pareto_cols].copy()
    plot_df.to_csv(tables_dir / "phase2_pareto_metrics.csv", index=False)

    # Pareto front: success vs cost
    pareto_points = [
        {"success_rate": float(r["success_rate"]), "avg_total_cost_usd": float(r["avg_total_cost_usd"])}
        for _, r in plot_df.iterrows()
    ]
    pareto_idx = _compute_pareto_front(pareto_points, maximize="success_rate", minimize="avg_total_cost_usd")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(plot_df["avg_total_cost_usd"], plot_df["success_rate"], s=80, zorder=3)
    for _, row in plot_df.iterrows():
        ax.annotate(row["condition_id"], (row["avg_total_cost_usd"], row["success_rate"]))
    if len(pareto_idx) >= 2:
        pf_x = [float(plot_df.iloc[i]["avg_total_cost_usd"]) for i in pareto_idx]
        pf_y = [float(plot_df.iloc[i]["success_rate"]) for i in pareto_idx]
        ax.plot(pf_x, pf_y, "r--", linewidth=1.5, label="Pareto front", zorder=2)
        ax.legend()
    ax.set_xlabel("Average Total Cost (USD)")
    ax.set_ylabel("Success Rate")
    ax.set_title("Phase2 Pareto: Success vs Cost")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "phase2_pareto_metrics.png")
    fig.savefig(plots_dir / "phase2_pareto.png")
    plt.close(fig)

    # Pareto: success vs latency
    if "p95_step_latency_ms" in work_df.columns:
        lat_df = work_df[["condition_id", "success_rate", "p95_step_latency_ms"]].copy()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(lat_df["p95_step_latency_ms"], lat_df["success_rate"], s=80, zorder=3)
        for _, row in lat_df.iterrows():
            ax.annotate(row["condition_id"], (row["p95_step_latency_ms"], row["success_rate"]))
        lat_points = [
            {"success_rate": float(r["success_rate"]), "p95_step_latency_ms": float(r["p95_step_latency_ms"])}
            for _, r in lat_df.iterrows()
        ]
        lat_pareto = _compute_pareto_front(lat_points, maximize="success_rate", minimize="p95_step_latency_ms")
        if len(lat_pareto) >= 2:
            pf_x = [float(lat_df.iloc[i]["p95_step_latency_ms"]) for i in lat_pareto]
            pf_y = [float(lat_df.iloc[i]["success_rate"]) for i in lat_pareto]
            ax.plot(pf_x, pf_y, "r--", linewidth=1.5, label="Pareto front")
            ax.legend()
        ax.set_xlabel("P95 Step Latency (ms)")
        ax.set_ylabel("Success Rate")
        ax.set_title("Phase2 Pareto: Success vs Latency")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "phase2_pareto_latency.png")
        plt.close(fig)

    # Pareto: success vs energy
    if "avg_total_energy_kwh" in work_df.columns and work_df["avg_total_energy_kwh"].notna().any():
        eng_df = work_df[work_df["avg_total_energy_kwh"].notna()][
            ["condition_id", "success_rate", "avg_total_energy_kwh"]
        ].copy()
        if not eng_df.empty:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(eng_df["avg_total_energy_kwh"], eng_df["success_rate"], s=80, zorder=3)
            for _, row in eng_df.iterrows():
                ax.annotate(row["condition_id"], (row["avg_total_energy_kwh"], row["success_rate"]))
            eng_points = [
                {"success_rate": float(r["success_rate"]), "avg_total_energy_kwh": float(r["avg_total_energy_kwh"])}
                for _, r in eng_df.iterrows()
            ]
            eng_pareto = _compute_pareto_front(eng_points, maximize="success_rate", minimize="avg_total_energy_kwh")
            if len(eng_pareto) >= 2:
                pf_x = [float(eng_df.iloc[i]["avg_total_energy_kwh"]) for i in eng_pareto]
                pf_y = [float(eng_df.iloc[i]["success_rate"]) for i in eng_pareto]
                ax.plot(pf_x, pf_y, "r--", linewidth=1.5, label="Pareto front")
                ax.legend()
            ax.set_xlabel("Avg Total Energy (kWh)")
            ax.set_ylabel("Success Rate")
            ax.set_title("Phase2 Pareto: Success vs Energy")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(plots_dir / "phase2_pareto_energy.png")
            plt.close(fig)

    fixed = plot_df[plot_df["condition_id"] == "phase2_fixed_best"]
    routed = plot_df[plot_df["condition_id"] == "phase2_routed"]
    if not fixed.empty and not routed.empty:
        # B-177 (/stress A1.4b-i codex B1, OOB): runner cost decomposition is
        # `total = model + router_overhead + obs_prepare` (see
        # `runner/main.py:1837-1839`, `aggregate_condition_metrics` emits
        # `avg_obs_prepare_cost_usd` at `metrics.py:356`). Pre-fix `_plot_phase2`
        # reconstructed routed_total_cost from only 2 of those 3 components, so
        # every Phase 2 net-saving JSON / CSV was biased UPWARD by exactly the
        # routed obs-prepare cost. Now use the canonical `avg_total_cost_usd`
        # directly (it already includes obs_prepare) and emit the full 4-way
        # decomposition so downstream readers can audit components.
        fixed_cost = float(fixed.iloc[0]["avg_total_cost_usd"])
        routed_total_cost = float(routed.iloc[0]["avg_total_cost_usd"])
        routed_model_cost = float(routed.iloc[0]["avg_total_model_cost_usd"])
        routed_overhead = float(routed.iloc[0]["avg_router_overhead_cost_usd"])
        # obs_prepare may be absent on legacy summaries (pre-§97); default to 0.
        routed_obs_prepare = float(
            routed.iloc[0].get("avg_obs_prepare_cost_usd", 0.0) or 0.0
        )
        # Net saving is baseline minus the canonical routed total (which
        # includes obs_prepare). Direct subtraction; do NOT use the legacy
        # `net_saving(baseline, model, overhead)` 2-component reconstruction.
        ns = fixed_cost - routed_total_cost
        # Sanity invariant: reconstructed sum should match canonical total within
        # rounding (catches future schema drift). Tolerate 1e-9 USD slack.
        recon = routed_model_cost + routed_overhead + routed_obs_prepare
        if abs(recon - routed_total_cost) > 1e-9:
            logger.warning(
                "_plot_phase2: routed cost decomposition mismatch — "
                "components sum %.9f USD but avg_total_cost_usd=%.9f USD "
                "(Δ=%.2e). Likely additional cost component not in {model, "
                "router_overhead, obs_prepare}. Net saving still uses canonical total.",
                recon, routed_total_cost, recon - routed_total_cost,
            )

        payload = {
            "baseline_total_cost": fixed_cost,
            "routed_model_cost": routed_model_cost,
            "routed_router_overhead_cost": routed_overhead,
            "routed_obs_prepare_cost": routed_obs_prepare,
            "routed_total_cost": routed_total_cost,
            "net_saving": ns,
        }
        with open(reports_dir / "phase2_net_saving.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        with open(reports_dir / "phase2_net_saving_decomposition.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        decomp = pd.DataFrame(
            [
                {"component": "baseline_total_cost", "value": fixed_cost},
                {"component": "routed_model_cost", "value": routed_model_cost},
                {"component": "routed_router_overhead_cost", "value": routed_overhead},
                {"component": "routed_obs_prepare_cost", "value": routed_obs_prepare},
                {"component": "routed_total_cost", "value": routed_total_cost},
                {"component": "net_saving", "value": ns},
            ]
        )
        decomp.to_csv(tables_dir / "phase2_net_saving_decomposition.csv", index=False)

        # Latency net saving — use end-to-end episode latency, not single-step P95.
        # P95 of step latency mixes step granularities and is not subtractable
        # across conditions; avg_total_latency_ms is the proper episode-level
        # measure produced by aggregate_condition_metrics (§97 audit).
        latency_col = (
            "avg_total_latency_ms" if "avg_total_latency_ms" in fixed.columns
            else "p95_step_latency_ms"
        )
        fixed_latency = float(fixed.iloc[0].get(latency_col, 0.0)) if latency_col in fixed.columns else 0.0
        routed_latency = float(routed.iloc[0].get(latency_col, 0.0)) if latency_col in routed.columns else 0.0
        routed_overhead_ms = float(routed.iloc[0].get("avg_router_overhead_ms", 0.0)) if "avg_router_overhead_ms" in work_df.columns else 0.0
        # net_saving_latency no longer subtracts overhead (already in routed total).
        latency_ns = net_saving_latency(fixed_latency, routed_latency)
        latency_payload = {
            "latency_basis": latency_col,
            "baseline_latency_ms": fixed_latency,
            "routed_latency_ms": routed_latency,
            "router_overhead_ms": routed_overhead_ms,  # diagnostic only
            "net_saving_latency_ms": latency_ns,
        }
        with open(reports_dir / "phase2_net_saving_latency.json", "w", encoding="utf-8") as f:
            json.dump(latency_payload, f, indent=2, ensure_ascii=False)

        # Energy net saving
        fixed_energy = fixed.iloc[0].get("avg_total_energy_kwh") if "avg_total_energy_kwh" in fixed.columns else None
        routed_energy = routed.iloc[0].get("avg_total_energy_kwh") if "avg_total_energy_kwh" in routed.columns else None
        if fixed_energy is not None and routed_energy is not None:
            energy_ns = net_saving_energy(float(fixed_energy), float(routed_energy), None)
            energy_payload = {
                "baseline_energy_kwh": float(fixed_energy),
                "routed_energy_kwh": float(routed_energy),
                "net_saving_energy_kwh": energy_ns,
            }
            with open(reports_dir / "phase2_net_saving_energy.json", "w", encoding="utf-8") as f:
                json.dump(energy_payload, f, indent=2, ensure_ascii=False)


def _plot_phase3(cond_df, plots_dir: Path, tables_dir: Path) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    out_df = cond_df[["condition_id", "success_rate", "avg_total_cost_usd", "p95_step_latency_ms"]].copy()
    out_df.to_csv(tables_dir / "phase3_module_ablation.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(out_df["condition_id"], out_df["success_rate"], alpha=0.7, label="success_rate")
    ax1.set_ylabel("Success Rate")
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis="x", rotation=30)

    ax2 = ax1.twinx()
    ax2.plot(out_df["condition_id"], out_df["avg_total_cost_usd"], color="red", marker="o", label="cost")
    ax2.set_ylabel("Avg Total Cost (USD)")

    fig.suptitle("Phase3 Module Ablation")
    fig.tight_layout()
    fig.savefig(plots_dir / "phase3_module_ablation.png")
    plt.close(fig)

    if out_df.empty:
        return

    if (out_df["condition_id"] == "phase3_none").any():
        base_row = out_df[out_df["condition_id"] == "phase3_none"].iloc[0]
    else:
        base_row = out_df.iloc[0]

    gain_df = out_df.copy()
    gain_df["base_condition_id"] = str(base_row["condition_id"])
    gain_df["delta_success"] = gain_df["success_rate"].astype(float) - float(base_row["success_rate"])
    gain_df["delta_cost"] = gain_df["avg_total_cost_usd"].astype(float) - float(base_row["avg_total_cost_usd"])
    gain_df["delta_latency"] = gain_df["p95_step_latency_ms"].astype(float) - float(base_row["p95_step_latency_ms"])
    gain_df = gain_df[
        [
            "condition_id",
            "base_condition_id",
            "delta_success",
            "delta_cost",
            "delta_latency",
        ]
    ]
    gain_df.to_csv(tables_dir / "phase3_module_gain_vs_base.csv", index=False)
