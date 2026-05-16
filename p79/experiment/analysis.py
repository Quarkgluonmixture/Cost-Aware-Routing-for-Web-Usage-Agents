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


def _load_na_task_ids(site: str, benchmark: str = "visualwebarena") -> set:
    """Return task_ids whose reference answer is N/A (unanswerable tasks)."""
    config_dir = _get_config_dir(benchmark)
    config_path = config_dir / f"test_{site}.json"
    if not config_path.exists():
        config_path = config_dir / f"test_{site}.raw.json"
    if not config_path.exists():
        logger.warning(
            "N/A task config not found for site=%s benchmark=%s (looked under %s); "
            "na_fp detection will be silently disabled for this site",
            site, benchmark, config_dir,
        )
        return set()
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)
        na_ids = set()
        for t in tasks:
            ref = t.get("eval", {}).get("reference_answers", {})
            if isinstance(ref, dict) and ref.get("fuzzy_match") == "N/A":
                na_ids.add(t["task_id"])
        return na_ids
    except Exception as exc:
        logger.warning(
            "Failed to parse N/A task config %s: %s; na_fp detection disabled",
            config_path, exc,
        )
        return set()


def scored_task_count(site: str, benchmark: str = "visualwebarena") -> int:
    """Number of tasks in the SCORED set for (site, benchmark).

    §139.8 single source of truth for "EXPECTED_N": total tasks in the site
    config minus the N/A (unanswerable) tasks excluded at load time
    (`task.exclude_na_tasks`, see `tasks.py::load_tasks`). Replaces the
    hardcoded `EXPECTED_N = {classifieds: 234, ...}` dicts scattered across
    analysis + maintenance scripts — those pre-exclusion counts are stale
    once N/A tasks are excluded. Returns 0 if the config cannot be read.
    """
    config_dir = _get_config_dir(benchmark)
    config_path = config_dir / f"test_{site}.json"
    if not config_path.exists():
        config_path = config_dir / f"test_{site}.raw.json"
    if not config_path.exists():
        logger.warning("scored_task_count: config not found for site=%s benchmark=%s", site, benchmark)
        return 0
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)
    except Exception as exc:
        logger.warning("scored_task_count: failed to read %s: %s", config_path, exc)
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
    rows: List[Dict[str, Any]] = []
    for summary_path in run_dir.glob("*/episodes/*_summary_v2.json"):
        with open(summary_path, "r", encoding="utf-8") as f:
            rows.append(json.load(f))
    return rows


def _synthesize_condition_summary(
    cond_dir: Path, ep_summaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a condition summary from condition_meta + episode summaries.

    Used for in-progress conditions that haven't finished yet.
    """
    meta_path = cond_dir / "condition_meta.json"
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    n = len(ep_summaries)
    if n == 0:
        return {}

    successes = sum(1 for e in ep_summaries if e.get("success"))
    steps = [e.get("steps", 0) or e.get("total_steps", 0) or 0 for e in ep_summaries]
    costs = [e.get("total_cost_usd", 0) or 0 for e in ep_summaries]
    model_costs = [e.get("total_model_cost_usd", 0) or 0 for e in ep_summaries]
    energies = [e.get("total_energy_kwh", 0) or 0 for e in ep_summaries]
    co2es = [e.get("total_co2e_kg", 0) or 0 for e in ep_summaries]
    # p95 latency: use per-episode p95, then take overall p95
    ep_p95s = [e.get("p95_step_latency_ms", 0) or 0 for e in ep_summaries]
    import numpy as _np
    p95_lat = float(_np.percentile(ep_p95s, 95)) if ep_p95s else 0.0
    # Retries, no_op, page_unchanged
    retries = [e.get("retries", 0) or 0 for e in ep_summaries]
    no_ops = [e.get("no_op_rate", 0) or 0 for e in ep_summaries]
    page_unch = [e.get("page_unchanged_rate", 0) or 0 for e in ep_summaries]

    # Aggregate trigger_distribution and state_change_reason_distribution
    from collections import Counter
    trigger_agg: Counter = Counter()
    state_change_agg: Counter = Counter()
    for e in ep_summaries:
        td = e.get("trigger_distribution")
        if isinstance(td, dict):
            trigger_agg.update({str(k): int(v) for k, v in td.items()})
        scr = e.get("state_change_reason_distribution")
        if isinstance(scr, dict):
            state_change_agg.update({str(k): int(v) for k, v in scr.items()})

    modules = meta.get("modules", {})

    return {
        "condition_id": meta.get("condition_id", cond_dir.name),
        "seed": meta.get("seed", 42),
        "phase": meta.get("phase", "phase1"),
        "backend_id": meta.get("backend_id", ""),
        "som_on": meta.get("som_on", False),
        "observation_mode": meta.get("observation_mode", "unknown"),
        "router_on": meta.get("router_on", False),
        "module_flags": modules,
        "episodes": n,
        "success_rate": successes / n,
        "avg_steps": sum(steps) / n,
        "p95_step_latency_ms": p95_lat,
        "avg_total_model_cost_usd": sum(model_costs) / n,
        "avg_total_cost_usd": sum(costs) / n,
        "avg_router_overhead_cost_usd": 0.0,
        "avg_total_energy_kwh": sum(energies) / n,
        "avg_total_co2e_kg": sum(co2es) / n,
        "avg_retries": sum(retries) / n,
        "avg_no_op_rate": sum(no_ops) / n,
        "avg_page_unchanged_rate": sum(page_unch) / n,
        "avg_escalation_count": 0.0,
        "trigger_distribution": dict(trigger_agg),
        "state_change_reason_distribution": dict(state_change_agg),
        "avg_checklist_completion_rate": None,
        "checklist_failure_episode_rate": None,
        "benchmark_noise_rate": 0.0,
        "wasted_energy_kwh": 0.0,
        "avg_wasted_cost_usd": 0.0,
        "avg_wasted_energy_kwh": 0.0,
        "cost_efficiency_ratio": 0.0,
        "_synthesized": True,  # mark as not from runner
    }


def _collect_condition_summaries(run_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen_conds: set = set()
    # 1. Completed conditions (have condition_summary_v2.json)
    for summary_path in run_dir.glob("*/condition_summary_v2.json"):
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            rows.append(data)
            seen_conds.add(summary_path.parent.name)

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
    rows: List[Dict[str, Any]] = []
    for step_path in run_dir.glob("*/episodes/*_steps_v2.jsonl"):
        rows.extend(read_jsonl_dedup(step_path))
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

    # Noise report
    if not ep_df.empty:
        if "benchmark_noise" in ep_df.columns:
            noise_df = ep_df[ep_df["benchmark_noise"] == True]  # noqa: E712
        else:
            noise_df = pd.DataFrame()
        if not noise_df.empty:
            noise_counts = noise_df.groupby("benchmark_noise_category").size().reset_index(name="count")
            noise_counts.to_csv(tables_dir / "benchmark_noise_report.csv", index=False)

    if ep_df.empty:
        return

    # --- Cumulative success rate curve ---
    if "success" in ep_df.columns and "task_id" in ep_df.columns:
        sr_df = ep_df[["task_id", "success"]].copy()
        sr_df["success"] = pd.to_numeric(sr_df["success"], errors="coerce").fillna(0)
        sr_df = sr_df.sort_values("task_id")
        sr_df["cumulative_success_rate"] = sr_df["success"].expanding().mean()
        sr_df.to_csv(tables_dir / "cumulative_success_rate.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(sr_df["task_id"], sr_df["cumulative_success_rate"])
        ax.set_xlabel("Task ID")
        ax.set_ylabel("Cumulative Success Rate")
        ax.set_ylim(0, 1)
        ax.set_title(f"Success Rate (N/A excluded at task-load) — {cond_id}")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "cumulative_success_rate.png")
        plt.close(fig)

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
            ax.set_xlabel("P95 Step Latency per Episode (s)")
            ax.set_ylabel("Count")
            ax.set_title(f"Latency Distribution — {cond_id}")
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
        results["bootstrap_ci"][cid] = {
            "success_rate": float(successes.mean()),
            "ci_lower_95": ci_lo,
            "ci_upper_95": ci_hi,
            "n_episodes": int(len(successes)),
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
    noise_dir = analysis_dir / "benchmark_noise"
    results_dir.mkdir(parents=True, exist_ok=True)
    noise_dir.mkdir(parents=True, exist_ok=True)

    # B-174: reset parse-failure collector at start of each run; emitted to
    # analysis/parse_failures.csv at the end so silent JSON drops become audit-visible.
    _TO_MAPPING_PARSE_FAILURES.clear()

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

    # Mark N/A reference tasks (unanswerable — reference answer is "N/A")
    if not ep_df.empty and "benchmark_site" in ep_df.columns and "task_id" in ep_df.columns:
        import pandas as _pd
        na_ids_by_site: Dict[str, set] = {}
        for site in ep_df["benchmark_site"].unique():
            na_ids_by_site[site] = _load_na_task_ids(str(site), _benchmark)
        ep_df["is_na_reference"] = ep_df.apply(
            lambda r: (
                False
                if _pd.isna(r["task_id"])
                else int(r["task_id"]) in na_ids_by_site.get(r["benchmark_site"], set())
            ),
            axis=1,
        )
        na_count = int(ep_df["is_na_reference"].sum())
        if na_count > 0:
            logger.info("Flagged %d episodes as N/A reference tasks", na_count)
            na_summary = ep_df[ep_df["is_na_reference"]][["benchmark_site", "task_id", "condition_id", "success"]].copy()
            na_summary.to_csv(noise_dir / "na_reference_tasks.csv", index=False)
    else:
        ep_df["is_na_reference"] = False

    # --- §139.8: post-hoc adjusted_success layer retired ---
    # na_fp + eval_fp are now fixed at the source — empty-pred guard in the
    # VWA evaluator (master bug B-91) + N/A task exclusion at load time
    # (`tasks.py::load_tasks`, `task.exclude_na_tasks`). The runner's
    # `success` is already the canonical paper-grade outcome, so there is no
    # post-hoc override. `raw_success` is kept as an alias (== `success`) for
    # backward compatibility with downstream readers that still reference it.
    if not ep_df.empty and "success" in ep_df.columns:
        ep_df["raw_success"] = ep_df["success"].copy()

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

    if not ep_df.empty:
        if "benchmark_noise" in ep_df.columns:
            noise_df = ep_df[ep_df["benchmark_noise"] == True]  # noqa: E712
        else:
            noise_df = pd.DataFrame()
        if not noise_df.empty:
            noise_counts = noise_df.groupby("benchmark_noise_category").size().reset_index(name="count")
            noise_counts.to_csv(ov_tables / "benchmark_noise_report.csv", index=False)
        else:
            pd.DataFrame(columns=["benchmark_noise_category", "count"]).to_csv(
                ov_tables / "benchmark_noise_report.csv", index=False
            )

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

    # Note: §139.8 retired the post-hoc adjusted_success layer entirely
    # (was: `compute_adjusted_success_batch` + a second `na_adjusted`
    # double-filter pass, both removed). `ep_df["success"]` is now the
    # runner's canonical outcome with no override; `is_na_reference`
    # remains only as a diagnostic flag for the noise-report CSV.
    na_total = int(ep_df["is_na_reference"].sum()) if (
        not ep_df.empty and "is_na_reference" in ep_df.columns
    ) else 0

    # B-174: emit collected JSON-parse failures so audit can see what got dropped.
    if _TO_MAPPING_PARSE_FAILURES:
        pf_df = pd.DataFrame(_TO_MAPPING_PARSE_FAILURES)
        pf_df.to_csv(analysis_dir / "parse_failures.csv", index=False)
        logger.warning(
            "analyze_run: %d _to_mapping parse failures recorded → %s",
            len(_TO_MAPPING_PARSE_FAILURES), analysis_dir / "parse_failures.csv",
        )

    with open(analysis_dir / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_dir": str(root),
                "phase": phase,
                "condition_count": int(len(cond_df)),
                "episode_count": int(len(ep_df)),
                "step_count": int(len(step_df)),
                "na_reference_task_count": na_total,
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
    bars = ax.bar(mode_order, success,
                  color=[_palette[i % len(_palette)] for i in range(len(mode_order))])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Success Rate")
    # B-171 (/stress A1.4b-i Claude A5 + gemini C1): "(adjusted)" prose
    # remnant from pre-§139.8 era retired post-hoc layer. `success` is canonical
    # now; only N/A tasks are excluded (at task-load, see `tasks.py::load_tasks`).
    ax.set_title("Phase 1 Representation Screening (N/A excluded at task-load)")
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

    work_df = cond_df.copy()
    if "avg_total_model_cost_usd" not in work_df.columns:
        if {"avg_total_cost_usd", "avg_router_overhead_cost_usd"}.issubset(work_df.columns):
            work_df["avg_total_model_cost_usd"] = (
                work_df["avg_total_cost_usd"].astype(float) - work_df["avg_router_overhead_cost_usd"].astype(float)
            )
        else:
            work_df["avg_total_model_cost_usd"] = 0.0

    plot_df = work_df[
        [
            "condition_id",
            "success_rate",
            "avg_total_model_cost_usd",
            "avg_router_overhead_cost_usd",
            "avg_total_cost_usd",
        ]
    ].copy()
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
        fixed_cost = float(fixed.iloc[0]["avg_total_cost_usd"])
        routed_model_cost = float(routed.iloc[0]["avg_total_model_cost_usd"])
        routed_overhead = float(routed.iloc[0]["avg_router_overhead_cost_usd"])
        routed_total_cost = routed_model_cost + routed_overhead
        ns = net_saving(fixed_cost, routed_model_cost, routed_overhead)

        payload = {
            "baseline_total_cost": fixed_cost,
            "routed_model_cost": routed_model_cost,
            "routed_router_overhead_cost": routed_overhead,
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
