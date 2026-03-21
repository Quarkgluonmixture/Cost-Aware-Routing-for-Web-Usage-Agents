from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from p79.experiment.metrics import net_saving


def _collect_episode_summaries(run_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for summary_path in run_dir.glob("*/episodes/*_summary_v2.json"):
        with open(summary_path, "r", encoding="utf-8") as f:
            rows.append(json.load(f))
    return rows


def _collect_condition_summaries(run_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for summary_path in run_dir.glob("*/condition_summary_v2.json"):
        with open(summary_path, "r", encoding="utf-8") as f:
            rows.append(json.load(f))
    return rows


def _collect_step_records(run_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for step_path in run_dir.glob("*/episodes/*_steps_v2.jsonl"):
        with open(step_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _to_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def _flatten_state_change_reasons(cond_df) -> Any:
    import pandas as pd  # type: ignore

    rows: List[Dict[str, Any]] = []
    if cond_df.empty or "state_change_reason_distribution" not in cond_df.columns:
        return pd.DataFrame(columns=["condition_id", "reason", "count"])

    for _, row in cond_df.iterrows():
        dist = _to_mapping(row.get("state_change_reason_distribution", {}))
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
        dist = _to_mapping(row.get("trigger_distribution", {}))
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


def _analyze_checklist(step_df, ep_df, output_dir: Path) -> None:
    import matplotlib.pyplot as plt  # type: ignore
    import pandas as pd  # type: ignore

    curve_path = output_dir / "checklist_progress_curve.csv"
    fail_path = output_dir / "checklist_failure_distribution.csv"

    if step_df.empty or "checklist" not in step_df.columns:
        pd.DataFrame(columns=["condition_id", "step_idx", "avg_completion_rate"]).to_csv(curve_path, index=False)
        pd.DataFrame(
            columns=["condition_id", "episodes", "episodes_with_failed", "failure_rate", "avg_completion_rate"]
        ).to_csv(fail_path, index=False)
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
        fig.savefig(output_dir / "checklist_progress_curve.png")
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
        fig.savefig(output_dir / "checklist_failure_distribution.png")
        plt.close(fig)


def _plot_state_change_reason_distribution(cond_df, output_dir: Path, phase: str) -> None:
    import matplotlib.pyplot as plt  # type: ignore
    import pandas as pd  # type: ignore

    flat_df = _flatten_state_change_reasons(cond_df)
    flat_df.to_csv(output_dir / "state_change_reason_distribution.csv", index=False)
    if flat_df.empty:
        return

    pivot = flat_df.pivot_table(index="condition_id", columns="reason", values="count", aggfunc="sum", fill_value=0)
    pivot.to_csv(output_dir / "state_change_reason_distribution_pivot.csv")

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Reason Count")
    ax.set_xlabel("Condition")
    ax.set_title("State-Change Reason Distribution")
    ax.legend(title="Reason", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "state_change_reason_distribution.png")
    plt.close(fig)

    if phase == "phase2":
        fig, ax = plt.subplots(figsize=(9, 5))
        pivot.plot(kind="bar", stacked=True, ax=ax)
        ax.set_ylabel("Reason Count")
        ax.set_xlabel("Condition")
        ax.set_title("Phase2 State-Change Reason Distribution")
        ax.legend(title="Reason", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(output_dir / "phase2_state_change_reason_distribution.png")
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
        fig.savefig(output_dir / "phase3_state_change_reason_heatmap.png")
        plt.close(fig)


def _plot_trigger_distribution(cond_df, output_dir: Path, phase: str) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    flat_df = _flatten_trigger_distribution(cond_df)
    flat_df.to_csv(output_dir / "trigger_distribution.csv", index=False)
    if flat_df.empty:
        return

    pivot = flat_df.pivot_table(index="condition_id", columns="trigger", values="count", aggfunc="sum", fill_value=0)
    pivot.to_csv(output_dir / "trigger_distribution_pivot.csv")

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Trigger Count")
    ax.set_xlabel("Condition")
    ax.set_title("Trigger Distribution")
    ax.legend(title="Trigger", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "trigger_distribution.png")
    plt.close(fig)

    if phase == "phase2":
        fig, ax = plt.subplots(figsize=(9, 5))
        pivot.plot(kind="bar", stacked=True, ax=ax)
        ax.set_ylabel("Trigger Count")
        ax.set_xlabel("Condition")
        ax.set_title("Phase2 Trigger Distribution")
        ax.legend(title="Trigger", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(output_dir / "phase2_trigger_distribution.png")
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

    with open(root / "run_summary_v2.json", "r", encoding="utf-8") as f:
        run_summary = json.load(f)

    phase = run_summary.get("phase", "phase1")
    condition_rows = _collect_condition_summaries(root)
    episode_rows = _collect_episode_summaries(root)
    step_rows = _collect_step_records(root)

    cond_df = pd.DataFrame(condition_rows)
    ep_df = pd.DataFrame(episode_rows)
    step_df = pd.DataFrame(step_rows)

    if not cond_df.empty:
        cond_df.to_csv(analysis_dir / "condition_metrics.csv", index=False)
    if not ep_df.empty:
        ep_df.to_csv(analysis_dir / "episode_metrics.csv", index=False)
    if not step_df.empty:
        step_df.to_csv(analysis_dir / "step_metrics.csv", index=False)

    if not ep_df.empty:
        if "benchmark_noise" in ep_df.columns:
            noise_df = ep_df[ep_df["benchmark_noise"] == True]  # noqa: E712
        else:
            noise_df = pd.DataFrame()
        if not noise_df.empty:
            noise_counts = noise_df.groupby("benchmark_noise_category").size().reset_index(name="count")
            noise_counts.to_csv(analysis_dir / "benchmark_noise_report.csv", index=False)
        else:
            pd.DataFrame(columns=["benchmark_noise_category", "count"]).to_csv(
                analysis_dir / "benchmark_noise_report.csv", index=False
            )

    if phase == "phase1" and not cond_df.empty:
        _plot_phase1(cond_df, analysis_dir)
    elif phase == "phase2" and not cond_df.empty:
        _plot_phase2(cond_df, analysis_dir)
    elif phase == "phase3" and not cond_df.empty:
        _plot_phase3(cond_df, analysis_dir)

    if not cond_df.empty:
        _plot_state_change_reason_distribution(cond_df, analysis_dir, phase)
        _plot_trigger_distribution(cond_df, analysis_dir, phase)
    _analyze_checklist(step_df, ep_df, analysis_dir)

    with open(analysis_dir / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_dir": str(root),
                "phase": phase,
                "condition_count": int(len(cond_df)),
                "episode_count": int(len(ep_df)),
                "step_count": int(len(step_df)),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return analysis_dir


def _plot_phase1(cond_df, output_dir: Path) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    # Representation screening table
    table = cond_df[
        ["condition_id", "som_on", "observation_mode", "success_rate", "avg_total_cost_usd", "p95_step_latency_ms"]
    ]
    table.to_csv(output_dir / "phase1_representation_screening.csv", index=False)

    pivot = table.pivot_table(
        index="som_on",
        columns="observation_mode",
        values="success_rate",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot.values, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"SoM={v}" for v in pivot.index])
    ax.set_title("Phase1 Success Screening")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "phase1_representation_screening.png")
    fig.savefig(output_dir / "phase1_success_heatmap.png")
    plt.close(fig)


def _plot_phase2(cond_df, output_dir: Path) -> None:
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
    plot_df.to_csv(output_dir / "phase2_pareto_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(plot_df["avg_total_cost_usd"], plot_df["success_rate"], s=80)
    for _, row in plot_df.iterrows():
        ax.annotate(row["condition_id"], (row["avg_total_cost_usd"], row["success_rate"]))
    ax.set_xlabel("Average Total Cost (USD)")
    ax.set_ylabel("Success Rate")
    ax.set_title("Phase2 Pareto: Success vs Cost")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "phase2_pareto_metrics.png")
    fig.savefig(output_dir / "phase2_pareto.png")
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
        with open(output_dir / "phase2_net_saving.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        with open(output_dir / "phase2_net_saving_decomposition.json", "w", encoding="utf-8") as f:
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
        decomp.to_csv(output_dir / "phase2_net_saving_decomposition.csv", index=False)


def _plot_phase3(cond_df, output_dir: Path) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    out_df = cond_df[["condition_id", "success_rate", "avg_total_cost_usd", "p95_step_latency_ms"]].copy()
    out_df.to_csv(output_dir / "phase3_module_ablation.csv", index=False)

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
    fig.savefig(output_dir / "phase3_module_ablation.png")
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
    gain_df.to_csv(output_dir / "phase3_module_gain_vs_base.csv", index=False)
