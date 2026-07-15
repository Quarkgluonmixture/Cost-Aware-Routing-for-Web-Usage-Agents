#!/usr/bin/env python3
"""Render the OFFLINE/NON-GATE router cost--SR Pareto plane.

Output:
- results/phantom_paper/figures/fig_router_pareto_plane.png

The main panel is B0_classifieds.  A smaller B1_classifieds panel is explicitly
limited to the available 4/5-fold partial-OOF diagnostic.  All numeric inputs
come from ``pareto/router_pareto_analysis.json``; this figure performs no gate
analysis and never reads canonical Pass-2 artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = (
    ROOT
    / "results/phantom_paper/l1_router_offline_20260715/pareto/router_pareto_analysis.json"
)
OUT = ROOT / "results/phantom_paper/figures/fig_router_pareto_plane.png"
PRIOR_BASELINES = (
    ROOT
    / "results/phantom_paper/l1_router_offline_20260715/prior_baselines/router_prior_baselines.json"
)
OUT_WITH_BASELINES = PRIOR_BASELINES.parent / "fig_router_pareto_plane_with_baselines.png"

FIXED_BLUE = "#4C78A8"
ROUTER_PURPLE = "#7A5195"
ORACLE_GREEN = "#2E8B57"
CURVE_ORANGE = "#E07A2D"
KNN_RED = "#C44E52"
CASCADE_BROWN = "#8C613C"
VARDANYAN_TEAL = "#238B8E"
LAZYMCOT_PINK = "#C05A8A"
INK = "#202124"
GRID = "#E7E9EC"

SHORT_LABELS = {
    "fixed_dom": "DOM",
    "fixed_som": "SoM",
    "fixed_vision": "Vision",
    "fixed_phantom_text": "P-text",
    "fixed_phantom_prompt": "P-prompt",
    "fixed_phantom_som": "P-SoM",
    "router_oof": "Learned router",
    "router_partial_oof": "Router",
    "oracle": "Oracle",
}


def _points_by_id(cell: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {point["policy_id"]: point for point in cell["points"]}


def _plot_fixed_frontier(
    ax: plt.Axes,
    cell: dict[str, Any],
    points: dict[str, dict[str, Any]],
    *,
    linewidth: float = 1.5,
) -> None:
    front = sorted(
        (points[policy_id] for policy_id in cell["fixed_policy_frontier"]),
        key=lambda point: point["mean_cost_usd"],
    )
    if len(front) >= 2:
        ax.plot(
            [point["mean_cost_usd"] for point in front],
            [point["success_rate_pct"] for point in front],
            color="#3D5366",
            linestyle=(0, (4, 3)),
            linewidth=linewidth,
            zorder=2,
        )


def _annotate(
    ax: plt.Axes,
    point: dict[str, Any],
    text: str,
    offset: tuple[float, float],
    *,
    fontsize: float,
    color: str = INK,
) -> None:
    ax.annotate(
        text,
        xy=(point["mean_cost_usd"], point["success_rate_pct"]),
        xytext=offset,
        textcoords="offset points",
        fontsize=fontsize,
        color=color,
        ha="left" if offset[0] >= 0 else "right",
        va="bottom" if offset[1] >= 0 else "top",
        arrowprops={"arrowstyle": "-", "color": "#A7ABB1", "lw": 0.65},
        zorder=8,
    )


def _draw_b0(
    ax: plt.Axes,
    cell: dict[str, Any],
    curve: dict[str, Any] | None,
) -> None:
    points = _points_by_id(cell)

    if curve is not None:
        rows = curve["curve"]
        ax.plot(
            [row["mean_cost_usd"] for row in rows],
            [row["success_rate_pct"] for row in rows],
            color=CURVE_ORANGE,
            linewidth=1.25,
            alpha=0.65,
            zorder=1,
        )
        ax.scatter(
            [row["mean_cost_usd"] for row in rows],
            [row["success_rate_pct"] for row in rows],
            color=CURVE_ORANGE,
            s=18,
            alpha=0.62,
            edgecolor="white",
            linewidth=0.45,
            zorder=2,
        )
        curve_front = set(curve["curve_pareto_frontier"])
        frontier_rows = [row for row in rows if row["policy_id"] in curve_front]
        ax.scatter(
            [row["mean_cost_usd"] for row in frontier_rows],
            [row["success_rate_pct"] for row in frontier_rows],
            color=CURVE_ORANGE,
            s=58,
            edgecolor=INK,
            linewidth=0.75,
            zorder=4,
        )
        curve_offsets = {0.05: (-4, 12), 0.10: (8, 10)}
        for row in frontier_rows:
            threshold = float(row["threshold"])
            if threshold in curve_offsets:
                _annotate(
                    ax,
                    row,
                    f"Cost-aware τ={threshold:.2f}\n{row['success_rate_pct']:.1f}% @ ${row['mean_cost_usd']:.4f}",
                    curve_offsets[threshold],
                    fontsize=7.8,
                    color="#A85114",
                )

    _plot_fixed_frontier(ax, cell, points)
    fixed = [point for point in points.values() if point["category"] == "fixed"]
    ax.scatter(
        [point["mean_cost_usd"] for point in fixed],
        [point["success_rate_pct"] for point in fixed],
        color=FIXED_BLUE,
        s=86,
        edgecolor="white",
        linewidth=1.0,
        zorder=5,
    )
    best = points[cell["best_single_policy_id"]]
    ax.scatter(
        [best["mean_cost_usd"]],
        [best["success_rate_pct"]],
        facecolor="none",
        edgecolor="black",
        s=150,
        linewidth=2.0,
        zorder=6,
    )
    router = points["router_oof"]
    ax.scatter(
        [router["mean_cost_usd"]],
        [router["success_rate_pct"]],
        color=ROUTER_PURPLE,
        marker="D",
        s=95,
        edgecolor="white",
        linewidth=1.0,
        zorder=6,
    )
    oracle = points["oracle"]
    ax.scatter(
        [oracle["mean_cost_usd"]],
        [oracle["success_rate_pct"]],
        color=ORACLE_GREEN,
        marker="*",
        s=260,
        edgecolor="#174A2D",
        linewidth=0.8,
        zorder=7,
    )

    offsets = {
        "fixed_dom": (8, 5),
        "fixed_som": (-7, 11),
        "fixed_vision": (-17, -22),
        "fixed_phantom_text": (-12, 12),
        "fixed_phantom_prompt": (-12, 9),
        "fixed_phantom_som": (10, 11),
        "router_oof": (-8, -23),
        "oracle": (11, -15),
    }
    for policy_id in [
        "fixed_dom",
        "fixed_som",
        "fixed_vision",
        "fixed_phantom_text",
        "fixed_phantom_prompt",
        "fixed_phantom_som",
        "router_oof",
        "oracle",
    ]:
        point = points[policy_id]
        suffix = " (best single)" if policy_id == cell["best_single_policy_id"] else ""
        _annotate(
            ax,
            point,
            f"{SHORT_LABELS[policy_id]}{suffix}\n{point['success_rate_pct']:.1f}% @ ${point['mean_cost_usd']:.4f}",
            offsets[policy_id],
            fontsize=8.25,
            color=(
                ROUTER_PURPLE
                if policy_id == "router_oof"
                else ORACLE_GREEN if policy_id == "oracle" else INK
            ),
        )

    ax.set_title("B0 · Classifieds (complete 5-fold OOF)", loc="left", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Mean billed cost per task (API USD)", fontsize=9.5)
    ax.set_ylabel("Success rate (%)", fontsize=9.5)
    ax.set_xlim(0.0608, 0.0772)
    ax.set_ylim(12.5, 46.0)
    ax.text(
        0.98,
        0.965,
        "Dashed: fixed-policy frontier\nOrange: post-hoc binary-success OOF threshold curve",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.8,
        color="#555B61",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#D5D8DC", "alpha": 0.92},
    )


def _draw_b1_partial(ax: plt.Axes, partial: dict[str, Any]) -> None:
    points = _points_by_id(partial)
    _plot_fixed_frontier(ax, partial, points, linewidth=1.15)
    fixed = [point for point in points.values() if point["category"] == "fixed"]
    ax.scatter(
        [point["mean_cost_usd"] for point in fixed],
        [point["success_rate_pct"] for point in fixed],
        color=FIXED_BLUE,
        s=58,
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )
    best = points[partial["best_single_policy_id"]]
    ax.scatter(
        [best["mean_cost_usd"]],
        [best["success_rate_pct"]],
        facecolor="none",
        edgecolor="black",
        s=105,
        linewidth=1.6,
        zorder=5,
    )
    router = points["router_partial_oof"]
    oracle = points["oracle"]
    ax.scatter(
        [router["mean_cost_usd"]],
        [router["success_rate_pct"]],
        color=ROUTER_PURPLE,
        marker="D",
        s=76,
        edgecolor="white",
        linewidth=0.8,
        zorder=6,
    )
    ax.scatter(
        [oracle["mean_cost_usd"]],
        [oracle["success_rate_pct"]],
        color=ORACLE_GREEN,
        marker="*",
        s=190,
        edgecolor="#174A2D",
        linewidth=0.7,
        zorder=6,
    )
    offsets = {
        "fixed_dom": (5, -14),
        "fixed_som": (5, 7),
        "fixed_vision": (5, -13),
        "fixed_phantom_text": (-5, 7),
        "fixed_phantom_prompt": (-4, -16),
        "fixed_phantom_som": (-5, -16),
        "router_partial_oof": (-5, 8),
        "oracle": (7, -9),
    }
    for policy_id, offset in offsets.items():
        point = points[policy_id]
        suffix = "*" if policy_id == partial["best_single_policy_id"] else ""
        _annotate(
            ax,
            point,
            f"{SHORT_LABELS[policy_id]}{suffix}",
            offset,
            fontsize=6.8,
            color=(
                ROUTER_PURPLE
                if policy_id == "router_partial_oof"
                else ORACLE_GREEN if policy_id == "oracle" else INK
            ),
        )
    ax.set_title("B1 · Classifieds diagnostic", loc="left", fontsize=10.0, fontweight="bold")
    ax.set_xlabel("Mean billed cost/task\n(electricity-derived USD)", fontsize=8.2)
    ax.set_ylabel("Success rate (%)", fontsize=8.2)
    ax.set_xlim(0.039, 0.067)
    ax.set_ylim(2.0, 26.0)
    ax.text(
        0.04,
        0.95,
        "4/5 folds only (n=180)\nNOT a full-cell estimate",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
        color="#8A3B12",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#FFF4E8", "edgecolor": "#E4A16E", "alpha": 0.94},
    )


def _draw_prior_zoom(ax: plt.Axes, prior: dict[str, Any]) -> None:
    """Overlay kNN and the low-cost end of the primary cascade on the B0 zoom."""

    knn = prior["knn"]["points"]
    ax.scatter(
        [row["mean_cost_usd"] for row in knn],
        [row["success_rate_pct"] for row in knn],
        color=KNN_RED,
        marker="^",
        s=72,
        edgecolor="white",
        linewidth=0.9,
        zorder=7,
    )
    knn_offsets = {5: (15, -31), 10: (8, -15), 20: (-7, -17)}
    for row in knn:
        _annotate(
            ax,
            row,
            (
                f"kNN k=5\n{row['success_rate_pct']:.1f}% @ ${row['mean_cost_usd']:.4f}"
                if int(row["k"]) == 5
                else f"kNN k={row['k']}"
            ),
            knn_offsets[int(row["k"])],
            fontsize=7.0,
            color=KNN_RED,
        )

    signal = prior["cascade"]["primary_signal"]
    curve = prior["cascade"]["curves"][signal]
    visible = [row for row in curve if float(row["mean_cost_usd"]) <= 0.078]
    if visible:
        ax.plot(
            [row["mean_cost_usd"] for row in visible],
            [row["success_rate_pct"] for row in visible],
            color=CASCADE_BROWN,
            linewidth=1.4,
            linestyle=(0, (2, 2)),
            zorder=3,
        )
        ax.scatter(
            [row["mean_cost_usd"] for row in visible],
            [row["success_rate_pct"] for row in visible],
            color=CASCADE_BROWN,
            marker="s",
            s=42,
            edgecolor="white",
            linewidth=0.7,
            zorder=6,
        )
        for row in visible:
            if float(row["tau_quantile"]) == 0.0:
                continue  # Exactly overlaps always-Vision in this cell.
            _annotate(
                ax,
                row,
                f"Cascade q={row['tau_quantile']:.2f}",
                (8, 8),
                fontsize=7.0,
                color=CASCADE_BROWN,
            )
    ax.text(
        0.98,
        0.79,
        "Red triangles: RouteLLM-style kNN\nBrown squares: observed-confidence cascade",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.4,
        color="#555B61",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#D5D8DC", "alpha": 0.92},
    )


def _draw_b0_prior_full(
    ax: plt.Axes,
    cell: dict[str, Any],
    cost_aware: dict[str, Any] | None,
    prior: dict[str, Any],
) -> None:
    """Show the complete cascade cost span without compressing the main zoom."""

    points = _points_by_id(cell)
    fixed = [point for point in points.values() if point["category"] == "fixed"]
    ax.scatter(
        [point["mean_cost_usd"] for point in fixed],
        [point["success_rate_pct"] for point in fixed],
        color=FIXED_BLUE,
        s=46,
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
        label="Fixed",
    )
    router = points["router_oof"]
    ax.scatter(
        [router["mean_cost_usd"]],
        [router["success_rate_pct"]],
        color=ROUTER_PURPLE,
        marker="D",
        s=54,
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
        label="Locked LR",
    )
    oracle = points["oracle"]
    ax.scatter(
        [oracle["mean_cost_usd"]],
        [oracle["success_rate_pct"]],
        color=ORACLE_GREEN,
        marker="*",
        s=125,
        edgecolor="#174A2D",
        linewidth=0.6,
        zorder=6,
        label="Oracle",
    )
    if cost_aware is not None:
        rows = cost_aware["curve"]
        ax.plot(
            [row["mean_cost_usd"] for row in rows],
            [row["success_rate_pct"] for row in rows],
            color=CURVE_ORANGE,
            linewidth=1.1,
            alpha=0.7,
            label="OOF P(success) curve",
        )
    knn = prior["knn"]["points"]
    ax.scatter(
        [row["mean_cost_usd"] for row in knn],
        [row["success_rate_pct"] for row in knn],
        color=KNN_RED,
        marker="^",
        s=48,
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
        label="kNN",
    )
    litmined = {
        row["policy_id"]: row for row in prior["litmined_baselines"]["points"]
    }
    vardanyan = litmined["vardanyan_dom_to_vision_failure"]
    lazymcot = litmined["lazymcot_length_median_vision_to_som"]
    ax.scatter(
        [vardanyan["mean_cost_usd"]],
        [vardanyan["success_rate_pct"]],
        color=VARDANYAN_TEAL,
        marker="X",
        s=62,
        edgecolor="white",
        linewidth=0.7,
        zorder=6,
        label="Vardanyan-style",
    )
    ax.scatter(
        [lazymcot["mean_cost_usd"]],
        [lazymcot["success_rate_pct"]],
        color=LAZYMCOT_PINK,
        marker="P",
        s=62,
        edgecolor="white",
        linewidth=0.7,
        zorder=6,
        label="LazyMCoT-style",
    )
    _annotate(
        ax,
        vardanyan,
        f"DOM→Vision failure\n{vardanyan['success_rate_pct']:.1f}% @ ${vardanyan['mean_cost_usd']:.3f}",
        (-5, -35),
        fontsize=6.7,
        color=VARDANYAN_TEAL,
    )
    _annotate(
        ax,
        lazymcot,
        f"Length→SoM\n{lazymcot['success_rate_pct']:.1f}% @ ${lazymcot['mean_cost_usd']:.3f}",
        (15, 24),
        fontsize=6.7,
        color=LAZYMCOT_PINK,
    )
    signal = prior["cascade"]["primary_signal"]
    cascade = prior["cascade"]["curves"][signal]
    ax.plot(
        [row["mean_cost_usd"] for row in cascade],
        [row["success_rate_pct"] for row in cascade],
        color=CASCADE_BROWN,
        linewidth=1.4,
        marker="s",
        markersize=3.2,
        label=f"Cascade ({signal})",
        zorder=3,
    )
    # The q=0 endpoint exactly overlaps the fixed-Vision point in this cell, so
    # label only the far endpoint; annotating both makes the low-cost cluster illegible.
    for row in (cascade[-1],):
        _annotate(
            ax,
            row,
            f"q={row['tau_quantile']:.2f}\n{row['success_rate_pct']:.1f}% @ ${row['mean_cost_usd']:.3f}",
            (5, -5),
            fontsize=6.8,
            color=CASCADE_BROWN,
        )
    ax.set_title("B0 lit-mined envelope · full cascade cost span", loc="left", fontsize=10.0, fontweight="bold")
    ax.set_xlabel("Mean billed cost/task (API USD)", fontsize=8.2)
    ax.set_ylabel("Success rate (%)", fontsize=8.2)
    ax.set_xlim(0.052, 0.435)
    ax.set_ylim(12.5, 46.0)
    ax.legend(loc="lower right", frameon=False, fontsize=6.7)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=ANALYSIS)
    parser.add_argument("--with-baselines", action="store_true")
    parser.add_argument("--prior-baselines", type=Path, default=PRIOR_BASELINES)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    analysis_path = args.analysis.resolve()
    if not analysis_path.is_file():
        raise FileNotFoundError(
            f"Missing {analysis_path}; run scripts/analysis/router_pareto_analysis.py first"
        )
    payload = json.loads(analysis_path.read_text())
    if payload.get("gate_eligible") is not False or "OFFLINE/NON-GATE" not in payload.get("artifact_status", ""):
        raise ValueError("Figure input must be the OFFLINE/NON-GATE Pareto artifact")
    prior = None
    if args.with_baselines:
        prior_path = args.prior_baselines.resolve()
        if not prior_path.is_file():
            raise FileNotFoundError(
                f"Missing {prior_path}; run router_prior_baselines.py first"
            )
        prior = json.loads(prior_path.read_text())
        if prior.get("gate_eligible") is not False or prior.get("cell_id") != "B0_classifieds":
            raise ValueError("Prior-baseline input must be isolated B0 OFFLINE/NON-GATE output")

    plt.rcParams.update(
        {
            "font.size": 9.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )
    fig, (ax_main, ax_diag) = plt.subplots(
        1,
        2,
        figsize=(11.4, 5.45),
        gridspec_kw={"width_ratios": [2.35, 1.0], "wspace": 0.28},
    )
    _draw_b0(ax_main, payload["cells"]["B0_classifieds"], payload.get("cost_aware_variant"))
    if prior is not None:
        _draw_prior_zoom(ax_main, prior)
        _draw_b0_prior_full(
            ax_diag,
            payload["cells"]["B0_classifieds"],
            payload.get("cost_aware_variant"),
            prior,
        )
    else:
        partial = payload["cells"]["B1_classifieds"].get("partial_oof_diagnostic")
        if partial is None:
            ax_diag.text(0.5, 0.5, "B1 partial OOF\nnot available", ha="center", va="center", color="#777777")
            ax_diag.set_axis_off()
        else:
            _draw_b1_partial(ax_diag, partial)

    for ax in (ax_main, ax_diag):
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8.0)
    fig.suptitle(
        "Representation routing on the cost–success plane",
        x=0.055,
        ha="left",
        fontsize=13.2,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.055,
        0.01,
        "OFFLINE / NON-GATE / POST-HOC EXPLORATORY · replayed Pass-1 trajectory cost; oracle uses hindsight",
        ha="left",
        fontsize=8.0,
        color="#6B7075",
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.15, top=0.84, wspace=0.27)
    output = (
        args.output.resolve()
        if args.output is not None
        else (OUT_WITH_BASELINES if args.with_baselines else OUT).resolve()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
