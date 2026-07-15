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

FIXED_BLUE = "#4C78A8"
ROUTER_PURPLE = "#7A5195"
ORACLE_GREEN = "#2E8B57"
CURVE_ORANGE = "#E07A2D"
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


def main() -> None:
    if not ANALYSIS.is_file():
        raise FileNotFoundError(
            f"Missing {ANALYSIS}; run scripts/analysis/router_pareto_analysis.py first"
        )
    payload = json.loads(ANALYSIS.read_text())
    if payload.get("gate_eligible") is not False or "OFFLINE/NON-GATE" not in payload.get("artifact_status", ""):
        raise ValueError("Figure input must be the OFFLINE/NON-GATE Pareto artifact")

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
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
