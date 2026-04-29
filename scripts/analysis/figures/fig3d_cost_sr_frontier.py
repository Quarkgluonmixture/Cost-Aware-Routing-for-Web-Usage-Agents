#!/usr/bin/env python3
"""[Layer 3d] Efficiency — B0 vs B1 cost/SR frontier.

Output:
- results/phantom_paper/figures/fig3d_cost_sr_frontier.png

Layer 3d: B0 vs B1 cost gap and Pareto-frontier visualization.

See docs/checkpoints/paper_planning.md §3 Layer 3 framework.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results/visualwebarena/phase1"
COST_PER_MODE = ROOT / "docs/analysis/cross_sites/cost_per_mode.json"
OUT = ROOT / "results/phantom_paper/figures/fig3d_cost_sr_frontier.png"

MODE_COLORS = {
    "DOM": "#4c78a8",
    "SoM": "#f58518",
    "Vision": "#54a24b",
    "Phantom-SoM": "#b279a2",
    "Phantom-DOM": "#e45756",
}
MODEL_MARKERS = {"B0": "o", "B1": "s"}


@dataclass(frozen=True)
class ConditionSpec:
    model: str
    site: str
    mode: str
    condition_dir: Path
    expected_n: int


@dataclass(frozen=True)
class Point:
    model: str
    site: str
    mode: str
    n: int
    expected_n: int
    adjusted_sr: float
    raw_sr: float | None
    cost: float
    cost_class: str
    model_cost: float | None
    latency_s: float | None
    energy_kwh: float | None


BASE_SPECS = [
    ConditionSpec("B0", "classifieds", "DOM", RESULTS / "B0_3mode_classifieds_20260413/phase1_dom_router_0", 234),
    ConditionSpec("B0", "classifieds", "SoM", RESULTS / "B0_3mode_classifieds_20260413/phase1_som_router_0", 234),
    ConditionSpec("B0", "classifieds", "Vision", RESULTS / "B0_3mode_classifieds_20260413/phase1_vision_router_0", 234),
    ConditionSpec("B0", "classifieds", "Phantom-SoM", RESULTS / "B0_phantom_classifieds_20260426/phase1_phantom_som_router_0", 234),
    ConditionSpec("B0", "classifieds", "Phantom-DOM", RESULTS / "B0_phantom_dom_classifieds_20260427/phase1_phantom_dom_router_0", 234),
    ConditionSpec("B0", "reddit", "DOM", RESULTS / "B0_3mode_reddit_20260422/phase1_dom_router_0", 210),
    ConditionSpec("B0", "reddit", "SoM", RESULTS / "B0_3mode_reddit_20260422/phase1_som_router_0", 210),
    ConditionSpec("B0", "reddit", "Vision", RESULTS / "B0_3mode_reddit_20260422/phase1_vision_router_0", 210),
    ConditionSpec("B0", "reddit", "Phantom-SoM", RESULTS / "B0_phantom_reddit_20260428/phase1_phantom_som_router_0", 210),
    ConditionSpec("B0", "reddit", "Phantom-DOM", RESULTS / "B0_phantom_dom_reddit_20260427/phase1_phantom_dom_router_0", 210),
    ConditionSpec("B1", "classifieds", "DOM", RESULTS / "B1_3mode_classifieds_20260413/phase1_dom_router_0", 234),
    ConditionSpec("B1", "classifieds", "SoM", RESULTS / "B1_3mode_classifieds_20260413/phase1_som_router_0", 234),
    ConditionSpec("B1", "classifieds", "Vision", RESULTS / "B1_3mode_classifieds_20260413/phase1_vision_router_0", 234),
    ConditionSpec("B1", "reddit", "DOM", RESULTS / "B1_3mode_reddit_20260413/phase1_dom_router_0", 210),
    ConditionSpec("B1", "reddit", "SoM", RESULTS / "B1_3mode_reddit_20260413/phase1_som_router_0", 210),
    ConditionSpec("B1", "reddit", "Vision", RESULTS / "B1_3mode_reddit_20260413/phase1_vision_router_0", 210),
]

OPTIONAL_SPECS = [
    ConditionSpec("B1", "classifieds", "Phantom-SoM", RESULTS / "B1_phantom_classifieds_20260428/phase1_phantom_som_router_0", 234),
    ConditionSpec("B1", "reddit", "Phantom-SoM", RESULTS / "B1_phantom_reddit_20260426/phase1_phantom_som_router_0", 210),
]


def has_episode_summaries(spec: ConditionSpec) -> bool:
    return any((spec.condition_dir / "episodes").glob("*_summary_v2.json"))


SPECS = BASE_SPECS + [spec for spec in OPTIONAL_SPECS if has_episode_summaries(spec)]


def mode_key(mode: str) -> str:
    return {"Phantom-SoM": "P-SoM", "Phantom-DOM": "P-text"}.get(mode, mode)


def load_cost_per_mode() -> dict:
    if not COST_PER_MODE.exists():
        raise FileNotFoundError(
            f"Missing {COST_PER_MODE}; run `make aggregate-cost-electricity` first."
        )
    with COST_PER_MODE.open() as f:
        return json.load(f)


COST_TABLE = load_cost_per_mode()


def paper_cost(spec: ConditionSpec, summary: dict) -> tuple[float | None, str | None]:
    cell = (
        COST_TABLE.get("cells", {})
        .get(spec.model, {})
        .get(spec.site, {})
        .get(mode_key(spec.mode), {})
    )
    if cell.get("available") and cell.get("paper_cost_usd") is not None:
        return float(cell["paper_cost_usd"]), cell.get("paper_cost_class")
    if spec.model == "B0" and summary.get("avg_total_cost_usd") is not None:
        return float(summary["avg_total_cost_usd"]), "API_token_dollars"
    return None, cell.get("paper_cost_class")


def fmt_cost(value: float) -> str:
    return f"${value:.5f}" if value < 0.01 else f"${value:.3f}"


def task_id(path: Path) -> int:
    match = re.search(r"task_(\d+)_summary", path.name)
    if not match:
        raise ValueError(f"Cannot parse task id from {path}")
    return int(match.group(1))


def episode_summary(condition_dir: Path) -> dict[str, float | int | None]:
    episodes_dir = condition_dir / "episodes"
    files = sorted(episodes_dir.glob("*_summary_v2.json"))
    seen: set[int] = set()
    successes = 0
    total_cost = total_model_cost = total_latency_ms = total_energy_kwh = 0.0
    cost_n = model_cost_n = latency_n = energy_n = 0
    for path in files:
        tid = task_id(path)
        if tid in seen:
            print(f"[warn] duplicate task summary ignored: {path}", file=sys.stderr)
            continue
        seen.add(tid)
        with path.open() as f:
            record = json.load(f)
        successes += bool(record.get("adjusted_success", record.get("success", False)))
        if record.get("total_cost_usd") is not None:
            total_cost += float(record["total_cost_usd"])
            cost_n += 1
        if record.get("total_model_cost_usd") is not None:
            total_model_cost += float(record["total_model_cost_usd"])
            model_cost_n += 1
        if record.get("total_latency_ms") is not None:
            total_latency_ms += float(record["total_latency_ms"])
            latency_n += 1
        if record.get("total_energy_kwh") is not None:
            total_energy_kwh += float(record["total_energy_kwh"])
            energy_n += 1
    if not seen:
        raise FileNotFoundError(f"No episode summaries under {episodes_dir}")
    return {
        "adjusted_sr": 100.0 * successes / len(seen),
        "n": len(seen),
        "avg_total_cost_usd": total_cost / cost_n if cost_n else None,
        "avg_total_model_cost_usd": total_model_cost / model_cost_n if model_cost_n else None,
        "avg_total_latency_ms": total_latency_ms / latency_n if latency_n else None,
        "avg_total_energy_kwh": total_energy_kwh / energy_n if energy_n else None,
    }


def load_point(spec: ConditionSpec) -> Point | None:
    summary_path = spec.condition_dir / "condition_summary_v2.json"
    ep_summary = episode_summary(spec.condition_dir)
    if summary_path.exists():
        with summary_path.open() as f:
            summary = json.load(f)
    else:
        print(
            f"[warn] missing condition summary, using episode-summary fallback: {summary_path}",
            file=sys.stderr,
        )
        summary = {
            "avg_total_cost_usd": ep_summary["avg_total_cost_usd"],
            "avg_total_model_cost_usd": ep_summary["avg_total_model_cost_usd"],
            "avg_total_latency_ms": ep_summary["avg_total_latency_ms"],
            "avg_total_energy_kwh": ep_summary["avg_total_energy_kwh"],
            "success_rate": None,
        }
    cost, cost_class = paper_cost(spec, summary)
    if cost is None or float(cost) <= 0:
        print(
            f"[warn] skipping {spec.model} {spec.site} {spec.mode}: "
            f"paper_cost_usd={cost}",
            file=sys.stderr,
        )
        return None
    adjusted_sr = float(ep_summary["adjusted_sr"])
    n = int(ep_summary["n"])
    latency_ms = summary.get("avg_total_latency_ms")
    point = Point(
        model=spec.model,
        site=spec.site,
        mode=spec.mode,
        n=n,
        expected_n=spec.expected_n,
        adjusted_sr=adjusted_sr,
        raw_sr=summary.get("success_rate"),
        cost=float(cost),
        cost_class=cost_class or "unknown",
        model_cost=summary.get("avg_total_model_cost_usd"),
        latency_s=float(latency_ms) / 1000.0 if latency_ms is not None else None,
        energy_kwh=summary.get("avg_total_energy_kwh"),
    )
    latency_text = f"{point.latency_s:.1f}s" if point.latency_s is not None else "NA"
    site_label = "cls" if point.site == "classifieds" else "red"
    print(
        f"{point.model} {site_label} {point.mode}: SR={point.adjusted_sr:.2f}% "
        f"paper_cost={fmt_cost(point.cost)} class={point.cost_class} lat={latency_text}"
    )
    if point.n != point.expected_n:
        print(
            f"[warn] {point.model} {point.site} {point.mode}: "
            f"episodes n={point.n}/{point.expected_n} partial",
            file=sys.stderr,
        )
    return point


def pareto_frontier(points: list[Point]) -> list[Point]:
    frontier: list[Point] = []
    best_sr = -1.0
    for point in sorted(points, key=lambda p: (p.cost, -p.adjusted_sr)):
        if point.adjusted_sr > best_sr + 1e-9:
            frontier.append(point)
            best_sr = point.adjusted_sr
    return frontier


def annotate_point(ax: plt.Axes, point: Point, index: int) -> None:
    offsets = {
        ("classifieds", "B0", "DOM"): (10, 8),
        ("classifieds", "B0", "SoM"): (8, -22),
        ("classifieds", "B0", "Vision"): (-42, 10),
        ("classifieds", "B0", "Phantom-SoM"): (-76, 4),
        ("classifieds", "B0", "Phantom-DOM"): (-52, -18),
        ("classifieds", "B1", "DOM"): (12, 24),
        ("classifieds", "B1", "SoM"): (-44, 22),
        ("classifieds", "B1", "Vision"): (8, 8),
        ("classifieds", "B1", "Phantom-SoM"): (-64, -20),
        ("reddit", "B0", "DOM"): (8, 8),
        ("reddit", "B0", "SoM"): (8, -22),
        ("reddit", "B0", "Vision"): (-42, 10),
        ("reddit", "B0", "Phantom-SoM"): (-70, -20),
        ("reddit", "B0", "Phantom-DOM"): (-66, 12),
        ("reddit", "B1", "DOM"): (-54, -18),
        ("reddit", "B1", "SoM"): (12, 22),
        ("reddit", "B1", "Vision"): (-50, 24),
    }
    dx, dy = offsets.get((point.site, point.model, point.mode), (8, 8))
    ax.annotate(
        f"{point.mode}\n{point.adjusted_sr:.1f}%/{fmt_cost(point.cost)}",
        xy=(point.cost, point.adjusted_sr),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=7.5,
        color="#222222",
        arrowprops={"arrowstyle": "-", "color": "#bdbdbd", "lw": 0.7},
    )


def draw_pending(ax: plt.Axes, site: str) -> None:
    pending = "B1 Phantom-SoM pending/partial where shown"
    ax.text(
        0.98,
        0.04,
        pending,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#777777",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#bdbdbd", "linestyle": "dotted"},
    )


def point_by(points: list[Point], site: str, model: str, mode: str) -> Point | None:
    return next((point for point in points if point.site == site and point.model == model and point.mode == mode), None)


def add_deployment_callouts(ax: plt.Axes, site: str, site_points: list[Point]) -> None:
    dom = point_by(site_points, site, "B0", "DOM")
    som = point_by(site_points, site, "B0", "SoM")
    phantom = point_by(site_points, site, "B0", "Phantom-SoM") or point_by(site_points, site, "B0", "Phantom-DOM")
    if not dom or not som or not phantom:
        print(f"[warn] {site}: deployment callout skipped; missing DOM/SoM/Phantom point", file=sys.stderr)
        return

    color = MODE_COLORS.get(phantom.mode, "#7a3f6d")
    if site == "classifieds":
        box_xy = (0.014, 24.5)
        arrow_y = 18.1
        label_y_offset = 0.7
    else:
        box_xy = (0.014, 15.7)
        arrow_y = 12.6
        label_y_offset = 0.45

    callout = (
        f"{phantom.mode} ~= DOM cost\n"
        f"({phantom.adjusted_sr:.1f}% vs {dom.adjusted_sr:.1f}% adj SR)"
    )
    ax.annotate(
        callout,
        xy=(phantom.cost, phantom.adjusted_sr),
        xytext=box_xy,
        textcoords="data",
        fontsize=9,
        fontweight="bold",
        color=color,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f7edf4", "edgecolor": color, "alpha": 0.85},
        arrowprops={"arrowstyle": "->", "color": color, "lw": 1.5},
        zorder=5,
    )
    ax.annotate(
        "",
        xy=(dom.cost, dom.adjusted_sr),
        xytext=box_xy,
        textcoords="data",
        arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2, "alpha": 0.85},
        zorder=5,
    )

    ratio = som.cost / phantom.cost
    label = f"SoM/{phantom.mode} cost {ratio:.1f}x"
    x0, x1 = sorted([phantom.cost, som.cost])
    ax.annotate(
        "",
        xy=(x1, arrow_y),
        xytext=(x0, arrow_y),
        arrowprops={"arrowstyle": "<->", "color": "#9a3412", "lw": 1.5},
        zorder=4,
    )
    ax.text(
        (x0 + x1) / 2,
        arrow_y + label_y_offset,
        label,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#9a3412",
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#fff7ed", "edgecolor": "#9a3412", "alpha": 0.85},
        zorder=5,
    )
    print(
        f"[annot] {site}: callout_box={box_xy} DOM=({dom.cost:.4f},{dom.adjusted_sr:.2f}) "
        f"{phantom.mode}=({phantom.cost:.4f},{phantom.adjusted_sr:.2f}) "
        f"SoM=({som.cost:.4f},{som.adjusted_sr:.2f}) ratio={ratio:.2f}x arrow_y={arrow_y}"
    )


def add_class_gap_callout(ax: plt.Axes, site: str) -> None:
    ratios = COST_TABLE.get("deployment_class_ratios", {})
    ratio = ratios.get(site, {})
    if not ratio:
        return
    ax.text(
        0.03,
        0.93,
        (
            f"B0/B1 deployment-class gap: {ratio['ratio_B0_over_B1']:.0f}x\n"
            f"API \\${ratio['avg_B0_API_dollars']:.4f} vs electricity "
            f"\\${ratio['avg_B1_electricity_dollars']:.6f}/ep"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color="#7f1d1d",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fef2f2", "edgecolor": "#ef4444", "alpha": 0.9},
        zorder=6,
    )


def draw_site(ax: plt.Axes, site: str, points: list[Point]) -> None:
    site_points = [point for point in points if point.site == site]
    for index, point in enumerate(site_points):
        ax.scatter(
            point.cost,
            point.adjusted_sr,
            s=92,
            marker=MODEL_MARKERS[point.model],
            color=MODE_COLORS[point.mode],
            edgecolor="#222222",
            linewidth=0.8,
            zorder=3,
        )
        annotate_point(ax, point, index)

    frontier = pareto_frontier(site_points)
    if len(frontier) >= 2:
        ax.plot(
            [point.cost for point in frontier],
            [point.adjusted_sr for point in frontier],
            color="#111111",
            linewidth=1.8,
            linestyle="-",
            zorder=2,
            label="Pareto frontier",
        )
    elif frontier:
        ax.scatter(
            frontier[0].cost,
            frontier[0].adjusted_sr,
            s=180,
            facecolor="none",
            edgecolor="#111111",
            linewidth=1.6,
            zorder=2,
        )

    partials = [point for point in site_points if point.n != point.expected_n]
    if partials:
        partial_text = "\n".join(f"{p.model} {p.mode} n={p.n}/{p.expected_n} partial" for p in partials)
        partial_y = 0.08 if site == "reddit" else 0.96
        partial_va = "bottom" if site == "reddit" else "top"
        ax.text(
            0.03,
            partial_y,
            partial_text,
            transform=ax.transAxes,
            ha="left",
            va=partial_va,
            fontsize=8,
            color="#9a3412",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "#fff7ed", "edgecolor": "#fed7aa"},
        )
    draw_pending(ax, site)

    max_cost = max(point.cost for point in site_points)
    max_sr = max(point.adjusted_sr for point in site_points)
    min_cost = min(point.cost for point in site_points)
    ax.set_title(site.capitalize(), fontsize=12, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(min_cost * 0.65, max_cost * 1.45)
    ax.set_ylim(0, max_sr + 8.0)
    ax.grid(color="#dddddd", linewidth=0.8, which="both")
    ax.set_axisbelow(True)
    ax.set_xlabel("paper_cost_usd per task (log scale)")
    add_deployment_callouts(ax, site, site_points)
    add_class_gap_callout(ax, site)


def main() -> None:
    points = [point for spec in SPECS if (point := load_point(spec)) is not None]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.7), sharey=False)
    for ax, site in zip(axes, ["classifieds", "reddit"]):
        draw_site(ax, site, points)
    axes[0].set_ylabel("Adjusted success rate (%)")

    mode_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color, markeredgecolor="#222222", label=mode)
        for mode, color in MODE_COLORS.items()
    ]
    model_handles = [
        Line2D([0], [0], marker=marker, linestyle="", markerfacecolor="#eeeeee", markeredgecolor="#222222", label=model)
        for model, marker in MODEL_MARKERS.items()
    ]
    frontier_handle = Line2D([0], [0], color="#111111", lw=1.8, label="Pareto frontier")
    fig.legend(
        handles=mode_handles + model_handles + [frontier_handle],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=8,
        frameon=False,
        fontsize=8.5,
    )
    fig.suptitle("Cost vs Adjusted Success: Pareto Frontier", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.855,
        r"B0 reports API token \$; B1 reports electricity-equivalent \$ (different cost classes)",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="#7f1d1d",
    )
    fig.text(
        0.5,
        0.825,
        "B0/B1 ~100x deployment-class gap (reddit 98x, cls 105x)",
        ha="center",
        fontsize=9.5,
        color="#7f1d1d",
    )
    fig.text(
        0.5,
        0.025,
        "Cost source: cost_per_mode.json paper_cost_usd; adjusted SR is recomputed from episode-level adjusted_success.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.76))
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
