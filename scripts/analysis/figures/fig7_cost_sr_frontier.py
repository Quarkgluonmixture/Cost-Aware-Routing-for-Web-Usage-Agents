#!/usr/bin/env python3
"""Live cost-vs-adjusted-SR Pareto frontier for B0/B1 VWA baselines."""

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
OUT = ROOT / "results/phantom_paper/figures/fig7_cost_sr_frontier.png"

MODE_COLORS = {
    "DOM": "#4c78a8",
    "SoM": "#f58518",
    "Vision": "#54a24b",
    "Phantom-DOM": "#b279a2",
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
    model_cost: float | None
    latency_s: float | None
    energy_kwh: float | None


SPECS = [
    ConditionSpec("B0", "classifieds", "DOM", RESULTS / "B0_3mode_classifieds_20260413/phase1_dom_router_0", 234),
    ConditionSpec("B0", "classifieds", "SoM", RESULTS / "B0_3mode_classifieds_20260413/phase1_som_router_0", 234),
    ConditionSpec("B0", "classifieds", "Vision", RESULTS / "B0_3mode_classifieds_20260413/phase1_vision_router_0", 234),
    ConditionSpec("B0", "classifieds", "Phantom-DOM", RESULTS / "B0_phantom_dom_classifieds_20260427/phase1_phantom_dom_router_0", 234),
    ConditionSpec("B0", "reddit", "DOM", RESULTS / "B0_3mode_reddit_20260422/phase1_dom_router_0", 210),
    ConditionSpec("B0", "reddit", "SoM", RESULTS / "B0_3mode_reddit_20260422/phase1_som_router_0", 210),
    ConditionSpec("B0", "reddit", "Vision", RESULTS / "B0_3mode_reddit_20260422/phase1_vision_router_0", 210),
    ConditionSpec("B1", "classifieds", "DOM", RESULTS / "B1_3mode_classifieds_20260413/phase1_dom_router_0", 234),
    ConditionSpec("B1", "classifieds", "SoM", RESULTS / "B1_3mode_classifieds_20260413/phase1_som_router_0", 234),
    ConditionSpec("B1", "classifieds", "Vision", RESULTS / "B1_3mode_classifieds_20260413/phase1_vision_router_0", 234),
    ConditionSpec("B1", "reddit", "DOM", RESULTS / "B1_3mode_reddit_20260413/phase1_dom_router_0", 210),
    ConditionSpec("B1", "reddit", "SoM", RESULTS / "B1_3mode_reddit_20260413/phase1_som_router_0", 210),
    ConditionSpec("B1", "reddit", "Vision", RESULTS / "B1_3mode_reddit_20260413/phase1_vision_router_0", 210),
]


def task_id(path: Path) -> int:
    match = re.search(r"task_(\d+)_summary", path.name)
    if not match:
        raise ValueError(f"Cannot parse task id from {path}")
    return int(match.group(1))


def load_adjusted_sr(condition_dir: Path) -> tuple[float, int]:
    episodes_dir = condition_dir / "episodes"
    files = sorted(episodes_dir.glob("*_summary_v2.json"))
    seen: set[int] = set()
    successes = 0
    for path in files:
        tid = task_id(path)
        if tid in seen:
            print(f"[warn] duplicate task summary ignored: {path}", file=sys.stderr)
            continue
        seen.add(tid)
        with path.open() as f:
            record = json.load(f)
        successes += bool(record.get("adjusted_success", record.get("success", False)))
    if not seen:
        raise FileNotFoundError(f"No episode summaries under {episodes_dir}")
    return 100.0 * successes / len(seen), len(seen)


def load_point(spec: ConditionSpec) -> Point | None:
    summary_path = spec.condition_dir / "condition_summary_v2.json"
    if not summary_path.exists():
        print(f"[warn] missing condition summary: {summary_path}", file=sys.stderr)
        return None
    with summary_path.open() as f:
        summary = json.load(f)
    cost = summary.get("avg_total_cost_usd")
    if cost is None or float(cost) <= 0:
        print(
            f"[warn] skipping {spec.model} {spec.site} {spec.mode}: "
            f"avg_total_cost_usd={cost}",
            file=sys.stderr,
        )
        return None
    adjusted_sr, n = load_adjusted_sr(spec.condition_dir)
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
        model_cost=summary.get("avg_total_model_cost_usd"),
        latency_s=float(latency_ms) / 1000.0 if latency_ms is not None else None,
        energy_kwh=summary.get("avg_total_energy_kwh"),
    )
    latency_text = f"{point.latency_s:.1f}s" if point.latency_s is not None else "NA"
    site_label = "cls" if point.site == "classifieds" else "red"
    print(f"{point.model} {site_label} {point.mode}: SR={point.adjusted_sr:.2f}% cost=${point.cost:.4f} lat={latency_text}")
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
        ("classifieds", "B0", "Phantom-DOM"): (-52, -18),
        ("classifieds", "B1", "DOM"): (12, 24),
        ("classifieds", "B1", "SoM"): (-44, 22),
        ("classifieds", "B1", "Vision"): (8, 8),
        ("reddit", "B0", "DOM"): (8, 8),
        ("reddit", "B0", "SoM"): (8, -22),
        ("reddit", "B0", "Vision"): (-42, 10),
        ("reddit", "B1", "DOM"): (-54, -18),
        ("reddit", "B1", "SoM"): (12, 22),
        ("reddit", "B1", "Vision"): (-50, 24),
    }
    dx, dy = offsets.get((point.site, point.model, point.mode), (8, 8))
    ax.annotate(
        f"{point.mode}\n{point.adjusted_sr:.1f}%/${point.cost:.3f}",
        xy=(point.cost, point.adjusted_sr),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=7.5,
        color="#222222",
        arrowprops={"arrowstyle": "-", "color": "#bdbdbd", "lw": 0.7},
    )


def draw_pending(ax: plt.Axes, site: str) -> None:
    pending = "pending: B0 Phantom-SoM, B1 Phantom"
    if site == "reddit":
        pending += ", Phantom-DOM"
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
        ax.text(
            0.03,
            0.96,
            partial_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#9a3412",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "#fff7ed", "edgecolor": "#fed7aa"},
        )
    draw_pending(ax, site)

    max_cost = max(point.cost for point in site_points)
    max_sr = max(point.adjusted_sr for point in site_points)
    min_cost = min(point.cost for point in site_points)
    ax.set_title(site.capitalize(), fontsize=12, fontweight="bold")
    ax.set_xlim(max(0.0, min_cost - 0.004), max_cost + 0.008)
    ax.set_ylim(0, max_sr + 8.0)
    ax.grid(color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("avg_total_cost_usd per task")


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
        ncol=7,
        frameon=False,
        fontsize=8.5,
    )
    fig.suptitle("Cost vs Adjusted Success: Pareto Frontier", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.025,
        "Cost and latency come from condition_summary_v2.json; adjusted SR is recomputed from episode-level adjusted_success.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.84))
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
