#!/usr/bin/env python3
"""[Layer 3d] Efficiency — B0 vs B1 deployment-class cost gap.

Output:
- results/phantom_paper/figures/fig3d_cost_sr_frontier.png

Layer 3d: B0 (API token $) vs B1 (electricity-equivalent $) deployment-class gap.
Single message: ~100× gap (reddit 98×, cls 105×). For intra-baseline token-cost
ratio (P-SoM ≈ DOM), see fig3a_token_cost_intra_baseline.

Design choices (simplified vs prior cluttered version):
- 4 modes only: DOM / SoM / Vision / P-SoM (P-text excluded — no deployment-class
  signal beyond P-SoM; covered in fig1ab_cascade_diamond)
- Log-scale x-axis to fit B0 (~$0.04) and B1 (~$0.0004) in same plot
- No per-cell labels; modes via legend
- Single prominent annotation: deployment-class gap

See docs/checkpoints/paper_planning.md §3 Layer 3d framework.
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
    "Phantom-prompt": "#9467bd",
}
MODE_DISPLAY = {"Phantom-SoM": "P-SoM", "Phantom-prompt": "P-prompt"}
MODEL_MARKERS = {"B0": "o", "B1": "s"}


def _phantom_prompt_run(baseline: str, site: str) -> Path | None:
    candidates = sorted(RESULTS.glob(f"{baseline}_phantom_prompt_{site}_*/phase1_phantom_prompt_router_0"))
    return candidates[-1] if candidates else None


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
    adj_sr: float
    cost: float


SPECS = [
    ConditionSpec("B0", "classifieds", "DOM", RESULTS / "B0_3mode_classifieds_20260413/phase1_dom_router_0", 234),
    ConditionSpec("B0", "classifieds", "SoM", RESULTS / "B0_3mode_classifieds_20260413/phase1_som_router_0", 234),
    ConditionSpec("B0", "classifieds", "Vision", RESULTS / "B0_3mode_classifieds_20260413/phase1_vision_router_0", 234),
    ConditionSpec("B0", "classifieds", "Phantom-SoM", RESULTS / "B0_phantom_som_classifieds_20260426/phase1_phantom_som_router_0", 234),
    ConditionSpec("B0", "reddit", "DOM", RESULTS / "B0_3mode_reddit_20260422/phase1_dom_router_0", 210),
    ConditionSpec("B0", "reddit", "SoM", RESULTS / "B0_3mode_reddit_20260422/phase1_som_router_0", 210),
    ConditionSpec("B0", "reddit", "Vision", RESULTS / "B0_3mode_reddit_20260422/phase1_vision_router_0", 210),
    ConditionSpec("B0", "reddit", "Phantom-SoM", RESULTS / "B0_phantom_som_reddit_20260428/phase1_phantom_som_router_0", 210),
    ConditionSpec("B1", "classifieds", "DOM", RESULTS / "B1_3mode_classifieds_20260413/phase1_dom_router_0", 234),
    ConditionSpec("B1", "classifieds", "SoM", RESULTS / "B1_3mode_classifieds_20260413/phase1_som_router_0", 234),
    ConditionSpec("B1", "classifieds", "Vision", RESULTS / "B1_3mode_classifieds_20260413/phase1_vision_router_0", 234),
    ConditionSpec("B1", "reddit", "DOM", RESULTS / "B1_3mode_reddit_20260413/phase1_dom_router_0", 210),
    ConditionSpec("B1", "reddit", "SoM", RESULTS / "B1_3mode_reddit_20260413/phase1_som_router_0", 210),
    ConditionSpec("B1", "reddit", "Vision", RESULTS / "B1_3mode_reddit_20260413/phase1_vision_router_0", 210),
]
OPTIONAL_SPECS = [
    ConditionSpec("B1", "classifieds", "Phantom-SoM", RESULTS / "B1_phantom_som_classifieds_20260428/phase1_phantom_som_router_0", 234),
    ConditionSpec("B1", "reddit", "Phantom-SoM", RESULTS / "B1_phantom_reddit_20260426/phase1_phantom_som_router_0", 210),
]
# Phantom-prompt entries (B0/B1 × cls/red) when run dirs exist
for _b in ("B0", "B1"):
    for _site, _expected in (("reddit", 210), ("classifieds", 234)):
        _run = _phantom_prompt_run(_b, _site)
        if _run is not None:
            OPTIONAL_SPECS.append(ConditionSpec(_b, _site, "Phantom-prompt", _run, _expected))


def has_episodes(spec: ConditionSpec) -> bool:
    return any((spec.condition_dir / "episodes").glob("*_summary_v2.json"))


SPECS = SPECS + [s for s in OPTIONAL_SPECS if has_episodes(s)]


def task_id(path: Path) -> int:
    m = re.search(r"task_(\d+)_summary", path.name)
    if not m:
        raise ValueError(path.name)
    return int(m.group(1))


def load_cost_per_mode() -> dict:
    if not COST_PER_MODE.exists():
        sys.exit(f"missing {COST_PER_MODE}; run `make aggregate-cost-electricity`")
    return json.loads(COST_PER_MODE.read_text())


def paper_cost(table: dict, spec: ConditionSpec) -> float | None:
    cell = table.get("cells", {}).get(spec.model, {}).get(spec.site, {}).get(MODE_DISPLAY.get(spec.mode, spec.mode), {})
    return float(cell["paper_cost_usd"]) if cell.get("available") and cell.get("paper_cost_usd") is not None else None


def episode_adj_sr(condition_dir: Path) -> tuple[float, int]:
    seen: set[int] = set()
    succ = 0
    for path in sorted((condition_dir / "episodes").glob("*_summary_v2.json")):
        tid = task_id(path)
        if tid in seen:
            continue
        seen.add(tid)
        rec = json.loads(path.read_text())
        succ += bool(rec.get("adjusted_success", rec.get("success", False)))
    n = len(seen)
    return (100.0 * succ / n if n else 0.0, n)


def load_point(spec: ConditionSpec, cost_table: dict) -> Point | None:
    cost = paper_cost(cost_table, spec)
    if cost is None or cost <= 0:
        print(f"[skip] {spec.model} {spec.site} {spec.mode}: no paper_cost", file=sys.stderr)
        return None
    adj_sr, n = episode_adj_sr(spec.condition_dir)
    if n == 0:
        print(f"[skip] {spec.model} {spec.site} {spec.mode}: no episodes", file=sys.stderr)
        return None
    return Point(spec.model, spec.site, spec.mode, n, spec.expected_n, adj_sr, cost)


def draw_panel(ax: plt.Axes, site: str, points: list[Point], cost_table: dict) -> None:
    site_points = [p for p in points if p.site == site]
    for p in site_points:
        ax.scatter(p.cost, p.adj_sr, s=160, marker=MODEL_MARKERS[p.model],
                   color=MODE_COLORS[p.mode], edgecolor="#222222", linewidth=1.2, zorder=3)

    # Cluster labels (B0 / B1) instead of per-cell labels
    b0_costs = [p.cost for p in site_points if p.model == "B0"]
    b1_costs = [p.cost for p in site_points if p.model == "B1"]
    if b0_costs:
        b0_x = sum(b0_costs) / len(b0_costs)
        ax.text(b0_x, max(p.adj_sr for p in site_points if p.model == "B0") + 3.5,
                "B0 (API token $)", ha="center", fontsize=10, fontweight="bold", color="#222222")
    if b1_costs:
        b1_x = sum(b1_costs) / len(b1_costs)
        ax.text(b1_x, max(p.adj_sr for p in site_points if p.model == "B1") + 3.5,
                "B1 (electricity $)", ha="center", fontsize=10, fontweight="bold", color="#222222")

    # Deployment-class gap annotation (the headline message of fig3d)
    ratios = cost_table.get("deployment_class_ratios", {}).get(site, {})
    if ratios:
        ax.text(
            0.5, 0.96,
            (f"B0/B1 deployment-class gap: {ratios['ratio_B0_over_B1']:.0f}×\n"
             f"API \\${ratios['avg_B0_API_dollars']:.4f}/ep  vs  "
             f"electricity \\${ratios['avg_B1_electricity_dollars']:.6f}/ep"),
            transform=ax.transAxes, ha="center", va="top",
            fontsize=10.5, fontweight="bold", color="#7f1d1d",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "#fef2f2", "edgecolor": "#ef4444", "alpha": 0.92},
            zorder=6,
        )

    # Partial-data note (e.g. B1 P-SoM 210/234)
    partials = [p for p in site_points if p.n != p.expected_n]
    if partials:
        ax.text(
            0.02, 0.04,
            "\n".join(f"{p.model} {MODE_DISPLAY.get(p.mode, p.mode)} partial n={p.n}/{p.expected_n}" for p in partials),
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7.5, color="#777777",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#bdbdbd", "linestyle": "dotted"},
        )

    max_cost = max(p.cost for p in site_points)
    min_cost = min(p.cost for p in site_points)
    max_sr = max(p.adj_sr for p in site_points)
    ax.set_title(site.capitalize(), fontsize=12, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(min_cost * 0.45, max_cost * 2.0)
    ax.set_ylim(0, max_sr + 14.0)
    ax.grid(color="#e8e8e8", linewidth=0.8, which="both")
    ax.set_axisbelow(True)
    ax.set_xlabel("paper_cost_usd per task (log scale)")


def main() -> None:
    cost_table = load_cost_per_mode()
    points = [p for spec in SPECS if (p := load_point(spec, cost_table)) is not None]
    if not points:
        sys.exit("no points loaded")

    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0), sharey=False)
    for ax, site in zip(axes, ("classifieds", "reddit")):
        draw_panel(ax, site, points, cost_table)
    axes[0].set_ylabel("Adjusted success rate (%)")

    # Legend: 4 modes × 2 baselines
    mode_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color,
               markeredgecolor="#222222", markersize=10, label=MODE_DISPLAY.get(mode, mode))
        for mode, color in MODE_COLORS.items()
    ]
    model_handles = [
        Line2D([0], [0], marker=marker, linestyle="", markerfacecolor="#cccccc",
               markeredgecolor="#222222", markersize=10, label=model)
        for model, marker in MODEL_MARKERS.items()
    ]
    fig.legend(handles=mode_handles + model_handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.05), ncol=6, frameon=False, fontsize=10)

    fig.suptitle("Deployment-Class Cost Gap: B0 (API) vs B1 (local electricity)",
                 fontsize=14, fontweight="bold")
    fig.text(
        0.5, 0.94,
        r"Different cost classes — B0 reports API token \$, B1 reports electricity-equivalent \$ "
        r"($0.12/kWh UK industrial). Not directly ratio-comparable.",
        ha="center", fontsize=9, color="#555555",
    )
    fig.text(
        0.5, -0.03,
        "Cost source: cost_per_mode.json paper_cost_usd. For intra-baseline P-SoM/DOM token-cost ratio (Layer 3a), see fig3a.",
        ha="center", fontsize=8, color="#666666",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
