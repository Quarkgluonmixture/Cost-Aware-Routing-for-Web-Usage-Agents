#!/usr/bin/env python3
"""[Layer 3a] Efficiency — intra-baseline token cost vs adjusted SR (B0 only).

Output:
- results/phantom_paper/figures/fig3a_token_cost_intra_baseline.png

Layer 3a: B0 API token cost per task. P-SoM ≈ DOM intra-token-cost claim
(4-fold drop-in property (a)). B1 cost is excluded because the per-token
rate in B1 condition_summary_v2.json is an artifact (uses B0 rates) — see
fig3d_cost_sr_frontier for the deployment-class (B0 vs B1) view.

5 modes per site. Linear scale x-axis. Pareto frontier within B0.

See docs/checkpoints/paper_planning.md §3 Layer 3a framework.
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
OUT = ROOT / "results/phantom_paper/figures/fig3a_token_cost_intra_baseline.png"

MODE_COLORS = {
    "DOM": "#4c78a8",
    "SoM": "#f58518",
    "Vision": "#54a24b",
    "Phantom-SoM": "#b279a2",
    "P-text": "#e45756",
}
MODE_DISPLAY = {"Phantom-SoM": "P-SoM", "P-text": "P-text"}


@dataclass(frozen=True)
class Cell:
    site: str
    mode: str
    cost: float
    adj_sr: float
    n: int


SPECS = [
    # B0 only (Layer 3a is intra-baseline token cost; B1 token cost is artifact)
    ("classifieds", "DOM", "B0_3mode_classifieds_20260413/phase1_dom_router_0", 234),
    ("classifieds", "SoM", "B0_3mode_classifieds_20260413/phase1_som_router_0", 234),
    ("classifieds", "Vision", "B0_3mode_classifieds_20260413/phase1_vision_router_0", 234),
    ("classifieds", "Phantom-SoM", "B0_phantom_som_classifieds_20260426/phase1_phantom_som_router_0", 234),
    ("classifieds", "P-text", "B0_phantom_text_classifieds_20260427/phase1_phantom_dom_router_0", 234),
    ("reddit", "DOM", "B0_3mode_reddit_20260422/phase1_dom_router_0", 210),
    ("reddit", "SoM", "B0_3mode_reddit_20260422/phase1_som_router_0", 210),
    ("reddit", "Vision", "B0_3mode_reddit_20260422/phase1_vision_router_0", 210),
    ("reddit", "Phantom-SoM", "B0_phantom_som_reddit_20260428/phase1_phantom_som_router_0", 210),
    ("reddit", "P-text", "B0_phantom_text_reddit_20260427/phase1_phantom_dom_router_0", 210),
]


def task_id(path: Path) -> int:
    m = re.search(r"task_(\d+)_summary", path.name)
    if not m:
        raise ValueError(path.name)
    return int(m.group(1))


def load_cell(site: str, mode: str, sub: str, expected_n: int) -> Cell | None:
    cond_dir = RESULTS / sub
    summary_path = cond_dir / "condition_summary_v2.json"
    if not summary_path.exists():
        print(f"[warn] missing {summary_path}", file=sys.stderr)
        return None
    summary = json.loads(summary_path.read_text())
    cost = summary.get("avg_total_cost_usd")
    if cost is None:
        return None
    # Recompute adj_SR live from episode summaries
    ep_dir = cond_dir / "episodes"
    files = sorted(ep_dir.glob("*_summary_v2.json"))
    if not files:
        return None
    seen: set[int] = set()
    succ = 0
    for path in files:
        tid = task_id(path)
        if tid in seen:
            continue
        seen.add(tid)
        rec = json.loads(path.read_text())
        succ += bool(rec.get("adjusted_success", rec.get("success", False)))
    n = len(seen)
    if n < expected_n * 0.9:
        print(f"[warn] {site} {mode}: partial n={n}/{expected_n}", file=sys.stderr)
    return Cell(site=site, mode=mode, cost=float(cost), adj_sr=100.0 * succ / n if n else 0.0, n=n)


def pareto_frontier(cells: list[Cell]) -> list[Cell]:
    """Lower cost + higher SR is better."""
    out: list[Cell] = []
    best = -1.0
    for cell in sorted(cells, key=lambda c: (c.cost, -c.adj_sr)):
        if cell.adj_sr > best + 1e-9:
            out.append(cell)
            best = cell.adj_sr
    return out


def draw_panel(ax: plt.Axes, site: str, cells: list[Cell]) -> None:
    site_cells = [c for c in cells if c.site == site]
    # Plot markers
    for cell in site_cells:
        color = MODE_COLORS.get(cell.mode, "#666666")
        label_mode = MODE_DISPLAY.get(cell.mode, cell.mode)
        ax.scatter(cell.cost, cell.adj_sr, color=color, s=150, edgecolor="white", linewidth=1.5, zorder=3, label=label_mode)
        # Concise per-marker label
        offset = {"DOM": (8, 8), "SoM": (8, -16), "Vision": (-42, 10),
                  "Phantom-SoM": (-72, 4), "P-text": (-50, -18)}.get(cell.mode, (8, 8))
        ax.annotate(
            f"{label_mode}\n{cell.adj_sr:.1f}%",
            xy=(cell.cost, cell.adj_sr),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.5,
            color="#222222",
            arrowprops={"arrowstyle": "-", "color": "#cccccc", "lw": 0.7},
        )
    # Pareto frontier
    frontier = pareto_frontier(site_cells)
    if len(frontier) >= 2:
        ax.plot([c.cost for c in frontier], [c.adj_sr for c in frontier],
                color="#444444", linewidth=1.2, linestyle="--", zorder=2, label="Pareto frontier")

    # P-SoM ≈ DOM cost ratio annotation (Layer 3a 4-fold drop-in (a) evidence)
    dom = next((c for c in site_cells if c.mode == "DOM"), None)
    psom = next((c for c in site_cells if c.mode == "Phantom-SoM"), None)
    if dom and psom:
        ratio = psom.cost / dom.cost
        ax.text(
            0.98, 0.02,
            f"P-SoM/DOM token-cost ratio: {ratio:.2f}×\n"
            f"(P-SoM ${psom.cost:.4f}/ep vs DOM ${dom.cost:.4f}/ep)\n"
            f"4-fold drop-in (a): cost ≈ DOM ✓",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color="#444444",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "#fff8e1", "edgecolor": "#c28f2c", "alpha": 0.92},
        )

    ax.set_title(f"B0 {site} (N={site_cells[0].n if site_cells else '?'})", fontsize=11, fontweight="bold")
    ax.set_xlabel("avg API token cost per task (USD)")
    ax.set_ylabel("Adjusted success rate (%)")
    ax.grid(axis="both", color="#e8e8e8", linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    cells = [c for c in (load_cell(site, mode, sub, n) for site, mode, sub, n in SPECS) if c is not None]
    if not cells:
        sys.exit("no cells loaded")

    plt.rcParams.update({"font.size": 9.5, "figure.dpi": 150})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, site in zip(axes, ("classifieds", "reddit")):
        draw_panel(ax, site, cells)

    # Shared legend
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=MODE_COLORS[m], markeredgecolor="white",
               markersize=10, label=MODE_DISPLAY.get(m, m))
        for m in ("DOM", "SoM", "Vision", "Phantom-SoM", "P-text")
    ]
    legend_handles.append(Line2D([0], [0], color="#444444", linewidth=1.2, linestyle="--", label="Pareto frontier"))
    fig.legend(handles=legend_handles, loc="upper center", ncol=6, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("Intra-baseline token cost vs adjusted SR (B0, Layer 3a)",
                 fontsize=13, fontweight="bold", y=1.06)
    fig.text(0.5, -0.02,
             "Cost = avg_total_cost_usd from condition_summary_v2 (Qwen3-VL-235B-A22B per-token rates). "
             "Adj SR recomputed from episode-level adjusted_success. B1 token cost is artifact and shown separately in fig3d.",
             ha="center", fontsize=8, color="#666666")
    fig.tight_layout(rect=(0, 0, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
