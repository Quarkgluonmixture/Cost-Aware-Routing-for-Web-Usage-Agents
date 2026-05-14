#!/usr/bin/env python3
"""[Efficiency 3a] Efficiency dimension — intra-baseline cost vs adjusted SR (B0 + B1).

Output:
- results/phantom_paper/figures/fig3a_token_cost_intra_baseline.png

Efficiency 3a: per-baseline cost vs adjusted SR with per-panel Pareto frontier.
- B0 panels: x = avg_total_cost_usd (API token $).
- B1 panels: x = avg_total_energy_kwh * $0.12/kWh (UK industrial electricity $).
  B1 token-cost field is computed with B0 per-token rates and is therefore an
  artifact; do NOT use it. Electricity cost is the deployment-class proxy.

Pareto frontier is computed within each panel (no cross-baseline ratios).

See docs/checkpoints/paper_planning.md §3 Efficiency dimension (sub-code 3a) framework.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from scripts.analysis.lib.run_registry import get_cells
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from scripts.analysis.lib.run_registry import get_cells


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/phantom_paper/figures/fig3a_token_cost_intra_baseline.png"

ELECTRICITY_USD_PER_KWH = 0.12  # UK industrial 2026

MODE_COLORS = {
    "DOM": "#4c78a8",
    "SoM": "#f58518",
    "Vision": "#54a24b",
    "P-SoM": "#b279a2",
    "P-text": "#e45756",
    "P-prompt": "#9467bd",
}
MODE_DISPLAY = {"P-SoM": "P-SoM", "P-text": "P-text", "P-prompt": "P-prompt"}


@dataclass(frozen=True)
class Cell:
    baseline: str
    site: str
    mode: str
    cost: float
    adj_sr: float
    n: int


SPECS = [
    (cell.baseline, cell.site, cell.mode, cell.run_dir / cell.condition_subdir, cell.expected_n)
    for baseline in ("B0", "B1")
    for site in ("classifieds", "reddit")
    for cell in get_cells(baseline=baseline, site=site)
]


def task_id(path: Path) -> int:
    m = re.search(r"task_(\d+)_summary", path.name)
    if not m:
        raise ValueError(path.name)
    return int(m.group(1))


def load_cell(baseline: str, site: str, mode: str, cond_dir: Path, expected_n: int) -> Cell | None:
    summary_path = cond_dir / "condition_summary_v2.json"
    if not summary_path.exists():
        print(f"[warn] missing {summary_path}", file=sys.stderr)
        return None
    summary = json.loads(summary_path.read_text())
    if baseline == "B0":
        cost = summary.get("avg_total_cost_usd")
    else:
        kwh = summary.get("avg_total_energy_kwh")
        cost = None if kwh is None else float(kwh) * ELECTRICITY_USD_PER_KWH
    if cost is None:
        print(f"[warn] {baseline} {site} {mode}: no cost field available in {summary_path}", file=sys.stderr)
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
        succ += bool(rec.get("success", False))  # §139.8: adjusted_success retired
    n = len(seen)
    if n < expected_n * 0.9:
        print(f"[warn] {baseline} {site} {mode}: partial n={n}/{expected_n}", file=sys.stderr)
    return Cell(baseline=baseline, site=site, mode=mode, cost=float(cost),
                adj_sr=100.0 * succ / n if n else 0.0, n=n)


def pareto_frontier(cells: list[Cell]) -> list[Cell]:
    """Lower cost + higher SR is better."""
    out: list[Cell] = []
    best = -1.0
    for cell in sorted(cells, key=lambda c: (c.cost, -c.adj_sr)):
        if cell.adj_sr > best + 1e-9:
            out.append(cell)
            best = cell.adj_sr
    return out


def draw_panel(ax: plt.Axes, baseline: str, site: str, cells: list[Cell]) -> None:
    panel_cells = [c for c in cells if c.baseline == baseline and c.site == site]
    cost_unit = "API token cost per task (USD)" if baseline == "B0" else "electricity-equivalent cost per task (USD)"
    cost_class = "API$" if baseline == "B0" else "electricity$"
    for cell in panel_cells:
        color = MODE_COLORS.get(cell.mode, "#666666")
        label_mode = MODE_DISPLAY.get(cell.mode, cell.mode)
        ax.scatter(cell.cost, cell.adj_sr, color=color, s=140, edgecolor="white", linewidth=1.5, zorder=3)
        offset = {"DOM": (8, 8), "SoM": (8, -16), "Vision": (-42, 10),
                  "P-SoM": (-72, 4), "P-text": (-50, -18),
                  "P-prompt": (-72, -22)}.get(cell.mode, (8, 8))
        ax.annotate(
            f"{label_mode}\n{cell.adj_sr:.1f}%",
            xy=(cell.cost, cell.adj_sr),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.0,
            color="#222222",
            arrowprops={"arrowstyle": "-", "color": "#cccccc", "lw": 0.7},
        )

    # P-prompt placeholder annotation when this panel lacks a P-prompt point
    if not any(c.mode == "P-prompt" for c in panel_cells):
        ax.text(
            0.02, 0.97,
            "P-prompt: pending",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8.0, color=MODE_COLORS["P-prompt"],
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#f3eaf7", "edgecolor": MODE_COLORS["P-prompt"], "alpha": 0.7, "linestyle": "dotted"},
        )
    frontier = pareto_frontier(panel_cells)
    if len(frontier) >= 2:
        ax.plot([c.cost for c in frontier], [c.adj_sr for c in frontier],
                color="#444444", linewidth=1.2, linestyle="--", zorder=2)

    # Annotate P-SoM/DOM cost ratio when both present
    dom = next((c for c in panel_cells if c.mode == "DOM"), None)
    psom = next((c for c in panel_cells if c.mode == "P-SoM"), None)
    if dom and psom and dom.cost > 0:
        ratio = psom.cost / dom.cost
        ax.text(
            0.98, 0.02,
            f"P-SoM/DOM cost ratio: {ratio:.2f}×\n"
            f"(P-SoM ${psom.cost:.4f}/ep vs DOM ${dom.cost:.4f}/ep)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.0, color="#444444",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "#fff8e1", "edgecolor": "#c28f2c", "alpha": 0.92},
        )

    n_label = panel_cells[0].n if panel_cells else "?"
    ax.set_title(f"{baseline} {site} ({cost_class}, N={n_label})", fontsize=10.5, fontweight="bold")
    ax.set_xlabel(cost_unit, fontsize=9)
    ax.set_ylabel("Adjusted success rate (%)", fontsize=9)
    ax.grid(axis="both", color="#e8e8e8", linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    cells = [c for c in (load_cell(b, s, m, sub, n) for b, s, m, sub, n in SPECS) if c is not None]
    if not cells:
        # 2026-05-10: fail-soft placeholder so make figures chain doesn't crash
        # when no cells loaded (typical when source aggregator outputs empty
        # post-F01 manifest grade chain).
        from pathlib import Path
        out = Path(__file__).resolve().parents[3] / "results/phantom_paper/figures/fig3a_token_cost_intra_baseline.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "fig3a_token_cost_intra_baseline\n\n[data pending]\n\nno cells loaded — run aggregators first",
                ha="center", va="center", fontsize=11, color="gray")
        ax.set_axis_off()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[fig3a] placeholder written → {out}")
        return

    plt.rcParams.update({"font.size": 9.5, "figure.dpi": 150})
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    panel_specs = [("B0", "classifieds"), ("B0", "reddit"), ("B1", "classifieds"), ("B1", "reddit")]
    for ax, (baseline, site) in zip(axes, panel_specs):
        draw_panel(ax, baseline, site, cells)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=MODE_COLORS[m], markeredgecolor="white",
               markersize=10, label=MODE_DISPLAY.get(m, m))
        for m in ("DOM", "SoM", "Vision", "P-SoM", "P-text", "P-prompt")
    ]
    legend_handles.append(Line2D([0], [0], color="#444444", linewidth=1.2, linestyle="--", label="Pareto frontier (per-panel)"))
    fig.legend(handles=legend_handles, loc="upper center", ncol=6, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, 1.04))

    fig.suptitle("Intra-baseline cost vs adjusted SR (B0 API$ / B1 electricity$)",
                 fontsize=13, fontweight="bold", y=1.06)
    fig.text(0.5, -0.02,
             "B0 = API token \\$. B1 = electricity-equivalent (\\$0.12/kWh UK industrial). Per-panel Pareto only.",
             ha="center", fontsize=8.5, color="#666666")
    fig.tight_layout(rect=(0, 0, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
