#!/usr/bin/env python3
"""[Efficiency 3c] Efficiency dimension — per-step latency by mode.

Output:
- results/phantom_paper/figures/fig3c_latency_per_step.png

Efficiency 3c: mean per-step latency computed as avg_total_latency_ms /
avg_steps, with p95 step latency markers when available.

See docs/checkpoints/paper_planning.md §3 Efficiency dimension framework.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    venv_python = Path(__file__).resolve().parents[3] / ".venv/bin/python3"
    if venv_python.exists() and Path(sys.executable) != venv_python:
        os.execv(str(venv_python), [str(venv_python), *sys.argv])
    raise

try:
    from scripts.analysis.lib.run_registry import BASELINES, PAPER_MODES, get_cells
except ModuleNotFoundError:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from scripts.analysis.lib.run_registry import BASELINES, PAPER_MODES, get_cells

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/phantom_paper/figures/fig3c_latency_per_step.png"

MODE_COLORS = {
    "DOM": "#4c78a8",
    "SoM": "#f58518",
    "Vision": "#54a24b",
    "P-text": "#e45756",
    "P-prompt": "#9467bd",
    "P-SoM": "#b279a2",
}
# 3-model deep-update 2026-05-18: PANELS dynamic from BASELINES registry
# (was 4 hardcoded pre-B2 Gemma3-VL inclusion 2026-05-14). Latency canonical
# = retry-adjusted per A2.7 estimand spec (B-1410 2026-05-18); per-step
# computation total_minus_retry_ms / avg_steps when field present (TODO
# wire condition_summary `avg_total_latency_minus_retry_ms` once aggregator
# B-1410 round-2 propagates field to per-condition output).
PANELS = [
    (baseline, site, f"{baseline} {'cls' if site == 'classifieds' else 'red'}")
    for baseline in BASELINES
    for site in ("classifieds", "reddit")
]


@dataclass(frozen=True)
class LatencyCell:
    mean_step_ms: float
    p95_step_ms: float | None
    n: int


def load_panel(baseline: str, site: str) -> dict[str, LatencyCell]:
    out: dict[str, LatencyCell] = {}
    for cell in get_cells(baseline=baseline, site=site):
        path = cell.condition_summary_path
        if not path.exists():
            print(f"[warn] {baseline} {site} {cell.mode} missing condition_summary_v2.json", file=sys.stderr)
            continue
        summary = json.loads(path.read_text())
        total = summary.get("avg_total_latency_ms")
        steps = summary.get("avg_steps")
        if total is None or steps in (None, 0):
            print(f"[warn] {baseline} {site} {cell.mode} missing latency fields", file=sys.stderr)
            continue
        out[cell.mode] = LatencyCell(
            mean_step_ms=float(total) / float(steps),
            p95_step_ms=None if summary.get("p95_step_latency_ms") is None else float(summary["p95_step_latency_ms"]),
            n=int(summary.get("episodes") or cell.actual_n),
        )
    return out


def ratio_text(cells: dict[str, LatencyCell], baseline: str, site: str) -> str | None:
    psom = cells.get("P-SoM")
    som = cells.get("SoM")
    dom = cells.get("DOM")
    lines = []
    if psom and som and som.mean_step_ms > 0:
        ratio = psom.mean_step_ms / som.mean_step_ms
        lines.append(f"P-SoM/SoM {ratio:.2f}×")
        if baseline == "B0" and ratio >= 1.0:
            print(f"[warn] anomaly to investigate: {baseline} {site} P-SoM latency >= SoM", file=sys.stderr)
    if psom and dom and dom.mean_step_ms > 0:
        lines.append(f"P-SoM/DOM {psom.mean_step_ms / dom.mean_step_ms:.2f}×")
    return "\n".join(lines) if lines else None


def draw_panel(ax: plt.Axes, baseline: str, site: str, title: str) -> None:
    cells = load_panel(baseline, site)
    x = np.arange(len(PAPER_MODES))
    values = [cells[m].mean_step_ms if m in cells else 0.0 for m in PAPER_MODES]
    bars = ax.bar(
        x,
        values,
        width=0.68,
        color=[MODE_COLORS[m] if m in cells else "#d9d9d9" for m in PAPER_MODES],
        edgecolor="white",
        linewidth=0.8,
    )
    p95_x = []
    p95_y = []
    for i, mode in enumerate(PAPER_MODES):
        cell = cells.get(mode)
        if cell is None:
            bars[i].set_hatch("//")
            ax.text(i, 150, "N/A\npending", ha="center", va="bottom", fontsize=7, color="#666666")
            print(f"[warn] {baseline} {site} missing latency for {mode}", file=sys.stderr)
            continue
        ax.text(i, cell.mean_step_ms + 180, f"{cell.mean_step_ms:.0f}", ha="center", va="bottom", fontsize=7.3)
        if cell.p95_step_ms is not None:
            p95_x.append(i)
            p95_y.append(cell.p95_step_ms)
    if p95_x:
        ax.scatter(p95_x, p95_y, marker="D", s=28, color="#222222", label="p95 step", zorder=3)
    note = ratio_text(cells, baseline, site)
    if note:
        ax.text(0.98, 0.95, note, transform=ax.transAxes, ha="right", va="top", fontsize=8.0, bbox={"boxstyle": "round,pad=0.3", "facecolor": "#fff8e1", "edgecolor": "#c28f2c", "alpha": 0.92})
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.set_xticks(x, PAPER_MODES, rotation=30, ha="right")
    ax.set_ylabel("Mean per-step latency (ms)")
    ymax = max(values + p95_y + [1000])
    ax.set_ylim(0, ymax * 1.18)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9.5, "figure.dpi": 150})
    # 3-model deep-update 2026-05-18: (n_baselines, 2 sites) grid.
    n_baselines = len(BASELINES)
    fig, axes = plt.subplots(n_baselines, 2, figsize=(13.5, max(6.0, 4.0 * n_baselines)), sharey=False)
    if n_baselines == 1:
        axes = axes.reshape(1, -1)
    for ax, (baseline, site, title) in zip(axes.ravel(), PANELS):
        draw_panel(ax, baseline, site, title)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Efficiency 3c — per-step latency separated from token cost", fontsize=14, fontweight="bold", y=1.02)
    fig.text(0.5, 0.02, "Mean = avg_total_latency_ms / avg_steps from condition_summary_v2.json; diamonds show p95_step_latency_ms when present.", ha="center", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
