#!/usr/bin/env python3
"""[Micro 2b] Micro dimension — target-page hit rate by mode and site.

Output:
- results/phantom_paper/figures/fig2b_target_hit_rate.png

Micro 2b: target-page hit rate per mode x site, with axis-1 and axis-2
paired contrast annotations.

See docs/checkpoints/paper_planning.md §3 Micro dimension framework.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

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
IN_JSON = ROOT / "docs/analysis/cross_sites/axis1_microbehavior.json"
OUT = ROOT / "results/phantom_paper/figures/fig2b_target_hit_rate.png"

MODE_COLORS = {
    "DOM": "#4c78a8",
    "SoM": "#f58518",
    "Vision": "#54a24b",
    "P-text": "#e45756",
    "P-prompt": "#9467bd",
    "P-SoM": "#b279a2",
}
MODE_ALIASES = {"Phantom-SoM": "P-SoM", "Phantom-prompt": "P-prompt"}
# 3-model deep-update 2026-05-18: drive PANELS from BASELINES registry
# (was 4 hardcoded entries pre-B2 Gemma3-VL inclusion 2026-05-14).
PANELS = [
    (baseline, site, f"{baseline} {'cls' if site == 'classifieds' else 'red'}")
    for baseline in BASELINES
    for site in ("classifieds", "reddit")
]


def canonical_mode(mode: str) -> str:
    return MODE_ALIASES.get(mode, mode)


def load_json() -> dict[str, Any]:
    return json.loads(IN_JSON.read_text())


def preaggregated_rates(data: dict[str, Any], baseline: str, site: str) -> dict[str, float]:
    raw = data.get("metrics_per_task_per_mode", {}).get(baseline, {}).get(site, {}) or {}
    out: dict[str, float] = {}
    for mode, row in raw.items():
        if not isinstance(row, dict):
            continue
        value = row.get("target_hit_rate")
        if value is not None:
            out[canonical_mode(mode)] = 100.0 * float(value)
    return out


def fallback_rates(baseline: str, site: str) -> dict[str, float]:
    """Best-effort fallback from condition summaries if the micro JSON is missing."""
    out: dict[str, float] = {}
    for cell in get_cells(baseline=baseline, site=site):
        summary_path = cell.condition_summary_path
        if not summary_path.exists():
            print(f"[warn] {baseline} {site} {cell.mode} missing {summary_path.name}", file=sys.stderr)
            continue
        summary = json.loads(summary_path.read_text())
        for key in ("target_hit_rate", "avg_target_hit_rate"):
            if key in summary:
                out[cell.mode] = 100.0 * float(summary[key])
                break
    return out


def contrast_delta(data: dict[str, Any], baseline: str, site: str, key: str) -> float | None:
    row = data.get("axis_contrasts", {}).get(baseline, {}).get(site, {}).get(key, {}) or {}
    if row.get("skipped") or row.get("target_hit_rate_diff_pct_pts") is None:
        return None
    return float(row["target_hit_rate_diff_pct_pts"])


def draw_delta(ax: plt.Axes, x0: int, x1: int, y: float, label: str, color: str) -> None:
    ax.annotate(
        "",
        xy=(x1, y),
        xytext=(x0, y),
        arrowprops={"arrowstyle": "<->", "color": color, "lw": 1.2},
    )
    ax.text((x0 + x1) / 2, y + 1.5, label, ha="center", va="bottom", fontsize=7.5, color=color)


def draw_panel(ax: plt.Axes, data: dict[str, Any], baseline: str, site: str, title: str) -> None:
    rates = preaggregated_rates(data, baseline, site)
    if not rates:
        print(f"[warn] {baseline} {site} missing target_hit_rate in {IN_JSON}", file=sys.stderr)
        rates = fallback_rates(baseline, site)

    x = np.arange(len(PAPER_MODES))
    values = [rates.get(mode, 0.0) for mode in PAPER_MODES]
    bars = ax.bar(
        x,
        values,
        width=0.68,
        color=[MODE_COLORS[m] if m in rates else "#d9d9d9" for m in PAPER_MODES],
        edgecolor="white",
        linewidth=0.8,
    )
    for bar, mode, value in zip(bars, PAPER_MODES, values):
        if mode not in rates:
            bar.set_hatch("//")
            ax.text(bar.get_x() + bar.get_width() / 2, 2, "N/A\npending", ha="center", va="bottom", fontsize=7, color="#666666")
            print(f"[warn] {baseline} {site} missing target_hit_rate for {mode}", file=sys.stderr)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.1f}%", ha="center", va="bottom", fontsize=7.3)

    y_top = max(values + [1.0])
    axis1 = contrast_delta(data, baseline, site, "axis_1_text")
    axis2 = contrast_delta(data, baseline, site, "axis_2_prompt")
    if axis1 is not None and "DOM" in PAPER_MODES and "P-text" in PAPER_MODES:
        draw_delta(ax, PAPER_MODES.index("DOM"), PAPER_MODES.index("P-text"), y_top + 6.0, f"axis 1 {axis1:+.1f}pp", "#2f5f9f")
    if axis2 is not None and "P-text" in PAPER_MODES and "P-SoM" in PAPER_MODES:
        draw_delta(ax, PAPER_MODES.index("P-text"), PAPER_MODES.index("P-SoM"), y_top + 12.0, f"axis 2 {axis2:+.1f}pp", "#c56b1d")

    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.set_xticks(x, PAPER_MODES, rotation=30, ha="right")
    ax.set_ylim(0, min(100, y_top + 22.0))
    ax.set_ylabel("Target-page hit rate (%)")
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    data = load_json()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9.5, "figure.dpi": 150})
    # 3-model deep-update 2026-05-18: subplot grid = (n_baselines, 2 sites).
    # Was hardcoded (2, 2) for B0+B1. With B2 → (3, 2). figsize height scales.
    n_baselines = len(BASELINES)
    fig, axes = plt.subplots(n_baselines, 2, figsize=(13.5, max(6.0, 4.0 * n_baselines)), sharey=True)
    if n_baselines == 1:
        axes = axes.reshape(1, -1)
    for ax, (baseline, site, title) in zip(axes.ravel(), PANELS):
        draw_panel(ax, data, baseline, site, title)
    fig.suptitle("Micro 2b — target-page hit rate by representation mode", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.02, "Source: docs/analysis/cross_sites/axis1_microbehavior.json; deltas are paired right-minus-left contrasts.", ha="center", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
