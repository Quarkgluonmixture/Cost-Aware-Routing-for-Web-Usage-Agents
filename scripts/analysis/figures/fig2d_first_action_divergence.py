#!/usr/bin/env python3
"""[Micro 2d] Micro dimension — first-action divergence by axis pair.

Output:
- results/phantom_paper/figures/fig2d_first_action_divergence.png

Micro 2d: percent of paired tasks whose first action_type differs between
axis-control mode pairs.

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
    from scripts.analysis.lib.run_registry import PAPER_MODES, get_cells
except ModuleNotFoundError:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from scripts.analysis.lib.run_registry import PAPER_MODES, get_cells

ROOT = Path(__file__).resolve().parents[3]
IN_JSON = ROOT / "docs/analysis/cross_sites/axis1_microbehavior.json"
OUT = ROOT / "results/phantom_paper/figures/fig2d_first_action_divergence.png"

PAIR_SPECS = [
    ("axis_1_text", "DOM↔P-text", "axis 1", "#4c78a8"),
    ("axis_2_prompt_alt", "DOM↔P-prompt", "axis 2", "#f58518"),
    ("axis_2_prompt", "P-text↔P-SoM", "axis 2", "#f58518"),
    ("axis_1_text_alt", "P-prompt↔P-SoM", "axis 1", "#4c78a8"),
    ("compound_dom_to_psom", "DOM↔P-SoM", "compound", "#9467bd"),
]
PANELS = [
    ("B0", "classifieds", "B0 cls"),
    ("B0", "reddit", "B0 red"),
    ("B1", "classifieds", "B1 cls"),
    ("B1", "reddit", "B1 red"),
]


def load_json() -> dict[str, Any]:
    return json.loads(IN_JSON.read_text())


def divergence_value(data: dict[str, Any], baseline: str, site: str, contrast: str) -> tuple[float | None, int]:
    row = data.get("axis_contrasts", {}).get(baseline, {}).get(site, {}).get(contrast, {}) or {}
    if row.get("skipped") or row.get("first_action_divergence_rate") is None:
        return None, int(row.get("n") or 0)
    return 100.0 * float(row["first_action_divergence_rate"]), int(row.get("n") or 0)


def draw_panel(ax: plt.Axes, data: dict[str, Any], baseline: str, site: str, title: str) -> None:
    x = np.arange(len(PAIR_SPECS))
    values: list[float] = []
    ns: list[int] = []
    colors: list[str] = []
    missing: list[bool] = []
    for contrast, _label, _axis, color in PAIR_SPECS:
        value, n = divergence_value(data, baseline, site, contrast)
        values.append(0.0 if value is None else value)
        ns.append(n)
        colors.append("#d9d9d9" if value is None else color)
        missing.append(value is None)
        if value is None:
            print(f"[warn] {baseline} {site} missing first_action_divergence_rate for {contrast}", file=sys.stderr)
    bars = ax.bar(x, values, width=0.68, color=colors, edgecolor="white", linewidth=0.8)
    for bar, value, n, is_missing in zip(bars, values, ns, missing):
        if is_missing:
            bar.set_hatch("//")
            ax.text(bar.get_x() + bar.get_width() / 2, 2, "N/A\npending", ha="center", va="bottom", fontsize=7, color="#666666")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}%\nN={n}", ha="center", va="bottom", fontsize=7.2)
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.set_xticks(x, [label for _key, label, _axis, _color in PAIR_SPECS], rotation=25, ha="right")
    ax.set_ylim(0, max(values + [1]) + 13)
    ax.set_ylabel("First action_type divergence (%)")
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    data = load_json()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9.5, "figure.dpi": 150})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), sharey=True)
    for ax, (baseline, site, title) in zip(axes.ravel(), PANELS):
        draw_panel(ax, data, baseline, site, title)
    fig.suptitle("Micro 2d — first-action divergence across axis-control pairs", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.02, "Source: docs/analysis/cross_sites/axis1_microbehavior.json; colors encode axis family.", ha="center", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
