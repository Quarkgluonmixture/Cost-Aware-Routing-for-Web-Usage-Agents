#!/usr/bin/env python3
"""[Outcome 0b] FP rate per mode — paper §1 hygiene figure.

Reads docs/analysis/cross_sites/sr_fp_per_mode.json and emits a grouped bar
chart of false-positive rate (raw SR − adjusted SR) per mode, split by
FP breakdown components (na_fp / eval_fp / visual_fp) when available.

Output: results/phantom_paper/figures/fig0b_fp_rate_per_mode.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "docs/analysis/cross_sites/sr_fp_per_mode.json"
OUT = ROOT / "results/phantom_paper/figures/fig0b_fp_rate_per_mode.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

MODES = ["DOM", "SoM", "Vision", "P-text", "P-prompt", "P-SoM"]
MODE_COLOR = {
    "DOM": "#4c78a8", "SoM": "#f58518", "Vision": "#54a24b",
    "P-text": "#e45756", "P-prompt": "#9467bd", "P-SoM": "#b279a2",
}


def emit_placeholder(reason: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, f"fig0b_fp_rate_per_mode\n\n[data pending]\n\n{reason}",
            ha="center", va="center", fontsize=11, color="gray")
    ax.set_axis_off()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig0b] placeholder written → {OUT} (reason: {reason})")


def main():
    if not SRC.exists():
        emit_placeholder(f"missing source {SRC.relative_to(ROOT)}")
        return
    data = json.loads(SRC.read_text())
    rows = data.get("summary_table", [])
    if not rows:
        emit_placeholder("summary_table empty — run aggregators first")
        return

    # Group by (baseline, site) → mode → fp_pct
    cells = {}
    for r in rows:
        bs = (r.get("baseline", "?"), r.get("site", "?"))
        mode = r.get("mode", "?")
        cells.setdefault(bs, {})[mode] = r

    cell_keys = sorted(cells.keys())
    n_cells = len(cell_keys)
    n_modes = len(MODES)

    fig, axes = plt.subplots(1, n_cells, figsize=(4.2 * n_cells, 4.5), sharey=True)
    if n_cells == 1:
        axes = [axes]

    for ax, ck in zip(axes, cell_keys):
        cell_data = cells[ck]
        x = np.arange(n_modes)
        fp_pct = [cell_data.get(m, {}).get("fp_rate_pct", np.nan) for m in MODES]
        colors = [MODE_COLOR.get(m, "gray") for m in MODES]
        bars = ax.bar(x, fp_pct, color=colors, edgecolor="black", linewidth=0.5)
        for b, v in zip(bars, fp_pct):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.1f}%",
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(MODES, fontsize=9, rotation=20, ha="right")
        ax.set_title(f"{ck[0]} · {ck[1]}", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("FP rate (% of N tasks)", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(0, max(5, (max([v for v in fp_pct if not np.isnan(v)], default=0)) * 1.3))

    fig.suptitle("False-positive rate per mode  (paper §1 0b — adjusted-SR hygiene)",
                 fontsize=12, y=1.02)
    fig.text(0.01, 0.005,
             f"Source: {SRC.relative_to(ROOT)}  |  FP rate = raw SR − adjusted SR (% of N)",
             fontsize=6.5, color="gray")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig0b] wrote {OUT}")


if __name__ == "__main__":
    main()
