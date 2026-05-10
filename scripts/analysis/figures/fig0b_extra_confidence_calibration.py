#!/usr/bin/env python3
"""[Outcome 0b-extra] Confidence calibration AUROC heatmap — paper §1 4-fold (c)
'signal AUROC ≥ baseline' citation figure.

Reads docs/analysis/cross_sites/mechanism_per_task.json E3_confidence_calibration
block and emits an AUROC heatmap (rows = (baseline, site), cols = mode) showing
best self-confidence routing signal AUROC.

Two panels: behavioral AUROC and verbal AUROC.

Output: results/phantom_paper/figures/fig0b_extra_confidence_calibration.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "docs/analysis/cross_sites/mechanism_per_task.json"
OUT = ROOT / "results/phantom_paper/figures/fig0b_extra_confidence_calibration.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

MODES = ["DOM", "SoM", "Vision", "P-text", "P-prompt", "P-SoM"]


def emit_placeholder(reason: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, f"fig0b_extra_confidence_calibration\n\n[data pending]\n\n{reason}",
            ha="center", va="center", fontsize=11, color="gray")
    ax.set_axis_off()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig0b_extra] placeholder written → {OUT} (reason: {reason})")


def main():
    if not SRC.exists():
        emit_placeholder(f"missing source {SRC.relative_to(ROOT)}")
        return
    data = json.loads(SRC.read_text())
    e3 = data.get("E3_confidence_calibration", {}).get("cells", {})
    if not e3:
        emit_placeholder("E3_confidence_calibration.cells empty")
        return

    # Group by (model, site) → mode → AUROC
    rows = {}
    for ck, cv in e3.items():
        parts = ck.split("/")
        if len(parts) != 3:
            continue
        model, site, mode = parts
        rows.setdefault((model, site), {})[mode] = cv
    if not rows:
        emit_placeholder("no parseable cells in E3")
        return

    row_keys = sorted(rows.keys())
    n_rows, n_cols = len(row_keys), len(MODES)

    behavioral = np.full((n_rows, n_cols), np.nan)
    verbal = np.full((n_rows, n_cols), np.nan)
    for i, rk in enumerate(row_keys):
        for j, m in enumerate(MODES):
            cell = rows[rk].get(m)
            if cell is None:
                continue
            b = cell.get("AUROC_behavioral_max")
            v = cell.get("AUROC_verbal")
            if b is not None:
                behavioral[i, j] = b
            if v is not None:
                verbal[i, j] = v

    fig, axes = plt.subplots(1, 2, figsize=(13.0, max(3.5, 0.7 * n_rows + 2)), sharey=True)

    for ax, grid, title in zip(axes, [behavioral, verbal], ["Behavioral AUROC (max signal)", "Verbal AUROC"]):
        masked = np.ma.masked_invalid(grid)
        im = ax.imshow(masked, cmap="RdYlGn", aspect="auto", vmin=0.4, vmax=0.9)
        for i in range(n_rows):
            for j in range(n_cols):
                if np.isnan(grid[i, j]):
                    ax.text(j, i, "—", ha="center", va="center", color="gray", fontsize=9)
                else:
                    color = "white" if grid[i, j] > 0.75 or grid[i, j] < 0.55 else "black"
                    ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center",
                            color=color, fontsize=9)
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(MODES, fontsize=9, rotation=15)
        if ax is axes[0]:
            ax.set_yticks(range(n_rows))
            ax.set_yticklabels([f"{m} · {s}" for m, s in row_keys], fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.axhline(-0.5, color="black", linewidth=0.5)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)

    fig.suptitle("Confidence calibration AUROC per (baseline, site) × mode\n"
                 "paper §1 0b-extra — 4-fold drop-in property (c) 'signal AUROC ≥ baseline'",
                 fontsize=11, y=1.04)
    fig.text(0.01, 0.005,
             f"Source: {SRC.relative_to(ROOT)} (E3 block)  |  AUROC > 0.5 = signal informative; chance = 0.5",
             fontsize=6.5, color="gray")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig0b_extra] wrote {OUT}")


if __name__ == "__main__":
    main()
