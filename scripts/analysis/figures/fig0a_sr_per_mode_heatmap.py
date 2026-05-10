#!/usr/bin/env python3
"""[Outcome 0a] SR per mode heatmap — paper §1 main hook citation figure.

Reads docs/analysis/cross_sites/sr_fp_per_mode.json (produced by
scripts/analysis/aggregate_sr_fp_per_mode.py) and emits a 2D heatmap with:
  - rows: (baseline, site) tuples (B0 cls / B0 reddit / B1 cls / B1 reddit / etc.)
  - cols: observation modes (DOM / SoM / Vision / P-text / P-prompt / P-SoM)
  - cell value: adjusted SR%

Cell annotations show: raw SR / adjusted SR / N tasks / FP count.

Empty-data fallback (P79_ALLOW_EMPTY=1 or no rows): emits a placeholder PNG
with "data pending" text so make analysis FAST=1 doesn't fail.

Output: results/phantom_paper/figures/fig0a_sr_per_mode_heatmap.png
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "docs/analysis/cross_sites/sr_fp_per_mode.json"
OUT = ROOT / "results/phantom_paper/figures/fig0a_sr_per_mode_heatmap.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

MODES = ["DOM", "SoM", "Vision", "P-text", "P-prompt", "P-SoM"]


def emit_placeholder(reason: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, f"fig0a_sr_per_mode_heatmap\n\n[data pending]\n\n{reason}",
            ha="center", va="center", fontsize=11, color="gray")
    ax.set_axis_off()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig0a] placeholder written → {OUT} (reason: {reason})")


def main():
    if not SRC.exists():
        emit_placeholder(f"missing source {SRC.relative_to(ROOT)}")
        return
    data = json.loads(SRC.read_text())
    rows = data.get("summary_table", [])
    if not rows:
        emit_placeholder("summary_table empty — run `make aggregate-sr-fp` after baseline runs")
        return

    # Build cell grid: row_keys=(baseline, site), col=mode → adj_sr_pct
    seen_rows = {}
    for r in rows:
        baseline = r.get("baseline", "?")
        site = r.get("site", "?")
        mode = r.get("mode", "?")
        adj = r.get("adjusted_sr_pct", r.get("adj_sr_pct"))
        if adj is None:
            continue
        key = (baseline, site)
        seen_rows.setdefault(key, {})[mode] = r

    if not seen_rows:
        emit_placeholder("no valid (baseline, site, mode) tuples in summary_table")
        return

    row_keys = sorted(seen_rows.keys())
    grid = np.full((len(row_keys), len(MODES)), np.nan)
    annot = [["" for _ in MODES] for _ in row_keys]
    for i, rk in enumerate(row_keys):
        for j, m in enumerate(MODES):
            r = seen_rows[rk].get(m)
            if r is None:
                continue
            adj = r.get("adjusted_sr_pct", r.get("adj_sr_pct"))
            grid[i, j] = adj
            raw = r.get("raw_sr_pct", r.get("raw_sr_pct"))
            n = r.get("n", r.get("n_tasks"))
            fp = r.get("fp_count", 0)
            annot[i][j] = (
                f"{adj:.1f}%\n"
                f"({raw:.1f}% raw)\n"
                f"N={n}, fp={fp}"
            )

    fig, ax = plt.subplots(figsize=(11.0, max(3.5, 0.8 * len(row_keys) + 2)))
    cmap = plt.get_cmap("YlGnBu")
    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0,
                   vmax=max(25, np.nanmax(grid) if np.isfinite(grid).any() else 25))
    for i in range(len(row_keys)):
        for j in range(len(MODES)):
            if np.isnan(grid[i, j]):
                ax.text(j, i, "—", ha="center", va="center", color="gray", fontsize=9)
            else:
                color = "white" if grid[i, j] > 12 else "black"
                ax.text(j, i, annot[i][j], ha="center", va="center", color=color, fontsize=7.5)

    ax.set_xticks(range(len(MODES)))
    ax.set_xticklabels(MODES, fontsize=10)
    ax.set_yticks(range(len(row_keys)))
    ax.set_yticklabels([f"{b} · {s}" for b, s in row_keys], fontsize=10)
    ax.set_title("Adjusted success rate (%) per (baseline, site) × mode\npaper §1 0a — main hook",
                 fontsize=12)
    fig.colorbar(im, ax=ax, label="Adjusted SR (%)", fraction=0.025, pad=0.02)
    fig.text(0.01, 0.005,
             f"Source: {SRC.relative_to(ROOT)}  |  raw SR / N / FP count shown per cell",
             fontsize=6.5, color="gray")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig0a] wrote {OUT}")


if __name__ == "__main__":
    main()
