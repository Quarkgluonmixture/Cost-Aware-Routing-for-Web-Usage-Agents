#!/usr/bin/env python3
"""[Layer 0g viz] Outcome — routing signal AUROC heatmap.

Output:
- results/phantom_paper/figures/fig0g_routing_auroc_heatmap.png

Visual companion to Layer 0g routing AUROC evidence.

See docs/checkpoints/paper_planning.md §3 Layer 0 framework.

fig11: Routing signal AUROC heatmap — Section 6 paper main evidence.

Reads results/phantom_paper/auroc_cross_condition.csv (per-cell × per-signal
AUROC + 95% CI) and visualizes as cross-condition × signal heatmap.

Rows: (baseline, site, mode) cells (~20 across full matrix).
Cols: signal types (verbalized / token-level / behavioral).
Color: AUROC value (0.5 random — 1.0 perfect; threshold 0.7 routing-usable).

Output: results/phantom_paper/figures/fig0g_routing_auroc_heatmap.png
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / "results/phantom_paper/auroc_cross_condition.csv"
OUT = ROOT / "results/phantom_paper/figures/fig0g_routing_auroc_heatmap.png"

# Featured signals to display (one of each type)
FEATURED_SIGNALS = [
    ("ep_mean_verbalized", "verbalized\nmean"),
    ("ep_min_verbalized",  "verbalized\nmin"),
    ("ep_mean_logprob",    "logprob\nmean"),
    ("ep_min_logprob",     "logprob\nmin"),
    ("max_repeat_streak",  "behavioral\nmax_repeat"),
    ("action_diversity",   "behavioral\naction_div"),
    ("url_revisit_count",  "behavioral\nurl_revisit"),
]


def main() -> None:
    if not CSV.exists():
        print(f"[warn] {CSV} not found — run `make routing-auroc` first")
        return
    df = pd.read_csv(CSV)
    if df.empty:
        print("[warn] empty AUROC table")
        return

    # Build cell label
    df["cell"] = df["baseline"] + " " + df["site"] + " " + df["mode"]
    cells = df["cell"].drop_duplicates().tolist()

    # Pivot: rows = cells, cols = featured signals
    sig_cols = [s for s, _ in FEATURED_SIGNALS]
    sig_labels = [lbl for _, lbl in FEATURED_SIGNALS]

    matrix = np.full((len(cells), len(sig_cols)), np.nan)
    for i, cell in enumerate(cells):
        cell_df = df[df["cell"] == cell]
        for j, sig in enumerate(sig_cols):
            row = cell_df[cell_df["signal"] == sig]
            if not row.empty and pd.notna(row.iloc[0]["AUROC"]):
                matrix[i, j] = float(row.iloc[0]["AUROC"])

    fig, ax = plt.subplots(figsize=(9.5, max(4.5, 0.42 * len(cells) + 1.2)))
    plt.rcParams.update({"font.size": 9})

    # Custom colormap: emphasize 0.5-1.0 range; AUROC < 0.5 = inverse signal (rare)
    cmap = plt.cm.RdYlGn
    vmin, vmax = 0.4, 0.9

    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(sig_cols)))
    ax.set_xticklabels(sig_labels, fontsize=8.5, ha="center")
    ax.set_yticks(np.arange(len(cells)))
    ax.set_yticklabels(cells, fontsize=9)

    # Cell text annotations
    for i in range(len(cells)):
        for j in range(len(sig_cols)):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="#999")
            else:
                color = "#000" if v < 0.78 else "#fff"
                marker = "★" if v >= 0.7 else ""
                ax.text(j, i, f"{v:.2f}{marker}", ha="center", va="center",
                        fontsize=8.5, color=color, fontweight="bold" if v >= 0.7 else "normal")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("AUROC", fontsize=10)
    cbar.ax.axhline(0.7, color="#000", linewidth=1.0)
    cbar.ax.text(2.5, 0.7, " usable\n threshold", fontsize=7.5, color="#000")

    ax.set_title("Routing signal AUROC heatmap (★ = ≥ 0.7 routing-usable)",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Signal type", fontsize=10)
    fig.text(0.5, 0.012,
             "Rows: (baseline × site × mode) cells. Cols: 4 logprob/verbalized + 3 behavioral signals. "
             "Higher AUROC → signal better predicts task success → cheaper trigger feature for routing.",
             ha="center", fontsize=8.0, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
