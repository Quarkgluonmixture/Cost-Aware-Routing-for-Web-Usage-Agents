#!/usr/bin/env python3
"""[Outcome 4-bucket failure-mode] Per-cell stacked-bar failure-mode
distribution — paper §5 deployment-class failure characterization figure.

Reads docs/analysis/cross_sites/failure_modes_per_cell.json (output of
scripts/analysis/aggregate_failure_modes.py) and emits a horizontal
stacked-bar chart, one bar per cell (baseline × site × mode), bucket
colors fixed across cells.

Output: results/phantom_paper/figures/fig_failure_modes_per_cell.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "docs/analysis/cross_sites/failure_modes_per_cell.json"
OUT = ROOT / "results/phantom_paper/figures/fig_failure_modes_per_cell.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Bucket order + colors (consistent across figures)
BUCKETS = [
    "early-finish/wrong-commit",
    "search-loop",
    "visual-hijack/click-loop",
    "element-misground",
    "missing-context",
    "max-steps-other",
    "error/noise",
    "other-failure",
]
BUCKET_COLOR = {
    "early-finish/wrong-commit": "#d62728",   # red — premature commit
    "search-loop":               "#ff7f0e",   # orange — DOM-style search loop
    "visual-hijack/click-loop":  "#9467bd",   # purple — SoM-style click loop
    "element-misground":         "#8c564b",   # brown — target unreachable
    "missing-context":           "#7f7f7f",   # gray — no progress / stuck
    "max-steps-other":           "#bcbd22",   # olive — generic timeout
    "error/noise":               "#17becf",   # teal — env/parse/noise
    "other-failure":             "#e377c2",   # pink — unmapped catch-all
}

MODE_ORDER = ["DOM", "SoM", "Vision", "P-text", "P-prompt", "P-SoM"]


def emit_placeholder(reason: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, f"fig_failure_modes_per_cell\n\n[data pending]\n\n{reason}",
            ha="center", va="center", fontsize=11, color="gray")
    ax.set_axis_off()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_failure_modes] placeholder written → {OUT} (reason: {reason})")


def main():
    if not SRC.exists():
        emit_placeholder(f"missing source {SRC.relative_to(ROOT)}")
        return
    data = json.loads(SRC.read_text())
    cells = data.get("cells", {})
    if not cells:
        emit_placeholder("failure_modes_per_cell.cells empty")
        return

    # Sort: baseline → site → mode (canonical order)
    # /stress A1.20 P1-5-A (2026-05-17, Claude): add B2:2 (Gemma3-VL 2026-05-14)
    # canonical baseline order. Pre-fix `.get(b, 9)` sorted B2 as 9 catch-all →
    # B2 panels scattered to end instead of being adjacent to B0/B1.
    def cell_sort_key(name: str):
        parts = name.split("/")
        if len(parts) != 3:
            return (9, 9, 9)
        b, s, m = parts
        baseline_order = {"B0": 0, "B1": 1, "B2": 2}.get(b, 9)
        site_order = {"classifieds": 0, "reddit": 1, "shopping": 2}.get(s, 9)
        mode_order = MODE_ORDER.index(m) if m in MODE_ORDER else 9
        return (baseline_order, site_order, mode_order)

    sorted_keys = sorted(cells.keys(), key=cell_sort_key)
    n = len(sorted_keys)

    # Build matrix (cell × bucket) % of failed
    matrix = np.zeros((n, len(BUCKETS)))
    n_failed = []
    n_total = []
    for i, ck in enumerate(sorted_keys):
        cell = cells[ck]
        n_failed.append(cell.get("failed_count", 0))
        n_total.append(cell.get("total_episodes", 0))
        for j, b in enumerate(BUCKETS):
            v = cell.get("buckets", {}).get(b, {})
            matrix[i, j] = v.get("pct_of_failed", 0)

    fig, ax = plt.subplots(figsize=(13.5, max(4, 0.42 * n + 1.5)))
    left = np.zeros(n)
    y_pos = np.arange(n)
    for j, bucket in enumerate(BUCKETS):
        widths = matrix[:, j]
        if widths.sum() == 0:
            continue
        ax.barh(y_pos, widths, left=left, color=BUCKET_COLOR[bucket],
                edgecolor="white", linewidth=0.4, label=bucket)
        for i, w in enumerate(widths):
            if w >= 8:
                ax.text(left[i] + w / 2, i, f"{w:.0f}%",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if bucket in ("missing-context",
                                                     "early-finish/wrong-commit",
                                                     "visual-hijack/click-loop") else "black")
        left += widths

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{ck}  (N={n_total[i]}, fail={n_failed[i]})"
                       for i, ck in enumerate(sorted_keys)], fontsize=8.5)
    ax.set_xlabel("% of failed episodes (per cell)", fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_title("Failure-mode bucket distribution per cell  (paper §5 — deployment-class characterization)",
                 fontsize=11.5)
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9, ncol=2)
    ax.grid(True, axis="x", alpha=0.3)
    fig.text(0.01, 0.005,
             f"Source: {SRC.relative_to(ROOT)}  |  7-bucket paper-grade taxonomy "
             f"(5 core + 2 catch-alls) + 1 dynamic, see aggregate_failure_modes.py docstring (A1.19 B-432)",
             fontsize=6.5, color="gray")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_failure_modes] wrote {OUT}")


if __name__ == "__main__":
    main()
