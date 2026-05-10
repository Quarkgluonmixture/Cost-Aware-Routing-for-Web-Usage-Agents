#!/usr/bin/env python3
"""[Efficiency 3b] Image embedding / total-token cost per mode — paper §1
4-fold drop-in property (a) 'cost ≈ DOM' citation figure.

Reads docs/analysis/cross_sites/cost_per_mode.json and emits a grouped bar
chart showing avg API token cost per episode per mode, annotated with
P-SoM/DOM ratio (the deployment-cost-equivalence claim).

Output: results/phantom_paper/figures/fig3b_image_token_gap.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "docs/analysis/cross_sites/cost_per_mode.json"
OUT = ROOT / "results/phantom_paper/figures/fig3b_image_token_gap.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

MODES = ["DOM", "SoM", "Vision", "P-text", "P-prompt", "P-SoM"]
MODE_COLOR = {
    "DOM": "#4c78a8", "SoM": "#f58518", "Vision": "#54a24b",
    "P-text": "#e45756", "P-prompt": "#9467bd", "P-SoM": "#b279a2",
}


def emit_placeholder(reason: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, f"fig3b_image_token_gap\n\n[data pending]\n\n{reason}",
            ha="center", va="center", fontsize=11, color="gray")
    ax.set_axis_off()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig3b] placeholder written → {OUT} (reason: {reason})")


def main():
    if not SRC.exists():
        emit_placeholder(f"missing source {SRC.relative_to(ROOT)}")
        return
    data = json.loads(SRC.read_text())
    cells = data.get("cells", {})
    if not cells:
        emit_placeholder("cost_per_mode.cells empty")
        return

    # Flatten to {(baseline, site): {mode: avg_cost_usd}}
    flat = {}
    for baseline, sites in cells.items():
        if not isinstance(sites, dict):
            continue
        for site, modes in sites.items():
            if not isinstance(modes, dict):
                continue
            row = {}
            for m, v in modes.items():
                if not isinstance(v, dict):
                    continue
                cost = v.get("avg_token_cost_usd_yaml_rate") or v.get("paper_cost_usd")
                if cost is not None:
                    row[m] = float(cost)
            if row:
                flat[(baseline, site)] = row

    if not flat:
        emit_placeholder("no cost rows found in cost_per_mode")
        return

    row_keys = sorted(flat.keys())
    n_rows = len(row_keys)

    fig, axes = plt.subplots(1, n_rows, figsize=(4.2 * n_rows, 4.5), sharey=False)
    if n_rows == 1:
        axes = [axes]

    for ax, rk in zip(axes, row_keys):
        row = flat[rk]
        dom_cost = row.get("DOM")
        x = np.arange(len(MODES))
        costs = [row.get(m, np.nan) for m in MODES]
        colors = [MODE_COLOR.get(m, "gray") for m in MODES]
        bars = ax.bar(x, costs, color=colors, edgecolor="black", linewidth=0.5)

        for b, m, c in zip(bars, MODES, costs):
            if np.isnan(c):
                continue
            label = f"${c:.4f}"
            if dom_cost and not np.isnan(c):
                ratio = c / dom_cost
                label += f"\n{ratio:.2f}× DOM"
            ax.text(b.get_x() + b.get_width() / 2, c + max(costs) * 0.02,
                    label, ha="center", va="bottom", fontsize=7.5)

        if dom_cost is not None:
            ax.axhline(dom_cost, color="#4c78a8", linestyle="--", alpha=0.6, linewidth=1,
                       label="DOM baseline")
            ax.legend(loc="upper right", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(MODES, fontsize=9, rotation=20, ha="right")
        ax.set_title(f"{rk[0]} · {rk[1]}", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("avg API token cost / episode ($)", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(0, max(costs) * 1.25 if any(not np.isnan(c) for c in costs) else 1)

    fig.suptitle("Average API token cost per episode  (paper §1 3b — 4-fold drop-in (a) 'cost ≈ DOM')",
                 fontsize=11, y=1.02)
    fig.text(0.01, 0.005,
             f"Source: {SRC.relative_to(ROOT)}  |  ratio vs DOM annotated per bar",
             fontsize=6.5, color="gray")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig3b] wrote {OUT}")


if __name__ == "__main__":
    main()
