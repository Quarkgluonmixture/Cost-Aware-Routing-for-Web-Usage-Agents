#!/usr/bin/env python3
"""[Outcome supporting] Paper §5 mechanism — Stage 2 activation patching pilot.

Generates three paper-grade consolidated figures from the 7 cls+reddit cells:

1. fig_mech_4cell_curves.png — Cells A/B/F/G overlay (cross-site × cross-direction
   bidirectional mid-layer mechanism). Primary §5 figure.
2. fig_mech_real_vs_random.png — Cell A (real patching) vs Cell E (random
   injection). Negative control proving content-specificity. Reviewer-rebuttal.
3. fig_mech_2x2_selection_bias.png — Cells A/B/C/D 2x2 grid showing direction-tier
   interaction (strong-tier bidirectional / reverse-tier direction-locked).

All three pull per-task per-layer overlap_to_target from each cell's
patching_continuation_results.json (canonical Stage 2 output schema).
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "results/phantom_paper/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CELLS = {
    "A": ROOT / "results/mechanistic/stage2b_curated_b1_cls_myriad",
    "B": ROOT / "results/mechanistic/stage2c_reverse_curated_b1_cls_myriad",
    "C": ROOT / "results/mechanistic/stage2b_2x2_fwd_revtasks_myriad",
    "D": ROOT / "results/mechanistic/stage2c_2x2_rev_strongtasks_myriad",
    "E": ROOT / "results/mechanistic/stage2b_celle_random_cls_strong_myriad",
    "F": ROOT / "results/mechanistic/stage2b_cellf_fwd_reddit_strong_myriad",
    "G": ROOT / "results/mechanistic/stage2c_cellg_rev_reddit_reverse_myriad",
}

CELL_META = {
    "A": dict(label="cls fwd · strong (N=24)",   site="cls",    direction="fwd", tier="strong",  l17_holm="0.011 ✓"),
    "B": dict(label="cls rev · reverse (N=15)",  site="cls",    direction="rev", tier="reverse", l17_holm="0.033 ✓"),
    "C": dict(label="cls fwd · reverse (N=15)",  site="cls",    direction="fwd", tier="reverse", l17_holm="0.257 ✗"),
    "D": dict(label="cls rev · strong (N=24)",   site="cls",    direction="rev", tier="strong",  l17_holm="0.010 ✓"),
    "E": dict(label="cls fwd · strong · random (N=24)", site="cls", direction="fwd", tier="random", l17_holm="n/a"),
    "F": dict(label="reddit fwd · strong (N=24)", site="reddit", direction="fwd", tier="strong",  l17_holm="0.004 ✓✓"),
    "G": dict(label="reddit rev · reverse (N=15)", site="reddit", direction="rev", tier="reverse", l17_holm="0.036 ✓"),
}


def load_cell_curves(cell_id: str) -> dict:
    """Load per-task per-layer overlap_to_target for a cell.

    Returns dict with arrays keyed by layer index 0..35:
      mean_overlap, std_overlap, mean_ld, std_ld, n_tasks
    """
    path = CELLS[cell_id] / "patching_continuation_results.json"
    data = json.load(open(path))
    per_task = data["per_task"]
    n_tasks = len(per_task)
    per_layer_overlap = {l: [] for l in range(36)}
    per_layer_ld = {l: [] for l in range(36)}
    for t in per_task:
        for lr in t["per_layer"]:
            L = lr["layer"]
            per_layer_overlap[L].append(lr.get("token_overlap_to_target"))
            per_layer_ld[L].append(lr.get("ld_to_target"))
    layers = sorted(per_layer_overlap.keys())
    mean_overlap = np.array([statistics.mean(per_layer_overlap[L]) for L in layers])
    std_overlap = np.array([statistics.pstdev(per_layer_overlap[L]) for L in layers])
    mean_ld = np.array([statistics.mean(per_layer_ld[L]) for L in layers])
    std_ld = np.array([statistics.pstdev(per_layer_ld[L]) for L in layers])
    return dict(
        layers=np.array(layers),
        mean_overlap=mean_overlap, std_overlap=std_overlap,
        mean_ld=mean_ld, std_ld=std_ld,
        n_tasks=n_tasks,
    )


def fig_4cell_curves():
    """Cells A/B/F/G overlay — primary mechanism evidence."""
    cells_used = ["A", "B", "F", "G"]
    colors = {"A": "#1f77b4", "B": "#ff7f0e", "F": "#2ca02c", "G": "#d62728"}
    linestyles = {"A": "-", "B": "--", "F": "-", "G": "--"}

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.0))
    for cid in cells_used:
        d = load_cell_curves(cid)
        meta = CELL_META[cid]
        ax.plot(d["layers"], d["mean_overlap"],
                label=f"Cell {cid}: {meta['label']}  L17 p_Holm={meta['l17_holm']}",
                color=colors[cid], linestyle=linestyles[cid], linewidth=2.0,
                marker="o" if cid in ("A", "F") else "s", markersize=4)
        ax.fill_between(d["layers"],
                        d["mean_overlap"] - d["std_overlap"],
                        d["mean_overlap"] + d["std_overlap"],
                        color=colors[cid], alpha=0.10)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.6, label="L35 baseline (no patch)")
    ax.axvspan(11, 17, alpha=0.08, color="purple", label="L11–L17 mid-layer mechanism")
    ax.set_xlabel("Layer index (Qwen3-VL-4B has 36 transformer layers)")
    ax.set_ylabel(r"overlap$_{\rm token}$(patched, target generation)")
    ax.set_title("4-Cell mid-layer mechanism replication\n(cls × reddit × forward × reverse all show Holm-sig L17)",
                 fontsize=11)
    ax.set_ylim(0.5, 1.05)
    ax.set_xlim(-0.5, 35.5)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.text(0.01, 0.005,
             "Source: results/mechanistic/stage2{b,c}_*_myriad/patching_continuation_results.json  |  "
             "Holm: stage2_layer_significance.py  |  shaded = ±1σ across tasks",
             fontsize=6.5, color="gray")
    fig.tight_layout()
    out = OUT_DIR / "fig_mech_4cell_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[fig_mech_4cell_curves] wrote {out}")
    plt.close(fig)


def fig_real_vs_random():
    """Cell A (real patching) vs Cell E (random injection) — content-specificity."""
    a = load_cell_curves("A")
    e = load_cell_curves("E")

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.0))
    ax.plot(a["layers"], a["mean_overlap"],
            label=f"Cell A: real patching (cls fwd · strong N={a['n_tasks']})",
            color="#1f77b4", linewidth=2.2, marker="o", markersize=4)
    ax.fill_between(a["layers"],
                    a["mean_overlap"] - a["std_overlap"],
                    a["mean_overlap"] + a["std_overlap"],
                    color="#1f77b4", alpha=0.15)

    ax.plot(e["layers"], e["mean_overlap"],
            label=f"Cell E: random injection (same 24 tasks, random source)",
            color="#d62728", linewidth=2.2, linestyle="--", marker="x", markersize=5)
    ax.fill_between(e["layers"],
                    e["mean_overlap"] - e["std_overlap"],
                    e["mean_overlap"] + e["std_overlap"],
                    color="#d62728", alpha=0.15)

    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.6)
    ax.axvspan(11, 17, alpha=0.08, color="purple", label="L11–L17 mid-layer region")

    # Annotate baselines
    ax.annotate(f"Real L35 baseline = {a['mean_overlap'][-1]:.3f}\n(target reproduces itself)",
                xy=(35, a["mean_overlap"][-1]), xytext=(25, 0.85),
                fontsize=8.5, color="#1f77b4",
                arrowprops=dict(arrowstyle="->", color="#1f77b4", alpha=0.6))
    ax.annotate(f"Random L35 = {e['mean_overlap'][-1]:.3f}\n(random injection breaks even L35)",
                xy=(35, e["mean_overlap"][-1]), xytext=(20, 0.30),
                fontsize=8.5, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728", alpha=0.6))

    ax.set_xlabel("Layer index")
    ax.set_ylabel(r"overlap$_{\rm token}$(patched, target generation)")
    ax.set_title("Content-specificity negative control: real vs random injection\n"
                 "(Real shows selective L17 disruption; random is uniformly destructive across all layers)",
                 fontsize=11)
    ax.set_ylim(-0.05, 1.10)
    ax.set_xlim(-0.5, 35.5)
    ax.legend(fontsize=9, loc="center right")
    ax.grid(True, alpha=0.3)
    fig.text(0.01, 0.005,
             "Source: stage2b_curated_b1_cls_myriad (Cell A) + stage2b_celle_random_cls_strong_myriad (Cell E)  |  "
             "shaded = ±1σ",
             fontsize=6.5, color="gray")
    fig.tight_layout()
    out = OUT_DIR / "fig_mech_real_vs_random.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[fig_mech_real_vs_random] wrote {out}")
    plt.close(fig)


def fig_2x2_selection_bias():
    """2x2 panel grid Cells A/B/C/D — direction × tier interaction."""
    cells_used = ["A", "C", "D", "B"]
    layout = {
        "A": (0, 0, "Strong-tier (composite ≥ 1.0)"),    # row=fwd, col=strong
        "C": (0, 1, "Reverse-tier (composite ≤ -1.5)"),  # row=fwd, col=reverse
        "D": (1, 0, "Strong-tier (composite ≥ 1.0)"),    # row=rev, col=strong
        "B": (1, 1, "Reverse-tier (composite ≤ -1.5)"),  # row=rev, col=reverse
    }
    row_labels = ["Forward direction\n(SoM → P-SoM patch)", "Reverse direction\n(P-SoM → SoM patch)"]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), sharey=True, sharex=True)
    for cid in cells_used:
        r, c, col_title = layout[cid]
        ax = axes[r, c]
        d = load_cell_curves(cid)
        meta = CELL_META[cid]
        is_holm_sig = "✓" in meta["l17_holm"]
        line_color = "#2ca02c" if is_holm_sig else "#d62728"
        ax.plot(d["layers"], d["mean_overlap"],
                color=line_color, linewidth=2.2, marker="o", markersize=3.5)
        ax.fill_between(d["layers"],
                        d["mean_overlap"] - d["std_overlap"],
                        d["mean_overlap"] + d["std_overlap"],
                        color=line_color, alpha=0.15)
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
        ax.axvspan(11, 17, alpha=0.08, color="purple")
        ax.scatter([17], [d["mean_overlap"][17]], s=130, marker="*",
                   color="black" if is_holm_sig else "gray", zorder=5,
                   label=f"L17 Δ={d['mean_overlap'][17]-d['mean_overlap'][35]:+.3f}, p_Holm={meta['l17_holm']}")
        if r == 0:
            ax.set_title(col_title, fontsize=10)
        if c == 0:
            ax.set_ylabel(row_labels[r] + "\n\noverlap to target", fontsize=9)
        if r == 1:
            ax.set_xlabel("Layer", fontsize=9)
        ax.text(0.03, 0.95, f"Cell {cid}\n{meta['label']}",
                transform=ax.transAxes, fontsize=8.5, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="gray"))
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.25)
        ax.set_ylim(0.55, 1.05)

    fig.suptitle("2x2 selection-bias control (cls)\n"
                 "Strong-tier tasks support BOTH directions; reverse-tier tasks are direction-locked",
                 fontsize=12, y=1.00)
    fig.text(0.01, 0.005,
             "Cells (left to right, top to bottom): A=fwd×strong / C=fwd×reverse / D=rev×strong / B=rev×reverse  |  "
             "green = L17 Holm-sig, red = NULL",
             fontsize=6.5, color="gray")
    fig.tight_layout()
    out = OUT_DIR / "fig_mech_2x2_selection_bias.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[fig_mech_2x2_selection_bias] wrote {out}")
    plt.close(fig)


def main():
    print("Loading cells:", list(CELLS.keys()))
    fig_4cell_curves()
    fig_real_vs_random()
    fig_2x2_selection_bias()
    print(f"All 3 figures written under {OUT_DIR}")


if __name__ == "__main__":
    main()
