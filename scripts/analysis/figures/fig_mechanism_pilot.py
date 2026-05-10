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
    "A":  ROOT / "results/mechanistic/stage2b_curated_b1_cls_myriad",
    "B":  ROOT / "results/mechanistic/stage2c_reverse_curated_b1_cls_myriad",
    "C":  ROOT / "results/mechanistic/stage2b_2x2_fwd_revtasks_myriad",
    "D":  ROOT / "results/mechanistic/stage2c_2x2_rev_strongtasks_myriad",
    "E":  ROOT / "results/mechanistic/stage2b_celle_random_cls_strong_myriad",
    "F":  ROOT / "results/mechanistic/stage2b_cellf_fwd_reddit_strong_myriad",
    "G":  ROOT / "results/mechanistic/stage2c_cellg_rev_reddit_reverse_myriad",
    "Cr": ROOT / "results/mechanistic/stage2b_cellcr_reddit_fwd_revtier_myriad",
    "Dr": ROOT / "results/mechanistic/stage2c_celldr_reddit_rev_strongtier_myriad",
    "Er": ROOT / "results/mechanistic/stage2b_celler_reddit_fwd_random_myriad",
}

CELL_META = {
    "A":  dict(label="cls fwd · strong (N=24)",   site="cls",    direction="fwd", tier="strong",  l17_holm="0.011 ✓"),
    "B":  dict(label="cls rev · reverse (N=15)",  site="cls",    direction="rev", tier="reverse", l17_holm="0.033 ✓"),
    "C":  dict(label="cls fwd · reverse (N=15)",  site="cls",    direction="fwd", tier="reverse", l17_holm="0.257 ✗"),
    "D":  dict(label="cls rev · strong (N=24)",   site="cls",    direction="rev", tier="strong",  l17_holm="0.010 ✓"),
    "E":  dict(label="cls fwd · strong · random (N=24)", site="cls", direction="fwd", tier="random", l17_holm="n/a"),
    "F":  dict(label="reddit fwd · strong (N=24)", site="reddit", direction="fwd", tier="strong",  l17_holm="0.004 ✓✓"),
    "G":  dict(label="reddit rev · reverse (N=15)", site="reddit", direction="rev", tier="reverse", l17_holm="0.036 ✓"),
    "Cr": dict(label="reddit fwd · reverse (N=15)", site="reddit", direction="fwd", tier="reverse", l17_holm="0.012 ✓"),
    "Dr": dict(label="reddit rev · strong (N=24)",  site="reddit", direction="rev", tier="strong",  l17_holm="0.041 ✓"),
    "Er": dict(label="reddit fwd · strong · random (N=24)", site="reddit", direction="fwd", tier="random", l17_holm="n/a"),
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
    """Real vs random injection on both cls and reddit (2-panel) — content-specificity closed on both sites."""
    panels = [
        ("A",  "E",  "cls",    "Cell A: real (cls fwd · strong N=24)",       "Cell E: random injection (same 24 tasks)"),
        ("F",  "Er", "reddit", "Cell F: real (reddit fwd · strong N=24)",    "Cell E-r: random injection (same 24 tasks)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.0), sharey=True)
    for ax, (real_id, rnd_id, site, real_label, rnd_label) in zip(axes, panels):
        real = load_cell_curves(real_id)
        rnd = load_cell_curves(rnd_id)
        ax.plot(real["layers"], real["mean_overlap"], label=real_label,
                color="#1f77b4", linewidth=2.2, marker="o", markersize=4)
        ax.fill_between(real["layers"],
                        real["mean_overlap"] - real["std_overlap"],
                        real["mean_overlap"] + real["std_overlap"],
                        color="#1f77b4", alpha=0.15)
        ax.plot(rnd["layers"], rnd["mean_overlap"], label=rnd_label,
                color="#d62728", linewidth=2.2, linestyle="--", marker="x", markersize=5)
        ax.fill_between(rnd["layers"],
                        rnd["mean_overlap"] - rnd["std_overlap"],
                        rnd["mean_overlap"] + rnd["std_overlap"],
                        color="#d62728", alpha=0.15)
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.6)
        ax.axvspan(11, 17, alpha=0.08, color="purple",
                   label="L11–L17 mid-layer" if site == "cls" else None)
        ax.annotate(f"Real L35 = {real['mean_overlap'][-1]:.3f}",
                    xy=(35, real["mean_overlap"][-1]), xytext=(22, 0.88),
                    fontsize=8, color="#1f77b4",
                    arrowprops=dict(arrowstyle="->", color="#1f77b4", alpha=0.5))
        ax.annotate(f"Random L35 = {rnd['mean_overlap'][-1]:.3f}",
                    xy=(35, rnd["mean_overlap"][-1]), xytext=(18, 0.30),
                    fontsize=8, color="#d62728",
                    arrowprops=dict(arrowstyle="->", color="#d62728", alpha=0.5))
        ax.set_title(f"{site.upper()}", fontsize=11)
        ax.set_xlabel("Layer index")
        if site == "cls":
            ax.set_ylabel(r"overlap$_{\rm token}$(patched, target generation)")
        ax.set_ylim(-0.05, 1.10)
        ax.set_xlim(-0.5, 35.5)
        ax.legend(fontsize=8.5, loc="center right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Content-specificity negative control on both sites\n"
                 "(Real patching shows selective L17 disruption; random injection is uniformly destructive — content-specificity confirmed)",
                 fontsize=12, y=1.02)
    fig.text(0.01, 0.005,
             "Sources: stage2b_curated_b1_cls + celle_random_cls (cls) / cellf_fwd_reddit + celler_reddit_fwd_random (reddit)  |  shaded = ±1σ",
             fontsize=6.5, color="gray")
    fig.tight_layout()
    out = OUT_DIR / "fig_mech_real_vs_random.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[fig_mech_real_vs_random] wrote {out}")
    plt.close(fig)


def fig_2x2_selection_bias():
    """2x2 panel grid stacked for cls (top) + reddit (bottom) — direction × tier interaction across sites."""
    # cls 2x2: A=(fwd,strong) C=(fwd,rev) D=(rev,strong) B=(rev,rev)
    # reddit 2x2: F=(fwd,strong) Cr=(fwd,rev) Dr=(rev,strong) G=(rev,rev)
    layout_cls    = {"A": (0, 0), "C": (0, 1), "D": (1, 0), "B": (1, 1)}
    layout_reddit = {"F": (0, 0), "Cr": (0, 1), "Dr": (1, 0), "G": (1, 1)}
    col_titles = ["Strong-tier (composite ≥ 1.0)", "Reverse-tier (composite ≤ -1.5)"]
    row_labels = ["Forward\n(SoM → P-SoM)", "Reverse\n(P-SoM → SoM)"]

    fig, all_axes = plt.subplots(2, 4, figsize=(16.0, 7.5), sharey=True, sharex=True,
                                  gridspec_kw={"wspace": 0.10, "hspace": 0.20})

    def plot_panel(ax, cid, show_row_label=False, show_col_title=False, col_title=""):
        d = load_cell_curves(cid)
        meta = CELL_META[cid]
        is_holm_sig = "✓" in meta["l17_holm"]
        line_color = "#2ca02c" if is_holm_sig else "#d62728"
        ax.plot(d["layers"], d["mean_overlap"],
                color=line_color, linewidth=2.0, marker="o", markersize=3)
        ax.fill_between(d["layers"],
                        d["mean_overlap"] - d["std_overlap"],
                        d["mean_overlap"] + d["std_overlap"],
                        color=line_color, alpha=0.15)
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
        ax.axvspan(11, 17, alpha=0.08, color="purple")
        ax.scatter([17], [d["mean_overlap"][17]], s=110, marker="*",
                   color="black" if is_holm_sig else "gray", zorder=5)
        ax.text(0.03, 0.95, f"Cell {cid}\nL17 Δ={d['mean_overlap'][17]-d['mean_overlap'][35]:+.3f}\np_Holm={meta['l17_holm']}",
                transform=ax.transAxes, fontsize=7.5, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="gray"))
        if show_col_title:
            ax.set_title(col_title, fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(0.55, 1.05)

    # cls (left 2 columns)
    for cid, (r, c) in layout_cls.items():
        ax = all_axes[r, c]
        plot_panel(ax, cid, show_col_title=(r == 0), col_title=col_titles[c])
        if c == 0:
            ax.set_ylabel(row_labels[r] + "\n\noverlap to target", fontsize=8.5)
        if r == 1:
            ax.set_xlabel("Layer", fontsize=8.5)

    # reddit (right 2 columns)
    for cid, (r, c) in layout_reddit.items():
        ax = all_axes[r, c + 2]
        plot_panel(ax, cid, show_col_title=(r == 0), col_title=col_titles[c])
        if r == 1:
            ax.set_xlabel("Layer", fontsize=8.5)

    # Site dividers above each block
    fig.text(0.27, 0.96, "cls (4-cell)", fontsize=12, fontweight="bold", ha="center", color="#444")
    fig.text(0.72, 0.96, "reddit (4-cell)", fontsize=12, fontweight="bold", ha="center", color="#444")
    fig.suptitle("2x2 selection-bias control on both sites\n"
                 "cls: 3/4 Holm-sig (Cell C NULL → direction-tier conditional)  |  "
                 "reddit: 4/4 Holm-sig (universal mid-layer mechanism)",
                 fontsize=12, y=1.04)
    fig.text(0.01, 0.005,
             "8 cells (cls A/B/C/D + reddit F/G/Cr/Dr)  |  green = L17 Holm-sig, red = NULL  |  ★ marks L17 mid-layer probe",
             fontsize=6.5, color="gray")
    fig.tight_layout()
    out = OUT_DIR / "fig_mech_2x2_selection_bias.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[fig_mech_2x2_selection_bias] wrote {out}")
    plt.close(fig)


def fig_8cell_l17_forest():
    """At-a-glance forest plot of all 8 patching cells (cls + reddit, both directions, both tiers).
    x-axis = L17 Δ overlap_to_target, y-axis = cell label, marker color = Holm-sig vs NULL."""
    forest_cells = ["A", "B", "C", "D", "F", "G", "Cr", "Dr"]
    holm_pvals = {"A": 0.011, "B": 0.033, "C": 0.257, "D": 0.010,
                  "F": 0.004, "G": 0.036, "Cr": 0.012, "Dr": 0.041}

    rows = []
    for cid in forest_cells:
        d = load_cell_curves(cid)
        l17_mean = d["mean_overlap"][17]
        l35_mean = d["mean_overlap"][35]
        delta = l17_mean - l35_mean
        # Bootstrap 95% CI on Δ
        per_t_l17 = []
        per_t_l35 = []
        data = json.load(open(CELLS[cid] / "patching_continuation_results.json"))
        for t in data["per_task"]:
            for lr in t["per_layer"]:
                if lr.get("layer") == 17:
                    per_t_l17.append(lr.get("token_overlap_to_target"))
                elif lr.get("layer") == 35:
                    per_t_l35.append(lr.get("token_overlap_to_target"))
        diffs = np.array(per_t_l17) - np.array(per_t_l35)
        rng = np.random.default_rng(42)
        boot_means = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(2000)])
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
        rows.append((cid, delta, ci_lo, ci_hi, holm_pvals[cid], CELL_META[cid]["site"], CELL_META[cid]["label"]))

    rows.sort(key=lambda r: (r[5], r[1]))  # site then delta

    fig, ax = plt.subplots(1, 1, figsize=(10.0, 6.0))
    y_pos = list(range(len(rows)))
    for i, (cid, delta, lo, hi, p_holm, site, label) in enumerate(rows):
        is_sig = p_holm < 0.05
        color = "#2ca02c" if is_sig else "#d62728"
        ax.errorbar(delta, i, xerr=[[delta - lo], [hi - delta]],
                    fmt="o", color=color, markersize=10, linewidth=2, capsize=5)
        ax.text(0.18, i, f"Cell {cid}: {label}  (p_Holm={p_holm:.3f}{' ✓' if is_sig else ' ✗'})",
                fontsize=9, verticalalignment="center")
    ax.axvline(0, color="black", linestyle="-", alpha=0.4, linewidth=0.8)
    ax.axvline(-0.10, color="purple", linestyle=":", alpha=0.5, linewidth=0.8, label="-10pp threshold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.set_xlabel(r"L17 Δ overlap$_{\rm token}$ vs L35 baseline (95% bootstrap CI)")
    ax.set_xlim(-0.45, 0.45)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_title("L17 mid-layer mechanism — 8-cell forest plot\n"
                 "(cls 3/4 + reddit 4/4 = 7/8 Holm-sig, only cls Cell C NULL)",
                 fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.25, axis="x")
    fig.text(0.01, 0.005,
             "Source: per-cell patching_continuation_results.json (Stage 2 patching pilots)  |  "
             "bootstrap n=2000, seed=42  |  green=Holm-sig, red=NULL",
             fontsize=6.5, color="gray")
    fig.tight_layout()
    out = OUT_DIR / "fig_mech_8cell_l17_forest.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[fig_mech_8cell_l17_forest] wrote {out}")
    plt.close(fig)


def main():
    print("Loading cells:", list(CELLS.keys()))
    fig_4cell_curves()
    fig_real_vs_random()
    fig_2x2_selection_bias()
    fig_8cell_l17_forest()
    print(f"All 4 figures written under {OUT_DIR}")


if __name__ == "__main__":
    main()
