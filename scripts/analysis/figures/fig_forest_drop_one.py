#!/usr/bin/env python3
"""[Outcome supporting] Forest plot — per-arm drop-one lift across cells.

Reads `results/phantom_paper/phantom_lift.csv` (T0a-augmented with Holm/BH/TOST).
Renders 3 stacked panels (P-text / P-SoM / P-prompt), each showing per-cell
drop-one lift point estimate + raw 95% bootstrap CI + Holm-adjusted-significance
marker + TOST equivalence band (δ=0.5pp).

Output: `results/phantom_paper/figures/fig_forest_drop_one.png`

Per the pre-registered SECONDARY family (m = N_cells × 3 arms), Holm-Bonferroni
adjusted p-value gates the sig marker. Raw 95% CI is shown as the visual primary
because Holm-adjusted CI is rank-dependent within family and confuses the reader;
sig marker carries the multi-comparison gating.

T0b of `docs/reference/EVIDENCE_LAYER_AUDIT.md` action queue.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CSV_IN = ROOT / "results/phantom_paper/phantom_lift.csv"
OUT = ROOT / "results/phantom_paper/figures/fig_forest_drop_one.png"

# Per-arm config: (csv arm prefix, display label, color)
ARMS = [
    ("4pdom",    "P-text",   "#9e6da8"),
    ("4psom",    "P-SoM",    "#b279a2"),
    ("4pprompt", "P-prompt", "#9467bd"),
]

TOST_DELTA_PP = 1.0  # locked by preregistration.md §4; was 0.5 (codex audit 2026-05-13)
SIG_ALPHA = 0.05


def _f(x):
    """CSV float parser tolerating empty / 'None' strings."""
    if x is None or x == "" or x == "None":
        return None
    return float(x)


def load_rows() -> list[dict]:
    if not CSV_IN.exists():
        sys.exit(f"missing {CSV_IN}; run scripts/analysis/aggregate_phantom_lift.py first")
    with CSV_IN.open() as f:
        reader = csv.DictReader(f)
        return list(reader)


def cell_label(row: dict) -> str:
    return f"{row['baseline']} {row['site']}"


def draw_arm_panel(ax: plt.Axes, rows: list[dict], arm_code: str, label: str,
                   color: str) -> None:
    """One panel: y = cells, x = lift (pp), per-cell point + CI + sig."""
    # Filter cells that have data for this arm
    plotted = []
    for r in rows:
        lift = _f(r.get(f"lift_{arm_code}_vs_3_pp"))
        if lift is None:
            continue
        ci_lo = _f(r.get(f"lift_{arm_code}_vs_3_ci95_lo_pp"))
        ci_hi = _f(r.get(f"lift_{arm_code}_vs_3_ci95_hi_pp"))
        holm_p = _f(r.get(f"mcnemar_{arm_code}_vs_3_p_holm"))
        tost_p = _f(r.get(f"tost_{arm_code}_vs_3_p"))
        h = _f(r.get(f"cohen_h_{arm_code}_vs_3"))
        plotted.append({
            "label": cell_label(r),
            "lift": lift,
            "ci_lo": ci_lo if ci_lo is not None else lift,
            "ci_hi": ci_hi if ci_hi is not None else lift,
            "holm_p": holm_p,
            "tost_p": tost_p,
            "h": h,
        })

    # TOST equivalence band (δ=0.5pp): shade [-δ, +δ]
    ax.axvspan(-TOST_DELTA_PP, TOST_DELTA_PP, alpha=0.18, color="#888888", zorder=0,
               label=f"TOST equivalence band (±{TOST_DELTA_PP}pp)")
    # Null line
    ax.axvline(0, color="#444444", linewidth=0.8, linestyle="--", zorder=1)

    if not plotted:
        ax.text(0.5, 0.5, f"no cells with {label} data yet",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="#888888", style="italic")
        ax.set_yticks([])
        ax.set_title(f"{label} drop-one lift (vs 3-mode)", fontsize=11,
                     fontweight="bold", color=color)
        return

    y = np.arange(len(plotted))
    lifts = np.array([p["lift"] for p in plotted])
    err_lo = np.array([max(0.0, p["lift"] - p["ci_lo"]) for p in plotted])
    err_hi = np.array([max(0.0, p["ci_hi"] - p["lift"]) for p in plotted])

    # Holm-significant cells = filled, non-sig = open marker
    sig_mask = np.array([
        (p["holm_p"] is not None and p["holm_p"] < SIG_ALPHA) for p in plotted
    ])

    # Error bars (raw 95% CI), all cells
    ax.errorbar(
        lifts, y, xerr=[err_lo, err_hi],
        fmt="none", ecolor=color, elinewidth=2.0, capsize=4, capthick=1.5, zorder=2,
    )
    # Filled markers (sig)
    if sig_mask.any():
        ax.scatter(lifts[sig_mask], y[sig_mask], s=110, marker="o",
                   facecolor=color, edgecolor="#222222", linewidth=1.2, zorder=3,
                   label="Holm-adj p < 0.05")
    # Open markers (non-sig)
    if (~sig_mask).any():
        ax.scatter(lifts[~sig_mask], y[~sig_mask], s=110, marker="o",
                   facecolor="white", edgecolor=color, linewidth=1.5, zorder=3,
                   label="Holm-adj p ≥ 0.05")

    # Annotations: lift value + Holm p + Cohen's h
    for i, p in enumerate(plotted):
        h_txt = f"h={p['h']:.3f}" if p["h"] is not None else ""
        holm_txt = (f"Holm p={p['holm_p']:.3f}"
                    if p["holm_p"] is not None else "Holm p=—")
        annot = f"  {p['lift']:+.2f}pp  ({holm_txt}, {h_txt})"
        ax.text(p["ci_hi"] + 0.15, i, annot, va="center", fontsize=8.0,
                color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels([p["label"] for p in plotted], fontsize=9)
    ax.invert_yaxis()  # First cell on top
    ax.set_title(f"{label} drop-one lift (vs 3-mode)", fontsize=11,
                 fontweight="bold", color=color)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    fig, axes = plt.subplots(len(ARMS), 1, figsize=(11.5, 7.5), sharex=True)
    if len(ARMS) == 1:
        axes = [axes]
    for ax, (code, label, color) in zip(axes, ARMS):
        draw_arm_panel(ax, rows, code, label, color)
    axes[-1].set_xlabel("Drop-one lift (pp; vs 3-mode oracle = DOM ∪ SoM ∪ Vision)",
                        fontsize=10)

    # Determine x range based on data + leave annotation room
    all_lifts = []
    all_ci = []
    for r in rows:
        for code, _, _ in ARMS:
            ci_hi = _f(r.get(f"lift_{code}_vs_3_ci95_hi_pp"))
            ci_lo = _f(r.get(f"lift_{code}_vs_3_ci95_lo_pp"))
            if ci_hi is not None:
                all_ci.append(ci_hi)
            if ci_lo is not None:
                all_ci.append(ci_lo)
            lift = _f(r.get(f"lift_{code}_vs_3_pp"))
            if lift is not None:
                all_lifts.append(lift)
    if all_ci:
        x_max = max(all_ci) + 6.0  # leave room for annotation
        x_min = min(min(all_ci), -1.0)
        for ax in axes:
            ax.set_xlim(x_min, x_max)

    # Shared legend: build from first panel
    handles, labels = axes[0].get_legend_handles_labels()
    # Dedup
    seen = set()
    legend_items = []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            legend_items.append((h, l))
    if legend_items:
        fig.legend([h for h, _ in legend_items], [l for _, l in legend_items],
                    loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3,
                    frameon=False, fontsize=9)

    fig.suptitle(
        "Forest plot — per-arm drop-one lift across cells (raw 95% bootstrap CI; Holm-gated sig)",
        fontsize=12.5, fontweight="bold",
    )
    fig.text(
        0.5, 0.02,
        "Filled marker = Holm-adjusted McNemar p < 0.05 within SECONDARY family "
        "(m = N_cells × 3 arms). Open marker = Holm p ≥ 0.05. Gray band = TOST "
        f"equivalence margin ±{TOST_DELTA_PP}pp (effects within band are practically zero).",
        ha="center", fontsize=8.5, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
