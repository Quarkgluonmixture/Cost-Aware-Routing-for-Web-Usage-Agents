#!/usr/bin/env python3
"""[Outcome supporting] Meta-analytic forest plot — per-arm pooled estimate
+ I² + Q + per-cell weight squares.

Reads `results/phantom_paper/meta_phantom_lift.csv` (T0c output) +
`results/phantom_paper/phantom_lift.csv` (T0a output, for per-cell rows).
Renders 3 panels (P-text / P-SoM / P-prompt) with classical forest convention:

- Per-cell row: square sized by RE weight (% of pooled), horizontal 95% CI line
- Pooled row at bottom: diamond (center = pooled estimate, width = pooled 95% CI)
- I² + Q + p_Q + τ² annotation in panel
- Vertical line at lift = 0 (null)
- TOST equivalence band shaded ±0.5pp

Output: `results/phantom_paper/figures/fig_meta_forest.png`

T0d of `docs/reference/EVIDENCE_LAYER_AUDIT.md` action queue.

Difference from `fig_forest_drop_one.py` (T0b): this adds the meta-pooled
diamond + heterogeneity statistics + weight-sized squares (classical forest
convention). T0b is per-cell-only, no pooling.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
LIFT_CSV = ROOT / "results/phantom_paper/phantom_lift.csv"
META_CSV = ROOT / "results/phantom_paper/meta_phantom_lift.csv"
OUT = ROOT / "results/phantom_paper/figures/fig_meta_forest.png"

# Per-arm panel config: (csv arm prefix, meta arm_code, label, color, role)
# Reordered 2026-05-03: P-SoM HERO first; P-text/P-prompt as STRUCTURAL ABLATION arms
# (per `EVIDENCE_LAYER_AUDIT.md` §2 Hero+Structural+Framing-rule pre-registration).
ARMS = [
    ("4psom",    "4psom_vs_3",    "P-SoM drop-in",    "#b279a2", "HERO"),
    ("4pdom",    "4pdom_vs_3",    "P-text drop-in",   "#9e6da8", "ABLATION"),
    ("4pprompt", "4pprompt_vs_3", "P-prompt drop-in", "#9467bd", "ABLATION"),
]

ROLE_BADGE = {
    "HERO":     "DEPLOYMENT HERO (H1, gating)",
    "ABLATION": "STRUCTURAL ABLATION (H4 exploratory; H3 axis evidence)",
}
ROLE_DIAMOND_STYLE = {
    "HERO":     dict(facecolor="#000000", edgecolor="#000000", linewidth=1.2, alpha=1.0),
    "ABLATION": dict(facecolor="#cccccc", edgecolor="#444444", linewidth=1.0, alpha=0.7),
}

TOST_DELTA_PP = 0.5


def _f(x):
    if x is None or x == "" or x == "None":
        return None
    return float(x)


def load_per_cell(arm_csv_prefix: str) -> list[dict]:
    rows = []
    with LIFT_CSV.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            theta = _f(r.get(f"lift_{arm_csv_prefix}_vs_3_pp"))
            ci_lo = _f(r.get(f"lift_{arm_csv_prefix}_vs_3_ci95_lo_pp"))
            ci_hi = _f(r.get(f"lift_{arm_csv_prefix}_vs_3_ci95_hi_pp"))
            if theta is None or ci_lo is None or ci_hi is None:
                continue
            se = (ci_hi - ci_lo) / (2 * 1.96)
            if se <= 0:
                continue
            rows.append({
                "label": f"{r['baseline']} {r['site']}",
                "theta": theta,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "se": se,
            })
    return rows


def load_meta(arm_code: str) -> dict | None:
    if not META_CSV.exists():
        return None
    with META_CSV.open() as f:
        for r in csv.DictReader(f):
            if r["arm_code"] == arm_code:
                return r
    return None


def draw_arm_panel(ax: plt.Axes, arm_csv_prefix: str, meta_arm_code: str,
                   label: str, color: str, role: str = "ABLATION") -> None:
    cells = load_per_cell(arm_csv_prefix)
    meta = load_meta(meta_arm_code)

    # TOST band + null line
    ax.axvspan(-TOST_DELTA_PP, TOST_DELTA_PP, alpha=0.18, color="#888888", zorder=0)
    ax.axvline(0, color="#444444", linewidth=0.8, linestyle="--", zorder=1)

    if not cells:
        ax.text(0.5, 0.5, f"no cells for {label}", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="#888888", style="italic")
        ax.set_yticks([])
        ax.set_title(label, fontsize=11, fontweight="bold", color=color)
        return

    # RE weight per cell (use meta's tau² if available, else FE weights)
    tau2 = float(meta["tau2"]) if meta and meta.get("tau2") not in (None, "") else 0.0
    var_star = np.array([c["se"] ** 2 + tau2 for c in cells])
    w_star = 1.0 / var_star
    weights = w_star / w_star.sum()

    n_rows = len(cells) + (1 if meta else 0)  # cells + diamond row
    y_positions = np.arange(n_rows)
    cell_y = y_positions[:len(cells)]

    # Per-cell rows: weight-sized squares + CI line
    for i, c in enumerate(cells):
        # CI line
        ax.plot([c["ci_lo"], c["ci_hi"]], [cell_y[i], cell_y[i]],
                color=color, linewidth=1.5, zorder=2)
        # Square sized by weight (visual area ∝ weight)
        size = 80 + 320 * weights[i]  # 80-400 size range
        ax.scatter(c["theta"], cell_y[i], marker="s", s=size,
                   facecolor=color, edgecolor="#222222", linewidth=0.8, zorder=3)

    # Pooled diamond — HERO = filled black; ABLATION = gray outlined
    if meta:
        re_y = y_positions[-1]
        re_theta = float(meta["theta_re"])
        re_ci_lo = float(meta["ci_lo"])
        re_ci_hi = float(meta["ci_hi"])
        diamond_x = [re_ci_lo, re_theta, re_ci_hi, re_theta]
        diamond_y = [re_y, re_y - 0.28, re_y, re_y + 0.28]
        style = ROLE_DIAMOND_STYLE[role]
        ax.fill(diamond_x, diamond_y, **style, zorder=4)
        # Outline only for ABLATION (already drawn by fill, but emphasize edge)
        if role == "ABLATION":
            ax.plot(diamond_x + [diamond_x[0]], diamond_y + [diamond_y[0]],
                    color=style["edgecolor"], linewidth=style["linewidth"], zorder=4)
        # Pooled summary line at zero
        ax.axhline(re_y - 0.65, color="#cccccc", linewidth=0.5, linestyle=":")

    # Y-tick labels
    yticks = list(cell_y)
    yticklabels = [c["label"] for c in cells]
    if meta:
        yticks.append(y_positions[-1])
        k = meta["k_cells"]
        yticklabels.append(f"Pooled (RE, k={k})")
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=9)
    ax.invert_yaxis()

    # Annotations: per-cell lift + weight; pooled lift + I² + Q
    x_anno = ax.get_xlim()[1] if False else None  # set after autoscale
    for i, c in enumerate(cells):
        ax.text(c["ci_hi"] + 0.15, cell_y[i],
                f"  {c['theta']:+.2f}pp [{c['ci_lo']:.2f},{c['ci_hi']:.2f}] "
                f"  w={weights[i]*100:.0f}%",
                va="center", fontsize=8.0, color="#333333")
    if meta:
        I2 = float(meta["I2"])
        Q = float(meta["Q"])
        df = int(meta["df"])
        p_Q = meta.get("p_Q")
        p_Q_str = (f"{float(p_Q):.3f}" if p_Q not in (None, "") else "—")
        re_theta = float(meta["theta_re"])
        re_ci_lo = float(meta["ci_lo"])
        re_ci_hi = float(meta["ci_hi"])
        p_re_holm = meta.get("p_re_holm", "")
        holm_str = (f"Holm p={float(p_re_holm):.3f}" if p_re_holm not in (None, "")
                    else "")
        ax.text(re_ci_hi + 0.15, y_positions[-1],
                f"  {re_theta:+.2f}pp [{re_ci_lo:.2f},{re_ci_hi:.2f}]  "
                f"I²={I2:.0f}%  Q({df})={Q:.2f} p={p_Q_str}  {holm_str}",
                va="center", fontsize=8.0, color="#222222", fontweight="bold")

    # Title with role badge
    title_color = "#000000" if role == "HERO" else color
    title_weight = "bold"
    ax.set_title(f"{label}  —  {ROLE_BADGE[role]}", fontsize=11,
                 fontweight=title_weight, color=title_color)
    # HERO panel gets a stronger frame to emphasize epistemic priority
    if role == "HERO":
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
            spine.set_color("#000000")
    ax.grid(axis="x", color="#dddddd", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not LIFT_CSV.exists():
        sys.exit(f"missing {LIFT_CSV}; run aggregate_phantom_lift.py first")
    if not META_CSV.exists():
        sys.exit(f"missing {META_CSV}; run aggregate_phantom_meta.py first")

    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    fig, axes = plt.subplots(len(ARMS), 1, figsize=(13.0, 8.0), sharex=True)
    if len(ARMS) == 1:
        axes = [axes]
    for ax, (csv_prefix, meta_code, label, color, role) in zip(axes, ARMS):
        draw_arm_panel(ax, csv_prefix, meta_code, label, color, role)
    axes[-1].set_xlabel("Drop-one lift (pp; vs 3-mode oracle = DOM ∪ SoM ∪ Vision)",
                        fontsize=10)

    # Set generous x range to leave room for annotations
    all_max = []
    all_min = []
    with LIFT_CSV.open() as f:
        for r in csv.DictReader(f):
            for prefix, _, _, _, _ in ARMS:
                hi = _f(r.get(f"lift_{prefix}_vs_3_ci95_hi_pp"))
                lo = _f(r.get(f"lift_{prefix}_vs_3_ci95_lo_pp"))
                if hi is not None:
                    all_max.append(hi)
                if lo is not None:
                    all_min.append(lo)
    if all_max:
        x_max = max(all_max) + 9.0  # extra room for verbose annotation
        x_min = min(min(all_min), -1.0)
        for ax in axes:
            ax.set_xlim(x_min, x_max)

    fig.suptitle(
        "Meta-analytic forest — Hero (P-SoM, gating) vs Structural Ablation (P-text/P-prompt, exploratory)",
        fontsize=12.5, fontweight="bold",
    )
    fig.text(
        0.5, 0.025,
        "Per-cell square sized by random-effect weight; horizontal line = 95% bootstrap CI. "
        "**Filled black diamond** (top panel) = HERO pooled estimate (P-SoM, paper hook gating). "
        "**Gray outlined diamond** (lower 2 panels) = STRUCTURAL ABLATION pooled estimate (P-text / "
        "P-prompt, exploratory; H4 magnitude not paper-claim gating; H3 axis evidence in separate Venn fig). "
        "I² = % variation due to between-cell heterogeneity (low/mod/subs/cons per Higgins-Thompson). "
        f"Gray band = TOST equivalence margin ±{TOST_DELTA_PP}pp.",
        ha="center", fontsize=8.5, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
