#!/usr/bin/env python3
"""[Paper §8 Appendix-D sensitivity figure] Meta-analytic forest plot — DL/HKSJ
per-arm pooled estimate + I² + Q + per-cell weight squares.

⚠️ **FIGURE STATUS (B-1307 /stress A2.3d P0-2-A*, 2026-05-18)**: This figure is
**APPENDIX-D SENSITIVITY** per prereg §2 H1 decision 3A (FE primary, no τ²).
The paper §1 hero forest plot must use fixed-effects pool from
`phase1_full_prereg_decision.{csv,json}` (B-1301 bootstrap percentile primary
gate) — NOT the DL/HKSJ pooled diamond rendered here. This figure is reserved
for paper §8 Appendix-D heterogeneity sensitivity reporting only.

Reads `results/phantom_paper/meta_phantom_lift.csv` (DL/HKSJ appendix-only
producer per B-1016 MD-warning header) + `results/phantom_paper/phantom_lift.csv`
(T0a output, for per-cell rows). Renders 3 panels (P-text / P-SoM / P-prompt)
with classical forest convention:

- Per-cell row: square sized by RE weight (% of pooled), horizontal 95% CI line
- Pooled row at bottom: diamond (center = pooled estimate, width = pooled 95% CI)
- I² + Q + p_Q + τ² annotation in panel
- Vertical line at lift = 0 (null)

Output: `results/phantom_paper/figures/fig_meta_forest.png` (Appendix-D figure)

T0d of `docs/reference/EVIDENCE_LAYER_AUDIT.md` action queue — now scoped to
Appendix sensitivity tier.

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
    ("4psom",    "4psom_vs_3",    "P-SoM drop-in",    "#b279a2", "APPENDIX_EXPLORATORY"),
    ("4pdom",    "4pdom_vs_3",    "P-text drop-in",   "#9e6da8", "APPENDIX_EXPLORATORY"),
    ("4pprompt", "4pprompt_vs_3", "P-prompt drop-in", "#9467bd", "APPENDIX_EXPLORATORY"),
]

# /stress A1.20 P0-4-AC (2026-05-17, A1.19 B-437 figure-layer propagation gap):
# the 3→5-mode lift estimand (`4psom_vs_3` etc.) was demoted to APPENDIX exploratory
# per A1.19 B-184/B-437 (aggregate_phantom_lift.md prose rewrite). Figure label here
# was still "DEPLOYMENT HERO (H1, gating)" — stale. True paper §1 H1 PRIMARY is
# `aggregate_phase1_prereg_gate.{csv,json,md}` (P-SoM drop-one over 6-mode universe
# FE inverse-variance pool; per preregistration.md §2 H1 decision "3A" 2026-05-14).
# All 3 arms in THIS figure are appendix-only.
ROLE_BADGE = {
    "APPENDIX_EXPLORATORY": "APPENDIX exploratory (3→5-mode legacy lift; cf. phase1_prereg_gate.{csv,md} for H1 PRIMARY)",
    # Legacy compat keys retained so any code path inadvertently passing old key
    # gets graceful behavior rather than KeyError; both deprecate alongside A1.19.
    "HERO":     "APPENDIX exploratory (legacy HERO label; see A1.19 B-437)",
    "ABLATION": "APPENDIX exploratory (legacy ABLATION label; see A1.19 B-437)",
}
ROLE_DIAMOND_STYLE = {
    # Same diamond style across all 3 arms (no HERO emphasis anymore).
    "APPENDIX_EXPLORATORY": dict(facecolor="#cccccc", edgecolor="#444444",
                                  linewidth=1.0, alpha=0.7),
    "HERO":     dict(facecolor="#cccccc", edgecolor="#444444", linewidth=1.0, alpha=0.7),
    "ABLATION": dict(facecolor="#cccccc", edgecolor="#444444", linewidth=1.0, alpha=0.7),
}

TOST_DELTA_PP = 1.0  # locked by preregistration.md §4; was 0.5 (codex audit 2026-05-13)


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

    # Pooled diamond — APPENDIX_EXPLORATORY all gray-outlined (A1.20 P0-4 fix)
    if meta:
        re_y = y_positions[-1]
        re_theta = float(meta["theta_re"])
        re_ci_lo = float(meta["ci_lo"])
        re_ci_hi = float(meta["ci_hi"])
        diamond_x = [re_ci_lo, re_theta, re_ci_hi, re_theta]
        diamond_y = [re_y, re_y - 0.28, re_y, re_y + 0.28]
        style = ROLE_DIAMOND_STYLE.get(role, ROLE_DIAMOND_STYLE["APPENDIX_EXPLORATORY"])
        ax.fill(diamond_x, diamond_y, **style, zorder=4)
        ax.plot(diamond_x + [diamond_x[0]], diamond_y + [diamond_y[0]],
                color=style["edgecolor"], linewidth=style["linewidth"], zorder=4)
        # /stress A1.20 P0-1-AC* (2026-05-17, A1.19 B-431 figure-layer fill):
        # HKSJ diamond now renders alongside DL-Wald iff CSV has the HKSJ columns
        # (aggregate_phantom_meta.py emits hk_theta_re / hk_ci_lo / hk_ci_hi cols
        # per same /stress fix). HKSJ is decision-grade at k≤10 (IntHout 2014);
        # DL-Wald shown for backward-compat with archive prose. Render HKSJ as a
        # smaller open diamond above DL-Wald row + dotted line so visual diff is
        # explicit.
        hk_theta = meta.get("hk_theta_re")
        hk_ci_lo = meta.get("hk_ci_lo")
        hk_ci_hi = meta.get("hk_ci_hi")
        if hk_theta not in (None, "") and hk_ci_lo not in (None, "") and hk_ci_hi not in (None, ""):
            hk_y = re_y - 0.55  # slight offset above DL-Wald diamond
            hk_theta_f = float(hk_theta)
            hk_ci_lo_f = float(hk_ci_lo)
            hk_ci_hi_f = float(hk_ci_hi)
            hk_dx = [hk_ci_lo_f, hk_theta_f, hk_ci_hi_f, hk_theta_f]
            hk_dy = [hk_y, hk_y - 0.18, hk_y, hk_y + 0.18]
            ax.fill(hk_dx, hk_dy, facecolor="white", edgecolor="#000000",
                    linewidth=1.4, alpha=0.9, zorder=5)
            ax.plot(hk_dx + [hk_dx[0]], hk_dy + [hk_dy[0]],
                    color="#000000", linewidth=1.4, zorder=5)
            ax.text(hk_ci_hi_f + 0.15, hk_y,
                    f"  HKSJ {hk_theta_f:+.2f}pp [{hk_ci_lo_f:.2f},{hk_ci_hi_f:.2f}] "
                    f"(decision-grade at k≤10 per IntHout 2014)",
                    va="center", fontsize=7.5, color="#000000",
                    fontstyle="italic")
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

    # Title with role badge. /stress A1.20 P0-4 (2026-05-17): no HERO frame
    # anymore — all 3 arms are APPENDIX exploratory per A1.19 B-437 demote.
    badge_text = ROLE_BADGE.get(role, ROLE_BADGE["APPENDIX_EXPLORATORY"])
    ax.set_title(f"{label}  —  {badge_text}", fontsize=11,
                 fontweight="bold", color=color)
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
        "Meta-analytic forest — APPENDIX exploratory (3→5-mode legacy lift; "
        "see `phase1_prereg_gate.{csv,md}` for paper §1 H1 PRIMARY hero)",
        fontsize=12.5, fontweight="bold",
    )
    fig.text(
        0.5, 0.025,
        "/stress A1.20 P0-4-AC (2026-05-17, A1.19 B-437 figure-layer propagation gap closed): "
        "the 3→5-mode lift estimand was demoted to APPENDIX exploratory per A1.19 B-184/B-437. "
        "True paper §1 H1 PRIMARY = P-SoM drop-one over 6-mode universe FE inverse-variance pool "
        "(producer: `aggregate_phase1_prereg_gate.py`, output: `phase1_prereg_gate.{csv,md}`). "
        "Per-cell square sized by random-effect weight; horizontal line = 95% bootstrap CI. "
        "**Gray diamond** = DerSimonian-Laird Wald RE pool (legacy descriptive only). "
        "**White outlined diamond** (when present) = Hartung-Knapp-Sidik-Jonkman adjustment "
        "(decision-grade at k≤10 per IntHout et al. 2014; A1.19 B-431). "
        "I² = % variation due to between-cell heterogeneity (low/mod/subs/cons per Higgins-Thompson). "
        f"Gray band = TOST equivalence margin ±{TOST_DELTA_PP}pp.",
        ha="center", fontsize=8.0, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
