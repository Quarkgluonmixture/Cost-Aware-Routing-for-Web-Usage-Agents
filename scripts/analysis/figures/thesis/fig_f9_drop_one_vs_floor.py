#!/usr/bin/env python3
"""Thesis F9 — each arm's irreplaceable coverage, against the noise it must clear.

drop-one asks: if this arm were removed from the 6-mode portfolio, how much
oracle success would be lost? A positive value means the arm solves something no
other arm solves. Across the six VWA cells the direction is consistent — but
direction is not magnitude, and this figure is built so the two cannot be
confused.

The threshold band is the point of the figure. `noise_floor_inventory.md` §1b
shows that a single rerun of the SAME arm moves |ΔSR| with a standard deviation
of the same order as the measured floor itself, so an effect needs roughly
3.8-4.2pp before one rerun would be unlikely to produce it alone. Nearly every
drop-one value sits below that line.

This is why C2 was downgraded on 2026-08-10: the structure is real in direction
and does not clear the noise in magnitude. Both halves are drawn here rather
than one being asserted in prose.

⚠️ The source rows carry grade=NON_PAPER_GRADE and that is surfaced in the
caption rather than quietly dropped.

Output: final_dissertation/figures/fig_f9_drop_one_vs_floor.{png,pdf}
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "results/phantom_paper/fig0c_drop_one_bootstrap_ci.csv"
SRC_FLOOR = ROOT / "docs/analysis/cross_sites/noise_floor_inventory.md"
OUT = ROOT / "final_dissertation/figures/fig_f9_drop_one_vs_floor"

# Grouped by WHAT IS SENT, verified from the step records: the four text-side
# modes all have image_payload_bytes == 0; SoM sends both; Vision sends only the
# image. The text side is itself the 2x2 of text format x prompt style, which the
# `mark_count` field encodes directly (30 = [SOM_MARKS] text, 0 = AXTree).
SIDES = [("Text side  ·  no image sent", ["DOM", "P-text", "P-prompt", "P-SoM"]),
         ("Combined  ·  text + image", ["SoM"]),
         ("Visual  ·  image only", ["Vision"])]
ORDER = [m for _s, ms in SIDES for m in ms]
SIDE_OF = {m: i for i, (_s, ms) in enumerate(SIDES) for m in ms}
C_SIDE = ["#0072B2", "#009E73", "#D55E00"]
C_TH = "#333333"

RE_SPREAD = re.compile(
    r"\|\s*`B\d[.\w-]+`\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*[\d.]+pp\s*\|\s*"
    r"\*\*[\d.]+pp\*\*\s*\|\s*([\d.]+)pp\s*\|")


def load():
    rows = [r for r in csv.DictReader(SRC.open(encoding="utf-8"))
            if r.get("drop_one_loss_pp")]
    if not rows:
        raise SystemExit(f"{SRC}: no usable rows")
    partial = [r for r in rows if r.get("is_partial") == "True"]
    if partial:
        raise SystemExit(f"{SRC}: {len(partial)} partial rows — refusing to plot")
    spreads = [float(x) for x in RE_SPREAD.findall(
        SRC_FLOOR.read_text(encoding="utf-8"))]
    if not spreads:
        raise SystemExit(f"{SRC_FLOOR}: no one-sided spread column found")
    grades = {r.get("grade") for r in rows}
    return rows, (min(spreads), max(spreads)), grades


def build(ax, rows, band):
    lo, hi = band
    ax.axvspan(lo, hi, color=C_TH, alpha=0.13, zorder=0, lw=0)
    ax.axvline(lo, color=C_TH, lw=1.3, ls="--", zorder=1)

    y = list(range(len(ORDER)))[::-1]
    for yi, mode in zip(y, ORDER):
        sel = [r for r in rows if r["mode"] == mode]
        col = C_SIDE[SIDE_OF[mode]]
        for j, r in enumerate(sorted(sel, key=lambda r: float(r["drop_one_loss_pp"]))):
            v = float(r["drop_one_loss_pp"])
            off = (j - (len(sel) - 1) / 2) * 0.115
            ax.plot([float(r["ci95_low_pp"]), float(r["ci95_high_pp"])],
                    [yi + off] * 2, color=col, lw=1.1, alpha=0.35, zorder=2)
            ax.scatter([v], [yi + off], s=34, color=col, zorder=4)
        pos = sum(1 for r in sel if float(r["drop_one_loss_pp"]) > 0)
        ax.text(-0.28, yi, f"{pos}/{len(sel)} > 0", va="center", ha="right",
                fontsize=7.8, color=col, fontweight="bold")

    ax.set_yticks(y, ORDER, fontsize=9.4)
    ax.set_xlim(-1.35, max(float(r["ci95_high_pp"]) for r in rows) + 0.6)
    ax.set_ylim(-0.75, len(ORDER) - 0.25)
    ax.set_xlabel("oracle success lost if this arm is removed  [pp]  "
                  "(point = drop-one, line = bootstrap 95% CI)", fontsize=8.8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#F2F2F2", lw=0.8)
    ax.set_axisbelow(True)
    ax.text(hi + 0.12, len(ORDER) - 0.55,
            f"above {lo:.2f}–{hi:.2f}pp a single rerun\nof the same arm is "
            "unlikely to produce it",
            fontsize=7.8, color=C_TH, va="center", linespacing=1.5)
    ax.legend(handles=[Patch(color=C_SIDE[i], label=lab)
                       for i, (lab, _m) in enumerate(SIDES)],
              loc="lower right", frameon=False, fontsize=8.0)
    # side separators
    for i in range(len(SIDES) - 1):
        cut = sum(len(ms) for _s, ms in SIDES[:i + 1])
        ax.axhline(len(ORDER) - cut - 0.5, color="#DDDDDD", lw=1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    rows, band, grades = load()
    cells = sorted({r["site_baseline"] for r in rows})
    # "phantom" = the three text-side arms other than DOM: same side, so they
    # differ only in text format and prompt style.
    phantom = set(SIDES[0][1]) - {"DOM"}
    ph = [r for r in rows if r["mode"] in phantom]
    ph_v = sorted(float(r["drop_one_loss_pp"]) for r in ph)
    ph_pos = sum(1 for x in ph_v if x > 0)
    above = [r for r in rows if float(r["drop_one_loss_pp"]) >= band[0]]

    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    build(ax, rows, band)
    ax.set_title("Every arm is uniquely useful somewhere — and almost none of it "
                 "clears the rerun noise",
                 fontsize=11.4, fontweight="bold", loc="left", pad=42)
    ax.text(0.0, 1.012,
            f"The four text-side arms differ only in text format and prompt "
            f"style, so they overlap heavily: {ph_pos} of {len(ph)} phantom "
            f"values are positive but span only {ph_v[0]:.2f}–{ph_v[-1]:.2f}pp. "
            f"The larger\nunique contributions come from crossing sides. Even so, "
            f"only {len(above)} of {len(rows)} values reach the band where one "
            "rerun would be an unlikely explanation.",
            transform=ax.transAxes, fontsize=8.4, color="#444444",
            linespacing=1.5, va="bottom")
    fig.text(0.012, 0.005,
             f"Source: results/phantom_paper/fig0c_drop_one_bootstrap_ci.csv "
             f"({len(cells)} cells × 6 modes, complete-case, "
             f"grade={'/'.join(sorted(grades))}); threshold band from "
             "noise_floor_inventory.md §1b (derived from one cell's three "
             "replicated arms).",
             fontsize=7.0, color="#888888")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=220, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"wrote {a.out}.png / .pdf   ({len(rows)} arm-cells over {len(cells)} "
          f"cells; phantom {ph_pos}/{len(ph)} positive, {ph_v[0]:.2f}–"
          f"{ph_v[-1]:.2f}pp; {len(above)}/{len(rows)} reach {band[0]:.2f}pp)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
