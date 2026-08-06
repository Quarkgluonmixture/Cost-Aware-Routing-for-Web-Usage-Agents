#!/usr/bin/env python3
"""The two ceilings a perfect per-task choice could reach, per cell.

Left panel: the six-arm union against the best single mode, with two reference
marks on the same row --- what one genuinely different arm adds (tick) and what
one rerun of an arm already in hand adds (dark segment, where a replicate exists).
Right panel: the cost ceiling at matched success, the bound that survives the
rerun control because it adds no arm.

Committed 2026-08-06. The Overleaf copy was a PDF with no script behind it, so the
panel spacing could not be adjusted by anyone who did not have the original
machine --- the same failure mode recorded for the two forest plots. This puts it
back under version control. Style matches the sibling forest plots (palette, type
scale, spine treatment) so the data figures still read as one set.

Usage:
    python3 scripts/analysis/figures/fig_ceilings.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "docs" / "analysis" / "cross_sites" / "routing_ceiling.json"

# Shared with the forest plots so the three data figures read as one set.
BLUE = "#2A78D6"
ORANGE = "#EB6834"
GREY = "#52514E"
TEAL = "#2CA58D"
FIGSIZE = (3.30, 2.05)

PRETTY = {"cls_B0": "cls·B0", "cls_B1": "cls·B1", "cls_B2": "cls·B2",
          "red_B0": "red·B0", "red_B1": "red·B1", "red_B2": "red·B2",
          "wa_red_B0": "WA·B0", "wa_red_B1": "WA·B1"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path.home() / "overleaf-aaai27" / "figures")
    ap.add_argument("--policy", default="leak_kept", choices=["leak_kept", "leak_zeroed"],
                    help="leakage convention; the paper reports leak_kept and treats "
                         "zeroing as a sensitivity analysis")
    args = ap.parse_args()

    cells = json.loads(SRC.read_text())["cells"]
    unknown = [c["cell"] for c in cells if c["cell"] not in PRETTY]
    if unknown:
        raise SystemExit(f"{SRC.name} has cells this plot cannot label: {unknown}")

    # Descending best-single-mode, so the reader scans capability top to bottom.
    rows = sorted(cells, key=lambda c: c[args.policy]["best_sr_pct"], reverse=True)

    plt.rcParams.update({"font.size": 6, "axes.linewidth": 0.6,
                         "xtick.labelsize": 6, "ytick.labelsize": 6})
    # width_ratios keeps the cost panel narrow; wspace is deliberately tight so the
    # cost axis sits close to the success axis rather than drifting to the margin.
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=FIGSIZE, sharey=True,
        gridspec_kw={"width_ratios": [2.35, 1.0], "wspace": 0.10})

    for i, c in enumerate(rows):
        blk = c[args.policy]
        best = blk["best_sr_pct"]
        union = blk["oracle_sr_pct"]

        # Baseline-to-union span, drawn first so the marks sit on top of it.
        axL.plot([best, union], [i, i], color="#C9C8C4", lw=1.6,
                 solid_capstyle="butt", zorder=1)

        # What one rerun of an arm already in hand buys, where a replicate exists.
        draws = c.get("rerun_draws_pp")
        if draws:
            axL.plot([best + min(draws), best + max(draws)], [i, i],
                     color=GREY, lw=2.4, solid_capstyle="butt", zorder=2)

        # What one genuinely different arm buys, at the same arm count.
        axL.plot([best + c["arm_matched_gain_pp"]], [i], marker="|", ms=6.0,
                 markeredgewidth=1.4, color=ORANGE, zorder=4)

        axL.plot([best], [i], marker="o", ms=3.0, color=BLUE, zorder=5)
        axL.plot([union], [i], marker="o", ms=3.4, mfc="white",
                 mec=BLUE, mew=1.0, zorder=5)

        # triage_, not oracle_: the paper's cost ceiling keeps the best mode
        # everywhere and reroutes only the never-solved tasks, which is the
        # 9.5-30.6% range quoted in the text. oracle_ is a different (larger)
        # quantity and would silently contradict it.
        axR.barh(i, blk["triage_cost_saving_pct"], height=0.55,
                 color=TEAL, zorder=2)

    axL.set_yticks(range(len(rows)))
    axL.set_yticklabels([PRETTY[c["cell"]] for c in rows])
    axL.invert_yaxis()
    axL.set_xlabel("success rate (%)", fontsize=6)
    axR.set_xlabel("cost saved (%)", fontsize=6)
    axR.set_xlim(left=0)

    for ax in (axL, axR):
        ax.tick_params(length=2, width=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="x", color="#E5E4E0", lw=0.5, zorder=0)
        ax.set_axisbelow(True)

    handles = [
        plt.Line2D([], [], color=BLUE, lw=0, marker="o", ms=3.0, label="best single mode"),
        plt.Line2D([], [], color=BLUE, lw=0, marker="o", ms=3.4, mfc="white",
                   mec=BLUE, mew=1.0, label="six-arm union"),
        plt.Line2D([], [], color=ORANGE, lw=0, marker="|", ms=6.0,
                   markeredgewidth=1.4, label="+1 distinct arm"),
        plt.Line2D([], [], color=GREY, lw=2.4, label="+1 rerun (measured)"),
    ]
    # tight_layout is unaware of a figure-level legend, so lay the axes out first
    # and let bbox_inches="tight" grow the canvas around the legend afterwards.
    fig.tight_layout(pad=0.25)
    fig.legend(handles=handles, fontsize=5.5, frameon=False, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, 1.0),
               handlelength=1.3, columnspacing=1.1, handletextpad=0.5)

    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / "fig_ceilings.pdf"
    fig.savefig(dest, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {dest}")
    saves = [c[args.policy]["triage_cost_saving_pct"] for c in rows]
    print("cost saved (%): " + ", ".join(
        f"{PRETTY[c['cell']]} {s:.1f}" for c, s in zip(rows, saves)))
    print(f"range {min(saves):.1f}-{max(saves):.1f} (text says 9.5-30.6)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
