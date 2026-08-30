#!/usr/bin/env python3
"""Thesis F7 — cost against success, one panel per cell, dominated modes shown.

Two rules govern this figure and both exist because of a specific way it could
mislead.

1. Dominated modes are drawn, not hidden. A frontier plot that shows only the
   non-dominated set makes every mode look reasonable; the interesting fact is
   that several modes cost more AND succeed less than an alternative.

2. Panels are never compared across the API/local boundary. Both axes are
   `total_billed_cost_usd`, but the SCHEDULE differs in kind: B0's is a vendor
   price, B1/B2's a device-amortisation rate. Each panel therefore carries its
   own x-axis and its own unit label, and no shared axis invites the ratio.
   Neither axis is measured electricity — that is a separate quantity, reported
   only in the sustainability discussion.

Output: final_dissertation/figures/fig_f7_cost_sr_frontier.{png,pdf}

Run via ``make thesis-figures``, not on its own: this script writes only the
working tree, and the copy LaTeX embeds is refreshed by that target.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as S  # noqa: E402
from _ordering_parse import (  # noqa: E402
    C_SIDE, MODES, SIDE_OF, SIDES, load)

OUT = (Path(__file__).resolve().parents[4]
       / "final_dissertation/figures/fig_f7_cost_sr_frontier")


def pareto(points):
    """Indices of non-dominated points: cheaper is better, higher SR is better."""
    keep = []
    for i, (ci, si) in enumerate(points):
        if not any((cj <= ci and sj >= si) and (cj < ci or sj > si)
                   for j, (cj, sj) in enumerate(points) if j != i):
            keep.append(i)
    return keep


# Shape carries the mode; colour keeps carrying the side. This was forced by a
# measurement, not a preference: the panel is 180pt wide and the six mode names
# laid side by side need 183pt, so on a panel whose points cluster --- and they
# all do, the figure existing precisely because six modes cost nearly the same
# --- NO arrangement of six text labels fits. Four attempts at cleverer offsets
# each produced a different overlap ("P-SoMprompt" in one, labels escaping into
# the panel title in another). Only the labels the prose actually argues about,
# the non-dominated ones, are still written out; there are at most three per
# panel, so those always fit.
MODE_MARKER = {"DOM": "o", "SoM": "s", "Vision": "D",
               "P-text": "^", "P-prompt": "v", "P-SoM": "P"}

# Modes the surrounding prose names by hand even though they are dominated, so
# a reader following the sentence to the panel can still find them. ch4 §4.2
# contrasts Vision against P-SoM on cls-B0 and against SoM on cls-B2, and says
# Vision is "the fourth most expensive of six" on wa_red-B0. Without these three
# the text points at a mark that carries no name.
PROSE_NAMED = {"cls\u00b7B0": {"P-SoM"}, "cls\u00b7B2": {"SoM"},
               "wa_red\u00b7B0": {"Vision"}}


# Offsets tried in order. Every one keeps the label touching its own mark.
_CAND = [(0, 7, "center", "bottom"), (0, -8, "center", "top"),
         (8, 0, "left", "center"), (-8, 0, "right", "center"),
         (8, 7, "left", "bottom"), (-8, -8, "right", "top"),
         (0, 17, "center", "bottom"), (0, -18, "center", "top")]


def _front_labels(ax, labels):
    """Name at most four marks per panel without printing one over another.

    Alternating above/below --- by list order OR by cost order --- is blind to
    the OTHER axis: on wa_red-B0 the P-text mark sits at 35% success and Vision
    at 19%, so "below P-text" and "above Vision" met in the middle and printed
    as "P-teVision". Four labels in a 180x124pt panel do fit; they just have to
    be placed against measured boxes rather than by a parity rule. Called after
    set_xlim/set_ylim --- an offset in points means nothing until the scale is
    final.
    """
    bb = ax.get_window_extent()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    sx, sy = bb.width / (x1 - x0), bb.height / (y1 - y0)
    ch_w, ln_h = S.FS_VALUE * 0.58, S.FS_VALUE * 1.25
    placed = []
    for m, c, sr_ in sorted(labels, key=lambda t: -t[2]):
        px, py = (c - x0) * sx, (sr_ - y0) * sy
        w = len(m) * ch_w
        for dx, dy, ha, va in _CAND:
            cx = px + dx + (w / 2 if ha == "left" else -w / 2 if ha == "right" else 0)
            cy = py + dy + (ln_h / 2 if va == "bottom" else -ln_h / 2 if va == "top" else 0)
            if all(abs(cx - ox) >= (w + ow) / 2 or abs(cy - oy) >= (ln_h + oh) / 2
                   for ox, oy, ow, oh in placed):
                break
        placed.append((cx, cy, w, ln_h))
        ax.annotate(m, (c, sr_), textcoords="offset points", xytext=(dx, dy),
                    ha=ha, va=va, fontsize=S.FS_VALUE, color="#222222")


def panel(ax, cell, n, sr, cost, unit):
    pts = [(cost[(cell, m)], sr[(cell, m)]) for m in MODES]
    front = sorted(pareto(pts), key=lambda i: pts[i][0])
    ax.plot([pts[i][0] for i in front], [pts[i][1] for i in front],
            color="#666666", lw=1.1, ls="-", zorder=1, alpha=0.7)
    labels = []
    for i, m in enumerate(MODES):
        c, s = pts[i]
        on = i in front
        ax.scatter([c], [s], s=74 if on else 46, color=C_SIDE[SIDE_OF[m]],
                   marker=MODE_MARKER.get(m, "o"),
                   edgecolor="#333333" if on else "none",
                   lw=1.1, zorder=3, alpha=1.0 if on else 0.55)
        if on or m in PROSE_NAMED.get(S.cell_label(cell), ()):
            labels.append((m, c, s))
    ax.set_title(f"{S.cell_label(cell)}   $n$={n}", fontsize=S.FS_PANEL,
                 loc="left", pad=6)
    ax.set_xlabel(unit, fontsize=S.FS_VALUE)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    lo = min(p[0] for p in pts)
    hi = max(p[0] for p in pts)
    ax.set_xlim(lo - (hi - lo) * 0.18, hi + (hi - lo) * 0.18)
    ymax = max(p[1] for p in pts)
    ax.set_ylim(-ymax * 0.18, ymax * 1.32)
    # After the limits are set, never before: an offset in points is only
    # meaningful once the axes scale is final.
    _front_labels(ax, labels)
    return len(front)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    cells, n_of, sr, cost = load()

    S.apply()
    fig, axes = plt.subplots(4, 2, figsize=(S.PRINT_W_IN, 7.4))
    fronts = []
    for ax, cell in zip(axes.ravel(), cells):
        # ⚠️ Both are `total_billed_cost_usd` — a TOKEN-priced quantity. B0's
        # per-1k schedule is the vendor's actual billing rate; B1/B2's is a
        # device-amortisation rate. Neither is measured electricity: for B1 DOM
        # the plotted 0.0595 sits ~88x above the 0.000677 that energy x tariff
        # gives. Naming the axis "electricity" would be naming a different
        # quantity than the one drawn.
        unit = ("billed \\$ / episode" if "B0" in cell
                else "amortised \\$ / episode")
        fronts.append(panel(ax, cell, n_of[cell], sr, cost, unit))
    for ax in axes.ravel()[len(cells):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("task success rate (%)", fontsize=S.FS_LABEL)

    side_h = [plt.Line2D([], [], marker="o", ls="", ms=6,
                         color=C_SIDE[i], label=lab)
              for i, (lab, _m) in enumerate(SIDES)]
    mode_h = [plt.Line2D([], [], marker=MODE_MARKER[m], ls="", ms=5.5,
                         color="#555555", label=m) for m in MODES]
    leg = fig.legend(handles=side_h, loc="lower center", ncol=3, frameon=False,
                     bbox_to_anchor=(0.5, -0.012))
    fig.add_artist(leg)
    fig.legend(handles=mode_h, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.055), handletextpad=0.35,
               columnspacing=1.1, fontsize=S.FS_VALUE)

    # No banner, no explanatory block, no provenance footnote. Which points are
    # non-dominated, why the panels do not share an x axis, and where the cost
    # estimand comes from are all caption material; drawing them here forced the
    # reader through three paragraphs before reaching the first data point.
    # Reserve a strip at the foot for the shared key so it never lands on the
    # bottom row's axis label.
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=220, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"wrote {a.out}.png / .pdf   (non-dominated per cell: {fronts})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
