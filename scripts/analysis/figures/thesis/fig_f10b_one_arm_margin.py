#!/usr/bin/env python3
"""Thesis F10b — one extra representation vs one extra rerun, at equal arm count.

This is the like-for-like comparison F8 deliberately refuses to make. Adding one
arm to a single-mode baseline raises the oracle ceiling by

    |{added arm solves} \\ {baseline solves}| / n

and that functional is identical whether the added arm is a DIFFERENT
representation or a RERUN of the same one — so at one-arm margin the two are
directly comparable (noise_floor_inventory.md §2). This is the defuse the A2
attack asked for; the answer is that they are indistinguishable.

Two honesty constraints are drawn rather than written:

1. Only two cells carry a measured rerun floor. The other six are shown with
   their representation gain and an explicit "no floor measured" marker, never
   with a borrowed band.
2. "Inside the band" is not the same as "inside the noise". §1b shows the band's
   own edge sits within one standard deviation of itself, so an effect needs
   ~3.8-4.2pp before a single rerun would be unlikely to produce it. That
   threshold is derived from ONE cell and is labelled as such.

Numbers are parsed from the source markdown, never hardcoded.

Output: final_dissertation/figures/fig_f10b_one_arm_margin.{png,pdf}
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "docs/analysis/cross_sites/noise_floor_inventory.md"
OUT = ROOT / "final_dissertation/figures/fig_f10b_one_arm_margin"

C_REP = "#0072B2"     # a different representation
C_RERUN = "#D55E00"   # a rerun of the same one
C_THRESH = "#333333"

# §2 one-arm table
RE_ARM = re.compile(
    r"\|\s*(B\d)\s*·\s*([\w-]+)\s*\(n=(\d+)[^)]*\)\s*\|\s*(\w+)\s*@\s*([\d.]+)%"
    r"\s*\|\s*\*\*([\d.]+)pp\*\*\s*\((\w+)\)\s*\|\s*([^|]*)\|")
# §1b spread table -> one-sided 95% column
RE_SPREAD = re.compile(
    r"\|\s*`(B\d[.\w-]+)`\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)pp\s*\|\s*"
    r"\*\*([\d.]+)pp\*\*\s*\|\s*([\d.]+)pp\s*\|")
RE_BAND = re.compile(r"([\d.]+)\s*[–-]\s*([\d.]+)pp")


def parse(src: Path):
    text = src.read_text(encoding="utf-8")

    rows = []
    for model, site, n, bmode, bsr, gain, gmode, floor_cell in RE_ARM.findall(text):
        m = RE_BAND.search(floor_cell)
        rows.append({"cell": f"{site} · {model}", "n": int(n),
                     "best_mode": bmode, "best_sr": float(bsr),
                     "gain": float(gain), "gain_mode": gmode,
                     "floor": (float(m.group(1)), float(m.group(2))) if m else None})
    spreads = [float(s[5]) for s in RE_SPREAD.findall(text)]
    if len(rows) < 6 or not spreads:
        raise SystemExit(f"{src}: parsed {len(rows)} arm rows / {len(spreads)} "
                         "spread rows — refusing to plot a partial figure")
    rows.sort(key=lambda r: (r["floor"] is None, -r["gain"]))
    return rows, (min(spreads), max(spreads))


def build(ax, rows, thresh):
    tlo, thi = thresh
    y = list(range(len(rows)))[::-1]

    ax.axvspan(tlo, thi, color=C_THRESH, alpha=0.10, zorder=0, lw=0)
    ax.axvline(tlo, color=C_THRESH, lw=1.0, ls=":", zorder=1)
    ax.axvline(thi, color=C_THRESH, lw=1.0, ls=":", zorder=1)

    for yi, r in zip(y, rows):
        if r["floor"]:
            lo, hi = r["floor"]
            ax.plot([lo, hi], [yi, yi], color=C_RERUN, lw=9, alpha=0.30,
                    zorder=2, solid_capstyle="butt")
            ax.plot([lo, lo], [yi - 0.19, yi + 0.19], color=C_RERUN, lw=2,
                    zorder=3)
            ax.plot([hi, hi], [yi - 0.19, yi + 0.19], color=C_RERUN, lw=2,
                    zorder=3)
            ax.text(hi + 0.25, yi - 0.30, f"rerun {lo:.2f}–{hi:.2f}pp",
                    va="center", fontsize=7.4, color=C_RERUN)
        ax.scatter([r["gain"]], [yi], s=92, color=C_REP, zorder=5)
        ax.text(r["gain"] + 0.22, yi + 0.28, f"{r['gain']:.2f}pp "
                f"(+{r['gain_mode']})", va="center", fontsize=8.2, color=C_REP,
                fontweight="bold")

    # "no floor" belongs in the tick label, not the plotting area — at small
    # gains the marker sits exactly where such a note would go.
    ax.set_yticks(y, [f"{r['cell']}\nbest: {r['best_mode']} @ {r['best_sr']:.1f}%"
                      + ("" if r["floor"] else "\n(no rerun floor measured)")
                      for r in rows], fontsize=8.4)
    ax.set_xlim(-0.3, max(max(r["gain"] for r in rows),
                          max((r["floor"][1] for r in rows if r["floor"]),
                              default=0)) + 3.4)
    ax.set_ylim(-0.8, len(rows) - 0.15)
    ax.set_xlabel("ceiling gain from adding ONE arm to the best single mode  [pp]",
                  fontsize=9)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#F1F1F1", lw=0.8)
    ax.set_axisbelow(True)

    ax.scatter([], [], s=92, color=C_REP,
               label="+1 arm = a DIFFERENT representation")
    ax.plot([], [], color=C_RERUN, lw=9, alpha=0.30,
            label="+1 arm = a RERUN of the same representation (measured)")
    ax.plot([], [], color=C_THRESH, lw=1.0, ls=":",
            label=f"{tlo:.2f}–{thi:.2f}pp: above this a single rerun is unlikely "
                  "to produce it (one cell)")
    ax.legend(loc="lower right", frameon=False, fontsize=8.2,
              handletextpad=0.6, borderpad=0.2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    rows, thresh = parse(a.src)
    with_floor = [r for r in rows if r["floor"]]
    inside = [r for r in with_floor
              if r["floor"][0] <= r["gain"] <= r["floor"][1]]

    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    build(ax, rows, thresh)
    ax.set_title("At equal arm count, a different representation is worth no more "
                 "than a rerun",
                 fontsize=11, fontweight="bold", loc="left", pad=42)
    ax.text(0.0, 1.015,
            f"In {len(inside)} of the {len(with_floor)} cells that carry a "
            "measured floor, the representation arm lands INSIDE the rerun band. "
            "The other six cells have no floor, so\nnothing is claimed for them — "
            "a band is never borrowed across cells. This is the like-for-like "
            "comparison; the five-arm ceiling is in F8.",
            transform=ax.transAxes, fontsize=8.4, color="#444444",
            linespacing=1.5, va="bottom")
    fig.text(0.012, 0.005,
             "Source: noise_floor_inventory.md §2 (one-arm margin) + §1b "
             "(null spread). ⚠️ The WA row pools five modes over ten shared "
             "tasks and is not a floor for its own baseline arm — see the source.",
             fontsize=7.4, color="#666666")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=220, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"wrote {a.out}.png / .pdf   ({len(rows)} cells, {len(with_floor)} "
          f"with floor, {len(inside)} inside band, threshold "
          f"{thresh[0]:.2f}-{thresh[1]:.2f}pp)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
