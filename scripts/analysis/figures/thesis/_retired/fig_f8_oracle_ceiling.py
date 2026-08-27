#!/usr/bin/env python3
"""Thesis F8 — the oracle ceiling, read next to what repetition alone buys (C1).

`noise_floor_inventory.md` §3 ① lets the ceiling claim stand but attaches a
condition to it: "the headline needs the rerun baseline printed next to it".
This figure is that requirement made structural — the bars are the ceiling, the
band is what one rerun of the SAME arm already delivers.

The same source refuses one thing explicitly, and so does this figure:

    "the 6-mode ceiling gain (five arms added) is NOT comparable to a one-rerun
     floor and is reported separately, labelled with its arm count"

So the band is NOT subtracted from the bars, and both carry their arm count in
the label. Drawing a 5-arm gain minus a 1-arm floor would be exactly the
arithmetic the analysis forbids; showing them side by side, labelled, is what it
asks for. The one-arm-margin comparison that IS legitimate lives in F10b.

Numbers are parsed from the source markdown, never hardcoded.

Output: final_dissertation/figures/fig_f8_oracle_ceiling.{png,pdf}
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
OUT = ROOT / "final_dissertation/figures/fig_f8_oracle_ceiling"

C_BAR = "#0072B2"
C_BAND = "#D55E00"
C_BEST = "#999999"

# §2 five-arm table: | B0 · VWA-cls (n=224) | 27.23% | 43.30% | 16.07pp |
RE_GAIN = re.compile(
    r"\|\s*(B\d)\s*·\s*([\w-]+)\s*\(n=(\d+)[^)]*\)\s*\|\s*([\d.]+)%\s*\|\s*"
    r"([\d.]+)%\s*\|\s*([\d.]+)pp\s*\|")
# §1 floors table: | `B0.cls.dom` | scope text | 224 | **7.14pp** | **4.91pp** | 12.05% |
RE_FLOOR = re.compile(
    r"\|\s*`(B\d[.\w-]+)`[^|]*\|\s*[^|]*[A-Za-z][^|]*\|\s*(\d+)\s*\|\s*"
    r"\*\*([\d.]+)pp\*\*\s*\|\s*\*\*([\d.]+)pp\*\*\s*\|\s*([\d.]+)%\s*\|")


def parse(src: Path):
    text = src.read_text(encoding="utf-8")

    gains = []
    for model, site, n, best, oracle, gain in RE_GAIN.findall(text):
        gains.append({"cell": f"{site} · {model}", "n": int(n),
                      "best": float(best), "oracle": float(oracle),
                      "gain": float(gain)})
    floors = [float(a) for _p, _n, a, b, _d in RE_FLOOR.findall(text)]
    floors += [float(b) for _p, _n, _a, b, _d in RE_FLOOR.findall(text)]

    if len(gains) < 6 or not floors:
        raise SystemExit(f"{src}: parsed {len(gains)} gains / {len(floors)} "
                         "floor values — refusing to plot a partial figure")
    gains.sort(key=lambda g: g["gain"], reverse=True)
    return gains, (min(floors), max(floors))


def build(ax, gains, band):
    lo, hi = band
    y = list(range(len(gains)))[::-1]

    ax.axvspan(lo, hi, color=C_BAND, alpha=0.13, zorder=0, lw=0)
    ax.axvline(lo, color=C_BAND, lw=1.2, ls="--", zorder=1)
    ax.axvline(hi, color=C_BAND, lw=1.2, ls="--", zorder=1)

    for yi, g in zip(y, gains):
        ax.barh(yi, g["gain"], height=0.52, color=C_BAR, zorder=3)
        ax.text(g["gain"] + 0.28, yi, f"{g['gain']:.2f}pp", va="center",
                fontsize=8.8, color=C_BAR, fontweight="bold")
        ax.text(-0.35, yi, f"best single {g['best']:.1f}%  →  oracle "
                           f"{g['oracle']:.1f}%", va="center", ha="right",
                fontsize=7.6, color=C_BEST)

    ax.set_yticks(y, [g["cell"] for g in gains], fontsize=9.2)
    ax.set_xlim(-9.2, max(g["gain"] for g in gains) + 2.6)
    ax.set_ylim(-0.75, len(gains) + 0.55)
    ax.set_xlabel("success-rate gain of the 6-mode oracle over the best single "
                  "fixed mode  [pp]", fontsize=9)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#EFEFEF", lw=0.8)
    ax.set_axisbelow(True)
    # the negative half of the x range is only a text gutter, not data
    ax.set_xticks([t for t in ax.get_xticks() if t >= 0])

    ax.text((lo + hi) / 2, len(gains) + 0.45,
            f"one rerun of the\nSAME arm buys\n{lo:.1f}–{hi:.1f}pp",
            ha="center", va="top", fontsize=8.4, color=C_BAND,
            fontweight="bold", linespacing=1.5, zorder=4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    gains, band = parse(a.src)

    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    build(ax, gains, band)
    ax.set_title("The ceiling is real — and it is read next to what repetition "
                 "alone delivers",
                 fontsize=11, fontweight="bold", loc="left", pad=40)
    ax.text(0.0, 1.015,
            "⚠️  The bars add FIVE arms; the band adds ONE. They are not "
            "subtractable, and no arithmetic is done across them here — the band "
            "is printed because\nthe headline gain should never be read without "
            "it. The like-for-like one-arm comparison is in F10b.",
            transform=ax.transAxes, fontsize=8.4, color="#444444",
            linespacing=1.5, va="bottom")
    fig.text(0.012, 0.005,
             f"Source: noise_floor_inventory.md (§1 floors, §2 five-arm gains). "
             f"{len(gains)} cells across two benchmarks. Oracle is retrospective "
             "and not deployable.",
             fontsize=7.6, color="#666666")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=220, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"wrote {a.out}.png / .pdf   ({len(gains)} cells, "
          f"rerun band {band[0]:.2f}-{band[1]:.2f}pp)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
