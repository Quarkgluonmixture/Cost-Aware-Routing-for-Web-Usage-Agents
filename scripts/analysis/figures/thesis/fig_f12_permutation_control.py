#!/usr/bin/env python3
"""Thesis F12 — how much of each cell's saving a signal-free pipeline reproduces.

The saving reported for a learned triage policy is picked by a threshold sweep,
so some of it is selection, not signal. The control is a permutation null: shuffle
the whole task bundle (y, succ, cost) against X, rerun the SAME sweep, and see
what saving survives with the labels destroyed.

Two design details are carried from the analysis rather than re-decided here:

  the permutation unit is the BUNDLE, not y alone. Permuting only y leaves the
  label disconnected from the outcomes that define it, and its error is not even
  one-directional (measured at B=200: cls/B1 0.478->0.503 but red/B2 0.040->0.005).

  p is the plus-one estimator (k+1)/(B+1), because k/B can report exactly 0 for an
  event that simply was not sampled — Phipson & Smyth, "Permutation p-values should
  never be zero".

Holm thresholds are recomputed here from the cell count actually present, which is
the defect that made the prose in this file's source drift (B-1974): m is a runtime
property, never a constant.

Output: final_dissertation/figures/fig_f12_permutation_control.{png,pdf}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "docs/analysis/cross_sites/router_triage_learnability_with_wa.json"
OUT = ROOT / "final_dissertation/figures/fig_f12_permutation_control"

C_OBS = "#0072B2"
C_NULL = "#999999"
C_WIN = "#009E73"
C_MUTE = "#777777"
PRETTY = {"classifieds": "cls", "reddit": "red", "wa_reddit": "WA-red"}


def load(src: Path):
    d = json.loads(src.read_text(encoding="utf-8"))
    cells = d["cells"]
    rows = [{"name": f"{PRETTY.get(c['site'], c['site'])}·{c['baseline_model']}",
             "obs": c["observed_lossless_saving_pct"],
             "null": c["null_shuffle_saving_median_pct"],
             "p": c["null_shuffle_p"], "n": c["n"]} for c in cells]
    if len(rows) < 6:
        raise SystemExit(f"{src}: {len(rows)} cells — refusing to plot partial")
    # Holm, step-down, over exactly the cells present.
    m = len(rows)
    for i, r in enumerate(sorted(rows, key=lambda r: r["p"])):
        r["thresh"] = 0.05 / (m - i)
    stop = False
    for r in sorted(rows, key=lambda r: r["p"]):
        r["reject"] = (not stop) and r["p"] < r["thresh"]
        if not r["reject"]:
            stop = True
    b = cells[0]["n_shuffles"]
    return rows, m, b


def build(ax, rows):
    rows = sorted(rows, key=lambda r: r["obs"], reverse=True)
    y = list(range(len(rows)))[::-1]
    for yi, r in zip(y, rows):
        ax.plot([r["null"], r["obs"]], [yi, yi], color="#E4E4E4", lw=6,
                solid_capstyle="round", zorder=1)
        ax.scatter([r["null"]], [yi], s=62, color=C_NULL, zorder=4)
        ax.scatter([r["obs"]], [yi], s=88,
                   color=C_WIN if r["reject"] else C_OBS, zorder=5)
        tag = (f"p={r['p']:.4g} < {r['thresh']:.4f}  ✓ survives Holm"
               if r["reject"] else f"p={r['p']:.3f}")
        ax.text(max(r["obs"], r["null"]) + 0.7, yi, tag, va="center",
                fontsize=7.6, color=C_WIN if r["reject"] else C_MUTE,
                fontweight="bold" if r["reject"] else "normal")
    ax.set_yticks(y, [r["name"] for r in rows], fontsize=9.0)
    ax.set_xlim(-1.0, max(r["obs"] for r in rows) + 12)
    ax.set_ylim(-0.75, len(rows) - 0.25)
    ax.set_xlabel("SR-lossless cost saving  [%]", fontsize=9)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#F1F1F1", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(handles=[
        Patch(color=C_OBS, label="observed saving (threshold sweep on real labels)"),
        Patch(color=C_NULL, label="median saving with labels destroyed "
                                  "(bundle permutation)"),
        Patch(color=C_WIN, label="survives Holm correction")],
        loc="lower right", frameon=False, fontsize=8.0, handletextpad=0.6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    rows, m, b = load(SRC)
    n_rej = sum(1 for r in rows if r["reject"])
    reproduced = [r for r in rows
                  if r["obs"] > 0.05 and r["null"] >= 0.8 * r["obs"]]

    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    build(ax, rows)
    ax.set_title(f"With the labels destroyed, the sweep still finds most of the "
                 f"saving in {len(reproduced)} of {m} cells",
                 fontsize=11.4, fontweight="bold", loc="left", pad=40)
    ax.text(0.0, 1.012,
            f"Grey = what a signal-free pipeline reproduces. {n_rej} of {m} "
            f"cells survive Holm at α=0.05. Surviving the null is necessary, not "
            "sufficient: it says the saving\nis not an artefact of the sweep, not "
            "that the policy is worth deploying — for that comparison see F13.",
            transform=ax.transAxes, fontsize=8.4, color="#444444",
            linespacing=1.5, va="bottom")
    fig.text(0.012, 0.005,
             f"B={b:,} permutations per cell; unit is the whole task bundle "
             "(y, succ, cost) against X, not y alone. p is the plus-one estimator "
             f"(k+1)/(B+1). Holm computed over the m={m} cells present.",
             fontsize=7.0, color="#888888")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=220, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"wrote {a.out}.png / .pdf   (m={m}, B={b}, {n_rej} survive Holm, "
          f"{len(reproduced)} largely reproduced by the null)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
