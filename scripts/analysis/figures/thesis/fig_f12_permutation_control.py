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

Run via ``make thesis-figures``, not on its own: this script writes only the
working tree, and the copy LaTeX embeds is refreshed by that target.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as S  # noqa: E402
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
        tag = f"p={r['p']:.4g}" if r["reject"] else f"p={r['p']:.3f}"
        ax.text(max(r["obs"], r["null"]) + 0.7, yi, tag, va="center",
                fontsize=S.FS_VALUE, color=C_WIN if r["reject"] else C_MUTE,
                fontweight="bold" if r["reject"] else "normal")
    ax.set_yticks(y, [r["name"] for r in rows], fontsize=S.FS_LABEL)
    ax.set_xlim(-1.0, max(r["obs"] for r in rows) + 12)
    ax.set_ylim(-0.75, len(rows) - 0.25)
    ax.set_xlabel("SR-lossless cost saving  [%]")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[Patch(color=C_OBS, label="real labels"),
                       Patch(color=C_NULL, label="labels destroyed"),
                       Patch(color=C_WIN, label="survives Holm")],
              loc="lower right", frameon=False, handletextpad=0.6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    rows, m, b = load(SRC)
    n_rej = sum(1 for r in rows if r["reject"])
    reproduced = [r for r in rows
                  if r["obs"] > 0.05 and r["null"] >= 0.8 * r["obs"]]

    S.apply()
    fig, ax = plt.subplots(figsize=(S.PRINT_W_IN, 3.3))
    build(ax, rows)
    # Permutation count, the permuted unit, the p estimator and the reading
    # that surviving the null is necessary but not sufficient are caption text.

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
