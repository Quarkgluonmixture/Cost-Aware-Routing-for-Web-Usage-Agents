#!/usr/bin/env python3
"""Thesis F11 — how much of the "predictability" is the corpus's own answer key.

Two feature sets, fitted on the same cells:

  20 features  VWA-only. Includes `reasoning_difficulty` and `has_reference_image`,
               both of which ship with the VisualWebArena task configs.
  18 features  the matched set used when WebArena is included. Those two columns
               are dropped on EVERY cell, so the six VWA cells are refitted, not
               subsetted.

Dropping them costs AUROC in five of the six VWA cells, and the single strongest
covariate in the 20-feature fits is `reasoning_difficulty` in five of six. The
source file says it plainly: the model "is reading the benchmark's own statement
of how hard the task is, which no deployment has."

That is why this figure answers attack A1 ("the negative result is just
under-sampling") more cheaply than a learning curve would: the issue is not how
many labels there are, it is that the one column carrying stable signal does not
exist at serving time — and, per F4, does not exist in WebArena at all.

Numbers parsed from both analysis markdowns, never hardcoded.

Output: final_dissertation/figures/fig_f11_feature_set_ablation.{png,pdf}
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
SRC20 = ROOT / "docs/analysis/cross_sites/router_triage_learnability.md"
SRC18 = ROOT / "docs/analysis/cross_sites/router_triage_learnability_with_wa.md"
OUT = ROOT / "final_dissertation/figures/fig_f11_feature_set_ablation"

C20 = "#CC79A7"     # includes corpus-shipped annotations
C18 = "#0072B2"     # deployment-faithful subset
C_DROP = "#D55E00"
C_MUTE = "#777777"

# | classifieds·B0 | 224 | 43.3 | **0.726** | 0.711 | +0.015 | `reasoning_difficulty` |
RE_ROW = re.compile(
    r"\|\s*([\w]+)·(B\d)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*\*\*([\d.]+)\*\*\s*\|"
    r"\s*([\d.]+)\s*\|\s*([+-][\d.]+)\s*\|\s*`(\w+)`\s*\|")


def parse(src: Path) -> dict:
    out = {}
    for site, model, n, solv, auroc, best, _delta, feat in RE_ROW.findall(
            src.read_text(encoding="utf-8")):
        out[f"{site}·{model}"] = {"n": int(n), "solv": float(solv),
                                  "auroc": float(auroc), "best": float(best),
                                  "feat": feat}
    return out


def build(ax, a20, a18, order):
    y = list(range(len(order)))[::-1]
    for yi, cell in zip(y, order):
        r18 = a18[cell]
        r20 = a20.get(cell)
        if r20:
            ax.plot([r20["auroc"], r18["auroc"]], [yi, yi], color="#DDDDDD",
                    lw=6, solid_capstyle="round", zorder=1)
            ax.scatter([r20["auroc"]], [yi], s=78, color=C20, zorder=4)
            drop = r18["auroc"] - r20["auroc"]
            ax.add_patch(FancyArrowPatch(
                (r20["auroc"], yi), (r18["auroc"], yi), arrowstyle="-|>",
                mutation_scale=11, lw=1.3, zorder=3, shrinkA=6, shrinkB=6,
                color=C_DROP if drop < 0 else "#009E73"))
            ax.text(min(r20["auroc"], r18["auroc"]) - 0.012, yi,
                    f"{drop:+.3f}", va="center", ha="right", fontsize=7.8,
                    color=C_DROP if drop < 0 else "#009E73", fontweight="bold")
        ax.scatter([r18["auroc"]], [yi], s=78, color=C18, zorder=5)
        note = f"strongest single covariate: {r20['feat']}" if r20 else \
               f"18-feature only  ·  {r18['feat']}"
        ax.text(0.885, yi, note, va="center", fontsize=7.2,
                color=C_MUTE if r20 else "#AAAAAA")

    ax.axvline(0.5, color="#BBBBBB", lw=1.0, ls=":")
    ax.text(0.502, len(order) - 0.45, "chance", fontsize=7.4, color="#999999")
    ax.set_yticks(y, order, fontsize=9.0)
    ax.set_xlim(0.44, 1.10)
    ax.set_ylim(-0.75, len(order) - 0.25)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xlabel("AUROC of the triage label (task-held-out CV)", fontsize=9)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#F1F1F1", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(handles=[
        Patch(color=C20, label="20 features — includes the corpus's own "
                               "difficulty label and reference-image flag"),
        Patch(color=C18, label="18 features — those two dropped on every cell "
                               "(what a deployment could actually use)")],
        loc="lower left", bbox_to_anchor=(0.0, -0.30), frameon=False,
        fontsize=8.0, handletextpad=0.6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    a20, a18 = parse(SRC20), parse(SRC18)
    if len(a18) < 6 or len(a20) < 6:
        raise SystemExit(f"parsed {len(a20)} / {len(a18)} cells — refusing to "
                         "plot a partial figure")
    order = [c for c in a18 if c in a20] + [c for c in a18 if c not in a20]
    drops = [a18[c]["auroc"] - a20[c]["auroc"] for c in a20 if c in a18]
    n_down = sum(1 for x in drops if x < 0)
    modal = max({a20[c]["feat"] for c in a20},
                key=lambda f: sum(1 for c in a20 if a20[c]["feat"] == f))
    n_modal = sum(1 for c in a20 if a20[c]["feat"] == modal)

    fig, ax = plt.subplots(figsize=(11.2, 5.4))
    build(ax, a20, a18, order)
    ax.set_title("Drop the two columns the corpus provides, and the signal "
                 f"falls in {n_down} of {len(drops)} cells",
                 fontsize=11.4, fontweight="bold", loc="left", pad=40)
    ax.text(0.0, 1.012,
            f"`{modal}` is the strongest single covariate in {n_modal} of "
            f"{len(a20)} of the 20-feature fits — it is an annotation shipped "
            "with the task config, not something a router\ncould read at serving "
            "time. WebArena carries neither column at all (see F4), which is why "
            "the matched set exists.",
            transform=ax.transAxes, fontsize=8.4, color="#444444",
            linespacing=1.5, va="bottom")
    fig.text(0.012, 0.005,
             "Sources: router_triage_learnability.md (20-feature, VWA) and "
             "router_triage_learnability_with_wa.md (18-feature, matched). "
             "The 18-feature fits are NOT a subset — every cell was refitted.",
             fontsize=7.0, color="#888888")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=220, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"wrote {a.out}.png / .pdf   ({len(order)} cells, {n_down}/{len(drops)} "
          f"drop, modal feature {modal} in {n_modal}/{len(a20)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
