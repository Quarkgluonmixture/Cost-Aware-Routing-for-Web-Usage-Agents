#!/usr/bin/env python3
"""Thesis F4 — what the six corpora actually are, and where they differ (rubric #10).

Four panels, chosen so that each one answers a question the reader would
otherwise have to take on trust:

  A  how many tasks are there, and how many are actually scored
     (the run set vs scored set double accounting that explains why 205 and 203,
      or 435 and 432, both appear in this project's tables)
  B  which annotations each corpus ships with
     — this is the panel that matters most downstream: WebArena carries neither
       reference images nor a difficulty label, and F11 shows that the difficulty
       label is doing much of the predictive work on VWA
  C  what "success" is even measured by
  D  how long the instructions are

Deliberately NOT claimed here: that a missing reference image means the task
needs no visual grounding. It means the task SPECIFICATION differs — attack A3
forced that downgrade and the panel label keeps it.

Numbers parsed from docs/analysis/benchmark_eda/corpus_eda.json.

Output: final_dissertation/figures/fig_f4_corpus_eda.{png,pdf}

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
SRC = ROOT / "docs/analysis/benchmark_eda/corpus_eda.json"
OUT = ROOT / "final_dissertation/figures/fig_f4_corpus_eda"

C_SCORED = "#0072B2"
C_DIFF = "#7FBFA6"    # difficulty label present; a lighter partner to C_YES
C_DROP = "#C9C9C9"
C_YES = "#009E73"
C_NO = "#D55E00"
C_MUTE = "#777777"
EVAL_C = {"url_match": "#0072B2", "program_html": "#009E73",
          "string_match": "#E69F00", "page_image_query": "#CC79A7"}


def load(src: Path):
    rows = json.loads(src.read_text(encoding="utf-8"))
    if len(rows) < 6:
        raise SystemExit(f"{src}: only {len(rows)} corpora — refusing to plot")
    for r in rows:
        r["label"] = f"{r['benchmark']}-{r['site'].replace('shopping_admin', 'shop-admin').replace('shopping', 'shop').replace('classifieds', 'cls')}"
    return rows


def build(fig, rows):
    axes = fig.subplots(4, 1, sharey=True,
                        gridspec_kw={"height_ratios": [1, 1.3, 1, 1]})
    y = list(range(len(rows)))[::-1]
    labels = [r["label"] for r in rows]

    # ---- A: run set vs scored set ----
    ax = axes[0]
    for yi, r in zip(y, rows):
        drop = r["n_tasks"] - r["n_scored"]
        ax.barh(yi, r["n_scored"], height=0.6, color=C_SCORED, zorder=3)
        ax.barh(yi, drop, left=r["n_scored"], height=0.6, color=C_DROP, zorder=3)
        ax.text(r["n_tasks"] + 8, yi, f"{r['n_scored']}", va="center",
                fontsize=S.FS_LABEL, color=C_SCORED, fontweight="bold")
        if drop:
            ax.text(r["n_tasks"] + 46, yi, f"(−{drop})", va="center",
                    fontsize=S.FS_VALUE, color=C_MUTE)
    ax.set_yticks(y, labels, fontsize=S.FS_LABEL)
    ax.set_xlim(0, max(r["n_tasks"] for r in rows) * 1.24)
    S.panel_label(ax, "A   Scored set")
    ax.set_xlabel("tasks")
    ax.legend(handles=[Patch(color=C_SCORED, label="scored"),
                       Patch(color=C_DROP, label="excluded")],
              loc="lower right", frameon=False)

    # ---- B: annotations the corpus ships with ----
    # Two bars per corpus rather than two lines of text per corpus. The earlier
    # version wrote "0/106 ref-image" and "no difficulty" beside every row, and
    # at six rows in a short strip the two lines collided. A count at the end of
    # the upper bar carries the same information in one glance.
    ax = axes[1]
    for yi, r in zip(y, rows):
        ref = 100.0 * r["with_reference_image"] / r["n_tasks"]
        has_diff = r.get("reasoning_difficulty") is not None
        ax.barh(yi + 0.17, ref, height=0.30, color=C_YES, zorder=3)
        ax.barh(yi - 0.19, 100 if has_diff else 0, height=0.30, color=C_DIFF,
                zorder=3)
        ax.text(max(ref, 0) + 1.8, yi + 0.17,
                f"{r['with_reference_image']}/{r['n_tasks']}", va="center",
                fontsize=S.FS_VALUE, color=C_YES if ref else C_NO,
                fontweight="bold" if not ref else "normal")
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 25, 50, 75, 100])
    S.panel_label(ax, "B   Annotations shipped with the corpus")
    ax.set_xlabel("% of tasks")
    ax.legend(handles=[Patch(color=C_YES, label="reference image"),
                       Patch(color=C_DIFF, label="difficulty label")],
              loc="lower right", frameon=False, ncol=2)

    # ---- C: what success is measured by ----
    ax = axes[2]
    kinds = sorted({k for r in rows for k in r["eval_type_counts"]},
                   key=lambda k: -sum(r["eval_type_counts"].get(k, 0) for r in rows))
    for yi, r in zip(y, rows):
        tot = sum(r["eval_type_counts"].values()) or 1
        left = 0.0
        for k in kinds:
            v = 100.0 * r["eval_type_counts"].get(k, 0) / tot
            if v <= 0:
                continue
            ax.barh(yi, v, left=left, height=0.6,
                    color=EVAL_C.get(k, "#999999"), zorder=3)
            if v >= 13:
                ax.text(left + v / 2, yi, f"{v:.0f}", ha="center", va="center",
                        fontsize=S.FS_VALUE, color="white", fontweight="bold")
            left += v
    ax.set_yticks(y, labels, fontsize=S.FS_LABEL)
    ax.set_xlim(0, 100)
    S.panel_label(ax, "C   Success criterion")
    ax.set_xlabel("% of tasks")
    fig = ax.get_figure()
    fig.legend(handles=[Patch(color=EVAL_C.get(k, "#999999"), label=k)
                        for k in kinds],
               loc="lower center", bbox_to_anchor=(0.5, 0.005), frameon=False,
               ncol=4, handletextpad=0.4, columnspacing=1.1)

    # ---- D: instruction length ----
    ax = axes[3]
    for yi, r in zip(y, rows):
        w = r["intent_words"]
        ax.plot([w["p10"], w["p90"]], [yi, yi], color="#BBBBBB", lw=5,
                solid_capstyle="round", zorder=2)
        ax.scatter([w["median"]], [yi], s=64, color=C_SCORED, zorder=4)
        ax.text(w["p90"] + 1.2, yi, f"{w['median']:.0f}", va="center",
                fontsize=S.FS_LABEL, color=C_SCORED, fontweight="bold")
    ax.set_yticks(y, labels, fontsize=S.FS_LABEL)
    ax.set_xlim(0, max(r["intent_words"]["p90"] for r in rows) + 8)
    S.panel_label(ax, "D   Instruction length")
    ax.set_xlabel("words")

    for ax in axes:
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    rows = load(a.src)
    defects = [(r["label"], d) for r in rows for d in r.get("corpus_defects", [])]
    no_ref = [r["label"] for r in rows if r["with_reference_image"] == 0]

    S.apply()
    fig = plt.figure(figsize=(S.PRINT_W_IN, 6.6))
    build(fig, rows)
    # Which corpora ship no reference image, the warning that a missing
    # annotation is a difference in specification rather than in visual demand,
    # and the two annotation typos found while building this figure are all
    # reported in the caption. They are still computed here and printed below.
    fig.subplots_adjust(left=0.20, right=0.97, top=0.97, bottom=0.11,
                        hspace=0.85)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=220, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"wrote {a.out}.png / .pdf   ({len(rows)} corpora, "
          f"{len(defects)} corpus defects, no-ref: {', '.join(no_ref)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
