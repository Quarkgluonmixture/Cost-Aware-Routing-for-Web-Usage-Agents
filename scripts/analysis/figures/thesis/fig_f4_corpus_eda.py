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
SRC = ROOT / "docs/analysis/benchmark_eda/corpus_eda.json"
OUT = ROOT / "final_dissertation/figures/fig_f4_corpus_eda"

C_SCORED = "#0072B2"
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
    axes = fig.subplots(2, 2)
    y = list(range(len(rows)))[::-1]
    labels = [r["label"] for r in rows]

    # ---- A: run set vs scored set ----
    ax = axes[0][0]
    for yi, r in zip(y, rows):
        drop = r["n_tasks"] - r["n_scored"]
        ax.barh(yi, r["n_scored"], height=0.6, color=C_SCORED, zorder=3)
        ax.barh(yi, drop, left=r["n_scored"], height=0.6, color=C_DROP, zorder=3)
        ax.text(r["n_tasks"] + 8, yi, f"{r['n_scored']}", va="center",
                fontsize=8.4, color=C_SCORED, fontweight="bold")
        if drop:
            ax.text(r["n_tasks"] + 46, yi, f"(−{drop})", va="center",
                    fontsize=7.6, color=C_MUTE)
    ax.set_yticks(y, labels, fontsize=8.6)
    ax.set_xlim(0, max(r["n_tasks"] for r in rows) * 1.24)
    ax.set_title("A   Tasks in the corpus, and tasks actually scored",
                 fontsize=9.8, fontweight="bold", loc="left", pad=8)
    ax.set_xlabel("tasks", fontsize=8)
    ax.legend(handles=[Patch(color=C_SCORED, label="scored"),
                       Patch(color=C_DROP, label="excluded (N/A or protocol)")],
              loc="lower right", frameon=False, fontsize=7.6)

    # ---- B: annotations the corpus ships with ----
    ax = axes[0][1]
    for yi, r in zip(y, rows):
        ref = 100.0 * r["with_reference_image"] / r["n_tasks"]
        has_diff = r.get("reasoning_difficulty") is not None
        ax.barh(yi + 0.17, ref, height=0.32, color=C_YES if ref else C_NO,
                zorder=3)
        ax.text(max(ref, 0) + 1.6, yi + 0.17,
                f"{r['with_reference_image']}/{r['n_tasks']} ref-image",
                va="center", fontsize=7.4,
                color=C_YES if ref else C_NO)
        ax.barh(yi - 0.19, 100 if has_diff else 0, height=0.32,
                color=C_YES if has_diff else C_NO, zorder=3, alpha=0.55)
        ax.text(1.6 if not has_diff else 101.6, yi - 0.19,
                "difficulty label" if has_diff else "no difficulty label",
                va="center", fontsize=7.4, color=C_YES if has_diff else C_NO)
    ax.set_yticks(y, labels, fontsize=8.6)
    ax.set_xlim(0, 168)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_title("B   Annotations the corpus ships with  (a deployment has neither)",
                 fontsize=9.8, fontweight="bold", loc="left", pad=8)
    ax.set_xlabel("% of tasks", fontsize=8)

    # ---- C: what success is measured by ----
    ax = axes[1][0]
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
                        fontsize=7.0, color="white", fontweight="bold")
            left += v
    ax.set_yticks(y, labels, fontsize=8.6)
    ax.set_xlim(0, 100)
    ax.set_title("C   What counts as success", fontsize=9.8, fontweight="bold",
                 loc="left", pad=8)
    ax.set_xlabel("% of tasks", fontsize=8)
    ax.legend(handles=[Patch(color=EVAL_C.get(k, "#999999"), label=k)
                       for k in kinds],
              loc="lower center", bbox_to_anchor=(0.5, -0.60), frameon=False,
              fontsize=7.4, ncol=4, handletextpad=0.4, columnspacing=1.1)

    # ---- D: instruction length ----
    ax = axes[1][1]
    for yi, r in zip(y, rows):
        w = r["intent_words"]
        ax.plot([w["p10"], w["p90"]], [yi, yi], color="#BBBBBB", lw=5,
                solid_capstyle="round", zorder=2)
        ax.scatter([w["median"]], [yi], s=64, color=C_SCORED, zorder=4)
        ax.text(w["p90"] + 1.2, yi, f"{w['median']:.0f}", va="center",
                fontsize=8.2, color=C_SCORED, fontweight="bold")
    ax.set_yticks(y, labels, fontsize=8.6)
    ax.set_xlim(0, max(r["intent_words"]["p90"] for r in rows) + 8)
    ax.set_title("D   Instruction length  (median, with 10–90th percentile)",
                 fontsize=9.8, fontweight="bold", loc="left", pad=8)
    ax.set_xlabel("words", fontsize=8)

    for row in axes:
        for ax in row:
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.tick_params(axis="y", length=0)
            ax.tick_params(axis="x", labelsize=7.6)
            ax.grid(axis="x", color="#F1F1F1", lw=0.8)
            ax.set_axisbelow(True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    rows = load(a.src)
    defects = [(r["label"], d) for r in rows for d in r.get("corpus_defects", [])]
    no_ref = [r["label"] for r in rows if r["with_reference_image"] == 0]

    fig = plt.figure(figsize=(13.0, 7.4))
    build(fig, rows)
    fig.suptitle("Six corpora, and the places where they are not interchangeable",
                 fontsize=12, fontweight="bold", x=0.008, ha="left", y=0.985)
    fig.text(0.008, 0.945,
             f"{', '.join(no_ref)} ship no reference image and no difficulty "
             "label. That is a difference in task SPECIFICATION — it does not by "
             "itself mean those tasks need less visual grounding.",
             fontsize=8.4, color="#B34700")
    foot = ("Source: docs/analysis/benchmark_eda/corpus_eda.json. "
            "Panel A's two numbers are why both run-set and scored-set counts "
            "appear in this thesis; only the scored set is ever used for a rate.")
    if defects:
        foot += ("\nCorpus defects found while building this figure: "
                 + "; ".join(f"{lab} task {d['task_id']} {d['field']}="
                             f"“{d['value']}”" for lab, d in defects)
                 + " — typos in the benchmark's own annotations, left uncorrected "
                   "and excluded from any difficulty tally.")
    fig.text(0.008, 0.012, foot, fontsize=7.0, color="#888888", linespacing=1.55)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.885, bottom=0.20,
                        hspace=0.62, wspace=0.30)

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
