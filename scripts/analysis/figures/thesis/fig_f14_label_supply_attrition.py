#!/usr/bin/env python3
"""Thesis F14 — label-supply attrition and the trainability threshold.

Form follows the P4 search recommendation (negative_result_figure_design.md
§13-17): a funnel/Sankey answers "how much is lost", but the takeaway here is
"what is left falls below the threshold for training a classifier at all" — a
threshold-crossing problem, not a flow-composition one. So: connected dots per
cell, absolute counts on the axis, and an explicit threshold line.

One honest deviation from that template. P4 proposes four stages
(All -> Solved -> Labels available -> Trainable), but in this project stages 2
and 3 are the SAME set: a which-mode label exists exactly when some mode solved
the task. That identity is the mechanism, not an accounting detail, so it is
drawn as one stage and labelled, rather than padded into two.

The real threshold is also not on the label total. `router_label_supply_diagnosis`
requires >=2 classes each surviving N_MIN_CLASS_TRAIN=10 in a 5-fold split — so
panel B puts the threshold where it actually bites: on the number of usable
classes. Faking a per-stage task count we do not have would be worse than
showing the two panels the data actually supports.

Numbers are PARSED from the source markdown, never hardcoded — a figure whose
numbers silently drift from their source is the failure mode this thesis has hit
three times (see 实验笔记 §450.3/.4/.10).

Output: final_dissertation/figures/fig_f14_label_supply_attrition.{png,pdf}
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
SRC = ROOT / "docs/analysis/cross_sites/router_label_supply_diagnosis.md"
OUT = ROOT / "final_dissertation/figures/fig_f14_label_supply_attrition"

C_ALL = "#999999"
C_LAB = "#0072B2"
C_NO = "#D55E00"
C_YES = "#009E73"

PRETTY = {
    "B0_classifieds": "classifieds · B0", "B0_reddit": "reddit · B0",
    "B1_classifieds": "classifieds · B1", "B1_reddit": "reddit · B1",
    "B2_classifieds": "classifieds · B2", "B2_reddit": "reddit · B2",
}


def parse(src: Path) -> tuple[list[dict], int]:
    """§1 (universe, labels, solvable) + §2 (surviving classes) + N_MIN."""
    text = src.read_text(encoding="utf-8")

    m = re.search(r"N_MIN_CLASS_TRAIN=(\d+)", text)
    if not m:
        raise SystemExit(f"{src}: N_MIN_CLASS_TRAIN not found — refusing to guess")
    n_min = int(m.group(1))

    rows: dict[str, dict] = {}
    # §1 | cell | scored universe | trainable labels | solvable | classes present |
    for cell, uni, lab, solv, cls in re.findall(
            r"\|\s*(B\d_\w+)\s*\|\s*(\d+)\s*\|\s*\*\*(\d+)\*\*\s*\|\s*"
            r"([\d.]+)%\s*\|\s*(\d+)/(?:\d+)\s*\|", text):
        rows[cell] = {"cell": cell, "universe": int(uni), "labels": int(lab),
                      "solvable": float(solv), "classes": int(cls)}
    # §2 | cell | labels | classes | surviving min-class filter | trainable |
    for cell, _lab, _cls, surv, trainable in re.findall(
            r"\|\s*(B\d_\w+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\([^)]*\)\s*\|"
            r"\s*(?:\*\*)?(yes|no)(?:\*\*)?\s*\|", text):
        if cell in rows:
            rows[cell]["surviving"] = int(surv)
            rows[cell]["trainable"] = (trainable == "yes")

    out = [rows[c] for c in PRETTY if c in rows]
    missing = [c for c in PRETTY if c not in rows]
    bad = [r["cell"] for r in out if "surviving" not in r]
    if missing or bad or len(out) != 6:
        raise SystemExit(f"{src}: parse incomplete (missing={missing}, "
                         f"no-surviving={bad}, n={len(out)}) — refusing to plot "
                         "a partial figure")
    return out, n_min


def build(fig, rows, n_min):
    axA, axB = fig.subplots(1, 2, width_ratios=[2.15, 1.0])
    y = list(range(len(rows)))[::-1]

    # ---------- panel A: the attrition ----------
    for yi, r in zip(y, rows):
        axA.plot([r["labels"], r["universe"]], [yi, yi], color="#C8C8C8", lw=1.6,
                 zorder=1, solid_capstyle="round")
        axA.scatter([r["universe"]], [yi], s=52, color=C_ALL, zorder=3)
        axA.scatter([r["labels"]], [yi], s=76, color=C_LAB, zorder=4)
        axA.text(r["universe"] + 5, yi, f"{r['universe']}", va="center",
                 fontsize=8.4, color="#666666")
        axA.text(r["labels"] - 5, yi + 0.30, f"{r['labels']}", va="center",
                 ha="right", fontsize=9.2, color=C_LAB, fontweight="bold")
        axA.text(r["labels"] - 5, yi - 0.32, f"{r['solvable']:.1f}% solvable",
                 va="center", ha="right", fontsize=7.4, color="#777777")

    axA.set_yticks(y, [PRETTY[r["cell"]] for r in rows], fontsize=9.4)
    axA.set_xlim(-38, 258)
    axA.set_ylim(-0.9, len(rows) - 0.1)
    axA.set_xlabel("tasks", fontsize=9)
    axA.set_title("A.  A which-mode label exists only where some mode succeeded",
                  fontsize=10, fontweight="bold", loc="left", pad=26)
    axA.scatter([], [], s=52, color=C_ALL, label="scored task universe")
    axA.scatter([], [], s=76, color=C_LAB, label="tasks with a which-mode label")
    axA.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), frameon=False,
               fontsize=8.4, ncol=2, handletextpad=0.35, columnspacing=1.4)
    for s in ("top", "right", "left"):
        axA.spines[s].set_visible(False)
    axA.tick_params(axis="y", length=0)
    axA.grid(axis="x", color="#EEEEEE", lw=0.8)
    axA.set_axisbelow(True)

    # the implication that drives the whole chapter
    axA.text(0.5, -0.30,
             "Label Exists  ⇒  Task Solved   —   resplitting cannot manufacture "
             "events, only redistribute them",
             transform=axA.transAxes, ha="center", fontsize=8.8, color=C_LAB,
             style="italic")

    # ---------- panel B: where the threshold actually bites ----------
    axB.axvspan(-0.5, 2, color="#F2F2F2", zorder=0)
    axB.axvline(2, color=C_NO, lw=1.8, ls="--", zorder=2)
    for yi, r in zip(y, rows):
        col = C_YES if r["trainable"] else C_NO
        axB.plot([0, r["surviving"]], [yi, yi], color=col, lw=2.6, alpha=0.30,
                 zorder=2, solid_capstyle="round")
        axB.scatter([r["surviving"]], [yi], s=86, color=col, zorder=4)
        axB.text(r["surviving"] + 0.18, yi, f"{r['surviving']}", va="center",
                 fontsize=9.0, color=col, fontweight="bold")

    axB.set_yticks(y, ["" for _ in rows])
    axB.set_xlim(-0.5, 6.4)
    axB.set_ylim(-0.9, len(rows) - 0.1)
    axB.set_xticks(range(7))
    axB.set_xlabel(f"classes with ≥ {n_min} labelled tasks", fontsize=9)
    axB.set_title("B.  Two classes are needed to discriminate at all",
                  fontsize=10, fontweight="bold", loc="left", pad=26)
    axB.text(0.95, len(rows) - 0.55, "not trainable", fontsize=8.4,
             color=C_NO, ha="center", style="italic")
    axB.text(2.25, len(rows) - 0.55, "trainable", fontsize=8.4, color=C_YES,
             style="italic")
    for s in ("top", "right", "left"):
        axB.spines[s].set_visible(False)
    axB.tick_params(axis="y", length=0)
    axB.grid(axis="x", color="#EEEEEE", lw=0.8)
    axB.set_axisbelow(True)

    n_bad = sum(1 for r in rows if not r["trainable"])
    axB.text(0.5, -0.30,
             f"{n_bad} of {len(rows)} cells never reach it",
             transform=axB.transAxes, ha="center", fontsize=9.4, color=C_NO,
             fontweight="bold")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    rows, n_min = parse(a.src)
    fig = plt.figure(figsize=(12.4, 4.9))
    build(fig, rows, n_min)
    fig.subplots_adjust(left=0.135, right=0.985, top=0.80, bottom=0.20,
                        wspace=0.12)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=220, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"wrote {a.out}.png / .pdf   (parsed {len(rows)} cells, "
          f"N_MIN_CLASS_TRAIN={n_min}, from {a.src.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
