#!/usr/bin/env python3
"""Thesis F2 — literature map: what each cluster optimises, and what is left over.

Guide §8.1 rejects the chronological survey ("Paper A did X. Paper B did Y.") and
asks related work to DERIVE the gap. So this map is organised by decision
variable, not by year: each cluster is labelled with the thing it chooses, and
the centre states the choice none of them makes.

Cluster membership and paper titles are parsed from the P2 search result, which
was verified paper-by-paper against the arXiv API (27 IDs, 0 mismatches — see
search_results/VERIFICATION.md). The one-line characterisation of each cluster
is MY summary of that column, not a parsed field, and is marked as such here so
a later reader does not mistake it for sourced text.

Output: final_dissertation/figures/fig_f2_literature_map.{png,pdf}
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "final_dissertation/search_results/web_agent_representation_routing_related_work.md"
OUT = ROOT / "final_dissertation/figures/fig_f2_literature_map"

C_CLUSTER = "#0072B2"
C_GAP = "#D55E00"
C_MUTE = "#666666"

# Author-written synthesis of the "what it optimises" column, one line per
# cluster. Not parsed — kept here so the provenance stays honest.
CLUSTERS = {
    "1": ("Web / GUI agent page representations",
          "chooses WHICH ELEMENTS survive into the prompt —\n"
          "the representation is a fixed configuration, not a per-step decision"),
    "2": ("Model & modality routing, cascades",
          "chooses WHICH MODEL answers —\nthe input representation is held fixed"),
    "3": ("Confidence-based deferral, selective prediction",
          "chooses WHETHER TO ANSWER —\nthe signal usually needs the full inference first"),
    "4": ("Cost-aware / adaptive inference",
          "chooses HOW MUCH COMPUTE to spend —\ndepth or width, not what the model sees"),
}
POS = {"1": (0.5, 76.0), "2": (50.5, 76.0), "3": (0.5, 8.0), "4": (50.5, 8.0)}
BW, BH = 49.0, 21.0


def parse(src: Path):
    out: dict[str, list[tuple[str, str]]] = {k: [] for k in CLUSTERS}
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        col = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(col) < 5 or col[0] not in out:
            continue
        title = re.sub(r"[*`]", "", col[1])
        short = re.split(r"[::]", title)[0].strip()
        short = re.sub(r"\s*\(.*?\)\s*", "", short)
        if len(short) > 22:
            short = short[:21].rstrip() + "…"
        out[col[0]].append((short, col[3]))
    missing = [k for k, v in out.items() if len(v) < 3]
    if missing:
        raise SystemExit(f"{src}: clusters {missing} under-populated — "
                         "refusing to plot a partial map")
    return out


def build(ax, papers):
    ax.set_xlim(0, 100)
    ax.set_ylim(4, 99)
    ax.axis("off")

    for key, (name, what) in CLUSTERS.items():
        x, y = POS[key]
        n = len(papers[key])
        ax.add_patch(FancyBboxPatch((x, y), BW, BH,
                                    boxstyle="round,pad=0.4,rounding_size=1.1",
                                    ec=C_CLUSTER, fc="#F4F9FD", lw=1.5))
        ax.text(x + 1.9, y + BH - 3.4, name, fontsize=9.6, fontweight="bold",
                color=C_CLUSTER, va="center")
        ax.text(x + BW - 1.9, y + BH - 3.4, f"{n} papers", fontsize=8.4,
                color=C_CLUSTER, va="center", ha="right")
        ax.text(x + 1.9, y + BH - 8.6, what, fontsize=8.2, color="#333333",
                va="center", linespacing=1.5)
        names = ", ".join(t for t, _ in papers[key][:3])
        ax.text(x + 1.9, y + 2.6, f"e.g. {names}", fontsize=7.0, color=C_MUTE,
                va="center")

    # the gap at the centre
    gx, gy, gw, gh = 12.0, 41.0, 76.0, 21.6
    ax.add_patch(FancyBboxPatch((gx, gy), gw, gh,
                                boxstyle="round,pad=0.5,rounding_size=1.3",
                                ec=C_GAP, fc="#FFF6F0", lw=2.4, zorder=4))
    ax.text(gx + gw / 2, gy + gh - 5.0,
            "None of them chooses the REPRESENTATION, per step, in advance",
            ha="center", fontsize=10.6, fontweight="bold", color=C_GAP, zorder=5)
    ax.text(gx + gw / 2, gy + 7.6,
            "Given the same agent on the same step: which page representation "
            "should it receive —\nand can that be decided from signals available "
            "BEFORE paying for the expensive one?",
            ha="center", fontsize=9.0, color="#333333", zorder=5, linespacing=1.6)

    for key in CLUSTERS:
        x, y = POS[key]
        cx = x + BW / 2
        y0 = y if y > 50 else y + BH
        y1 = gy + gh + 0.4 if y > 50 else gy - 0.4  # noqa: E501
        ax.add_patch(FancyArrowPatch((cx, y0), (cx, y1), arrowstyle="-|>",
                                     mutation_scale=13, lw=1.3, color="#AAAAAA",
                                     zorder=2, shrinkA=0, shrinkB=0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    papers = parse(a.src)
    total = sum(len(v) for v in papers.values())

    fig, ax = plt.subplots(figsize=(12.2, 6.4))
    build(ax, papers)
    ax.set_title("Four literatures choose four different things — "
                 "none of them chooses this one",
                 fontsize=12, fontweight="bold", loc="left", pad=16)
    fig.text(0.012, 0.012,
             f"{total} papers, clustered by decision variable rather than by year "
             "(Guide §8.1). Every arXiv ID was checked against the arXiv API "
             "before use (0 mismatches; see search_results/VERIFICATION.md).\n"
             "The one-line characterisation under each cluster heading is the "
             "author's synthesis of that cluster, not a quotation.",
             fontsize=7.0, color="#888888", linespacing=1.55)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=220, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"wrote {a.out}.png / .pdf   ({total} papers: "
          + ", ".join(f"c{k}={len(v)}" for k, v in sorted(papers.items())) + ")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
