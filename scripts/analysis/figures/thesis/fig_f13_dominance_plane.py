#!/usr/bin/env python3
"""Thesis F13 — baseline-normalised dominance plane (C4 headline).

Form follows the P4 search recommendation (negative_result_figure_design.md
§3.1-§5), which explicitly rules out the obvious alternative: eight separate
per-cell Pareto panels make the reader look eight times before they can conclude
"none of them wins". Instead every cell is normalised so that the policy a router
must beat sits at the origin:

    x = log2(cost / cost_always_cheapest)      y = SR - SR_always_cheapest  [pp]

log-ratio on x because baseline cost differs by cell; percentage POINTS on y
because base SR can be ~2%, where a +1pp move would read as "+50%" in relative
terms and badly overstate itself (P4 §4.1).

The quadrant x<=0, y>=0 is exactly "Pareto-dominates the free fixed policy". The
claim `0 of 8` then becomes a visual fact: no learned point lies in it. The
oracle triage points ARE drawn, and they do land in it — which is the honest
contrast (the headroom exists; the learned policy cannot reach it) and also a
reminder that the oracle is retrospective, not deployable.

Numbers are read from the analysis JSON, never hardcoded.

Output: final_dissertation/figures/fig_f13_dominance_plane.{png,pdf}
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as S  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "docs/analysis/cross_sites/router_triage_learnability_with_wa.json"
OUT = ROOT / "final_dissertation/figures/fig_f13_dominance_plane"

C_NESTED = "#D55E00"   # what a deployment actually gets
C_LOSSLESS = "#E8A87C"  # in-sample optimistic variant
C_ORACLE = "#009E73"   # retrospective upper bound
C_WIN = "#009E73"

PRETTY = {"classifieds": "cls", "reddit": "red", "wa_reddit": "WA-red"}


def load(src: Path):
    d = json.loads(src.read_text(encoding="utf-8"))
    cells = d["cells"]
    if not cells:
        raise SystemExit(f"{src}: no cells")
    out = []
    for c in cells:
        ac = c.get("always_cheapest")
        if not ac or not ac.get("mean_cost"):
            raise SystemExit(f"{src}: {c['site']} missing always_cheapest — "
                             "refusing to normalise against a guess")
        rec = {"name": f"{PRETTY.get(c['site'], c['site'])}·{c['baseline_model']}",
               "n": c["n"], "pts": {}}
        for key in ("learned_nested_honest", "learned_lossless", "oracle_triage"):
            p = c.get(key)
            if not p:
                continue
            rec["pts"][key] = (math.log2(p["mean_cost"] / ac["mean_cost"]),
                               p["sr_pct"] - ac["sr_pct"])
        out.append(rec)
    return out, d.get("protocol", {})


def build(ax, rows):
    xs = [x for r in rows for x, _ in r["pts"].values()]
    ys = [y for r in rows for _, y in r["pts"].values()]
    xlo, xhi = min(xs + [0]) - 0.12, max(xs + [0]) + 0.12
    ylo, yhi = min(ys + [0]) - 1.2, max(ys + [0]) + 1.6

    # the win region
    ax.add_patch(Rectangle((xlo, 0), -xlo, yhi, facecolor=C_WIN, alpha=0.075,
                           zorder=0, lw=0))
    ax.text(xlo + (-xlo) / 2, yhi * 0.94,
            "Pareto-dominates\nthe free fixed policy\n(cheaper AND no worse)",
            ha="center", va="top", fontsize=S.FS_LABEL, color=C_WIN,
            fontweight="bold", linespacing=1.5, zorder=2)

    ax.axhline(0, color="#999999", lw=1.0, zorder=1)
    ax.axvline(0, color="#999999", lw=1.0, zorder=1)
    ax.scatter([0], [0], s=110, marker="*", color="#000000", zorder=6)
    ax.annotate("always-cheapest\n(costs nothing to implement)", (0, 0),
                textcoords="offset points", xytext=(-6, -34), ha="right",
                fontsize=S.FS_LABEL, color="#333333", linespacing=1.4)

    for r in rows:
        if "oracle_triage" in r["pts"]:
            x, y = r["pts"]["oracle_triage"]
            ax.scatter([x], [y], s=58, marker="s", facecolor="none",
                       edgecolor=C_ORACLE, lw=1.5, zorder=4)
        if "learned_lossless" in r["pts"]:
            x, y = r["pts"]["learned_lossless"]
            ax.scatter([x], [y], s=34, marker="^", color=C_LOSSLESS, zorder=4)
        if "learned_nested_honest" in r["pts"]:
            x, y = r["pts"]["learned_nested_honest"]
            ax.scatter([x], [y], s=72, color=C_NESTED, zorder=5)
            ax.annotate(r["name"], (x, y), textcoords="offset points",
                        xytext=(7, 4), fontsize=S.FS_LABEL, color=C_NESTED)

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("$\\log_2(\\mathrm{cost}/\\mathrm{cost}_{\\mathrm{cheapest}})$")
    ax.set_ylabel("success rate over always-cheapest  [pp]")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax.scatter([], [], s=72, color=C_NESTED, label="learned, nested")
    ax.scatter([], [], s=34, marker="^", color=C_LOSSLESS,
               label="learned, in-sample")
    ax.scatter([], [], s=58, marker="s", facecolor="none", edgecolor=C_ORACLE,
               lw=1.5, label="oracle")
    ax.legend(loc="lower right", frameon=False, handletextpad=0.4,
              borderpad=0.2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    rows, proto = load(a.src)
    n_win = sum(1 for r in rows
                if (p := r["pts"].get("learned_nested_honest")) and p[0] <= 0
                and p[1] >= 0)
    # How many oracle points reach the win region? If almost none do, the negative
    # result is partly a property of the comparator and must be said out loud.
    n_or = sum(1 for r in rows
               if (p := r["pts"].get("oracle_triage")) and p[0] <= 0 and p[1] >= 0)
    S.apply()
    fig, ax = plt.subplots(figsize=(S.PRINT_W_IN, 4.2))
    build(ax, rows)
    # How many cells reach the win region, how many oracle points do, and the
    # protocol constants are reported in the caption and in the text; the plot
    # itself only has to make the emptiness of the win region visible.
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=220, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"wrote {a.out}.png / .pdf   ({len(rows)} cells, "
          f"{n_win} in win region)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
