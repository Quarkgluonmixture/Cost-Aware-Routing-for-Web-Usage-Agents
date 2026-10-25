#!/usr/bin/env python3
"""Camera-ready additions to the VLM4RWD paper (deliverables/vlm4rwd/), read from the products.

Every number comes from a product JSON under docs/analysis/cross_sites/; nothing is typed here.
These answer the two reviewer points the submitted version left open (issue_vlm4rwd_reviews_2026-09-29):
the SR-cost trade-off (1MR9 #2, AC) and the coverage of the rerun control (1MR9 #3).

  figures/fig_frontier.pdf  one cell's (cost, SR) plane over the six modes + pooled gain over the
                            mixture frontier (representation_routing_frontier)
  tables/tab_frontier.tex   curve max vs deployable gain, six modes, eight cells pooled
                            (representation_routing_frontier, routing_gain_upper_bounds)
  tables/tab_rerun.tex      every registered same-condition rerun, per cell, next to the gain of one
                            distinct added arm (noise_floor_inventory)

Run with any Python that has matplotlib; regenerate whenever a source product changes.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")   # byte-stable PDFs: a rerun with unchanged data is no diff
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
CS = REPO / "docs/analysis/cross_sites"
DST = REPO / "deliverables/vlm4rwd"

SLOT = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
TEXTW = 5.5   # NeurIPS text width, inches

plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.linewidth": 0.6, "savefig.bbox": "tight", "pdf.fonttype": 42})

# the paper's cell ids (tab42 / Table 1 order) -> how each product keys the same cell
CELLS = ("cls_B0", "cls_B1", "cls_B2", "red_B0", "red_B1", "red_B2", "wa_red_B0", "wa_red_B1")
PAIR_KEY = {"cls": "classifieds", "red": "reddit", "wa_red": "WA-reddit"}
TT = {"cls_B0": r"cls$\cdot$B0", "cls_B1": r"cls$\cdot$B1", "cls_B2": r"cls$\cdot$B2",
      "red_B0": r"red$\cdot$B0", "red_B1": r"red$\cdot$B1", "red_B2": r"red$\cdot$B2",
      "wa_red_B0": r"wa\_red$\cdot$B0", "wa_red_B1": r"wa\_red$\cdot$B1"}
MODE = {"dom": "DOM", "som": "SoM", "vision": "Vision", "ptext": "P-text", "pprompt": "P-prompt", "psom": "P-SoM"}


def _load(name):
    return json.loads((CS / f"{name}.json").read_text(encoding="utf-8"))


def _m(s: str) -> str:
    """Typeset minus signs as math minus."""
    return s.replace("-", "$-$")


def _write(rel: str, lines: list[str]) -> None:
    p = DST / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fig_frontier():
    """(a) cls_B0's (cost, SR) plane: the six fixed modes, their mixture frontier, the two out-of-fold
    router curves; (b) the pooled frontier gain over the normalised budget u with the label-shuffle
    null's pointwise 95th percentile."""
    fr = _load("representation_routing_frontier")
    cell = next(c for c in fr["cells"] if c["cell_id"] == "cls_B0")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(TEXTW, 2.0), gridspec_kw={"wspace": 0.34})

    fm = cell["fixed_modes"]
    hull = [(p["cost"] * 100, p["sr_pct"]) for p in cell["fixed_hull"]]
    xs = [c["cost"] * 100 for k in ("six_head", "triage") for c in cell[k]["curve"]] + \
         [v["cost"] * 100 for v in fm.values()]
    lo, hi = min(xs) - 0.1, max(xs) + 0.1
    hx = [lo] + [h[0] for h in hull] + [hi]
    hy = [hull[0][1]] + [h[1] for h in hull] + [hull[-1][1]]
    ax.fill_between(hx, 0, hy, color=GRID, alpha=0.55, lw=0, zorder=0)
    ax.plot(hx, hy, color=MUTED, lw=1.0, zorder=1, label="fixed modes + mixtures")
    # a threshold sweep is not monotone in cost, so its points are drawn unconnected
    for k, col, mk, lab in (("six_head", SLOT[1], "o", "per-mode heads (sweep)"),
                            ("triage", SLOT[2], "x", "triage (sweep)")):
        ax.scatter([c["cost"] * 100 for c in cell[k]["curve"]], [c["sr_pct"] for c in cell[k]["curve"]],
                   color=col, marker=mk, s=7 if mk == "o" else 9, lw=0.7, zorder=2, label=lab,
                   facecolors="none" if mk == "o" else col)
    for mode, v in fm.items():
        ax.scatter(v["cost"] * 100, v["sr_pct"], color=INK, marker="D", s=12, zorder=4,
                   edgecolors="#fcfcfb", linewidths=0.5)
        ax.annotate(mode, (v["cost"] * 100, v["sr_pct"]), xytext=(3, -7 if mode == "Vision" else 2),
                    textcoords="offset points", fontsize=6.0, color=INK2)
    srs = [v["sr_pct"] for v in fm.values()] + [c["sr_pct"] for k in ("six_head", "triage") for c in cell[k]["curve"]]
    ax.set_xlim(lo, hi)
    ax.set_ylim(min(srs) - 2, max(srs) + 7)   # headroom above the highest point for the legend
    ax.set_xlabel("mean billed cost per task (cents)")
    ax.set_ylabel("success rate (%)")
    ax.set_title(r"(a) cls$\cdot$B0, six modes, out-of-fold curves", fontsize=7.5, color=INK2, loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=5.8, handletextpad=0.3, borderaxespad=0.2)

    pf = fr["pooled"]
    u = pf["u_grid"]
    for k, col, ls, lab in (("six_head", SLOT[1], "-", "per-mode heads"), ("triage", SLOT[2], "--", "triage")):
        bx.plot(u, pf[k]["mean_gain_pp"], color=col, lw=1.0, ls=ls, label=lab)
    bx.plot(u, pf["triage"]["null_pointwise_q95_pp"], color=MUTED, lw=0.8, ls=":",
            label="shuffle null, 95th pct (triage)")
    bx.axhline(0, color=MUTED, lw=0.6, zorder=0)
    bx.set_xlabel("budget u (0 = cheapest mode, 1 = dearest)")
    bx.set_ylabel("gain over frontier (pp)")
    bx.set_title(f"(b) pooled over {len(pf['cells'])} cells", fontsize=7.5, color=INK2, loc="left")
    bx.legend(frameon=False, loc="lower right", fontsize=6.0)
    for a in (ax, bx):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    (DST / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(DST / "figures" / "fig_frontier.pdf")
    plt.close(fig)


def tab_frontier():
    """Curve max over the frontier (selected after seeing test outcomes) vs the deployable,
    train-chosen policy; six modes, eight cells pooled."""
    fr = _load("representation_routing_frontier")
    dep = _load("routing_gain_upper_bounds")
    L = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{5pt}",
         r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
         r"router & curve max ($p$) & cells above frontier & deployable [5\%, 95\%] \\", r"\midrule"]
    for k, lab in (("six_head", "per-mode heads"), ("triage", "triage")):
        f, d = fr["pooled"][k], dep["pooled"][k]
        L.append(f"{lab} & {_m(f'+{f['max_pp']:.2f}')} ({f['null_p']:.3f}) & "
                 f"{f['n_cells_with_gain']} of {len(fr['pooled']['cells'])} & "
                 f"{_m(f'{d['observed']:+.2f} [{d['lower05']:+.2f}, {d['upper95']:+.2f}]')} \\\\")
    o = fr["pooled"]["oracle"]
    L.append(f"outcome oracle & {_m(f'+{o['max_pp']:.2f}')} (hindsight) & "
             f"{o['n_cells_with_gain']} of {len(fr['pooled']['cells'])} & --- \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Routing gain over the six fixed modes and all their random mixtures, in success-rate points, "
          r"equal-weight mean over the eight cells. \emph{Curve max}: the best point of an out-of-fold threshold "
          r"sweep, read on a budget normalised per cell, chosen after seeing test outcomes, with a label-shuffle "
          r"$p$ that refits the whole curve per draw (so the choice of point is inside the null; not corrected "
          r"across the two routers). \emph{Deployable}: the operating point chosen on training folds only, task "
          r"bootstrap; its interval is read as a magnitude, not a test, because the threshold is re-selected in "
          r"every draw. Exploratory: designed after the runs were collected.}",
          r"% sources: representation_routing_frontier (pooled), routing_gain_upper_bounds (pooled)",
          r"\label{tab:frontier}", r"\end{table}"]
    _write("tables/tab_frontier.tex", L)


def _rng(a: float, b: float, fmt: str = "{:.2f}") -> str:
    return fmt.format(a) if abs(a - b) < 1e-9 else f"{fmt.format(a)}--{fmt.format(b)}"


def tab_rerun():
    """Every registered same-condition rerun in the paper's eight cells, per cell, next to what one
    distinct added arm buys (same functional, same arm count, same n)."""
    inv = _load("noise_floor_inventory")
    by_cell: dict[str, list[dict]] = {c: [] for c in CELLS}
    extra = []
    for p in inv["clean_pairs"]:
        bl = p["label"].split(".")[0]
        site = next((k for k, v in PAIR_KEY.items() if v == p["site"]), None)
        cid = f"{site}_{bl}" if site else None
        (by_cell[cid] if cid in by_cell else extra).append(p)
    L = [r"\begin{table}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
         r"\begin{tabular}{@{}llrrrrl@{}}", r"\toprule",
         r"cell & rerun arms & $n$ & discordance & rerun adds & distinct arm adds & reading \\", r"\midrule"]
    n_pairs = n_cells = 0
    for cid in CELLS:
        ps, mg = by_cell[cid], inv["margins"][cid]
        gain, gmode = mg["gain_1_best_distinct_arm_pp"], mg["gain_1_best_distinct_arm_mode"]
        gtxt = f"+{gain:.2f} ({MODE[gmode.removeprefix('sr_')]})"
        if not ps:
            L.append(rf"\texttt{{{TT[cid]}}} & none & {mg['n']} & --- & --- & {gtxt} & no rerun \\")
            continue
        n_pairs += len(ps)
        n_cells += 1
        drops = [x for p in ps for x in (p["self_drop_a_to_b_pp"], p["self_drop_b_to_a_pp"])]
        disc = [p["discordance_pct"] for p in ps]
        arms = ", ".join(MODE[p["label"].split(".")[-1]] for p in ps) if len(ps) < 6 else "all six"
        reading = "inside rerun range" if gain <= max(drops) + 1e-9 else "exceeds rerun range"
        L.append(rf"\texttt{{{TT[cid]}}} & {arms} & {ps[0]['n']} & {_rng(min(disc), max(disc))}\% & "
                 rf"{_rng(min(drops), max(drops))} & {gtxt} & {reading} \\")
    L += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{Every registered same-condition rerun in the eight cells: {n_pairs} pairs in {n_cells} cells. "
          r"\emph{Discordance}: share of tasks whose outcome differs between the two runs. \emph{Rerun adds}: "
          r"$|\{a \text{ solves}\} \setminus \{b \text{ solves}\}|/n$ over both directions of every pair, the "
          r"ceiling gain bought by adding a rerun as one extra arm. \emph{Distinct arm adds}: the same functional "
          r"for the best single genuinely different mode added to the cell's best single mode. Same cell, same "
          r"$n$, same arm count, so the two columns are directly comparable; \emph{reading} only says whether the "
          r"distinct arm's gain lies within the range of the cell's reruns. Every run, rerun included, carries "
          r"environment drift as well as serving nondeterminism. The B2 cells solve 2--4\% of tasks, so their "
          r"columns are one or two tasks wide. "
          + (r"Outside the eight cells and not used here: "
             + ", ".join(rf"\texttt{{{p['label']}}}" for p in extra) + ". " if extra else "")
          + r"Source: \texttt{noise\_floor\_inventory.json}.}",
          r"% source: noise_floor_inventory (clean_pairs, margins); regenerate when a pair is registered",
          r"\label{tab:rerun}", r"\end{table}"]
    _write("tables/tab_rerun.tex", L)


def main() -> int:
    fig_frontier()
    tab_frontier()
    tab_rerun()
    print(f"wrote {DST}/figures/fig_frontier.pdf, tables/tab_frontier.tex, tables/tab_rerun.tex", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
