#!/usr/bin/env python3
"""Figures for the COLING 2027 draft (deliverables/coling2027/figures/), read from the products.

Every plotted number comes from a product JSON under docs/analysis/cross_sites/; nothing is typed
here. Palette: the dataviz reference categorical slots 1-3 (validated 2026-10-08, light surface),
with marker shape as the second channel so identity never rests on colour alone.

  fig_deploy.pdf      success rate of the three deployment arms per cell (§1 of the frame)
  fig_screenshot.pdf  image-only contrast SoM - P-SoM on rule-flagged vs other tasks (§2)
  fig_dstudy.pdf      decision study: label reliability against runs per task
  fig_frontier.pdf    one cell's (cost, SR) plane + pooled gain over the mixture frontier
  tables/tab_routing.tex, tables/tab_cells.tex
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
OUT = REPO / "deliverables/coling2027/figures"

SLOT = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
ARM_STYLE = {"DOM": (SLOT[1], "o"), "SoM": (SLOT[2], "s"), "Vision": (SLOT[3], "^")}
COLW = 3.15   # ACL single-column width, inches

plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.linewidth": 0.6, "savefig.bbox": "tight", "pdf.fonttype": 42})

LABEL = {"cls": "VWA-classifieds", "red": "VWA-reddit", "wared": "WA-reddit", "shop": "VWA-shopping"}


def _load(name):
    return json.loads((CS / f"{name}.json").read_text(encoding="utf-8"))


def _pretty(cell: str) -> str:
    site, bl = cell.split("_")[0], cell.split("_")[1]
    return f"{LABEL[site]} · {bl}"


def fig_deploy():
    three = {c["cell_id"]: c["frontier"]["fixed_modes"] for c in _load("routing_three_arm")["cells"]}
    ext = {v["variant"]: v["frontier"]["fixed_modes"] for v in _load("routing_extension_cells")["variants"]}
    rows = [("cls_B5", ext["cls_B5"])] + [(k, three[k]) for k in ("cls_B0", "cls_B1", "cls_B2")] + \
           [(k, three[k]) for k in ("red_B0", "red_B1", "red_B2", "wared_B0", "wared_B1")] + \
           [("shop_B0", ext["shop_B0_clean"]), ("shop_B1", ext["shop_B1_clean"])]
    fig, ax = plt.subplots(figsize=(COLW, 2.9))
    for y, (cell, fm) in enumerate(reversed(rows)):
        ax.axhline(y, color=GRID, lw=0.4, zorder=0)
        for arm, (col, mk) in ARM_STYLE.items():
            if arm in fm:
                ax.scatter(fm[arm]["sr_pct"], y, color=col, marker=mk, s=22, zorder=3,
                           edgecolors="#fcfcfb", linewidths=0.6, label=arm if y == 0 else None)
        if "Vision" not in fm:
            ax.text(max(fm[a]["sr_pct"] for a in fm if a in ARM_STYLE) + 2.0, y, "(Vision n/a)",
                    va="center", fontsize=6, color=MUTED)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([_pretty(c) + (" (clean)" if c.startswith("shop") else "") for c, _ in reversed(rows)])
    ax.set_xlabel("success rate (%)")
    ax.set_xlim(0, 44)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    h, l = ax.get_legend_handles_labels()
    order = [l.index(a) for a in ARM_STYLE if a in l]
    ax.legend([h[i] for i in order], [l[i] for i in order], frameon=False, ncol=3, loc="lower center",
              bbox_to_anchor=(0.45, 1.0), handletextpad=0.2, columnspacing=1.0)
    fig.savefig(OUT / "fig_deploy.pdf")
    plt.close(fig)


def fig_screenshot():
    ext = _load("visual_intent_routing")["extension"]["cells"]
    rows = [("cls_B2", "VWA-classifieds · B2"), ("cls_B1", "VWA-classifieds · B1"),
            ("cls_B0", "VWA-classifieds · B0"), ("cls_B0_rerun", "VWA-classifieds · B0 (rerun)"),
            ("cls_B5", "VWA-classifieds · B5"), ("red_B1", "VWA-reddit · B1"),
            ("red_B0", "VWA-reddit · B0"), ("red_B0_rerun", "VWA-reddit · B0 (rerun)")]
    fig, ax = plt.subplots(figsize=(COLW, 2.5))
    for y, (key, lab) in enumerate(reversed(rows)):
        r = ext[key]["som-psom"]
        for part, col, mk, dy in (("flagged", SLOT[1], "o", 0.14), ("rest", SLOT[2], "D", -0.14)):
            p = r[part]
            ax.plot(p["ci"], [y + dy] * 2, color=col, lw=1.0, solid_capstyle="round", zorder=2)
            ax.scatter(p["est_pp"], y + dy, color=col, marker=mk, s=18, zorder=3, edgecolors="#fcfcfb",
                       linewidths=0.6, label=(f"rule-flagged tasks" if part == "flagged" else "other tasks")
                       if y == 0 else None)
    ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lab for _, lab in reversed(rows)])
    ax.set_xlabel("SoM − P-SoM (pp), 95% CI")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.4, 1.0), handletextpad=0.2)
    fig.savefig(OUT / "fig_screenshot.pdf")
    plt.close(fig)


def fig_dstudy():
    """Reliability of a k-run-averaged task x arm label, three deployment arms (routing_three_arm)."""
    nulls = _load("routing_three_arm")["null_cells"]
    ks = list(range(1, 11))
    fig, ax = plt.subplots(figsize=(COLW, 1.9))
    for (c, (col, mk)) in zip(nulls, ((SLOT[1], "o"), (SLOT[2], "s"), (SLOT[3], "^"))):
        s_int, s_err = c["decomposition"]["interaction"], c["decomposition"]["noise"]
        rel = [s_int / (s_int + s_err / k) for k in ks]
        ax.plot(ks, rel, color=col, lw=1.2, marker=mk, ms=3.5, label=_pretty(c["cell_id"]))
    ax.axhline(0.5, color=MUTED, lw=0.6, ls="--", zorder=0)
    ax.set_xlabel("runs per task averaged into the label (k)")
    ax.set_ylabel("label reliability")
    ax.set_ylim(0, 1)
    ax.set_xticks(ks)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=6.5)
    fig.savefig(OUT / "fig_dstudy.pdf")
    plt.close(fig)


def fig_frontier():
    """(a) one cell's (cost, SR) plane: fixed arms, their mixture frontier, the two out-of-fold
    router curves; (b) the pooled frontier gain over the normalised budget u with the label-shuffle
    null's pointwise 95th percentile. Three deployment arms (routing_three_arm)."""
    three = _load("routing_three_arm")
    cell = next(c for c in three["cells"] if c["cell_id"] == "cls_B0")["frontier"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(2 * COLW + 0.3, 2.05), gridspec_kw={"wspace": 0.32})

    fm = cell["fixed_modes"]
    hull = [(p["cost"] * 100, p["sr_pct"]) for p in cell["fixed_hull"]]
    xs = [c["cost"] * 100 for k in ("six_head", "triage") for c in cell[k]["curve"]] + [h[0] for h in hull] + \
         [v["cost"] * 100 for v in fm.values()]
    lo, hi = min(xs) - 0.1, max(xs) + 0.1
    hx = [lo] + [h[0] for h in hull] + [hi]
    hy = [hull[0][1]] + [h[1] for h in hull] + [hull[-1][1]]
    ax.fill_between(hx, 0, hy, color=GRID, alpha=0.55, lw=0, zorder=0)
    ax.plot(hx, hy, color=MUTED, lw=1.0, zorder=1, label="fixed arms + mixtures")
    # a threshold sweep is not monotone in cost, so its points are drawn unconnected
    for k, col, mk, lab in (("six_head", INK2, "o", "per-arm heads (threshold sweep)"),
                            ("triage", INK, "x", "triage (threshold sweep)")):
        ax.scatter([c["cost"] * 100 for c in cell[k]["curve"]], [c["sr_pct"] for c in cell[k]["curve"]],
                   color=col, marker=mk, s=7 if mk == "o" else 9, lw=0.7, zorder=2, label=lab,
                   facecolors="none" if mk == "o" else col)
    for arm, (col, mk) in ARM_STYLE.items():
        ax.scatter(fm[arm]["cost"] * 100, fm[arm]["sr_pct"], color=col, marker=mk, s=26, zorder=4,
                   edgecolors="#fcfcfb", linewidths=0.6)
        ax.annotate(arm, (fm[arm]["cost"] * 100, fm[arm]["sr_pct"]), xytext=(3, -8 if arm == "Vision" else 3),
                    textcoords="offset points", fontsize=6.5, color=INK2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(14, 33.5)
    ax.set_xlabel("mean cost per task (US cents)")
    ax.set_ylabel("success rate (%)")
    ax.set_title("(a) VWA-classifieds · B0, out-of-fold curves", fontsize=7.5, color=INK2, loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=6.0, handletextpad=0.3, borderaxespad=0.2)

    pf = three["pooled_frontier"]
    u = pf["u_grid"]
    for k, col, ls, lab in (("six_head", INK2, "-", "per-arm heads"), ("triage", INK, "--", "triage")):
        bx.plot(u, pf[k]["mean_gain_pp"], color=col, lw=1.0, ls=ls, label=lab)
    bx.plot(u, pf["triage"]["null_pointwise_q95_pp"], color=SLOT[2], lw=0.8, ls=":",
            label="shuffle null, 95th pct (triage)")
    bx.axhline(0, color=MUTED, lw=0.6, zorder=0)
    bx.set_xlabel("normalised budget u")
    bx.set_ylabel("gain over frontier (pp)")
    bx.set_title("(b) pooled over 8 cells", fontsize=7.5, color=INK2, loc="left")
    bx.legend(frameon=False, loc="upper right", fontsize=6.2)
    for a in (ax, bx):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.savefig(OUT / "fig_frontier.pdf")
    plt.close(fig)


def tab_cells():
    """Appendix table: success rate and mean billed cost of the three deployment arms per cell."""
    three = {c["cell_id"]: c for c in _load("routing_three_arm")["cells"]}
    ext = {v["variant"]: v for v in _load("routing_extension_cells")["variants"]}
    rows = [(k, three[k]) for k in ("cls_B0", "cls_B1", "cls_B2", "red_B0", "red_B1", "red_B2",
                                    "wared_B0", "wared_B1")] + \
           [("cls_B5", ext["cls_B5"]), ("shop_B0_clean", ext["shop_B0_clean"]), ("shop_B1_clean", ext["shop_B1_clean"])]
    L = [r"\begin{table}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
         r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{DOM} & \multicolumn{2}{c}{\SoM} & \multicolumn{2}{c}{Vision} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}",
         r"cell & $n$ & SR & cost & SR & cost & SR & cost \\", r"\midrule"]
    for k, c in rows:
        fm = c["frontier"]["fixed_modes"]
        n = c.get("n_tasks") or len(c["frontier"].get("tasks", [])) or ""
        site, bl = k.split("_")[0], k.split("_")[1]
        name = {"cls": "cls", "red": "VWA-red", "wared": "WA-red", "shop": "shop"}[site] + f" {bl}"
        cells = []
        for arm in ("DOM", "SoM", "Vision"):
            cells += ([f"{fm[arm]['sr_pct']:.1f}", f"{fm[arm]['cost'] * 100:.2f}"] if arm in fm else ["--", "--"])
        if k == "cls_B5":
            L.append(r"\addlinespace[2pt]")
        L.append(f"{name} & {n} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Success rate (\%) and mean billed cost per task (US cents) of the three deployment arms. "
          r"B0 and GPT-5.6 (B5) costs are API invoices; B1 and B2 are token-priced estimates for local "
          r"serving, comparable within a cell only. Shopping rows are the clean cells (Appendix~\ref{app:harness}); "
          r"GPT-5.6 Vision is excluded. Sources: \texttt{routing\_three\_arm}, \texttt{routing\_extension\_cells}.}",
          r"\label{tab:cells}", r"\end{table}"]
    (OUT.parent / "tables" / "tab_cells.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


def _m(s: str) -> str:
    """Typeset minus signs as math minus."""
    return s.replace("-", "$-$")


def tab_routing():
    """Table: curve max over the frontier (selected after seeing test outcomes) vs the deployable,
    train-chosen policy. Pooled rows over the 8 cells; extension cells per variant."""
    three = _load("routing_three_arm")
    six_fr = _load("representation_routing_frontier")["pooled"]
    six_dep = _load("routing_gain_upper_bounds")["pooled"]
    ext = {v["variant"]: v for v in _load("routing_extension_cells")["variants"]}
    rows = []
    for k, lab in (("six_head", "per-arm heads"), ("triage", "triage")):
        f = three["pooled_frontier"][k]
        d = three["pooled_deployable"][k]
        rows.append(("3 arms, 8 cells pooled", lab, f"+{f['max_pp']:.2f} ({f['null_p']:.3f})",
                     f"{d['observed']:+.2f} [{d['lower05']:+.2f}, {d['upper95']:+.2f}]"))
    for k, lab in (("six_head", "per-arm heads"), ("triage", "triage")):
        f = six_fr[k]
        d = six_dep[k]
        rows.append(("6 arms, 8 cells pooled", lab, f"+{f['max_pp']:.2f} ({f['null_p']:.3f})",
                     f"{d['observed']:+.2f} [{d['lower05']:+.2f}, {d['upper95']:+.2f}]"))
    for vid, name in (("cls_B5", "GPT-5.6, classifieds (5 arms)"), ("shop_B0_clean", "Shopping B0 (3 arms)"),
                      ("shop_B1_clean", "Shopping B1 (6 arms)")):
        v = ext[vid]
        for k, lab in (("six_head", "per-arm heads"), ("triage", "triage")):
            s = v["frontier"][k]["summary"]
            d = v["deployable"]
            rows.append((name, lab, f"{s['max_excess_pp']:+.2f} ({s['null_p']:.3f})",
                         f"{d['observed'][k]:+.2f} [{d['lower05'][k]:+.2f}, {d['upper95'][k]:+.2f}]"))
    L = [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{6pt}",
         r"\begin{tabular}{@{}llrr@{}}", r"\toprule",
         r"cells & policy & curve max (p) & deployable [5\%, 95\%] \\", r"\midrule"]
    last = None
    for cells, lab, a, b in rows:
        if last is not None and cells != last:
            L.append(r"\addlinespace[2pt]")
        L.append(f"{cells if cells != last else ''} & {lab} & {_m(a)} & {_m(b)} \\\\")
        last = cells
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Routing gain over the fixed arms and their random mixtures, in SR points. \emph{Curve max}: the best "
          r"point of an out-of-fold curve, chosen after seeing test outcomes, with its label-shuffle $p$ (pooled rows: "
          r"max over a normalised budget). \emph{Deployable}: the operating point chosen on training folds only, task "
          r"bootstrap. Sources: \texttt{routing\_three\_arm}, \texttt{representation\_routing\_frontier}, "
          r"\texttt{routing\_gain\_upper\_bounds}, \texttt{routing\_extension\_cells}.}",
          r"\label{tab:routing}", r"\end{table*}"]
    (OUT.parent / "tables").mkdir(exist_ok=True)
    (OUT.parent / "tables" / "tab_routing.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_deploy()
    fig_screenshot()
    fig_dstudy()
    fig_frontier()
    tab_routing()
    tab_cells()
    print(f"wrote {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
