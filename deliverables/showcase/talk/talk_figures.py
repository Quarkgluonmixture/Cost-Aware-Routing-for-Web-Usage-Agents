#!/usr/bin/env python3
"""Talk figures — the poster's panels redrawn for a room that has just seen the demo.

Same sources as `poster_figures_v9.py`; only the words change. The demo names the
views LOOK / READ / BOTH, so every figure here does too, and the three image-free
variants are named by what they are (READ as a numbered list, READ with BOTH's
instructions, BOTH without the picture). No "oracle", no "pp", no cell codes.

One figure the poster does not have: `talk_hindsight.png`, best single view versus
perfect per-task hindsight in each setting. It replaces the six-view matrix and the
three-arm Venn on the `hindsight` slide. It draws the five-view gain only — the rerun
comparison is arm-count matched and lives in the slide foot, because
`noise_floor_inventory.md` §2 forbids setting a five-view gain against a one-rerun
floor.

Usage::

    .venv/bin/python3 deliverables/showcase/talk/talk_figures.py
"""
from __future__ import annotations

import json
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "scripts" / "analysis" / "figures" / "thesis"))
sys.path.insert(0, str(HERE.parent))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402

import _style as S  # noqa: E402

OUT = HERE / "fig"
PROFILE = REPO / "docs/analysis/cross_sites/per_mode_four_dimension_profile_with_wa.json"
NOISE = REPO / "docs/analysis/cross_sites/noise_floor_inventory.json"
DPI = 170

# what the demo calls each view; the profile uses the thesis names, the other sources lower-case keys
VIEW = {"Vision": "LOOK · screenshot", "SoM": "BOTH · marked screenshot", "DOM": "READ · page text",
        "P-text": "READ as a numbered list", "P-prompt": "READ + BOTH's instructions",
        "P-SoM": "BOTH without the picture"}
SHORT = {"Vision": "LOOK", "SoM": "BOTH", "DOM": "READ", "P-text": "READ as a list",
         "P-prompt": "READ + BOTH's prompt", "P-SoM": "BOTH, no picture"}
KEY = {"vision": "Vision", "som": "SoM", "dom": "DOM", "ptext": "P-text", "pprompt": "P-prompt", "psom": "P-SoM"}
COLOUR = {"Vision": S.C_IMAGE, "SoM": S.C_BOTH, "DOM": S.C_TEXT, "P-text": S.C_TEXT,
          "P-prompt": S.C_TEXT, "P-SoM": S.C_TEXT}
ORDER = ["Vision", "SoM", "DOM", "P-text", "P-prompt", "P-SoM"]   # the two screenshot views first


def _base() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#444444",
                         "axes.labelcolor": S.C_INK, "xtick.color": S.C_INK, "ytick.color": S.C_INK})


def _save(fig, name: str) -> None:
    out = OUT / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {name:24} {out.stat().st_size // 1024} KB")


def behaviour() -> None:
    cells = json.loads(PROFILE.read_text())["cells"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True)
    for ax, (metric, word) in zip(axes, [("scroll_frac", "scrolling"), ("type_frac", "typing")]):
        for y, m in enumerate(ORDER):
            v = [100 * c["per_mode"][m][metric] for c in cells if m in c["per_mode"]]
            assert len(v) == 8, (m, metric, len(v))
            ax.plot([min(v), max(v)], [y, y], lw=5, color=COLOUR[m], alpha=0.3, solid_capstyle="round")
            ax.plot([st.median(v)], [y], "o", ms=14, color=COLOUR[m])
        ax.axhline(1.5, color=S.C_MUTED, lw=1, ls=(0, (4, 4)))
        ax.set_xlabel(f"% of steps spent {word}", fontsize=18)
        ax.tick_params(axis="x", labelsize=15)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(ORDER)))
    axes[0].set_yticklabels([VIEW[m] for m in ORDER], fontsize=16)
    for lab, m in zip(axes[0].get_yticklabels(), ORDER):
        lab.set_color(COLOUR[m])
    axes[0].invert_yaxis()
    axes[1].text(1.0, 0.5, "with the\nscreenshot", transform=axes[1].get_yaxis_transform(),
                 ha="right", va="center", fontsize=13, color=S.C_MUTED)
    axes[1].text(1.0, 3.5, "text only", transform=axes[1].get_yaxis_transform(),
                 ha="right", va="center", fontsize=13, color=S.C_MUTED)
    fig.text(0.99, -0.02, "dot = median over 8 settings · line = range", ha="right", fontsize=13, color=S.C_MUTED)
    fig.tight_layout(w_pad=2.5)
    _save(fig, "talk_behaviour.png")


def failure() -> None:
    import poster_figures_v9 as V9          # the numbers stay in one place (REALM Table 41)
    (_, _, text_rows), (_, _, image_rows) = V9.FAILURE
    blocks = [("TEXT ONLY  —  READ and its three variants", S.C_TEXT, text_rows),
              ("WITH THE SCREENSHOT  —  LOOK and BOTH", S.C_IMAGE, image_rows)]
    slots, spans, y = [], [], 0.0
    for head, col, items in blocks:
        slots.append(("head", head, col, y)); y += 1.2
        first = y
        for lab, v in items:
            slots.append(("bar", (lab, v), col, y)); y += 1.0
        spans.append((first - 0.5, y - 0.5))
        y += 0.55
    fig, ax = plt.subplots(figsize=(13.5, 4.9))
    for kind, payload, col, yy in slots:
        if kind == "head":
            ax.text(0.0, yy - 0.15, payload, ha="left", va="center", fontsize=16, fontweight="bold", color=col)
            continue
        lab, v = payload
        ax.barh(yy, v, height=0.64, color=col, alpha=0.9, zorder=3)
        ax.text(v + 0.04, yy, f"{v:.1f}×", va="center", fontsize=17, fontweight="bold", color=col, zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
        ax.text(-0.05, yy, lab, ha="right", va="center", fontsize=16, color=S.C_INK)
    for lo, hi in spans:      # the 1x line runs through the bars only, never through a block heading
        ax.plot([1.0, 1.0], [lo, hi], color="#555555", lw=1.6, zorder=5)
    ax.text(1.0, y - 0.25, "1× = as often as that side fails this way everywhere", ha="left", va="center",
            fontsize=13, color="#555555")
    ax.set_xlim(0, 2.75)
    ax.set_ylim(y, -0.8)
    ax.set_xticks([]); ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    _save(fig, "talk_failure.png")


ROWS = [("cls_B0", "classifieds", "large model"), ("cls_B1", "classifieds", "small model"),
        ("cls_B2", "classifieds", "small model, other family"), ("red_B0", "reddit", "large model"),
        ("red_B1", "reddit", "small model"), ("red_B2", "reddit", "small model, other family"),
        ("wa_red_B0", "WebArena reddit", "large model"), ("wa_red_B1", "WebArena reddit", "small model")]


def hindsight() -> None:
    margins = json.loads(NOISE.read_text())["margins"]
    assert set(margins) == {r[0] for r in ROWS}, sorted(margins)
    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    for y, (cell, site, model) in enumerate(ROWS):
        m = margins[cell]
        best, hind, gain = m["best_single_sr_pct"], m["oracle_6mode_sr_pct"], m["gain_5_arms_added_pp"]
        assert abs((hind - best) - gain) < 0.01, cell
        view = KEY[m["best_mode"].removeprefix("sr_")]
        ax.annotate("", xy=(hind, y), xytext=(best, y),
                    arrowprops=dict(arrowstyle="-|>", color=S.C_INK, lw=2.2, mutation_scale=18), zorder=3)
        ax.plot([best], [y], "o", ms=14, color=COLOUR[view], zorder=4)
        ax.text(best - 1.1, y, SHORT[view], ha="right", va="center", fontsize=13.5, color=COLOUR[view], fontweight="bold")
        ax.text(hind + 1.0, y, f"+{gain:.1f}", ha="left", va="center", fontsize=15, color=S.C_INK)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([f"{site} · {model}" for _, site, model in ROWS], fontsize=15)
    ax.invert_yaxis()
    ax.set_xlim(-8, 60)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.tick_params(axis="x", labelsize=14); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("tasks solved (%)", fontsize=17)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=12, color=S.C_MUTED, label="best single view in that setting"),
                       Line2D([], [], marker=">", ls="-", ms=9, color=S.C_INK, label="perfect hindsight: any of the six views, per task")],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=14)
    fig.tight_layout()
    _save(fig, "talk_hindsight.png")


def routing() -> None:
    import fig_f13_dominance_plane as f
    rows, _ = f.load(f.SRC)
    xs = [x for r in rows for x, _ in r["pts"].values()]
    ys = [y for r in rows for _, y in r["pts"].values()]
    xlo, xhi = min(xs + [0]) - 0.1, max(xs + [0]) + 0.08
    ylo, yhi = min(ys + [0]) - 1.2, max(ys + [0]) + 1.6
    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    ax.add_patch(Rectangle((xlo, 0), -xlo, yhi, facecolor=f.C_WIN, alpha=0.09, lw=0, zorder=0))
    ax.text(xlo + 0.01, yhi * 0.97, "WIN\ncheaper and\nno worse", ha="left", va="top", fontsize=14,
            color=f.C_WIN, fontweight="bold", linespacing=1.15)
    ax.axhline(0, color="#999999", lw=1); ax.axvline(0, color="#999999", lw=1)
    ax.scatter([0], [0], s=260, marker="*", color="#000000", zorder=6)
    ax.annotate("always the\ncheapest view", (0, 0), textcoords="offset points", xytext=(-12, -40),
                ha="right", fontsize=13, color="#333333")
    for r in rows:
        p = r["pts"]
        if "oracle_triage" in p:
            ax.scatter(*p["oracle_triage"], s=150, marker="s", facecolor="none", edgecolor=f.C_ORACLE, lw=2.2, zorder=4)
        if "learned_lossless" in p:
            ax.scatter(*p["learned_lossless"], s=110, marker="^", color=f.C_LOSSLESS, zorder=4)
        if "learned_nested_honest" in p:
            ax.scatter(*p["learned_nested_honest"], s=170, color=f.C_NESTED, zorder=5)
    ratios = [r for r in (0.8, 0.9, 1.0, 1.1, 1.25, 1.5) if xlo <= np.log2(r) <= xhi]
    ax.set_xticks([np.log2(r) for r in ratios])
    ax.set_xticklabels(["same" if r == 1.0 else f"{(r - 1) * 100:+.0f}%" for r in ratios], fontsize=14)
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.set_xlabel("cost, compared with always using the cheapest view", fontsize=16)
    ax.set_ylabel("tasks solved per 100,\ncompared with it", fontsize=15)
    ax.tick_params(axis="y", labelsize=14)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=12, color=f.C_NESTED, label="learned choice, tested on tasks it never saw"),
                       Line2D([], [], marker="^", ls="", ms=11, color=f.C_LOSSLESS, label="learned choice, tested on its own training tasks"),
                       Line2D([], [], marker="s", ls="", ms=11, mfc="none", mec=f.C_ORACLE, mew=2, label="perfect hindsight")],
              loc="upper right", frameon=False, fontsize=13.5)
    fig.tight_layout()
    _save(fig, "talk_routing.png")


def label_supply() -> None:
    import poster_figures as P8
    rows, trainable = {}, {}
    text = P8.LABEL_SUPPLY_MD.read_text(encoding="utf-8")
    for m in re.finditer(r"^\| (B\d)_(classifieds|reddit) \| (\d+) \| \*\*(\d+)\*\* \| ([\d.]+)% \| (\d)/6 \|", text, re.M):
        rows[(m.group(2), m.group(1))] = int(m.group(4))
    for m in re.finditer(r"^\| (B\d)_(classifieds|reddit) \| (\d+) \| \d \| [^|]+ \| (\*\*no\*\*|yes) \|", text, re.M):
        trainable[(m.group(2), m.group(1))] = (m.group(4) == "yes")
    assert len(rows) == 6 and len(trainable) == 6, (rows, trainable)
    sr = {(c["site"], c["baseline_model"]): c["baseline_policy"]["sr_pct"]
          for c in json.loads(P8.LEARN_JSON.read_text())["cells"]}
    fig, ax = plt.subplots(figsize=(13.5, 5.0))
    for key, n in rows.items():
        ax.scatter([sr[key]], [n], s=420, facecolors=S.C_INK if trainable[key] else "white",
                   edgecolors=S.C_INK, linewidths=2.6, zorder=3)
    ax.set_xlim(0, 32); ax.set_ylim(0, 112)
    ax.set_xlabel("tasks the best single view solves (%)\n← more to gain from choosing", fontsize=16, linespacing=1.5)
    ax.set_ylabel("usable “which view” examples", fontsize=15)
    ax.tick_params(labelsize=14)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend([Line2D([], [], marker="o", ls="", ms=15, mfc=S.C_INK, mec=S.C_INK),
               Line2D([], [], marker="o", ls="", ms=15, mfc="white", mec=S.C_INK, mew=2.4)],
              ["enough to train a classifier", "not enough"], loc="upper left", frameon=False, fontsize=15)
    fig.text(0.99, -0.02, "six VisualWebArena settings", ha="right", fontsize=13, color=S.C_MUTED)
    fig.tight_layout()
    _save(fig, "talk_label_supply.png")


def scaling_supply() -> None:
    """Schematic fixed-yield scaling; no measured points or fitted power law.

    At a fixed probability p that any view solves a task, E[labels] = p*N.
    Arbitrary p values separate the lines visually; they are not fitted success
    rates. The common target is a label budget, NOT a trainability threshold:
    class balance and the min-class filter also matter (label-supply diagnosis §2).
    The empirical 2–4x estimate stays outside this schematic, in the slide footer.
    """
    ink, muted = "#12162E", "#5D6787"
    fig, ax = plt.subplots(figsize=(13.5, 5.2), facecolor="white")
    ax.set_facecolor("white")
    tasks = np.geomspace(1, 100, 300)
    target = 8.0
    levels = [(0.65, "More successes", "#5049F9"),
              (0.27, "", "#AB5FCE"),
              (0.11, "Fewer successes", "#36B1FE")]
    for rate, label, colour in levels:
        ax.plot(tasks, rate * tasks, color=colour, lw=3.3, zorder=3)
        crossing = target / rate
        if label:
            ax.plot([crossing, crossing], [0.3, target], color=colour, lw=1.2,
                    ls=(0, (4, 4)), alpha=.65)
        # These are theoretical intersections, deliberately not empirical dots.
        if label:
            ax.annotate(label, (103, rate * 100), xytext=(8, 0),
                        textcoords="offset points", color=colour, fontsize=17,
                        va="center", fontweight="bold")
    ax.axhline(target, color=muted, lw=1.3, ls=(0, (5, 4)), zorder=1)
    ax.text(1.1, target * 1.15, "Same label target", color=muted, fontsize=16)
    left, right = target / levels[0][0], target / levels[-1][0]
    ax.annotate("", (right, .48), (left, .48),
                arrowprops=dict(arrowstyle="<->", color=ink, lw=1.7))
    ax.text(np.sqrt(left * right), .65, "More tasks for the same target",
            fontsize=16, color=ink, ha="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1, 300); ax.set_ylim(.3, 100)
    ax.set_xticks([1, 10, 100], labels=["", "", ""])
    ax.set_yticks([1, 10, 100], labels=["", "", ""])
    ax.minorticks_off()
    ax.grid(True, which="major", color="#e5e8f1", lw=.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Tasks collected →  (log scale)", fontsize=18, labelpad=10, color=ink)
    ax.set_ylabel("Usable training examples →\n(log scale)", fontsize=18, labelpad=12, color=ink)
    ax.tick_params(length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b9c1d5")
    fig.tight_layout()
    fig.savefig(OUT / "talk_scaling.pdf", bbox_inches="tight", facecolor="white")
    _save(fig, "talk_scaling.png")


def capability_supply() -> None:
    """The six real settings against the line where a choice becomes trainable.

    x = share of tasks some view solves (how capable the agent is); y = examples of the
    second most common right view, the class that decides trainability. The line is
    `scale_needed × second_largest_class` from `router_undersampling_control.json` §D
    (10 training rows per class in a 5-fold split = 12.5), asserted equal across rows.
    The arrow is a direction, not a fitted trend: today's mix of right views and today's
    benchmark size are assumed, which makes any crossing a lower bound (笔记 §453.2).
    Crossing the line means a choice can be trained and tested, not that it wins.
    """
    rows = json.loads((REPO / "docs/analysis/cross_sites/router_undersampling_control.json")
                      .read_text())["whichmode_scale"]
    needs = {round(r["scale_needed"] * r["second_largest_class"], 6) for r in rows}
    assert len(rows) == 6 and len(needs) == 1, (len(rows), needs)
    need = needs.pop()
    ink, muted, accent = "#12162E", "#5D6787", "#5049F9"
    fig, ax = plt.subplots(figsize=(13.5, 5.2), facecolor="white")
    x0, x1, y0, y1 = 4, 100, 2, 60
    ax.fill_between([x0, x1], need, y1, color="#EEF0FE", zorder=0)
    ax.axhline(need, color=ink, lw=2.2, zorder=2)
    ax.text(x0 * 1.08, need * 1.18, "Enough examples to train a choice", fontsize=16,
            color=ink, fontweight="bold")
    ax.text(x0 * 1.08, need * 0.74, "Too few", fontsize=16, color=muted)
    # direction only: examples grow in step with solved tasks (slope 1 on log-log)
    ax.annotate("", (70, 70 * 0.42), (8.5, 8.5 * 0.42),
                arrowprops=dict(arrowstyle="simple,head_width=1.6,head_length=1.4,tail_width=0.55",
                                color="#AB5FCE", alpha=.28, lw=0), zorder=1)
    ax.text(95, 10, "stronger agents →\nmore solved tasks →\nmore examples",
            fontsize=15, color="#8A45AE", ha="right", va="top", linespacing=1.3)
    for r in rows:
        above = r["second_largest_class"] >= need
        ax.scatter([r["solvable_pct"]], [r["second_largest_class"]], s=380, zorder=4,
                   facecolors=accent if above else "white", edgecolors=accent, linewidths=2.6)
        ax.annotate("large model" if r["cell"].startswith("B0") else "small model",
                    (r["solvable_pct"], r["second_largest_class"]), xytext=(14, 0),
                    textcoords="offset points", fontsize=13, color=muted, va="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xticks([5, 10, 20, 50, 100], labels=["5%", "10%", "20%", "50%", "100%"])
    ax.set_yticks([2, 5, 10, 20, 50], labels=["2", "5", "10", "20", "50"])
    ax.minorticks_off()
    ax.tick_params(labelsize=14, length=0)
    ax.grid(True, which="major", color="#e5e8f1", lw=.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Tasks the agent solves with some view →  (log scale)", fontsize=17, labelpad=8, color=ink)
    ax.set_ylabel("Examples of the second\nmost common right view", fontsize=16, labelpad=10, color=ink)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b9c1d5")
    fig.tight_layout()
    _save(fig, "talk_capability.png")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    _base()
    for fn in (behaviour, failure, hindsight, routing, label_supply, scaling_supply, capability_supply):
        fn()


if __name__ == "__main__":
    main()
