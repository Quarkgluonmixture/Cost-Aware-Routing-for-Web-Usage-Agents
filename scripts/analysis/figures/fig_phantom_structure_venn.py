#!/usr/bin/env python3
"""[Outcome supporting] Paper §1 centerpiece — Phantom space 2-axis empirical
structure visualization (3-circle Venn per cell).

Shows per cell the task-set overlap among P-text / P-SoM / P-prompt:

- If circles heavily overlap → phantom space collapsed (1-D, axis decomposition
  not empirically validated; paper hook degrades to 04-30 fallback "P-SoM only")
- If circles have substantial unique regions → phantom space is multi-region
  (axis decomposition empirically validated; paper hook is "phantom routing
  space" with 2-axis structural claim)

This figure directly visualizes H3 (structural) gating evidence from
`phantom_lift.md` — but as a unified shape rather than as a table of counts.

Output: `results/phantom_paper/figures/fig_phantom_structure_venn.png`

Dependencies: `matplotlib_venn` (already installed in `.venv`).

T0d-bis (paper §1 centerpiece, post-T0d) of `EVIDENCE_LAYER_AUDIT.md` queue.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

try:
    from matplotlib_venn import venn3, venn3_circles, venn2, venn2_circles
except ImportError:  # pragma: no cover
    sys.exit("matplotlib_venn not installed; pip install matplotlib_venn")

try:
    from scripts.analysis.lib.run_registry import get_cells, BASELINES
    from scripts.analysis.lib.canonical_task_universe import expected_scored_ids
    from scripts.analysis.figures.lib.panels import paper_grade_panels
except ModuleNotFoundError:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from scripts.analysis.lib.run_registry import get_cells, BASELINES
    from scripts.analysis.lib.canonical_task_universe import expected_scored_ids
    from scripts.analysis.figures.lib.panels import paper_grade_panels

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/phantom_paper/figures/fig_phantom_structure_venn.png"

# /stress A1.20 P0-3-ABC* (2026-05-17): drive panel list from run_registry +
# scored_task_count canonical N (was: hardcoded 4 B0+B1 tuples).
PANELS = [
    (s.baseline, s.site, s.title)
    for s in paper_grade_panels()
]

# Phantom arm colors aligned with `fig0c_drop_one_oracle.py`
ARM_COLORS = {
    "P-text":   "#9e6da8",
    "P-SoM":    "#b279a2",
    "P-prompt": "#9467bd",
}


def task_id(path: Path) -> int:
    m = re.search(r"task_(\d+)_summary", path.name)
    if not m:
        raise ValueError(f"Cannot parse task id from {path}")
    return int(m.group(1))


def load_success_set(ep_dir: Path) -> tuple[set[int], set[int]]:
    files = sorted(ep_dir.glob("*_summary_v2.json"))
    if not files:
        return set(), set()
    succ: set[int] = set()
    obs: set[int] = set()
    for path in files:
        try:
            rec = json.loads(path.read_text())
        except Exception:
            continue
        tid = task_id(path)
        obs.add(tid)
        # /stress A1.20 P1-2-AB (2026-05-17, B-283 sibling): strict `is True`.
        if rec.get("success") is True:
            succ.add(tid)
    return succ, obs


def draw_panel(ax: plt.Axes, baseline: str, site: str, title: str) -> None:
    """One Venn panel for a (baseline, site) cell."""
    # F40 audit fix 2026-05-09: respect P79_AGGREGATOR_GRADE so legacy
    # archived cells produce sensitivity figures.
    import os as _os
    env_grade = _os.environ.get("P79_AGGREGATOR_GRADE", "")
    grade_filter = [g.strip() for g in env_grade.split(",") if g.strip()] or None
    cells = list(get_cells(baseline=baseline, site=site, grade=grade_filter))
    mode_dirs = {c.mode: c.episodes_dir for c in cells}

    needed = ["P-text", "P-SoM"]
    optional = "P-prompt"

    sets = {}
    obs = {}
    for mode in needed + [optional]:
        if mode not in mode_dirs:
            continue
        s, o = load_success_set(mode_dirs[mode])
        if not o:
            continue
        sets[mode] = s
        obs[mode] = o

    missing = [m for m in needed if m not in sets]
    if missing:
        ax.text(0.5, 0.5, f"missing: {', '.join(missing)}",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="#888888", style="italic")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")
        return

    # Common task universe (intersection of observed) ∩ canonical scored set.
    #
    # B-1907 (/stress Mode B codex follow-up, 2026-07-27): the intersection
    # alone is the COLLECTED universe, so the AMENDMENT_08 protocol-excluded
    # reddit tasks sat inside the per-arm "uniquely solves" regions this figure
    # exists to show.  Measured impact before the fix: B0_reddit P-SoM-only
    # 6→5 (task 160) and B2_reddit 2→1 (task 58).  Task 160 is a tier-A passive
    # false positive, so the pre-fix panel credited P-SoM with uniquely solving
    # a task nothing actually solved — in the one evidence slot §1 leans on
    # after H1 failed.  Same defect class as B-1901, different call site.
    scored_ids, _scored_sha = expected_scored_ids(site)
    common = set.intersection(*obs.values()) & set(scored_ids)
    n = len(common)
    if n == 0:
        ax.text(0.5, 0.5, "no common task universe",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="#888888", style="italic")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")
        return

    # Restrict to common universe
    sets_r = {m: s & common for m, s in sets.items()}

    if optional in sets_r:
        # 3-circle Venn
        v = venn3([sets_r["P-text"], sets_r["P-SoM"], sets_r[optional]],
                  set_labels=("P-text", "P-SoM", optional),
                  set_colors=(ARM_COLORS["P-text"], ARM_COLORS["P-SoM"],
                              ARM_COLORS[optional]),
                  alpha=0.55, ax=ax)
        venn3_circles([sets_r["P-text"], sets_r["P-SoM"], sets_r[optional]],
                      linewidth=1.0, linestyle="solid", color="#333333", ax=ax)
        # Annotate set sizes inside set labels
        for label_id, mode in [("A", "P-text"), ("B", "P-SoM"), ("C", optional)]:
            lab = v.get_label_by_id(label_id)
            if lab:
                lab.set_text(f"{mode}\n(N solved={len(sets_r[mode])})")
                lab.set_fontsize(9)
                lab.set_fontweight("bold")
    else:
        # 2-circle Venn
        v = venn2([sets_r["P-text"], sets_r["P-SoM"]],
                  set_labels=("P-text", "P-SoM"),
                  set_colors=(ARM_COLORS["P-text"], ARM_COLORS["P-SoM"]),
                  alpha=0.55, ax=ax)
        venn2_circles([sets_r["P-text"], sets_r["P-SoM"]],
                      linewidth=1.0, linestyle="solid", color="#333333", ax=ax)
        for label_id, mode in [("A", "P-text"), ("B", "P-SoM")]:
            lab = v.get_label_by_id(label_id)
            if lab:
                lab.set_text(f"{mode}\n(N solved={len(sets_r[mode])})")
                lab.set_fontsize(9)
                lab.set_fontweight("bold")

    ax.set_title(f"{title}  (N={n} common tasks)", fontsize=11.5,
                 fontweight="bold")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    # /stress A1.20 P0-3: layout grows with panel count (3 baselines × 2 sites = 6).
    n_panels = len(PANELS)
    n_cols = min(2, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13.5, 5.5 * n_rows))
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]
    for ax, (baseline, site, title) in zip(axes_flat, PANELS):
        draw_panel(ax, baseline, site, title)
    for extra in list(axes_flat)[n_panels:]:
        extra.set_visible(False)

    fig.suptitle(
        "Phantom space 2-axis empirical structure — task-set overlap "
        "(H3 structural evidence)",
        fontsize=13.5, fontweight="bold", y=0.99,
    )
    fig.text(
        0.5, 0.02,
        "Per cell: each circle = task set solved by that phantom arm; circle area "
        "encodes # tasks solved; intersection counts are subset cardinalities. "
        "If circles heavily overlap → phantom space is collapsed to a single point "
        "(axis decomposition not empirical). If unique regions are non-trivial "
        "(≥2 tasks per axis, H3 commit floor) → phantom space is multi-region (M1/M2 "
        "axis decomposition empirically validated). 2-circle Venn = P-prompt data "
        "absent for that cell; measured zero-success P-prompt is still rendered as "
        "a 3-circle panel.",
        ha="center", fontsize=9.0, color="#555555", wrap=True,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
