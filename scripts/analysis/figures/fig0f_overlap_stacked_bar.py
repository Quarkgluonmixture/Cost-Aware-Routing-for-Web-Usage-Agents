#!/usr/bin/env python3
"""[Outcome 0f] Outcome dimension — overlap-depth distribution for solve pools.

Output:
- results/phantom_paper/figures/fig0f_overlap_stacked_bar.png

Outcome 0f: solve-pool overlap depth across observation modes.

See docs/checkpoints/paper_planning.md §3 Outcome dimension framework.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

try:
    from scripts.analysis.lib.run_registry import PAPER_MODES, get_cells
    from scripts.analysis.figures.lib.panels import paper_grade_panels
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from scripts.analysis.lib.run_registry import PAPER_MODES, get_cells
    from scripts.analysis.figures.lib.panels import paper_grade_panels

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/phantom_paper/figures/fig0f_overlap_stacked_bar.png"

MODE_ORDER = PAPER_MODES
MODE_LABELS = ["DOM", "SoM", "Vision", "P-text", "P-prompt", "P-SoM"]
COLORS = {
    "DOM": "#4c78a8",
    "SoM": "#f58518",
    "Vision": "#54a24b",
    "P-SoM": "#b279a2",
    "P-text": "#e45756",
    "P-prompt": "#9467bd",
}
DEPTH_ALPHA = {1: 1.0, 2: 0.75, 3: 0.60, 4: 0.45, 5: 0.30, 6: 0.20}


# /stress A1.20 P0-3-ABC* + P1-1-AB (2026-05-17): PANELS from shared lib helper.
PANELS = [
    {
        "key": s.key,
        "title": s.title,
        "expected": s.expected_n,
        "modes": dict(s.modes),
        "is_placeholder": s.is_placeholder,
    }
    for s in paper_grade_panels()
]


def task_id(path: Path) -> int:
    match = re.search(r"task_(\d+)_summary", path.name)
    if not match:
        raise ValueError(f"Cannot parse task id from {path}")
    return int(match.group(1))


def load_success_set(ep_dir: Path) -> tuple[set[int], set[int]]:
    files = sorted(ep_dir.glob("*_summary_v2.json"))
    if not files:
        print(f"[warn] no episode summaries under {ep_dir}", file=sys.stderr)
        return set(), set()
    successes: set[int] = set()
    observed: set[int] = set()
    for path in files:
        with path.open() as f:
            record = json.load(f)
        tid = task_id(path)
        observed.add(tid)
        # /stress A1.20 P1-2-AB (2026-05-17, B-283 sibling): strict `is True`.
        if record.get("success") is True:
            successes.add(tid)
    return successes, observed


def panel_data(panel: dict) -> tuple[dict[str, set[int]], dict[str, int]]:
    sets: dict[str, set[int]] = {}
    observed_counts: dict[str, int] = {}
    for mode, ep_dir in panel["modes"].items():
        successes, observed = load_success_set(ep_dir)
        sets[mode] = successes
        observed_counts[mode] = len(observed)
        if len(observed) != panel["expected"]:
            print(
                f"[warn] {panel['key']} {mode}: observed n={len(observed)}/{panel['expected']}",
                file=sys.stderr,
            )
    return sets, observed_counts


def depth_counts(mode: str, sets: dict[str, set[int]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for tid in sets[mode]:
        depth = sum(1 for success_set in sets.values() if tid in success_set)
        counts[depth] = counts.get(depth, 0) + 1
    return counts


def text_color(alpha: float) -> str:
    return "white" if alpha >= 0.7 else "#222222"


def draw_panel(ax: plt.Axes, panel: dict, require_six_mode_complete: bool = True) -> None:
    """Render per-cell stacked-bar overlap depth.

    /stress A1.20 P0-6-B* (2026-05-17, codex Mode B OOB): when `require_six_mode_complete=True`
    (DEFAULT, paper-grade safe), cells missing any of the 6 paper modes render an
    explicit "incomplete — uniqueness inflation risk" placeholder rather than silently
    computing depth over available modes. Pre-fix: "unique=1" meant unique-among-loaded-
    modes (k modes), NOT unique-among-6-mode-universe → if P-prompt / B2 / etc. are
    absent, unique counts inflate **exactly in direction of "hidden 4th arm" structural
    claim** (confirmation bias direction-aligned with paper §1 hypothesis). Now: gate
    rejects incomplete cells from inference; sensitivity sweep (override flag) for
    Phase 1a partial data inspection.

    Override via `P79_FIG0F_ALLOW_INCOMPLETE=1` env var (Phase 1a inspection mode).
    """
    import os as _os
    env_allow = _os.environ.get("P79_FIG0F_ALLOW_INCOMPLETE", "0") in ("1", "true", "yes")
    sets, observed_counts = panel_data(panel)
    available_modes = list(panel["modes"])
    six_required = {"DOM", "SoM", "Vision", "P-text", "P-prompt", "P-SoM"}
    missing_modes = sorted(six_required - set(available_modes))
    if missing_modes and require_six_mode_complete and not env_allow:
        # Paper-grade gate: refuse to render uniqueness on incomplete cell.
        ax.text(
            0.5, 0.5,
            f"INCOMPLETE CELL — uniqueness inflation risk\n\n"
            f"Missing modes: {', '.join(missing_modes)}\n\n"
            f"per /stress A1.20 P0-6: 'unique=N' over <6-mode universe is\n"
            f"biased toward 'hidden 4th arm' structural claim.\n\n"
            f"Re-render after Phase 1a 6-mode data lands, or set\n"
            f"P79_FIG0F_ALLOW_INCOMPLETE=1 for inspection sensitivity.",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=8.5, color="#a93226", style="italic",
            bbox={"boxstyle": "round,pad=0.6", "facecolor": "#fdf2f2",
                  "edgecolor": "#a93226", "linewidth": 1.2},
        )
        ax.set_title(f"{panel['title']} — INCOMPLETE", fontsize=11,
                     fontweight="bold", color="#a93226")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    k = len(available_modes)
    max_depth = max(5, k)
    totals = {mode: len(sets.get(mode, set())) for mode in MODE_ORDER if mode in sets}
    max_total = max(totals.values() or [1])
    placeholder_height = max(2.0, max_total * 0.08)

    for x, mode in enumerate(MODE_ORDER):
        if mode not in sets:
            ax.bar(
                x,
                placeholder_height,
                color="#eeeeee",
                edgecolor="#999999",
                hatch="//",
                width=0.66,
                linewidth=0.9,
            )
            ax.text(x, placeholder_height + 0.8, "N/A\npending", ha="center", va="bottom", fontsize=8, color="#666666")
            continue

        counts = depth_counts(mode, sets)
        bottom = 0
        unique = counts.get(1, 0)
        for depth in range(1, max_depth + 1):
            count = counts.get(depth, 0)
            if count == 0:
                continue
            alpha = DEPTH_ALPHA.get(depth, 0.25)
            bar = ax.bar(
                x,
                count,
                bottom=bottom,
                color=to_rgba(COLORS[mode], alpha),
                edgecolor=COLORS[mode],
                width=0.66,
                linewidth=0.7,
            )[0]
            y = bottom + count / 2
            label_kwargs = {
                "ha": "center",
                "va": "center",
                "fontsize": 8,
                "color": text_color(alpha),
                "fontweight": "bold" if depth == 1 else "normal",
            }
            if count >= 3:
                ax.text(x, y, str(count), **label_kwargs)
            else:
                ax.text(
                    x + 0.36,
                    y,
                    str(count),
                    ha="left",
                    va="center",
                    fontsize=7.5,
                    color="#222222",
                    fontweight="bold" if depth == 1 else "normal",
                )
            if depth == 1 and count > 0 and count < 3:
                bar.set_linewidth(1.2)
            bottom += count

        observed = observed_counts[mode]
        denom = observed if observed else panel["expected"]
        sr = 100.0 * totals[mode] / denom if denom else 0.0
        suffix = panel.get("notes", {}).get(mode, "")
        ax.text(x, totals[mode] + 1.0, f"{sr:.1f}%{suffix}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        unique_y = max(0.8, totals[mode] * (0.38 if totals[mode] <= 8 else 0.55))
        ax.text(
            x + 0.38,
            unique_y,
            f"unique={unique}",
            ha="left",
            va="center",
            fontsize=8,
            color=COLORS[mode],
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.72},
        )

        verbose = [f"d{depth}={counts.get(depth, 0)}" for depth in range(2, max_depth + 1)]
        print(f"{panel['key']} {mode}: total={totals[mode]} unique={unique} " + " ".join(verbose))

    ax.set_title(f"{panel['title']} (N={panel['expected']}, K={k})", fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(MODE_ORDER)), MODE_LABELS)
    ax.set_ylabel("Solved tasks")
    ax.set_ylim(0, max_total + max(8, max_total * 0.20))
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9.5, "figure.dpi": 150})
    # /stress A1.20 P0-3: layout grows with panel count.
    n_panels = len(PANELS)
    n_cols = min(2, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(13.5, 4.75 * n_rows), sharey=False)
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]
    for ax, panel in zip(axes_flat, PANELS):
        # Placeholder placeholders already handled inside draw_panel via
        # require_six_mode_complete gate (P0-6). is_placeholder cells will
        # render the gate's "incomplete cell" text.
        draw_panel(ax, panel)
    for extra in list(axes_flat)[n_panels:]:
        extra.set_visible(False)

    depth_handles = [
        Patch(facecolor=to_rgba("#555555", DEPTH_ALPHA[depth]), edgecolor="#555555", label=f"depth={depth}")
        for depth in [1, 2, 3, 4, 5, 6]
    ]
    pending_handle = Patch(facecolor="#eeeeee", edgecolor="#999999", hatch="//", label="N/A pending")
    fig.legend(handles=depth_handles + [pending_handle], loc="upper center", ncol=6, frameon=False, fontsize=8.5)
    fig.suptitle("Mode Independence and Shared Solve Pools", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.025,
        "Stacked segments = number of modes that solved each task. Depth=1 (solid) means uniquely solved by that arm — "
        "primary evidence for hidden 4th routing arm. N/A pending B1 phantom re-run.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
