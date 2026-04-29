#!/usr/bin/env python3
"""[Layer 0f] Outcome — overlap-depth distribution for solve pools.

Output:
- results/phantom_paper/figures/fig0f_overlap_stacked_bar.png

Layer 0f: solve-pool overlap depth across observation modes.

See docs/checkpoints/paper_planning.md §3 Layer 0 framework.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results/visualwebarena/phase1"
OUT = ROOT / "results/phantom_paper/figures/fig0f_overlap_stacked_bar.png"

MODE_ORDER = ["DOM", "SoM", "Vision", "Phantom-SoM", "P-text"]
MODE_LABELS = ["DOM", "SoM", "Vision", "Phantom\nSoM", "Phantom\nDOM"]
COLORS = {
    "DOM": "#4c78a8",
    "SoM": "#f58518",
    "Vision": "#54a24b",
    "Phantom-SoM": "#b279a2",
    "P-text": "#e45756",
}
DEPTH_ALPHA = {1: 1.0, 2: 0.75, 3: 0.55, 4: 0.40, 5: 0.25}

PANELS = [
    {
        "key": "B0 cls",
        "title": "B0 classifieds",
        "expected": 234,
        "modes": {
            "DOM": RESULTS / "B0_3mode_classifieds_20260413/phase1_dom_router_0/episodes",
            "SoM": RESULTS / "B0_3mode_classifieds_20260413/phase1_som_router_0/episodes",
            "Vision": RESULTS / "B0_3mode_classifieds_20260413/phase1_vision_router_0/episodes",
            "Phantom-SoM": RESULTS / "B0_phantom_som_classifieds_20260426/phase1_phantom_som_router_0/episodes",
            "P-text": RESULTS / "B0_phantom_text_classifieds_20260427/phase1_phantom_dom_router_0/episodes",
        },
            },
    {
        "key": "B0 red",
        "title": "B0 reddit",
        "expected": 210,
        "modes": {
            "DOM": RESULTS / "B0_3mode_reddit_20260422/phase1_dom_router_0/episodes",
            "SoM": RESULTS / "B0_3mode_reddit_20260422/phase1_som_router_0/episodes",
            "Vision": RESULTS / "B0_3mode_reddit_20260422/phase1_vision_router_0/episodes",
            "Phantom-SoM": RESULTS / "B0_phantom_som_reddit_20260428/phase1_phantom_som_router_0/episodes",
            "P-text": RESULTS / "B0_phantom_text_reddit_20260427/phase1_phantom_dom_router_0/episodes",
        },
            },
    {
        "key": "B1 cls",
        "title": "B1 classifieds",
        "expected": 234,
        "modes": {
            "DOM": RESULTS / "B1_3mode_classifieds_20260413/phase1_dom_router_0/episodes",
            "SoM": RESULTS / "B1_3mode_classifieds_20260413/phase1_som_router_0/episodes",
            "Vision": RESULTS / "B1_3mode_classifieds_20260413/phase1_vision_router_0/episodes",
        },
        "pending": ["Phantom-SoM", "P-text"],
    },
    {
        "key": "B1 red",
        "title": "B1 reddit",
        "expected": 210,
        "modes": {
            "DOM": RESULTS / "B1_3mode_reddit_20260413/phase1_dom_router_0/episodes",
            "SoM": RESULTS / "B1_3mode_reddit_20260413/phase1_som_router_0/episodes",
            "Vision": RESULTS / "B1_3mode_reddit_20260413/phase1_vision_router_0/episodes",
        },
        "pending": ["Phantom-SoM", "P-text"],
    },
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
        if bool(record.get("adjusted_success", record.get("success", False))):
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


def draw_panel(ax: plt.Axes, panel: dict) -> None:
    sets, observed_counts = panel_data(panel)
    available_modes = list(panel["modes"])
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
    ax.set_ylabel("Solved tasks (adjusted_success)")
    ax.set_ylim(0, max_total + max(8, max_total * 0.20))
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9.5, "figure.dpi": 150})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), sharey=False)
    for ax, panel in zip(axes.flat, PANELS):
        draw_panel(ax, panel)

    depth_handles = [
        Patch(facecolor=to_rgba("#555555", DEPTH_ALPHA[depth]), edgecolor="#555555", label=f"depth={depth}")
        for depth in [1, 2, 3, 4, 5]
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
