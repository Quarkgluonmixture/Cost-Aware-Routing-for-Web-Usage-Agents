#!/usr/bin/env python3
"""Drop-one oracle loss for B0/B1 VWA observation arms.

All available cells are computed from episode-level ``adjusted_success`` sets.
B0 Phantom-SoM uses stale pre-rederive fallback summaries; B1 Phantom-SoM is
drawn as unavailable until the cleared runs are regenerated.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results/visualwebarena/phase1"
OUT = ROOT / "results/phantom_paper/figures/fig2_drop_one_oracle.png"

MODES = ["DOM", "SoM", "Vision", "Phantom-SoM"]
COLORS = {
    "DOM": "#4c78a8",
    "SoM": "#f58518",
    "Vision": "#54a24b",
    "Phantom-SoM": "#b279a2",
}

PANELS = [
    {
        "key": "b0_cls",
        "title": "B0 classifieds",
        "expected": 234,
        "modes": {
            "DOM": RESULTS / "B0_3mode_classifieds_20260413/phase1_dom_router_0/episodes",
            "SoM": RESULTS / "B0_3mode_classifieds_20260413/phase1_som_router_0/episodes",
            "Vision": RESULTS / "B0_3mode_classifieds_20260413/phase1_vision_router_0/episodes",
            "Phantom-SoM": RESULTS / "B0_phantom_classifieds_20260426/phase1_phantom_som_router_0/episodes",
        },
        "notes": {"Phantom-SoM": "fresh re-run"},
    },
    {
        "key": "b0_red",
        "title": "B0 reddit",
        "expected": 210,
        "modes": {
            "DOM": RESULTS / "B0_3mode_reddit_20260422/phase1_dom_router_0/episodes",
            "SoM": RESULTS / "B0_3mode_reddit_20260422/phase1_som_router_0/episodes",
            "Vision": RESULTS / "B0_3mode_reddit_20260422/phase1_vision_router_0/episodes",
            "Phantom-SoM": RESULTS / "run_reddit_1777238854_ef9c4b/phase1_phantom_som_router_0/episodes",
        },
        "notes": {"Phantom-SoM": "fresh re-run"},
    },
    {
        "key": "b1_cls",
        "title": "B1 classifieds",
        "expected": 234,
        "modes": {
            "DOM": RESULTS / "B1_3mode_classifieds_20260413/phase1_dom_router_0/episodes",
            "SoM": RESULTS / "B1_3mode_classifieds_20260413/phase1_som_router_0/episodes",
            "Vision": RESULTS / "B1_3mode_classifieds_20260413/phase1_vision_router_0/episodes",
        },
        "missing": "Phantom-SoM N/A",
    },
    {
        "key": "b1_red",
        "title": "B1 reddit",
        "expected": 210,
        "modes": {
            "DOM": RESULTS / "B1_3mode_reddit_20260413/phase1_dom_router_0/episodes",
            "SoM": RESULTS / "B1_3mode_reddit_20260413/phase1_som_router_0/episodes",
            "Vision": RESULTS / "B1_3mode_reddit_20260413/phase1_vision_router_0/episodes",
        },
        "missing": "Phantom-SoM N/A",
    },
]

SECTION103_LOSS = {
    "b0_cls": {"DOM": 2.14, "SoM": 7.69, "Vision": 3.85, "Phantom-SoM": 1.71},
    "b0_red": {"DOM": 1.43, "SoM": 2.86, "Vision": 1.90, "Phantom-SoM": 2.38},
}


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


def load_panel_sets(panel: dict) -> dict[str, set[int]]:
    sets: dict[str, set[int]] = {}
    for mode, ep_dir in panel["modes"].items():
        successes, observed = load_success_set(ep_dir)
        sets[mode] = successes
        if len(observed) != panel["expected"]:
            print(
                f"[warn] {panel['title']} {mode}: n={len(observed)} "
                f"expected={panel['expected']}",
                file=sys.stderr,
            )
    return sets


def drop_one_losses(sets: dict[str, set[int]], expected: int) -> dict[str, float]:
    union_all = set().union(*sets.values()) if sets else set()
    losses: dict[str, float] = {}
    for mode in sets:
        without = set().union(*(s for m, s in sets.items() if m != mode))
        losses[mode] = 100.0 * (len(union_all) - len(without)) / expected
    return losses


def draw_panel(ax: plt.Axes, panel: dict) -> None:
    sets = load_panel_sets(panel)
    losses = drop_one_losses(sets, panel["expected"])
    if panel["key"] in SECTION103_LOSS:
        for mode, verified in SECTION103_LOSS[panel["key"]].items():
            if mode in losses and abs(losses[mode] - verified) > 0.25:
                print(
                    f"[warn] {panel['title']} {mode}: live/fallback drop-one "
                    f"{losses[mode]:.2f} pp vs §103 {verified:.2f} pp",
                    file=sys.stderr,
                )

    x = np.arange(len(MODES))
    values = [losses.get(mode, 0.0) for mode in MODES]
    colors = [COLORS[mode] if mode in losses else "#d4d4d4" for mode in MODES]
    bars = ax.bar(x, values, color=colors, width=0.66)
    for bar, mode, value in zip(bars, MODES, values):
        if mode not in losses:
            bar.set_hatch("//")
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0.23,
                "N/A",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#666666",
            )
            continue
        label = f"{value:.2f}" + ("*" if panel.get("notes", {}).get(mode) else "")
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.15,
            label,
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax.set_title(f"{panel['title']} (N={panel['expected']})", fontsize=11, fontweight="bold")
    ax.set_xticks(x, ["DOM", "SoM", "Vision", "Phantom"])
    ax.set_ylim(0, 9.0)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4), sharey=True)
    for ax, panel in zip(axes.flat, PANELS):
        draw_panel(ax, panel)
    for ax in axes[:, 0]:
        ax.set_ylabel("Oracle loss when arm is removed (pp)")
    fig.suptitle("Drop-One Oracle: Incremental Routing Value", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.025,
        "Higher bars mean the representation solves tasks not recovered by the other plotted arms. "
        "* B0 Phantom-SoM uses stale pre-rederive fallback summaries; B1 Phantom-SoM pending re-run.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
