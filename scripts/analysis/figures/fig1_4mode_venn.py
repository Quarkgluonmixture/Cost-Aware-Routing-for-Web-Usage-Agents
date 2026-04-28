#!/usr/bin/env python3
"""Draw live success-overlap sketches for B0/B1 VWA observation arms.

The B0/B1 DOM/SoM/Vision cells are read from current episode-level
``adjusted_success`` summaries. B0 Phantom-SoM uses the pre-rederive backup
episode summaries because the fresh runs are currently being regenerated. B1
Phantom-SoM is intentionally marked unavailable.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results/visualwebarena/phase1"
OUT = ROOT / "results/phantom_paper/figures/fig1_4mode_venn.png"

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
            },
    {
        "key": "b0_red",
        "title": "B0 reddit",
        "expected": 210,
        "modes": {
            "DOM": RESULTS / "B0_3mode_reddit_20260422/phase1_dom_router_0/episodes",
            "SoM": RESULTS / "B0_3mode_reddit_20260422/phase1_som_router_0/episodes",
            "Vision": RESULTS / "B0_3mode_reddit_20260422/phase1_vision_router_0/episodes",
            "Phantom-SoM": RESULTS / "B0_phantom_reddit_20260428/phase1_phantom_som_router_0/episodes",
        },
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
        "missing": "Phantom-SoM N/A pending re-run",
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
        "missing": "Phantom-SoM N/A pending re-run",
    },
]

SECTION103_SR = {
    "b0_cls": {"DOM": 14.10, "SoM": 21.37, "Vision": 13.68, "Phantom-SoM": 11.97},
    "b0_red": {"DOM": 9.52, "SoM": 10.48, "Vision": 6.67, "Phantom-SoM": 10.95},
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


def panel_sets(panel: dict) -> tuple[dict[str, set[int]], dict[str, int]]:
    sets: dict[str, set[int]] = {}
    observed_counts: dict[str, int] = {}
    expected = panel["expected"]
    for mode, ep_dir in panel["modes"].items():
        successes, observed = load_success_set(ep_dir)
        sets[mode] = successes
        observed_counts[mode] = len(observed)
        if len(observed) != expected:
            print(
                f"[warn] {panel['title']} {mode}: n={len(observed)} expected={expected}",
                file=sys.stderr,
            )
        if panel["key"] in SECTION103_SR and mode in SECTION103_SR[panel["key"]]:
            live_sr = 100.0 * len(successes) / expected
            verified = SECTION103_SR[panel["key"]][mode]
            if abs(live_sr - verified) > 0.25:
                print(
                    f"[warn] {panel['title']} {mode}: live/fallback adjusted SR "
                    f"{live_sr:.2f}% vs §103 {verified:.2f}%",
                    file=sys.stderr,
                )
    return sets, observed_counts


def unique_count(mode: str, sets: dict[str, set[int]]) -> int:
    others = set().union(*(s for m, s in sets.items() if m != mode))
    return len(sets[mode] - others)


def draw_panel(ax: plt.Axes, panel: dict) -> None:
    sets, observed_counts = panel_sets(panel)
    modes = list(panel["modes"])
    expected = panel["expected"]
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-2.25, 2.25)

    if len(modes) == 4:
        positions = {
            "DOM": (-0.75, 0.45),
            "SoM": (0.75, 0.45),
            "Vision": (-0.75, -0.45),
            "Phantom-SoM": (0.75, -0.45),
        }
        label_positions = {
            "DOM": (-2.35, 1.65),
            "SoM": (0.35, 1.65),
            "Vision": (-2.35, -1.72),
            "Phantom-SoM": (0.25, -1.72),
        }
        radius = 1.22
    else:
        positions = {
            "DOM": (-0.85, 0.25),
            "SoM": (0.85, 0.25),
            "Vision": (0.0, -0.7),
        }
        label_positions = {
            "DOM": (-2.35, 1.45),
            "SoM": (0.55, 1.45),
            "Vision": (-0.9, -1.82),
        }
        radius = 1.16

    for mode in modes:
        ax.add_patch(
            Circle(
                positions[mode],
                radius,
                facecolor=COLORS[mode],
                edgecolor=COLORS[mode],
                alpha=0.24,
                lw=2.0,
            )
        )

    for mode in modes:
        sr = 100.0 * len(sets[mode]) / expected
        suffix = "*" if panel.get("notes", {}).get(mode) else ""
        x, y = label_positions[mode]
        ax.text(
            x,
            y,
            f"{mode}{suffix}\nSR {sr:.2f}%\nunique {unique_count(mode, sets)}",
            ha="left",
            va="center",
            fontsize=8.5,
            color="#222222",
            fontweight="bold",
        )

    common = set.intersection(*(sets[m] for m in modes)) if modes else set()
    oracle = set().union(*sets.values()) if sets else set()
    ax.text(
        0,
        0.03,
        f"all {len(common)}\noracle {len(oracle)}",
        ha="center",
        va="center",
        fontsize=10,
        color="#222222",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.86},
    )

    if panel.get("missing"):
        ax.text(0, -2.08, panel["missing"], ha="center", va="center", fontsize=8.5, color="#777777")

    if any(n != expected for n in observed_counts.values()):
        observed = ", ".join(f"{m} n={n}" for m, n in observed_counts.items())
        ax.text(0, 2.03, observed, ha="center", va="center", fontsize=7.5, color="#9a3412")

    ax.set_title(f"{panel['title']} (N={expected})", fontsize=12, fontweight="bold")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.5))
    for ax, panel in zip(axes.flat, PANELS):
        draw_panel(ax, panel)
    fig.suptitle("Observation Arms: Success-Pool Overlap", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.025,
        "Labels show adjusted SR and tasks uniquely solved by that arm within the plotted set. "
        "B1 Phantom-SoM/Phantom-DOM is unavailable pending re-run.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
