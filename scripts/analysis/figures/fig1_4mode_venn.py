#!/usr/bin/env python3
"""Draw 4-arm success-overlap sketches for B0 VWA classifieds and reddit.

The marginal SR and drop-one annotations are pinned to the verified numbers in
docs/checkpoints/实验笔记.md §103. Full 4-way region labels are read from the
current episode summary JSON when available; if those summaries disagree with
§103 adjusted marginals, the script prints a warning and keeps the §103 labels.
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

MODE_ORDER = ["DOM", "SoM", "Vision", "Phantom"]
COLORS = {
    "DOM": "#4c78a8",
    "SoM": "#f58518",
    "Vision": "#54a24b",
    "Phantom": "#b279a2",
}

EPISODES = {
    "classifieds": {
        "DOM": RESULTS / "B0_3mode_classifieds_20260413/phase1_dom_router_0/episodes",
        "SoM": RESULTS / "B0_3mode_classifieds_20260413/phase1_som_router_0/episodes",
        "Vision": RESULTS / "B0_3mode_classifieds_20260413/phase1_vision_router_0/episodes",
        "Phantom": RESULTS / "B0_phantom_classifieds_20260426/phase1_phantom_som_router_0/episodes",
    },
    "reddit": {
        "DOM": RESULTS / "B0_3mode_reddit_20260422/phase1_dom_router_0/episodes",
        "SoM": RESULTS / "B0_3mode_reddit_20260422/phase1_som_router_0/episodes",
        "Vision": RESULTS / "B0_3mode_reddit_20260422/phase1_vision_router_0/episodes",
        "Phantom": RESULTS / "run_reddit_1777238854_ef9c4b/phase1_phantom_som_router_0/episodes",
    },
}

VERIFIED = {
    "classifieds": {
        "N": 234,
        "sr": {"DOM": 14.10, "SoM": 21.37, "Vision": 13.68, "Phantom": 11.97},
        "drop": {"DOM": 2.14, "SoM": 7.69, "Vision": 3.85, "Phantom": 1.71},
    },
    "reddit": {
        "N": 210,
        "sr": {"DOM": 9.52, "SoM": 10.48, "Vision": 6.67, "Phantom": 10.95},
        "drop": {"DOM": 1.43, "SoM": 2.86, "Vision": 1.90, "Phantom": 2.38},
    },
}


def task_id(path: Path) -> int:
    match = re.search(r"task_(\d+)_summary", path.name)
    if not match:
        raise ValueError(f"Cannot parse task id from {path}")
    return int(match.group(1))


def load_success_set(ep_dir: Path) -> set[int]:
    successes: set[int] = set()
    files = list(ep_dir.rglob("*_summary_v2.json"))
    for path in files:
        with path.open() as f:
            record = json.load(f)
        if bool(record.get("adjusted_success", record.get("success", False))):
            successes.add(task_id(path))
    return successes


def region_counts(site: str) -> dict[tuple[str, ...], int]:
    sets = {mode: load_success_set(path) for mode, path in EPISODES[site].items()}
    all_tasks = set().union(*sets.values())
    counts: dict[tuple[str, ...], int] = {}
    for tid in all_tasks:
        key = tuple(mode for mode in MODE_ORDER if tid in sets[mode])
        if key:
            counts[key] = counts.get(key, 0) + 1

    n = VERIFIED[site]["N"]
    for mode, success_set in sets.items():
        computed = 100.0 * len(success_set) / n
        verified = VERIFIED[site]["sr"][mode]
        if abs(computed - verified) > 0.25:
            print(
                f"[warn] {site} {mode}: episode summaries give {computed:.2f}% "
                f"adjusted SR, §103 uses {verified:.2f}%",
                file=sys.stderr,
            )
    return counts


def draw_site(ax: plt.Axes, site: str) -> None:
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-2.35, 2.35)
    ax.set_ylim(-2.15, 2.15)
    positions = {
        "DOM": (-0.75, 0.45),
        "SoM": (0.75, 0.45),
        "Vision": (-0.75, -0.45),
        "Phantom": (0.75, -0.45),
    }
    for mode in MODE_ORDER:
        circle = Circle(
            positions[mode],
            1.22,
            facecolor=COLORS[mode],
            edgecolor=COLORS[mode],
            alpha=0.24,
            lw=2.0,
        )
        ax.add_patch(circle)

    v = VERIFIED[site]
    label_offsets = {
        "DOM": (-1.55, 1.68),
        "SoM": (0.55, 1.68),
        "Vision": (-1.80, -1.72),
        "Phantom": (0.35, -1.72),
    }
    for mode in MODE_ORDER:
        x, y = label_offsets[mode]
        ax.text(
            x,
            y,
            f"{mode}\nSR {v['sr'][mode]:.2f}%\nunique {v['drop'][mode]:.2f} pp",
            ha="left",
            va="center",
            fontsize=9,
            color="#222222",
            fontweight="bold",
        )

    counts = region_counts(site)
    # Label a compact subset of regions; these are descriptive, not area-scaled.
    region_positions = {
        ("DOM",): (-1.48, 0.78),
        ("SoM",): (1.48, 0.78),
        ("Vision",): (-1.48, -0.78),
        ("Phantom",): (1.48, -0.78),
        ("DOM", "SoM"): (0.0, 1.12),
        ("DOM", "Vision"): (-1.32, 0.0),
        ("SoM", "Phantom"): (1.32, 0.0),
        ("Vision", "Phantom"): (0.0, -1.12),
        ("DOM", "SoM", "Vision", "Phantom"): (0.0, 0.0),
    }
    for key, (x, y) in region_positions.items():
        value = counts.get(key, 0)
        if value:
            ax.text(x, y, str(value), ha="center", va="center", fontsize=10, color="#222222")

    ax.set_title(f"{site.capitalize()} (N={v['N']}, adjusted)", fontsize=13, fontweight="bold")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 140})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.7))
    for ax, site in zip(axes, ["classifieds", "reddit"]):
        draw_site(ax, site)
    fig.suptitle("Four Observation Arms: Success-Pool Overlap", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.03,
        "Circle labels use §103 verified adjusted SR and drop-one oracle loss; interior counts are read from current episode JSON for orientation.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
