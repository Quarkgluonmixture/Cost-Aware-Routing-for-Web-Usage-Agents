#!/usr/bin/env python3
"""Drop-one oracle loss for four B0 VWA observation arms.

Bars use the §103 verified adjusted same-task oracle losses.
Episode summaries are loaded only as a sanity check so the script fails loudly
if the expected result directories disappear.
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

MODES = ["DOM", "SoM", "Vision", "Phantom"]
COLORS = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]

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

VERIFIED_LOSS = {
    "classifieds": {"DOM": 2.14, "SoM": 7.69, "Vision": 3.85, "Phantom": 1.71},
    "reddit": {"DOM": 1.43, "SoM": 2.86, "Vision": 1.90, "Phantom": 2.38},
}
N = {"classifieds": 234, "reddit": 210}


def task_id(path: Path) -> int:
    return int(re.search(r"task_(\d+)_summary", path.name).group(1))


def load_success_set(ep_dir: Path) -> set[int]:
    files = list(ep_dir.rglob("*_summary_v2.json"))
    if not files:
        raise FileNotFoundError(f"No episode summaries under {ep_dir}")
    successes = set()
    for path in files:
        with path.open() as f:
            rec = json.load(f)
        if bool(rec.get("adjusted_success", rec.get("success", False))):
            successes.add(task_id(path))
    return successes


def sanity_check() -> None:
    for site, mode_paths in EPISODES.items():
        sets = {mode: load_success_set(path) for mode, path in mode_paths.items()}
        union = set().union(*sets.values())
        for mode in MODES:
            without = set().union(*(sets[m] for m in MODES if m != mode))
            computed = 100.0 * (len(union) - len(without)) / N[site]
            verified = VERIFIED_LOSS[site][mode]
            if abs(computed - verified) > 0.25:
                print(
                    f"[warn] {site} {mode}: episode-derived drop-one {computed:.2f} pp, "
                    f"§103 verified {verified:.2f} pp",
                    file=sys.stderr,
                )


def main() -> None:
    sanity_check()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)
    ymax = 8.6
    for ax, site in zip(axes, ["classifieds", "reddit"]):
        values = [VERIFIED_LOSS[site][m] for m in MODES]
        x = np.arange(len(MODES))
        bars = ax.bar(x, values, color=COLORS, width=0.68)
        ax.set_title(f"{site.capitalize()} (N={N[site]}, adjusted)")
        ax.set_xticks(x, MODES)
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", color="#dddddd", linewidth=0.8)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.18,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    axes[0].set_ylabel("Oracle loss when arm is removed (percentage points)")
    fig.suptitle("Drop-One Oracle: Incremental Routing Value", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.02,
        "Higher bars mean the representation solves more tasks not recovered by the other three arms.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
