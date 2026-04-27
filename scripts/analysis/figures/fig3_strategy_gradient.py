#!/usr/bin/env python3
"""Strategy-gradient bars for reddit observation/prompt variants.

The plotted Phantom-DOM ablation points use the verified N=48 notes supplied in
§103/current checkpoint context. The script also counts local episode/step JSON
where present as a sanity check, but does not replace verified values on mismatch.
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
OUT = ROOT / "docs/analysis/figures/fig3_strategy_gradient.png"

MODES = ["DOM", "Vision", "SoM", "Phantom-SoM", "Phantom-DOM"]
COLORS = {
    "DOM": "#4c78a8",
    "Vision": "#54a24b",
    "SoM": "#f58518",
    "Phantom-SoM": "#b279a2",
    "Phantom-DOM": "#e45756",
}

# Verified notes:
# - Full reddit gradient (§103): DOM search 27/type 38/steps 12.7;
#   Phantom-SoM search 20/type 32/steps 9.9; SoM search 12/type 23/steps 8.1.
# - N=48 ablation/user context: DOM search-loop 22.7; Phantom-SoM and
#   Phantom-DOM search-loop 10.8; 5/5 macro metrics Phantom-DOM = Phantom-SoM.
# - N=26 table in §103 supplies type/scroll/self-correction anchors for the
#   ablation subset; we keep Phantom-DOM equal to Phantom-SoM per the N=48 note.
VERIFIED = {
    "Search-loop %": {
        "DOM": 22.7,
        "Vision": None,
        "SoM": 12.0,
        "Phantom-SoM": 10.8,
        "Phantom-DOM": 10.8,
    },
    "Type action %": {
        "DOM": 40.2,
        "Vision": None,
        "SoM": 23.0,
        "Phantom-SoM": 20.4,
        "Phantom-DOM": 20.4,
    },
    "Scroll action %": {
        "DOM": 15.2,
        "Vision": None,
        "SoM": None,
        "Phantom-SoM": 26.2,
        "Phantom-DOM": 26.2,
    },
    "Self-correction / ep": {
        "DOM": 0.31,
        "Vision": None,
        "SoM": None,
        "Phantom-SoM": 0.35,
        "Phantom-DOM": 0.35,
    },
}

STEP_DIRS = {
    "DOM": RESULTS / "B0_3mode_reddit_20260422/phase1_dom_router_0/episodes",
    "Vision": RESULTS / "B0_3mode_reddit_20260422/phase1_vision_router_0/episodes",
    "SoM": RESULTS / "B0_3mode_reddit_20260422/phase1_som_router_0/episodes",
    "Phantom-SoM": RESULTS / "run_reddit_1777238854_ef9c4b/phase1_phantom_som_router_0/episodes",
    "Phantom-DOM": RESULTS / "B0_phantom_dom_reddit_20260427/phase1_phantom_dom_router_0/episodes",
}


def read_steps(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def step_task_id(path: Path) -> int:
    return int(re.search(r"task_(\d+)_steps", path.name).group(1))


def compute_available_metrics() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for mode, ep_dir in STEP_DIRS.items():
        files = list(ep_dir.rglob("reddit_task_*_steps_v2.jsonl"))
        if not files:
            print(f"[warn] no step JSONL found for {mode}; using verified constants only", file=sys.stderr)
            continue
        total = typed = scrolled = 0
        search_loop_eps = 0
        selfcorr = 0
        for path in files:
            steps = read_steps(path)
            total += len(steps)
            search_steps = 0
            for idx, step in enumerate(steps):
                action_type = step.get("action_type") or (step.get("action") or {}).get("action_type")
                if action_type == "type":
                    typed += 1
                if action_type == "scroll":
                    scrolled += 1
                url = step.get("obs_url", "")
                next_url = steps[idx + 1].get("obs_url", "") if idx + 1 < len(steps) else ""
                if "/search" in url or (action_type == "type" and "/search" in next_url):
                    search_steps += 1
                action = step.get("action") or {}
                thought = action.get("thought", "").lower() if isinstance(action, dict) else ""
                if any(token in thought for token in ("mistake", "wrong", "try again", "go back")):
                    selfcorr += 1
            if search_steps >= 2:
                search_loop_eps += 1
        n = len({step_task_id(p) for p in files})
        if total:
            out[mode] = {
                "Search-loop %": 100.0 * search_loop_eps / n,
                "Type action %": 100.0 * typed / total,
                "Scroll action %": 100.0 * scrolled / total,
                "Self-correction / ep": selfcorr / n,
            }
    return out


def main() -> None:
    available = compute_available_metrics()
    if "Phantom-DOM" in available:
        observed = available["Phantom-DOM"]["Search-loop %"]
        verified = VERIFIED["Search-loop %"]["Phantom-DOM"]
        if verified is not None and abs(observed - verified) > 0.5:
            print(
                f"[warn] Phantom-DOM derived search-loop {observed:.1f}% differs from "
                f"verified {verified:.1f}%; plotting verified value",
                file=sys.stderr,
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9.5, "figure.dpi": 150})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    axes = axes.ravel()

    for ax, metric in zip(axes, VERIFIED):
        values = [VERIFIED[metric][mode] for mode in MODES]
        x = np.arange(len(MODES))
        heights = [0 if value is None else value for value in values]
        bars = ax.bar(x, heights, color=[COLORS[m] for m in MODES], width=0.68)
        for bar, value in zip(bars, values):
            if value is None:
                bar.set_alpha(0.18)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.2,
                    "n/a",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#666666",
                )
            else:
                label = f"{value:.1f}" if "Self" not in metric else f"{value:.2f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(0.35, 0.02 * max(heights or [1])),
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        ax.set_title(metric)
        ax.set_xticks(x, MODES, rotation=25, ha="right")
        ax.grid(axis="y", color="#dddddd", linewidth=0.8)
        ax.set_axisbelow(True)
        if "Self" not in metric:
            ax.set_ylabel("Percent")
        else:
            ax.set_ylabel("Count per episode")

    fig.suptitle("Reddit Strategy Gradient: Representation Changes Exploration", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.02,
        "Verified points from §103/current N=48 ablation notes; n/a means no §103-verified value for that mode/metric.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
