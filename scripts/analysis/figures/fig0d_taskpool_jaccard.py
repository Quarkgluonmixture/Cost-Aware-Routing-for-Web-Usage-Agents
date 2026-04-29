#!/usr/bin/env python3
"""[Outcome 0d] Outcome — task-pool Jaccard heatmap (5-mode).

Output:
- results/phantom_paper/figures/fig0d_taskpool_jaccard.png

The output filename is retained for draft compatibility, but the figure is now
a 5-mode task-pool overlap matrix rather than a geometric Venn sketch.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results/visualwebarena/phase1"
OUT = ROOT / "results/phantom_paper/figures/fig0d_taskpool_jaccard.png"

COLORS = {
    "DOM": "#4c78a8",
    "SoM": "#f58518",
    "Vision": "#54a24b",
    "Phantom-SoM": "#b279a2",
    "P-text": "#e45756",
    "Phantom-prompt": "#9467bd",
}

MODE_ORDER = ["DOM", "P-text", "Phantom-prompt", "Phantom-SoM", "SoM", "Vision"]


def _phantom_prompt_dir(baseline: str, site: str) -> Path | None:
    candidates = sorted(RESULTS.glob(f"{baseline}_phantom_prompt_{site}_*/phase1_phantom_prompt_router_0/episodes"))
    return candidates[-1] if candidates else None

PANELS = [
    {
        "key": "b0_cls",
        "title": "B0 classifieds",
        "expected": 234,
        "modes": {
            "DOM": RESULTS / "B0_3mode_classifieds_20260413/phase1_dom_router_0/episodes",
            "P-text": RESULTS / "B0_phantom_text_classifieds_20260427/phase1_phantom_dom_router_0/episodes",
            "Phantom-SoM": RESULTS / "B0_phantom_som_classifieds_20260426/phase1_phantom_som_router_0/episodes",
            "SoM": RESULTS / "B0_3mode_classifieds_20260413/phase1_som_router_0/episodes",
            "Vision": RESULTS / "B0_3mode_classifieds_20260413/phase1_vision_router_0/episodes",
        },
    },
    {
        "key": "b0_red",
        "title": "B0 reddit",
        "expected": 210,
        "modes": {
            "DOM": RESULTS / "B0_3mode_reddit_20260422/phase1_dom_router_0/episodes",
            "P-text": RESULTS / "B0_phantom_text_reddit_20260427/phase1_phantom_dom_router_0/episodes",
            "Phantom-SoM": RESULTS / "B0_phantom_som_reddit_20260428/phase1_phantom_som_router_0/episodes",
            "SoM": RESULTS / "B0_3mode_reddit_20260422/phase1_som_router_0/episodes",
            "Vision": RESULTS / "B0_3mode_reddit_20260422/phase1_vision_router_0/episodes",
        },
    },
    {
        "key": "b1_cls",
        "title": "B1 classifieds",
        "expected": 234,
        "modes": {
            "DOM": RESULTS / "B1_3mode_classifieds_20260413/phase1_dom_router_0/episodes",
            "Phantom-SoM": RESULTS / "B1_phantom_som_classifieds_20260428/phase1_phantom_som_router_0/episodes",
            "SoM": RESULTS / "B1_3mode_classifieds_20260413/phase1_som_router_0/episodes",
            "Vision": RESULTS / "B1_3mode_classifieds_20260413/phase1_vision_router_0/episodes",
        },
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
    },
]
# Auto-attach Phantom-prompt run dir per panel when it exists on disk
for _panel in PANELS:
    _baseline_short, _site_short = _panel["key"].split("_", 1)
    _baseline = "B0" if _baseline_short.lower() == "b0" else "B1"
    _site_full = "classifieds" if _site_short == "cls" else ("reddit" if _site_short == "red" else _site_short)
    _pp = _phantom_prompt_dir(_baseline, _site_full)
    if _pp is not None:
        _panel["modes"]["Phantom-prompt"] = _pp


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
        if tid in observed:
            print(f"[warn] duplicate summary ignored in count: {path}", file=sys.stderr)
            continue
        observed.add(tid)
        if bool(record.get("adjusted_success", record.get("success", False))):
            successes.add(tid)
    return successes, observed


def jaccard(left: set[int], right: set[int]) -> tuple[float, int, int]:
    inter = len(left & right)
    union = len(left | right)
    return (inter / union if union else 1.0), inter, union


def panel_sets(panel: dict) -> tuple[dict[str, set[int]], dict[str, set[int]], dict[str, int], list[str]]:
    sets: dict[str, set[int]] = {}
    obs: dict[str, set[int]] = {}
    observed_counts: dict[str, int] = {}
    expected = panel["expected"]
    panel_modes = [m for m in MODE_ORDER if m in panel["modes"]]
    for mode in panel_modes:
        successes, observed = load_success_set(panel["modes"][mode])
        sets[mode] = successes
        obs[mode] = observed
        observed_counts[mode] = len(observed)
        if len(observed) != expected:
            print(
                f"[warn] {panel['title']} {mode}: n={len(observed)} expected={expected}",
                file=sys.stderr,
            )
    return sets, obs, observed_counts, panel_modes


def draw_panel(ax: plt.Axes, panel: dict, cmap) -> None:
    sets, obs, observed_counts, panel_modes = panel_sets(panel)
    expected = panel["expected"]
    n_modes = len(panel_modes)
    matrix = np.zeros((n_modes, n_modes), dtype=float)
    annotations: list[list[str]] = [["" for _ in panel_modes] for _ in panel_modes]

    for i, left in enumerate(panel_modes):
        for j, right in enumerate(panel_modes):
            # Restrict both success sets to the joint-observed task universe so
            # a partial mode (e.g. P-prompt at n=134) doesn't artificially
            # under-count overlap on its unobserved tasks.
            joint_obs = obs[left] & obs[right]
            left_s = sets[left] & joint_obs
            right_s = sets[right] & joint_obs
            value, inter, union = jaccard(left_s, right_s)
            matrix[i, j] = value
            if i == j:
                # Self-Jaccard on observed pool reports per-mode SR over its own pool
                n_obs = len(obs[left])
                sr = 100.0 * len(sets[left]) / n_obs if n_obs else 0.0
                partial_mark = "*" if n_obs < expected else ""
                annotations[i][j] = f"1.00\n{len(sets[left])}/{n_obs}{partial_mark}\nSR {sr:.1f}%"
            else:
                annotations[i][j] = f"{value:.2f}\n{inter}/{union}"

    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap=cmap)
    ax.set_xticks(range(n_modes), panel_modes, rotation=35, ha="right", fontsize=8.0)
    ax.set_yticks(range(n_modes), panel_modes, fontsize=8.0)
    ax.set_title(f"{panel['title']} (N={expected})", fontsize=11, fontweight="bold")
    ax.set_xlabel("mode")
    ax.set_ylabel("mode")

    for tick, mode in zip(ax.get_xticklabels(), panel_modes):
        tick.set_color(COLORS[mode])
        tick.set_fontweight("bold")
    for tick, mode in zip(ax.get_yticklabels(), panel_modes):
        tick.set_color(COLORS[mode])
        tick.set_fontweight("bold")

    for i in range(n_modes):
        for j in range(n_modes):
            color = "white" if matrix[i, j] < 0.42 else "#111111"
            ax.text(j, i, annotations[i][j], ha="center", va="center", fontsize=6.5, color=color)

    observed_note = ", ".join(f"{mode} n={observed_counts[mode]}" for mode in panel_modes if observed_counts[mode] != expected)
    if observed_note:
        ax.text(
            0.5,
            -0.28,
            observed_note,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color="#9a3412",
        )
    return im


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    cmap = LinearSegmentedColormap.from_list("overlap", ["#b91c1c", "#fef2f2", "#dbeafe", "#1d4ed8"])
    fig, axes = plt.subplots(1, 4, figsize=(22.0, 5.8), constrained_layout=True)
    image = None
    for ax, panel in zip(axes.flat, PANELS):
        image = draw_panel(ax, panel, cmap)
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, pad=0.03)
        cbar.set_label("Jaccard overlap of adjusted-success task pools")
    fig.suptitle("5-Mode Outcome Overlap: Task-Pool Jaccard Matrix", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        -0.02,
        "Cells show Jaccard overlap and intersection/union counts; diagonals show adjusted successes and SR.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
