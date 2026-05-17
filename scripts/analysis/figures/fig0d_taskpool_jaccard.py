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

try:
    from scripts.analysis.lib.run_registry import get_cells
    from scripts.analysis.figures.lib.panels import paper_grade_panels
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from scripts.analysis.lib.run_registry import get_cells
    from scripts.analysis.figures.lib.panels import paper_grade_panels


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/phantom_paper/figures/fig0d_taskpool_jaccard.png"

COLORS = {
    "DOM": "#4c78a8",
    "SoM": "#f58518",
    "Vision": "#54a24b",
    "P-SoM": "#b279a2",
    "P-text": "#e45756",
    "P-prompt": "#9467bd",
}

MODE_ORDER = ["DOM", "P-text", "P-prompt", "P-SoM", "SoM", "Vision"]


# /stress A1.20 P0-3-ABC* + P1-1-AB (2026-05-17): PANELS from shared lib helper
# (was: hardcoded B0+B1, stale N=234/210). Now B0+B1+B2 + canonical N=224/205.
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
        if tid in observed:
            print(f"[warn] duplicate summary ignored in count: {path}", file=sys.stderr)
            continue
        observed.add(tid)
        # /stress A1.20 P1-2-AB (2026-05-17, B-283 sibling): strict `is True`.
        if record.get("success") is True:
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
    # /stress A1.20 P0-3: placeholder cells (B2 pre-Phase-1a-fire) render "pending"
    # tile rather than silent skip.
    if panel.get("is_placeholder") or not panel["modes"]:
        ax.text(0.5, 0.5,
                f"{panel['title']}\n\n(pending Phase 1a)\nN={panel['expected']}",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="#888888", style="italic")
        ax.set_title(panel["title"], fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        return None
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
    # /stress A1.20 P0-3: layout = (n_panels // n_cols)×n_cols, grows with B2.
    n_panels = len(PANELS)
    n_cols = min(3, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7.0 * n_cols, 5.8 * n_rows),
                             constrained_layout=True)
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]
    image = None
    for ax, panel in zip(axes_flat, PANELS):
        result = draw_panel(ax, panel, cmap)
        if result is not None:
            image = result
    for extra in list(axes_flat)[n_panels:]:
        extra.set_visible(False)
    if image is not None:
        cbar = fig.colorbar(image,
                            ax=(axes.ravel().tolist() if hasattr(axes, "ravel") else [axes]),
                            shrink=0.82, pad=0.03)
        # /stress A1.20 P2-2-A (2026-05-17): canonical `success` per §139.8 retire.
        cbar.set_label("Jaccard overlap of success task pools (canonical, post-§139.8)")
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
