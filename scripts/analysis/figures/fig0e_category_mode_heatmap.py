#!/usr/bin/env python3
"""[Layer 0e] Outcome — per-category adjusted SR heatmap.

Output:
- results/phantom_paper/figures/fig0e_category_mode_heatmap.png

Layer 0e: category × observation-mode success-rate evidence.

See docs/checkpoints/paper_planning.md §3 Layer 0 framework.
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
OUT = ROOT / "results/phantom_paper/figures/fig0e_category_mode_heatmap.png"

CATEGORIES = ["A", "B", "C", "D"]
CATEGORY_LABELS = [
    "A\ntext-only",
    "B\nref-image",
    "C\npage-screen",
    "D\nuncertain",
]
MODES = ["DOM", "SoM", "Vision", "Phantom-SoM", "P-text"]

PANELS = [
    {
        "key": "B0 classifieds",
        "site": "classifieds",
        "expected": 234,
        "audit": ROOT / "docs/analysis/cross_sites/codex_audit_classifieds.json",
        "modes": {
            "DOM": RESULTS / "B0_3mode_classifieds_20260413/phase1_dom_router_0/episodes",
            "SoM": RESULTS / "B0_3mode_classifieds_20260413/phase1_som_router_0/episodes",
            "Vision": RESULTS / "B0_3mode_classifieds_20260413/phase1_vision_router_0/episodes",
            "Phantom-SoM": RESULTS / "B0_phantom_som_classifieds_20260426/phase1_phantom_som_router_0/episodes",
            "P-text": RESULTS / "B0_phantom_text_classifieds_20260427/phase1_phantom_dom_router_0/episodes",
        },
        "notes": {"Phantom-SoM": "fresh re-run"},
    },
    {
        "key": "B0 reddit",
        "site": "reddit",
        "expected": 210,
        "audit": ROOT / "docs/analysis/cross_sites/codex_audit_reddit.json",
        "modes": {
            "DOM": RESULTS / "B0_3mode_reddit_20260422/phase1_dom_router_0/episodes",
            "SoM": RESULTS / "B0_3mode_reddit_20260422/phase1_som_router_0/episodes",
            "Vision": RESULTS / "B0_3mode_reddit_20260422/phase1_vision_router_0/episodes",
            "Phantom-SoM": RESULTS / "B0_phantom_som_reddit_20260428/phase1_phantom_som_router_0/episodes",
            "P-text": RESULTS / "B0_phantom_text_reddit_20260427/phase1_phantom_dom_router_0/episodes",
        },
    },
    {
        "key": "B1 classifieds",
        "site": "classifieds",
        "expected": 234,
        "audit": ROOT / "docs/analysis/cross_sites/codex_audit_classifieds.json",
        "modes": {
            "DOM": RESULTS / "B1_3mode_classifieds_20260413/phase1_dom_router_0/episodes",
            "SoM": RESULTS / "B1_3mode_classifieds_20260413/phase1_som_router_0/episodes",
            "Vision": RESULTS / "B1_3mode_classifieds_20260413/phase1_vision_router_0/episodes",
            "Phantom-SoM": RESULTS / "B1_phantom_som_classifieds_20260428/phase1_phantom_som_router_0/episodes",
        },
    },
    {
        "key": "B1 reddit",
        "site": "reddit",
        "expected": 210,
        "audit": ROOT / "docs/analysis/cross_sites/codex_audit_reddit.json",
        "modes": {
            "DOM": RESULTS / "B1_3mode_reddit_20260413/phase1_dom_router_0/episodes",
            "SoM": RESULTS / "B1_3mode_reddit_20260413/phase1_som_router_0/episodes",
            "Vision": RESULTS / "B1_3mode_reddit_20260413/phase1_vision_router_0/episodes",
        },
    },
]


def task_id(path: Path) -> int:
    match = re.search(r"task_(\d+)_summary", path.name)
    if not match:
        raise ValueError(f"Cannot parse task id from {path}")
    return int(match.group(1))


def load_audit(path: Path) -> dict[int, str]:
    data = json.load(path.open())
    categories: dict[int, str] = {}
    for row in data:
        category = str(row["category"])[0]
        if category not in CATEGORIES:
            raise ValueError(f"Unknown category {row['category']} in {path}")
        categories[int(row["task_id"])] = category
    return categories


def load_successes(ep_dir: Path) -> tuple[set[int], set[int]]:
    files = sorted(ep_dir.glob("*_summary_v2.json"))
    if not files:
        print(f"[warn] no episode summaries under {ep_dir}", file=sys.stderr)
        return set(), set()
    observed: set[int] = set()
    successes: set[int] = set()
    for path in files:
        with path.open() as f:
            record = json.load(f)
        tid = task_id(path)
        observed.add(tid)
        if bool(record.get("adjusted_success", record.get("success", False))):
            successes.add(tid)
    return successes, observed


def build_matrix(panel: dict) -> tuple[np.ndarray, list[list[str]]]:
    audit = load_audit(panel["audit"])
    category_tasks = {cat: {tid for tid, c in audit.items() if c == cat} for cat in CATEGORIES}
    matrix = np.full((len(CATEGORIES), len(MODES)), np.nan)
    labels: list[list[str]] = [["" for _ in MODES] for _ in CATEGORIES]

    for j, mode in enumerate(MODES):
        ep_dir = panel["modes"].get(mode)
        if ep_dir is None:
            for i in range(len(CATEGORIES)):
                labels[i][j] = "—\npending"
            continue
        successes, observed = load_successes(ep_dir)
        if len(observed) != panel["expected"]:
            print(
                f"[warn] {panel['key']} {mode}: n={len(observed)} expected={panel['expected']} "
                f"({panel.get('notes', {}).get(mode, 'live')})",
                file=sys.stderr,
            )
        for i, category in enumerate(CATEGORIES):
            denom_tasks = category_tasks[category] & observed
            n = len(denom_tasks)
            if n == 0:
                labels[i][j] = "N/A"
                continue
            value = 100.0 * len(successes & denom_tasks) / n
            matrix[i, j] = value
            suffix = "*" if panel.get("notes", {}).get(mode) else ""
            labels[i][j] = f"{value:.1f}%{suffix}\n(n={n})"
    return matrix, labels


def draw_panel(ax: plt.Axes, panel: dict, vmax: float) -> None:
    matrix, labels = build_matrix(panel)
    cmap = plt.colormaps["YlGnBu"].copy()
    cmap.set_bad("#f2f2f2")
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax.set_title(panel["key"], fontsize=11, fontweight="bold")
    ax.set_xticks(np.arange(len(MODES)), ["DOM", "SoM", "Vision", "Phantom\nSoM", "Phantom\nDOM"], fontsize=8.5)
    ax.set_yticks(np.arange(len(CATEGORIES)), CATEGORY_LABELS, fontsize=8.5)
    ax.tick_params(axis="both", length=0)
    for i in range(len(CATEGORIES)):
        for j in range(len(MODES)):
            value = matrix[i, j]
            color = "white" if np.isfinite(value) and value >= vmax * 0.58 else "#222222"
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=7.2, color=color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(MODES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CATEGORIES), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    return im


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    fig, axes = plt.subplots(1, 4, figsize=(22.0, 5.6), constrained_layout=True)
    ims = [draw_panel(ax, panel, vmax=32.0) for ax, panel in zip(axes, PANELS)]
    fig.colorbar(ims[0], ax=axes, shrink=0.82, label="Adjusted success rate (%)")
    fig.suptitle("Codex Audit Category x Observation Mode (B0 + B1)", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        -0.03,
        "B0 covers 5 modes per site (paper-grade fresh). B1 cls covers 4 modes (Phantom-SoM available, P-text pending). B1 reddit phantom modes pending.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
