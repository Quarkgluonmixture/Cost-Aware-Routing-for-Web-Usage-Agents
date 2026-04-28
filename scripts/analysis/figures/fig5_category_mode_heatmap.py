#!/usr/bin/env python3
"""B0 adjusted SR by Codex audit category and observation mode."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results/visualwebarena/phase1"
OUT = ROOT / "results/phantom_paper/figures/fig5_category_mode_heatmap.png"

CATEGORIES = ["A", "B", "C", "D"]
CATEGORY_LABELS = [
    "A\ntext-only",
    "B\nref-image",
    "C\npage-screen",
    "D\nuncertain",
]
MODES = ["DOM", "SoM", "Vision", "Phantom-SoM", "Phantom-DOM"]

SITES = {
    "classifieds": {
        "expected": 234,
        "audit": ROOT / "docs/analysis/cross_sites/codex_audit_classifieds.json",
        "modes": {
            "DOM": RESULTS / "B0_3mode_classifieds_20260413/phase1_dom_router_0/episodes",
            "SoM": RESULTS / "B0_3mode_classifieds_20260413/phase1_som_router_0/episodes",
            "Vision": RESULTS / "B0_3mode_classifieds_20260413/phase1_vision_router_0/episodes",
            "Phantom-SoM": RESULTS / "B0_phantom_classifieds_20260426/phase1_phantom_som_router_0/episodes",
            "Phantom-DOM": RESULTS / "B0_phantom_dom_classifieds_20260427/phase1_phantom_dom_router_0/episodes",
        },
        "notes": {"Phantom-SoM": "fresh re-run"},
    },
    "reddit": {
        "expected": 210,
        "audit": ROOT / "docs/analysis/cross_sites/codex_audit_reddit.json",
        "modes": {
            "DOM": RESULTS / "B0_3mode_reddit_20260422/phase1_dom_router_0/episodes",
            "SoM": RESULTS / "B0_3mode_reddit_20260422/phase1_som_router_0/episodes",
            "Vision": RESULTS / "B0_3mode_reddit_20260422/phase1_vision_router_0/episodes",
            "Phantom-SoM": RESULTS / "run_reddit_1777238854_ef9c4b/phase1_phantom_som_router_0/episodes",
            "Phantom-DOM": RESULTS / "B0_phantom_dom_reddit_20260427/phase1_phantom_dom_router_0/episodes",
        },
        "notes": {"Phantom-SoM": "fresh re-run", "Phantom-DOM": "partial live"},
    },
}


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


def build_matrix(site: str) -> tuple[np.ndarray, list[list[str]]]:
    spec = SITES[site]
    audit = load_audit(spec["audit"])
    category_tasks = {cat: {tid for tid, c in audit.items() if c == cat} for cat in CATEGORIES}
    matrix = np.full((len(CATEGORIES), len(MODES)), np.nan)
    labels: list[list[str]] = [["" for _ in MODES] for _ in CATEGORIES]

    for j, mode in enumerate(MODES):
        successes, observed = load_successes(spec["modes"][mode])
        if len(observed) != spec["expected"]:
            print(
                f"[warn] {site} {mode}: n={len(observed)} expected={spec['expected']} "
                f"({spec.get('notes', {}).get(mode, 'live')})",
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
            suffix = "*" if spec.get("notes", {}).get(mode) else ""
            labels[i][j] = f"{value:.1f}%{suffix}\n(n={n})"
    return matrix, labels


def draw_site(ax: plt.Axes, site: str, vmax: float) -> None:
    matrix, labels = build_matrix(site)
    cmap = plt.colormaps["YlGnBu"].copy()
    cmap.set_bad("#f2f2f2")
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax.set_title(f"B0 {site}", fontsize=12, fontweight="bold")
    ax.set_xticks(np.arange(len(MODES)), ["DOM", "SoM", "Vision", "Phantom\nSoM", "Phantom\nDOM"])
    ax.set_yticks(np.arange(len(CATEGORIES)), CATEGORY_LABELS)
    ax.tick_params(axis="both", length=0)
    for i in range(len(CATEGORIES)):
        for j in range(len(MODES)):
            value = matrix[i, j]
            color = "white" if np.isfinite(value) and value >= vmax * 0.58 else "#222222"
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=8.2, color=color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(MODES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CATEGORIES), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    return im


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), constrained_layout=True)
    ims = [draw_site(ax, site, vmax=32.0) for ax, site in zip(axes, ["classifieds", "reddit"])]
    fig.colorbar(ims[0], ax=axes, shrink=0.82, label="Adjusted success rate (%)")
    fig.suptitle("Codex Audit Category x Observation Mode", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        -0.03,
        "* Phantom-SoM uses stale pre-rederive fallback summaries; reddit Phantom-DOM is partial live data while the run is still in progress.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
