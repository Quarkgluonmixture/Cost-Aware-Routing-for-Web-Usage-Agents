#!/usr/bin/env python3
"""[Outcome 0e] Outcome dimension — per-category adjusted SR heatmap.

Output:
- results/phantom_paper/figures/fig0e_category_mode_heatmap.png

Outcome 0e: category × observation-mode success-rate evidence.

See docs/checkpoints/paper_planning.md §3 Outcome dimension framework.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from scripts.analysis.lib.run_registry import PAPER_MODES, get_cells
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from scripts.analysis.lib.run_registry import PAPER_MODES, get_cells

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/phantom_paper/figures/fig0e_category_mode_heatmap.png"

CATEGORIES = ["A", "B", "C", "D"]
CATEGORY_LABELS = [
    "A\ntext-only",
    "B\nref-image",
    "C\npage-screen",
    "D\nuncertain",
]
MODES = PAPER_MODES
MODE_LABELS = {"DOM": "DOM", "SoM": "SoM", "Vision": "Vision", "P-SoM": "P-SoM", "P-text": "P-text", "P-prompt": "P-prompt"}


def _panel(key: str, baseline: str, site: str, expected: int, audit: Path) -> dict:
    return {
        "key": key,
        "baseline": baseline,
        "site": site,
        "expected": expected,
        "audit": audit,
        "modes": {cell.mode: cell.episodes_dir for cell in get_cells(baseline=baseline, site=site)},
    }

PANELS = [
    _panel("B0 classifieds", "B0", "classifieds", 234, ROOT / "docs/analysis/cross_sites/codex_audit_classifieds.json"),
    _panel("B0 reddit", "B0", "reddit", 210, ROOT / "docs/analysis/cross_sites/codex_audit_reddit.json"),
    _panel("B1 classifieds", "B1", "classifieds", 234, ROOT / "docs/analysis/cross_sites/codex_audit_classifieds.json"),
    _panel("B1 reddit", "B1", "reddit", 210, ROOT / "docs/analysis/cross_sites/codex_audit_reddit.json"),
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
    ax.set_xticks(np.arange(len(MODES)), [MODE_LABELS[m] for m in MODES], fontsize=8.0)
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
    fig, axes = plt.subplots(1, 4, figsize=(25.0, 5.6), constrained_layout=True)
    ims = [draw_panel(ax, panel, vmax=32.0) for ax, panel in zip(axes, PANELS)]
    fig.colorbar(ims[0], ax=axes, shrink=0.82, label="Adjusted success rate (%)")
    fig.suptitle("Codex Audit Category x Observation Mode (B0 + B1)", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        -0.03,
        "B0/B1 cells and pending modes come from run_manifest.yaml; P-SoM/P-text/P-prompt are omitted until marked paper-grade.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
