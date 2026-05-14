#!/usr/bin/env python3
"""[Micro 2f] Micro dimension — first-divergence step distribution.

Output:
- results/phantom_paper/figures/fig2f_first_divergence.png

Micro 2f: first step where paired trajectories differ in action_type or
element_id, binned by early/mid/late divergence timing.

See docs/checkpoints/paper_planning.md §3 Micro dimension framework.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    venv_python = Path(__file__).resolve().parents[3] / ".venv/bin/python3"
    if venv_python.exists() and Path(sys.executable) != venv_python:
        os.execv(str(venv_python), [str(venv_python), *sys.argv])
    raise

try:
    from scripts.analysis.lib.run_registry import PAPER_MODES, get_cells
except ModuleNotFoundError:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from scripts.analysis.lib.run_registry import PAPER_MODES, get_cells

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/phantom_paper/figures/fig2f_first_divergence.png"

PAIR_SPECS = [
    ("DOM", "P-text", "DOM↔P-text"),
    ("DOM", "P-prompt", "DOM↔P-prompt"),
    ("P-text", "P-SoM", "P-text↔P-SoM"),
    ("P-prompt", "P-SoM", "P-prompt↔P-SoM"),
    ("DOM", "P-SoM", "DOM↔P-SoM"),
]
PANELS = [
    ("B0", "classifieds", "B0 cls"),
    ("B0", "reddit", "B0 red"),
    ("B1", "classifieds", "B1 cls"),
    ("B1", "reddit", "B1 red"),
]
BIN_LABELS = ["step 0", "1-3", "4-10", "11+", "no divergence"]
BIN_COLORS = ["#7c2d12", "#f58518", "#f2cf5b", "#8ab17d", "#bdbdbd"]


def task_id(path: Path, suffix: str) -> int:
    match = re.search(r"task_(\d+)_" + suffix, path.name)
    if not match:
        raise ValueError(path.name)
    return int(match.group(1))


def read_steps(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"[warn] malformed JSONL ignored: {path}", file=sys.stderr)
    return rows


def read_successes(ep_dir: Path) -> dict[int, bool]:
    out: dict[int, bool] = {}
    for path in sorted(ep_dir.glob("*_summary_v2.json")):
        tid = task_id(path, "summary")
        rec = json.loads(path.read_text())
        out[tid] = bool(rec.get("success", False))  # §139.8: adjusted_success retired
    return out


def action_signature(step: dict[str, Any]) -> tuple[Any, Any]:
    action = step.get("action")
    if not isinstance(action, dict):
        action = {}
    action_type = step.get("action_type") or action.get("action_type")
    element_id = step.get("element_id") if step.get("element_id") is not None else action.get("element_id")
    return action_type, element_id


def first_divergence(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> tuple[int, bool]:
    n = min(len(left), len(right))
    for idx in range(n):
        if action_signature(left[idx]) != action_signature(right[idx]):
            return idx, True
    if len(left) != len(right):
        return n, True
    return n, False


def bin_index(step: int, diverged: bool) -> int:
    if not diverged:
        return 4
    if step == 0:
        return 0
    if step <= 3:
        return 1
    if step <= 10:
        return 2
    return 3


def load_mode_steps(baseline: str, site: str) -> tuple[dict[str, dict[int, list[dict[str, Any]]]], dict[str, dict[int, bool]]]:
    steps_by_mode: dict[str, dict[int, list[dict[str, Any]]]] = {}
    success_by_mode: dict[str, dict[int, bool]] = {}
    for cell in get_cells(baseline=baseline, site=site):
        mode_steps: dict[int, list[dict[str, Any]]] = {}
        for path in sorted(cell.episodes_dir.glob(f"{site}_task_*_steps_v2.jsonl")):
            mode_steps[task_id(path, "steps")] = read_steps(path)
        if mode_steps:
            steps_by_mode[cell.mode] = mode_steps
            success_by_mode[cell.mode] = read_successes(cell.episodes_dir)
    return steps_by_mode, success_by_mode


def pair_distribution(
    steps_by_mode: dict[str, dict[int, list[dict[str, Any]]]],
    left: str,
    right: str,
) -> tuple[list[int], list[int], int]:
    if left not in steps_by_mode or right not in steps_by_mode:
        return [0, 0, 0, 0, 0], [], 0
    task_ids = sorted(set(steps_by_mode[left]) & set(steps_by_mode[right]))
    counts = [0, 0, 0, 0, 0]
    divergent_steps: list[int] = []
    for tid in task_ids:
        step, diverged = first_divergence(steps_by_mode[left][tid], steps_by_mode[right][tid])
        counts[bin_index(step, diverged)] += 1
        if diverged:
            divergent_steps.append(step)
    return counts, divergent_steps, len(task_ids)


def solved_delta_note(
    steps_by_mode: dict[str, dict[int, list[dict[str, Any]]]],
    success_by_mode: dict[str, dict[int, bool]],
    left: str,
    right: str,
) -> str | None:
    if left not in steps_by_mode or right not in steps_by_mode or left not in success_by_mode or right not in success_by_mode:
        return None
    task_ids = sorted(set(steps_by_mode[left]) & set(steps_by_mode[right]) & set(success_by_mode[left]) & set(success_by_mode[right]))
    sym = [tid for tid in task_ids if success_by_mode[left][tid] != success_by_mode[right][tid]]
    if not sym:
        return None
    steps = [first_divergence(steps_by_mode[left][tid], steps_by_mode[right][tid])[0] for tid in sym]
    early = 100.0 * sum(step <= 3 for step in steps) / len(steps)
    return f"Δsolve N={len(sym)}, med={statistics.median(steps):.0f}, ≤3={early:.0f}%"


def draw_panel(ax: plt.Axes, baseline: str, site: str, title: str) -> None:
    steps_by_mode, success_by_mode = load_mode_steps(baseline, site)
    y = np.arange(len(PAIR_SPECS))
    lefts = np.zeros(len(PAIR_SPECS))
    for i, (left, right, label) in enumerate(PAIR_SPECS):
        counts, divergent_steps, n = pair_distribution(steps_by_mode, left, right)
        if n == 0:
            print(f"[warn] {baseline} {site} missing first-divergence pair {left}<->{right}", file=sys.stderr)
            ax.text(2, i, "N/A pending", va="center", ha="left", fontsize=8, color="#666666")
            continue
        pct = [100.0 * c / n for c in counts]
        start = 0.0
        for value, color in zip(pct, BIN_COLORS):
            ax.barh(i, value, left=start, color=color, edgecolor="white", linewidth=0.6)
            start += value
        median = statistics.median(divergent_steps) if divergent_steps else float("nan")
        note = f"N={n}, med={median:.0f}" if divergent_steps else f"N={n}, no div"
        if baseline == "B0" and site == "reddit" and left == "P-text" and right == "P-SoM":
            extra = solved_delta_note(steps_by_mode, success_by_mode, left, right)
            if extra:
                note += f"\n{extra}"
        ax.text(102, i, note, va="center", ha="left", fontsize=7.5, color="#333333")
        lefts[i] = start
    ax.set_yticks(y, [label for _left, _right, label in PAIR_SPECS])
    ax.invert_yaxis()
    ax.set_xlim(0, 125)
    ax.set_xlabel("% paired tasks")
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.grid(axis="x", color="#e8e8e8", linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9.5, "figure.dpi": 150})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), sharex=True)
    for ax, (baseline, site, title) in zip(axes.ravel(), PANELS):
        draw_panel(ax, baseline, site, title)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in BIN_COLORS]
    fig.legend(handles, BIN_LABELS, loc="upper center", ncol=len(BIN_LABELS), frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Micro 2f — first-divergence timing by axis-control pair", fontsize=14, fontweight="bold", y=1.035)
    fig.text(0.5, 0.02, "First divergence = first step where action_type or element_id differs; earlier termination counts at truncation step.", ha="center", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
