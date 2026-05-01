#!/usr/bin/env python3
"""[Micro 2c] Micro dimension — search-keyword reuse distribution.

Output:
- results/phantom_paper/figures/fig2c_keyword_repeat.png

Micro 2c: max repeated typed keyword/query per trajectory, shown as
per-mode task-level distributions.

See docs/checkpoints/paper_planning.md §3 Micro dimension framework.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
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
OUT = ROOT / "results/phantom_paper/figures/fig2c_keyword_repeat.png"

MODE_COLORS = {
    "DOM": "#4c78a8",
    "SoM": "#f58518",
    "Vision": "#54a24b",
    "P-text": "#e45756",
    "P-prompt": "#9467bd",
    "P-SoM": "#b279a2",
}
PANELS = [
    ("B0", "classifieds", "B0 cls"),
    ("B0", "reddit", "B0 red"),
    ("B1", "classifieds", "B1 cls"),
    ("B1", "reddit", "B1 red"),
]


def task_id(path: Path) -> int:
    match = re.search(r"task_(\d+)_steps", path.name)
    if not match:
        raise ValueError(path.name)
    return int(match.group(1))


def action_type(step: dict[str, Any]) -> str | None:
    action = step.get("action")
    nested = action.get("action_type") if isinstance(action, dict) else None
    return step.get("action_type") or nested


def typed_text(step: dict[str, Any]) -> str | None:
    action = step.get("action")
    if not isinstance(action, dict):
        action = {}
    text = action.get("text") or step.get("text")
    if text is None:
        return None
    text = re.sub(r"\s+", " ", str(text).strip().lower())
    return text or None


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


def max_keyword_repeat(path: Path) -> int:
    keywords = [typed_text(step) for step in read_steps(path) if action_type(step) == "type"]
    counts = Counter(k for k in keywords if k)
    return max(counts.values(), default=0)


def panel_distributions(baseline: str, site: str) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for cell in get_cells(baseline=baseline, site=site):
        files = sorted(cell.episodes_dir.glob(f"{site}_task_*_steps_v2.jsonl"))
        if not files:
            print(f"[warn] {baseline} {site} {cell.mode} missing steps JSONL", file=sys.stderr)
            continue
        seen: set[int] = set()
        values = []
        for path in files:
            tid = task_id(path)
            if tid in seen:
                continue
            seen.add(tid)
            values.append(max_keyword_repeat(path))
        out[cell.mode] = values
    return out


def annotate_delta(ax: plt.Axes, values: dict[str, list[int]], left: str, right: str, y: float, label: str, color: str) -> None:
    if not values.get(left) or not values.get(right):
        return
    delta = float(np.median(values[right]) - np.median(values[left]))
    x0, x1 = PAPER_MODES.index(left) + 1, PAPER_MODES.index(right) + 1
    ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops={"arrowstyle": "<->", "color": color, "lw": 1.1})
    ax.text((x0 + x1) / 2, y + 0.25, f"{label} Δmed {delta:+.1f}", ha="center", va="bottom", fontsize=7.3, color=color)


def draw_panel(ax: plt.Axes, baseline: str, site: str, title: str) -> None:
    dists = panel_distributions(baseline, site)
    data = [dists.get(mode, []) for mode in PAPER_MODES]
    positions = np.arange(1, len(PAPER_MODES) + 1)
    non_empty = [vals for vals in data if vals]
    if non_empty:
        bp = ax.boxplot(data, positions=positions, widths=0.56, patch_artist=True, showmeans=False, whis=1.5)
        for patch, mode, vals in zip(bp["boxes"], PAPER_MODES, data):
            patch.set_facecolor(MODE_COLORS[mode] if vals else "#d9d9d9")
            patch.set_alpha(0.78 if vals else 0.35)
            if not vals:
                patch.set_hatch("//")
        for key in ("medians", "whiskers", "caps"):
            for item in bp[key]:
                item.set_color("#333333")
                item.set_linewidth(1.0)
        for flier in bp["fliers"]:
            flier.set_marker(".")
            flier.set_alpha(0.35)
    for x, mode, vals in zip(positions, PAPER_MODES, data):
        if vals:
            ax.text(x, np.median(vals) + 0.12, f"med {np.median(vals):.0f}", ha="center", va="bottom", fontsize=7)
        else:
            ax.text(x, 0.25, "N/A\npending", ha="center", va="bottom", fontsize=7, color="#666666")
            print(f"[warn] {baseline} {site} missing max_keyword_repeat distribution for {mode}", file=sys.stderr)
    ymax = max((max(vals) for vals in non_empty), default=1)
    annotate_delta(ax, dists, "DOM", "P-text", ymax + 1.0, "axis 1", "#2f5f9f")
    annotate_delta(ax, dists, "P-text", "P-SoM", ymax + 2.3, "axis 2", "#c56b1d")
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.set_xticks(positions, PAPER_MODES, rotation=30, ha="right")
    ax.set_ylim(0, ymax + 4.0)
    ax.set_ylabel("Max repeated typed keyword/query")
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9.5, "figure.dpi": 150})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), sharey=False)
    for ax, (baseline, site, title) in zip(axes.ravel(), PANELS):
        draw_panel(ax, baseline, site, title)
    fig.suptitle("Micro 2c — search-keyword reuse per trajectory", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.02, "Distribution computed from episodes/*_steps_v2.jsonl for visualization; each point is one task trajectory.", ha="center", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
