#!/usr/bin/env python3
"""[Macro 1c] Macro dimension — strategy-gradient visualization.

Output:
- results/phantom_paper/figures/fig1c_strategy_gradient.png

Macro 1c: search-loop, type, scroll, and self-correction strategy gradient.

See docs/checkpoints/paper_planning.md §3 Macro dimension framework.

Strategy-gradient bars for reddit and classifieds observation variants.

Both rows are computed live from available step JSONL files (full 5-mode B0
data: B0_3mode_<site> + B0_phantom_som_<site> + B0_phantom_text_<site>).
The §103 / N=48 anchor values are kept as a sanity-check reference but are
no longer used for plotting; live values are reported instead.
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
OUT = ROOT / "results/phantom_paper/figures/fig1c_strategy_gradient.png"

MODES = PAPER_MODES
METRICS = ["Search-loop %", "Type action %", "Scroll action %", "Self-correction / ep"]
COLORS = {
    "DOM": "#4c78a8",
    "Vision": "#54a24b",
    "SoM": "#f58518",
    "P-SoM": "#b279a2",
    "P-text": "#e45756",
    "P-prompt": "#9467bd",
}

# Verified notes:
# - Full reddit gradient (§103): DOM search 27/type 38/steps 12.7;
#   Phantom-SoM search 20/type 32/steps 9.9; SoM search 12/type 23/steps 8.1.
# - N=48 ablation/user context: DOM search-loop 22.7; Phantom-SoM and
#   P-text search-loop 10.8; 5/5 macro metrics P-text = Phantom-SoM.
# - N=26 table in §103 supplies type/scroll/self-correction anchors for the
#   ablation subset; we keep P-text equal to Phantom-SoM per the N=48 note.
REDDIT_VERIFIED = {
    "Search-loop %": {
        "DOM": 22.7,
        "Vision": None,
        "SoM": 12.0,
        "P-SoM": 10.8,
        "P-text": 10.8,
        "P-prompt": None,
    },
    "Type action %": {
        "DOM": 40.2,
        "Vision": None,
        "SoM": 23.0,
        "P-SoM": 20.4,
        "P-text": 20.4,
        "P-prompt": None,
    },
    "Scroll action %": {
        "DOM": 15.2,
        "Vision": None,
        "SoM": None,
        "P-SoM": 26.2,
        "P-text": 26.2,
        "P-prompt": None,
    },
    "Self-correction / ep": {
        "DOM": 0.31,
        "Vision": None,
        "SoM": None,
        "P-SoM": 0.35,
        "P-text": 0.35,
        "P-prompt": None,
    },
}

STEP_DIRS: dict[str, dict[str, dict[str, Path]]] = {
    baseline: {
        site: {cell.mode: cell.episodes_dir for cell in get_cells(baseline=baseline, site=site)}
        for site in ("reddit", "classifieds")
    }
    for baseline in ("B0", "B1")
}
ROW_SPECS = [
    ("B0", "reddit", "B0 Reddit"),
    ("B0", "classifieds", "B0 Classifieds"),
    ("B1", "reddit", "B1 Reddit"),
    ("B1", "classifieds", "B1 Classifieds"),
]
SEARCH_MARKERS = {"reddit": ("/search",), "classifieds": ("page=search", "/search")}


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
    match = re.search(r"task_(\d+)_steps", path.name)
    if not match:
        raise ValueError(f"Cannot parse task id from {path}")
    return int(match.group(1))


def is_search_url(site: str, url: str) -> bool:
    return any(marker in url for marker in SEARCH_MARKERS[site])


def compute_available_metrics(site: str, step_dirs: dict[str, Path]) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for mode in MODES:
        ep_dir = step_dirs.get(mode)
        if ep_dir is None:
            out[mode] = {metric: None for metric in METRICS}
            continue
        files = sorted(ep_dir.glob(f"{site}_task_*_steps_v2.jsonl"))
        if not files:
            print(f"[warn] {site} {mode}: no step JSONL found; plotting n/a or verified anchor", file=sys.stderr)
            out[mode] = {metric: None for metric in METRICS}
            continue
        total = typed = scrolled = 0
        search_loop_eps = 0
        selfcorr = 0
        task_ids = set()
        for path in files:
            task_ids.add(step_task_id(path))
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
                if is_search_url(site, url) or (action_type == "type" and is_search_url(site, next_url)):
                    search_steps += 1
                action = step.get("action") or {}
                thought = action.get("thought", "").lower() if isinstance(action, dict) else ""
                if any(token in thought for token in ("mistake", "wrong", "try again", "go back")):
                    selfcorr += 1
            if search_steps >= 2:
                search_loop_eps += 1
        n = len(task_ids)
        if total and n:
            out[mode] = {
                "Search-loop %": 100.0 * search_loop_eps / n,
                "Type action %": 100.0 * typed / total,
                "Scroll action %": 100.0 * scrolled / total,
                "Self-correction / ep": selfcorr / n,
            }
        else:
            out[mode] = {metric: None for metric in METRICS}
    return out


def all_values() -> dict[tuple[str, str], dict[str, dict[str, float | None]]]:
    out: dict[tuple[str, str], dict[str, dict[str, float | None]]] = {}
    for baseline, baseline_dirs in STEP_DIRS.items():
        for site, site_dirs in baseline_dirs.items():
            metrics = compute_available_metrics(site, site_dirs)
            out[(baseline, site)] = metrics
            if baseline == "B0" and site == "reddit":
                # Sanity check live values against §103 N=48 anchors (warn-only)
                for mode in MODES:
                    live = metrics.get(mode, {}).get("Search-loop %")
                    anchor = REDDIT_VERIFIED["Search-loop %"].get(mode)
                    if live is not None and anchor is not None and abs(live - anchor) > 3.0:
                        print(
                            f"[warn] B0 reddit {mode} live search-loop {live:.1f}% differs from "
                            f"§103 N=48 anchor {anchor:.1f}% (>3pp)",
                            file=sys.stderr,
                        )
    return out


def print_verbose(values: dict[tuple[str, str], dict[str, dict[str, float | None]]]) -> None:
    for (baseline, site), site_metrics in values.items():
        prefix = f"{baseline} {'red' if site == 'reddit' else 'cls'}"
        for mode in MODES:
            metrics = site_metrics.get(mode, {})
            search = metrics.get("Search-loop %")
            typed = metrics.get("Type action %")
            scroll = metrics.get("Scroll action %")
            selfcorr = metrics.get("Self-correction / ep")
            fmt = lambda value: "n/a" if value is None else f"{value:.2f}"
            print(
                f"{prefix} {mode}: search_loop={fmt(search)} "
                f"type={fmt(typed)} scroll={fmt(scroll)} selfcorr={fmt(selfcorr)}"
            )


def draw_panel(ax: plt.Axes, row_idx: int, metric: str, values: dict[str, dict[str, float | None]]) -> None:
    metric_values = [values.get(mode, {}).get(metric) for mode in MODES]
    heights = [0 if value is None else value for value in metric_values]
    x = np.arange(len(MODES))
    bars = ax.bar(x, heights, color=[COLORS[mode] for mode in MODES], width=0.68)
    for bar, mode, value in zip(bars, MODES, metric_values):
        if value is None:
            bar.set_alpha(0.18)
            bar.set_hatch("//")
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0.2,
                "n/a",
                ha="center",
                va="bottom",
                fontsize=7.0,
                color="#666666",
            )
            continue
        label = f"{value:.1f}" if "Self" not in metric else f"{value:.2f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(0.35, 0.02 * max(heights or [1])),
            label,
            ha="center",
            va="bottom",
            fontsize=7.0,
        )
    if row_idx == 0:
        ax.set_title(metric, fontsize=10.0, fontweight="bold")
    ax.set_xticks(x, MODES, rotation=0, fontsize=7.0)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ymax = max([v for v in metric_values if v is not None] or [1])
    ax.set_ylim(0, ymax * (1.28 if "Self" not in metric else 1.45) + (1.0 if "Self" not in metric else 0.05))
    if metric == "Search-loop %":
        ax.set_ylabel("Percent")
    elif metric == "Self-correction / ep":
        ax.set_ylabel("Count / ep")


def main() -> None:
    values = all_values()
    print_verbose(values)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8.5, "figure.dpi": 150})
    fig, axes = plt.subplots(4, 4, figsize=(15, 13.5))

    for row, (baseline, site, _label) in enumerate(ROW_SPECS):
        site_metrics = values[(baseline, site)]
        for col, metric in enumerate(METRICS):
            draw_panel(axes[row, col], row, metric, site_metrics)

    # Row labels (left side)
    n_rows = len(ROW_SPECS)
    for idx, (_baseline, _site, label) in enumerate(ROW_SPECS):
        # Y position at vertical center of each row in figure coords
        y = 1.0 - (idx + 0.5) / n_rows * 0.85 - 0.04
        fig.text(0.012, y, label, rotation=90, va="center", ha="center", fontsize=11, fontweight="bold")

    fig.suptitle("Strategy Gradient: Representation Changes Exploration Shape (B0 + B1)", fontsize=13, fontweight="bold")
    fig.text(
        0.5,
        0.012,
        "All rows live-computed from step JSONL (B0 5-mode + B1 partial). "
        "B1 cls includes P-SoM; B1 reddit phantom modes pending (n/a hatched). "
        "OSClass search detection uses 'page=search' / '/search' and measures search-page coverage; "
        "cross-site comparison invalid for the search-loop column.",
        ha="center",
        fontsize=7.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0.035, 0.04, 1, 0.95))
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
