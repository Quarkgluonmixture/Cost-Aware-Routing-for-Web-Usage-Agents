#!/usr/bin/env python3
"""[Layer 2 visualization] Micro behavior decision divergence heatmap.

Output:
- results/phantom_paper/figures/fig2_micro_divergence_heatmap.png
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[3]
IN_JSON = ROOT / "docs/analysis/cross_sites/axis1_microbehavior.json"
OUT_LEGACY = ROOT / "results/phantom_paper/figures/fig2_micro_divergence_heatmap.png"
OUT_B0 = ROOT / "results/phantom_paper/figures/fig2_micro_divergence_heatmap_B0.png"
OUT_B1 = ROOT / "results/phantom_paper/figures/fig2_micro_divergence_heatmap_B1.png"

SITES = ["reddit", "classifieds"]
CONTRASTS = [
    ("axis_1_text", "axis 1\ntext"),
    ("axis_2_prompt", "axis 2\nprompt"),
    ("axis_3_image", "axis 3\nimage"),
    ("compound_dom_to_psom", "DOM->\nP-SoM"),
    ("endpoint_dom_to_som", "DOM->\nSoM"),
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def as_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def matrix(data: dict[str, Any], baseline: str, key: str, *, scale: float = 1.0, absolute: bool = False) -> np.ndarray:
    rows = []
    for site in SITES:
        row = []
        for contrast, _label in CONTRASTS:
            cell = data.get("axis_contrasts", {}).get(baseline, {}).get(site, {}).get(contrast, {}) or {}
            if cell.get("skipped") or cell.get("n", 0) == 0:
                row.append(float("nan"))
                continue
            value = as_float(cell.get(key))
            if absolute and not math.isnan(value):
                value = abs(value)
            row.append(value * scale if not math.isnan(value) else value)
        rows.append(row)
    return np.array(rows, dtype=float)


def annotate(ax: plt.Axes, values: np.ndarray, labels: list[list[str]], *, dark_threshold: float | None = None) -> None:
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            color = "#111111"
            if dark_threshold is not None and not math.isnan(value) and value < dark_threshold:
                color = "white"
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=7.5, color=color, linespacing=1.25)


def configure_axes(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.set_xticks(range(len(CONTRASTS)), [label for _key, label in CONTRASTS], fontsize=8.2)
    ax.set_yticks(range(len(SITES)), SITES, fontsize=8.5)
    ax.tick_params(length=0)


def render_baseline(data: dict[str, Any], baseline: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

    overlap_cmap = LinearSegmentedColormap.from_list("overlap", ["#b91c1c", "#fef2f2", "#dbeafe", "#1d4ed8"])
    divergence_cmap = LinearSegmentedColormap.from_list("divergence", ["#eff6ff", "#fee2e2", "#b91c1c"])

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 6.8), constrained_layout=True)

    def label_or_na(value: float, fmt: str) -> str:
        if math.isnan(value):
            return "n/a"
        return fmt.format(value)

    url_j = matrix(data, baseline, "url_jaccard_mean")
    url_div = matrix(data, baseline, "url_decision_divergence", scale=100.0)
    im0 = axes[0, 0].imshow(url_j, vmin=0.0, vmax=1.0, cmap=overlap_cmap)
    configure_axes(axes[0, 0], "URL-path Jaccard (lower = more divergence)")
    annotate(
        axes[0, 0],
        url_j,
        [[("n/a" if math.isnan(url_j[i, j]) else f"{url_j[i, j]:.3f}\n{url_div[i, j]:.1f}pp div") for j in range(url_j.shape[1])] for i in range(url_j.shape[0])],
        dark_threshold=0.42,
    )
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.045, pad=0.02)

    target = matrix(data, baseline, "target_hit_rate_diff_pct_pts")
    finite_target = np.abs(target)[~np.isnan(target)]
    lim = max(1.0, float(finite_target.max())) if finite_target.size else 1.0
    im1 = axes[0, 1].imshow(target, cmap="RdBu", norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim))
    configure_axes(axes[0, 1], "Target-hit diff (right-minus-left, pp)")
    annotate(axes[0, 1], target, [[label_or_na(target[i, j], "{:+.1f}pp") for j in range(target.shape[1])] for i in range(target.shape[0])])
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.045, pad=0.02)

    first = matrix(data, baseline, "first_action_divergence_rate", scale=100.0)
    finite_first = first[~np.isnan(first)]
    fmax = max(1.0, float(finite_first.max())) if finite_first.size else 1.0
    im2 = axes[1, 0].imshow(first, vmin=0.0, vmax=fmax, cmap=divergence_cmap)
    configure_axes(axes[1, 0], "First-action divergence (%)")
    annotate(axes[1, 0], first, [[label_or_na(first[i, j], "{:.1f}%") for j in range(first.shape[1])] for i in range(first.shape[0])])
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.045, pad=0.02)

    repeat = matrix(data, baseline, "max_keyword_repeat_diff")
    finite_repeat = np.abs(repeat)[~np.isnan(repeat)]
    repeat_lim = max(0.1, float(finite_repeat.max())) if finite_repeat.size else 0.1
    im3 = axes[1, 1].imshow(repeat, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-repeat_lim, vcenter=0.0, vmax=repeat_lim))
    configure_axes(axes[1, 1], "Keyword-repeat diff (right-minus-left)")
    annotate(axes[1, 1], repeat, [[label_or_na(repeat[i, j], "{:+.2f}") for j in range(repeat.shape[1])] for i in range(repeat.shape[0])])
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.045, pad=0.02)

    fig.suptitle(
        f"Layer 2 Micro Behavior ({baseline}) — per-task URL-path overlap (lower = more decision divergence)",
        fontsize=14,
        fontweight="bold",
    )
    note = "Source: docs/analysis/cross_sites/axis1_microbehavior.json; cascade contrasts are right-minus-left except Jaccard."
    if baseline == "B1":
        note += " B1 P-text data pending; only DOM↔P-SoM compound + image contrast computed (cls only)."
    fig.text(0.5, -0.01, note, ha="center", fontsize=8.5, color="#555555")
    fig.savefig(out_path, bbox_inches="tight")
    print(out_path)


def main() -> None:
    data = read_json(IN_JSON)
    render_baseline(data, "B0", OUT_B0)
    render_baseline(data, "B1", OUT_B1)
    # Maintain legacy filename: copy B0 file content (or re-render to legacy path) for old draft refs
    render_baseline(data, "B0", OUT_LEGACY)


if __name__ == "__main__":
    main()
