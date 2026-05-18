#!/usr/bin/env python3
"""[Micro 2e] Micro dimension — cross-site validity ratio.

Output:
- results/phantom_paper/figures/fig2e_cross_site_validity.png

Micro 2e: ratio of reddit effect magnitude to classifieds effect magnitude
for micro decision metrics across axis contrasts.

See docs/checkpoints/paper_planning.md §3 Micro dimension framework.
"""

from __future__ import annotations

import json
import math
import os
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
    from scripts.analysis.lib.run_registry import BASELINES, PAPER_MODES, get_cells
except ModuleNotFoundError:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from scripts.analysis.lib.run_registry import BASELINES, PAPER_MODES, get_cells

ROOT = Path(__file__).resolve().parents[3]
IN_JSON = ROOT / "docs/analysis/cross_sites/axis1_microbehavior.json"
OUT = ROOT / "results/phantom_paper/figures/fig2e_cross_site_validity.png"

AXES = [
    ("axis_1_text", "axis 1\nDOM→P-text"),
    ("axis_2_prompt", "axis 2\nP-text→P-SoM"),
    ("compound_dom_to_psom", "compound\nDOM→P-SoM"),
]
# 3-model deep-update 2026-05-18: B2 color = #17becf (matplotlib tab cyan,
# distinct from B0 blue / B1 purple, and from MODE_COLORS Vision green
# #54a24b to avoid visual collision with baseline column).
BASELINE_COLORS = {"B0": "#4c78a8", "B1": "#9467bd", "B2": "#17becf"}


def load_json() -> dict[str, Any]:
    return json.loads(IN_JSON.read_text())


def abs_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return abs(float(value))
    except (TypeError, ValueError):
        return None


def metric_vector(row: dict[str, Any], keyword_norm: float) -> list[float]:
    vals: list[float] = []
    url = abs_float(row.get("url_decision_divergence"))
    target = abs_float(row.get("target_hit_rate_diff"))
    keyword = abs_float(row.get("max_keyword_repeat_diff"))
    first = abs_float(row.get("first_action_divergence_rate"))
    for value in (url, target, first):
        if value is not None and not math.isnan(value):
            vals.append(value)
    if keyword is not None and keyword_norm > 0:
        vals.append(keyword / keyword_norm)
    return vals


def effect_magnitude(data: dict[str, Any], baseline: str, site: str, contrast: str, keyword_norm: float) -> float | None:
    row = data.get("axis_contrasts", {}).get(baseline, {}).get(site, {}).get(contrast, {}) or {}
    if row.get("skipped") or row.get("n", 0) == 0:
        return None
    vals = metric_vector(row, keyword_norm)
    if not vals:
        return None
    return float(np.mean(vals))


def keyword_norm_for(data: dict[str, Any], baseline: str, contrast: str) -> float:
    vals = []
    for site in ("classifieds", "reddit"):
        row = data.get("axis_contrasts", {}).get(baseline, {}).get(site, {}).get(contrast, {}) or {}
        value = abs_float(row.get("max_keyword_repeat_diff"))
        if value is not None:
            vals.append(value)
    return max(vals, default=1.0)


def ratio_for(data: dict[str, Any], baseline: str, contrast: str) -> float | None:
    norm = keyword_norm_for(data, baseline, contrast)
    red = effect_magnitude(data, baseline, "reddit", contrast, norm)
    cls = effect_magnitude(data, baseline, "classifieds", contrast, norm)
    if red is None or cls is None or cls == 0:
        print(f"[warn] {baseline} missing cross-site validity inputs for {contrast}", file=sys.stderr)
        return None
    return red / cls


def interpretation(value: float) -> str:
    if value > 1.15:
        return "reddit-amplified"
    if value < 0.85:
        return "cls-amplified"
    return "symmetric"


def main() -> None:
    data = load_json()
    # 3-model deep-update 2026-05-18: iterate BASELINES registry (was hardcoded
    # ("B0","B1")). For missing baseline (e.g., B2 pre-Phase-1a-fire), ratio_for
    # warns + returns None → bar renders as N/A hatched per existing graceful-skip.
    ratios = {
        baseline: [ratio_for(data, baseline, contrast) for contrast, _label in AXES]
        for baseline in BASELINES
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9.5, "figure.dpi": 150})
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    x = np.arange(len(AXES))
    # 3-model deep-update 2026-05-18: bar width + offsets computed for N baselines
    # so 2→3→N expansion stays evenly spaced without touching layout each time.
    n_b = len(BASELINES)
    width = min(0.34, 0.85 / max(n_b, 1))
    offsets = {b: (i - (n_b - 1) / 2) * width for i, b in enumerate(BASELINES)}
    for baseline in BASELINES:
        values = [0.0 if v is None else v for v in ratios[baseline]]
        bars = ax.bar(x + offsets[baseline], values, width=width, label=baseline, color=BASELINE_COLORS[baseline], edgecolor="white", linewidth=0.8)
        for bar, value in zip(bars, ratios[baseline]):
            if value is None:
                bar.set_color("#d9d9d9")
                bar.set_hatch("//")
                ax.text(bar.get_x() + bar.get_width() / 2, 0.07, "N/A\npending", ha="center", va="bottom", fontsize=7, color="#666666")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.05, f"{value:.2f}×\n{interpretation(value)}", ha="center", va="bottom", fontsize=7.5)

    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
    ax.text(len(AXES) - 0.05, 1.03, "1.0 = symmetric generalization", ha="right", va="bottom", fontsize=8, color="#333333")
    cv = data.get("cross_site_validity", {}) or {}
    note = (
        "Aggregates Micro 2a URL divergence, 2b target-hit abs diff, 2c normalized keyword-repeat diff, "
        "and 2d first-action divergence. "
        f"Existing axis-1 decision/macro ratios: red={cv.get('B0_reddit_ratio', float('nan')):.2f}, "
        f"cls={cv.get('B0_classifieds_ratio', float('nan')):.2f}."
    )
    ax.set_title("Micro 2e — reddit/classifieds effect ratio", fontsize=13, fontweight="bold")
    ax.set_xticks(x, [label for _contrast, label in AXES])
    ax.set_ylabel("Effect ratio (reddit / classifieds)")
    ymax = max([v for vals in ratios.values() for v in vals if v is not None] + [1.0])
    ax.set_ylim(0, ymax + 0.7)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    fig.text(0.5, -0.01, note, ha="center", fontsize=8.3, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
