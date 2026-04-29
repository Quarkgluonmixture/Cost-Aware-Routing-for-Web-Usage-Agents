#!/usr/bin/env python3
"""[Layer 1/2 supporting] Mechanism schematic for 3-axis cascade controls.

Output:
- results/phantom_paper/figures/fig1ab_cascade_diamond.png

Schematic 3-axis cascade diamond: text-payload, prompt, and image axes.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results/visualwebarena/phase1"
AXIS_JSON = ROOT / "docs/analysis/cross_sites/axis_effect_size.json"
OUT = ROOT / "results/phantom_paper/figures/fig1ab_cascade_diamond.png"

EXPECTED_N = {"reddit": 210, "classifieds": 234}
SITE_SHORT = {"reddit": "red", "classifieds": "cls"}

STEP_DIRS = {
    "reddit": {
        "DOM": RESULTS / "B0_3mode_reddit_20260422/phase1_dom_router_0/episodes",
        "P-text": RESULTS / "B0_phantom_dom_reddit_20260427/phase1_phantom_dom_router_0/episodes",
        "P-SoM": RESULTS / "B0_phantom_reddit_20260428/phase1_phantom_som_router_0/episodes",
    },
    "classifieds": {
        "DOM": RESULTS / "B0_3mode_classifieds_20260413/phase1_dom_router_0/episodes",
        "P-text": RESULTS / "B0_phantom_dom_classifieds_20260427/phase1_phantom_dom_router_0/episodes",
        "P-SoM": RESULTS / "B0_phantom_classifieds_20260426/phase1_phantom_som_router_0/episodes",
    },
}

SEARCH_MARKERS = {"reddit": ("/search",), "classifieds": ("page=search", "/search")}


def task_id(path: Path, suffix: str) -> int:
    match = re.search(r"task_(\d+)_" + suffix, path.name)
    if not match:
        raise ValueError(path.name)
    return int(match.group(1))


def read_steps(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def action_type(step: dict[str, Any]) -> str | None:
    action = step.get("action")
    nested = action.get("action_type") if isinstance(action, dict) else None
    return step.get("action_type") or nested


def mode_metrics(site: str, ep_dir: Path) -> dict[str, float | int | None]:
    files = sorted(ep_dir.glob(f"{site}_task_*_steps_v2.jsonl"))
    if not files:
        return {"n": 0, "search_loop_pct": None, "finish_rate_pct": None, "n_steps_mean": None}

    seen: set[int] = set()
    loops = 0
    finishes = 0
    total_steps = 0
    for path in files:
        tid = task_id(path, "steps")
        if tid in seen:
            print(f"[warn] duplicate steps ignored: {path}", file=sys.stderr)
            continue
        seen.add(tid)
        steps = read_steps(path)
        total_steps += len(steps)
        search_steps = 0
        for i, step in enumerate(steps):
            at = action_type(step)
            url = step.get("obs_url", "") or ""
            next_url = steps[i + 1].get("obs_url", "") if i + 1 < len(steps) else ""
            is_search = any(marker in url for marker in SEARCH_MARKERS[site])
            triggers_search = at == "type" and any(marker in next_url for marker in SEARCH_MARKERS[site])
            if is_search or triggers_search:
                search_steps += 1
        if search_steps >= 2:
            loops += 1
        if steps and action_type(steps[-1]) == "finish":
            finishes += 1

    n = len(seen)
    return {
        "n": n,
        "search_loop_pct": 100.0 * loops / n if n else None,
        "finish_rate_pct": 100.0 * finishes / n if n else None,
        "n_steps_mean": total_steps / n if n else None,
    }


def prompt_status(site: str) -> tuple[str, dict[str, float | int | None] | None]:
    candidates = sorted(RESULTS.glob(f"B0_phantom_prompt_{site}_*/phase1_phantom_prompt_router_0/episodes"))
    ep_dir = candidates[-1] if candidates else RESULTS / f"B0_phantom_prompt_{site}_queued/phase1_phantom_prompt_router_0/episodes"
    files = sorted(ep_dir.glob(f"{site}_task_*_summary_v2.json")) if ep_dir.exists() else []
    n = len({task_id(path, "summary") for path in files})
    expected = EXPECTED_N[site]
    if n >= min(200, expected):
        return "complete", mode_metrics(site, ep_dir)
    if n > 0:
        return f"in progress: {n}/{expected}", None
    return "queued", None


def fmt(value: float | int | None, suffix: str = "", digits: int = 1) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}{suffix}"


def stats_lines(stats_by_site: dict[str, dict[str, float | int | None]]) -> str:
    lines = []
    for site in ["reddit", "classifieds"]:
        stats = stats_by_site.get(site, {})
        n = stats.get("n")
        lines.append(
            f"{SITE_SHORT[site]} N={n}: loop {fmt(stats.get('search_loop_pct'), '%')} | "
            f"finish {fmt(stats.get('finish_rate_pct'), '%')} | steps {fmt(stats.get('n_steps_mean'))}"
        )
    return "\n".join(lines)


def cell(ax, xy, w, h, title, subtitle, face, edge="#333333", title_size=12):
    rect = Rectangle(xy, w, h, facecolor=face, edgecolor=edge, lw=1.4)
    ax.add_patch(rect)
    x, y = xy
    ax.text(x + w / 2, y + h * 0.72, title, ha="center", va="center", fontsize=title_size, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.39, subtitle, ha="center", va="center", fontsize=8.0, color="#333333", linespacing=1.35)


def read_axis_n_note() -> str:
    try:
        data = json.loads(AXIS_JSON.read_text())
    except Exception:
        return "Axis metrics read live from step traces; axis_effect_size.json unavailable."
    checks = data.get("validation", {}).get("n_checks", {})
    bad = [key for key, row in checks.items() if isinstance(row, dict) and not row.get("pass")]
    if bad:
        return f"Live step metrics; axis_effect_size n-check warnings: {len(bad)} cells."
    expected = data.get("validation", {}).get("expected_n", EXPECTED_N)
    return f"Live step metrics aligned with axis_effect_size.json n-checks: reddit N={expected.get('reddit')}, cls N={expected.get('classifieds')}."


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    fig, ax = plt.subplots(figsize=(12.0, 7.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(5.0, 6.62, "3-Axis Cascade Diamond Ablation", ha="center", fontsize=16, fontweight="bold")
    ax.text(
        5.0,
        6.24,
        "Each cell isolates one axis swap from DOM (text-payload x prompt x image-no); "
        "P-text and P-prompt are controlled mismatch phantom modes",
        ha="center",
        fontsize=9.8,
        color="#444444",
    )

    ax.text(3.25, 5.62, "DOM prompt", ha="center", fontsize=12, fontweight="bold")
    ax.text(6.75, 5.62, "SoM prompt", ha="center", fontsize=12, fontweight="bold")
    ax.text(0.70, 4.45, "AXTree obs", ha="center", rotation=90, fontsize=12, fontweight="bold")
    ax.text(0.70, 2.05, "[SOM_MARKS] obs", ha="center", rotation=90, fontsize=12, fontweight="bold")

    metrics = {
        site: {mode: mode_metrics(site, ep_dir) for mode, ep_dir in modes.items()}
        for site, modes in STEP_DIRS.items()
    }
    prompt_lines = []
    prompt_real: dict[str, dict[str, float | int | None]] = {}
    for site in ["reddit", "classifieds"]:
        status, stats = prompt_status(site)
        if stats is None:
            prompt_lines.append(f"{SITE_SHORT[site]} P-prompt ({status})")
        else:
            prompt_real[site] = stats
    prompt_subtitle = stats_lines(prompt_real) if len(prompt_real) == 2 else "\n".join(prompt_lines)
    prompt_face = "#dce8f5" if len(prompt_real) == 2 else "#eeeeee"
    prompt_edge = "#333333" if len(prompt_real) == 2 else "#888888"

    cell(ax, (1.55, 3.58), 3.28, 1.62, "DOM", stats_lines({s: metrics[s]["DOM"] for s in metrics}), "#dce8f5")
    cell(ax, (5.15, 3.58), 3.28, 1.62, "P-prompt", prompt_subtitle, prompt_face, edge=prompt_edge)
    cell(ax, (1.55, 1.28), 3.28, 1.62, "P-text", stats_lines({s: metrics[s]["P-text"] for s in metrics}), "#f8dddd")
    cell(ax, (5.15, 1.28), 3.28, 1.62, "P-SoM", stats_lines({s: metrics[s]["P-SoM"] for s in metrics}), "#eadff0")

    ax.add_patch(
        FancyArrowPatch(
            (1.20, 4.36),
            (1.20, 2.10),
            arrowstyle="<->",
            mutation_scale=18,
            lw=2.0,
            color="#222222",
        )
    )
    ax.text(
        1.20,
        3.20,
        "Axis 1\ntext payload",
        ha="center",
        va="center",
        fontsize=9.5,
        bbox=dict(facecolor="white", edgecolor="none", pad=2.5),
    )

    ax.add_patch(
        FancyArrowPatch(
            (3.18, 5.38),
            (6.80, 5.38),
            arrowstyle="<->",
            mutation_scale=18,
            lw=2.0,
            color="#222222",
        )
    )
    ax.text(
        5.0,
        5.05,
        "Axis 2: prompt prior",
        ha="center",
        va="center",
        fontsize=9.5,
        bbox=dict(facecolor="white", edgecolor="none", pad=2.5),
    )

    ax.add_patch(
        FancyArrowPatch(
            (8.70, 2.08),
            (8.70, 4.36),
            arrowstyle="<->",
            mutation_scale=18,
            lw=1.6,
            color="#666666",
            linestyle="dashed",
        )
    )
    ax.text(9.18, 3.22, "Axis 3 image\n(no image here)", ha="center", va="center", fontsize=9.0, color="#555555")

    ax.text(5.0, 0.35, read_axis_n_note(), ha="center", fontsize=8.5, color="#555555")
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
