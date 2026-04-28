#!/usr/bin/env python3
"""Drop-one oracle loss for B0/B1 VWA observation arms.

All available cells are computed from episode-level ``adjusted_success`` sets.
B0 Phantom-SoM/Phantom-DOM use fresh paper-grade clean re-run; B1 Phantom-SoM is
drawn as unavailable pending re-run.
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
OUT = ROOT / "results/phantom_paper/figures/fig2_drop_one_oracle.png"

MODES = ["DOM", "SoM", "Vision", "Phantom-SoM"]
COLORS = {
    "DOM": "#4c78a8",
    "SoM": "#f58518",
    "Vision": "#54a24b",
    "Phantom-SoM": "#b279a2",
}

PANELS = [
    {
        "key": "b0_cls",
        "title": "B0 classifieds",
        "expected": 234,
        "modes": {
            "DOM": RESULTS / "B0_3mode_classifieds_20260413/phase1_dom_router_0/episodes",
            "SoM": RESULTS / "B0_3mode_classifieds_20260413/phase1_som_router_0/episodes",
            "Vision": RESULTS / "B0_3mode_classifieds_20260413/phase1_vision_router_0/episodes",
            "Phantom-SoM": RESULTS / "B0_phantom_classifieds_20260426/phase1_phantom_som_router_0/episodes",
        },
            },
    {
        "key": "b0_red",
        "title": "B0 reddit",
        "expected": 210,
        "modes": {
            "DOM": RESULTS / "B0_3mode_reddit_20260422/phase1_dom_router_0/episodes",
            "SoM": RESULTS / "B0_3mode_reddit_20260422/phase1_som_router_0/episodes",
            "Vision": RESULTS / "B0_3mode_reddit_20260422/phase1_vision_router_0/episodes",
            "Phantom-SoM": RESULTS / "B0_phantom_reddit_20260428/phase1_phantom_som_router_0/episodes",
        },
            },
    {
        "key": "b1_cls",
        "title": "B1 classifieds",
        "expected": 234,
        "modes": {
            "DOM": RESULTS / "B1_3mode_classifieds_20260413/phase1_dom_router_0/episodes",
            "SoM": RESULTS / "B1_3mode_classifieds_20260413/phase1_som_router_0/episodes",
            "Vision": RESULTS / "B1_3mode_classifieds_20260413/phase1_vision_router_0/episodes",
        },
        "missing": "Phantom-SoM N/A",
    },
    {
        "key": "b1_red",
        "title": "B1 reddit",
        "expected": 210,
        "modes": {
            "DOM": RESULTS / "B1_3mode_reddit_20260413/phase1_dom_router_0/episodes",
            "SoM": RESULTS / "B1_3mode_reddit_20260413/phase1_som_router_0/episodes",
            "Vision": RESULTS / "B1_3mode_reddit_20260413/phase1_vision_router_0/episodes",
        },
        "missing": "Phantom-SoM N/A",
    },
]

SECTION103_LOSS = {
    "b0_cls": {"DOM": 2.14, "SoM": 7.69, "Vision": 3.85, "Phantom-SoM": 1.71},
    "b0_red": {"DOM": 1.43, "SoM": 2.86, "Vision": 1.90, "Phantom-SoM": 2.38},
}


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
        observed.add(tid)
        if bool(record.get("adjusted_success", record.get("success", False))):
            successes.add(tid)
    return successes, observed


def load_panel_sets(panel: dict) -> dict[str, set[int]]:
    sets: dict[str, set[int]] = {}
    for mode, ep_dir in panel["modes"].items():
        successes, observed = load_success_set(ep_dir)
        sets[mode] = successes
        if len(observed) != panel["expected"]:
            print(
                f"[warn] {panel['title']} {mode}: n={len(observed)} "
                f"expected={panel['expected']}",
                file=sys.stderr,
            )
    return sets


def drop_one_losses(sets: dict[str, set[int]], expected: int) -> dict[str, float]:
    union_all = set().union(*sets.values()) if sets else set()
    losses: dict[str, float] = {}
    for mode in sets:
        without = set().union(*(s for m, s in sets.items() if m != mode))
        losses[mode] = 100.0 * (len(union_all) - len(without)) / expected
    return losses


def bootstrap_drop_one_ci(
    sets: dict[str, set[int]],
    expected: int,
    n_bootstrap: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """Bootstrap 95% CI for drop-one oracle loss per mode.

    Resamples N tasks (with replacement) from observed task universe; for each
    resample, recomputes drop-one loss; returns (low, high) percentiles.
    """
    rng = np.random.default_rng(seed)
    if not sets:
        return {}
    universe = sorted(set().union(*sets.values()))
    if not universe:
        return {mode: (0.0, 0.0) for mode in sets}
    n = len(universe)
    arr = np.asarray(universe)
    # Pre-build per-mode boolean masks indexed by universe order
    mode_masks = {mode: np.isin(arr, list(s)) for mode, s in sets.items()}
    losses_samples = {mode: np.empty(n_bootstrap) for mode in sets}
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        union_succ = np.zeros(n, dtype=bool)
        for mask in mode_masks.values():
            union_succ |= mask[idx]
        union_count = int(union_succ.sum())
        for mode, mask in mode_masks.items():
            without = np.zeros(n, dtype=bool)
            for m, mk in mode_masks.items():
                if m == mode:
                    continue
                without |= mk[idx]
            losses_samples[mode][b] = 100.0 * (union_count - int(without.sum())) / expected
    alpha = (1 - ci) / 2
    return {
        mode: (
            float(np.quantile(samples, alpha)),
            float(np.quantile(samples, 1 - alpha)),
        )
        for mode, samples in losses_samples.items()
    }


def draw_panel(ax: plt.Axes, panel: dict, csv_rows: list[dict]) -> None:
    sets = load_panel_sets(panel)
    losses = drop_one_losses(sets, panel["expected"])
    cis = bootstrap_drop_one_ci(sets, panel["expected"])
    if panel["key"] in SECTION103_LOSS:
        for mode, verified in SECTION103_LOSS[panel["key"]].items():
            if mode in losses and abs(losses[mode] - verified) > 0.25:
                print(
                    f"[warn] {panel['title']} {mode}: live/fallback drop-one "
                    f"{losses[mode]:.2f} pp vs §103 {verified:.2f} pp",
                    file=sys.stderr,
                )

    x = np.arange(len(MODES))
    values = [losses.get(mode, 0.0) for mode in MODES]
    colors = [COLORS[mode] if mode in losses else "#d4d4d4" for mode in MODES]
    err_low = [max(0.0, values[i] - cis.get(MODES[i], (values[i], values[i]))[0]) for i in range(len(MODES))]
    err_high = [max(0.0, cis.get(MODES[i], (values[i], values[i]))[1] - values[i]) for i in range(len(MODES))]
    err_arr = [
        [err_low[i] if MODES[i] in losses else 0.0 for i in range(len(MODES))],
        [err_high[i] if MODES[i] in losses else 0.0 for i in range(len(MODES))],
    ]
    bars = ax.bar(x, values, color=colors, width=0.66, yerr=err_arr,
                  ecolor="#333333", capsize=3, error_kw={"linewidth": 0.9})
    for bar, mode, value in zip(bars, MODES, values):
        if mode not in losses:
            bar.set_hatch("//")
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0.23,
                "N/A",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#666666",
            )
            continue
        ci_low, ci_high = cis.get(mode, (value, value))
        label = f"{value:.2f}\n[{ci_low:.2f},{ci_high:.2f}]"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ci_high + 0.15,
            label,
            ha="center",
            va="bottom",
            fontsize=7.0,
        )
        csv_rows.append({
            "panel": panel["key"],
            "site_baseline": panel["title"],
            "mode": mode,
            "drop_one_loss_pp": round(value, 4),
            "ci95_low_pp": round(ci_low, 4),
            "ci95_high_pp": round(ci_high, 4),
            "n_tasks": panel["expected"],
        })
    ax.set_title(f"{panel['title']} (N={panel['expected']})", fontsize=11, fontweight="bold")
    ax.set_xticks(x, ["DOM", "SoM", "Vision", "Phantom"])
    ax.set_ylim(0, 11.0)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    import csv as _csv
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4), sharey=True)
    csv_rows: list[dict] = []
    for ax, panel in zip(axes.flat, PANELS):
        draw_panel(ax, panel, csv_rows)
    for ax in axes[:, 0]:
        ax.set_ylabel("Oracle loss when arm is removed (pp, 95% bootstrap CI)")
    fig.suptitle("Drop-One Oracle: Incremental Routing Value (95% bootstrap CI, n=1000)", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.025,
        "Higher bars mean the representation solves tasks not recovered by the other plotted arms. "
        "Error bars: 95% bootstrap CI (n=1000 resamples). B1 Phantom-SoM/Phantom-DOM pending re-run.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)
    csv_path = OUT.parent / "fig2_drop_one_bootstrap_ci.csv"
    with csv_path.open("w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=[
            "panel", "site_baseline", "mode",
            "drop_one_loss_pp", "ci95_low_pp", "ci95_high_pp", "n_tasks",
        ])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(csv_path)


if __name__ == "__main__":
    main()
