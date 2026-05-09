#!/usr/bin/env python3
"""[Outcome supporting] Outcome dimension — drop-one oracle loss visualization.

Outputs:
- results/phantom_paper/figures/fig0c_drop_one_oracle.png  (figure)
- results/phantom_paper/fig0c_drop_one_bootstrap_ci.csv     (data sidecar, alongside other paper-grade aggregations)

Supporting visualization for oracle/drop-one solve-pool evidence.

Drop-one oracle loss for B0/B1 VWA observation arms.

All available cells are computed from episode-level ``adjusted_success`` sets.
B0 Phantom-SoM/P-text use fresh paper-grade clean re-run; B1 Phantom-SoM is
drawn as unavailable pending re-run.
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
OUT = ROOT / "results/phantom_paper/figures/fig0c_drop_one_oracle.png"

MODES = PAPER_MODES
COLORS = {
    "DOM": "#4c78a8",
    "SoM": "#f58518",
    "Vision": "#54a24b",
    "P-SoM": "#b279a2",
    "P-text": "#9e6da8",
    "P-prompt": "#9467bd",
}
MODE_LABELS = {
    "DOM": "DOM",
    "SoM": "SoM",
    "Vision": "Vision",
    "P-SoM": "P-SoM",
    "P-text": "P-text",
    "P-prompt": "P-prompt",
}


def _mode_dirs(baseline: str, site: str) -> dict[str, Path]:
    # F40 audit fix 2026-05-09: respect P79_AGGREGATOR_GRADE so legacy
    # archived cells produce sensitivity figures.
    import os as _os
    env_grade = _os.environ.get("P79_AGGREGATOR_GRADE", "")
    grade_filter = [g.strip() for g in env_grade.split(",") if g.strip()] or None
    return {cell.mode: cell.episodes_dir for cell in get_cells(baseline=baseline, site=site, grade=grade_filter)}


def _panel(key: str, title: str, baseline: str, site: str, expected: int) -> dict:
    return {
        "key": key,
        "title": title,
        "expected": expected,
        "modes": _mode_dirs(baseline, site),
    }

# Drop-one oracle uses union/intersection over the **common observed** task
# universe per panel (so a partial in-flight run doesn't artificially shrink
# other modes' unique-task counts). Set "common_universe": True (default) to
# restrict to intersection across modes; partial Phantom rows annotated with
# their N/expected coverage.
PANELS = [
    _panel("b0_cls", "B0 classifieds", "B0", "classifieds", 234),
    _panel("b0_red", "B0 reddit", "B0", "reddit", 210),
    _panel("b1_cls", "B1 classifieds", "B1", "classifieds", 234),
    _panel("b1_red", "B1 reddit", "B1", "reddit", 210),
]

SECTION103_LOSS = {
    "b0_cls": {"DOM": 2.14, "SoM": 7.69, "Vision": 3.85, "P-SoM": 1.71},
    "b0_red": {"DOM": 1.43, "SoM": 2.86, "Vision": 1.90, "P-SoM": 2.38},
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


def load_panel_sets(panel: dict) -> tuple[dict[str, set[int]], dict[str, set[int]], set[int], set[str]]:
    """Returns (succ_sets_intersected, observed_sets, common_universe, partial_modes).

    Drop-one is computed on the intersection of **complete** modes only.
    Partial modes (observed < 0.9 × expected) are excluded from the oracle
    union/intersection so they don't artificially shrink the comparable
    universe (otherwise modes that diverge on the full pool can become
    accidentally redundant in the partial intersection — e.g. B0 reddit
    Vision drop-one collapsed to 0 when P-prompt at n=134 was included).
    """
    sets: dict[str, set[int]] = {}
    obs: dict[str, set[int]] = {}
    partial_modes: set[str] = set()
    for mode, ep_dir in panel["modes"].items():
        successes, observed = load_success_set(ep_dir)
        sets[mode] = successes
        obs[mode] = observed
        if observed and len(observed) < int(panel["expected"] * 0.9):
            partial_modes.add(mode)
            print(
                f"[note] {panel['title']} {mode}: partial n={len(observed)}/"
                f"{panel['expected']} — excluded from drop-one oracle",
                file=sys.stderr,
            )
        elif observed and len(observed) != panel["expected"]:
            print(
                f"[note] {panel['title']} {mode}: near-complete n={len(observed)}/"
                f"{panel['expected']} — included",
                file=sys.stderr,
            )
    complete_obs = {m: o for m, o in obs.items() if m not in partial_modes and o}
    common = set.intersection(*complete_obs.values()) if complete_obs else set()
    sets_r = {m: s & common for m, s in sets.items() if m not in partial_modes}
    return sets_r, obs, common, partial_modes


def drop_one_losses(sets: dict[str, set[int]], expected: int) -> dict[str, float]:
    """Drop-one expressed as percentage of the *common universe* (denominator)."""
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
    sets_r, obs, common, partial_modes_set = load_panel_sets(panel)
    # Drop-one denominator = N_common across complete (non-partial) modes.
    n_common = len(common) if common else panel["expected"]
    losses = drop_one_losses(sets_r, n_common)
    cis = bootstrap_drop_one_ci(sets_r, n_common)
    partial_modes = sorted(partial_modes_set)
    if panel["key"] in SECTION103_LOSS:
        for mode, verified in SECTION103_LOSS[panel["key"]].items():
            if mode in losses and abs(losses[mode] - verified) > 0.25:
                print(
                    f"[note] {panel['title']} {mode}: live drop-one "
                    f"{losses[mode]:.2f} pp vs §103 {verified:.2f} pp (full→common subset)",
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
            n_obs = len(obs.get(mode, set()))
            label = f"partial\nN={n_obs}/{panel['expected']}" if mode in partial_modes_set else "N/A"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0.23,
                label,
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#666666",
            )
            continue
        ci_low, ci_high = cis.get(mode, (value, value))
        # Mark partial modes with † for clarity in figure
        marker = "†" if mode in partial_modes else ""
        label = f"{value:.2f}{marker}\n[{ci_low:.2f},{ci_high:.2f}]"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ci_high + 0.15,
            label,
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
        csv_rows.append({
            "panel": panel["key"],
            "site_baseline": panel["title"],
            "mode": mode,
            "drop_one_loss_pp": round(value, 4),
            "ci95_low_pp": round(ci_low, 4),
            "ci95_high_pp": round(ci_high, 4),
            "n_common": n_common,
            "n_expected": panel["expected"],
            "is_partial": mode in partial_modes,
        })
    n_label = f"N={n_common}" if n_common == panel["expected"] else f"N={n_common}/{panel['expected']}†"
    ax.set_title(f"{panel['title']} ({n_label})", fontsize=10.5, fontweight="bold")
    ax.set_xticks(x, [MODE_LABELS[m] for m in MODES], fontsize=8.5, rotation=20, ha="right")
    ax.set_ylim(0, 13.0)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    import csv as _csv
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.6), sharey=True)
    csv_rows: list[dict] = []
    for ax, panel in zip(axes.flat, PANELS):
        draw_panel(ax, panel, csv_rows)
    for ax in axes[:, 0]:
        ax.set_ylabel("Oracle loss when arm is removed (pp, 95% bootstrap CI)")
    fig.suptitle("Drop-One Oracle: Incremental Routing Value (up to 6-mode, 95% bootstrap CI, n=1000)", fontsize=13.5, fontweight="bold")
    fig.text(
        0.5,
        0.025,
        "Higher bars = representation solves tasks not recovered by the other plotted arms. "
        "P-SoM = Phantom-SoM, P-text = AXTree+DOM-prompt+no-image, P-prompt = AXTree+SoM-prompt+no-image. "
        "† = partial / common-universe subset (B0 reddit P-prompt run live; "
        "B1 phantom_som chain in flight). N=common observed across all modes per panel. CI from 1000-resample bootstrap.",
        ha="center",
        fontsize=8.0,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)
    # Data sidecar lives alongside other cross-condition aggregations
    # (phantom_lift.csv, auroc_cross_condition.csv, run_summary_collect.json),
    # not inside figures/ — figures/ is for PNGs only.
    csv_path = OUT.parent.parent / "fig0c_drop_one_bootstrap_ci.csv"
    with csv_path.open("w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=[
            "panel", "site_baseline", "mode",
            "drop_one_loss_pp", "ci95_low_pp", "ci95_high_pp",
            "n_common", "n_expected", "is_partial",
        ])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(csv_path)


if __name__ == "__main__":
    main()
