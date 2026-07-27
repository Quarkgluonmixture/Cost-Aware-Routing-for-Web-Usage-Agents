#!/usr/bin/env python3
"""[Outcome supporting] Outcome dimension — drop-one oracle loss visualization.

Outputs:
- results/phantom_paper/figures/fig0c_drop_one_oracle.png  (figure)
- results/phantom_paper/fig0c_drop_one_bootstrap_ci.csv     (data sidecar, alongside other paper-grade aggregations)

Supporting visualization for oracle/drop-one solve-pool evidence.

Drop-one oracle loss for the six planned B0/B1/B2 × Classifieds/Reddit cells.
Paper-grade numeric rows require the exact canonical six-mode/task-set contract;
``--allow-partial`` is an explicitly NON_PAPER_GRADE exploratory route.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from scripts.analysis.lib.run_registry import PAPER_MODES
    from scripts.analysis.figures.lib.panels import paper_grade_panels
    from scripts.analysis.lib.canonical_task_universe import (
        expected_scored_ids,
        protocol_excluded_in_universe,
        task_id_set_sha256,
    )
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from scripts.analysis.lib.run_registry import PAPER_MODES
    from scripts.analysis.figures.lib.panels import paper_grade_panels
    from scripts.analysis.lib.canonical_task_universe import (
        expected_scored_ids,
        protocol_excluded_in_universe,
        task_id_set_sha256,
    )

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


def _spec_to_legacy_panel(spec) -> dict:
    """Convert PanelSpec to legacy `_panel` dict shape used by draw_panel.

    /stress A1.20 P0-3-ABC* + P1-1-AB (2026-05-17, 3-AI overlap): replaces
    hardcoded `PANELS = [_panel(..., 234), _panel(..., 210)]` (B0+B1 only,
    stale N=234/210) with `paper_grade_panels()` from lib/panels.py — pulls
    B0/B1/B2 + canonical N=224/205 from run_registry + scored_task_count.
    """
    return {
        "key": spec.key,
        "title": spec.title,
        "baseline": spec.baseline,
        "site": spec.site,
        "expected": spec.expected_n,
        "modes": dict(spec.modes),
        "is_placeholder": spec.is_placeholder,
    }


# Panel topology is registry-driven; task-universe validation happens inside
# ``load_panel_sets`` against the canonical scored IDs.
PANELS = [_spec_to_legacy_panel(s) for s in paper_grade_panels()]

# /stress A1.20 P1-4-A (2026-05-17): SECTION103_LOSS drift-detection dict deleted.
# Pre-fix: hardcoded B0 cls/red only, stale numbers (B0 cls=1.71 vs intro hero 3.33);
# drift mechanism stale + B1/B2 blind. Drift detection now lives in aggregator-level
# `validate_run.py` + post-flight QA. Figure-internal sanity check retired (redundant).
SECTION103_LOSS = {}


def task_id(path: Path) -> int:
    match = re.search(r"task_(\d+)_summary", path.name)
    if not match:
        raise ValueError(f"Cannot parse task id from {path}")
    return int(match.group(1))


def load_success_set(ep_dir: Path) -> tuple[set[int], set[int]]:
    # B-549 (/stress A1.5 P0-2-AB* Claude+codex OOB sibling propagation,
    # 2026-05-17): switch plain `json.load()` → `load_episode_summary_strict`
    # with `reject_needs_reevaluation=True`. Pre-fix any B-486 quarantined
    # episode (crash-before-evaluator) would enter `observed` set with
    # `success` falsy → drop-one oracle denominator polluted → paper figure
    # 0c (hero 4-fold drop-in property visualization) silently mis-counted.
    # Lenient mode + reject_quarantine → loader returns None for both
    # quarantined rows AND type-mismatched rows; skip both. Same pattern as
    # `aggregate_sr_fp_per_mode.py:85` (post-B-549).
    from p79.experiment.io_utils import load_episode_summary_strict
    files = sorted(ep_dir.glob("*_summary_v2.json"))
    if not files:
        print(f"[warn] no episode summaries under {ep_dir}", file=sys.stderr)
        return set(), set()
    successes: set[int] = set()
    observed: set[int] = set()
    for path in files:
        record = load_episode_summary_strict(
            path, mode="lenient", reject_needs_reevaluation=True,
        )
        if record is None:
            continue  # corrupt / type-mismatch / B-486 quarantine — already logged
        tid = task_id(path)
        observed.add(tid)
        # /stress A1.20 P1-2-AB (2026-05-17, B-283 sibling propagation): strict
        # `is True` instead of `bool(record.get(...))`. JSON string "false" is
        # Python truthy under bool() → SR silently inflated. `success` field
        # must be strict bool per post-§139.8 canonical (B-283 strict loader).
        if record.get("success") is True:
            successes.add(tid)
    return successes, observed


def load_panel_sets(
    panel: dict,
    *,
    allow_partial: bool = False,
) -> tuple[
    dict[str, set[int]],
    dict[str, set[int]],
    set[int],
    set[str],
    dict,
]:
    """Load and validate one panel's six-arm task universe.

    Paper-grade output requires exactly the six canonical modes and requires
    every mode's observed task IDs to equal the site's canonical scored set.
    The old 90% near-complete route is intentionally gone.  Exploratory output
    is available only through ``allow_partial=True`` and is explicitly marked
    NON_PAPER_GRADE by the caller.
    """
    expected_override = panel.get("expected_task_ids")
    if expected_override is None:
        expected_ids, expected_sha = expected_scored_ids(panel["site"])
    else:
        expected_ids = frozenset(int(t) for t in expected_override)
        expected_sha = task_id_set_sha256(expected_ids)

    # B-1908 (cont.): the runner keeps COLLECTING the AMENDMENT_08 protocol-
    # excluded tasks, so a landed reddit condition observes 205 IDs against a
    # 203-task scored set.  Restrict each arm to the scored universe and count
    # the excluded IDs separately; without this every reddit panel failed the
    # exact check for holding exactly the episodes the protocol asked for.
    _excluded = (
        frozenset()
        if expected_override is not None
        else protocol_excluded_in_universe(panel["site"])
    )

    registered_modes = set(panel.get("modes", {}))
    canonical_modes = set(MODES)
    sets: dict[str, set[int]] = {}
    obs: dict[str, set[int]] = {}
    excluded_seen: dict[str, list[int]] = {}
    for mode in MODES:
        ep_dir = panel.get("modes", {}).get(mode)
        if ep_dir is None:
            successes, observed = set(), set()
        else:
            successes, observed = load_success_set(ep_dir)
        excluded_seen[mode] = sorted(observed & _excluded)
        sets[mode] = successes & expected_ids
        obs[mode] = observed & expected_ids

    partial_modes = {m for m in MODES if obs[m] != expected_ids}
    complete_exact = (
        registered_modes == canonical_modes
        and not partial_modes
        and len(expected_ids) == int(panel["expected"])
    )
    errors: list[str] = []
    if registered_modes != canonical_modes:
        errors.append(
            "mode set mismatch: "
            f"missing={sorted(canonical_modes - registered_modes)} "
            f"extra={sorted(registered_modes - canonical_modes)}"
        )
    for mode in MODES:
        if obs[mode] != expected_ids:
            errors.append(
                f"{mode} task set mismatch: observed={len(obs[mode])}/"
                f"expected={len(expected_ids)} missing={len(expected_ids - obs[mode])} "
                f"extra={len(obs[mode] - expected_ids)}"
            )

    if complete_exact:
        common = set(expected_ids)
        participating_modes = list(MODES)
    elif allow_partial:
        participating_modes = [m for m in MODES if obs[m]]
        common = (
            set.intersection(*(obs[m] for m in participating_modes))
            if participating_modes
            else set()
        )
        if len(participating_modes) < 2:
            errors.append("fewer than two observed modes for exploratory drop-one")
        if not common:
            errors.append("empty common task universe")
    else:
        participating_modes = []
        common = set()

    sets_r = {m: sets[m] & common for m in participating_modes} if common else {}
    diagnostic_modes = [m for m in MODES if obs[m]]
    diagnostic_common = (
        set.intersection(*(obs[m] for m in diagnostic_modes))
        if diagnostic_modes else set()
    )
    portfolio_modes = participating_modes or diagnostic_modes
    task_sha = (
        expected_sha if complete_exact
        else task_id_set_sha256(common or diagnostic_common)
    )
    meta = {
        "portfolio_modes": portfolio_modes,
        "n_modes_unique": len(set(portfolio_modes)),
        "task_set_sha256": task_sha,
        "expected_task_set_sha256": expected_sha,
        "complete_exact": complete_exact,
        "errors": errors,
    }
    return sets_r, obs, common, partial_modes, meta


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
    common_task_ids: set[int] | frozenset[int],
    n_bootstrap: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """Bootstrap 95% CI for drop-one oracle loss per mode.

    Resamples the explicit common task universe, including tasks on which every
    arm fails.  The universe must never be inferred from a success-set union.
    """
    rng = np.random.default_rng(seed)
    if not sets:
        return {}
    universe = sorted(int(t) for t in common_task_ids)
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
            losses_samples[mode][b] = 100.0 * (union_count - int(without.sum())) / n
    alpha = (1 - ci) / 2
    return {
        mode: (
            float(np.quantile(samples, alpha)),
            float(np.quantile(samples, 1 - alpha)),
        )
        for mode, samples in losses_samples.items()
    }


def _panel_error_row(
    panel: dict, meta: dict, captured_at: str, error: str, *,
    grade: str = "PAPER_GRADE",
) -> dict:
    return {
        "row_type": "panel_error",
        "grade": grade,
        "captured_at": captured_at,
        "panel": panel["key"],
        "site_baseline": panel["title"],
        "baseline": panel.get("baseline", ""),
        "site": panel.get("site", ""),
        "mode": "",
        "drop_one_loss_pp": "",
        "ci95_low_pp": "",
        "ci95_high_pp": "",
        "n_common": 0,
        "n_expected": panel["expected"],
        "portfolio_modes": json.dumps(meta.get("portfolio_modes", [])),
        "n_modes_unique": meta.get("n_modes_unique", 0),
        "task_set_sha256": meta.get("task_set_sha256", ""),
        "complete_exact": False,
        "is_partial": True,
        "error": error,
    }


def draw_panel(
    ax: plt.Axes,
    panel: dict,
    csv_rows: list[dict],
    *,
    allow_partial: bool,
    captured_at: str,
) -> bool:
    """Draw a panel and return whether its paper-grade contract is exact."""
    # /stress A1.20 P0-3 (2026-05-17): placeholder cells (e.g., B2 pre-Phase-1a-fire)
    # render explicit "pending" tile rather than silent skip.
    if panel.get("is_placeholder") or not panel["modes"]:
        ax.text(0.5, 0.5,
                f"{panel['title']}\n\n(pending Phase 1a paper-grade fire)\n"
                f"N expected = {panel['expected']}",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="#888888", style="italic",
                bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f9f9f9",
                      "edgecolor": "#cccccc"})
        ax.set_title(panel["title"], fontsize=10.5, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        meta = {
            "portfolio_modes": [], "n_modes_unique": 0,
            "task_set_sha256": "", "complete_exact": False,
        }
        csv_rows.append(_panel_error_row(
            panel, meta, captured_at, "placeholder or no registered modes",
            grade="NON_PAPER_GRADE" if allow_partial else "PAPER_GRADE",
        ))
        return False
    sets_r, obs, common, partial_modes_set, meta = load_panel_sets(
        panel, allow_partial=allow_partial,
    )
    if not meta["complete_exact"] and not allow_partial:
        error = "; ".join(meta["errors"])
        ax.text(
            0.5, 0.5, f"{panel['title']}\n\nPANEL ERROR\n{error}",
            ha="center", va="center", transform=ax.transAxes, fontsize=8,
            color="#990000", wrap=True,
        )
        ax.set_title(panel["title"], fontsize=10.5, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        csv_rows.append(_panel_error_row(panel, meta, captured_at, error))
        return False
    if not sets_r or not common:
        error = "; ".join(meta["errors"]) or "no numeric exploratory universe"
        csv_rows.append(_panel_error_row(
            panel, meta, captured_at, error,
            grade="NON_PAPER_GRADE" if allow_partial else "PAPER_GRADE",
        ))
        ax.text(0.5, 0.5, f"{panel['title']}\n\nNO NUMERIC OUTPUT\n{error}",
                ha="center", va="center", transform=ax.transAxes, fontsize=8,
                color="#990000", wrap=True)
        ax.set_xticks([])
        ax.set_yticks([])
        return False
    # Drop-one denominator = N_common across complete (non-partial) modes.
    n_common = len(common) if common else panel["expected"]
    losses = drop_one_losses(sets_r, n_common)
    cis = bootstrap_drop_one_ci(sets_r, common)
    partial_modes = sorted(partial_modes_set)
    # P1-4-A: SECTION103_LOSS now empty (deleted retired drift dict); loop preserved
    # as no-op for future drift-detection re-injection if needed.
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
            "row_type": "numeric",
            "grade": "NON_PAPER_GRADE" if allow_partial else "PAPER_GRADE",
            "captured_at": captured_at,
            "panel": panel["key"],
            "site_baseline": panel["title"],
            "baseline": panel.get("baseline", ""),
            "site": panel.get("site", ""),
            "mode": mode,
            "drop_one_loss_pp": round(value, 4),
            "ci95_low_pp": round(ci_low, 4),
            "ci95_high_pp": round(ci_high, 4),
            "n_common": n_common,
            "n_expected": panel["expected"],
            "portfolio_modes": json.dumps(meta["portfolio_modes"]),
            "n_modes_unique": meta["n_modes_unique"],
            "task_set_sha256": meta["task_set_sha256"],
            "complete_exact": meta["complete_exact"],
            "is_partial": not meta["complete_exact"] or mode in partial_modes,
            "error": "" if meta["complete_exact"] else "; ".join(meta["errors"]),
        })
    n_label = f"N={n_common}" if n_common == panel["expected"] else f"N={n_common}/{panel['expected']}†"
    ax.set_title(f"{panel['title']} ({n_label})", fontsize=10.5, fontweight="bold")
    ax.set_xticks(x, [MODE_LABELS[m] for m in MODES], fontsize=8.5, rotation=20, ha="right")
    ax.set_ylim(0, 13.0)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    return bool(meta["complete_exact"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--csv-out", type=Path, default=None)
    ap.add_argument(
        "--allow-partial", action="store_true",
        help="Exploratory only: emit NON_PAPER_GRADE numeric rows on the observed common subset.",
    )
    args = ap.parse_args(argv)
    out_path = args.out
    csv_path = args.csv_out or out_path.parent.parent / "fig0c_drop_one_bootstrap_ci.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    captured_at = datetime.now(timezone.utc).isoformat()
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    # /stress A1.20 P0-3: layout grows automatically with N panels (3 baselines
    # × 2 sites = 6 panels for Phase 1a, 2×4 grid). lib/panels.PanelSpec drives.
    n_panels = len(PANELS)
    n_cols = 2
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(13.0, max(4.3, 4.3 * n_rows)), sharey=True)
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]
    csv_rows: list[dict] = []
    exact_panels: list[bool] = []
    for ax, panel in zip(axes_flat, PANELS):
        exact_panels.append(draw_panel(
            ax, panel, csv_rows,
            allow_partial=args.allow_partial,
            captured_at=captured_at,
        ))
    # Hide unused subplot if odd panel count.
    for extra in list(axes_flat)[n_panels:]:
        extra.set_visible(False)
    for row in (axes if n_rows > 1 else [axes]):
        if hasattr(row, "__iter__"):
            row[0].set_ylabel("Oracle loss when arm is removed (pp, 95% bootstrap CI)")
        else:
            row.set_ylabel("Oracle loss when arm is removed (pp, 95% bootstrap CI)")
    fig.suptitle("Drop-One Oracle: Incremental Routing Value (strict 6-mode, 95% bootstrap CI, n=1000)", fontsize=13.5, fontweight="bold")
    fig.text(
        0.5,
        0.025,
        "Higher bars = representation solves tasks not recovered by the other plotted arms. "
        "P-SoM = Phantom-SoM, P-text = AXTree+DOM-prompt+no per-step screenshot, P-prompt = AXTree+SoM-prompt+no per-step screenshot. "
        "† = explicit --allow-partial exploratory common-universe subset. "
        "Paper-grade numeric rows require exact six modes and the canonical task-ID set. "
        "(denominator value shown in each panel title; pp lifts are expressed as percentages "
        "of THIS panel's N_common, not against site's expected_n) — /stress A1.20 P1-9-C (2026-05-17). "
        "Canonical N per site from scored_task_count post-§139.8: cls=224 / red=205. "
        "CI from 1000-resample bootstrap.",
        ha="center",
        fontsize=7.5,
        color="#555555",
    )
    if args.allow_partial:
        fig.text(0.5, 0.5, "NON_PAPER_GRADE — PARTIAL EXPLORATION",
                 fontsize=24, color="#CC0000", alpha=0.18, ha="center",
                 va="center", rotation=18, zorder=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))

    # B-1908 (/stress Mode B codex follow-up, 2026-07-27): validate BEFORE
    # writing.  Pre-fix the figure and CSV were written first and the exact
    # paper-grade check returned 2 afterwards, so a failing run still left a
    # freshly-timestamped PNG on disk.  Anyone reading `ls -lt` — or a Makefile
    # step that ignores exit status — would treat an unvalidated panel as the
    # current figure.  Fail closed with nothing written instead.
    validation_failed = not args.allow_partial and not all(exact_panels)
    if validation_failed:
        print(
            "error: one or more panels failed exact paper-grade validation — "
            "no figure or CSV written (pass --allow-partial to produce a "
            "watermarked exploratory version)",
            file=sys.stderr,
        )
        return 2

    fig.savefig(out_path, bbox_inches="tight")
    print(out_path)
    # Data sidecar lives alongside other cross-condition aggregations
    # (phantom_lift.csv, auroc_cross_condition.csv, run_summary_collect.json),
    # not inside figures/ — figures/ is for PNGs only.
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "row_type", "grade", "captured_at", "panel", "site_baseline",
            "baseline", "site", "mode",
            "drop_one_loss_pp", "ci95_low_pp", "ci95_high_pp",
            "n_common", "n_expected", "portfolio_modes", "n_modes_unique",
            "task_set_sha256", "complete_exact", "is_partial", "error",
        ])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
