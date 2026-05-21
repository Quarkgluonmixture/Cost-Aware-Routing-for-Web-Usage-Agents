#!/usr/bin/env python3
"""Router archive diagnostic — pre-Phase-1a gates for H9/H10 preregistration lock.

Runs 3 primary gates on existing B0 archive (cls + reddit × 6 modes) to validate
v3 router design choices BEFORE Phase 1a launch. Outputs the per-cell verdicts
needed to finalize preregistration §C patches (estimand δ calibration + anchor
fallback) and unblock OSF DOI commit.

Gates:
    G-1  Label entropy (P2 viability gate per codex Mode B finding F6)
    G-2  Best-single-mode anchor Kendall tau across 100 x 5-fold resamples
         (anchor-flicker fallback gate per Mode A finding F5 / preregistration C2)
    G-4  Router-vs-anchor noise SD MC (delta_h9/delta_h10 fine-calibration gate
         per preregistration C1 rationale)

Gates G-3 (threshold validation) / G-5 (intent-regex coverage) / G-6 (hijack
threshold) are deferred to a second pass — they need step-1 JSONL browser-state
data that requires reading thousands of step records.

Outputs:
    docs/checkpoints/router_archive_diagnostic_<YYYY-MM-DD>.md
    + per-gate JSON sidecars in same dir

Usage:
    python3 scripts/analysis/router_archive_diagnostic.py [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# MODES single-source (F6-followup / B-1821) — see p1_archive_simulation for rationale.
# Canonical ascending-cost order makes gate_g1_label_entropy's "argmax success,
# tie-broken by mode index" prefer the cheap HERO phantom_som over som/vision on ties.
from p79.policies.router_features import MODES


REPO = Path(__file__).resolve().parents[2]
PHASE1_ROOT = REPO / "results/visualwebarena/phase1"


# Archive run map: (baseline, site, mode) -> condition subdir relative to PHASE1_ROOT
# B0 cls + reddit x 6 modes. DOM/SoM/Vision come from the 3mode run; phantom
# modes have dedicated runs.
ARCHIVE_RUNS: dict[tuple[str, str, str], str] = {
    ("B0", "classifieds", "dom"):           "B0_3mode_classifieds_20260413/phase1_dom_router_0",
    ("B0", "classifieds", "som"):           "B0_3mode_classifieds_20260413/phase1_som_router_0",
    ("B0", "classifieds", "vision"):        "B0_3mode_classifieds_20260413/phase1_vision_router_0",
    # NB: B0_phantom_text dir was renamed from B0_phantom_dom (CLAUDE.md note); internal condition_id retains phantom_dom_router_0
    ("B0", "classifieds", "phantom_text"):  "B0_phantom_text_classifieds_20260427/phase1_phantom_dom_router_0",
    ("B0", "classifieds", "phantom_prompt"):"B0_phantom_prompt_classifieds_20260504/phase1_phantom_prompt_router_0",
    ("B0", "classifieds", "phantom_som"):   "B0_phantom_som_classifieds_20260426/phase1_phantom_som_router_0",
    ("B0", "reddit", "dom"):                "B0_3mode_reddit_20260422/phase1_dom_router_0",
    ("B0", "reddit", "som"):                "B0_3mode_reddit_20260422/phase1_som_router_0",
    ("B0", "reddit", "vision"):             "B0_3mode_reddit_20260422/phase1_vision_router_0",
    ("B0", "reddit", "phantom_text"):       "B0_phantom_text_reddit_20260427/phase1_phantom_dom_router_0",
    ("B0", "reddit", "phantom_prompt"):     "B0_phantom_prompt_reddit_20260429/phase1_phantom_prompt_router_0",
    ("B0", "reddit", "phantom_som"):        "B0_phantom_som_reddit_20260428/phase1_phantom_som_router_0",
}

CELLS = [("B0", "classifieds"), ("B0", "reddit")]


def load_cell(baseline: str, site: str, min_ep: int = 50) -> tuple[dict[int, dict[str, bool]], list[str], list[str]]:
    """Load (task_id → mode → success) restricted to tasks present in ALL retained modes.

    Modes with fewer than `min_ep` episodes are excluded as 'partial archive'
    (e.g., B0 phantom_prompt cls only has 4 episodes — aborted run).

    Returns (matrix, retained_modes, skipped_modes_with_reason).
    """
    per_mode: dict[str, dict[int, bool]] = {}
    retained: list[str] = []
    skipped: list[str] = []
    for mode in MODES:
        sub = ARCHIVE_RUNS.get((baseline, site, mode))
        if sub is None:
            skipped.append(f"{mode}: no archive entry mapped")
            continue
        ep_dir = PHASE1_ROOT / sub / "episodes"
        if not ep_dir.is_dir():
            skipped.append(f"{mode}: dir missing {ep_dir}")
            continue
        per_task: dict[int, bool] = {}
        for f in ep_dir.glob(f"{site}_task_*_summary_v2.json"):
            try:
                rec = json.loads(f.read_text())
            except json.JSONDecodeError:
                continue
            tid = int(rec["task_id"])
            per_task[tid] = bool(rec.get("success", False))
        if len(per_task) < min_ep:
            skipped.append(f"{mode}: only {len(per_task)} ep (< {min_ep}) — partial/aborted archive")
            continue
        per_mode[mode] = per_task
        retained.append(mode)
    if not per_mode:
        return {}, retained, skipped
    common = set.intersection(*(set(per_mode[m]) for m in retained))
    matrix: dict[int, dict[str, bool]] = {}
    for tid in sorted(common):
        matrix[tid] = {m: per_mode[m][tid] for m in retained}
    return matrix, retained, skipped


# ----------------------------------------------------------------------------
# G-1 — Label entropy (P2 viability gate)
# ----------------------------------------------------------------------------
def gate_g1_label_entropy(matrix: dict[int, dict[str, bool]], modes: list[str]) -> dict:
    """Label = argmax_m success(t, m), tie-broken by mode index order.

    Returns label histogram + Shannon entropy + majority-baseline SR.
    Gate threshold: entropy >= log(2) (= 0.693) means labels span >= 2 modes
    meaningfully; below this P2 collapses to single-mode prediction = trivially
    matched by best-single-mode baseline (no learnable signal).
    """
    labels: list[str] = []
    for tid, row in matrix.items():
        # argmax across retained modes; ties broken by `modes` list order (stable)
        best_mode = max(modes, key=lambda m: (row[m], -modes.index(m)))
        labels.append(best_mode)
    counts = Counter(labels)
    n = len(labels)
    probs = [c / n for c in counts.values()]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    majority_mode = counts.most_common(1)[0][0]
    majority_baseline_sr = sum(row[majority_mode] for row in matrix.values()) / n
    return {
        "n_tasks": n,
        "label_histogram": dict(counts),
        "entropy": entropy,
        "entropy_threshold_log2": math.log(2),
        "p2_viable": entropy >= math.log(2),
        "majority_mode": majority_mode,
        "majority_baseline_sr_pct": 100.0 * majority_baseline_sr,
    }


# ----------------------------------------------------------------------------
# G-2 — Anchor stability (Kendall tau across resamples)
# ----------------------------------------------------------------------------
def gate_g2_anchor_stability(matrix: dict[int, dict[str, bool]], modes: list[str],
                              n_resamples: int = 100,
                              n_folds: int = 5, seed: int = 42) -> dict:
    """For each of n_resamples random 5-fold splits, record best-single-mode per
    train-fold of each outer fold. Then compute the Kendall tau-rank correlation
    of mode-ranking across all (n_resamples x n_folds) resamples vs a reference
    ranking = full-dataset mode-rank.

    Below tau=0.7 = anchor flickers, trigger preregistration C2 fallback.
    Reports per-cell tau + majority-winner-across-resamples mode.
    """
    rng = np.random.default_rng(seed)
    task_ids = list(matrix.keys())
    n = len(task_ids)
    if n < n_folds:
        return {"error": f"n_tasks {n} < n_folds {n_folds}", "kendall_tau": None}

    def sr_per_mode(idxs: list[int]) -> dict[str, float]:
        return {m: sum(matrix[task_ids[i]][m] for i in idxs) / len(idxs) for m in modes}

    full_sr = sr_per_mode(list(range(n)))
    full_rank = sorted(modes, key=lambda m: -full_sr[m])
    full_rank_idx = {m: full_rank.index(m) for m in modes}

    winners: list[str] = []
    tau_per_run: list[float] = []
    for _ in range(n_resamples):
        perm = rng.permutation(n)
        fold_size = n // n_folds
        for k in range(n_folds):
            test_idxs = list(perm[k * fold_size:(k + 1) * fold_size])
            train_idxs = [i for i in perm if i not in set(test_idxs)]
            sr = sr_per_mode(train_idxs)
            winner = max(modes, key=lambda m: sr[m])
            winners.append(winner)
            run_rank = sorted(modes, key=lambda m: -sr[m])
            run_rank_idx = {m: run_rank.index(m) for m in modes}
            # Kendall tau across retained-mode ranking
            pairs = [(a, b) for a in modes for b in modes if a < b]
            concordant = sum(1 for a, b in pairs
                             if (full_rank_idx[a] - full_rank_idx[b]) *
                                (run_rank_idx[a] - run_rank_idx[b]) > 0)
            discordant = sum(1 for a, b in pairs
                             if (full_rank_idx[a] - full_rank_idx[b]) *
                                (run_rank_idx[a] - run_rank_idx[b]) < 0)
            denom = len(pairs)
            tau_per_run.append((concordant - discordant) / denom if denom else 1.0)
    winner_counts = Counter(winners)
    majority_winner, majority_count = winner_counts.most_common(1)[0]
    mean_tau = float(np.mean(tau_per_run))
    return {
        "n_resamples": n_resamples,
        "n_folds": n_folds,
        "full_dataset_best_single_mode": full_rank[0],
        "full_dataset_mode_rank": full_rank,
        "kendall_tau_mean": mean_tau,
        "kendall_tau_stable": mean_tau >= 0.7,
        "anchor_winner_counts": dict(winner_counts),
        "majority_winner": majority_winner,
        "majority_winner_fraction": majority_count / len(winners),
        "anchor_flicker_fallback_needed": mean_tau < 0.7,
    }


# ----------------------------------------------------------------------------
# G-4 — Router-vs-anchor noise SD MC (delta calibration)
# ----------------------------------------------------------------------------
def gate_g4_noise_sd(matrix: dict[int, dict[str, bool]], modes: list[str],
                      n_bootstrap: int = 1000,
                      seed: int = 42) -> dict:
    """Under null-hypothesis simulation (no real router signal), bootstrap the
    lift of "oracle router that always picks best-single-mode-per-task on test
    fold" vs best-single-mode-fixed-per-cell. SD of this lift across bootstraps
    = noise floor for H9/H10 delta calibration. If SD > 0.5pp, raise delta to
    2 x SD per preregistration C1 rationale.
    """
    rng = np.random.default_rng(seed)
    task_ids = list(matrix.keys())
    n = len(task_ids)

    # Best-single-mode on FULL cell (anchor proxy)
    full_sr = {m: sum(matrix[tid][m] for tid in task_ids) / n for m in modes}
    best_mode = max(modes, key=lambda m: full_sr[m])

    lifts: list[float] = []
    for _ in range(n_bootstrap):
        resample = rng.choice(task_ids, size=n, replace=True)
        # Anchor on resampled set
        bs_sr_per_mode = {m: sum(matrix[tid][m] for tid in resample) / n for m in modes}
        anchor_sr = bs_sr_per_mode[best_mode]
        # Oracle "router" = pick best per task (upper bound, but used here just for SD estimate)
        oracle_sr = sum(max(matrix[tid][m] for m in modes) for tid in resample) / n
        lifts.append(100.0 * (oracle_sr - anchor_sr))
    return {
        "n_bootstrap": n_bootstrap,
        "anchor_mode": best_mode,
        "oracle_lift_mean_pp": float(np.mean(lifts)),
        "oracle_lift_sd_pp": float(np.std(lifts, ddof=1)),
        "oracle_lift_p5_pp": float(np.percentile(lifts, 5)),
        "oracle_lift_p95_pp": float(np.percentile(lifts, 95)),
        "delta_h9_h10_calibrated_pp": max(1.0, 2.0 * float(np.std(lifts, ddof=1))),
        "delta_raise_needed": float(np.std(lifts, ddof=1)) > 0.5,
    }


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def run_all() -> dict:
    out: dict = {"run_date": datetime.utcnow().isoformat() + "Z", "cells": {}}
    for baseline, site in CELLS:
        key = f"{baseline}_{site}"
        try:
            matrix, retained, skipped = load_cell(baseline, site)
        except Exception as e:
            out["cells"][key] = {"error": str(e)}
            continue
        if not matrix:
            out["cells"][key] = {"error": "no retained modes", "skipped": skipped}
            continue
        out["cells"][key] = {
            "n_common_tasks": len(matrix),
            "modes_retained": retained,
            "modes_skipped": skipped,
            "g1_label_entropy": gate_g1_label_entropy(matrix, retained),
            "g2_anchor_stability": gate_g2_anchor_stability(matrix, retained),
            "g4_noise_sd": gate_g4_noise_sd(matrix, retained),
        }
    return out


def render_markdown(report: dict) -> str:
    lines = ["# Router archive diagnostic — pre-Phase-1a gate verdicts",
             "",
             f"Run date: {report['run_date']}",
             f"Archive: B0 cls + reddit x 6 modes (paper-grade pre-bug archive)",
             "",
             "## Gate summary",
             "",
             "| Cell | n tasks | G-1 entropy | P2 viable? | G-2 Kendall tau | Anchor stable? | G-4 noise SD (pp) | Delta raise? |",
             "|---|---|---|---|---|---|---|---|"]
    for ck, cv in report["cells"].items():
        if "error" in cv:
            lines.append(f"| {ck} | ERROR: {cv['error']} | | | | | | |")
            continue
        if cv.get("modes_skipped"):
            lines.append(f"| {ck} (partial: {len(cv['modes_retained'])}/{len(MODES)} modes) | {cv['n_common_tasks']} | ... | ... | ... | ... | ... | ... |")
        g1 = cv["g1_label_entropy"]
        g2 = cv["g2_anchor_stability"]
        g4 = cv["g4_noise_sd"]
        lines.append(
            f"| {ck} | {cv['n_common_tasks']} | {g1['entropy']:.3f} | "
            f"{'YES' if g1['p2_viable'] else 'NO'} | {g2['kendall_tau_mean']:.3f} | "
            f"{'YES' if g2['kendall_tau_stable'] else 'NO'} | {g4['oracle_lift_sd_pp']:.2f} | "
            f"{'YES' if g4['delta_raise_needed'] else 'NO'} |"
        )
    lines += ["", "## Per-cell detail", ""]
    for ck, cv in report["cells"].items():
        if "error" in cv:
            continue
        g1, g2, g4 = cv["g1_label_entropy"], cv["g2_anchor_stability"], cv["g4_noise_sd"]
        lines += [
            f"### {ck}",
            "",
            f"**G-1 label entropy**: H = {g1['entropy']:.3f} (threshold log(2) = {g1['entropy_threshold_log2']:.3f}); "
            f"P2 {'VIABLE' if g1['p2_viable'] else 'NOT VIABLE — H10 DEFER condition triggers'}.",
            f"Label histogram: `{g1['label_histogram']}`. Majority mode `{g1['majority_mode']}` baseline SR = {g1['majority_baseline_sr_pct']:.2f}%.",
            "",
            f"**G-2 anchor Kendall tau**: mean tau = {g2['kendall_tau_mean']:.3f} "
            f"({'STABLE' if g2['kendall_tau_stable'] else 'FLICKER — preregistration C2 fallback triggers'}). "
            f"Full-cell best-single-mode = `{g2['full_dataset_best_single_mode']}`; "
            f"majority-winner-across-resamples = `{g2['majority_winner']}` ({g2['majority_winner_fraction']:.1%}).",
            f"Anchor winner distribution: `{g2['anchor_winner_counts']}`.",
            "",
            f"**G-4 router-vs-anchor noise SD**: SD = {g4['oracle_lift_sd_pp']:.2f}pp, "
            f"mean oracle lift = {g4['oracle_lift_mean_pp']:.2f}pp [{g4['oracle_lift_p5_pp']:.2f}, {g4['oracle_lift_p95_pp']:.2f}]. "
            f"delta_h9/delta_h10 calibrated to {g4['delta_h9_h10_calibrated_pp']:.2f}pp "
            f"({'RAISE from 1.0pp default' if g4['delta_raise_needed'] else 'keep default 1.0pp'}).",
            "",
        ]
    lines += ["",
              "## Verdicts for preregistration §C lock",
              "",
              "1. **§C1 H9/H10 delta calibration**: see per-cell G-4 SD column. If any cell SD > 0.5pp, raise that cell's delta to 2 x SD (preregistration §C1 rationale).",
              "2. **§C2 anchor-flicker fallback**: see per-cell G-2 'Anchor stable?' column. If NO for any cell, that cell's anchor switches to majority-winner-across-resamples (preregistration §C2 wording).",
              "3. **§C1 H10 DEFER condition**: see per-cell G-1 'P2 viable?' column. If NO for any cell, H10 family collapses to {H9} only per preregistration §C1 wording.",
              "",
              "## TODO second pass (browser-state gates G-3 / G-5 / G-6)",
              "",
              "- G-3 P1 threshold validation (bucket SR gap at dom_size=12000 + dom_complexity=500 on archive) — needs step-1 JSONL browser-state read",
              "- G-5 runtime intent regex coverage vs codex audit Cat A/B/C/D (target >= 70%) — needs task.intent + codex_audit_*.json cross-check",
              "- G-6 hijack threshold validation ((B1+B2) + density > 90 markers cell SR ranking) — needs B1 + B2 archive (currently B0 only)",
              ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="docs/checkpoints", type=Path)
    args = ap.parse_args()

    np.random.seed(args.seed)
    report = run_all()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.utcnow().strftime("%Y-%m-%d")
    json_path = args.out_dir / f"router_archive_diagnostic_{today}.json"
    md_path = args.out_dir / f"router_archive_diagnostic_{today}.md"
    json_path.write_text(json.dumps(report, indent=2, default=str))
    md_path.write_text(render_markdown(report))
    print(f"wrote {json_path}", file=sys.stderr)
    print(f"wrote {md_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
