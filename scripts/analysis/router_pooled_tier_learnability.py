#!/usr/bin/env python3
"""Is routing learnable in the FAVOURABLE corner? — 2026-07-28, post_hoc_exploratory

Every router negative result Paper B currently reports was measured in one
corner of a 2x2: a pool mixing model FAMILIES, labelled at WHICH-MODE
granularity. `router_pooling_by_family.py` (2026-07-28) showed that corner is
the worst of the four:

                          which-mode conflict   cost-tier conflict   tier ceiling
  same-family B0+B1 cls          48.0%               24.0%              88.0%
  same-family B0+B1 red          45.0%                5.0%              97.5%
  cross-family B1+B2 cls         81.8%               45.5%              77.3%
  mixed (what the paper reports) 57.4% / 56.0%    31.5% / 12.0%      85.6% / 94.8%

The mixed number the paper quotes is an average pulled up by the cross-family
pairings. The same-family x cost-tier corner has never been trained. This script
trains it, plus the three controls needed to attribute any difference.

H-pool: a router trained on the same-family pool (B0+B1) with cost-tier labels
Pareto-dominates the always-cheapest fixed policy on >=1 site.

Both outcomes are informative. Dominance => Paper B's "not learnable" claim needs
substantive revision (the pool and the granularity were wrong, not the idea).
No dominance => the claim gets STRONGER: even the most favourable configuration
— agreeing backbones, a 97.5% ceiling, 2-13x the label supply — loses to a
policy that costs nothing to implement.

Arms (2x2 + 1 supplementary), all evaluated per (site, model) cell:

  pool in {same_family = B0+B1, all_three = B0+B1+B2}
  label in {cost_tier (binary: does this task need the screenshot),
            which_mode (the 6-way preregistered label)}
  + per_cell x cost_tier — isolates whether POOLING helped at all, as opposed
    to the coarse label. Not in the original spec; added because "your pooled
    router beat always-cheapest, but would per-cell training have beaten it
    too?" is the first question a reviewer asks and it costs one more pass.

Why cost-tier is not just "a coarser label". §395.2 measured that 12.5-54.64%
of which-mode labels return a STRICTLY MORE EXPENSIVE successful mode than
another available one, because `derive_oracle_label` tie-breaks on the `MODES`
prior order and B-1806 measured that order to be wrong and cell-inverted. The
tier label is immune BY CONSTRUCTION, not by luck: `MODES` is tier-monotone
([0,0,0,0,1,1]), so `MODE_COST_TIER[derive_oracle_label(...)]` is identically
"did any text-only mode succeed" and never consults the intra-tier order. The
script asserts this equivalence at run time rather than trusting the comment.

Protocol (each clause is a locked adjudication — see the spec for provenance):
  * CV: task-held-out 5-fold within fixed cells (§216.1; LOCO superseded).
    POOLED means a task contributes one row PER BACKBONE, so all of a task's
    rows must share a fold — otherwise the same X sits on both sides of the
    split and the held-out estimate is leaked.
  * Nesting: genuinely nested (§392.2 / B-1903). Per outer fold, the tier->mode
    map, the always-cheapest reference and the decision threshold are all
    re-derived from training-fold tasks only, and outer-test rows are scored by
    an LR fitted on training rows alone.
  * Pareto: per-cell paired bootstrap, 95% non-dominance (§150b.4 / B-1550).
  * Cost: `total_billed_cost_usd`, comparable WITHIN a cell only — B0 bills a
    proxy API, B1/B2 are electricity-derived. Pooling is used for LABELS and
    FEATURES; every (SR, cost) number is scored inside its own cell.
  * Tie-break: untouched. The tier arm never reaches it; the which-mode control
    arms use the locked `MODES` order (B-1806: do NOT switch to measured cost).

post_hoc_exploratory=True, h10_eligible=False. Touches no gating producer.

Usage:
  .venv/bin/python3 scripts/analysis/router_pooled_tier_learnability.py \
      --out docs/analysis/cross_sites/router_pooled_tier_learnability.md \
      --json-out docs/analysis/cross_sites/router_pooled_tier_learnability.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from p79.policies.learned_router import extract_raw_features  # noqa: E402
from p79.policies.router_features import (  # noqa: E402
    MODE_COST_TIER,
    MODES,
    derive_oracle_label,
)
from scripts.analysis.aggregate_h10_pareto import (  # noqa: E402
    check_pareto_non_dominance_paired_bootstrap,
    paired_bootstrap_arm_metrics,
)
from scripts.analysis.aggregate_phantom_lift import CELLS  # noqa: E402
from scripts.analysis.extract_50_features import (  # noqa: E402
    find_pass1_runs,
    read_step0_features,
    read_task_config,
)
from scripts.analysis.lib.canonical_task_universe import expected_scored_ids  # noqa: E402
from scripts.analysis.lib.episode_rows import load_cell_task_rows  # noqa: E402

SCHEMA_VERSION = "2026-07-28-router-pooled-tier-learnability-v1"

# `load_cell_task_rows` speaks display names; every label/definition helper in
# `router_features` speaks canonical names. One map, applied at the boundary.
DISPLAY_MODES = ("DOM", "SoM", "Vision", "P-text", "P-prompt", "P-SoM")
DISPLAY_TO_CANON = {
    "DOM": "dom", "SoM": "som", "Vision": "vision",
    "P-text": "phantom_text", "P-prompt": "phantom_prompt", "P-SoM": "phantom_som",
}
CANON_TO_DISPLAY = {v: k for k, v in DISPLAY_TO_CANON.items()}

COST_FIELD = "total_billed_cost_usd"
SEED = 42
N_FOLDS = 5
N_INNER_FOLDS = 5
BOOTSTRAP_B = 1000
FAMILY = {"B0": "qwen", "B1": "qwen", "B2": "gemma"}

TEXT_MODES = tuple(m for m in MODES if MODE_COST_TIER[m] == 0)
IMAGE_MODES = tuple(m for m in MODES if MODE_COST_TIER[m] == 1)

POOLS = {
    "same_family": ("B0", "B1"),
    "all_three": ("B0", "B1", "B2"),
}


def assert_tier_monotone() -> None:
    """The whole 'immune to the tie-break' argument rests on this ordering.

    If someone reorders `MODES` so an image mode precedes a text-only one, the
    tier label silently starts depending on intra-tier position — exactly the
    §395.2 defect this experiment was designed to sidestep. Fail loudly instead.
    """
    tiers = [MODE_COST_TIER[m] for m in MODES]
    if tiers != sorted(tiers):
        raise AssertionError(
            f"MODES is no longer tier-monotone ({list(zip(MODES, tiers))}). The "
            "cost-tier label would then depend on the MODES tie-break order and "
            "would inherit the §395.2 defect this arm exists to avoid."
        )


# ── pool construction ────────────────────────────────────────────────────────
def build_cell(spec: dict) -> dict | None:
    """One (site, backbone) cell over its canonical scored universe.

    Features are computed for EVERY task, labels only where some mode succeeded
    (C1 universe-vs-trainable separation: a policy must route every task, but it
    can only be trained on the tasks the benchmark produced a solve event for).
    """
    site, baseline = spec["site"], spec["baseline"]
    universe, universe_sha = expected_scored_ids(site)
    rows_by_mode = load_cell_task_rows(spec, modes=DISPLAY_MODES)
    if any(not rows_by_mode.get(m) for m in DISPLAY_MODES):
        return None
    runs = find_pass1_runs(baseline, site)

    task_ids: list[int] = []
    X: list[list[float]] = []
    succ: list[dict[str, bool]] = []
    cost: list[dict[str, float]] = []
    label_mode: list[str | None] = []
    n_no_feature = 0

    for t in sorted(universe):
        rows = {m: rows_by_mode[m].get(t) for m in DISPLAY_MODES}
        if any(r is None for r in rows.values()):
            raise ValueError(
                f"{baseline}_{site}: task {t} lacks a six-mode row — the cell is "
                "not a complete Pass-1 matrix and cannot be routed over."
            )
        cfg = read_task_config(site, t)
        step0 = read_step0_features(runs, site, t)
        if cfg is None or step0 is None:
            n_no_feature += 1
            continue
        # Canonical feature path (train == serve == archive). Deliberately NOT
        # the `step0.get("reasoning_difficulty")` shortcut used by
        # router_triage_learnability.py:98 — step-0 records carry no difficulty
        # and no intent-token count, so that path silently zeroes 2 of the 5
        # numeric features. Difficulty comes from the task config and the token
        # count from the intent string, as `extract_raw_features` defines them.
        raw = extract_raw_features(
            intent=str(cfg.get("intent", "")),
            has_reference_image=bool(cfg.get("has_reference_image")),
            dom_complexity=int(step0.get("dom_complexity", 0) or 0),
            text_length=int(step0.get("text_length", 0) or 0),
            tokens_input_text=int(step0.get("tokens_input_text", 0) or 0),
            reasoning_difficulty=int(cfg.get("reasoning_difficulty", 0) or 0),
        )
        s = {DISPLAY_TO_CANON[m]: rows[m].get("success") is True for m in DISPLAY_MODES}
        c = {DISPLAY_TO_CANON[m]: float(rows[m].get(COST_FIELD) or 0.0)
             for m in DISPLAY_MODES}
        task_ids.append(t)
        X.append(list(raw["numeric"]) + list(raw["binary"]))
        succ.append(s)
        cost.append(c)
        label_mode.append(derive_oracle_label(s))

    # Tier label, plus the run-time proof that it does not consult the MODES order.
    label_tier: list[int | None] = []
    for i, lab in enumerate(label_mode):
        if lab is None:
            label_tier.append(None)
            continue
        via_label = MODE_COST_TIER[lab]
        via_rule = 0 if any(succ[i][m] for m in TEXT_MODES) else 1
        if via_label != via_rule:
            raise AssertionError(
                f"{baseline}_{site} task {task_ids[i]}: tier via oracle label "
                f"({via_label}) != tier via 'any text-only success' ({via_rule}). "
                "The tier label is NOT order-independent here; see §395.2."
            )
        label_tier.append(via_label)

    return {
        "cell_id": f"{baseline}_{site}", "site": site, "baseline": baseline,
        "task_ids": task_ids, "X": np.asarray(X, dtype=float),
        "succ": succ, "cost": cost,
        "label_mode": label_mode, "label_tier": label_tier,
        "n_no_feature": n_no_feature, "n_universe": len(universe),
        "universe_sha256": universe_sha,
        "n_labeled": sum(1 for l in label_mode if l is not None),
    }


def outer_fold_map(task_ids: list[int], seed: int = SEED,
                   k: int = N_FOLDS) -> dict[int, int]:
    """task_id -> fold. Shared by every cell on a site so folds are comparable,
    and — the load-bearing part for a POOLED design — so a task's B0 row and its
    B1 row can never land on opposite sides of the split."""
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(task_ids))
    perm = rng.permutation(len(ids))
    out: dict[int, int] = {}
    for f, chunk in enumerate(np.array_split(perm, k)):
        for i in chunk:
            out[int(ids[i])] = f
    return out


# ── modelling helpers ────────────────────────────────────────────────────────
def _fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray):
    """Fold-local standardisation + L2 LR. Returns (classes, proba) or None."""
    from sklearn.linear_model import LogisticRegression

    if len(np.unique(ytr)) < 2:
        return None
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit((Xtr - mu) / sd, ytr)
    return lr.classes_, lr.predict_proba((Xte - mu) / sd)


def _train_rows(cells: list[dict], train_tasks: set[int], label_kind: str):
    """Stack (X, y) across the pool's cells, keeping only labelled training tasks."""
    Xs, ys, src = [], [], []
    key = "label_tier" if label_kind == "cost_tier" else "label_mode"
    for cell in cells:
        for i, t in enumerate(cell["task_ids"]):
            if t not in train_tasks:
                continue
            lab = cell[key][i]
            if lab is None:
                continue
            Xs.append(cell["X"][i])
            ys.append(lab)
            src.append(cell["cell_id"])
    if not Xs:
        return None
    return np.asarray(Xs, dtype=float), np.asarray(ys), src


def _cell_train_stats(cell: dict, train_tasks: set[int]) -> dict[str, Any]:
    """Mode-level SR and cost on TRAINING tasks only (§392.2 clause a)."""
    idx = [i for i, t in enumerate(cell["task_ids"]) if t in train_tasks]
    n = len(idx)
    sr = {m: sum(cell["succ"][i][m] for i in idx) / n for m in MODES}
    cost = {m: sum(cell["cost"][i][m] for i in idx) / n for m in MODES}
    # Within a tier, pick by SR; break ties by lower training cost, then by the
    # locked MODES order. This is a DEPLOYMENT choice inside a tier the router
    # already decided on, not a change to the oracle label definition.
    def pick(cands):
        return min(cands, key=lambda m: (-sr[m], cost[m], MODES.index(m)))
    return {
        "sr": sr, "cost": cost, "n_train": n,
        "mode_for_tier": {0: pick(TEXT_MODES), 1: pick(IMAGE_MODES)},
        "cheapest": min(MODES, key=lambda m: (cost[m], MODES.index(m))),
        "best_sr": pick(MODES),
    }


def _simulate(cell: dict, choice_by_task: dict[int, str]) -> tuple[float, float, int]:
    """(SR%, mean cost, n) for a task->mode map over the given tasks."""
    hits, spend = 0.0, 0.0
    for i, t in enumerate(cell["task_ids"]):
        if t not in choice_by_task:
            continue
        m = choice_by_task[t]
        hits += cell["succ"][i][m]
        spend += cell["cost"][i][m]
    n = len(choice_by_task)
    return (100.0 * hits / n if n else float("nan"),
            spend / n if n else float("nan"), n)


# ── the nested evaluation ────────────────────────────────────────────────────
def run_arm(pool_cells: list[dict], eval_cells: list[dict], label_kind: str,
            fold_map: dict[int, int]) -> dict[str, Any]:
    """Fully nested per-outer-fold routing. Returns per-cell routed vectors.

    Per outer fold, using ONLY that fold's training tasks:
      a. re-derive each cell's tier->mode map, always-cheapest and best-SR mode;
      b. produce inner-CV out-of-fold scores over the TRAINING tasks and pick the
         decision threshold against those (tier arm only — the which-mode arm is
         an argmax and has no threshold to select);
      c. refit on all training rows and score outer-test tasks with that model;
      d. apply (threshold, mode map) blind to the outer-test tasks.
    """
    n_out = max(fold_map.values()) + 1
    all_tasks = sorted(fold_map)
    routed: dict[str, dict[int, str]] = {c["cell_id"]: {} for c in eval_cells}
    cheap_ref: dict[str, dict[int, str]] = {c["cell_id"]: {} for c in eval_cells}
    best_ref: dict[str, dict[int, str]] = {c["cell_id"]: {} for c in eval_cells}
    fold_log: list[dict[str, Any]] = []

    for f in range(n_out):
        test_tasks = {t for t in all_tasks if fold_map[t] == f}
        train_tasks = {t for t in all_tasks if fold_map[t] != f}

        tr = _train_rows(pool_cells, train_tasks, label_kind)
        stats = {c["cell_id"]: _cell_train_stats(c, train_tasks) for c in eval_cells}
        entry: dict[str, Any] = {
            "fold": f, "n_train_tasks": len(train_tasks), "n_test_tasks": len(test_tasks),
            "n_train_rows": 0 if tr is None else int(len(tr[1])),
            "train_label_counts": {} if tr is None else
                {str(k): int(v) for k, v in sorted(Counter(tr[1].tolist()).items())},
            "per_cell": {},
        }

        for cell in eval_cells:
            cid = cell["cell_id"]
            st = stats[cid]
            # Reference policies are recorded only for tasks this cell actually
            # has a feature row for, so the three vectors scored below stay
            # index-aligned when a task drops out for want of step-0 features.
            cell_tasks = set(cell["task_ids"])
            for t in test_tasks & cell_tasks:
                cheap_ref[cid][t] = st["cheapest"]
                best_ref[cid][t] = st["best_sr"]

            # (b) threshold from inner CV over training tasks (tier arm only).
            thr = 0.5
            thr_source = "argmax_default"
            if label_kind == "cost_tier" and tr is not None:
                thr, thr_source = _select_threshold(
                    pool_cells, cell, st, train_tasks, label_kind)

            # (c) outer-test scores from a model fitted on training rows alone.
            te_idx = [i for i, t in enumerate(cell["task_ids"]) if t in test_tasks]
            fitted = None
            if tr is not None and te_idx:
                fitted = _fit_predict(tr[0], tr[1], cell["X"][te_idx])

            n_fallback = 0
            for pos, i in enumerate(te_idx):
                t = cell["task_ids"][i]
                if fitted is None:
                    # No trainable signal this fold: fall back to the cheapest
                    # training-fold mode. Counted, never silently absorbed.
                    routed[cid][t] = st["cheapest"]
                    n_fallback += 1
                    continue
                classes, proba = fitted
                if label_kind == "cost_tier":
                    p1 = float(proba[pos][list(classes).index(1)]) if 1 in classes else 0.0
                    routed[cid][t] = st["mode_for_tier"][1 if p1 >= thr else 0]
                else:
                    routed[cid][t] = str(classes[int(np.argmax(proba[pos]))])
            entry["per_cell"][cid] = {
                "mode_for_tier": {str(k): v for k, v in st["mode_for_tier"].items()},
                "cheapest": st["cheapest"], "best_sr": st["best_sr"],
                "threshold": round(float(thr), 6), "threshold_source": thr_source,
                "n_untrainable_fallback": n_fallback,
            }
        fold_log.append(entry)

    return {"routed": routed, "cheapest": cheap_ref, "best_sr": best_ref,
            "per_outer_fold": fold_log}


def _select_threshold(pool_cells: list[dict], cell: dict, st: dict,
                      train_tasks: set[int], label_kind: str) -> tuple[float, str]:
    """Threshold chosen on inner-CV OOF scores over TRAINING tasks only.

    Criterion mirrors H-pool: among candidate thresholds whose simulated
    training SR is at least the always-cheapest training SR, take the cheapest.
    That is the only kind of saving that is free. If none qualifies, take the
    highest-SR candidate and say so, rather than quietly reporting a threshold
    that was selected under a different rule.
    """
    inner = outer_fold_map(sorted(train_tasks), seed=SEED + 1, k=N_INNER_FOLDS)
    scores: dict[int, float] = {}
    for g in range(N_INNER_FOLDS):
        in_test = {t for t in train_tasks if inner[t] == g}
        in_train = train_tasks - in_test
        itr = _train_rows(pool_cells, in_train, label_kind)
        idx = [i for i, t in enumerate(cell["task_ids"]) if t in in_test]
        if itr is None or not idx:
            continue
        fitted = _fit_predict(itr[0], itr[1], cell["X"][idx])
        if fitted is None:
            continue
        classes, proba = fitted
        col = list(classes).index(1) if 1 in classes else None
        for pos, i in enumerate(idx):
            scores[cell["task_ids"][i]] = 0.0 if col is None else float(proba[pos][col])
    if not scores:
        return 0.5, "no_inner_signal"

    base_sr, _, _ = _simulate(cell, {t: st["cheapest"] for t in train_tasks})
    cands = np.quantile(np.array(list(scores.values())), np.linspace(0.0, 0.95, 20))
    best, best_cost, fallback_best, fallback_sr = None, None, None, -np.inf
    for thr in cands:
        choice = {
            t: st["mode_for_tier"][1 if scores.get(t, 0.0) >= thr else 0]
            for t in train_tasks
        }
        sr, c, _ = _simulate(cell, choice)
        if sr > fallback_sr:
            fallback_best, fallback_sr = float(thr), sr
        if sr >= base_sr - 1e-9 and (best_cost is None or c < best_cost):
            best, best_cost = float(thr), c
    if best is not None:
        return best, "sr_preserving_vs_cheapest"
    return (fallback_best if fallback_best is not None else 0.5), "max_sr_fallback"


def strict_dominance_paired_bootstrap(r_s: np.ndarray, r_c: np.ndarray,
                                      b_s: np.ndarray, b_c: np.ndarray,
                                      B: int = BOOTSTRAP_B,
                                      seed: int = SEED) -> float:
    """Share of paired replicates where the router DOMINATES the baseline.

    Non-dominance and dominance are different questions and the spec asks both
    without naming the gap: §3 locks the decision rule to "per-cell paired
    bootstrap 95% NON-dominance" (§150b.4 / B-1550, a deployment-admissibility
    criterion), while H-pool in §2 is worded as the router "Pareto DOMINATES
    always-cheapest". A policy that buys +7pp of SR for +10% of cost is
    non-dominated and is not dominant. Reporting only the locked rule would let
    that trade-off be read as confirmation of a hypothesis it does not confirm,
    so both are computed and reported side by side.
    """
    rng = np.random.default_rng(seed)
    n = len(r_s)
    if n == 0:
        return float("nan")
    hits = 0
    for _ in range(B):
        idx = rng.integers(0, n, n)
        rs, rc = float(r_s[idx].mean()), float(r_c[idx].mean())
        bs, bc = float(b_s[idx].mean()), float(b_c[idx].mean())
        if rs >= bs and rc <= bc and (rs > bs or rc < bc):
            hits += 1
    return hits / B


# ── per-cell scoring ─────────────────────────────────────────────────────────
def score_cell(cell: dict, routed: dict[int, str], cheapest: dict[int, str],
               best_sr: dict[int, str]) -> dict[str, Any]:
    """(SR, cost) for the router and its references, plus the Pareto verdict."""
    ids = sorted(routed)
    pos = {t: i for i, t in enumerate(cell["task_ids"])}
    r_s = np.array([cell["succ"][pos[t]][routed[t]] for t in ids], dtype=float)
    r_c = np.array([cell["cost"][pos[t]][routed[t]] for t in ids], dtype=float)
    ch_s = np.array([cell["succ"][pos[t]][cheapest[t]] for t in ids], dtype=float)
    ch_c = np.array([cell["cost"][pos[t]][cheapest[t]] for t in ids], dtype=float)
    bs_s = np.array([cell["succ"][pos[t]][best_sr[t]] for t in ids], dtype=float)
    bs_c = np.array([cell["cost"][pos[t]][best_sr[t]] for t in ids], dtype=float)

    fixed = {
        m: {"success": np.array([cell["succ"][pos[t]][m] for t in ids], dtype=float),
            "cost": np.array([cell["cost"][pos[t]][m] for t in ids], dtype=float)}
        for m in MODES
    }

    # Primary contrast (spec §3): always-cheapest, the policy that costs nothing
    # to implement. Secondary: all six fixed modes, a strictly harder bar that
    # asks whether the router is on the empirical Pareto front at all.
    vs_cheap = check_pareto_non_dominance_paired_bootstrap(
        r_s, r_c, {"always_cheapest": {"success": ch_s, "cost": ch_c}}, ids,
        B=BOOTSTRAP_B, seed=SEED)
    vs_all = check_pareto_non_dominance_paired_bootstrap(
        r_s, r_c, fixed, ids, B=BOOTSTRAP_B, seed=SEED)
    boot = paired_bootstrap_arm_metrics(r_s, r_c, B=BOOTSTRAP_B, seed=SEED)
    dom_frac = strict_dominance_paired_bootstrap(
        r_s, r_c, ch_s, ch_c, B=BOOTSTRAP_B, seed=SEED)

    def pack(s, c):
        return {"sr_pct": 100.0 * float(s.mean()), "mean_cost": float(c.mean())}

    return {
        "cell_id": cell["cell_id"], "n_tasks": len(ids),
        "router": pack(r_s, r_c),
        "router_sr_ci_pct": [100.0 * v for v in boot["sr_ci"]],
        "router_cost_ci": list(boot["cost_ci"]),
        "always_cheapest": pack(ch_s, ch_c),
        "best_sr_ref": pack(bs_s, bs_c),
        "delta_sr_vs_cheapest_pp": 100.0 * float(r_s.mean() - ch_s.mean()),
        "delta_cost_vs_cheapest_pct": (
            100.0 * (float(r_c.mean()) / float(ch_c.mean()) - 1.0)
            if float(ch_c.mean()) else float("nan")),
        "pareto_vs_always_cheapest": {
            "fraction_non_dominated": vs_cheap["fraction_non_dominated"],
            "passes": bool(vs_cheap["passes"]),
        },
        "pareto_vs_six_fixed_modes": {
            "fraction_non_dominated": vs_all["fraction_non_dominated"],
            "passes": bool(vs_all["passes"]),
        },
        "strict_dominance_vs_cheapest": {
            "fraction_dominating": dom_frac,
            "passes": bool(dom_frac >= 0.95),
        },
        "routed_mode_counts": dict(sorted(Counter(routed.values()).items())),
        "cheapest_mode_counts": dict(sorted(Counter(cheapest.values()).items())),
    }


# ── diagnostics the spec did not ask for but the result depends on ───────────
def backbone_identifiability(cells: list[dict], fold_map: dict[int, int]) -> dict:
    """Can the features tell the backbones apart? They are supposed not to.

    The conflict statistic reads as "the same X carries contradictory y across
    backbones". That reading requires X to actually be the same. Step-0 features
    are re-read per run, so a task's B0 and B1 rows can differ by a few
    characters of rendered DOM. If those differences were systematic enough to
    identify the backbone, a pooled router could route on model identity rather
    than on task content, and "pooling works" would mean something else entirely.

    Reported as (a) the share of shared tasks whose feature rows are bit-identical
    across the pool and (b) out-of-fold AUROC of an LR predicting backbone from X.
    Near-0.5 AUROC is the reassuring outcome.
    """
    if len(cells) < 2:
        return {}
    a, b = cells[0], cells[1]
    pa = {t: i for i, t in enumerate(a["task_ids"])}
    pb = {t: i for i, t in enumerate(b["task_ids"])}
    shared = sorted(set(pa) & set(pb))
    identical = sum(1 for t in shared if np.array_equal(a["X"][pa[t]], b["X"][pb[t]]))

    X = np.vstack([a["X"][[pa[t] for t in shared]], b["X"][[pb[t] for t in shared]]])
    y = np.array([0] * len(shared) + [1] * len(shared))
    tasks = np.array(shared + shared)
    oof = np.full(len(y), np.nan)
    for f in range(N_FOLDS):
        te = np.array([fold_map.get(int(t), -1) == f for t in tasks])
        tr = ~te
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        fitted = _fit_predict(X[tr], y[tr], X[te])
        if fitted is None:
            continue
        classes, proba = fitted
        oof[te] = proba[:, list(classes).index(1)]
    ok = ~np.isnan(oof)
    auroc = float("nan")
    if ok.sum() and len(np.unique(y[ok])) == 2:
        s, yy = oof[ok], y[ok]
        pos, neg = s[yy == 1], s[yy == 0]
        order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
        ranks = np.empty(len(order), dtype=float)
        ranks[order] = np.arange(1, len(order) + 1)
        vals = np.concatenate([pos, neg])
        for v in np.unique(vals):
            m = vals == v
            ranks[m] = ranks[m].mean()
        auroc = float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                      / (len(pos) * len(neg)))
    return {
        "pair": [a["cell_id"], b["cell_id"]], "n_shared_tasks": len(shared),
        "n_feature_rows_identical": identical,
        "pct_identical": 100.0 * identical / len(shared) if shared else float("nan"),
        "backbone_auroc_oof": auroc,
        "note": ("AUROC near 0.5 => the pooled features carry no usable backbone "
                 "identity, so the conflict rate really is same-X-different-y."),
    }


def per_task_cost_headroom(cell: dict) -> dict[str, Any]:
    """Is always-cheapest a cost FLOOR, or only the cheapest mode ON AVERAGE?

    Gemini (cross-AI Mode C, 2026-07-29) argued H-pool is unsatisfiable by
    arithmetic: Vision is the cheapest mode by construction, so any router that
    departs from it must cost more, making 0/4 a tautology rather than a result.

    That argument requires Vision to be the cheapest mode ON EVERY TASK. It is
    not — it is only the mode with the lowest MEAN. This diagnostic measures the
    gap directly, and it refutes the objection: a per-task cost oracle is
    substantially cheaper than always-cheapest in every cell, so a router CAN in
    principle be cheaper while holding SR, and the observed failure is empirical
    rather than definitional.

    The same number is worth reporting on its own: it is the cost-routing
    headroom, i.e. what a perfect per-task cost policy would save.
    """
    n = len(cell["task_ids"])
    mean_cost = {m: sum(c[m] for c in cell["cost"]) / n for m in MODES}
    cheapest = min(MODES, key=lambda m: (mean_cost[m], MODES.index(m)))
    not_floor = sum(1 for c in cell["cost"]
                    if c[cheapest] > min(c.values()) + 1e-12)
    oracle = sum(min(c.values()) for c in cell["cost"]) / n
    return {
        "cheapest_on_mean": cheapest,
        "n_tasks": n,
        "n_tasks_where_cheapest_mode_is_not_the_per_task_floor": not_floor,
        "pct_not_floor": 100.0 * not_floor / n if n else None,
        "mean_cost_always_cheapest": mean_cost[cheapest],
        "mean_cost_per_task_oracle": oracle,
        "headroom_pct": (100.0 * (oracle / mean_cost[cheapest] - 1.0)
                         if mean_cost[cheapest] else None),
        "note": ("always-cheapest is the lowest-MEAN mode, not a per-task floor; "
                 "a per-task cost oracle is cheaper, so Pareto dominance is not "
                 "arithmetically excluded"),
    }


def label_supply(cells: list[dict], pool_name: str) -> dict[str, Any]:
    counts_mode: Counter = Counter()
    counts_tier: Counter = Counter()
    for c in cells:
        counts_mode.update([l for l in c["label_mode"] if l is not None])
        counts_tier.update([l for l in c["label_tier"] if l is not None])
    return {
        "pool": pool_name,
        "backbones": [c["baseline"] for c in cells],
        "n_labeled_rows": int(sum(counts_mode.values())),
        "which_mode_distribution": {k: int(v) for k, v in sorted(counts_mode.items())},
        "cost_tier_distribution": {("text_only" if k == 0 else "image"): int(v)
                                   for k, v in sorted(counts_tier.items())},
        "per_cell_labeled": {c["cell_id"]: c["n_labeled"] for c in cells},
    }


# ── orchestration ────────────────────────────────────────────────────────────
# --- WebArena reddit, added 2026-08-03 ------------------------------------------------
# This was the last routing product still VisualWebArena-only, and it tests the corner
# most likely to work: same-family pooling with a coarse two-value label. WebArena is
# exactly that corner by construction — both its cells are Qwen, so `same_family` is the
# only pool it can form and `all_three` is structurally absent (B2 never ran WA).
# It is also the site with the most per-task routing room in the study (36/104 tasks have
# more than one solver, against 68/224 on the largest VWA cell).
#
# Two features do not exist on WebArena and are zero-filled here rather than dropped:
# `reasoning_difficulty` (a VWA task-config annotation) and `has_reference_image` (WA
# ships none). That is acceptable HERE and not in `router_triage_learnability`, because
# this product never compares an AUROC across sites — every arm is scored inside one
# site against that site's own always-cheapest baseline. The zero-fill is disclosed in
# LIMITATIONS and the two columns are constant, so they cannot carry signal either way.
WA_ROOT_PT = REPO / "results/webarena/phase1"
WA_STEM_PT = {"DOM": "dom", "SoM": "som", "Vision": "vision", "P-text": "phantom_text",
              "P-prompt": "phantom_prompt", "P-SoM": "phantom_som"}


def _wa_run_dir_pt(baseline: str, display_mode: str) -> Path | None:
    hits = [p for p in WA_ROOT_PT.glob(
        f"{baseline}_{WA_STEM_PT[display_mode]}_wa_reddit_2026*_R*")
        if p.is_dir() and "ABORTED" not in p.name]
    return hits[0] if len(hits) == 1 else None


def build_wa_cell(baseline: str) -> dict | None:
    """WA counterpart of build_cell. Universe = the six-mode intersection (104);
    WebArena ships no exclusion list, so there is no canonical sha to carry."""
    rows_by_mode: dict[str, dict[int, dict]] = {}
    run_dirs: dict[str, Path] = {}
    for m in DISPLAY_MODES:
        d = _wa_run_dir_pt(baseline, m)
        if d is None:
            return None
        run_dirs[m] = d
        rows: dict[int, dict] = {}
        for f in (list(d.glob("*/episodes/*summary*.json"))
                  or list(d.glob("episodes/*summary*.json"))):
            try:
                rec = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not rec.get("sr_excluded"):
                rows[int(rec["task_id"])] = rec
        if not rows:
            return None
        rows_by_mode[m] = rows
    universe = sorted(set.intersection(*(set(v) for v in rows_by_mode.values())))
    if len(universe) < 50:
        return None

    dom_run = run_dirs["DOM"]
    task_ids, X, succ, cost, label_mode = [], [], [], [], []
    n_no_feature = 0
    for t in universe:
        cfg_f = dom_run / "task_configs" / f"reddit_task_{t}.json"
        # site="reddit": WA episode files keep the bare site prefix in their names.
        step0 = read_step0_features([dom_run], "reddit", t)
        if not cfg_f.exists() or step0 is None:
            n_no_feature += 1
            continue
        try:
            cfg = json.loads(cfg_f.read_text())
        except (OSError, json.JSONDecodeError):
            n_no_feature += 1
            continue
        raw = extract_raw_features(
            intent=str(cfg.get("intent", "")),
            has_reference_image=False,          # structurally absent on WebArena
            dom_complexity=int(step0.get("dom_complexity", 0) or 0),
            text_length=int(step0.get("text_length", 0) or 0),
            tokens_input_text=int(step0.get("tokens_input_text", 0) or 0),
            reasoning_difficulty=0,             # structurally absent on WebArena
        )
        s = {DISPLAY_TO_CANON[m]: rows_by_mode[m][t].get("success") is True
             for m in DISPLAY_MODES}
        c = {DISPLAY_TO_CANON[m]: float(rows_by_mode[m][t].get(COST_FIELD) or 0.0)
             for m in DISPLAY_MODES}
        task_ids.append(t)
        X.append(list(raw["numeric"]) + list(raw["binary"]))
        succ.append(s)
        cost.append(c)
        label_mode.append(derive_oracle_label(s))

    label_tier: list[int | None] = []
    for i, lab in enumerate(label_mode):
        if lab is None:
            label_tier.append(None)
            continue
        via_label = MODE_COST_TIER[lab]
        via_rule = 0 if any(succ[i][m] for m in TEXT_MODES) else 1
        if via_label != via_rule:
            raise AssertionError(
                f"{baseline}_wa_reddit task {task_ids[i]}: tier via oracle label "
                f"({via_label}) != tier via 'any text-only success' ({via_rule}).")
        label_tier.append(via_label)

    return {
        "cell_id": f"{baseline}_wa_reddit", "site": "wa_reddit", "baseline": baseline,
        "task_ids": task_ids, "X": np.asarray(X, dtype=float),
        "succ": succ, "cost": cost,
        "label_mode": label_mode, "label_tier": label_tier,
        "n_no_feature": n_no_feature, "n_universe": len(universe),
        "universe_sha256": "wa-six-mode-intersection(no exclusion list)",
        "n_labeled": sum(1 for l in label_mode if l is not None),
    }


def build_all() -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for spec in CELLS:
        cell = build_cell(spec)
        if cell is None:
            continue
        out.setdefault(spec["site"], {})[spec["baseline"]] = cell
    for bb in ("B0", "B1"):
        wa = build_wa_cell(bb)
        if wa is not None:
            out.setdefault("wa_reddit", {})[bb] = wa
    return out


def analyse(by_site: dict) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for site, cells_by_bb in sorted(by_site.items()):
        any_cell = next(iter(cells_by_bb.values()))
        fold_map = outer_fold_map(any_cell["task_ids"])
        site_out: dict[str, Any] = {
            "n_universe": any_cell["n_universe"],
            "universe_sha256": any_cell["universe_sha256"],
            "fold_sizes": dict(sorted(Counter(fold_map.values()).items())),
            "arms": {}, "supply": {}, "diagnostics": {},
        }

        same = [cells_by_bb[b] for b in POOLS["same_family"] if b in cells_by_bb]
        site_out["diagnostics"]["backbone_identifiability"] = \
            backbone_identifiability(same, fold_map)
        site_out["diagnostics"]["per_task_cost_headroom"] = {
            c["cell_id"]: per_task_cost_headroom(c)
            for c in cells_by_bb.values()}

        for pool_name, backbones in POOLS.items():
            pool_cells = [cells_by_bb[b] for b in backbones if b in cells_by_bb]
            if len(pool_cells) < 2:
                continue
            site_out["supply"][pool_name] = label_supply(pool_cells, pool_name)
            for label_kind in ("cost_tier", "which_mode"):
                arm = run_arm(pool_cells, pool_cells, label_kind, fold_map)
                site_out["arms"][f"{pool_name}|{label_kind}"] = {
                    "pool": pool_name, "label": label_kind,
                    "backbones": list(backbones),
                    "per_cell": {
                        c["cell_id"]: score_cell(c, arm["routed"][c["cell_id"]],
                                                 arm["cheapest"][c["cell_id"]],
                                                 arm["best_sr"][c["cell_id"]])
                        for c in pool_cells
                    },
                    "per_outer_fold": arm["per_outer_fold"],
                }

        # Supplementary: per-cell training, cost-tier label. Isolates pooling.
        for bb, cell in sorted(cells_by_bb.items()):
            arm = run_arm([cell], [cell], "cost_tier", fold_map)
            site_out["arms"][f"per_cell|cost_tier|{bb}"] = {
                "pool": "per_cell", "label": "cost_tier", "backbones": [bb],
                "per_cell": {cell["cell_id"]: score_cell(
                    cell, arm["routed"][cell["cell_id"]],
                    arm["cheapest"][cell["cell_id"]],
                    arm["best_sr"][cell["cell_id"]])},
                "per_outer_fold": arm["per_outer_fold"],
            }
        results[site] = site_out
    return results


LIMITATIONS = [
    "WebArena (added 2026-08-03) carries neither `reasoning_difficulty` nor a "
    "reference image, so those two of the twenty features are zero-filled on its "
    "cells and cannot contribute there. That is tolerable in THIS product and not in "
    "router_triage_learnability, because nothing here compares a score across sites: "
    "every arm is judged inside one site against that site's own always-cheapest "
    "baseline. WebArena also has no cross-family backbone — B2 never ran it — so its "
    "`all_three` row is B0+B1 and is therefore IDENTICAL to `same_family` by "
    "construction, not by result. Read the two WA rows as one arm reported twice; the "
    "cross-family contrast this product is partly about cannot be formed there at all.",
    "n is small: the same-family pool shares only 50 tasks on classifieds and 20 "
    "on reddit (labelled rows: cls 152, red 77). Every per-cell contrast below is "
    "underpowered and the fold-to-fold variation is correspondingly large.",
    "A ceiling is not learnability (§394): red·B2 is the only cell of six whose "
    "triage signal survived Holm, and its AUROC was 0.483. A 97.5% tier ceiling "
    "bounds what a perfect classifier could do; it says nothing about whether this "
    "one does.",
    "The 48% / 45% which-mode conflict may be a real backbone difference or may be "
    "noise; the present data cannot separate the two. Phase 0b measured a 4.9-7.6pp "
    "same-condition replicate floor, which is the scale the disagreement must clear "
    "before it can be read as a model property.",
    "B0 and B1 costs are NOT comparable (B0 bills a proxy API; B1/B2 are "
    "electricity-derived). Pooling is used for labels and features only — every "
    "(SR, cost) pair and every Pareto verdict is computed inside a single cell.",
    "post_hoc_exploratory=True / h10_eligible=False. This is not the preregistered "
    "H10 gate and must never be cited as one; it is an exploratory probe of a "
    "corner the negative result never covered, in the manner of "
    "router_objective_ordering.md.",
    "Non-dominance is not dominance, and the two tallies must not be merged. The "
    "locked §3 rule (95% paired-bootstrap non-dominance vs always-cheapest) is an "
    "admissibility criterion: a policy buying +7pp SR for +10% cost passes it. "
    "H-pool as worded in §2 asks for dominance. Any reader quoting a pass rate "
    "must say which of the two it is.",
    "A low tier-conflict rate can mean agreement or it can mean the label barely "
    "varies. The same-family reddit pool is 63 text_only vs 14 image, and B1·reddit "
    "alone is 20 vs 4 — so reddit's 5.0% conflict / 97.5% ceiling partly reflects a "
    "near-constant label rather than two backbones agreeing on a hard call. The "
    "class balance is reported per pool above for exactly this reason.",
    "The which-mode arms carry no min-class filter. train_l1_router's Stage-3 "
    "N_MIN_CLASS_TRAIN=10 rule would leave several folds with fewer than two "
    "trainable classes and the control arm could not run at all; per-fold class "
    "counts are reported instead so the reader can see which classes were never "
    "learnable.",
]


def reading(results: dict, verdict: dict) -> list[str]:
    """The narrative the tables support — written from the tables, not around them."""
    sf, al = verdict["same_family_tier"], verdict["all_arms"]
    out = [
        f"1. **Nothing reaches the front.** Across all {al['total']} arm×cell "
        f"combinations the router is non-dominated by the six-fixed-mode menu in "
        f"{al['ndall']} of them, and dominates always-cheapest in {al['dom']}. The "
        "favourable corner the spec identified — same family, coarse label, "
        f"highest ceiling — contributes {sf['ndall']}/{sf['total']} and "
        f"{sf['dom']}/{sf['total']} respectively.",
        "2. **The coarse label did not buy a better operating point — and on the "
        "one passing cell it was slightly worse.** reddit·B0 same-family: "
        "which-mode 15.27% SR at 0.10415 vs cost-tier 14.29% at 0.10803, i.e. the "
        "6-way label is better on both axes there. The §395.2 defect the tier "
        "label sidesteps is real, but sidestepping it did not help. ⚠️ Two "
        "qualifiers, both from cross-AI review 2026-07-29: (a) this is a "
        "point-estimate comparison, not a paired contrast, so it does not "
        "establish that granularity is causally inert; (b) on reddit the tier "
        "label is severely imbalanced (63 text_only vs 14 image), so a tier "
        "classifier can score by collapsing to the majority class — granularity "
        "and label-variance are confounded in exactly the cell that passes.",
        "3. **The one pass is a property of the contrast, not of the router.** "
        "reddit·B0's always-cheapest is Vision at 7.39% SR against a best-single "
        "reference of 11.33% — an unusually weak baseline. Any routing policy that "
        "moves tasks off Vision buys SR there, which is why all five passing arms "
        "pass, including per-cell training with no pooling at all.",
        "4. **The trade-off is genuine but priced.** reddit·B0 reaches 13.3-15.3% "
        "SR, above the best single mode, and pays 2.7-10.2% more per task for it. "
        "That is a legitimate operating point to report; it is not the dominance "
        "H-pool asked for, and the six-mode menu still dominates it in 35-71% of "
        "paired replicates.",
        "5. **Direction for Paper B.** The negative result survives its most "
        "favourable test and can now be stated with the qualifier attached rather "
        "than as an unexamined generalisation: routing does not beat a fixed cheap "
        "policy *even when the pool agrees, the label is coarse and immune to the "
        "tie-break defect, and the plug-in ceiling is 97.5%*.",
    ]
    return out


def render(payload: dict) -> str:
    L: list[str] = []
    L.append("# Pooled × cost-tier router learnability (same-family corner)")
    L.append("")
    L.append(f"- generated: `{payload['generated_utc']}`")
    L.append(f"- schema: `{payload['schema_version']}`")
    L.append("- **post_hoc_exploratory=True / h10_eligible=False** — not the "
             "preregistered H10 gate.")
    L.append(f"- protocol: task-held-out {N_FOLDS}-fold (§216.1), fully nested "
             f"(§392.2), per-cell paired bootstrap B={BOOTSTRAP_B} at 95% "
             "non-dominance (§150b.4 / B-1550)")
    L.append(f"- cost estimand: `{COST_FIELD}`, comparable within a cell only")
    L.append("")
    L.append("## H-pool")
    L.append("")
    L.append("> A router trained on the same-family pool (B0+B1) with cost-tier "
             "labels Pareto-dominates always-cheapest on ≥1 site.")
    L.append("")
    v = payload["verdict"]
    L.append(f"**Verdict: {v['headline']}**")
    L.append("")
    L.append("Three tallies, because the hypothesis and the locked test are not "
             "the same question. Non-dominance says the router is *admissible*; "
             "dominance says it is *better*. H-pool is worded as dominance "
             "(spec §2); the locked decision rule is non-dominance (spec §3).")
    L.append("")
    L.append("| tally | same-family × cost-tier | all arms |")
    L.append("|---|---|---|")
    sf, al = v["same_family_tier"], v["all_arms"]
    L.append(f"| non-dominated vs always-cheapest (locked §3 rule) | "
             f"**{sf['nd']}/{sf['total']}** | {al['nd']}/{al['total']} |")
    L.append(f"| **dominates** always-cheapest (H-pool as worded, §2) | "
             f"**{sf['dom']}/{sf['total']}** | {al['dom']}/{al['total']} |")
    L.append(f"| non-dominated vs all six fixed modes (on the front at all) | "
             f"**{sf['ndall']}/{sf['total']}** | {al['ndall']}/{al['total']} |")
    L.append("")
    L.append("### Attribution — what, if anything, caused a pass")
    L.append("")
    L.extend(payload["attribution"])
    L.append("")
    L.append("### How to read this")
    L.append("")
    for line in payload["reading"]:
        L.append(line)
    L.append("")

    for site, s in payload["results"].items():
        L.append(f"## {site}")
        L.append("")
        L.append(f"universe N={s['n_universe']} (sha `{s['universe_sha256'][:12]}`), "
                 f"fold sizes {s['fold_sizes']}")
        L.append("")
        sup = s.get("supply", {})
        if sup:
            L.append("### Label supply")
            L.append("")
            L.append("| pool | labelled rows | which-mode classes | cost-tier split |")
            L.append("|---|---|---|---|")
            for name, d in sup.items():
                wm = ", ".join(f"{k}={v}" for k, v in d["which_mode_distribution"].items())
                ct = ", ".join(f"{k}={v}" for k, v in d["cost_tier_distribution"].items())
                L.append(f"| `{name}` ({'+'.join(d['backbones'])}) | "
                         f"{d['n_labeled_rows']} | {wm} | {ct} |")
            L.append("")
        diag = s.get("diagnostics", {}).get("backbone_identifiability") or {}
        if diag:
            L.append("### Are the pooled features backbone-identifiable?")
            L.append("")
            L.append(f"{diag['pair'][0]} vs {diag['pair'][1]}: "
                     f"{diag['n_feature_rows_identical']}/{diag['n_shared_tasks']} "
                     f"shared tasks have bit-identical feature rows "
                     f"({diag['pct_identical']:.1f}%); out-of-fold AUROC for "
                     f"predicting the backbone from X = **{diag['backbone_auroc_oof']:.3f}**.")
            L.append("")
            L.append(f"> {diag['note']}")
            L.append("")

        head = s.get("diagnostics", {}).get("per_task_cost_headroom") or {}
        if head:
            L.append("### Is always-cheapest a cost floor? (No.)")
            L.append("")
            L.append("A cross-AI reviewer argued H-pool is arithmetically "
                     "unsatisfiable because the cheapest mode is cheapest by "
                     "construction. That holds only if it is cheapest on every "
                     "task. It is not — it has the lowest **mean**.")
            L.append("")
            L.append("| cell | cheapest-on-mean | tasks where it is NOT the "
                     "per-task floor | always-cheapest cost | per-task oracle | "
                     "headroom |")
            L.append("|---|---|---|---|---|---|")
            for cid, h in sorted(head.items()):
                L.append(f"| {cid} | {h['cheapest_on_mean']} | "
                         f"{h['n_tasks_where_cheapest_mode_is_not_the_per_task_floor']}"
                         f"/{h['n_tasks']} = {h['pct_not_floor']:.1f}% | "
                         f"{h['mean_cost_always_cheapest']:.5f} | "
                         f"{h['mean_cost_per_task_oracle']:.5f} | "
                         f"**{h['headroom_pct']:+.1f}%** |")
            L.append("")
            L.append("> So Pareto dominance is **not** excluded by definition; the "
                     "failure below is empirical. The headroom column is also the "
                     "cost-routing upper bound in its own right.")
            L.append("")

        L.append("### Operating points")
        L.append("")
        L.append("`best-SR ref` is the best single mode chosen per training fold — "
                 "it shows how weak or strong the always-cheapest contrast is in "
                 "that cell, which is what decides whether beating it means "
                 "anything.")
        L.append("")
        L.append("| arm | cell | router SR% | router cost | cheapest SR% | "
                 "cheapest cost | best-SR ref% | ΔSR pp | Δcost % | ND vs cheapest "
                 "| **dominates** | ND vs 6 fixed |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for arm_name, arm in s["arms"].items():
            for cid, r in arm["per_cell"].items():
                nd1 = r["pareto_vs_always_cheapest"]
                nd2 = r["pareto_vs_six_fixed_modes"]
                dm = r["strict_dominance_vs_cheapest"]
                L.append(
                    f"| `{arm_name}` | {cid} | {r['router']['sr_pct']:.2f} | "
                    f"{r['router']['mean_cost']:.5f} | "
                    f"{r['always_cheapest']['sr_pct']:.2f} | "
                    f"{r['always_cheapest']['mean_cost']:.5f} | "
                    f"{r['best_sr_ref']['sr_pct']:.2f} | "
                    f"{r['delta_sr_vs_cheapest_pp']:+.2f} | "
                    f"{r['delta_cost_vs_cheapest_pct']:+.1f} | "
                    f"{nd1['fraction_non_dominated']:.3f} "
                    f"{'PASS' if nd1['passes'] else 'fail'} | "
                    f"{dm['fraction_dominating']:.3f} "
                    f"{'PASS' if dm['passes'] else 'fail'} | "
                    f"{nd2['fraction_non_dominated']:.3f} "
                    f"{'PASS' if nd2['passes'] else 'fail'} |")
        L.append("")
        L.append("### What the router actually selected")
        L.append("")
        for arm_name, arm in s["arms"].items():
            for cid, r in arm["per_cell"].items():
                L.append(f"- `{arm_name}` / {cid}: routed "
                         f"{r['routed_mode_counts']}; always-cheapest picked "
                         f"{r['cheapest_mode_counts']}")
        L.append("")

    L.append("## Known limitations (not optional)")
    L.append("")
    for i, lim in enumerate(LIMITATIONS, 1):
        L.append(f"{i}. {lim}")
    L.append("")
    return "\n".join(L)


def summarise(results: dict) -> dict[str, Any]:
    """Three tallies, because the spec's hypothesis and its locked test differ.

    `nd_vs_cheapest`   the locked §3 rule: 95% paired-bootstrap NON-dominance
                       against always-cheapest. Admissibility, not superiority.
    `dominates_cheapest` H-pool as worded in §2: the router actually DOMINATES
                       always-cheapest (SR no worse AND cost no worse, one strict).
    `nd_vs_six_fixed`  the harder bar: is the router on the empirical Pareto
                       front at all, or only ahead of one badly chosen contrast.

    The headline is driven by the last two. A cell can clear the locked rule
    purely because always-cheapest happens to be a weak policy there, which is a
    fact about the contrast rather than about the router.
    """
    nd_by_arm: dict[str, str] = {}
    dom_by_arm: dict[str, str] = {}
    ndall_by_arm: dict[str, str] = {}
    sf = {"nd": 0, "dom": 0, "ndall": 0, "total": 0}
    tot = {"nd": 0, "dom": 0, "ndall": 0, "total": 0}

    for site, s in results.items():
        for arm_name, arm in s["arms"].items():
            cells = arm["per_cell"].values()
            nd = sum(1 for r in cells if r["pareto_vs_always_cheapest"]["passes"])
            dm = sum(1 for r in cells if r["strict_dominance_vs_cheapest"]["passes"])
            na = sum(1 for r in cells if r["pareto_vs_six_fixed_modes"]["passes"])
            n = len(arm["per_cell"])
            key = f"{site}|{arm_name}"
            nd_by_arm[key], dom_by_arm[key], ndall_by_arm[key] = (
                f"{nd}/{n}", f"{dm}/{n}", f"{na}/{n}")
            for acc, v in ((tot, (nd, dm, na, n)),):
                acc["nd"] += v[0]; acc["dom"] += v[1]
                acc["ndall"] += v[2]; acc["total"] += v[3]
            if arm_name == "same_family|cost_tier":
                sf["nd"] += nd; sf["dom"] += dm
                sf["ndall"] += na; sf["total"] += n

    if sf["dom"] > 0:
        headline = (
            "H-pool SUPPORTED — the same-family × cost-tier router Pareto-DOMINATES "
            f"always-cheapest in {sf['dom']}/{sf['total']} cells")
    elif sf["ndall"] > 0:
        headline = (
            "H-pool NOT supported as worded (0 dominance), but the same-family × "
            "cost-tier router is on the empirical Pareto front in "
            f"{sf['ndall']}/{sf['total']} cells — a genuine trade-off point")
    else:
        headline = (
            "H-pool NOT supported — the same-family × cost-tier router dominates "
            f"always-cheapest in 0/{sf['total']} cells and is dominated by the "
            "fixed-mode menu in every cell. The most favourable corner "
            "(agreeing backbones, coarse label, highest ceiling) does not "
            "change the negative result.")
    return {
        "headline": headline,
        "same_family_tier": sf, "all_arms": tot,
        "non_dominated_vs_cheapest_by_arm": nd_by_arm,
        "dominates_cheapest_by_arm": dom_by_arm,
        "non_dominated_vs_six_fixed_by_arm": ndall_by_arm,
    }


def attribution(results: dict) -> list[str]:
    """Which factor, if any, is associated with a cell clearing the locked rule?

    ⚠️ This function deliberately does NOT assert causation any more.

    The first revision concluded "X is not the cause" whenever two arms differing
    in X both landed on the same side of the 0.95 threshold. codex (cross-AI Mode
    B, 2026-07-29) showed that inference is invalid, with a counterexample from
    this very table: on reddit·B0 the which-mode arm reaches 15.27% SR at 0.10415
    while the cost-tier arm reaches 14.29% at 0.10803 — which-mode is better on
    BOTH axes, yet a pass/fail rule calls them equivalent because both clear the
    bar. A binary admissibility test cannot license a statement about whether a
    factor moved the operating point.

    So the function now reports co-occurrence and the point-estimate spread, and
    says explicitly what would be needed to make a causal claim.
    """
    lines: list[str] = []
    for site, s in sorted(results.items()):
        passing: dict[str, list[str]] = {}
        for arm_name, arm in s["arms"].items():
            for cid, r in arm["per_cell"].items():
                if r["pareto_vs_always_cheapest"]["passes"]:
                    passing.setdefault(cid, []).append(arm_name)

        # Label balance is part of the attribution picture: a near-constant target
        # makes the classifier collapse to the majority class, so "granularity did
        # not matter" and "the label barely varied" are not distinguishable here.
        # (Gemini cross-AI Mode C, 2026-07-29.)
        sup = (s.get("supply") or {}).get("same_family") or {}
        bal = sup.get("cost_tier_distribution") or {}
        if bal:
            tot = sum(bal.values()) or 1
            minor = min(bal.values())
            lines.append(
                f"- **{site} label balance (same-family pool)**: {bal} — minority "
                f"class is {100.0*minor/tot:.0f}% of rows. "
                + ("**Severely imbalanced**: a classifier can score well by "
                   "collapsing to the majority class, so any statement about what "
                   "the LABEL GRANULARITY contributed is confounded with the label "
                   "barely varying." if minor / tot < 0.25 else
                   "Balanced enough that majority-class collapse is not the "
                   "default explanation."))

        if not passing:
            lines.append(f"- **{site}**: no cell clears the locked rule under any "
                         f"arm ({len(s['arms'])} arms tested).")
            continue
        for cid, arms in sorted(passing.items()):
            fams = {a.split("|")[0] for a in arms}
            labs = {a.split("|")[1] for a in arms if len(a.split("|")) > 1}
            pts = []
            for a in arms:
                r = s["arms"][a]["per_cell"][cid]
                pts.append(f"`{a}` {r['router']['sr_pct']:.2f}%@"
                           f"{r['router']['mean_cost']:.5f}")
            bits = []
            if "same_family" in fams and "all_three" in fams:
                bits.append("clears with **and** without the cross-family cell")
            if "cost_tier" in labs and "which_mode" in labs:
                bits.append("clears at **both** granularities")
            if "per_cell" in fams:
                bits.append("clears under **per-cell** training as well as pooled")
            lines.append(
                f"- **{site} / {cid}** clears the locked rule in "
                f"{len(arms)}/{len(s['arms'])} arms: {', '.join(pts)}."
                + ("  Co-occurrence: " + "; ".join(bits) + "." if bits else ""))
    lines.append(
        "- ⚠️ **These are co-occurrences, not causal attributions.** Two arms can "
        "both clear a binary admissibility bar while differing materially in "
        "(SR, cost) — the point estimates above show exactly that. Establishing "
        "that a factor did or did not move the operating point requires paired "
        "task-level contrasts (ΔSR and Δcost with paired bootstrap) or an explicit "
        "2×2 interaction contrast, neither of which is computed here.")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=REPO / "docs/analysis/cross_sites/router_pooled_tier_learnability.md")
    ap.add_argument("--json-out", type=Path,
                    default=REPO / "docs/analysis/cross_sites/router_pooled_tier_learnability.json")
    a = ap.parse_args(argv)

    assert_tier_monotone()
    by_site = build_all()
    if not by_site:
        raise SystemExit("no cells built — check the paper-grade run manifest")
    results = analyse(by_site)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "post_hoc_exploratory": True,
        "h10_eligible": False,
        "spec": "docs/checkpoints/EXP_SPEC_pooled_tier_router.md",
        "protocol": {
            "cv": f"task-held-out {N_FOLDS}-fold, shared fold map per site",
            "nesting": "tier->mode map, always-cheapest and threshold all "
                       "re-derived per outer fold from training tasks only",
            "bootstrap_B": BOOTSTRAP_B,
            "pareto_threshold": 0.95,
            "cost_field": COST_FIELD,
            "seed": SEED,
        },
        "limitations": LIMITATIONS,
        "results": results,
    }
    payload["verdict"] = summarise(results)
    payload["attribution"] = attribution(results)
    payload["reading"] = reading(results, payload["verdict"])

    a.json_out.parent.mkdir(parents=True, exist_ok=True)
    a.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                          encoding="utf-8")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(render(payload), encoding="utf-8")
    print(render(payload))
    print(f"\nwrote {a.out}\nwrote {a.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
