#!/usr/bin/env python3
"""Is the TRIAGE half of routing learnable? — 2026-07-27, post_hoc_exploratory

`router_objective_ordering.py` split the oracle's advantage into two halves that
need very different labels:

    triage   send tasks NO mode will solve to the cheapest mode.
             -38% to -45% cost at ZERO SR change, 6/6 cells.
             Label = binary "is this solvable by anything", defined for EVERY
             task in the cell (203 / 224 labels).
    route    choose among the modes that do solve it.
             +3.45 to +16.07pp SR, but only -0.2% to -11.4% cost.
             Label = which-mode, defined only on solved tasks (16-97 per cell),
             and 笔记 §383.4 established it is not learnable at that supply.

The unlearnable half is the one carrying the SR gain. The half carrying almost
all the cost gain has 2-13x the label supply and has never been tested. This
script tests it.

Two things decide whether the answer means anything:

  * A trivial baseline is mandatory. 笔记 §367 found the 18-feature LR matched a
    3-covariate scalar on the which-mode task (ΔAUROC -0.013/+0.007, CI spanning
    0) — "learnable" there meant "learnable by anything". Every AUROC below is
    reported next to a prior-only baseline and the best single raw feature.
  * AUROC is not the deliverable. The deliverable is what the policy DOES: send
    predicted-hopeless tasks to the cheapest mode and measure realized SR and
    cost against the best-single-mode baseline. A triage that saves 40% while
    dropping 3pp of SR is not a win; the threshold sweep below finds the
    SR-preserving operating point, if one exists.

Protocol: task-held-out 5-fold CV within each fixed cell (mirrors the registered
H10 split, prereg §4 "Router train/test split"), seed 42, per-cell L2 logistic
regression on the 20 raw features (5 numeric + 15 binary), fold-local
standardisation. TF-IDF text features are deliberately omitted — §367 showed
they add nothing over covariates here, and leaving them out keeps the fold-local
vocabulary leak surface at zero.

post_hoc_exploratory=True, h10_eligible=False. Touches no gating producer.

Usage:
  .venv/bin/python3 scripts/analysis/router_triage_learnability.py \
      --out docs/analysis/cross_sites/router_triage_learnability.md \
      --json-out results/phantom_paper/router_triage_learnability.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from p79.policies.router_features import INTENT_REGEX, compute_intent_binaries  # noqa: E402
from scripts.analysis.aggregate_phantom_lift import CELLS  # noqa: E402
from scripts.analysis.extract_50_features import (  # noqa: E402
    find_pass1_runs,
    read_step0_features,
    read_task_config,
)
from scripts.analysis.lib.canonical_task_universe import expected_scored_ids  # noqa: E402
from scripts.analysis.lib.episode_rows import load_cell_task_rows  # noqa: E402

SIX_MODES = ("DOM", "SoM", "Vision", "P-text", "P-prompt", "P-SoM")
COST_FIELD = "total_billed_cost_usd"
NUMERIC = ["dom_complexity", "text_length", "tokens_input_text",
           "intent_token_count", "reasoning_difficulty"]
BINARY = ["has_reference_image"] + sorted(INTENT_REGEX.keys())
SEED = 42
N_FOLDS = 5

# Permutation count for the label-shuffle null.
#
# Raised 200 -> 10000 on 2026-07-28 (stress finding #11). The plus-one estimator
# (k+1)/(B+1) floors at 1/(B+1), so B=200 could not report anything below 0.004975
# — and red·B2, the one cell that survives Holm, sat exactly there with k=0. Its
# p was therefore not a measurement of how extreme the saving is, it was "zero of
# 200". Worse, whether that cell could clear its Holm threshold (0.05/6 = 0.008333)
# was decided by B rather than by the data: at B=100 the floor is 0.009901 and no
# amount of signal could have passed. B=10000 floors at 9.999e-5, two orders below
# the threshold, so the verdict is data-determined across any plausible B.
#
# Cost: ~40 s at B=200, ~30 min at B=10000 (6 cells, 5-fold LR refit per draw).
N_SHUFFLE = 10000


def _feature_row(runs, site: str, tid: int) -> tuple[list[float], list[int]] | None:
    cfg = read_task_config(site, tid)
    if cfg is None:
        return None
    step0 = read_step0_features(runs, site, tid)
    if step0 is None:
        return None
    num = [float(step0.get(k, 0.0) or 0.0) for k in NUMERIC]
    bins = compute_intent_binaries(str(cfg.get("intent", "")))
    binv = [int(bool(cfg.get("image")))] + [int(bins[k]) for k in sorted(INTENT_REGEX.keys())]
    return num, binv


def build_cell(cell: dict) -> dict | None:
    site, baseline = cell["site"], cell["baseline"]
    universe, _ = expected_scored_ids(site)
    rows_by_mode = load_cell_task_rows(cell, modes=SIX_MODES)
    if any(not rows_by_mode.get(m) for m in SIX_MODES):
        return None
    runs = find_pass1_runs(baseline, site)

    tids, X, y, succ, cost = [], [], [], [], []
    n_no_feature = 0
    for t in sorted(universe):
        rows = {m: rows_by_mode[m].get(t) for m in SIX_MODES}
        if any(r is None for r in rows.values()):
            return None
        fr = _feature_row(runs, site, t)
        if fr is None:
            n_no_feature += 1
            continue
        num, binv = fr
        s = {m: rows[m].get("success") is True for m in SIX_MODES}
        c = {m: float(rows[m].get(COST_FIELD) or 0.0) for m in SIX_MODES}
        tids.append(t)
        X.append(num + binv)
        y.append(int(any(s.values())))          # triage label: solvable by anything
        succ.append(s)
        cost.append(c)
    if len(tids) < 50:
        return None
    return {
        "site": site, "baseline": baseline, "task_ids": tids,
        "X": np.asarray(X, dtype=float), "y": np.asarray(y, dtype=int),
        "succ": succ, "cost": cost, "n_no_feature": n_no_feature,
    }


def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # rank-based (handles ties correctly)
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    vals = np.concatenate([pos, neg])
    for v in np.unique(vals):
        m = vals == v
        ranks[m] = ranks[m].mean()
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def oof_scores(X: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Out-of-fold scores for the LR and for the trivial baselines."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, N_FOLDS)
    oof_lr = np.full(len(y), np.nan)
    oof_prior = np.full(len(y), np.nan)
    best_feat_oof = {j: np.full(len(y), np.nan) for j in range(X.shape[1])}

    for f in folds:
        tr = np.setdiff1d(idx, f)
        if len(np.unique(y[tr])) < 2:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1.0
        Xtr, Xte = (X[tr] - mu) / sd, (X[f] - mu) / sd
        lr = LogisticRegression(max_iter=2000, C=1.0)
        lr.fit(Xtr, y[tr])
        oof_lr[f] = lr.predict_proba(Xte)[:, 1]
        oof_prior[f] = y[tr].mean()               # constant = train-fold base rate
        for j in range(X.shape[1]):
            # single-feature score, oriented by the train fold so it cannot peek
            sign = 1.0 if np.corrcoef(Xtr[:, j], y[tr])[0, 1] >= 0 else -1.0
            best_feat_oof[j][f] = sign * Xte[:, j]
    return {"lr": oof_lr, "prior": oof_prior, "per_feature": best_feat_oof}


def policy_at_threshold(cell: dict, score: np.ndarray, thr: float,
                        best_mode: str, cheap_mode: str) -> dict:
    """Predicted-hopeless → cheapest mode; otherwise → best-SR mode."""
    n = len(cell["y"])
    hits, spend = 0, 0.0
    n_sent_cheap = 0
    for i in range(n):
        use = cheap_mode if score[i] < thr else best_mode
        if use == cheap_mode:
            n_sent_cheap += 1
        hits += cell["succ"][i][use]
        spend += cell["cost"][i][use]
    return {"sr_pct": 100.0 * hits / n, "mean_cost": spend / n,
            "n_sent_cheap": n_sent_cheap, "threshold": float(thr)}


def evaluate(cell_spec: dict) -> dict | None:
    cell = build_cell(cell_spec)
    if cell is None:
        return None
    X, y = cell["X"], cell["y"]
    n = len(y)

    per_mode_sr = {m: 100.0 * sum(s[m] for s in cell["succ"]) / n for m in SIX_MODES}
    per_mode_cost = {m: sum(c[m] for c in cell["cost"]) / n for m in SIX_MODES}
    best_mode = max(SIX_MODES, key=lambda m: per_mode_sr[m])
    cheap_mode = min(SIX_MODES, key=lambda m: per_mode_cost[m])

    sc = oof_scores(X, y)
    auroc_lr = _auroc(y, sc["lr"])
    feat_names = NUMERIC + BINARY
    per_feat = {feat_names[j]: _auroc(y, v) for j, v in sc["per_feature"].items()}
    best_feat = max(per_feat, key=lambda k: (per_feat[k] if per_feat[k] == per_feat[k] else -1))

    # Oracle triage = perfect knowledge of y.
    oracle = policy_at_threshold(cell, y.astype(float), 0.5, best_mode, cheap_mode)
    baseline = {"sr_pct": per_mode_sr[best_mode], "mean_cost": per_mode_cost[best_mode]}
    # The baseline that actually matters. Comparing a triage policy only against
    # best-single flatters it: "send everything to the cheapest mode" is a fixed
    # policy costing nothing to implement, and in a cell where the cheapest mode
    # already ties on SR it wins outright. A learned triage has to beat THIS.
    always_cheap = {"sr_pct": per_mode_sr[cheap_mode], "mean_cost": per_mode_cost[cheap_mode]}

    # ---- Honest operating point: FULLY NESTED cross-validation.
    #
    # The sweep further down picks its threshold by looking at realized outcomes
    # on the WHOLE cell, so its "SR-lossless" point is in-sample with respect to
    # the threshold even though the scores are out-of-fold. Claude, codex and
    # gemini each flagged that independently (2026-07-27). It is retained because
    # the permutation null shares the same selection step — so those p-values
    # compare like with like — but it is NOT an achievable operating point.
    #
    # B-1903 (fixed 2026-07-27): the first attempt at an honest point, added the
    # same day, was only HALF nested and codex caught it. Three leaks:
    #   (1) it scored held-out rows with `sc["lr"]`, the GLOBAL out-of-fold score
    #       — whose fold split is a different random permutation, so the model
    #       behind a training row's score was fitted on data including the
    #       current outer test rows;
    #   (2) `best_mode` / `cheap_mode` came from whole-cell SR/cost (L195-196),
    #       i.e. chosen with knowledge of the outcomes being predicted;
    #   (3) `base_tr_sr`, the SR floor the threshold must preserve, was computed
    #       against that whole-cell-selected `best_mode`.
    # Evidence it mattered: even half-nested, the SR delta was already
    # -1.786…-0.893, 0.0 pp — the "lossless" reading was a selection artifact.
    #
    # Properly nested, per outer fold, using ONLY that fold's training rows:
    #   a. re-select best_mode / cheap_mode from training-row SR and cost;
    #   b. produce inner-CV out-of-fold scores over the training rows, and pick
    #      the threshold against those (never against outer-test rows);
    #   c. refit the LR on all training rows and score the outer-test rows with
    #      that model alone;
    #   d. apply (threshold, modes) blind to the outer-test rows.
    # Nothing crossing into an outer test fold has seen it.
    from sklearn.linear_model import LogisticRegression

    def _fit_score(Xtr_raw, ytr, Xte_raw):
        """Standardise on train, fit, return P(y=1) for the test rows."""
        if len(np.unique(ytr)) < 2:
            return None
        mu, sd = Xtr_raw.mean(0), Xtr_raw.std(0)
        sd = np.where(sd == 0, 1.0, sd)
        lr = LogisticRegression(max_iter=2000, C=1.0)
        lr.fit((Xtr_raw - mu) / sd, ytr)
        return lr.predict_proba((Xte_raw - mu) / sd)[:, 1]

    nested_hits = nested_spend = 0.0
    nested_sent_cheap = 0
    nested_folds: list[dict] = []
    rng_n = np.random.default_rng(SEED)
    idx_n = rng_n.permutation(n)
    outer_folds = np.array_split(idx_n, N_FOLDS)

    for f in outer_folds:
        tr = np.setdiff1d(idx_n, f)

        # (a) modes re-selected on training rows only.
        tr_sr = {m: sum(cell["succ"][i][m] for i in tr) / len(tr) for m in SIX_MODES}
        tr_cost = {m: sum(cell["cost"][i][m] for i in tr) / len(tr) for m in SIX_MODES}
        best_o = max(SIX_MODES, key=lambda m: tr_sr[m])
        cheap_o = min(SIX_MODES, key=lambda m: tr_cost[m])

        # (b) inner-CV OOF scores over the training rows.
        inner_oof = np.full(len(tr), np.nan)
        rng_i = np.random.default_rng(SEED + 1)
        tr_shuf = rng_i.permutation(len(tr))
        for g in np.array_split(tr_shuf, N_FOLDS):
            in_tr = np.setdiff1d(tr_shuf, g)
            p = _fit_score(X[tr][in_tr], y[tr][in_tr], X[tr][g])
            if p is not None:
                inner_oof[g] = p

        # (c) threshold chosen against inner OOF + training outcomes only.
        thr_star = -np.inf
        if not np.all(np.isnan(inner_oof)):
            base_tr_sr = 100.0 * tr_sr[best_o]
            cands = np.quantile(inner_oof[~np.isnan(inner_oof)], np.linspace(0.0, 0.95, 20))
            best_c = None
            for thr in cands:
                h = c = 0.0
                for pos, i in enumerate(tr):
                    if np.isnan(inner_oof[pos]):
                        use = best_o
                    else:
                        use = cheap_o if inner_oof[pos] < thr else best_o
                    h += cell["succ"][i][use]
                    c += cell["cost"][i][use]
                if 100.0 * h / len(tr) >= base_tr_sr - 1e-9 and (best_c is None or c < best_c):
                    thr_star, best_c = float(thr), c

        # (d) score the outer-test rows with a model fitted on training rows only.
        score_te = _fit_score(X[tr], y[tr], X[f])
        fold_cheap = 0
        for pos, i in enumerate(f):
            s = None if score_te is None else score_te[pos]
            use = best_o if s is None else (cheap_o if s < thr_star else best_o)
            fold_cheap += use == cheap_o
            nested_sent_cheap += use == cheap_o
            nested_hits += cell["succ"][i][use]
            nested_spend += cell["cost"][i][use]
        nested_folds.append({
            "n_test": int(len(f)), "best_mode": best_o, "cheap_mode": cheap_o,
            "threshold": None if thr_star == -np.inf else round(float(thr_star), 6),
            "n_sent_cheap": int(fold_cheap),
        })

    policies_nested = {
        "sr_pct": 100.0 * nested_hits / n, "mean_cost": nested_spend / n,
        "n_sent_cheap": int(nested_sent_cheap),
        "per_outer_fold": nested_folds,
        "note": (
            "FULLY nested (B-1903): per outer fold the modes are re-selected on "
            "training rows, the threshold is chosen on inner-CV out-of-fold "
            "scores over training rows, and the outer-test rows are scored by an "
            "LR fitted on training rows only. Nothing that touches an outer test "
            "fold has seen it. This is the only achievable operating point here; "
            "the whole-cell sweep below is selection-contaminated by construction."
        ),
    }

    # Threshold sweep on the OOF score; report the operating point that keeps SR
    # whole (the only kind of saving that is free) and the best-cost point within
    # a 1pp SR give-back.
    sweep = []
    for thr in np.quantile(sc["lr"][~np.isnan(sc["lr"])], np.linspace(0.0, 0.95, 20)):
        sweep.append(policy_at_threshold(cell, sc["lr"], float(thr), best_mode, cheap_mode))
    lossless = [p for p in sweep if p["sr_pct"] >= baseline["sr_pct"] - 1e-9]
    within_1pp = [p for p in sweep if p["sr_pct"] >= baseline["sr_pct"] - 1.0]
    best_lossless = min(lossless, key=lambda p: p["mean_cost"]) if lossless else None
    best_1pp = min(within_1pp, key=lambda p: p["mean_cost"]) if within_1pp else None

    # Label-shuffle null. Re-run the whole CV on permuted labels: any saving the
    # pipeline reports on noise is saving the threshold sweep manufactured by
    # picking the best point post hoc, not saving the classifier earned. Reported
    # as the fraction of shuffles matching or beating the observed lossless saving.
    # B-1902: permute the whole task BUNDLE, not just `y`.
    #
    # The first version permuted `y` and then evaluated the resulting scores
    # against the ORIGINAL per-task `succ` / `cost` dicts. A permuted "solvable"
    # label was therefore disconnected from the outcomes that define solvability
    # and policy cost, so the null was not a permutation null for the policy being
    # tested. Permuting the (y, succ, cost) triple against X breaks the
    # feature->outcome link while keeping each task's outcome bundle internally
    # consistent — which is the actual null of interest.
    #
    # The direction of the error is NOT uniform, which is why it mattered:
    # measured cls/B1 0.4776 -> 0.5025 (was anti-conservative) but
    # red/B2 0.0398 -> 0.0050 (was CONSERVATIVE by 8x). Under the corrected null
    # red/B2 crosses Holm at m=6, reversing an earlier "nothing survives" claim.
    #
    # Also switched to the plus-one Monte Carlo p-value (k+1)/(B+1): k/B can
    # report 0 for an event that simply was not sampled in B draws, and is
    # unambiguously anti-conservative at small k.
    rng_null = np.random.default_rng(SEED + 1)
    null_savings = []
    for _ in range(N_SHUFFLE):
        perm = rng_null.permutation(n)
        y_b = y[perm]
        succ_b = [cell["succ"][i] for i in perm]
        cost_b = [cell["cost"][i] for i in perm]
        cell_b = dict(cell, y=y_b, succ=succ_b, cost=cost_b)
        try:
            sp = oof_scores(X, y_b)["lr"]
        except Exception:
            continue
        sw = [policy_at_threshold(cell_b, sp, float(t), best_mode, cheap_mode)
              for t in np.quantile(sp[~np.isnan(sp)], np.linspace(0.0, 0.95, 20))]
        ll = [q for q in sw if q["sr_pct"] >= baseline["sr_pct"] - 1e-9]
        null_savings.append(
            100.0 * (1 - min(q["mean_cost"] for q in ll) / baseline["mean_cost"]) if ll else 0.0
        )
    # Draws that raise are skipped above, so a silent exception would shrink B for
    # one cell only and the plus-one floor reported for it would be wrong (codex Mode
    # B finding 7). Fail loud rather than report a p against a denominator nobody sees.
    if len(null_savings) != N_SHUFFLE:
        raise RuntimeError(
            f"{cell['site']}/{cell['baseline']}: {len(null_savings)} of {N_SHUFFLE} "
            "permutation draws completed. A short draw count changes this cell's "
            "plus-one floor, so the reported p is not comparable across cells."
        )
    observed_saving = (100.0 * (1 - best_lossless["mean_cost"] / baseline["mean_cost"])
                       if best_lossless else 0.0)
    n_exceed = int(sum(1 for v in null_savings if v >= observed_saving - 1e-12))
    p_null = ((n_exceed + 1) / (len(null_savings) + 1)) if null_savings else float("nan")

    return {
        "site": cell["site"], "baseline_model": cell["baseline"], "n": n,
        "n_no_feature": cell["n_no_feature"],
        "solvable_rate_pct": 100.0 * y.mean(),
        "best_mode": best_mode, "cheap_mode": cheap_mode,
        "baseline_policy": baseline,
        "auroc_lr": auroc_lr,
        "auroc_prior": 0.5,
        "auroc_best_single_feature": per_feat[best_feat],
        "best_single_feature": best_feat,
        "auroc_lr_minus_best_feature": auroc_lr - per_feat[best_feat],
        "always_cheapest": always_cheap,
        "oracle_triage": oracle,
        "learned_lossless": best_lossless,
        "learned_within_1pp": best_1pp,
        "learned_nested_honest": policies_nested,
        "observed_lossless_saving_pct": observed_saving,
        "null_shuffle_saving_median_pct": (float(np.median(null_savings))
                                           if null_savings else float("nan")),
        "null_shuffle_p": p_null,
        "null_p_estimator": "(k+1)/(B+1) plus-one Monte Carlo",
        "null_permutation_unit": "task bundle (y, succ_by_mode, cost_by_mode) vs X",
        "n_shuffles": len(null_savings),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    global N_SHUFFLE
    ap.add_argument("--out", type=Path)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--n-shuffle", type=int, default=N_SHUFFLE,
                    help="permutation draws for the label-shuffle null; the plus-one "
                         "estimator floors at 1/(B+1), so B bounds the smallest reportable p")
    args = ap.parse_args()
    N_SHUFFLE = args.n_shuffle

    res = [r for r in (evaluate(c) for c in CELLS) if r]
    L = ["# Is the triage half of routing learnable?\n",
         "`post_hoc_exploratory=True`, `h10_eligible=False`. Task-held-out 5-fold CV "
         "per cell, seed 42, L2 LR on the 20 raw features, fold-local standardisation.\n",
         "Triage policy: predicted-hopeless → cheapest mode, otherwise → best-SR mode. "
         "The oracle row knows the true label; the learned rows use out-of-fold scores.\n"]

    L.append("\n## 1. Can the label be predicted at all?\n")
    L.append("| cell | n | solvable % | AUROC LR | AUROC best single feature | Δ | that feature |")
    L.append("|---|---|---|---|---|---|---|")
    for r in res:
        L.append(
            f"| {r['site']}·{r['baseline_model']} | {r['n']} | {r['solvable_rate_pct']:.1f} | "
            f"**{r['auroc_lr']:.3f}** | {r['auroc_best_single_feature']:.3f} | "
            f"{r['auroc_lr_minus_best_feature']:+.3f} | `{r['best_single_feature']}` |"
        )
    L.append("\nA prior-only predictor scores 0.500 by construction. The single-feature "
             "column is the §367 check: if the LR does not clear it, 'learnable' means "
             "'learnable by one covariate', which is not a router.\n")

    L.append("\n## 2. What does the policy actually buy?\n")
    L.append("| cell | policy | SR % | mean cost | ΔSR | Δcost | sent to cheapest |")
    L.append("|---|---|---|---|---|---|---|")
    for r in res:
        b = r["baseline_policy"]
        L.append(f"| {r['site']}·{r['baseline_model']} | best-single (`{r['best_mode']}`) | "
                 f"{b['sr_pct']:.2f} | {b['mean_cost']:.5f} | — | — | — |")
        ac = r["always_cheapest"]
        L.append(f"| | **always-cheapest (`{r['cheap_mode']}`)** — the fixed policy a "
                 f"router must beat | {ac['sr_pct']:.2f} | {ac['mean_cost']:.5f} | "
                 f"{ac['sr_pct']-b['sr_pct']:+.2f}pp | "
                 f"{100*(ac['mean_cost']/b['mean_cost']-1):+.1f}% | {r['n']}/{r['n']} |")
        o = r["oracle_triage"]
        L.append(f"| | oracle triage | {o['sr_pct']:.2f} | {o['mean_cost']:.5f} | "
                 f"{o['sr_pct']-b['sr_pct']:+.2f}pp | "
                 f"{100*(o['mean_cost']/b['mean_cost']-1):+.1f}% | {o['n_sent_cheap']}/{r['n']} |")
        for key, label in (("learned_nested_honest", "**learned, nested threshold (honest)**"),
                           ("learned_lossless", "learned, SR-lossless (in-sample threshold)"),
                           ("learned_within_1pp", "learned, ≤1pp give-back (in-sample threshold)")):
            p = r[key]
            if p is None:
                L.append(f"| | {label} | — | — | none exists | — | — |")
                continue
            L.append(f"| | {label} | {p['sr_pct']:.2f} | {p['mean_cost']:.5f} | "
                     f"{p['sr_pct']-b['sr_pct']:+.2f}pp | "
                     f"{100*(p['mean_cost']/b['mean_cost']-1):+.1f}% | "
                     f"{p['n_sent_cheap']}/{r['n']} |")

    L.append("\n## 3. Is the saving real, or manufactured by the threshold sweep?\n")
    L.append("| cell | observed SR-lossless saving | median under shuffled labels | p |")
    L.append("|---|---|---|---|")
    # `.3f` rendered the B=10000 floor (4.9995e-4) as "0.000", i.e. as if the
    # plus-one estimator had reported an impossible zero — the exact thing the
    # estimator exists to prevent (codex Mode B finding 7, 2026-07-28).
    def _fmt_p(p: float) -> str:
        return f"{p:.4g}" if p < 1e-3 else f"{p:.3f}"

    for r in res:
        L.append(f"| {r['site']}·{r['baseline_model']} | "
                 f"{r['observed_lossless_saving_pct']:.1f}% | "
                 f"{r['null_shuffle_saving_median_pct']:.1f}% | "
                 f"{_fmt_p(r['null_shuffle_p'])} |")
    _b = res[0]["n_shuffles"]
    L.append(f"\nSmallest reportable p at B={_b} is 1/(B+1) = {1.0 / (_b + 1):.2e}; "
             "Holm's tightest threshold over six cells is 0.05/6 = 8.33e-3. B is therefore "
             "not what decides any cell's verdict (it was at B=200, where the floor 4.98e-3 "
             "sat inside the threshold and the surviving cell reported exactly it).\n")
    L.append(f"\n{res[0]['n_shuffles']} permutations per cell. The permutation unit is the whole "
             "task bundle (y, succ, cost) against X — permuting only `y` leaves the label "
             "disconnected from the outcomes that define it, and its error is not "
             "one-directional (measured at B=200: cls/B1 0.478→0.503 but red/B2 "
             "0.040→0.005; both figures are from that era, not from the current B). "
             "p is the plus-one Monte Carlo estimator (k+1)/(B+1). The sweep still picks its "
             "operating point post hoc, so this column is how much of the observed saving a "
             "signal-free pipeline reproduces.\n")

    L.append("\n## 4. Verdict\n")
    ps = sorted((r["null_shuffle_p"], f"{r['site']}·{r['baseline_model']}") for r in res)
    m = len(ps)
    holm = []
    for i, (pv, name) in enumerate(ps):
        thresh = 0.05 / (m - i)
        holm.append((name, pv, thresh, pv < thresh))
        if pv >= thresh:
            break
    n_rej = sum(1 for _n, _p, _t, _ok in holm if _ok)
    _surv = [h for h in holm if h[3]]
    L.append(f"Holm at α=0.05 over the m=6 cells tested (the sweep was run once per cell, "
             f"so the family is the six cells) — **{n_rej} of 6 reject**:\n")
    for name, pv, thresh, ok in holm:
        # The step-down stops at the first non-rejection; saying "no cell survives"
        # there is wrong whenever an earlier step already rejected.
        verdict = "reject null" if ok else "**stop — this and all larger p unrejected**"
        L.append(f"- {name}: p={_fmt_p(pv)} vs {thresh:.4f} → {verdict}")
    beat_cheap = [r for r in res
                  if r["learned_lossless"] is not None
                  and r["learned_lossless"]["mean_cost"] < r["always_cheapest"]["mean_cost"]
                  and r["learned_lossless"]["sr_pct"] >= r["always_cheapest"]["sr_pct"]]
    L.append(f"\nCells where the learned triage Pareto-beats the trivial always-cheapest "
             f"fixed policy: **{len(beat_cheap)} of {len(res)}**"
             + (f" ({', '.join(f'{r[chr(39)+chr(39)] if False else r['site']}·{r['baseline_model']}' for r in beat_cheap)})" if beat_cheap else "")
             + ".\n")
    L.append("Read together — and note this is a **narrower** negative than an earlier "
             "draft of this file claimed. In five of six cells the label is predictable "
             "(AUROC 0.651-0.717, and unlike the which-mode task it clears the best single "
             "covariate in 4/6). Two cells yield no SR-lossless saving at all; two more "
             "yield savings a signal-free pipeline reproduces (p ~= 0.50). "
             # Generated, not hardcoded: this sentence carried `p=0.005` from the
             # B=200 era for one commit after B rose to 10000 (codex finding 7).
             + (f"**{'One cell' if len(_surv) == 1 else str(len(_surv)) + ' cells'}, "
                + ", ".join(f"{n} (p={_fmt_p(p)} vs {t:.4f})" for n, p, t, _o in _surv)
                + f", {'has' if len(_surv) == 1 else 'have'} a saving that survives Holm "
                "at m=6** — "
                if _surv else "**No cell's saving survives Holm at m=6** — ")
             + "under the corrected bundle-permutation null; the earlier y-only null reported "
             "0.040 for reddit/B2 and supported a blanket 'nothing survives' claim, which "
             "was wrong.\n")
    L.append("⚠️ **The sixth cell is the significant one, and its AUROC is 0.483** — below "
             "chance, and below its own best single covariate (0.711). That is not a "
             "contradiction: the two quantities measure different things, and on this data "
             "they come apart. AUROC scores the GLOBAL ranking; the saving comes from the "
             "TAIL. reddit/B2 sends 192 of 203 tasks (95%) to the cheap mode with no SR "
             "loss — in a cell where only 7.4% of tasks are solvable at all, almost nothing "
             "in that 95% was ever going to succeed. It differs from the free "
             "always-cheapest policy by five percent of the task allocation, and those 11 "
             "retained tasks happen to hold 4 successes (8 vs 4). The permutation null is "
             "detecting that tail enrichment, not a globally ordered score.\n"
             "So the honest phrasing is NOT 'the label is predictable, yet triage fails'. "
             "It is: **at 2-27% base SR, a high AUROC is neither necessary nor sufficient — "
             "what decides whether triage saves anything is whether a handful of tail tasks "
             "land on the right side, and at n=203 that handful is 4 successes.**\n")
    L.append("What still holds, and is the load-bearing statement: **no cell's learned "
             "triage Pareto-beats the trivial always-cheapest fixed policy** (0 of 6). In "
             "reddit/B2 specifically the learned policy keeps 1.97pp more SR than "
             "always-cheapest but pays ~2.4% more cost — a genuine trade-off point, not a "
             "dominating one, and not something a deployment would prefer without a stated "
             "SR price. So: a detectable signal in one of six cells, worth less than the "
             "policy you get for free.\n")
    L.append("⚠️ What is and is not out-of-sample. The **nested** row is now FULLY nested "
             "(B-1903, 2026-07-27): per outer fold the modes are re-selected from training-row "
             "SR/cost, the threshold is chosen against inner-CV out-of-fold scores over the "
             "training rows only, and the outer-test rows are scored by an LR fitted on "
             "training rows alone — nothing that touches an outer test fold has seen it. "
             "An earlier revision of this file claimed a nested operating point while "
             "reusing the GLOBAL out-of-fold scores (whose folds include the outer test rows) "
             "and a whole-cell choice of `best_mode`/`cheap_mode`; codex caught that, and the "
             "numbers here supersede it. The remaining caveat is that the threshold **sweep** "
             "rows are still post hoc by construction, which is why the permutation null "
             "shares that same selection step — the null is what keeps the swept saving "
             "honest, and the nested row is what an actual deployment would get.\n")
    L.append("One thing the nested design exposes that the whole-cell version hid: "
             "`best_mode` is **not stable across folds**. In reddit·B0 the five outer folds "
             "select DOM, DOM, SoM, SoM, DOM. A pipeline that picks one best mode from all "
             "realized outcomes is therefore not merely optimistic about the threshold — it "
             "is reporting a mode choice that its own resampling does not reproduce.\n")
    L.append("Contrast with the which-mode half: that one fails on label SUPPLY (16-97 "
             "labels per cell, 笔记 §383.4). Triage has the labels and the AUROC and still "
             "does not beat a fixed policy — a different failure mode, at 2-27% base SR "
             "where almost every task is hopeless and 'always take the cheap one' is already "
             "close to optimal.\n")

    text = "\n".join(L) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(
            {"post_hoc_exploratory": True, "h10_eligible": False,
             "protocol": {"folds": N_FOLDS, "seed": SEED, "n_shuffle": N_SHUFFLE,
                          "min_reportable_p": 1.0 / (N_SHUFFLE + 1),
                          "holm_tightest_threshold_m6": 0.05 / 6,
                          "features": NUMERIC + BINARY, "model": "L2 logistic regression"},
             "cells": res}, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
