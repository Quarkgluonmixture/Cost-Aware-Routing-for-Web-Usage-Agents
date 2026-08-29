#!/usr/bin/env python3
"""Is the ACTION half of routing learnable, once the target matches the deployment's
actual decision? — 2026-08-29, post_hoc_exploratory

`router_triage_learnability.py` tested whether a router can predict
`y_i = 1[any mode solves task i]` and found the label technically predictable
(AUROC clears its own best single feature in most cells) but the policy that
predictor buys does not Pareto-beat always-cheapest anywhere except by
construction-neutral cost saving. That target was never the deployment's actual
question. `oracle_triage` — the thesis's stated "ceiling of the half of the
problem we can actually attempt" (`final_dissertation/tex/chapters/
ch6_endtoend.tex:26`) — only ever routes between two fixed arms per cell,
`best_mode` (globally strongest by SR) and `cheap_mode` (globally cheapest), but
assigns them BY CLASS: every task with y=1 goes to best_mode, whether or not
best_mode is the arm that actually solves that task.

A perfect-knowledge policy that assigns PER TASK instead — the "two-arm action
oracle" — reaches at least `oracle_triage`'s success in every cell, and on these
logs also spends less in 8 of 8. Only the SUCCESS half of that is guaranteed:
whenever `oracle_triage`'s class-level pick succeeds, the per-task oracle can
choose the same arm, so SR never drops. The COST half is empirical, not
necessary — see the counterexample in `evaluate()`. This is NOT a wider action
space than the thesis already commits to — same two arms `oracle_triage` already
uses — it is the retrospective optimum OVER them, which `oracle_triage` is not.

That pair is itself chosen by two independent extrema (strongest-by-SR,
cheapest-by-cost), which is what `oracle_triage` does and therefore the right
thing to hold fixed for this comparison — but it is NOT the best pair available.
`enumerate_pairs()` scores all 30 ordered pairs per cell; on 2 of 8 cells another
pair dominates the selected one on both axes. The selected pair is therefore
reported as a CONDITIONAL oracle over the triage-selected arms, never as a
two-arm ceiling.

This script asks whether the per-task pick is LEARNABLE, using the label

    z_i = 1  iff best_mode succeeds AND cheap_mode fails on task i
    z_i = 0  otherwise (both succeed, both fail, or only cheap_mode succeeds)

    Both-succeed and both-fail cases default to cheap_mode; they are NOT broken
    by realized cost, because the realized cost of a failed episode encodes how
    long it ran before giving up — outcome information a pre-action router
    cannot see. See `derive_two_arm_label`.

Unlike the six-way which-mode label (defined only on the 15-97 solved-by-someone
rows per cell — 笔记 §383.4), z is defined for EVERY task in the cell. But note
what that does and does not buy: its POSITIVE class carries only 2.2-14.4% of
rows (5-22 tasks per cell), comparable to or thinner than the six-way label's
supply. The row count is not the binding constraint; the positive-class count is,
and switching target does not relieve it.

Protocol matches `router_triage_learnability.py` exactly so the two studies are
comparable apples-to-apples: task-held-out 5-fold CV, seed 42, L2 logistic
regression on the same feature set, a FULLY NESTED honest operating point (per
outer fold: re-select best_mode/cheap_mode AND re-derive z from training rows
only, inner-CV threshold selection against training rows only, refit-and-score
outer-test rows blind — mirrors the B-1903 fix), and a bundle-permutation
label-shuffle null (B=10000, plus-one Monte Carlo p, Holm correction across
cells). Every piece that is not the label itself — `build_cell`, `build_wa_cell`,
`oof_scores`, `policy_at_threshold`, the feature set, the fold seeding — is
IMPORTED from `router_triage_learnability.py`, not reimplemented, so there is no
train/serve or protocol drift between the two studies (the exact failure mode
`p79/policies/router_features.py` was written to prevent).

post_hoc_exploratory=True, h10_eligible=False. Touches no gating producer.

Usage:
  .venv/bin/python3 scripts/analysis/two_arm_action_learnability.py \
      --out docs/analysis/cross_sites/two_arm_action_learnability.md \
      --json-out results/phantom_paper/two_arm_action_learnability.json \
      --with-wa
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.router_triage_learnability import (  # noqa: E402
    ALL_FEATURES,
    CELLS,
    N_FOLDS,
    SEED,
    SIX_MODES,
    VWA_ONLY_FEATURES,
    WA_CELLS,
    _auroc,
    build_cell,
    build_wa_cell,
    oof_scores,
    policy_at_threshold,
)

ACTIVE_NAMES: list[str] = list(ALL_FEATURES)
ACTIVE_IDX: list[int] = list(range(len(ALL_FEATURES)))

# Same rationale and same value as router_triage_learnability.py: the plus-one
# estimator (k+1)/(B+1) floors at 1/(B+1); B=10000 floors two orders below the
# tightest Holm threshold any plausible cell count produces.
N_SHUFFLE = 10000


def per_mode_stats(cell: dict, idx) -> tuple[dict, dict]:
    idx = list(idx)
    n = len(idx)
    sr = {m: 100.0 * sum(cell["succ"][i][m] for i in idx) / n for m in SIX_MODES}
    cost = {m: sum(cell["cost"][i][m] for i in idx) / n for m in SIX_MODES}
    return sr, cost


# Canonical tie-break order. `max()`/`min()` return the FIRST extremal element,
# so on an exact tie the winner is decided by iteration order. Iterating
# SIX_MODES (a display order) let presentation silently decide a modelling
# choice, and 9 of 40 outer-fold selections in the 8-cell run hit an exact
# max-SR tie. Ordering the scan by `router_features.MODES` — the repo's single
# canonical mode order, ascending prior cost — makes the tie-break the same
# rule the label pipeline already uses, rather than a second, undeclared one.
_RF_ORDER = {"DOM": 0, "P-SoM": 1, "P-text": 2, "P-prompt": 3, "SoM": 4, "Vision": 5}
_TIEBREAK_MODES = tuple(sorted(SIX_MODES, key=lambda m: _RF_ORDER[m]))


def pick_arms(cell: dict, idx) -> tuple[str, str]:
    """best_mode = strongest-by-SR, cheap_mode = cheapest-by-cost, over `idx`.

    ⚠️ These two are chosen by INDEPENDENT extrema on two axes, not as a jointly
    optimal pair. That is deliberate: this pair is exactly the action space
    `oracle_triage` already operates over (see `router_triage_learnability.py`
    :365-366), and the whole point of this script is to compare per-task with
    per-class assignment *on the same two arms*. It is therefore NOT the best
    pair available — `enumerate_pairs()` below measures how far off it is, and
    the report calls this a conditional oracle rather than a ceiling.

    Ties broken by `_TIEBREAK_MODES` (canonical ascending-prior-cost order),
    not by SIX_MODES display order."""
    sr, cost = per_mode_stats(cell, idx)
    best = max(_TIEBREAK_MODES, key=lambda m: sr[m])
    cheap = min(_TIEBREAK_MODES, key=lambda m: cost[m])
    return best, cheap


def enumerate_pairs(cell: dict, idx) -> dict:
    """P1-3 diagnostic: is the (best,cheap) pair even on the two-arm frontier?

    Scores the per-task oracle over EVERY ordered pair of distinct modes and
    reports whether any pair dominates the selected one on both axes. On the
    8-cell run this fires for 2 cells, which is why the selected pair is
    reported as a conditional oracle, not as a two-arm ceiling."""
    idx = list(idx)
    sel_b, sel_c = pick_arms(cell, idx)

    def pair_oracle(b: str, c: str) -> tuple[float, float]:
        hits = spend = 0.0
        for i in idx:
            use = b if (cell["succ"][i][b] and not cell["succ"][i][c]) else c
            hits += cell["succ"][i][use]
            spend += cell["cost"][i][use]
        return 100.0 * hits / len(idx), spend / len(idx)

    sel_sr, sel_cost = pair_oracle(sel_b, sel_c)
    dominators = []
    for b in SIX_MODES:
        for c in SIX_MODES:
            if b == c:
                continue
            sr_bc, cost_bc = pair_oracle(b, c)
            if sr_bc > sel_sr and cost_bc <= sel_cost:
                dominators.append({"best": b, "cheap": c,
                                   "sr_pct": sr_bc, "mean_cost": cost_bc})
    dominators.sort(key=lambda d: (-d["sr_pct"], d["mean_cost"]))
    return {
        "selected_pair": [sel_b, sel_c],
        "selected_oracle": {"sr_pct": sel_sr, "mean_cost": sel_cost},
        "n_dominating_pairs": len(dominators),
        "best_dominating_pair": dominators[0] if dominators else None,
    }


def derive_two_arm_label(cell: dict, idx, best_mode: str, cheap_mode: str) -> np.ndarray:
    """z_i = 1 -> oracle sends task i to best_mode, 0 -> cheap_mode. Defined for
    EVERY task (unlike the six-way which-mode label).

    z_i = 1 iff best_mode succeeds AND cheap_mode fails on task i; every other
    case (both succeed, both fail, or only cheap_mode succeeds) is z_i = 0.

    Both-succeed and both-fail cases default to cheap_mode rather than
    tie-breaking on REALIZED cost. An earlier version of this label broke ties
    by whichever arm was cheaper on that specific episode, which made z's base
    rate on `reddit.B2` exceed `y`'s solvable rate (20.2% > 7.4%) — impossible
    if z=1 only ever meant "best_mode solved it". The realized cost of a
    FAILED episode is a function of how many steps it took before giving up,
    not of anything a pre-action feature can see; `derive_cost_oracle_label`
    in `p79/policies/router_features.py` documents exactly this trap for the
    six-way label ("puts outcome information into a target a pre-action
    router has to predict from pre-action features") and rejects it as
    canonical for the same reason. This label does not repeat that mistake:
    the SR ceiling `policy_at_threshold(cell, z.astype(float), 0.5, best_mode,
    cheap_mode)` reaches is IDENTICAL either way (both-succeed/both-fail tasks
    contribute the same success/failure regardless of which arm is nominally
    "picked"); only the ceiling's cost accounting becomes a fixed, honest
    default instead of an outcome-chased minimum."""
    idx = list(idx)
    z = np.zeros(len(idx), dtype=int)
    for pos, i in enumerate(idx):
        sb, sc = cell["succ"][i][best_mode], cell["succ"][i][cheap_mode]
        z[pos] = 1 if (sb and not sc) else 0
    return z


def evaluate(cell_spec: dict) -> dict | None:
    cell = build_wa_cell(cell_spec) if cell_spec.get("_wa") else build_cell(cell_spec)
    if cell is None:
        return None
    X = cell["X"][:, ACTIVE_IDX]
    y = cell["y"]  # triage label, kept only for the AUROC(z) vs AUROC(y) side-by-side
    n = len(y)
    all_idx = list(range(n))

    dead_features = [ACTIVE_NAMES[j] for j in range(X.shape[1]) if np.std(X[:, j]) == 0]

    best_mode, cheap_mode = pick_arms(cell, all_idx)
    z = derive_two_arm_label(cell, all_idx, best_mode, cheap_mode)
    z_base_rate = 100.0 * z.mean()  # % of tasks the oracle sends to best_mode

    sc_z = oof_scores(X, z)
    auroc_z = _auroc(z, sc_z["lr"])
    sc_y = oof_scores(X, y)
    auroc_y = _auroc(y, sc_y["lr"])
    feat_names = ACTIVE_NAMES
    per_feat_z = {feat_names[j]: _auroc(z, v) for j, v in sc_z["per_feature"].items()}
    best_feat_z = max(per_feat_z, key=lambda k: (per_feat_z[k] if per_feat_z[k] == per_feat_z[k] else -1))

    per_mode_sr, per_mode_cost = per_mode_stats(cell, all_idx)
    best_single = {"sr_pct": per_mode_sr[best_mode], "mean_cost": per_mode_cost[best_mode]}
    always_cheap = {"sr_pct": per_mode_sr[cheap_mode], "mean_cost": per_mode_cost[cheap_mode]}

    # Two perfect-knowledge policies over the SAME two arms {best_mode, cheap_mode}:
    #   oracle_triage:  y-driven, by-CLASS assignment (all "solvable" -> best_mode)
    #   oracle_two_arm: z-driven, by-TASK assignment (whichever arm actually wins)
    #
    # ⚠️ Only the SR ordering is guaranteed by construction. Any success the
    # class-level pick achieves is reproducible by the per-task pick, so
    # SR(two_arm) >= SR(triage) always. COST IS NOT GUARANTEED, and an earlier
    # revision of this file wrongly asserted full Pareto dominance "by
    # construction". Counterexample (3 tasks, arms B/C):
    #   succ B=[1,1,0] C=[0,0,1];  cost B=[100,100,0] C=[0,0,150]
    #   => B is best-by-SR (2/3 vs 1/3), C is cheapest-by-mean-cost (50 vs 66.7)
    #   => triage sends all three to B: SR 2/3, cost 66.7
    #   => two_arm sends task 3 to C:  SR 3/3, cost 116.7  -- higher SR, HIGHER cost
    # The 8-of-8 dominance this script reports is therefore an empirical
    # property of these logs (a failed episode here usually burns the whole step
    # budget, so recovering a success tends to cost less, not more), not a
    # theorem. Report it as measured, never as necessary.
    oracle_triage = policy_at_threshold(cell, y.astype(float), 0.5, best_mode, cheap_mode)
    oracle_two_arm = policy_at_threshold(cell, z.astype(float), 0.5, best_mode, cheap_mode)
    pair_diag = enumerate_pairs(cell, all_idx)

    # ---- Honest operating point: FULLY NESTED cross-validation (mirrors
    # router_triage_learnability.py's B-1903 fix). z is NOT fixed in advance —
    # it is RE-DERIVED per outer fold from that fold's training rows only,
    # because z depends on which two arms (best_o, cheap_o) are in play, and
    # those are themselves re-selected on training rows only. Nothing that
    # touches an outer test fold has seen it.
    from sklearn.linear_model import LogisticRegression

    def _fit_score(Xtr_raw, ytr, Xte_raw):
        if len(np.unique(ytr)) < 2:
            return None
        mu, sd = Xtr_raw.mean(0), Xtr_raw.std(0)
        sd = np.where(sd == 0, 1.0, sd)
        lr = LogisticRegression(max_iter=2000, C=1.0)
        lr.fit((Xtr_raw - mu) / sd, ytr)
        return lr.predict_proba((Xte_raw - mu) / sd)[:, 1]

    nested_hits = nested_spend = 0.0
    nested_sent_best = 0
    # Cross-fitted comparators (P1-4): always-cheap / always-best evaluated on
    # the same outer-test rows with the same fold-local arm choice.
    xf_cheap_hits = xf_cheap_spend = 0.0
    xf_best_hits = xf_best_spend = 0.0
    nested_folds: list[dict] = []
    rng_n = np.random.default_rng(SEED)
    idx_n = rng_n.permutation(n)
    outer_folds = np.array_split(idx_n, N_FOLDS)

    for f in outer_folds:
        tr = np.setdiff1d(idx_n, f)
        tr_list = list(tr)

        # (a) modes re-selected on training rows only.
        best_o, cheap_o = pick_arms(cell, tr_list)
        # (a2) z RE-DERIVED on training rows only, against (best_o, cheap_o).
        z_tr = derive_two_arm_label(cell, tr_list, best_o, cheap_o)

        # (b) inner-CV OOF scores over the training rows, predicting z_tr.
        inner_oof = np.full(len(tr_list), np.nan)
        rng_i = np.random.default_rng(SEED + 1)
        tr_shuf = rng_i.permutation(len(tr_list))
        for g in np.array_split(tr_shuf, N_FOLDS):
            in_tr = np.setdiff1d(tr_shuf, g)
            if len(np.unique(z_tr[in_tr])) < 2:
                continue
            p = _fit_score(X[tr][in_tr], z_tr[in_tr], X[tr][g])
            if p is not None:
                inner_oof[g] = p

        # (c) threshold chosen against inner OOF + training outcomes only.
        #     Preserve always-cheapest's training-fold SR floor (thesis
        #     convention: "Δ columns are against always-cheapest") while
        #     minimising cost. Unlike the triage label, z's oracle is NOT
        #     SR-neutral against best_o alone, so best_o is the wrong floor
        #     here — always-cheapest is the free comparator a router has to
        #     beat, and is the correct one to preserve.
        # P0-1: the candidate set must contain always-cheapest, and the
        # fallback must BE always-cheapest.
        #
        # Inherited from the triage script, this loop initialised
        # `thr_star = -np.inf` and drew candidates from quantiles 0..0.95.
        # Under `use = best if score >= thr else cheap`, -inf means "send
        # everything to BEST" and no candidate ever means "send everything to
        # CHEAP" (that requires +inf). The triage script's SR floor was
        # always-best, so its -inf fallback was coherent; this script's floor
        # is always-CHEAPEST, so both the fallback and the missing endpoint
        # pointed the wrong way. The symptom was unmissable in hindsight: 6 of
        # 8 cells reported a NEGATIVE "best saving", which is impossible when
        # the free always-cheapest policy is genuinely inside the feasible set.
        thr_star = np.inf
        if not np.all(np.isnan(inner_oof)):
            base_tr_sr = 100.0 * sum(cell["succ"][i][cheap_o] for i in tr_list) / len(tr_list)
            cands = np.append(
                np.quantile(inner_oof[~np.isnan(inner_oof)], np.linspace(0.0, 0.95, 20)),
                np.inf,                      # = always-cheapest, the free comparator
            )
            best_c = None
            for thr in cands:
                h = c = 0.0
                for pos, i in enumerate(tr_list):
                    if np.isnan(inner_oof[pos]):
                        use = cheap_o
                    else:
                        use = best_o if inner_oof[pos] >= thr else cheap_o
                    h += cell["succ"][i][use]
                    c += cell["cost"][i][use]
                if 100.0 * h / len(tr_list) >= base_tr_sr - 1e-9 and (best_c is None or c < best_c):
                    thr_star, best_c = float(thr), c

        # (d) score outer-test rows with an LR fitted on training rows only,
        #     trained against z_tr (never the outer-test z).
        #
        # P1-4: accumulate the CROSS-FITTED comparators on the same test rows
        # and the same fold-local arms. Comparing a nested router (which pays
        # for arm-selection uncertainty out of its training fold) against a
        # whole-cell always-cheapest (which was handed the min-cost arm by
        # full-data hindsight) puts the two on different information sets and
        # inflates the router's apparent cost penalty — measured at 4x on
        # wa_reddit.B0, whose 5th fold picks a different cheap arm than the
        # whole cell does.
        score_te = _fit_score(X[tr], z_tr, X[f]) if len(np.unique(z_tr)) >= 2 else None
        fold_best = 0
        for pos, i in enumerate(f):
            s = None if score_te is None else score_te[pos]
            use = cheap_o if s is None else (best_o if s >= thr_star else cheap_o)
            fold_best += use == best_o
            nested_sent_best += use == best_o
            nested_hits += cell["succ"][i][use]
            nested_spend += cell["cost"][i][use]
            xf_cheap_hits += cell["succ"][i][cheap_o]
            xf_cheap_spend += cell["cost"][i][cheap_o]
            xf_best_hits += cell["succ"][i][best_o]
            xf_best_spend += cell["cost"][i][best_o]
        nested_folds.append({
            "n_test": int(len(f)), "best_mode": best_o, "cheap_mode": cheap_o,
            "threshold": (None if not np.isfinite(thr_star)
                          else round(float(thr_star), 6)),
            "threshold_is_always_cheapest": bool(thr_star == np.inf),
            "n_sent_best": int(fold_best),
        })

    policies_nested = {
        "sr_pct": 100.0 * nested_hits / n, "mean_cost": nested_spend / n,
        "n_sent_best": int(nested_sent_best),
        "per_outer_fold": nested_folds,
        "note": (
            "FULLY nested, z re-derived per outer fold from training rows only "
            "(mirrors router_triage_learnability.py B-1903). This is the only "
            "achievable operating point here; the whole-cell sweep below is "
            "selection-contaminated by construction."
        ),
    }
    # The two comparators an honest nested comparison is entitled to: same test
    # rows, same fold-local arms, no full-data hindsight (P1-4).
    xfit_always_cheap = {"sr_pct": 100.0 * xf_cheap_hits / n,
                         "mean_cost": xf_cheap_spend / n}
    xfit_best_single = {"sr_pct": 100.0 * xf_best_hits / n,
                        "mean_cost": xf_best_spend / n}

    # Whole-cell threshold sweep (in-sample threshold, optimistic — kept only
    # as the object the label-shuffle null below is run against, exactly as
    # router_triage_learnability.py does for its own sweep).
    # P0-1 applies here too: +inf (= always-cheapest) must be in the sweep, or
    # the "best SR-preserving saving" can come out negative, which is only
    # possible when the free policy was excluded from the feasible set.
    _sweep_thr = np.append(
        np.quantile(sc_z["lr"][~np.isnan(sc_z["lr"])], np.linspace(0.0, 0.95, 20)), np.inf)
    sweep = [policy_at_threshold(cell, sc_z["lr"], float(thr), best_mode, cheap_mode)
             for thr in _sweep_thr]
    lossless = [p for p in sweep if p["sr_pct"] >= always_cheap["sr_pct"] - 1e-9]
    best_lossless = min(lossless, key=lambda p: p["mean_cost"]) if lossless else None

    # Bundle-permutation label-shuffle null (B-1902-safe): permute (z, succ,
    # cost) jointly against X. best_mode/cheap_mode are whole-cell aggregates
    # over REALIZED outcomes and therefore permutation-invariant, so they are
    # fixed across draws exactly as router_triage_learnability.py fixes them.
    rng_null = np.random.default_rng(SEED + 1)
    null_savings = []
    for _ in range(N_SHUFFLE):
        perm = rng_null.permutation(n)
        z_b = z[perm]
        succ_b = [cell["succ"][i] for i in perm]
        cost_b = [cell["cost"][i] for i in perm]
        cell_b = dict(cell, succ=succ_b, cost=cost_b)
        try:
            sp = oof_scores(X, z_b)["lr"]
        except Exception:
            continue
        # Same candidate set as the observed sweep, +inf included (P0-1): the
        # null must be free to reach always-cheapest exactly where the observed
        # statistic is, or the two are not comparable.
        sw = [policy_at_threshold(cell_b, sp, float(t), best_mode, cheap_mode)
              for t in np.append(
                  np.quantile(sp[~np.isnan(sp)], np.linspace(0.0, 0.95, 20)), np.inf)]
        ll = [q for q in sw if q["sr_pct"] >= always_cheap["sr_pct"] - 1e-9]
        null_savings.append(
            100.0 * (1 - min(q["mean_cost"] for q in ll) / always_cheap["mean_cost"]) if ll else 0.0
        )
    if len(null_savings) != N_SHUFFLE:
        raise RuntimeError(
            f"{cell['site']}/{cell['baseline']}: {len(null_savings)} of {N_SHUFFLE} "
            "permutation draws completed. A short draw count changes this cell's "
            "plus-one floor, so the reported p is not comparable across cells."
        )
    observed_saving = (100.0 * (1 - best_lossless["mean_cost"] / always_cheap["mean_cost"])
                        if best_lossless else 0.0)
    n_exceed = int(sum(1 for v in null_savings if v >= observed_saving - 1e-12))
    p_null = ((n_exceed + 1) / (len(null_savings) + 1)) if null_savings else float("nan")

    return {
        "site": cell["site"], "baseline_model": cell["baseline"], "n": n,
        "n_features": len(ACTIVE_NAMES), "dead_features": dead_features,
        "n_no_feature": cell["n_no_feature"],
        "best_mode": best_mode, "cheap_mode": cheap_mode,
        "z_base_rate_pct": z_base_rate,
        "solvable_rate_pct": 100.0 * y.mean(),
        "auroc_z": auroc_z, "auroc_y": auroc_y,
        "auroc_best_single_feature_z": per_feat_z[best_feat_z],
        "best_single_feature_z": best_feat_z,
        "auroc_z_minus_best_feature": auroc_z - per_feat_z[best_feat_z],
        "best_single": best_single,
        "always_cheapest": always_cheap,
        "xfit_always_cheapest": xfit_always_cheap,
        "xfit_best_single": xfit_best_single,
        "pair_frontier_diagnostic": pair_diag,
        "oracle_triage": oracle_triage,
        "oracle_two_arm": oracle_two_arm,
        "learned_two_arm_nested": policies_nested,
        "learned_two_arm_lossless": best_lossless,
        "observed_lossless_saving_pct": observed_saving,
        "null_shuffle_saving_median_pct": (float(np.median(null_savings))
                                           if null_savings else float("nan")),
        "null_shuffle_p": p_null,
        "null_p_estimator": "(k+1)/(B+1) plus-one Monte Carlo",
        "null_permutation_unit": "task bundle (z, succ_by_mode, cost_by_mode) vs X",
        "n_shuffles": len(null_savings),
    }


def _fmt_p(p: float) -> str:
    return f"{p:.4g}" if p < 1e-3 else f"{p:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    global N_SHUFFLE
    ap.add_argument("--out", type=Path)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--n-shuffle", type=int, default=N_SHUFFLE)
    ap.add_argument("--with-wa", action="store_true")
    ap.add_argument("--cells", type=str, default=None,
                     help="comma-separated 'site.baseline' filter for a fast smoke test, "
                          "e.g. 'classifieds.B0,reddit.B2'")
    args = ap.parse_args()
    N_SHUFFLE = args.n_shuffle

    global ACTIVE_NAMES, ACTIVE_IDX
    cells = list(CELLS)
    if args.with_wa:
        cells += WA_CELLS
        ACTIVE_NAMES = [f for f in ALL_FEATURES if f not in VWA_ONLY_FEATURES]
        ACTIVE_IDX = [ALL_FEATURES.index(f) for f in ACTIVE_NAMES]
    if args.cells:
        wanted = set(args.cells.split(","))
        cells = [c for c in cells if f"{c['site']}.{c['baseline']}" in wanted]

    res = [r for r in (evaluate(c) for c in cells) if r]
    n_feat = len(ACTIVE_NAMES)

    L = ["# Is the ACTION half of routing learnable?\n",
         f"`post_hoc_exploratory=True`, `h10_eligible=False`. Task-held-out {N_FOLDS}-fold "
         f"CV per cell, seed {SEED}, L2 LR on the {n_feat} raw features (same feature set "
         "as `router_triage_learnability.py`), fold-local standardisation.\n",
         "Label z: oracle's per-task pick between {best_mode, cheap_mode} — 1 iff best_mode "
         "succeeds AND cheap_mode fails, else 0 (both-succeed/both-fail default to cheap_mode, "
         "not a realized-cost tie-break — see `derive_two_arm_label`). Defined for EVERY task, "
         "but its POSITIVE class is thin: 2.2-14.4% base rate, comparable to or thinner than "
         "the six-way which-mode label's per-cell supply (笔记 §490.4).\n"]

    # Rerun band for net SR (笔记 §470 / aggregate_noise_floor_inventory CLEAN_PAIRS).
    # An effect this size or smaller is not separable from running the same
    # condition twice, so the cell-count headline must be split, not pooled.
    RERUN_BAND_PP = 2.23

    L.append("\n## 1. Per-task vs per-class assignment over the same two arms\n")
    L.append("| cell | n | oracle_triage SR/cost | oracle_two_arm SR/cost | ΔSR | Δcost | ΔSR vs rerun band |")
    L.append("|---|---|---|---|---|---|---|")
    # A cell that clears the band by less than one task's worth of SR is not
    # separated from it. One task is 100/n pp, so anything inside that margin is
    # a rounding artefact of the threshold, not a measurement that beat it.
    _knife = []
    for r in res:
        ot, ta = r["oracle_triage"], r["oracle_two_arm"]
        dsr = ta["sr_pct"] - ot["sr_pct"]
        one_task_pp = 100.0 / r["n"]
        if dsr > RERUN_BAND_PP and (dsr - RERUN_BAND_PP) < one_task_pp:
            band = "**knife-edge**"
            _knife.append((f"{r['site']}·{r['baseline_model']}", dsr - RERUN_BAND_PP))
        elif dsr > RERUN_BAND_PP:
            band = "clear"
        else:
            band = "**within band**"
        L.append(f"| {r['site']}·{r['baseline_model']} | {r['n']} | "
                 f"{ot['sr_pct']:.2f}/{ot['mean_cost']:.5f} | "
                 f"{ta['sr_pct']:.2f}/{ta['mean_cost']:.5f} | "
                 f"{dsr:+.2f}pp | "
                 f"{100*(ta['mean_cost']/ot['mean_cost']-1):+.1f}% | {band} |")
    n_dom = sum(1 for r in res if r["oracle_two_arm"]["sr_pct"] > r["oracle_triage"]["sr_pct"]
                and r["oracle_two_arm"]["mean_cost"] <= r["oracle_triage"]["mean_cost"])
    n_clear = sum(1 for r in res
                  if r["oracle_two_arm"]["sr_pct"] - r["oracle_triage"]["sr_pct"] > RERUN_BAND_PP)
    L.append(
        f"\n**Two statements, and they are not the same statement.** "
        f"(a) *Guaranteed by construction*: SR(two_arm) ≥ SR(triage) in every cell — any "
        f"success the class-level pick achieves is reproducible by the per-task pick. "
        f"(b) *Measured on these logs, not guaranteed*: the per-task pick is also cheaper, "
        f"giving full Pareto dominance in {n_dom} of {len(res)} cells. **Cost dominance is "
        f"NOT implied by the construction** — with a strong-but-dear arm and a weak-but-cheap "
        f"one, recovering a success by switching arms can cost more, not less (worked "
        f"counterexample in `evaluate()`). It holds here because a failed episode on these "
        f"benchmarks usually burns the whole step budget.\n")
    _knife_note = ""
    if _knife:
        _knife_note = (
            " ⚠️ **" + ", ".join(f"{nm} clears it by only {mg:.4f}pp" for nm, mg in _knife)
            + f"** — less than one task's worth of SR at this n, i.e. inside the rounding of "
            f"the threshold itself. Counting {'it' if len(_knife) == 1 else 'them'} as cleared "
            f"is an artefact; the defensible count is {n_clear - len(_knife)}.")
    L.append(
        f"**Of the {len(res)} cells, {n_clear} carry a ΔSR above the {RERUN_BAND_PP}pp "
        f"rerun band**; the rest move by less than re-running one unchanged condition would. "
        f"The direction is safe in all {len(res)} (it is guaranteed), but the magnitude is "
        f"only separable from noise in {n_clear - len(_knife)}--{n_clear}.{_knife_note}\n")

    L.append("\n### 1b. Is the selected pair even on the two-arm frontier?\n")
    L.append("| cell | selected (best, cheap) | its oracle SR/cost | dominating pairs | best alternative |")
    L.append("|---|---|---|---|---|")
    for r in res:
        pd_ = r["pair_frontier_diagnostic"]
        sel = pd_["selected_pair"]
        so = pd_["selected_oracle"]
        bd = pd_["best_dominating_pair"]
        alt = (f"({bd['best']}, {bd['cheap']}) {bd['sr_pct']:.2f}/{bd['mean_cost']:.5f}"
               if bd else "—")
        L.append(f"| {r['site']}·{r['baseline_model']} | ({sel[0]}, {sel[1]}) | "
                 f"{so['sr_pct']:.2f}/{so['mean_cost']:.5f} | "
                 f"{pd_['n_dominating_pairs']}/30 | {alt} |")
    n_notfrontier = sum(1 for r in res if r["pair_frontier_diagnostic"]["n_dominating_pairs"] > 0)
    L.append(
        f"\n⚠️ **The selected pair is dominated by some other pair in {n_notfrontier} of "
        f"{len(res)} cells.** `best_mode` and `cheap_mode` are two independent extrema, not a "
        "jointly optimal pair. That is the correct thing to hold fixed here — it is exactly "
        "the action space `oracle_triage` operates over, and the comparison in §1 is only "
        "meaningful on the same two arms — but it means **`oracle_two_arm` is a conditional "
        "oracle over the triage-selected pair, not a two-arm ceiling.** Any claim of the form "
        "'this is the best a two-arm policy could do' is unsupported.\n")

    L.append("\n## 2. Is z more or less predictable than the triage label y?\n")
    L.append("| cell | n | z base rate % (sent to best_mode) | solvable % (y) | AUROC(z) | AUROC(y) | AUROC(z) best single feat | that feature |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in res:
        L.append(f"| {r['site']}·{r['baseline_model']} | {r['n']} | "
                 f"{r['z_base_rate_pct']:.1f} | {r['solvable_rate_pct']:.1f} | "
                 f"**{r['auroc_z']:.3f}** | {r['auroc_y']:.3f} | "
                 f"{r['auroc_best_single_feature_z']:.3f} | `{r['best_single_feature_z']}` |")
    _clears = [r for r in res if r["auroc_z_minus_best_feature"] > 0]
    _z_worse = sum(1 for r in res if r["auroc_z"] < r["auroc_y"])
    _z_chance = sum(1 for r in res if r["auroc_z"] < 0.5)
    L.append(f"\nThe z-LR clears its own best single covariate in {len(_clears)} of "
             f"{len(res)} cells. AUROC(z) < AUROC(y) in {_z_worse} of {len(res)}, and falls "
             f"below chance in {_z_chance}.\n")
    L.append(
        "⚠️ **Do not read this as 'z is harder because its positive class is smaller'.** "
        "AUROC is a ranking statistic and is insensitive to class balance in expectation; "
        "a thin positive class inflates the VARIANCE of the estimate (5-22 positives per "
        "cell), it does not depress its expected value. What a low AUROC(z) says is that "
        "these features do not separate the specific margin z encodes — *this task needs the "
        "dear arm and the cheap one will not do* — which is a strictly finer distinction than "
        "y's *something solves this*. Scarcity and feature inadequacy are different diagnoses "
        "and this table cannot separate them; report both as open.\n")

    L.append("\n## 3. What does the LEARNED (honest, fully nested) two-arm policy buy?\n")
    L.append(
        "The nested router pays for arm-selection uncertainty out of its own training fold. "
        "Comparing it against a whole-cell always-cheapest — which was handed the min-cost arm "
        "by full-data hindsight — puts the two on different information sets. The Δ columns "
        "below are therefore against the **cross-fitted** comparator: always-cheapest evaluated "
        "on the same outer-test rows with the same fold-local arm. Whole-cell rows are kept as "
        "descriptive oracles, marked *(whole-cell)*.\n")
    L.append("| cell | policy | SR % | mean cost | ΔSR vs xfit-cheap | Δcost vs xfit-cheap | sent to best_mode |")
    L.append("|---|---|---|---|---|---|---|")
    for r in res:
        xc = r["xfit_always_cheapest"]
        L.append(f"| {r['site']}·{r['baseline_model']} | **always-cheapest (cross-fitted)** | "
                 f"{xc['sr_pct']:.2f} | {xc['mean_cost']:.5f} | ref. | ref. | 0/{r['n']} |")
        xb = r["xfit_best_single"]
        L.append(f"| | best-single (cross-fitted) | {xb['sr_pct']:.2f} | {xb['mean_cost']:.5f} | "
                 f"{xb['sr_pct']-xc['sr_pct']:+.2f}pp | "
                 f"{100*(xb['mean_cost']/xc['mean_cost']-1):+.1f}% | {r['n']}/{r['n']} |")
        nn = r["learned_two_arm_nested"]
        L.append(f"| | **learned two-arm, nested (honest)** | {nn['sr_pct']:.2f} | "
                 f"{nn['mean_cost']:.5f} | {nn['sr_pct']-xc['sr_pct']:+.2f}pp | "
                 f"{100*(nn['mean_cost']/xc['mean_cost']-1):+.1f}% | "
                 f"{nn['n_sent_best']}/{r['n']} sent best |")
        ac = r["always_cheapest"]
        ot = r["oracle_triage"]
        L.append(f"| | oracle triage *(whole-cell)* | {ot['sr_pct']:.2f} | {ot['mean_cost']:.5f} | "
                 f"{ot['sr_pct']-ac['sr_pct']:+.2f}pp | "
                 f"{100*(ot['mean_cost']/ac['mean_cost']-1):+.1f}% | "
                 f"{ot['n_sent_cheap']}/{r['n']} sent cheap |")
        ta = r["oracle_two_arm"]
        L.append(f"| | oracle two-arm, conditional on the pair *(whole-cell)* | "
                 f"{ta['sr_pct']:.2f} | {ta['mean_cost']:.5f} | "
                 f"{ta['sr_pct']-ac['sr_pct']:+.2f}pp | "
                 f"{100*(ta['mean_cost']/ac['mean_cost']-1):+.1f}% | "
                 f"{ta['n_sent_cheap']}/{r['n']} sent cheap |")

    def _pareto(a, b):
        return (a["sr_pct"] >= b["sr_pct"] and a["mean_cost"] <= b["mean_cost"]
                and (a["sr_pct"] > b["sr_pct"] or a["mean_cost"] < b["mean_cost"]))

    beat_xfit = [r for r in res
                 if _pareto(r["learned_two_arm_nested"], r["xfit_always_cheapest"])]
    beat_whole = [r for r in res
                  if _pareto(r["learned_two_arm_nested"], r["always_cheapest"])]
    L.append(
        f"\n**Learned nested two-arm policy Pareto-beats the cross-fitted always-cheapest in "
        f"{len(beat_xfit)} of {len(res)} cells** (against the whole-cell comparator it is "
        f"{len(beat_whole)} of {len(res)}). The comparator choice moves the effect size, not "
        "the verdict.\n")
    # A "win" of a hundredth of a percent of cost at identical SR is a tie that
    # rounded the right way, and saying 1-of-8 without saying which one invites
    # exactly the reading the number cannot support.
    for r in beat_xfit:
        nn, xc = r["learned_two_arm_nested"], r["xfit_always_cheapest"]
        dsr = nn["sr_pct"] - xc["sr_pct"]
        dcp = 100.0 * (nn["mean_cost"] / xc["mean_cost"] - 1)
        L.append(
            f"\n⚠️ Read that {len(beat_xfit)} closely. **{r['site']}·{r['baseline_model']}**: "
            f"{dsr:+.2f}pp SR and {dcp:+.2f}% cost — an identical success rate and a cost "
            f"difference of ${abs(nn['mean_cost'] - xc['mean_cost']):.6f} per episode, on a "
            f"cell whose base success is {xc['sr_pct']:.2f}% ({round(xc['sr_pct'] * r['n'] / 100)} "
            f"of {r['n']} tasks). It satisfies the Pareto definition and it is not a result "
            "anyone would deploy on.\n")
    if not beat_xfit:
        L.append("\nNo cell qualifies, so there is no borderline win to characterise.\n")

    L.append("\n## 4. Is the sweep saving real, or manufactured?\n")
    L.append("| cell | observed SR-floor-preserving saving vs always-cheap | median under shuffled z | p |")
    L.append("|---|---|---|---|")
    for r in res:
        L.append(f"| {r['site']}·{r['baseline_model']} | "
                 f"{r['observed_lossless_saving_pct']:.1f}% | "
                 f"{r['null_shuffle_saving_median_pct']:.1f}% | "
                 f"{_fmt_p(r['null_shuffle_p'])} |")
    ps = sorted((r["null_shuffle_p"], f"{r['site']}·{r['baseline_model']}") for r in res)
    m = len(ps)
    holm = []
    for i, (pv, name) in enumerate(ps):
        thresh = 0.05 / (m - i)
        holm.append((name, pv, thresh, pv < thresh))
        if pv >= thresh:
            break
    n_rej = sum(1 for _n, _p, _t, _ok in holm if _ok)
    L.append(f"\nHolm at α=0.05 over m={m} cells — **{n_rej} of {m} reject**:\n")
    for name, pv, thresh, ok in holm:
        verdict = "reject null" if ok else "**stop — this and all larger p unrejected**"
        L.append(f"- {name}: p={_fmt_p(pv)} vs {thresh:.4f} → {verdict}")

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
                          "holm_tightest_threshold": 0.05 / max(len(res), 1),
                          "n_cells": len(res), "features": ACTIVE_NAMES,
                          "n_features": len(ACTIVE_NAMES), "model": "L2 logistic regression"},
             "cells": res}, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
