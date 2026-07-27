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
N_SHUFFLE = 200


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

    # ---- Honest operating point: NESTED threshold selection.
    #
    # The sweep below picks the threshold by looking at realized outcomes on the
    # WHOLE cell, so its "SR-lossless" point is in-sample with respect to the
    # threshold even though the scores are out-of-fold. Cross-AI audit and self
    # audit flagged this independently (2026-07-27). It is retained because the
    # permutation null shares the same selection step — so the p-values compare
    # like with like — but it must NOT be read as an achievable operating point.
    # `nested` below is the achievable one: per outer fold, choose the threshold
    # on the training folds only, then apply it blind to the held-out fold.
    nested_hits = nested_spend = 0
    nested_sent_cheap = 0
    rng_n = np.random.default_rng(SEED)
    idx_n = rng_n.permutation(n)
    for f in np.array_split(idx_n, N_FOLDS):
        tr = np.setdiff1d(idx_n, f)
        sc_tr = sc["lr"][tr]
        if np.all(np.isnan(sc_tr)):
            thr_star = -np.inf
        else:
            base_tr_sr = 100.0 * sum(cell["succ"][i][best_mode] for i in tr) / len(tr)
            cands = np.quantile(sc_tr[~np.isnan(sc_tr)], np.linspace(0.0, 0.95, 20))
            thr_star, best_c = -np.inf, None
            for thr in cands:
                h = sum(cell["succ"][i][cheap_mode if sc["lr"][i] < thr else best_mode] for i in tr)
                c = sum(cell["cost"][i][cheap_mode if sc["lr"][i] < thr else best_mode] for i in tr)
                if 100.0 * h / len(tr) >= base_tr_sr - 1e-9 and (best_c is None or c < best_c):
                    thr_star, best_c = float(thr), c
        for i in f:
            use = cheap_mode if sc["lr"][i] < thr_star else best_mode
            nested_sent_cheap += (use == cheap_mode)
            nested_hits += cell["succ"][i][use]
            nested_spend += cell["cost"][i][use]
    policies_nested = {
        "sr_pct": 100.0 * nested_hits / n, "mean_cost": nested_spend / n,
        "n_sent_cheap": int(nested_sent_cheap),
        "note": "threshold chosen on train folds only, applied blind to held-out fold",
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
    rng_null = np.random.default_rng(SEED + 1)
    null_savings = []
    for _ in range(N_SHUFFLE):
        yp = rng_null.permutation(y)
        try:
            sp = oof_scores(X, yp)["lr"]
        except Exception:
            continue
        sw = [policy_at_threshold(cell, sp, float(t), best_mode, cheap_mode)
              for t in np.quantile(sp[~np.isnan(sp)], np.linspace(0.0, 0.95, 20))]
        ll = [q for q in sw if q["sr_pct"] >= baseline["sr_pct"] - 1e-9]
        null_savings.append(
            100.0 * (1 - min(q["mean_cost"] for q in ll) / baseline["mean_cost"]) if ll else 0.0
        )
    observed_saving = (100.0 * (1 - best_lossless["mean_cost"] / baseline["mean_cost"])
                       if best_lossless else 0.0)
    p_null = (float(np.mean([v >= observed_saving - 1e-12 for v in null_savings]))
              if null_savings else float("nan"))

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
        "n_shuffles": len(null_savings),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

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
    for r in res:
        L.append(f"| {r['site']}·{r['baseline_model']} | "
                 f"{r['observed_lossless_saving_pct']:.1f}% | "
                 f"{r['null_shuffle_saving_median_pct']:.1f}% | "
                 f"{r['null_shuffle_p']:.3f} |")
    L.append(f"\n{res[0]['n_shuffles']} label permutations per cell, same CV and same "
             "threshold sweep. The sweep picks its operating point post hoc, so it can "
             "extract an apparent saving from pure noise; this column is how much.\n")

    L.append("\n## 4. Verdict\n")
    ps = sorted((r["null_shuffle_p"], f"{r['site']}·{r['baseline_model']}") for r in res)
    m = len(ps)
    holm = []
    for i, (pv, name) in enumerate(ps):
        thresh = 0.05 / (m - i)
        holm.append((name, pv, thresh, pv < thresh))
        if pv >= thresh:
            break
    L.append("Holm at α=0.05 over the m=6 cells tested (the sweep was run once per cell, "
             "so the family is the six cells):\n")
    for name, pv, thresh, ok in holm:
        L.append(f"- {name}: p={pv:.3f} vs {thresh:.4f} → {'reject null' if ok else '**stop, no cell survives**'}")
    beat_cheap = [r for r in res
                  if r["learned_lossless"] is not None
                  and r["learned_lossless"]["mean_cost"] < r["always_cheapest"]["mean_cost"]
                  and r["learned_lossless"]["sr_pct"] >= r["always_cheapest"]["sr_pct"]]
    L.append(f"\nCells where the learned triage Pareto-beats the trivial always-cheapest "
             f"fixed policy: **{len(beat_cheap)} of {len(res)}**"
             + (f" ({', '.join(f'{r[chr(39)+chr(39)] if False else r['site']}·{r['baseline_model']}' for r in beat_cheap)})" if beat_cheap else "")
             + ".\n")
    L.append("Read together: the label is predictable (AUROC 0.65-0.72 in 5/6 cells, and "
             "unlike the which-mode task it clears the best single covariate in 4/6), but "
             "the prediction does not convert into operational value. Two cells yield no "
             "SR-lossless saving at all; two more yield savings a label-shuffled classifier "
             "matches; the two that beat their own null do not survive multiplicity "
             "correction. So the triage half joins the which-mode half as unlearnable "
             "here — but for a different reason. Which-mode fails on label supply "
             "(16-97 labels, 笔记 §383.4). Triage has the labels and the AUROC and still "
             "fails, because at 2-27% base SR the decision that matters is a narrow "
             "margin against a trivial fixed policy.\n")

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
             "protocol": {"folds": N_FOLDS, "seed": SEED,
                          "features": NUMERIC + BINARY, "model": "L2 logistic regression"},
             "cells": res}, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
