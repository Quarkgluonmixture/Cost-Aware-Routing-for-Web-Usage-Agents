#!/usr/bin/env python3
"""Is the routing negative result a mechanism, or just too few rows?

This closes the one attack on the dissertation that had no data behind the
answer. Stated at its strongest (Gemini, cold read, 2026-08-11):

    "The author dresses up a tiny sample size as a profound structural law. A
     cell with n=203 and a 7% success rate yields ~15 positive examples. Of
     course you cannot train a classifier on that. If n were 10,000 you would
     have 700 labels -- plenty. The negative result is an artifact of the
     author's compute budget, not the environment."

It is the cheapest possible rebuttal of a negative result, which is exactly why
it has to be answered with measurements rather than with prose. Four of them,
each isolating a different thing the objection could mean.

A. LEARNING CURVE -- does more of the same data help?
   Subsample the TRAINING rows only, stratified, leaving each outer test fold
   untouched, and sweep the training fraction. If out-of-fold performance is
   still climbing at full data, the objection stands: we are data-starved. If it
   has flattened, more rows of this kind buy nothing. Repeated over many seeds
   because at these n a single split is noise.

B. IN-SAMPLE SEPARABILITY -- is there signal to find at all?
   Fit a near-unregularised model on ALL rows and score it on those same rows,
   against the same model fitted to PERMUTED labels. This is the diagnostic that
   splits the objection cleanly:
     * train AUROC(real) >> train AUROC(permuted)  ->  signal exists, and only
       generalisation is failing  ->  the undersampling reading is right;
     * train AUROC(real)  ~ train AUROC(permuted)  ->  the model can do no
       better than memorise noise  ->  more rows of the same columns will not
       help, and the constraint is the feature set, not n.

C. THE COLUMN THAT DOES NOT EXIST AT SERVING TIME
   Run A and B twice: once on the 20-feature set that includes VisualWebArena's
   own `reasoning_difficulty` annotation, and once on the 18-feature deployment
   set that drops it. If the deployment curve is flat and low while the
   annotated curve is not, the binding constraint is a missing column rather
   than a missing row -- a claim about the regime, not about the budget.

D. HOW MANY TASKS WOULD THE WHICH-MODE HALF NEED?
   Labels are minted at rate (solvable %), so the required benchmark size for a
   given label budget is arithmetic. This answers "if n were 10,000" in the
   units the objection uses.

Average precision is reported alongside AUROC throughout, because a separate
criticism of this work is that AUROC flatters a model on 93%-negative data. AP
is the tail-sensitive counterpart and costs nothing to add here.

Output: docs/analysis/cross_sites/router_undersampling_control.{md,json}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from scripts.analysis.aggregate_phantom_lift import CELLS  # noqa: E402
from scripts.analysis import router_triage_learnability as L  # noqa: E402

OUT_MD = REPO / "docs/analysis/cross_sites/router_undersampling_control.md"
OUT_JSON = REPO / "docs/analysis/cross_sites/router_undersampling_control.json"

FRACTIONS = (0.25, 0.40, 0.55, 0.70, 0.85, 1.00)
N_SPLITS = 5
N_REPEAT = 20
SEED = 42
# Matches router_triage_learnability: the two VWA-only columns a deployment
# cannot have.
VWA_ONLY = ("reasoning_difficulty", "has_reference_image")


def feature_index(drop_vwa_only: bool) -> list[int]:
    names = list(L.ALL_FEATURES)
    if not drop_vwa_only:
        return list(range(len(names)))
    return [i for i, nm in enumerate(names) if nm not in VWA_ONLY]


def _fit_score(Xtr, ytr, Xte, yte):
    """Fold-local standardisation, then L2 LR. Returns (auroc, ap) or None."""
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return None
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    lr = LogisticRegression(max_iter=5000)
    lr.fit((Xtr - mu) / sd, ytr)
    p = lr.predict_proba((Xte - mu) / sd)[:, 1]
    return roc_auc_score(yte, p), average_precision_score(yte, p)


def learning_curve(X, y, idx, rng_seed=SEED):
    """A: out-of-fold AUROC/AP as the TRAINING set is thinned, test folds intact."""
    Xa = X[:, idx]
    out = {}
    for f in FRACTIONS:
        aucs, aps = [], []
        for rep in range(N_REPEAT):
            rng = np.random.default_rng(rng_seed + rep)
            skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                  random_state=rng_seed + rep)
            for tr, te in skf.split(Xa, y):
                if f < 1.0:
                    # Stratified thinning of the training rows only.
                    keep = []
                    for cls in (0, 1):
                        pool = tr[y[tr] == cls]
                        k = max(2, int(round(len(pool) * f)))
                        keep.append(rng.choice(pool, size=min(k, len(pool)),
                                               replace=False))
                    tr = np.concatenate(keep)
                r = _fit_score(Xa[tr], y[tr], Xa[te], y[te])
                if r is not None:
                    aucs.append(r[0]); aps.append(r[1])
        if not aucs:
            continue
        out[f] = {"n_train_median": int(round(len(y) * (N_SPLITS - 1) / N_SPLITS * f)),
                  "auroc_mean": float(np.mean(aucs)), "auroc_sd": float(np.std(aucs)),
                  "ap_mean": float(np.mean(aps)), "ap_sd": float(np.std(aps)),
                  "n_fits": len(aucs)}
    return out


def insample_separability(X, y, idx, n_perm=200, rng_seed=SEED):
    """B: can a near-unregularised model separate these rows AT ALL?

    Scored on the rows it was fitted to. That is the point: this is a
    memorisation test, not a performance estimate. The permuted-label arm is the
    only thing that makes the number interpretable -- it is how much a model of
    this capacity memorises from n rows and p columns of pure noise.
    """
    Xa = X[:, idx]
    mu, sd = Xa.mean(0), Xa.std(0)
    sd[sd == 0] = 1.0
    Z = (Xa - mu) / sd
    lr = LogisticRegression(C=1e6, max_iter=20000)
    lr.fit(Z, y)
    real_auc = roc_auc_score(y, lr.predict_proba(Z)[:, 1])
    real_ap = average_precision_score(y, lr.predict_proba(Z)[:, 1])

    rng = np.random.default_rng(rng_seed)
    perm = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        m = LogisticRegression(C=1e6, max_iter=20000)
        m.fit(Z, yp)
        perm.append(roc_auc_score(yp, m.predict_proba(Z)[:, 1]))
    perm = np.asarray(perm)
    return {"train_auroc_real": float(real_auc), "train_ap_real": float(real_ap),
            "train_auroc_perm_mean": float(perm.mean()),
            "train_auroc_perm_p95": float(np.quantile(perm, 0.95)),
            "excess_over_perm": float(real_auc - perm.mean()),
            "p_one_sided": float((1 + (perm >= real_auc).sum()) / (1 + len(perm))),
            "n_perm": n_perm, "n_rows": int(len(y)), "n_cols": int(Xa.shape[1])}


def whichmode_scale(supply_json: Path):
    """D: how big would the benchmark have to be to mint enough which-mode labels?"""
    d = json.loads(supply_json.read_text(encoding="utf-8"))
    rows = []
    # Two classes must each retain >=10 training rows under a 5-fold split, so
    # each needs >= 10 / (4/5) = 12.5 minted labels.
    need_per_class = 10 * N_SPLITS / (N_SPLITS - 1)
    for cell, v in d["supply"]["per_cell"].items():
        dist = sorted(v["label_distribution"].values(), reverse=True)
        second = dist[1] if len(dist) > 1 else 0
        scale = need_per_class / second if second > 0 else float("inf")
        rows.append({"cell": cell, "n_universe": v["n_universe"],
                     "minted": v["n_trainable_labels"],
                     "solvable_pct": v["solvable_rate_pct"],
                     "second_largest_class": second,
                     "scale_needed": scale,
                     "tasks_needed": (None if scale == float("inf")
                                      else int(round(v["n_universe"] * scale)))})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=N_REPEAT)
    ap.add_argument("--perms", type=int, default=200)
    a = ap.parse_args()
    globals()["N_REPEAT"] = a.repeats

    results = {}
    for spec in CELLS:
        cell = L.build_cell(dict(spec))
        if cell is None:
            print(f"  skip {spec['site']}·{spec['baseline']} (no rows)")
            continue
        key = f"{spec['site']}·{spec['baseline']}"
        X = np.asarray(cell["X"], dtype=float)
        y = np.asarray(cell["y"], dtype=int)
        entry = {"n": int(len(y)), "positives": int(y.sum())}
        for tag, drop in (("f20_annotated", False), ("f18_deployment", True)):
            idx = feature_index(drop)
            entry[tag] = {"n_features": len(idx),
                          "curve": learning_curve(X, y, idx),
                          "insample": insample_separability(X, y, idx, a.perms)}
        results[key] = entry
        c = entry["f18_deployment"]["curve"]
        lo, hi = c[FRACTIONS[0]]["auroc_mean"], c[1.00]["auroc_mean"]
        ins = entry["f18_deployment"]["insample"]
        print(f"  {key:<18} n={entry['n']:>3} pos={entry['positives']:>3} | "
              f"18-feat AUROC {lo:.3f}@25% -> {hi:.3f}@100% (Δ{hi - lo:+.3f}) | "
              f"in-sample {ins['train_auroc_real']:.3f} vs perm "
              f"{ins['train_auroc_perm_mean']:.3f} (p={ins['p_one_sided']:.3f})")

    scale = whichmode_scale(REPO / "results/phantom_paper/router_label_supply_diagnosis.json")
    payload = {"post_hoc_exploratory": True, "h10_eligible": False,
               "fractions": list(FRACTIONS), "n_splits": N_SPLITS,
               "n_repeat": a.repeats, "seed": SEED,
               "cells": results, "whichmode_scale": scale}
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _render(payload)
    print(f"\nwrote {OUT_MD.relative_to(REPO)}")
    return 0


def _render(p: dict) -> None:
    L_ = ["# Is the routing negative result undersampling?", "",
          "> Regenerate: `.venv/bin/python3 scripts/analysis/router_undersampling_control.py`",
          "> `post_hoc_exploratory=True`, `h10_eligible=False` — this answers an "
          "attack on a negative result; it is not a gate.", "",
          f"Triage label, {p['n_splits']}-fold stratified CV × {p['n_repeat']} "
          "repeats. Training rows are thinned; **test folds are never thinned**, "
          "so the curve isolates training size. AP is reported beside AUROC "
          "because these cells are 57–93% negative and AUROC alone flatters a "
          "model there.", "",
          "## A + C. Learning curve, annotated vs deployment feature set", "",
          "| cell | n | pos | set | AUROC @25% | @55% | @100% | Δ(25→100) | AP @100% |",
          "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for cell, e in p["cells"].items():
        for tag, lab in (("f20_annotated", "20 (incl. benchmark difficulty)"),
                         ("f18_deployment", "**18 (deployment)**")):
            c = e[tag]["curve"]
            g = lambda f, k: c[str(f)][k] if str(f) in c else c[f][k]  # noqa: E731
            L_.append(f"| `{cell}` | {e['n']} | {e['positives']} | {lab} | "
                      f"{g(0.25,'auroc_mean'):.3f} | {g(0.55,'auroc_mean'):.3f} | "
                      f"{g(1.00,'auroc_mean'):.3f} | "
                      f"{g(1.00,'auroc_mean') - g(0.25,'auroc_mean'):+.3f} | "
                      f"{g(1.00,'ap_mean'):.3f} |")
    L_ += ["",
           "**Saturation.** The comparison that decides A is not the total rise but "
           "whether the increments shrink. The second half of the sweep adds more "
           "absolute rows than the first; if it nonetheless buys less AUROC, the "
           "curve is saturating and further rows of the same kind are worth "
           "progressively less.", "",
           "| cell | set | Δ AUROC 25→55% | Δ AUROC 55→100% | ratio | verdict |",
           "|---|---|---:|---:|---:|---|"]
    for cell, e in p["cells"].items():
        for tag, lab in (("f20_annotated", "20"), ("f18_deployment", "**18**")):
            c = e[tag]["curve"]
            g = lambda f: (c[str(f)] if str(f) in c else c[f])["auroc_mean"]  # noqa: E731
            d1, d2 = g(0.55) - g(0.25), g(1.00) - g(0.55)
            ratio = (d2 / d1) if abs(d1) > 1e-9 else float("nan")
            verdict = ("saturating" if d2 < d1 else "still climbing")
            L_.append(f"| `{cell}` | {lab} | {d1:+.3f} | {d2:+.3f} | "
                      f"{ratio:.2f} | {verdict} |")

    L_ += ["", "## B. In-sample separability (near-unregularised, C=1e6)", "",
           "Scored on the rows it was fitted to — a memorisation test, made "
           "interpretable only by the permuted-label arm beside it.", "",
           "| cell | set | train AUROC (real) | train AUROC (permuted labels) | excess | p |",
           "|---|---|---:|---:|---:|---:|"]
    for cell, e in p["cells"].items():
        for tag, lab in (("f20_annotated", "20"), ("f18_deployment", "**18**")):
            i = e[tag]["insample"]
            L_.append(f"| `{cell}` | {lab} | {i['train_auroc_real']:.3f} | "
                      f"{i['train_auroc_perm_mean']:.3f} "
                      f"(p95 {i['train_auroc_perm_p95']:.3f}) | "
                      f"{i['excess_over_perm']:+.3f} | {i['p_one_sided']:.4f} |")
    L_ += ["", "## D. Benchmark size the which-mode half would need", "",
           "Two classes must each retain ≥10 training rows under a 5-fold split, "
           "so each needs ≥12.5 minted labels. Labels are minted at the solvable "
           "rate, so the required benchmark size is arithmetic.", "",
           "| cell | tasks now | minted | solvable % | 2nd-largest class | × needed | tasks needed |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for r in p["whichmode_scale"]:
        sc = ("—" if r["tasks_needed"] is None else f"{r['scale_needed']:.1f}×")
        tn = ("∞" if r["tasks_needed"] is None else f"**{r['tasks_needed']:,}**")
        L_.append(f"| `{r['cell']}` | {r['n_universe']} | {r['minted']} | "
                  f"{r['solvable_pct']:.1f} | {r['second_largest_class']} | {sc} | {tn} |")
    L_ += [""]
    OUT_MD.write_text("\n".join(L_), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
