#!/usr/bin/env python3
"""Leave-one-SITE-out transfer for the abstention router.

`abstention_learnability.py` shows "should this task be attempted at all" is learnable
*within* a cell (task-level 5-fold, AUROC 0.615-0.864). That is the weaker of the two
questions a reviewer asks. The NAACL attack surface is **generalisation**: does a policy
fitted on one site work on a site it has never seen? Within-cell CV cannot answer it —
every fold shares the site's task distribution, its DOM idiom, and its evaluator.

This script holds the feature set, the label function and the estimator fixed and changes
only the split: train on one site, test on the other.

Two protocols, because they answer different questions:

  A. **matched-model** — train (model, site A) -> test (model, site B). Model held fixed,
     so the only thing that changes is the site. 6 transfers.
  B. **pooled** — train all three cells of site A -> test each cell of site B. More
     training rows and cross-model diversity; the question is whether that buys transfer.

Both report the within-cell held-out AUROC of the TEST cell alongside, because that is
the ceiling this transfer is trying to reach, and a transfer number alone is unreadable.

The label-shuffle null is the same control §6 and `abstention_learnability` use: identical
features and split, training labels permuted. Under transfer the null matters MORE than it
does within-cell — a transferred classifier can look non-trivial purely by ranking on a
feature whose scale differs between sites.

*** THE OPERATING POINT IS CHOSEN WITHOUT THE TEST SITE ***

§465 adjudicated that held-out *prediction* and held-out *policy* are different claims, and
that any "saves X% under a Y loss budget" number needs its threshold picked by an inner CV
rather than swept over the test labels. Transfer makes that sharper: the threshold here is
selected by an inner 5-fold **inside the training site only** and then applied unseen to
the other site. Nothing about the test site — not its labels, not its base rate, not its
score distribution — touches the threshold.

⚠️ Calibration does not transfer even when ranking does. The universal-fail base rate runs
56.7% (B0 cls) to 92.9% (B2 cls), so a threshold that abstains on a sane fraction at home
can abstain on nearly everything abroad. AUROC is rank-based and blind to that; the
operating-point table is not, and the two therefore disagree by construction. Both are
reported. Reading only the AUROC would overstate what transfers.

Regenerate:
    .venv/bin/python3 scripts/analysis/abstention_site_transfer.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.abstention_learnability import (  # noqa: E402
    NO_SUCCESS,
    load_costs,
    patched_records,
    roc_auc,
    sr_cell_id,
)
from scripts.analysis.router_pooled_tier_learnability import (  # noqa: E402
    N_FOLDS,
    SEED,
    _fit_predict,
    outer_fold_map,
)

LOG = logging.getLogger("abstention_transfer")
SITES = ["classifieds", "reddit"]
MODELS = ["B0", "B1", "B2"]
LOSS_BUDGETS = [0.0, 0.02, 0.05, 0.10]

# P0-2 (/stress A F1, 2026-08-16): the null is a DISTRIBUTION, not a draw.
# v1 took ONE permutation and used the comparison as a binary verdict ("nothing
# was learned that survives the site change"). The AUROC null's own spread is
# `sqrt((n1+n0+1)/(12*n1*n0))` (Hanley-McNeil), which on the B2 cells is
# 0.075-0.078 — while the observed gaps there were -0.016/-0.017, i.e. 0.2 SD.
# A single draw cannot separate those. 200 permutations, and the reported
# reference is the 95th percentile of the null, not one sample.
N_PERM = 200

OUT_MD = REPO / "docs/analysis/cross_sites/abstention_site_transfer.md"
OUT_JSON = REPO / "docs/analysis/cross_sites/abstention_site_transfer.json"
WITHIN = REPO / "docs/analysis/cross_sites/abstention_learnability.json"


def cell_xy(rec: dict) -> tuple[np.ndarray, np.ndarray, list[int]]:
    X = np.hstack([np.asarray(rec["X_numeric"], float),
                   np.asarray(rec["X_binary"], float)])
    y = np.array([1 if lab == NO_SUCCESS else 0 for lab in rec["labels"]], dtype=int)
    return X, y, list(rec["task_ids"])


def stack(recs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for r in recs:
        X, y, _ = cell_xy(r)
        Xs.append(X)
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


def _positive_proba(fit, Xte: np.ndarray) -> np.ndarray | None:
    """`_fit_predict` returns (classes, proba); pull the column for class 1."""
    if fit is None:
        return None
    classes, proba = fit
    idx = np.where(classes == 1)[0]
    if len(idx) == 0:
        return None
    return proba[:, idx[0]]


def transfer(train_recs: list[dict], test_rec: dict, rng: np.random.Generator) -> dict:
    Xtr, ytr = stack(train_recs)
    Xte, yte, tids = cell_xy(test_rec)

    p = _positive_proba(_fit_predict(Xtr, ytr, Xte), Xte)
    if p is None:
        return {"error": "single-class training set"}

    # Same features, same split, permuted training labels — N_PERM times.
    null = []
    for _ in range(N_PERM):
        pn = _positive_proba(_fit_predict(Xtr, rng.permutation(ytr), Xte), Xte)
        if pn is not None:
            a = roc_auc(yte, pn)
            if not np.isnan(a):
                null.append(a)
    null_arr = np.asarray(null, float)
    auroc = roc_auc(yte, p)
    n1 = float(yte.sum())
    n0 = float(len(yte) - n1)
    # Hanley-McNeil analytic null SD, reported alongside the empirical one so a
    # reader can see the two agree (or not) rather than trusting the resample alone.
    sd_analytic = float(np.sqrt((n1 + n0 + 1) / (12.0 * n1 * n0))) if n1 and n0 else float("nan")

    return {
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "train_base_rate": float(ytr.mean()),
        "test_base_rate": float(yte.mean()),
        "auroc": auroc,
        "null_n": int(len(null_arr)),
        "null_mean": float(null_arr.mean()) if len(null_arr) else float("nan"),
        "null_sd_empirical": float(null_arr.std(ddof=1)) if len(null_arr) > 1 else float("nan"),
        "null_sd_analytic": sd_analytic,
        "null_p95": float(np.percentile(null_arr, 95)) if len(null_arr) else float("nan"),
        # one-sided permutation p, plus-one corrected (a Monte-Carlo p can never be 0)
        "perm_p": (float((null_arr >= auroc).sum() + 1) / (len(null_arr) + 1)
                   if len(null_arr) else float("nan")),
        "_p": p,
        "_y": yte,
        "_tids": tids,
    }


def pick_threshold_on_train(train_rec: dict, budget: float) -> tuple[float, dict]:
    """Threshold meeting `budget` on an inner CV **inside the training cell**.

    Takes ONE cell, not a list. P1-15 (/stress A F5): the previous signature accepted
    `list[dict]` and split by ROW index, while the project's canonical `outer_fold_map`
    is keyed by TASK id precisely so "a task's B0 row and its B1 row can never land on
    opposite sides of the split" — load-bearing for any POOLED design. With a list of
    pooled cells this leaked the threshold selection; it was only safe because every
    call site passed a single cell. Narrowing the signature makes that structural rather
    than accidental, and the split below is now task-keyed like the canonical helper
    (`abstention_learnability.evaluate_cell` uses `j % N_FOLDS` over sorted task ids).

    Also returns the QUANTIZATION record. P0-1 (/stress A F2 + gemini G5): a percentage
    budget on a small solvable set collapses to a handful of integers — B1_reddit has 24
    solvable tasks, so ≤0% and ≤2% both permit floor(0.02*24)=0 losses and are therefore
    the SAME policy. Reporting them as separate frontier points implies a sweep that does
    not exist. The integer is now carried and printed.
    """
    X, y, tids = cell_xy(train_rec)
    n = len(y)
    fold_of = {t: j % N_FOLDS for j, t in enumerate(sorted(tids))}
    oof = np.full(n, np.nan)
    for f in range(N_FOLDS):
        te = np.array([i for i, t in enumerate(tids) if fold_of[t] == f])
        tr = np.array([i for i, t in enumerate(tids) if fold_of[t] != f])
        if te.size == 0 or tr.size == 0:
            continue
        p = _positive_proba(_fit_predict(X[tr], y[tr], X[te]), X[te])
        if p is not None:
            oof[te] = p
    ok = ~np.isnan(oof)
    solvable = int((y[ok] == 0).sum())
    quant = {"solvable_train": solvable,
             "budget_tasks": int(np.floor(budget * solvable)) if solvable else 0}
    if not ok.any() or solvable == 0:
        return float("inf"), quant
    o, yy = oof[ok], y[ok]
    allowed = quant["budget_tasks"]
    best = float("inf")
    for thr in np.unique(o):
        if int(((o >= thr) & (yy == 0)).sum()) <= allowed:
            best = min(best, float(thr))
    return best, quant


def operating_point(res: dict, thr: float, costs, test_cell_id: str) -> dict:
    """Apply a threshold picked elsewhere. Loss = solvable tasks abstained on.

    `test_cell_id` is the EXTRACTOR's name (`B0_reddit`); `load_costs` is keyed by
    `per_task_sr.csv`'s name (`red_B0`). Passing the extractor name straight through
    yields an all-NaN cost column, and `nansum` returns 0.0 for that rather than
    raising — "no cost data" would then print as "$0.00 saved". This walked into
    exactly the trap `abstention_learnability.py:78-81` documents; `cost_coverage`
    below is the guard that caught it.
    """
    p, y, tids = res["_p"], res["_y"], res["_tids"]
    abstain = p >= thr if np.isfinite(thr) else np.zeros(len(p), bool)
    solvable = int((y == 0).sum())
    lost = int((abstain & (y == 0)).sum())
    baseline, site = test_cell_id.split("_", 1)
    cost_key = sr_cell_id(baseline, site)
    c = np.array([costs.get((cost_key, t), np.nan) for t in tids], float)
    total = float(np.nansum(c))
    saved = float(np.nansum(np.where(abstain, c, 0.0)))
    return {
        "threshold": None if not np.isfinite(thr) else thr,
        "abstain_rate": float(abstain.mean()),
        "solvable_lost": lost,
        "solvable_total": solvable,
        "loss_rate": (lost / solvable) if solvable else float("nan"),
        "cost_total_usd": total,
        "cost_saved_usd": saved,
        "cost_saved_pct": (100.0 * saved / total) if total > 0 else float("nan"),
        "cost_coverage": float(np.isfinite(c).mean()),
    }


def duplication_audit(by_cell: dict, site: str) -> dict:
    """Why the pooled protocol is withdrawn, measured rather than asserted.

    P0-3 (/stress gemini G3): pooling B0+B1+B2 rows of one site stacks three rows per
    task. The abstention LABEL is per-(task, model) — "did THIS model's six modes solve
    it" — but the FEATURES are step-0 observation statistics plus task config, which are
    largely model-invariant for a fixed task. So the pooled design feeds the classifier
    near-duplicate x with conflicting y, and it can only learn average-LLM solvability;
    testing it on one model's cell is a target mismatch.

    This counts the duplication instead of arguing about it.
    """
    cells = [by_cell[f"{m}_{site}"] for m in MODELS if f"{m}_{site}" in by_cell]
    if len(cells) < 2:
        return {"error": "need >=2 cells"}
    xs, ys, tid = [], [], []
    for c in cells:
        X, y, t = cell_xy(c)
        xs.append({tt: X[i] for i, tt in enumerate(t)})
        ys.append({tt: int(y[i]) for i, tt in enumerate(t)})
        tid.append(set(t))
    common = sorted(set.intersection(*tid))
    ident = sum(1 for t in common
                if all(np.allclose(xs[0][t], xs[k][t]) for k in range(1, len(xs))))
    conflict = sum(1 for t in common if len({yy[t] for yy in ys}) > 1)
    both = sum(1 for t in common
               if all(np.allclose(xs[0][t], xs[k][t]) for k in range(1, len(xs)))
               and len({yy[t] for yy in ys}) > 1)
    return {"site": site, "n_cells": len(cells), "n_common_tasks": len(common),
            "identical_features": ident, "conflicting_labels": conflict,
            "identical_and_conflicting": both,
            "pct_identical_and_conflicting": 100.0 * both / len(common) if common else float("nan")}


def render_md(d: dict) -> str:
    L = []
    L.append("---")
    L.append("type: analysis")
    L.append("status: complete")
    L.append("purpose: does an abstention policy fitted on one site work on a site it has never seen")
    L.append("scope_warning: 6 VWA cells, 2 sites. The abstention label is N=1 per mode. "
             "AUROC is rank-based and blind to calibration drift; the operating-point table is not, "
             "and they disagree by construction. Read both.")
    L.append("producer: scripts/analysis/abstention_site_transfer.py")
    L.append(f"generated: {d['generated']}")
    L.append("---")
    L.append("")
    L.append("# Abstention across sites: does the policy transfer?")
    L.append("")
    L.append("Regenerate: `.venv/bin/python3 scripts/analysis/abstention_site_transfer.py`")
    L.append("")
    L.append("`abstention_learnability` answers the *within-cell* question with task-level 5-fold. "
             "Every fold there shares the site's task distribution, DOM idiom and evaluator, so it "
             "cannot speak to generalisation. Here the feature set, label function and estimator are "
             "unchanged and only the split moves: **train on one site, test on the other**.")
    L.append("")

    L.append("## 1. Matched-model transfer (model fixed, site swapped)")
    L.append("")
    L.append(f"The null column is the 95th percentile of **{d['n_perm']} label permutations**, "
             "not a single draw, and the permutation p is plus-one corrected. A one-draw null "
             "cannot support a verdict: the AUROC null's own SD on the sparsest cells here is "
             "~0.08, which is larger than several of the gaps being judged.")
    L.append("")
    L.append("| train | test | n test | base rate train → test | **transfer AUROC** | "
             "null p95 | null SD (emp / analytic) | perm p | within-cell (ceiling) |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in d["matched"]:
        w = d["within_cell_auroc"].get(r["test_cell"])
        L.append(
            f"| `{r['train_cell']}` | `{r['test_cell']}` | {r['n_test']} | "
            f"{100*r['train_base_rate']:.1f}% → {100*r['test_base_rate']:.1f}% | "
            f"**{r['auroc']:.3f}** | {r['null_p95']:.3f} | "
            f"{r['null_sd_empirical']:.3f} / {r['null_sd_analytic']:.3f} | "
            f"{r['perm_p']:.3f} | {'—' if w is None else f'{w:.3f}'} |")
    L.append("")

    L.append("## 2. ⚠️ The pooled protocol is WITHDRAWN")
    L.append("")
    L.append("An earlier version of this artifact also pooled all three models of one site "
             "(≈672 rows) and tested on each cell of the other, reporting that pooling "
             "cleared the null in 5 of 6 and even beat one cell's own within-cell ceiling. "
             "**That comparison is withdrawn**, because the pooled design does not estimate "
             "the quantity it is tested against:")
    L.append("")
    for a in d.get("duplication", []):
        if a.get("error"):
            continue
        L.append(f"- **{a['site']}**, {a['n_cells']} cells, {a['n_common_tasks']} shared tasks: "
                 f"the feature vector is **byte-identical across all models on "
                 f"{a['identical_features']} tasks "
                 f"({100*a['identical_features']/a['n_common_tasks']:.1f}%)**, the label "
                 f"conflicts across models on {a['conflicting_labels']} "
                 f"({100*a['conflicting_labels']/a['n_common_tasks']:.1f}%), and "
                 f"**{a['identical_and_conflicting']} tasks "
                 f"({a['pct_identical_and_conflicting']:.1f}%) are both at once**.")
    L.append("")
    L.append("The abstention label is per-(task, model) — *did this model's six modes solve it* "
             "— while the features are step-0 observation statistics and task config, which are "
             "largely model-invariant. Pooling therefore trains on same-x-different-y triples and "
             "can only recover *average-LLM solvability*; scoring it against a single model's "
             "cell is a target mismatch, not a transfer result. The numbers remain in the JSON "
             "under `pooled` for the record; nothing in this document rests on them.")
    L.append("")

    L.append("## 3. Transferred operating point — threshold never sees the test site")
    L.append("")
    L.append("The threshold is chosen by an inner 5-fold **inside the training site** at the stated "
             "solvable-loss budget, then applied unseen to the other site. This is the cross-site "
             "analogue of the nested column in `abstention_learnability` §3, and it is a held-out "
             "*policy*, not merely a held-out prediction (§465).")
    L.append("")
    L.append("⚠️ **The percentage budget is quantised.** What the inner CV actually enforces is "
             "an integer: `floor(budget × solvable_train)`. On a small solvable set several "
             "percentage rows collapse onto the SAME integer and are therefore the same policy — "
             "the `budget→tasks` column below makes that visible instead of implying a sweep "
             "that does not exist.")
    L.append("")
    L.append("| train | test | budget | **budget→tasks** | abstain rate | solvable lost | "
             "realised loss | saved |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in d["operating_points"]:
        op = r["op"]
        q = r.get("quant") or {}
        lost = f"{op['solvable_lost']}/{op['solvable_total']}"
        loss = "—" if np.isnan(op["loss_rate"]) else f"{100*op['loss_rate']:.1f}%"
        saved = "—" if np.isnan(op["cost_saved_pct"]) else f"{op['cost_saved_pct']:.1f}%"
        bt = f"{q.get('budget_tasks','?')} / {q.get('solvable_train','?')}"
        L.append(f"| `{r['train_cell']}` | `{r['test_cell']}` | ≤{100*r['budget']:.0f}% | "
                 f"**{bt}** | {100*op['abstain_rate']:.1f}% | {lost} | {loss} | {saved} |")
    L.append("")
    L.append("⚠️ **A budget met at home is not a budget met abroad.** The budget column is what the "
             "threshold bought on the training site; the realised-loss column is what it actually cost "
             "on the test site. Where the two diverge, the cause is base-rate drift (56.7%–92.9% across "
             "these cells), and it is exactly the failure mode within-cell CV cannot show.")
    L.append("")
    L.append("⚠️ Cost coverage is the fraction of test-cell tasks with a cost figure in "
             "`per_task_sr.csv`; that product is on the leak-kept, n=205 reddit convention while the "
             "labels here come from the canonical n=203 universe (§462.1). The join is by task id, so "
             "surplus rows are simply unused, but the saved-dollar column inherits the cost side's "
             "convention.")
    L.append("")

    # --- §4 derived, never typed (§455: a hardcoded paragraph is how a subrange with the
    #     sign change removed got quoted downstream for weeks) -------------------------
    L.append("## 4. What transfers, and what does not")
    L.append("")

    sig = [r for r in d["matched"] if r["perm_p"] <= 0.05]
    ind = [r for r in d["matched"] if r["perm_p"] > 0.05]
    L.append(
        f"**Ranking transfers, on the cells that have the events to show it.** "
        f"{len(sig)} of {len(d['matched'])} matched transfers clear their permutation null at "
        f"p≤0.05. "
        + (f"The {len(ind)} that {'does' if len(ind) == 1 else 'do'} not — "
           + ", ".join(f"`{r['train_cell']}`→`{r['test_cell']}` (p={r['perm_p']:.3f})" for r in ind)
           + f" — {'is' if len(ind) == 1 else 'are'} **indeterminate, not negative**: "
             "the test cell carries so few solvable tasks "
             "that the null's own spread (SD "
           + "/".join(f"{r['null_sd_analytic']:.3f}" for r in ind)
           + ") is of the same order as any effect one could hope to see there." if ind else ""))
    L.append("")

    honoured = [o for o in d["operating_points"]
                if not np.isnan(o["op"]["loss_rate"]) and o["op"]["loss_rate"] <= o["budget"] + 1e-9]
    worst = max(d["operating_points"], key=lambda o: (o["op"]["loss_rate"]
                                                      if not np.isnan(o["op"]["loss_rate"]) else -1))
    # How many of the nominal budget rows are actually distinct policies?
    bykey: dict[tuple[str, str], set] = {}
    for o in d["operating_points"]:
        bykey.setdefault((o["train_cell"], o["test_cell"]), set()).add(
            (o.get("quant") or {}).get("budget_tasks"))
    collapsed = sum(len(LOSS_BUDGETS) - len(v) for v in bykey.values())
    L.append(
        f"**The operating point does not transfer, and the ranking does not warn you.** "
        f"{len(honoured)} of {len(d['operating_points'])} transferred thresholds kept the realised "
        f"loss inside the budget they were bought at. The worst is "
        f"`{worst['train_cell']}` → `{worst['test_cell']}` at ≤{100*worst['budget']:.0f}%: it "
        f"abstains on {100*worst['op']['abstain_rate']:.1f}% of the test site and loses "
        f"{worst['op']['solvable_lost']}/{worst['op']['solvable_total']} "
        f"({100*worst['op']['loss_rate']:.1f}%) of its solvable tasks.")
    L.append("")
    L.append(
        "⚠️ **What this is NOT.** An earlier version attributed the failure to *base-rate drift* "
        "and paired it with the observation that the highest-AUROC direction fails worst. Both were "
        "withdrawn against this table. Base-rate drift does not order the failures: the largest "
        "drop (" + ", ".join(
            f"{100*(r['test_base_rate']-r['train_base_rate']):+.1f}pp"
            for r in sorted(d["matched"], key=lambda r: r["test_base_rate"] - r["train_base_rate"])[:1])
        + ") stays inside budget while a near-zero drop overruns it; and pairing the max-AUROC "
          f"direction with the max-loss direction over {len(d['matched'])} transfers is a "
          f"1-in-{len(d['matched'])} coincidence under the null, not a mechanism.")
    L.append("")
    L.append(
        f"**What the table does support** is quantisation. Of the "
        f"{len(LOSS_BUDGETS)}×{len(bykey)} nominal budget rows, **{collapsed} are duplicates of "
        f"another row in the same direction** — the integer `floor(budget × solvable_train)` "
        "repeats when the solvable set is small. The budget axis is coarser than it looks, and a "
        "policy bought at one nominal budget can be the identical policy sold at another. That is "
        "the cross-site form of the distinction §465 drew within a cell: **a held-out prediction is "
        "not a held-out policy** — and here the policy is not even a continuum.")
    L.append("")
    L.append("⚠️ Two sites is two points. Nothing here licenses a claim about transfer to a *third* "
             "site; what it licenses is that ranking survived the one site change available on the "
             "cells that had the events to test it, and calibration did not.")
    return "\n".join(L) + "\n"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--allow-incomplete", action="store_true")
    a = ap.parse_args()

    recs = patched_records(allow_incomplete=a.allow_incomplete)
    by_cell = {cid: r for cid, r in recs.items()}
    LOG.info("cells: %s", sorted(by_cell))

    within = {}
    if WITHIN.is_file():
        wj = json.loads(WITHIN.read_text())
        # `cells` is a LIST of per-cell dicts, and the key is `auroc_heldout`
        # (not `auroc`). Both were guessed wrong on the first pass; a dict-shaped
        # read failed loudly, but a wrong key name would have silently produced an
        # all-"—" ceiling column that reads as "no within-cell number exists".
        for v in (wj.get("cells") or []):
            if isinstance(v, dict) and v.get("auroc_heldout") is not None:
                within[str(v["cell_id"])] = float(v["auroc_heldout"])

    costs = load_costs()
    rng = np.random.default_rng(SEED)

    matched, pooled, ops = [], [], []

    for m in MODELS:
        for tr_site, te_site in [("classifieds", "reddit"), ("reddit", "classifieds")]:
            tr_id, te_id = f"{m}_{tr_site}", f"{m}_{te_site}"
            if tr_id not in by_cell or te_id not in by_cell:
                LOG.warning("skip matched %s -> %s (missing)", tr_id, te_id)
                continue
            r = transfer([by_cell[tr_id]], by_cell[te_id], rng)
            if r.get("error"):
                LOG.warning("skip matched %s -> %s: %s", tr_id, te_id, r["error"])
                continue
            for b in LOSS_BUDGETS:
                thr, quant = pick_threshold_on_train(by_cell[tr_id], b)
                ops.append({"train_cell": tr_id, "test_cell": te_id, "budget": b,
                            "quant": quant,
                            "op": operating_point(r, thr, costs, te_id)})
            matched.append({**{k: v for k, v in r.items() if not k.startswith("_")},
                            "train_cell": tr_id, "test_cell": te_id})

    for tr_site, te_site in [("classifieds", "reddit"), ("reddit", "classifieds")]:
        tr_recs = [by_cell[f"{m}_{tr_site}"] for m in MODELS if f"{m}_{tr_site}" in by_cell]
        for m in MODELS:
            te_id = f"{m}_{te_site}"
            if te_id not in by_cell or not tr_recs:
                continue
            r = transfer(tr_recs, by_cell[te_id], rng)
            if r.get("error"):
                continue
            pooled.append({**{k: v for k, v in r.items() if not k.startswith("_")},
                           "train_site": tr_site, "test_cell": te_id})

    out = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "n_folds_inner": N_FOLDS,
        "n_perm": N_PERM,
        "duplication": [duplication_audit(by_cell, s) for s in SITES],
        "within_cell_auroc": within,
        "matched": matched,
        "pooled": pooled,
        "operating_points": ops,
    }
    # Fail loud on an empty cost join rather than reporting "$0.00 saved" — the
    # naming mismatch between the extractor and per_task_sr.csv makes this a live
    # trap, not a hypothetical one (it fired once while writing this script).
    covs = [o["op"]["cost_coverage"] for o in ops]
    if covs and max(covs) == 0.0:
        raise RuntimeError(
            "cost join returned zero coverage for every operating point — the "
            "(cell_id, task_id) key is wrong. Expected per_task_sr.csv names like "
            "'red_B0', got extractor names like 'B0_reddit'."
        )
    if covs and min(covs) < 0.9:
        LOG.warning("cost coverage below 0.9 on some cells: min=%.3f", min(covs))

    a.out_json.write_text(json.dumps(out, indent=2, default=float))
    a.out_md.write_text(render_md(out))
    print(f"[md]   {a.out_md}")
    print(f"[json] {a.out_json}")

    print(f"\n=== matched-model site transfer (AUROC / null p95 / perm p / ceiling), "
          f"{N_PERM} permutations ===")
    for r in matched:
        w = within.get(r["test_cell"])
        verdict = "clears" if r["perm_p"] <= 0.05 else "INDETERMINATE"
        print(f"  {r['train_cell']:>17} -> {r['test_cell']:<17} "
              f"{r['auroc']:.3f} / {r['null_p95']:.3f} / p={r['perm_p']:.3f} / "
              f"{'--' if w is None else f'{w:.3f}'}   {verdict}")
    print("\n=== pooled: WITHDRAWN (target mismatch) — duplication audit ===")
    for a in out["duplication"]:
        if a.get("error"):
            continue
        print(f"  {a['site']:<12} 共有 task {a['n_common_tasks']:>4} | "
              f"特征三模型相同 {a['identical_features']:>4} "
              f"({100*a['identical_features']/a['n_common_tasks']:.1f}%) | "
              f"标签冲突 {a['conflicting_labels']:>4} | "
              f"**两者同时 {a['identical_and_conflicting']:>4} "
              f"({a['pct_identical_and_conflicting']:.1f}%)**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
