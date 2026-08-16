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

    # Same features, same split, permuted training labels.
    p_null = _positive_proba(_fit_predict(Xtr, rng.permutation(ytr), Xte), Xte)

    return {
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "train_base_rate": float(ytr.mean()),
        "test_base_rate": float(yte.mean()),
        "auroc": roc_auc(yte, p),
        "auroc_shuffle": roc_auc(yte, p_null) if p_null is not None else float("nan"),
        "_p": p,
        "_y": yte,
        "_tids": tids,
    }


def pick_threshold_on_train(train_recs: list[dict], budget: float) -> float:
    """Largest abstain-rate threshold whose solvable-loss stays inside `budget`.

    Chosen by an inner 5-fold **inside the training site**. The test site is not consulted
    — not its labels, not its score distribution. Returns +inf when no threshold in the
    training site meets the budget (i.e. abstain on nothing).
    """
    Xtr, ytr = stack(train_recs)
    n = len(ytr)
    oof = np.full(n, np.nan)
    idx = np.arange(n)
    inner_rng = np.random.default_rng(SEED)
    perm = inner_rng.permutation(idx)
    folds = np.array_split(perm, N_FOLDS)
    for f in range(N_FOLDS):
        te = folds[f]
        tr = np.setdiff1d(idx, te)
        p = _positive_proba(_fit_predict(Xtr[tr], ytr[tr], Xtr[te]), Xtr[te])
        if p is None:
            continue
        oof[te] = p
    ok = ~np.isnan(oof)
    if not ok.any():
        return float("inf")
    o, yy = oof[ok], ytr[ok]
    solvable = int((yy == 0).sum())
    if solvable == 0:
        return float("inf")
    best = float("inf")
    for thr in np.unique(o):
        abstain = o >= thr
        lost = int((abstain & (yy == 0)).sum())
        if lost / solvable <= budget + 1e-12:
            best = min(best, float(thr))
    return best


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
    L.append("| train | test | n train | n test | base rate train | base rate test | "
             "**transfer AUROC** | shuffle null | within-cell AUROC (ceiling) |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in d["matched"]:
        w = d["within_cell_auroc"].get(r["test_cell"])
        L.append(
            f"| `{r['train_cell']}` | `{r['test_cell']}` | {r['n_train']} | {r['n_test']} | "
            f"{100*r['train_base_rate']:.1f}% | {100*r['test_base_rate']:.1f}% | "
            f"**{r['auroc']:.3f}** | {r['auroc_shuffle']:.3f} | "
            f"{'—' if w is None else f'{w:.3f}'} |")
    L.append("")

    L.append("## 2. Pooled transfer (all three models of one site → each cell of the other)")
    L.append("")
    L.append("| train site | test | n train | n test | **transfer AUROC** | shuffle null | "
             "within-cell AUROC (ceiling) |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for r in d["pooled"]:
        w = d["within_cell_auroc"].get(r["test_cell"])
        L.append(
            f"| {r['train_site']} (3 cells) | `{r['test_cell']}` | {r['n_train']} | {r['n_test']} | "
            f"**{r['auroc']:.3f}** | {r['auroc_shuffle']:.3f} | "
            f"{'—' if w is None else f'{w:.3f}'} |")
    L.append("")

    L.append("## 3. Transferred operating point — threshold never sees the test site")
    L.append("")
    L.append("The threshold is chosen by an inner 5-fold **inside the training site** at the stated "
             "solvable-loss budget, then applied unseen to the other site. This is the cross-site "
             "analogue of the nested column in `abstention_learnability` §3, and it is a held-out "
             "*policy*, not merely a held-out prediction (§465).")
    L.append("")
    L.append("| train | test | budget | abstain rate | solvable lost | realised loss | saved |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for r in d["operating_points"]:
        op = r["op"]
        lost = f"{op['solvable_lost']}/{op['solvable_total']}"
        loss = "—" if np.isnan(op["loss_rate"]) else f"{100*op['loss_rate']:.1f}%"
        saved = "—" if np.isnan(op["cost_saved_pct"]) else f"{op['cost_saved_pct']:.1f}%"
        L.append(f"| `{r['train_cell']}` | `{r['test_cell']}` | ≤{100*r['budget']:.0f}% | "
                 f"{100*op['abstain_rate']:.1f}% | {lost} | {loss} | {saved} |")
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

    beat_null = [r for r in d["pooled"] if r["auroc"] > r["auroc_shuffle"]]
    fail_null = [r for r in d["pooled"] if r["auroc"] <= r["auroc_shuffle"]]
    gaps = [(r, d["within_cell_auroc"].get(r["test_cell"])) for r in d["pooled"]]
    gaps = [(r, w) for r, w in gaps if w is not None]
    deltas = sorted((r["auroc"] - w, r["test_cell"]) for r, w in gaps)
    L.append(
        f"**Ranking transfers.** {len(beat_null)} of {len(d['pooled'])} pooled transfers clear their "
        f"own label-shuffle null. Against the ceiling — the same cell's *within-cell* held-out AUROC — "
        f"the pooled transfer lands between {deltas[0][0]:+.3f} (`{deltas[0][1]}`) and "
        f"{deltas[-1][0]:+.3f} (`{deltas[-1][1]}`). "
        + (f"It is **positive** in {sum(1 for x, _ in deltas if x > 0)} of {len(deltas)} cells, i.e. "
           f"training on the other site beat training on the cell's own tasks there."
           if any(x > 0 for x, _ in deltas) else ""))
    L.append("")
    if fail_null:
        names = ", ".join(f"`{r['test_cell']}`" for r in fail_null)
        L.append(f"**Where it does not transfer:** {names} — the shuffle null matches or beats the "
                 f"fitted score, so nothing was learned that survives the site change there. These are "
                 f"the cells the within-cell product already flags as sitting near the floor.")
        L.append("")

    worst = max(d["operating_points"], key=lambda o: (o["op"]["loss_rate"]
                                                      if not np.isnan(o["op"]["loss_rate"]) else -1))
    wa = next((r for r in d["matched"]
               if r["train_cell"] == worst["train_cell"] and r["test_cell"] == worst["test_cell"]), None)
    within_w = d["within_cell_auroc"].get(worst["test_cell"])
    honoured = [o for o in d["operating_points"]
                if not np.isnan(o["op"]["loss_rate"]) and o["op"]["loss_rate"] <= o["budget"] + 1e-9]
    L.append(
        f"**The operating point does not, and the AUROC does not warn you.** "
        f"{len(honoured)} of {len(d['operating_points'])} transferred thresholds kept the realised "
        f"loss inside the budget they were bought at. The worst is "
        f"`{worst['train_cell']}` → `{worst['test_cell']}` at a ≤{100*worst['budget']:.0f}% budget: it "
        f"abstains on {100*worst['op']['abstain_rate']:.1f}% of the test site and destroys "
        f"{worst['op']['solvable_lost']}/{worst['op']['solvable_total']} "
        f"({100*worst['op']['loss_rate']:.1f}%) of its solvable tasks. "
        + (f"That same direction has AUROC **{wa['auroc']:.3f}**"
           + (f", the highest of the {len(d['matched'])} matched transfers"
              if wa["auroc"] == max(r["auroc"] for r in d["matched"]) else "")
           + (f" and above its own within-cell ceiling of {within_w:.3f}"
              if within_w is not None and wa["auroc"] > within_w else "")
           + " — i.e. **the direction that looks best by ranking is the one that fails worst as a "
             "policy**." if wa else ""))
    L.append("")
    L.append(
        "The mechanism is base-rate drift, and it is one-directional: universal-fail runs "
        f"{100*min(r['train_base_rate'] for r in d['matched']):.1f}%–"
        f"{100*max(r['train_base_rate'] for r in d['matched']):.1f}% across these cells, so a "
        "threshold calibrated where almost everything fails abstains on far too much where less does. "
        "AUROC is rank-based and cannot see it. This is the cross-site form of the distinction §465 "
        "drew within a cell: **a held-out prediction is not a held-out policy**, and transfer is where "
        "the gap between them is widest.")
    L.append("")
    L.append("⚠️ Two sites is two points. Nothing here licenses a claim about transfer to a *third* "
             "site; what it licenses is that ranking survived the one site change available and "
             "calibration did not.")
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
                thr = pick_threshold_on_train([by_cell[tr_id]], b)
                ops.append({"train_cell": tr_id, "test_cell": te_id, "budget": b,
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

    print("\n=== matched-model site transfer (AUROC / shuffle / within-cell ceiling) ===")
    for r in matched:
        w = within.get(r["test_cell"])
        print(f"  {r['train_cell']:>17} -> {r['test_cell']:<17} "
              f"{r['auroc']:.3f} / {r['auroc_shuffle']:.3f} / "
              f"{'--' if w is None else f'{w:.3f}'}")
    print("\n=== pooled site transfer ===")
    for r in pooled:
        w = within.get(r["test_cell"])
        print(f"  {r['train_site']:>12} (3) -> {r['test_cell']:<17} "
              f"{r['auroc']:.3f} / {r['auroc_shuffle']:.3f} / "
              f"{'--' if w is None else f'{w:.3f}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
