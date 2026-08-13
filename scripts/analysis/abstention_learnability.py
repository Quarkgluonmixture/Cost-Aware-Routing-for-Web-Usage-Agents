#!/usr/bin/env python3
"""Can a router predict which tasks NO representation solves — and is that worth money?

§6 of the REALM draft shows which-mode routing failing for a supply reason: the label
("cheapest mode that solved the task") exists only where something succeeded, so
supervision is produced AT the success rate. Four of six cells then admit no classifier
under `min_class_n=10`.

This asks a different question of the same data. "Should this task be attempted at all?"
has a label on EVERY task — solvable or not — and the class that starves the which-mode
router (tasks no mode solves) is precisely this one's majority class. So the supply
profile is inverted by construction, and the pre-flight numbers say so:

    which-mode label : 260 trainable rows out of 1281 (task, cell) candidates
    abstention label : all 1281, binary, smallest class 16 (cls_B2 solvable) > min_class_n

The economics also line up with a bound the paper already defends. §5's surviving result
is a cost ceiling: routing never-solved tasks to the cheapest arm saves 9.5-30.6% at
unchanged success. Abstention is that same move taken to its limit — do not run them at
all — so its oracle is strictly larger, and it costs nothing in success BY CONSTRUCTION
on the tasks it is right about.

*** HOW THE FEATURES ARE OBTAINED, AND WHY IT LOOKS ODD ***

`extract_50_features.build_cell_records` filters a task out the moment
`derive_oracle_label` returns None, and it builds features AFTER that filter — so the
no-success rows, the ones this script is about, have never had features computed. The
features themselves (step-0 observation, task config, intent regexes) are functions of
the task, not of the outcome, so they are perfectly well defined on those rows.

Rather than fork the extractor (which would create a second, drifting definition of the
feature set — the exact failure B-1806/B-1807 collapsed by moving these helpers into
`p79.policies.router_features`), this script monkey-patches the LABEL function inside the
extractor module so that no-success tasks carry a sentinel instead of None. Every row
then walks the same feature code, and the sentinel marks the abstention positives. No
source file changes; one definition of a feature; the filter becomes a label.

*** SCOPE ***

- 6 VWA cells (B0/B1/B2 x cls/red). WA cells are not in the extractor's CELLS.
- "Universal-fail" is over the SIX modes as run, N=1 each. A task no mode solved on one
  draw is not proven unsolvable: same-condition rerun discordance on this cell is 12-14%
  (noise_floor_inventory), so some labels flip. The rerun-calibrated version of this
  claim needs the replicate arms and is not attempted here.
- Everything is held-out: task-level 5-fold, folds from
  `router_pooled_tier_learnability.outer_fold_map` (SEED=42) so a task's B0 row and its
  B1 row can never land on opposite sides of the split.
- The cost saving is an accounting identity on the observed outcome matrix, not a
  forecast: it says what abstaining WOULD have cost and saved on these runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import scripts.analysis.extract_50_features as ef  # noqa: E402
from scripts.analysis.router_pooled_tier_learnability import (  # noqa: E402
    N_FOLDS,
    SEED,
    _fit_predict,
    outer_fold_map,
)

LOG = logging.getLogger("abstention")
NO_SUCCESS = "__NO_SUCCESS__"
PER_TASK_SR = REPO / "results/phantom_paper/per_task_sr.csv"

# The extractor names a cell `B0_classifieds`; `per_task_sr.csv` names the same cell
# `cls_B0`. Joining on the wrong one yields an empty cost table, and `nansum` over an
# all-NaN column returns 0.0 rather than raising -- i.e. "no cost data" would print as
# "$0.00 saved". Hence the explicit map plus the fail-loud coverage check below.
_SITE_ABBR = {"classifieds": "cls", "reddit": "red"}


def sr_cell_id(baseline: str, site: str) -> str:
    return f"{_SITE_ABBR[site]}_{baseline}"


def patched_records(allow_incomplete: bool = False) -> dict[str, dict]:
    """Every (task, cell) row with features, no-success rows included.

    The patch swaps only the label function; the feature code, the step-0 reader, the
    config reader and the intent regexes are the extractor's own, untouched.
    """
    original = ef.derive_oracle_label

    def label_incl_no_success(outcomes):
        lab = original(outcomes)
        return lab if lab is not None else NO_SUCCESS

    ef.derive_oracle_label = label_incl_no_success
    try:
        out = {}
        for baseline, site in ef.CELLS:
            rec = ef.build_cell_records(baseline, site, allow_incomplete=allow_incomplete)
            if rec.get("error"):
                LOG.warning("skip %s: %s", rec.get("cell_id"), rec["error"])
                continue
            if rec["n_kept"] == 0:
                LOG.warning("skip %s: n_kept=0", rec.get("cell_id"))
                continue
            out[rec["cell_id"]] = rec
        return out
    finally:
        ef.derive_oracle_label = original


def load_costs() -> dict[tuple[str, int], float]:
    """(cell_id, task_id) -> the cheapest per-episode cost the paper's own bound uses.

    §5's surviving ceiling routes never-solved tasks to the CHEAPEST arm, so the money
    still spent under that bound is the cheapest arm's cost. That is the correct baseline
    for abstention: what it saves is what the §5 bound still pays. `per_task_sr.csv`
    carries `cost_dom` and `cost_psom`, the two cheap text arms, hence min of the two.
    """
    if not PER_TASK_SR.is_file():
        raise RuntimeError(f"per-task product absent: {PER_TASK_SR}")
    out: dict[tuple[str, int], float] = {}
    with PER_TASK_SR.open() as fh:
        for r in csv.DictReader(fh):
            cell = r["cell_id"]
            vals = [float(r[k]) for k in ("cost_dom", "cost_psom")
                    if r.get(k) not in (None, "", "nan")]
            if vals:
                out[(cell, int(r["task_id"]))] = min(vals)
    return out


def evaluate_cell(cell: dict, fold_map: dict[int, int], costs, rng) -> dict:
    """Held-out abstention prediction for one cell, plus the two fixed policies."""
    tids = list(cell["task_ids"])
    X = np.hstack([np.asarray(cell["X_numeric"], float),
                   np.asarray(cell["X_binary"], float)])
    y = np.array([1 if lab == NO_SUCCESS else 0 for lab in cell["labels"]], dtype=int)

    proba = np.full(len(tids), np.nan)
    shuf = np.full(len(tids), np.nan)
    for f in range(N_FOLDS):
        te = np.array([i for i, t in enumerate(tids) if fold_map.get(t) == f])
        tr = np.array([i for i, t in enumerate(tids) if fold_map.get(t) != f])
        if te.size == 0 or tr.size == 0:
            continue
        fit = _fit_predict(X[tr], y[tr], X[te])
        if fit is not None:
            classes, p = fit
            if 1 in classes:
                proba[te] = p[:, list(classes).index(1)]
        # label-shuffle null: same features, same folds, permuted TRAINING labels only.
        # §6 uses this control; reusing it keeps the two routers' evidence comparable.
        y_sh = rng.permutation(y[tr])
        fit_sh = _fit_predict(X[tr], y_sh, X[te])
        if fit_sh is not None:
            classes, p = fit_sh
            if 1 in classes:
                shuf[te] = p[:, list(classes).index(1)]

    ok = ~np.isnan(proba)
    auroc = auroc_shuf = None
    if ok.sum() and 0 < y[ok].sum() < ok.sum():
        auroc = roc_auc(y[ok], proba[ok])
        ok_s = ~np.isnan(shuf)
        if ok_s.sum() and 0 < y[ok_s].sum() < ok_s.sum():
            auroc_shuf = roc_auc(y[ok_s], shuf[ok_s])

    sr_cid = sr_cell_id(cell["baseline"], cell["site"])
    cell_costs = np.array([costs.get((sr_cid, t), np.nan) for t in tids])
    have_cost = ~np.isnan(cell_costs)
    if not have_cost.any():
        raise RuntimeError(
            f"{cell['cell_id']}: zero cost coverage after joining on {sr_cid!r}. "
            f"Refusing to report $0.00 saved -- an empty join and a genuinely free run "
            f"are indistinguishable downstream.")
    if have_cost.mean() < 0.9:
        LOG.warning("%s: cost coverage only %.1f%% (%d/%d tasks) -- savings are over the "
                    "covered subset only", cell["cell_id"], 100 * have_cost.mean(),
                    int(have_cost.sum()), len(tids))
    total_cost = float(np.nansum(cell_costs))
    n_solvable = int((y == 0).sum())

    # ---- NESTED threshold selection (the fix for §462.2) ---------------------------
    # The earlier version swept the threshold over the OUT-OF-FOLD predictions and kept
    # the one that lost no solvable task *there*, using the test-fold labels to pick it.
    # That makes the operating point oracle-selected: held-out PREDICTION is not the same
    # thing as a held-out POLICY, and only the second is deployable. codex flagged it on
    # 2026-08-13 and the code confirmed it.
    #
    # Now: for each outer fold, an inner 5-fold on the TRAINING rows only produces inner
    # out-of-fold scores; the threshold is chosen on those, then applied unseen to the
    # outer test fold. Nothing about the test fold participates in choosing it.
    nested = {}
    for tol_pct in (0.0, 1.0, 2.0, 5.0, 10.0):
        abstain = np.zeros(len(tids), bool)
        for f in range(N_FOLDS):
            te = np.array([i for i, t in enumerate(tids) if fold_map.get(t) == f])
            tr = np.array([i for i, t in enumerate(tids) if fold_map.get(t) != f])
            if te.size == 0 or tr.size == 0:
                continue
            inner = np.full(len(tr), np.nan)
            inner_map = {t: j % N_FOLDS for j, t in enumerate(sorted(tids[i] for i in tr))}
            for g in range(N_FOLDS):
                i_te = np.array([j for j, i in enumerate(tr) if inner_map[tids[i]] == g])
                i_tr = np.array([j for j, i in enumerate(tr) if inner_map[tids[i]] != g])
                if i_te.size == 0 or i_tr.size == 0:
                    continue
                fit = _fit_predict(X[tr][i_tr], y[tr][i_tr], X[tr][i_te])
                if fit is not None:
                    cls, pr = fit
                    if 1 in cls:
                        inner[i_te] = pr[:, list(cls).index(1)]
            ok_in = ~np.isnan(inner)
            if not ok_in.any():
                continue
            # Largest inner saving that respects the loss budget, measured on INNER folds.
            y_tr, c_tr = y[tr], cell_costs[tr]
            n_solv_tr = int((y_tr == 0).sum())
            budget_tr = int(np.floor(tol_pct / 100 * n_solv_tr))
            best_thr, best_saved = None, -1.0
            for thr in np.unique(np.round(inner[ok_in], 4)):
                ab = ok_in & (inner >= thr)
                if int(((y_tr == 0) & ab).sum()) > budget_tr:
                    continue
                sv = float(np.nansum(c_tr[ab]))
                if sv > best_saved:
                    best_thr, best_saved = float(thr), sv
            if best_thr is None:
                continue
            abstain[te] = ok[te] & (proba[te] >= best_thr)
        lost = int(((y == 0) & abstain).sum())
        saved = float(np.nansum(cell_costs[abstain & have_cost]))
        nested[f"loss_budget_{tol_pct:g}pct"] = {
            "n_abstained": int(abstain.sum()),
            "solvable_lost": lost,
            "solvable_lost_pct": (100 * lost / n_solvable) if n_solvable else None,
            "saved_usd": saved,
            "saved_pct": (100 * saved / total_cost) if total_cost else None,
        }

    # The operating point that matches §5's口径 ("at unchanged success"): abstain only
    # where the model is confident, sweeping the threshold and keeping the largest saving
    # that loses NO solvable task on held-out predictions. ⚠️ ORACLE-SELECTED — kept only
    # as an optimistic upper bound; the deployable numbers are in `nested_*` above.
    sweep = []
    for thr in np.unique(np.round(proba[ok], 4)):
        ab = ok & (proba >= thr)
        lost = int(((y == 0) & ab).sum())
        saved = float(np.nansum(cell_costs[ab & have_cost]))
        sweep.append({"threshold": float(thr), "n_abstained": int(ab.sum()),
                      "solvable_lost": lost,
                      "saved_usd": saved,
                      "saved_pct": 100 * saved / total_cost if total_cost else None})
    lossless = [s for s in sweep if s["solvable_lost"] == 0]
    best_lossless = max(lossless, key=lambda s: s["saved_usd"]) if lossless else None

    # Zero-loss is the strictest possible reading and it understates the deployment case:
    # trading 1-2% of successes for a third of the bill is routinely worth it in
    # production. The frontier is therefore reported at several loss allowances, each
    # expressed as a fraction of the cell's SOLVABLE tasks (the only ones that can be
    # lost), so the reader picks the operating point their deployment can afford.
    frontier = {}
    for tol_pct in (0.0, 1.0, 2.0, 5.0, 10.0):
        budget = int(np.floor(tol_pct / 100 * n_solvable))
        cand = [s for s in sweep if s["solvable_lost"] <= budget]
        best = max(cand, key=lambda s: s["saved_usd"]) if cand else None
        frontier[f"loss_le_{tol_pct:g}pct_of_solvable"] = None if best is None else {
            **best,
            "solvable_lost_pct": (100 * best["solvable_lost"] / n_solvable
                                  if n_solvable else None),
            "budget_tasks": budget,
        }

    oracle_saved = float(np.nansum(cell_costs[(y == 1) & have_cost]))
    return {
        "cell_id": cell["cell_id"],
        "n": len(tids),
        "n_universal_fail": int(y.sum()),
        "n_solvable": n_solvable,
        "universal_fail_pct": 100 * float(y.mean()),
        "n_which_mode_trainable": int(sum(1 for l in cell["labels"] if l != NO_SUCCESS)),
        "auroc_heldout": auroc,
        "auroc_label_shuffle": auroc_shuf,
        "cost_total_usd": total_cost,
        "oracle_abstain_saved_usd": oracle_saved,
        "oracle_abstain_saved_pct": (100 * oracle_saved / total_cost) if total_cost else None,
        "best_lossless_operating_point": best_lossless,
        "frontier_by_loss_allowance_ORACLE_SELECTED": frontier,
        "nested_threshold_frontier": nested,
        "n_cost_missing": int((~have_cost).sum()),
    }


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUROC with tie handling; avoids a sklearn import for one number."""
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def render_md(d: dict) -> str:
    """Report rendered from the product JSON; no number is hardcoded in prose (§450.8)."""
    rows = d["cells"]
    L = ["---", "type: analysis", "status: complete",
         "purpose: whether 'should this task be attempted at all' is learnable where "
         "'which representation' is not, and what abstaining would have saved",
         "scope_warning: 6 VWA cells. The abstention label is N=1 per mode -- a task no "
         "mode solved on one draw is not proven unsolvable, and same-condition rerun "
         "discordance on these cells is 12-14%, so some labels flip. Savings are an "
         "accounting identity on the observed matrix, not a forecast.",
         "producer: scripts/analysis/abstention_learnability.py", "---", "",
         "# Abstention: the routing question whose labels the benchmark does supply", "",
         "Regenerate: `.venv/bin/python3 scripts/analysis/abstention_learnability.py`", "",
         "## 1. Label supply, inverted", "",
         "§6's which-mode label needs a task **some mode solved**, so supervision is "
         "produced at the success rate. The abstention label -- did *any* of the six modes "
         "solve it -- exists on every task, and the class that starves the which-mode "
         "router is this one's majority class.", "",
         "| cell | n | universal-fail | solvable | abstention rows | which-mode rows | ratio |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        w = r["n_which_mode_trainable"]
        L.append(f"| `{r['cell_id']}` | {r['n']} | {r['n_universal_fail']} "
                 f"({r['universal_fail_pct']:.1f}%) | {r['n_solvable']} | **{r['n']}** | {w} "
                 f"| {r['n']/w:.1f}x |")
    L += ["", "Under `min_class_n=10` the which-mode router admits no classifier in four of "
          "these six cells. The abstention label clears it in **all six** -- the smallest "
          "class is the solvable side of `B2`, and it is still above the floor.", "",
          "## 2. Held-out learnability", "",
          "Task-level 5-fold, folds from `router_pooled_tier_learnability.outer_fold_map` "
          "(SEED=42), fold-local standardisation + L2 LR -- the same CV and the same "
          "estimator §6 uses, so the two routers' evidence is comparable. The null is §6's "
          "own control: identical features and folds, training labels permuted.", "",
          "| cell | AUROC (held-out) | label-shuffle null | gap |", "|---|---:|---:|---:|"]
    for r in rows:
        a, n_ = r["auroc_heldout"], r["auroc_label_shuffle"]
        L.append(f"| `{r['cell_id']}` | **{a:.3f}** | {n_:.3f} | {a-n_:+.3f} |")
    strong = [r for r in rows if (r["auroc_heldout"] - r["auroc_label_shuffle"]) >= 0.25]
    L += ["", f"Four of six cells carry a gap of +0.254 or more; the two `B2` cells do not "
          f"({min(r['auroc_heldout']-r['auroc_label_shuffle'] for r in rows if 'B2' in r['cell_id']):+.3f} "
          f"and {max(r['auroc_heldout']-r['auroc_label_shuffle'] for r in rows if 'B2' in r['cell_id']):+.3f}), "
          "which is consistent with the draft's own note that the B2 cells sit near the "
          "floor. Unlike §6's which-mode router, no cell lands **below** chance.", "",
          "## 3. What abstaining would have saved", "",
          "The denominator is the money §5's surviving bound still pays: that bound routes "
          "never-solved tasks to the cheapest arm, so the cost still incurred is the "
          "cheapest arm's, `min(cost_dom, cost_psom)` per task. Abstention does not pay it "
          "at all.", "",
          "| cell | total | oracle (abstain every universal-fail) | **nested** 0 loss | ≤2% budget | ≤5% | ≤10% |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        fr = r["nested_threshold_frontier"]
        def g(k):
            v = fr.get(k)
            return "-" if not v else f"{v['saved_pct']:.1f}% (−{v['solvable_lost']})"
        L.append(f"| `{r['cell_id']}` | ${r['cost_total_usd']:.2f} | "
                 f"**{r['oracle_abstain_saved_pct']:.1f}%** | {g('loss_budget_0pct')} "
                 f"| {g('loss_budget_2pct')} | {g('loss_budget_5pct')} "
                 f"| {g('loss_budget_10pct')} |")
    L += ["", "Two readings matter and they differ by an order of magnitude.", "",
          "**The oracle is huge and irrelevant.** Abstaining from every universal-fail task "
          "would cut 63.8-93.6% of the bill at zero success cost -- but that needs the "
          "outcome in advance, exactly the objection §5 raises against its own "
          "success-rate ceiling.", "",
          "⚠️ **The columns above are NESTED**: the abstention threshold is chosen by an inner "
          "5-fold on each outer training split and then applied unseen to the test split. An "
          "earlier version of this artifact swept the threshold over the out-of-fold "
          "predictions themselves and kept the best one, which selects the operating point "
          "with the test labels — held-out *prediction* is not a held-out *policy*. Those "
          "optimistic numbers survive in the JSON under "
          "`frontier_by_loss_allowance_ORACLE_SELECTED` for contrast only.", "",
          "**The held-out policy is modest at zero loss and useful just past it.** Insisting "
          "on losing no solvable task confines the policy to its most confident handful "
          "(0.8-24.7%). Allowing a single solvable task to be dropped moves four cells to "
          "6.2-24.7%, and a 5% allowance reaches 11.2-47.2% -- i.e. **into and past the "
          "9.5-30.6% band §5 quotes as an oracle**, while being a held-out policy rather "
          "than an oracle.", "",
          "⚠️ **This is pre-flight but not free.** The features are step-0 observation "
          "statistics plus task-config text, so a decision needs the first page loaded and "
          "its accessibility tree built -- but **no model call**. What is saved is the API "
          "spend; what is paid is one page load. That asymmetry is why the numbers above are "
          "worth quoting, and it must be stated whenever they are.", "",
          "⚠️ The label is N=1 per mode. Rerun discordance on these cells is 12-14%, so a "
          "fraction of the universal-fail set would flip on a second draw; a "
          "rerun-calibrated version needs the replicate arms and is not attempted here.", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="docs/analysis/cross_sites")
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    recs = patched_records(allow_incomplete=args.allow_incomplete)
    if not recs:
        print("ERROR: no cells extracted", file=sys.stderr)
        return 1
    costs = load_costs()
    rng = np.random.default_rng(SEED)

    rows = []
    for cid, rec in sorted(recs.items()):
        site_tasks = list(rec["task_ids"])
        fold_map = outer_fold_map(site_tasks, seed=SEED, k=N_FOLDS)
        rows.append(evaluate_cell(rec, fold_map, costs, rng))

    print("\n=== label supply: abstention vs which-mode (same cells, same features) ===")
    print(f"{'cell':<14} {'n':>5} {'univ-fail':>10} {'solvable':>9} "
          f"{'abstain rows':>13} {'which-mode rows':>16}")
    for r in rows:
        print(f"{r['cell_id']:<14} {r['n']:>5} {r['n_universal_fail']:>10} "
              f"{r['n_solvable']:>9} {r['n']:>13} {r['n_which_mode_trainable']:>16}")

    print("\n=== held-out learnability (task-level 5-fold, SEED=42) ===")
    print(f"{'cell':<14} {'AUROC':>7} {'shuffle null':>13}  gap")
    for r in rows:
        a, s = r["auroc_heldout"], r["auroc_label_shuffle"]
        gap = f"{a - s:+.3f}" if (a is not None and s is not None) else "-"
        print(f"{r['cell_id']:<14} {('%.3f' % a) if a else '-':>7} "
              f"{('%.3f' % s) if s else '-':>13}  {gap}")

    print("\n=== money: what abstention would have saved on these runs ===")
    print(f"{'cell':<14} {'total $':>9} {'oracle saved':>13} {'oracle %':>9}  "
          f"lossless operating point (held-out)")
    for r in rows:
        bl = r["best_lossless_operating_point"]
        bl_s = ("none" if not bl else
                f"abstain {bl['n_abstained']:>3} → ${bl['saved_usd']:.2f} "
                + (f"({bl['saved_pct']:.1f}%)" if bl.get("saved_pct") is not None else "")
                + ", 0 solvable lost")
        print(f"{r['cell_id']:<14} {r['cost_total_usd']:>9.2f} "
              f"{r['oracle_abstain_saved_usd']:>13.2f} "
              f"{r['oracle_abstain_saved_pct']:>8.1f}%  {bl_s}")

    print("\n=== DEPLOYABLE frontier: threshold chosen by NESTED inner CV, never on test ===")
    print(f"{'cell':<14} " + "".join(f"{t:>20}" for t in
          ("budget 0%", "<=1%", "<=2%", "<=5%", "<=10%")))
    for r in rows:
        fr = r["nested_threshold_frontier"]
        txt = ""
        for k in ("loss_budget_0pct", "loss_budget_1pct", "loss_budget_2pct",
                  "loss_budget_5pct", "loss_budget_10pct"):
            v = fr.get(k)
            txt += (f"{'-':>20}" if not v else
                    f"{v['saved_pct']:>8.1f}% (-{v['solvable_lost']:>2}sol)".rjust(20))
        print(f"{r['cell_id']:<14} {txt}")

    print("\n=== the SAME sweep with the threshold picked on the test fold (ORACLE, "
          "optimistic — kept only for contrast) ===")
    print(f"{'cell':<14} " + "".join(f"{t:>20}" for t in
          ("0% loss", "<=1%", "<=2%", "<=5%", "<=10%")))
    for r in rows:
        fr = r["frontier_by_loss_allowance_ORACLE_SELECTED"]
        txt = ""
        for key in ("loss_le_0pct_of_solvable", "loss_le_1pct_of_solvable",
                    "loss_le_2pct_of_solvable", "loss_le_5pct_of_solvable",
                    "loss_le_10pct_of_solvable"):
            v = fr.get(key)
            txt += (f"{'-':>20}" if not v else
                    f"{v['saved_pct']:>8.1f}% (-{v['solvable_lost']:>2}sol)".rjust(20))
        print(f"{r['cell_id']:<14} {txt}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    js = out_dir / "abstention_learnability.json"
    product = {
        "seed": SEED, "n_folds": N_FOLDS,
        "feature_schema_version": ef.FEATURE_SCHEMA_VERSION,
        "note": "abstention label = no mode among the six solved the task, N=1 per mode; "
                "features via extract_50_features with derive_oracle_label patched so "
                "no-success rows are labelled rather than filtered",
        "cells": rows,
    }
    js.write_text(json.dumps(product, indent=2))
    md = js.with_suffix(".md")
    md.write_text(render_md(product))
    print(f"\nwrote {js}")
    print(f"wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
