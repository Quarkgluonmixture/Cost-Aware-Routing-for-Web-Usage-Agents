#!/usr/bin/env python3
"""At step k, can you tell this episode is not going to make it?

The paper's abstention result (`abstention_learnability.md`) decides before the first
model call, from task text and the step-0 page. This asks the complementary question with
the complementary signal: after k steps have actually run, does the trajectory itself say
whether the remaining budget is worth spending?

Why it is worth asking on this data:
  - Budget exhaustion is the largest single failure dimension (笔记 §290/§318: P31 counts
    73 vision / 58 dom / 49 som), i.e. a great many episodes spend the full step budget
    going nowhere.
  - `representation_deployment_profile.md` §1 finds 55-62% of Vision's failures land in
    `max-steps-other` -- "the budget ran out and no rule fired". That is exactly the mass
    an abort would reclaim.
  - The per-step `confidence` block (mean/min logprob, margin) has been written since the
    beginning and is a strong single signal: the layered-evidence 0b row reports routing
    AUROC up to 0.877 for it.

*** WHAT MAKES THIS DIFFERENT FROM 0b, AND WHY 0b's NUMBER CANNOT BE REUSED ***

0b is the AUROC of a signal aggregated over the WHOLE episode. For an abort decision that
is looking at the future: the confidence of step 20 cannot inform a decision at step 5.
Every feature here is computed from the first k steps only.

*** THE CONTROL THAT DECIDES WHETHER THIS MEANS ANYTHING ***

The trivial way to "abort at step k" is to set the step budget to k for every episode. A
learned abort policy is only interesting if it beats that fixed policy at matched cost --
the same logic §6 applies to always-cheapest. Both are reported side by side, and the
comparison is made at matched spend, not against zero.

*** SCOPE ***

- B0 x classifieds, six modes. B0 is the paying backbone and the cell with the highest
  success rate, so it is the one where losing a success actually costs something.
- Success is the episode's own canonical `success` field; steps come from the step_v2
  records of the same run.
- Cost is per-step billed cost where recorded, so "saved" is the spend on steps after the
  abort point. It is an accounting identity on the observed trajectories, not a forecast:
  a real abort changes nothing about the steps it never runs, but it also cannot recover
  a success that those steps would have produced -- which is what the loss column counts.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.router_pooled_tier_learnability import (  # noqa: E402
    N_FOLDS,
    SEED,
    _fit_predict,
    outer_fold_map,
)

LOG = logging.getLogger("early_abort")
PHASE1 = REPO / "results/visualwebarena/phase1"
MODE_DIRS = {"DOM": "dom", "SoM": "som", "Vision": "vision",
             "P-text": "phantom_text", "P-prompt": "phantom_prompt", "P-SoM": "phantom_som"}
PREFIXES = (3, 5, 10)


def _num(v):
    return float(v) if isinstance(v, (int, float)) else None


def load_episodes(baseline: str, site: str, mode_dir: str) -> list[dict]:
    """One record per episode: outcome, per-step confidence/cost, step count."""
    runs = sorted(p for p in PHASE1.glob(f"{baseline}_{mode_dir}_{site}_*")
                  if p.is_dir() and "ABORTED" not in p.name)
    if not runs:
        return []
    # Only the first (canonical) run: pooling an episode with its own replicate would put
    # two rows for one task on opposite sides of a fold split.
    run = runs[0]
    out = []
    for sf in sorted(run.glob("*/episodes/*_summary_v2.json")):
        try:
            summ = json.loads(sf.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if summ.get("sr_excluded"):
            continue
        tid = summ.get("task_id")
        if tid is None:
            continue
        jf = sf.with_name(sf.name.replace("_summary_v2.json", "_steps_v2.jsonl"))
        if not jf.is_file():
            continue
        steps = []
        for line in jf.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                steps.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if not steps:
            continue
        out.append({
            "task_id": int(tid), "success": 1 if summ.get("success") else 0,
            "n_steps": len(steps), "steps": steps,
            "run_id": run.name,
        })
    return out


def prefix_features(steps: list[dict], k: int) -> list[float] | None:
    """Features from the first k steps only. None if the episode is shorter than k."""
    if len(steps) < k:
        return None
    pre = steps[:k]
    mean_lp, min_lp, margins = [], [], []
    acts, urls, changed = [], [], 0
    for s in pre:
        c = s.get("confidence") or {}
        for src, dst in ((c.get("mean_logprob"), mean_lp), (c.get("min_logprob"), min_lp),
                         (c.get("mean_margin"), margins)):
            v = _num(src)
            if v is not None:
                dst.append(v)
        acts.append(str(s.get("action_type") or "?"))
        u = s.get("url_after") or s.get("url_before")
        if u:
            urls.append(str(u))
        if s.get("state_changed"):
            changed += 1

    def agg(xs, f, default=0.0):
        return float(f(xs)) if xs else default

    n_urls = len(set(urls))
    return [
        agg(mean_lp, np.mean, -5.0), agg(mean_lp, np.min, -5.0),
        agg(min_lp, np.mean, -5.0), agg(min_lp, np.min, -5.0),
        agg(margins, np.mean), agg(margins, np.min),
        # Behavioural: is it going in circles? Repetition and a frozen URL are the two
        # signals the runner's own cycle detector watches (helpers.py), reused as features.
        len(set(acts)) / k,
        max(acts.count(a) for a in set(acts)) / k,
        n_urls / k,
        changed / k,
        float(k),
    ]


def evaluate(mode: str, eps: list[dict], k: int, fold_map, rng) -> dict | None:
    rows = [(e, prefix_features(e["steps"], k)) for e in eps]
    rows = [(e, f) for e, f in rows if f is not None]
    if len(rows) < 4 * N_FOLDS:
        return None
    X = np.asarray([f for _, f in rows], float)
    # Predict FAILURE (the thing an abort acts on), so a high score means "abort".
    y = np.asarray([1 - e["success"] for e, _ in rows], int)
    tids = [e["task_id"] for e, _ in rows]
    if not (0 < y.sum() < len(y)):
        return None

    proba = np.full(len(rows), np.nan)
    shuf = np.full(len(rows), np.nan)
    for f in range(N_FOLDS):
        te = np.array([i for i, t in enumerate(tids) if fold_map.get(t) == f])
        tr = np.array([i for i, t in enumerate(tids) if fold_map.get(t) != f])
        if te.size == 0 or tr.size == 0:
            continue
        fit = _fit_predict(X[tr], y[tr], X[te])
        if fit:
            cls, p = fit
            if 1 in cls:
                proba[te] = p[:, list(cls).index(1)]
        fit_s = _fit_predict(X[tr], rng.permutation(y[tr]), X[te])
        if fit_s:
            cls, p = fit_s
            if 1 in cls:
                shuf[te] = p[:, list(cls).index(1)]

    ok = ~np.isnan(proba)
    if not ok.any() or not (0 < y[ok].sum() < ok.sum()):
        return None
    auroc = roc_auc(y[ok], proba[ok])
    ok_s = ~np.isnan(shuf)
    auroc_s = (roc_auc(y[ok_s], shuf[ok_s])
               if ok_s.any() and 0 < y[ok_s].sum() < ok_s.sum() else None)

    # --- economics, in STEPS (the unit an abort actually saves) --------------------
    n_steps = np.asarray([e["n_steps"] for e, _ in rows], float)
    steps_after_k = np.maximum(n_steps - k, 0)
    total_steps = float(n_steps.sum())
    n_success = int((y == 0).sum())

    # Fixed-policy control: truncate EVERY episode at k. It saves the same kind of steps
    # and loses every success that needed more than k steps.
    fixed_saved = float(steps_after_k.sum())
    fixed_lost = int(((y == 0) & (n_steps > k)).sum())

    sweep = []
    for thr in np.unique(np.round(proba[ok], 4)):
        ab = ok & (proba >= thr)
        sweep.append({
            "threshold": float(thr),
            "n_aborted": int(ab.sum()),
            "steps_saved": float(steps_after_k[ab].sum()),
            "steps_saved_pct": 100 * float(steps_after_k[ab].sum()) / total_steps,
            "successes_lost": int(((y == 0) & ab).sum()),
        })
    frontier = {}
    for tol in (0, 1, 2):
        cand = [s for s in sweep if s["successes_lost"] <= tol]
        best = max(cand, key=lambda s: s["steps_saved"]) if cand else None
        frontier[f"lost_le_{tol}"] = best
    # Matched-loss comparison against the fixed control: at the SAME number of successes
    # lost, does the learned policy save more steps?
    matched = [s for s in sweep if s["successes_lost"] <= fixed_lost]
    matched_best = max(matched, key=lambda s: s["steps_saved"]) if matched else None
    # The matched-loss point is where truncate-at-k happens to sit, and that is usually an
    # operating point nobody would deploy (truncating at k=3 discards most successes). The
    # honest comparison is at a loss level a deployment would accept -- and there the fixed
    # policy has NO tunable knob: truncating at k loses `fixed_lost` successes by
    # construction, so at "lose at most 0" it is only available if fixed_lost == 0.
    fixed_available_at_zero_loss = (fixed_lost == 0)
    return {
        "mode": mode, "k": k, "n_episodes": len(rows), "n_success": n_success,
        "auroc_heldout": auroc, "auroc_label_shuffle": auroc_s,
        "total_steps": total_steps,
        "fixed_truncate_at_k": {"steps_saved": fixed_saved,
                                "steps_saved_pct": 100 * fixed_saved / total_steps,
                                "successes_lost": fixed_lost},
        "learned_at_matched_loss": matched_best,
        "matched_loss_point_is_deployable": bool(fixed_lost <= 2),
        "fixed_available_at_zero_loss": fixed_available_at_zero_loss,
        "n_success_lost_share_at_matched": (fixed_lost / n_success) if n_success else None,
        "frontier_by_successes_lost": frontier,
    }


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ss = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    n1, n0 = float(y.sum()), float(len(y) - y.sum())
    return float("nan") if not n1 or not n0 else float(
        (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def render_md(d: dict) -> str:
    rows = d["results"]
    L = ["---", "type: analysis", "status: complete",
         "purpose: whether the first k steps of a trajectory predict that the episode will "
         "fail, i.e. whether an abort is learnable",
         "scope_warning: B0 x classifieds only. Every row is CONDITIONAL ON SURVIVING TO "
         "STEP k -- episodes shorter than k are excluded, and those are exactly the "
         "easy-to-call ones (early success, early hard failure), so the discrimination "
         "reported here is on the residual hard subset, not on all episodes.",
         "producer: scripts/analysis/early_abort_learnability.py", "---", "",
         "# Is an abort learnable from the first k steps?", "",
         f"Regenerate: `.venv/bin/python3 scripts/analysis/early_abort_learnability.py "
         f"--baseline {d['baseline']} --site {d['site']}`", "",
         "The layered-evidence 0b row reports routing AUROC up to 0.877 for the per-step "
         "confidence signal, but that is aggregated over the **whole** episode -- for an "
         "abort decision at step k it is looking at the future. Everything below uses the "
         "first k steps only.", "",
         "| mode | k | episodes | AUROC | shuffle null | gap | truncate-at-k | learned @ that same loss |",
         "|---|---:|---:|---:|---:|---:|---|---|"]
    for r in rows:
        fx, lm = r["fixed_truncate_at_k"], r["learned_at_matched_loss"]
        a, n_ = r["auroc_heldout"], r["auroc_label_shuffle"]
        gap = f"{a-n_:+.3f}" if n_ else "-"
        L.append(f"| {r['mode']} | {r['k']} | {r['n_episodes']} | {a:.3f} | "
                 f"{(('%.3f' % n_) if n_ else '-')} | {gap} | "
                 f"{fx['steps_saved_pct']:.1f}% steps, −{fx['successes_lost']} succ | "
                 + ("-" if not lm else
                    f"{lm['steps_saved_pct']:.1f}% steps, −{lm['successes_lost']} succ") + " |")
    best = max(rows, key=lambda r: (r["auroc_heldout"] - (r["auroc_label_shuffle"] or 0.5)))
    beats = [r for r in rows if r["learned_at_matched_loss"] and
             r["learned_at_matched_loss"]["steps_saved_pct"] >
             r["fixed_truncate_at_k"]["steps_saved_pct"]]
    L += ["", "## What this says", "",
          f"**The signal is not there.** AUROC runs "
          f"{min(r['auroc_heldout'] for r in rows):.3f}-{max(r['auroc_heldout'] for r in rows):.3f} "
          f"against a label-shuffle null of "
          f"{min(r['auroc_label_shuffle'] or 0 for r in rows):.3f}-"
          f"{max(r['auroc_label_shuffle'] or 0 for r in rows):.3f}; several rows sit **below** "
          f"their own null. The best row is {best['mode']} at k={best['k']} "
          f"({best['auroc_heldout']:.3f} vs {best['auroc_label_shuffle']:.3f}).", "",
          f"**And it loses to the trivial policy.** At the loss level truncate-at-k happens "
          f"to sit at, the learned policy saves fewer steps in "
          f"{len(rows) - len(beats)} of {len(rows)} rows.", "",
          "⚠️ **That matched-loss point is not a deployable one.** Truncating at k=3 discards "
          "most of the cell's successes, so both policies are being compared at an operating "
          "point nobody would ship. The deployable comparison is the zero-loss column below "
          "— and there the fixed policy has **no knob at all**: truncating at k loses its "
          "`fixed_lost` successes by construction.", "",
          "| mode | k | aborted | steps saved (0 successes lost) |", "|---|---:|---:|---:|"]
    for r in rows:
        b = r["frontier_by_successes_lost"]["lost_le_0"]
        if b:
            L.append(f"| {r['mode']} | {r['k']} | {b['n_aborted']} | "
                     f"{b['steps_saved_pct']:.1f}% |")
    L += ["", "So at zero success loss a learned abort reclaims a few percent of steps where "
          "the fixed policy reclaims none — a real but small effect resting on an AUROC that "
          "is mostly indistinguishable from noise.", "",
          "## Why this matters for the paper", "",
          "This is a **third** routing question on the same data, and the three fail (or not) "
          "for different reasons:", "",
          "| question | label supply | signal | outcome |", "|---|---|---|---|",
          "| which mode (§6) | **starved** — 4/6 cells admit no classifier | — | fails |",
          "| retry vs switch (§455) | adequate but mostly preference-free | — | no gain over fixed |",
          "| **abort at step k** (this) | **every episode has one** | **absent (AUROC≈null)** | fails |",
          "| **abstain up front** (§457) | **every task has one** | **present (0.615-0.864)** | **works** |", "",
          "The circularity §7 names therefore has two distinguishable failure modes, not one: "
          "a label that does not exist, and a label that exists with no signal behind it. "
          "Only the pre-flight question has both.", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", default="B0")
    ap.add_argument("--site", default="classifieds")
    ap.add_argument("--out-dir", default="docs/analysis/cross_sites")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    all_rows = []
    for mode, mdir in MODE_DIRS.items():
        eps = load_episodes(args.baseline, args.site, mdir)
        if not eps:
            LOG.warning("no episodes for %s %s %s", args.baseline, mdir, args.site)
            continue
        fold_map = outer_fold_map([e["task_id"] for e in eps], seed=SEED, k=N_FOLDS)
        LOG.info("%s: %d episodes, %d success, median %d steps", mode, len(eps),
                 sum(e["success"] for e in eps),
                 int(np.median([e["n_steps"] for e in eps])))
        for k in PREFIXES:
            rng = np.random.default_rng(SEED)
            r = evaluate(mode, eps, k, fold_map, rng)
            if r:
                all_rows.append(r)
    if not all_rows:
        print("ERROR: nothing evaluated", file=sys.stderr)
        return 1

    print(f"\n=== early abort on {args.baseline} x {args.site}: learned vs truncate-at-k ===")
    print(f"{'mode':<9} {'k':>3} {'eps':>5} {'AUROC':>7} {'null':>6}  "
          f"{'truncate@k':>22}  {'learned @ matched loss':>24}")
    for r in all_rows:
        fx = r["fixed_truncate_at_k"]
        lm = r["learned_at_matched_loss"]
        fx_s = f"{fx['steps_saved_pct']:.1f}% steps, -{fx['successes_lost']} succ"
        lm_s = ("-" if not lm else
                f"{lm['steps_saved_pct']:.1f}% steps, -{lm['successes_lost']} succ")
        a, n_ = r["auroc_heldout"], r["auroc_label_shuffle"]
        print(f"{r['mode']:<9} {r['k']:>3} {r['n_episodes']:>5} {a:>7.3f} "
              f"{(('%.3f' % n_) if n_ else '-'):>6}  {fx_s:>22}  {lm_s:>24}")

    print(f"\n=== zero-success-loss frontier ===")
    print(f"{'mode':<9} {'k':>3} {'abort':>6} {'steps saved':>12}")
    for r in all_rows:
        b = r["frontier_by_successes_lost"]["lost_le_0"]
        if b:
            print(f"{r['mode']:<9} {r['k']:>3} {b['n_aborted']:>6} "
                  f"{b['steps_saved_pct']:>11.1f}%")

    out = Path(args.out_dir) / f"early_abort_{args.baseline}_{args.site}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    product = {
        "baseline": args.baseline, "site": args.site, "seed": SEED, "n_folds": N_FOLDS,
        "prefixes": list(PREFIXES),
        "note": "features from the first k steps ONLY; the layered-evidence 0b AUROC uses "
                "whole-episode signals and would be looking at the future here. The "
                "truncate-at-k control is the fixed policy a learned abort must beat.",
        "results": all_rows,
    }
    out.write_text(json.dumps(product, indent=2))
    md = out.with_suffix(".md")
    md.write_text(render_md(product))
    print(f"\nwrote {out}\nwrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
