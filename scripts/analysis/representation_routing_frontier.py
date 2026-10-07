#!/usr/bin/env python3
"""Representation routing on the SR–cost plane: the frontier, not a dominance verdict
— 2026-10-07, post_hoc_exploratory

Why this exists. VLM4RWD (AC xab8 and reviewer 1MR9, 2026-09-29) said the same thing:
"failing to outperform a fixed policy on both metrics does not necessarily make routing
unhelpful", and the SR–cost trade-off was not explored. Every routing product here so far
returns a *point* per cell and a dominance verdict against best-single / always-cheapest
(`router_triage_learnability`, `router_objective_ordering`, `rule_routing_pareto`). A
router that trades SR for cost at a rate no fixed policy offers is useful even when it
dominates nothing, and a point-vs-point test cannot see it. (实验笔记 §539)

What is compared, per cell, on one (mean cost, SR) plane:

  fixed frontier   the six fixed modes AND every random mixture of them: the upper concave
                   envelope of the six points, flat after the best-SR mode. Mixing is a
                   deployable policy (send a random share of tasks to each of two modes), so
                   a router has to beat the hull, not just the six corners. The hull is
                   computed in-sample on the cell's own outcomes — the fixed side gets to
                   pick its modes knowing the answer, which is conservative for the router.
  six-head curve   one L2 LR success head per mode (P(success_m | features)), cheapest mode
                   whose P >= tau, else the training-fold best-SR mode; tau swept 0..1. This
                   is the construction `router_pareto_analysis.build_cost_aware_success_curve`
                   ran on cls_B0 alone (笔记 §406.10); here it is refitted on the matched
                   18-feature set for the eight cross-mode units.
  triage curve     one LR on "solvable by any mode"; predicted-hopeless -> cheapest mode,
                   else best-SR mode (fold-local); threshold swept over score quantiles.
                   The half of routing that has labels for every task (§383.4 / triage).
  oracle curve     hindsight per-task choice maximising success − λ·cost, λ swept. The
                   ceiling, so a learned excess can be read as a share of the headroom.

Every learned point is out-of-fold: task-held-out 5-fold CV, seed 42, the fold split of
`router_triage_learnability.oof_scores`, modes and decision costs re-selected on training
rows only. The *curve* is therefore honest; its *maximum* over tau is a selection over 21
points. The label-shuffle null (task bundle (y, success, cost) permuted against X, as in
B-1902) recomputes the whole curve per draw, so the reported p for the max excess carries
the same selection.

Reading the excess: excess(point) = SR(point) − H(cost(point)), H = fixed frontier. A
learned point cheaper than every fixed mode is scored against the cheapest mode's SR (as if
that SR were available at the lower cost) — again conservative for the router.

Cost = `total_billed_cost_usd`, mean per task, comparable within a cell only (B0 bills a
proxy API; B1/B2 are electricity-derived). Never read a cost axis across cells.

Pooling across cells (笔记 §540). Cost units differ, so cells are pooled on a normalised
budget u = (cost − cheapest fixed mode) / (dearest fixed mode − cheapest fixed mode), u in
[0, 1]. Per cell and u, the frontier gain is how much the attainable SR rises when the
curve's operating points are added to the fixed modes: envelope(fixed ∪ curve)(u) −
envelope(fixed)(u), ≥ 0 by construction (a router is deployable alongside the fixed modes
and mixable with them, the same reasoning that puts mixtures on the fixed side). The pooled
curve is the equal-weight mean of the eight cells' gain curves; its max over u is tested
against the same label-shuffle draws, pooled draw by draw (draw b of every cell averaged,
then the max over u), so the selection over u is inside the null too. The per-point excess
above and the gain differ only in that the gain lets the curve's own points be mixed; the
per-cell tests stay on the per-point excess.

Scope: the 8 cross-mode units (product_scope.yaml XMODE). Features: the matched 18 of
`router_triage_learnability --with-wa` on every cell (WA has no reference images and no
reasoning annotation), so VWA and WA rows are fitted on the same columns.

post_hoc_exploratory=True, h10_eligible=False. Touches no gating producer.

Usage (writes the md, the JSON and both PNGs next to them by default):
  python scripts/analysis/representation_routing_frontier.py
  python scripts/analysis/representation_routing_frontier.py --n-shuffle 200 --no-fig \
      --out /tmp/f.md --json-out /tmp/f.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis import router_triage_learnability as rt  # noqa: E402
from scripts.analysis.aggregate_phantom_lift import CELLS  # noqa: E402

MODES = rt.SIX_MODES
FEATURES = [f for f in rt.ALL_FEATURES if f not in rt.VWA_ONLY_FEATURES]
FEAT_IDX = [rt.ALL_FEATURES.index(f) for f in FEATURES]
SEED = rt.SEED
N_FOLDS = rt.N_FOLDS
TAUS = [round(float(t), 2) for t in np.linspace(0.0, 1.0, 21)]
TRIAGE_QUANTILES = [round(float(q), 2) for q in np.linspace(0.0, 1.0, 21)]
N_SHUFFLE = 1000
EPS = 1e-9
U_GRID = [round(float(u), 2) for u in np.linspace(0.0, 1.0, 101)]
CURVES = ("six_head", "triage", "oracle")

OUT_MD = REPO / "docs/analysis/cross_sites/representation_routing_frontier.md"
OUT_JSON = REPO / "docs/analysis/cross_sites/representation_routing_frontier.json"
COST_BASES = {   # 笔记 §545: the frontier under each cost basis the local-cost audit distinguishes
    "billed": {"field": None, "unit": "USD (`total_billed_cost_usd`; B1/B2 are token-priced electricity estimates)",
               "baselines": None},
    "wallclock": {"field": "total_latency_canonical_ms", "unit": "seconds of wall-clock episode time "
                  "(`total_latency_canonical_ms`), the occupancy a deployment pays for", "baselines": None},
    "gpu_time": {"field": None, "unit": "seconds of model inference (Σ steps `latency_ms.backend_infer`); "
                 "locally served backbones only — for B0 this would be API latency, not GPU time",
                 "baselines": ("B1", "B2")},
}
FIG_CELLS = "representation_routing_frontier_cells.png"
FIG_POOLED = "representation_routing_frontier_pooled.png"

SITE_KEY = {"classifieds": "cls", "reddit": "red", "wa_reddit": "wared", "shopping": "shop"}   # shopping: extension cells only (§547)
SITE_LABEL = {"classifieds": "classifieds", "reddit": "reddit", "wa_reddit": "wa_reddit", "shopping": "shopping"}


# ---------------------------------------------------------------------------- data

def _gpu_seconds(spec: dict, task_ids: list[int]) -> np.ndarray:
    """Per (task, mode): sum over steps of latency_ms.backend_infer, in seconds. A missing step
    file or field is an error, never a zero (笔记 §545 measured 0 missing on the 5 local cells)."""
    from p79.experiment.io_utils import read_jsonl_dedup

    out = np.zeros((len(task_ids), len(MODES)))
    for j, m in enumerate(MODES):
        if spec.get("_wa"):
            d = rt._wa_run_dir(spec["baseline"], m)
            hits = {int(p.name.split("_task_")[1].split("_steps")[0]): p
                    for p in d.glob("*/episodes/*_steps_v2.jsonl")}
        else:
            d = Path(spec["modes"][m])
            hits = {int(p.name.split("_task_")[1].split("_steps")[0]): p
                    for p in d.glob("*_steps_v2.jsonl")}
        for i, t in enumerate(task_ids):
            if t not in hits:
                raise FileNotFoundError(f"{spec['site']} {spec['baseline']} {m}: no steps file for task {t}")
            ms = [(r.get("latency_ms") or {}).get("backend_infer") for r in read_jsonl_dedup(hits[t])]
            if not ms or any(v is None for v in ms):
                raise ValueError(f"{hits[t]}: backend_infer missing on {sum(v is None for v in ms)} step(s)")
            out[i, j] = sum(ms) / 1000.0
    return out


def load_cell(spec: dict, basis: str = "billed") -> dict | None:
    field = COST_BASES[basis]["field"] or rt.COST_FIELD
    cell = (rt.build_wa_cell(spec, cost_field=field) if spec.get("_wa")
            else rt.build_cell(spec, cost_field=field))
    if cell is None:
        return None
    S = np.array([[bool(s[m]) for m in MODES] for s in cell["succ"]], dtype=float)
    C = np.array([[float(c[m]) for m in MODES] for c in cell["cost"]], dtype=float)
    if basis == "gpu_time":
        C = _gpu_seconds(spec, cell["task_ids"])
    elif basis == "wallclock":
        C = C / 1000.0                      # ms -> s
    if basis != "billed" and not (C > 0).all():
        # build_cell maps a missing field to 0.0; on a time basis that is a missing value, not a
        # free episode.
        raise ValueError(f"{spec['site']} {spec['baseline']}: {int((C <= 0).sum())} non-positive "
                         f"{basis} cost(s) — a missing field, not a free episode")
    return {"site": cell["site"], "baseline": cell["baseline"], "task_ids": cell["task_ids"],
            "X": cell["X"][:, FEAT_IDX], "y": cell["y"], "S": S, "C": C}


def fold_split(n: int) -> list[np.ndarray]:
    """Same permutation and split as `router_triage_learnability.oof_scores`."""
    idx = np.random.default_rng(SEED).permutation(n)
    return np.array_split(idx, N_FOLDS)


def _fit_proba(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    """Fold-local standardisation + L2 LR (C=1.0, as the triage product). A single-class
    training fold has nothing to separate: return that class as a constant probability."""
    from sklearn.linear_model import LogisticRegression

    if len(np.unique(ytr)) < 2:
        return np.full(len(Xte), float(ytr[0]))
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit((Xtr - mu) / sd, ytr)
    return lr.predict_proba((Xte - mu) / sd)[:, 1]


def policy_point(S: np.ndarray, C: np.ndarray, sel: np.ndarray) -> tuple[float, float]:
    rows = np.arange(len(sel))
    return float(C[rows, sel].mean()), float(100.0 * S[rows, sel].mean())


# ---------------------------------------------------------------------------- frontier

def fixed_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Upper concave envelope of (cost, SR) points, cut at the best-SR vertex: the set of
    (cost, SR) reachable by randomly mixing fixed modes, with free disposal of cost."""
    pts = sorted(set(points), key=lambda p: (p[0], -p[1]))
    hull: list[tuple[float, float]] = []
    for p in pts:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            # pop hull[-1] if it lies on or below the chord hull[-2] -> p
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) >= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    top = max(range(len(hull)), key=lambda i: (hull[i][1], -hull[i][0]))
    return hull[: top + 1]


def hull_sr_at(hull: list[tuple[float, float]], cost: float) -> float:
    if cost <= hull[0][0]:
        return hull[0][1]          # below every fixed cost: scored against the cheapest mode
    for (x1, y1), (x2, y2) in zip(hull, hull[1:]):
        if cost <= x2:
            return y1 + (y2 - y1) * (cost - x1) / (x2 - x1)
    return hull[-1][1]


def frontier_gain(fixed_pts: list[tuple[float, float]], curve_pts: list[tuple[float, float]],
                  ) -> np.ndarray:
    """SR (pp) the attainable envelope gains on the U_GRID when the curve's points join the
    fixed modes. u = 0 is the cheapest fixed mode's mean cost, u = 1 the dearest's."""
    costs = [c for c, _ in fixed_pts]
    lo, hi = min(costs), max(costs)
    base = fixed_hull(fixed_pts)
    both = fixed_hull(list(fixed_pts) + list(curve_pts))
    return np.array([hull_sr_at(both, lo + u * (hi - lo)) - hull_sr_at(base, lo + u * (hi - lo))
                     for u in U_GRID])


# ---------------------------------------------------------------------------- curves

def six_head_curve(X, S, C, folds) -> list[dict]:
    n = len(S)
    P = np.zeros_like(S)
    dec_cost = np.zeros_like(C)          # per task: its training fold's mean cost per mode
    fallback = np.zeros(n, dtype=int)    # per task: its training fold's best-SR mode
    for f in folds:
        tr = np.setdiff1d(np.arange(n), f)
        tr_cost, tr_sr = C[tr].mean(0), S[tr].mean(0)
        dec_cost[f] = tr_cost
        fallback[f] = min(range(S.shape[1]), key=lambda j: (-tr_sr[j], tr_cost[j], j))
        for j in range(S.shape[1]):
            P[f, j] = _fit_proba(X[tr], S[tr, j].astype(int), X[f])
    out = []
    for tau in TAUS:
        elig = P >= tau - EPS
        masked = np.where(elig, dec_cost, np.inf)
        sel = np.where(elig.any(1), masked.argmin(1), fallback)
        cost, sr = policy_point(S, C, sel)
        out.append({"tau": tau, "cost": cost, "sr_pct": sr,
                    "n_fallback": int((~elig.any(1)).sum())})
    return out


def triage_curve(X, y, S, C, folds) -> list[dict]:
    n = len(S)
    score = np.zeros(n)
    best = np.zeros(n, dtype=int)
    cheap = np.zeros(n, dtype=int)
    for f in folds:
        tr = np.setdiff1d(np.arange(n), f)
        tr_cost, tr_sr = C[tr].mean(0), S[tr].mean(0)
        best[f] = min(range(S.shape[1]), key=lambda j: (-tr_sr[j], tr_cost[j], j))
        cheap[f] = min(range(S.shape[1]), key=lambda j: (tr_cost[j], j))
        score[f] = _fit_proba(X[tr], y[tr], X[f])
    out = []
    thresholds = [float(np.quantile(score, q)) for q in TRIAGE_QUANTILES] + [np.inf]
    labels = TRIAGE_QUANTILES + ["all"]
    for q, thr in zip(labels, thresholds):
        to_cheap = score < thr if np.isfinite(thr) else np.ones(n, dtype=bool)
        sel = np.where(to_cheap, cheap, best)
        cost, sr = policy_point(S, C, sel)
        out.append({"quantile": q, "cost": cost, "sr_pct": sr, "n_sent_cheap": int(to_cheap.sum())})
    return out


def oracle_curve(S, C) -> list[dict]:
    """Hindsight: per task argmax success − λ·cost/scale, ties to the cheaper mode."""
    scale = float(np.median(C.mean(0))) or 1.0
    lams = [0.0] + [float(v) for v in np.logspace(-3, 3, 61)]
    out = []
    for lam in lams:
        util = S - lam * C / scale - 1e-12 * C   # tiny cost term = cheaper tie-break at λ=0
        sel = util.argmax(1)
        cost, sr = policy_point(S, C, sel)
        out.append({"lambda": lam, "cost": cost, "sr_pct": sr})
    return out


def excess_summary(curve: list[dict], hull, param: str) -> dict:
    for p in curve:
        p["excess_pp"] = p["sr_pct"] - hull_sr_at(hull, p["cost"])
    top = max(curve, key=lambda p: (p["excess_pp"], -p["cost"]))
    return {
        "max_excess_pp": top["excess_pp"],
        "at": {param: top[param], "cost": top["cost"], "sr_pct": top["sr_pct"]},
        "n_points_above_hull": int(sum(p["excess_pp"] > EPS for p in curve)),
        "n_points": len(curve),
    }


# ---------------------------------------------------------------------------- per cell

def evaluate(cell: dict, n_shuffle: int) -> dict:
    X, y, S, C = cell["X"], cell["y"], cell["S"], cell["C"]
    n = len(y)
    folds = fold_split(n)

    fixed = {m: {"cost": float(C[:, j].mean()), "sr_pct": float(100 * S[:, j].mean())}
             for j, m in enumerate(cell.get("modes", MODES))}   # extension cells carry fewer modes (§547)
    hull = fixed_hull([(v["cost"], v["sr_pct"]) for v in fixed.values()])
    hull_modes = [next(m for m, v in fixed.items() if (v["cost"], v["sr_pct"]) == h) for h in hull]

    six = six_head_curve(X, S, C, folds)
    tri = triage_curve(X, y, S, C, folds)
    orc = oracle_curve(S, C)
    s_six = excess_summary(six, hull, "tau")
    s_tri = excess_summary(tri, hull, "quantile")
    s_orc = excess_summary(orc, hull, "lambda")

    fixed_pts = [(v["cost"], v["sr_pct"]) for v in fixed.values()]

    def _gain(curve: list[dict]) -> np.ndarray:
        return frontier_gain(fixed_pts, [(p["cost"], p["sr_pct"]) for p in curve])

    gain = {"six_head": _gain(six), "triage": _gain(tri), "oracle": _gain(orc)}

    # Label-shuffle null for the max excess: permute the task bundle against X and redo
    # both curves end to end (fold-local fits, sweep, max). The hull is a property of the
    # cell's aggregate outcomes and is invariant to the permutation. Each draw's gain curve
    # is kept (in memory only) for the cross-cell pooled null.
    rng = np.random.default_rng(SEED + 1)
    null_six, null_tri = [], []
    null_gain = {"six_head": np.zeros((n_shuffle, len(U_GRID))),
                 "triage": np.zeros((n_shuffle, len(U_GRID)))}
    for b in range(n_shuffle):
        perm = rng.permutation(n)
        yb, Sb, Cb = y[perm], S[perm], C[perm]
        c6 = six_head_curve(X, Sb, Cb, folds)
        ct = triage_curve(X, yb, Sb, Cb, folds)
        null_six.append(excess_summary(c6, hull, "tau")["max_excess_pp"])
        null_tri.append(excess_summary(ct, hull, "quantile")["max_excess_pp"])
        null_gain["six_head"][b] = _gain(c6)
        null_gain["triage"][b] = _gain(ct)

    def _p(obs: float, null: list[float]) -> float | None:
        if not null:
            return None
        k = sum(v >= obs - 1e-12 for v in null)
        return (k + 1) / (len(null) + 1)

    for s, null in ((s_six, null_six), (s_tri, null_tri)):
        s["null_p"] = _p(s["max_excess_pp"], null)
        s["null_median_pp"] = float(np.median(null)) if null else None
        s["null_q95_pp"] = float(np.quantile(null, 0.95)) if null else None
        s["share_of_oracle_headroom"] = (s["max_excess_pp"] / s_orc["max_excess_pp"]
                                         if s_orc["max_excess_pp"] > EPS else None)

    return {
        "site": SITE_LABEL[cell["site"]], "baseline": cell["baseline"],
        "cell_id": f"{SITE_KEY[cell['site']]}_{cell['baseline']}",
        "n_tasks": n, "solvable_pct": float(100 * y.mean()),
        "fixed_modes": fixed,
        "fixed_hull": [{"mode": m, "cost": c, "sr_pct": s} for m, (c, s) in zip(hull_modes, hull)],
        "six_head": {"summary": s_six, "curve": six},
        "triage": {"summary": s_tri, "curve": tri},
        "oracle": {"summary": s_orc, "curve": orc},
        "frontier_gain": {
            "u0_cost": min(c for c, _ in fixed_pts), "u1_cost": max(c for c, _ in fixed_pts),
            **{k: {"gain_pp": gain[k].tolist(), "max_gain_pp": float(gain[k].max()),
                   "at_u": U_GRID[int(gain[k].argmax())]} for k in CURVES},
        },
        "n_shuffle": n_shuffle,
        "_null_gain": null_gain,
    }


def run_cell(spec: dict, n_shuffle: int, basis: str = "billed") -> dict:
    t0 = time.time()
    cell = load_cell(spec, basis)
    if cell is None:
        raise RuntimeError(f"cell {spec} did not build; every XMODE unit is expected to")
    res = evaluate(cell, n_shuffle)
    print(f"{res['cell_id']}: six-head {res['six_head']['summary']['max_excess_pp']:+.2f}pp "
          f"triage {res['triage']['summary']['max_excess_pp']:+.2f}pp "
          f"oracle {res['oracle']['summary']['max_excess_pp']:+.2f}pp ({time.time() - t0:.0f}s)",
          file=sys.stderr, flush=True)
    return res


def pool(results: list[dict]) -> dict:
    """Equal-weight mean of the cells' gain curves on U_GRID; for the learned curves, the max
    over u tested against the pooled null (draw b averaged across cells, then max over u)."""
    out = {"weighting": "equal per cell", "cells": [r["cell_id"] for r in results],
           "u_grid": U_GRID}
    for k in CURVES:
        mean = np.mean([r["frontier_gain"][k]["gain_pp"] for r in results], axis=0)
        i = int(mean.argmax())
        s = {"mean_gain_pp": mean.tolist(), "max_pp": float(mean[i]), "at_u": U_GRID[i],
             "n_cells_with_gain": int(sum(r["frontier_gain"][k]["max_gain_pp"] > EPS for r in results))}
        if k != "oracle":
            draws = np.mean([r["_null_gain"][k] for r in results], axis=0)   # (B, len(U_GRID))
            null_max = draws.max(1)
            s["null_p"] = (int((null_max >= mean[i] - 1e-12).sum()) + 1) / (len(null_max) + 1)
            s["null_max_median_pp"] = float(np.median(null_max))
            s["null_max_q95_pp"] = float(np.quantile(null_max, 0.95))
            s["null_pointwise_q95_pp"] = np.quantile(draws, 0.95, axis=0).tolist()
        out[k] = s
    orc_max = out["oracle"]["max_pp"]
    for k in ("six_head", "triage"):
        out[k]["share_of_oracle_max"] = out[k]["max_pp"] / orc_max if orc_max > EPS else None
    return out


def holm(pvals: dict[str, float | None], alpha: float = 0.05) -> dict[str, bool]:
    items = sorted(((p, k) for k, p in pvals.items() if p is not None))
    m = len(items)
    out, alive = {}, True
    for i, (p, k) in enumerate(items):
        alive = alive and p <= alpha / (m - i)
        out[k] = alive
    return out


# ---------------------------------------------------------------------------- render

def _f(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}"


def render_pooled(payload: dict) -> list[str]:
    P, cells = payload["pooled"], payload["cells"]
    m, B = len(cells), payload["protocol"]["n_shuffle"]
    L = [
        "## 2. Pooled across cells: the frontier on a normalised budget",
        "",
        "Cost units differ between cells, so each cell's budget is normalised: **u = 0 is its "
        "cheapest fixed mode, u = 1 its dearest** (101 grid points). Per cell and u, the "
        "**frontier gain** is the SR the attainable envelope gains when the curve's operating "
        "points are added to the fixed modes (and may be mixed with them) — ≥ 0 by "
        f"construction. The pooled curve is the equal-weight mean over the {m} cells. Its max "
        f"over u is tested against the same B={B} label-shuffle draws, averaged across cells "
        "draw by draw before taking the max, so the choice of u is inside the null. A cell's max "
        "gain is not §1's max excess: it can be higher (the curve's own points may be mixed) or "
        "slightly lower (it is read on the grid inside [0, 1], so a peak between grid points or "
        "outside the fixed cost range is missed). The per-cell tests stay those of §1.",
        "",
        "| curve | pooled max gain | at u | cells with any gain | null max: median / q95 | p | share of oracle max |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, name in (("six_head", "six-head (OOF)"), ("triage", "triage (OOF)"), ("oracle", "oracle (hindsight)")):
        s = P[k]
        null = (f"{s['null_max_median_pp']:.2f} / {s['null_max_q95_pp']:.2f}pp"
                if "null_p" in s else "—")
        share = s.get("share_of_oracle_max")
        L.append(f"| {name} | +{s['max_pp']:.2f}pp | {s['at_u']} | {s['n_cells_with_gain']} of {m} "
                 f"| {null} | {_f(s.get('null_p'), 4)} | {_f(share and 100 * share, 0) + '%' if share is not None else '—'} |")
    L += [
        "",
        "Per cell (gain is the max over u of that cell's curve, in SR pp):",
        "",
        "| cell | u = 0 → 1 cost | six-head max gain (u) | triage max gain (u) | oracle max gain (u) |",
        "|---|---|---|---|---|",
    ]
    for c in cells:
        g = c["frontier_gain"]
        L.append(f"| {c['cell_id']} | {g['u0_cost']:.5f} → {g['u1_cost']:.5f} "
                 + " ".join(f"| +{g[k]['max_gain_pp']:.2f}pp ({g[k]['at_u']})" for k in CURVES) + " |")
    L += [
        "",
        "Pooled gain at selected budgets (pp; null = pointwise q95 of the pooled draws, not a test):",
        "",
        "| u | six-head | null q95 | triage | null q95 | oracle |",
        "|---|---|---|---|---|---|",
    ]
    for u in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        i = U_GRID.index(u)
        L.append(f"| {u} | {P['six_head']['mean_gain_pp'][i]:.2f} | {P['six_head']['null_pointwise_q95_pp'][i]:.2f} "
                 f"| {P['triage']['mean_gain_pp'][i]:.2f} | {P['triage']['null_pointwise_q95_pp'][i]:.2f} "
                 f"| {P['oracle']['mean_gain_pp'][i]:.2f} |")
    L += ["", f"![pooled frontier gain]({FIG_POOLED})", ""]
    return L


def render(payload: dict) -> str:
    cells = payload["cells"]
    m = len(cells)
    B = payload["protocol"]["n_shuffle"]
    L = [
        "# Representation routing on the SR–cost plane — the frontier, not a dominance verdict",
        "",
        "Generated by `scripts/analysis/representation_routing_frontier.py`. `post_hoc_exploratory=True`, "
        "`h10_eligible=False`. Chronicle: 实验笔记 §539.",
        "",
        "Answers VLM4RWD AC xab8 / 1MR9 #2: does a learned representation router reach (cost, SR) "
        "points that no fixed policy — **including random mixtures of fixed modes** — reaches? "
        "The fixed frontier is the upper concave envelope of the six fixed modes (in-sample, so the "
        "fixed side picks its modes knowing the outcomes). Learned curves are out-of-fold "
        f"(task-held-out {N_FOLDS}-fold, seed {SEED}, matched {len(FEATURES)} features, modes and "
        "decision costs re-selected on training rows). Excess = SR of a curve point minus the "
        "frontier's SR at the same mean cost; a point cheaper than every fixed mode is scored "
        "against the cheapest mode's SR.",
        "",
        f"Cost basis **{payload['protocol'].get('cost_basis', 'billed')}**: "
        f"{payload['protocol'].get('cost_unit', COST_BASES['billed']['unit'])}, mean per task, "
        "**comparable within a cell only**."
        + ("" if payload["protocol"].get("cost_basis", "billed") == "billed" else
           " Sensitivity sibling of `representation_routing_frontier` (笔记 §545): same pipeline, "
           "same seeds, only the cost axis changes; the per-cell `p` and Holm are recomputed here."),
        "",
        f"`p` = label-shuffle null for the curve's **max** excess (task bundle permuted against X, "
        f"whole curve refitted per draw, B={B}, plus-one estimator), so the selection over the "
        f"sweep is inside the null. Holm across the {m} cells, per curve.",
        "",
        "## 1. Headline: how far above the fixed frontier does each curve get?",
        "",
        "| cell | n | fixed frontier (modes on hull) | oracle max excess | six-head max excess (τ) | p | Holm | triage max excess (q) | p | Holm |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in cells:
        s6, st, so = c["six_head"]["summary"], c["triage"]["summary"], c["oracle"]["summary"]
        hull = " → ".join(h["mode"] for h in c["fixed_hull"])
        L.append(
            f"| {c['cell_id']} | {c['n_tasks']} | {hull} | +{so['max_excess_pp']:.2f}pp "
            f"| {s6['max_excess_pp']:+.2f}pp ({s6['at']['tau']}) | {_f(s6['null_p'], 4)} "
            f"| {'pass' if c['six_head']['holm'] else '—'} "
            f"| {st['max_excess_pp']:+.2f}pp ({st['at']['quantile']}) | {_f(st['null_p'], 4)} "
            f"| {'pass' if c['triage']['holm'] else '—'} |"
        )
    n6 = sum(c["six_head"]["summary"]["max_excess_pp"] > EPS for c in cells)
    nt = sum(c["triage"]["summary"]["max_excess_pp"] > EPS for c in cells)
    h6 = sum(c["six_head"]["holm"] for c in cells)
    ht = sum(c["triage"]["holm"] for c in cells)
    L += [
        "",
        f"- Six-head curve rises above the fixed frontier anywhere in **{n6} of {m}** cells; "
        f"the max excess survives its null under Holm in **{h6} of {m}**.",
        f"- Triage curve: above the frontier in **{nt} of {m}**; Holm survivors **{ht} of {m}**.",
        "- Share of the oracle's headroom the six-head max excess captures, per cell: "
        + ", ".join(f"{c['cell_id']} {_f(c['six_head']['summary']['share_of_oracle_headroom'] and 100 * c['six_head']['summary']['share_of_oracle_headroom'], 0)}%"
                    for c in cells) + ".",
        "",
        "A positive max excess is a point estimate on one run per arm; read it against the null "
        "column, and — where a cell has one — against its rerun band (which band definition to use "
        "is the open §530.4 #2 decision, so no band is applied here).",
        "",
    ]
    L += render_pooled(payload)
    L += [
        "## 3. Per cell: the curves",
        "",
        f"![SR–cost plane per cell]({FIG_CELLS})",
        "",
        "Fixed modes and frontier, then each learned curve's points that sit on or above the "
        "frontier (all points are in the JSON).",
        "",
    ]
    for c in cells:
        L += [f"### {c['cell_id']} (n={c['n_tasks']}, solvable {c['solvable_pct']:.1f}%)", "",
              "| fixed mode | mean cost | SR % | on frontier |", "|---|---|---|---|"]
        on = {h["mode"] for h in c["fixed_hull"]}
        for mname, v in sorted(c["fixed_modes"].items(), key=lambda kv: kv[1]["cost"]):
            L.append(f"| {mname} | {v['cost']:.5f} | {v['sr_pct']:.2f} | {'yes' if mname in on else ''} |")
        for name, key in (("six-head", "tau"), ("triage", "quantile")):
            pts = [p for p in c[name.replace('-', '_')]["curve"] if p["excess_pp"] > -EPS]
            L += ["", f"{name} points on/above the frontier: "
                  + (", ".join(f"{key}={p[key]} → {p['sr_pct']:.2f}% @ {p['cost']:.5f} ({p['excess_pp']:+.2f}pp)"
                               for p in pts) if pts else "none")]
        L.append("")
    return "\n".join(L).rstrip() + "\n"


def plot_pooled(payload: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P = payload["pooled"]
    u = P["u_grid"]
    style = {"six_head": ("#2a6fdb", "six-head (OOF)"), "triage": ("#d9822b", "triage (OOF)"),
             "oracle": ("#3a9a5b", "oracle (hindsight)")}
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    for k in CURVES:
        col, lab = style[k]
        a.plot(u, P[k]["mean_gain_pp"], color=col, lw=1.6, label=lab)
    a.set_title(f"mean frontier gain over {len(P['cells'])} cells", fontsize=9)
    for k in ("six_head", "triage"):
        col, lab = style[k]
        b.plot(u, P[k]["mean_gain_pp"], color=col, lw=1.6, label=lab)
        b.plot(u, P[k]["null_pointwise_q95_pp"], color=col, lw=1.0, ls="--",
               label=f"{lab.split(' ')[0]}: label-shuffle q95")
    b.set_title("learned curves vs their label-shuffle null (pointwise q95)", fontsize=9)
    for ax in (a, b):
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xlabel("normalised budget u (0 = cheapest fixed mode, 1 = dearest)", fontsize=8)
        ax.set_ylabel("SR gain over fixed modes + mixtures (pp)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def plot_cells(payload: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = payload["cells"]
    cols = 4
    rows = (len(cells) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), squeeze=False)
    for ax, c in zip(axes.flat, cells):
        hull = c["fixed_hull"]
        top = max(p["cost"] for k in ("six_head", "triage", "oracle") for p in c[k]["curve"])
        hx = [h["cost"] for h in hull] + [max(top, hull[-1]["cost"])]
        hy = [h["sr_pct"] for h in hull] + [hull[-1]["sr_pct"]]
        ax.plot(hx, hy, color="0.35", lw=1.6, label="fixed frontier (mixtures)")
        for mname, v in c["fixed_modes"].items():
            ax.scatter(v["cost"], v["sr_pct"], color="0.35", s=18, zorder=3)
            ax.annotate(mname, (v["cost"], v["sr_pct"]), fontsize=6, xytext=(2, 2),
                        textcoords="offset points", color="0.3")
        for key, col, lab in (("six_head", "#2a6fdb", "six-head (OOF)"),
                              ("triage", "#d9822b", "triage (OOF)"),
                              ("oracle", "#3a9a5b", "oracle (hindsight)")):
            pts = sorted(c[key]["curve"], key=lambda p: p["cost"])
            ax.plot([p["cost"] for p in pts], [p["sr_pct"] for p in pts], marker=".", ms=3,
                    lw=1.0, color=col, label=lab, alpha=0.9 if key != "oracle" else 0.6)
        ax.set_title(c["cell_id"], fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("mean cost per task (cell units)", fontsize=7)
        ax.set_ylabel("SR %", fontsize=7)
    for ax in list(axes.flat)[len(cells):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_MD)
    ap.add_argument("--json-out", type=Path, default=OUT_JSON)
    ap.add_argument("--n-shuffle", type=int, default=N_SHUFFLE)
    ap.add_argument("--jobs", type=int, default=8, help="cells evaluated in parallel processes")
    ap.add_argument("--no-fig", action="store_true",
                    help=f"skip {FIG_CELLS} / {FIG_POOLED} (written next to --out by default; "
                         "billed basis only)")
    ap.add_argument("--cost-basis", choices=sorted(COST_BASES), default="billed",
                    help="billed writes the main product; the others write a _<basis> sibling")
    args = ap.parse_args()
    if args.cost_basis != "billed":
        if args.out == OUT_MD:
            args.out = OUT_MD.with_name(f"{OUT_MD.stem}_{args.cost_basis}.md")
        if args.json_out == OUT_JSON:
            args.json_out = OUT_JSON.with_name(f"{OUT_JSON.stem}_{args.cost_basis}.json")
        args.no_fig = True

    specs = list(CELLS) + list(rt.WA_CELLS)
    keep = COST_BASES[args.cost_basis]["baselines"]
    if keep:
        specs = [s for s in specs if s["baseline"] in keep]
    # Cells are independent and each seeds its own RNG, so --jobs changes wall time only.
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        results = list(ex.map(run_cell, specs, [args.n_shuffle] * len(specs),
                              [args.cost_basis] * len(specs)))

    for key in ("six_head", "triage"):
        verdict = holm({r["cell_id"]: r[key]["summary"]["null_p"] for r in results})
        for r in results:
            r[key]["holm"] = bool(verdict.get(r["cell_id"], False))

    pooled = pool(results)
    for r in results:
        del r["_null_gain"]
    print(f"pooled: six-head +{pooled['six_head']['max_pp']:.2f}pp (p={pooled['six_head']['null_p']:.4f}) "
          f"triage +{pooled['triage']['max_pp']:.2f}pp (p={pooled['triage']['null_p']:.4f}) "
          f"oracle +{pooled['oracle']['max_pp']:.2f}pp", file=sys.stderr)

    payload = {
        "post_hoc_exploratory": True, "h10_eligible": False,
        "producer": "scripts/analysis/representation_routing_frontier.py",
        "protocol": {
            "folds": N_FOLDS, "seed": SEED, "features": FEATURES,
            "taus": TAUS, "triage_quantiles": TRIAGE_QUANTILES,
            "n_shuffle": args.n_shuffle,
            "null_unit": "task bundle (y, success_by_mode, cost_by_mode) permuted against X",
            "p_estimator": "(k+1)/(B+1)", "multiplicity": "Holm across cells, per curve",
            "cost_field": rt.COST_FIELD,
            "cost_basis": args.cost_basis, "cost_unit": COST_BASES[args.cost_basis]["unit"],
            "excess_definition": "SR(point) - fixed-frontier SR at the same mean cost; below the "
                                 "cheapest fixed cost the cheapest mode's SR is used",
            "frontier_gain_definition": "envelope(fixed ∪ curve points) - envelope(fixed), in SR pp, "
                                        "on u = (cost - cheapest fixed) / (dearest fixed - cheapest fixed)",
            "pooled_null": "per draw b, mean of the cells' gain curves, then max over u",
        },
        "cells": results,
        "pooled": pooled,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    args.out.write_text(render(payload), encoding="utf-8")
    if not args.no_fig:
        plot_cells(payload, args.out.parent / FIG_CELLS)
        plot_pooled(payload, args.out.parent / FIG_POOLED)
    print(f"wrote {args.out} and {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
