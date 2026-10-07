#!/usr/bin/env python3
"""How large a representation-routing gain do the data rule out? One-sided upper bounds for a
deployable, cross-fitted router on the SR–cost plane — 2026-10-07, post_hoc_exploratory (笔记 §544)

Why this exists. `representation_routing_frontier` (§539 / §540) says how far learned routing
curves rise above the fixed modes and their random mixtures and whether that beats a label-shuffle
null. A non-significant excess does not say how large a gain the data exclude — the question a
reviewer asks of a negative result ("no effect, or no power?").

Why not bound the frontier product's own statistic. Its statistic is a max over 21 operating
points. A percentile bootstrap of a max is not a valid interval (the functional is not smooth);
tried first, it gave per-cell draws from −2.7 to +14.7pp on cls_B0 and bounds of 8–12pp that say
more about the bootstrap than about the data (笔记 §544.1).

Estimand here: the excess over the fixed frontier (fixed modes + random mixtures, full sample, as
in the frontier product) of ONE policy per curve family whose operating point is chosen on
training data only. In each outer fold, tau (six-head) or the triage quantile is the value with
the largest excess on the training rows' own frontier, from in-sample predictions; it is applied
to the held-out fold. The five folds' decisions form one policy over every task. This is what a
deployer could actually run, so its excess is the honest "gain", with no selection on test data.

Bootstrap. Tasks resampled with replacement (B=1000, seed 7), whole pipeline refitted per
resample; the 5-fold split is over DISTINCT task ids, so copies of a resampled task sit in the
same fold. Upper bound = 95th percentile. Pooled = equal-weight mean over the 8 cells,
bootstrapped draw by draw. References that need no band decision: one task (100/n pp) and the
observed |ΔSR| between the cell's registered same-condition reruns (noise_floor_inventory;
reference only — the band definition is the open §530.4 #2 decision).

Usage:
  python scripts/analysis/routing_gain_upper_bounds.py
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

from scripts.analysis import representation_routing_frontier as rrf  # noqa: E402
from scripts.analysis import router_triage_learnability as rt  # noqa: E402
from scripts.analysis.aggregate_phantom_lift import CELLS  # noqa: E402

N_BOOT = 1000
SEED = 7
NOISE_JSON = REPO / "docs/analysis/cross_sites/noise_floor_inventory.json"
OUT_MD = REPO / "docs/analysis/cross_sites/routing_gain_upper_bounds.md"
OUT_JSON = REPO / "docs/analysis/cross_sites/routing_gain_upper_bounds.json"
LABEL_TO_CELL = {"cls": "cls", "red": "red", "wared": "wared"}


def group_folds(groups: np.ndarray) -> list[np.ndarray]:
    """5 folds over distinct ids (same permutation rule as the frontier product), as row indices."""
    uniq = np.unique(groups)
    perm = np.random.default_rng(rrf.SEED).permutation(len(uniq))
    parts = np.array_split(uniq[perm], rrf.N_FOLDS)
    return [np.flatnonzero(np.isin(groups, p)) for p in parts]


def _hull(S, C):
    return rrf.fixed_hull([(float(C[:, j].mean()), float(100 * S[:, j].mean())) for j in range(S.shape[1])])


def _excess(S, C, sel, hull) -> tuple[float, float]:
    cost, sr = rrf.policy_point(S, C, sel)
    return sr - rrf.hull_sr_at(hull, cost), cost


def crossfit(X, y, S, C, folds) -> dict:
    """One deployable policy per curve family: in each outer fold the operating point (tau, or the
    triage quantile) is the one with the largest excess over the TRAINING rows' own frontier,
    using in-sample predictions on those rows; it is then applied to the held-out fold. The five
    folds' decisions form one policy over every task, scored against the full-sample frontier
    exactly as the frontier product scores its curve points."""
    n, m = S.shape
    sel6 = np.zeros(n, dtype=int)
    selt = np.zeros(n, dtype=int)
    for f in folds:
        tr = np.setdiff1d(np.arange(n), f)
        Str, Ctr = S[tr], C[tr]
        hull_tr = _hull(Str, Ctr)
        tr_cost, tr_sr = Ctr.mean(0), Str.mean(0)
        best = min(range(m), key=lambda j: (-tr_sr[j], tr_cost[j], j))
        cheap = min(range(m), key=lambda j: (tr_cost[j], j))
        # six-head
        P_tr = np.column_stack([rrf._fit_proba(X[tr], Str[:, j].astype(int), X[tr]) for j in range(m)])
        P_te = np.column_stack([rrf._fit_proba(X[tr], Str[:, j].astype(int), X[f]) for j in range(m)])

        def pick(P, tau):
            elig = P >= tau - rrf.EPS
            masked = np.where(elig, tr_cost[None, :], np.inf)
            return np.where(elig.any(1), masked.argmin(1), best)

        tau = max(rrf.TAUS, key=lambda t: (_excess(Str, Ctr, pick(P_tr, t), hull_tr)[0],
                                           -_excess(Str, Ctr, pick(P_tr, t), hull_tr)[1]))
        sel6[f] = pick(P_te, tau)
        # triage
        s_tr = rrf._fit_proba(X[tr], y[tr], X[tr])
        s_te = rrf._fit_proba(X[tr], y[tr], X[f])
        thresholds = [float(np.quantile(s_tr, q)) for q in rrf.TRIAGE_QUANTILES] + [np.inf]

        def route(s, thr):
            return np.where(s < thr, cheap, best)

        thr = max(thresholds, key=lambda v: (_excess(Str, Ctr, route(s_tr, v), hull_tr)[0],
                                             -_excess(Str, Ctr, route(s_tr, v), hull_tr)[1]))
        selt[f] = route(s_te, thr)
    hull = _hull(S, C)
    return {"six_head": _excess(S, C, sel6, hull)[0], "triage": _excess(S, C, selt, hull)[0]}


def run_cell(spec: dict, n_boot: int) -> dict:
    t0 = time.time()
    cell = rrf.load_cell(spec)
    if cell is None:
        raise RuntimeError(f"cell {spec} did not build")
    X, y, S, C = cell["X"], cell["y"], cell["S"], cell["C"]
    n = len(y)
    cid = f"{rrf.SITE_KEY[cell['site']]}_{cell['baseline']}"
    obs = crossfit(X, y, S, C, rrf.fold_split(n))     # the frontier product's split
    rng = np.random.default_rng(SEED)
    boot = {"six_head": [], "triage": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        st = crossfit(X[idx], y[idx], S[idx], C[idx], group_folds(idx))
        for k in boot:
            boot[k].append(st[k])
    print(f"{cid}: six-head {obs['six_head']:+.2f} ub {np.quantile(boot['six_head'], .95):.2f} · "
          f"triage {obs['triage']:+.2f} ub {np.quantile(boot['triage'], .95):.2f} ({time.time() - t0:.0f}s)",
          file=sys.stderr, flush=True)
    return {"cell_id": cid, "n_tasks": n, "observed": obs,
            "upper95": {k: float(np.quantile(v, 0.95)) for k, v in boot.items()},
            "lower05": {k: float(np.quantile(v, 0.05)) for k, v in boot.items()},
            "_boot": boot}


def rerun_reference() -> dict[str, list[float]]:
    inv = json.loads(NOISE_JSON.read_text(encoding="utf-8"))
    out: dict[str, list[float]] = {}
    for cp in inv["clean_pairs"]:
        bl, site, _ = cp["label"].split(".")
        out.setdefault(f"{LABEL_TO_CELL[site]}_{bl}", []).append(float(cp["abs_mean_diff_pp"]))
    return out


def render(p: dict) -> str:
    L = [
        "# Routing gain the data rule out — upper bounds for a cross-fitted router on the SR–cost plane",
        "",
        "Generated by `scripts/analysis/routing_gain_upper_bounds.py`. `post_hoc_exploratory=True`. "
        "Chronicle: 实验笔记 §544. Companion to `representation_routing_frontier` (§539 / §540).",
        "",
        "**Estimand**: excess SR over the fixed frontier (six fixed modes and their random mixtures) of one "
        "deployable policy per curve family, whose operating point is chosen on training folds only and "
        "applied to the held-out fold. It is not the frontier product's max excess, which selects the "
        "best of 21 points after seeing the test outcomes. "
        f"Bounds: task bootstrap, B={p['n_boot']}, whole pipeline refitted, folds over distinct task ids.",
        "",
        "References: **one task** = 100/n pp; **rerun |ΔSR|** = range of |SR(run a) − SR(run b)| over the "
        "cell's registered same-condition pairs — a reference, not a band (§530.4 #2 is open).",
        "",
        "| cell | n | one task | rerun \\|ΔSR\\| (pairs) | six-head excess | [5%, **95%**] | triage excess | [5%, **95%**] |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in p["cells"]:
        ref = c["rerun_abs_diff_pp"]
        refs = "—" if not ref else f"{min(ref):.2f}–{max(ref):.2f} ({len(ref)})"
        L.append(f"| {c['cell_id']} | {c['n_tasks']} | {100 / c['n_tasks']:.2f} | {refs} "
                 f"| {c['observed']['six_head']:+.2f} | [{c['lower05']['six_head']:+.2f}, **{c['upper95']['six_head']:.2f}**] "
                 f"| {c['observed']['triage']:+.2f} | [{c['lower05']['triage']:+.2f}, **{c['upper95']['triage']:.2f}**] |")
    po = p["pooled"]
    L += ["", "## Pooled (equal-weight mean over the 8 cells)", "",
          "| curve | mean excess | [5%, **95%**] |", "|---|---|---|"]
    for k, lab in (("six_head", "six-head"), ("triage", "triage")):
        L.append(f"| {lab} | {po[k]['observed']:+.2f} | [{po[k]['lower05']:+.2f}, **{po[k]['upper95']:.2f}**] |")
    L += ["", "An excess can be negative: a policy fixed in advance can land below the mixture frontier.", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--out", type=Path, default=OUT_MD)
    ap.add_argument("--json-out", type=Path, default=OUT_JSON)
    args = ap.parse_args()
    if args.n_boot != N_BOOT and (args.out == OUT_MD or args.json_out == OUT_JSON):
        raise SystemExit("a non-default --n-boot must not overwrite the tracked product")
    specs = list(CELLS) + list(rt.WA_CELLS)
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        cells = list(ex.map(run_cell, specs, [args.n_boot] * len(specs)))
    pooled = {}
    for k in ("six_head", "triage"):
        draws = np.mean([c["_boot"][k] for c in cells], axis=0)
        pooled[k] = {"observed": float(np.mean([c["observed"][k] for c in cells])),
                     "upper95": float(np.quantile(draws, 0.95)), "lower05": float(np.quantile(draws, 0.05))}
    ref = rerun_reference()
    for c in cells:
        c.pop("_boot")
        c["rerun_abs_diff_pp"] = ref.get(c["cell_id"], [])
    payload = {"post_hoc_exploratory": True, "producer": "scripts/analysis/routing_gain_upper_bounds.py",
               "n_boot": args.n_boot, "seed": SEED, "cells": cells, "pooled": pooled}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    args.out.write_text(render(payload), encoding="utf-8")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
