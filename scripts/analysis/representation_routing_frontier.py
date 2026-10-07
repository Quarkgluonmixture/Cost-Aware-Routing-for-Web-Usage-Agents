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

Scope: the 8 cross-mode units (product_scope.yaml XMODE). Features: the matched 18 of
`router_triage_learnability --with-wa` on every cell (WA has no reference images and no
reasoning annotation), so VWA and WA rows are fitted on the same columns.

post_hoc_exploratory=True, h10_eligible=False. Touches no gating producer.

Usage (writes both files by default):
  python scripts/analysis/representation_routing_frontier.py
  python scripts/analysis/representation_routing_frontier.py --n-shuffle 200 --fig /tmp/f.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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

OUT_MD = REPO / "docs/analysis/cross_sites/representation_routing_frontier.md"
OUT_JSON = REPO / "docs/analysis/cross_sites/representation_routing_frontier.json"

SITE_KEY = {"classifieds": "cls", "reddit": "red", "wa_reddit": "wared"}
SITE_LABEL = {"classifieds": "classifieds", "reddit": "reddit", "wa_reddit": "wa_reddit"}


# ---------------------------------------------------------------------------- data

def load_cell(spec: dict) -> dict | None:
    cell = rt.build_wa_cell(spec) if spec.get("_wa") else rt.build_cell(spec)
    if cell is None:
        return None
    S = np.array([[bool(s[m]) for m in MODES] for s in cell["succ"]], dtype=float)
    C = np.array([[float(c[m]) for m in MODES] for c in cell["cost"]], dtype=float)
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
        fallback[f] = min(range(len(MODES)), key=lambda j: (-tr_sr[j], tr_cost[j], j))
        for j in range(len(MODES)):
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
        best[f] = min(range(len(MODES)), key=lambda j: (-tr_sr[j], tr_cost[j], j))
        cheap[f] = min(range(len(MODES)), key=lambda j: (tr_cost[j], j))
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
             for j, m in enumerate(MODES)}
    hull = fixed_hull([(v["cost"], v["sr_pct"]) for v in fixed.values()])
    hull_modes = [next(m for m, v in fixed.items() if (v["cost"], v["sr_pct"]) == h) for h in hull]

    six = six_head_curve(X, S, C, folds)
    tri = triage_curve(X, y, S, C, folds)
    orc = oracle_curve(S, C)
    s_six = excess_summary(six, hull, "tau")
    s_tri = excess_summary(tri, hull, "quantile")
    s_orc = excess_summary(orc, hull, "lambda")

    # Label-shuffle null for the max excess: permute the task bundle against X and redo
    # both curves end to end (fold-local fits, sweep, max). The hull is a property of the
    # cell's aggregate outcomes and is invariant to the permutation.
    rng = np.random.default_rng(SEED + 1)
    null_six, null_tri = [], []
    for _ in range(n_shuffle):
        perm = rng.permutation(n)
        yb, Sb, Cb = y[perm], S[perm], C[perm]
        null_six.append(excess_summary(six_head_curve(X, Sb, Cb, folds), hull, "tau")["max_excess_pp"])
        null_tri.append(excess_summary(triage_curve(X, yb, Sb, Cb, folds), hull, "quantile")["max_excess_pp"])

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
        "n_shuffle": n_shuffle,
    }


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
        "Cost is `total_billed_cost_usd` per task, **comparable within a cell only**.",
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
        "## 2. Per cell: the curves",
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


def plot(payload: dict, path: Path) -> None:
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
    axes.flat[0].legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor="white")


# ---------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_MD)
    ap.add_argument("--json-out", type=Path, default=OUT_JSON)
    ap.add_argument("--n-shuffle", type=int, default=N_SHUFFLE)
    ap.add_argument("--fig", type=Path, help="optional PNG of the eight planes (not tracked)")
    args = ap.parse_args()

    specs = list(CELLS) + list(rt.WA_CELLS)
    results = []
    for spec in specs:
        t0 = time.time()
        cell = load_cell(spec)
        if cell is None:
            raise RuntimeError(f"cell {spec} did not build; every XMODE unit is expected to")
        res = evaluate(cell, args.n_shuffle)
        print(f"{res['cell_id']}: six-head {res['six_head']['summary']['max_excess_pp']:+.2f}pp "
              f"triage {res['triage']['summary']['max_excess_pp']:+.2f}pp "
              f"oracle {res['oracle']['summary']['max_excess_pp']:+.2f}pp ({time.time() - t0:.0f}s)",
              file=sys.stderr)
        results.append(res)

    for key in ("six_head", "triage"):
        verdict = holm({r["cell_id"]: r[key]["summary"]["null_p"] for r in results})
        for r in results:
            r[key]["holm"] = bool(verdict.get(r["cell_id"], False))

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
            "excess_definition": "SR(point) - fixed-frontier SR at the same mean cost; below the "
                                 "cheapest fixed cost the cheapest mode's SR is used",
        },
        "cells": results,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    args.out.write_text(render(payload), encoding="utf-8")
    if args.fig:
        plot(payload, args.fig)
    print(f"wrote {args.out} and {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
