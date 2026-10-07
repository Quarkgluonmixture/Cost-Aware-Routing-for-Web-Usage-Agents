#!/usr/bin/env python3
"""Does a learned representation router's gain survive a rerun and an unseen template? —
2026-10-07, post_hoc_exploratory (笔记 §546)

Why this exists. The frontier product (§539 / §540) and its upper bounds (§544) fit and score a
router on ONE run per arm, with folds over tasks. Two things can make a small gain look real
there: the router may learn the run's own success noise (a label that a rerun rewrites), and it
may learn task templates (VWA tasks come in templates of near-identical intents; a task-level
fold puts siblings on both sides). The zero-preset Codex review of 2026-10-07 ranked this check
first.

Policy: the cross-fitted, deployable router of `routing_gain_upper_bounds.crossfit` — per outer
fold, the operating point (tau for six-head, quantile for triage) is chosen on the training rows
of the FIT run, then applied to the held-out rows; the decisions are scored on the EVAL run
against the eval run's own fixed frontier (six modes + random mixtures, in-sample, conservative
for the router).

Readings per cell:
  A→A task folds        the §544 estimand (reference; must match it)
  A→A template folds    same, folds grouped by intent_template_id — every cell
  A→B task / template   fit on run a, scored on run b — the three fully replicated cells
  B→A template          the symmetric check; NOT a second independent sample
  diagnostic            per task, the cheapest mode that succeeded on run a (else the cheapest
                        mode), scored on run b. It uses labels a deployment never has: it is
                        how much a single run's per-task preference persists, NOT a deployable
                        policy and NOT a Bayes bound (a predictor of the true success
                        probabilities would beat one noisy label).
Uncertainty: template-cluster bootstrap (B=500, seed 11) of the A→A and A→B template-fold
estimates, pipeline refitted per resample.

Usage:
  python scripts/analysis/routing_crossrun_template_validation.py
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
from scripts.analysis.lib.episode_rows import load_task_rows  # noqa: E402
from scripts.analysis.lib.replicate_pairs import cell_pairs, full_paired_cells  # noqa: E402
from scripts.analysis.routing_gain_upper_bounds import _excess, _hull  # noqa: E402

N_BOOT = 500
SEED = 11
OUT_MD = REPO / "docs/analysis/cross_sites/routing_crossrun_template_validation.md"
OUT_JSON = REPO / "docs/analysis/cross_sites/routing_crossrun_template_validation.json"
SITE_KEY = {"classifieds": "cls", "reddit": "red", "wa_reddit": "wared"}
PAIR_MODE = dict(zip(rt.SIX_MODES, ("dom", "som", "vision", "ptext", "pprompt", "psom")))


# ---------------------------------------------------------------------------- data

def templates(spec: dict, task_ids: list[int]) -> np.ndarray:
    out = []
    for t in task_ids:
        if spec.get("_wa"):
            cfg = rt._wa_task_config(spec["baseline"], t)
        else:
            f = Path(spec["modes"]["DOM"]).parents[1] / "task_configs" / f"{spec['site']}_task_{t}.json"
            cfg = json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
        if cfg is None or cfg.get("intent_template_id") is None:
            raise KeyError(f"{spec['site']} {spec['baseline']} task {t}: no intent_template_id")
        out.append(int(cfg["intent_template_id"]))
    return np.array(out)


def run_b(spec: dict, cell: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """(S_b, C_b) aligned to cell['task_ids'] and rrf.MODES, or None if not fully replicated.
    Fails loud if run a of the registered pairs is not the run load_cell read."""
    bl, sk = spec["baseline"], SITE_KEY[spec["site"]]
    if (bl, sk) not in full_paired_cells():
        return None
    pairs = cell_pairs(bl, sk)
    tids = cell["task_ids"]
    Sa = np.zeros((len(tids), len(rrf.MODES)))
    Sb, Cb = np.zeros_like(Sa), np.zeros_like(Sa)
    for j, m in enumerate(rrf.MODES):
        a_dir, b_dir = pairs[PAIR_MODE[m]]
        ra, rb = load_task_rows(a_dir / "episodes"), load_task_rows(b_dir / "episodes")
        for i, t in enumerate(tids):
            Sa[i, j] = bool(ra[t].get("success"))
            Sb[i, j] = bool(rb[t].get("success"))
            Cb[i, j] = float(rb[t][rt.COST_FIELD])
    if not np.array_equal(Sa, cell["S"]):
        raise RuntimeError(f"{bl}·{sk}: run a of CLEAN_PAIRS differs from the run load_cell read "
                           f"on {int((Sa != cell['S']).sum())} (task, mode) entries")
    return Sb, Cb


def folds_by(groups: np.ndarray) -> list[np.ndarray]:
    uniq = np.unique(groups)
    perm = np.random.default_rng(rrf.SEED).permutation(len(uniq))
    return [np.flatnonzero(np.isin(groups, p)) for p in np.array_split(uniq[perm], rrf.N_FOLDS)]


# ---------------------------------------------------------------------------- policies

def crossfit(X, Sf, Cf, Se, Ce, folds) -> dict:
    """Operating point chosen on the FIT run's training rows; decisions scored on the EVAL run."""
    n, m = Sf.shape
    yf = Sf.max(1).astype(int)
    sel6 = np.zeros(n, dtype=int)
    selt = np.zeros(n, dtype=int)
    for f in folds:
        tr = np.setdiff1d(np.arange(n), f)
        Str, Ctr = Sf[tr], Cf[tr]
        hull_tr = _hull(Str, Ctr)
        tr_cost, tr_sr = Ctr.mean(0), Str.mean(0)
        best = min(range(m), key=lambda j: (-tr_sr[j], tr_cost[j], j))
        cheap = min(range(m), key=lambda j: (tr_cost[j], j))
        P_tr = np.column_stack([rrf._fit_proba(X[tr], Str[:, j].astype(int), X[tr]) for j in range(m)])
        P_te = np.column_stack([rrf._fit_proba(X[tr], Str[:, j].astype(int), X[f]) for j in range(m)])

        def pick(P, tau):
            elig = P >= tau - rrf.EPS
            return np.where(elig.any(1), np.where(elig, tr_cost[None, :], np.inf).argmin(1), best)

        tau = max(rrf.TAUS, key=lambda t: (_excess(Str, Ctr, pick(P_tr, t), hull_tr)[0],
                                           -_excess(Str, Ctr, pick(P_tr, t), hull_tr)[1]))
        sel6[f] = pick(P_te, tau)
        s_tr = rrf._fit_proba(X[tr], yf[tr], X[tr])
        s_te = rrf._fit_proba(X[tr], yf[tr], X[f])
        thresholds = [float(np.quantile(s_tr, q)) for q in rrf.TRIAGE_QUANTILES] + [np.inf]

        def route(s, thr):
            return np.where(s < thr, cheap, best)

        thr = max(thresholds, key=lambda v: (_excess(Str, Ctr, route(s_tr, v), hull_tr)[0],
                                             -_excess(Str, Ctr, route(s_tr, v), hull_tr)[1]))
        selt[f] = route(s_te, thr)
    hull = _hull(Se, Ce)
    return {"six_head": _excess(Se, Ce, sel6, hull)[0], "triage": _excess(Se, Ce, selt, hull)[0]}


def persistence_diagnostic(Sa, Ca, Sb, Cb) -> float:
    cheap_order = np.argsort(Ca.mean(0), kind="stable")
    sel = np.empty(len(Sa), dtype=int)
    for i in range(len(Sa)):
        ok = [j for j in cheap_order if Sa[i, j]]
        sel[i] = ok[0] if ok else cheap_order[0]
    return _excess(Sb, Cb, sel, _hull(Sb, Cb))[0]


# ---------------------------------------------------------------------------- per cell

def run_cell(spec: dict, n_boot: int) -> dict:
    t0 = time.time()
    cell = rrf.load_cell(spec)
    X, S, C = cell["X"], cell["S"], cell["C"]
    n = len(S)
    cid = f"{rrf.SITE_KEY[cell['site']]}_{cell['baseline']}"
    tmpl = templates(spec, cell["task_ids"])
    tf = folds_by(tmpl)
    out = {"cell_id": cid, "n_tasks": n, "n_templates": int(len(np.unique(tmpl))),
           "a_a_task": crossfit(X, S, C, S, C, rrf.fold_split(n)),
           "a_a_template": crossfit(X, S, C, S, C, tf)}
    b = run_b(spec, cell)
    if b is not None:
        Sb, Cb = b
        out["a_b_task"] = crossfit(X, S, C, Sb, Cb, rrf.fold_split(n))
        out["a_b_template"] = crossfit(X, S, C, Sb, Cb, tf)
        out["b_a_template"] = crossfit(X, Sb, Cb, S, C, tf)
        out["diagnostic_a_pref_on_b"] = persistence_diagnostic(S, C, Sb, Cb)
        out["diagnostic_b_pref_on_a"] = persistence_diagnostic(Sb, Cb, S, C)
    rng = np.random.default_rng(SEED)
    uniq = np.unique(tmpl)
    members = {g: np.flatnonzero(tmpl == g) for g in uniq}
    boot = {"a_a_template": {"six_head": [], "triage": []}}
    if b is not None:
        boot["a_b_template"] = {"six_head": [], "triage": []}
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[g] for g in draw])
        # Label copies by the ORIGINAL template id, so every copy of a template drawn more than
        # once lands in one fold. Until 2026-10-08 this used the draw position, which split
        # copies of the same template (and so of the same tasks) across train and test
        # (found by the zero-preset Codex frame review; 笔记 §549).
        g = np.concatenate([np.full(len(members[t]), t) for t in draw])
        f = folds_by(g)
        r = crossfit(X[idx], S[idx], C[idx], S[idx], C[idx], f)
        for k in r:
            boot["a_a_template"][k].append(r[k])
        if b is not None:
            r = crossfit(X[idx], S[idx], C[idx], Sb[idx], Cb[idx], f)
            for k in r:
                boot["a_b_template"][k].append(r[k])
    out["template_boot"] = {est: {k: [float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95))]
                                  for k, v in d.items()} for est, d in boot.items()}
    print(f"{cid}: A→A task {out['a_a_task']} template {out['a_a_template']}"
          + (f" A→B template {out['a_b_template']}" if b is not None else "")
          + f" ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    return out


# ---------------------------------------------------------------------------- render

def render(p: dict) -> str:
    def iv(c, est, k):
        b = c["template_boot"].get(est)
        return "" if b is None else f" [{b[k][0]:+.2f}, {b[k][1]:+.2f}]"

    L = [
        "# Does the router's gain survive a rerun and an unseen template?",
        "",
        "Generated by `scripts/analysis/routing_crossrun_template_validation.py`. `post_hoc_exploratory=True`. "
        "Chronicle: 实验笔记 §546. Estimand: excess SR (pp) over the eval run's fixed frontier (six modes + random "
        "mixtures) of the cross-fitted, deployable router of `routing_gain_upper_bounds` (§544): operating point "
        "chosen on the fit run's training rows only.",
        "",
        f"Brackets: 90% template-cluster bootstrap (B={p['n_boot']}), pipeline refitted per resample.",
        "",
        "## 1. Template-held-out (every cell)",
        "",
        "| cell | n | templates | six-head: task folds | template folds [90%] | triage: task folds | template folds [90%] |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in p["cells"]:
        L.append(f"| {c['cell_id']} | {c['n_tasks']} | {c['n_templates']} "
                 f"| {c['a_a_task']['six_head']:+.2f} | {c['a_a_template']['six_head']:+.2f}{iv(c, 'a_a_template', 'six_head')} "
                 f"| {c['a_a_task']['triage']:+.2f} | {c['a_a_template']['triage']:+.2f}{iv(c, 'a_a_template', 'triage')} |")
    L += ["", "## 2. Fit on one run, score on its rerun (fully replicated cells)", "",
          "| cell | curve | A→A template | A→B task | A→B template [90%] | B→A template (symmetric check) |",
          "|---|---|---|---|---|---|"]
    for c in p["cells"]:
        if "a_b_task" not in c:
            continue
        for k, lab in (("six_head", "six-head"), ("triage", "triage")):
            L.append(f"| {c['cell_id']} | {lab} | {c['a_a_template'][k]:+.2f} | {c['a_b_task'][k]:+.2f} "
                     f"| {c['a_b_template'][k]:+.2f}{iv(c, 'a_b_template', k)} | {c['b_a_template'][k]:+.2f} |")
    L += ["", "## 3. Diagnostic: how much does one run's per-task preference persist?", "",
          "Per task, the cheapest mode that succeeded on one run (else the cheapest mode), scored on the other run "
          "against that run's frontier. Uses labels no deployment has — **not a deployable policy and not a Bayes "
          "bound**; it measures the persistence of a single run's preference.", "",
          "| cell | run a preference on run b | run b preference on run a |", "|---|---|---|"]
    for c in p["cells"]:
        if "diagnostic_a_pref_on_b" in c:
            L.append(f"| {c['cell_id']} | {c['diagnostic_a_pref_on_b']:+.2f} | {c['diagnostic_b_pref_on_a']:+.2f} |")
    L.append("")
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
    ub = REPO / "docs/analysis/cross_sites/routing_gain_upper_bounds.json"
    if ub.exists() and args.n_boot == N_BOOT:
        ref = {c["cell_id"]: c["observed"] for c in json.loads(ub.read_text(encoding="utf-8"))["cells"]}
        for c in cells:
            for k in ("six_head", "triage"):
                if abs(ref[c["cell_id"]][k] - c["a_a_task"][k]) > 1e-9:
                    raise RuntimeError(f"{c['cell_id']} {k}: A→A task folds {c['a_a_task'][k]} != §544 {ref[c['cell_id']][k]}")
    payload = {"post_hoc_exploratory": True, "producer": "scripts/analysis/routing_crossrun_template_validation.py",
               "n_boot": args.n_boot, "seed": SEED, "cells": cells}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    args.out.write_text(render(payload), encoding="utf-8")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
