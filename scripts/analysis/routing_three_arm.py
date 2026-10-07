#!/usr/bin/env python3
"""The routing suite on the deployment question itself: DOM vs Vision vs SoM — 2026-10-08,
post_hoc_exploratory (笔记 §550)

Why this exists. The practical question the paper asks is three-way: text (DOM), screenshot
(Vision) or both (SoM); practitioners do not distinguish the four screenshot-free variants
(user, 2026-08-01). Every routing reading so far (§539–§546) was computed over all six arms, so
the evidence answered a six-way question the text does not ask. The zero-preset Codex frame
review of 2026-10-08 flagged the mismatch. This product recomputes the same readings with the
decision space restricted to the three deployment arms. Nothing else changes: same cells, same
tasks, same features, same seeds, same estimators.

Readings:
  frontier     §539 / §540 — learned curves vs the three fixed arms and their random mixtures;
               label-shuffle null B=1000; Holm across the 8 cells per curve; pooled frontier gain
               on the normalised budget with its pooled null.
  deployable   §544 — operating point chosen on training folds; task bootstrap B=1000; pooled.
  template     §546 — deployable excess with folds grouped by intent_template_id; A→B (fit on
               run a, score on run b) on the three fully replicated cells.
  null model   §543 — no-interaction (Rasch) null conditioned on margins, curveball sampling,
               B=2000, on the three fully replicated cells; Holm across them; decomposition and
               D-study.

Usage:
  python scripts/analysis/routing_three_arm.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis import representation_routing_frontier as rrf  # noqa: E402
from scripts.analysis import router_triage_learnability as rt  # noqa: E402
from scripts.analysis import task_mode_interaction_null as tmn  # noqa: E402
from scripts.analysis.aggregate_phantom_lift import CELLS  # noqa: E402
from scripts.analysis.lib.replicate_pairs import full_paired_cells  # noqa: E402
from scripts.analysis.routing_crossrun_template_validation import crossfit as crossfit_eval  # noqa: E402
from scripts.analysis.routing_crossrun_template_validation import folds_by, run_b, templates  # noqa: E402
from scripts.analysis.routing_gain_upper_bounds import crossfit, group_folds  # noqa: E402

ARMS = ("DOM", "SoM", "Vision")
ARM_IDX = [rrf.MODES.index(m) for m in ARMS]
NULL_KEYS = ("dom", "som", "vision")
NULL_IDX = [tmn.MODE_KEYS.index(k) for k in NULL_KEYS]
N_SHUFFLE = 1000
N_BOOT = 1000
SEED_BOOT = 7
OUT_MD = REPO / "docs/analysis/cross_sites/routing_three_arm.md"
OUT_JSON = REPO / "docs/analysis/cross_sites/routing_three_arm.json"


def run_cell(spec: dict, n_shuffle: int, n_boot: int) -> dict:
    t0 = time.time()
    full = rrf.load_cell(spec)
    S, C = full["S"][:, ARM_IDX], full["C"][:, ARM_IDX]
    X = full["X"]
    y = S.max(1).astype(int)
    cell = dict(full, S=S, C=C, y=y, modes=list(ARMS))
    n = len(y)
    fr = rrf.evaluate(cell, n_shuffle)
    dep = crossfit(X, y, S, C, rrf.fold_split(n))
    rng = np.random.default_rng(SEED_BOOT)
    boot = {"six_head": [], "triage": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        st = crossfit(X[idx], y[idx], S[idx], C[idx], group_folds(idx))
        for k in boot:
            boot[k].append(st[k])
    tf = folds_by(templates(spec, full["task_ids"]))
    out = {
        "cell_id": fr["cell_id"], "n_tasks": n,
        "frontier": {k: fr[k] for k in ("fixed_modes", "fixed_hull", "six_head", "triage", "oracle")},
        "frontier_gain": fr["frontier_gain"], "_null_gain": fr["_null_gain"],
        "deployable": {"observed": dep, "_boot": boot},
        "template_folds": crossfit_eval(X, S, C, S, C, tf),
    }
    b = run_b(spec, full)
    if b is not None:
        Sb, Cb = b[0][:, ARM_IDX], b[1][:, ARM_IDX]
        out["a_b_template"] = crossfit_eval(X, S, C, Sb, Cb, tf)
    print(f"{out['cell_id']}: arm-selector max {fr['six_head']['summary']['max_excess_pp']:+.2f} "
          f"(p {fr['six_head']['summary']['null_p']:.3f}) triage max {fr['triage']['summary']['max_excess_pp']:+.2f} "
          f"(p {fr['triage']['summary']['null_p']:.3f}) · deployable {dep['six_head']:+.2f}/{dep['triage']:+.2f} "
          f"({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    return out


def null_cell(cell: tuple[str, str]) -> dict:
    bl, sk = cell
    tasks, Y6 = tmn.load(bl, sk)
    Y = Y6[:, NULL_IDX, :]
    obs = tmn.statistics(Y)
    nd = tmn.null_draws(Y, tmn.N_NULL, tmn.THIN_PER_ROW, tmn.SEED)
    v = [d["sigma2_int"] for d in nd]
    dec = tmn.anova(Y)
    return {"cell_id": tmn.cell_id(bl, sk), "n_tasks": len(tasks),
            "sigma2_int": obs["sigma2_int"], "null_mean": float(np.mean(v)),
            "null_q95": float(np.quantile(v, 0.95)), "p_upper": tmn.p_upper(obs["sigma2_int"], v),
            "decomposition": dec, "decomposition_ci95": tmn.bootstrap(Y, tmn.SEED),
            "k_needed": tmn.k_needed(dec["interaction"], dec["noise"]),
            "reliability_k1": tmn.reliability(dec["interaction"], dec["noise"], 1)}


def render(p: dict) -> str:
    cells = p["cells"]
    L = [
        "# The routing suite on the deployment question: DOM vs Vision vs SoM",
        "",
        "Generated by `scripts/analysis/routing_three_arm.py`. `post_hoc_exploratory=True`. Chronicle: 实验笔记 §550. "
        "Same cells, tasks, features, seeds and estimators as the six-arm suite (§539 / §543 / §544 / §546); only the "
        "decision space is restricted to the three deployment arms. `six-head` here means one success head per arm "
        "(three heads).",
        "",
        "## 1. Frontier: max excess over the three fixed arms and their random mixtures",
        "",
        f"Label-shuffle null B={p['n_shuffle']}; Holm across the {len(cells)} cells per curve.",
        "",
        "| cell | n | frontier arms | oracle max | arm-selector max (p) | Holm | triage max (p) | Holm |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in cells:
        f = c["frontier"]
        s6, st = f["six_head"]["summary"], f["triage"]["summary"]
        L.append(f"| {c['cell_id']} | {c['n_tasks']} | {' → '.join(h['mode'] for h in f['fixed_hull'])} "
                 f"| +{f['oracle']['summary']['max_excess_pp']:.2f} | {s6['max_excess_pp']:+.2f} ({s6['null_p']:.4f}) "
                 f"| {'pass' if c['holm']['six_head'] else '—'} | {st['max_excess_pp']:+.2f} ({st['null_p']:.4f}) "
                 f"| {'pass' if c['holm']['triage'] else '—'} |")
    po = p["pooled_frontier"]
    L += ["", "Pooled frontier gain over the 8 cells (normalised budget, §540 definition):", "",
          "| curve | pooled max gain | at u | p |", "|---|---|---|---|"]
    for k, lab in (("six_head", "arm-selector"), ("triage", "triage"), ("oracle", "oracle")):
        s = po[k]
        L.append(f"| {lab} | +{s['max_pp']:.2f} | {s['at_u']} | {s.get('null_p', float('nan')):.4f} |"
                 if "null_p" in s else f"| {lab} | +{s['max_pp']:.2f} | {s['at_u']} | — |")
    L += ["", "## 2. Deployable router (operating point chosen on training folds)", "",
          f"Task bootstrap B={p['n_boot']}: [5%, **95%**]. Template folds: same estimand, folds by template. "
          "A→B: fit on run a, scored on run b (fully replicated cells).", "",
          "| cell | arm-selector | [5%, **95%**] | template | A→B template | triage | [5%, **95%**] | template | A→B template |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        d = c["deployable"]
        ab = c.get("a_b_template")
        L.append(f"| {c['cell_id']} | {d['observed']['six_head']:+.2f} | [{d['lower05']['six_head']:+.2f}, "
                 f"**{d['upper95']['six_head']:.2f}**] | {c['template_folds']['six_head']:+.2f} "
                 f"| {'—' if ab is None else format(ab['six_head'], '+.2f')} "
                 f"| {d['observed']['triage']:+.2f} | [{d['lower05']['triage']:+.2f}, **{d['upper95']['triage']:.2f}**] "
                 f"| {c['template_folds']['triage']:+.2f} | {'—' if ab is None else format(ab['triage'], '+.2f')} |")
    pd_ = p["pooled_deployable"]
    L += ["", "| pooled (8 cells) | arm-selector | [5%, **95%**] | triage | [5%, **95%**] |", "|---|---|---|---|---|",
          f"| mean | {pd_['six_head']['observed']:+.2f} | [{pd_['six_head']['lower05']:+.2f}, **{pd_['six_head']['upper95']:.2f}**] "
          f"| {pd_['triage']['observed']:+.2f} | [{pd_['triage']['lower05']:+.2f}, **{pd_['triage']['upper95']:.2f}**] |",
          "", "## 3. No-interaction null on the three arms (fully replicated cells)", "",
          f"Same null and statistic as §543 (σ²_int = reproducible task×arm interaction), B={p['n_null']}; Holm across the cells.", "",
          "| cell | n | σ²_int | null mean / q95 | p | Holm | interaction / noise [95%] | single-run reliability | k for reliability ≥ 0.5 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c in p["null_cells"]:
        d, ci = c["decomposition"], c["decomposition_ci95"]
        r = d["interaction_over_noise"]
        k = c["k_needed"]
        L.append(f"| {c['cell_id']} | {c['n_tasks']} | {c['sigma2_int']:.4f} | {c['null_mean']:.4f} / {c['null_q95']:.4f} "
                 f"| {c['p_upper']:.4f} | {'pass' if c['holm'] else '—'} | {r:.2f} [{ci['interaction_over_noise'][0]:.2f}, "
                 f"{ci['interaction_over_noise'][1]:.2f}] | {c['reliability_k1']:.2f} | {'—' if k is None else int(k)} |")
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--n-shuffle", type=int, default=N_SHUFFLE)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--out", type=Path, default=OUT_MD)
    ap.add_argument("--json-out", type=Path, default=OUT_JSON)
    args = ap.parse_args()
    if (args.n_shuffle, args.n_boot) != (N_SHUFFLE, N_BOOT) and (args.out == OUT_MD or args.json_out == OUT_JSON):
        raise SystemExit("non-default B must not overwrite the tracked product")
    specs = list(CELLS) + list(rt.WA_CELLS)
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        cells = list(ex.map(run_cell, specs, [args.n_shuffle] * len(specs), [args.n_boot] * len(specs)))
        nulls = list(ex.map(null_cell, full_paired_cells()))
    for k in ("six_head", "triage"):
        verdict = rrf.holm({c["cell_id"]: c["frontier"][k]["summary"]["null_p"] for c in cells})
        for c in cells:
            c.setdefault("holm", {})[k] = bool(verdict[c["cell_id"]])
    pooled_frontier = rrf.pool(cells)
    pooled_dep = {}
    for k in ("six_head", "triage"):
        draws = np.mean([c["deployable"]["_boot"][k] for c in cells], axis=0)
        pooled_dep[k] = {"observed": float(np.mean([c["deployable"]["observed"][k] for c in cells])),
                         "lower05": float(np.quantile(draws, 0.05)), "upper95": float(np.quantile(draws, 0.95))}
    for c in cells:
        c.pop("_null_gain")
        b = c["deployable"].pop("_boot")
        c["deployable"]["lower05"] = {k: float(np.quantile(v, 0.05)) for k, v in b.items()}
        c["deployable"]["upper95"] = {k: float(np.quantile(v, 0.95)) for k, v in b.items()}
    nv = rrf.holm({c["cell_id"]: c["p_upper"] for c in nulls})
    for c in nulls:
        c["holm"] = bool(nv[c["cell_id"]])
    payload = {"post_hoc_exploratory": True, "producer": "scripts/analysis/routing_three_arm.py",
               "arms": list(ARMS), "n_shuffle": args.n_shuffle, "n_boot": args.n_boot, "n_null": tmn.N_NULL,
               "cells": cells, "pooled_frontier": pooled_frontier, "pooled_deployable": pooled_dep,
               "null_cells": nulls}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    args.out.write_text(render(payload), encoding="utf-8")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
