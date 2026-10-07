#!/usr/bin/env python3
"""Identification interval for the prospective budget-router test on shop_B1 — 2026-10-07,
post_hoc_exploratory (笔记 §542)

Why this exists. `budget_router_prospective_eval.py` truncates each episode at its cap and
counts a success only when the episode had ENDED within the cap (`_trunc`). That reads every
success that ran past the cap as lost. For a success where the agent never issued a stop
(`agent_finished=False`), the evaluator judged the state after the last step; whether that state
was already reached by the cap is not in the logs (cu_pareto_operating_envelope_audit_2026-09-22.md,
the "识别区间" section, which named this test as the one to re-check). The zero-preset Codex
review of 2026-10-07 asked for the interval of the *differences* the pre-declared criterion
reads, not of each policy's SR separately.

Per task t and cap c, the outcome under truncation is one of:
  success   c >= steps and the full episode succeeded
  failure   the full episode failed (monotonicity of failure is assumed: a state judged wrong at
            the end is not assumed to have been right earlier), or c = 0 (abstain), or c < steps
            and the agent stopped on its own at the end (its last actions are part of the
            solution — the 09-22 audit's convention)
  unknown   c < steps, the full episode succeeded, and agent_finished is False
Each unknown (t, c) is its own variable in [0, 1]; nothing is assumed about how unknowns at
different caps relate (no monotonicity in c). Every difference the criterion reads —
learned − fixed-cap, learned − random — is linear in those variables, and the variables are
SHARED by the policies that use the same (t, c), so its exact range is the constant plus the
sum of the negative (or positive) coefficients.

Baselines, as pre-declared in the eval script: fixed cap k with cost closest to the learned
policy's (k in 3..30, ties to the smaller k; cost is exact, unknowns do not move it); random =
the learned policy's caps randomly permuted over tasks. The eval script averages 200 draws; here
the expectation over all permutations is exact (each task gets cap c with probability equal to
the share of tasks the learned policy sends to c).

Sampling uncertainty, on top of identification: task bootstrap and template-cluster bootstrap
(B=2000, seed 0), recomputing the matched fixed cap and the random expectation per resample;
the reported interval is [2.5th percentile of the lower bound, 97.5th of the upper bound], and
a Bonferroni version over the four primary comparisons (2 arms × 2 baselines).

Usage:
  python scripts/analysis/budget_router_identification.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from p79.experiment.io_utils import read_jsonl_dedup  # noqa: E402
from scripts.analysis import budget_router_prospective_eval as bre  # noqa: E402
from scripts.analysis.lib.canonical_task_universe import expected_scored_ids  # noqa: E402

ARMS = {
    "ptext": "results/visualwebarena/phase1/B1_phantom_text_shopping_20260908",
    "pprompt": "results/visualwebarena/phase1/B1_phantom_prompt_shopping_20260911",
}
PRIMARY_POLICY = "two_tier"
N_BOOT = 2000
OUT_MD = REPO / "docs/analysis/vwa_shopping/budget_router_identification.md"
OUT_JSON = REPO / "docs/analysis/vwa_shopping/budget_router_identification.json"


# ---------------------------------------------------------------------------- data

def load_arm(name: str, run: str, seen: set[int], scored: set[int]) -> list[dict]:
    """Same episode set as `budget_router_prospective_eval.evaluate`, PRIMARY prospective only."""
    eps = []
    for sp in glob.glob(os.path.join(REPO / run, "*", "episodes", "*_steps_v2.jsonl")):
        tid = int(os.path.basename(sp).split("_task_")[1].split("_steps")[0])
        summ = sp.replace("_steps_v2.jsonl", "_summary_v2.json")
        if not os.path.exists(summ) or tid not in scored:
            continue
        if name == "ptext" and tid in seen:
            continue
        recs = read_jsonl_dedup(sp)
        with open(summ, encoding="utf-8") as f:
            sj = json.load(f)
        if "agent_finished" not in sj:
            raise KeyError(f"{summ}: no agent_finished field; the interval cannot be formed")
        eps.append({"task": tid, "success": bool(sj.get("success")),
                    "finished": bool(sj["agent_finished"]),
                    "cost": [(r.get("cost_usd") or {}).get("total") or 0 for r in recs]})
    return sorted(eps, key=lambda e: e["task"])


def templates() -> dict[int, int]:
    out = {}
    for f in glob.glob(str(REPO / bre.TASK_CFG / "*.json")):
        with open(f, encoding="utf-8") as fh:
            j = json.load(fh)
        if j.get("task_id") is not None:
            out[int(j["task_id"])] = int(j["intent_template_id"])
    return out


# ---------------------------------------------------------------------------- linear forms

def status(e: dict, cap: int) -> str:
    n = len(e["cost"])
    if cap <= 0 or not e["success"]:
        return "fail"
    if cap >= n:
        return "success"
    return "fail" if e["finished"] else "unknown"


def sr_form(eps: list[dict], weights: list[dict[int, float]]) -> tuple[float, dict]:
    """SR (%) as const + Σ coef·u over unknown (task, cap) variables. weights[i] maps cap ->
    probability that episode i runs at that cap (one entry for a deterministic policy)."""
    n = len(eps)
    const, coef = 0.0, {}
    for e, w in zip(eps, weights):
        for cap, p in w.items():
            s = status(e, cap)
            if s == "success":
                const += p
            elif s == "unknown":
                key = (e["task"], cap)
                coef[key] = coef.get(key, 0.0) + p
    return 100 * const / n, {k: 100 * v / n for k, v in coef.items()}


def diff_range(a: tuple[float, dict], b: tuple[float, dict]) -> tuple[float, float]:
    """Exact range of SR(a) − SR(b) over shared unknowns in [0, 1]."""
    c = a[0] - b[0]
    keys = set(a[1]) | set(b[1])
    d = [a[1].get(k, 0.0) - b[1].get(k, 0.0) for k in keys]
    return c + sum(min(0.0, v) for v in d), c + sum(max(0.0, v) for v in d)


def level_range(a: tuple[float, dict]) -> tuple[float, float]:
    return a[0], a[0] + sum(a[1].values())


def mean_cost(eps: list[dict], caps: list[int]) -> float:
    return sum(sum(e["cost"][:min(c, len(e["cost"]))]) for e, c in zip(eps, caps)) / len(eps)


def evaluate(eps: list[dict], caps: list[int]) -> dict:
    full = sr_form(eps, [{10 ** 6: 1.0}] * len(eps))[0]
    learned = sr_form(eps, [{c: 1.0} for c in caps])
    c_learned = mean_cost(eps, caps)
    k = min((abs(mean_cost(eps, [kk] * len(eps)) - c_learned), kk) for kk in range(3, 31))[1]
    fixed = sr_form(eps, [{k: 1.0}] * len(eps))
    shares = {c: caps.count(c) / len(caps) for c in set(caps)}
    random = sr_form(eps, [shares] * len(eps))
    # loss_learned − loss_baseline = SR_baseline − SR_learned; the criterion wants it < 0
    return {
        "n": len(eps), "full_sr_pct": full, "fixed_cap": k,
        "cost_saved_pct": 100 * (1 - c_learned / mean_cost(eps, [10 ** 6] * len(eps))),
        "sr_pct": {"learned": level_range(learned), "fixed": level_range(fixed),
                   "random": level_range(random)},
        "learned_minus_fixed_loss_pp": diff_range(fixed, learned),
        "learned_minus_random_loss_pp": diff_range(random, learned),
        "n_unknown_vars": len(set(learned[1]) | set(fixed[1]) | set(random[1])),
    }


def bootstrap(eps: list[dict], caps: list[int], groups: list[int], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    uniq = sorted(set(groups))
    members = {g: [i for i, x in enumerate(groups) if x == g] for g in uniq}
    out = {"fixed": [], "random": []}
    for _ in range(N_BOOT):
        idx = [i for g in rng.choice(uniq, size=len(uniq), replace=True) for i in members[g]]
        r = evaluate([eps[i] for i in idx], [caps[i] for i in idx])
        out["fixed"].append(r["learned_minus_fixed_loss_pp"])
        out["random"].append(r["learned_minus_random_loss_pp"])

    def ci(pairs, alpha):
        lo = np.quantile([p[0] for p in pairs], alpha / 2)
        hi = np.quantile([p[1] for p in pairs], 1 - alpha / 2)
        return [float(lo), float(hi)]

    return {k: {"ci95": ci(v, 0.05), "ci_bonferroni4": ci(v, 0.05 / 4)} for k, v in out.items()}


# ---------------------------------------------------------------------------- render

def _iv(r, nd=2):
    lo, hi = r
    return f"{lo:+.{nd}f}" if abs(hi - lo) < 1e-9 else f"[{lo:+.{nd}f}, {hi:+.{nd}f}]"


def render(payload: dict) -> str:
    L = [
        "# Budget router on shop_B1 — identification interval for the prospective test",
        "",
        "Generated by `scripts/analysis/budget_router_identification.py`. `post_hoc_exploratory=True`. "
        "Chronicle: 实验笔记 §542. The pre-declared test and its point readings are in "
        "`budget_router_prospective_eval.py` (§510.3 / §515.3); this file adds what the logs cannot decide.",
        "",
        "A success that ran past its cap without the agent stopping on its own is **unknown** under "
        "truncation: the evaluator judged the final state, and when that state was first reached is "
        "not logged. Each unknown (task, cap) is a free variable in [0, 1], shared by every policy "
        "that runs that task at that cap; no monotonicity in the cap is assumed. A success that ran "
        "past its cap and ended with the agent's own stop counts as lost (the 09-22 audit's "
        "convention); a failure counts as a failure at every cap. The point readings of the eval "
        "script are the corner where every unknown is lost.",
        "",
        "Differences are **loss(learned) − loss(baseline)** in SR pp; the pre-declared criterion "
        "wants both below zero. Random = the learned caps permuted over tasks, exact expectation.",
        "",
        "| arm | policy | n | full SR | unknown vars | learned − fixed cap (k) | learned − random | criterion holds for every completion? |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm, by_pol in payload["arms"].items():
        for pol, r in by_pol.items():
            f, rd = r["learned_minus_fixed_loss_pp"], r["learned_minus_random_loss_pp"]
            if f[1] < 0 and rd[1] < 0:
                verdict = "yes"
            elif f[0] >= 0 or rd[0] >= 0:
                verdict = "no — fails for every completion"
            else:
                verdict = "no — depends on the unknowns"
            L.append(f"| {arm} | {pol}{' (primary)' if pol == PRIMARY_POLICY else ''} | {r['n']} "
                     f"| {r['full_sr_pct']:.2f}% | {r['n_unknown_vars']} | {_iv(f)} (cap {r['fixed_cap']}) "
                     f"| {_iv(rd)} | {verdict} |")
    L += ["", "SR levels (%; interval where unknowns move it):", "",
          "| arm | policy | learned | fixed cap | random | cost saved |", "|---|---|---|---|---|---|"]
    for arm, by_pol in payload["arms"].items():
        for pol, r in by_pol.items():
            s = r["sr_pct"]
            L.append(f"| {arm} | {pol} | {_iv(s['learned'])} | {_iv(s['fixed'])} | {_iv(s['random'])} "
                     f"| {r['cost_saved_pct']:.0f}% |")
    L += ["", "## Sampling uncertainty (primary policy)", "",
          f"B={N_BOOT}, seed 0; the matched fixed cap and the random expectation are recomputed per "
          "resample. Interval = [2.5th percentile of the identification lower bound, 97.5th of the "
          "upper bound]; the Bonferroni column covers the four primary comparisons.",
          "",
          "| arm | resampling unit | learned − fixed: 95% | Bonferroni | learned − random: 95% | Bonferroni |",
          "|---|---|---|---|---|---|"]
    for arm, by_unit in payload["bootstrap"].items():
        for unit, b in by_unit.items():
            L.append(f"| {arm} | {unit} | {_iv(b['fixed']['ci95'])} | {_iv(b['fixed']['ci_bonferroni4'])} "
                     f"| {_iv(b['random']['ci95'])} | {_iv(b['random']['ci_bonferroni4'])} |")
    L += ["", f"Templates: {payload['n_templates']} on the scored shopping set.", ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------- main

def main() -> int:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    fz = json.loads(bre.FREEZE.read_text(encoding="utf-8"))
    seen = set(fz["seen_ptext_ids"])
    scored, _ = expected_scored_ids(bre.SITE)
    tmpl = templates()
    payload = {"post_hoc_exploratory": True,
               "producer": "scripts/analysis/budget_router_identification.py",
               "freeze": str(bre.FREEZE.relative_to(REPO)).replace("\\", "/"),
               "n_boot": N_BOOT, "arms": {}, "bootstrap": {},
               "n_templates": len({tmpl[t] for t in scored if t in tmpl})}
    for arm, run in ARMS.items():
        eps = load_arm(arm, run, seen, set(scored))
        payload["arms"][arm] = {}
        for pol in fz["policies"]:
            caps = [int(fz["tiers"][pol].get(str(e["task"]), 30)) for e in eps]
            payload["arms"][arm][pol] = evaluate(eps, caps)
        caps = [int(fz["tiers"][PRIMARY_POLICY].get(str(e["task"]), 30)) for e in eps]
        payload["bootstrap"][arm] = {
            "task": bootstrap(eps, caps, [e["task"] for e in eps], seed=0),
            "template": bootstrap(eps, caps, [tmpl[e["task"]] for e in eps], seed=0),
        }
        r = payload["arms"][arm][PRIMARY_POLICY]
        print(f"{arm}: n={r['n']} unknown={r['n_unknown_vars']} learned−fixed {_iv(r['learned_minus_fixed_loss_pp'])} "
              f"learned−random {_iv(r['learned_minus_random_loss_pp'])}", file=sys.stderr)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(f"wrote {OUT_MD.relative_to(REPO)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
