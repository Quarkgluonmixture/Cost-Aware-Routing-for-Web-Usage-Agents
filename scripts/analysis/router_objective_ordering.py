#!/usr/bin/env python3
"""Does the ORDER you put SR and cost in change what routing is worth? — 2026-07-27

Motivated by two results that landed the same day:
  * H1 failed at k=6 (pooled drop-one 0.75pp vs the 1.0pp bar), so "P-SoM is the
    deployment arm" is off the table and the question becomes what ANY routing
    policy is worth on the (SR, cost) plane;
  * 笔记 §383.4 measured that ~1/4 of the oracle training labels are decided by
    the hardcoded `MODES` order in `router_features.py:101` rather than by data.
    Tie-breaking on cost instead is the principled fix, and it is exactly the
    "SR first, then cost" policy below.

A note on what is and is not a distinct policy. For a SINGLE-SHOT choice,
"maximize SR, break ties on cost" and "minimize cost among modes that succeed"
are the same rule — the cheapest solver is the first solver in cost order. The
orderings only genuinely diverge when one objective is allowed to override the
other (always-cheapest ignores SR entirely) or when the choice becomes
sequential (a cascade pays for each attempt). So four policies, not two synonyms.

Policies, all evaluated per (site, model) cell over that cell's canonical scored
universe:

  FIXED (no routing — what you get without a router)
    single:<mode>      each of the six modes on its own
    best_sr            the cell's highest-SR single mode
    cheapest           the cell's lowest-mean-cost single mode

  ORACLE (per-task choice with hindsight — upper bounds, not achievable)
    oracle_sr          succeed iff ANY mode succeeds; cost = mean over the mode
                       an SR-only oracle would pick (first solver in MODES order,
                       i.e. the arbitrary tie-break §383.4 measured)
    oracle_sr_cost     SR first, cost as tie-break: among solvers take the
                       cheapest; when nothing solves, take the cheapest overall
                       (an oracle that knows it will fail should waste least)
    cascade_cost_first cost first, escalate on failure: attempt modes in
                       ascending mean-cost order, stop at the first success;
                       cost is the SUM of attempts. This is the only ordering
                       where "cost first" is not just "ignore SR".

Cost estimand = `total_billed_cost_usd` (paper §1 canonical, memory
`project_cost_latency_canonical_estimand`). Costs are comparable WITHIN a cell
only — B0 bills a proxy API while B1/B2 are electricity-derived — so every
policy is scored inside its own cell and nothing is pooled across models.

post_hoc_exploratory: this is a descriptive policy comparison, NOT an H10
gating artifact. It does not touch any gating producer.

Usage:
  .venv/bin/python3 scripts/analysis/router_objective_ordering.py
  .venv/bin/python3 scripts/analysis/router_objective_ordering.py --out docs/analysis/cross_sites/router_objective_ordering.md
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.aggregate_phantom_lift import CELLS  # noqa: E402
from scripts.analysis.lib.canonical_task_universe import expected_scored_ids  # noqa: E402
from scripts.analysis.lib.episode_rows import load_cell_task_rows  # noqa: E402

SIX_MODES = ("DOM", "SoM", "Vision", "P-text", "P-prompt", "P-SoM")
COST_FIELD = "total_billed_cost_usd"


def _cell_matrix(cell: dict) -> tuple[list[int], dict, dict] | None:
    """Return (tasks, success[mode][task], cost[mode][task]) over the scored set."""
    universe, _ = expected_scored_ids(cell["site"])
    rows_by_mode = load_cell_task_rows(cell, modes=SIX_MODES)
    succ: dict[str, dict[int, bool]] = {}
    cost: dict[str, dict[int, float]] = {}
    for m in SIX_MODES:
        rows = rows_by_mode.get(m) or {}
        if not rows:
            return None
        succ[m] = {}
        cost[m] = {}
        for t in universe:
            r = rows.get(t)
            if r is None:
                return None
            succ[m][t] = r.get("success") is True
            c = r.get(COST_FIELD)
            cost[m][t] = float(c) if isinstance(c, (int, float)) else float("nan")
    return sorted(universe), succ, cost


def _mean(xs) -> float:
    xs = [x for x in xs if x == x]        # drop NaN
    return statistics.fmean(xs) if xs else float("nan")


def evaluate(cell: dict) -> dict | None:
    got = _cell_matrix(cell)
    if got is None:
        return None
    tasks, succ, cost = got
    n = len(tasks)

    per_mode = {
        m: {
            "sr_pct": 100.0 * sum(succ[m][t] for t in tasks) / n,
            "mean_cost": _mean(cost[m][t] for t in tasks),
        }
        for m in SIX_MODES
    }
    best_sr_mode = max(SIX_MODES, key=lambda m: per_mode[m]["sr_pct"])
    cheapest_mode = min(SIX_MODES, key=lambda m: per_mode[m]["mean_cost"])
    # Ascending mean cost — the escalation order a cost-first cascade would use.
    cost_order = sorted(SIX_MODES, key=lambda m: per_mode[m]["mean_cost"])

    policies: dict[str, dict] = {}
    for m in SIX_MODES:
        policies[f"single:{m}"] = dict(per_mode[m], kind="fixed")
    policies["best_sr"] = dict(per_mode[best_sr_mode], kind="fixed", pick=best_sr_mode)
    policies["cheapest"] = dict(per_mode[cheapest_mode], kind="fixed", pick=cheapest_mode)

    # --- the two SR-first variants differ ONLY in how they break a tie among
    # solvers. The fallback for tasks no mode solves is held identical (cheapest)
    # in both, because those tasks are 60-95% of the set here: letting the two
    # policies use different fallbacks would fold a "DOM vs Vision on unsolvable
    # tasks" difference into what is supposed to measure the tie-break alone.
    # (An earlier version of this script did exactly that and inflated the gap.)
    solved = 0
    cost_arbitrary = []      # tie-break = hardcoded MODES order (§383.4)
    cost_cheapest = []       # tie-break = cost
    tie_savings = []
    n_ties = 0
    for t in tasks:
        solvers = [m for m in SIX_MODES if succ[m][t]]
        fallback = min(cost[m][t] for m in SIX_MODES)
        if not solvers:
            cost_arbitrary.append(fallback)
            cost_cheapest.append(fallback)
            continue
        solved += 1
        c_arb = cost[solvers[0]][t]                              # first in MODES order
        c_cheap = cost[min(solvers, key=lambda m: cost[m][t])][t]
        cost_arbitrary.append(c_arb)
        cost_cheapest.append(c_cheap)
        if len(solvers) > 1:
            n_ties += 1
            tie_savings.append(c_arb - c_cheap)
    sr = 100.0 * solved / n
    policies["oracle_sr"] = {
        "kind": "oracle", "sr_pct": sr, "mean_cost": _mean(cost_arbitrary),
        "n_multi_solver_tasks": n_ties,
        "note": "tie among solvers broken by hardcoded MODES order",
    }
    policies["oracle_sr_cost"] = {
        "kind": "oracle", "sr_pct": sr, "mean_cost": _mean(cost_cheapest),
        "n_multi_solver_tasks": n_ties,
        "mean_saving_per_tied_task": _mean(tie_savings) if tie_savings else 0.0,
        "total_saving_over_set": sum(tie_savings) if tie_savings else 0.0,
        "note": "identical fallback to oracle_sr on unsolved tasks; only the tie-break differs",
    }

    # --- Decomposition: where does the oracle's cost advantage actually come from?
    #
    # `oracle_sr_cost` beats best-single on both axes, but the win has two very
    # differently-learnable halves, and lumping them hides the one that matters:
    #
    #   triage_only  keep the best-SR mode, but drop to the cheapest mode on tasks
    #                NO mode solves. SR is unchanged by construction; all the gain
    #                is cost. The label it needs is binary "is this task solvable
    #                by anything", available for EVERY task in the cell (n=203/224)
    #                — a completely different label-supply regime from the
    #                which-mode label that 笔记 §383.4 showed is unlearnable at
    #                16-97 labels per cell.
    #   route_only   choose optimally among solvers, but stay on the best-SR mode
    #                when nothing solves. All the SR gain, little of the cost gain.
    #                This is the half that needs the scarce which-mode label.
    tri_cost, rte_cost = [], []
    for t in tasks:
        solvers = [m for m in SIX_MODES if succ[m][t]]
        if solvers:
            tri_cost.append(cost[best_sr_mode][t])
            rte_cost.append(cost[min(solvers, key=lambda m: cost[m][t])][t])
        else:
            cheap = min(cost[m][t] for m in SIX_MODES)
            tri_cost.append(cheap)
            rte_cost.append(cost[best_sr_mode][t])
    policies["triage_only"] = {
        "kind": "oracle-half", "sr_pct": per_mode[best_sr_mode]["sr_pct"],
        "mean_cost": _mean(tri_cost),
        "note": "best-SR mode, cheapest on unsolvable; needs only a solvable/not label",
    }
    policies["route_only"] = {
        "kind": "oracle-half", "sr_pct": sr, "mean_cost": _mean(rte_cost),
        "note": "optimal among solvers, best-SR mode on unsolvable; needs the scarce which-mode label",
    }
    n_unsolvable = sum(1 for t in tasks if not any(succ[m][t] for m in SIX_MODES))
    policies["_unsolvable"] = {"kind": "meta", "n": n_unsolvable,
                               "pct": 100.0 * n_unsolvable / n}

    # --- cascade_cost_first: escalate in ascending cost, pay for every attempt
    casc_solved = 0
    casc_cost = []
    attempts_hist: dict[int, int] = {}
    for t in tasks:
        spent = 0.0
        k = 0
        hit = False
        for m in cost_order:
            k += 1
            spent += cost[m][t]
            if succ[m][t]:
                hit = True
                break
        casc_solved += hit
        casc_cost.append(spent)
        attempts_hist[k] = attempts_hist.get(k, 0) + 1
    policies["cascade_cost_first"] = {
        "kind": "oracle-free", "sr_pct": 100.0 * casc_solved / n,
        "mean_cost": _mean(casc_cost),
        "attempts_hist": dict(sorted(attempts_hist.items())),
        "escalation_order": list(cost_order),
    }

    return {
        "site": cell["site"], "baseline": cell["baseline"], "n_tasks": n,
        "best_sr_mode": best_sr_mode, "cheapest_mode": cheapest_mode,
        "policies": policies,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    results = [r for r in (evaluate(c) for c in CELLS) if r]
    L: list[str] = ["# Routing policies — does the objective ORDER matter?\n",
                    "Generated by `scripts/analysis/router_objective_ordering.py`. "
                    "`post_hoc_exploratory=True`, `h10_eligible=False` — descriptive, "
                    "not a gating artifact.\n",
                    "Cost = `total_billed_cost_usd` (paper §1 canonical estimand), mean per "
                    "task, **comparable within a cell only** (B0 bills a proxy API; B1/B2 are "
                    "electricity-derived).\n"]

    for r in results:
        p = r["policies"]
        L.append(f"\n## {r['site']} · {r['baseline']}  (n={r['n_tasks']})\n")
        L.append("| policy | kind | SR % | mean cost | ΔSR vs best-single | Δcost vs best-single |")
        L.append("|---|---|---|---|---|---|")
        b = p["best_sr"]
        for name in ([f"single:{m}" for m in SIX_MODES]
                     + ["cheapest", "best_sr", "triage_only", "route_only",
                        "oracle_sr", "oracle_sr_cost", "cascade_cost_first"]):
            v = p[name]
            L.append(
                f"| `{name}` | {v['kind']} | {v['sr_pct']:.2f} | {v['mean_cost']:.5f} | "
                f"{v['sr_pct'] - b['sr_pct']:+.2f}pp | "
                f"{100.0 * (v['mean_cost'] / b['mean_cost'] - 1):+.1f}% |"
            )
        osr, osc = p["oracle_sr"], p["oracle_sr_cost"]
        L.append(
            f"\n- **Tie-break, isolated**: both rows have identical SR ({osr['sr_pct']:.2f}%) "
            f"and identical treatment of the tasks nothing solves, so the whole gap is the "
            f"tie-break: {osr['mean_cost']:.5f} → {osc['mean_cost']:.5f} "
            f"({100.0*(osc['mean_cost']/osr['mean_cost']-1):+.1f}%). "
            f"Only **{osc['n_multi_solver_tasks']} of {r['n_tasks']}** tasks have >1 solver — "
            f"those are the ones §383.4's hardcoded `MODES` order was silently deciding — "
            f"and switching their tie-break to cost saves "
            f"{osc['mean_saving_per_tied_task']:.5f} each."
        )
        tri, rte, un = p["triage_only"], p["route_only"], p["_unsolvable"]
        L.append(
            f"- **Where the oracle's advantage comes from**: {un['n']}/{r['n_tasks']} "
            f"({un['pct']:.1f}%) of tasks are solved by NO mode. `triage_only` — keep the "
            f"best-SR mode but spend the cheapest on those — gives "
            f"{100.0*(tri['mean_cost']/b['mean_cost']-1):+.1f}% cost at **zero** SR change, "
            f"and needs only a binary solvable/not label (available for all {r['n_tasks']} "
            f"tasks). `route_only` — choose among solvers, best-SR elsewhere — gives the "
            f"{rte['sr_pct'] - b['sr_pct']:+.2f}pp SR at "
            f"{100.0*(rte['mean_cost']/b['mean_cost']-1):+.1f}% cost, and needs the "
            f"which-mode label that only exists on the {r['n_tasks'] - un['n']} solved tasks."
        )
        c = p["cascade_cost_first"]
        L.append(
            f"- **cost-first cascade** reaches {c['sr_pct']:.2f}% "
            f"({c['sr_pct'] - b['sr_pct']:+.2f}pp vs best single) at "
            f"{100.0*(c['mean_cost']/b['mean_cost']-1):+.1f}% cost; escalation order "
            f"{' → '.join(c['escalation_order'])}; attempts histogram {c['attempts_hist']}."
        )

    text = "\n".join(L) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps({"post_hoc_exploratory": True, "h10_eligible": False,
                        "cost_field": COST_FIELD, "cells": results},
                       indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
