"""The routing ceiling table — what a perfect per-task choice could buy, and what it costs to say so.

This is the product behind the paper's second ablation table. It exists because the numbers it
needs were scattered across two products and had never been put on one page:

  * `router_objective_ordering.json` holds per-cell single-mode / oracle success and cost.
  * `noise_floor_inventory.json` holds the arm-matched marginal gain and the rerun floor.

Reading either alone invites the error the whole table is built to prevent: the headline ceiling
is a SIX-ARM union quoted against a ONE-ARM baseline, while the honest arm-matched comparison
(add one arm) lands in the same range as rerunning the arm you already have. Both columns are
therefore rendered side by side and neither is emitted without the other.

Two ceilings, deliberately both:

  A  success-rate ceiling   any mode solves it  ->  higher SR, but not separable from resampling
  B  cost ceiling           same tasks solved   ->  lower cost, unaffected by arm count

  B' triage                 keep the best mode, send only the unsolvable tasks to the cheapest.
                            SR is unchanged BY CONSTRUCTION (an unsolvable task stays unsolved),
                            so this is the one ceiling no arm-count objection reaches.

Leaked-success policy (user decision 2026-08-04, option B):
    `require_reset` is a no-op on reddit, so subscriptions accumulate across a run's episodes and
    a later task can be scored against state an earlier one created. `audit_reddit_sidebar_leakage`
    identifies the scored successes credited without the episode ever visiting the forum the
    evaluator reads. Those episodes are set to **0 with the denominator unchanged** — an attempted
    and unaccomplished task is a 0, not a missing row. Both policies are computed and printed;
    the zeroed one is primary. Nothing is silently replaced.

    Scope note: this script applies the policy to ITS OWN table. Other products still carry the
    leak-kept figures; `leakage_sensitivity` is where that difference is reported for the fusion
    contrasts. Stated here rather than left implicit.

Regenerate:
    .venv/bin/python3 scripts/analysis/aggregate_routing_ceiling.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.aggregate_phantom_lift import CELLS  # noqa: E402
from scripts.analysis.lib.canonical_task_universe import expected_scored_ids  # noqa: E402
from scripts.analysis.router_objective_ordering import (  # noqa: E402
    SIX_MODES, WA_CELLS, _cell_matrix, _mean, _wa_matrix,
)

# WebArena has no AMENDMENT exclusion list; its universe is the six-mode intersection,
# the convention `aggregate_fusion_premium.load_wa` also uses.
WA_UNIVERSE_N = 104

AUDIT_JSON = REPO / "docs/analysis/cross_sites/reddit_sidebar_leakage_audit_with_wa.json"
AUDIT_FALLBACK = REPO / "docs/analysis/cross_sites/reddit_sidebar_leakage_audit.json"
FLOOR_JSON = REPO / "docs/analysis/cross_sites/noise_floor_inventory.json"
OUT_JSON = REPO / "docs/analysis/cross_sites/routing_ceiling.json"
OUT_MD = REPO / "docs/analysis/cross_sites/routing_ceiling.md"

# audit writes DOM / SoM / Vision / P-text ...; the matrices are keyed the same way, so the
# only mapping needed is the cell id.
CELL_KEY = {"classifieds": "cls", "reddit": "red", "wa_reddit": "wa_red"}


class MissingInput(RuntimeError):
    """Fail loud rather than emit a ceiling computed over the wrong set."""


def load_leaks() -> set[tuple[str, str, int]]:
    """Read leaked (cell_key, mode, task) triples from the audit's own JSON.

    Not hardcoded: if the audit is rerun after a universe or ruleset change this must follow
    it. A hardcoded list would keep answering the previous question.
    """
    src = AUDIT_JSON if AUDIT_JSON.is_file() else AUDIT_FALLBACK
    if not src.is_file():
        raise MissingInput(f"{src} missing — run audit_reddit_sidebar_leakage.py --with-wa first")
    d = json.loads(src.read_text())
    rows = d.get("rows")
    if rows is None:
        raise MissingInput(f"{src}: no `rows` key — audit output shape changed")
    out: set[tuple[str, str, int]] = set()
    for r in rows:
        if r.get("verdict") != "LEAKED":
            continue
        # The audit also lists passive-satisfiable successes on AMENDMENT_08-excluded tasks.
        # Those are already outside every SR in the paper; zeroing them here would double-count
        # the exclusion.
        if not r.get("in_scored_universe", True):
            continue
        # `baseline` is B0/B1/B2 and the audit is reddit-only, so the VWA cell key is red_<b>.
        # A WA row would carry its own site; handled defensively rather than assumed absent.
        site = str(r.get("site") or "reddit")
        key = f"{CELL_KEY.get(site, site)}_{r['baseline']}"
        out.add((key, str(r["mode"]), int(r["task_id"])))
    if not out:
        raise MissingInput(f"{src}: parsed but contained no in-universe LEAKED rows")
    declared = d.get("n_leaked")
    if declared is not None and len(out) != declared:
        print(f"[note] audit declares n_leaked={declared}; {len(out)} are in the scored "
              f"universe — using the in-universe set")
    return out


def load_floor() -> dict[str, dict]:
    """Arm-matched marginal gain per cell, plus the rerun draws where a replicate exists."""
    if not FLOOR_JSON.is_file():
        raise MissingInput(f"{FLOOR_JSON} missing — run aggregate_noise_floor_inventory.py first")
    d = json.loads(FLOOR_JSON.read_text())
    margins = d.get("margins")
    if not margins:
        raise MissingInput(f"{FLOOR_JSON}: no `margins` key")
    reruns: dict[str, list[float]] = {}
    for p in d.get("clean_pairs") or []:
        # label looks like "B0.cls.dom"; the replicate belongs to that (baseline, site) cell
        parts = str(p.get("label", "")).split(".")
        if len(parts) < 3:
            continue
        base, site_short = parts[0], parts[1]
        key = f"{'cls' if site_short.startswith('cls') else 'red'}_{base}"
        for f in ("self_drop_a_to_b_pp", "self_drop_b_to_a_pp"):
            v = p.get(f)
            if isinstance(v, (int, float)):
                reruns.setdefault(key, []).append(float(v))
    wa = d.get("wa_floor") or {}
    wa_draws = [wa.get("self_drop_a_to_b_pp"), wa.get("self_drop_b_to_a_pp")]
    wa_draws = [float(v) for v in wa_draws if isinstance(v, (int, float))]
    return {"margins": margins, "reruns": reruns, "wa_reruns": wa_draws}


def _apply_leaks(cell_key: str, succ: dict, leaks: set) -> tuple[dict, int]:
    """Return a success matrix with leaked episodes set to 0. Denominator untouched."""
    out = {m: dict(v) for m, v in succ.items()}
    n = 0
    for (k, mode, task) in leaks:
        if k != cell_key:
            continue
        if mode not in out:
            raise MissingInput(f"{cell_key}: leaked row names mode {mode!r}, not in {list(out)}")
        if task not in out[mode]:
            # The task is outside this matrix's universe — a real inconsistency, not a no-op.
            raise MissingInput(f"{cell_key}/{mode}: leaked task {task} absent from the matrix")
        if out[mode][task]:
            out[mode][task] = False
            n += 1
    return out, n


def evaluate(tasks: list[int], succ: dict, cost: dict) -> dict:
    """All ceilings for one cell under one leak policy."""
    n = len(tasks)
    per_mode = {m: {"sr_pct": 100 * sum(succ[m][t] for t in tasks) / n,
                    "mean_cost": _mean([cost[m][t] for t in tasks])} for m in SIX_MODES}
    best_mode = max(SIX_MODES, key=lambda m: per_mode[m]["sr_pct"])
    cheap_mode = min(SIX_MODES, key=lambda m: per_mode[m]["mean_cost"])

    solved = {t: [m for m in SIX_MODES if succ[m][t]] for t in tasks}
    n_any = sum(1 for t in tasks if solved[t])
    n_multi = sum(1 for t in tasks if len(solved[t]) > 1)

    # A — success ceiling: solved iff any mode solves it. Cost is that of the cheapest solver,
    # which is the most favourable reading; the point of A is the SR, not its price.
    oracle_cost = _mean([min((cost[m][t] for m in solved[t]), default=cost[cheap_mode][t])
                         for t in tasks])

    # B' — triage: keep the best-SR mode everywhere EXCEPT tasks no mode solves, which go to the
    # cheapest. SR is unchanged by construction: an unsolvable task is unsolved either way.
    triage_cost = _mean([cost[cheap_mode][t] if not solved[t] else cost[best_mode][t]
                         for t in tasks])

    best_cost = per_mode[best_mode]["mean_cost"]
    return {
        "n_tasks": n,
        "best_mode": best_mode, "best_sr_pct": per_mode[best_mode]["sr_pct"],
        "best_mean_cost": best_cost,
        "cheapest_mode": cheap_mode,
        "oracle_sr_pct": 100 * n_any / n,
        "headroom_pp": 100 * n_any / n - per_mode[best_mode]["sr_pct"],
        "oracle_mean_cost": oracle_cost,
        "oracle_cost_saving_pct": 100 * (best_cost - oracle_cost) / best_cost,
        "triage_sr_pct": per_mode[best_mode]["sr_pct"],          # unchanged, by construction
        "triage_mean_cost": triage_cost,
        "triage_cost_saving_pct": 100 * (best_cost - triage_cost) / best_cost,
        "n_multi_solver": n_multi,
        "multi_solver_share_pct": 100 * n_multi / n,
        "unsolvable_share_pct": 100 * (n - n_any) / n,
        "per_mode": per_mode,
    }


def main() -> int:
    leaks = load_leaks()
    floor = load_floor()
    rows = []

    specs = [(c, False) for c in CELLS] + [(c, True) for c in WA_CELLS]
    for cell, is_wa in specs:
        got = _wa_matrix(cell["baseline"]) if is_wa else _cell_matrix(cell)
        if got is None:
            print(f"[skip] {cell['site']}/{cell['baseline']}: matrix unavailable")
            continue
        tasks, succ, cost = got
        key = f"{CELL_KEY[cell['site']]}_{cell['baseline']}"

        # This script reads episodes only through `_cell_matrix` / `_wa_matrix`, which
        # restrict to the canonical scored set — but it reads them through an IMPORT, which
        # `test_universe_consumption_lint` cannot see (it detects direct filename literals
        # and one named helper, not a third-level transitive read). So the denominator is
        # asserted here rather than inherited on trust: a reddit cell arriving with 205
        # tasks would mean the AMENDMENT_08 exclusions had leaked back in, and every
        # ceiling below it would be computed over the wrong universe.
        want = WA_UNIVERSE_N if is_wa else len(expected_scored_ids(cell["site"])[0])
        if len(tasks) != want:
            raise MissingInput(
                f"{key}: {len(tasks)} tasks, canonical scored universe has {want} — "
                f"the upstream matrix stopped restricting to it")

        kept = evaluate(tasks, succ, cost)
        zeroed_succ, n_zeroed = _apply_leaks(key, succ, leaks)
        zeroed = evaluate(tasks, zeroed_succ, cost)

        m = floor["margins"].get(key, {})
        reruns = floor["wa_reruns"] if is_wa else floor["reruns"].get(key, [])
        rows.append({
            "cell": key, "site": cell["site"], "baseline": cell["baseline"],
            "n_leaked_zeroed": n_zeroed,
            "leak_kept": kept, "leak_zeroed": zeroed,
            "arm_matched_gain_pp": m.get("gain_1_best_distinct_arm_pp"),
            "arm_matched_gain_mode": m.get("gain_1_best_distinct_arm_mode"),
            "all_arms_gain_pp": m.get("gain_5_arms_added_pp"),
            "rerun_draws_pp": sorted(reruns) or None,
        })

    if not rows:
        raise MissingInput("no cells evaluated — nothing to write")

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "leak_policy": "primary = leaked successes set to 0, denominator unchanged "
                       "(user decision 2026-08-04); leak_kept retained for comparison",
        "n_leaked_in_universe": len(leaks),
        "cost_field": "total_billed_cost_usd",
        "cells": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=1) + "\n")
    OUT_MD.write_text(render(payload))
    print(f"[ok] {OUT_JSON.relative_to(REPO)}  ({len(rows)} cells, "
          f"{sum(r['n_leaked_zeroed'] for r in rows)} successes zeroed)")
    return 0


def _f(v, spec="+.2f", dash="—"):
    return dash if v is None else format(v, spec)


def render(p: dict) -> str:
    L = ["# Routing ceiling — what a perfect per-task choice could buy", ""]
    L.append(f"- leak policy: **{p['leak_policy']}**")
    L.append(f"- {p['n_leaked_in_universe']} leaked successes in the scored universe")
    L.append("- **The success ceiling is a six-arm union against a one-arm baseline.** The "
             "arm-matched columns are the honest comparison and are printed beside it.")
    L.append("")
    L.append("## Primary (leaked successes zeroed)")
    L.append("")
    L.append("| cell | n | best single mode | ceiling A: any mode | headroom | "
             "ceiling B': triage cost | +1 arm | rerun once |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in sorted(p["cells"], key=lambda x: -x["leak_zeroed"]["oracle_sr_pct"]):
        z = r["leak_zeroed"]
        rr = r["rerun_draws_pp"]
        rr_s = "—" if not rr else (f"{rr[0]:.2f}" if len(rr) == 1
                                   else f"{min(rr):.2f}–{max(rr):.2f}")
        L.append(f"| `{r['cell']}` | {z['n_tasks']} | {z['best_mode']} "
                 f"{z['best_sr_pct']:.2f}% | **{z['oracle_sr_pct']:.2f}%** | "
                 f"{z['headroom_pp']:+.2f}pp | SR unchanged, "
                 f"**{-z['triage_cost_saving_pct']:+.1f}%** | "
                 f"{_f(r['arm_matched_gain_pp'])}pp | {rr_s}pp |")
    L.append("")
    L.append("## Effect of the leak policy")
    L.append("")
    L.append("| cell | zeroed | best SR kept → zeroed | ceiling kept → zeroed |")
    L.append("|---|---|---|---|")
    for r in sorted(p["cells"], key=lambda x: -x["n_leaked_zeroed"]):
        k, z = r["leak_kept"], r["leak_zeroed"]
        L.append(f"| `{r['cell']}` | {r['n_leaked_zeroed']} | "
                 f"{k['best_sr_pct']:.2f}% → {z['best_sr_pct']:.2f}% | "
                 f"{k['oracle_sr_pct']:.2f}% → {z['oracle_sr_pct']:.2f}% |")
    L.append("")
    L.append("## Why the ceiling is hard to reach")
    L.append("")
    L.append("| cell | no mode solves | >1 solver (the routable set) |")
    L.append("|---|---|---|")
    for r in sorted(p["cells"], key=lambda x: x["leak_zeroed"]["unsolvable_share_pct"]):
        z = r["leak_zeroed"]
        L.append(f"| `{r['cell']}` | {z['unsolvable_share_pct']:.1f}% | "
                 f"{z['n_multi_solver']}/{z['n_tasks']} = {z['multi_solver_share_pct']:.1f}% |")
    L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
