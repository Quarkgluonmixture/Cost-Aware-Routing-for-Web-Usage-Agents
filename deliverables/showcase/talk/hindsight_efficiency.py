#!/usr/bin/env python3
"""What picking the right view per task would buy — success, time and estimated CO2e.

Talk slide `hindsight` (2026-09-15). The success and cost half already exists in
`scripts/analysis/router_objective_ordering.py` (policy `oracle_sr_cost`); this script
re-uses that module's cells, task universe and picks, and adds the two quantities the
talk asks for on the SAME picks:

  * time   = mean `total_latency_canonical_ms` per task (the project's latency estimand)
  * CO2e   = the demo's token-based estimate (`demo/build_demo_data.py` CARBON constants,
             sourced in `demo/README.md` "The CO2e row"), summed from each episode's step
             records because the formula needs input and output tokens separately.
             B0 settings only: the energy-per-token constants were measured for B0's model
             (Qwen3-VL-235B-A22B); they do not apply to the 4B backbones.

Picks, identical to `oracle_sr_cost`: among the views that solved a task take the one with
the lowest billed cost; when no view solved it, take the setting's cheapest view. It is
compared with the setting's best single view (highest success rate). Both are hindsight
policies over recorded runs — an upper bound, not something a deployed agent achieved.

Two self-checks fail loudly: the recomputed mean cost must equal the module's
`oracle_sr_cost`, and each episode's summed step tokens must equal its `total_tokens`.

Usage::

    .venv/bin/python3 deliverables/showcase/talk/hindsight_efficiency.py
"""
from __future__ import annotations

import glob
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))
sys.path.insert(0, str(HERE.parent / "demo"))

import router_objective_ordering as R  # noqa: E402
from p79.experiment.io_utils import read_jsonl_dedup  # noqa: E402
from scripts.analysis.lib.episode_rows import load_cell_task_rows  # noqa: E402
import build_demo_data as DEMO  # noqa: E402

OUT = HERE / "hindsight_efficiency.json"
LAT = "total_latency_canonical_ms"


def _wa_episode_dirs(baseline: str) -> dict[str, list[Path]]:
    """The same run directories `R._wa_matrix` resolves, one list of episode dirs per view."""
    dirs = {}
    for m, stem in R.WA_STEM.items():
        hits = [Path(p) for p in glob.glob(str(REPO / R.WA_ROOT / f"{baseline}_{stem}_wa_reddit_2026*_R*"))
                if Path(p).is_dir() and "ABORTED" not in p]
        assert len(hits) == 1, (baseline, m, hits)
        eps = [p for p in hits[0].glob("*/episodes") if p.is_dir()] or [hits[0] / "episodes"]
        dirs[m] = eps
    return dirs


def _rows_wa(baseline: str) -> dict[str, dict[int, dict]]:
    rows = {}
    for m, eps in _wa_episode_dirs(baseline).items():
        rows[m] = {}
        for ep in eps:
            for f in ep.glob("*summary*.json"):
                rec = json.loads(f.read_text())
                if rec.get("sr_excluded"):
                    continue
                rec["_episodes_dir"] = str(ep)
                rows[m][int(rec["task_id"])] = rec
    return rows


def _rows_vwa(cell: dict) -> dict[str, dict[int, dict]]:
    rows = load_cell_task_rows(cell, modes=R.SIX_MODES)
    for m in R.SIX_MODES:
        for rec in rows[m].values():
            rec["_episodes_dir"] = str(cell["modes"][m])
    return rows


def _tokens(rec: dict) -> tuple[int, int]:
    ep = Path(rec["_episodes_dir"])
    hits = list(ep.glob(f"*_task_{int(rec['task_id'])}_steps_v2.jsonl"))
    assert len(hits) == 1, (ep, rec["task_id"], hits)
    t_in = t_out = 0
    for step in read_jsonl_dedup(str(hits[0])):
        tok = step.get("tokens") or {}
        t_in += tok.get("input") or 0
        t_out += tok.get("output") or 0
    assert t_in + t_out == rec.get("total_tokens"), (hits[0], t_in + t_out, rec.get("total_tokens"))
    return t_in, t_out


def cell_report(cell: dict) -> dict:
    ev = R.evaluate(cell)
    got = R._wa_matrix(cell["baseline"]) if cell.get("_wa") else R._cell_matrix(cell)
    if ev is None or got is None:      # the module's own main() drops these cells the same way
        return {"site": cell["site"], "baseline": cell["baseline"], "skipped": "incomplete six-view matrix"}
    tasks, succ, cost = got
    best, cheapest = ev["best_sr_mode"], ev["cheapest_mode"]
    rows = _rows_wa(cell["baseline"]) if cell.get("_wa") else _rows_vwa(cell)

    picks = {}
    for t in tasks:
        solvers = [m for m in R.SIX_MODES if succ[m][t]]
        picks[t] = min(solvers, key=lambda m: cost[m][t]) if solvers else cheapest

    mean_cost_pick = st.fmean(cost[picks[t]][t] for t in tasks)
    want = ev["policies"]["oracle_sr_cost"]["mean_cost"]
    assert abs(mean_cost_pick - want) < 1e-9, (cell["site"], cell["baseline"], mean_cost_pick, want)

    def lat(m, t):
        r = rows[m][t]
        return float(r.get(LAT) if r.get(LAT) is not None else r["total_latency_ms"]) / 1000.0

    sr_best = 100 * st.fmean(succ[best][t] for t in tasks)
    sr_pick = 100 * st.fmean(succ[picks[t]][t] or any(succ[m][t] for m in R.SIX_MODES) for t in tasks)
    out = {
        "site": cell["site"], "baseline": cell["baseline"], "n": len(tasks),
        "best_single_view": best, "cheapest_view": cheapest,
        "sr_best_pct": round(sr_best, 2), "sr_hindsight_pct": round(sr_pick, 2),
        "sr_gain_pp": round(sr_pick - sr_best, 2),
        "cost_change_pct": round(100 * (mean_cost_pick / st.fmean(cost[best][t] for t in tasks) - 1), 1),
        "latency_best_s": round(st.fmean(lat(best, t) for t in tasks), 1),
        "latency_hindsight_s": round(st.fmean(lat(picks[t], t) for t in tasks), 1),
    }
    out["latency_change_pct"] = round(100 * (out["latency_hindsight_s"] / out["latency_best_s"] - 1), 1)

    if cell["baseline"] == "B0":
        def co2(m, t):
            return DEMO._co2_g(*_tokens(rows[m][t]))
        best_lo = st.fmean(co2(best, t)[0] for t in tasks); best_hi = st.fmean(co2(best, t)[1] for t in tasks)
        pick_lo = st.fmean(co2(picks[t], t)[0] for t in tasks); pick_hi = st.fmean(co2(picks[t], t)[1] for t in tasks)
        ch_lo, ch_hi = 100 * (pick_lo / best_lo - 1), 100 * (pick_hi / best_hi - 1)
        out.update({"co2_best_g": [round(best_lo, 3), round(best_hi, 3)],
                    "co2_hindsight_g": [round(pick_lo, 3), round(pick_hi, 3)],
                    "co2_change_pct": [round(min(ch_lo, ch_hi), 1), round(max(ch_lo, ch_hi), 1)]})
    return out


def main() -> None:
    cells = list(R.CELLS) + R.WA_CELLS
    report = [cell_report(c) for c in cells]
    OUT.write_text(json.dumps({
        "what": "hindsight per-task view choice (oracle_sr_cost picks) vs best single view: success, cost, "
                "latency (total_latency_canonical_ms) and token-estimated CO2e (B0 only)",
        "source_module": "scripts/analysis/router_objective_ordering.py", "carbon_constants": DEMO.CARBON,
        "cells": report}, indent=1, default=str) + "\n")
    print(f"{'setting':22} {'best':9} {'SR best→hind':>14} {'ΔSR':>7} {'Δcost':>7} {'time best→hind (s)':>20} {'Δtime':>7} {'ΔCO2e (est.)':>16}")
    for r in report:
        if r.get("skipped"):
            print(f"{r['site'] + ' · ' + r['baseline']:22} skipped: {r['skipped']}")
            continue
        co2 = f"{r['co2_change_pct'][0]:+.1f}…{r['co2_change_pct'][1]:+.1f}%" if "co2_change_pct" in r else "—"
        print(f"{r['site'] + ' · ' + r['baseline']:22} {r['best_single_view']:9} "
              f"{r['sr_best_pct']:6.2f}→{r['sr_hindsight_pct']:6.2f} {r['sr_gain_pp']:+7.2f} {r['cost_change_pct']:+6.1f}% "
              f"{r['latency_best_s']:8.1f}→{r['latency_hindsight_s']:8.1f} {r['latency_change_pct']:+6.1f}% {co2:>16}")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
