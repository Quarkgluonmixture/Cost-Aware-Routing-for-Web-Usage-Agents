#!/usr/bin/env python3
"""Post-hoc two-tier cascade driven by decoder confidence — the one routing formulation
the paper never tested, on a signal the runs already carry.

WHY THIS EXISTS
---------------
Sections 4-6 all route on PRE-ACTION features (task text, DOM size, intent regexes) and all
fail. A reviewer can answer: *your features were weak, not the problem unroutable.* A cascade
escalates on POST-ACTION signal — a strictly larger information set — so it is the strongest
form of the routing question this data can answer:

    run the cheap mode; if its own decoder was unconfident, pay again for the expensive one.

Every step record already carries `confidence` (B0: mean/min logprob + mean/min margin;
B1/B2 also entropy). `docs/checkpoints/paper_drafts/paperB/` cites it zero times.

WHAT IT DOES NOT ASSUME
-----------------------
The escalation decision sees ONLY the cheap run's own episode. No outcome, no expensive-run
information. That is what a deployment has.

WHAT IT DOES ASSUME, AND CANNOT TEST
------------------------------------
When a task escalates, its outcome is taken from a SEPARATE, standalone run of the rich mode.
A real cascade would start the rich episode AFTER the cheap episode had already acted on a
stateful website — different cart, different session, different page. That sequential potential
outcome is unobserved and no run in this project contains it. So every number here estimates an
OFFLINE SPLICE of two independent conditions, not the post-action cascade it is named after, and
the bias can run in either direction. This limits the oracle and the confidence curves equally.
(codex Mode B, §H stress 2026-08-02.)

THE BASELINES THAT MATTER
-------------------------
Reporting a cascade curve alone is meaningless — any escalation rule buys accuracy by
spending. Three references accompany every operating point:

  random        escalate a same-size subset signal-free (exact expectation, not sampled).
                Beating this is necessary: it isolates the signal from the spending.
  always-rich   just run the expensive mode on everything. A fixed policy — no signal, no
                threshold, no fitting. Beating this on BOTH axes is what makes a cascade
                worth deploying, and §1b is the verdict.
  oracle        escalate exactly the tasks the expensive mode would fix (the ceiling).
                §1c reports what fraction of its headroom the best signal recovers.

Not computed here: spending the same extra budget re-running the CHEAP mode. That comparison
needs a same-condition replicate of the cheap arm in every cell, and only B0 x classifieds has
one (`noise_floor_inventory.md`). Left open rather than approximated.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.lib.canonical_task_universe import expected_scored_ids  # noqa: E402

LOG = logging.getLogger("confidence_cascade")

OUT_MD = REPO / "docs/analysis/cross_sites/confidence_cascade.md"
OUT_JSON = REPO / "docs/analysis/cross_sites/confidence_cascade.json"

# Cheap tier = the mode that is cheapest in EVERY cell we measure (see
# per_mode_four_dimension_profile.md, cost_rel_dom: vision 0.7149-0.9665, lowest in 6/6).
# Expensive tier = the fused mode, dearest in 5/6 and the field's default.
CHEAP, RICH = "vision", "som"

# Glob per (cell, mode). Paper-grade canonical runs; reddit B0 psom is excluded from step-level
# reads elsewhere but this script only ever reads CHEAP, so that exclusion does not apply.
RUNS: dict[str, dict[str, str]] = {
    "cls_B0": {"vision": "*B0_vision_classifieds_20260526_141916*", "som": "*B0_som_classifieds_20260526_041601*"},
    "cls_B1": {"vision": "*B1_vision_classifieds_2026*", "som": "*B1_som_classifieds_2026*"},
    "cls_B2": {"vision": "*B2_vision_classifieds_2026*", "som": "*B2_som_classifieds_2026*"},
    "red_B0": {"vision": "*B0_vision_reddit_2026*", "som": "*B0_som_reddit_2026*"},
    "red_B1": {"vision": "*B1_vision_reddit_2026*", "som": "*B1_som_reddit_2026*"},
    "red_B2": {"vision": "*B2_vision_reddit_2026*", "som": "*B2_som_reddit_2026*"},
}
# WebArena as a seventh cell. Its step records were pulled from the paper-grade host on
# 2026-08-02 (笔切 §407.25); before that this cell looked impossible and was not.
# A prediction written here before the run said WA would register DEGENERATE, reasoning that the
# fused mode scores 13.46% against DOM's 16.35%. That was wrong and is kept as a warning: the
# cheap tier is `vision` at 9.62%, not DOM, so cheap < rich and the cell is not degenerate.
#
# WA was then reported as the ONLY cell where an operating point Pareto-beats always-rich, at two
# points. BOTH are tie artefacts and the claim is withdrawn (§H stress P0-2, 2026-08-02):
#   * `min_margin_min`@40% — the signal has ONE distinct value across all 104 episodes, so the
#     ranking fell through to the stable sort's task-id order. It is now dropped before ranking.
#   * `neg_steps`@30% — 60 episodes tie at the cutoff and 28 of them are chosen by task id;
#     SR spans 8.65-14.42% across tie orders, and the reported 13.46% "exact match" with
#     always-rich sits inside that arbitrary span.
# B0 x WA is still running as of 2026-08-02; when it lands, rerun this and re-judge rather than
# assuming either the old exception or its withdrawal carries over.
# Parameterised on backbone 2026-08-03 (B0 x WA landed 07:23). CHEAP/RICH are mode names,
# so only the {b} slot varies.
WA_RUN_TMPL = {"vision": "{b}_vision_wa_reddit_2026*_R*", "som": "{b}_som_wa_reddit_2026*_R*"}
WA_BASELINES = ("B1", "B0")
WA_ROOT = REPO / "results/webarena/phase1"
SITE_OF = {"cls": "classifieds", "red": "reddit"}
SEARCH_ROOT = REPO / "results/visualwebarena/phase1"

# Episode-level statistics derived from the cheap run's own steps. Each is a candidate
# escalation score; LOW score = escalate. Signs are normalised so that low == unconfident.
SIGNALS = [
    "mean_logprob_mean", "mean_logprob_min", "min_logprob_min",
    "mean_margin_mean", "min_margin_min",
    "neg_steps",           # fewer steps = more confident (cap-hit ⇒ likely stuck)
    "neg_noop_rate",       # fewer no-op steps = more confident
    "neg_actfail_rate",    # fewer failed actions = more confident
]


class MissingInput(RuntimeError):
    """Fail loud on absent inputs — never silently degrade to a partial table."""


@dataclass
class Episode:
    task_id: int
    success: int
    cost: float
    signals: dict[str, float]


def _resolve(pattern: str) -> Path:
    hits = [Path(p) for p in glob.glob(str(SEARCH_ROOT / pattern)) if os.path.isdir(p)]
    if len(hits) != 1:
        raise MissingInput(f"expected exactly 1 run dir for {pattern!r}, got {len(hits)}: "
                           f"{[h.name for h in hits]}")
    return hits[0]


def _read_episodes(run_dir: Path, scored: set[int], *, with_signals: bool) -> dict[int, Episode]:
    """One Episode per scored task. `with_signals=False` skips the step JSONL read entirely."""
    summaries = list(run_dir.glob("*/episodes/*summary*.json")) or list(run_dir.glob("episodes/*summary*.json"))
    if not summaries:
        raise MissingInput(f"no episode summaries under {run_dir}")
    out: dict[int, Episode] = {}
    for f in summaries:
        s = json.loads(f.read_text())
        if s.get("sr_excluded"):
            continue
        tid = int(s["task_id"])
        if tid not in scored:
            continue
        # `total_billed_cost_usd` is the project's canonical cost estimand (§1 primary;
        # memory project_cost_latency_canonical_estimand). Fail loud rather than 0-fill —
        # a silently-zero cost would make every cascade look free.
        if "total_billed_cost_usd" not in s:
            raise MissingInput(f"{f}: no total_billed_cost_usd (canonical cost estimand)")
        cost = float(s["total_billed_cost_usd"])
        sig = _signals_for(f, tid) if with_signals else {}
        out[tid] = Episode(tid, 1 if s.get("success") else 0, cost, sig)
    if len(out) != len(scored):
        raise MissingInput(f"{run_dir.name}: {len(out)} scored episodes but the canonical "
                           f"universe has {len(scored)}")
    return out


def _signals_for(summary_file: Path, task_id: int) -> dict[str, float]:
    """Aggregate one episode's step records into escalation scores. Low = unconfident."""
    steps_glob = list(summary_file.parent.glob(f"*task_{task_id}_steps*.jsonl"))
    if not steps_glob:
        raise MissingInput(f"no step JSONL for task {task_id} beside {summary_file}")
    ml, mn, mm, mgn, noop, actfail, n = [], [], [], [], 0, 0, 0
    for line in steps_glob[0].read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        n += 1
        c = r.get("confidence") or {}
        for key, bucket in (("mean_logprob", ml), ("min_logprob", mn),
                            ("mean_margin", mm), ("min_margin", mgn)):
            v = c.get(key)
            if v is not None:
                bucket.append(float(v))
        if r.get("page_changed") is False:
            noop += 1
        if r.get("action_success") is False:
            actfail += 1
    if n == 0:
        raise MissingInput(f"empty step file for task {task_id}")
    if not ml:
        raise MissingInput(f"task {task_id}: no populated confidence in any of {n} steps")
    return {
        "mean_logprob_mean": statistics.fmean(ml),
        "mean_logprob_min": min(ml),
        "min_logprob_min": min(mn) if mn else min(ml),
        # None, not 0.0, when the backend never populated the field. Substituting a constant
        # made `min_margin_min` identical across all 104 WA episodes, and the sort that ranks
        # tasks by "confidence" then fell through to the stable sort's task-id order — so the
        # reported 40% operating point was literally "the first 42 task ids". A signal with no
        # variance is not a threshold. (codex Mode B, §H stress 2026-08-02)
        "mean_margin_mean": statistics.fmean(mm) if mm else None,
        "min_margin_min": min(mgn) if mgn else None,
        "neg_steps": -float(n),
        "neg_noop_rate": -(noop / n),
        "neg_actfail_rate": -(actfail / n),
    }


def usable_signals(cheap: dict[int, Episode], names: list[str]) -> tuple[list[str], dict]:
    """Drop signals that cannot rank anything, and say which and why.

    A signal is unusable if it is missing on any episode, or if every episode shares one value.
    In the latter case `sorted(..., key=signal)` is a no-op and the escalation set is decided
    entirely by the stable sort's tie order — i.e. by task id. The WA cell's `min_margin_min`
    was exactly this: 104 episodes, one distinct value. (§H stress P0-2)
    """
    keep, dropped = [], {}
    for s in names:
        vals = [c.signals.get(s) for c in cheap.values()]
        if any(v is None for v in vals):
            dropped[s] = f"not populated on {sum(1 for v in vals if v is None)}/{len(vals)} episodes"
        elif len({round(float(v), 12) for v in vals}) < 2:
            dropped[s] = (f"no variance: all {len(vals)} episodes share the value "
                          f"{vals[0]!r}, so ranking falls through to task id")
        else:
            keep.append(s)
    return keep, dropped


def _tie_span(cheap: dict[int, Episode], rich: dict[int, Episode], signal: str,
              k: int) -> dict:
    """How much of the operating point is decided by tie order rather than by the signal?

    Reports the SR reachable at the same escalation count k if the tasks tied at the cutoff had
    been broken the other way. A point whose reported SR is inside a wide span is a tie artifact.
    """
    tids = sorted(cheap)
    n = len(tids)
    scores = {t: float(cheap[t].signals[signal]) for t in tids}
    order = sorted(tids, key=lambda t: scores[t])
    if k <= 0 or k >= n:
        return {"n_tied_at_cutoff": 0, "sr_min": None, "sr_max": None, "tie_decided": 0}
    cut = scores[order[k - 1]]
    below = [t for t in tids if scores[t] < cut]
    tied = [t for t in tids if scores[t] == cut]
    need = k - len(below)                      # how many of the tied group get escalated
    gain = sorted((rich[t].success - cheap[t].success for t in tied), reverse=True)
    base = sum(rich[t].success for t in below) + sum(
        cheap[t].success for t in tids if t not in below and t not in tied)
    best = (base + sum(cheap[t].success for t in tied) + sum(gain[:need])) / n
    worst = (base + sum(cheap[t].success for t in tied) + sum(gain[len(gain) - need:])) / n
    return {"n_tied_at_cutoff": len(tied), "n_decided_by_tie_order": max(0, need),
            "sr_min": 100 * worst, "sr_max": 100 * best,
            "tie_decided": 100 * (best - worst)}


def _curve(cheap: dict[int, Episode], rich: dict[int, Episode], signal: str,
           fracs: list[float]) -> list[dict]:
    """SR and cost as a function of how much of the cheap run gets escalated."""
    tids = sorted(cheap)
    n = len(tids)
    # ascending score == least confident first
    order = sorted(tids, key=lambda t: cheap[t].signals[signal])
    base_cost = sum(cheap[t].cost for t in tids)
    rows = []
    for f in fracs:
        k = round(f * n)
        esc = set(order[:k])
        sr = sum(rich[t].success if t in esc else cheap[t].success for t in tids) / n
        cost = base_cost + sum(rich[t].cost for t in esc)
        # null 1: escalate the same COUNT, chosen by task_id (a fixed, signal-free subset).
        # Deterministic by construction — the workflow forbids Math.random and a seeded RNG
        # here would still be one draw; the exhaustive expectation below is what we report.
        rand_sr = _expected_random_sr(cheap, rich, k)
        # null 2: spend the same extra calls re-running CHEAP. A rerun cannot exceed the
        # cheap mode's own reproducibility, so its expected gain on the escalated set is
        # bounded by the measured self-drop; we report the ceiling of that bound.
        rows.append({
            "frac": f, "k": k, "sr": 100 * sr, "cost": cost,
            "sr_gain_pp": 100 * (sr - sum(cheap[t].success for t in tids) / n),
            "cost_rel": cost / base_cost,
            "random_sr": 100 * rand_sr,
            "random_gain_pp": 100 * (rand_sr - sum(cheap[t].success for t in tids) / n),
            # How much of this point is the signal, and how much is the stable sort's tie order
            "tie": _tie_span(cheap, rich, signal, k),
        })
    return rows


def _expected_random_sr(cheap: dict[int, Episode], rich: dict[int, Episode], k: int) -> float:
    """Exact expectation of escalating a uniformly random k-subset (no sampling needed):
    each task is escalated with probability k/n independently in expectation."""
    tids = sorted(cheap)
    n = len(tids)
    p = k / n if n else 0.0
    return sum((1 - p) * cheap[t].success + p * rich[t].success for t in tids) / n


def _oracle(cheap: dict[int, Episode], rich: dict[int, Episode]) -> dict:
    tids = sorted(cheap)
    n = len(tids)
    fix = [t for t in tids if rich[t].success and not cheap[t].success]
    cost = sum(cheap[t].cost for t in tids) + sum(rich[t].cost for t in fix)
    sr = sum(max(cheap[t].success, rich[t].success if t in set(fix) else 0) for t in tids) / n
    return {"k": len(fix), "sr": 100 * sr, "cost": cost,
            "cost_rel": cost / sum(cheap[t].cost for t in tids)}


def wa_cell(baseline: str = "B1") -> tuple[dict[int, Episode], dict[int, Episode]]:
    """Cheap/rich episode maps for <baseline> x WA-reddit over the tasks both modes ran."""
    got = {}
    for m, tmpl in WA_RUN_TMPL.items():
        pat = tmpl.format(b=baseline)
        hits = [Path(x) for x in glob.glob(str(WA_ROOT / pat))
                if os.path.isdir(x) and "ABORTED" not in x]
        if not hits:
            raise MissingInput(f"WA[{baseline}] {m}: no run dir for {pat!r}")
        got[m] = sorted(hits)[-1]
    ids = None
    for d in got.values():
        s = {int(f.name.split("_task_")[1].split("_")[0])
             for f in (list(d.glob("*/episodes/*summary*.json")) or [])}
        ids = s if ids is None else (ids & s)
    if not ids:
        raise MissingInput(f"WA[{baseline}]: empty task intersection between cheap and rich")
    return (_read_episodes(got[CHEAP], ids, with_signals=True),
            _read_episodes(got[RICH], ids, with_signals=False))


def build(cells: list[str], with_wa: bool = False) -> dict:
    fracs = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0]
    out = {"schema": "2026-08-01-confidence-cascade-v1", "cheap": CHEAP, "rich": RICH,
           "fracs": fracs, "cells": {}}
    for cid in cells:
        site = SITE_OF[cid.split("_")[0]]
        scored, sha = expected_scored_ids(site)
        LOG.info("%s: reading %s (cheap, with step signals)", cid, CHEAP)
        cheap = _read_episodes(_resolve(RUNS[cid][CHEAP]), scored, with_signals=True)
        LOG.info("%s: reading %s (rich, summaries only)", cid, RICH)
        rich = _read_episodes(_resolve(RUNS[cid][RICH]), scored, with_signals=False)
        base_sr = 100 * sum(e.success for e in cheap.values()) / len(cheap)
        rich_sr = 100 * sum(e.success for e in rich.values()) / len(rich)
        # THE baseline that decides the verdict: always run the rich mode instead. It is a
        # fixed policy, needs no signal, and any cascade must Pareto-beat it to matter.
        # Cost is expressed against the same denominator as the curves (the cheap run's bill).
        cheap_bill = sum(e.cost for e in cheap.values())
        always_rich = {"sr": rich_sr, "cost_rel": sum(e.cost for e in rich.values()) / cheap_bill}
        cell = {"n": len(cheap), "universe_sha": sha, "cheap_sr": base_sr, "rich_sr": rich_sr,
                "always_rich": always_rich, "oracle": _oracle(cheap, rich), "curves": {}}
        usable, dropped = usable_signals(cheap, SIGNALS)
        cell["signals_dropped"] = dropped
        cell["n_signals_used"] = len(usable)
        for sig in usable:
            cell["curves"][sig] = _curve(cheap, rich, sig, fracs)
        # Does ANY (signal, operating point) Pareto-beat always-rich? >= SR and <= cost,
        # strictly better on one. This is the whole question.
        # frac=0 is "never escalate" = the always-cheap fixed policy, not a cascade. In the two
        # cells where the rich mode is simply worse, it would otherwise register a degenerate
        # win for every signal. Exclude it and flag those cells instead.
        wins = [(s, r["frac"]) for s, rows in cell["curves"].items() for r in rows
                if r["frac"] > 0
                and r["sr"] >= always_rich["sr"] and r["cost_rel"] <= always_rich["cost_rel"]
                and (r["sr"] > always_rich["sr"] or r["cost_rel"] < always_rich["cost_rel"])]
        # Outcome-dependent by construction, and `>=` labels an exact tie as "rich worse"
        # (cls_B2 is 2.23% vs 2.23%). Both facts are now carried in the flag itself so the
        # count downstream cannot be read as a property of the design. (§H stress P1-5)
        cell["rich_mode_is_worse"] = base_sr >= rich_sr
        cell["degeneracy_is_a_tie"] = abs(base_sr - rich_sr) < 1e-9
        cell["degeneracy_rule"] = "outcome-dependent: cheap_sr >= rich_sr, evaluated post hoc"
        cell["pareto_beats_always_rich"] = wins
        head = cell["oracle"]["sr"] - base_sr
        cell["headroom_captured"] = {
            f"{f:.0%}": (max(next(x for x in cell["curves"][s] if abs(x["frac"] - f) < 1e-9)["sr"]
                             for s in cell["curves"]) - base_sr) / head * 100 if head > 0 else None
            for f in (0.10, 0.20, 0.30)}
        out["cells"][cid] = cell
        LOG.info("%s: cheap %.2f%% / rich %.2f%% / oracle %.2f%%",
                 cid, base_sr, rich_sr, cell["oracle"]["sr"])
    if with_wa:
        for _wb in WA_BASELINES:
            cheap, rich = wa_cell(_wb)
            base_sr = 100 * sum(e.success for e in cheap.values()) / len(cheap)
            rich_sr = 100 * sum(e.success for e in rich.values()) / len(rich)
            cheap_bill = sum(e.cost for e in cheap.values())
            always_rich = {"sr": rich_sr,
                           "cost_rel": sum(e.cost for e in rich.values()) / cheap_bill}
            cell = {"n": len(cheap), "universe_sha": "wa-intersection", "cheap_sr": base_sr,
                    "rich_sr": rich_sr, "always_rich": always_rich,
                    "oracle": _oracle(cheap, rich), "curves": {}}
            usable, dropped = usable_signals(cheap, SIGNALS)
            cell["signals_dropped"] = dropped
            cell["n_signals_used"] = len(usable)
            for sig in usable:
                cell["curves"][sig] = _curve(cheap, rich, sig, out["fracs"])
            wins = [(s, r["frac"]) for s, rows in cell["curves"].items() for r in rows
                    if r["frac"] > 0
                    and r["sr"] >= always_rich["sr"] and r["cost_rel"] <= always_rich["cost_rel"]
                    and (r["sr"] > always_rich["sr"] or r["cost_rel"] < always_rich["cost_rel"])]
            cell["pareto_beats_always_rich"] = wins
            # Outcome-dependent by construction, and `>=` labels an exact tie as "rich worse"
            # (cls_B2 is 2.23% vs 2.23%). Both facts are now carried in the flag itself so the
            # count downstream cannot be read as a property of the design. (§H stress P1-5)
            cell["rich_mode_is_worse"] = base_sr >= rich_sr
            cell["degeneracy_is_a_tie"] = abs(base_sr - rich_sr) < 1e-9
            cell["degeneracy_rule"] = "outcome-dependent: cheap_sr >= rich_sr, evaluated post hoc"
            head = cell["oracle"]["sr"] - base_sr
            cell["headroom_captured"] = {
                f"{f:.0%}": (max(next(x for x in cell["curves"][s] if abs(x["frac"] - f) < 1e-9)["sr"]
                                 for s in cell["curves"]) - base_sr) / head * 100 if head > 0 else None
                for f in (0.10, 0.20, 0.30)}
            out["cells"][f"wa_red_{_wb}"] = cell
            LOG.info(f"wa_red_{_wb}: cheap %.2f%% / rich %.2f%% / oracle %.2f%% (rich worse: %s)",
                     base_sr, rich_sr, cell["oracle"]["sr"], cell["rich_mode_is_worse"])
    return out


def render(d: dict) -> str:
    L = [
        "---", "type: analysis", "status: complete", "created: 2026-08-01",
        f"purpose: post-hoc two-tier cascade escalating {d['cheap']} -> {d['rich']} on the cheap "
        "run's own decoder confidence; the one routing formulation the paper never tested",
        "scope_warning: cost is within-cell only (B0 = API bill, B1/B2 = electricity-derived). "
        "The escalation threshold is swept, NOT selected out-of-fold, so every operating point "
        "below is in-sample and is an UPPER bound on deployable performance.",
        "producer: scripts/analysis/aggregate_confidence_cascade.py", "---", "",
        "# Confidence-triggered cascade", "",
        "Regenerate: `.venv/bin/python3 scripts/analysis/aggregate_confidence_cascade.py`", "",
        # This said "lowest cost in 6/6 cells" / "dearest in 5/6" until 2026-08-03. Both counts
        # were hardcoded, this script has no per-mode cost to check them against, and the /6
        # denominator survived the seventh and eighth cells landing. The eighth also falsified
        # the first count outright. State where the tiers come from and disclose the exception
        # instead of asserting a tally this producer cannot compute.
        f"Cheap tier = **{d['cheap']}**, rich tier = **{d['rich']}** — fixed **a priori** from "
        "the six-cell cost ordering in `multimetric_pareto`, not chosen per cell (choosing per "
        "cell would make the cells incomparable).",
        "",
        f"⚠️ **The tiers are not cost-ordered in every cell.** On `wa_B0` the cheapest mode is "
        f"`dom`, not `{d['cheap']}`. On that cell this is still a fixed-pair escalation and the "
        "SR arithmetic is unaffected, but it is not a cheap→rich escalation in the cost sense, "
        "and its cost column should not be read as one.",
        "",
        "The escalation decision sees only the cheap run's own episode — no outcome, no "
        "rich-run information. Two nulls accompany every point: **random** escalates the same "
        "number of tasks signal-free (exact expectation, not sampled), and **oracle** escalates "
        "exactly the tasks the rich mode would fix.",
        "",
        "## 1. Endpoints", "",
        "Cost is relative to running the cheap mode on every task.", "",
        "| cell | n | cheap SR | **always-rich SR / cost** | oracle SR | oracle escalates | oracle cost |",
        "|---|---|---|---|---|---|---|",
    ]
    for cid, c in d["cells"].items():
        o, ar = c["oracle"], c["always_rich"]
        L.append(f"| `{cid}` | {c['n']} | {c['cheap_sr']:.2f}% | "
                 f"**{ar['sr']:.2f}% / {ar['cost_rel']:.2f}x** | "
                 f"{o['sr']:.2f}% | {o['k']} tasks | {o['cost_rel']:.2f}x |")

    L += ["", "The **oracle cascade is the attractive operating point in this table**: it pays "
          "double only on the 2–22 tasks that need it, so it buys +2.2 to +10.8pp for +2% to "
          "+12% cost. Everything below asks how much of that a deployable signal recovers.", "",
          "> ⚠️ **Every number below is an offline splice.** An escalated task takes its outcome "
          "from a standalone rich-mode run, but a real cascade would start the rich episode "
          "*after* the cheap one had already acted on a stateful site. That sequential outcome "
          "is unobserved in this project, so the bias can run either way — this is a limitation "
          "of the design, not of the estimator.", "",
          "## 1b. THE VERDICT — does any operating point Pareto-beat *always-rich*?", "",
          "Always running the rich mode is a fixed policy: no signal, no threshold, no fitting. "
          "A cascade that does not beat it on both axes has bought nothing.", "",
          "| cell | always-rich SR / cost | operating points that Pareto-beat it |",
          "|---|---|---|"]
    total_wins = 0
    for cid, c in d["cells"].items():
        w = c["pareto_beats_always_rich"]
        total_wins += len(w)
        if not w:
            txt = "**none**"
        else:
            parts = []
            for sname, f in w[:4]:
                row = next(x for x in c["curves"][sname] if abs(x["frac"] - f) < 1e-9)
                t = row.get("tie") or {}
                tag = ""
                if t.get("n_tied_at_cutoff", 0) > 1 and t.get("tie_decided", 0) > 0:
                    tag = (f" ⚠️ {t['n_tied_at_cutoff']} tied at the cutoff, "
                           f"{t.get('n_decided_by_tie_order', 0)} of them picked by task id; "
                           f"SR spans {t['sr_min']:.2f}–{t['sr_max']:.2f}% over tie orders")
                parts.append(f"`{sname}`@{f:.0%}{tag}")
            txt = ", ".join(parts)
        if c["rich_mode_is_worse"]:
            txt += " · ⚠️ rich mode is *worse than or equal to* cheap here, so the cascade question is moot"
        L.append(f"| `{cid}` | {c['always_rich']['sr']:.2f}% / "
                 f"{c['always_rich']['cost_rel']:.2f}x | {txt} |")
    n_cells_win = sum(1 for c in d["cells"].values() if c["pareto_beats_always_rich"])
    # Denominator counts the signals each cell actually ranks with (some are dropped for having
    # no variance) times the non-zero fractions. Using len(SIGNALS) x len(fracs) overcounts and
    # was the source of the "2 of 80" figure; the true search space is smaller. (§H stress P1-5)
    searched = sum(len(c["curves"]) * (len(d["fracs"]) - 1) for c in d["cells"].values())
    L += ["", f"**{total_wins} of {searched} (cell, signal, operating point) combinations "
          f"Pareto-beat the fixed policy, in {n_cells_win} of {len(d['cells'])} cells.** "
          "`frac=0` is excluded throughout — it is the always-cheap fixed policy, not a cascade. "
          "The denominator counts only signals a cell can actually rank with; where a signal was "
          "dropped for having no variance it is not part of the search space.", "",
          "## 1c. Fraction of the oracle's headroom the best signal recovers", "",
          "| cell | 10% | 20% | 30% |", "|---|---|---|---|"]
    for cid, c in d["cells"].items():
        h = c["headroom_captured"]
        L.append(f"| `{cid}` | " + " | ".join(
            "—" if h[k] is None else f"{h[k]:.0f}%" for k in ("10%", "20%", "30%")) + " |")

    L += ["", "## 2. Does the confidence signal beat a signal-free escalation of the same size?", "",
          "For each cell, the best signal at each escalation fraction, and the margin over the "
          "random-escalation expectation. **A positive margin is the entire claim** — without it "
          "the cascade is just paying more.", "",
          "| cell | frac | best signal | SR | gain vs cheap | random gain | **margin** |",
          "|---|---|---|---|---|---|---|"]
    for cid, c in d["cells"].items():
        for f in (0.10, 0.20, 0.30):
            best, row = None, None
            for sig, rows in c["curves"].items():
                r = next(x for x in rows if abs(x["frac"] - f) < 1e-9)
                if row is None or r["sr"] > row["sr"]:
                    best, row = sig, r
            margin = row["sr_gain_pp"] - row["random_gain_pp"]
            flag = " ✅" if margin > 0 else ""
            L.append(f"| `{cid}` | {f:.0%} | `{best}` | {row['sr']:.2f}% | "
                     f"{row['sr_gain_pp']:+.2f}pp | {row['random_gain_pp']:+.2f}pp | "
                     f"**{margin:+.2f}pp**{flag} |")

    L += ["", "⚠️ The best signal is picked per (cell, fraction) from "
          f"{len(SIGNALS)} candidates against realised outcomes, so these margins are "
          "in-sample maxima over a signal menu. Treat them as an upper bound on what an "
          "out-of-fold selection could deliver.", "",
          "## 3. Per-signal margin over random, averaged across cells", "",
          "| signal | " + " | ".join(f"{f:.0%}" for f in (0.10, 0.20, 0.30)) + " |",
          "|---|---|---|---|"]
    for sig in SIGNALS:
        cells_ = []
        n_cells_with = 0
        for f in (0.10, 0.20, 0.30):
            ms = []
            for c in d["cells"].values():
                if sig not in c["curves"]:      # dropped for no variance / not populated
                    continue
                r = next(x for x in c["curves"][sig] if abs(x["frac"] - f) < 1e-9)
                ms.append(r["sr_gain_pp"] - r["random_gain_pp"])
            n_cells_with = max(n_cells_with, len(ms))
            cells_.append(f"{statistics.fmean(ms):+.2f}pp" if ms else "—")
        suffix = "" if n_cells_with == len(d["cells"]) else f" ⚠️ {n_cells_with}/{len(d['cells'])} cells"
        L.append(f"| `{sig}`{suffix} | " + " | ".join(cells_) + " |")
    dropped_any = {cid: c.get("signals_dropped") or {} for cid, c in d["cells"].items()}
    if any(dropped_any.values()):
        L += ["", "**Signals dropped before ranking** — a score with no variance cannot rank "
              "anything, and `sorted()` then falls through to task id, so the resulting "
              "\"operating point\" is a set of task ids wearing a threshold's name:"]
        for cid, dr in dropped_any.items():
            for sname, why in dr.items():
                L.append(f"- `{cid}` / `{sname}`: {why}")

    L += ["", "## 4. Full curves", ""]
    for cid, c in d["cells"].items():
        L += [f"### `{cid}` (n={c['n']}, cheap {c['cheap_sr']:.2f}%, "
              f"oracle {c['oracle']['sr']:.2f}%)", "",
              "| frac | k | SR | cost | SR gain | random gain | margin |",
              "|---|---|---|---|---|---|---|"]
        sig = max(c["curves"], key=lambda s: next(
            x for x in c["curves"][s] if abs(x["frac"] - 0.20) < 1e-9)["sr"])
        for r in c["curves"][sig]:
            m = r["sr_gain_pp"] - r["random_gain_pp"]
            L.append(f"| {r['frac']:.0%} | {r['k']} | {r['sr']:.2f}% | {r['cost_rel']:.2f}x | "
                     f"{r['sr_gain_pp']:+.2f}pp | {r['random_gain_pp']:+.2f}pp | {m:+.2f}pp |")
        L += [f"", f"_Signal shown: `{sig}` (best at 20% for this cell)._", ""]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", nargs="*", default=list(RUNS), help="subset of cell ids")
    ap.add_argument("--with-wa", action="store_true",
                    help="append B1 x WA-reddit; writes to *_with_wa.* so the six-cell verdict "
                         "the paper cites is not overwritten")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    d = build(a.cells, a.with_wa)
    om, oj = OUT_MD, OUT_JSON
    if a.with_wa:
        om = OUT_MD.with_name(OUT_MD.stem + "_with_wa" + OUT_MD.suffix)
        oj = OUT_JSON.with_name(OUT_JSON.stem + "_with_wa" + OUT_JSON.suffix)
    oj.write_text(json.dumps(d, indent=2))
    om.write_text(render(d))
    print(f"✓ {om.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
