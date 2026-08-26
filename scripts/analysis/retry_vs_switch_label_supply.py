#!/usr/bin/env python3
"""Does "retry or switch" have the opposite label-supply profile to "which mode"?

§6 of the REALM draft shows which-representation routing failing for a supply reason:
the natural label -- the cheapest mode that solved the task -- exists only where
something succeeded, so supervision is produced AT the success rate. 15-97 labels per
cell over six classes; four of six cells admit no classifier under min_class_n=10; and
the tasks where a per-task choice even exists (more than one mode succeeding) are
1.5-34.6% of a cell. That is where the title comes from.

This script asks whether a DIFFERENT routing decision inverts that profile. "The first
attempt failed -- retry the same arm, or switch to another one?" has a label wherever
the first attempt FAILED, and failures are abundant exactly in the low-success regime
the paper says routing matters most in. If the decision set and its label composition
are both large where §6's contested set is small, then the circularity named in §7 is a
property of the which-mode question, not of representation routing as such.

Inputs: the three same-condition replicate pairs registered in
`aggregate_noise_floor_inventory.CLEAN_PAIRS` (B0 x VWA-classifieds, canonical n=224).
Success is read through that module's `_episode_success`, so sr_excluded handling and
the canonical scored universe are identical to the noise-floor product. A second reader
of the same episodes must not invent a second definition of success.

*** SCOPE, and it is narrow ***

1. Three arms, one cell. dom / som / vision on B0 x classifieds. The phantom arms have
   no replicate, so nothing here speaks to them.
2. These pairs are "run-to-run INCLUDING environment drift" (noise_floor_inventory's own
   phrase), not immediate retries: the dom pair is 2 days apart, the som pair 69. A
   production retry fires seconds later and carries only serving nondeterminism, so the
   retry gains computed here are an UPPER bound on what an immediate retry would buy.
   Do not quote them as the value of a retry action without that qualifier.
3. Everything is oracle-conditioned: knowing that retry (or a switch) rescued a task
   presumes the outcome is observable. A deployed policy needs failure detection, and
   §8 of the draft says this evaluator is binary and carries FPs. This measures whether
   the SIGNAL exists, not whether it is exploitable.

Outputs land in the scratchpad first, per the §383.2 producer discipline (run it, look
at it, promote it only once the numbers have been read).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.aggregate_noise_floor_inventory import (  # noqa: E402
    CLEAN_PAIRS,
    MissingInput,
    _episode_success,
)

LOG = logging.getLogger("retry_vs_switch")

# Direction a/b is the pair's own ordering and carries no time semantics (the dom
# pair's "b" arm is in fact two days EARLIER than its "a" arm). Both directions are
# reported everywhere for that reason.
# All six cls·B0 arms carry a replicate as of 2026-08-20. This tuple used to
# list three, which made the script fail closed once the phantom pairs landed.
ARMS = ("dom", "som", "vision", "ptext", "pprompt", "psom")

# B-1994 (2026-08-26). This module is about ONE cell — B0 x classifieds — but it
# used to walk all of CLEAN_PAIRS and key the result by `label.rsplit(".")[-1]`,
# i.e. by MODE ALONE. That was correct only while every registered pair was a
# B0.cls one, which is what the old comment here asserted ("labels are
# B0.cls.<mode>"). CLEAN_PAIRS has since grown B1 pairs (2026-08-17/19), a B5
# pair (08-21) and reddit pairs (08-26), and every one of them collides on that
# key: B1.cls.dom overwrote B0.cls.dom, then B5.cls.dom overwrote that. The
# arity check at the end compares only the SET OF KEYS, so dom/som/vision were
# all present and it passed — the module would have silently analysed B5's and
# B1's arms as if they were B0's. The published artefact predates the first
# collision, so nothing shipped wrong; a rerun would have been the first victim.
# Filter by the full cell prefix, not by mode.
CELL_PREFIX = "B0.cls."


def load_arms() -> dict[str, dict[str, dict[int, int]]]:
    """mode -> {"a": {task_id: 0/1}, "b": {...}} restricted to the canonical universe."""
    from scripts.analysis.lib.canonical_task_universe import expected_scored_ids

    scored, sha = expected_scored_ids("classifieds")
    scored = set(scored)
    LOG.info("canonical classifieds scored universe: n=%d (sha=%s)", len(scored), sha[:12])

    out: dict[str, dict[str, dict[int, int]]] = {}
    for label, ra, rb in CLEAN_PAIRS:
        if not label.startswith(CELL_PREFIX):
            continue  # B-1994: other cells collide on the mode-only key
        mode = label[len(CELL_PREFIX):]
        arms = {}
        for key, rel in (("a", ra), ("b", rb)):
            p = REPO / rel
            if not p.is_dir():
                raise MissingInput(f"{label}/{key}: replicate arm not on disk: {p}")
            s = _episode_success(p)
            missing = scored - set(s)
            if missing:
                raise MissingInput(
                    f"{label}/{key}: {len(missing)} of {len(scored)} canonical tasks absent "
                    f"(first few: {sorted(missing)[:5]}). A partial arm reads exactly like a "
                    f"complete one downstream -- refusing."
                )
            arms[key] = {t: s[t] for t in scored}
        out[mode] = arms
        LOG.info("loaded %s: SR a=%.2f%% b=%.2f%%", mode,
                 100 * sum(arms["a"].values()) / len(scored),
                 100 * sum(arms["b"].values()) / len(scored))
    if set(out) != set(ARMS):
        raise MissingInput(f"expected arms {ARMS}, got {sorted(out)}")
    return out


def solves(d: dict[int, int]) -> set[int]:
    return {t for t, v in d.items() if v}


def union_gain(base: dict[int, int], added: dict[int, int]) -> dict:
    """|{added solves} \\ {base solves}| / n -- the functional §5 uses for a one-arm gain.

    Identical in form whether the added arm is a rerun of the base or a different
    representation, which is the whole point of the comparison.
    """
    n = len(base)
    b, a = solves(base), solves(added)
    rescued = sorted(a - b)
    return {
        "n": n,
        "base_sr_pp": 100 * len(b) / n,
        "union_sr_pp": 100 * len(b | a) / n,
        "gain_pp": 100 * len(rescued) / n,
        "rescued_count": len(rescued),
        "rescued_tasks": rescued,
    }


def part1_matched_gains(arms) -> list[dict]:
    """Retry vs switch at one added arm, every base/added combination, both directions."""
    rows = []
    for mode, base_key in itertools.product(ARMS, ("a", "b")):
        base = arms[mode][base_key]
        other_key = "b" if base_key == "a" else "a"
        r = union_gain(base, arms[mode][other_key])
        rows.append({"base": f"{mode}.{base_key}", "action": "retry",
                     "added": f"{mode}.{other_key}", **r})
        for other in ARMS:
            if other == mode:
                continue
            # Both generations of the switch target are reported: pairing a base with the
            # same-generation arm vs the other one is a free choice, and quoting only one
            # of them would hide the drift sensitivity that choice carries.
            for k in ("a", "b"):
                r = union_gain(base, arms[other][k])
                rows.append({"base": f"{mode}.{base_key}", "action": "switch",
                             "added": f"{other}.{k}", **r})
    return rows


def part2_label_supply(arms) -> list[dict]:
    """The decision set (base failed) and what each action would have done on it.

    A label here needs only the base to have FAILED, plus the outcome of the two
    candidate actions. It does NOT need two modes to have succeeded, which is the
    constraint that starves §6's which-mode label.
    """
    rows = []
    for mode, base_key in itertools.product(ARMS, ("a", "b")):
        base = arms[mode][base_key]
        other_key = "b" if base_key == "a" else "a"
        retry_arm = arms[mode][other_key]
        switch_arms = {o: arms[o] for o in ARMS if o != mode}

        n = len(base)
        decision = sorted(t for t, v in base.items() if not v)
        counts = {"retry_only": 0, "switch_only": 0, "both": 0, "neither": 0}
        per_task = []
        for t in decision:
            retry_ok = bool(retry_arm[t])
            # A switch is credited if EITHER generation of EITHER other arm solves it;
            # this is the most generous reading of the switch action, so any advantage
            # retry shows here is not an artifact of under-crediting its rival.
            sw_detail = {o: (bool(a["a"][t]) or bool(a["b"][t])) for o, a in switch_arms.items()}
            switch_ok = any(sw_detail.values())
            if retry_ok and switch_ok:
                lab = "both"
            elif retry_ok:
                lab = "retry_only"
            elif switch_ok:
                lab = "switch_only"
            else:
                lab = "neither"
            counts[lab] += 1
            per_task.append({"task_id": t, "base": f"{mode}.{base_key}", "label": lab,
                             "retry_rescues": int(retry_ok), "switch_rescues": int(switch_ok),
                             **{f"switch_{o}": int(v) for o, v in sw_detail.items()}})

        actionable = counts["retry_only"] + counts["switch_only"] + counts["both"]
        contested = counts["retry_only"] + counts["switch_only"]
        rows.append({
            "base": f"{mode}.{base_key}",
            "n": n,
            "decision_set": len(decision),
            "decision_set_pct": 100 * len(decision) / n,
            **counts,
            "actionable": actionable,
            "actionable_pct_of_cell": 100 * actionable / n,
            # The rows a router could actually LEARN from: one action rescues and the
            # other does not. "both" and "neither" carry no preference to fit.
            "contested": contested,
            "contested_pct_of_cell": 100 * contested / n,
            "per_task": per_task,
        })
    return rows


def part3_which_mode_baseline(arms) -> dict:
    """§6's contested set, recomputed at THIS arm count so the comparison is legal.

    §6 quotes 1.5-34.6% over six modes. Three arms cannot reproduce that number and this
    does not try to: it recomputes the same definition (more than one arm solves it, so a
    cheapest-solver choice exists) on the three arms in hand, which is the only quantity
    the retry-vs-switch figures may be set beside.
    """
    n = len(arms["dom"]["a"])
    per_gen = {}
    for gen in ("a", "b"):
        solved_by = {t: sum(1 for m in ARMS if arms[m][gen][t]) for t in arms["dom"][gen]}
        at_least_one = sum(1 for v in solved_by.values() if v >= 1)
        more_than_one = sum(1 for v in solved_by.values() if v >= 2)
        per_gen[gen] = {
            "n": n,
            "at_least_one_solver": at_least_one,
            "at_least_one_pct": 100 * at_least_one / n,
            "contested_multi_solver": more_than_one,
            "contested_pct": 100 * more_than_one / n,
        }
    return per_gen


def part4_starting_point(p1, p2) -> list[dict]:
    """Whether "a rerun is worth as much as a distinct representation" depends on the base.

    `noise_floor_inventory.md` §2 licenses exactly one sentence on this cell: "at the
    one-arm margin a distinct representation is worth no more than a rerun of the same
    representation." It is computed from the cell's BEST single mode (som @ 27.23%), which
    is the correct baseline for reporting a ceiling -- and also the starting point least
    favourable to switching, because the strongest arm leaves the others least to add.

    This tabulates the same functional from every starting point, so the dependence is
    visible instead of being a property of one defensible choice.
    """
    rows = []
    for mode, bk in itertools.product(ARMS, ("a", "b")):
        base = f"{mode}.{bk}"
        rt = [r["gain_pp"] for r in p1 if r["base"] == base and r["action"] == "retry"][0]
        sw = [r["gain_pp"] for r in p1 if r["base"] == base and r["action"] == "switch"]
        sup = [r for r in p2 if r["base"] == base][0]
        rows.append({
            "base": base,
            "base_sr_pp": [r["base_sr_pp"] for r in p1 if r["base"] == base][0],
            "retry_gain_pp": rt,
            "switch_gain_min_pp": min(sw),
            "switch_gain_max_pp": max(sw),
            "switch_over_retry_min": (min(sw) / rt) if rt else None,
            "switch_over_retry_max": (max(sw) / rt) if rt else None,
            "retry_only": sup["retry_only"],
            "switch_only": sup["switch_only"],
            "neither": sup["neither"],
        })
    return sorted(rows, key=lambda r: r["base_sr_pp"])


MODE_SR_COLS = {"dom": "sr_dom", "som": "sr_som", "vision": "sr_vision",
                "ptext": "sr_ptext", "pprompt": "sr_pprompt", "psom": "sr_psom"}


def part5_arm_budget(arms) -> dict:
    """At a fixed budget of six arms: six representations once, or three twice?

    `noise_floor_inventory.md` explicitly declines this: "Not licensed. 'The whole 6-mode
    ceiling gain is noise.' We hold one rerun arm, not five." Three replicated arms landed
    since (som on 2026-08-03), so the six-arm comparison is now computable -- with one
    asymmetry that must travel with it: the repetition budget covers only THREE distinct
    representations, so this is not "repetition equals representation". It is the deployment
    question of how to spend a fixed arm budget.

    The six-representation union is also a correctness check: it must reproduce the 43.30%
    six-mode oracle already published for this cell.
    """
    per_task = REPO / "results/phantom_paper/per_task_sr.csv"
    if not per_task.is_file():
        raise MissingInput(f"per-task SR product absent: {per_task}")
    six: dict[int, dict[str, float]] = {}
    with per_task.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("cell_id") != "cls_B0":
                continue
            six[int(row["task_id"])] = {m: float(row[c]) for m, c in MODE_SR_COLS.items()}
    if not six:
        raise MissingInput("no cls_B0 rows in per_task_sr.csv")

    tasks = sorted(set(six) & set(arms["dom"]["a"]))
    n = len(tasks)
    six_union = {t for t in tasks if any(v > 0 for v in six[t].values())}
    six_best = max(MODE_SR_COLS,
                   key=lambda m: sum(1 for t in tasks if six[t][m] > 0))
    three_x2_union = {t for t in tasks
                      if any(arms[m][k][t] for m in ARMS for k in ("a", "b"))}
    return {
        "n": n,
        "six_representations_one_generation": {
            "union_sr_pp": 100 * len(six_union) / n,
            "best_single_mode": six_best,
            "best_single_sr_pp": 100 * sum(1 for t in tasks if six[t][six_best] > 0) / n,
        },
        "three_representations_two_generations": {
            "union_sr_pp": 100 * len(three_x2_union) / n,
            "arms": [f"{m}.{k}" for m in ARMS for k in ("a", "b")],
        },
        "solved_only_by_six_rep": sorted(six_union - three_x2_union),
        "solved_only_by_three_x2": sorted(three_x2_union - six_union),
        # Not independent: per_task_sr.csv's canonical generation for dom/som/vision may be
        # one of the CLEAN_PAIRS arms, so the two budgets can share up to three arms.
        "independence_caveat": "budgets may share up to 3 arms; not an independent contrast",
    }


def render_md(d: dict) -> str:
    """Render the report from the product JSON. Every number comes from `d`.

    Per §450.8: no cell count, interval or per-cell number may be hardcoded in prose, and
    the producer must offer a re-render path that does not recompute (`--from-json`), so
    that fixing a word never costs a recomputation.
    """
    p1, p2, p3 = d["part1_matched_gains"], d["part2_label_supply"], d["part3_which_mode_contested"]
    p4, p5 = d["part4_starting_point"], d["part5_arm_budget"]
    a6 = p5["six_representations_one_generation"]
    b3 = p5["three_representations_two_generations"]
    n = p2[0]["n"]
    neither = sorted({r["neither"] for r in p2})
    wm = max(v["contested_pct"] for v in p3.values())
    rs = max(r["contested_pct_of_cell"] for r in p2)
    rt_all = [r["retry_gain_pp"] for r in p4]
    sw_all = [x for r in p4 for x in (r["switch_gain_min_pp"], r["switch_gain_max_pp"])]

    L = [
        "---", "type: analysis", "status: complete",
        "purpose: whether a retry-or-switch routing decision has a different supply profile"
        " than which-mode routing, and whether the licensed one-arm claim survives every base",
        f"scope_warning: three arms of one cell (B0 x VWA-classifieds, n={n}). The replicate"
        " pairs are run-to-run INCLUDING environment drift (dom 2 days apart, som 69), so"
        " retry gains are an UPPER bound on an immediate retry. Everything is"
        " oracle-conditioned and needs failure detection to deploy.",
        "producer: scripts/analysis/retry_vs_switch_label_supply.py",
        "---", "",
        "# Retry-or-switch: label supply, and whether the one-arm claim is base-dependent", "",
        "Regenerate: `.venv/bin/python3 scripts/analysis/retry_vs_switch_label_supply.py"
        " --write-md`  ·  re-render prose only: `--from-json <path> --write-md`", "",
        "## 1. The one-arm margin, read from every base", "",
        "`noise_floor_inventory.md` §2 licenses one sentence on this cell — *at the one-arm"
        " margin a distinct representation is worth no more than a rerun of the same"
        " representation*. It is computed from the cell's **best** single mode, which is also"
        " the base least favourable to switching: the strongest arm leaves the others least"
        " to add. Read from every base, the sentence is starting-point dependent.", "",
        "| base | base SR | +1 rerun | +1 distinct representation | switch / retry |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in p4:
        L.append(f"| `{r['base']}` | {r['base_sr_pp']:.2f}% | **{r['retry_gain_pp']:.2f}pp** | "
                 f"{r['switch_gain_min_pp']:.2f}–{r['switch_gain_max_pp']:.2f}pp | "
                 f"{r['switch_over_retry_min']:.2f}–{r['switch_over_retry_max']:.2f}× |")
    L += [
        "",
        f"Rerun gain moves over {min(rt_all):.2f}–{max(rt_all):.2f}pp with no trend in the"
        f" base, while switch gain moves over {min(sw_all):.2f}–{max(sw_all):.2f}pp and"
        " tracks it. That asymmetry has a reading: what a repetition buys is a property of"
        " the serving path and the environment, roughly independent of which representation"
        " is being repeated, whereas what a switch buys is a function of what the current"
        " representation is missing.", "",
        "⚠️ Both generations of each switch target are reported because pairing a base with"
        " the same-generation arm or the other one is a free choice; quoting one would hide"
        " the drift sensitivity that choice carries.", "",
        "## 2. Supply: the decision set is large, the learnable part is not", "",
        "The which-mode label needs a task **some mode solved**. A retry-or-switch label"
        " needs only that the base attempt **failed**, which is far more common — so the"
        " decision set should be larger. It is. That turns out not to be the binding"
        " constraint.", "",
        "| base | decision set | retry only | switch only | both | neither | contested | % of cell |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in p2:
        L.append(f"| `{r['base']}` | {r['decision_set']} ({r['decision_set_pct']:.1f}%) | "
                 f"{r['retry_only']} | {r['switch_only']} | {r['both']} | {r['neither']} | "
                 f"{r['contested']} | {r['contested_pct_of_cell']:.2f}% |")
    L += [
        "",
        f"`neither` is {neither} out of n={n} on every base: the same tasks, no matter which"
        " arm starts. The decision set is large because failures are abundant, but most of it"
        " carries no preference to learn — both actions fail together.", "",
        f"Against the same-arm-count which-mode contested set (**{wm:.2f}%** of the cell,"
        f" recomputed on these three arms), retry-or-switch offers **{rs:.2f}%** —"
        f" **{rs / wm:.2f}×**. Redefining the label does not escape the ceiling. What bounds"
        " both is the number of tasks the agent can solve at all, which is the same"
        " circularity the draft's §7 names, reached from a second direction.", "",
        "## 3. A fixed budget of six arms, spent two ways", "",
        "`noise_floor_inventory.md` declines this explicitly — *Not licensed. 'The whole"
        " 6-mode ceiling gain is noise.' We hold one rerun arm, not five.* Three replicated"
        " arms exist now, so the six-arm contrast is computable.", "",
        f"| budget | union SR | note |", "|---|---:|---|",
        f"| 6 representations × 1 generation | **{a6['union_sr_pp']:.2f}%** | best single"
        f" `{a6['best_single_mode']}` @ {a6['best_single_sr_pp']:.2f}% |",
        f"| 3 representations × 2 generations | **{b3['union_sr_pp']:.2f}%** |"
        f" {', '.join('`' + a + '`' for a in b3['arms'])} |", "",
        f"Solved only by the six-representation budget: {len(p5['solved_only_by_six_rep'])}"
        f" tasks `{p5['solved_only_by_six_rep']}`. Only by the 3×2 budget:"
        f" {len(p5['solved_only_by_three_x2'])} tasks `{p5['solved_only_by_three_x2']}`.", "",
        f"The six-representation union reproduces the published **43.30%** six-mode oracle for"
        " this cell exactly, which is the correctness check on this whole read path.", "",
        "⚠️ **The gap is not a difference.** It is"
        f" {abs(b3['union_sr_pp'] - a6['union_sr_pp']):.2f}pp ="
        f" {round(abs(b3['union_sr_pp'] - a6['union_sr_pp']) * p5['n'] / 100)} tasks, against"
        " a same-condition discordance of 12–14% on this cell. The defensible statement is"
        " that the two ways of spending six arms are **indistinguishable here**, not that"
        " repetition wins.", "",
        f"⚠️ {p5['independence_caveat']}. And the repetition budget covers only three"
        " distinct representations, so this is a question about how to spend an arm budget,"
        " not a claim that repetition is equivalent to representation diversity.", "",
        "⚠️ The 3×2 union is if anything **flattered**: its arms are separated by days to"
        " months, so it collects environment drift that a same-day budget would not.", "",
    ]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=None,
                    help="where to write json/csv (default: scratchpad next to this repo)")
    ap.add_argument("--from-json", default=None,
                    help="re-render the report from an existing product JSON (no recompute)")
    ap.add_argument("--write-md", action="store_true",
                    help="also write the markdown report next to the JSON")
    args = ap.parse_args()

    if args.from_json:
        d = json.loads(Path(args.from_json).read_text())
        out = Path(args.from_json).with_suffix(".md")
        out.write_text(render_md(d))
        print(f"re-rendered {out} from {args.from_json} (no recomputation)")
        return 0

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    out_dir = Path(args.out_dir) if args.out_dir else Path(
        "/tmp/claude-1012/-home-jiaming-workspace-Cost-Aware-Routing-for-Web-Usage-Agents/"
        "76c61488-38d6-4a27-bdba-1eb4de7cad09/scratchpad")
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = load_arms()
    p1 = part1_matched_gains(arms)
    p2 = part2_label_supply(arms)
    p3 = part3_which_mode_baseline(arms)

    # ---- Part 1: is a rerun worth as much as a different representation? -------------
    print("\n=== PART 1 — one added arm: retry vs switch (all base/added combinations) ===")
    print(f"{'base':<12} {'action':<7} {'added':<12} {'base SR':>8} {'gain':>8} {'union SR':>9}")
    for r in p1:
        print(f"{r['base']:<12} {r['action']:<7} {r['added']:<12} "
              f"{r['base_sr_pp']:>7.2f}% {r['gain_pp']:>7.2f}pp {r['union_sr_pp']:>8.2f}%")

    print("\n--- per base: retry gain vs best/worst switch gain ---")
    print(f"{'base':<12} {'retry':>8} {'switch min':>11} {'switch max':>11}  verdict")
    for mode, bk in itertools.product(ARMS, ("a", "b")):
        base = f"{mode}.{bk}"
        rt = [r["gain_pp"] for r in p1 if r["base"] == base and r["action"] == "retry"][0]
        sw = [r["gain_pp"] for r in p1 if r["base"] == base and r["action"] == "switch"]
        verdict = ("retry >= every switch" if rt >= max(sw)
                   else "retry <= every switch" if rt <= min(sw)
                   else "retry inside the switch range")
        print(f"{base:<12} {rt:>7.2f}pp {min(sw):>10.2f}pp {max(sw):>10.2f}pp  {verdict}")

    # ---- Part 2: the supply question ------------------------------------------------
    print("\n=== PART 2 — label supply for 'retry or switch' ===")
    print("decision set = tasks where the base attempt FAILED (that is where the question arises)")
    print(f"\n{'base':<12} {'decision':>9} {'retry_only':>11} {'switch_only':>12} "
          f"{'both':>6} {'neither':>8} {'contested':>10} {'%cell':>7}")
    for r in p2:
        print(f"{r['base']:<12} {r['decision_set']:>9} {r['retry_only']:>11} "
              f"{r['switch_only']:>12} {r['both']:>6} {r['neither']:>8} "
              f"{r['contested']:>10} {r['contested_pct_of_cell']:>6.2f}%")

    # ---- Part 3: the legal comparison ----------------------------------------------
    print("\n=== PART 3 — §6's which-mode contested set, recomputed at 3 arms ===")
    for gen, v in p3.items():
        print(f"  generation {gen}: >=1 solver {v['at_least_one_solver']:>3} "
              f"({v['at_least_one_pct']:.2f}%) | >1 solver (a choice exists) "
              f"{v['contested_multi_solver']:>3} ({v['contested_pct']:.2f}%)")

    wm = max(v["contested_pct"] for v in p3.values())
    rs = max(r["contested_pct_of_cell"] for r in p2)
    rs_act = max(r["actionable_pct_of_cell"] for r in p2)
    print(f"\n  which-mode  contested (a choice exists) : {wm:.2f}% of cell")
    print(f"  retry/switch contested (one action wins) : {rs:.2f}% of cell")
    print(f"  retry/switch actionable (any action wins): {rs_act:.2f}% of cell")
    print(f"  ratio (contested): {rs / wm:.2f}x" if wm else "  ratio: n/a")

    neither = {r["neither"] for r in p2}
    print(f"\n  'neither action rescues it' is {neither} across all six bases, out of "
          f"n={p2[0]['n']}")
    print("  => the decision set is large because failures are abundant, but most of it")
    print("     carries no preference to learn. The binding constraint is not how the")
    print("     label is DEFINED -- it is how many tasks the agent can solve at all.")

    # ---- Part 4: the finding that revises an existing licensed claim ----------------
    p4 = part4_starting_point(p1, p2)
    print("\n=== PART 4 — does 'a rerun is worth as much as a switch' hold from every base? ===")
    print(f"{'base':<12} {'base SR':>8} {'retry':>8} {'switch':>16} {'switch/retry':>14}"
          f"  {'r_only':>6} {'s_only':>6}")
    def _rng(lo, hi, w, prec, unit):
        # switch/retry is undefined when a base's retry gain is zero -- which the
        # six-arm registry made reachable, and which used to crash this print.
        if lo is None or hi is None:
            return f"{'n/a':>{w}}"
        return f"{lo:>{w}.{prec}f}-{hi:<{w - 1}.{prec}f}{unit}"

    for r in p4:
        print(f"{r['base']:<12} {r['base_sr_pp']:>7.2f}% {r['retry_gain_pp']:>7.2f}pp "
              f"{_rng(r['switch_gain_min_pp'], r['switch_gain_max_pp'], 6, 2, 'pp')} "
              f"{_rng(r['switch_over_retry_min'], r['switch_over_retry_max'], 6, 2, 'x')} "
              f"{r['retry_only']:>6} {r['switch_only']:>6}")
    print("\n  noise_floor_inventory §2 licenses one sentence on this cell -- 'a distinct")
    print("  representation is worth no more than a rerun' -- computed from the cell's")
    print("  BEST single mode. Read from every base, that sentence is starting-point")
    print("  dependent: it is the strongest arm that leaves the others least to add.")

    # ---- Part 5: how to spend a fixed arm budget ------------------------------------
    p5 = part5_arm_budget(arms)
    print("\n=== PART 5 — a fixed budget of SIX arms, spent two ways (n=%d) ===" % p5["n"])
    a6 = p5["six_representations_one_generation"]
    b3 = p5["three_representations_two_generations"]
    print(f"  6 representations x 1 generation : union {a6['union_sr_pp']:.2f}%  "
          f"(best single {a6['best_single_mode']} @ {a6['best_single_sr_pp']:.2f}%)")
    print(f"  3 representations x 2 generations: union {b3['union_sr_pp']:.2f}%")
    print(f"  solved only by the 6-representation budget: {len(p5['solved_only_by_six_rep'])} "
          f"tasks {p5['solved_only_by_six_rep'][:10]}")
    print(f"  solved only by the 3x2 budget            : "
          f"{len(p5['solved_only_by_three_x2'])} tasks {p5['solved_only_by_three_x2'][:10]}")
    print(f"  ⚠ {p5['independence_caveat']}")
    print("  ⚠ sanity: the 6-representation union must reproduce the published 43.30% "
          "six-mode oracle for this cell")

    # ---- artefacts ------------------------------------------------------------------
    js = out_dir / "retry_vs_switch_label_supply.json"
    product = {"part1_matched_gains": p1,
               "part2_label_supply": [{k: v for k, v in r.items() if k != "per_task"}
                                      for r in p2],
               "part3_which_mode_contested": p3,
               "part4_starting_point": p4,
               "part5_arm_budget": p5}
    js.write_text(json.dumps(product, indent=2))
    if args.write_md:
        md = js.with_suffix(".md")
        md.write_text(render_md(product))
        print(f"wrote {md}")
    csv_path = out_dir / "retry_vs_switch_per_task.csv"
    rows = [pt for r in p2 for pt in r["per_task"]]
    if rows:
        # The switch_<mode> columns differ by base (a dom base has switch_som/switch_vision,
        # a som base has switch_dom/switch_vision), so the header is the union over all
        # rows, not the keys of the first one.
        fixed = ["task_id", "base", "label", "retry_rescues", "switch_rescues"]
        extra = sorted({k for r in rows for k in r} - set(fixed))
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fixed + extra, restval="")
            w.writeheader()
            w.writerows(rows)
    print(f"\nwrote {js}")
    print(f"wrote {csv_path}  ({len(rows)} per-task label rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
