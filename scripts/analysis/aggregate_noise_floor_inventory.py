#!/usr/bin/env python3
"""Noise-floor inventory — every measured run-to-run floor, side by side with the
marginal oracle-ceiling gain it has to be compared against.

Why this exists
---------------
`phase0b_noise_floor.md` established the B0-classifieds floors and asked, as its own
top open item (§7.1), for *a locally-served (B1) same-mode replicate*, noting it was
queued behind the WA reddit run. That replicate already exists and had been overlooked:
the WA 10-task pilot and the WA full-104 run are the **same condition** (the full base
only deletes `task.task_ids.reddit`), so their task overlap is a same-condition rerun.

The second thing this computes is the comparison the floors were always for. A floor is
only interpretable against a gain measured with the *same functional and the same arm
count*. Adding one arm to a single-mode baseline raises the oracle ceiling by

    |{added arm solves} \\ {baseline solves}| / n

and that is true whether the added arm is a **different representation** or a **rerun of
the same one**. Those two are therefore directly comparable at one-arm margin; the
6-mode ceiling gain (five arms added) is NOT comparable to a one-rerun floor and is
reported separately, labelled with its arm count.

Scope discipline (§302 category-error retraction, §300.2 cross-GPU drift)
-------------------------------------------------------------------------
Every row carries its own scope. **No arithmetic is performed across rows.** The only
comparisons made are within one (model, site) cell at equal arm count.

Usage
-----
    .venv/bin/python3 scripts/analysis/aggregate_noise_floor_inventory.py
    .venv/bin/python3 scripts/analysis/aggregate_noise_floor_inventory.py --require-complete
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import logging
import math
import os
import sys
from pathlib import Path

LOG = logging.getLogger("noise-floor-inventory")

REPO = Path(__file__).resolve().parents[2]
# Run from the command line, sys.path[0] is scripts/analysis/ and `import scripts.…`
# raises. §2b records what that cost once already: axis_effect_size caught the ImportError,
# warned, returned an empty directory map, and wrote a full report in which every negative
# finding was vacuously true over an empty set — exit 0. Put the root on the path instead
# of catching the symptom.
sys.path.insert(0, str(REPO))
PER_TASK_CSV = REPO / "results/phantom_paper/per_task_sr.csv"
OUT_MD = REPO / "docs/analysis/cross_sites/noise_floor_inventory.md"
OUT_JSON = REPO / "docs/analysis/cross_sites/noise_floor_inventory.json"

MODE_KEYS = ["sr_dom", "sr_som", "sr_vision", "sr_ptext", "sr_pprompt", "sr_psom"]

# --- clean same-condition replicate pairs (classifieds, n=224) -----------------------
# (label, run_a, run_b). Direction matters: self_drop(a->b) = |a solves \ b solves| / n.
# Registration here is load-bearing twice over: `validate_fire_manifest.py` reads this
# list via ast.literal_eval and exempts each arm_b from ghost detection, so an UNregistered
# deliberate replicate is reported as contamination and halts aggregation (B-1951).
# ⏳ IN FLIGHT 2026-08-30 — B1 x reddit x {som, dom}, run ids
# B1_som_reddit_20260830_093638_..._R4567 and the dom cell that follows it.
# Declared before the fire in pre_run/b1_reddit_chain_launch_intent_20260826.md;
# they are the replicate arm for the two cells already bound to the July runs.
#
# ⚠️ WHEN THEY LAND, validate_fire_manifest WILL REPORT THEM AS COMPLETE GHOSTS,
# and the watchdog will fail-closed with a halt marker. That is EXPECTED, not a
# fault: ghost detection excuses a second complete run only if it is registered
# here (validate_fire_manifest.py:227 reads this list via registered_replicate_
# run_ids), and a run cannot be registered before it exists. The same sequence
# produced the 2026-08-21/23/24 reddit "ghosts" recorded in ledger §487.7.
# Do not spend a session diagnosing that alert --- the diagnostic text is also
# truncated away by experiment_watchdog.py:1185 (§492.3), so it will say nothing.
# Register the pair here once both cells are verified at 205 episodes.

CLEAN_PAIRS = [
    ("B0.cls.dom",
     "results/visualwebarena/phase1/B0_dom_classifieds_20260525_194618_553890342_530647_R21557/phase1_dom_router_0",
     "results/repro_replicates/B0_dom_classifieds_R31194_clean_replicate/phase1_dom_router_0"),
    ("B0.cls.vision",
     "results/visualwebarena/phase1/B0_vision_classifieds_20260526_141916_610351680_689390_R32024/phase1_vision_router_0",
     "results/repro_replicates/B0_vision_classifieds_R24792_clean_replicate/phase1_vision_router_0"),
    # The arm claims 1 and 3 are ABOUT. Queued 2026-08-03 precisely because the band those
    # claims are read against was measured on DOM and Vision and borrowed for SoM.
    ("B0.cls.som",
     "results/visualwebarena/phase1/B0_som_classifieds_20260526_041601_863239369_602235_R5313/phase1_som_router_0",
     "results/visualwebarena/phase1/B0_som_classifieds_20260803_084743_413015398_3677519_R30696/phase1_som_router_0"),
    # --- The three phantom arms, registered 2026-08-19 --------------------------------
    # Landed 2026-08-17/18 as cells 1-3 of the floor chain, the paid segment ($48). These
    # are the arms the phantom-space claims are ABOUT, and until now the band they were
    # read against was measured on DOM/Vision and borrowed. Launch intent was declared
    # before the chain went out (pre_run/floor_chain_launch_intent_20260817.md), so per
    # §469.7 these are registered as declared — not selected after seeing the numbers.
    #
    # POWER: d ≈ n × SR × 0.59 → P-text 20.7 / P-prompt 26.1 / P-SoM 20.7, all above the
    # d≥10 bar (§468 / B-1972), so these rows carry intervals rather than inventory.
    #
    # First use of all six arms together: 实验笔记 §470.3 computes the 2^6 assignment
    # envelope for unique-solve, the one metric replicate_metric_noise.json had exempted
    # as "cross-mode by construction" — an exemption that only held while some arm lacked
    # a replicate. See scripts/analysis/unique_solve_noise_envelope.py.
    ("B0.cls.ptext",
     "results/visualwebarena/phase1/B0_phantom_text_classifieds_20260526_233303_901232655_764510_R31183/phase1_phantom_text_router_0",
     "results/visualwebarena/phase1/B0_phantom_text_classifieds_20260817_092244_763693821_1962797_R20043/phase1_phantom_text_router_0"),
    ("B0.cls.pprompt",
     "results/visualwebarena/phase1/B0_phantom_prompt_classifieds_20260528_040546_107246795_987141_R14655/phase1_phantom_prompt_router_0",
     "results/visualwebarena/phase1/B0_phantom_prompt_classifieds_20260817_184335_813828144_2037698_R12207/phase1_phantom_prompt_router_0"),
    ("B0.cls.psom",
     "results/visualwebarena/phase1/B0_phantom_som_classifieds_20260527_191300_844420226_914570_R32031/phase1_phantom_som_router_0",
     "results/visualwebarena/phase1/B0_phantom_som_classifieds_20260818_040525_430521618_2113605_R13257/phase1_phantom_som_router_0"),
    # Cell 4 of the floor chain, registered 2026-08-19 as declared. This was the cell the
    # intent file called "first B1 floor with any power" (d≈16.6, the highest of the five
    # B1 cells). It came back at EXACTLY 0.00% discordance with zero step-count differences
    # across all 224 episodes — a byte-level reproduction, not merely the same outcomes.
    #
    # Registered anyway, per the declared policy: a floor of zero is a result, and the
    # whole point of declaring the list up front is that inconvenient cells get registered
    # too. Read together with B1.cls.som (also 0.00%), the pair says the ~12% seen on every
    # B0 arm is a property of B0's serving stack, not of the benchmark, the agent, or VWA.
    # Consequence worth stating: a B1 floor cannot bound an effect measured on B0 — see
    # 实验笔记 §470.5 / §471.
    ("B1.cls.vision",
     "results/visualwebarena/phase1/B1_vision_classifieds_20260605_012235_349047872_327631_R28622/phase1_vision_router_0",
     "results/visualwebarena/phase1/B1_vision_classifieds_20260818_132213_338416266_2187432_R26207/phase1_vision_router_0"),
    # B1 arm, registered 2026-08-17. The replicate (R28065) ran to a full 224 episodes on
    # 2026-08-16 as the first cell of the floor chain, but was never registered here, so
    # `validate_fire_manifest.py` classified it a COMPLETE ghost against canonical R31705
    # and raised the fail-closed halt marker seen at 08:04Z. It is a deliberate
    # same-condition replicate, not contamination — registering is the correct resolution
    # (the alternative, deleting a finished 224-episode run, discards real data).
    #
    # ⚠️ POWER: read this row as descriptive only. The floor's measurability scales as
    # d ≈ n × SR × 0.59; at B1's classifieds SoM success rate this lands near d≈8-10,
    # below the d<10 bar this project set for reporting a CI (§468 / B-1972). It is
    # inventory, not an interval. The B0 rows (d≈27-32) are what carry claims.
    ("B1.cls.som",
     "results/visualwebarena/phase1/B1_som_classifieds_20260604_072456_562166453_226675_R31705/phase1_som_router_0",
     "results/visualwebarena/phase1/B1_som_classifieds_20260816_150005_118867835_1831739_R28065/phase1_som_router_0"),
    # Registered floor chain cell 5, intent 20260817 + AMENDMENT 20260819; descriptive only (d≈8.3 < 10).
    ("B1.cls.dom",
     "results/visualwebarena/phase1/B1_dom_classifieds_20260603_103630_477435114_112846_R17188/phase1_dom_router_0",
     "results/visualwebarena/phase1/B1_dom_classifieds_20260819_071630_802155540_2293406_R14980/phase1_dom_router_0"),
    # Registered reframe chain, intent reframe_chain_launch_intent_20260819.md.
    ("B5.cls.dom",
     "results/visualwebarena/phase1/B5_dom_classifieds_20260820_202158_076182888_2491046_R29736/phase1_dom_router_0",
     "results/visualwebarena/phase1/B5_dom_classifieds_20260821_065801_251938920_2585213_R15476/phase1_dom_router_0"),
    # --- The first non-classifieds rows, registered 2026-08-26 ----------------
    # Phase B of the reframe chain, declared before the fire in
    # pre_run/reframe_chain_launch_intent_20260819.md ("B1-B3 | B0 x red x
    # {P-text, P-prompt, P-SoM}"), so these are registered as declared — not
    # selected after seeing the numbers (§469.7).
    #
    # WHY THEY MATTER: every floor above is classifieds. §470.3 could bound the
    # unique-solve counts only on that one cell, and claim A of the reframe
    # ("within the text side, format and prompt change WHICH tasks succeed, not
    # HOW MANY, and the change sits inside the noise") was read against a band
    # measured on a single site. These three give the text side a second site.
    #
    # POWER: d ~ n x SR x 0.59 at the archived reddit phantom SRs = 13.0-15.9,
    # above the d>=10 bar (§468 / B-1972), so these rows carry intervals.
    #
    # SITE DRIFT RULED OUT before registering, not assumed: the archive arms are
    # from 2026-06/07 and the replicates from 2026-08, two months apart, and the
    # self_drop asymmetry (12 vs 4 on P-text) had the shape of a one-way drift.
    # `compare_cross_run_same_condition.py` on all three pairs returns
    # start_url_mismatch = 0 with all 205 step-0 landings identical, and every
    # flip classified model_nondeterm. The asymmetry also does not point one way
    # across arms (P-text and P-prompt favour the archive, P-SoM the replicate),
    # which a site ageing one direction could not produce.
    ("B0.red.ptext",
     "results/visualwebarena/phase1/B0_phantom_text_reddit_20260629_140253_060787566_3384189_R32139/phase1_phantom_text_router_0",
     "results/visualwebarena/phase1/B0_phantom_text_reddit_20260821_165404_673791669_2663733_R2359/phase1_phantom_text_router_0"),
    ("B0.red.pprompt",
     "results/visualwebarena/phase1/B0_phantom_prompt_reddit_20260709/phase1_phantom_prompt_router_0",
     "results/visualwebarena/phase1/B0_phantom_prompt_reddit_20260823_075453_423269667_2958720_R11669/phase1_phantom_prompt_router_0"),
    ("B0.red.psom",
     "results/visualwebarena/phase1/B0_phantom_som_reddit_20260701_223127_661875492_3649813_R28173/phase1_phantom_som_router_0",
     "results/visualwebarena/phase1/B0_phantom_som_reddit_20260824_152956_638287802_3173740_R26550/phase1_phantom_som_router_0"),
    # --- The first LOCAL non-classifieds rows, registered 2026-09-06 ------------
    # Declared before the fire in pre_run/b1_reddit_chain_launch_intent_20260826.md
    # (cells R1/R2), so these are registered as declared (§469.7).
    #
    # WHY THEY MATTER: C1 (serving_mode_floor) grouped the floor by serving path, but
    # its LOCAL group was single-site — every local measurement in this project was
    # classifieds. The intent file called that "the weakest part of C1" and ran these
    # two to make the grouping falsifiable across sites rather than to refine it.
    #
    # POWER: declared underpowered BEFORE the fire — d ~ n x SR x 0.59 projected
    # som 9.9 / dom 8.2, both under the d>=10 bar (§468 / B-1972). Observed discordance
    # came in far below even that: som 4, dom 7. These rows are INVENTORY, not
    # intervals, and are not licensed to carry a CI anywhere downstream.
    #
    # OUTCOME vs the pre-declared landing points (som 2.0% / dom 3.4%):
    #   - "0-3% => grouping holds across two sites"     : som lands here
    #   - "3-7.39% => cross-site version not available" : dom lands here, at the very
    #     bottom of that band (0.4pp above 3%, 2.5pp below the API group's lower bound)
    #   - ">=7.39% => C1 dead as stated"                : NOT triggered
    # The intent file named the middle band "easiest to spin and hardest to report",
    # which is exactly where dom landed. Reported as such rather than rounded into the
    # comfortable band: the honest reading is that the two groups remain separated but
    # the local group is no longer uniformly under 3%.
    ("B1.red.som",
     "results/visualwebarena/phase1/B1_som_reddit_20260706/phase1_som_router_0",
     "results/visualwebarena/phase1/B1_som_reddit_20260830_093638_766738893_4123936_R4567/phase1_som_router_0"),
    ("B1.red.dom",
     "results/visualwebarena/phase1/B1_dom_reddit_20260703/phase1_dom_router_0",
     "results/visualwebarena/phase1/B1_dom_reddit_20260901_022017_161214272_197095_R1798/phase1_dom_router_0"),
    # --- B0 x reddit x {SoM, DOM, Vision}, registered 2026-09-06 ----------------
    # Declared before the fire in pre_run/b0_reddit_replicate_chain_launch_intent_20260902.md
    # (cells A1/A2/A3), landed 2026-09-02/04/06. Registered as declared (§469.7).
    #
    # WHY THEY MATTER: they complete the SIX-ARM set for B0 x reddit, which is what the
    # 2^6 unique-solve envelope needs (the envelope is not defined on five arms). Before
    # these, §470.3's cross-side asymmetry — the surviving hero's only quantitative
    # backing — rested on cls-B0 alone. See scripts/analysis/unique_solve_noise_envelope.py
    # --cell red_b0.
    #
    # POWER, as declared: som d=17.5 and dom d=17.5 carry intervals; vision d=9.3 was
    # declared INVENTORY-ONLY before it ran, 0.7 short of the bar, and is "not licensed
    # to carry a CI in any downstream table".
    # ⚠️ Its OBSERVED discordance is 12, i.e. above the d>=10 bar its PROJECTION missed.
    # Do not silently promote it on that basis: the bar was set on the projection, and
    # re-reading power from the realised count is exactly the post-hoc move the
    # declared-power discipline exists to prevent. It stays inventory until that rule is
    # changed deliberately and in writing.
    #
    # SITE DRIFT RULED OUT, not assumed (this was Reading 2 of the intent file). The
    # open CLAIM_UNVERIFIED at 台账 §478.4 suspected B0.red.ptext's 7.39% of being a
    # June<->August site drift rather than a floor, on the shape of its 3:1 flip
    # asymmetry. The intent declared: if the three new arms also come back >=2:1 and
    # same-signed, 7.39% is unusable. They came back 11/6, 13/7, 6/6 — all under 2:1 —
    # so the 3:1 is that arm's own property and 7.39% stands as a floor.
    # `compare_cross_run_same_condition.py` returns start_url_mismatch = 0 on all three,
    # with every flip classified model_nondeterm.
    ("B0.red.som",
     "results/visualwebarena/phase1/B0_som_reddit_20260627_035453_162107997_3024022_R20936/phase1_som_router_0",
     "results/visualwebarena/phase1/B0_som_reddit_20260902_194848_784669986_474818_R11761/phase1_som_router_0"),
    ("B0.red.dom",
     "results/visualwebarena/phase1/B0_dom_reddit_20260625_154833_928747130_2827521_R11344/phase1_dom_router_0",
     "results/visualwebarena/phase1/B0_dom_reddit_20260904_010441_409707415_681024_R9525/phase1_dom_router_0"),
    ("B0.red.vision",
     "results/visualwebarena/phase1/B0_vision_reddit_20260628_094255_184327569_3222015_R17559/phase1_vision_router_0",
     "results/visualwebarena/phase1/B0_vision_reddit_20260905_081817_462107405_893625_R17511/phase1_vision_router_0"),
]

# A replicate that is still running has a task set that merely LOOKS like a scored universe:
# `_pair_stats` intersects the two runs, so a pair short of the canonical set silently
# reports a floor at n=222 with no indication that two tasks never ran. Today's lesson
# (B-1928, B-1929, the empty failure_modes product) is that a partially-complete artifact
# reads exactly like a complete one, so the completeness check is explicit and fails loud.
# `--allow-partial-replicate` exists for interim inspection and marks the output.
REQUIRE_FULL_UNIVERSE = True

# --- WA reddit: pilot (10-task registered draw) vs full-104, same condition ----------
# prereg 8.8 / B-1296 registered pilot sample, reproducible from _wa_pilot_task_sample.py
WA_REGISTERED_PILOT_TASKS = [581, 584, 597, 598, 607, 635, 641, 652, 715, 729]
WA_ROOT = "results/webarena/phase1"
WA_PAIRS = {  # mode -> (pilot glob, full glob)
    "dom": ("B1_dom_wa_reddit_20260727", "B1_dom_wa_reddit_20260727_180024*"),
    "som": ("B1_som_wa_reddit_20260727", "B1_som_wa_reddit_20260728_090436*"),
    "vision": ("B1_vision_wa_reddit_20260727", "B1_vision_wa_reddit_20260729_002545*"),
    "ptext": ("B1_phantom_text_wa_reddit_20260727", "B1_phantom_text_wa_reddit_20260729_154551*"),
    "pprompt": ("B1_phantom_prompt_wa_reddit_20260727", "B1_phantom_prompt_wa_reddit_20260730_073250*"),
    # psom's 20260727 dir is NOT the registered pilot (task ids 27..584, only 2/10
    # overlap) -- it is a restarted partial of the full run. Kept out of the clean
    # estimate and reported separately.
    "psom": ("B1_phantom_som_wa_reddit_20260727", "B1_phantom_som_wa_reddit_20260730_231304*"),
}
WA_CLEAN_MODES = ["dom", "som", "vision", "ptext", "pprompt"]
# Backbone-agnostic mode -> run-dir stem, for WA cells that have no registered pilot
# (B0 x WA landed 2026-08-03). WA_PAIRS above stays B1-only: it encodes the pilot pairing.
WA_MODE_STEM = {"dom": "dom", "som": "som", "vision": "vision",
                "ptext": "phantom_text", "pprompt": "phantom_prompt", "psom": "phantom_som"}


class MissingInput(RuntimeError):
    """Fail-loud on absent inputs -- never silently degrade to a partial inventory."""


def _episode_success(condition_dir: Path) -> dict[int, int]:
    """task_id -> 0/1, skipping sr_excluded episodes (canonical scored universe)."""
    out: dict[int, int] = {}
    hits = list(condition_dir.glob("episodes/*summary*.json"))
    if not hits:
        hits = list(condition_dir.glob("*/episodes/*summary*.json"))
    for f in hits:
        try:
            s = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise MissingInput(f"unreadable episode summary {f}: {exc}") from exc
        if s.get("sr_excluded"):
            continue
        out[int(s["task_id"])] = 1 if s.get("success") else 0
    return out


def _resolve_one(pattern: str) -> Path:
    hits = [Path(p) for p in glob.glob(str(REPO / WA_ROOT / pattern))
            if os.path.isdir(p) and "ABORTED" not in p]
    if len(hits) != 1:
        raise MissingInput(f"expected exactly 1 run dir for {pattern!r}, got {len(hits)}")
    return hits[0]


def _pair_stats(a: dict[int, int], b: dict[int, int], restrict: list[int] | None = None) -> dict:
    common = sorted(set(a) & set(b))
    if restrict is not None:
        common = [t for t in common if t in set(restrict)]
    if not common:
        raise MissingInput("replicate pair has empty task intersection")
    n = len(common)
    a_not_b = [t for t in common if a[t] and not b[t]]
    b_not_a = [t for t in common if b[t] and not a[t]]
    # --- the mean-difference functional, and the sampling spread it is a draw FROM ----
    # |SR(a) - SR(b)| is one observation of a random quantity, not a bound on it. Under
    # the exchangeability null (the two runs are the same condition, so each discordant
    # task flips either way with probability 1/2) the mean difference is
    #     D = (2X - d) / n,  X ~ Binom(d, 1/2),  d = discordant count
    # so Var(2X - d) = 4 * d * 1/4 = d  and  SD(D) = sqrt(d) / n.
    # Quoting max(two draws) as "the measured floor" understates it whenever that max is
    # of the same order as this SD -- which is the case on both B0 x classifieds pairs.
    d_count = len(a_not_b) + len(b_not_a)
    null_sd_pp = math.sqrt(d_count) / n * 100 if d_count else 0.0
    return {
        "n": n,
        "sr_a": sum(a[t] for t in common) / n,
        "sr_b": sum(b[t] for t in common) / n,
        "self_drop_a_to_b_pp": len(a_not_b) / n * 100,
        "self_drop_b_to_a_pp": len(b_not_a) / n * 100,
        "discordance_pct": d_count / n * 100,
        "discordant_count": d_count,
        "mean_diff_pp": (len(a_not_b) - len(b_not_a)) / n * 100,
        "abs_mean_diff_pp": abs(len(a_not_b) - len(b_not_a)) / n * 100,
        "null_sd_mean_diff_pp": null_sd_pp,
        # |ΔSR| is a FOLDED quantity but null_sd is the SD of the unfolded D, so comparing
        # them directly (as §1b did) reads a perfectly-null observation as "same order as one
        # SD". Under D~N(0,σ) the folded mean is E|D| = σ·sqrt(2/π) ≈ 0.798σ — that is what an
        # observed |ΔSR| should be judged against.
        "null_expected_abs_pp": math.sqrt(2 / math.pi) * null_sd_pp,
        "null_one_sided_95_pp": 1.645 * null_sd_pp,
        "null_two_sided_95_pp": 1.960 * null_sd_pp,
        "flip_tasks_a_to_b": a_not_b,
        "flip_tasks_b_to_a": b_not_a,
    }


# Second field of a CLEAN_PAIRS label -> the site whose scored universe the pair
# must be restricted to. Added 2026-08-26 with the first non-classifieds rows.
# Before that the universe was `expected_scored_ids("classifieds")`, hardcoded,
# which is fail-CLOSED for a reddit pair rather than wrong: reddit task ids are
# not a subset of the classifieds universe, so `missing` would be all 224 and
# REQUIRE_FULL_UNIVERSE would raise. Registering a reddit pair therefore had to
# start here. The scope string was hardcoded the same way — the twin of the
# 2026-08-20 baseline fix, which caught "B0 x classifieds" being printed on
# every B1 row.
_SITE_OF_LABEL = {"cls": "classifieds", "red": "reddit", "shop": "shopping"}


def compute_clean_pairs(allow_partial: bool = False) -> list[dict]:
    from scripts.analysis.lib.canonical_task_universe import expected_scored_ids
    _universe: dict[str, set] = {}
    rows = []
    for label, ra, rb in CLEAN_PAIRS:
        _baseline, _site_key = label.split(".")[0], label.split(".")[1]
        try:
            _site = _SITE_OF_LABEL[_site_key]
        except KeyError:
            raise MissingInput(
                f"{label}: unknown site key {_site_key!r}; extend _SITE_OF_LABEL "
                f"(known: {sorted(_SITE_OF_LABEL)})") from None
        if _site not in _universe:
            _ids, _sha = expected_scored_ids(_site)
            _universe[_site] = set(_ids)
        scored = _universe[_site]
        pa, pb = REPO / ra, REPO / rb
        for p in (pa, pb):
            if not p.is_dir():
                raise MissingInput(f"{label}: replicate arm not on disk: {p}")
        sa, sb = _episode_success(pa), _episode_success(pb)
        # Completeness gate — see REQUIRE_FULL_UNIVERSE. A replicate mid-flight yields a
        # perfectly well-formed floor over whatever it has finished, and nothing downstream
        # can tell that from a finished one.
        missing = sorted(scored - (set(sa) & set(sb)))
        if missing:
            msg = (f"{label}: {len(missing)} of {len(scored)} canonical tasks absent from the "
                   f"pair (first few: {missing[:5]}). A floor computed here would be over "
                   f"n={len(set(sa) & set(sb))} while every consumer reads it as n={len(scored)}.")
            if REQUIRE_FULL_UNIVERSE and not allow_partial:
                raise MissingInput(msg + "  Re-run when the replicate finishes, or pass "
                                         "--allow-partial-replicate to inspect it anyway.")
            LOG.warning("PARTIAL %s", msg)
        st = _pair_stats(sa, sb, restrict=sorted(scored))
        # The baseline was hardcoded "B0" here until 2026-08-20. CLEAN_PAIRS has held
        # B1 pairs since B1.cls.som was registered, so every B1 row in the published
        # table read "B0 x classifieds" — a table whose whole point is that B0 and B1
        # floors differ by an order of magnitude. Derive it from the label instead.
        st.update(label=label, site=_site,
                  scope=(f"{_baseline} x {_site}, canonical n={len(scored)}" if not missing
                         else f"{_baseline} x {_site}, PARTIAL n={st['n']} of {len(scored)}"),
                  partial=bool(missing), n_missing=len(missing),
                  arm_a=str(pa.relative_to(REPO)), arm_b=str(pb.relative_to(REPO)))
        rows.append(st)
        LOG.info("clean pair %s: self_drop %.2f / %.2f pp, discordance %.2f%% (n=%d)",
                 label, st["self_drop_a_to_b_pp"], st["self_drop_b_to_a_pp"],
                 st["discordance_pct"], st["n"])
    return rows


def compute_wa_floor() -> dict:
    """B1 x WA-reddit: registered-pilot rerun vs full-104, pooled over 5 clean modes."""
    per_mode, pooled_ab, pooled_ba, pooled_n = {}, 0, 0, 0
    for m in WA_CLEAN_MODES:
        pilot, full = WA_PAIRS[m]
        st = _pair_stats(_episode_success(_resolve_one(pilot)),
                         _episode_success(_resolve_one(full)),
                         restrict=WA_REGISTERED_PILOT_TASKS)
        per_mode[m] = st
        pooled_n += st["n"]
        pooled_ab += len(st["flip_tasks_a_to_b"])
        pooled_ba += len(st["flip_tasks_b_to_a"])
    pooled = {
        "n": pooled_n,
        "self_drop_a_to_b_pp": pooled_ab / pooled_n * 100,
        "self_drop_b_to_a_pp": pooled_ba / pooled_n * 100,
        "discordance_pct": (pooled_ab + pooled_ba) / pooled_n * 100,
        "per_mode": per_mode,
        # ⚠️ pooled_n is 5 modes x the SAME 10 tasks, so the 50 observations contain 10
        # independent tasks; treating them as 50 understates the spread by up to sqrt(5).
        # The per-mode floors (each n=10) are the like-for-like comparison against the VWA
        # per-arm floors at n=224, and the widest of them is the honest band edge.
        "n_independent_tasks": len(WA_REGISTERED_PILOT_TASKS),
        "pooling_note": ("pooled over 5 modes on one shared 10-task draw; per-mode floors "
                         "below are the unpooled comparison"),
        "per_mode_self_drop_max_pp": max(
            max(v["self_drop_a_to_b_pp"], v["self_drop_b_to_a_pp"]) for v in per_mode.values()),
        "per_mode_self_drop_min_pp": min(
            min(v["self_drop_a_to_b_pp"], v["self_drop_b_to_a_pp"]) for v in per_mode.values()),
        "scope": "B1 x WA-reddit, registered 10-task pilot draw x 5 modes",
    }
    LOG.info("WA B1 floor (pooled, clean): self_drop %.2f / %.2f pp, discordance %.2f%% (n=%d)",
             pooled["self_drop_a_to_b_pp"], pooled["self_drop_b_to_a_pp"],
             pooled["discordance_pct"], pooled_n)

    # psom's 20260727 dir is a restarted partial of the full run, not the pilot draw.
    pilot, full = WA_PAIRS["psom"]
    aux = _pair_stats(_episode_success(_resolve_one(pilot)), _episode_success(_resolve_one(full)))
    aux["scope"] = ("B1 x WA-reddit, P-SoM: 20260727 dir is a RESTARTED PARTIAL of the full "
                    "run (task ids 27..584, 2/10 overlap with the registered draw), NOT the "
                    "pilot -- different comparison, reported separately, never pooled above")
    pooled["psom_restart_pair"] = aux
    return pooled


def _marginal_gain(succ: dict[str, dict[int, int]], tasks: list[int]) -> dict:
    """Best single mode, 6-mode oracle, and the gain from adding ONE more arm."""
    n = len(tasks)
    srs = {m: sum(succ[m][t] for t in tasks) / n for m in succ}
    best = max(srs, key=lambda k: srs[k])
    oracle = sum(1 for t in tasks if any(succ[m][t] for m in succ)) / n
    marg = sorted(((m, sum(1 for t in tasks if succ[m][t] and not succ[best][t]) / n)
                   for m in succ if m != best), key=lambda kv: -kv[1])
    return {
        "n": n,
        "best_mode": best,
        "best_single_sr_pct": srs[best] * 100,
        "oracle_6mode_sr_pct": oracle * 100,
        "gain_5_arms_added_pp": (oracle - srs[best]) * 100,
        "gain_1_best_distinct_arm_pp": marg[0][1] * 100,
        "gain_1_best_distinct_arm_mode": marg[0][0],
        "gain_1_second_arm_pp": marg[1][1] * 100,
    }


def compute_vwa_margins() -> dict[str, dict]:
    """Per-cell marginal gains, restricted to the CANONICAL SCORED universe.

    `generate_per_task_sr.py` still emits the *collected* set — it is on the
    `UNIVERSE_TRIAGE_PENDING` ratchet in `tests/test_universe_consumption_lint.py`
    for exactly that reason — so reddit arrives with 205 rows including the two
    AMENDMENT_08 protocol-excluded tasks (58, 160). The paper scores 203. We
    intersect here rather than trusting the CSV; caught by the 2026-08-01 Phase 2
    cross-AI audit after the first version of this file quoted n=205 for reddit.
    """
    if not PER_TASK_CSV.exists():
        raise MissingInput(
            f"{PER_TASK_CSV} absent -- regenerate with:\n"
            "  .venv/bin/python3 scripts/analysis/generate_per_task_sr.py "
            "--out results/phantom_paper/per_task_sr.csv")
    sys.path.insert(0, str(REPO))
    from scripts.analysis.lib.canonical_task_universe import expected_scored_ids

    cells: dict[str, list[dict]] = {}
    for r in csv.DictReader(PER_TASK_CSV.open()):
        cells.setdefault(r["cell_id"], []).append(r)
    out = {}
    for cid, rows in sorted(cells.items()):
        site = rows[0]["site"]
        scored, sha = expected_scored_ids(site)
        kept = [r for r in rows if int(r["task_id"]) in scored]
        dropped = len(rows) - len(kept)
        if dropped:
            LOG.info("%s: dropped %d row(s) outside the canonical scored universe "
                     "(%s, n=%d, sha=%s)", cid, dropped, site, len(scored), sha[:12])
        if len(kept) != len(scored):
            raise MissingInput(
                f"{cid}: {len(kept)} rows after restriction but the canonical "
                f"{site} universe has {len(scored)} -- refusing to report a "
                "marginal gain on an incomplete cell")
        idx = list(range(len(kept)))
        succ = {m: {i: (1 if int(float(kept[i][m])) >= 1 else 0) for i in idx}
                for m in MODE_KEYS}
        out[cid] = _marginal_gain(succ, idx)
        out[cid]["universe_sha"] = sha
    return out


def compute_wa_margin(baseline: str = "B1") -> dict:
    """Arm-matched marginal gain on <baseline> x WA-reddit.

    B1 resolves through WA_PAIRS (its full-run globs carry the registered-pilot pairing);
    any other backbone globs on the run-id suffix. B0 x WA landed 2026-08-03 and has no
    pilot draw, so it contributes a margin but no floor -- the floor row stays B1-only.
    """
    succ = {}
    if baseline == "B1":
        for m, (_pilot, full) in WA_PAIRS.items():
            succ[m] = _episode_success(_resolve_one(full))
    else:
        for m, stem in WA_MODE_STEM.items():
            succ[m] = _episode_success(_resolve_one(f"{baseline}_{stem}_wa_reddit_2026*_R*"))
    tasks = sorted(set.intersection(*[set(v) for v in succ.values()]))
    return _marginal_gain(succ, tasks)


def render(data: dict) -> str:
    L: list[str] = []
    add = L.append
    add("---")
    add("type: analysis")
    add("status: complete")
    add(f"created: {data['generated_for_date']}")
    add("purpose: every measured run-to-run noise floor, next to the arm-count-matched "
        "oracle-ceiling gain it must be judged against")
    add("scope_warning: every number carries its own scope; do NOT do arithmetic across "
        "rows (§302 category-error retraction, §300.2 cross-GPU drift). The only "
        "comparisons drawn are within one (model, site) cell at equal arm count.")
    add("producer: scripts/analysis/aggregate_noise_floor_inventory.py")
    add("---")
    add("")
    add("# Noise-floor inventory")
    add("")
    add("Regenerate: `.venv/bin/python3 scripts/analysis/aggregate_noise_floor_inventory.py`")
    add("")
    add("## 1. Measured same-condition run-to-run floors")
    add("")
    add("`self_drop(a→b) = |{a solves} ∖ {b solves}| / n`. Two runs of ONE `(model, site, "
        "mode)`; the pair is a rerun, so this is the ceiling gain buyable by adding a "
        "**rerun** as one extra arm.")
    add("")
    add("| pair | scope | n | self_drop a→b | self_drop b→a | discordance |")
    add("|---|---|---|---|---|---|")
    for r in data["clean_pairs"]:
        add(f"| `{r['label']}` | {r['scope']} | {r['n']} | **{r['self_drop_a_to_b_pp']:.2f}pp** "
            f"| **{r['self_drop_b_to_a_pp']:.2f}pp** | {r['discordance_pct']:.2f}% |")
    w = data["wa_floor"]
    add(f"| `B1.wa-red` (**new**) | {w['scope']} | {w['n']} | **{w['self_drop_a_to_b_pp']:.2f}pp** "
        f"| **{w['self_drop_b_to_a_pp']:.2f}pp** | {w['discordance_pct']:.2f}% |")
    add("")
    add("### 1b. The mean-difference floor is two draws, not a bound")
    add("")
    add("The set-difference functional above is the one claim 1 needs. Claims 3 and 4 "
        "compare **mean** success rates between two modes, and the matched floor for that "
        "is `|SR(a) − SR(b)|` on the same replicate pairs — which is where the band "
        "`0.89–2.23pp` comes from. Those two numbers are **one observation each of a "
        "random quantity**, and the quantity's own spread is computable from the "
        "discordant counts already in the table above. Under the exchangeability null "
        "(same condition, so each discordant task flips either way with probability ½) "
        "`D = (2X − d)/n` with `X ~ Binom(d, ½)`, so `SD(D) = √d / n`:")
    add("")
    add("| pair | n | discordant d | observed \\|ΔSR\\| | **SD(ΔSR) under the null** | one-sided 95% | two-sided 95% |")
    add("|---|---|---|---|---|---|---|")
    band = data["floor_band"]
    for r in band["rows"]:
        add(f"| `{r['label']}` | {r['n']} | {r['discordant_count']} "
            f"| {r['abs_mean_diff_pp']:.2f}pp | **{r['null_sd_mean_diff_pp']:.2f}pp** "
            f"| {r['null_one_sided_95_pp']:.2f}pp | ±{r['null_two_sided_95_pp']:.2f}pp |")
    add("")
    add(f"⚠️ **The band's upper edge ({band['observed_max_pp']:.2f}pp) is of the same order "
        f"as one standard deviation ({band['null_sd_min_pp']:.2f}–{band['null_sd_max_pp']:.2f}pp).** "
        "So \"clears the band\" is not \"clears the noise\": an effect has to reach roughly "
        f"**{band['one_sided_95_min_pp']:.2f}–{band['one_sided_95_max_pp']:.2f}pp** before a "
        "single rerun would be unlikely to produce it by itself. Both readings are reported "
        "because they answer different questions — *what did repetition actually deliver* "
        "(the two draws) versus *what could repetition deliver* (the null spread). Reading a "
        "2.2pp effect against a 2.23pp \"measured floor\" is comparing a draw to a draw.")
    add("")
    add("🚫 **Scope of that threshold — it is NOT a general significance bar** (/stress gemini "
        "G1, 2026-08-16). `SD(ΔSR) = √d / n` is derived from **this pair's own discordance** "
        "`d`, i.e. from re-running ONE arm. A cross-mode contrast (say SoM − DOM) has its own, "
        "larger `d`, hence its own wider null; judging it against a rerun-derived bar borrows "
        "`Var(A − A′)` to adjudicate `A − B` and is a category error. The number above answers "
        "exactly one question — *could a single rerun of the same arm have manufactured this?* "
        "— which is the arm-count-matched comparison §2 makes. For any other contrast, compute "
        "that contrast's own off-diagonal counts (McNemar / its own permutation test).")
    add("")
    add("⚠️ This null assumes only exchangeability of the two runs; it does **not** model "
        "environment drift, which is one-directional and is what the P-SoM restart pair "
        "below shows. Where drift is present the true spread is larger than `√d / n`, so "
        "these thresholds are themselves a lower bound.")
    add("")
    add("### The B1 floor was not missing — it was unrecognised")
    add("")
    add("`phase0b_noise_floor.md` §7.1 lists *a locally-served (B1) same-mode replicate* as "
        "the top thing still needed, queued behind the WA run. It already existed: the WA "
        "10-task pilot and the WA full-104 run are the **same condition** — "
        "`exp_v2_wa_full_reddit_base.yaml` only deletes `task.task_ids.reddit` — so their "
        "task overlap is a same-condition rerun. Per-mode:")
    add("")
    add("| mode | paired n | \\|pilot ∖ full\\| | \\|full ∖ pilot\\| |")
    add("|---|---|---|---|")
    for m in WA_CLEAN_MODES:
        s = w["per_mode"][m]
        add(f"| {m} | {s['n']} | {len(s['flip_tasks_a_to_b'])} {s['flip_tasks_a_to_b'] or ''} "
            f"| {len(s['flip_tasks_b_to_a'])} {s['flip_tasks_b_to_a'] or ''} |")
    add("")
    aux = w["psom_restart_pair"]
    add(f"P-SoM is excluded from the pooled figure and reported alone: its `20260727` "
        f"directory is a **restarted partial of the full run**, not the registered pilot "
        f"draw (task ids 27..584, 2/10 overlap). On its {aux['n']} shared tasks it shows "
        f"{len(aux['flip_tasks_a_to_b'])} / {len(aux['flip_tasks_b_to_a'])} one-directional "
        f"flips ({aux['discordance_pct']:.2f}% discordance) — one-directional, which reads "
        "more like state drift than symmetric noise.")
    add("")
    add("**This refutes a live `CLAIM_UNVERIFIED`** — *\"B1 是完全确定性的 (do_sample=False "
        "贪婪解码 → 重跑 bit-identical)\"*. Step-level greedy determinism (§298.2 133/133, "
        "§397.10 within-group 1.000) is real and is **not the same property**: an episode "
        "also carries site state, wall-clock, and session lifetime. The 2026-07-29 decision "
        "*\"B1/B2 重跑地板不用测\"* rested on the step-level evidence and does not survive.")
    add("")
    add("⚠️ **This floor includes environment drift, not only stochasticity.** The two runs "
        "are days apart, and `require_reset` is a no-op on reddit (§402), so subscriptions "
        "accumulate across episodes. For the comparison in §2 that is correct — the paper's "
        "own conditions were also run at different times and carry the same drift — but the "
        "quantity must be named *run-to-run including environment drift*, never *decoding "
        "stochasticity*.")
    add("")
    add("## 2. The arm-count-matched comparison")
    add("")
    add("A floor is only interpretable against a gain of the **same functional at the same "
        "arm count**. Adding one arm to a single-mode baseline raises the oracle ceiling by "
        "`|{added} ∖ {baseline}| / n` — identical in form to `self_drop`, whether the added "
        "arm is a different representation or a rerun.")
    add("")
    add("| cell | best single mode | +1 best **distinct representation** | +1 **rerun** (measured floor) | verdict |")
    add("|---|---|---|---|---|")
    for cid, label, floor_txt, floor_lo, floor_hi in data["head_to_head"]:
        g = data["margins"][cid]
        one = g["gain_1_best_distinct_arm_pp"]
        if floor_lo is None:
            verdict = "no floor measured on this cell"
        elif one <= floor_hi:
            verdict = "**indistinguishable — inside the rerun band**"
        else:
            verdict = f"above the band by {one - floor_hi:.2f}pp"
        add(f"| {label} | {g['best_mode'].replace('sr_', '')} @ {g['best_single_sr_pct']:.2f}% "
            f"| **{one:.2f}pp** ({g['gain_1_best_distinct_arm_mode'].replace('sr_', '')}) "
            f"| {floor_txt} | {verdict} |")
    add("")
    add(f"Two cells carry a floor, and they differ in model family, benchmark and serving "
        f"path. On `B0 · VWA-cls` the extra representation lands **inside** the rerun band. "
        f"On `B1 · WA-red` it lands **inside** the honest band. ⚠️ Corrected 2026-08-04: it "
        f"read *just outside by 0.81pp* against a band pooled over 5 modes on one shared "
        f"10-task draw — 50 observations carrying 10 independent tasks. Against the "
        f"unpooled per-mode floors the gain is comfortably inside. "
        f"the same order, and on a floor estimated from only n=50. Neither cell shows a "
        f"representation arm worth appreciably more than a rerun arm; one shows it worth "
        f"no more at all.")
    add("")
    _cb = data.get("cls_band", {})
    if _cb.get("n_arms", 0) > 1:
        add(f"⚠️ **`B0 · VWA-cls` now carries {_cb['n_arms']} replicated arms, not one** "
            f"({_cb['arm_names']}) — the band above is the min/max over all of them. "
            + (f"The `som` pair landed 2026-08-03 and it is the one that matters most: "
               f"**claim 3 is about the fused arm, and until that day its floor was "
               f"borrowed from DOM and Vision.** The borrowed band turned out to be right "
               f"— SoM's own set-difference floor {_cb['som_lo']:.2f}–{_cb['som_hi']:.2f}pp "
               f"sits inside it and its mean-difference draw is {_cb['som_absdiff']:.2f}pp, "
               f"matching DOM's, so no number downstream moves. That is a robustness "
               f"result rather than a correction, and it is worth more than the numbers: "
               f"the claim no longer rests on an extrapolation."
               if _cb.get("has_som") else ""))
    add("")
    add("### What this licenses, and what it does not")
    add("")
    add("**Licensed on `B0 x VWA-cls`.** At the one-arm margin a distinct representation is "
        "worth no more than a rerun of the same representation: same cell, same `n`, same "
        "functional, same arm count.")
    add("")
    add("⚠️ **The WA row does not have that property and must not be read as if it did.** "
        "Its rerun figure pools five mode-specific pilot-vs-full comparisons over the "
        "**10 registered pilot tasks**, giving `pooled_n=50` panels over ten distinct tasks — "
        "while the distinct-arm column beside it is computed on **104** tasks starting from "
        "`dom`. Worse for the comparison, `dom`'s own pilot-vs-full is **0 flips in either "
        "direction**; the 2-4pp band is generated entirely by `vision` and `pprompt`. So it is "
        "not the rerun of the arm the row's baseline names, the two columns do not share `n`, "
        "and the five panels are correlated through the same ten tasks. Read the WA line as "
        "*some* arms of this cell move by 2-4pp under repetition, not as a floor for `dom`. "
        "(codex Mode B, §H stress P1-6, 2026-08-02.)")
    add("")
    add("**Not licensed.** *\"The whole 6-mode ceiling gain is noise.\"* We hold one rerun "
        "arm, not five, and reruns have their own diminishing returns. The five-arm gain is "
        "reported below **with its arm count attached** and is never set against a one-rerun "
        "floor.")
    add("")
    add("| cell | best single | 6-mode oracle | gain, **5 arms added** |")
    add("|---|---|---|---|")
    for cid, label, *_ in data["head_to_head"]:
        g = data["margins"][cid]
        add(f"| {label} | {g['best_single_sr_pct']:.2f}% | {g['oracle_6mode_sr_pct']:.2f}% "
            f"| {g['gain_5_arms_added_pp']:.2f}pp |")
    add("")
    add("## 3. Consequence for the paper's four-step spine")
    add("")
    add("| step | effect size | floor it meets | outcome |")
    add("|---|---|---|---|")
    add("| ① a real ceiling exists | 5-arm gain 4.39–16.07pp | one rerun buys 2.0–7.6pp | "
        "**survives, but the headline needs the rerun baseline printed next to it** |")
    add("| ② H3 axes are structural | 1.35 / 2.09pp pooled | lowest floor measured 2.0pp | "
        "**does not survive as a positive claim** — below even the most permissive floor |")
    add("| ③ structure < rerun floor | — | — | **this is the floor finding; now on two "
        "deployment forms, not one backbone** |")
    add("| ④ not learnable | 0/6 Pareto | — | **survives; noise only strengthens a "
        "negative result** |")
    add("")
    add("Noise destroys positive claims. Of this paper's load-bearing steps, ③ and ④ are "
        "negative, ② was already demoted to weak evidence on 2026-07-28, and ① is the only "
        "positive one left — which is why §2's caveat is the whole cost.")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", default="2026-08-01", help="stamp written into frontmatter")
    p.add_argument("--allow-partial-replicate", action="store_true",
                   help="compute a floor from a replicate that has not finished; the row "
                        "is marked PARTIAL and must not be quoted as a floor")
    p.add_argument("--require-complete", action="store_true",
                   help="exit 2 if any input is missing instead of raising")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        clean = compute_clean_pairs(allow_partial=args.allow_partial_replicate)
        wa_floor = compute_wa_floor()
        margins = compute_vwa_margins()
        margins["wa_red_B1"] = compute_wa_margin("B1")
        margins["wa_red_B0"] = compute_wa_margin("B0")
    except MissingInput as exc:
        LOG.error("missing input: %s", exc)
        return 2 if args.require_complete else 1

    # Derived over EVERY B0-classifieds replicate pair, not over a hardcoded dom+vision.
    # The SoM pair landed 2026-08-03 and appeared in §1 while this band, four lines below
    # it, still described two arms — a table and the prose above it disagreeing, which is
    # the defect class §4d of the summary is about. Adding a fourth pair now needs no edit.
    cls_pairs = [r for r in clean if r["label"].startswith("B0.cls.")]
    # Defined here, next to its sibling, because head_to_head below needs it. It
    # used to be defined 100 lines further down beside red_band, which made the
    # first attempt at a red_B0 head_to_head row raise UnboundLocalError.
    red_pairs = [r for r in clean if r["label"].startswith("B0.red.")]
    _drops = [v for r in cls_pairs
              for v in (r["self_drop_a_to_b_pp"], r["self_drop_b_to_a_pp"])]
    lo, hi = min(_drops), max(_drops)
    n_cls_arms = len(cls_pairs)
    cls_arm_names = ", ".join(r["label"].rsplit(".", 1)[-1] for r in cls_pairs)
    wlo = min(wa_floor["self_drop_a_to_b_pp"], wa_floor["self_drop_b_to_a_pp"])
    whi = max(wa_floor["self_drop_a_to_b_pp"], wa_floor["self_drop_b_to_a_pp"])

    # red_B0 joined 2026-08-26, the first non-classifieds row here. Its band covers
    # the TEXT SIDE only (three phantom arms); consumers must gate on
    # red_band["replicated_side"] before reading it against an image-bearing arm.
    _red_row = []
    if red_pairs:
        _rlo = min(v for r in red_pairs
                   for v in (r["self_drop_a_to_b_pp"], r["self_drop_b_to_a_pp"]))
        _rhi = max(v for r in red_pairs
                   for v in (r["self_drop_a_to_b_pp"], r["self_drop_b_to_a_pp"]))
        _red_row = [("red_B0", "B0 · VWA-red (n=%d, text side only)" % red_pairs[0]["n"],
                     "%.2f – %.2fpp" % (_rlo, _rhi), _rlo, _rhi)]

    head_to_head = [
        ("cls_B0", "B0 · VWA-cls (n=224)", f"{lo:.2f} – {hi:.2f}pp", lo, hi),
        *_red_row,
        # ⚠️ The pooled band (wlo..whi over n=50) understates the spread: those 50 obs are
        # 5 modes on ONE shared 10-task draw, so there are 10 independent tasks, not 50.
        # The per-mode floors are the like-for-like comparison and they span far wider —
        # judging an added-arm gain against the pooled band is judging it against a band
        # narrowed by counting correlated observations as independent.
        # NOTE: single-quoted keys inside these f-strings on purpose. Same-quote nesting
        # (PEP 701) parses only on Python 3.12+, and this file is READ BY AST on the fire
        # host, which runs 3.10 — `validate_fire_manifest.registered_replicate_run_ids()`
        # catches SyntaxError and fails CLOSED, so a 3.12-only line here silently empties
        # the replicate registry and every deliberate replicate is reported as a ghost.
        # Empirical 2026-08-17: that is exactly what happened after this file was rsynced
        # to the A100 — B0.cls.som went from "registered replicate" to GHOST with no edit
        # to its own entry. Keep this module parseable on 3.10.
        ("wa_red_B1",
         f"B1 · WA-red (n=104; floor = 5 modes × {wa_floor['n_independent_tasks']} shared tasks)",
         f"{wa_floor['per_mode_self_drop_min_pp']:.2f} – "
         f"{wa_floor['per_mode_self_drop_max_pp']:.2f}pp "
         f"*(pooled would read {wlo:.2f}–{whi:.2f})*",
         wa_floor["per_mode_self_drop_min_pp"], wa_floor["per_mode_self_drop_max_pp"]),
        ("wa_red_B0", "B0 · WA-red (n=104; no pilot → no floor)", "—", None, None),
        ("cls_B1", "B1 · VWA-cls (n=224)", "—", None, None),
        ("cls_B2", "B2 · VWA-cls (n=224)", "—", None, None),
        ("red_B1", "B1 · VWA-red (n=203)", "—", None, None),
        ("red_B2", "B2 · VWA-red (n=203)", "—", None, None),
    ]

    # --- the mean-difference band, and the sampling spread it is drawn from -----------
    # Consumers (aggregate_fusion_premium.py) read `floor_band` rather than hardcoding a
    # literal tuple: FLOOR_MEAN_PP was a hand-copied constant for months.
    band_rows = [
        {k: r[k] for k in ("label", "scope", "n", "discordant_count", "abs_mean_diff_pp",
                           "null_sd_mean_diff_pp", "null_one_sided_95_pp",
                           "null_two_sided_95_pp")}
        for r in clean
    ]
    _obs = [r["abs_mean_diff_pp"] for r in band_rows]
    _sd = [r["null_sd_mean_diff_pp"] for r in band_rows]
    _os95 = [r["null_one_sided_95_pp"] for r in band_rows]
    floor_band = {
        "functional": "mean difference |SR(a) - SR(b)| on same-condition replicate pairs",
        "rows": band_rows,
        # what repetition actually delivered -- two draws, NOT a bound
        "observed_min_pp": round(min(_obs), 2),
        "observed_max_pp": round(max(_obs), 2),
        "n_draws": len(band_rows),
        # what repetition COULD deliver -- the null spread those draws came from
        "null_sd_min_pp": round(min(_sd), 2),
        "null_sd_max_pp": round(max(_sd), 2),
        "one_sided_95_min_pp": round(min(_os95), 2),
        "one_sided_95_max_pp": round(max(_os95), 2),
        "reading": ("observed_* is a range of 2 draws and must never be quoted as a "
                    "threshold on its own; one_sided_95_* is the level an effect must "
                    "reach before a single rerun would be unlikely to produce it"),
    }

    _som = [r for r in cls_pairs if r["label"].endswith(".som")]
    cls_band = {"n_arms": n_cls_arms, "arm_names": cls_arm_names,
                "band_lo_pp": lo, "band_hi_pp": hi, "has_som": bool(_som)}
    if _som:
        _s0 = _som[0]
        cls_band.update(
            som_lo=min(_s0["self_drop_a_to_b_pp"], _s0["self_drop_b_to_a_pp"]),
            som_hi=max(_s0["self_drop_a_to_b_pp"], _s0["self_drop_b_to_a_pp"]),
            som_absdiff=_s0["abs_mean_diff_pp"])

    # Per-site companion to cls_band, added 2026-08-26 with the first reddit pairs.
    # WHY A SEPARATE BAND AND NOT A WIDER SHARED ONE: §477.2 settled that a
    # threshold is a PER-ARM quantity — taking a min/max across arms lets adding a
    # replicate on an unrelated arm weaken a conclusion about a different one. The
    # same argument applies across sites, only more so. floor_band stays the
    # all-pairs summary (its one_sided_95_* is unchanged by these rows, checked),
    # but a reddit effect must be read against reddit's own band, not against a
    # band whose min/max is set by classifieds.
    # NOTE: no same-quote f-string nesting below — this module is AST-parsed on the
    # A100 under Python 3.10 (see the PEP 701 note above); a 3.12-only line here
    # silently empties the replicate registry.
    red_band = {"n_arms": len(red_pairs)}
    if red_pairs:
        _rdrops = [v for r in red_pairs
                   for v in (r["self_drop_a_to_b_pp"], r["self_drop_b_to_a_pp"])]
        _rabs = [r["abs_mean_diff_pp"] for r in red_pairs]
        _r95 = [r["null_one_sided_95_pp"] for r in red_pairs]
        _rarm_names = ", ".join(r["label"].rsplit(".", 1)[-1] for r in red_pairs)
        red_band.update(
            arm_names=_rarm_names,
            band_lo_pp=min(_rdrops), band_hi_pp=max(_rdrops),
            observed_min_pp=min(_rabs), observed_max_pp=max(_rabs),
            one_sided_95_min_pp=min(_r95), one_sided_95_max_pp=max(_r95),
            n=red_pairs[0]["n"],
            # Which SIDE the replicated arms sit on (TERMS.md §1.1: text | combined |
            # visual). Downstream must check an effect's arm against this before
            # reading the band — §477.2 fixed thresholds as a per-arm quantity, and
            # a text-side floor is not a comparator for an image-bearing arm.
            replicated_arms=[r["label"].rsplit(".", 1)[-1] for r in red_pairs],
            replicated_side="text",
            side_of_arm=dict(dom="text", ptext="text", pprompt="text", psom="text",
                             som="combined", vision="visual"),
            reading=("text-side arms only (the three phantom modes); reddit has no "
                     "replicated dom/som/vision arm, so this band must not be read "
                     "as a floor for the image-bearing side of this site"))

    data = {"generated_for_date": args.date, "clean_pairs": clean, "wa_floor": wa_floor,
            "margins": margins, "head_to_head": head_to_head, "floor_band": floor_band,
            "red_band": red_band,
            "cls_band": cls_band}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    md = render(data)
    OUT_MD.write_text(md)
    LOG.info("wrote %s (%d bytes) + %s", OUT_MD.relative_to(REPO), len(md),
             OUT_JSON.relative_to(REPO))
    LOG.info("md sha256 %s", hashlib.sha256(md.encode()).hexdigest()[:16])
    return 0


if __name__ == "__main__":
    sys.exit(main())
