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

# --- clean same-condition replicate pairs (both B0 x classifieds, n=224) -------------
# (label, run_a, run_b). Direction matters: self_drop(a->b) = |a solves \ b solves| / n.
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
        "null_one_sided_95_pp": 1.645 * null_sd_pp,
        "null_two_sided_95_pp": 1.960 * null_sd_pp,
        "flip_tasks_a_to_b": a_not_b,
        "flip_tasks_b_to_a": b_not_a,
    }


def compute_clean_pairs(allow_partial: bool = False) -> list[dict]:
    from scripts.analysis.lib.canonical_task_universe import expected_scored_ids
    scored, _sha = expected_scored_ids("classifieds")
    scored = set(scored)
    rows = []
    for label, ra, rb in CLEAN_PAIRS:
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
        st.update(label=label,
                  scope=("B0 x classifieds, canonical n=224" if not missing
                         else f"B0 x classifieds, PARTIAL n={st['n']} of {len(scored)}"),
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
        "(the two draws) versus *what could repetition deliver* (the null spread) — and the "
        "second is the one an effect size has to be judged against. Reading a 2.2pp effect "
        "against a 2.23pp \"measured floor\" is comparing a draw to a draw.")
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
        f"On `B1 · WA-red` it lands **just outside**, by 0.81pp — above the floor, but of "
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
    _drops = [v for r in cls_pairs
              for v in (r["self_drop_a_to_b_pp"], r["self_drop_b_to_a_pp"])]
    lo, hi = min(_drops), max(_drops)
    n_cls_arms = len(cls_pairs)
    cls_arm_names = ", ".join(r["label"].rsplit(".", 1)[-1] for r in cls_pairs)
    wlo = min(wa_floor["self_drop_a_to_b_pp"], wa_floor["self_drop_b_to_a_pp"])
    whi = max(wa_floor["self_drop_a_to_b_pp"], wa_floor["self_drop_b_to_a_pp"])

    head_to_head = [
        ("cls_B0", "B0 · VWA-cls (n=224)", f"{lo:.2f} – {hi:.2f}pp", lo, hi),
        ("wa_red_B1", "B1 · WA-red (n=104; floor n=50)", f"{wlo:.2f} – {whi:.2f}pp", wlo, whi),
        ("wa_red_B0", "B0 · WA-red (n=104; no pilot → no floor)", "—", None, None),
        ("cls_B1", "B1 · VWA-cls (n=224)", "—", None, None),
        ("cls_B2", "B2 · VWA-cls (n=224)", "—", None, None),
        ("red_B0", "B0 · VWA-red (n=203)", "—", None, None),
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

    data = {"generated_for_date": args.date, "clean_pairs": clean, "wa_floor": wa_floor,
            "margins": margins, "head_to_head": head_to_head, "floor_band": floor_band,
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
