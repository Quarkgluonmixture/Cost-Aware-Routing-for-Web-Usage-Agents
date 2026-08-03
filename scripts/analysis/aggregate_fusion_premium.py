#!/usr/bin/env python3
"""Is the fused mode's accuracy advantage a counted observation or a tested one?

The claim in §4.2 is that the fused representation (SoM: annotated screenshot + mark legend,
dearest in 5/6 cells) does not earn its premium. As first written it rested on two things, and
both were weaker than they needed to be:

  1. "no cell clears the rerun floor" was a COUNT over seven cells, not a test, and
  2. the per-cell effect was SR(som) - max(SR(best text), SR(vision)), a maximum over two noisy
     quantities. Taking a max biases the comparator upward and therefore biases the fusion
     advantage DOWNWARD, in the direction that favours our own claim.

This script replaces both. Effects are against A PRIORI FIXED comparators, never a max:

  som - vision   the two image-bearing modes, one arm each
  som - dom      fusion against the standard text agent, one arm each

Each gets a paired bootstrap CI over tasks (the two arms see the same task set, so the pairing
is real), then a fixed-effect inverse-variance pool across cells, which is the same machinery
the pre-registration specifies for its primary gate. This analysis is POST HOC and exploratory:
it is not the pre-registered H1 and is not gated.

The pooled estimate is then read against the measured rerun band rather than against zero,
because "better than nothing" is not the question a deployment asks about a premium.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import math
import random
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.lib.canonical_task_universe import expected_scored_ids  # noqa: E402

LOG = logging.getLogger("fusion_premium")
OUT_MD = REPO / "docs/analysis/cross_sites/fusion_premium.md"
OUT_JSON = REPO / "docs/analysis/cross_sites/fusion_premium.json"

FUSED = "som"
COMPARATORS = ["vision", "dom"]
MODES = [FUSED, *COMPARATORS, "ptext", "pprompt", "psom"]
N_BOOT = 10000
# Deterministic: the workflow forbids unseeded randomness and a reproducible artifact must not
# move between runs. A fixed LCG is enough for a paired bootstrap over <= 224 items.
SEED = 20260802

# Measured run-to-run band, mean-difference scale. READ from noise_floor_inventory.json rather
# than hardcoded: it was the literal tuple `(0.89, 2.23)` here for months, i.e. a conclusion
# frozen in a producer while its own source recomputed on every run (the §4d defect class).
# The product supplies two readings and both are used below:
#   observed_*      the two draws repetition actually delivered -- NOT a threshold
#   one_sided_95_*  the level an effect must reach before one rerun is unlikely to produce it
NOISE_FLOOR_JSON = REPO / "docs/analysis/cross_sites/noise_floor_inventory.json"


def load_floor_band() -> dict:
    """Fail loud: a hardcoded fallback is how the old constant went stale unnoticed."""
    if not NOISE_FLOOR_JSON.exists():
        raise MissingInput(
            f"{NOISE_FLOOR_JSON} absent -- run aggregate_noise_floor_inventory.py first. "
            "No fallback band is applied on purpose.")
    band = json.loads(NOISE_FLOOR_JSON.read_text()).get("floor_band")
    if not band or not band.get("rows"):
        raise MissingInput(
            f"{NOISE_FLOOR_JSON} carries no floor_band -- regenerate the inventory.")
    return band


class MissingInput(RuntimeError):
    """Fail loud rather than pool over a partial cell set."""


class _LCG:
    """RETIRED 2026-08-02 — kept so an older artifact can be regenerated for comparison.

    `below()` takes a modulo of the LOW-ORDER bits of a 31-bit LCG, which are its least random
    ones. Measured against the closed-form empirical-bootstrap variance it understated the SE by
    8.5-8.6% on the two largest cells, which then corrupts the inverse-variance weights.
    (codex Mode B, §H stress.) `random.Random` is deterministic under a fixed seed and has no
    such defect; it is what `paired_effect` uses now.
    """

    def __init__(self, seed: int) -> None:
        self.s = seed & 0xFFFFFFFF

    def below(self, n: int) -> int:
        self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
        return self.s % n


def load_vwa() -> dict[str, dict[int, dict[str, int]]]:
    rows = list(csv.DictReader((REPO / "results/phantom_paper/per_task_sr.csv").open()))
    cells: dict[str, dict[int, dict[str, int]]] = {}
    for r in rows:
        scored, _ = expected_scored_ids(r["site"])
        tid = int(r["task_id"])
        if tid not in scored:
            continue
        cells.setdefault(r["cell_id"], {})[tid] = {m: int(float(r[f"sr_{m}"]) > 0) for m in MODES}
    for cid, d in cells.items():
        site = "classifieds" if cid.startswith("cls") else "reddit"
        n_expected = len(expected_scored_ids(site)[0])
        if len(d) != n_expected:
            raise MissingInput(f"{cid}: {len(d)} scored tasks, canonical universe has {n_expected}")
    return cells


WA_MODE_STEM = {"dom": "dom", "som": "som", "vision": "vision",
                "ptext": "phantom_text", "pprompt": "phantom_prompt", "psom": "phantom_som"}


def load_wa(baseline: str = "B1") -> dict[int, dict[str, int]]:
    """<baseline> x WebArena-reddit. B0 landed 2026-08-03, so the run dirs are no longer
    a fixed set of timestamps — glob on the run-id suffix and reject ABORTED skeletons."""
    pats = {m: f"{baseline}_{stem}_wa_reddit_2026*_R*" for m, stem in WA_MODE_STEM.items()}
    per: dict[str, dict[int, int]] = {}
    for m, pat in pats.items():
        hits = [Path(p) for p in glob.glob(str(REPO / "results/webarena/phase1" / pat))
                if os.path.isdir(p) and "ABORTED" not in p]
        if len(hits) != 1:
            raise MissingInput(f"WA[{baseline}] {m}: expected 1 run dir for {pat!r}, got {len(hits)}")
        d = {}
        for f in (list(hits[0].glob("*/episodes/*summary*.json"))
                  or list(hits[0].glob("episodes/*summary*.json"))):
            s = json.loads(f.read_text())
            if not s.get("sr_excluded"):
                d[int(s["task_id"])] = 1 if s.get("success") else 0
        per[m] = d
    common = sorted(set.intersection(*(set(v) for v in per.values())))
    if not common:
        raise MissingInput("WA: empty task intersection across the six modes")
    return {t: {m: per[m][t] for m in MODES} for t in common}


def paired_effect(tasks: dict[int, dict[str, int]], a: str, b: str) -> dict:
    """Mean difference SR(a) - SR(b) in points, with a paired bootstrap CI over tasks."""
    ids = sorted(tasks)
    n = len(ids)
    diffs = [tasks[t][a] - tasks[t][b] for t in ids]
    est = 100 * sum(diffs) / n
    rng = random.Random(SEED)
    boots = []
    for _ in range(N_BOOT):
        s = 0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        boots.append(100 * s / n)
    boots.sort()
    lo, hi = boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT) - 1]
    # SE from the bootstrap distribution; used for the inverse-variance pool below.
    mu = sum(boots) / len(boots)
    se = math.sqrt(sum((x - mu) ** 2 for x in boots) / (len(boots) - 1))
    # The empirical-bootstrap SE of a mean has a closed form; report it so the resampling SE
    # can be checked rather than trusted.
    var = sum((x - sum(diffs) / n) ** 2 for x in diffs) / n
    return {"n": n, "est_pp": est, "ci": [lo, hi], "se": se,
            "se_exact_pp": 100 * math.sqrt(var / n)}


def fe_pool(effects: list[dict]) -> dict:
    """Fixed-effect inverse-variance pool, matching the pre-registration's primary machinery."""
    usable = [e for e in effects if e["se"] > 0]
    if not usable:
        raise MissingInput("no cell has a positive SE; cannot pool")
    w = [1 / e["se"] ** 2 for e in usable]
    theta = sum(wi * e["est_pp"] for wi, e in zip(w, usable)) / sum(w)
    se = math.sqrt(1 / sum(w))
    return {"k": len(usable), "theta_pp": theta, "se": se,
            "ci": [theta - 1.96 * se, theta + 1.96 * se]}


def clustered_pool(cells: dict[str, dict[int, dict[str, int]]], a: str, b: str,
                   weights: dict[str, float]) -> dict:
    """FE pool whose SE respects that the cells are not independent.

    Within a site the three backbones are evaluated on the SAME task universe, so effects on
    `cls_B0` and `cls_B1` share their sampling noise. `fe_pool()` combines them as
    `sqrt(1/sum(w))`, the independent-cells formula, and understates the pooled SE. Here the
    bootstrap resamples TASKS ONCE PER SITE and evaluates every backbone in that site on the
    same resampled set, carrying the cross-backbone correlation through. Weights are held at
    their point-estimate values — recomputing them inside each replicate would make this a
    different, self-selecting estimator. (§H stress P1-1, 2026-08-02.)
    """
    def site_of(cid: str) -> str:
        return "wa" if cid.startswith("wa") else cid.split("_")[0]

    by_site: dict[str, list[str]] = {}
    for cid in cells:
        by_site.setdefault(site_of(cid), []).append(cid)
    site_ids = {site: sorted(cells[cids[0]]) for site, cids in by_site.items()}
    for site, cids in by_site.items():
        for cid in cids:
            if sorted(cells[cid]) != site_ids[site]:
                raise MissingInput(f"{cid} does not share {site}'s task universe; the "
                                   "clustered resampling assumes it does")
    diffs = {cid: [cells[cid][t][a] - cells[cid][t][b] for t in site_ids[site_of(cid)]]
             for cid in cells}
    wsum = sum(weights[cid] for cid in cells)
    rng = random.Random(SEED)
    boots = []
    for _ in range(N_BOOT):
        draw = {site: [rng.randrange(len(ids)) for _ in ids] for site, ids in site_ids.items()}
        theta = 0.0
        for cid in cells:
            d, idx = diffs[cid], draw[site_of(cid)]
            theta += weights[cid] * (100 * sum(d[i] for i in idx) / len(idx))
        boots.append(theta / wsum)
    boots.sort()
    mu = sum(boots) / len(boots)
    se = math.sqrt(sum((x - mu) ** 2 for x in boots) / (len(boots) - 1))
    point = sum(weights[cid] * (100 * sum(diffs[cid]) / len(diffs[cid]))
                for cid in cells) / wsum
    return {"k": len(cells), "theta_pp": point, "se": se,
            "ci": [boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT) - 1]],
            "clusters": {st_: len(i) for st_, i in site_ids.items()}}


def cochran_q(effects: list[dict]) -> dict:
    """Heterogeneity across cells. A common-effect (FE) pool only means something if it is small."""
    usable = [e for e in effects if e["se"] > 0]
    w = [1 / e["se"] ** 2 for e in usable]
    theta = sum(wi * e["est_pp"] for wi, e in zip(w, usable)) / sum(w)
    q = sum(wi * (e["est_pp"] - theta) ** 2 for wi, e in zip(w, usable))
    df = len(usable) - 1

    def chi2_sf(x: float, k: int) -> float:
        if k <= 0:
            return 1.0
        if k % 2 == 0:
            term = tot = 1.0
            for i in range(1, k // 2):
                term *= x / (2 * i)
                tot += term
            return math.exp(-x / 2) * tot
        t = math.erfc(math.sqrt(x / 2))
        term, tot = math.sqrt(2 * x / math.pi) * math.exp(-x / 2), 0.0
        for i in range(1, (k - 1) // 2 + 1):
            tot += term
            term *= x / (2 * i + 1)
        return t + tot

    return {"Q": q, "df": df, "p": chi2_sf(q, df),
            "I2_pct": max(0.0, 100 * (q - df) / q) if q > 0 else 0.0}


def build() -> dict:
    cells = load_vwa()
    # Both WA backbones. They share the site cluster in the pooled resample because they
    # are scored on the same 104-task universe — not independent draws (same reason the
    # three VWA backbones share their site's cluster).
    cells["wa_red_B1"] = load_wa("B1")
    cells["wa_red_B0"] = load_wa("B0")
    band = load_floor_band()
    out = {"schema": "2026-08-03-fusion-premium-v2", "post_hoc_exploratory": True,
           "n_boot": N_BOOT, "seed": SEED,
           "floor_mean_pp": [band["observed_min_pp"], band["observed_max_pp"]],
           "floor_band": band,
           "fused": FUSED, "comparators": COMPARATORS, "cells": {}, "pooled": {}}
    for cid, tasks in cells.items():
        out["cells"][cid] = {c: paired_effect(tasks, FUSED, c) for c in COMPARATORS}
        n = len(tasks)
        out.setdefault("sr", {})[cid] = {
            m: 100 * sum(v[m] for v in tasks.values()) / n for m in (FUSED, *COMPARATORS)}
        LOG.info("%s: %s", cid, {c: round(out["cells"][cid][c]["est_pp"], 2) for c in COMPARATORS})
    for c in COMPARATORS:
        eff = [out["cells"][cid][c] for cid in cells]
        out["pooled"][c] = fe_pool(eff)
        out["pooled"][c]["heterogeneity"] = cochran_q(eff)
        w = {cid: 1 / out["cells"][cid][c]["se"] ** 2 for cid in cells
             if out["cells"][cid][c]["se"] > 0}
        if len(w) == len(cells):
            out["pooled"][c]["clustered"] = clustered_pool(cells, FUSED, c, w)
    return out


def render(d: dict) -> str:
    lo, hi = d["floor_mean_pp"]
    band = d["floor_band"]
    null_lo, null_hi = band["one_sided_95_min_pp"], band["one_sided_95_max_pp"]
    L = ["---", "type: analysis", "status: complete", "created: 2026-08-02",
         "purpose: turn §4.2's fusion-premium claim from a count over cells into a paired test "
         "against a priori comparators",
         "post_hoc_exploratory: true",
         "scope_warning: not the pre-registered H1; not gated. The rerun band it is read against "
         f"({lo}-{hi}pp, mean-difference scale) is measured on two conditions and extrapolated to "
         "the rest.",
         "producer: scripts/analysis/aggregate_fusion_premium.py", "---", "",
         "# Does the fused mode earn its premium?", "",
         "Regenerate: `.venv/bin/python3 scripts/analysis/aggregate_fusion_premium.py`", "",
         "Effects are `SR(SoM) - SR(comparator)` in points, against **a priori fixed** "
         "comparators. An earlier version compared against the per-cell maximum of two "
         "alternatives; a maximum over noisy quantities biases the comparator up and so biased "
         "the fusion advantage down, in the direction that favoured the claim being made. "
         f"Intervals are paired bootstrap over tasks, {d['n_boot']:,} resamples, seed "
         f"{d['seed']}.", "",
         "## 1. Per cell", "",
         "| cell | n | SoM − Vision | 95% CI | SoM − DOM | 95% CI |", "|---|---|---|---|---|---|"]
    for cid, c in d["cells"].items():
        v, m = c["vision"], c["dom"]
        L.append(f"| `{cid}` | {v['n']} | {v['est_pp']:+.2f}pp | "
                 f"[{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}] | {m['est_pp']:+.2f}pp | "
                 f"[{m['ci'][0]:+.2f}, {m['ci'][1]:+.2f}] |")
    L += ["", "## 2. Fixed-effect pool", "",
          "| comparator | k | pooled θ | 95% CI | clears 0? | clears the observed band? "
          "| clears the rerun **null**? |",
          "|---|---|---|---|---|---|---|"]
    for c in d["comparators"]:
        p = d["pooled"][c]
        cl = p.get("clustered") or p
        z = "**yes**" if cl["ci"][0] > 0 else "no"
        f = "**yes**" if cl["ci"][0] > hi else "no"
        nn = "**yes**" if cl["ci"][0] > null_hi else "no"
        L.append(f"| SoM − {c} | {p['k']} | **{cl['theta_pp']:+.2f}pp** | "
                 f"[{cl['ci'][0]:+.2f}, {cl['ci'][1]:+.2f}] | {z} | {f} | {nn} |")
    L += ["", f"The band is the measured run-to-run mean-difference floor, {lo} to {hi}pp. "
          "Reading the pooled estimate against it rather than against zero is the point: a "
          "premium has to beat what repetition delivers for the same money, not merely beat "
          "nothing.", "",
          f"⚠️ **Two bands, and the last column is the one that answers the question.** "
          f"`{lo}–{hi}pp` is what {band['n_draws']} reruns *happened to* deliver — two draws "
          f"from a random quantity, not a bound on it. That quantity's own spread is "
          f"computable from the same pairs' discordant counts: `SD(ΔSR) = √d/n` gives "
          f"**{band['null_sd_min_pp']}–{band['null_sd_max_pp']}pp**, i.e. the band's upper "
          f"edge is about one standard deviation. An effect only becomes unlikely for a "
          f"single rerun to manufacture at roughly **{null_lo}–{null_hi}pp** (one-sided 95%). "
          "Both are reported; nothing here should be read as clearing noise on the strength "
          "of the observed band alone. → `noise_floor_inventory` §1b.", "",
          "**The interval above is the task-clustered one, and that choice changes an answer.** "
          "Within a site the three backbones are scored on the same task universe, so their "
          "effects share sampling noise; the textbook `sqrt(1/Σw)` treats them as independent "
          "and understates the pooled SE. Resampling tasks once per site and evaluating every "
          "backbone in that site on the same draw gives:", "",
          "| comparator | independent-cells CI | task-clustered CI | SE |", "|---|---|---|---|"]
    for c in d["comparators"]:
        p = d["pooled"][c]
        cl = p.get("clustered")
        if not cl:
            continue
        L.append(f"| SoM − {c} | [{p['ci'][0]:+.2f}, {p['ci'][1]:+.2f}] | "
                 f"**[{cl['ci'][0]:+.2f}, {cl['ci'][1]:+.2f}]** | "
                 f"{p['se']:.3f} → {cl['se']:.3f} |")
    # This verdict was hardcoded prose until 2026-08-03 ("the one interval that
    # excluded zero no longer does"). It was true of the six-cell pool, and it kept
    # printing itself unchanged after the seventh and eighth cells reversed it —
    # while the data-driven table three lines up said the opposite. Derive it.
    flipped, kept = [], []
    for c in d["comparators"]:
        p = d["pooled"][c]
        cl = p.get("clustered")
        if not cl:
            continue
        if p["ci"][0] > 0 and cl["ci"][0] <= 0:
            flipped.append(c)
        elif cl["ci"][0] > 0:
            kept.append((c, cl["ci"][0]))
    if flipped:
        L += ["", "Clustering removes the zero-exclusion for "
              + ", ".join(f"`SoM − {c}`" for c in flipped) + "."]
    elif kept:
        L += ["", "Clustering widens every interval without changing a verdict: "
              + ", ".join(f"`SoM − {c}` still excludes zero (lower bound {b:+.2f}pp)"
                          for c, b in kept)
              + f". ⚠️ Read that against the band, not against zero: a lower bound of "
                f"{min(b for _, b in kept):+.2f}pp sits below the rerun band's floor of "
                f"{lo}pp, so excluding zero here is **not** a premium claim — the "
                "`clears the rerun band?` column above is the one that answers the question."]
    else:
        L += ["", "No comparator's clustered interval excludes zero."]
    L += ["(codex Mode B, §H stress 2026-08-02; its predicted clustered SE of 0.741 matched.)",
          "",
          "⚠️ **And a fixed-effect pool is the wrong estimand here regardless.** Cochran's Q "
          "rejects a common effect for both comparators:", "",
          "| comparator | Q | df | p | I² |", "|---|---|---|---|---|"]
    for c in d["comparators"]:
        h = d["pooled"][c]["heterogeneity"]
        L.append(f"| SoM − {c} | {h['Q']:.2f} | {h['df']} | {h['p']:.2e} | {h['I2_pct']:.0f}% |")
    L += ["", "With I² at this level the pooled number describes no cell in particular. It is "
          "kept because the pre-registration names FE as the primary machinery, but the per-cell "
          "table in §1 and the workload split in §3 are what carry the finding — and the sign "
          "change across workloads in §3 is itself why a common-effect model cannot hold.", "",
          "## 3. Fusion against the channel that suits the workload", "",
          "The two columns read together show something neither shows alone. In every cell one of "
          "the two single channels is the stronger, and it is the visual one on all three "
          "classifieds cells and the text one on all four reddit splits. Against **that** channel, "
          "fusion's interval includes zero everywhere but one, where it is significantly "
          "negative.", "",
          "| cell | stronger single channel | SoM - that channel | 95% CI | excludes 0? "
          "| \\|effect\\| > rerun null? |",
          "|---|---|---|---|---|---|"]
    n_incl = 0
    weak_gains: dict[str, list[tuple[str, float]]] = {"dom": [], "vision": []}
    for cid, c in d["cells"].items():
        strong = "vision" if d["sr"][cid]["vision"] >= d["sr"][cid]["dom"] else "dom"
        weak = "dom" if strong == "vision" else "vision"
        e = c[strong]
        excl = e["ci"][0] > 0 or e["ci"][1] < 0
        n_incl += not excl
        weak_gains[weak].append((cid, c[weak]["est_pp"]))
        L.append(f"| `{cid}` | {strong} | {e['est_pp']:+.2f}pp | "
                 f"[{e['ci'][0]:+.2f}, {e['ci'][1]:+.2f}] | "
                 f"{'**yes, negative**' if excl else 'no'} | "
                 f"{'yes' if abs(e['est_pp']) > null_hi else 'no'} |")
    L += ["", f"**{n_incl} of {len(d['cells'])}** intervals include zero and the remaining one is "
          "negative, so in no cell does fusion beat the channel that suits the workload. "
          "⚠️ The word *significantly* does not belong on that sentence and has been removed: "
          "the comparator is chosen per cell using the same observed success rates the interval "
          "is computed from, so these CIs do not retain nominal coverage. Restoring coverage "
          "needs either a site→channel mapping fixed in advance, or a bootstrap that re-selects "
          "the comparator inside every resample. Until then this row is descriptive. "
          "(§H stress P1-2.) Against the channel that does *not* suit the workload it does "
          "better — but the full set must be quoted, not its top end:", ""]
    for weak, label in (("dom", "over DOM, where the visual channel is stronger"),
                        ("vision", "over Vision, where the text channel is stronger")):
        rows = sorted(weak_gains[weak], key=lambda kv: kv[1])
        L.append(f"- `SoM − {weak}` {label}: "
                 + ", ".join(f"{cid} {v:+.2f}pp" for cid, v in rows)
                 + f"  → range **{rows[0][1]:+.2f} to {rows[-1][1]:+.2f}pp**, "
                 + f"{sum(1 for _, v in rows if abs(v) > null_hi)} of {len(rows)} above the "
                   "rerun null.")
    L += ["",
          "⚠️ **Derived from the table above, not typed.** Until 2026-08-03 this paragraph "
          "hardcoded four of these numbers and named only the two largest on each side — the "
          "source of the `4.93-7.39pp` string quoted downstream. The dropped cells are not "
          "cosmetic: on the text-stronger side the smallest is **negative**, so a range "
          "starting at +4.93 is a subrange with the sign change removed. Which channel is "
          "stronger is "
          "read off each cell and is therefore post hoc, which is why both full columns appear in "
          "§1 and the pooled tests in §2 use comparators fixed in advance.", "",
          "## 4. What this does and does not settle", "",
          "It replaces a count over cells with an interval, and it removes the selection bias "
          "in the comparator. It does **not** supply a rerun floor for the fused mode itself, "
          "which is measured on DOM and Vision only; a same-condition SoM replicate is the "
          "experiment that would close that, and it is queued rather than done."]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    d = build()
    OUT_JSON.write_text(json.dumps(d, indent=2))
    OUT_MD.write_text(render(d))
    print(f"✓ {OUT_MD.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
