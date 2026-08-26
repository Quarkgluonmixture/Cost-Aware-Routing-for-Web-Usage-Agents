#!/usr/bin/env python3
"""Is the run-to-run reproducibility floor a property of the MODEL or of how it is SERVED?

This is the question the reframe chain was built to answer (`pre_run/
reframe_chain_launch_intent_20260819.md`, claim B). Until 2026-08-21 the project
held exactly one API-served backbone, so "B0 has a ~12% floor" and "API serving
has a ~12% floor" were the same sentence and neither could be told from the
other. B0 is MoE *and* proxy-served; B1 is dense *and* local. Perfectly confounded.

A second API model breaks it. This module does nothing but group the measured
floors by how the backbone was served and report whether the two groups separate.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
No mechanism. 实验笔记 §302.5 adjudicated that a paper-grade claim here stops at
an *observable provider-dependent noise floor*, and named three escape hatches as
unusable: "switch provider and it goes away", "MoE is the cause", "it is a
provider bug". The repo holds no expert-route log, no serving batch id, no
instance id and no model SHA, so none of those can be told apart from the client
side. This module reports WHERE the floor is, never WHY.

THE CONFOUND THAT SURVIVES, STATED UP FRONT
-------------------------------------------
We do not hold the same weights served both ways. The API group is
{Qwen3-VL-235B-A22B, GPT-5.6-terra} and the local group is {Qwen3-VL-4B}: serving
mode covaries with scale and with family. What the second API model removes is
the *family* explanation (two unrelated families, same floor); what it cannot
remove is scale. The honest form of the claim is therefore about the observed
grouping, not about a mechanism, and the one experiment that would settle it —
the same checkpoint served locally and through the API — is named in the output.

Regenerate: `.venv/bin/python3 scripts/analysis/serving_mode_floor.py`
"""
from __future__ import annotations

import json
import logging
import math
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
INV = REPO / "docs/analysis/cross_sites/noise_floor_inventory.json"
OUT_JSON = REPO / "docs/analysis/cross_sites/serving_mode_floor.json"
OUT_MD = REPO / "docs/analysis/cross_sites/serving_mode_floor.md"

LOG = logging.getLogger("serving-mode-floor")

# Baseline -> how it was served, and what it is. `serving` is the grouping variable;
# everything else is carried so the covariates are visible in the table rather than
# buried in this dict.
BACKBONE = {
    "B0": dict(serving="API", provider="AWS proxy", family="Qwen", arch="MoE 235B-A22B",
               name="Qwen3-VL-235B-A22B", weights="open"),
    "B1": dict(serving="local", provider="self-hosted bf16", family="Qwen", arch="dense 4B",
               name="Qwen3-VL-4B", weights="open"),
    "B2": dict(serving="local", provider="self-hosted bf16", family="Gemma", arch="dense 4B",
               name="google/gemma-3-4b-it", weights="open"),
    "B5": dict(serving="API", provider="AWS proxy", family="OpenAI", arch="undisclosed",
               name="GPT-5.6-terra", weights="closed"),
}
SITE_NAME = {"cls": "VWA-classifieds", "red": "VWA-reddit", "shop": "VWA-shopping"}

# d ~ n x SR x 0.59 (§468 / B-1972). Below d=10 a pair is inventory, not an interval.
D_BAR = 10.0
D_COEF = 0.59


class MissingInput(RuntimeError):
    """Fail loud rather than group over a partial inventory."""


def _load() -> list[dict]:
    if not INV.exists():
        raise MissingInput(f"{INV} absent — run aggregate_noise_floor_inventory.py first")
    pairs = json.loads(INV.read_text())["clean_pairs"]
    if not pairs:
        raise MissingInput(f"{INV} carries no clean_pairs")
    rows = []
    for p in pairs:
        base, site_key, arm = p["label"].split(".")
        if base not in BACKBONE:
            raise MissingInput(f"{p['label']}: unknown backbone {base!r}; extend BACKBONE")
        # ⚠️ UNIT TRAP, caught 2026-08-26 on first run: inside one and the same
        # clean_pairs record, `sr_a`/`sr_b` are FRACTIONS (0.25) while
        # `discordance_pct` is a PERCENTAGE (14.29). Dividing sr by 100 as if it
        # matched its neighbour shrank every d by 100x and reported all thirteen
        # arms as underpowered — a table that would have said the project has no
        # interval-carrying floor at all. Assert the unit rather than trust it.
        sr_frac = (p["sr_a"] + p["sr_b"]) / 2.0
        if not 0.0 <= sr_frac <= 1.0:
            raise MissingInput(
                f"{p['label']}: sr_a/sr_b look like percentages ({p['sr_a']}, {p['sr_b']}), "
                f"not fractions — the d formula would be off by 100x")
        sr = sr_frac * 100.0
        d = p["n"] * sr_frac * D_COEF
        rows.append(dict(
            label=p["label"], base=base, site=site_key, arm=arm,
            site_name=SITE_NAME.get(site_key, site_key),
            floor_pct=p["discordance_pct"], n=p["n"],
            sr_mean_pct=sr, sr_a=p["sr_a"], sr_b=p["sr_b"],
            d=d, powered=bool(d >= D_BAR),
            self_drop_lo=min(p["self_drop_a_to_b_pp"], p["self_drop_b_to_a_pp"]),
            self_drop_hi=max(p["self_drop_a_to_b_pp"], p["self_drop_b_to_a_pp"]),
            **{k: v for k, v in BACKBONE[base].items()},
        ))
    return rows


def _exact_separation_p(api: list[float], loc: list[float]) -> dict:
    """One-sided exact rank test: how surprising is a perfect split under exchangeability?

    Under H0 (both groups drawn from one distribution) every assignment of which
    len(loc) of the pooled floors are the local ones is equally likely. The p-value
    for "every local floor below every API floor" is therefore the count of
    assignments at least that extreme over C(N, k) — no distributional assumption,
    which matters at these group sizes.

    ⚠️ The observations are NOT independent: arms within a cell share a site, a
    backbone and a task universe. This is a descriptive separation statistic, not a
    hypothesis test the paper may gate on. It is reported to say how far from
    coincidence a perfect split is, given how few draws there are.
    """
    pooled = sorted(api + loc)
    k = len(loc)
    n_total = len(pooled)
    obs_max_loc = max(loc)
    at_least_as_extreme = 0
    total = 0
    for combo in combinations(range(n_total), k):
        total += 1
        if max(pooled[i] for i in combo) <= obs_max_loc:
            at_least_as_extreme += 1
    return {
        "p_one_sided_exact": at_least_as_extreme / total,
        "n_assignments": total,
        "n_at_least_as_extreme": at_least_as_extreme,
        "separated": max(loc) < min(api),
        "gap_pp": min(api) - max(loc),
        "caveat": ("arms within a cell are not independent (shared site, backbone, task "
                   "universe); descriptive separation statistic, NOT a gateable test"),
    }


def build() -> dict:
    rows = _load()
    api = [r for r in rows if r["serving"] == "API"]
    loc = [r for r in rows if r["serving"] == "local"]
    if not api or not loc:
        raise MissingInput(
            f"need both serving modes; have API={len(api)} local={len(loc)}")

    def _grp(rs: list[dict]) -> dict:
        f = [r["floor_pct"] for r in rs]
        pw = [r for r in rs if r["powered"]]
        return dict(
            n_arms=len(rs), floor_min_pct=min(f), floor_max_pct=max(f),
            floor_mean_pct=sum(f) / len(f),
            n_arms_powered=len(pw),
            floor_min_powered_pct=min((r["floor_pct"] for r in pw), default=None),
            floor_max_powered_pct=max((r["floor_pct"] for r in pw), default=None),
            families=sorted({r["family"] for r in rs}),
            sites=sorted({r["site_name"] for r in rs}),
            cells=sorted({f"{r['base']}.{r['site']}" for r in rs}),
        )

    g_api, g_loc = _grp(api), _grp(loc)
    sep = _exact_separation_p([r["floor_pct"] for r in api],
                              [r["floor_pct"] for r in loc])
    # The same test restricted to arms that carry an interval — the version a
    # reviewer will ask for, because an underpowered zero is not evidence of zero.
    api_pw = [r["floor_pct"] for r in api if r["powered"]]
    loc_pw = [r["floor_pct"] for r in loc if r["powered"]]
    sep_pw = (_exact_separation_p(api_pw, loc_pw)
              if api_pw and loc_pw else
              {"unavailable": "one group has no arm above the d>=10 bar"})

    return {
        "schema": "2026-08-26-serving-mode-floor-v1",
        "post_hoc_exploratory": False,
        "declared_in": "docs/checkpoints/pre_run/reframe_chain_launch_intent_20260819.md (claim B)",
        "functional": ("per-task discordance between two runs of an IDENTICAL condition, "
                       "grouped by how the backbone was served"),
        "d_bar": D_BAR, "d_formula": f"d ~ n x SR x {D_COEF}",
        "groups": {"API": g_api, "local": g_loc},
        "separation": sep,
        "separation_powered_only": sep_pw,
        "arms": sorted(rows, key=lambda r: (r["serving"], -r["floor_pct"])),
        "confound_not_removed": (
            "serving mode covaries with scale (235B/undisclosed vs 4B). A second API "
            "family removes the FAMILY explanation, not the SCALE one."),
        "settling_experiment": (
            "the same checkpoint served both ways — e.g. Qwen3-VL-4B through an API "
            "endpoint, or a 235B-class model self-hosted. Neither is in this project's "
            "compute envelope; naming it is the honest substitute."),
        "mechanism_policy": (
            "none offered. 实验笔记 §302.5: the claim stops at an observable "
            "provider-dependent floor; 'MoE is the cause', 'switch provider', and "
            "'provider bug' are all named unusable without a server-side audit artifact."),
        "independent_corroboration_local": (
            "实验笔记 §298.2: a controlled step-level probe on B1 (dense, local, temp=0) "
            "returned determinism 133/133 OK. The local group's near-zero floor is "
            "therefore not only a replicate-pair inference."),
        "coverage_gaps": [
            "B2 (local, Gemma) carries no replicate: at its SR (0.45-2.23%) d~1.8, far "
            "below the bar — the local group cannot be given a second family by measuring "
            "B2, which is a power limit, not a scheduling one",
            "B1 has no reddit replicate: the local group is one site",
            "B5 has no reddit replicate yet (_b5_reddit_chain.sh is armed for it)",
        ],
    }


def render(d: dict) -> str:
    ga, gl = d["groups"]["API"], d["groups"]["local"]
    sep = d["separation"]
    L = ["---", "type: analysis", "status: complete", "created: 2026-08-26",
         "purpose: test whether the reproducibility floor groups by how the backbone is "
         "served rather than by which backbone it is",
         "producer: scripts/analysis/serving_mode_floor.py", "---", "",
         "# Is the reproducibility floor a property of the model, or of the serving path?", "",
         f"Regenerate: `.venv/bin/python3 scripts/analysis/serving_mode_floor.py`", "",
         "Until a second API-served backbone landed (2026-08-21) this question could not be "
         "asked: the project held one API model and one local one, so *model* and *serving "
         "path* were the same variable. Every floor below is the same functional — per-task "
         "discordance between two runs of an identical condition.", "",
         "## 1. The two groups", "",
         "| serving | arms | families | sites | floor range | powered arms (d≥10) |",
         "|---|---|---|---|---|---|"]
    for name, g in (("**API**", ga), ("**local**", gl)):
        pw = (f"{g['floor_min_powered_pct']:.2f}–{g['floor_max_powered_pct']:.2f}%"
              if g["floor_min_powered_pct"] is not None else "—")
        L.append(f"| {name} | {g['n_arms']} | {', '.join(g['families'])} | "
                 f"{len(g['sites'])} | **{g['floor_min_pct']:.2f}–{g['floor_max_pct']:.2f}%** | "
                 f"{g['n_arms_powered']} ({pw}) |")
    L += ["",
          (f"The groups **{'do not overlap' if sep['separated'] else 'overlap'}**"
           + (f": the lowest API floor ({min(r['floor_pct'] for r in d['arms'] if r['serving']=='API'):.2f}%) "
              f"is {sep['gap_pp']:.2f}pp above the highest local one "
              f"({max(r['floor_pct'] for r in d['arms'] if r['serving']=='local'):.2f}%)."
              if sep["separated"] else ".")), "",
          f"Exact one-sided rank test on a perfect split: **p = {sep['p_one_sided_exact']:.4f}** "
          f"({sep['n_at_least_as_extreme']}/{sep['n_assignments']} assignments at least this "
          f"extreme). ⚠️ {sep['caveat']}.", ""]
    spw = d["separation_powered_only"]
    if "unavailable" in spw:
        L += [f"Restricted to powered arms: **{spw['unavailable']}**.", ""]
    else:
        L += [f"Restricted to arms carrying an interval (d≥10): separated="
              f"**{spw['separated']}**, p = {spw['p_one_sided_exact']:.4f}, "
              f"gap {spw['gap_pp']:.2f}pp.", ""]

    L += ["## 2. Every arm, with its power", "",
          "| serving | backbone | arch | site | arm | n | SR | floor | d | interval? |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in d["arms"]:
        L.append(f"| {r['serving']} | `{r['name']}` | {r['arch']} | {r['site_name']} | "
                 f"`{r['arm']}` | {r['n']} | {r['sr_mean_pct']:.2f}% | "
                 f"**{r['floor_pct']:.2f}%** | {r['d']:.1f} | "
                 f"{'yes' if r['powered'] else '**no — inventory only**'} |")

    L += ["", "## 3. What this does and does not license", "",
          "**Licensed.** The floor groups by serving path across two unrelated model "
          f"families ({', '.join(ga['families'])}) and {len(ga['sites'])} site(s) on the API "
          "side. Before this, one API model meant *model* and *serving* were the same "
          "variable; the second family removes the reading that the floor is a quirk of one "
          "architecture.", "",
          f"**Not removed: scale.** {d['confound_not_removed']} The experiment that would "
          f"settle it is {d['settling_experiment']}", "",
          f"**No mechanism.** {d['mechanism_policy']}", "",
          f"**Local side, independent corroboration.** {d['independent_corroboration_local']}",
          "", "**Coverage gaps.**"]
    for g in d["coverage_gaps"]:
        L.append(f"- {g}")
    L += ["", "## 4. Why it matters beyond this project", "",
          "Web-agent benchmarks report success rates as point estimates. If a condition "
          "rerun through an API disagrees with itself on a tenth of its tasks, then any "
          "reported difference smaller than that is not distinguishable from repetition — "
          "and the overwhelming majority of agent evaluations are run through exactly such "
          "an API, once. The local column is what makes this a statement about the serving "
          "path rather than about benchmarks being noisy in general.", ""]
    return "\n".join(L)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    d = build()
    OUT_JSON.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
    OUT_MD.write_text(render(d) + "\n")
    ga, gl = d["groups"]["API"], d["groups"]["local"]
    LOG.info("API   %d arms, %s families, floor %.2f-%.2f%%",
             ga["n_arms"], len(ga["families"]), ga["floor_min_pct"], ga["floor_max_pct"])
    LOG.info("local %d arms, %s families, floor %.2f-%.2f%%",
             gl["n_arms"], len(gl["families"]), gl["floor_min_pct"], gl["floor_max_pct"])
    LOG.info("separated=%s gap=%.2fpp p=%.4f",
             d["separation"]["separated"], d["separation"]["gap_pp"],
             d["separation"]["p_one_sided_exact"])
    LOG.info("wrote %s + %s", OUT_MD, OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
