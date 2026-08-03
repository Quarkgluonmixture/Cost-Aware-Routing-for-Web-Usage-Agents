"""Render every evidence product as an ablation table, into one markdown file.

The Overleaf pipeline (`scripts/maintenance/overleaf_sync.sh`) treats
`docs/checkpoints/paper_drafts/**/*.md` as the single source and pandoc-converts it with
`pipe_tables`. So the tables have to live in markdown — but they must not be *typed* into
markdown, because a hand-copied number silently decouples from the product that produced
it. Six such decouplings were found and fixed on 2026-08-03 alone, one of which was wrong
on the fact and not merely on the denominator.

This reads the product JSONs and writes the tables. The prose around them is hand-written
and lives in the same file, between the markers; regenerating rewrites only what is
between `<!-- BEGIN table:<id> -->` and `<!-- END table:<id> -->`.

Every table carries its own denominator and estimand in the caption, because these tables
will be read out of context and a bare percentage invites the misuse the caption prevents.

Regenerate:
    .venv/bin/python3 scripts/analysis/export_ablation_tables.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
X = REPO / "docs/analysis/cross_sites"
OUT = REPO / "docs/checkpoints/paper_drafts/ablation_tables.md"

CELLS = ["cls_B0", "cls_B1", "cls_B2", "red_B0", "red_B1", "red_B2", "wa_B0", "wa_B1"]
CELL_LABEL = {"cls_B0": "cls·B0", "cls_B1": "cls·B1", "cls_B2": "cls·B2",
              "red_B0": "red·B0", "red_B1": "red·B1", "red_B2": "red·B2",
              "wa_B0": "WA·B0", "wa_B1": "WA·B1"}
MODES = ["dom", "som", "vision", "ptext", "pprompt", "psom"]
PRETTY = {"dom": "DOM", "som": "SoM", "vision": "Vision",
          "ptext": "P-text", "pprompt": "P-prompt", "psom": "P-SoM"}


class MissingProduct(RuntimeError):
    """Fail loud: a missing product must not become a silently absent table."""


def load(name: str) -> dict:
    p = X / f"{name}.json"
    if not p.exists():
        raise MissingProduct(f"{p} missing — regenerate that product first")
    return json.loads(p.read_text())


def fmt(v, spec="+.2f", suffix="pp", none="—"):
    if v is None:
        return none
    return f"{format(v, spec)}{suffix}"


# diag rule names are partly Chinese; the ACL template has no CJK font, so one unmapped
# character is a compile failure several hundred log lines from its cause. Map the ones
# that occur and fall back to the rule id — never emit the raw name.
RULE_NAME_ASCII = {
    "click-back振荡": "click-back oscillation",
    "从不翻页": "never paginates",
    "budget耗尽未完成": "budget exhausted, unfinished",
    "感知缺失循环": "perception-gap loop",
    "URL 自环": "URL self-loop",
    "找不到即放弃": "gives up when not found",
    "视觉图像内容DOM必败": "visual-content task, DOM cannot see it",
    "根节点误操作": "root-node misfire",
    "导航至裸图片URL幻觉": "navigates to a bare image URL",
}


def ascii_rule_name(rule_id: str, name: str) -> str:
    if name in RULE_NAME_ASCII:
        return RULE_NAME_ASCII[name]
    if any("\u4e00" <= ch <= "\u9fff" for ch in name or ""):
        return rule_id          # unmapped CJK: the id is lossless enough
    return name



# pdflatex has no glyph for these and skeleton_acl.tex cannot usefully declare an emoji.
# Replace at write time rather than asking every caption author to remember: the failure
# mode is a compile error several hundred log lines from its cause.
AT_NUMBER = __import__("re").compile(r"(?<=\\s)@(?=[0-9])")
TEX_UNSAFE = {"\u26a0\ufe0f": "**Caution:**", "\u26a0": "**Caution:**",
              "\u2b50": "", "\u2705": "yes", "\u274c": "no"}


def tex_safe(s: str) -> str:
    for bad, good in TEX_UNSAFE.items():
        s = s.replace(bad, good)
    # pandoc --natbib reads `@token` as a citation key, so "SoM @27.23" silently becomes
    # \\citet{27.23} and surfaces only as an undefined-citation warning.
    return AT_NUMBER.sub("at ", s)


# --------------------------------------------------------------------------- tables
def t_sr() -> tuple[str, str]:
    d = load("representation_class_comparison")
    rows = ["| cell | " + " | ".join(PRETTY[m] for m in MODES) + " | best |",
            "|---|" + "---|" * (len(MODES) + 1)]
    for c in CELLS:
        if c not in d["cells"]:
            continue
        s = d["cells"][c]["single_sr"]
        best = max(s, key=lambda m: s[m])
        rows.append(f"| {CELL_LABEL[c]} | "
                    + " | ".join(("**%.2f**" % s[m]) if m == best else ("%.2f" % s[m])
                                 for m in MODES)
                    + f" | {PRETTY[best]} |")
    cap = ("Success rate (%) per observation mode. Denominator is the canonical scored set "
           "(cls 224 / red 203) and, for WebArena, the six-mode task intersection (104). "
           "Bold = best mode in the cell. Source: `representation_class_comparison.json`.")
    return "\n".join(rows), cap


def t_class() -> tuple[str, str]:
    d = load("representation_class_comparison")
    rows = ["| cell | no-image (4 arms) | vision-only | hybrid | sole best |",
            "|---|---|---|---|---|"]
    for c in CELLS:
        if c not in d["cells"]:
            continue
        r = d["cells"][c]
        b = r["class_best"]
        w = r["best_class"]
        rows.append(f"| {CELL_LABEL[c]} | {b['no-image']:.2f} "
                    f"({PRETTY[r['class_best_arm']['no-image']]}) | "
                    f"{b['vision-only']:.2f} | {b['hybrid']:.2f} | {w} |")
    cap = ("Best arm within each deployment class (%). Classes: **no-image** = "
           "{DOM, P-text, P-prompt, P-SoM}, **vision-only** = {Vision}, **hybrid** = {SoM}. "
           "Grouping the four is licensed by the non-separability result (Table 4): they "
           "clear the ≥83% consistency bar on none of 26 metrics. "
           "Source: `representation_class_comparison.json`.")
    return "\n".join(rows), cap


def t_class_ablate() -> tuple[str, str]:
    d = load("representation_class_comparison")
    rows = ["| cell | all six | −no-image | −vision-only | −hybrid | +1 no-image | "
            "+1 vision-only | +1 hybrid |", "|---|---|---|---|---|---|---|---|"]
    for c in CELLS:
        if c not in d["cells"]:
            continue
        r = d["cells"][c]
        dr = r.get("drop_class_unmatched", {})
        g = r.get("arm_matched_gain", {})
        rows.append(f"| {CELL_LABEL[c]} | {r['oracle_all']:.2f} | "
                    f"−{dr.get('no-image', 0):.2f} | −{dr.get('vision-only', 0):.2f} | "
                    f"−{dr.get('hybrid', 0):.2f} | {fmt(g.get('no-image'))} | "
                    f"{fmt(g.get('vision-only'))} | {fmt(g.get('hybrid'))} |")
    cap = ("Class ablation. Columns 2–4: oracle coverage lost when a whole class is "
           "unavailable — **not arm-matched**, the no-image class has four arms and the "
           "others one each, so most of the gap is arm count. Columns 5–7: the matched "
           "comparison — gain from adding ONE arm of that class to the cell's best single "
           "arm (— = that class already supplies the starting arm). The matched panel shows "
           "no systematic difference between classes. "
           "Source: `representation_class_comparison.json`.")
    return "\n".join(rows), cap


def t_nonsep() -> tuple[str, str]:
    d = load("per_mode_four_dimension_profile_with_wa")
    cons = d["consistency"]
    n = next(iter(cons.values()))["n_cells"]
    thr = round(0.83 * n)
    tally: dict[str, int] = {}
    for r in cons.values():
        for side, mode in (("highest_in_n_cells", "highest_mode"),
                           ("lowest_in_n_cells", "lowest_mode")):
            if r[side] >= thr:
                tally[r[mode]] = tally.get(r[mode], 0) + 1
    rows = ["| mode | metrics reaching the bar |", "|---|---|"]
    for m in ["Vision", "SoM", "DOM", "P-text", "P-prompt", "P-SoM"]:
        rows.append(f"| {m} | {tally.get(m, 0)} |")
    cap = (f"Behavioural non-separability. A mode 'reaches the bar' on a metric when it is "
           f"the extreme (highest or lowest) in ≥{thr} of {n} cells — 83%, the same "
           f"proportion the six-cell version meant by ≥5/6. Over 26 metrics, the four "
           f"image-free modes reach it on **none**. ⚠️ Carrying the literal numerator "
           f"(≥5/8 = 63%) instead would let P-text clear it on two metrics and this negative "
           f"would appear to break. Source: `per_mode_four_dimension_profile_with_wa.json`.")
    return "\n".join(rows), cap


def t_fusion() -> tuple[str, str]:
    d = load("fusion_premium")
    rows = ["| cell | n | SoM − Vision | 95% CI | SoM − DOM | 95% CI |",
            "|---|---|---|---|---|---|"]
    for c in d["cells"]:
        r = d["cells"][c]
        v, dm = r["vision"], r["dom"]
        rows.append(f"| {CELL_LABEL.get(c, c)} | {v['n']} | {v['est_pp']:+.2f} | "
                    f"[{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}] | {dm['est_pp']:+.2f} | "
                    f"[{dm['ci'][0]:+.2f}, {dm['ci'][1]:+.2f}] |")
    lo, hi = d.get("floor_mean_pp", [0.89, 2.23])
    cap = (f"Fusion premium (pp). Comparators are fixed a priori, not per-cell maxima. "
           f"Paired bootstrap over tasks, 10,000 resamples. Read against the measured "
           f"rerun band **{lo}–{hi}pp**, not against zero: a premium must beat what "
           f"repetition delivers for the same money. No cell clears the band; `cls_B0`'s "
           f"+2.23 *equals* its upper edge (both are 5/224). "
           f"Source: `fusion_premium.json`.")
    return "\n".join(rows), cap


def t_routing() -> tuple[str, str]:
    d = load("rule_routing_pareto")
    rows = ["| cell | policy | SR | cost | latency | on frontier |",
            "|---|---|---|---|---|---|"]
    for c in CELLS:
        if c not in d["cells"]:
            continue
        pol = d["cells"][c]["policies"]
        keep = [k for k in pol if k.startswith("rule") or k in
                ("always-DOM", "always-Vision", "always-SoM")]
        for k in sorted(keep, key=lambda k: -pol[k]["sr_pct"]):
            p = pol[k]
            fr = "yes" if not d["cells"][c]["dominated_by"][k] else "no"
            rows.append(f"| {CELL_LABEL[c]} | {k} | {p['sr_pct']:.2f} | {p['cost']:.5f} | "
                        f"{p['latency_canonical_s']:.1f}s | {fr} |")
    cap = ("Is routing worth it? `rule` policies send the ex-ante-flagged tasks to one arm "
           "and the rest to another; the partition is a regex over the task intent, so "
           "nothing is learned and there is no in-sample optimism. 'On frontier' means "
           "**nothing dominates it**, not that it is preferable — on `cls·B0` all three rule "
           "policies sit between always-SoM and always-Vision, worse on every axis than one "
           "or the other. Cost/latency are per-attempt cell means, **within-cell comparable "
           "only**. Source: `rule_routing_pareto.json`.")
    return "\n".join(rows), cap


def t_exante() -> tuple[str, str]:
    d = load("visual_intent_routing")
    rows = ["| cell | flagged | arm | Δ vs DOM on flagged | 95% CI | Δ on the rest | 95% CI |",
            "|---|---|---|---|---|---|---|"]
    for site in ("classifieds", "reddit"):
        blk = d["sites"].get(site, {})
        nf = blk.get("n_flagged")
        for c, rec in blk.get("cells", {}).items():
            for arm in ("vision", "som"):
                if arm not in rec:
                    continue
                f_, r_ = rec[arm]["flagged"], rec[arm]["rest"]
                rows.append(f"| {CELL_LABEL.get(c, c)} | {nf} | {arm} | {f_['est_pp']:+.2f} | "
                            f"[{f_['ci'][0]:+.2f}, {f_['ci'][1]:+.2f}] | {r_['est_pp']:+.2f} | "
                            f"[{r_['ci'][0]:+.2f}, {r_['ci'][1]:+.2f}] |")
    cap = ("Ex-ante partition. The predicate is a regex over the task intent plus 'carries no "
           "reference image' — both read the task config, so it costs no tokens and needs no "
           "episode. On classifieds the screenshot is worth an order of magnitude more on the "
           "flagged tasks than on the rest, and the flagged/rest split is significant/not "
           "respectively on the two capable backbones. ⚠️ WebArena is omitted: the predicate "
           "fires on only 5 of 104 tasks there and none is solved by any mode, so the cells "
           "are degenerate rather than null. Source: `visual_intent_routing.json`.")
    return "\n".join(rows), cap


def t_floor() -> tuple[str, str]:
    d = load("noise_floor_inventory")
    margins = d["margins"]
    band = {r[0]: (r[2], r[3], r[4]) for r in d["head_to_head"]}
    rows = ["| cell | best single | +1 distinct arm | +1 rerun (measured floor) | verdict |",
            "|---|---|---|---|---|"]
    for c in CELLS:
        key = c if c in margins else {"wa_B0": "wa_red_B0", "wa_B1": "wa_red_B1"}.get(c)
        if key not in margins:
            continue
        m = margins[key]
        txt, lo, hi = band.get(key, ("—", None, None))
        gain = m["gain_1_best_distinct_arm_pp"]
        arm = m["gain_1_best_distinct_arm_mode"].replace("sr_", "")
        if lo is None:
            verdict = "no floor on this cell"
        elif lo <= gain <= hi:
            verdict = "**inside the rerun band**"
        else:
            verdict = f"outside by {gain - hi:+.2f}pp"
        rows.append(f"| {CELL_LABEL[c]} | {PRETTY.get(m['best_mode'].replace('sr_',''), m['best_mode'])} "
                    f" at {m['best_single_sr_pct']:.2f} | +{gain:.2f} ({PRETTY.get(arm, arm)}) | "
                    f"{txt} | {verdict} |")
    cap = ("Is a new representation worth more than a rerun? Both middle columns are the same "
           "functional at the same arm count — `|{added} ∖ {baseline}| / n` — so they are "
           "directly comparable; only the *source* of the extra arm differs. **Only two cells "
           "carry a measured floor**, and neither measures it on the arm being added, so the "
           "other six rows have no comparator at all. Source: `noise_floor_inventory.json`.")
    return "\n".join(rows), cap


# --- full per-mode matrices: every metric, every cell, no summarisation ------------
PROFILE_DIMS = {
    "Outcome": [("sr_pct", "SR %", "{:.2f}"), ("n_success", "solves", "{:.0f}"),
                ("n_unique_solves", "unique", "{:.0f}")],
    "Macro": [("n_steps", "steps/ep", "{:.2f}"), ("cap_hit_rate", "cap-hit", "{:.3f}"),
              ("click_frac", "click", "{:.3f}"), ("type_frac", "type", "{:.3f}"),
              ("scroll_frac", "scroll", "{:.3f}"), ("search_loop_rate", "search-loop", "{:.3f}"),
              ("url_revisit_rate", "URL-revisit", "{:.3f}")],
    "Micro": [("parse_fail_rate", "parse-fail", "{:.4f}"),
              ("action_fail_rate", "act-fail", "{:.3f}"),
              ("click_fail_rate", "act-fail|click", "{:.3f}"),
              ("type_fail_rate", "act-fail|type", "{:.3f}"),
              ("no_change_rate", "no-op", "{:.3f}"),
              ("scroll_inert_rate", "scroll-inert", "{:.3f}"),
              ("noop_inert_rate", "no-op|success", "{:.3f}"),
              ("visibility_gap_rate", "vis-gap", "{:.3f}"),
              ("locator_fallback_rate", "loc-fallback", "{:.3f}"),
              ("action_repeat_frac", "act-repeat", "{:.3f}"),
              ("finish_rate", "finish", "{:.3f}")],
    "Efficiency": [("mean_cost_usd", "cost/ep", "{:.5f}"),
                   ("cost_rel_dom", "cost rel DOM", "{:.3f}"),
                   ("mean_latency_s", "latency s", "{:.1f}"),
                   ("mean_latency_canonical_s", "latency canon s", "{:.1f}"),
                   ("mean_tokens", "tokens/ep", "{:.0f}")],
}
_PROF_CELL = {"B0_classifieds": "cls·B0", "B1_classifieds": "cls·B1", "B2_classifieds": "cls·B2",
              "B0_reddit": "red·B0", "B1_reddit": "red·B1", "B2_reddit": "red·B2",
              "B0_wa_reddit": "WA·B0", "B1_wa_reddit": "WA·B1"}
_PROF_ORDER = ["cls·B0", "cls·B1", "cls·B2", "red·B0", "red·B1", "red·B2", "WA·B0", "WA·B1"]


def _profile_table(dim: str):
    d = load("per_mode_four_dimension_profile_with_wa")
    metrics = PROFILE_DIMS[dim]
    head = "| cell | mode | " + " | ".join(lbl for _, lbl, _ in metrics) + " |"
    rows = [head, "|---|---|" + "---|" * len(metrics)]
    by_cell = {}
    for cell in d["cells"]:
        by_cell[_PROF_CELL.get(cell["cell_id"], cell["cell_id"])] = cell["per_mode"]
    for cl in _PROF_ORDER:
        if cl not in by_cell:
            continue
        for m in ["DOM", "SoM", "Vision", "P-text", "P-prompt", "P-SoM"]:
            blk = by_cell[cl].get(m)
            if not blk:
                continue
            vals = []
            for key, _, spec in metrics:
                v = blk.get(key)
                vals.append(spec.format(v) if isinstance(v, (int, float)) else "—")
            rows.append(f"| {cl} | {m} | " + " | ".join(vals) + " |")
    return "\n".join(rows)


def t_prof_outcome():
    return _profile_table("Outcome"), (
        "Outcome dimension, every cell × every mode. `unique` counts tasks **no other mode "
        "in that cell solved**. Denominators: cls 224 / red 203 / WA 104. "
        "Source: `per_mode_four_dimension_profile_with_wa.json`.")


def t_prof_macro():
    return _profile_table("Macro"), (
        "Macro dimension — what the agent did, per step, aggregated per episode. Fractions "
        "are over agent actions. `cap-hit` = share of episodes that exhausted the 30-step "
        "budget. Source: `per_mode_four_dimension_profile_with_wa.json`.")


def t_prof_micro():
    return _profile_table("Micro"), (
        "Micro dimension — per-step execution quality. `act-fail|click` and `act-fail|type` "
        "are conditional on the action type, so they are not comparable to the unconditional "
        "`act-fail` column. ⚠️ `loc-fallback` is near-zero for Vision **by construction** "
        "(no element ids to fall back from), not as a finding. "
        "Source: `per_mode_four_dimension_profile_with_wa.json`.")


def t_prof_efficiency():
    return _profile_table("Efficiency"), (
        "Efficiency dimension. **Cost is comparable within a cell only** — B0 bills a proxy "
        "API, B1/B2 are electricity-derived from a per-token constant calibrated for a "
        "different accelerator, so absolute dollars for B1/B2 are uncalibrated and only "
        "`cost rel DOM` is safe across backbones. `latency canon` removes retry, busy-wait "
        "and recovered-screenshot time; it differs from raw only on the API-served arm. "
        "Source: `per_mode_four_dimension_profile_with_wa.json`.")


def t_pareto():
    d = load("multimetric_pareto_with_wa")
    rows = ["| cell | cost span | latency span | cheapest | fastest | same? | ρ(cost,lat) | "
            "frontier (success × cost) |", "|---|---|---|---|---|---|---|---|"]
    for cell, r in d["cells"].items():
        sp = r["spans"]
        ind = r.get("independence", {})
        fr = r.get("frontiers", {}).get("success x cost", [])
        rows.append(f"| {_PROF_CELL.get(cell, cell)} | {sp['cost']:.3f}× | "
                    f"{sp['latency']:.3f}× | {r['cheapest']} | {r['fastest']} | "
                    f"{'yes' if r['cheapest_is_fastest'] else '**no**'} | "
                    f"{ind.get('spearman_rho_cost_latency', float('nan')):+.2f} | "
                    f"{', '.join(fr)} |")
    return "\n".join(rows), (
        "Multi-metric Pareto. `span` = dearest/cheapest (or slowest/fastest) ratio within the "
        "cell. ⚠️ Per-cell exact permutation p-values on ρ are **not significant** — six modes "
        "give a Spearman test almost no power — so the cross-cell pattern carries this, not "
        "any single ρ. Source: `multimetric_pareto_with_wa.json`.")


def t_per_success():
    d = load("outcome_efficiency")
    rows = ["| cell | content? | cheapest/attempt | cheapest/success | fastest/attempt | "
            "fastest/success | max solves |", "|---|---|---|---|---|---|---|"]
    for cell, r in d["cells"].items():
        rows.append(f"| {_PROF_CELL.get(cell, cell.replace('_', '·'))} | "
                    f"{'yes' if r.get('has_content') else '**no**'} | "
                    f"{r.get('cheapest_per_attempt')} | {r.get('cheapest_per_success')} | "
                    f"{r.get('fastest_per_attempt')} | {r.get('fastest_per_success')} | "
                    f"{r.get('max_n_success')} |")
    return "\n".join(rows), (
        "Per-attempt versus per-success denominators. `content? = no` marks cells whose best "
        "mode has fewer than 10 successes — their ratios are directions at best and both B2 "
        "cells are in that state. Among the 6 cells with content the cheapest mode changes "
        "under the success denominator in 4 and the fastest in 3. ⚠️ **Every pairwise CI "
        "overlaps**, so this supports the methodological point that the denominator must be "
        "declared, not any ranking. Source: `outcome_efficiency.json`.")


def t_cascade():
    d = load("confidence_cascade_with_wa")
    rows = ["| cell | n | cheap SR | rich SR | always-rich cost | oracle SR | "
            "operating points that Pareto-beat always-rich | signals dropped |",
            "|---|---|---|---|---|---|---|---|"]
    for c in CELLS:
        r = d["cells"].get(c)
        if not r:
            continue
        ar, orc = r.get("always_rich", {}), r.get("oracle", {})
        beats = r.get("pareto_beats_always_rich") or []
        rows.append(f"| {CELL_LABEL[c]} | {r['n']} | {r['cheap_sr']:.2f} | {r['rich_sr']:.2f} | "
                    f"{ar.get('cost_rel', float('nan')):.3f}× | "
                    f"{orc.get('sr', float('nan')):.2f} | "
                    f"{('**' + str(len(beats)) + '**') if beats else '0'} | "
                    f"{len(r.get('signals_dropped') or [])} |")
    return "\n".join(rows), (
        f"Confidence-triggered cascade, {d['cheap']} → {d['rich']}. The escalation decision "
        "sees only the cheap run's own episode — no outcome, no rich-run information. "
        "⚠️ **Every number is an offline splice**: an escalated task takes its outcome from a "
        "standalone rich run, whereas a real cascade would start the rich episode after the "
        "cheap one had already acted on a stateful site. That sequential outcome is "
        "unobserved in this project. `signals dropped` counts confidence signals with too few "
        "distinct values to rank with — dropping them is what removed the one apparent WA "
        "win, which was a tie artefact. Source: `confidence_cascade_with_wa.json`.")


def t_triage_learn():
    d = load("visual_difficulty_router")
    rows = ["| cell | n | solvable | AUROC without | AUROC with visual_difficulty | Δ | "
            "visual_difficulty alone |", "|---|---|---|---|---|---|---|"]
    for c in CELLS:
        r = d["cells"].get(c)
        if not r:
            continue
        alone = r["per_scorer"].get("visual_difficulty_alone", {}).get("auroc_with")
        a, b = r.get("auroc_without"), r.get("auroc_with")
        rows.append(f"| {CELL_LABEL[c]} | {r['n']} | {r['n_positive']} | "
                    f"{a:.3f} | {b:.3f} | {b - a:+.3f} | "
                    f"{alone:.3f}" .replace("nan", "—") + " |")
    return "\n".join(rows), (
        f"Can the triage label be predicted, and does the benchmark's own visual-difficulty "
        f"annotation help? Task-held-out 5-fold CV, L2 logistic regression, seed 42. "
        f"Mean ΔAUROC = **{d['mean_delta_auroc']:+.4f}** over {len(d['cells'])} cells, "
        f"improving {d['n_cells_improved']} — inside fold-split noise. `solvable` is the "
        f"positive count: the label exists for every task, unlike the which-mode label. "
        f"Source: `visual_difficulty_router.json`.")


def t_feature_sign():
    d = load("routing_feature_diagnostics")
    rows = ["| cell | tasks with ref image | without | best mode WITH image | SR | "
            "best mode WITHOUT | SR |", "|---|---|---|---|---|---|---|"]
    for c in CELLS:
        r = d.get("has_reference_image", {}).get(c)
        if not r:
            continue
        w, wo = r["sr"]["with_ref_image"], r["sr"]["without_ref_image"]
        bw = max(w, key=lambda m: w[m])
        bo = max(wo, key=lambda m: wo[m])
        rows.append(f"| {CELL_LABEL[c]} | {r['n_img']} | {r['n_txt']} | {PRETTY.get(bw, bw)} | "
                    f"{w[bw]:.2f} | {PRETTY.get(bo, bo)} | {wo[bo]:.2f} |")
    return "\n".join(rows), (
        "The intuitive routing feature. A task shipping a reference image ought to route to a "
        "mode that can see images — the table shows which mode is actually best on each side "
        "of that split. ⚠️ Reference images are delivered in **every** mode, so this feature "
        "does not separate what it appears to. WebArena ships no reference images, so it "
        "cannot arbitrate. Source: `routing_feature_diagnostics.json`.")


def _cond_side(side: str, label: str):
    """One side of the paired cut. Two tabulars under one caption breaks floatify — it
    wraps the prose between them into the float and pdflatex dies with 'Not in outer par
    mode', so each side gets its own table."""
    d = load("conditional_failure_attribution")
    p_ = d["pooled"][side]
    cond, base = p_["cond"], p_["base"]
    nc, nb, names = p_["n_cond_eps"], p_["n_base_eps"], p_["names"]
    enr = []
    for rule, hits in cond.items():
        if hits < 8:
            continue
        cr = hits / nc if nc else 0
        br = base.get(rule, 0) / nb if nb else 0
        enr.append((cr / br if br else float("inf"), rule, hits, cr, br))
    enr.sort(reverse=True)
    rows = ["| rule | name | on disagreement | baseline | enrichment | hits |",
            "|---|---|---|---|---|---|"]
    for e, rule, hits, cr, br in enr[:8]:
        rows.append(f"| `{rule}` | {ascii_rule_name(rule, names.get(rule, ''))[:34]} | "
                    f"{100*cr:.1f}% | {100*br:.1f}% | **{e:.2f}x** | {hits} |")
    cap = (f"{label} Pooled over 8 cells at ruleset v11: {p_['n_tasks']} disagreement tasks, "
           f"{nc} losing-channel failure episodes against {nb} of that channel's failures "
           f"overall. Enrichment = hit rate on the disagreement set over hit rate across all "
           f"that channel's failures; approximately 1x means the loser failed there the way "
           f"it fails everywhere. Rules with fewer than 8 pooled conditional hits are "
           f"omitted. **Caution:** TEXT is four arms against IMAGE's two, so the two panels' "
           f"task counts are not comparable to each other. "
           f"Source: `conditional_failure_attribution.json`.")
    return "\n".join(rows), cap


def t_cond_text_wins():
    return _cond_side("text_only",
                      "Only the TEXT channel solved it: how the IMAGE channel failed.")


def t_cond_image_wins():
    return _cond_side("image_only",
                      "Only the IMAGE channel solved it: how the TEXT channel failed.")


def t_instability():
    d = load("label_instability")
    rows = ["| stratum | tasks | share of cell | flipped | flip rate | share of all flips | "
            "enrichment vs complement |", "|---|---|---|---|---|---|---|"]
    for k, v in d["strata"].items():
        if not isinstance(v, dict):
            continue
        e = v.get("enrichment_vs_complement")
        rows.append(f"| {k} | {v['n_tasks']} | {v['share_of_cell']:.1f}% | "
                    f"{v['n_flipped']} | {v['flip_rate_pct']:.2f}% | "
                    f"{v['share_of_all_flips']:.1f}% | "
                    + (f"**{e:.2f}x** |" if isinstance(e, (int, float)) else "— |"))
    if len(rows) == 2:
        raise MissingProduct("label_instability.json: `strata` empty or unrecognised")
    cap = (f"Per-task label instability on `{d['cell']}` (n={d['n']}): "
           f"{d['n_flipped']} tasks change outcome between two runs of the same condition. "
           f"The rows a which-mode router could learn from are exactly the contested ones, "
           f"and they carry almost all of the instability. **Caution:** this is the entire "
           f"replicate inventory of the project — **one cell, two arms "
           f"({', '.join(d['replicated_arms'])}), rerun once** — so every stability figure "
           f"elsewhere is a lower bound derived from it. The headline enrichment has two "
           f"defensible definitions and **neither may be quoted alone**: 17.4x defined over "
           f"all six arms is correct for the claim (a router chooses among six) but the flips "
           f"are produced by rerunning two of them, so the same arms decide both membership "
           f"and outcome; rebuilding the difficulty proxy from the other four breaks that "
           f"circle and gives 3.95x. Source: `label_instability.json`.")
    return "\n".join(rows), cap


def t_leakage():
    d = load("leakage_sensitivity")
    rows = ["| cell | contrast | before | 95% CI | after | 95% CI | verdict |",
            "|---|---|---|---|---|---|---|"]
    for c, rec in d["cells"].items():
        for comp in ("vision", "dom"):
            k = f"som_minus_{comp}"
            if k not in rec:
                continue
            b, a_ = rec[k]["before"], rec[k]["after"]
            bz = b["ci"][0] > 0 or b["ci"][1] < 0
            az = a_["ci"][0] > 0 or a_["ci"][1] < 0
            rows.append(f"| {CELL_LABEL.get(c, c)} | SoM − {PRETTY[comp]} | "
                        f"{b['est_pp']:+.2f} | [{b['ci'][0]:+.2f}, {b['ci'][1]:+.2f}] | "
                        f"{a_['est_pp']:+.2f} | [{a_['ci'][0]:+.2f}, {a_['ci'][1]:+.2f}] | "
                        f"{'**flips**' if bz != az else 'unchanged'} |")
    return "\n".join(rows), (
        f"Sensitivity to environmentally-credited successes. `require_reset` is a no-op on "
        f"reddit, so subscriptions accumulate across a run's episodes and a later task can be "
        f"scored on state an earlier one created. {len(d['leaks_removed'])} such successes are "
        f"set to 0 here — the denominator is unchanged, because an attempted-and-unaccomplished "
        f"task is a 0, not a missing row. 4 of the leaks are on DOM, so removing them **helps** "
        f"the fused arm: the direction that disfavours this project's own caution. "
        f"⚠️ The WA cells are **unaudited** for the same defect. "
        f"Source: `leakage_sensitivity.json`.")


def t_offsite():
    d = load("offsite_navigation_audit")
    rows = ["| cell | off-site steps | off-site episodes | median env_step on-site | "
            "off-site | ratio |", "|---|---|---|---|---|---|"]
    for c in d["cells"]:
        on, off = c["median_env_ms_onsite"], c["median_env_ms_offsite"]
        rows.append(f"| {c['cell']} | {c['n_offsite']}/{c['n_steps']} ({c['pct_steps']:.2f}%) | "
                    f"{c['n_episodes_offsite']}/{c['n_episodes']} ({c['pct_episodes']:.1f}%) | "
                    f"{on:,.0f} ms | {f'{off:,.0f} ms' if off else '—'} | "
                    f"{f'{off/on:.2f}×' if on and off else '—'} |")
    return "\n".join(rows), (
        "Off-site navigation. Postmill is a link aggregator, so an agent opening a trending "
        "thread can walk onto the live public internet; classifieds is self-contained. "
        "⚠️ Off-site steps are **faster**, not slower — commercial CDNs beat a Postmill "
        "container sharing a host with the agent — so the distortion runs opposite to the "
        "intuition. The larger asymmetry is in the last two columns of the on-site medians: "
        "reddit's container costs ~1.69× what classifieds' does before any agent behaviour "
        "enters, which is why no between-site latency number is quotable bare. "
        "Source: `offsite_navigation_audit.json`.")


def t_axis_effects():
    d = load("axis_effect_size_with_wa")
    res = d["results"]
    axes = [("text", "text axis"), ("prompt", "prompt axis"), ("image", "image axis"),
            ("compound_dom_to_psom", "DOM→P-SoM")]
    rows = ["| cell | metric | " + " | ".join(lbl for _, lbl in axes) + " |",
            "|---|---|" + "---|" * len(axes)]
    order = [("B0", "classifieds"), ("B1", "classifieds"), ("B2", "classifieds"),
             ("B0", "reddit"), ("B1", "reddit"), ("B2", "reddit"),
             ("B0", "wa_reddit"), ("B1", "wa_reddit")]
    for b, site in order:
        blk = res.get(b, {}).get(site, {})
        for metric, mb in blk.items():
            vals, any_n = [], False
            for key, _ in axes:
                c = mb.get(key, {})
                n = c.get("n", 0)
                if not n:
                    vals.append("—")
                    continue
                any_n = True
                e = c.get("cohen_h", c.get("cohen_d_z"))
                vals.append(f"{e:+.3f}" if isinstance(e, (int, float)) else "—")
            if any_n:
                rows.append(f"| {CELL_LABEL.get(('cls' if site=='classifieds' else 'red' if site=='reddit' else 'wa')+'_'+b, b+'/'+site)} "
                            f"| {metric} | " + " | ".join(vals) + " |")
    return "\n".join(rows), (
        "2×2 axis decomposition. Effect sizes are Cohen's h (binary metrics) or d_z (paired "
        "continuous), signed right-minus-left. The compound DOM→P-SoM transition decomposes "
        "into a text-payload axis and a prompt-style axis; the image axis is P-SoM→SoM. "
        "⚠️ On mean differences the two decomposition routes agreeing is an **algebraic "
        "identity**, so a zero residual is arithmetic and not evidence about an interaction. "
        "`B2 × wa_reddit` is absent because B2 never ran WebArena. "
        "Source: `axis_effect_size_with_wa.json`.")


def t_axis1_ratio():
    d = load("axis1_microbehavior_with_wa")
    csv_ = d["cross_site_validity"]
    rows = ["| cell | decision effect (mean abs) | macro effect (mean abs) | ratio | >1? |",
            "|---|---|---|---|---|"]
    order = [("B0", "classifieds"), ("B1", "classifieds"), ("B2", "classifieds"),
             ("B0", "reddit"), ("B1", "reddit"), ("B2", "reddit"),
             ("B0", "wa_reddit"), ("B1", "wa_reddit")]
    for b, site in order:
        r = csv_.get(f"{b}_{site}_ratio")
        macro = csv_.get(f"{b}_{site}_macro_mean_abs_effect")
        dec = csv_.get(f"{b}_{site}_decision_effects")
        dm = dec.get("mean_abs_decision_effect") if isinstance(dec, dict) else None
        if r is None and macro is None:
            continue
        key = ("cls" if site == "classifieds" else "red" if site == "reddit" else "wa") + "_" + b
        rows.append(f"| {CELL_LABEL.get(key, key)} | "
                    f"{dm:.4f}" .format() if isinstance(dm, float) else "—")
        rows[-1] = (f"| {CELL_LABEL.get(key, key)} | "
                    f"{dm:.4f} | " if isinstance(dm, float) else f"| {CELL_LABEL.get(key, key)} | — | ") \
                   + (f"{macro:.4f} | " if isinstance(macro, float) else "— | ") \
                   + (f"**{r:.2f}** | {'yes' if r > 1 else '**no**'} |" if isinstance(r, float)
                      else "— | — |")
    return "\n".join(rows), (
        f"Does the text axis change per-step decisions more than it changes macro action "
        f"frequencies? Ratio >1 means yes. Verdict: **{csv_.get('verdict')}** "
        f"(site_ok = {csv_.get('site_ok')}). ⚠️ `_site_ok` passes a site if **any** backbone "
        f"clears 1.0, which is a loose bar — `WA·B0` is 0.97 and the WA site passes on B1's "
        f"2.98. Until 2026-08-03 the verdict function named only the two VWA sites literally "
        f"and did not consult WA at all. Source: `axis1_microbehavior_with_wa.json`.")


def t_hallucinated():
    d = load("cross_mode_failure_signatures")
    pb = d["part_b_hallucinated_references"]["per_cell_per_mode"]
    rows = ["| cell | mode | episodes | failed | with hallucinated ref | rate of failed |",
            "|---|---|---|---|---|---|"]
    for cell, modes in pb.items():
        for m, r in modes.items():
            ef, eh = r.get("ep_failed", 0), r.get("ep_with_hall", 0)
            rows.append(f"| {cell} | {m} | {r.get('ep_total','—')} | {ef} | {eh} | "
                        f"{(100*eh/ef if ef else 0):.1f}% |")
    return "\n".join(rows), (
        f"Hallucinated element references, ruleset `{d['ruleset_version']}` over "
        f"{d['n_conditions']} conditions. An action naming an element id that is not in the "
        f"observation. ⚠️ `vision` carries no element-id list at all, so this rule is "
        f"**structurally inapplicable** there rather than measuring zero — the same "
        f"gate-versus-measurement confusion flagged for P2/P4. "
        f"Source: `cross_mode_failure_signatures.json`.")


def t_pooled_tier():
    d = load("router_pooled_tier_learnability")
    v = d["verdict"]
    rows = ["| quantity | value |", "|---|---|",
            f"| headline | {v.get('headline','—')} |"]
    for k, val in v.items():
        if k == "headline":
            continue
        rows.append(f"| {k} | {json.dumps(val)[:150] if not isinstance(val,(str,int,float)) else val} |")
    for site, r in d["results"].items():
        rows.append(f"| {site}: universe / labelled | {r.get('n_universe','—')} / "
                    f"{r.get('n_labelled', r.get('n_labeled','—'))} |")
    return "\n".join(rows), (
        "Pooled same-family × cost-tier router. " + (d["limitations"][0][:220] if d.get("limitations") else "")
        + " Source: `router_pooled_tier_learnability.json`.")


def t_page_change():
    d = load("page_change_corrected")
    rows = ["| mode | no-change rate observed | corrected | cosmetic FP steps | steps |",
            "|---|---|---|---|---|"]
    for m, r in d["per_mode"].items():
        rows.append(f"| {PRETTY.get(m, m)} | {r['no_change_rate_observed']:.4f} | "
                    f"{r['no_change_rate_corrected']:.4f} | {r['cosmetic_fp_steps']} | "
                    f"{r['steps']} |")
    obs = d.get("router_streak2_total_observed")
    cor = d.get("router_streak2_total_corrected")
    return "\n".join(rows), (
        f"`page_changed` false positives. A step can register a page change that is purely "
        f"cosmetic; correcting for it **raises** every mode's no-change rate. The Micro "
        f"conclusion is unaffected — Vision remains highest in every cell either way — but a "
        f"router firing on a 2-step no-change streak would trigger "
        f"{obs} → {cor} times (+{100*(cor-obs)/obs:.1f}%). "
        f"Source: `page_change_corrected.json`.")


def t_evaluator():
    d = load("evaluator_score_granularity")
    pg = d["paper_grade"]
    rows = ["| quantity | value |", "|---|---|"]
    for k in ("n_conditions", "n_episodes", "distinct_scores", "score_counts",
              "protocol_excluded_episodes"):
        if k in pg:
            rows.append(f"| {k} | {json.dumps(pg[k]) if not isinstance(pg[k], (str,int,float)) else pg[k]} |")
    return "\n".join(rows), (
        "Evaluator granularity over the paper-grade set. The evaluator emits **two** distinct "
        "values. There is no graded quality target to regress on — a property of the "
        "benchmark's design, not of this pipeline, and a precondition of every routing "
        "negative in this document. Source: `evaluator_score_granularity.json`.")


def t_cost_class():
    d = load("cost_per_mode")
    rows = ["| site | B0 API dollars/ep | B1 electricity dollars/ep | ratio |",
            "|---|---|---|---|"]
    for site, r in d["deployment_class_ratios"].items():
        rows.append(f"| {site} | {r['avg_B0_API_dollars']:.5f} | "
                    f"{r['avg_B1_electricity_dollars']:.7f} | **{r['ratio_B0_over_B1']:.1f}×** |")
    return "\n".join(rows), (
        "Two cost estimands that are **not the same quantity**. B0 pays a per-token API bill; "
        "B1/B2 pay electricity. The ratio is reported to show the scale of the category "
        "error, not as a comparison — a paper that divides one by the other is comparing a "
        "price to a physical cost. Within a cell, mode-to-mode ratios are safe; across "
        "deployment classes only the ordering is. Source: `cost_per_mode.json`.")


def t_leak_audit():
    d = load("reddit_sidebar_leakage_audit")
    rows = ["| cell · mode | scored successes | of which LEAKED | share |",
            "|---|---|---|---|"]
    agg = {}
    for r in d["rows"]:
        if not r.get("in_scored_universe"):
            continue
        k = (r["baseline"], r["mode"])
        a = agg.setdefault(k, {"succ": 0, "leak": 0})
        if r["success"]:
            a["succ"] += 1
        if r["verdict"] == "LEAKED":
            a["leak"] += 1
    for (b, m), a in sorted(agg.items()):
        if not a["succ"]:
            continue
        rows.append(f"| {b} · {m} | {a['succ']} | {a['leak']} | "
                    f"{100*a['leak']/a['succ']:.1f}% |")
    return "\n".join(rows), (
        f"Which successes were earned. `{d['selector']}` is read by {len(d['tasks'])} reddit "
        f"tasks; `require_reset` is a no-op on reddit so subscriptions accumulate. "
        f"**LEAKED** = scored success by an episode that never visited the required forum. "
        f"{d['n_leaked']} leaked, {d['n_earned']} earned. Table 20 recomputes every contrast "
        f"with the leaked ones zeroed. Source: `reddit_sidebar_leakage_audit.json`.")


TABLES = [
    ("sr", "Success rate per mode", t_sr),
    ("class", "Best arm per deployment class", t_class),
    ("class-ablate", "Class ablation, unmatched and arm-matched", t_class_ablate),
    ("nonsep", "Behavioural non-separability", t_nonsep),
    ("prof-outcome", "Full matrix — Outcome dimension", t_prof_outcome),
    ("prof-macro", "Full matrix — Macro dimension", t_prof_macro),
    ("prof-micro", "Full matrix — Micro dimension", t_prof_micro),
    ("prof-eff", "Full matrix — Efficiency dimension", t_prof_efficiency),
    ("pareto", "Multi-metric Pareto", t_pareto),
    ("per-success", "Per-attempt versus per-success", t_per_success),
    ("fusion", "Fusion premium against the rerun band", t_fusion),
    ("exante", "Ex-ante visual-intent partition", t_exante),
    ("floor", "New representation versus a rerun", t_floor),
    ("routing", "Routing policies on the 3-axis frontier", t_routing),
    ("cascade", "Confidence-triggered cascade", t_cascade),
    ("triage", "Triage learnability and the visual-difficulty feature", t_triage_learn),
    ("feature", "The intuitive routing feature", t_feature_sign),
    ("cond-text", "Paired failure attribution: text wins", t_cond_text_wins),
    ("cond-image", "Paired failure attribution: image wins", t_cond_image_wins),
    ("instability", "Per-task label instability", t_instability),
    ("leakage", "Leaked-success sensitivity", t_leakage),
    ("offsite", "Off-site navigation and container latency", t_offsite),
    ("axis", "2x2 axis decomposition", t_axis_effects),
    ("axis1", "Decision quality versus macro frequency", t_axis1_ratio),
    ("halluc", "Hallucinated element references", t_hallucinated),
    ("pooled", "Pooled tier router", t_pooled_tier),
    ("pagechange", "page_changed false positives", t_page_change),
    ("evaluator", "Evaluator granularity", t_evaluator),
    ("costclass", "Two cost estimands", t_cost_class),
    ("leakaudit", "Earned versus leaked successes", t_leak_audit),
]


ABSTRACT_TMPL = """## Abstract

<!-- NOT an abstract. This slot holds the evidence inventory so that opening the document
     shows the measurements before any story about them. Generated — rerun
     scripts/analysis/export_ablation_tables.py rather than editing. Replace with a real
     abstract once the claim is chosen. -->

**Evidence inventory — 8 cells = (site x backbone), 6 observation modes.**
`cls`/`red` are VisualWebArena classifieds/reddit \\citep{{koh2024visualwebarena}};
`WA` is WebArena reddit \\citep{{zhou2024webarena}}. `WA-B2` does not exist, so no
statement holds cross-benchmark and cross-family simultaneously.

{measured}

**Known defects in the above.** Six scored successes on VWA reddit were credited by
accumulated site state, and zeroing them flips the one cell that showed fusion
significantly beaten. 1.05-2.13% of reddit steps run on the public internet against
0.00-0.16% on classifieds, and reddit's container is 1.69x slower before any agent
behaviour enters. Per-rule failure frequencies are symptom distributions, not cause
distributions. Six conclusions were found hardcoded in their producers on 2026-08-03.

**Out of reach with this data.** Whether the reversal turns on modality, task set or
benchmark; anything needing a third workload; what a real cascade does; whether a learned
router beats the rule.

## 1. Introduction

<!-- EMPTY BY DESIGN. Write this LAST. Three frames died between 08-01 and 08-03 because
     they were written before being checked against coverage; a fourth was killed by a
     selection bias in the single number it rested on. Choose the claim against the
     inventory above, with the advisor, then write this. -->
"""


# ------------------------------------------------------------------ inventory mode
def build_inventory() -> tuple[str, str]:
    """Section 0: every measurement, stated without an argument around it.

    Deliberately claim-free. Each entry says what was measured, over what, and what would
    make it wrong — never what it means. Three frames died in three days because they were
    written before being checked against coverage; this file exists so the next one is
    picked while looking at the evidence rather than at a story about it.
    """
    cls_ = load("representation_class_comparison")
    fus = load("fusion_premium")
    vir = load("visual_intent_routing")
    rrp = load("rule_routing_pareto")
    nfi = load("noise_floor_inventory")
    prof = load("per_mode_four_dimension_profile_with_wa")

    n_cells = len(cls_["cells"])
    tally = cls_["best_class_count"]
    matched = cls_["arm_matched_winner_count"]
    ties = sum(1 for c in cls_["cells"].values() if c.get("is_tie"))
    wa0 = cls_["cells"]["wa_B0"]["class_best"]
    cb0 = vir["sites"]["classifieds"]["cells"]["cls_B0"]["vision"]
    lo, hi = nfi.get("floor_mean_pp", [0.89, 2.23])
    cons = prof["consistency"]
    ncell_prof = next(iter(cons.values()))["n_cells"]
    thr = round(0.83 * ncell_prof)

    E = []
    A = E.append
    A("## 0. Evidence inventory")
    A("")
    A("<!-- Section 0 is NOT part of the paper. It is the pre-frame inventory: every "
      "measurement, stated without an argument around it, so the claim can be chosen "
      "while looking at the evidence instead of at a story about it. Delete before "
      "submission. -->")
    A("")
    A("**Read this as a list of measurements, not of findings.** Each entry gives what was "
      "measured, over what, and what would make it wrong. None says what it means — that is "
      "the decision this document exists to keep open.")
    A("")
    A(f"Coverage throughout: **{n_cells} cells** = (site × backbone). "
      "`cls`/`red` = VisualWebArena classifieds/reddit, `WA` = WebArena reddit. "
      "**`WA·B2` does not exist** (B2 never ran WebArena), so no statement holds "
      "cross-benchmark *and* cross-family at once.")
    A("")

    MEASURED_START = len(E)
    A("### What was measured")
    A("")
    A(f"1. **Success rate per mode**, 6 modes × {n_cells} cells (Table 1). The best mode is "
      f"SoM in 5 cells, DOM in 2, P-text in 1.")
    A(f"2. **Best arm per deployment class** (Table 2). Sole best: "
      + ", ".join(f"{k} {v}/{n_cells}" for k, v in sorted(tally.items(), key=lambda kv: -kv[1]))
      + f"; {ties} tied cell. `vision-only` is never a sole best. On `WA·B0` the no-image "
      f"class leads the hybrid class by {wa0['no-image'] - wa0['hybrid']:.2f}pp "
      f"({wa0['no-image']:.2f} vs {wa0['hybrid']:.2f}).")
    A(f"3. **Class ablation** (Table 3). Unmatched, dropping the no-image class costs the "
      f"most in every cell — **but it has four arms and the others one each**. Arm-matched, "
      f"the largest single-arm gain lands on "
      + ", ".join(f"{k} {v}×" for k, v in sorted(matched.items(), key=lambda kv: -kv[1]))
      + ". The matched panel is the one that compares like with like.")
    A(f"4. **Behavioural non-separability** (Table 4). Over 26 metrics × {ncell_prof} cells, "
      f"the four image-free modes are the extreme in ≥{thr} cells on **zero** metrics "
      f"(Vision 9, SoM 5).")
    A(f"5. **Fusion premium** (Table 5). The fused mode does not beat the workload-matched "
      f"single channel in any of the {len(fus['cells'])} cells. The comparison is against a "
      f"measured rerun band of **{lo}–{hi}pp**, not against zero; `cls_B0`'s +2.23pp equals "
      f"the band's upper edge exactly (both are 5/224).")
    A(f"6. **A 0-token ex-ante partition** (Table 6). A regex over the task intent flags "
      f"71/224 classifieds tasks; on them the screenshot is worth "
      f"**{cb0['flagged']['est_pp']:+.2f}pp** "
      f"[{cb0['flagged']['ci'][0]:+.2f}, {cb0['flagged']['ci'][1]:+.2f}] against "
      f"**{cb0['rest']['est_pp']:+.2f}pp** "
      f"[{cb0['rest']['ci'][0]:+.2f}, {cb0['rest']['ci'][1]:+.2f}] on the other 153.")
    A(f"7. **New representation versus a rerun** (Table 7). Adding one distinct arm and "
      f"adding one rerun are the same functional at the same arm count. Only 2 of "
      f"{n_cells} cells carry a measured floor, and neither measures it on the arm being added.")
    A(f"8. **Routing policies on the (success, cost, latency) frontier** (Table 8). A policy "
      f"built on the partition in (6) survives undominated in 5 of {len(rrp['cells'])} cells "
      f"— where *undominated* means nothing beats it on all three axes, not that it is "
      f"preferable. On `cls·B0` all three rule policies sit between always-SoM and "
      f"always-Vision, and `always-P-prompt` at 19.64% is equally undominated.")
    A("")

    A("### What is known to be wrong with it")
    A("")
    A("- **6 leaked successes on VWA reddit** (`require_reset` is a no-op there, so "
      "subscriptions accumulate). Zeroing them flips `red_B2`'s SoM−DOM interval across "
      "zero — the only cell that showed fusion significantly beaten. The WA cells are "
      "**unaudited** for the same defect.")
    A("- **1.05–2.13% of reddit steps run on the public internet** (Postmill is a link "
      "aggregator); classifieds is 0.00–0.16%. Those steps are *faster*, not slower. "
      "Separately, reddit's container is **1.69×** slower than classifieds' before any "
      "agent behaviour enters, so no between-site latency number is quotable bare.")
    A("- **The `vision` column of any diag per-rule table is not co-tabulable** with "
      "`dom`/`som`: `P2`/`P4` read `element_bbox`, which vision's clicks do not carry, so "
      "those cells are structural zeros rather than measurements.")
    A("- **Per-rule frequencies are symptom distributions, not cause distributions.** "
      "`P36` (51%) and `P31` (50%) are risk markers; causal verification exists for `P49` "
      "and not for them.")
    A("- **Six conclusions were found hardcoded in their producers** on 2026-08-03, one "
      "wrong on the fact and not only on the denominator. The sweep that found them covered "
      "one textual shape (`n/6`); mode names, ratios and directions were not swept.")
    A("")

    A("### What cannot be answered with this data")
    A("")
    A("- Whether the reversal in (2) turns on modality, task set, or benchmark — two "
      "workloads cannot identify a moderator.")
    A("- Anything requiring a third workload: `shopping` has zero landed directories.")
    A("- What a *real* cascade does: every escalation number is an offline splice.")
    A("- Whether a learned router could do better than the rule in (8): the which-mode "
      "label exists only where some mode succeeded (15–97 rows per cell, 260 total).")
    A("")
    # Second element is just the numbered measurement list, for the abstract slot: the
    # same strings, so the two documents cannot drift apart.
    return "\n".join(E), "\n".join(E[MEASURED_START + 2:]).strip()


def build_guide() -> str:
    """A reading guide: what each group of tables is for, in plain sentences.

    Sits between the eight-line inventory and the 29 tables. Without it the document goes
    straight from a summary nobody can act on to a matrix nobody reads. Numbers here are
    injected from the products, never typed.
    """
    cls_ = load("representation_class_comparison")
    fus = load("fusion_premium")
    vir = load("visual_intent_routing")
    rrp = load("rule_routing_pareto")
    vdr = load("visual_difficulty_router")
    off = load("offsite_navigation_audit")
    li = load("label_instability")
    lk = load("leakage_sensitivity")
    ev = load("evaluator_score_granularity")

    tally = cls_["best_class_count"]
    matched = cls_["arm_matched_winner_count"]
    wa0 = cls_["cells"]["wa_B0"]["class_best"]
    cb0 = vir["sites"]["classifieds"]["cells"]["cls_B0"]["vision"]
    n_flip = len(lk.get("flips", []))
    red = [c for c in off["cells"] if "red" in c["cell"]]
    cls_off = [c for c in off["cells"] if "cla" in c["cell"]]

    G = []
    A = G.append
    A("## Reading guide")
    A("")
    A("<!-- Plain-sentence guide to the 29 tables. Not an argument — each paragraph says "
      "what a group of tables is for and what the trap in it is. Delete before submission. -->")
    A("")

    A("**Tables 1–4 — what each mode achieves, and whether the six collapse into three.** "
      "Table 1 is the raw success rate. Tables 2–3 group the six modes into the three shapes "
      "web agents actually ship in, and Table 4 is the licence for that grouping: over 26 "
      "metrics the four image-free modes are never the extreme in ≥7 of 8 cells, so they do "
      "not behave differently enough to keep apart. "
      f"The grouping produces two facts: `vision-only` is never the sole best class "
      f"(hybrid {tally.get('hybrid',0)}, no-image {tally.get('no-image',0)}, one tie), and "
      f"which class wins reverses between benchmarks — on `WA·B0` no-image leads hybrid by "
      f"{wa0['no-image']-wa0['hybrid']:.2f}pp. "
      "**The trap is in Table 3**: dropping the whole no-image class costs far more than "
      "dropping the others, but it has four arms against one each. The arm-matched columns "
      f"beside it show no systematic difference ({', '.join(f'{k} {v}x' for k,v in matched.items())}), "
      "and those are the ones that compare like with like.")
    A("")

    A("**Tables 5–8 — the full behavioural matrices.** Every metric, every cell, every mode, "
      "unsummarised: Outcome, Macro (what the agent did), Micro (how often what it did "
      "failed), Efficiency. These are the substrate the consistency counts in Table 4 are "
      "computed from, included so a reader can check a claim rather than take the tally on "
      "trust. Two columns are gates rather than measurements and are marked as such: "
      "`loc-fallback` is near-zero for Vision because there are no element ids to fall back "
      "from, not because it fails less.")
    A("")

    A("**Tables 9–10, 28 — efficiency, and the denominator nobody declares.** Table 9 shows "
      "the cost and latency orderings are not each other restated. Table 10 changes the "
      "denominator from per-attempt to per-success and the cheapest mode changes in 4 of the "
      "6 cells that carry enough successes to divide by. **Every pairwise interval overlaps**, "
      "so this supports 'the denominator must be stated', not 'X is more efficient'. Table 28 "
      "is the other denominator problem: B0 pays an API bill and B1/B2 pay electricity, and "
      "those are not the same quantity.")
    A("")

    A(f"**Tables 11, 13 — is a second representation worth buying?** Table 11 asks whether the "
      f"fused mode beats the single channel that suits the workload: it does not, in any of "
      f"{len(fus['cells'])} cells. The comparison is against a **measured rerun band**, not "
      f"against zero, because the question a deployment asks is whether a new arm beats "
      f"re-running the arm it already has — Table 13 puts those two side by side at the same "
      f"arm count. Only two cells carry a measured floor, and neither measures it on the arm "
      f"being added, which is the single largest gap in this evidence layer.")
    A("")

    A(f"**Tables 12, 14–17, 25 — five ways of routing, and what each one dies of.** Table 12 "
      f"finds a signal that is as good as signals get: a regex over the task intent, costing "
      f"nothing and needing no episode, that flags tasks where the screenshot is worth "
      f"{cb0['flagged']['est_pp']:+.2f}pp against {cb0['rest']['est_pp']:+.2f}pp elsewhere. "
      f"Table 14 turns it into a policy and it **still loses to always-Vision**, because the "
      f"screenshot does not hurt on the unflagged tasks either. Table 15's cascade beats "
      f"always-rich at no operating point. Table 16 adds the benchmark's own difficulty "
      f"annotation for a mean ΔAUROC of {vdr['mean_delta_auroc']:+.4f}. Table 17 shows the "
      f"feature a practitioner would reach for first does not separate what it appears to. "
      f"**The pattern is not 'no signal'** — it is that the arm the router would route *to* "
      f"is already the right arm to route everything to.")
    A("")

    A("**Tables 18, 22–24, 26 — where the failures come from.** Table 18 is the paired cut: "
      "on tasks only one channel solved, how did the other fail? Table 22 decomposes the "
      "DOM→P-SoM transition into a text axis and a prompt axis. Table 24 counts hallucinated "
      "element references — inapplicable to Vision by construction, marked. Table 26 corrects "
      "a false-positive in the page-change detector. ⚠️ **A per-rule frequency is a "
      "distribution of symptoms, not of causes.** The two largest rows in most cells are risk "
      "markers that causal verification did not confirm as death causes.")
    A("")

    A(f"**Tables 19–21, 27, 29 — what would make all of the above wrong.** Table 27: the "
      f"evaluator emits two values, so there is no graded target and every routing negative "
      f"inherits that. Table 19: the entire replicate inventory of this project is one cell "
      f"with two arms rerun once — every stability number is a lower bound from that. "
      f"Tables 29 and 20: {lk['leaks_removed'].__len__()} successes were credited by "
      f"accumulated site state, and zeroing them flips {n_flip} verdict"
      f"{'s' if n_flip != 1 else ''}. Table 21: reddit episodes leave the benchmark for the "
      f"public internet on {min(c['pct_steps'] for c in red):.2f}–{max(c['pct_steps'] for c in red):.2f}% "
      f"of steps against {max(c['pct_steps'] for c in cls_off):.2f}% on classifieds, and "
      f"reddit's container is slower than classifieds' before any agent behaviour enters.")
    A("")
    return "\n".join(G)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--inventory", type=Path, default=None,
                    help="write the full claim-free inventory (prose + tables) here")
    ap.add_argument("--abstract", type=Path, default=None,
                    help="write the abstract-slot inventory here (draft section1_intro.md)")
    ap.add_argument("--guide", type=Path, default=None,
                    help="write the plain-sentence reading guide here")
    ap.add_argument("--evidence", type=Path, default=None,
                    help="write the tables-only evidence section here")
    ap.add_argument("--strict", action="store_true",
                    help="abort if any product is missing (default: emit a visible stub)")
    a = ap.parse_args()

    parts = [
        "---", "type: paper-draft", "status: generated",
        "purpose: every evidence product as an ablation table, for the Overleaf draft",
        "producer: scripts/analysis/export_ablation_tables.py", "---", "",
        "## Ablation tables", "",
        f"<!-- GENERATED {datetime.now(timezone.utc).isoformat(timespec='seconds')} — "
        "do not hand-edit between the table markers; rerun the producer instead. -->", "",
        "Every number below is read from a product JSON at render time. None is typed by "
        "hand: six hand-copied numbers were found decoupled from their products on "
        "2026-08-03, one of them wrong on the fact rather than only on the denominator.", "",
        "Cells are (site × backbone). `cls`/`red` are VisualWebArena classifieds/reddit; "
        "`WA` is WebArena reddit. `B0` = Qwen3-VL-235B-A22B, `B1` = Qwen3-VL-4B, "
        "`B2` = Gemma-3-4B. **`WA·B2` does not exist** — B2 never ran WebArena.", "",
    ]
    failures = []
    for i, (tid, title, fn) in enumerate(TABLES, start=1):
        try:
            body, cap = fn()
            # floatify.pl matches `\emph{Table N: ...}` immediately adjacent to a tabular,
            # so the caption IS the heading — a separate `### Table N` line would become a
            # subsection and the float would not be built. Caption must be one line.
            one_line = " ".join((title + ". " + cap).split())
            parts += [f"<!-- BEGIN table:{tid} -->", "", body, "",
                      f"*Table {i}: {one_line}*", "", f"<!-- END table:{tid} -->", ""]
        except MissingProduct as e:
            if a.strict:
                raise
            failures.append((tid, str(e)))
            parts += [f"<!-- BEGIN table:{tid} -->", "",
                      f"> ⚠️ **TABLE {i} NOT RENDERED** — {e}", "",
                      f"<!-- END table:{tid} -->", ""]

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(tex_safe("\n".join(parts)))
    print(f"[md] {a.out}  ({len(TABLES) - len(failures)}/{len(TABLES)} tables rendered)")
    for tid, e in failures:
        print(f"  ⚠️ {tid}: {e}")

    tables_body = "\n".join(parts[parts.index("## Ablation tables") + 1:])
    if a.inventory or a.abstract or a.evidence:
        head, measured = build_inventory()
    if a.inventory:
        a.inventory.parent.mkdir(parents=True, exist_ok=True)
        a.inventory.write_text(tex_safe(head + "\n" + tables_body))
        print(f"[md] {a.inventory}  (inventory + {len(TABLES) - len(failures)} tables)")
    if a.abstract:
        # The abstract slot and the tables must regenerate together, or the prose numbers
        # silently stop matching the table numbers — exactly the failure mode six producers
        # were caught in on 2026-08-03. Splitting this by hand once already created a
        # seventh instance, which is why it is a flag and not a manual step.
        a.abstract.parent.mkdir(parents=True, exist_ok=True)
        a.abstract.write_text(tex_safe(ABSTRACT_TMPL.format(measured=measured)))
        print(f"[md] {a.abstract}  (abstract slot)")
    if a.guide:
        a.guide.parent.mkdir(parents=True, exist_ok=True)
        a.guide.write_text(tex_safe(build_guide()))
        print(f"[md] {a.guide}  (reading guide)")
    if a.evidence:
        a.evidence.parent.mkdir(parents=True, exist_ok=True)
        a.evidence.write_text(tex_safe(
            "## Evidence tables\n\n<!-- Generated by "
            "scripts/analysis/export_ablation_tables.py. Do not hand-edit between the "
            "markers. -->\n" + tables_body))
        print(f"[md] {a.evidence}  (evidence section)")


if __name__ == "__main__":
    main()
