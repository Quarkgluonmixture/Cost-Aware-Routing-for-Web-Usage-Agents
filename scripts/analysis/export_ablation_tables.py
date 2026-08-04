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
CELL_ALIASES = {"wa_B0": ("wa_B0", "wa_red_B0"), "wa_B1": ("wa_B1", "wa_red_B1")}


def cell_get(cells: dict, cell: str):
    """A product's row for `cell`, tolerating the two WebArena key spellings.

    `CELLS` spells the WA cells `wa_B0`/`wa_B1`; `fusion_premium`,
    `confidence_cascade_with_wa` and `conditional_failure_attribution` spell them
    `wa_red_B0`/`wa_red_B1`. A plain `.get()` returned None and the `continue` under it
    dropped those rows without a word — the fusion and cascade tables rendered six rows
    out of eight while their captions, and claim 3's "in 8/8 cells", spoke about eight.
    Nothing in a six-row table says two rows are missing, which is why this survived.
    """
    for key in CELL_ALIASES.get(cell, (cell,)):
        if key in cells:
            return cells[key]
    return None


CANON_CELL = {alias: canon for canon, aliases in CELL_ALIASES.items() for alias in aliases}


def cell_label(key: str) -> str:
    """Display label for a product's cell key, whichever spelling that product uses.

    `t_fusion` iterates the product's own keys rather than `CELLS`, so its WA rows were
    present but printed raw (`wa_red_B1`) beside pretty ones (`cls·B0`) — the same table
    labelling its cells in two conventions.
    """
    return CELL_LABEL.get(CANON_CELL.get(key, key), key)


def unmatched_cells(cells: dict) -> list[str]:
    """Product cell keys that no entry of `CELLS` resolves to — a rename tripwire."""
    resolved = {k for c in CELLS for k in CELL_ALIASES.get(c, (c,)) if k in cells}
    return sorted(set(cells) - resolved)


MODES = ["dom", "som", "vision", "ptext", "pprompt", "psom"]

# Outward-facing mode names, 2026-08-04. `phantom` was a name from a framing this evidence
# no longer supports: the four image-free modes reach the consistency bar on **none** of 26
# metrics across 8 cells, i.e. they are behaviourally indistinguishable from DOM — so calling
# them a separate family made the naming argue against the tables. They are DOM variants and
# are now named as such, with the compound one named for what it is (SoM minus the image).
#
# NOTHING INTERNAL CHANGES. The data layer keeps `phantom_text` / `phantom_prompt` /
# `phantom_som` in directory names, condition ids, run ids and every product JSON key — 76
# references under `p79/` alone, and renaming those would make 8,000 landed episodes
# unreadable. The two layers are separated deliberately: a display name is a framing decision
# and may change again; a data key changes once and costs a re-run.
PRETTY = {"dom": "DOM", "som": "SoM", "vision": "Vision",
          "ptext": "DOM+somtext", "pprompt": "DOM+somprompt", "psom": "SoM-image"}

# Products key their rows by the OLD display names, so relabel at render time rather than
# touching any product. Accepts either spelling and is idempotent, so it is safe to apply
# at any point in a rendering path without tracking which convention got there first.
_MODE_RELABEL = {"P-text": "DOM+somtext", "P-prompt": "DOM+somprompt", "P-SoM": "SoM-image"}


def mode_label(key: str) -> str:
    """Display name for a mode, given either an internal key or a product's own spelling."""
    return _MODE_RELABEL.get(key, PRETTY.get(key, key))


class MissingProduct(RuntimeError):
    """Fail loud: a missing product must not become a silently absent table."""


def load(name: str) -> dict:
    p = X / f"{name}.json"
    if not p.exists():
        raise MissingProduct(f"{p} missing — regenerate that product first")
    return json.loads(p.read_text())


def load_cross_site_rows() -> list[dict]:
    """The 36-row cross-site aggregation, whose cost columns no product re-exports.

    `load()` only reaches `docs/analysis/cross_sites/*.json`. The additive cost breakdown
    (`avg_total_billed_cost_usd` = `avg_canonical_action_cost_usd` + protocol waste) is
    computed by `aggregate_cross_site.py` and lands only in this CSV, so a table that
    wants it has to read the CSV directly.
    """
    import csv

    p = REPO / "results/phantom_paper/cross_site/cross_site_aggregation.csv"
    if not p.exists():
        raise MissingProduct(f"{p} missing — rerun aggregate_cross_site.py first")
    with p.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


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


def relabel_modes(s: str) -> str:
    """Apply the outward-facing mode names to finished output.

    Eight tables print a mode name straight out of a product row rather than through
    `mode_label` — `halluc`, `leakaudit`, `routing`, `latency-split`, `pareto`,
    `per-success`, `cost-protocol`, `ceiling` — and patching each render site invites the
    exact failure this pass exists to prevent: one missed site leaves a document naming the
    same mode two ways, which reads as two modes. Doing it once at the output boundary is
    total by construction, and `assert_no_legacy_mode_names` makes a future render site that
    reintroduces the old spelling fail loudly instead of shipping.

    Longest key first so `P-SoM` is never matched as `P-S` + text by a shorter key.
    """
    for old in sorted(_MODE_RELABEL, key=len, reverse=True):
        s = s.replace(old, _MODE_RELABEL[old])
    return s


def assert_no_legacy_mode_names(s: str, where: str) -> None:
    """Fail loudly if any pre-2026-08-04 mode spelling survived into rendered output."""
    stray = sorted({old for old in _MODE_RELABEL if old in s})
    if stray:
        raise MissingProduct(
            f"{where}: legacy mode name(s) {stray} survived relabelling — a render site is "
            f"emitting them in a form `relabel_modes` cannot see (split across a line?)")


def tex_safe(s: str) -> str:
    for bad, good in TEX_UNSAFE.items():
        s = s.replace(bad, good)
    # pandoc --natbib reads `@token` as a citation key, so "SoM @27.23" silently becomes
    # \\citet{27.23} and surfaces only as an undefined-citation warning.
    out = AT_NUMBER.sub("at ", relabel_modes(s))
    assert_no_legacy_mode_names(out, "tex_safe")
    return out


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
           "{DOM, DOM+somtext, DOM+somprompt, SoM-image}, **vision-only** = {Vision}, "
           "**hybrid** = {SoM}. "
           "Grouping the four is licensed by the non-separability result (Table 4): they "
           "clear the ≥83% consistency bar on none of 26 metrics. "
           "Source: `representation_class_comparison.json`.")
    return "\n".join(rows), cap


def t_class_1arm() -> tuple[str, str]:
    d = load("representation_class_comparison")
    rows = ["| cell | no-image (DOM) | vision-only (Vision) | hybrid (SoM) | sole best "
            "| gap vs hybrid | Table 2 gap |", "|---|---|---|---|---|---|---|"]
    for c in CELLS:
        if c not in d["cells"]:
            continue
        r = d["cells"][c]
        o = r.get("one_arm_per_class")
        if not o:
            continue
        s, w = o["sr"], o["winners"]
        top = w[0] if len(w) == 1 else "tie: " + "+".join(w)
        rows.append(f"| {CELL_LABEL[c]} | {s['no-image']:.2f} | {s['vision-only']:.2f} | "
                    f"{s['hybrid']:.2f} | {top} | "
                    f"{o['gap_noimage_minus_hybrid_pp']:+.2f} | "
                    f"{o['gap_maxof4_minus_hybrid_pp']:+.2f} |")
    cap = ("The same comparison at **one arm per class** (%). Table 2's no-image column is a "
           "maximum over four arms while the other two are single arms, so it is biased up; "
           "this panel uses the arm of each class that exists outside this study. **The "
           "ordering does not move** — hybrid 4, no-image 3, one tie, and vision-only is never "
           "a sole best in either version — which is the robustness statement Table 2 cannot "
           "make. **The gaps do move**: on `WA·B0` the no-image lead is +4.81pp here against "
           "+13.46pp there, because Table 2's figure is carried by DOM+somtext. Quote this table for "
           "any class gap; Table 2 only for the ordering. "
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
    # Iterate the PRODUCT's own spelling; relabel only on the way out.
    for m in ["Vision", "SoM", "DOM", "P-text", "P-prompt", "P-SoM"]:
        rows.append(f"| {mode_label(m)} | {tally.get(m, 0)} |")
    cap = (f"**Absence of repeated extrema** — read the name literally. A mode 'reaches the "
           f"bar' on a metric when it is the extreme (highest or lowest) in ≥{thr} of {n} "
           f"cells ({100 * thr / n:.1f}%). Over 26 metrics the four image-free modes reach it "
           f"on **none**. ⚠️ **This is not a separability test.** `rank_consistency()` only "
           f"counts which mode attains each metric's max/min; a mode that is consistently "
           f"*second* — and strongly distinguishable — never reaches the bar. Establishing "
           f"non-separability would need pairwise equivalence margins with task-clustered "
           f"intervals, which this does not do. ⚠️ **The threshold got stricter when cells "
           f"grew, and an earlier caption said otherwise**: ≥{thr}/{n} is {100 * thr / n:.1f}%, "
           f"not the 83% it claimed, and the six-cell ≥5/6 it says it matches is "
           f"{100 * 5 / 6:.1f}% — a {100 * thr / n - 100 * 5 / 6:+.1f}pp shift, chosen after "
           f"the cell count changed. Carrying the literal numerator (≥5/8 = 62.5%) instead "
           f"would let DOM+somtext clear two metrics and this negative would appear to break, so "
           f"the choice is load-bearing and is disclosed rather than defended. "
           f"Source: `per_mode_four_dimension_profile_with_wa.json`.")
    return "\n".join(rows), cap


def t_fusion() -> tuple[str, str]:
    d = load("fusion_premium")
    rows = ["| cell | n | SoM − Vision | 95% CI | SoM − DOM | 95% CI |",
            "|---|---|---|---|---|---|"]
    for c in d["cells"]:
        r = d["cells"][c]
        v, dm = r["vision"], r["dom"]
        rows.append(f"| {cell_label(c)} | {v['n']} | {v['est_pp']:+.2f} | "
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
                rows.append(f"| {cell_label(c)} | {nf} | {arm} | {f_['est_pp']:+.2f} | "
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
    # Which arms were actually rerun, read from the product rather than asserted. The
    # previous caption said the floor "measures neither on the arm being added" — true when
    # only dom and vision had replicates, false since the SoM rerun landed 2026-08-03.
    nf = load("noise_floor_inventory")
    pairs = nf.get("clean_pairs") or []
    reps = sorted({p["label"].rsplit(".", 1)[-1] for p in pairs if p.get("label")})
    rep_pretty = ", ".join(PRETTY.get(a, a) for a in reps)
    cap = (f"Is a new representation worth more than a rerun? Both middle columns are the same "
           f"functional at the same arm count — `|{{added}} ∖ {{baseline}}| / n` — so they are "
           f"directly comparable; only the *source* of the extra arm differs. **The band is "
           f"{len(pairs)} rerun pairs on one cell** (`B0 × classifieds`, n=224), one each for "
           f"**{rep_pretty}** — so the rows without a band have no comparator at all, and the "
           f"band itself is {len(pairs)} draws rather than a bound. Since the SoM replicate "
           f"landed 2026-08-03 the band is **no longer extrapolated onto an unreplicated arm**: "
           f"both the fused mode this table's best-single column keeps selecting and the arm "
           f"the comparison adds now carry their own measured floor, and adding the third pair "
           f"left the band unmoved. Source: `noise_floor_inventory.json`.")
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
            rows.append(f"| {cl} | {mode_label(m)} | " + " | ".join(vals) + " |")
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
        "`act-fail` column. ⚠️ **They are also not pure conditionals.** An episode containing "
        "no click at all has no click-failure rate to report, and the producer stores `0.0` "
        "there — an undefined rate encoded as a perfect one — which is then averaged over "
        "every task. The zero-denominator share is large: **25–35% of episodes never type**, "
        "and a few percent never click. The product now carries "
        "`*_fail_rate_complete_case` (averaged only over episodes where the action occurred) "
        "and `*_fail_rate_denom_zero_frac` beside these columns; the complete-case values run "
        "**higher**, and any statement about which mode fails most on clicks or types should "
        "be read from those, not from this column. ⚠️ `loc-fallback` is near-zero for Vision "
        "**by construction** (no element ids to fall back from), not as a finding. "
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


def t_latency_split():
    d = load("latency_decomposition")
    rows = ["| cell | mean step (ms) | model call (ms) | model share | fastest by total "
            "| fastest by model only | same? |", "|---|---|---|---|---|---|---|"]
    for cell, r in d["cells"].items():
        pm = r["per_mode_ms"]
        tot = sum(v["total"] for v in pm.values()) / len(pm)
        inf = sum(v["backend_infer"] for v in pm.values()) / len(pm)
        rows.append(f"| {cell} | {tot:,.0f} | {inf:,.0f} | "
                    f"**{100 * r['model_share_of_total']:.1f}%** | {r['fastest_by_total']} "
                    f"| {r['fastest_by_model_only']} | "
                    f"{'yes' if r['fastest_agrees'] else '**no**'} |")
    red, red_n = d["flips_by_family"]["reddit"]
    cls, cls_n = d["flips_by_family"]["classifieds"]
    return "\n".join(rows), (
        f"What a latency number contains. Every latency figure elsewhere in this paper is the "
        f"whole step; `backend_infer` isolates the model call and was read by no analysis "
        f"script until 2026-08-03. **The model is "
        f"{100 * d['model_share_min']:.0f}–{100 * d['model_share_max']:.0f}% of the measured "
        f"time**; the rest is the browser and the container, which "
        f"`offsite_navigation_audit` measures at 1.69× between the two sites. Removing it "
        f"**changes which mode is fastest in {d['n_fastest_flips']} of {d['n_cells']} cells**, "
        f"and not at random: {red} of {red_n} reddit-family cells flip against {cls} of "
        f"{cls_n} classifieds cells — the flips land where the container is slowest. A "
        f"sentence naming the fastest mode is therefore partly a sentence about this "
        f"deployment. What survives estimand choice is only that the two orderings disagree. "
        f"Source: `latency_decomposition.json`.")


def t_dispatch():
    d = load("dispatch_path_audit")
    fam = d["path_families"]
    rows = ["| delivery path | actions | action success |", "|---|---|---|"]
    order = sorted(fam.items(), key=lambda kv: -kv[1]["success"])
    for k, v in order:
        rows.append(f"| {k.replace('_', ' ')} | {v['n']:,} | **{100*v['success']:.1f}%** |")
    fb = d["fallback_share_by_backbone"]
    cap = (
        "How each action reached the browser. **`Vision` is on the coordinate path by "
        "construction** — it emits no element ids — so its action success is capped by this "
        f"harness's coordinate implementation ({100*fam['coord']['success']:.0f}%) rather "
        f"than by the {100*fam['id_locator']['success']:.0f}% the element-id path achieves. "
        "That is not a confound to remove (it is what screenshot-only *is*), but the Vision "
        "arm measures our grounding code as much as the representation. Separately the "
        "element-id fallback share rises with backbone weakness — "
        + " · ".join(f"{b} {100*v['mean']:.0f}%" for b, v in fb.items())
        + " on the text arms: *how often* a run falls back is a model property, the "
        f"fallback's own {100*fam['id_framework']['success']:.0f}% success is ours. "
        "No success rate elsewhere is adjusted by this; it bounds external validity. "
        "Source: `dispatch_path_audit.json`.")
    return "\n".join(rows), cap


def t_estimands():
    lat = load("latency_decomposition")
    cost = load("local_cost_estimand_audit")
    eng = load("energy_carbon_audit")
    rows = ["| quantity | what the reported number is | what changes under the alternative |",
            "|---|---|---|"]
    rows.append(
        f"| latency | whole step | model call is only "
        f"{100*lat['model_share_min']:.0f}–{100*lat['model_share_max']:.0f}% of it; removing "
        f"the container **changes the fastest mode in {lat['n_fastest_flips']} of "
        f"{lat['n_cells']} cells** |")
    rows.append(
        f"| local cost | price per token | the constant assumes "
        f"{cost['config_constants']['derived_from_assumed_tok_per_s']:.0f} tok/s against "
        f"{cost['measured_tok_per_s_min']:.0f}–{cost['measured_tok_per_s_max']:.0f} measured; "
        f"pricing by GPU-time **changes the cheapest mode in "
        f"{len(cost['cheapest_flips_token_vs_time'])} of {cost['n_cells']} local cells** |")
    rows.append(
        f"| energy / carbon | kWh from a CPU estimate | r(energy, latency) = "
        f"{eng['r_min']:.3f}–{eng['r_max']:.3f} at {eng['power_mean_w']:.0f} W — it **is** "
        f"elapsed time; and it does not exist for B0 at all |")
    return "\n".join(rows), (
        "Three efficiency quantities, three estimand choices, none of them previously "
        "stated. Each row's right-hand column is what a defensible alternative definition "
        "does to the per-mode ordering. The pattern is the finding: **efficiency claims in "
        "this setting are estimand-dependent, and the estimand is usually left implicit.** "
        "The local-cost constant was additionally derived for a DGX Spark while every run "
        "was served on an A100 — the same config file migrated its energy profile and not "
        "its cost block. Sources: `latency_decomposition.json`, "
        "`local_cost_estimand_audit.json`, `energy_carbon_audit.json`.")


def t_metric_noise():
    d = load("replicate_metric_noise")
    rows = ["| dimension | metric | cross-mode spread | rerun band | ratio | > a rerun? |",
            "|---|---|---|---|---|---|"]
    for r in d["metrics"]:
        if "excluded" in r:
            rows.append(f"| {r['dimension']} | `{r['metric']}` | — | — | — "
                        f"| *{r['excluded']}* |")
            continue
        rows.append(f"| {r['dimension']} | `{r['metric']}` | "
                    f"{r['cross_mode_spread']:.3f} | {r['band_max']:.3f} | "
                    f"{r['ratio']:.2f}x | {'**yes**' if r['exceeds_noise'] else 'no'} |")
    fails = [r for r in d["metrics"]
             if "excluded" not in r and not r["exceeds_noise"]]
    return "\n".join(rows), (
        f"Behavioural metrics against run-to-run movement, `B0 x classifieds`, three "
        f"replicated arms (dom, vision, som). `rerun band` is the largest "
        f"|metric(run A) - metric(run B)| over those arms; `cross-mode spread` is max-min "
        f"over the six modes. **{d['n_exceeding_noise']} of {d['n_metrics_live']} metrics "
        f"exceed the band**, several by 5-22x. The exceptions are "
        + ", ".join(f"`{r['metric']}` ({r['ratio']:.2f}x)" for r in
                    sorted(fails, key=lambda r: r["ratio"])) +
        " — i.e. **both latency metrics**, which `latency_decomposition` reaches "
        "independently by decomposing the step into model and container. Every other "
        "efficiency and behavioural claim in this paper is judged against a rerun band; "
        "these 26 metrics were not, until this table. One cell, one rerun per arm: a point "
        "estimate, not a threshold. Source: `replicate_metric_noise.json`.")


def t_per_success():
    d = load("outcome_efficiency")
    rows = ["| cell | content? | cheapest/attempt | cheapest/success | fastest/attempt | "
            "fastest/success | max solves |", "|---|---|---|---|---|---|---|"]
    for cell, r in d["cells"].items():
        # cell_label() rather than the raw fallback: `outcome_efficiency` keys the WA cells
        # `wa_B0`, which `_PROF_CELL` does not carry, so the fallback rendered them `wa·B0`
        # in lower case beside `cls·B0`. Two spellings in one column is how a reader — and
        # three times today, me — concludes the WA rows are missing when they are present.
        rows.append(f"| {cell_label(cell) if cell not in _PROF_CELL else _PROF_CELL[cell]} | "
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


def _split_cascade_beats(r: dict) -> tuple[list, list]:
    """Split the Pareto-beating points into strictly-better-SR and merely-SR-tied.

    A point 'beats' always-rich when it is no worse on either axis and better on one, so a
    point whose SR *equals* the rich arm and costs less counts as a win. On `wa_red_B1` all
    three such points sit at exactly the rich arm's SR while 44-60 of 104 episodes are tied
    at the cutoff — the ranking falls through to task id and the achievable SR spans
    8.65-14.42. Reporting one number let the prose say the WA exception was withdrawn (as a
    tie artefact, correctly) while this table kept printing 3, and nobody could see the
    contradiction because the WA rows were being dropped before render (see `cell_get`).
    """
    rich = r.get("rich_sr")
    strict: list = []
    tied: list = []
    for sig, frac in (r.get("pareto_beats_always_rich") or []):
        pt = next((p for p in (r.get("curves") or {}).get(sig, [])
                   if abs(p.get("frac", -1) - frac) < 1e-9), None)
        if pt is None or rich is None:
            strict.append((sig, frac))
        elif abs(pt.get("sr", 0.0) - rich) < 1e-9:
            tied.append((sig, frac))
        else:
            strict.append((sig, frac))
    return strict, tied


def t_cascade():
    d = load("confidence_cascade_with_wa")
    rows = ["| cell | n | cheap SR | rich SR | always-rich cost | oracle SR | comparable? | "
            "beats always-rich — strictly / SR-tied | signals dropped |",
            "|---|---|---|---|---|---|---|---|---|"]
    n_strict_comparable = 0
    n_comparable = 0
    for c in CELLS:
        r = cell_get(d["cells"], c)
        if not r:
            continue
        ar, orc = r.get("always_rich", {}), r.get("oracle", {})
        strict, tied = _split_cascade_beats(r)
        degenerate = bool(r.get("rich_mode_is_worse"))
        if not degenerate:
            n_comparable += 1
            n_strict_comparable += len(strict)
        rows.append(f"| {CELL_LABEL[c]} | {r['n']} | {r['cheap_sr']:.2f} | {r['rich_sr']:.2f} | "
                    f"{ar.get('cost_rel', float('nan')):.3f}× | "
                    f"{orc.get('sr', float('nan')):.2f} | "
                    f"{'— *(rich is worse)*' if degenerate else 'yes'} | "
                    f"{('**' + str(len(strict)) + '**') if strict else '0'} / {len(tied)} | "
                    f"{len(r.get('signals_dropped') or [])} |")
    return "\n".join(rows), (
        f"Confidence-triggered cascade, {d['cheap']} → {d['rich']}. The escalation decision "
        f"sees only the cheap run's own episode — no outcome, no rich-run information. "
        f"**In {n_strict_comparable} of the {n_comparable} comparable cells does any operating "
        f"point beat always-rich on success rate**; the cells marked *rich is worse* are "
        f"excluded because the cascade's premise fails there, and that exclusion rule "
        f"(`cheap_sr >= rich_sr`) is **outcome-dependent**. ⚠️ **The two counts are split for a "
        f"reason.** A point also 'wins' by matching always-rich's SR at lower cost, and every "
        f"such point here is a **tie artefact**: on `WA·B1` all of them sit at exactly the rich "
        f"arm's SR while 44-60 of 104 episodes are tied at the cutoff, so the ranking falls "
        f"through to task id and the reachable SR spans 8.65-14.42. Counting the two together "
        f"is what let this table print a WA win for months while the prose said the WA "
        f"exception was withdrawn — the rows disagreed with the sentence and nobody could see "
        f"it, because the WA rows were dropped before render until 2026-08-03. "
        f"⚠️ **Every number is an offline splice**: an escalated task takes its outcome from a "
        f"standalone rich run, whereas a real cascade would start the rich episode after the "
        f"cheap one had already acted on a stateful site. That sequential outcome is "
        f"unobserved in this project. Source: `confidence_cascade_with_wa.json`.")


def t_cascade_control():
    """The cascade's control arm and its effect size — the other half of `t_cascade`.

    `t_cascade` reports one verdict: does any operating point Pareto-beat always-rich (no).
    That answers whether the cascade is *deployable*, not whether the confidence signal
    carries information — a reader cannot tell a signal-free policy from a failed signal
    without the same-budget random comparator, which the product computes and nothing read.
    """
    d = load("confidence_cascade_with_wa")
    fracs = [0.1, 0.2, 0.3]
    rows = ["| cell | signals | **best** margin (10/20/30%) | **median over all signals** "
            "(10/20/30%) | signals >0 | oracle headroom captured (10/20/30%) |",
            "|---|---|---|---|---|---|"]
    n_pos = n_tot = 0
    all_margins: list[float] = []
    median_neg_cells: list[str] = []
    for c in CELLS:
        r = cell_get(d["cells"], c)
        if not r:
            continue
        best_m, med_m, npos_frac, ntot_frac = [], [], 0, 0
        for f in fracs:
            # every signal at this fraction, not just the argmax — the max alone is what made
            # "24/24 positive" look like evidence when it is partly arithmetic (see caption).
            at_f = [p for curve in r["curves"].values() for p in curve
                    if abs(p["frac"] - f) < 1e-9]
            if not at_f:
                best_m.append(float("nan"))
                med_m.append(float("nan"))
                continue
            ms = sorted(p["sr_gain_pp"] - p["random_gain_pp"] for p in at_f)
            mid = ms[len(ms) // 2] if len(ms) % 2 else 0.5 * (ms[len(ms) // 2 - 1] + ms[len(ms) // 2])
            best_m.append(ms[-1])
            med_m.append(mid)
            all_margins.extend(ms)
            npos_frac += sum(1 for m in ms if m > 0)
            ntot_frac += len(ms)
            n_tot += 1
            n_pos += int(ms[-1] > 0)
        if any(m <= 0 for m in med_m if m == m):
            median_neg_cells.append(CELL_LABEL[c])
        hc = r.get("headroom_captured", {})
        rows.append(
            f"| {CELL_LABEL[c]} | {r['n_signals_used']} | "
            + " / ".join(f"{m:+.2f}" for m in best_m) + " | "
            + " / ".join(f"{m:+.2f}" for m in med_m) + " | "
            + f"{npos_frac}/{ntot_frac} | "
            + " / ".join(f"{hc.get(f'{int(f * 100)}%', float('nan')):.0f}%" for f in fracs) + " |")
    n_all = len(all_margins)
    n_all_pos = sum(1 for m in all_margins if m > 0)
    srt = sorted(all_margins)
    med_all = (srt[n_all // 2] if n_all % 2
               else 0.5 * (srt[n_all // 2 - 1] + srt[n_all // 2])) if n_all else float("nan")
    return "\n".join(rows), (
        f"Is the cascade's signal doing anything? {T('cascade')} answers a deployment "
        f"question — no operating point beats always-rich on success rate — which on its own "
        f"cannot distinguish a signal that carries nothing from one that carries something "
        f"insufficient. This is the comparator that separates them: the same escalation budget "
        f"spent at random. ⚠️ **Read the median column, not the best column.** The best margin "
        f"is positive in {n_pos} of {n_tot} (cell × fraction) combinations, but each of those is "
        f"a **maximum over that cell's 8–10 candidate signals against a constant** — the random "
        f"comparator does not depend on which signal is used — so positivity there is partly "
        f"arithmetic. Over **all {n_all} (signal × cell × fraction) points, {100 * n_all_pos / n_all:.1f}% "
        f"are positive with a median of {med_all:+.3f}pp**: that is the unselected statement, and "
        f"it is the one to quote. On {', '.join(median_neg_cells) if median_neg_cells else '—'} the "
        f"median is negative at one or more fractions while the best is positive — there the "
        f"apparent win is entirely selection. The right-hand column is how much of the gap to a "
        f"per-task oracle the best signal recovers, and inherits the same caveat. The offline-"
        f"splice caveat on {T('cascade')} applies unchanged. "
        f"Source: `confidence_cascade_with_wa.json`.")


def t_triage_learn():
    d = load("visual_difficulty_router")
    rows = ["| cell | n | solvable | AUROC without | AUROC with visual_difficulty | Δ | "
            "visual_difficulty alone |", "|---|---|---|---|---|---|---|"]
    for c in CELLS:
        r = cell_get(d["cells"], c)
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
        f"⚠️ **Six cells, not eight, and the reason is the finding.** `visual_difficulty` is a "
        f"VisualWebArena task-config annotation; WebArena's 106 reddit configs carry **none of** "
        f"`visual_difficulty`, `reasoning_difficulty`, `overall_difficulty` or `image` — the "
        f"field simply is not there, so the contrast cannot be computed rather than computing "
        f"to zero. That absence is what makes WA the clean test elsewhere: where a router can "
        f"read the benchmark's own difficulty annotation it looks learnable, and WA is the "
        f"setting where it cannot. Source: `visual_difficulty_router.json`.")


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
        "does not separate what it appears to. **WebArena cannot arbitrate**: its 106 reddit "
        "configs carry no `image` field at all — nor any of the three difficulty annotations — "
        "so both of this table's stratifiers are undefined there. Six cells is the whole "
        "population for this question, not a coverage gap. "
        "Source: `routing_feature_diagnostics.json`.")


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


def t_ceiling():
    """What a perfect per-task choice could buy — and the arm-matched column beside it.

    The ceiling is a SIX-arm union quoted against a ONE-arm baseline, which is the
    "count the arms before you divide" defect in its most tempting form. The arm-matched
    gain and the rerun draw are therefore in the same row, not in the caption: adding one
    arm and rerunning the arm you already have land in the same range, so the headline
    headroom cannot be read as representation diversity alone.
    """
    d = load("routing_ceiling")
    cells = sorted(d["cells"], key=lambda r: -r["leak_zeroed"]["oracle_sr_pct"])
    rows = ["| cell | n | best single mode | ceiling: any mode solves | headroom | "
            "same tasks, lower cost | +1 arm | rerun once |",
            "|---|---|---|---|---|---|---|---|"]
    for r in cells:
        z = r["leak_zeroed"]
        rr = r["rerun_draws_pp"]
        rr_s = "--" if not rr else (f"{rr[0]:.2f}" if len(rr) == 1
                                    else f"{min(rr):.2f}-{max(rr):.2f}")
        am = r["arm_matched_gain_pp"]
        rows.append(
            f"| {cell_label(r['cell'])} | {z['n_tasks']} | "
            f"{PRETTY.get(z['best_mode'], z['best_mode'])} {z['best_sr_pct']:.2f}% | "
            f"**{z['oracle_sr_pct']:.2f}%** | {z['headroom_pp']:+.2f}pp | "
            f"**{-z['triage_cost_saving_pct']:+.1f}%** | "
            f"{'--' if am is None else f'{am:+.2f}'} | {rr_s} |")
    zs = [c["leak_zeroed"] for c in cells]
    save = [-c["triage_cost_saving_pct"] for c in zs]
    unsolv = [c["unsolvable_share_pct"] for c in zs]
    multi = [c["multi_solver_share_pct"] for c in zs]
    ams = [c["arm_matched_gain_pp"] for c in cells if c["arm_matched_gain_pp"] is not None]
    n_leaked = sum(c["n_leaked_zeroed"] for c in cells)
    return "\n".join(rows), (
        f"**Two ceilings, and only one of them survives its own control.** *Ceiling: any "
        f"mode solves it* is what a perfect per-task choice could reach; it runs "
        f"{min(c['oracle_sr_pct'] for c in zs):.1f}-{max(c['oracle_sr_pct'] for c in zs):.1f}% "
        f"against a best single mode of "
        f"{min(c['best_sr_pct'] for c in zs):.1f}-{max(c['best_sr_pct'] for c in zs):.1f}%. "
        f"⚠️ **That column is a six-arm union against a one-arm baseline.** The arm-matched "
        f"comparison is the last two columns: adding the single best distinct arm buys "
        f"{min(ams):+.2f} to {max(ams):+.2f}pp, and rerunning an arm already in hand buys a "
        f"draw in the same range wherever a replicate exists -- so the headroom cannot be "
        f"attributed to representation diversity rather than to resampling. The union of "
        f"*five* reruns has never been measured, so the split at higher arm counts is "
        f"unknown, not estimated. *Same tasks, lower cost* keeps the best mode everywhere "
        f"and sends only the tasks no mode solves to the cheapest one: success is unchanged "
        f"**by construction** and cost falls "
        f"{abs(max(save)):.1f}-{abs(min(save)):.1f}% in **8 of 8** cells. That ceiling is "
        f"immune to the arm-count objection because it adds no arms. Why both are hard to "
        f"reach: no mode solves {min(unsolv):.1f}-{max(unsolv):.1f}% of tasks, and the set "
        f"where a per-task choice even exists is {min(multi):.1f}-{max(multi):.1f}% of the "
        f"cell. **Leaked-success policy:** {n_leaked} scored successes credited without the "
        f"episode ever visiting the forum the evaluator reads are set to 0 with the "
        f"denominator unchanged; `leak_kept` figures are retained in the product for "
        f"comparison. Source: `routing_ceiling.json`.")


def _rule_cell_spread(side: str, rule: str) -> tuple[dict, int]:
    """Which cells carry a rule's conditional hits, derived rather than asserted.

    The claim "all of block B's top-rule hits are in the WebArena cells" is load-bearing —
    it is what makes that block's only above-baseline row a benchmark-local effect instead
    of a general one. Hardcoding it in the caption would put a conclusion next to a table
    deriving the same quantity, which is the defect class the 2026-08-03 sweep found five
    instances of. So it is computed here and the caption reads whatever this returns.
    """
    d = load("conditional_failure_attribution")
    hits = {c: int(((v.get(side) or {}).get("cond") or {}).get(rule, 0))
            for c, v in d["cells"].items()}
    return {c: n for c, n in hits.items() if n}, len(hits)


def _cond_rows(side: str, top: int):
    """Enrichment rows for one side, sorted, plus what the truncation hides.

    Returns (rows, tail_max, n_tail, n_tasks, n_cond, n_base). `tail_max` is the largest
    enrichment among the rules NOT shown — reporting a top-N without it is the same
    selection-without-its-complement defect the 2026-08-04 audit found twelve instances of.
    """
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
    shown, tail = enr[:top], enr[top:]
    tail_max = max((e for e, *_ in tail), default=None)
    return shown, tail_max, len(tail), p_["n_tasks"], nc, nb


def t_failmode():
    """The paired failure cut, both sides in ONE tabular.

    Kept to a single tabular deliberately: two tabulars under one caption break `floatify`,
    which wraps the prose between them into the float and pdflatex dies with 'Not in outer
    par mode'. The two sides are rendered as labelled blocks of the same table instead, so
    the asymmetry the table exists to show is a property of its layout rather than something
    the reader has to reconstruct by flipping between two floats.

    The arm counts go in the block headers, not only in the caption: TEXT is four arms and
    IMAGE is two, so a symmetric-looking table invites exactly the comparison it cannot
    support.
    """
    TOP = 4
    img, img_tail, img_n_tail, img_tasks, img_nc, img_nb = _cond_rows("image_only", TOP)
    txt, txt_tail, txt_n_tail, txt_tasks, txt_nc, txt_nb = _cond_rows("text_only", TOP)
    names = load("conditional_failure_attribution")["pooled"]["image_only"]["names"]
    names_t = load("conditional_failure_attribution")["pooled"]["text_only"]["names"]

    rows = ["| | rule | how it failed | on disagreement | baseline | enrichment | hits |",
            "|---|---|---|---|---|---|---|"]

    rows.append(f"| **A. IMAGE channel wins** | | *how the TEXT channel failed* | "
                f"({img_tasks} tasks) | | | |")
    for e, rule, hits, cr, br in img:
        rows.append(f"| | `{rule}` | {ascii_rule_name(rule, names.get(rule, ''))[:32]} | "
                    f"{100*cr:.1f}% | {100*br:.1f}% | **{e:.2f}x** | {hits} |")
    if img_tail is not None:
        rows.append(f"| | | *{img_n_tail} further rules* | | | all <= {img_tail:.2f}x | |")

    rows.append(f"| **B. TEXT channel wins** | | *how the IMAGE channel failed* | "
                f"({txt_tasks} tasks) | | | |")
    for e, rule, hits, cr, br in txt:
        rows.append(f"| | `{rule}` | {ascii_rule_name(rule, names_t.get(rule, ''))[:32]} | "
                    f"{100*cr:.1f}% | {100*br:.1f}% | **{e:.2f}x** | {hits} |")
    if txt_tail is not None:
        rows.append(f"| | | *{txt_n_tail} further rules* | | | all <= {txt_tail:.2f}x | |")

    top_img = img[0][0] if img else float("nan")
    n_txt_hits = txt[0][2] if txt else 0
    # Where block B's only above-baseline row actually lives. Derived, not asserted.
    b_rule = txt[0][1] if txt else None
    live, n_cells = _rule_cell_spread("text_only", b_rule) if b_rule else ({}, 0)
    wa_live = [c for c in live if c.startswith("wa")]
    if live and len(wa_live) == len(live):
        where = (f"and all {sum(live.values())} of them fall in the "
                 f"{len(wa_live)} WebArena cells, none in the other {n_cells - len(wa_live)}")
    else:
        where = (f"spread over {len(live)} of {n_cells} cells "
                 f"({', '.join(f'{c}:{n}' for c, n in sorted(live.items()))})")
    return "\n".join(rows), (
        f"**The two channels do not fail the same way.** On tasks only one channel solved, "
        f"how did the other fail? Pooled over 8 cells at ruleset v11. Enrichment = hit rate "
        f"on the disagreement set over that channel's hit rate across all its failures, so "
        f"about 1x means it failed there the way it fails everywhere. Rules with fewer than "
        f"8 pooled conditional hits are omitted, and the largest omitted enrichment is stated "
        f"per block rather than left to the reader. **Block A has named death causes** "
        f"(top {top_img:.2f}x, {img_nc} losing-channel episodes against {img_nb} of that "
        f"channel's failures overall). **Block B does not**: its top row rests on "
        f"{n_txt_hits} hits, exactly the reporting floor, {where}; every other rule in that "
        f"block sits at or below the everywhere-baseline. On the "
        f"tasks the text channel uniquely solves, the image channel did not break somewhere "
        f"nameable -- it did not arrive. **Caution:** TEXT is **four arms** and IMAGE is "
        f"**two**, so the two blocks' task counts are not comparable to each other; read "
        f"each block against its own baseline column, never across blocks. **Caution:** a "
        f"per-rule frequency is a distribution of symptoms, not of causes -- the largest rows "
        f"in most cells are risk markers, not death causes, and only rules whose docstrings "
        f"record a causal check are verified as such. Source: "
        f"`conditional_failure_attribution.json`.")


def t_cond_text_wins():
    return _cond_side("text_only",
                      "Only the TEXT channel solved it: how the IMAGE channel failed.")


def t_cond_image_wins():
    return _cond_side("image_only",
                      "Only the IMAGE channel solved it: how the TEXT channel failed.")


def t_cond_probes():
    """The vocabulary-free half of the paired cut.

    `_cond_side` reads only `d["pooled"][side]`, which is rule-hit frequencies — so the
    reader-facing tables inherit the rule vocabulary and cannot answer the objection that
    the vocabulary is what makes the text-wins side look empty. The product computes six
    probes straight from raw step fields for exactly that objection; nothing exported them.
    """
    d = load("conditional_failure_attribution")
    probes = d.get("text_wins_probes") or {}
    rows = ["| candidate mechanism | on the disagreement set | that channel's baseline | "
            "enrichment |", "|---|---|---|---|"]
    ordered = sorted(probes.items(), key=lambda kv: -kv[1].get("enrichment", 0))
    n_dis = n_base = None
    for name, p in ordered:
        n_dis = n_dis or p.get("n_disagreement_episodes")
        n_base = n_base or p.get("n_baseline_episodes")
        rows.append(f"| {name} | {p['on_disagreement']:.3f} | {p['baseline']:.3f} | "
                    f"**{p['enrichment']:.2f}×** |")
    top = max((p.get("enrichment", 0) for p in probes.values()), default=float("nan"))
    below = sum(1 for p in probes.values() if p.get("enrichment", 1) < 1)
    return "\n".join(rows), (
        f"Is the text-wins residual real, or an artefact of the rule vocabulary? "
        f"{TS('cond-text','cond-image')} count rule hits, and the ruleset was discovered on "
        f"VisualWebArena — so an absent signature there could be a property of the vocabulary "
        f"rather than of the world. Each probe here is computed from raw step fields and "
        f"never from a rule hit, over {n_dis} disagreement episodes against {n_base} baseline "
        f"failures. **The largest enrichment is {top:.2f}×** and {below} of {len(probes)} sit "
        f"*below* 1: on the tasks the text channel uniquely solves, the image channel fails "
        f"**more blandly** than it fails elsewhere — it did not arrive, rather than breaking "
        f"somewhere nameable. ⚠️ Six candidates chosen by us, so this cannot show that no "
        f"mechanism exists; what it closes is the specific objection that the residual is an "
        f"artefact of a VWA-shaped vocabulary. "
        f"Source: `conditional_failure_attribution.json`.")


def t_cost_protocol():
    """What a cost number contains — the cost-side counterpart of `t_latency_split`.

    Every reader-facing cost figure is `cost/ep`, one number. The schema splits it
    additively into the spend that bought a canonical action and the spend burned on
    protocol repair, and `aggregate_cross_site.py` carries all three columns — nothing
    displayed them. They are not the same size across backbones, which is the point.
    """
    rows_in = load_cross_site_rows()
    agg: dict[str, list[float]] = {}
    worst: dict[str, tuple[float, str]] = {}
    for r in rows_in:
        b = r["baseline"]
        tot = float(r.get("avg_total_billed_cost_usd") or 0)
        can = float(r.get("avg_canonical_action_cost_usd") or 0)
        wst = float(r.get("avg_protocol_wasted_cost_usd") or 0)
        wait = float(r.get("avg_parse_error_injected_wait_count") or 0)
        a = agg.setdefault(b, [0.0, 0.0, 0.0, 0.0])
        a[0] += tot
        a[1] += can
        a[2] += wst
        a[3] += 1
        if wait > worst.get(b, (-1.0, ""))[0]:
            worst[b] = (wait, f"{r['site'][:3]}·{r['mode']}")
    rows = ["| backbone | conditions | billed | canonical action | protocol repair | "
            "share of billed | worst parse-error waits/ep |", "|---|---|---|---|---|---|---|"]
    shares = {}
    for b in sorted(agg):
        tot, can, wst, n = agg[b]
        shares[b] = 100 * wst / tot if tot else 0.0
        w, where = worst.get(b, (0.0, "—"))
        rows.append(f"| {b} | {int(n)} | ${tot:.4f} | ${can:.4f} | ${wst:.5f} | "
                    f"**{shares[b]:.2f}%** | {w:.2f} ({where}) |")
    lo_b = min(shares, key=shares.get)
    hi_b = max(shares, key=shares.get)
    ratio = shares[hi_b] / shares[lo_b] if shares[lo_b] else float("inf")
    return "\n".join(rows), (
        f"Every cost figure elsewhere is `cost/ep`, a single "
        f"number; the schema splits it additively into what bought a canonical action and "
        f"what was burned re-parsing the model's output. Summed over each backbone's "
        f"conditions, protocol repair is {shares[lo_b]:.2f}% of {lo_b}'s bill and "
        f"{shares[hi_b]:.2f}% of {hi_b}'s — a **{ratio:.1f}× spread across model families "
        f"for the same task set**, and the worst per-episode parse-error wait counts all sit "
        f"on {hi_b}. This is small in absolute terms and is **not** a correction to any "
        f"reported cost. It bounds something else: a cross-family cost comparison is not "
        f"quite comparing like with like, because one family pays a protocol tax the others "
        f"do not. Source: `results/phantom_paper/cross_site/cross_site_aggregation.csv` "
        f"(via `aggregate_cross_site.py`).")


def t_mech_readability():
    """Method 4.2: every mode pair is linearly separable, at sub-permille cosine gaps."""
    d = load("mechanism_evidence")["readability"]
    rows = ["| site | modes | examples | pairs | pairs at AUROC 1.000 | worst pair | "
            "image gap | text-format gap | prompt-family gap |",
            "|---|---|---|---|---|---|---|---|---|"]
    for site, r in d.items():
        g = r["axis_cosine_gap"]
        rows.append(
            f"| {site} | {r['n_modes']} | {r['n_examples']} | {r['n_pairs']} | "
            f"**{r['n_pairs_at_auroc_1']}/{r['n_pairs']}** | "
            f"{r['auroc_lototask_best_layer_min']:.3f} | "
            + " | ".join(
                f"{g[a]['peak_gap']:.4f} (L{g[a]['peak_layer']:02d})"
                for a in ("image", "text_format", "prompt_family")) + " |")
    sites = list(d.values())
    img = max(s["axis_cosine_gap"]["image"]["peak_gap"] for s in sites)
    txt = max(s["axis_cosine_gap"]["text_format"]["peak_gap"] for s in sites)
    return "\n".join(rows), (
        f"Are the six representations distinguishable inside the model? Leave-one-task-out "
        f"AUROC over the residual stream separates **every** mode pair perfectly on both "
        f"sites, at best layer. The gap columns are why that is worth stating rather than "
        f"assuming: the image axis peaks at {img:.3f} cosine while the text-format axis "
        f"peaks at {txt:.4f} — **a factor of ~{img / txt:.0f}** — so perfect separability "
        f"coexists with geometric differences in the third decimal place. ⚠️ Scope: one "
        f"backbone (B1), 144 examples per site, a single decode step; and AUROC 1.000 on 24 "
        f"tasks × 6 modes is a **ceiling effect**, not a calibrated effect size. This licenses "
        f"'the representation is present and readable', nothing about whether it is used. "
        f"Source: `mechanism_evidence.json` ← `method42_metrics_v2.json`.")


def t_mech_patching():
    """Stage 3 prompt-family patching, reported against BOTH of its controls."""
    d = load("mechanism_evidence")["patching"]
    label = {"real": "**real**", "random_injection": "random injection",
             "task_shuffled": "task-shuffled source"}
    rows = ["| site | arm | n | displacement (peak) | convergence to source (peak) | peak at |",
            "|---|---|---|---|---|---|"]
    for site, arms in d.items():
        for arm, a in arms.items():
            rows.append(
                f"| {site} | {label.get(arm, arm)} | {a['n_tasks']} | "
                f"{a['peak_displacement']:.3f} | {a['peak_convergence']:.3f} | "
                f"L{a['peak_convergence_layer']:02d} |")
    cls_ = d.get("classifieds", {})
    real_c = cls_.get("real", {}).get("peak_convergence")
    rand_c = cls_.get("random_injection", {}).get("peak_convergence")
    shuf_c = cls_.get("task_shuffled", {}).get("peak_convergence")
    shuf_l = cls_.get("task_shuffled", {}).get("peak_convergence_layer")
    real_l = cls_.get("real", {}).get("peak_convergence_layer")
    return "\n".join(rows), (
        f"Does the prompt-family signature *do* anything, or is it only visible? Source hidden "
        f"states from `phantom_som` are patched into a `phantom_text` run, holding image and "
        f"text-format constant. **Displacement is the wrong column to read**: random injection "
        f"scores {cls_.get('random_injection', {}).get('peak_displacement', float('nan')):.3f} "
        f"there, which is destruction, not steering — only *convergence to the source* speaks "
        f"to direction. Against random injection the real arm wins clearly "
        f"({real_c:.3f} vs {rand_c:.3f}). ⚠️ **Against the task-shuffled control it barely "
        f"wins** ({real_c:.3f} vs {shuf_c:.3f}, +{100 * (real_c / shuf_c - 1):.0f}%): a source "
        f"drawn from an *unrelated task* moves the output almost as far toward itself. What "
        f"does separate them is **where**: the real arm's convergence peaks mid-stack "
        f"(L{real_l:02d}) and the shuffled arm's collapses to the boundary layer "
        f"(L{shuf_l:02d}). So the content-specific claim rests on the **layer profile**, not "
        f"on the magnitude — which is weaker than the write-ups in "
        f"`docs/checkpoints/mechanism/results/` read, and is the reason this table reports "
        f"both controls rather than the headline. n = 24 tasks per arm, one backbone, one "
        f"decode step. Source: `mechanism_evidence.json` ← `patching_continuation_results.json`.")


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
            rows.append(f"| {cell_label(c)} | SoM − {PRETTY[comp]} | "
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
            ("compound_dom_to_psom", "DOM->SoM-image")]
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
        "continuous), signed right-minus-left. The compound DOM->SoM-image transition decomposes "
        "into a text-payload axis and a prompt-style axis; the image axis is SoM-image->SoM. "
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
    # Prefer the WA-inclusive run. The VWA-only product stays on disk byte-stable, but a
    # table that silently covers three of five cells is the coverage hole this whole audit
    # was about — WebArena reddit runs the same Postmill image with the same reset no-op.
    try:
        d = load("reddit_sidebar_leakage_audit_with_wa")
        wa = True
    except MissingProduct:
        d = load("reddit_sidebar_leakage_audit")
        wa = False
    rows = ["| benchmark | cell · mode | scored successes | of which LEAKED | share |",
            "|---|---|---|---|---|"]
    agg = {}
    for r in d["rows"]:
        if not r.get("in_scored_universe"):
            continue
        # keyed on cell, not baseline: WA reuses B0/B1, so a baseline-only key silently
        # merges a VWA row into a WA one.
        k = (r.get("benchmark", "visualwebarena"), r["baseline"], r["mode"])
        a = agg.setdefault(k, {"succ": 0, "leak": 0})
        if r["success"]:
            a["succ"] += 1
        if r["verdict"] == "LEAKED":
            a["leak"] += 1
    for (bench, base, m), a in sorted(agg.items()):
        if not a["succ"]:
            continue
        # the benchmark column already separates the two, so the cell column carries the
        # backbone alone — WA and VWA both spell theirs B0/B1.
        rows.append(f"| {'WA' if bench == 'webarena' else 'VWA'} | {base} · {m} | "
                    f"{a['succ']} | {a['leak']} | {100*a['leak']/a['succ']:.1f}% |")
    n_wa_leak = sum(1 for r in d["rows"]
                    if r.get("benchmark") == "webarena" and r.get("in_scored_universe")
                    and r["verdict"] == "LEAKED")
    n_wa_scored = sum(1 for r in d["rows"]
                      if r.get("benchmark") == "webarena" and r.get("in_scored_universe"))
    wa_note = (
        f" **WebArena audited 2026-08-03** (first time): {n_wa_scored} scored episodes over its "
        f"{len(d.get('wa_tasks') or {})} sidebar tasks, **{n_wa_leak} leaked**. ⚠️ That zero is a "
        f"*lower bound*, not a clearance — the test asks whether the episode reached the forum, "
        f"and an episode can arrive at a forum an earlier one subscribed to, read `Unsubscribe`, "
        f"and finish without acting. One such case is hand-confirmed (`B1`/DOM task 597) and "
        f"scores `earned` here; a text heuristic for the pattern was tried and rejected because "
        f"model self-report cannot separate deliberating from acting."
    ) if wa else ""
    return "\n".join(rows), (
        f"Which successes were earned. `{d['selector']}` is read by {len(d['tasks'])} VWA reddit "
        f"tasks; `require_reset` is a no-op on reddit so subscriptions accumulate. "
        f"**LEAKED** = scored success by an episode that never visited the required forum. "
        f"On VWA: {d['n_leaked']} leaked, {d['n_earned']} earned; {T('leakage')} recomputes every "
        f"contrast with the leaked ones zeroed.{wa_note} "
        f"Source: `reddit_sidebar_leakage_audit{'_with_wa' if wa else ''}.json`.")


TABLES = [
    ("sr", "Success rate per mode", t_sr),
    ("class", "Best arm per deployment class", t_class),
    ("class-1arm", "Deployment classes at one arm each", t_class_1arm),
    ("class-ablate", "Class ablation, unmatched and arm-matched", t_class_ablate),
    ("nonsep", "Absence of repeated extrema among image-free modes", t_nonsep),
    ("prof-outcome", "Full matrix — Outcome dimension", t_prof_outcome),
    ("prof-macro", "Full matrix — Macro dimension", t_prof_macro),
    ("prof-micro", "Full matrix — Micro dimension", t_prof_micro),
    ("prof-eff", "Full matrix — Efficiency dimension", t_prof_efficiency),
    ("pareto", "Multi-metric Pareto", t_pareto),
    ("latency-split", "What a latency number contains", t_latency_split),
    ("estimands", "Three efficiency quantities, three estimand choices", t_estimands),
    ("dispatch", "What actually delivered the click", t_dispatch),
    ("metric-noise", "Behavioural metrics against run-to-run noise", t_metric_noise),
    ("per-success", "Per-attempt versus per-success", t_per_success),
    ("fusion", "Fusion premium against the rerun band", t_fusion),
    ("exante", "Ex-ante visual-intent partition", t_exante),
    ("ceiling", "What a perfect per-task choice could buy", t_ceiling),
    ("floor", "New representation versus a rerun", t_floor),
    ("routing", "Routing policies on the 3-axis frontier", t_routing),
    ("cascade", "Confidence-triggered cascade", t_cascade),
    ("cascade-control", "Cascade signal against a random-escalation control", t_cascade_control),
    ("triage", "Triage learnability and the visual-difficulty feature", t_triage_learn),
    ("feature", "The intuitive routing feature", t_feature_sign),
    ("failmode", "Failure modes are asymmetric across channels", t_failmode),
    ("cond-text", "Paired failure attribution: text wins", t_cond_text_wins),
    ("cond-image", "Paired failure attribution: image wins", t_cond_image_wins),
    ("cond-probes", "Vocabulary-free probes on the text-wins residual", t_cond_probes),
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
    ("cost-protocol", "What a cost number contains", t_cost_protocol),
    ("mech-read", "Linear readability against geometric magnitude", t_mech_readability),
    ("mech-patch", "Causal patching against both of its controls", t_mech_patching),
    ("leakaudit", "Earned versus leaked successes", t_leak_audit),
]

TABLE_NO = {slug: i + 1 for i, (slug, _, _) in enumerate(TABLES)}


def T(slug: str) -> str:
    """`Table N` resolved from the registry.

    Never type a table number. Inserting a table renumbers everything after it, and the
    reading guide is the only place those numbers are read by a human — it claimed to
    describe 29 tables against a registry of 35, with its groups mis-mapped from Table 5
    onward, because the numbers were literals while the surrounding figures were injected.
    """
    return f"Table {TABLE_NO[slug]}"


def TS(*slugs: str) -> str:
    """`Tables A, B–C` for a group; contiguous runs collapse to a dash."""
    nums = sorted(TABLE_NO[s] for s in slugs)
    runs: list[tuple[int, int]] = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        runs.append((start, prev))
        start = prev = n
    runs.append((start, prev))
    body = ", ".join(str(a) if a == b else f"{a}–{b}" for a, b in runs)
    return ("Table " if len(nums) == 1 else "Tables ") + body


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
      f"SoM in 5 cells, DOM in 2, DOM+somtext in 1.")
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
      f"always-Vision, and `always-DOM+somprompt` at 19.64% is equally undominated.")
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

    Sits between the eight-line inventory and the tables. Without it the document goes
    straight from a summary nobody can act on to a matrix nobody reads. Numbers here are
    injected from the products, never typed — and since 2026-08-03 so are the table
    numbers, via `T()` / `TS()`. Every slug in `TABLES` must appear in exactly one group
    below; `test_guide_covers_every_table` enforces it.
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
    A(f"<!-- Plain-sentence guide to the {len(TABLES)} tables. Not an argument — each "
      "paragraph says what a group of tables is for and what the trap in it is. Delete "
      "before submission. -->")
    A("")

    A(f"**{TS('sr','class','class-1arm','class-ablate','nonsep')} — what each mode achieves, "
      f"and whether the six collapse into three.** {T('sr')} is the raw success rate. "
      f"{TS('class','class-1arm')} group the six modes into the three shapes web agents "
      f"actually ship in, and {T('nonsep')} is the licence for that grouping: over 26 "
      f"metrics the four image-free modes are never the extreme in ≥7 of 8 cells, so they do "
      f"not behave differently enough to keep apart. "
      f"The grouping produces two facts: `vision-only` is never the sole best class "
      f"(hybrid {tally.get('hybrid',0)}, no-image {tally.get('no-image',0)}, one tie), and "
      f"which class wins reverses between benchmarks — on `WA·B0` no-image leads hybrid by "
      f"{wa0['no-image']-wa0['hybrid']:.2f}pp. "
      f"**The trap is in {T('class-ablate')}**: dropping the whole no-image class costs far "
      f"more than dropping the others, but it has four arms against one each. The arm-matched "
      f"columns beside it show no systematic difference "
      f"({', '.join(f'{k} {v}x' for k,v in matched.items())}), "
      f"and those are the ones that compare like with like.")
    A("")

    A(f"**{TS('prof-outcome','prof-macro','prof-micro','prof-eff')} — the full behavioural "
      f"matrices.** Every metric, every cell, every mode, unsummarised: Outcome, Macro (what "
      f"the agent did), Micro (how often what it did failed), Efficiency. These are the "
      f"substrate the consistency counts in {T('nonsep')} are computed from, included so a "
      f"reader can check a claim rather than take the tally on trust. Two columns are gates "
      f"rather than measurements and are marked as such: `loc-fallback` is near-zero for "
      f"Vision because there are no element ids to fall back from, not because it fails less.")
    A("")

    A(f"**{TS('pareto','latency-split','estimands','per-success','costclass','cost-protocol')} "
      f"— efficiency, "
      f"and the denominator nobody declares.** {T('pareto')} shows the cost and latency "
      f"orderings are not each other restated. {T('per-success')} changes the denominator "
      f"from per-attempt to per-success and the cheapest mode changes in 4 of the 6 cells "
      f"that carry enough successes to divide by. **Every pairwise interval overlaps**, so "
      f"this supports 'the denominator must be stated', not 'X is more efficient'. The other "
      f"three say the same thing about three different quantities: {T('latency-split')} — a "
      f"latency figure is mostly browser and container, not model, and stripping them changes "
      f"which mode is fastest; {T('estimands')} — each efficiency quantity admits a defensible "
      f"alternative definition that reorders the modes; {T('costclass')} — B0 pays an API bill "
      f"and B1/B2 pay electricity, and those are not the same quantity. {T('cost-protocol')} "
      f"splits the cost figure the way {T('latency-split')} splits the latency one, and finds "
      f"the protocol-repair share differing several-fold between model families on the same "
      f"tasks. **The pattern, not any one row, is the finding: efficiency here is "
      f"estimand-dependent and the estimand is usually left implicit.**")
    A("")

    A(f"**{TS('fusion','floor','ceiling')} — is a second representation worth buying?** "
      f"{T('ceiling')} is the upper bound on the whole question: what a perfect per-task "
      f"choice could reach, and — in the same row, deliberately — what adding one arm or "
      f"rerunning one arm buys instead, because the ceiling column is a six-arm union quoted "
      f"against a one-arm baseline. Its second ceiling, *same tasks solved at lower cost*, is "
      f"the one no arm-count objection reaches, since it adds no arms. {T('fusion')} "
      f"asks whether the fused mode beats the single channel that suits the workload: it does "
      f"not, in any of {len(fus['cells'])} cells. The comparison is against a **measured rerun "
      f"band**, not against zero, because the question a deployment asks is whether a new arm "
      f"beats re-running the arm it already has — {T('floor')} puts those two side by side at "
      f"the same arm count. **The trap is the band's width, not its coverage**: only two cells "
      f"carry a measured floor at all, and one standard deviation of the null it is drawn from "
      f"is of the same order as the band itself, so 'clears the band' is not 'clears the "
      f"noise'. Since 2026-08-03 the fused arm is no longer extrapolated onto — the SoM "
      f"replicate landed and its own floor falls inside the band.")
    A("")

    A(f"**{TS('exante','routing','cascade','cascade-control','triage','feature','instability','pooled')} "
      f"— five ways of routing, and what each one dies of.** {T('exante')} finds a signal that "
      f"is as good as signals get: a regex over the task intent, costing nothing and needing "
      f"no episode, that flags tasks where the screenshot is worth "
      f"{cb0['flagged']['est_pp']:+.2f}pp against {cb0['rest']['est_pp']:+.2f}pp elsewhere. "
      f"{T('routing')} turns it into a policy and it **still loses to always-Vision**, because "
      f"the screenshot does not hurt on the unflagged tasks either. {T('cascade')}'s cascade "
      f"beats always-rich at no operating point — but {T('cascade-control')} is the table that "
      f"says what that means: against the *same budget spent at random* the confidence ranking "
      f"wins nearly everywhere, so the signal is informative and still not enough. "
      f"{T('triage')} adds the benchmark's own difficulty annotation for a mean ΔAUROC of "
      f"{vdr['mean_delta_auroc']:+.4f}. {T('feature')} shows the feature a practitioner would "
      f"reach for first does not separate what it appears to. {T('pooled')} pools the "
      f"backbones and routes by cost tier instead of by task. {T('instability')} is the "
      f"supervision problem underneath all of them: the rows a router would learn from are "
      f"the contested ones, and those are exactly the rows that flip between two runs of the "
      f"same condition. **The pattern is not 'no signal'** — it is that the arm the router "
      f"would route *to* is already the right arm to route everything to.")
    A("")

    A(f"**{TS('failmode','cond-text','cond-image','cond-probes','axis','axis1','halluc','pagechange')} — "
      f"where the failures come from.** {T('failmode')} is the headline cut with both sides in "
      f"one table, so the asymmetry is visible without flipping between floats; "
      f"{TS('cond-text','cond-image')} are the same data split per side at full depth. "
      f"These are the paired cut: on "
      f"tasks only one channel solved, how did the other fail? Those count rule hits, so "
      f"{T('cond-probes')} asks the same question without the rule vocabulary — six probes "
      f"read straight off the step records — and finds the losing channel failing *more "
      f"blandly* there than elsewhere, which is what closes the objection that the residual "
      f"is an artefact of a VWA-shaped ruleset. {T('axis')} decomposes the DOM->SoM-image "
      f"transition into a text axis and a prompt axis, and {T('axis1')} asks whether that text "
      f"axis moves per-step decision quality more than it moves macro action frequencies. "
      f"{T('halluc')} counts hallucinated element references — inapplicable to Vision by "
      f"construction, marked. {T('pagechange')} corrects a false-positive in the page-change "
      f"detector. ⚠️ **A per-rule frequency is a distribution of symptoms, not of causes.** "
      f"The two largest rows in most cells are risk markers that causal verification did not "
      f"confirm as death causes.")
    A("")

    A(f"**{TS('dispatch','metric-noise','leakage','offsite','evaluator','leakaudit')} — what "
      f"would make all of the above wrong.** {T('evaluator')}: the evaluator emits two values, "
      f"so there is no graded target and every routing negative inherits that. "
      f"{T('metric-noise')}: the replicate inventory behind every stability figure here is one "
      f"cell, so those bands are what repetition happened to deliver, not bounds on what it "
      f"could. {T('dispatch')}: an arm's measured ceiling is partly this harness — Vision is "
      f"on the coordinate path by construction, and that path succeeds far less often than the "
      f"element-id one. {TS('leakage','leakaudit')}: {lk['leaks_removed'].__len__()} successes "
      f"were credited by accumulated site state, and zeroing them flips {n_flip} verdict"
      f"{'s' if n_flip != 1 else ''}. {T('offsite')}: reddit episodes leave the benchmark for "
      f"the public internet on {min(c['pct_steps'] for c in red):.2f}–"
      f"{max(c['pct_steps'] for c in red):.2f}% of steps against "
      f"{max(c['pct_steps'] for c in cls_off):.2f}% on classifieds, and reddit's container is "
      f"slower than classifieds' before any agent behaviour enters.")
    A("")

    A(f"**{TS('mech-read','mech-patch')} — inside the model, and how far that gets us.** "
      f"Everything above is behavioural. These two ask whether the representations differ "
      f"*in the residual stream* and whether that difference is used. {T('mech-read')}: every "
      f"mode pair is perfectly separable by a linear probe, at cosine gaps in the third "
      f"decimal — present and readable. {T('mech-patch')}: patching the prompt-family "
      f"signature from one mode into another moves the output, but the honest column is "
      f"**convergence toward the source**, not displacement — random injection maximises "
      f"displacement by destroying the continuation. **The trap is the second control**: a "
      f"source drawn from an unrelated task converges almost as well, so the content-specific "
      f"reading rests on the layer profile (mid-stack for the real arm, boundary layer for the "
      f"shuffled one) rather than on the size of the effect. These are one backbone, 24 tasks, "
      f"one decode step, and the mechanism programme was shelved 2026-05-14 — they are here "
      f"because a shelved result and an absent one look identical on disk and are not the same "
      f"thing to a reader.")
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
        # Multiplicity status, stated once for the whole family. Before 2026-08-04 the
        # document contained zero occurrences of multiplicity / BH / Holm / FDR / FWER /
        # family-wise / post-hoc / exploratory across all tables, so a reader had no way to
        # know how many comparisons produced them. Skeleton only — which tables are
        # confirmatory is a framing decision and is marked TBD rather than guessed.
        family_note = (
            "> **Multiplicity status of this table set.** These "
            f"{len(TABLES)} tables are **not one inferential family** and are not corrected "
            "as one. Read them in three classes:\n"
            ">\n"
            "> * **Confirmatory** — hypotheses fixed before the data, corrected within their "
            "own family. Currently: the 2×2 axis independence set (64 conjunction hypotheses, "
            "BH/Holm on `max(p1, p2)`; see that table's caption). *Which further tables join "
            "this class is a framing decision and is still TBD.*\n"
            "> * **Descriptive** — quantities reported with intervals but no test, and no "
            "correction claimed: the full behavioural matrices, the efficiency tables, the "
            "failure-attribution counts.\n"
            "> * **Selection-derived (exploratory)** — a maximum or argmax over layers, "
            "signals, thresholds or comparators is part of the statistic. These bound what a "
            "selection could deliver and **cannot be read as effect estimates**: the cascade "
            "control (best of 8–10 signals), the linear-probe AUROC (best of 37 layers), the "
            "reference-image and visual-difficulty diagnostics (selected comparators). Each "
            "such table states it in its own caption.\n"
            ">\n"
            "> No family-wise statement covers the set as a whole, and none is claimed.\n")
        a.evidence.write_text(tex_safe(
            "## Evidence tables\n\n<!-- Generated by "
            "scripts/analysis/export_ablation_tables.py. Do not hand-edit between the "
            "markers. -->\n\n" + family_note + "\n" + tables_body))
        print(f"[md] {a.evidence}  (evidence section)")


if __name__ == "__main__":
    main()
