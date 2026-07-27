#!/usr/bin/env python3
"""Cross-mode failure-signature aggregation over the 36 landed Phase-1a conditions.

Closes two traceability gaps found by the 2026-07-27 three-AI stress round:

  #17 (Mode A A-4 sibling) — paper A §4.3 claims "a deterministic failure-pattern
       scan over all 36 landed conditions, aggregated cross-mode", and that the
       top failure signatures are mode-invariant. `diag_rescan_all.py` produces the
       36 per-condition scans and asserts version consistency, but nothing wrote
       the aggregate. Part A below is that aggregate.

  #3 / A-5 — paper A §4.2's hallucinated-reference rates (0.04% / 0.39% / ...) lived
       only in a hand-written prose string at `write_digests.py:170`, sourced from a
       one-off measurement (笔记 §387.12). Part B recomputes them from step records
       and, because the 2x2 was run, reports the *decomposition* rather than a single
       compound contrast. The old §4.2 attributed a DOM-vs-P-SoM (both-knobs) ratio to
       the text knob alone; the decomposition shows the effect is an interaction.

Denominator note (Part B). The original measurement is a rate over **action-steps**
(click / type / select_option) across **all** episodes. `check_p44` in
diag_pattern_match.py returns early on successful episodes, so the rule's own hit
counts are the failed-episode view of the same signal; both denominators are emitted
here so the choice is visible rather than implied.

Usage:
  python3 scripts/analysis/aggregate_cross_mode_failure_signatures.py
  python3 scripts/analysis/aggregate_cross_mode_failure_signatures.py --scan-dir /tmp/diag_v8

Output:
  docs/analysis/cross_sites/cross_mode_failure_signatures.json
  docs/analysis/cross_sites/cross_mode_failure_signatures.md
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from diag_pattern_match import (  # noqa: E402
    MISSING_UNION_BOUND_RE, _action_type, _discover_episodes, _load_steps,
    _locator_errors,
)
from diag_rescan_all import CANONICAL, _discover_cls, scan  # noqa: E402

sys.path.insert(0, str(ROOT))
from scripts.analysis.lib.canonical_task_universe import restrict_to_scored  # noqa: E402

PHASE1 = ROOT / "results" / "visualwebarena" / "phase1"
OUT_DIR = ROOT / "docs" / "analysis" / "cross_sites"

# Display order. `vision` carries no element-id list at all, so the id-space rules
# (P43/P44/P45) are structurally inapplicable there — kept in the table, excluded
# from the "text-bearing" spread column that the §4.3 claim rests on.
MODES = ["dom", "som", "vision", "phantom_text", "phantom_prompt", "phantom_som"]
TEXT_BEARING = [m for m in MODES if m != "vision"]
ACTION_TYPES = ("click", "type", "select_option")

# The phantom quadrant: (text payload, prompt family). This is the 2x2 whose
# interaction Part B measures.
QUADRANT = {
    "dom": ("AXTree", "DOM"),
    "phantom_text": ("legend", "DOM"),
    "phantom_prompt": ("AXTree", "SoM"),
    "phantom_som": ("legend", "SoM"),
}
PAPER_NAME = {
    "dom": "DOM", "som": "SoM", "vision": "Vision",
    "phantom_text": "P-text", "phantom_prompt": "P-prompt", "phantom_som": "P-SoM",
}


def parse_key(key: str) -> tuple[str, str, str]:
    """`B0_phantom_som_reddit` -> (B0, phantom_som, reddit)."""
    parts = key.split("_")
    return parts[0], "_".join(parts[1:-1]), parts[-1]


def resolve_targets() -> dict[str, str]:
    targets = dict(CANONICAL)
    targets.update(_discover_cls())
    return targets


def load_scans(scan_dir: Path, targets: dict[str, str]) -> dict[str, dict]:
    """Read (or produce) one diag scan JSON per condition, asserting one ruleset."""
    scan_dir.mkdir(parents=True, exist_ok=True)
    scans = {}
    for key, run in sorted(targets.items()):
        path = scan_dir / f"{key}.json"
        if path.exists():
            scans[key] = json.loads(path.read_text(encoding="utf-8"))
            continue
        d = scan(key, run, scan_dir)
        if d is None:
            raise SystemExit(f"FATAL: scan failed for {key} ({run})")
        scans[key] = d

    versions = collections.Counter(d.get("ruleset_version", "?") for d in scans.values())
    if len(versions) != 1:
        raise SystemExit(
            f"FATAL: mixed ruleset_version {dict(versions)} — cross-mode aggregation is "
            "barred by the /diag discover-then-freeze discipline. Rerun diag_rescan_all.py."
        )
    if len(scans) != 36:
        raise SystemExit(f"FATAL: expected 36 landed conditions, resolved {len(scans)}")
    return scans


def part_a(scans: dict[str, dict]) -> dict:
    """Episode-level hit rate per (rule, mode), pooled over the six cells."""
    num: dict[tuple[str, str], int] = collections.defaultdict(int)
    den: dict[str, int] = collections.defaultdict(int)
    names: dict[str, str] = {}
    for key, d in scans.items():
        _, mode, site = parse_key(key)
        # Same SCORED restriction as Part B. The scan JSONs carry every collected
        # episode, so without this the reddit denominators would be 205 rather than
        # the 203 every other rate in the paper uses.
        by_task = {int(ep["task_id"]): ep for ep in d["results"]
                   if ep.get("task_id") is not None}
        kept, _prov = restrict_to_scored(by_task, site)
        den[mode] += len(kept)
        for ep in kept.values():
            for rule in {h["rule_id"] for h in ep["hits"]}:
                num[(rule, mode)] += 1
            for h in ep["hits"]:
                names[h["rule_id"]] = h["rule_name"]

    total_eps = sum(den[m] for m in MODES)
    rules = []
    for rule in sorted({r for r, _ in num}):
        per_mode = {m: 100.0 * num[(rule, m)] / den[m] for m in MODES}
        text_only = [per_mode[m] for m in TEXT_BEARING]
        rules.append({
            "rule_id": rule,
            "rule_name": names.get(rule, ""),
            "overall_pct": 100.0 * sum(num[(rule, m)] for m in MODES) / total_eps,
            "per_mode_pct": per_mode,
            "spread_all_modes_pp": max(per_mode.values()) - min(per_mode.values()),
            "spread_text_bearing_pp": max(text_only) - min(text_only),
            "structurally_zero_under_vision": per_mode["vision"] == 0.0,
        })
    rules.sort(key=lambda r: -r["overall_pct"])
    return {
        "episodes_per_mode": {m: den[m] for m in MODES},
        "episodes_total": total_eps,
        "rules": rules,
    }


def measure_refs(run_dir: Path, site: str) -> dict:
    """Hallucinated-reference counts under two denominators.

    Two, because they answer different questions and the answers differ. The
    action-step rate weights an episode by how many actions it took, so an episode
    that deadlocks on one invalid id for thirty steps contributes thirty times as
    much as an episode that misfires once. Episode incidence — the share of
    episodes with at least one hallucinated reference — removes that weighting.
    Any claim in the paper is stated only if it holds under both (self-audit
    2026-07-28: three of five candidate claims did not).

    Restricted to the canonical SCORED task set (B-1906 discipline). This reads
    episodes through `_discover_episodes`, so it never names `*_summary_v2.json`
    and the universe-consumption lint's source grep cannot see it — the call to
    `restrict_to_scored` below is what keeps the denominator honest.
    """
    out = {"ep_total": 0, "ep_failed": 0, "ep_with_hall": 0,
           "act_all": 0, "act_failed": 0, "hall_all": 0, "hall_failed": 0}
    per_task: dict[int, dict] = {}
    for steps_path, summary_path, _cfg in _discover_episodes(run_dir, None, None):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        steps = _load_steps(steps_path)
        if not steps:
            continue
        tid = summary.get("task_id", steps[0].get("task_id"))
        if tid is None:
            continue
        ok = bool(summary.get("success"))
        rec = {"ok": ok, "act": 0, "hall": 0}
        seen: set = set()
        for s in steps:
            if _action_type(s) not in ACTION_TYPES:
                continue
            rec["act"] += 1
            hit = False
            for err in _locator_errors(s):
                if MISSING_UNION_BOUND_RE.search(err):
                    k = (s.get("step_idx"), err)
                    if k not in seen:
                        seen.add(k)
                        hit = True
            if hit:
                rec["hall"] += 1
        per_task[int(tid)] = rec

    kept, prov = restrict_to_scored(per_task, site)
    for rec in kept.values():
        out["ep_total"] += 1
        out["ep_failed"] += 0 if rec["ok"] else 1
        out["ep_with_hall"] += 1 if rec["hall"] else 0
        out["act_all"] += rec["act"]
        out["hall_all"] += rec["hall"]
        if not rec["ok"]:
            out["act_failed"] += rec["act"]
            out["hall_failed"] += rec["hall"]
    out["universe_sha256"] = prov.get("content_task_ids_sha256")
    return out


def part_b(targets: dict[str, str]) -> dict:
    """Hallucinated-reference rate per (site, backbone, mode) + the 2x2 decomposition."""
    cells: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for key, run in sorted(targets.items()):
        model, mode, site = parse_key(key)
        m = measure_refs(PHASE1 / run, site)
        m["rate_all_pct"] = 100.0 * m["hall_all"] / m["act_all"] if m["act_all"] else 0.0
        m["rate_failed_pct"] = (100.0 * m["hall_failed"] / m["act_failed"]
                                if m["act_failed"] else 0.0)
        m["episode_incidence_pct"] = (100.0 * m["ep_with_hall"] / m["ep_total"]
                                      if m["ep_total"] else 0.0)
        m["run_id"] = run
        cells[f"{site}·{model}"][mode] = m

    def _decompose(field: str) -> list[dict]:
        rows = []
        for cell, bymode in sorted(cells.items()):
            if not all(m in bymode for m in QUADRANT):
                continue
            r = {m: bymode[m][field] for m in QUADRANT}
            rows.append({
                "cell": cell,
                "rates_pct": {PAPER_NAME[m]: r[m] for m in QUADRANT},
                # text knob (AXTree -> legend) held at each prompt family
                "text_effect_at_DOM_prompt_pp": r["phantom_text"] - r["dom"],
                "text_effect_at_SoM_prompt_pp": r["phantom_som"] - r["phantom_prompt"],
                # prompt knob (DOM -> SoM) held at each text payload
                "prompt_effect_on_AXTree_pp": r["phantom_prompt"] - r["dom"],
                "prompt_effect_on_legend_pp": r["phantom_som"] - r["phantom_text"],
                "lowest_arm": PAPER_NAME[min(r, key=lambda m: r[m])],
                "highest_arm": PAPER_NAME[max(r, key=lambda m: r[m])],
            })
        return rows

    def _summary(rows: list[dict]) -> dict:
        def c(pred) -> int:
            return sum(1 for d in rows if pred(d))
        return {
            "n_cells": len(rows),
            "cells_where_P_SoM_lowest": c(lambda d: d["lowest_arm"] == "P-SoM"),
            "cells_where_P_prompt_highest": c(lambda d: d["highest_arm"] == "P-prompt"),
            "cells_text_effect_negative_at_SoM_prompt":
                c(lambda d: d["text_effect_at_SoM_prompt_pp"] < 0),
            "cells_text_effect_negative_at_DOM_prompt":
                c(lambda d: d["text_effect_at_DOM_prompt_pp"] < 0),
            # The interaction itself: is the legend's reduction bigger under the
            # SoM prompt than under the DOM prompt?
            "cells_reduction_larger_at_SoM_prompt":
                c(lambda d: d["text_effect_at_SoM_prompt_pp"]
                  < d["text_effect_at_DOM_prompt_pp"]),
            "cells_prompt_effect_negative_on_legend":
                c(lambda d: d["prompt_effect_on_legend_pp"] < 0),
            "cells_prompt_effect_negative_on_AXTree":
                c(lambda d: d["prompt_effect_on_AXTree_pp"] < 0),
        }

    by_step = _decompose("rate_all_pct")
    by_episode = _decompose("episode_incidence_pct")
    s_step, s_ep = _summary(by_step), _summary(by_episode)
    robust = {k: (s_step[k], s_ep[k]) for k in s_step if k != "n_cells"}
    return {
        "per_cell_per_mode": {c: {PAPER_NAME[m]: v for m, v in bm.items()}
                              for c, bm in cells.items()},
        "quadrant_definition": {PAPER_NAME[m]: {"text": t, "prompt": p}
                                for m, (t, p) in QUADRANT.items()},
        "decomposition_by_action_step": by_step,
        "decomposition_by_episode_incidence": by_episode,
        "summary_by_action_step": s_step,
        "summary_by_episode_incidence": s_ep,
        # A claim is quotable in the paper only where both denominators agree at 6/6.
        "denominator_robust_at_6_of_6": sorted(
            k for k, (a, b) in robust.items() if a == b == s_step["n_cells"]),
    }


def render_md(a: dict, b: dict, ruleset: str) -> str:
    L = []
    L.append("# Cross-mode failure signatures — 36 landed Phase-1a conditions\n")
    L.append(f"Ruleset `{ruleset}` · {a['episodes_total']} episodes over 36 conditions "
             "(2 sites x 3 backbones x 6 modes).\n")
    L.append("Regenerate: `python3 scripts/analysis/aggregate_cross_mode_failure_signatures.py`\n")

    L.append("\n## Part A — signature frequency by mode (paper A §4.3)\n")
    L.append("Episode-level hit rate: share of episodes in which the signature fires at "
             "least once, pooled over the six (site, backbone) cells.\n")
    L.append("`vision` carries no element-id list, so id-space rules are structurally "
             "inapplicable there; the last column excludes it.\n")
    hdr = "| rule | name | overall % | " + " | ".join(PAPER_NAME[m] for m in MODES) \
          + " | spread (all) | spread (text-bearing) |"
    L.append("\n" + hdr)
    L.append("|" + "---|" * (len(MODES) + 5))
    for r in a["rules"][:12]:
        cells = " | ".join(f"{r['per_mode_pct'][m]:.1f}" for m in MODES)
        L.append(f"| {r['rule_id']} | {r['rule_name']} | {r['overall_pct']:.1f} | {cells} "
                 f"| {r['spread_all_modes_pp']:.1f} | {r['spread_text_bearing_pp']:.1f} |")

    top4 = a["rules"][:4]
    L.append("\n**Top four signatures**: " + ", ".join(
        f"{r['rule_id']} ({r['overall_pct']:.1f}%)" for r in top4) + ".")
    L.append("Spread across the five text-bearing modes: " + ", ".join(
        f"{r['rule_id']} {r['spread_text_bearing_pp']:.1f} pp" for r in top4) + ".")
    L.append("Including `vision`: " + ", ".join(
        f"{r['rule_id']} {r['spread_all_modes_pp']:.1f} pp" for r in top4) + ".")

    L.append("\n## Part B — hallucinated element references (paper A §4.2)\n")
    L.append("Two denominators, because they disagree. **action-step** = share of "
             "click / type / select_option steps naming an absent id, which weights an "
             "episode by how many actions it took; **episode incidence** = share of episodes "
             "with at least one such step, which does not. A thirty-step deadlock on one "
             "invalid id moves the first a great deal and the second by one episode.\n")
    L.append("Restricted to the canonical SCORED task set.\n")
    for field, title in (("rate_all_pct", "By action-step"),
                         ("episode_incidence_pct", "By episode incidence")):
        L.append(f"\n### {title}\n")
        L.append("| cell | " + " | ".join(PAPER_NAME[m] for m in MODES) + " |")
        L.append("|" + "---|" * (len(MODES) + 1))
        for cell in sorted(b["per_cell_per_mode"]):
            bm = b["per_cell_per_mode"][cell]
            row = " | ".join(f"{bm[PAPER_NAME[m]][field]:.3f}" if PAPER_NAME[m] in bm
                             else "—" for m in MODES)
            L.append(f"| {cell} | {row} |")

    L.append("\n### The 2x2: which knob moves the rate\n")
    L.append("P-text = legend text under the DOM prompt; P-prompt = AXTree text under the "
             "SoM prompt. So the text knob is read at a fixed prompt family and vice versa.\n")
    for key, title in (("decomposition_by_action_step", "By action-step"),
                       ("decomposition_by_episode_incidence", "By episode incidence")):
        L.append(f"\n**{title}**\n")
        L.append("| cell | text @ DOM prompt | text @ SoM prompt | prompt @ AXTree | "
                 "prompt @ legend | lowest | highest |")
        L.append("|" + "---|" * 7)
        for d in b[key]:
            L.append(f"| {d['cell']} | {d['text_effect_at_DOM_prompt_pp']:+.3f} | "
                     f"{d['text_effect_at_SoM_prompt_pp']:+.3f} | "
                     f"{d['prompt_effect_on_AXTree_pp']:+.3f} | "
                     f"{d['prompt_effect_on_legend_pp']:+.3f} | {d['lowest_arm']} | "
                     f"{d['highest_arm']} |")

    ss, se = b["summary_by_action_step"], b["summary_by_episode_incidence"]
    n = ss["n_cells"]
    L.append("\n### Which statements survive both denominators\n")
    L.append("| statement | by action-step | by episode incidence | quotable |")
    L.append("|---|---|---|---|")
    labels = [
        ("cells_reduction_larger_at_SoM_prompt",
         "legend's reduction is larger under the SoM prompt than the DOM prompt"),
        ("cells_text_effect_negative_at_SoM_prompt",
         "legend lowers the rate under the SoM prompt"),
        ("cells_prompt_effect_negative_on_legend",
         "SoM prompt lowers the rate when the text is the legend"),
        ("cells_where_P_SoM_lowest", "P-SoM is the lowest arm"),
        ("cells_where_P_prompt_highest", "P-prompt is the highest arm"),
        ("cells_text_effect_negative_at_DOM_prompt",
         "legend lowers the rate under the DOM prompt"),
        ("cells_prompt_effect_negative_on_AXTree",
         "SoM prompt lowers the rate when the text is the AXTree"),
    ]
    for k, text in labels:
        ok = "**yes**" if ss[k] == se[k] == n else "no"
        L.append(f"| {text} | {ss[k]}/{n} | {se[k]}/{n} | {ok} |")
    L.append("\nOnly the rows marked **yes** are stated in the paper. The interaction claim "
             "rests on the first row: the legend's effect on reference hallucination depends "
             "on which prompt it is paired with, in every cell under either denominator. The "
             "arms in which the prompt's advertised id scheme and the text's actual id scheme "
             "agree behave differently from the two mismatched arms.\n")
    L.append("Rows marked *no* are real under one denominator and not the other, which is "
             "why the second denominator is computed at all rather than assumed to agree.\n")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scan-dir", type=Path, default=None,
                    help="reuse an existing diag scan dir (default: fresh temp scan)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    targets = resolve_targets()
    scan_dir = args.scan_dir or Path(tempfile.mkdtemp(prefix="diag_xmode_"))
    print(f"scan dir: {scan_dir}")
    scans = load_scans(scan_dir, targets)
    ruleset = next(iter(scans.values()))["ruleset_version"]
    print(f"ruleset_version: {ruleset} (36/36 consistent)")

    a = part_a(scans)
    print("Part A done — top rules:",
          ", ".join(r["rule_id"] for r in a["rules"][:4]))
    b = part_b(targets)
    print("Part B done — denominator-robust at 6/6:", b["denominator_robust_at_6_of_6"])
    print("  by action-step:      ", b["summary_by_action_step"])
    print("  by episode incidence:", b["summary_by_episode_incidence"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "producer": "scripts/analysis/aggregate_cross_mode_failure_signatures.py",
        "ruleset_version": ruleset,
        "n_conditions": len(scans),
        "run_ids": {k: v for k, v in sorted(targets.items())},
        "part_a_signature_frequency": a,
        "part_b_hallucinated_references": b,
    }
    (args.out_dir / "cross_mode_failure_signatures.json").write_text(
        json.dumps(payload, indent=1), encoding="utf-8")
    (args.out_dir / "cross_mode_failure_signatures.md").write_text(
        render_md(a, b, ruleset), encoding="utf-8")
    print(f"wrote {args.out_dir}/cross_mode_failure_signatures.{{json,md}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
