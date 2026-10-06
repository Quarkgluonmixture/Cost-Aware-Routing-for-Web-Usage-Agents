#!/usr/bin/env python3
"""Tier-1 failure attribution for VWA shopping (B0 x3, B1 x6), read against the other sites.

Shopping had no /diag coverage at all until 2026-10-06 (实验笔记 §531.9): the 41 digests
cover classifieds / reddit / WA only. This product is the mechanical half — what the v11
rule set says about shopping failures — and states, before anything else, how much of the
failure mass that rule set does NOT explain on a site it was never discovered on.

Inputs
  results/diag_scans/v11_vwa_shop/*.json   one scan per shopping condition (pinned runs:
                                           run_manifest.yaml `extension:`)
  results/diag_scans/v11_vwa/*.json        classifieds + reddit, same ruleset (comparison)
  results/diag_scans/v11_wa/*.json         WA reddit, same ruleset (comparison)
Every count is over the SCORED universe (shopping 432 after AMENDMENT_09/10).

Outputs
  docs/analysis/vwa_shopping/_tier1_summary.{md,json}

Regenerate: python scripts/analysis/shopping_diag_tier1.py
"""
from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.lib.canonical_task_universe import expected_scored_ids  # noqa: E402

SCANS = REPO / "results/diag_scans"
SHOP = SCANS / "v11_vwa_shop"
OUT_DIR = REPO / "docs/analysis/vwa_shopping"
RULESET = "11-intent-text-fallback"

# Success-side benchmark-FP rules (rule names carry "FP").
FP_RULES = {"P28", "P29", "P40", "P41"}
# Hits on shopping that were read by hand on 2026-10-06 and found to be rule misfires, not
# evaluator errors (§531.9). Keyed (scan stem, task_id, rule). The rules themselves are left
# unchanged under discover-then-freeze; the fixes are queued for the v12 rule batch.
REVIEWED_NOT_FP = {
    ("B0_som_shopping", 183, "P41"):
        "eval HAS a positive check (`required_values: ['== 3']`, order quantity); P41 only "
        "recognises must_include / exact_match / fuzzy_match as positive",
    ("B0_som_shopping", 200, "P40"):
        "agent opened the item page (…/spanish-cow-milk-cheese-mahon-1-pound.html) and read "
        "'1 pound' from the title; P40's item-page markers (page=item, product_id=, /product/) "
        "do not match Magento product URLs",
    ("B0_vision_shopping", 200, "P40"): "same trajectory shape as B0_som_shopping task 200",
}
# Risk markers, not causes: they fire on most long failures (see digest interpretation notes).
RISK_MARKERS = {"P31", "P36"}


class MissingInput(RuntimeError):
    pass


def _site_of(stem: str) -> tuple[str, str]:
    if stem.endswith("_wa_reddit"):
        return "reddit", "webarena"
    for s in ("classifieds", "reddit", "shopping"):
        if stem.endswith(s):
            return s, "visualwebarena"
    raise MissingInput(f"cannot parse site from {stem}")


def summarise(path: Path, site: str, bench: str) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    if d.get("ruleset_version") != RULESET:
        raise MissingInput(f"{path.name}: ruleset {d.get('ruleset_version')!r}, expected {RULESET}")
    if d.get("config_missing"):
        raise MissingInput(f"{path.name}: {d['config_missing']} episodes scanned without config")
    scored = expected_scored_ids(site, bench)[0]
    rows = [e for e in d["results"] if e["task_id"] in scored]
    if len({e["task_id"] for e in rows}) != len(scored):
        raise MissingInput(f"{path.name}: {len(rows)} scored episodes, universe has {len(scored)}")
    fail = [e for e in rows if not e["success"]]
    rules = collections.Counter(h["rule_id"] for e in fail for h in e["hits"])
    non_marker = [e for e in fail if any(h["rule_id"] not in RISK_MARKERS for h in e["hits"])]
    fp = [(e["task_id"], "success" if e["success"] else "fail", h["rule_id"])
          for e in rows for h in e["hits"] if h["rule_id"] in FP_RULES]
    return {
        "run_id": d.get("run_id"), "n": len(rows), "success": len(rows) - len(fail),
        "failed": len(fail),
        "failed_with_hit": sum(1 for e in fail if e["hits"]),
        "failed_no_hit": sum(1 for e in fail if not e["hits"]),
        "failed_only_risk_markers": len([e for e in fail if e["hits"]]) - len(non_marker),
        "scaffold_episodes": sum(1 for e in fail if any(h.get("is_scaffold") for h in e["hits"])),
        "fp_rule_hits": fp,
        "top_rules": rules.most_common(8),
        "no_hit_task_ids": sorted(e["task_id"] for e in fail if not e["hits"]),
    }


def build() -> dict:
    shop = sorted(SHOP.glob("*_shopping.json"))
    if len(shop) != 9:
        raise MissingInput(f"expected 9 shopping scans in {SHOP}, found {len(shop)}")
    cells = {p.stem: summarise(p, "shopping", "visualwebarena") for p in shop}
    for stem, c in cells.items():
        c["fp_rule_hits"] = [
            {"task_id": t, "outcome": o, "rule": r,
             "review": REVIEWED_NOT_FP.get((stem, t, r), "NOT REVIEWED")}
            for t, o, r in c["fp_rule_hits"]]
    share = {}
    for name, d in (("classifieds", "v11_vwa"), ("reddit", "v11_vwa"), ("wa_reddit", "v11_wa")):
        vals = []
        for p in sorted((SCANS / d).glob("*.json")):
            if name == "wa_reddit" or p.stem.endswith(name):
                s = summarise(p, *_site_of(p.stem))
                vals.append(s["failed_no_hit"] / s["failed"])
        share[name] = vals
    share["shopping"] = [c["failed_no_hit"] / c["failed"] for c in cells.values()]
    return {"schema": "2026-10-06-shopping-diag-tier1-v1", "ruleset_version": RULESET,
            "cells": cells,
            "no_hit_share_by_site": {k: {"k": len(v), "median": statistics.median(v),
                                         "min": min(v), "max": max(v)} for k, v in share.items()}}


def render(d: dict) -> str:
    L = ["---", "type: analysis", "status: tier-1 only",
         "purpose: failure attribution coverage for VWA shopping, read against the other sites",
         "producer: scripts/analysis/shopping_diag_tier1.py", "---", "",
         "# VWA shopping — Tier-1 failure attribution (ruleset v11)", "",
         "Regenerate: `python scripts/analysis/shopping_diag_tier1.py`", "",
         "Mechanical half only: rule hits from `diag_pattern_match.py` at the same ruleset as the "
         "classifieds / reddit / WA conditions, over the scored universe (432). No Tier-2 / Tier-3 reading yet "
         "except the benchmark-FP flags in §3.", "",
         "## 1. How much of the failure mass the rule set explains, by site", "",
         "Share of FAILED episodes on which no rule fires. The rules were discovered on "
         "classifieds and reddit; a higher share on a new site is the size of what Tier-1 "
         "cannot speak for there.", "",
         "| site | conditions | median | range |", "|---|---|---|---|"]
    for k, v in d["no_hit_share_by_site"].items():
        L.append(f"| {k} | {v['k']} | **{100*v['median']:.0f}%** | "
                 f"{100*v['min']:.0f}–{100*v['max']:.0f}% |")
    L += ["", "## 2. Per condition", "",
          "| condition | SR | failed | no rule | only risk markers (P31/P36) | scaffold | top rules on failures |",
          "|---|---|---|---|---|---|---|"]
    for stem, c in d["cells"].items():
        top = ", ".join(f"{r} {n}" for r, n in c["top_rules"][:5])
        L.append(f"| `{stem}` | {100*c['success']/c['n']:.2f}% ({c['success']}/{c['n']}) | "
                 f"{c['failed']} | {c['failed_no_hit']} ({100*c['failed_no_hit']/c['failed']:.0f}%) | "
                 f"{c['failed_only_risk_markers']} | {c['scaffold_episodes']} | {top} |")
    n_scaf = sum(c["scaffold_episodes"] for c in d["cells"].values())
    L += ["", f"Scaffold-bug rules fire on **{n_scaf}** shopping failures across the nine conditions.",
          "", "⚠️ Rule counts are symptoms, not causes: P31 (budget exhausted) and P36 "
          "(degenerate walk) fire on most long failures. The `only risk markers` column counts "
          "failures explained by nothing more specific than those two.", "",
          "## 3. Benchmark-FP flags", ""]
    flags = [(stem, f) for stem, c in d["cells"].items() for f in c["fp_rule_hits"]]
    if not flags:
        L.append("None.")
    else:
        L += ["| condition | task | outcome | rule | hand review |", "|---|---|---|---|---|"]
        for stem, f in flags:
            L.append(f"| `{stem}` | {f['task_id']} | {f['outcome']} | {f['rule']} | {f['review']} |")
        unrev = [f for _, f in flags if f["review"] == "NOT REVIEWED"]
        L += ["", f"**{len(flags) - len(unrev)} of {len(flags)}** flags read by hand; "
              + ("all were rule misfires, so the benchmark-FP count on shopping is **0** at Tier-1. "
                 if not unrev else
                 f"**{len(unrev)} not yet reviewed** — do not quote a benchmark-FP count. ")
              + "The misfires are queued for the v12 rule batch rather than patched here, "
                "because a rule change obliges a full rescan (discover-then-freeze)."]
    L += ["", "## 4. What is still open", "",
          "- The no-rule failures (§1) are unattributed. They need Tier-2 reading — a sample per "
          "condition is enough to say whether they hide a scaffold or evaluator class the rules "
          "miss. Task ids are in the JSON (`no_hit_task_ids`).",
          "- B0 shopping has no phantom arms, so only three of its modes appear."]
    return "\n".join(L) + "\n"


def main() -> int:
    d = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "_tier1_summary.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
    (OUT_DIR / "_tier1_summary.md").write_text(render(d), encoding="utf-8")
    print(f"✓ {(OUT_DIR / '_tier1_summary.md').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
