#!/usr/bin/env python3
"""Benchmark EDA over the VWA / WA task corpora — thesis rubric #10.

What a dissertation reader needs before any result table means anything: how
many tasks, of what kind, how hard, and how the sites differ from each other.
None of that exists anywhere in the repo yet (`known.py "benchmark EDA"` → 0
matches), and the 8-page REALM paper had no room for it.

Reads the `.raw.json` configs (placeholder URLs, host-independent) so the output
is reproducible on any machine without a live site.

Three things worth knowing about the corpora, all of which this script surfaces:

  * VWA ships three per-task difficulty labels (`reasoning_difficulty`,
    `visual_difficulty`, `overall_difficulty`). They are human annotations from
    the benchmark authors, not derived from any model's behaviour — so crossing
    them against our measured SR is a genuine external check, not circular.
  * WA configs carry no `image` and no difficulty fields. Cross-benchmark tables
    must therefore be built on the intersecting columns only; the script marks
    absent fields as None rather than defaulting them to a value, so a missing
    annotation can never be silently read as "easy" or "no image".
  * `intent_template_id` groups tasks that are the same question with different
    slot fills. The count of DISTINCT templates is the honest measure of task
    diversity — 234 classifieds tasks are not 234 independent questions.

Usage:
    .venv/bin/python3 scripts/analysis/benchmark_eda.py
    .venv/bin/python3 scripts/analysis/benchmark_eda.py --json-only

Outputs: docs/analysis/benchmark_eda/{corpus_eda.md,corpus_eda.json}
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Canonical definitions, imported rather than reimplemented. A second copy of
# "which tasks are scored" is exactly how the corpus table and the results
# tables drift apart.
from p79.experiment.tasks import PROTOCOL_EXCLUSIONS, _is_na_task  # noqa: E402

CFG = REPO / "external/visualwebarena/config_files"
OUT = REPO / "docs/analysis/benchmark_eda"

CORPORA = [
    ("VWA", "visualwebarena", "classifieds", CFG / "vwa/test_classifieds.raw.json"),
    ("VWA", "visualwebarena", "reddit", CFG / "vwa/test_reddit.raw.json"),
    ("VWA", "visualwebarena", "shopping", CFG / "vwa/test_shopping.raw.json"),
    ("WA", "webarena", "reddit", CFG / "wa/test_reddit.raw.json"),
    ("WA", "webarena", "shopping", CFG / "wa/test_shopping.raw.json"),
    ("WA", "webarena", "shopping_admin", CFG / "wa/test_shopping_admin.raw.json"),
]

# Difficulty vocabularies the benchmark authors intended. Anything outside these
# is reported as a corpus defect, never silently folded into the nearest label —
# a typo'd level that gets auto-corrected is a stratified analysis quietly
# running on a different partition than the one it documents.
DIFFICULTY_LEVELS = {"easy", "medium", "hard"}


def _pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.1f}%" if d else "—"


def profile(bench: str, bench_key: str, site: str, path: Path) -> dict[str, Any]:
    tasks = json.loads(path.read_text())
    n = len(tasks)

    # --- scored-set derivation -------------------------------------------
    # Two exclusion layers with different timing, and the distinction matters:
    #   N/A      — dropped at task-LOAD time; the runner never sees them
    #   protocol — dropped at ANALYSIS time; episodes are collected on purpose
    # Both are preregistered (§139.8 / AMENDMENT_08-10).
    na_ids = sorted(t["task_id"] for t in tasks if _is_na_task(t))
    protocol = PROTOCOL_EXCLUSIONS.get((bench_key, site), ())
    protocol_ids = sorted(e.task_id for e in protocol)
    scored = n - len(na_ids) - len(protocol_ids)

    # --- corpus defects ---------------------------------------------------
    defects = []
    for t in tasks:
        for field in ("reasoning_difficulty", "visual_difficulty", "overall_difficulty"):
            val = t.get(field)
            if val is not None and val not in DIFFICULTY_LEVELS:
                defects.append({"task_id": t["task_id"], "field": field, "value": val})

    # eval_types is a list per task; a task can carry more than one checker.
    eval_combo = Counter()
    eval_single = Counter()
    for t in tasks:
        types = tuple(sorted((t.get("eval") or {}).get("eval_types") or []))
        eval_combo[types or ("<none>",)] += 1
        for et in types:
            eval_single[et] += 1

    # `image` is VWA-only and may be a str or a list of reference images.
    def n_images(t: dict) -> int:
        img = t.get("image")
        if img is None:
            return 0
        return len(img) if isinstance(img, list) else 1

    img_counts = [n_images(t) for t in tasks]
    intents = [t.get("intent", "") for t in tasks]
    words = [len(i.split()) for i in intents]
    chars = [len(i) for i in intents]

    # Difficulty labels: VWA-only. Absent -> None, never a default.
    def dist(field: str) -> dict[str, int] | None:
        vals = [t.get(field) for t in tasks]
        if all(v is None for v in vals):
            return None
        return dict(Counter(str(v) for v in vals).most_common())

    templates = [t.get("intent_template_id") for t in tasks]
    n_templates = len({x for x in templates if x is not None})
    per_template = Counter(x for x in templates if x is not None)

    return {
        "benchmark": bench,
        "site": site,
        "n_tasks": n,
        "n_na_excluded": len(na_ids),
        "na_task_ids": na_ids,
        "n_protocol_excluded": len(protocol_ids),
        "protocol_exclusions": [
            {"task_id": e.task_id, "tier": e.tier, "amendment": e.amendment} for e in protocol
        ],
        "n_scored": scored,
        "corpus_defects": defects,
        "n_distinct_templates": n_templates or None,
        "tasks_per_template_median": (
            statistics.median(per_template.values()) if per_template else None
        ),
        "tasks_per_template_max": max(per_template.values()) if per_template else None,
        "require_login": sum(1 for t in tasks if t.get("require_login")),
        "require_reset": sum(1 for t in tasks if t.get("require_reset")),
        "with_reference_image": sum(1 for c in img_counts if c > 0),
        "reference_images_total": sum(img_counts),
        "multi_image_tasks": sum(1 for c in img_counts if c > 1),
        "eval_type_counts": dict(eval_single.most_common()),
        "eval_combos": {" + ".join(k): v for k, v in eval_combo.most_common()},
        "intent_words": {
            "median": statistics.median(words),
            "mean": round(statistics.mean(words), 1),
            "p10": sorted(words)[max(0, int(0.10 * len(words)) - 1)],
            "p90": sorted(words)[min(len(words) - 1, int(0.90 * len(words)))],
            "max": max(words),
        },
        "intent_chars": {
            "median": statistics.median(chars),
            "mean": round(statistics.mean(chars), 1),
            "max": max(chars),
        },
        "reasoning_difficulty": dist("reasoning_difficulty"),
        "visual_difficulty": dist("visual_difficulty"),
        "overall_difficulty": dist("overall_difficulty"),
    }


def render(profiles: list[dict]) -> str:
    L: list[str] = []
    L.append("# Benchmark corpus EDA — VWA + WA\n")
    L.append(
        "Generated by `scripts/analysis/benchmark_eda.py` from the `.raw.json` task "
        "configs (placeholder URLs, host-independent). Thesis rubric #10.\n"
    )
    L.append("\n## 0. From corpus to scored set\n")
    L.append(
        "Every success rate in this thesis has one of the `scored` numbers below as "
        "its denominator. The two exclusion layers differ in *when* they apply, and "
        "the distinction is operational, not cosmetic:\n\n"
        "- **N/A** — removed at task-**load** time; the runner never sees these "
        "episodes. All are `string_match` with `reference_answers.fuzzy_match == "
        '"N/A"`. Under our no-N/A-exit agent prompt they are un-passable, and the '
        "evaluator cannot separate a reasoned N/A judgement from an early exit "
        "(§139.8, preregistered).\n"
        "- **protocol** — removed at **analysis** time only; the runner keeps "
        "collecting them deliberately, so the exclusion stays auditable and "
        "reversible (AMENDMENT_08–10).\n"
    )
    L.append(
        "\nBecause the layers fire at different times, **two different counts are "
        "both correct** and the project uses both. Quote whichever one your "
        "denominator actually is:\n"
    )
    L.append("| corpus | corpus size | − N/A | **run set** | − protocol | **scored set** |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for p in profiles:
        prot = (
            f"{p['n_protocol_excluded']} "
            f"({', '.join(str(e['task_id']) for e in p['protocol_exclusions'])})"
            if p["n_protocol_excluded"]
            else "0"
        )
        run_set = p["n_tasks"] - p["n_na_excluded"]
        L.append(
            f"| {p['benchmark']} {p['site']} | {p['n_tasks']} | {p['n_na_excluded']} | "
            f"**{run_set}** | {prot} | **{p['n_scored']}** |"
        )
    L.append(
        "\n<sub>A protocol count of 0 is the CURRENT state (no amendment has "
        "excluded a task on that corpus yet), not a property of the corpus.</sub>\n"
    )
    L.append(
        "\n- **run set** = episodes a completed run actually contains. Verified "
        "against real runs: `B1_dom_classifieds` → 224 episodes, `B1_dom_reddit` → "
        "**205**, WA `B1_dom_wa_reddit` → **104**.\n"
        "- **scored set** = denominator of every reported success rate: VWA reddit "
        "**203**, VWA shopping **432**.\n\n"
        "This is the source of a long-standing double set of numbers in the project "
        "notes — reddit appears as both 205 and 203, shopping as both 435 and 432. "
        "Neither is wrong; they are the run set and the scored set.\n\n"
        "> **How to use this as a check.** On a corpus that HAS protocol exclusions "
        "(currently VWA reddit and VWA shopping) a complete run's episode count "
        "should be **greater than** the scored number, because protocol-excluded "
        "episodes are collected on purpose. Where the exclusion count is 0 — VWA "
        "classifieds and all three WA sites today — the two are **equal by "
        "construction**, and equality says nothing either way. An earlier version of "
        "this note stated the check unconditionally; the table above is its own "
        "counterexample (cls 224 == 224).\n"
    )

    defect_rows = [(p, d) for p in profiles for d in p["corpus_defects"]]
    if defect_rows:
        L.append("\n### 0.1 Corpus defects found in the shipped annotations\n")
        L.append(
            "Difficulty labels outside the intended `{easy, medium, hard}` vocabulary. "
            "Reported rather than auto-corrected: a typo'd level folded silently into "
            "the nearest label means a stratified analysis runs on a different "
            "partition than the one it documents.\n"
        )
        L.append("| corpus | task_id | field | value | likely intent |")
        L.append("|---|---:|---|---|---|")
        for p, d in defect_rows:
            guess = {"hrad": "hard", "mediun": "medium"}.get(str(d["value"]), "—")
            L.append(
                f"| {p['benchmark']} {p['site']} | {d['task_id']} | `{d['field']}` | "
                f"`{d['value']}` | `{guess}` |"
            )
        L.append(
            "\nEach affects a single task, so the effect on any aggregate is "
            "negligible; the reason to record them is that a difficulty-stratified "
            "table would otherwise silently grow an extra one-task stratum.\n"
        )

    L.append("\n## 1. Corpus at a glance\n")
    L.append("| corpus | tasks | distinct templates | tasks/template (med / max) | login | reset | with ref image |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for p in profiles:
        tpl = p["n_distinct_templates"]
        L.append(
            f"| {p['benchmark']} {p['site']} | {p['n_tasks']} | {tpl or '—'} | "
            f"{p['tasks_per_template_median'] or '—'} / {p['tasks_per_template_max'] or '—'} | "
            f"{p['require_login']} | {p['require_reset']} | "
            f"{p['with_reference_image']} ({_pct(p['with_reference_image'], p['n_tasks'])}) |"
        )
    L.append(
        "\n**Task diversity is well below task count.** A template is one question "
        "shape with different slot fills, so the distinct-template column is the "
        "honest denominator for 'how many different things are we asking'.\n"
    )

    L.append("\n## 2. Reference images — the axis that separates the two benchmarks\n")
    L.append("| corpus | tasks with ≥1 ref image | total ref images | multi-image tasks |")
    L.append("|---|---:|---:|---:|")
    for p in profiles:
        L.append(
            f"| {p['benchmark']} {p['site']} | {p['with_reference_image']} "
            f"({_pct(p['with_reference_image'], p['n_tasks'])}) | "
            f"{p['reference_images_total']} | {p['multi_image_tasks']} |"
        )
    L.append(
        "\nWA carries no reference images at all — a clean, measurable difference "
        "in what the task *specifies*, and the reason WA is worth treating as a "
        "distribution shift rather than a third VWA site.\n\n"
        "> ⚠️ **Do not overstate what this removes.** An earlier version of this "
        "note claimed the absence of reference images means \"the visual-grounding "
        "demand that motivates SoM is simply absent\". That conflates two things: "
        "**visual matching** (compare the page against a supplied target image — "
        "genuinely gone on WA) and **visual grounding** (locate and act on elements "
        "of the *current* screen — still required, and what SoM marks actually "
        "serve). What shifts is the task specification, not necessarily the "
        "modality requirement. Establishing the stronger claim would need evidence "
        "that DOM-only closes the gap to SoM on WA but not on VWA.\n"
    )

    L.append("\n## 3. Evaluation machinery\n")
    all_types = sorted({t for p in profiles for t in p["eval_type_counts"]})
    L.append("| corpus | " + " | ".join(all_types) + " |")
    L.append("|---" * (len(all_types) + 1) + "|")
    for p in profiles:
        cells = [str(p["eval_type_counts"].get(t, 0)) for t in all_types]
        L.append(f"| {p['benchmark']} {p['site']} | " + " | ".join(cells) + " |")
    L.append("\nCombinations actually used (a task may carry more than one checker):\n")
    for p in profiles:
        combos = ", ".join(f"`{k}` ×{v}" for k, v in list(p["eval_combos"].items())[:6])
        L.append(f"- **{p['benchmark']} {p['site']}** — {combos}")

    L.append("\n\n## 4. Difficulty annotations (VWA only)\n")
    L.append(
        "Human labels shipped by the benchmark authors, independent of any model's "
        "behaviour — so crossing them against measured SR is an external check, not "
        "a circular one. WA ships none; those rows are omitted rather than defaulted.\n"
    )
    for axis in ("reasoning_difficulty", "visual_difficulty", "overall_difficulty"):
        rows = [p for p in profiles if p[axis]]
        if not rows:
            continue
        levels = sorted({k for p in rows for k in p[axis]})
        L.append(f"\n**{axis.replace('_', ' ')}**\n")
        L.append("| corpus | " + " | ".join(levels) + " |")
        L.append("|---" * (len(levels) + 1) + "|")
        for p in rows:
            cells = [
                f"{p[axis].get(lv, 0)} ({_pct(p[axis].get(lv, 0), p['n_tasks'])})"
                for lv in levels
            ]
            L.append(f"| {p['benchmark']} {p['site']} | " + " | ".join(cells) + " |")

    L.append("\n\n## 5. Instruction length\n")
    L.append("| corpus | words (p10 / med / p90 / max) | chars (med / max) |")
    L.append("|---|---|---|")
    for p in profiles:
        w, c = p["intent_words"], p["intent_chars"]
        L.append(
            f"| {p['benchmark']} {p['site']} | {w['p10']} / {w['median']} / {w['p90']} / {w['max']} | "
            f"{c['median']} / {c['max']} |"
        )
    L.append(
        "\nIntent length matters for cost accounting: it is the one prompt component "
        "identical across all six observation modes, so it sets a floor that no "
        "representation choice can reduce.\n"
    )
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-only", action="store_true")
    args = ap.parse_args()

    profiles = []
    for bench, bench_key, site, path in CORPORA:
        if not path.exists():
            print(f"  skip (absent): {path.relative_to(REPO)}")
            continue
        profiles.append(profile(bench, bench_key, site, path))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "corpus_eda.json").write_text(json.dumps(profiles, indent=1, ensure_ascii=False))
    if not args.json_only:
        (OUT / "corpus_eda.md").write_text(render(profiles))

    for p in profiles:
        print(
            f"{p['benchmark']:4s} {p['site']:16s} n={p['n_tasks']:4d} "
            f"templates={str(p['n_distinct_templates']):>5s} "
            f"ref_img={p['with_reference_image']:4d} "
            f"intent_words_med={p['intent_words']['median']}"
        )
    print(f"\nwrote -> {OUT.relative_to(REPO)}/corpus_eda.{{md,json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
