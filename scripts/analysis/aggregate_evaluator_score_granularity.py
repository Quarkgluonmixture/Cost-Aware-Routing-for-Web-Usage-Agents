#!/usr/bin/env python3
"""Distinct evaluator score values across the landed Phase-1a conditions.

Closes traceability gap #16 (2026-07-27 three-AI stress, Mode A A-4 = codex #16).
Paper B appendix B.1 rules out continuous-label routing on the grounds that
VisualWebArena's evaluator emits no graded signal. The claim was carried by a
hand-written triple (7,963 episodes; 7,278 / 685 split) with no product behind it.

The substantive claim (the evaluator emits exactly two values) is checkable and is
what this script checks. It reports every distinct `score` value it finds rather
than asserting the expected two, so a third value would surface as data instead of
being masked by a passing assertion.

Universe: the `grade: paper-grade` cells of results/phantom_paper/run_manifest.yaml
— the same 36 landed conditions paper A §4.3 aggregates over. Archived (pre-fix)
conditions are counted separately and never pooled into the headline number.

Usage:
  python3 scripts/analysis/aggregate_evaluator_score_granularity.py

Output:
  docs/analysis/cross_sites/evaluator_score_granularity.json
  docs/analysis/cross_sites/evaluator_score_granularity.md
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analysis.lib.canonical_task_universe import restrict_to_scored  # noqa: E402
from scripts.analysis.lib.run_registry import load_manifest  # noqa: E402

PHASE1 = ROOT / "results" / "visualwebarena" / "phase1"
OUT_DIR = ROOT / "docs" / "analysis" / "cross_sites"


def condition_dir(cell: dict) -> Path | None:
    run_dir = cell.get("run_dir", "")
    sub = cell.get("condition_subdir", "")
    if not run_dir:
        return None
    base = Path(run_dir)
    if not base.is_absolute():
        base = PHASE1 / base
    d = base / sub / "episodes" if sub else base / "episodes"
    return d if d.is_dir() else None


_TASK_RE = re.compile(r"_task_(\d+)_summary_v2\.json$")


def tally(cell: dict) -> dict | None:
    """Score value counts for one condition, split scored vs protocol-excluded.

    B-1906 lint (test_universe_consumption_lint): reading `*_summary_v2.json`
    without consulting the canonical universe silently folds the AMENDMENT_08
    protocol-excluded reddit tasks (58, 160) into the denominator. The headline
    here is therefore the SCORED set, matching every other denominator in the
    paper; the excluded episodes are counted separately rather than dropped,
    because for this particular claim — that the evaluator emits only two values
    — they are additional evidence and it would be odd to discard them silently.
    """
    d = condition_dir(cell)
    if d is None:
        return None
    by_task: dict[int, float | None] = {}
    missing = 0
    non_numeric = 0
    for f in sorted(d.glob("*_summary_v2.json")):
        m = _TASK_RE.search(f.name)
        if m is None:
            continue
        try:
            s = json.loads(f.read_text(encoding="utf-8")).get("score")
        except (json.JSONDecodeError, OSError):
            continue
        tid = int(m.group(1))
        if s is None:
            missing += 1
            by_task[tid] = None
        elif not isinstance(s, (int, float)) or isinstance(s, bool):
            non_numeric += 1
            by_task[tid] = None
        else:
            by_task[tid] = float(s)

    kept, prov = restrict_to_scored(by_task, cell["site"], label=cell.get("run_dir", ""))
    scores: collections.Counter = collections.Counter(
        v for v in kept.values() if v is not None)
    excluded_scores: collections.Counter = collections.Counter(
        v for t, v in by_task.items() if t not in kept and v is not None)
    return {
        "run_dir": cell.get("run_dir"),
        "condition_subdir": cell.get("condition_subdir"),
        "n_episodes": len(by_task),
        "n_scored": sum(scores.values()),
        "n_score_missing": missing,
        "n_score_non_numeric": non_numeric,
        "value_counts": {str(k): v for k, v in sorted(scores.items())},
        "protocol_excluded": {
            "n_episodes": len(by_task) - len(kept),
            "value_counts": {str(k): v for k, v in sorted(excluded_scores.items())},
        },
        "universe_provenance": {
            k: prov[k] for k in (
                "canonical_task_universe_sha256", "content_task_ids_sha256",
                "n_expected", "n_kept") if k in prov
        },
    }


def collect(cells: list[dict], grade: str) -> dict:
    per_condition = {}
    pooled: collections.Counter = collections.Counter()
    excluded: collections.Counter = collections.Counter()
    missing = non_numeric = 0
    unresolved = []
    for c in cells:
        if c.get("grade") != grade:
            continue
        key = f"{c['baseline']}_{c['mode']}_{c['site']}"
        t = tally(c)
        if t is None:
            unresolved.append(key)
            continue
        per_condition[key] = t
        for k, v in t["value_counts"].items():
            pooled[float(k)] += v
        for k, v in t["protocol_excluded"]["value_counts"].items():
            excluded[float(k)] += v
        missing += t["n_score_missing"]
        non_numeric += t["n_score_non_numeric"]
    return {
        "grade": grade,
        "n_conditions_resolved": len(per_condition),
        "n_conditions_unresolved": len(unresolved),
        "unresolved": unresolved,
        "n_episodes_scored": sum(pooled.values()),
        "n_score_missing": missing,
        "n_score_non_numeric": non_numeric,
        "distinct_values": sorted(pooled),
        "value_counts": {str(k): pooled[k] for k in sorted(pooled)},
        "protocol_excluded": {
            "n_episodes": sum(excluded.values()),
            "distinct_values": sorted(excluded),
            "value_counts": {str(k): excluded[k] for k in sorted(excluded)},
        },
        "per_condition": per_condition,
    }


def render_md(pg: dict, arch: dict) -> str:
    vals = pg["distinct_values"]
    counts = pg["value_counts"]
    L = ["# Evaluator score granularity — landed Phase-1a conditions\n"]
    L.append("Regenerate: `python3 scripts/analysis/aggregate_evaluator_score_granularity.py`\n")
    L.append("Universe: `grade: paper-grade` cells of `results/phantom_paper/run_manifest.yaml`, "
             "restricted to the canonical SCORED task set so the denominator matches every "
             "other rate in the paper. AMENDMENT_08 protocol-excluded episodes are counted "
             "separately below rather than dropped.\n")
    L.append(f"\n**{pg['n_conditions_resolved']} conditions · "
             f"{pg['n_episodes_scored']} scored episodes.**\n")
    L.append(f"\nDistinct evaluator `score` values observed: **{len(vals)}** "
             f"({', '.join(f'{v:g}' for v in vals)}).\n")
    L.append("\n| score | episodes | share |")
    L.append("|---|---|---|")
    tot = pg["n_episodes_scored"]
    for v in vals:
        n = counts[str(v)]
        L.append(f"| {v:g} | {n} | {100.0 * n / tot:.2f}% |")
    if pg["n_score_missing"] or pg["n_score_non_numeric"]:
        L.append(f"\nEpisodes with no numeric score: {pg['n_score_missing']} missing, "
                 f"{pg['n_score_non_numeric']} non-numeric.")
    if pg["unresolved"]:
        L.append(f"\n⚠️ Unresolved conditions ({len(pg['unresolved'])}): "
                 + ", ".join(pg["unresolved"]))
    ex = pg["protocol_excluded"]
    L.append(f"\nProtocol-excluded episodes (AMENDMENT_08, reddit tasks 58 and 160): "
             f"{ex['n_episodes']}, distinct values "
             f"{', '.join(f'{v:g}' for v in ex['distinct_values']) or 'none'}. Counting them "
             f"in would give {tot + ex['n_episodes']} episodes and change no conclusion.\n")
    L.append("\nThe evaluator is binary, so no graded quality target exists to regress on. "
             "This is a property of the benchmark's evaluation design, not of our pipeline.\n")
    L.append(f"\nArchived (pre-fix) conditions, reported separately and never pooled into the "
             f"above: {arch['n_conditions_resolved']} conditions, "
             f"{arch['n_episodes_scored']} scored episodes, distinct values "
             f"{', '.join(f'{v:g}' for v in arch['distinct_values']) or 'none'}.\n")

    L.append("\n## Per condition\n")
    L.append("| condition | episodes | " + " | ".join(f"score {v:g}" for v in vals) + " |")
    L.append("|" + "---|" * (len(vals) + 2))
    for key in sorted(pg["per_condition"]):
        t = pg["per_condition"][key]
        row = " | ".join(str(t["value_counts"].get(str(v), 0)) for v in vals)
        L.append(f"| {key} | {t['n_scored']} | {row} |")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    cells = load_manifest()["cells"]
    pg = collect(cells, "paper-grade")
    arch = collect(cells, "archived")
    print(f"paper-grade: {pg['n_conditions_resolved']} conditions, "
          f"{pg['n_episodes_scored']} episodes, values {pg['distinct_values']}")
    print(f"archived:    {arch['n_conditions_resolved']} conditions, "
          f"{arch['n_episodes_scored']} episodes, values {arch['distinct_values']}")
    if pg["unresolved"]:
        print(f"⚠️ unresolved paper-grade conditions: {pg['unresolved']}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "producer": "scripts/analysis/aggregate_evaluator_score_granularity.py",
        "paper_grade": pg,
        "archived": arch,
    }
    (args.out_dir / "evaluator_score_granularity.json").write_text(
        json.dumps(payload, indent=1), encoding="utf-8")
    (args.out_dir / "evaluator_score_granularity.md").write_text(
        render_md(pg, arch), encoding="utf-8")
    print(f"wrote {args.out_dir}/evaluator_score_granularity.{{json,md}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
