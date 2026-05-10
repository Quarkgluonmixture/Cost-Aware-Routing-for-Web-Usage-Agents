#!/usr/bin/env python3
"""Cross-cell failure-mode bucket aggregator (paper §5 evidence).

Walks all phase1 runs under `results/visualwebarena/phase1/`, reads
`<run>/analysis/reason_diagnostics/condition_reason_summary.csv`, and maps
the fine-grained reason buckets (fail_early_finish / fail_max_steps_* /
etc.) into the **5-bucket paper-grade taxonomy** documented in
`docs/analysis/phantom_paper/phantom_dom_vs_som_diagnostic.md` §4:

  - early-finish/wrong-commit
  - search-loop
  - visual-hijack/click-loop
  - element-misground
  - missing-context
  - (max-steps-other + error catch-alls reported separately)

Output:
  docs/analysis/cross_sites/failure_modes_per_cell.json
  docs/analysis/cross_sites/failure_modes_per_cell.md

Cell key: (baseline, site, mode). Baseline + site derived from run_id
prefix (e.g. `B0_phantom_som_reddit_20260428` → B0 / reddit), mode from
condition_id pattern `phase1_<mode>_router_*`.
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PHASE1_DIR = ROOT / "results/visualwebarena/phase1"
OUT_JSON = ROOT / "docs/analysis/cross_sites/failure_modes_per_cell.json"
OUT_MD = ROOT / "docs/analysis/cross_sites/failure_modes_per_cell.md"


# Map fine-grained reason bucket → 5-bucket paper taxonomy
PAPER_TAXONOMY = {
    "early-finish/wrong-commit": {
        "fail_early_finish",
        "fail_finish_eval_mismatch",
        "fail_finish_wrong_url_not_found",
        "fail_finish_wrong_url_left_target",
        "fail_finish_wrong_url_price_mismatch",
        "fail_finish_claim_missing",
        "fail_finish_empty_answer",
    },
    "search-loop": {"fail_max_steps_search_repeat"},
    "visual-hijack/click-loop": {"fail_max_steps_click_back_loop"},
    "element-misground": {"fail_max_steps_target_unreachable"},
    "missing-context": {"fail_no_progress", "fail_incomplete_or_stuck"},
    "max-steps-other": {"fail_max_steps"},
    "error/noise": {
        "fail_env_error",
        "fail_parse_error",
        "fail_summary_error",
        "fail_benchmark_noise",
    },
}

ALL_FINE_BUCKETS_IN_TAXONOMY = {b for s in PAPER_TAXONOMY.values() for b in s}


def fine_to_paper(fine: str) -> str:
    for paper_bucket, fine_set in PAPER_TAXONOMY.items():
        if fine in fine_set:
            return paper_bucket
    return "other-failure"


# baseline + site from run_id
RUN_RE = re.compile(r"^(B[01])_(?:3mode_|phantom_[a-z]+_|[a-z]+_)?(classifieds|reddit|shopping)")


def parse_run(run_id: str):
    m = RUN_RE.match(run_id)
    if not m:
        return None, None
    return m.group(1), m.group(2)


# condition_id → mode
COND_RE = re.compile(r"^phase1_([a-z_]+)_router_\d+$")
COND_MODE_MAP = {
    "dom": "DOM",
    "som": "SoM",
    "vision": "Vision",
    "phantom_som": "P-SoM",
    "phantom_text": "P-text",
    "phantom_prompt": "P-prompt",
}


def parse_cond(cond_id: str):
    m = COND_RE.match(cond_id)
    if not m:
        return None
    return COND_MODE_MAP.get(m.group(1))


def main():
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    # cells[(baseline, site, mode)] = {paper_bucket: count, ...}
    cells: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    # cell_totals[(baseline, site, mode)] = total episodes (including success)
    cell_totals: dict[tuple[str, str, str], int] = defaultdict(int)
    sources: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    unmapped_fine: dict[str, int] = defaultdict(int)

    if not PHASE1_DIR.exists():
        print(f"[failure_modes] phase1 dir missing at {PHASE1_DIR} — emitting empty output")
        result = {"cells": {}, "method": "no phase1 runs found",
                  "paper_taxonomy": {k: sorted(v) for k, v in PAPER_TAXONOMY.items()}}
        OUT_JSON.write_text(json.dumps(result, indent=2))
        OUT_MD.write_text("# Failure modes per cell\n\nNo phase1 runs found.\n")
        return

    for run_dir in sorted(PHASE1_DIR.glob("B*")):
        if not run_dir.is_dir():
            continue
        baseline, site = parse_run(run_dir.name)
        if not baseline or not site:
            continue
        cond_csv = run_dir / "analysis/reason_diagnostics/condition_reason_summary.csv"
        if not cond_csv.exists():
            continue
        with cond_csv.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                cond_id = row.get("condition_id", "")
                mode = parse_cond(cond_id)
                if not mode:
                    continue
                bucket_fine = row.get("reason_bucket", "")
                try:
                    count = int(row.get("count", 0))
                except ValueError:
                    continue
                if count <= 0:
                    continue
                cell_key = (baseline, site, mode)
                cell_totals[cell_key] += count
                if bucket_fine == "success":
                    cells[cell_key]["success"] += count
                    continue
                paper_bucket = fine_to_paper(bucket_fine)
                if paper_bucket == "other-failure":
                    unmapped_fine[bucket_fine] += count
                cells[cell_key][paper_bucket] += count
                sources[cell_key].append(run_dir.name)

    # Build output JSON
    result = {
        "method": "fine-grained reason_bucket → 5-bucket paper taxonomy (taxonomy doc: docs/analysis/phantom_paper/phantom_dom_vs_som_diagnostic.md §4)",
        "paper_taxonomy": {k: sorted(v) for k, v in PAPER_TAXONOMY.items()},
        "unmapped_fine_buckets": dict(sorted(unmapped_fine.items())),
        "cells": {},
    }
    for ck, buckets in sorted(cells.items()):
        total = cell_totals[ck]
        failed = total - buckets.get("success", 0)
        bucket_pct = {}
        for b, c in buckets.items():
            if b == "success":
                continue
            bucket_pct[b] = {"count": c, "pct_of_failed": (c / failed * 100) if failed else 0.0,
                             "pct_of_total": (c / total * 100) if total else 0.0}
        result["cells"][f"{ck[0]}/{ck[1]}/{ck[2]}"] = {
            "baseline": ck[0], "site": ck[1], "mode": ck[2],
            "total_episodes": total,
            "success_count": buckets.get("success", 0),
            "failed_count": failed,
            "buckets": bucket_pct,
            "source_runs": sorted(set(sources[ck])),
        }

    OUT_JSON.write_text(json.dumps(result, indent=2))
    print(f"[failure_modes] wrote {OUT_JSON}")

    # Build markdown
    md_lines = [
        "# Failure modes per cell (paper §5 — 5-bucket taxonomy)",
        "",
        "5-bucket paper taxonomy mapped from fine-grained reason_bucket "
        "(see `aggregate_failure_modes.py` PAPER_TAXONOMY).",
        "",
        "## Per-cell breakdown",
        "",
    ]
    for ck, info in sorted(result["cells"].items()):
        md_lines.append(f"### {ck} (N={info['total_episodes']}, failed={info['failed_count']})")
        md_lines.append("")
        md_lines.append("| Paper bucket | Count | % of failed | % of total |")
        md_lines.append("|---|---:|---:|---:|")
        for b in sorted(info["buckets"].keys()):
            bv = info["buckets"][b]
            md_lines.append(f"| {b} | {bv['count']} | {bv['pct_of_failed']:.1f}% | {bv['pct_of_total']:.1f}% |")
        md_lines.append("")
    if unmapped_fine:
        md_lines.append("## Unmapped fine-grained buckets (catch-all)")
        md_lines.append("")
        for k, v in sorted(unmapped_fine.items()):
            md_lines.append(f"- `{k}`: {v}")
    OUT_MD.write_text("\n".join(md_lines))
    print(f"[failure_modes] wrote {OUT_MD}")


if __name__ == "__main__":
    main()
