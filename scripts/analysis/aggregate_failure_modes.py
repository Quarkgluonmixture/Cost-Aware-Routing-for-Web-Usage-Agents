#!/usr/bin/env python3
"""Cross-cell failure-mode bucket aggregator (paper §5 evidence).

Walks all phase1 runs under `results/visualwebarena/phase1/`, reads
`<run>/analysis/reason_diagnostics/condition_reason_summary.csv`, and maps
the fine-grained reason buckets (fail_early_finish / fail_max_steps_* /
etc.) into a **7-bucket paper-grade taxonomy** (5 main + 2 catch-alls)
documented in `docs/analysis/phantom_paper/phantom_dom_vs_som_diagnostic.md` §4:

  Core 5 (paper §5 main figure rows):
  - early-finish/wrong-commit
  - search-loop
  - visual-hijack/click-loop
  - element-misground
  - missing-context

  Catch-all 2 (paper §5 appendix transparency rows):
  - max-steps-other     (fail_max_steps not matched to specific behavioral bucket)
  - error/noise         (env/parse/summary/benchmark_noise infrastructure failures)

  Plus dynamic `other-failure` row for any fine-grained bucket not in PAPER_TAXONOMY
  (catch-all-of-catch-alls; should be empty on paper-grade data, surfaces taxonomy
  drift if non-empty).

/stress A1.19 P1-4-AC (2026-05-17, Claude+Gemini overlap): pre-fix docstring +
filename + paper §5 prose said "5-bucket taxonomy" but PAPER_TAXONOMY dict had 7
keys + `other-failure` catch-all = 8 effective buckets. Reviewer 5-vs-7 mismatch.
Fix: docstring + code now explicit "5 core + 2 catch-alls + 1 dynamic" (the prose
in paper §5 will be reconciled in next codex round per Q11=C bottom-tier default).

Output:
  docs/analysis/cross_sites/failure_modes_per_cell.json
  docs/analysis/cross_sites/failure_modes_per_cell.md

Cell key: (baseline, site, mode). Baseline + site derived from run_id
prefix (e.g. `B0_phantom_som_reddit_20260428` → B0 / reddit), mode from
condition_id pattern `phase1_<mode>_router_*`.

/stress A1.19 P1-8-A (2026-05-17, Claude): multi-rerun dedup. Pre-fix `RUN_RE`
matched baseline+site prefix only; same (baseline, site, mode) cell with multiple
paper-grade runs (rerun, B-184 lock cycle) was counted ADDITIVELY → failure-mode
distribution silently inflated 1.5-2× across reruns. `source_runs.append` tracked
runs but no dedup gate on cell_totals. Fix: per-cell `seen_runs: set[str]` guard
skips already-counted runs and surfaces a stderr warning so user audits the
manifest before paper §5 prose locks.
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
        # P1-5 (/stress accounting audit 2026-05-21, B-1782): off-site goto blocked
        # by the VWA-origin whitelist — a model-emitted off-policy action, sibling
        # of parse_error (protocol failure). Member of existing bucket → no 5+2
        # taxonomy-count drift.
        "fail_policy_blocked_offsite",
    },
}

ALL_FINE_BUCKETS_IN_TAXONOMY = {b for s in PAPER_TAXONOMY.values() for b in s}


def fine_to_paper(fine: str) -> str:
    for paper_bucket, fine_set in PAPER_TAXONOMY.items():
        if fine in fine_set:
            return paper_bucket
    return "other-failure"


# B-297 fix (2026-05-16, A1.8): regex `B[01]` previously skipped B2 (Gemma3-VL,
# added 2026-05-14 per advisor) → B2 failure data structurally vanished from
# cross-site evidence. `B[0-2]` includes all 3 baselines.
RUN_RE = re.compile(r"^(B[0-2])_(?:3mode_|phantom_[a-z]+_|[a-z]+_)?(classifieds|reddit|shopping)")


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
    # B-297 fix (2026-05-16, A1.8): legacy alias `phantom_dom` maps to P-text
    # for archive backward-compat (B-261 fix retired phantom_dom obs_mode but
    # 3 existing run dirs are still named `phase1_phantom_dom_router_0/`).
    "phantom_dom": "P-text",
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
    # /stress A1.19 P1-8-A (2026-05-17): per-cell seen-runs dedup to prevent
    # multi-rerun additive counting. If same (baseline, site, mode) has >1 paper-grade
    # run dir on disk (B-184 rerun cycles), pre-fix double-counted episodes →
    # failure_count silently inflated. Now skip already-counted run + stderr warn.
    seen_runs_per_cell: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    import sys as _sys

    if not PHASE1_DIR.exists():
        print(f"[failure_modes] phase1 dir missing at {PHASE1_DIR} — emitting empty output")
        result = {"cells": {}, "method": "no phase1 runs found",
                  "paper_taxonomy": {k: sorted(v) for k, v in PAPER_TAXONOMY.items()}}
        OUT_JSON.write_text(json.dumps(result, indent=2))
        OUT_MD.write_text("# Failure modes per cell\n\nNo phase1 runs found.\n")
        return

    # C4 fix 2026-05-24: replace PHASE1_DIR.glob("B*") with registry lookup.
    # Pre-fix: glob picked up ALL run dirs including pre-bug/archived/in-flight
    # runs that should not enter paper §5 failure-mode distribution. Registry
    # get_all_cells(grade_filter=["paper-grade"]) returns only manifest-promoted
    # paper-grade cells — the same gate used by all other paper aggregators.
    # Derive unique run_dirs from the registry; fall back to glob if manifest
    # missing (dev smoke use-case).
    try:
        import sys as _sys
        sys_path_backup = list(_sys.path)
        _sys.path.insert(0, str(ROOT))
        from scripts.analysis.lib.run_registry import get_all_cells as _get_all_cells
        _registry_cells = _get_all_cells(grade_filter=["paper-grade"])
        # Collect unique run dirs that have a condition_reason_summary.csv
        _registry_run_dirs: dict[str, tuple[str, str]] = {}  # run_dir.name → (baseline, site)
        for _cs in _registry_cells:
            _run_dirs_key = _cs.run_dir.name
            if _run_dirs_key not in _registry_run_dirs:
                _registry_run_dirs[_run_dirs_key] = (_cs.baseline, _cs.site)
        _candidate_dirs = sorted(
            [_cs.run_dir for _cs in _registry_cells],
            key=lambda p: p.name,
        )
        # Deduplicate — multiple cells share the same run_dir (one per mode)
        _seen_paths: set[Path] = set()
        _unique_run_dirs: list[Path] = []
        for _p in _candidate_dirs:
            if _p not in _seen_paths:
                _seen_paths.add(_p)
                _unique_run_dirs.append(_p)
        if not _unique_run_dirs:
            print("[failure_modes] WARN: registry returned 0 paper-grade run_dirs — "
                  "falling back to PHASE1_DIR.glob('B*') for dev smoke", file=_sys.stderr)
            _unique_run_dirs = sorted([d for d in PHASE1_DIR.glob("B*") if d.is_dir()])
        else:
            print(f"[failure_modes] registry: {len(_unique_run_dirs)} unique paper-grade run_dirs",
                  file=_sys.stderr)
    except Exception as _reg_exc:
        import sys as _sys
        print(f"[failure_modes] WARN: registry lookup failed ({_reg_exc}), "
              "falling back to PHASE1_DIR.glob('B*')", file=_sys.stderr)
        _unique_run_dirs = sorted([d for d in PHASE1_DIR.glob("B*") if d.is_dir()])

    for run_dir in _unique_run_dirs:
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
                # P1-8-A dedup: per-cell unique run_dir gate.
                if run_dir.name in seen_runs_per_cell[cell_key]:
                    # Already counted this cell's episodes from a prior row in same run
                    # OR (the bug case) from a sibling run that produced identical cell.
                    continue
                cell_totals[cell_key] += count
                if bucket_fine == "success":
                    cells[cell_key]["success"] += count
                    continue
                paper_bucket = fine_to_paper(bucket_fine)
                if paper_bucket == "other-failure":
                    unmapped_fine[bucket_fine] += count
                cells[cell_key][paper_bucket] += count
                sources[cell_key].append(run_dir.name)
        # Mark this run as counted for all cells it contributed to during this file pass.
        # (Run-level mark applied after row loop so all (baseline, site, mode) keys
        # within this run_dir's csv are recorded.)
        for cell_key in list(cell_totals.keys()):
            if cell_key[0] == baseline and cell_key[1] == site:
                if run_dir.name not in seen_runs_per_cell[cell_key]:
                    seen_runs_per_cell[cell_key].add(run_dir.name)
    # Surface multi-rerun warning so user can audit:
    multi_run_cells = {
        ck: runs for ck, runs in seen_runs_per_cell.items() if len(runs) > 1
    }
    if multi_run_cells:
        for ck, runs in multi_run_cells.items():
            print(
                f"[failure_modes] WARN P1-8-A: cell {ck[0]}/{ck[1]}/{ck[2]} has "
                f"{len(runs)} paper-grade runs on disk ({sorted(runs)}); "
                f"counts come from FIRST encountered; audit run_manifest for "
                f"the canonical paper-grade run.",
                file=_sys.stderr,
            )

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
