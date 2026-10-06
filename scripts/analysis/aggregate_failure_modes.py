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
# 2026-09-11: `B\d` so extension backbones (B5 = GPT-5.6, run_manifest `extension:`) parse
# too. Which runs are read is still decided by the registry, not by this regex; cells whose
# baseline is outside the preregistered set go to `extension_cells`, never to `cells`.
RUN_RE = re.compile(r"^(B\d)_(?:3mode_|phantom_[a-z]+_|[a-z]+_)?(classifieds|reddit|shopping)")


def output_section(source_runs, extension_run_names, baseline: str, prereg_baselines) -> str:
    """Which output key a cell belongs to: "cells" (preregistered set) or "extension_cells".

    Decided by manifest section when the registry was read: any source run registered under
    `extension:` sends the cell to `extension_cells`. Baseline alone is not enough — shopping
    B0/B1 (registered under `extension:` 2026-10-06, 实验笔记 §531.5) have preregistered
    baselines, and the old baseline-only rule put them into `cells`. Without a registry
    (dev-smoke glob fallback) the baseline rule is all there is.
    """
    if extension_run_names is not None:
        return "extension_cells" if any(r in extension_run_names for r in source_runs) else "cells"
    return "cells" if baseline in prereg_baselines else "extension_cells"


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
    # Insertion-ordered: element 0 is the run that supplies the cell (first-wins).
    seen_runs_per_cell: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    import sys as _sys

    if not PHASE1_DIR.exists():
        print(f"[failure_modes] phase1 dir missing at {PHASE1_DIR} — emitting empty output")
        result = {"cells": {}, "method": "no phase1 runs found",
                  "paper_taxonomy": {k: sorted(v) for k, v in PAPER_TAXONOMY.items()}}
        OUT_JSON.write_text(json.dumps(result, indent=2))
        OUT_MD.write_text("# Failure modes per cell\n\nNo phase1 runs found.\n")
        return

    _extension_run_names: set[str] | None = None  # None = registry not read (glob fallback)
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
        # include_extension: extension backbones (B5) are read here but written to the
        # separate `extension_cells` key below — `cells` keeps its prereg-only scope for
        # fig_failure_modes_per_cell + representation_deployment_profile.
        _registry_cells = _get_all_cells(grade_filter=["paper-grade"], include_extension=True)
        _default_run_names = {c.run_dir.name for c in _get_all_cells(grade_filter=["paper-grade"])}
        _extension_run_names = {c.run_dir.name for c in _registry_cells} - _default_run_names
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
            _extension_run_names = None
        else:
            print(f"[failure_modes] registry: {len(_unique_run_dirs)} unique paper-grade run_dirs",
                  file=_sys.stderr)
    except Exception as _reg_exc:
        import sys as _sys
        print(f"[failure_modes] WARN: registry lookup failed ({_reg_exc}), "
              "falling back to PHASE1_DIR.glob('B*')", file=_sys.stderr)
        _extension_run_names = None
        _unique_run_dirs = sorted([d for d in PHASE1_DIR.glob("B*") if d.is_dir()])

    # WebArena (2026-10-07, 实验笔记 §538): WA runs are not in run_manifest.yaml and their names
    # (B1_dom_wa_reddit_...) do not match RUN_RE, so they were never read. They come from the same
    # canonical-run resolver the four-dimension profile uses (glob minus registered replicates)
    # and go to their own `wa_cells` key; `cells` keeps its preregistered scope.
    from scripts.analysis.per_mode_four_dimension_profile import wa_spec
    _wa_runs: dict[str, tuple[str, str]] = {}
    for _wb in ("B0", "B1"):
        for _ep in wa_spec(_wb)["modes"].values():
            _rd = Path(_ep).parent.parent
            _wa_runs[_rd.name] = (_wb, "wa_reddit")
            _unique_run_dirs.append(_rd)

    _missing_csv: list[str] = []
    _unparsed: list[str] = []
    for run_dir in _unique_run_dirs:
        if not run_dir.is_dir():
            continue
        baseline, site = _wa_runs.get(run_dir.name) or parse_run(run_dir.name)
        if not baseline or not site:
            _unparsed.append(run_dir.name)
            continue
        cond_csv = run_dir / "analysis/reason_diagnostics/condition_reason_summary.csv"
        if not cond_csv.exists():
            _missing_csv.append(run_dir.name)
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
                # P1-8-A dedup: a cell is counted from the FIRST run that supplies it.
                # Fixed 2026-09-11: the old gate tested only the current run's own name, so a
                # second run of the same cell was SUMMED in, not skipped — moot while the
                # registry holds one run per cell, but the warning below claims first-wins.
                runs = seen_runs_per_cell[cell_key]
                if run_dir.name not in runs:
                    runs.append(run_dir.name)
                if runs[0] != run_dir.name:
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
        # (A post-loop pass used to mark this run on EVERY (baseline, site) cell seen so
        # far, so each sibling-mode run dir of a backbone looked like a second run of every
        # earlier mode → spurious P1-8-A warnings, e.g. B2/reddit/SoM "has" B2_vision_reddit.
        # Marking now happens per row, only on the cell the row belongs to.)
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
        "extension_cells": {},
        "wa_cells": {},
        "wa_note": (
            "WebArena reddit (B0, B1): canonical runs, registered replicates excluded. Same "
            "taxonomy; kept apart because WA is a different benchmark and is not in the "
            "preregistered cell set."
        ),
        "extension_note": (
            "Cells registered under run_manifest `extension:` — B5 = GPT-5.6 and the shopping "
            "site (B0/B1). Same taxonomy, kept out of `cells` so consumers scoped to the "
            "preregistered set (figures, deployment profile) are unchanged. Shopping caveat: "
            "B-2002 (search box submits old+new query) hits the text arms far more than Vision, "
            "so shopping mode-to-mode bucket differences are not clean (实验笔记 §531.9)."
        ),
    }
    try:
        from scripts.analysis.lib.run_registry import BASELINES as _PREREG_BASELINES
    except Exception:
        _PREREG_BASELINES = ["B0", "B1", "B2"]
    for ck, buckets in sorted(cells.items()):
        total = cell_totals[ck]
        failed = total - buckets.get("success", 0)
        bucket_pct = {}
        for b, c in buckets.items():
            if b == "success":
                continue
            bucket_pct[b] = {"count": c, "pct_of_failed": (c / failed * 100) if failed else 0.0,
                             "pct_of_total": (c / total * 100) if total else 0.0}
        target = (result["wa_cells"] if ck[1] == "wa_reddit" else
                  result[output_section(seen_runs_per_cell[ck], _extension_run_names,
                                        ck[0], _PREREG_BASELINES)])
        target[f"{ck[0]}/{ck[1]}/{ck[2]}"] = {
            "baseline": ck[0], "site": ck[1], "mode": ck[2],
            "total_episodes": total,
            "success_count": buckets.get("success", 0),
            "failed_count": failed,
            "buckets": bucket_pct,
            "source_runs": sorted(set(sources[ck])),
        }

    # --- fail loud on an empty product ------------------------------------------------
    # This script wrote a complete-looking, entirely empty document on exit 0 from
    # (at least) 2026-08-03 back, because every run_dir was skipped by the `cond_csv`
    # existence check above. A finished-looking artifact raises no questions, so the
    # emptiness survived three coverage sweeps. Refuse to write instead.
    import sys as _sys
    if _missing_csv:
        print(f"[failure_modes] WARN: {len(_missing_csv)}/{len(_unique_run_dirs)} run_dirs "
              f"have no analysis/reason_diagnostics/condition_reason_summary.csv "
              f"(e.g. {_missing_csv[0]}). Regenerate with: "
              f"make analyze RUN=results/.../<run>   (or analyze_reason_diagnostics.py "
              f"--run-dir <run>)", file=_sys.stderr)
    if _unparsed:
        print(f"[failure_modes] WARN: {len(_unparsed)} run_dir name(s) did not parse: "
              f"{_unparsed[:3]}", file=_sys.stderr)
    if not result["cells"]:
        raise SystemExit(
            f"[failure_modes] REFUSING to write an empty product: 0 cells from "
            f"{len(_unique_run_dirs)} paper-grade run_dirs "
            f"({len(_missing_csv)} missing reason_diagnostics, {len(_unparsed)} unparsed "
            f"names). The previous behaviour was to write a header-only markdown and "
            f"exit 0, which is how this went unnoticed. Fix the input, then rerun.")

    # 2026-10-07 (实验笔记 §533.2): the stderr WARN above was the only trace that B0/reddit/P-SoM
    # (R28173) had no reason summary — the cell was absent from 2026-08-03 to 10-07 and the
    # product read as complete. Missing runs now go into the product itself.
    result["missing_reason_diagnostics"] = sorted(_missing_csv)
    OUT_JSON.write_text(json.dumps(result, indent=2))
    print(f"[failure_modes] wrote {OUT_JSON}")

    # Build markdown
    md_lines = [
        "# Failure modes per cell (paper §5 — 5-bucket taxonomy)",
        "",
        "5-bucket paper taxonomy mapped from fine-grained reason_bucket "
        "(see `aggregate_failure_modes.py` PAPER_TAXONOMY).",
        "",
    ]
    if _missing_csv:
        md_lines += [f"⚠️ **{len(_missing_csv)} registered run(s) have no reason summary and are "
                     f"absent below**: " + ", ".join(f"`{r}`" for r in sorted(_missing_csv)), ""]
    md_lines += ["## Per-cell breakdown", ""]
    def _cell_md(ck, info):
        md_lines.append(f"### {ck} (N={info['total_episodes']}, failed={info['failed_count']})")
        md_lines.append("")
        md_lines.append("| Paper bucket | Count | % of failed | % of total |")
        md_lines.append("|---|---:|---:|---:|")
        for b in sorted(info["buckets"].keys()):
            bv = info["buckets"][b]
            md_lines.append(f"| {b} | {bv['count']} | {bv['pct_of_failed']:.1f}% | {bv['pct_of_total']:.1f}% |")
        md_lines.append("")

    for ck, info in sorted(result["cells"].items()):
        _cell_md(ck, info)
    if result["wa_cells"]:
        md_lines += ["## WebArena reddit", "", result["wa_note"], ""]
        for ck, info in sorted(result["wa_cells"].items()):
            _cell_md(ck, info)
    if result["extension_cells"]:
        md_lines += ["## Extension cells (outside the preregistered cell set)", "",
                     result["extension_note"], ""]
        for ck, info in sorted(result["extension_cells"].items()):
            _cell_md(ck, info)
    if unmapped_fine:
        md_lines.append("## Unmapped fine-grained buckets (catch-all)")
        md_lines.append("")
        for k, v in sorted(unmapped_fine.items()):
            md_lines.append(f"- `{k}`: {v}")
    OUT_MD.write_text("\n".join(md_lines))
    print(f"[failure_modes] wrote {OUT_MD}")


if __name__ == "__main__":
    main()
