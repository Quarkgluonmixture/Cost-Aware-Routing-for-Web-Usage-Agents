#!/usr/bin/env python3
"""Per (site, model, mode) aggregator for select_option dispatch telemetry.

B-482 (/stress A1.25 GRL Chunk 2 P0-2-AB* + P1-1-BC*, 2026-05-17): paper §3
evidence layer for the structured select_option dispatch return (B-481).
Pre-fix `step_record.select_option_meta.success` was overpromised — JS
`return;` for no-match / wrong-coord / no-label / successful-click all set
`success=True` because the only signal was "page.evaluate() did not throw".
This aggregator consumes the **post-B-481 structured fields** (`matched`,
`match_stage`, `target_type`, plus optional `selected_text_before/after`,
`clicked_text`) and computes the true ON_OPTION rate.

Reads B-450 split schema (`select_option_meta_primary` + `_retry`) when
present; falls back to legacy `select_option_meta` for archive rows from
before A1.25 GRL Chunk 2. Pre-B-481 rows (which lack `matched` /
`match_stage`) are surfaced under a `pre_b481_unknown_matched` bucket so
operators can see how much of an aggregate is in the legacy semantics.

Usage:
    python3 scripts/analysis/aggregate_select_option_dispatch.py \\
        --run-dir results/visualwebarena/phase1/<RUN_ID>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional


def _safe_load_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue


def _extract_primary_meta(step: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """B-450-aware: prefer `_primary`; fallback to legacy."""
    primary = step.get("select_option_meta_primary")
    if primary is not None:
        return primary if isinstance(primary, dict) else None
    legacy = step.get("select_option_meta")
    return legacy if isinstance(legacy, dict) else None


def _classify_step(step: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return classification info for a select_option step, or None if not one."""
    meta = _extract_primary_meta(step)
    if meta is None or meta.get("action_kind") != "select_option":
        return None
    return meta


def aggregate_run(run_dir: Path) -> Dict[str, Any]:
    cells: Dict[tuple, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for cond_dir in sorted(run_dir.iterdir()):
        if not cond_dir.is_dir():
            continue
        meta_path = cond_dir / "condition_meta.json"
        if not meta_path.exists():
            continue
        try:
            cmeta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        site = cmeta.get("benchmark_site") or cmeta.get("site") or "unknown"
        model = cmeta.get("backend_id") or "unknown"
        mode = cmeta.get("observation_mode") or "unknown"
        cell_key = (site, model, mode)

        episodes_dir = cond_dir / "episodes"
        if not episodes_dir.exists():
            continue
        for episode_dir in sorted(episodes_dir.iterdir()):
            if not episode_dir.is_dir():
                continue
            steps_jsonl = episode_dir / "steps.jsonl"
            for step in _safe_load_jsonl(steps_jsonl):
                meta = _classify_step(step)
                if meta is None:
                    continue
                bucket = cells[cell_key]
                bucket["invoked"] += 1

                dispatch_path = meta.get("dispatch_path")
                if dispatch_path == "missing_obs":
                    bucket["dispatched_missing_obs"] += 1
                elif dispatch_path == "element_id":
                    bucket["dispatched_element_id"] += 1
                elif dispatch_path == "coordinate":
                    bucket["dispatched_coordinate"] += 1

                # B-481 structured fields — pre-B-481 archive rows lack these.
                if "matched" not in meta:
                    bucket["pre_b481_unknown_matched"] += 1
                    # legacy `success=True` here meant "JS didn't throw",
                    # which we surface separately for archive transparency.
                    if meta.get("success") is True:
                        bucket["pre_b481_legacy_success_true"] += 1
                    continue
                if meta.get("matched") is True:
                    bucket["matched_true"] += 1
                    stage = meta.get("match_stage") or "unknown"
                    bucket[f"stage_{stage}"] += 1
                    target = meta.get("target_type") or "unknown"
                    bucket[f"target_{target}"] += 1
                else:
                    bucket["matched_false"] += 1
                    err = meta.get("error") or "unknown_error"
                    bucket[f"error_{err}"] += 1

    out: Dict[str, Any] = {"cells": []}
    for (site, model, mode), counts in sorted(cells.items()):
        invoked = counts.get("invoked", 0)
        matched = counts.get("matched_true", 0)
        match_rate = (matched / invoked) if invoked > 0 else None
        # fuzzy share = fraction of matched steps that took the 'fuzzy' tier
        # (vs exact / ci / index). Probes prompt-vs-runtime "exact-text"
        # contract drift per cell.
        fuzzy = counts.get("stage_fuzzy", 0)
        fuzzy_share = (fuzzy / matched) if matched > 0 else None
        out["cells"].append({
            "site": site,
            "model": model,
            "mode": mode,
            "invoked": invoked,
            "matched_true": counts.get("matched_true", 0),
            "matched_false": counts.get("matched_false", 0),
            "pre_b481_unknown_matched": counts.get("pre_b481_unknown_matched", 0),
            "pre_b481_legacy_success_true": counts.get("pre_b481_legacy_success_true", 0),
            "match_stage": {
                k.removeprefix("stage_"): v
                for k, v in counts.items() if k.startswith("stage_")
            },
            "target_type": {
                k.removeprefix("target_"): v
                for k, v in counts.items() if k.startswith("target_")
            },
            "error_taxonomy": {
                k.removeprefix("error_"): v
                for k, v in counts.items() if k.startswith("error_")
            },
            "dispatched": {
                "element_id": counts.get("dispatched_element_id", 0),
                "coordinate": counts.get("dispatched_coordinate", 0),
                "missing_obs": counts.get("dispatched_missing_obs", 0),
            },
            "match_rate": match_rate,
            "fuzzy_share_of_matched": fuzzy_share,
            "note": (
                "match_rate is the true ON_OPTION proxy (matched + dispatched); "
                "pre_b481_unknown_matched marks rows that predate the B-481 "
                "structured-return fix where `success=True` only meant 'JS did "
                "not throw'. Reviewer should treat unknown_matched > 0 as "
                "evidence-layer hole in that cell."
            ),
        })
    return out


def format_markdown(agg: Dict[str, Any]) -> str:
    lines = [
        "| Site | Model | Mode | Invoked | Matched | Unknown(pre-B481) | match_rate | fuzzy_share |",
        "|------|-------|------|---------|---------|-------------------|------------|-------------|",
    ]
    for cell in agg["cells"]:
        rate = cell["match_rate"]
        rate_str = f"{rate:.3f}" if rate is not None else "N/A"
        fshare = cell["fuzzy_share_of_matched"]
        fshare_str = f"{fshare:.3f}" if fshare is not None else "N/A"
        lines.append(
            f"| {cell['site']} | {cell['model']} | {cell['mode']} "
            f"| {cell['invoked']} | {cell['matched_true']} | {cell['pre_b481_unknown_matched']} "
            f"| {rate_str} | {fshare_str} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    if not args.run_dir.is_dir():
        print(f"ERROR: run-dir does not exist: {args.run_dir}", file=sys.stderr)
        return 2

    agg = aggregate_run(args.run_dir)
    if args.out_json:
        args.out_json.write_text(json.dumps(agg, indent=2), encoding="utf-8")
        print(f"Wrote JSON: {args.out_json}", file=sys.stderr)
    else:
        print(json.dumps(agg, indent=2))

    md = format_markdown(agg)
    if args.out_md:
        args.out_md.write_text(md, encoding="utf-8")
        print(f"Wrote markdown: {args.out_md}", file=sys.stderr)
    else:
        print("\n" + md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
