#!/usr/bin/env python3
"""Per (site, model, mode) aggregator for locator-route dispatch telemetry.

B-448 (/stress A1.25 P0-1-ABC* + P1-7-B, 2026-05-17): paper §3 evidence layer
for the locator-route ON_TARGET fix (B-01/02/33 family). Pre-fix the
`step_record.locator_route_meta` field was written by runner but **no
aggregator computed per-cell rates** — the field was "存在主义" not
actually consumed. Codex Mode B P1-7 catch:
``rg locator_route_meta p79/experiment/metrics.py scripts/analysis``
returned nothing before this script existed.

Reads B-440 split schema (`locator_route_meta_primary` + `_retry`) when
present; falls back to legacy `locator_route_meta` for archive rows from
before A1.19. Outputs per (site, model, mode) JSON + Markdown table with:
- invoked (total steps where locator-route was invoked)
- walk_success (success=True)
- walk_fail (success=False, fallback to framework bbox-center)
- retry_overwritten (retry fired after primary walk-fail — historical
  archive only; post-A1.25 telemetry is split so this is informational)
- unconditional ON_TARGET proxy: P(walk_success) (lower bound; actual
  ON_TARGET requires gallery labeling which is out of scope here)

Usage:
    python3 scripts/analysis/aggregate_locator_route_metrics.py \\
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
    """Yield parsed JSON lines, skipping corrupt/empty lines."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue


def _extract_primary_meta(step: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """B-440-aware: prefer `_primary` field; fallback to legacy `locator_route_meta`."""
    primary = step.get("locator_route_meta_primary")
    if primary is not None:
        return primary if isinstance(primary, dict) else None
    legacy = step.get("locator_route_meta")
    return legacy if isinstance(legacy, dict) else None


def _classify_step(step: Dict[str, Any]) -> Optional[str]:
    """Return classification bucket or None if not a locator-route step.

    Buckets:
      - 'walk_success': primary meta has success=True
      - 'walk_fail':    primary meta has success=False
      - None:           step did not invoke locator-route (scroll/wait/coord-only)
    """
    meta = _extract_primary_meta(step)
    if meta is None:
        return None
    success = meta.get("success")
    if success is True:
        return "walk_success"
    if success is False:
        return "walk_fail"
    return None


def _retry_fired(step: Dict[str, Any]) -> bool:
    """Did the baseline_retry_on_no_progress path fire on this step?"""
    if step.get("locator_route_meta_retry") is not None:
        return True
    return bool(step.get("retry_action_applied"))


def aggregate_run(run_dir: Path) -> Dict[str, Any]:
    """Aggregate locator-route telemetry across all conditions in a run.

    Returns dict mapping (site, model, mode) → counts dict.
    """
    cells: Dict[tuple, Dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    # Iterate condition directories
    for cond_dir in sorted(run_dir.iterdir()):
        if not cond_dir.is_dir():
            continue
        meta_path = cond_dir / "condition_meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        site = meta.get("benchmark_site") or meta.get("site") or "unknown"
        model = meta.get("backend_id") or "unknown"
        mode = meta.get("observation_mode") or "unknown"
        cell_key = (site, model, mode)

        episodes_dir = cond_dir / "episodes"
        if not episodes_dir.exists():
            continue
        for episode_dir in sorted(episodes_dir.iterdir()):
            if not episode_dir.is_dir():
                continue
            steps_jsonl = episode_dir / "steps.jsonl"
            for step in _safe_load_jsonl(steps_jsonl):
                bucket = _classify_step(step)
                if bucket is None:
                    continue
                cells[cell_key][bucket] += 1
                cells[cell_key]["invoked"] += 1
                if _retry_fired(step):
                    cells[cell_key]["retry_overwritten"] += 1

    # Compute derived rates
    out: Dict[str, Any] = {"cells": []}
    for (site, model, mode), counts in sorted(cells.items()):
        invoked = counts["invoked"]
        ws = counts["walk_success"]
        wf = counts["walk_fail"]
        rate = (ws / invoked) if invoked > 0 else None
        out["cells"].append({
            "site": site,
            "model": model,
            "mode": mode,
            "invoked": invoked,
            "walk_success": ws,
            "walk_fail": wf,
            "retry_overwritten": counts["retry_overwritten"],
            "walk_success_rate": rate,
            "note": (
                "walk_success_rate is the success-conditional proxy for "
                "ON_TARGET; walk_fail cases fall back to framework "
                "mouse.click(bbox_center) = B-33 buggy path. Unconditional "
                "ON_TARGET requires gallery labeling (out of scope)."
            ),
        })
    return out


def format_markdown(agg: Dict[str, Any]) -> str:
    """Render aggregator output as a markdown table for paper §3.5."""
    lines = [
        "| Site | Model | Mode | Invoked | Walk-success | Walk-fail | Retry fired | walk_success_rate |",
        "|------|-------|------|---------|--------------|-----------|-------------|-------------------|",
    ]
    for cell in agg["cells"]:
        rate = cell["walk_success_rate"]
        rate_str = f"{rate:.3f}" if rate is not None else "N/A"
        lines.append(
            f"| {cell['site']} | {cell['model']} | {cell['mode']} "
            f"| {cell['invoked']} | {cell['walk_success']} | {cell['walk_fail']} "
            f"| {cell['retry_overwritten']} | {rate_str} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="Path to a run directory containing condition subdirs")
    parser.add_argument("--out-json", type=Path, default=None,
                        help="Write aggregator JSON to this path (default: stdout)")
    parser.add_argument("--out-md", type=Path, default=None,
                        help="Write markdown table to this path (default: stdout)")
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
