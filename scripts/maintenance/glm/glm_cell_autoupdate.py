#!/usr/bin/env python3
"""Cell frontmatter auto-update — sync _status/cells/*.md from condition_summary_v2.json.

For each cell note, find matching VWA run via frontmatter (baseline + site + mode),
parse latest condition_summary_v2.json, update structured fields in cell frontmatter:
- status: active → done if episodes complete, active if running
- progress: ep_count / N * 100
- sr_raw: success_rate * 100 (rounded 2 decimals)
- pid: cleared if cell done

Adj_sr / drop_one updates require `make analysis` cross-condition output (TODO future).

Run via cron @5min or `make glm-update-cells` ad-hoc.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import yaml

REPO = Path(__file__).resolve().parents[3]
STATUS_CELLS = REPO / "docs/checkpoints/_status/cells"
PHASE1_DIR = REPO / "results/visualwebarena/phase1"

# Mode (frontmatter) → condition_id substring
MODE_TO_COND = {
    "DOM": "dom",
    "SoM": "som",
    "Vision": "vision",
    "P-text": "phantom_text",
    "P-SoM": "phantom_som",
    "P-prompt": "phantom_prompt",
}
# Site (frontmatter) → run_dir site segment
SITE_NORM = {"classifieds": "classifieds", "reddit": "reddit", "shopping": "shopping"}
# Expected N per site
EXPECTED_N = {"classifieds": 234, "reddit": 210, "shopping": 466}


def parse_frontmatter(text: str) -> tuple[Optional[dict], str, str]:
    """Returns (fm_dict, fm_raw, body) or (None, '', text) if no frontmatter."""
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not m:
        return None, "", text
    fm_raw = m.group(1)
    body = m.group(2)
    try:
        fm = yaml.safe_load(fm_raw) or {}
    except yaml.YAMLError:
        fm = None
    return fm, fm_raw, body


def serialize_frontmatter(fm: dict) -> str:
    return yaml.safe_dump(fm, allow_unicode=True, sort_keys=False, default_flow_style=False).strip()


def find_matching_runs(baseline: str, site: str, mode: str) -> list[Path]:
    """Find condition_summary_v2.json matching this (baseline, site, mode)."""
    site_seg = SITE_NORM.get(site, site)
    cond_keyword = MODE_TO_COND.get(mode, mode.lower())

    matches = []
    if not PHASE1_DIR.exists():
        return matches
    for run_dir in PHASE1_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        if not run_dir.name.startswith(baseline + "_"):
            continue
        if site_seg not in run_dir.name:
            continue
        for cond_dir in run_dir.iterdir():
            if not cond_dir.is_dir():
                continue
            summary = cond_dir / "condition_summary_v2.json"
            if not summary.exists():
                continue
            # Match condition by observation_mode field (more reliable than dir name)
            try:
                d = json.loads(summary.read_text(encoding="utf-8"))
            except Exception:
                continue
            obs_mode = d.get("observation_mode", "")
            if obs_mode == cond_keyword or cond_keyword in cond_dir.name:
                matches.append(summary)
    return matches


def latest_summary(paths: list[Path]) -> Optional[Path]:
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def update_cell(cell_path: Path, dry_run: bool = False, force: bool = False) -> tuple[bool, str]:
    """Returns (updated, message).

    Safety: cells with status=active AND pid set are SKIPPED (assume in-flight
    run not yet written condition_summary; auto-update would overwrite with
    stale archived run data). Override with --force.
    """
    text = cell_path.read_text(encoding="utf-8")
    fm, fm_raw, body = parse_frontmatter(text)
    if fm is None:
        return False, "no frontmatter"

    baseline = fm.get("baseline")
    site = fm.get("site")
    mode = fm.get("mode")
    if not (baseline and site and mode):
        return False, "missing baseline/site/mode"

    # Safety: skip active+pid cells (in-flight, would match stale archived run)
    if not force and fm.get("status") == "active" and fm.get("pid"):
        return False, f"skip (active, pid={fm['pid']}; use --force to override)"

    matches = find_matching_runs(baseline, site, mode)
    summary_path = latest_summary(matches)
    if not summary_path:
        return False, f"no matching run for {baseline}/{site}/{mode}"

    try:
        d = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"summary parse error: {e}"

    episodes = d.get("episodes", 0)
    expected_n = EXPECTED_N.get(site, fm.get("n", 234))
    sr = d.get("success_rate")

    new_fm = dict(fm)
    changed_fields = []

    progress_pct = round(100 * episodes / expected_n) if expected_n else 0
    if new_fm.get("progress") != progress_pct:
        new_fm["progress"] = progress_pct
        changed_fields.append(f"progress→{progress_pct}")

    if episodes >= expected_n and new_fm.get("status") != "done":
        new_fm["status"] = "done"
        new_fm.pop("pid", None)  # clear PID on done
        changed_fields.append("status→done")
    elif episodes < expected_n and new_fm.get("status") == "pending":
        new_fm["status"] = "active"
        changed_fields.append("status→active")

    if sr is not None:
        sr_pct = round(sr * 100, 2)
        if new_fm.get("sr_raw") != sr_pct:
            new_fm["sr_raw"] = sr_pct
            changed_fields.append(f"sr_raw→{sr_pct}")

    new_fm["n"] = expected_n

    if not changed_fields:
        return False, "no change"

    if dry_run:
        return True, f"would update: {', '.join(changed_fields)}"

    new_text = "---\n" + serialize_frontmatter(new_fm) + "\n---\n" + body
    cell_path.write_text(new_text, encoding="utf-8")
    return True, f"updated: {', '.join(changed_fields)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="actually write (default dry-run)")
    parser.add_argument("--force", action="store_true", help="overwrite active+pid cells too")
    parser.add_argument("--cell", type=Path, help="single cell file to update")
    args = parser.parse_args()
    dry_run = not args.apply

    cells = [args.cell] if args.cell else sorted(STATUS_CELLS.glob("cell_*.md"))
    print(f"📋 Scanning {len(cells)} cell notes" + (" (dry-run)" if dry_run else " (APPLY)"))

    n_updated = 0
    for cell in cells:
        updated, msg = update_cell(cell, dry_run=dry_run, force=args.force)
        marker = "✏️" if updated else "  "
        print(f"  {marker} {cell.name}: {msg}")
        if updated:
            n_updated += 1

    print(f"\n{'Would update' if dry_run else 'Updated'} {n_updated}/{len(cells)} cells.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
