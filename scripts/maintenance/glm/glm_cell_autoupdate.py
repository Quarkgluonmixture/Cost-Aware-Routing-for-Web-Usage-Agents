#!/usr/bin/env python3
"""Cell frontmatter auto-update — sync _status/cells/*.md from condition_summary_v2.json.

For each cell note, find matching VWA run via frontmatter (baseline + site + mode),
parse latest condition_summary_v2.json, update structured fields in cell frontmatter:
- status: active → done if episodes complete; done → active if NEW run detected
  (re-run support); pending → active when first run starts
- progress: ep_count / N * 100
- sr_raw: success_rate * 100 (rounded 2 decimals)
- last_run_id: run dir name of latest summary (re-run detection key)
- pid: cleared if cell done; auto-detected from pgrep on re-run start
- history: append-only list of {run_id, finalized_at, ep, sr_raw} on completion

Adj_sr / drop_one updates require `make analysis` cross-condition output (TODO future).

Writes a changelog line per change to logs/cron/cell_changelog.jsonl
(consumed by glm_playbook_refresh §2 automation status synthesis).

Run via cron @10min or `make glm-update-cells` ad-hoc.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml


def _pid_alive(pid) -> bool:
    """Liveness probe: signal 0 raises ProcessLookupError if PID gone, no actual signal sent."""
    if pid is None:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # PID exists, owned by another user
    except (OSError, ValueError):
        return False

REPO = Path(__file__).resolve().parents[3]
STATUS_CELLS = REPO / "docs/checkpoints/_status/cells"
PHASE1_DIRS = {
    "vwa": REPO / "results/visualwebarena/phase1",
    "wa": REPO / "results/webarena/phase1",
}
CHANGELOG = REPO / "logs/cron/cell_changelog.jsonl"

# Mode (frontmatter) → list of condition_id keywords (first = canonical, rest = legacy alias).
# P-text ↔ phantom_dom is legacy mode value (paper-facing renamed 2026-04-29);
# pre-rename runs still have observation_mode="phantom_dom" / cond_dir "phase1_phantom_dom_router_0".
MODE_TO_COND = {
    "DOM": ["dom"],
    "SoM": ["som"],
    "Vision": ["vision"],
    "P-text": ["phantom_text", "phantom_dom"],
    "P-SoM": ["phantom_som"],
    "P-prompt": ["phantom_prompt"],
}
# Site (frontmatter) → run_dir site segment
SITE_NORM = {
    "classifieds": "classifieds",
    "reddit": "reddit",
    "shopping": "shopping",
    "shopping_admin": "shopping_admin",
}
# Expected N keyed by (benchmark, site) — VWA reddit 210 ≠ WA reddit 106
EXPECTED_N = {
    ("vwa", "classifieds"): 234,
    ("vwa", "reddit"): 210,
    ("vwa", "shopping"): 466,
    ("wa", "reddit"): 106,
    ("wa", "shopping"): 192,
    ("wa", "shopping_admin"): 182,
}


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


def find_matching_runs(baseline: str, site: str, mode: str, benchmark: str = "vwa") -> list[Path]:
    """Find condition_summary_v2.json matching (baseline, site, mode, benchmark).

    Site is anchored against `_<site>_<8-digit>` to avoid `shopping` substring
    collision with `shopping_admin` run dirs.
    """
    site_seg = SITE_NORM.get(site, site)
    cond_keywords = MODE_TO_COND.get(mode, [mode.lower()])
    phase_dir = PHASE1_DIRS.get(benchmark)

    matches = []
    if phase_dir is None or not phase_dir.exists():
        return matches
    site_pat = re.compile(rf"_{re.escape(site_seg)}_\d{{8}}")
    for run_dir in phase_dir.iterdir():
        if not run_dir.is_dir():
            continue
        if not run_dir.name.startswith(baseline + "_"):
            continue
        if not site_pat.search(run_dir.name):
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
            if obs_mode in cond_keywords or any(k in cond_dir.name for k in cond_keywords):
                matches.append(summary)
    return matches


def latest_summary(paths: list[Path]) -> Optional[Path]:
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def detect_pid(run_id: str) -> Optional[int]:
    """Try to recover PID for a re-run by pgrep matching condition run_id segment.

    On re-run, runner spawns a new process whose argv contains the run dir name
    (or a config that resolves to it). Best-effort match; returns None if unsure.
    """
    try:
        out = subprocess.run(
            ["pgrep", "-af", "run_experiment"],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return None
    for line in out.stdout.splitlines():
        if run_id and run_id in line:
            try:
                return int(line.split(None, 1)[0])
            except (ValueError, IndexError):
                pass
    return None


def append_changelog(cell_name: str, changes: list[str]) -> None:
    """Append a JSONL row per cron tick that produced changes."""
    if not changes:
        return
    try:
        CHANGELOG.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "cell": cell_name,
            "changes": changes,
        }
        with CHANGELOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass  # changelog best-effort, never break the cron job


def update_cell(cell_path: Path, dry_run: bool = False, force: bool = False) -> tuple[bool, str]:
    """Returns (updated, message).

    Safety: cells with status=active AND pid set are SKIPPED (assume in-flight
    run not yet written condition_summary; auto-update would overwrite with
    stale archived run data). Override with --force.

    Re-run detection: if latest run_dir.name != stored last_run_id, treat as
    fresh re-run — flip status done→active, attempt to detect PID via pgrep,
    and append previous (run_id, sr_raw) to history before overwriting.
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

    # Safety: skip active+pid cells if PID is alive (in-flight, would match stale archived run).
    # If PID is set but dead (runner died/crashed/finished without clearing), proceed with update
    # and clear the stale pid below — prevents cron from skipping forever.
    pid_was_dead = None
    if not force and fm.get("status") == "active" and fm.get("pid"):
        if _pid_alive(fm["pid"]):
            return False, f"skip (active, pid={fm['pid']} alive; use --force to override)"
        pid_was_dead = fm["pid"]

    benchmark = fm.get("benchmark", "vwa")
    matches = find_matching_runs(baseline, site, mode, benchmark)
    summary_path = latest_summary(matches)
    if not summary_path:
        return False, f"no matching run for {benchmark}/{baseline}/{site}/{mode}"

    try:
        d = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"summary parse error: {e}"

    episodes = d.get("episodes", 0)
    expected_n = EXPECTED_N.get((benchmark, site), fm.get("n", 234))
    sr = d.get("success_rate")
    new_run_id = summary_path.parent.parent.name  # results/.../<run_id>/<cond>/condition_summary_v2.json
    prev_run_id = fm.get("last_run_id")
    is_new_run = bool(prev_run_id) and prev_run_id != new_run_id

    new_fm = dict(fm)
    changed_fields = []

    # Clear stale pid if liveness check above flagged it as dead
    if pid_was_dead is not None:
        new_fm.pop("pid", None)
        changed_fields.append(f"pid_dead({pid_was_dead})_cleared")

    # Re-run detected: archive prior canonical sr_raw to history, flip done→active
    if is_new_run and fm.get("status") == "done":
        history = list(fm.get("history") or [])
        history.append({
            "run_id": prev_run_id,
            "finalized_at": fm.get("finalized_at"),
            "sr_raw": fm.get("sr_raw"),
            "ep": fm.get("n"),
        })
        new_fm["history"] = history
        new_fm["status"] = "active"
        new_fm.pop("finalized_at", None)
        changed_fields.append(f"rerun_detected({prev_run_id[:12]}→{new_run_id[:12]})")
        changed_fields.append("status→active")
        # Attempt PID recovery
        recovered_pid = detect_pid(new_run_id)
        if recovered_pid:
            new_fm["pid"] = recovered_pid
            changed_fields.append(f"pid→{recovered_pid}")

    if new_fm.get("last_run_id") != new_run_id:
        new_fm["last_run_id"] = new_run_id
        if not is_new_run:  # avoid double-logging for re-run case
            changed_fields.append("last_run_id→" + new_run_id[:24])

    progress_pct = round(100 * episodes / expected_n) if expected_n else 0
    if new_fm.get("progress") != progress_pct:
        new_fm["progress"] = progress_pct
        changed_fields.append(f"progress→{progress_pct}")

    if episodes >= expected_n and new_fm.get("status") != "done":
        new_fm["status"] = "done"
        new_fm["finalized_at"] = datetime.now(timezone.utc).date().isoformat()
        new_fm.pop("pid", None)
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
    append_changelog(cell_path.name, changed_fields)
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
