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
# Expected N keyed by (benchmark, site) — VWA reddit 210 ≠ WA reddit 106.
# §139.8: scored counts (total − N/A tasks excluded at load time) from the
# single source of truth. Pre-exclusion: vwa 234/210/466, wa 106/192/182.
from p79.experiment.analysis import scored_task_count as _scored_task_count

_BENCH_FULL = {"vwa": "visualwebarena", "wa": "webarena"}
EXPECTED_N = {
    (_b, _s): _scored_task_count(_s, _BENCH_FULL[_b])
    for _b, _s in (
        ("vwa", "classifieds"), ("vwa", "reddit"), ("vwa", "shopping"),
        ("wa", "reddit"), ("wa", "shopping"), ("wa", "shopping_admin"),
    )
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


def find_matching_runs(baseline: str, site: str, mode: str, benchmark: str = "vwa") -> list[dict]:
    """Find run condition dirs matching (baseline, site, mode, benchmark).

    Site is anchored against `_<site>_<8-digit>` to avoid `shopping` substring
    collision with `shopping_admin` run dirs.

    Returns list of dicts (one per matched condition dir), each with:
      - cond_dir: Path to condition directory
      - summary_path: Path to condition_summary_v2.json, or None if not yet generated
      - run_id: outer run dir name (e.g. B1_phantom_prompt_classifieds_20260501)
      - observation_mode: extracted from summary if available, else from condition_meta.json
      - episode_count: count of episode_*_summary_v2.json files in episodes/
        (proxy for in-flight progress when full summary not yet generated)
      - is_inflight: True if no condition_summary_v2.json but episodes are accumulating

    Includes BOTH finalized runs (with summary) and in-flight runs (only
    condition_meta.json + partial episodes/). Caller (update_cell) handles each case.
    """
    site_seg = SITE_NORM.get(site, site)
    cond_keywords = MODE_TO_COND.get(mode, [mode.lower()])
    phase_dir = PHASE1_DIRS.get(benchmark)

    matches: list[dict] = []
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
            meta = cond_dir / "condition_meta.json"

            # Resolve observation_mode from summary (preferred) or condition_meta.
            obs_mode = ""
            if summary.exists():
                try:
                    d = json.loads(summary.read_text(encoding="utf-8"))
                    obs_mode = d.get("observation_mode", "")
                except Exception:
                    pass
            if not obs_mode and meta.exists():
                try:
                    md = json.loads(meta.read_text(encoding="utf-8"))
                    obs_mode = md.get("observation_mode", md.get("mode", ""))
                except Exception:
                    pass

            # Filter by mode. Prefer strict obs_mode equality (canonical field
            # in summary/meta). Fallback to cond_dir mode-segment strict
            # equality only when obs_mode is unavailable — substring match is
            # unsafe ("dom" ⊂ "phantom_dom" caused DOM cells to mis-match
            # phantom_dom runs).
            if obs_mode:
                if obs_mode not in cond_keywords:
                    continue
            else:
                m = re.match(r"phase1_(.+)_router_\d+$", cond_dir.name)
                cond_seg = m.group(1) if m else ""
                if cond_seg not in cond_keywords:
                    continue

            # Count in-flight progress via episodes/ dir
            episodes_dir = cond_dir / "episodes"
            episode_count = 0
            if episodes_dir.exists():
                episode_count = sum(1 for _ in episodes_dir.glob("*_summary_v2.json"))

            # Skip empty-scaffolded conditions (condition_meta.json present but
            # no real runner activity — typically launch-prepared-then-cancelled).
            # Real in-flight: at least one *_steps_v2.jsonl exists (runner has
            # started writing step data, even before first episode summary).
            # ep_count=0 alone is insufficient — runner mid-task-0 has steps
            # but no ep summary yet.
            if not summary.exists() and episode_count == 0:
                has_steps = episodes_dir.exists() and any(
                    episodes_dir.glob("*_steps_v2.jsonl")
                )
                if not has_steps:
                    continue

            matches.append({
                "cond_dir": cond_dir,
                "summary_path": summary if summary.exists() else None,
                "run_id": run_dir.name,
                "observation_mode": obs_mode,
                "episode_count": episode_count,
                "is_inflight": not summary.exists() and episode_count > 0,
            })
    return matches


def latest_match(matches: list[dict]) -> Optional[dict]:
    """Pick the most-recent match.

    B-910 (/stress A2.2 P1-11-B* codex F5 OOB, 2026-05-17): in-flight re-run
    prefer when mtime newer than finalized. Pre-fix sort key was
    `(1, summary_mtime)` always-beats `(0, cond_dir_mtime)` regardless of
    actual mtime ordering — so after a cell finalized and operator immediately
    re-fired a new in-flight run, the OLD finalized summary kept winning the
    latest_match() until the new run finalized. The frontmatter window
    `status=done, pid=None` persisted through the very window most likely to
    trigger same-site collision detection failure (manual rescue / master
    orchestrator both see "done, no pid" and may attempt new launches).

    New rule: prefer in-flight run when its cond_dir mtime is strictly newer
    than the latest finalized summary mtime — that's the "operator just
    re-fired" signal. Otherwise keep original finalized-prefer ordering.

    Combined with B-905 P79_CHAIN_PID env-bypass hardening + B-907 watchdog
    flock + B-906 GLM hook sleep 300 covers reddit cold-start: the re-run
    detection window is now correct at the GLM cron read layer too.
    """
    if not matches:
        return None

    finalized = [m for m in matches if m.get("summary_path")]
    inflight = [m for m in matches if not m.get("summary_path")]

    if finalized and inflight:
        latest_final_mtime = max(m["summary_path"].stat().st_mtime for m in finalized)
        latest_inflight = max(inflight, key=lambda m: m["cond_dir"].stat().st_mtime)
        latest_inflight_mtime = latest_inflight["cond_dir"].stat().st_mtime
        if latest_inflight_mtime > latest_final_mtime:
            # Operator just re-fired — prefer the in-flight match (B-910).
            return latest_inflight

    # Legacy behavior: finalized-prefer + mtime fallback.
    def sort_key(m: dict):
        if m["summary_path"]:
            return (1, m["summary_path"].stat().st_mtime)
        return (0, m["cond_dir"].stat().st_mtime)

    return max(matches, key=sort_key)


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
    match = latest_match(matches)
    if not match:
        return False, f"no matching run for {benchmark}/{baseline}/{site}/{mode}"

    expected_n = EXPECTED_N.get((benchmark, site), fm.get("n", 234))
    new_run_id = match["run_id"]
    prev_run_id = fm.get("last_run_id")
    is_new_run = bool(prev_run_id) and prev_run_id != new_run_id

    # Resolve episodes + sr from finalized summary (preferred) or in-flight episodes/ count.
    if match["summary_path"]:
        try:
            d = json.loads(match["summary_path"].read_text(encoding="utf-8"))
        except Exception as e:
            return False, f"summary parse error: {e}"
        episodes = d.get("episodes", 0)
        sr = d.get("success_rate")
    else:
        # In-flight: derived progress from episodes/ count, no aggregate sr yet
        episodes = match["episode_count"]
        sr = None

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
        # Audit (B) 2026-05-09: fire `make analysis FAST=1` in background
        # when a cell flips active→done so paper figures stay synced
        # with the latest paper-grade cell. Best-effort; never blocks
        # cron tick. Log under logs/cron/post_finalize_analysis.log.
        try:
            repo_root = cell_path.resolve().parents[3]
            log_path = repo_root / "logs" / "cron" / "post_finalize_analysis.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            cell_id = cell_path.stem
            with log_path.open("a") as _lf:
                _lf.write(
                    f"[{datetime.now(timezone.utc).isoformat()}] "
                    f"trigger make analysis FAST=1 (cell {cell_id} done)\n"
                )
            subprocess.Popen(
                ["nohup", "make", "-C", str(repo_root), "analysis", "FAST=1"],
                stdout=log_path.open("a"),
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            changed_fields.append("trigger:make-analysis-FAST")
        except Exception as _e:
            # Cron never blocks on best-effort hooks.
            print(f"[autoupdate] make analysis trigger failed: {_e}",
                  file=sys.stderr)
    elif episodes < expected_n and new_fm.get("status") == "pending":
        new_fm["status"] = "active"
        changed_fields.append("status→active")
        # Try to recover PID for first-time pending→active flip
        recovered_pid = detect_pid(new_run_id)
        if recovered_pid:
            new_fm["pid"] = recovered_pid
            changed_fields.append(f"pid→{recovered_pid}")

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

    # B-869 (/stress A1.23 P1-13 C, 2026-05-17): .git/index.lock pre-check.
    # Obsidian Git plugin auto-pulls every 10min on Windows side; if cron
    # tick lands during git's index update, our write races the merge →
    # git can inject `<<<<<<< HEAD ... =======` conflict markers into YAML
    # frontmatter → next parse_frontmatter / yaml.safe_load silently fails
    # → cell.md permanently corrupt. Skip this tick, retry next cron round.
    _git_lock = REPO / ".git" / "index.lock"
    if _git_lock.exists():
        return False, f"skip (git index.lock present at {_git_lock}; Obsidian Git pull in flight)"

    # B-853 (A1.15b Chunk γ P2-6): atomic frontmatter write. Pre-fix used
    # direct `write_text()` which is NOT atomic — cron tick crash mid-write
    # leaves cell .md half-written (invalid YAML). Obsidian Bases YAML parse
    # silently drops the row → cells.base shows wrong/missing state →
    # operator/user makes wrong launch decision. Now: write to temp file in
    # same dir + os.replace() atomic rename (POSIX semantics).
    #
    # B-860 (/stress A1.23 P0-3 AB* OOB, 2026-05-17): add `fcntl.flock(LOCK_EX)`
    # so this cron writer is serialized vs concurrent writers from
    # `auto_pull_myriad_cell.sh:228-244` (GONE-event-triggered cell quarantine
    # frontmatter update). Pre-fix B-853 closed within-process torn-write race,
    # but two separate processes (cron tick + GONE-event auto_pull) could
    # still race the tmp+rename sequence (tmp file collision, partial
    # writes visible via inotify by Obsidian). flock on a sibling lockfile
    # (not the cell.md itself — we re-replace the inode atomically) provides
    # cross-process mutual exclusion. Plus `_fsync_dir(parent)` ensures
    # the dir-entry rename hits stable storage (mirror logger_v2 B-198).
    import fcntl
    new_text = "---\n" + serialize_frontmatter(new_fm) + "\n---\n" + body
    _tmp_path = cell_path.with_suffix(cell_path.suffix + ".tmp")
    _lock_path = cell_path.with_suffix(cell_path.suffix + ".lock")
    try:
        with open(_lock_path, "w") as _lock_f:
            fcntl.flock(_lock_f.fileno(), fcntl.LOCK_EX)
            try:
                # Re-check .git/index.lock under lock (another writer may have
                # passed pre-check but git pull races caught it mid-write).
                if _git_lock.exists():
                    return False, f"skip (git index.lock appeared mid-tick)"
                _tmp_path.write_text(new_text, encoding="utf-8")
                os.replace(_tmp_path, cell_path)
                # B-860: fsync dir entry so rename hits stable storage.
                try:
                    _fd = os.open(str(cell_path.parent), os.O_RDONLY)
                    try:
                        os.fsync(_fd)
                    finally:
                        os.close(_fd)
                except OSError:
                    pass  # platform doesn't support dir fsync; not fatal
            finally:
                fcntl.flock(_lock_f.fileno(), fcntl.LOCK_UN)
    except Exception:
        # Cleanup temp file on any error; preserve original cell content.
        try:
            _tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise
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
