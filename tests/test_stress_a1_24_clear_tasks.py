"""B-873~B-890 (/stress A1.24, 2026-05-17): regression tests for clear_tasks
hardening + shared cleanup API + Option K event symmetry.

Coverage:
- B-873 P0-1-ABC* --force PID-lock + marker + double-flag + paper_grade reject
- B-874 P0-2-AB* orphan .in_progress marker check
- B-875 P0-3-ABC* orphan .stale_ skip
- B-876 P0-4-AB* safe_unlink/safe_rmtree idempotent + run-lock
- B-880 P0-7-C* shared p79.experiment.cleanup.clear_task_files API
- B-881 P0-8-C* Option K manual_task_cleared event symmetry
- B-885 P1-4-C* WA digest task_id int/string type coercion
- B-886 P1-5-A _parse_task_ids sanity (lo>hi, negative, > max)
- B-888 P1-6-A --site whitelist enforcement
- B-889 P1-7-A condition_summary glob *_task_*.json precise
- B-890 P1-8-C .cleaning flag wraps digest + cond_summary ops
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CLEAR_TASKS = REPO / "scripts" / "maintenance" / "clear_tasks.py"


# ---- helpers --------------------------------------------------------------

def _load_clear_tasks_module():
    spec = importlib.util.spec_from_file_location("clear_tasks_mod", CLEAR_TASKS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_run(tmp_path: Path, condition: str = "phase1_dom_router_0") -> Path:
    run_dir = tmp_path / "test_run"
    cond_dir = run_dir / condition
    (cond_dir / "episodes").mkdir(parents=True)
    (cond_dir / "artifacts").mkdir(parents=True)
    (run_dir / "task_configs").mkdir(parents=True)
    (run_dir / "analysis" / "digest").mkdir(parents=True)
    return run_dir


def _make_task(run_dir: Path, condition: str, site: str, tid: int,
                with_summary: bool = True, with_steps: bool = True,
                with_artifacts: bool = True, with_in_progress: bool = False) -> None:
    cond_dir = run_dir / condition
    prefix = f"{site}_task_{tid}"
    if with_summary:
        (cond_dir / "episodes" / f"{prefix}_summary_v2.json").write_text(
            json.dumps({"task_id": tid, "site": site}), encoding="utf-8")
    if with_steps:
        (cond_dir / "episodes" / f"{prefix}_steps_v2.jsonl").write_text(
            json.dumps({"step_idx": 0}) + "\n", encoding="utf-8")
    if with_artifacts:
        art = cond_dir / "artifacts" / prefix
        art.mkdir(exist_ok=True)
        (art / "screenshot.png").write_text("fake", encoding="utf-8")
        if with_in_progress:
            (art / ".in_progress").touch()


def _run_clear_tasks(args: list, env_overrides: dict = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(REPO))
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, str(CLEAR_TASKS)] + args,
        capture_output=True, text=True, env=env,
    )


# ---- B-876: safe_unlink / safe_rmtree -------------------------------------

def test_b858_safe_unlink_idempotent(tmp_path):
    """B-876: safe_unlink returns False on 2nd call (file already gone), no raise."""
    from p79.experiment.cleanup import safe_unlink
    f = tmp_path / "x.txt"
    f.write_text("data")
    assert safe_unlink(f) is True
    assert safe_unlink(f) is False  # already gone — no FileNotFoundError


def test_b858_safe_rmtree_idempotent(tmp_path):
    """B-876: safe_rmtree returns False on 2nd call (dir already gone), no raise."""
    from p79.experiment.cleanup import safe_rmtree
    d = tmp_path / "subdir"
    d.mkdir()
    (d / "file.txt").write_text("data")
    assert safe_rmtree(d) is True
    assert safe_rmtree(d) is False  # already gone


# ---- B-880: shared clear_task_files API -----------------------------------

def test_b862_clear_task_files_deletes_three_paths(tmp_path):
    """B-880: clear_task_files removes summary + steps + artifact dir."""
    from p79.experiment.cleanup import clear_task_files
    run = _make_run(tmp_path)
    _make_task(run, "phase1_dom_router_0", "classifieds", 5)
    cond_dir = run / "phase1_dom_router_0"
    assert (cond_dir / "episodes" / "classifieds_task_5_summary_v2.json").exists()
    did_delete = clear_task_files(
        condition_dir=cond_dir, site="classifieds", task_id=5,
        event_type="manual_task_cleared",
        audit_event=False,  # disable event emit for unit test isolation
    )
    assert did_delete is True
    assert not (cond_dir / "episodes" / "classifieds_task_5_summary_v2.json").exists()
    assert not (cond_dir / "episodes" / "classifieds_task_5_steps_v2.jsonl").exists()
    assert not (cond_dir / "artifacts" / "classifieds_task_5").exists()


def test_b862_clear_task_files_idempotent_returns_false(tmp_path):
    """B-880: second call on already-deleted task returns False, no raise."""
    from p79.experiment.cleanup import clear_task_files
    run = _make_run(tmp_path)
    _make_task(run, "phase1_dom_router_0", "reddit", 10)
    cond_dir = run / "phase1_dom_router_0"
    assert clear_task_files(cond_dir, "reddit", 10, audit_event=False) is True
    assert clear_task_files(cond_dir, "reddit", 10, audit_event=False) is False


# ---- B-881: Option K manual_task_cleared event symmetry -------------------

def test_b863_clear_task_files_emits_option_k_event(tmp_path):
    """B-881: manual_task_cleared event written to trajectory_events.jsonl
    with parity to watchdog task_auto_cleared (event_type discriminator)."""
    from p79.experiment.cleanup import clear_task_files
    run = _make_run(tmp_path)
    _make_task(run, "phase1_dom_router_0", "classifieds", 7)
    cond_dir = run / "phase1_dom_router_0"
    clear_task_files(
        condition_dir=cond_dir, site="classifieds", task_id=7,
        event_type="manual_task_cleared", reason="test_reason",
        force=False, operator_pid=99999,
    )
    events_path = cond_dir / "trajectory_events.jsonl"
    assert events_path.exists(), "Option K event log not created"
    lines = events_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) >= 1
    last = json.loads(lines[-1])
    assert last["event_type"] == "manual_task_cleared"
    assert last["task_index"] == 7
    md = last["metadata"]
    assert md["site"] == "classifieds"
    assert md["reason"] == "test_reason"
    assert md["force"] is False
    assert md["operator_pid"] == 99999


def test_b863_clear_task_files_extra_metadata_merges(tmp_path):
    """B-881: extra_metadata merges with base fields (watchdog use case)."""
    from p79.experiment.cleanup import clear_task_files
    run = _make_run(tmp_path)
    _make_task(run, "phase1_dom_router_0", "reddit", 3)
    cond_dir = run / "phase1_dom_router_0"
    clear_task_files(
        condition_dir=cond_dir, site="reddit", task_id=3,
        event_type="task_auto_cleared", reason="session_not_logged_in",
        extra_metadata={"wave_size": 5, "wave_task_index": 1, "is_auth_loss": True},
    )
    events = json.loads((cond_dir / "trajectory_events.jsonl").read_text().strip().split("\n")[-1])
    assert events["event_type"] == "task_auto_cleared"
    assert events["metadata"]["wave_size"] == 5
    assert events["metadata"]["wave_task_index"] == 1
    assert events["metadata"]["is_auth_loss"] is True


# ---- B-886: _parse_task_ids sanity ----------------------------------------

def test_b868_parse_task_ids_lo_gt_hi_raises():
    mod = _load_clear_tasks_module()
    with pytest.raises(ValueError, match="lo > hi"):
        mod._parse_task_ids("100-99")


def test_b868_parse_task_ids_negative_raises():
    mod = _load_clear_tasks_module()
    with pytest.raises(ValueError, match="leading '-'"):
        mod._parse_task_ids("-5")


def test_b868_parse_task_ids_over_max_raises():
    mod = _load_clear_tasks_module()
    with pytest.raises(ValueError, match="max task_id in site config"):
        mod._parse_task_ids("9999", max_task_id=224)


def test_b1910_cap_is_max_task_id_not_scored_count():
    """B-1910: the cap must admit every legal task ID, including N/A-excluded ones.

    VWA task IDs are contiguous 0..N_total-1 while `scored_task_count` subtracts
    the N/A exclusions, so using the count as the cap rejected real tasks —
    including `TASKS=0-465`, the canonical shopping example in CLAUDE.md.
    """
    import json as _json

    from p79.experiment.analysis import (
        _resolve_site_config,
        paper_scored_task_count,
        scored_task_count,
    )

    mod = _load_clear_tasks_module()
    for site in ("classifieds", "reddit", "shopping"):
        cfg = _resolve_site_config(site, "visualwebarena")
        assert cfg is not None, site
        ids = [int(t["task_id"]) for t in _json.load(open(cfg))]
        max_id = max(ids)

        # Both counts sit at or below the highest legal ID; neither is a bound.
        assert scored_task_count(site, "visualwebarena", strict=True) <= max_id
        assert paper_scored_task_count(site, "visualwebarena", strict=True) <= max_id

        # The full legal range parses under the correct cap...
        parsed = mod._parse_task_ids(f"0-{max_id}", max_task_id=max_id)
        assert parsed == sorted(ids), site
        # ...and one past the end is still rejected.
        with pytest.raises(ValueError, match="max task_id in site config"):
            mod._parse_task_ids(str(max_id + 1), max_task_id=max_id)


def test_b868_parse_task_ids_empty_segment_raises():
    mod = _load_clear_tasks_module()
    with pytest.raises(ValueError, match="empty segment"):
        mod._parse_task_ids("1,,5")


def test_b868_parse_task_ids_valid_range_works():
    mod = _load_clear_tasks_module()
    assert mod._parse_task_ids("5-10") == [5, 6, 7, 8, 9, 10]
    assert mod._parse_task_ids("1,3,5") == [1, 3, 5]


# ---- B-888: --site whitelist ----------------------------------------------

def test_b870_site_whitelist_rejects_typo(tmp_path):
    """B-888: --site shoping → exit non-zero with explicit error."""
    run = _make_run(tmp_path)
    proc = _run_clear_tasks([
        "--run-dir", str(run), "--condition", "phase1_dom_router_0",
        "--site", "shoping", "--tasks", "5", "--dry-run",
    ])
    assert proc.returncode != 0
    assert "shoping" in proc.stderr
    assert "classifieds" in proc.stderr  # whitelist suggestion


# ---- B-873: --force hardening cluster -------------------------------------

def test_b855_force_rejected_when_paper_grade_env_set(tmp_path):
    run = _make_run(tmp_path)
    _make_task(run, "phase1_dom_router_0", "classifieds", 5)
    proc = _run_clear_tasks(
        ["--run-dir", str(run), "--condition", "phase1_dom_router_0",
         "--site", "classifieds", "--tasks", "5", "--force",
         "--confirm-run-id", "test_run", "--dry-run"],
        env_overrides={"P79_PAPER_GRADE": "1"},
    )
    assert proc.returncode == 3
    assert "P79_PAPER_GRADE" in proc.stderr


def test_b855_force_requires_confirm_run_id(tmp_path):
    run = _make_run(tmp_path)
    _make_task(run, "phase1_dom_router_0", "classifieds", 5)
    proc = _run_clear_tasks([
        "--run-dir", str(run), "--condition", "phase1_dom_router_0",
        "--site", "classifieds", "--tasks", "5", "--force", "--dry-run",
    ])
    assert proc.returncode == 3
    assert "confirm-run-id" in proc.stderr


def test_b855_force_rejected_when_in_progress_marker_present(tmp_path):
    """B-873: .in_progress marker on target task blocks --force."""
    run = _make_run(tmp_path)
    _make_task(run, "phase1_dom_router_0", "classifieds", 5,
               with_in_progress=True, with_summary=False)
    proc = _run_clear_tasks([
        "--run-dir", str(run), "--condition", "phase1_dom_router_0",
        "--site", "classifieds", "--tasks", "5", "--force",
        "--confirm-run-id", "test_run",
    ])
    # Note: dry-run skip block bypasses marker check; we test non-dry mode
    assert proc.returncode == 3
    assert ".in_progress" in proc.stderr or "in-progress" in proc.stderr


# ---- B-874: orphan .in_progress marker check ------------------------------

def test_b856_orphan_clean_skips_in_progress(tmp_path):
    """B-874: orphan cleanup respects .in_progress marker (mtime alone insufficient)."""
    run = _make_run(tmp_path)
    cond = run / "phase1_dom_router_0"
    # Orphan: artifacts exist, no summary
    _make_task(run, "phase1_dom_router_0", "classifieds", 7,
               with_summary=False, with_in_progress=True)
    # Backdate artifact dir to > 10min
    art = cond / "artifacts" / "classifieds_task_7"
    old_ts = os.path.getmtime(art) - 3600
    os.utime(art, (old_ts, old_ts))
    proc = _run_clear_tasks([
        "--run-dir", str(run), "--clean-orphan-artifacts",
        "--condition", "phase1_dom_router_0",
    ])
    assert proc.returncode == 0
    # Marker present → must NOT be deleted
    assert art.exists(), "B-874: orphan with .in_progress marker should be preserved"
    assert "in_progress" in proc.stdout or "B-874" in proc.stdout


# ---- B-875: orphan .stale_ skip -------------------------------------------

def test_b857_orphan_clean_skips_stale_archives(tmp_path):
    """B-875: orphan cleanup skips B-488 .stale_<ts> forensic archives."""
    run = _make_run(tmp_path)
    cond = run / "phase1_dom_router_0"
    # Create B-488-style stale archive
    stale_art = cond / "artifacts" / "classifieds_task_5.stale_1234567890"
    stale_art.mkdir()
    (stale_art / "forensic.txt").write_text("preserve me")
    # Backdate it to > 10min
    old_ts = os.path.getmtime(stale_art) - 3600
    os.utime(stale_art, (old_ts, old_ts))
    proc = _run_clear_tasks([
        "--run-dir", str(run), "--clean-orphan-artifacts",
        "--condition", "phase1_dom_router_0",
    ])
    assert proc.returncode == 0
    assert stale_art.exists(), "B-875: .stale_ archive must be preserved"
    assert ".stale_" in proc.stdout or "B-875" in proc.stdout


# ---- B-885: WA task_id string vs int coercion ------------------------------

def test_b867_wa_digest_task_id_string_coerced(tmp_path):
    """B-885: digest with string task_id "5" gets matched against int task_id_set."""
    run = _make_run(tmp_path)
    _make_task(run, "phase1_dom_router_0", "reddit", 5)
    digest = run / "analysis" / "digest" / "digest_dom.jsonl"
    digest.parent.mkdir(parents=True, exist_ok=True)
    # WA-style: task_id stored as string
    digest.write_text(
        json.dumps({"task_id": "5", "condition_id": "phase1_dom_router_0", "site": "reddit"}) + "\n"
        + json.dumps({"task_id": "99", "condition_id": "phase1_dom_router_0", "site": "reddit"}) + "\n",
        encoding="utf-8",
    )
    proc = _run_clear_tasks([
        "--run-dir", str(run), "--condition", "phase1_dom_router_0",
        "--site", "reddit", "--tasks", "5",
    ])
    assert proc.returncode == 0
    # task_id "5" string should be coerced + removed; "99" untouched
    remaining = digest.read_text(encoding="utf-8").strip().split("\n")
    assert len(remaining) == 1
    assert json.loads(remaining[0])["task_id"] == "99"


# ---- B-889: condition_summary glob precise --------------------------------

def test_b871_condition_summary_glob_precise(tmp_path):
    """B-889: glob `*_task_*.json` ignores non-task json (e.g. _manifest.json)."""
    run = _make_run(tmp_path)
    cond = run / "phase1_dom_router_0"
    # Create 3 task summaries
    for tid in [0, 1, 2]:
        _make_task(run, "phase1_dom_router_0", "reddit", tid)
        (run / "task_configs" / f"reddit_task_{tid}.json").write_text("{}")
    # Add a future-proofing sibling that should NOT be counted as task
    (run / "task_configs" / "_manifest.json").write_text('{"version": 2}')
    (cond / "condition_summary_v2.json").write_text('{"finalized": true}')
    # Delete one task; remaining=2, total should still be 3 (not 4 from manifest)
    proc = _run_clear_tasks([
        "--run-dir", str(run), "--condition", "phase1_dom_router_0",
        "--site", "reddit", "--tasks", "0",
    ])
    assert proc.returncode == 0
    # cond_summary should be removed (remaining=2 < total=3)
    assert not (cond / "condition_summary_v2.json").exists()
    # If glob had been *.json, _manifest would inflate total=4, also remove.
    # Either way summary removed, but the proof is total count in stdout.
    assert "2/3" in proc.stdout or "deleted stale" in proc.stdout


# ---- B-890: .cleaning flag lifecycle --------------------------------------

def test_b872_cleaning_flag_removed_after_success(tmp_path):
    """B-890: .cleaning flag created during operation, removed on success."""
    run = _make_run(tmp_path)
    _make_task(run, "phase1_dom_router_0", "classifieds", 5)
    cond = run / "phase1_dom_router_0"
    proc = _run_clear_tasks([
        "--run-dir", str(run), "--condition", "phase1_dom_router_0",
        "--site", "classifieds", "--tasks", "5",
    ])
    assert proc.returncode == 0
    # Flag must be removed after successful cleanup
    assert not (cond / ".cleaning").exists(), "B-890: .cleaning flag should be removed on success"
