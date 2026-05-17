"""Regression tests for /stress A1.23 concurrency + race contract fixes.

Cycle 2026-05-17:
- 3-AI hostile reviewer audit (Claude Mode A / codex Mode B / gemini Mode C)
- 14 fixes B-858 ~ B-871 (5 P0 + 7 P1 + 2 P2; P1-11 GLMM covariate dropped
  per user 2026-05-17: "我没有人为 race-induced contamination")
- Option B runtime singleton lease dropped per user decision Q2=A; only
  Option A SIGTERM handler landed for P0-2.

Tests verify invariants exposed by the fixes:
- P0-1 leaf cross-mode collision check helper exists + integrated in 4 leaf scripts
- P0-2 SIGTERM handler converts to KeyboardInterrupt
- P0-3 cell frontmatter atomic write + flock + git-lock-aware
- P0-4 myriad auto_pull failure markers + paper_grade env propagation
- P0-5 auth storage_state atomic write helper
- P1-6 deletion-intent rename in cleanup module + watchdog hooks
- P1-7 docker reset timeout uses setsid + --kill-after
- P1-8 myriad_watcher STATE_FILE atomic write
- P1-9 mid-run staging re-merge counter
- P1-10 phantom siblings have dirty-cell FATAL block
- P1-12 stale-resume fingerprint includes paper_grade env
- P1-13 cron .git/index.lock detection
- P2-14 auto_pull yaml.safe_dump
- P2-15 watchdog DOM prefix 15000 char
"""

from __future__ import annotations

import inspect
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# P0-1 (B-858) — leaf cross-mode collision check
# ---------------------------------------------------------------------------


def test_b858_assert_no_cross_mode_collision_helper_exists():
    """The helper `assert_no_cross_mode_collision` lives in
    `_lib_paper_grade_gates.sh` and is callable by 4 leaf scripts."""
    lib_path = REPO / "scripts" / "queues" / "_lib_paper_grade_gates.sh"
    content = lib_path.read_text(encoding="utf-8")
    assert "assert_no_cross_mode_collision()" in content, (
        "B-858 helper missing from _lib_paper_grade_gates.sh"
    )
    assert "B-858" in content
    assert "A1.23 P0-1" in content


@pytest.mark.parametrize("leaf_script", [
    "queue_baseline.sh",
    "queue_phantom_som.sh",
    "queue_phantom_text.sh",
    "queue_phantom_prompt.sh",
])
def test_b858_leaf_invokes_collision_check(leaf_script):
    """Every leaf script (baseline + 3 phantom siblings) invokes the helper
    after the RUN_ID-specific pgrep fall-through."""
    leaf_path = REPO / "scripts" / "queues" / leaf_script
    content = leaf_path.read_text(encoding="utf-8")
    assert "assert_no_cross_mode_collision " in content, (
        f"B-858 helper not invoked in {leaf_script}"
    )


# ---------------------------------------------------------------------------
# P0-2 (B-859) — SIGTERM handler
# ---------------------------------------------------------------------------


def test_b859_runner_imports_signal():
    """runner/main.py must import signal module at top-level for SIGTERM
    handler registration in `run()`."""
    runner_path = REPO / "p79" / "experiment" / "runner" / "main.py"
    content = runner_path.read_text(encoding="utf-8")
    assert "import signal" in content, "B-859 missing 'import signal'"


def test_b859_sigterm_handler_registered_in_run():
    """`run()` method registers SIGTERM handler raising KeyboardInterrupt."""
    runner_path = REPO / "p79" / "experiment" / "runner" / "main.py"
    content = runner_path.read_text(encoding="utf-8")
    assert "signal.signal(signal.SIGTERM" in content
    assert "B-859" in content
    assert "raise KeyboardInterrupt" in content


# ---------------------------------------------------------------------------
# P0-3 (B-860) + P1-13 (B-869) — cell frontmatter atomic + flock + git-lock
# ---------------------------------------------------------------------------


def test_b860_glm_cell_autoupdate_uses_flock():
    """glm_cell_autoupdate.py uses fcntl.flock(LOCK_EX) when writing cell
    frontmatter."""
    p = REPO / "scripts" / "maintenance" / "glm" / "glm_cell_autoupdate.py"
    content = p.read_text(encoding="utf-8")
    assert "fcntl.flock" in content, "B-860 missing fcntl.flock"
    assert "LOCK_EX" in content


def test_b869_glm_cell_autoupdate_detects_git_index_lock():
    """glm_cell_autoupdate.py skips writes when .git/index.lock present
    (Obsidian Git pull in flight)."""
    p = REPO / "scripts" / "maintenance" / "glm" / "glm_cell_autoupdate.py"
    content = p.read_text(encoding="utf-8")
    assert "B-869" in content
    assert ".git" in content
    assert "index.lock" in content


def test_b860_auto_pull_uses_flock_and_safe_dump():
    """auto_pull_myriad_cell.sh's Phase 3 Python heredoc uses fcntl.flock +
    yaml.safe_dump (B-860 + B-870)."""
    p = REPO / "scripts" / "maintenance" / "auto_pull_myriad_cell.sh"
    content = p.read_text(encoding="utf-8")
    assert "fcntl.flock" in content, "B-860 missing fcntl.flock in auto_pull"
    assert "yaml.safe_dump" in content, "B-870 missing yaml.safe_dump"


# ---------------------------------------------------------------------------
# P0-4 (B-861) — myriad auto_pull failure surface + env propagation
# ---------------------------------------------------------------------------


def test_b861_myriad_watcher_uses_log_redirection_not_devnull():
    """myriad_watcher.py _dispatch_gone_hook redirects child stdout to
    autopull log file, not DEVNULL."""
    p = REPO / "scripts" / "maintenance" / "glm" / "myriad_watcher.py"
    content = p.read_text(encoding="utf-8")
    assert "B-861" in content
    assert "autopull_log" in content or "autopull_" in content
    # Old broken pattern should be gone
    assert 'stdout=subprocess.DEVNULL,\n                    stderr=subprocess.DEVNULL,\n                    start_new_session=True,\n                )\n                return prefix' not in content


def test_b861_myriad_watcher_propagates_paper_grade_env():
    """myriad_watcher.py propagates P79_PAPER_GRADE=1 into child env."""
    p = REPO / "scripts" / "maintenance" / "glm" / "myriad_watcher.py"
    content = p.read_text(encoding="utf-8")
    assert "P79_PAPER_GRADE" in content
    assert "env=" in content or "child_env" in content


def test_b861_myriad_watcher_has_stale_lock_detection():
    """myriad_watcher.py defines _detect_stale_autopull_locks function."""
    p = REPO / "scripts" / "maintenance" / "glm" / "myriad_watcher.py"
    content = p.read_text(encoding="utf-8")
    assert "_detect_stale_autopull_locks" in content


def test_b861_auto_pull_writes_done_marker():
    """auto_pull_myriad_cell.sh writes .done marker on success + removes
    .lock."""
    p = REPO / "scripts" / "maintenance" / "auto_pull_myriad_cell.sh"
    content = p.read_text(encoding="utf-8")
    assert ".done" in content
    assert "LOCK_PATH" in content or "lock_path" in content.lower()


# ---------------------------------------------------------------------------
# P0-5 (B-862) — auth storage_state atomic write
# ---------------------------------------------------------------------------


def test_b862_auth_storage_state_atomic_write():
    """auth_refresh.py wraps ctx.storage_state(path=...) in tmp+rename."""
    p = REPO / "p79" / "utils" / "auth_refresh.py"
    content = p.read_text(encoding="utf-8")
    assert "B-862" in content
    # The script-string-embedded fix references .tmp + os.replace
    assert "_atomic_tmp" in content
    assert "_atomic_os.replace" in content


# ---------------------------------------------------------------------------
# P1-6 (B-863) — deletion-intent rename
# ---------------------------------------------------------------------------


def test_b863_cleanup_module_has_deletion_intent_helpers():
    """p79.experiment.cleanup exports deletion_intent_rename and
    purge_pending_deletes."""
    from p79.experiment import cleanup
    assert hasattr(cleanup, "deletion_intent_rename")
    assert hasattr(cleanup, "purge_pending_deletes")


def test_b863_deletion_intent_rename_creates_pending_marker(tmp_path):
    """deletion_intent_rename moves the path aside with a .pending_delete.<ts>
    suffix."""
    from p79.experiment.cleanup import deletion_intent_rename
    target = tmp_path / "task_0_summary_v2.json"
    target.write_text('{"task_id": 0}', encoding="utf-8")
    pending = deletion_intent_rename(target)
    assert pending is not None
    assert not target.exists()
    assert pending.exists()
    assert ".pending_delete." in pending.name


def test_b863_purge_pending_deletes_only_reaps_old_markers(tmp_path):
    """purge_pending_deletes only removes markers older than threshold."""
    from p79.experiment.cleanup import purge_pending_deletes
    now = int(time.time())
    fresh = tmp_path / f"task_0_summary.json.pending_delete.{now}"
    old = tmp_path / f"task_1_summary.json.pending_delete.{now - 1000}"
    fresh.write_text("fresh", encoding="utf-8")
    old.write_text("old", encoding="utf-8")
    n = purge_pending_deletes(tmp_path, older_than_secs=300)
    assert n == 1
    assert fresh.exists()  # still within threshold
    assert not old.exists()  # reaped


def test_b863_clear_task_files_supports_deletion_intent():
    """clear_task_files has a `deletion_intent` keyword parameter."""
    from p79.experiment.cleanup import clear_task_files
    sig = inspect.signature(clear_task_files)
    assert "deletion_intent" in sig.parameters
    assert sig.parameters["deletion_intent"].default is False


# ---------------------------------------------------------------------------
# P1-7 (B-864) — docker reset timeout setsid + --kill-after
# ---------------------------------------------------------------------------


def test_b864_docker_reset_uses_process_group_kill():
    """_lib_paper_grade_gates.sh:reset_and_auth_gate wraps reset in
    `timeout --kill-after ... setsid bash` for PGID-level kill."""
    p = REPO / "scripts" / "queues" / "_lib_paper_grade_gates.sh"
    content = p.read_text(encoding="utf-8")
    assert "B-864" in content
    assert "--kill-after" in content
    assert "setsid bash" in content


def test_b864_inner_bash_traps_sigterm():
    """The inner bash command traps SIGTERM to attempt docker stop."""
    p = REPO / "scripts" / "queues" / "_lib_paper_grade_gates.sh"
    content = p.read_text(encoding="utf-8")
    assert "trap" in content
    assert "docker stop" in content
    assert "SIGTERM" in content


# ---------------------------------------------------------------------------
# P1-8 (B-865) — STATE_FILE atomic + dedup marker
# ---------------------------------------------------------------------------


def test_b865_state_file_atomic_write():
    """myriad_watcher.py STATE_FILE write uses tmp + os.replace + fsync_dir."""
    p = REPO / "scripts" / "maintenance" / "glm" / "myriad_watcher.py"
    content = p.read_text(encoding="utf-8")
    assert "B-865" in content
    assert "_tmp_state" in content
    assert "os.replace(_tmp_state, STATE_FILE)" in content
    assert "os.fsync" in content


def test_b865_dispatch_dedup_via_done_marker():
    """_dispatch_gone_hook checks .done / .lock markers before re-dispatching
    auto_pull."""
    p = REPO / "scripts" / "maintenance" / "glm" / "myriad_watcher.py"
    content = p.read_text(encoding="utf-8")
    assert "done_path.exists()" in content
    assert "lock_path.exists()" in content


# ---------------------------------------------------------------------------
# P1-9 (B-866) — mid-run staging re-merge
# ---------------------------------------------------------------------------


def test_b866_runner_has_mid_run_remerge_counter():
    """runner/main.py uses _staging_remerge_counter every 10 tasks to call
    merge_staging_trajectory_events."""
    p = REPO / "p79" / "experiment" / "runner" / "main.py"
    content = p.read_text(encoding="utf-8")
    assert "_staging_remerge_counter" in content
    assert "B-866" in content
    assert "% 10 == 0" in content


# ---------------------------------------------------------------------------
# P1-10 (B-867) — phantom sibling dirty-cell hard fail
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phantom_script", [
    "queue_phantom_som.sh",
    "queue_phantom_text.sh",
    "queue_phantom_prompt.sh",
])
def test_b867_phantom_siblings_have_dirty_cell_fatal(phantom_script):
    """All phantom siblings have the (P79_PAPER_GRADE=1 + RESET_BEFORE=1
    + existing RUN_ID) FATAL block sibling-propagated from B-756."""
    p = REPO / "scripts" / "queues" / phantom_script
    content = p.read_text(encoding="utf-8")
    assert "B-867" in content
    assert 'P79_PAPER_GRADE:-0' in content and 'RESET_BEFORE:-0' in content
    assert "FATAL" in content


# ---------------------------------------------------------------------------
# P1-12 (B-868) — stale-resume fingerprint paper_grade env
# ---------------------------------------------------------------------------


def test_b868_resume_fingerprint_includes_env_paper_grade():
    """_compute_resume_fingerprint includes both paper_grade_yaml and
    paper_grade_env, computing _pg_effective as OR of both."""
    p = REPO / "p79" / "experiment" / "runner" / "main.py"
    content = p.read_text(encoding="utf-8")
    assert "B-868" in content
    assert "_pg_env" in content
    assert "_pg_yaml" in content
    assert "_pg_effective" in content
    assert "paper_grade_env" in content
    assert "paper_grade_yaml" in content


# ---------------------------------------------------------------------------
# P2-15 (B-871) — DOM prefix 15000 char
# ---------------------------------------------------------------------------


def test_b871_check_session_health_uses_15k_prefix():
    """_check_session_health reads dom_path[:15000] (raised from 5000)."""
    p = REPO / "scripts" / "maintenance" / "experiment_watchdog.py"
    content = p.read_text(encoding="utf-8")
    assert "[:15000]" in content
    assert "B-871" in content


# ---------------------------------------------------------------------------
# Bash syntax sanity for all touched scripts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", [
    "scripts/queues/_lib_paper_grade_gates.sh",
    "scripts/queues/queue_baseline.sh",
    "scripts/queues/queue_phantom_som.sh",
    "scripts/queues/queue_phantom_text.sh",
    "scripts/queues/queue_phantom_prompt.sh",
    "scripts/maintenance/auto_pull_myriad_cell.sh",
])
def test_bash_syntax_valid(script):
    """All bash scripts touched by A1.23 fixes pass `bash -n` syntax check."""
    result = subprocess.run(
        ["bash", "-n", str(REPO / script)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"{script} bash -n failed: {result.stderr}"


# ---------------------------------------------------------------------------
# Python compilability for all touched modules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module", [
    "p79/experiment/runner/main.py",
    "p79/experiment/cleanup.py",
    "p79/utils/auth_refresh.py",
    "scripts/maintenance/experiment_watchdog.py",
    "scripts/maintenance/glm/glm_cell_autoupdate.py",
    "scripts/maintenance/glm/myriad_watcher.py",
])
def test_python_module_compiles(module):
    """All Python modules touched by A1.23 fixes pass py_compile."""
    import py_compile
    py_compile.compile(str(REPO / module), doraise=True)
