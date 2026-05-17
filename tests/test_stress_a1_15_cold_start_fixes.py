"""Invariant tests for /stress A1.15 cold-start fixes (Chunk a).

Covers:
- B-741 (P0-3-B codex OOB): `_parse_ts` returns timezone-aware UTC datetime
  so mixed naive/aware cross-comparison does NOT raise `TypeError`.
- B-742 (P1-2-B codex OOB): Option K Hook E emits `auth_refresh_no_clear`
  trajectory event after `_auto_refresh_auth`.
- B-743 (digest retire): watchdog no longer exports digest-pipeline functions
  (`_run_auto_digest`, `_check_digest_completions`, `_purge_digest_records`,
  `_purge_digest_records_batch`, `_count_failed_episodes_by_mode`,
  `_DIGEST_MODES`, `_get_active_digest_modes`); state schema no longer carries
  `seen_digest_completions`; argparse no longer accepts `--glm-config` or
  `--digest-dir`; queue scripts no longer pass these flags.
"""
from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# B-741 — _parse_ts returns aware UTC, mixed comparison must not crash
# ---------------------------------------------------------------------------

def _load_aggregator():
    """Direct-load aggregator module without going through scripts/ as a package."""
    spec = importlib.util.spec_from_file_location(
        "aggregate_trajectory_covariates",
        REPO / "scripts" / "analysis" / "aggregate_trajectory_covariates.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_b741_parse_ts_returns_aware_utc_for_naive_input():
    """B-741: naive ISO-8601 input gets timezone.utc attached, not raw naive."""
    mod = _load_aggregator()
    dt = mod._parse_ts("2026-05-17T12:00:00")  # naive
    assert dt is not None
    assert dt.tzinfo is timezone.utc, (
        f"Expected aware UTC, got tzinfo={dt.tzinfo!r} — B-741 fix not active"
    )


def test_b741_parse_ts_aware_input_preserved():
    """B-741: aware input keeps its offset."""
    mod = _load_aggregator()
    dt = mod._parse_ts("2026-05-17T12:00:00+00:00")
    assert dt is not None
    assert dt.tzinfo is not None
    assert dt == datetime(2026, 5, 17, 12, 0, 0, tzinfo=timezone.utc)


def test_b741_parse_ts_z_suffix_normalized():
    """B-741: trailing 'Z' (UTC indicator) gets normalized to +00:00."""
    mod = _load_aggregator()
    dt = mod._parse_ts("2026-05-17T12:00:00Z")
    assert dt is not None
    assert dt.tzinfo is not None
    assert dt == datetime(2026, 5, 17, 12, 0, 0, tzinfo=timezone.utc)


def test_b741_mixed_aware_naive_comparison_does_not_raise():
    """B-741 root cause: comparing aware+offset vs naive must NOT crash.

    Pre-fix: `datetime.fromisoformat('2026-05-17T12:00:00+00:00') < datetime.fromisoformat('2026-05-17T12:00:01')`
    raises `TypeError: can't compare offset-naive and offset-aware datetimes`,
    breaking Option K aggregator on ANY mixed-format input.
    Post-fix: both parsed to aware UTC, comparison succeeds.
    """
    mod = _load_aggregator()
    aware = mod._parse_ts("2026-05-17T12:00:00+00:00")
    naive = mod._parse_ts("2026-05-17T12:00:01")
    z_suf = mod._parse_ts("2026-05-17T12:00:02Z")
    # All three combinations must succeed without TypeError
    assert aware < naive
    assert naive < z_suf
    assert aware < z_suf


def test_b741_parse_ts_unparseable_returns_none():
    """B-741: unparseable input still returns None (graceful degradation preserved)."""
    mod = _load_aggregator()
    assert mod._parse_ts("not-a-timestamp") is None
    assert mod._parse_ts("") is None
    assert mod._parse_ts(None) is None


# ---------------------------------------------------------------------------
# B-742 — Option K Hook E auth_refresh_no_clear emit
# ---------------------------------------------------------------------------

def test_b742_schema_declares_auth_refresh_no_clear():
    """B-742: logger_v2 schema must list `auth_refresh_no_clear` event_type."""
    schema_text = (REPO / "p79" / "experiment" / "logger_v2.py").read_text()
    assert "auth_refresh_no_clear" in schema_text, (
        "B-742: `auth_refresh_no_clear` event_type missing from logger_v2 schema "
        "(was declared in §163.3 Option K but watchdog never emitted pre-B-742)"
    )


def test_b742_watchdog_emits_auth_refresh_no_clear():
    """B-742: watchdog `_auto_refresh_auth` call site emits Option K Hook E."""
    wd_text = (REPO / "scripts" / "maintenance" / "experiment_watchdog.py").read_text()
    assert "B-742" in wd_text, "B-742 fix marker missing in watchdog"
    # The emit block immediately after _auto_refresh_auth must log auth_refresh_no_clear
    # via the trajectory event API (cell-level: task_index=None).
    assert "event_type=\"auth_refresh_no_clear\"" in wd_text, (
        "B-742: watchdog must emit event_type='auth_refresh_no_clear' after refresh attempt"
    )
    assert "task_index=None" in wd_text, (
        "B-742: auth_refresh_no_clear is cell-level (task_index=None)"
    )


# ---------------------------------------------------------------------------
# B-743 — digest pipeline retirement
# ---------------------------------------------------------------------------

def test_b743_digest_functions_removed():
    """B-743: watchdog no longer defines digest-pipeline functions."""
    wd_text = (REPO / "scripts" / "maintenance" / "experiment_watchdog.py").read_text()
    for removed_fn in (
        "def _run_auto_digest(",
        "def _check_digest_completions(",
        "def _purge_digest_records(",
        "def _purge_digest_records_batch(",
        "def _count_failed_episodes_by_mode(",
        "def _get_active_digest_modes(",
    ):
        assert removed_fn not in wd_text, (
            f"B-743: digest function `{removed_fn}` should be removed but still present"
        )
    # The _DIGEST_MODES module-level constant should also be gone
    assert "\n_DIGEST_MODES = (" not in wd_text, (
        "B-743: `_DIGEST_MODES` constant should be removed"
    )


def test_b743_state_schema_no_seen_digest_completions():
    """B-743: state schema (_save_state payload) must NOT include seen_digest_completions."""
    wd_text = (REPO / "scripts" / "maintenance" / "experiment_watchdog.py").read_text()
    # Allow as a comment only (B-743 retirement marker), but not as live JSON key.
    # Check no live `"seen_digest_completions": ...` outside of removal comments.
    for line in wd_text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        assert '"seen_digest_completions"' not in line, (
            f"B-743: live `seen_digest_completions` JSON key found: {line!r}"
        )


def test_b743_argparse_no_glm_config_no_digest_dir():
    """B-743: argparse must NOT register --glm-config or --digest-dir."""
    wd_text = (REPO / "scripts" / "maintenance" / "experiment_watchdog.py").read_text()
    for removed_flag in ('"--glm-config"', '"--digest-dir"'):
        assert removed_flag not in wd_text, (
            f"B-743: argparse flag {removed_flag} must be removed"
        )


@pytest.mark.parametrize(
    "queue_script",
    [
        "queue_baseline.sh",
        "queue_phantom_som.sh",
        "queue_phantom_text.sh",
        "queue_phantom_prompt.sh",
        "queue_phantom_dom.sh",
        "queue_router_learned.sh",
    ],
)
def test_b743_queue_scripts_no_glm_config_no_digest_dir(queue_script):
    """B-743 sibling propagation: queue scripts no longer pass digest-pipeline flags."""
    qs_text = (REPO / "scripts" / "queues" / queue_script).read_text()
    assert "--glm-config" not in qs_text, (
        f"B-743: {queue_script} still passes --glm-config (must be removed)"
    )
    assert "--digest-dir" not in qs_text, (
        f"B-743: {queue_script} still passes --digest-dir (must be removed)"
    )
    assert "WATCHDOG_DIGEST" not in qs_text, (
        f"B-743: {queue_script} still defines WATCHDOG_DIGEST (dead variable, must be removed)"
    )


def test_b743_watchdog_imports_unchanged_for_active_paths():
    """B-743 regression guard: removing digest functions must NOT break Option K hooks.

    The watchdog still emits Option K events (task_auto_cleared / reset_post_interrupt /
    auth_refresh_no_clear) via `from p79.experiment.logger_v2 import log_trajectory_event_external`.
    This import path must remain intact.
    """
    wd_text = (REPO / "scripts" / "maintenance" / "experiment_watchdog.py").read_text()
    assert "from p79.experiment.logger_v2 import log_trajectory_event_external" in wd_text, (
        "B-743 regression: Option K trajectory event import must remain in watchdog"
    )


def test_b743_make_reason_diag_target_preserved():
    """B-743: operator path `make reason-diag` must still exist (analyze_reason_diagnostics.py kept)."""
    makefile = (REPO / "Makefile").read_text()
    assert "reason-diag:" in makefile, (
        "B-743: `make reason-diag` target removed by mistake — should be preserved "
        "(only watchdog auto-trigger was retired, not the operator-facing analysis script)"
    )
    reason_script = REPO / "scripts" / "analysis" / "analyze_reason_diagnostics.py"
    assert reason_script.exists(), (
        "B-743: scripts/analysis/analyze_reason_diagnostics.py removed by mistake — "
        "should be preserved for `make reason-diag` operator path"
    )


def test_b743_glm_batch_digest_preserved_as_standalone():
    """B-743: glm_batch_digest.py preserved as standalone (operator manual debug tool)."""
    glm_digest = REPO / "scripts" / "maintenance" / "glm" / "glm_batch_digest.py"
    assert glm_digest.exists(), (
        "B-743: glm_batch_digest.py removed by mistake — should remain as standalone "
        "operator manual tool (only watchdog auto-trigger was retired)"
    )


# ---------------------------------------------------------------------------
# B-761 — pgrep self-match fix (Path B)
# ---------------------------------------------------------------------------

def _load_watchdog():
    """Direct-load watchdog module with sys.modules registration so @dataclass works."""
    import importlib.util
    import sys as _sys
    spec = importlib.util.spec_from_file_location(
        "ew_module",
        REPO / "scripts" / "maintenance" / "experiment_watchdog.py",
    )
    mod = importlib.util.module_from_spec(spec)
    _sys.modules["ew_module"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_b744_no_pgrep_in_orphan_cleanup():
    """B-761: orphan cleanup no longer calls pgrep (self-match bug source)."""
    wd_text = (REPO / "scripts" / "maintenance" / "experiment_watchdog.py").read_text()
    # The pgrep call for live-runner detection in orphan cleanup must be gone.
    # (pgrep MAY still appear elsewhere — that's fine, we only care about orphan-cleanup path)
    # Check: B-761 marker present + no `pgrep -fa run_experiment` pattern.
    assert "B-761" in wd_text, "B-761 fix marker missing"
    # The specific buggy pgrep pattern (with f-string for run_dir.name) should be removed
    assert 'pgrep", "-fa", f"run_experiment.*{run_dir.name}"' not in wd_text, (
        "B-761: pgrep self-match pattern still present in orphan cleanup"
    )


def test_b744_runner_pid_path_preferred():
    """B-761: orphan cleanup uses `args.runner_pid` + `os.kill(pid, 0)` instead of pgrep."""
    wd_text = (REPO / "scripts" / "maintenance" / "experiment_watchdog.py").read_text()
    # Look for the explicit `args.runner_pid is not None` check in B-761 region
    assert "args.runner_pid is not None" in wd_text, (
        "B-761: orphan cleanup must check args.runner_pid first (Path B)"
    )
    # `os.kill(args.runner_pid, 0)` is the liveness probe (matches B-761 fix path).
    assert "os.kill(args.runner_pid, 0)" in wd_text, (
        "B-761: must use os.kill(args.runner_pid, 0) for liveness probe"
    )


# ---------------------------------------------------------------------------
# B-762 — --reset-state amnesia + --recover-and-quarantine
# ---------------------------------------------------------------------------

def test_b745_recover_and_quarantine_flag_present():
    """B-762: argparse must register --recover-and-quarantine flag."""
    wd_text = (REPO / "scripts" / "maintenance" / "experiment_watchdog.py").read_text()
    assert '"--recover-and-quarantine"' in wd_text, (
        "B-762: --recover-and-quarantine flag not registered"
    )


def test_b745_attempt_partial_state_recovery_present():
    """B-762: `_attempt_partial_state_recovery` helper function exists."""
    mod = _load_watchdog()
    assert hasattr(mod, "_attempt_partial_state_recovery"), (
        "B-762: _attempt_partial_state_recovery helper missing"
    )


def test_b745_partial_recovery_salvages_session_contaminated(tmp_path):
    """B-762: partial recovery salvages session_contaminated from corrupt JSON."""
    mod = _load_watchdog()
    # Simulate corrupt state file: valid prefix, truncated mid-value
    corrupt = tmp_path / "state.json.corrupt.123"
    corrupt.write_text(
        '{\n'
        '  "_schema_version": "v2",\n'
        '  "seen_keys": ["a", "b"],\n'
        '  "session_contaminated": {"reddit": [["cid1", "/path/cdir", 5, "reddit", "key1"]]},\n'
        '  "error_retry_counts": {"cid1/reddit_task_5": 2},\n'
        '  "session_loss_streak": {"reddit": 3},\n'
        '  "TRUNCATED_HERE'  # corrupt
    )
    recovered = mod._attempt_partial_state_recovery(corrupt)
    assert "session_contaminated" in recovered, "B-762: session_contaminated should salvage"
    assert "error_retry_counts" in recovered, "B-762: error_retry_counts should salvage"
    assert "session_loss_streak" in recovered, "B-762: session_loss_streak should salvage"
    # Verify content correctness
    assert recovered["session_contaminated"]["reddit"][0][2] == 5
    assert recovered["error_retry_counts"]["cid1/reddit_task_5"] == 2
    assert recovered["session_loss_streak"]["reddit"] == 3


def test_b745_emit_state_reset_event_helper_present():
    """B-762: `_emit_state_reset_event` helper exists and emits Option K event."""
    mod = _load_watchdog()
    assert hasattr(mod, "_emit_state_reset_event"), (
        "B-762: _emit_state_reset_event helper missing"
    )


# ---------------------------------------------------------------------------
# B-763 — _classify_episode retry-classification code fix (Q1=A)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "err_str,expected_cat",
    [
        # Session
        ("Session expired", "session"),
        ("Not logged in to site", "session"),
        ("login required for /admin", "session"),
        # Auth (after session check)
        ("AUTH_REQUIRED: please relogin", "auth"),
        ("HTTP 401 Unauthorized", "auth"),
        ("HTTP 403 Forbidden access denied", "auth"),
        ("Invalid credentials", "auth"),
        # Connection
        ("Connection refused at endpoint:9980", "connection"),
        ("ECONNREFUSED 127.0.0.1:7770", "connection"),
        ("DNS resolution failed", "connection"),
        # Timeout
        ("Request timed out after 30s", "timeout"),
        ("ETIMEDOUT on Playwright nav", "timeout"),
        ("Deadline exceeded for click", "timeout"),
        # Noise
        ("NavigationError: ERR_ABORTED", "noise"),
        ("Page closed unexpectedly", "noise"),
        ("Playwright Target closed", "noise"),
        # Should NOT match (fall-through to code_bug)
        ("Generic ValueError", None),
        ("TypeError: unsupported", None),
        ("AttributeError: foo", None),
    ],
)
def test_b746_classify_error_string(err_str, expected_cat):
    """B-763: substring classifier returns expected noise category or None."""
    mod = _load_watchdog()
    got = mod._classify_error_string(err_str)
    assert got == expected_cat, (
        f"B-763: _classify_error_string({err_str!r}) → {got!r}, expected {expected_cat!r}"
    )


def test_b746_classify_episode_session_dispatches_to_error_session():
    """B-763: _classify_episode now returns `error(session)` not `error(code_bug)` for session errors."""
    mod = _load_watchdog()
    reason = mod._classify_episode(
        {"error": "Session expired during login"}, {}, max_steps=30
    )
    assert reason == "error(session)", (
        f"B-763: session-class error must classify as error(session), got {reason!r}"
    )


def test_b746_classify_episode_code_bug_fallthrough_preserved():
    """B-763: non-noise errors still fall through to error(code_bug)."""
    mod = _load_watchdog()
    reason = mod._classify_episode(
        {"error": "Generic TypeError: unsupported operand"}, {}, max_steps=30
    )
    assert reason == "error(code_bug)", (
        f"B-763: non-noise error must still classify as error(code_bug), got {reason!r}"
    )


def test_b746_classify_episode_evaluator_error_preserved():
    """B-763: evaluator_error prefix takes precedence over noise substring."""
    mod = _load_watchdog()
    # `evaluator_error: connection refused` — contains "connection" substring,
    # but evaluator prefix should win (paper §3 estimand purity).
    reason = mod._classify_episode(
        {"error": "evaluator_error: connection refused"}, {}, max_steps=30
    )
    assert reason == "error(evaluator)", (
        f"B-763: evaluator_error prefix must take precedence, got {reason!r}"
    )


def test_b746_classify_episode_benchmark_noise_preserved():
    """B-763: benchmark_noise=True path still wins over substring match."""
    mod = _load_watchdog()
    reason = mod._classify_episode(
        {
            "error": "Connection refused (noise marker)",
            "benchmark_noise": True,
            "benchmark_noise_category": "magento_302",
        },
        {}, max_steps=30,
    )
    assert reason == "error(magento_302)", (
        f"B-763: benchmark_noise=True must take precedence with category, got {reason!r}"
    )


# ---------------------------------------------------------------------------
# B-764 — SIGUSR1 fail-loud on registration failure
# ---------------------------------------------------------------------------

def test_b747_sigusr1_fail_loud():
    """B-764: SIGUSR1 register failure must SystemExit(2), not silently pass."""
    wd_text = (REPO / "scripts" / "maintenance" / "experiment_watchdog.py").read_text()
    assert "B-764" in wd_text, "B-764 marker missing"
    # The except block should NO LONGER be `except Exception: pass`
    assert "except (OSError, ValueError)" in wd_text, (
        "B-764: SIGUSR1 except clause must catch OSError/ValueError specifically"
    )
    # The block should raise SystemExit(2)
    assert "raise SystemExit(2)" in wd_text, (
        "B-764: SIGUSR1 fail path must `raise SystemExit(2)` not silently pass"
    )
    # Must NOT contain the old silent pass pattern
    silent_pass_pattern = (
        "signal.signal(signal.SIGUSR1, _on_force_report_signal)\n"
        "    except Exception:\n"
        "        # Some environments may not support SIGUSR1 registration.\n"
        "        pass"
    )
    assert silent_pass_pattern not in wd_text, (
        "B-764: old silent-pass pattern still present (must be removed)"
    )
