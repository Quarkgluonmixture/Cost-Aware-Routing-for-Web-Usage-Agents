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
