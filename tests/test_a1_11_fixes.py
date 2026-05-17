"""/stress A1.11 cold-start fix anti-regression tests (B-717 ~ B-730).

Covers paper-grade contracts surfaced by the 3-AI cold-start audit
(Claude Mode A + codex Mode B + gemini Mode C) on the utility / CLI /
logging substrate:

- B-717 P0-1 AC* OOB: subprocess credentials read from env, not argv
- B-722 P1-6 AC: run_id has ms-precision + PID
- B-724 P1-8 AB* OOB: LogCleanupConfig default is dry_run, helpers require confirmed=True
- B-726 P1-10 B: refresh_site_auth auto-creates auth_dir parent
- B-727 P1-11 B: AUTH_REFRESH_TIMEOUT ValueError caught (soft-fail contract)
- B-728 P1-12 A: outcome= tag in refresh_site_auth log on every return path
- B-730 P1-14 BC* OOB: torch._p79_nvrtc_prod_fallback_count counter
- B-730 P1-14 BC* OOB: _cpu_prod_tensor honours kwargs["dtype"] over input dtype
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# ─── B-717 P0-1 AC* OOB: argv credential security ──────────────────────────
def test_b717_no_credentials_in_subprocess_script():
    """Verify the auth_refresh script body does NOT f-string-interpolate creds.

    Pre-fix: script body contained literal `page.locator('#email').fill('<password>')`
    via `{username!r}` / `{password!r}` interpolation → visible in `ps auxe` on
    shared multi-user DGX host.
    Post-fix: script reads from env, body has `os.environ.get('P79_AUTH_USER')`.
    """
    src = (REPO_ROOT / "p79" / "utils" / "auth_refresh.py").read_text()
    # Forbidden: f-string credential interpolation patterns
    assert "{username!r}" not in src, (
        "B-717 regression: {username!r} found — creds back in subprocess argv"
    )
    assert "{password!r}" not in src, (
        "B-717 regression: {password!r} found — creds back in subprocess argv"
    )
    # Required: env-based read inside subprocess script
    assert "os.environ.get('P79_AUTH_USER')" in src, (
        "B-717 contract: subprocess must read P79_AUTH_USER from env, not argv"
    )
    assert "os.environ.get('P79_AUTH_PASS')" in src, (
        "B-717 contract: subprocess must read P79_AUTH_PASS from env, not argv"
    )


def test_b717_env_includes_p79_auth_keys():
    """Verify env propagation block sets P79_AUTH_USER/PASS for the subprocess."""
    src = (REPO_ROOT / "p79" / "utils" / "auth_refresh.py").read_text()
    assert 'env["P79_AUTH_USER"] = username' in src
    assert 'env["P79_AUTH_PASS"] = password' in src


# ─── B-722 P1-6 AC: run_id includes ms + PID ───────────────────────────────
def test_b722_run_id_format_ms_pid():
    """Default run_id must be `run_<ms_timestamp>_<pid>` to avoid same-second collision."""
    src = (REPO_ROOT / "p79" / "cli" / "run_experiment.py").read_text()
    # Match `f"run_{int(time.time() * 1000)}_{os.getpid()}"` (allow whitespace variation)
    pattern = re.compile(
        r'f["\']run_\{int\(time\.time\(\)\s*\*\s*1000\)\}_\{os\.getpid\(\)\}["\']'
    )
    assert pattern.search(src), (
        "B-722 regression: run_id default missing ms precision + PID — same-second "
        "concurrent dispatch will collide on POSIX-second"
    )


# ─── B-724 P1-8 AB* OOB: LogCleanupConfig safety ───────────────────────────
def test_b724_log_cleanup_config_default_is_dry_run():
    """Default LogCleanupConfig().dry_run must be True (pre-fix it was False)."""
    from p79.utils.log_cleanup import LogCleanupConfig
    cfg = LogCleanupConfig()
    assert cfg.dry_run is True, (
        "B-724 regression: LogCleanupConfig default dry_run=False — direct callers "
        "of cleanup_logs/cleanup_results would silently rmtree paper-grade evidence"
    )


def test_b724_cleanup_logs_requires_confirmed(tmp_path, caplog):
    """`cleanup_logs(dir, LogCleanupConfig(dry_run=False))` without confirmed=True
    must still dry-run (safety gate)."""
    from p79.utils.log_cleanup import LogCleanupConfig, cleanup_logs

    # Create dummy old log file
    old_log = tmp_path / "old.log"
    old_log.write_text("dummy")
    # Backdate via os.utime to 100 days ago
    old_mtime = time.time() - (100 * 86400)
    os.utime(old_log, (old_mtime, old_mtime))

    cfg = LogCleanupConfig(max_log_age_days=30, dry_run=False)
    with caplog.at_level(logging.WARNING):
        cleanup_logs(tmp_path, cfg, pattern="*.log")  # no confirmed=True

    # File must still exist (safety gate forced dry-run)
    assert old_log.exists(), (
        "B-724 regression: cleanup_logs deleted file without confirmed=True"
    )
    assert any("B-724" in r.message or "safety gate" in r.message for r in caplog.records), (
        "B-724 contract: must log safety-gate warning when confirmed=False"
    )


def test_b724_cleanup_results_requires_confirmed(tmp_path, caplog):
    """`cleanup_results(dir, LogCleanupConfig(dry_run=False))` without confirmed=True
    must still dry-run."""
    from p79.utils.log_cleanup import LogCleanupConfig, cleanup_results

    run_dir = tmp_path / "run_old"
    run_dir.mkdir()
    (run_dir / "marker.txt").write_text("evidence")
    # Backdate via os.utime
    old_mtime = time.time() - (100 * 86400)
    os.utime(run_dir, (old_mtime, old_mtime))

    cfg = LogCleanupConfig(max_log_age_days=30, dry_run=False)
    with caplog.at_level(logging.WARNING):
        cleanup_results(tmp_path, cfg, max_run_age_days=90)  # no confirmed=True

    assert run_dir.exists(), (
        "B-724 regression: cleanup_results rmtree'd run_dir without confirmed=True"
    )
    assert (run_dir / "marker.txt").exists()


# ─── B-726 P1-10 B: auth_dir auto-create ───────────────────────────────────
def test_b726_refresh_creates_auth_dir(tmp_path):
    """refresh_site_auth must auto-create the auth_dir parent."""
    from p79.utils.auth_refresh import refresh_site_auth

    # Use a deep nested path that does not exist
    deep_auth = tmp_path / "a" / "b" / "c" / "auth"
    assert not deep_auth.exists()

    # Stub env so _load_account does not raise
    with patch.dict(
        os.environ, {"VWA_CLASSIFIEDS_USER": "u", "VWA_CLASSIFIEDS_PASS": "p"},
        clear=False,
    ):
        # We don't care about login success — just that mkdir runs before failure
        # Use a base_url that will definitely fail subprocess (network unreachable host)
        # so refresh returns False, but auth_dir gets created en route.
        result = refresh_site_auth(
            "classifieds",
            deep_auth,
            base_urls={"classifieds": "http://0.0.0.0:1"},
        )
        assert result is False  # subprocess will fail (no server on 0.0.0.0:1)
    assert deep_auth.is_dir(), (
        "B-726 regression: auth_dir not auto-created before subprocess attempt"
    )


# ─── B-727 P1-11 B: AUTH_REFRESH_TIMEOUT misconfig ─────────────────────────
def test_b727_auth_refresh_timeout_value_error_caught(tmp_path, caplog):
    """Typo'd AUTH_REFRESH_TIMEOUT must return False, NOT raise ValueError."""
    from p79.utils.auth_refresh import refresh_site_auth

    with patch.dict(os.environ, {
        "VWA_CLASSIFIEDS_USER": "u",
        "VWA_CLASSIFIEDS_PASS": "p",
        "AUTH_REFRESH_TIMEOUT": "not-a-number",
    }, clear=False):
        with caplog.at_level(logging.WARNING):
            result = refresh_site_auth(
                "classifieds", tmp_path,
                base_urls={"classifieds": "http://0.0.0.0:1"},
            )
    assert result is False, (
        "B-727 regression: misconfig AUTH_REFRESH_TIMEOUT escaped soft-fail contract"
    )
    assert any("misconfig" in r.message for r in caplog.records), (
        "B-727 contract: outcome=misconfig log tag expected"
    )


def test_b727_auth_refresh_timeout_nonpositive_rejected(tmp_path, caplog):
    """AUTH_REFRESH_TIMEOUT=0 or negative also rejected as misconfig."""
    from p79.utils.auth_refresh import refresh_site_auth

    with patch.dict(os.environ, {
        "VWA_CLASSIFIEDS_USER": "u",
        "VWA_CLASSIFIEDS_PASS": "p",
        "AUTH_REFRESH_TIMEOUT": "0",
    }, clear=False):
        with caplog.at_level(logging.WARNING):
            result = refresh_site_auth(
                "classifieds", tmp_path,
                base_urls={"classifieds": "http://0.0.0.0:1"},
            )
    assert result is False
    assert any("misconfig" in r.message for r in caplog.records)


# ─── B-728 P1-12 A: outcome= log tag ───────────────────────────────────────
def test_b728_outcome_log_tag_present_in_source():
    """Every return path of refresh_site_auth must log an outcome= tag."""
    src = (REPO_ROOT / "p79" / "utils" / "auth_refresh.py").read_text()
    # outcome tags introduced by B-728
    required_tags = (
        "outcome=ok",
        "outcome=cred_wrong",
        "outcome=env_missing",
        "outcome=playwright_error",
        "outcome=timeout",
        "outcome=playwright_crash",
        "outcome=misconfig",
        "outcome=dir_not_writable",
    )
    for tag in required_tags:
        assert tag in src, (
            f"B-728 regression: missing outcome tag {tag!r} — log-grep-based "
            f"failure diagnosis cannot distinguish failure mode"
        )


# ─── B-730 P1-14 BC* OOB: CUDA workaround dtype + counter ──────────────────
def test_b730_fallback_counter_attribute():
    """torch._p79_nvrtc_prod_fallback_count must initialize lazily without error."""
    import torch
    # B-730 fallback initializes counter on first _cpu_prod_tensor call. We assert
    # the *source* increments via getattr-with-default (safe even when not yet set).
    src = (REPO_ROOT / "p79" / "utils" / "torch_cuda_workarounds.py").read_text()
    assert "torch._p79_nvrtc_prod_fallback_count = getattr(" in src, (
        "B-730 regression: fallback counter missing — env_snapshot cannot record "
        "nvrtc_fallback_fired_count for paper-grade reproducibility audit"
    )


def test_b730_dtype_kwarg_honoured_in_source():
    """_cpu_prod_tensor must honour kwargs['dtype'] over input tensor dtype."""
    src = (REPO_ROOT / "p79" / "utils" / "torch_cuda_workarounds.py").read_text()
    assert '_target_dtype = kwargs.get("dtype", tensor.dtype)' in src, (
        "B-730 regression: dtype kwarg silently overridden by input tensor dtype"
    )
    assert "out = out.to(dtype=_target_dtype)" in src
