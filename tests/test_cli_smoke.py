"""CLI entrypoint smoke test — /stress A1.12 P0-5 AB (2026-05-17, B-669).

Companion to `tests/test_learned_router_runtime.py` (which covers the
learned-router predictor). `p79/cli/run_experiment.py` (64 LOC) is the
CLI entrypoint that queue scripts dispatch into:
  - `scripts/queues/queue_baseline.sh` → `python scripts/run_experiment.py`
  - `pyproject.toml [project.scripts]` registers `run-experiment` entry point

Pre-fix the CLI had 0 test coverage. A regression in `argparse` definition
(e.g., `--config` becomes `--config-file`) would mass-fail ALL queue scripts
silently at first paper-grade fire — `make launch` would echo `usage:` then
exit non-zero, but cron sidecars might misclassify as "site unreachable".

This smoke verifies the CLI parser surface contract via subprocess (so we
exercise the entry-point exactly as queue scripts do).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cli_run_experiment_help_exits_zero():
    """`python -m p79.cli.run_experiment --help` exits 0 + prints usage."""
    proc = subprocess.run(
        [sys.executable, "-m", "p79.cli.run_experiment", "--help"],
        capture_output=True, text=True, timeout=30,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, (
        f"--help exited {proc.returncode}:\nSTDOUT:\n{proc.stdout}\n"
        f"STDERR:\n{proc.stderr}"
    )
    assert "usage:" in proc.stdout.lower() or "usage:" in proc.stderr.lower(), (
        f"--help output missing 'usage:' marker:\n{proc.stdout}\n---\n{proc.stderr}"
    )


def test_cli_run_experiment_accepts_canonical_flags():
    """Argparse contract: --config (required) + --run_id / --phase / --max_steps /
    --log_path (optional) must remain stable.

    Queue scripts hardcode these flag names. Renaming any of them = silent
    paper-grade launch breakage on `make launch`.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "p79.cli.run_experiment", "--help"],
        capture_output=True, text=True, timeout=30,
        cwd=str(REPO_ROOT),
    )
    help_text = (proc.stdout + proc.stderr).lower()
    for flag in ("--config", "--run_id", "--phase", "--max_steps", "--log_path"):
        assert flag in help_text, (
            f"CLI argparse missing canonical flag {flag!r}:\n{help_text}"
        )


def test_cli_run_experiment_missing_config_exits_nonzero():
    """Required --config missing → argparse exits 2 (standard) with error msg.

    Verifies the `required=True` contract on --config so queue scripts that
    omit it (regression) surface immediately.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "p79.cli.run_experiment"],
        capture_output=True, text=True, timeout=30,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode != 0, (
        f"missing --config should be non-zero exit; got {proc.returncode}"
    )


def test_cli_module_importable_without_running_main():
    """Import p79.cli.run_experiment module without invoking main()."""
    import p79.cli.run_experiment as m
    assert callable(m.main), "p79.cli.run_experiment.main must be callable"
