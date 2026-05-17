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


# ─── B-729 (/stress A1.11 P1-13 A*, 2026-05-17): analyze_experiment CLI smoke ──
# Sibling miss from A1.12 B-674 — that fix added smoke for run_experiment but left
# analyze_experiment uncovered. Same contract: queue scripts + watchdog auto-analysis
# hardcode `--run_dir` flag; argparse regression silently mass-fails downstream.


def test_analyze_cli_help_exits_zero():
    """`python -m p79.cli.analyze_experiment --help` exits 0 + prints usage."""
    proc = subprocess.run(
        [sys.executable, "-m", "p79.cli.analyze_experiment", "--help"],
        capture_output=True, text=True, timeout=30,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, (
        f"--help exited {proc.returncode}:\nSTDOUT:\n{proc.stdout}\n"
        f"STDERR:\n{proc.stderr}"
    )
    assert "usage:" in proc.stdout.lower() or "usage:" in proc.stderr.lower()


def test_analyze_cli_requires_run_dir():
    """Required --run_dir missing → argparse exits non-zero."""
    proc = subprocess.run(
        [sys.executable, "-m", "p79.cli.analyze_experiment"],
        capture_output=True, text=True, timeout=30,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode != 0, (
        f"missing --run_dir should be non-zero exit; got {proc.returncode}"
    )


def test_analyze_module_importable():
    """Import p79.cli.analyze_experiment without invoking main()."""
    import p79.cli.analyze_experiment as m
    assert callable(m.main), "p79.cli.analyze_experiment.main must be callable"


# ─── B-718 (/stress A1.11 P0-2 B* OOB, 2026-05-17): bootstrap import order ────
# Codex Mode B repro: absolute-path invocation of either CLI died with
# `ModuleNotFoundError: No module named 'p79'`. The fix moved sys.path bootstrap
# BEFORE p79.* imports.


def test_run_experiment_absolute_path_invocation():
    """`python3 /abs/path/p79/cli/run_experiment.py --help` works without `pip install -e .`."""
    abs_path = REPO_ROOT / "p79" / "cli" / "run_experiment.py"
    proc = subprocess.run(
        [sys.executable, str(abs_path), "--help"],
        capture_output=True, text=True, timeout=30,
        cwd="/tmp",  # NOT inside repo — sys.path injection must self-bootstrap
        env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"},  # minimal env, no PYTHONPATH
    )
    assert proc.returncode == 0, (
        f"absolute-path invocation should succeed (B-718 fix);\n"
        f"rc={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert "ModuleNotFoundError" not in proc.stderr, (
        f"B-718 regression — ModuleNotFoundError back:\n{proc.stderr}"
    )


def test_analyze_experiment_absolute_path_invocation():
    """Same B-718 contract for analyze_experiment CLI."""
    abs_path = REPO_ROOT / "p79" / "cli" / "analyze_experiment.py"
    proc = subprocess.run(
        [sys.executable, str(abs_path), "--help"],
        capture_output=True, text=True, timeout=30,
        cwd="/tmp",
        env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"},
    )
    assert proc.returncode == 0, (
        f"absolute-path invocation should succeed (B-718 fix);\n"
        f"rc={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert "ModuleNotFoundError" not in proc.stderr
