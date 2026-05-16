"""Shell-script smoke / syntax tests — /stress A1.12 P0-4 (起步 scope).

Pre-2026-05-16 status: ZERO automated tests for `_lib_paper_grade_gates.sh`,
`queue_phase1_paper_grade.sh`, `queue_chain.sh`, or `queue_baseline.sh`
despite these being the launch-side defense layer (A1.13/A1.14 batch:
B-303 chain leakage, B-304 resume discontinuity, B-224 auth gate
hard-fail propagation all live in shell).

This is the **起步** version (per /stress A1.12 T2-2 default ~1h scope):
- `bash -n` syntax check covers parse-level regressions
- `declare -F` verifies the lib exports its declared function surface
- `mint_run_id` is exercised in FORCE_NEW=1 dry mode (pure function: just
  echoes a fresh run_id without side effects)

Full subprocess-level happy-path coverage of `reset_and_auth_gate` /
`init_paper_grade_env` deferred (requires mocking VWA env / docker /
proxy api key — separate batch).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Canonical shell-script surface — paper-grade launch path.
SHELL_SCRIPTS = [
    "scripts/queues/_lib_paper_grade_gates.sh",
    "scripts/queues/queue_phase1_paper_grade.sh",
    "scripts/queues/queue_chain.sh",
    "scripts/queues/queue_baseline.sh",
    "scripts/queues/queue_phantom_som.sh",
    "scripts/queues/queue_phantom_dom.sh",
    "scripts/queues/queue_phantom_text.sh",
    "scripts/queues/queue_phantom_prompt.sh",
    "scripts/preflight_v2.sh",
]

LIB_PATH = REPO_ROOT / "scripts/queues/_lib_paper_grade_gates.sh"

# Functions the lib MUST export (A1.13/A1.14 contract; any rename without
# updating downstream queue scripts is a silent break).
LIB_EXPECTED_FUNCS = [
    "init_paper_grade_env",
    "load_proxy_api_key",
    "mint_run_id",
    "reset_and_auth_gate",
]


@pytest.mark.parametrize("rel_path", SHELL_SCRIPTS)
def test_shell_script_parses_clean(rel_path):
    """`bash -n` (no-exec parse) on every paper-grade launch script."""
    path = REPO_ROOT / rel_path
    if not path.exists():
        pytest.skip(f"{rel_path} not present (queue file rename?)")
    proc = subprocess.run(
        ["bash", "-n", str(path)],
        capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode == 0, (
        f"bash -n failed for {rel_path}:\n{proc.stderr}"
    )


def test_lib_paper_grade_gates_exports_expected_functions():
    """Source the lib and verify the contract surface is intact.

    `declare -F` lists function names defined in the current shell. Queue
    scripts call `init_paper_grade_env` / `mint_run_id` / `reset_and_auth_gate`
    by name; renaming any of them without updating callers = silent break.
    """
    assert LIB_PATH.exists(), f"shared lib missing at {LIB_PATH}"
    func_list = "\n".join(f'declare -F {f}' for f in LIB_EXPECTED_FUNCS)
    cmd = f'source "{LIB_PATH}" && {func_list}'
    proc = subprocess.run(
        ["bash", "-c", cmd], capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode == 0, (
        f"lib source / declare -F failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    # `declare -F <name>` with name argument echoes just `<name>` when defined,
    # nothing otherwise. (vs `declare -F` with no arg = `declare -f <name>` form).
    defined_funcs = set(proc.stdout.split())
    missing = [f for f in LIB_EXPECTED_FUNCS if f not in defined_funcs]
    assert not missing, (
        f"lib does not export functions {missing} after source — caller breakage. "
        f"declare -F output:\n{proc.stdout}"
    )


def test_mint_run_id_force_new_emits_timestamped_id(tmp_path):
    """`mint_run_id` in FORCE_NEW mode echoes a fresh run_id without touching
    the filesystem outside `tmp_path`.

    Pure function exercise — proves the lib actually runs end-to-end in a
    controlled subprocess, not just parses.
    """
    # mint_run_id takes (cfg_name, output_root, log_prefix) positionally per
    # lib source. FORCE_NEW=1 forces a fresh timestamped run_id.
    output_root = tmp_path / "results"
    output_root.mkdir()
    cmd = (
        f'set -e; source "{LIB_PATH}"; '
        f'FORCE_NEW=1 mint_run_id "smoke_cfg" "{output_root}" "smoke-test"'
    )
    proc = subprocess.run(
        ["bash", "-c", cmd], capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode == 0, (
        f"mint_run_id failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    # mint_run_id echoes the run_id; output should contain "smoke_cfg_" prefix.
    assert "smoke_cfg_" in proc.stdout, (
        f"mint_run_id output missing smoke_cfg_ prefix:\n{proc.stdout}"
    )


def test_mint_run_id_nanosecond_collision_defense():
    """Two back-to-back mint_run_id FORCE_NEW calls must produce distinct ids.

    P1-2 fix (A1.13 codex+gemini): RUN_ID suffix includes nanos + PID + RANDOM
    so concurrent paper-grade sessions cannot collide on second-precision
    timestamps. This test rapid-fires two mints and asserts inequality.
    """
    cmd = (
        f'set -e; source "{LIB_PATH}"; '
        f'FORCE_NEW=1 R1=$(mint_run_id "x" /tmp "test" 2>/dev/null); '
        f'FORCE_NEW=1 R2=$(mint_run_id "x" /tmp "test" 2>/dev/null); '
        f'echo "$R1"; echo "$R2"'
    )
    proc = subprocess.run(
        ["bash", "-c", cmd], capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode == 0, f"mint_run_id pair failed: {proc.stderr}"
    lines = [line for line in proc.stdout.strip().split("\n") if line.strip()]
    # Each invocation echoes 1+ lines including the run_id line. The last line
    # from each invocation is the printed $R1 / $R2.
    assert len(lines) >= 2, f"expected ≥2 lines, got {len(lines)}: {lines}"
    assert lines[-1] != lines[-2], (
        f"RUN_ID collision: back-to-back FORCE_NEW produced identical ids:\n"
        f"{lines[-2]}\n{lines[-1]}\n"
        f"Nanos/PID/RANDOM suffix may have regressed (A1.13 P1-2)."
    )


def test_queue_chain_three_baseline_collision_check_present():
    """B0/B1/B2 same-site collision check must live in queue_chain.sh.

    /stress A1.13 batch + memory `feedback_experiment_launch_rules`: same site
    same time only ONE baseline can run (server-side session race). The
    3-baseline aware collision check must be present in queue_chain.sh.
    """
    chain = REPO_ROOT / "scripts/queues/queue_chain.sh"
    text = chain.read_text(encoding="utf-8")
    # The collision check uses pgrep on run_experiment + site name. Look for
    # the broad surface (any of B0/B1/B2 / Gemma reference + pgrep) — exact
    # implementation can vary but the defense must exist.
    assert "pgrep" in text, "queue_chain.sh missing pgrep collision check"
    assert "run_experiment" in text, (
        "queue_chain.sh collision check must target run_experiment processes"
    )


def test_no_python_smoke_when_bash_missing():
    """Sanity guard: this whole file assumes bash. If bash absent, skip clean."""
    if shutil.which("bash") is None:
        pytest.skip("bash not available on this host")
