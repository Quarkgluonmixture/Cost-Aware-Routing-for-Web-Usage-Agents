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


def _eval_config_for_cmd(orch_path: Path, cmd_arg: str) -> tuple[int, str, str]:
    """Helper: extract `config_for_cmd` from orchestrator + exercise it.

    Sources just the function via sed range (avoids triggering the orchestrator's
    top-level `case "$MODE"` which would run dry_run / check_gates / launch).
    Returns (rc, stdout, stderr).
    """
    bash_cmd = (
        f'eval "$(sed -n \'/^config_for_cmd()/,/^}}$/p\' "{orch_path}")"; '
        f'config_for_cmd "{cmd_arg}"'
    )
    proc = subprocess.run(
        ["bash", "-c", bash_cmd], capture_output=True, text=True, timeout=10,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr


@pytest.mark.parametrize("baseline,site,expected_filename", [
    ("B0", "classifieds", "configs/exp_v2_B0_phantom_som_classifieds.yaml"),
    ("B1", "reddit", "configs/exp_v2_B1_phantom_som_reddit.yaml"),
    ("B2", "shopping", "configs/exp_v2_B2_phantom_som_shopping.yaml"),
])
def test_config_for_cmd_phantom_som_canonical_path(baseline, site, expected_filename):
    """B-672 (/stress A1.14 P0-1, codex Mode B F2 OOB): orchestrator config_for_cmd
    must map queue_phantom_som.sh chain command to exp_v2_<bl>_phantom_som_<site>.yaml.

    Pre-fix built `..._phantom_<site>.yaml` (missing `_som_` infix), causing Gate 7
    to FAIL all 6 P-SoM cells on `launch all` (B0/B1/B2 × cls/red).
    Launch-blocking. The actual config files (per `queue_phantom_som.sh:61-63`
    CFG_NAME builder) include the `_som_` infix.
    """
    orch = REPO_ROOT / "scripts/queues/queue_phase1_paper_grade.sh"
    rc, out, err = _eval_config_for_cmd(orch, f"queue_phantom_som.sh {baseline} {site}")
    assert rc == 0, f"config_for_cmd extraction failed: {err}"
    assert out == expected_filename, (
        f"phantom_som config name typo regressed (B-672). "
        f"Expected {expected_filename}, got {out}"
    )


def test_config_for_cmd_phantom_dom_back_compat_alias():
    """B-672 (A1.14): queue_phantom_dom.sh is a back-compat symlink to
    queue_phantom_text.sh; config_for_cmd must map both to the same
    `phantom_text` config family (canonical mode value per A1.13 B-630).
    """
    orch = REPO_ROOT / "scripts/queues/queue_phase1_paper_grade.sh"
    rc1, text_out, _ = _eval_config_for_cmd(orch, "queue_phantom_text.sh B0 classifieds")
    rc2, dom_out, _ = _eval_config_for_cmd(orch, "queue_phantom_dom.sh B0 classifieds")
    assert rc1 == 0 and rc2 == 0
    assert text_out == dom_out == "configs/exp_v2_B0_phantom_text_classifieds.yaml", (
        f"phantom_dom/text alias divergence: text={text_out!r}, dom={dom_out!r}"
    )


def test_config_for_cmd_unknown_script_fails_loud():
    """B-672 (A1.14): config_for_cmd default branch must emit `UNKNOWN_SCRIPT:<name>`
    so Gate 7 can fail-loud instead of silently skipping config existence check
    (pre-fix returned empty string → `[ -n "$cfg_path" ]` falsy → silent bypass).
    """
    orch = REPO_ROOT / "scripts/queues/queue_phase1_paper_grade.sh"
    rc, out, err = _eval_config_for_cmd(orch, "queue_made_up.sh B0 classifieds")
    assert rc == 0, f"config_for_cmd extraction failed: {err}"
    assert out.startswith("UNKNOWN_SCRIPT:"), (
        f"default branch must fail-loud with UNKNOWN_SCRIPT marker, got: {out!r}"
    )
    assert "queue_made_up.sh" in out, f"error marker must echo the unknown script: {out!r}"


def test_config_for_cmd_phantom_som_actual_files_exist():
    """B-672 sanity: the canonical configs for Phase 1a P-SoM cells must exist
    on disk. If this fails, either:
      (a) configs/exp_v2_*_phantom_som_*.yaml files were deleted/renamed (real bug)
      (b) config_for_cmd output diverged from on-disk reality (B-672 regressed)
    """
    expected_files = [
        f"configs/exp_v2_{bl}_phantom_som_{site}.yaml"
        for bl in ("B0", "B1", "B2")
        for site in ("classifieds", "reddit")
    ]
    missing = [f for f in expected_files if not (REPO_ROOT / f).exists()]
    assert not missing, (
        f"Phase 1a P-SoM config files missing on disk: {missing}. "
        f"Gate 7 will hard-fail any `launch all` until these exist."
    )


def _eval_check_openai_api_key(preflight_path: Path, env_key: str | None) -> tuple[int, str, str]:
    """Helper: extract `check_openai_api_key` from preflight + exercise it.

    Uses pass()/fail() shims to capture function semantics without exiting the
    test process; mirrors the orchestrator's expectation that fail() = log + rc=1.
    Returns (matched_fail_count, fail_message, all_stderr).
    """
    bash_cmd = (
        f'source <(sed -n "/^check_openai_api_key()/,/^}}$/p" "{preflight_path}"); '
        f'EXIT_CODE=0; '
        f'pass() {{ echo "[PASS] $1"; }}; '
        f'fail() {{ echo "[FAIL] $1"; EXIT_CODE=1; }}; '
        f'check_openai_api_key; '
        f'echo "RC=$EXIT_CODE"'
    )
    env = {**os.environ}
    if env_key is None:
        env.pop("OPENAI_API_KEY", None)
        env["OPENAI_API_KEY"] = ""
    else:
        env["OPENAI_API_KEY"] = env_key
    proc = subprocess.run(
        ["bash", "-c", bash_cmd], capture_output=True, text=True, timeout=10,
        env=env,
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_check_openai_api_key_unset_fails():
    """B-679 (/stress A1.14 Chunk b P1-9, Claude unique OOB): preflight must fail
    if OPENAI_API_KEY is unset — paper-grade VWA LLM judge calls OpenAI for N/A
    task evaluation (helper_functions.py:613+707). Pre-fix DUMMY_P79_PRECHECK
    placeholder masked this requirement; runtime crashed at first N/A task.
    """
    preflight = REPO_ROOT / "scripts/preflight_v2.sh"
    rc, stdout, _ = _eval_check_openai_api_key(preflight, env_key="")
    assert "[FAIL]" in stdout, f"unset OPENAI_API_KEY must fail, got: {stdout!r}"
    assert "OPENAI_API_KEY not set" in stdout
    assert "RC=1" in stdout, f"failure must set EXIT_CODE=1: {stdout!r}"


def test_check_openai_api_key_dummy_placeholder_fails():
    """B-679: DUMMY placeholder must be detected and rejected."""
    preflight = REPO_ROOT / "scripts/preflight_v2.sh"
    rc, stdout, _ = _eval_check_openai_api_key(preflight, env_key="DUMMY_P79_PRECHECK")
    assert "[FAIL]" in stdout, f"DUMMY placeholder must fail: {stdout!r}"
    assert "placeholder" in stdout
    assert "RC=1" in stdout


def test_check_openai_api_key_realistic_passes():
    """B-679: a real-looking key (≥20 chars, no DUMMY/PLACEHOLDER) passes."""
    preflight = REPO_ROOT / "scripts/preflight_v2.sh"
    # Synthetic key — NOT a real OpenAI key, just shaped like one for shape-check.
    rc, stdout, _ = _eval_check_openai_api_key(
        preflight, env_key="sk-proj-test_only_paper_grade_smoke_NOT_REAL_KEY_1234567890"
    )
    assert "[PASS]" in stdout, f"real-shape key must pass: {stdout!r}"
    assert "RC=0" in stdout, f"PASS path must keep EXIT_CODE=0: {stdout!r}"


def test_check_openai_api_key_short_key_fails():
    """B-679: keys <20 chars (e.g., truncated copy-paste) flagged suspicious."""
    preflight = REPO_ROOT / "scripts/preflight_v2.sh"
    rc, stdout, _ = _eval_check_openai_api_key(preflight, env_key="sk-short")
    assert "[FAIL]" in stdout, f"short key must fail: {stdout!r}"
    assert "suspiciously short" in stdout
    assert "RC=1" in stdout


def test_check_vwa_submodule_lock_sha_first_order():
    """B-682 (/stress A1.14 Chunk c P1-7 codex F7 unique OOB): preflight
    `check_vwa_submodule_lock` must check SHA before branch (SHA is immutable
    evidence; branch is social metadata). Pre-fix checked branch first → rejected
    detached-HEAD checkouts at correct SHA (the canonical OSF reproducibility
    workflow). Static check verifies the SHA-comparison block comes before the
    branch-mismatch warn in the function body.
    """
    preflight = REPO_ROOT / "scripts/preflight_v2.sh"
    text = preflight.read_text(encoding="utf-8")
    sha_block_start = text.find('"${actual_sha}" != "${expected_sha}"')
    branch_warn = text.find('"${actual_branch}" != "${expected_branch}" && "${actual_branch}" != "HEAD"')
    assert sha_block_start > 0, "SHA comparison missing from check_vwa_submodule_lock"
    assert branch_warn > 0, "branch mismatch + HEAD-allowance comparison missing"
    assert sha_block_start < branch_warn, (
        f"SHA check must come BEFORE branch check (B-682 SHA-first order). "
        f"SHA pos={sha_block_start}, branch pos={branch_warn}"
    )


def test_check_vwa_submodule_lock_ancestor_fallback_present():
    """B-683 (/stress A1.14 Chunk c P1-10 Claude unique): preflight must allow
    forward-sync via `git merge-base --is-ancestor` fallback. Without this,
    every submodule advance requires manual SHA bump in preflight_v2.sh.
    `EXPECTED_SHA_STRICT=1` env reverts to exact-match for OSF strict mode.
    """
    preflight = REPO_ROOT / "scripts/preflight_v2.sh"
    text = preflight.read_text(encoding="utf-8")
    assert "merge-base --is-ancestor" in text, (
        "B-683 ancestor fallback missing — `git merge-base --is-ancestor expected_sha HEAD` "
        "must be in check_vwa_submodule_lock"
    )
    assert "EXPECTED_SHA_STRICT" in text, (
        "B-683 strict-mode override missing — `EXPECTED_SHA_STRICT=1` should revert to "
        "exact-match for OSF audit runs"
    )


def test_check_vwa_submodule_lock_smoke_current_state():
    """B-682/B-683 smoke: current external/visualwebarena state should pass
    check_vwa_submodule_lock under set -u (no unset-var error).
    Validates the function runs end-to-end without bash syntax issues.
    """
    preflight = REPO_ROOT / "scripts/preflight_v2.sh"
    vwa = REPO_ROOT / "external/visualwebarena"
    if not vwa.exists() or not (vwa / ".git").exists():
        pytest.skip("VWA submodule not initialized")
    bash_cmd = f'''
set -u
PROJECT_DIR="{REPO_ROOT}"
ALLOW_MISSING_EVALUATOR=0
EXIT_CODE=0
pass() {{ echo "[PASS] $1"; }}
fail() {{ echo "[FAIL] $1"; EXIT_CODE=1; }}
warn() {{ echo "[WARN] $1"; }}
print_check() {{ echo "[$1] $2"; }}
source <(sed -n "/^check_vwa_submodule_lock()/,/^}}$/p" "{preflight}")
check_vwa_submodule_lock
echo "RC=$EXIT_CODE"
'''
    proc = subprocess.run(
        ["bash", "-c", bash_cmd], capture_output=True, text=True, timeout=15,
    )
    # Either PASS (exact match), or WARN (ancestor fallback) — both acceptable.
    # FAIL means SHA doesn't match AND isn't an ancestor; that's a real regression.
    assert "RC=0" in proc.stdout, (
        f"check_vwa_submodule_lock failed under current submodule state:\n"
        f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
    )


def test_check_provenance_baseline_helper_present():
    """B-681 (/stress A1.14 Chunk c P1-6 Claude+codex 2-AI AB): orchestrator
    provenance gates must check git-tracked + clean + schema, not just file-exists.
    `_check_provenance_baseline` helper consolidates the 4-layer check.
    """
    orch = REPO_ROOT / "scripts/queues/queue_phase1_paper_grade.sh"
    text = orch.read_text(encoding="utf-8")
    assert "_check_provenance_baseline" in text, "B-681 helper function missing"
    # Verify the 4 layers are wired in the helper:
    assert "git ls-files --error-unmatch" in text, "git-tracked check missing"
    assert "git diff --quiet HEAD" in text, "clean-vs-HEAD check missing"
    assert "captured_at" in text and "host" in text, "JSON schema check missing"


def test_no_python_smoke_when_bash_missing():
    """Sanity guard: this whole file assumes bash. If bash absent, skip clean."""
    if shutil.which("bash") is None:
        pytest.skip("bash not available on this host")
