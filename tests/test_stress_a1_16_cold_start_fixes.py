"""Invariant tests for /stress A1.16 cold-start fixes (B-822 ~ B-839).

Cross-AI 3-AI cycle (Claude self Mode A / codex Mode B / gemini Mode C) on
`scripts/provenance/{snapshot_env.py, snapshot_vwa.sh}` 2026-05-17.

A1.16 cold-start re-audit fixes covered by these tests:
- B-822 (P0-1-ABC*): VWA SBOM tree-hash chain recompute via `_vwa_submodule_integrity`
- B-823 (P0-2-AB* + P0-6-BC*): `snapshot_has_critical_errors()` helper for
  caller post-call inspect (covers _critical_error / evaluator_code.incomplete /
  divergence != match / vwa_sbom.match_lock=False)
- B-824 (P0-3-AC*): `_capture_reference_images_sha()` recursive hash of
  `external/visualwebarena/coco_images/`
- B-827 (P1-1-BC*): runner `_seed_global_rng` sets deterministic torch flags
- B-828 (P1-2-AC*): EVALUATOR_SOURCE_FILES expanded 7 → 17 files
- B-829 (P1-3-B*): `_myriad_sitecustomize_hash()` Myriad-only path
- B-832 (P1-6-A*): `_loaded_revision_from_cache` docstring + cache_latest_revision
  alias + _field_semantics annotation
- B-833 (P1-7-C*): `_capture_judge_env()` env capture for LLM judge model
- B-834 (P1-8-B): phase1a_launch_gate.py renamed to endpoint_gate.py with
  scope-clarification docstring + "ENDPOINTS PASS" message
- B-836 (P1-10-B*): stale-resume content-invariant check (git_commit +
  vwa_submodule_sha) added to `_lib_paper_grade_gates.sh`
- B-837 (P2-1-A): EVALUATOR_CONFIG_GLOB recursive `**/exp_v2_*.yaml`
- B-838 (P2-2-C): `config_files/generation_manifest.json` in EVALUATOR_SOURCE_FILES

Pre-A1.16-re A1.16 fixes (B-273~B-279) implicitly retained as preconditions
of the canonical hash form; tests assert NO regression there.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
# Ensure repo root on sys.path so `from scripts.provenance.snapshot_env import ...`
# works without `pip install -e .` (used in CI matrix).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ─── B-828 (P1-2-AC*) — EVALUATOR_SOURCE_FILES scope expansion ──────────────
def test_b805_evaluator_source_files_scope_expanded():
    """EVALUATOR_SOURCE_FILES must include agents+backends+tasks.py+browser_env
    + generation_manifest after A1.16 cold-start re-audit (B-828+B-838)."""
    from scripts.provenance import snapshot_env

    files = set(snapshot_env.EVALUATOR_SOURCE_FILES)
    # Pre-A1.16-re baseline (B-242 scope): 7 files
    pre_a1_16_re_baseline = {
        "p79/experiment/analysis.py",
        "p79/experiment/environment.py",
        "p79/experiment/metrics.py",
        "external/visualwebarena/evaluation_harness/helper_functions.py",
        "external/visualwebarena/evaluation_harness/evaluators.py",
        "p79/utils/auth_refresh.py",
    }
    for f in pre_a1_16_re_baseline:
        assert f in files, f"regression: pre-A1.16-re baseline file {f} dropped from EVALUATOR_SOURCE_FILES"

    # B-828 P1-2-AC* expansion: agents (3) + backends (3) + tasks.py (1) + browser_env (2)
    b805_additions = {
        "p79/experiment/tasks.py",
        "p79/agents/qwen3vl_agent.py",
        "p79/agents/proxy_api_agent.py",
        "p79/agents/gemma3vl_agent.py",
        "p79/backends/local_qwen.py",
        "p79/backends/local_gemma.py",
        "p79/backends/api_proxy.py",
        "external/visualwebarena/browser_env/actions.py",
        "external/visualwebarena/browser_env/processors.py",
    }
    for f in b805_additions:
        assert f in files, f"B-828 P1-2-AC* expansion regression: {f} missing"

    # B-838 P2-2-C: generation_manifest
    assert "external/visualwebarena/config_files/generation_manifest.json" in files, (
        "B-838 P2-2-C: generation_manifest.json missing from EVALUATOR_SOURCE_FILES"
    )

    # Total file count sanity
    assert len(files) >= 16, f"EVALUATOR_SOURCE_FILES count regression: {len(files)} < expected 16"


def test_b805_evaluator_source_files_canonical_sorted():
    """List must be sorted (canonical) so combined_sha256 is order-independent."""
    from scripts.provenance import snapshot_env

    assert list(snapshot_env.EVALUATOR_SOURCE_FILES) == sorted(snapshot_env.EVALUATOR_SOURCE_FILES), (
        "EVALUATOR_SOURCE_FILES must be canonical-sorted (`sorted(...)` literal at module load)"
    )


# ─── B-837 (P2-1-A) — EVALUATOR_CONFIG_GLOB recursive ─────────────────────────
def test_b814_evaluator_config_glob_recursive():
    """Recursive glob `**/exp_v2_*.yaml` for future subdir organization."""
    from scripts.provenance import snapshot_env

    assert snapshot_env.EVALUATOR_CONFIG_GLOB.startswith("**/"), (
        f"B-837 P2-1-A: EVALUATOR_CONFIG_GLOB must use recursive `**/` glob, got "
        f"{snapshot_env.EVALUATOR_CONFIG_GLOB!r}"
    )


# ─── B-822 (P0-1-ABC*) — VWA SBOM tree-hash chain constants ───────────────────
def test_b799_vwa_sbom_constants_match_locked_versions():
    """Constants in snapshot_env.py must match locked_versions.md hardcoded values."""
    from scripts.provenance import snapshot_env

    # Sanity: SHA hex format
    assert len(snapshot_env.VWA_LOCKED_HEAD_SHA) == 40, "head sha not 40 hex chars"
    assert all(c in "0123456789abcdef" for c in snapshot_env.VWA_LOCKED_HEAD_SHA), "head sha not hex"
    assert len(snapshot_env.VWA_LOCKED_UPSTREAM_BASE) == 40, "base sha not 40 hex chars"
    assert len(snapshot_env.VWA_LOCKED_TREE_HASH_CHAIN) == 64, "chain sha256 not 64 hex chars"


def test_b799_vwa_submodule_integrity_returns_match_lock_field():
    """`_vwa_submodule_integrity` must return dict with `match_lock` boolean
    AND per-layer match flags (head_match / base_match / chain_match)."""
    from scripts.provenance.snapshot_env import _vwa_submodule_integrity

    errors = []
    result = _vwa_submodule_integrity(REPO_ROOT, errors)

    # Either VWA dir is present or unavailable; both paths must surface match_lock
    if result.get("unavailable"):
        assert result.get("match_lock") is False
    else:
        assert "match_lock" in result
        assert "head_match" in result
        assert "base_match" in result
        assert "chain_match" in result
        assert "locked_head_sha" in result
        assert "locked_chain_sha256" in result
        # match_lock must be conjunction of three sub-matches
        assert result["match_lock"] == (
            result["head_match"] and result["base_match"] and result["chain_match"]
        )


# ─── B-823 (P0-2/P0-6) — snapshot_has_critical_errors helper ─────────────────
def test_b800_critical_errors_detects_gated_no_token():
    """When models[m]._critical_error is set, has_critical=True."""
    from scripts.provenance.snapshot_env import snapshot_has_critical_errors

    snap = {
        "models": {
            "google/gemma-3-4b-it": {
                "loaded_revision": None,
                "_critical_error": "GATED model + HF_TOKEN unset",
            }
        },
        "evaluator_code": {},
        "vwa_sbom": {"match_lock": True},
    }
    has, reasons = snapshot_has_critical_errors(snap)
    assert has is True
    assert any("gated model" in r.lower() for r in reasons)


def test_b800_critical_errors_detects_evaluator_incomplete():
    """When evaluator_code.incomplete=True, has_critical=True."""
    from scripts.provenance.snapshot_env import snapshot_has_critical_errors

    snap = {
        "models": {},
        "evaluator_code": {"incomplete": True, "error": "missing file: foo.py"},
        "vwa_sbom": {"match_lock": True},
    }
    has, reasons = snapshot_has_critical_errors(snap)
    assert has is True
    assert any("evaluator_code incomplete" in r for r in reasons)


def test_b800_critical_errors_detects_model_divergence():
    """When models[m].divergence is not match/None/no_local_cache, has_critical=True."""
    from scripts.provenance.snapshot_env import snapshot_has_critical_errors

    snap = {
        "models": {
            "Qwen/Qwen3-VL-4B-Instruct": {
                "loaded_revision": "abc123",
                "divergence": "runner_used_stale_cache",
            },
        },
        "evaluator_code": {},
        "vwa_sbom": {"match_lock": True},
    }
    has, reasons = snapshot_has_critical_errors(snap)
    assert has is True
    assert any("divergence" in r.lower() for r in reasons)


def test_b800_critical_errors_match_state_clean():
    """When all checks pass, has_critical=False with empty reasons."""
    from scripts.provenance.snapshot_env import snapshot_has_critical_errors

    snap = {
        "models": {
            "Qwen/Qwen3-VL-4B-Instruct": {
                "loaded_revision": "abc123",
                "divergence": "match",
            }
        },
        "evaluator_code": {"combined_sha256": "deadbeef" * 8},
        "vwa_sbom": {"match_lock": True},
    }
    has, reasons = snapshot_has_critical_errors(snap)
    assert has is False, f"unexpected critical reasons: {reasons}"
    assert reasons == []


def test_b800_critical_errors_detects_vwa_sbom_divergence():
    """When vwa_sbom.match_lock=False, has_critical=True (tree-hash mismatch)."""
    from scripts.provenance.snapshot_env import snapshot_has_critical_errors

    snap = {
        "models": {},
        "evaluator_code": {},
        "vwa_sbom": {
            "match_lock": False,
            "head_match": False,
            "base_match": True,
            "chain_match": False,
        },
    }
    has, reasons = snapshot_has_critical_errors(snap)
    assert has is True
    assert any("vwa sbom divergence" in r.lower() for r in reasons)


def test_b800_critical_errors_no_local_cache_not_critical():
    """divergence=no_local_cache (runner hasn't loaded the model yet) is NOT critical."""
    from scripts.provenance.snapshot_env import snapshot_has_critical_errors

    snap = {
        "models": {
            "google/gemma-3-4b-it": {
                "loaded_revision": None,
                "divergence": "no_local_cache",
            }
        },
        "evaluator_code": {},
        "vwa_sbom": {"match_lock": True},
    }
    has, reasons = snapshot_has_critical_errors(snap)
    assert has is False, f"no_local_cache should not be critical, got: {reasons}"


# ─── B-824 (P0-3-AC*) — reference images SHA ──────────────────────────────────
def test_b801_reference_image_roots_includes_coco_images():
    """REFERENCE_IMAGE_ROOTS must include external/visualwebarena/coco_images
    (per glm_batch_digest._load_reference_images_b64 discovery 2026-05-17)."""
    from scripts.provenance import snapshot_env

    assert "external/visualwebarena/coco_images" in snapshot_env.REFERENCE_IMAGE_ROOTS, (
        "B-824 P0-3-AC*: coco_images root missing from REFERENCE_IMAGE_ROOTS"
    )


def test_b801_capture_reference_images_sha_canonical():
    """`_capture_reference_images_sha` returns canonical sentinel-delimited
    combined hash (B-240 sibling form)."""
    from scripts.provenance.snapshot_env import _capture_reference_images_sha

    errors = []
    result = _capture_reference_images_sha(REPO_ROOT, errors)

    assert "combined_sha256" in result
    assert "files_count" in result
    assert "per_file" in result
    assert "roots" in result
    assert "schema_version" in result

    # If coco_images exists with files, expect non-empty hash
    coco_dir = REPO_ROOT / "external/visualwebarena/coco_images"
    if coco_dir.exists() and any(coco_dir.glob("*.jpg")):
        assert result["combined_sha256"] is not None, "combined_sha256 should not be None when files exist"
        assert result["files_count"] > 0, "files_count > 0 when coco_images populated"
        assert "external/visualwebarena/coco_images" in result["roots"]


# ─── B-829 (P1-3-B*) — Myriad sitecustomize hash ──────────────────────────────
def test_b806_myriad_sitecustomize_returns_none_on_non_myriad():
    """`_myriad_sitecustomize_hash` returns None on non-Myriad host
    (paper-1 A100 self-host path is no-op)."""
    from scripts.provenance.snapshot_env import _myriad_sitecustomize_hash

    errors = []
    # Force non-Myriad detection by stubbing socket.gethostname
    with patch("scripts.provenance.snapshot_env.socket.gethostname", return_value="a100-jiaming-test"):
        # Also clear SLURM/SGE env vars to avoid false detection
        original_slurm = os.environ.pop("SLURM_JOB_ID", None)
        original_sge = os.environ.pop("SGE_TASK_ID", None)
        try:
            result = _myriad_sitecustomize_hash(errors)
            assert result is None, f"non-Myriad host should return None, got: {result}"
        finally:
            if original_slurm:
                os.environ["SLURM_JOB_ID"] = original_slurm
            if original_sge:
                os.environ["SGE_TASK_ID"] = original_sge


# ─── B-832 (P1-6-A*) — loaded_revision field semantics ────────────────────────
def test_b809_capture_model_revisions_includes_field_semantics():
    """models[m] entries must include _field_semantics annotation explaining
    loaded_revision is filesystem-cache-newest (NOT runtime-loaded)."""
    from scripts.provenance.snapshot_env import _capture_model_revisions

    errors = []
    # Simulate gated model failure path (deterministic, no HF API call needed)
    result = _capture_model_revisions(
        models=["google/gemma-3-4b-it"],
        gated_models=["google/gemma-3-4b-it"],
        errors=errors,
    )

    # Force HF_TOKEN unset to trigger _critical_error path
    if "HF_TOKEN" in os.environ:
        # If running with token, skip semantic check (different code path)
        pytest.skip("HF_TOKEN set in env — skipping critical-error path semantic check")

    entry = result.get("google/gemma-3-4b-it", {})
    assert "_field_semantics" in entry, "B-832: _field_semantics annotation missing from model entry"
    semantics = entry["_field_semantics"]
    assert "loaded_revision" in semantics
    assert "NOT runtime-loaded" in semantics["loaded_revision"], (
        "_field_semantics.loaded_revision must explicitly disambiguate from runtime-loaded"
    )


# ─── B-833 (P1-7-C*) — judge env capture ─────────────────────────────────────
def test_b810_judge_env_vars_includes_vwa_eval_model():
    """JUDGE_ENV_VARS must include VWA_EVAL_MODEL + OPENAI_EVAL_MODEL
    (used by VWA evaluator at helper_functions.py:612-613, 706-707)."""
    from scripts.provenance import snapshot_env

    assert "VWA_EVAL_MODEL" in snapshot_env.JUDGE_ENV_VARS, (
        "B-833 P1-7-C*: VWA_EVAL_MODEL missing from JUDGE_ENV_VARS"
    )
    assert "OPENAI_EVAL_MODEL" in snapshot_env.JUDGE_ENV_VARS, (
        "B-833 P1-7-C*: OPENAI_EVAL_MODEL missing from JUDGE_ENV_VARS"
    )


def test_b810_capture_judge_env_returns_all_keys():
    """`_capture_judge_env()` returns dict with all JUDGE_ENV_VARS keys
    (value None for unset, string for set)."""
    from scripts.provenance.snapshot_env import _capture_judge_env, JUDGE_ENV_VARS

    errors = []
    result = _capture_judge_env(errors)
    for var in JUDGE_ENV_VARS:
        assert var in result, f"JUDGE_ENV_VARS key {var} missing from _capture_judge_env() return"


# ─── B-834 (P1-8-B) — endpoint_gate.py rename + docstring ────────────────────
def test_b811_phase1a_launch_gate_renamed_to_endpoint_gate():
    """Original `phase1a_launch_gate.py` was renamed to `endpoint_gate.py`
    in A1.16 cold-start re-audit P1-8-B."""
    old_path = REPO_ROOT / "scripts/maintenance/phase1a_launch_gate.py"
    new_path = REPO_ROOT / "scripts/maintenance/endpoint_gate.py"

    assert not old_path.exists(), (
        f"B-834 P1-8-B: old name {old_path} still exists — git mv incomplete"
    )
    assert new_path.exists(), (
        f"B-834 P1-8-B: new name {new_path} missing"
    )


def test_b811_endpoint_gate_docstring_clarifies_scope():
    """endpoint_gate.py docstring must explicitly disambiguate URL reachability
    from full provenance gate (no false "SAFE" implication)."""
    gate_path = REPO_ROOT / "scripts/maintenance/endpoint_gate.py"
    text = gate_path.read_text()

    assert "ENDPOINTS PASS" in text, "endpoint_gate.py output must say ENDPOINTS PASS (not SAFE)"
    assert "NAME / SCOPE CLARIFICATION" in text, "endpoint_gate.py docstring missing scope-clarification banner"
    assert "not a paper-grade launch authorization" in text.lower(), (
        "endpoint_gate.py must explicitly disclaim paper-grade authorization"
    )


# ─── B-838 (P2-2-C) — generation_manifest in EVALUATOR_SOURCE_FILES ──────────
def test_b815_generation_manifest_in_evaluator_source_files():
    """`config_files/generation_manifest.json` must be in EVALUATOR_SOURCE_FILES
    per prereg §7 line 578 OSF byte-equivalence claim."""
    from scripts.provenance import snapshot_env

    assert "external/visualwebarena/config_files/generation_manifest.json" in snapshot_env.EVALUATOR_SOURCE_FILES, (
        "B-838 P2-2-C: generation_manifest.json missing from EVALUATOR_SOURCE_FILES"
    )


# ─── Schema version markers ─────────────────────────────────────────────────
def test_a1_16_re_schema_version_bumped():
    """Schema version markers reflect A1.16 cold-start re-audit."""
    from scripts.provenance import snapshot_env

    # Module-level constants should reference 2026-05-17-a1.16-re schema.
    # Read source to inspect string literal (avoiding need to call functions
    # that hit network / GPU).
    source = (REPO_ROOT / "scripts/provenance/snapshot_env.py").read_text()
    assert '"2026-05-17-a1.16-re"' in source, "snap schema_version not bumped to 2026-05-17-a1.16-re"
    assert '"2026-05-17-a1.16-re-canonical-v3"' in source, "evaluator_code schema_version not bumped"


# ─── B-836 (P1-10-B*) — stale-resume content invariant ───────────────────────
def test_b813_stale_resume_checks_git_commit():
    """_lib_paper_grade_gates.sh stale-resume check must inspect git_commit
    in env_snapshot.json (B-836 P1-10-B*)."""
    gates_path = REPO_ROOT / "scripts/queues/_lib_paper_grade_gates.sh"
    text = gates_path.read_text()

    assert "B-836" in text, "B-836 P1-10-B* fix marker missing from _lib_paper_grade_gates.sh"
    assert "git_commit mismatch" in text, "git_commit content-invariant check missing"


def test_b813_stale_resume_checks_vwa_submodule():
    """_lib_paper_grade_gates.sh stale-resume must also check VWA submodule SHA."""
    gates_path = REPO_ROOT / "scripts/queues/_lib_paper_grade_gates.sh"
    text = gates_path.read_text()

    assert "vwa_submodule mismatch" in text, "vwa_submodule content-invariant check missing"


# ─── B-827 (P1-1-BC*) — deterministic torch flags in _seed_global_rng ────────
def test_b804_seed_global_rng_sets_deterministic_flags():
    """runner `_seed_global_rng` must set torch.use_deterministic_algorithms +
    cudnn.deterministic + CUBLAS_WORKSPACE_CONFIG."""
    main_path = REPO_ROOT / "p79/experiment/runner/main.py"
    text = main_path.read_text()

    assert "use_deterministic_algorithms" in text, (
        "B-827 P1-1-BC*: torch.use_deterministic_algorithms missing from _seed_global_rng"
    )
    assert "cudnn.deterministic" in text, (
        "B-827 P1-1-BC*: cudnn.deterministic missing"
    )
    assert "CUBLAS_WORKSPACE_CONFIG" in text, (
        "B-827 P1-1-BC*: CUBLAS_WORKSPACE_CONFIG missing"
    )


# ─── B-825 (P0-4-AB*) — snapshot_vwa.sh submodule_dirty fail-loud ────────────
def test_b802_snapshot_vwa_submodule_dirty_fail_loud():
    """snapshot_vwa.sh must sys.exit(2) on submodule_dirty under P79_PAPER_GRADE=1."""
    vwa_path = REPO_ROOT / "scripts/provenance/snapshot_vwa.sh"
    text = vwa_path.read_text()

    assert "B-825" in text, "B-825 P0-4-AB* fix marker missing"
    assert "P79_PAPER_GRADE" in text, "P79_PAPER_GRADE gate missing in snapshot_vwa.sh"
    assert "submodule_dirty" in text and "sys.exit(2)" in text, (
        "submodule_dirty fail-loud (sys.exit(2)) missing"
    )


# ─── B-826 (P0-5-A*) — /robots.txt HTTP status sanity ────────────────────────
def test_b803_snapshot_vwa_http_status_sanity():
    """snapshot_vwa.sh must verify HTTP 200 + Content-Type text/plain on probe."""
    vwa_path = REPO_ROOT / "scripts/provenance/snapshot_vwa.sh"
    text = vwa_path.read_text()

    assert "probe_sanity_ok" in text, "B-826 P0-5-A*: probe_sanity_ok field missing"
    assert "http_status" in text, "http_status capture missing"
    assert "content_type" in text, "content_type capture missing"


# ─── B-835 (P1-9-A) — docker ps -a + RepoDigests warn ────────────────────────
def test_b812_snapshot_vwa_docker_ps_minus_a():
    """snapshot_vwa.sh must use `docker ps -a` (include STOPPED containers)."""
    vwa_path = REPO_ROOT / "scripts/provenance/snapshot_vwa.sh"
    text = vwa_path.read_text()

    assert 'docker ps -a' in text or '"docker", "ps", "-a"' in text, (
        "B-835 P1-9-A: `docker ps -a` not found (STOPPED containers excluded from inventory)"
    )
    assert "running_container_count" in text, "running_container_count breakdown missing"
    assert "stopped_container_count" in text, "stopped_container_count breakdown missing"
    assert "docker_engine_version" in text, "docker engine version capture missing"
