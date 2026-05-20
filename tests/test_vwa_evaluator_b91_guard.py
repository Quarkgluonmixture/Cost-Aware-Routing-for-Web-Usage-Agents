"""VWA submodule B-91 empty-prediction guard invariant — /stress A1.12 P0-5.

B-91 fix (2026-05-14, submodule branch p79-patches): both `llm_fuzzy_match`
and `llm_ua_match` in `external/visualwebarena/evaluation_harness/helper_functions.py`
return `0.0` immediately when `pred` is empty or whitespace-only.

Pre-fix vector: agent never calls `finish()` → P79 runner injects fake stop
action with `answer=""` → upstream LLM judge could score `""` as `'correct'`
when reference also empty-like → string_match false positive inflates SR
~2-3pp on N/A-heavy tasks.

This test exercises the guard at runtime (no OpenAI key needed — guard
returns before API call), AND cross-checks the submodule SHA pin so a
silent submodule reset can't bypass the contract.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VWA_SUBMODULE = REPO_ROOT / "external" / "visualwebarena"

# A1.18-re lock (memory `reference_vwa_submodule_p79_patches`, 2026-05-17,
# updated post Chunk 1 11-fix sweep). Tree-hash chain SBOM in prereg §7
# per B-580.
EXPECTED_SUBMODULE_SHA = "2c15d66d120f8498633ae65057aa50a34b3e93e0"  # Fire-6 C1b re-lock (was ac33d2f)
# B-663 (/stress A1.12 P0-4 C* gemini, 2026-05-17): tree-hash chain is the
# IMMUTABLE witness per prereg §7 — HEAD SHA above is mutable under `git push
# --force-with-lease`. The tree-hash chain recipe is environment-independent
# (vs `git diff base..HEAD | sha256sum` which varied on diff.algorithm /
# core.autocrlf / git version). Recipe:
#   git rev-list <upstream-base>..HEAD --format=tformat:'%H %T' | sha256sum
UPSTREAM_BASE_SHA = "89f5af29305c3d1e9f97ce4421462060a70c9a03"
EXPECTED_TREE_HASH_CHAIN = "2696d0a61e2f70536f247ebb225f51c262b657d8b8b7b407f8581b75757a8bae"  # Fire-6 C1b re-lock (was 752caeb)


@pytest.fixture(autouse=True)
def _vwa_on_path():
    """Add submodule root to sys.path for `from evaluation_harness ...`.

    Set placeholder env vars so module-level imports don't error:
    - `OPENAI_API_KEY`: VWA provider modules read at import time
    - `DATASET`: `browser_env/env_config.py` reads at import time
    - VWA site URLs: `browser_env/env_config.py` reads at import time
    The B-91 guard runs before any actual API call so placeholders suffice.
    """
    import os
    os.environ.setdefault("OPENAI_API_KEY", "DUMMY_P79_B91_GUARD_TEST")
    os.environ.setdefault("DATASET", "visualwebarena")
    os.environ.setdefault("SHOPPING", "http://localhost:7770")
    os.environ.setdefault("REDDIT", "http://localhost:9999")
    os.environ.setdefault("WIKIPEDIA", "http://localhost:8888")
    os.environ.setdefault("CLASSIFIEDS", "http://localhost:9980")
    os.environ.setdefault("HOMEPAGE", "http://localhost:4399")
    os.environ.setdefault("CLASSIFIEDS_RESET_TOKEN", "dummy_token_for_b91_test")
    if str(VWA_SUBMODULE) not in sys.path:
        sys.path.insert(0, str(VWA_SUBMODULE))
    yield


# ─── Direct runtime exercise of the guard ───────────────────────────────────
def test_llm_fuzzy_match_returns_zero_for_empty_pred():
    from evaluation_harness.helper_functions import llm_fuzzy_match
    # `pred=""` triggers `if not pred or not pred.strip(): return 0.0` at line 589
    # before any OpenAI call — no API key required.
    assert llm_fuzzy_match("", "anything", "what was the answer?") == 0.0


def test_llm_fuzzy_match_returns_zero_for_whitespace_pred():
    from evaluation_harness.helper_functions import llm_fuzzy_match
    assert llm_fuzzy_match("   \n\t  ", "anything", "q") == 0.0


def test_llm_ua_match_returns_zero_for_empty_pred():
    from evaluation_harness.helper_functions import llm_ua_match
    # Same guard at line 677 of the same file.
    assert llm_ua_match("", "anything", "q") == 0.0


def test_llm_ua_match_returns_zero_for_whitespace_pred():
    from evaluation_harness.helper_functions import llm_ua_match
    assert llm_ua_match("\t  \n", "anything", "q") == 0.0


# ─── SBOM cross-check — defense in depth ────────────────────────────────────
def test_vwa_submodule_sha_matches_a1_18_lock():
    """If submodule got `git reset` / merge-from-upstream, the guard may not be
    present at the source paths this test imports. SHA pin is the OSF lock
    boundary; preflight + Makefile pre-launch-check both verify, but a unit
    test caught by `make test` surfaces drift earlier.
    """
    if not (VWA_SUBMODULE / ".git").exists() and not (VWA_SUBMODULE / ".git").is_file():
        pytest.skip("VWA submodule not initialized (git submodule update --init)")
    proc = subprocess.run(
        ["git", "-C", str(VWA_SUBMODULE), "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode == 0, f"git rev-parse failed: {proc.stderr}"
    actual = proc.stdout.strip()
    assert actual == EXPECTED_SUBMODULE_SHA, (
        f"VWA submodule SHA drift: expected {EXPECTED_SUBMODULE_SHA}, got {actual}. "
        f"If this is an intentional bump, update EXPECTED_SUBMODULE_SHA in this file "
        f"+ Makefile LOCK_SHA + scripts/preflight_v2.sh expected_sha + "
        f"memory `reference_vwa_submodule_p79_patches`."
    )


def test_vwa_submodule_tree_hash_chain_matches_prereg_witness():
    """B-663 (P0-4 C* gemini OOB): tree-hash chain is the OSF immutable witness.

    Pre-fix: `test_vwa_submodule_sha_matches_a1_18_lock` only checks HEAD SHA,
    which prereg §7 explicitly notes is mutable under
    `git push --force-with-lease`. The tree-hash chain over the commit history
    from upstream-base to HEAD is byte-deterministic across git versions / OS
    environments because it uses git's content-addressable object IDs (%H + %T).

    OSF replayer must independently re-derive this hash to verify the
    `p79-patches` branch history hasn't been rewritten between lock time and
    audit time. Hardcoding it in the test mirrors prereg §7's contract.
    """
    if not (VWA_SUBMODULE / ".git").exists() and not (VWA_SUBMODULE / ".git").is_file():
        pytest.skip("VWA submodule not initialized (git submodule update --init)")
    cmd = [
        "git", "-C", str(VWA_SUBMODULE), "rev-list",
        f"{UPSTREAM_BASE_SHA}..HEAD",
        "--format=tformat:%H %T",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0, (
        f"git rev-list failed (upstream-base {UPSTREAM_BASE_SHA} may be unreachable): "
        f"{proc.stderr}"
    )
    actual = hashlib.sha256(proc.stdout.encode("utf-8")).hexdigest()
    assert actual == EXPECTED_TREE_HASH_CHAIN, (
        f"VWA tree-hash chain mismatch:\n"
        f"  expected: {EXPECTED_TREE_HASH_CHAIN}\n"
        f"  actual:   {actual}\n"
        f"If this is an intentional submodule bump, update EXPECTED_TREE_HASH_CHAIN "
        f"in this file + preregistration.md §7 witness + memory file. "
        f"Otherwise the p79-patches branch history was rewritten — OSF contract broken."
    )


def test_b91_guard_source_present_at_both_callsites():
    """Defense in depth: even if SHA matches, verify guard *code* literally lives
    at the documented callsites (B-91 grep contract, paper §3.5 disclosure)."""
    hf_path = VWA_SUBMODULE / "evaluation_harness/helper_functions.py"
    if not hf_path.exists():
        pytest.skip(f"helper_functions.py not present at {hf_path}")
    src = hf_path.read_text(encoding="utf-8")
    guard = "if not pred or not pred.strip():"
    count = src.count(guard)
    assert count >= 2, (
        f"B-91 guard `{guard}` expected at ≥2 sites (llm_fuzzy_match + llm_ua_match), "
        f"found {count}. Check submodule for unintended reset / merge."
    )
