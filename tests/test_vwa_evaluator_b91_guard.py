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

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VWA_SUBMODULE = REPO_ROOT / "external" / "visualwebarena"

# A1.18-re lock (memory `reference_vwa_submodule_p79_patches`, 2026-05-17,
# updated post Chunk 1 11-fix sweep). Tree-hash chain SBOM in prereg §7
# per B-580.
EXPECTED_SUBMODULE_SHA = "2f9b0b47175a1bffa01e13100e3075e212161a89"


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
