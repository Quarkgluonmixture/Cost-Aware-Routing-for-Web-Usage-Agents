"""Wave 1 regression: queue_chain.sh baseline-aware wallclock regex.

Catches the Fire-4 RCA bug where `=~ _B0_` failed to match the actual run_id
`B0_dom_classifieds_...` because run_id has no leading underscore. The fixed
regex `(^|_)B0_` must match B0 prefix AND `_B0_` substring patterns, while
NOT matching B1 / B2 patterns.

This test invokes bash directly because the regex lives in shell, not Python.
"""

import shutil
import subprocess

import pytest


def _bash_regex_match(pattern: str, regex: str) -> bool:
    """Return True if bash `[[ "$pattern" =~ $regex ]]` matches."""
    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    cmd = f'pattern="{pattern}"; if [[ "$pattern" =~ {regex} ]]; then echo MATCH; else echo NOMATCH; fi'
    result = subprocess.run(
        ["bash", "-c", cmd], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() == "MATCH"


@pytest.mark.parametrize(
    "run_id,expected",
    [
        # B0 prefix (typical Phase 1a paper-grade run_id) — must match
        ("B0_dom_classifieds_20260519_174455_061534393_281104_R12090", True),
        ("B0_som_reddit_20260520_010203_abcdef_pid_R9999", True),
        ("B0_phantom_som_classifieds_20260520_080808_x_1_R1", True),
        # B0 as substring with underscore boundary (legacy formats) — must match
        ("phase1_B0_dom_classifieds_20260519_174455", True),
        ("rerun_B0_vision_reddit_20260520", True),
        # B1 / B2 prefix — must NOT match (4h budget falls through to else)
        ("B1_dom_classifieds_20260519_174455_x_pid_R1", False),
        ("B2_som_reddit_20260520_010203_y_pid_R2", False),
        ("B1_phantom_text_classifieds_20260520_080808", False),
        # Edge: NB0 / XB0 / SubB0 — must NOT match (no underscore/start boundary)
        ("NB0_dom_classifieds_20260519", False),
        ("XB0_test", False),
    ],
)
def test_baseline_aware_regex(run_id: str, expected: bool) -> None:
    """Fire-4 Wave 1 regression: `(^|_)B0_` regex must correctly distinguish B0 from B1/B2."""
    assert _bash_regex_match(run_id, r"(^|_)B0_") is expected


def test_buggy_regex_demonstrates_bug() -> None:
    """Demonstrates the pre-fix regex `_B0_` FAILED on prefix B0 patterns.

    This test exists to document the original bug + lock in the fix. If this
    test ever fails (i.e., `_B0_` starts matching prefix `B0_...`), bash
    semantics changed and the fix in queue_chain.sh:135 needs re-validation.
    """
    # Pre-fix regex: requires underscore BEFORE B0 → fails on prefix
    assert _bash_regex_match("B0_dom_classifieds_20260519_174455", r"_B0_") is False
    # Fix regex: matches prefix OR underscore-bounded
    assert _bash_regex_match("B0_dom_classifieds_20260519_174455", r"(^|_)B0_") is True
