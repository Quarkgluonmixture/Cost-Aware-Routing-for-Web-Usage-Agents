"""SHA pin single-source-of-truth invariant — /stress A1.12 P1-6 A (2026-05-17).

B-664 fix. Pre-fix: `EXPECTED_SUBMODULE_SHA` was hardcoded at 4 independent
sites with no SoT enforcement:
  - tests/test_vwa_evaluator_b91_guard.py:30
  - scripts/preflight_v2.sh: expected_sha variable
  - Makefile: LOCK_SHA variable
  - docs/checkpoints/pre_run/preregistration.md §7

2026-05-17 sprint bumped the SHA 3× (eb5cbd8 → 1c3a615 → 2f9b0b4) each
requiring 4-site manual sync. Missing any 1 site = silent OSF contract drift.

This test extracts the SHA literal from each site and asserts pairwise equality
+ equality against `git -C external/visualwebarena rev-parse HEAD`. Drift = RED
CI immediately, instead of latent paper-grade SBOM contract break.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VWA_SUBMODULE = REPO_ROOT / "external" / "visualwebarena"

SHA_RE = re.compile(r"[0-9a-f]{40}")


def _extract_first_40hex(path: Path, anchor_substr: str) -> str:
    """Read `path`, find first line containing `anchor_substr`, return first
    40-hex match on that line (or any subsequent line within a small window).
    Raises if no match — better than silently passing on a missing anchor.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if anchor_substr in line:
            # Search this line + next 3 lines for a 40-hex SHA literal
            for j in range(i, min(i + 4, len(lines))):
                m = SHA_RE.search(lines[j])
                if m:
                    return m.group(0)
    raise RuntimeError(
        f"No 40-hex SHA found near anchor {anchor_substr!r} in {path}"
    )


def test_sha_pin_test_constant_matches_preflight():
    test_sha = _extract_first_40hex(
        REPO_ROOT / "tests" / "test_vwa_evaluator_b91_guard.py",
        "EXPECTED_SUBMODULE_SHA",
    )
    preflight_sha = _extract_first_40hex(
        REPO_ROOT / "scripts" / "preflight_v2.sh",
        "expected_sha=",
    )
    assert test_sha == preflight_sha, (
        f"SHA drift: test ({test_sha[:8]}) vs preflight ({preflight_sha[:8]})"
    )


def test_sha_pin_test_constant_matches_makefile():
    test_sha = _extract_first_40hex(
        REPO_ROOT / "tests" / "test_vwa_evaluator_b91_guard.py",
        "EXPECTED_SUBMODULE_SHA",
    )
    makefile_sha = _extract_first_40hex(
        REPO_ROOT / "Makefile", 'LOCK_SHA="',
    )
    assert test_sha == makefile_sha, (
        f"SHA drift: test ({test_sha[:8]}) vs Makefile ({makefile_sha[:8]})"
    )


def test_sha_pin_test_constant_matches_preregistration():
    test_sha = _extract_first_40hex(
        REPO_ROOT / "tests" / "test_vwa_evaluator_b91_guard.py",
        "EXPECTED_SUBMODULE_SHA",
    )
    prereg_sha = _extract_first_40hex(
        REPO_ROOT / "docs" / "checkpoints" / "pre_run" / "preregistration.md",
        "rev-parse HEAD",
    )
    assert test_sha == prereg_sha, (
        f"SHA drift: test ({test_sha[:8]}) vs preregistration ({prereg_sha[:8]}). "
        f"If you bumped the submodule, update preregistration.md §7 too."
    )


def test_sha_pin_test_constant_matches_actual_submodule_head():
    """Final check: all 4 hardcoded constants must equal the real git HEAD."""
    if not (VWA_SUBMODULE / ".git").exists() and not (VWA_SUBMODULE / ".git").is_file():
        pytest.skip("VWA submodule not initialized (git submodule update --init)")
    test_sha = _extract_first_40hex(
        REPO_ROOT / "tests" / "test_vwa_evaluator_b91_guard.py",
        "EXPECTED_SUBMODULE_SHA",
    )
    proc = subprocess.run(
        ["git", "-C", str(VWA_SUBMODULE), "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode == 0
    actual_head = proc.stdout.strip()
    assert test_sha == actual_head, (
        f"All 4 hardcoded constants are in sync with each other but DRIFT from "
        f"actual git HEAD: pinned={test_sha[:8]} vs HEAD={actual_head[:8]}. "
        f"Either the submodule was bumped without updating the pins, or someone "
        f"updated the pins to a wrong SHA. Verify intent."
    )
