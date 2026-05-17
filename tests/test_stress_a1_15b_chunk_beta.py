"""Invariant tests for A1.15b Chunk β fixes (B-845, B-846, B-847).

Cross-AI 3-AI cycle output:
- B-845 P1-2 phantom mode central normalize (Mode B codex OOB catch)
- B-846 P1-3 reference image parents[3] path + _extract_site fallback (codex OOB)
- B-847 P1-9 rfind balanced-brace JSON extraction (codex OOB)

Scope: `scripts/maintenance/glm/{glm_batch_digest, glm_diagnosis_sidecar}.py`.
Test pattern: importlib + sys.modules registration (matches A1.15 + A1.16
test style for watchdog substrate loading).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
    """Load a script-style module via importlib + sys.modules registration."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bd():
    return _load_module("bd_a15b", "scripts/maintenance/glm/glm_batch_digest.py")


@pytest.fixture(scope="module")
def ds():
    return _load_module("ds_a15b", "scripts/maintenance/glm/glm_diagnosis_sidecar.py")


# ---- B-845: _normalize_obs_mode ----

@pytest.mark.parametrize(
    "raw,canonical",
    [
        ("phantom_som", "som"),
        ("phantom_dom", "dom"),
        ("phantom_text", "dom"),
        ("phantom_prompt", "dom"),
        ("som", "som"),
        ("dom", "dom"),
        ("vision", "vision"),
        ("", ""),
        ("  Phantom_SOM  ", "som"),
        ("PHANTOM_DOM", "dom"),
    ],
)
def test_b845_normalize_obs_mode(bd, raw, canonical):
    """Phantom modes map to canonical {dom, som, vision} digest bucket."""
    assert bd._normalize_obs_mode(raw) == canonical


def test_b845_normalize_preserves_unknown(bd):
    """Unknown modes pass through unchanged (with strip+lower)."""
    assert bd._normalize_obs_mode("unknown_mode") == "unknown_mode"


# ---- B-846: _extract_site + reference image path ----

@pytest.mark.parametrize(
    "case,expected",
    [
        ({"site": "reddit"}, "reddit"),                              # explicit valid
        ({"site": "classifieds"}, "classifieds"),                    # explicit valid
        ({"site": "shopping_admin"}, "shopping_admin"),              # explicit valid
        ({"run_dir": "B0_3mode_classifieds_20260413"}, "classifieds"),
        ({"run_dir": "B1_3mode_shopping_20260413"}, "shopping"),
        ({"run_dir": "B0_3mode_shopping_admin_20260413"}, "shopping_admin"),
        ({"run_dir": "B1_phantom_som_reddit_20260501"}, "reddit"),
        ({"site": "invalid_site"}, ""),                              # explicit invalid → fallback
        ({"run_dir": "no_site_anchor"}, ""),                         # no _site_\d{8} pattern
        ({"condition_id": "phase1_dom_router_0"}, ""),               # cond_id only insufficient
        ({}, ""),                                                    # empty case
    ],
)
def test_b846_extract_site(bd, case, expected):
    """_extract_site walks (site → run_dir → fallback) inference chain."""
    assert bd._extract_site(case) == expected


def test_b846_extract_site_shopping_admin_anchored(bd):
    """Anchored regex avoids shopping/shopping_admin substring collision.

    Pre-anchor: `"shopping"` would substring-match
    `B0_3mode_shopping_admin_20260413`. Anchored `_shopping_\\d{8}` won't.
    Longest-first sweep ensures shopping_admin matches before shopping.
    """
    # The crucial test: shopping_admin must NOT collapse to shopping
    case_admin = {"run_dir": "B0_3mode_shopping_admin_20260413"}
    case_shop = {"run_dir": "B1_3mode_shopping_20260413"}
    assert bd._extract_site(case_admin) == "shopping_admin"
    assert bd._extract_site(case_shop) == "shopping"
    # Verify they don't collide
    assert bd._extract_site(case_admin) != bd._extract_site(case_shop)


def test_b846_reference_image_path_resolves_to_repo_root(bd):
    """parents[3] from glm_batch_digest.py resolves to repo root, not scripts/."""
    bd_path = Path(bd.__file__).resolve()
    # glm_batch_digest.py lives at scripts/maintenance/glm/
    # parents[3] should = repo root
    repo = bd_path.parents[3]
    vwa_root = repo / "external" / "visualwebarena"
    # Verify path actually exists (paper §3 ref-image claim requires this dir)
    assert vwa_root.exists(), f"VWA root must exist post-fix: {vwa_root}"
    # Verify the OLD (broken) path doesn't exist
    broken_root = bd_path.parents[2] / "external" / "visualwebarena"
    assert not broken_root.exists(), (
        f"pre-fix path should not exist: {broken_root}"
    )


# ---- B-847: _extract_balanced_json ----

def test_b847_simple_flat(ds):
    assert ds._extract_balanced_json('{"a":1}') == '{"a":1}'


def test_b847_nested_outer_wins(ds):
    """OUTERMOST balanced object returned, not innermost (the bug)."""
    nested = 'prefix {"outer": [{"inner": 1}, {"inner": 2}]} suffix'
    got = ds._extract_balanced_json(nested)
    expected = '{"outer": [{"inner": 1}, {"inner": 2}]}'
    assert got == expected
    # Confirm OLD rfind approach would have given wrong answer
    # rfind("{") = position of last { = position before "inner": 2
    rfind_start = nested.rfind("{")
    rfind_end = nested.rfind("}")
    rfind_result = nested[rfind_start : rfind_end + 1]
    assert rfind_result != expected, "rfind should give wrong answer for nested"


def test_b847_string_with_braces(ds):
    """Braces inside string literals don't shift depth."""
    quoted = '{"msg": "this has { in it"}'
    assert ds._extract_balanced_json(quoted) == quoted


def test_b847_escaped_quote(ds):
    r"""Backslash-escaped quotes (\") handled correctly."""
    esc = '{"msg": "a \\"quoted{\\" } b"}'
    assert ds._extract_balanced_json(esc) == esc


def test_b847_unbalanced_returns_none(ds):
    assert ds._extract_balanced_json('{"a": 1') is None


def test_b847_no_braces_returns_none(ds):
    assert ds._extract_balanced_json('no braces here') is None


def test_b847_empty_returns_none(ds):
    assert ds._extract_balanced_json('') is None


def test_b847_multiple_top_level_returns_first(ds):
    """Greedy: first balanced object found wins."""
    multi = '{"first": 1} {"second": 2}'
    assert ds._extract_balanced_json(multi) == '{"first": 1}'


def test_b847_glm_thinking_model_realistic(ds):
    """Realistic GLM reasoning_content shape: prose + JSON wrapped in
    markdown fence + trailing prose. Verify outer JSON object survives."""
    glm_response = '''Let me analyze the failure mode here.

The agent's trajectory shows a clear pattern of getting stuck.

```json
{
  "failure_diagnosis": [
    {"step": 0, "type": "navigation"},
    {"step": 5, "type": "redirect_loop"}
  ],
  "root_cause": "Magento 302 cycle",
  "confidence": 0.85
}
```

Hope this helps.'''
    got = ds._extract_balanced_json(glm_response)
    assert got is not None
    assert '"failure_diagnosis"' in got
    assert '"root_cause"' in got
    assert '"confidence"' in got
    # Verify it's parseable JSON
    import json
    parsed = json.loads(got)
    assert len(parsed["failure_diagnosis"]) == 2
    assert parsed["root_cause"] == "Magento 302 cycle"
