"""Invariant tests for A1.15b Chunk δ fixes (B-855 + B-856).

- B-855 P1-6 glm_client.py extraction (5 GLM helpers moved out of
  glm_diagnosis_sidecar 1996 LOC into standalone glm_client.py ~200 LOC)
- B-856 P1-10 atomic state writes (temp+rename) + fail-loud load_state
  + mandatory fcntl.LOCK_EX in batch digest JSONL append

Loads modules via importlib + sys.modules registration (matches A1.15 +
A1.16 + Chunk β test pattern for substrate testing).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
GLM_DIR = REPO_ROOT / "scripts/maintenance/glm"


def _load_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gc():
    """Load glm_client.py — must be importable as standalone."""
    # Ensure glm/ in sys.path so sibling imports (glm_client used in
    # glm_diagnosis_sidecar) resolve.
    sys.path.insert(0, str(GLM_DIR))
    yield _load_module("gc_a15bd", "scripts/maintenance/glm/glm_client.py")
    sys.path.remove(str(GLM_DIR))


@pytest.fixture(scope="module")
def ds_d(gc):
    """Load glm_diagnosis_sidecar.py with glm_client already in path."""
    return _load_module("ds_a15bd", "scripts/maintenance/glm/glm_diagnosis_sidecar.py")


# ---- B-855: glm_client extraction + back-compat ----

def test_b855_glm_client_public_api(gc):
    """5 public helpers exposed."""
    for name in (
        "load_glm_config",
        "call_glm_chat",
        "candidate_glm_urls",
        "is_vision_model",
        "extract_balanced_json",
    ):
        assert hasattr(gc, name), f"missing public API: {name}"


def test_b855_glm_client_backcompat_aliases(gc):
    """Underscored back-compat aliases for existing callers."""
    for name in (
        "_load_glm_config",
        "_call_glm_chat",
        "_candidate_glm_urls",
        "_is_vision_model",
        "_extract_balanced_json",
    ):
        assert hasattr(gc, name), f"missing back-compat: {name}"


def test_b855_diagnosis_sidecar_reexports_glm_helpers(ds_d):
    """glm_diagnosis_sidecar re-exports the 5 GLM helpers from glm_client."""
    for name in (
        "_load_glm_config",
        "_call_glm_chat",
        "_candidate_glm_urls",
        "_is_vision_model",
        "_extract_balanced_json",
    ):
        assert hasattr(ds_d, name), f"sidecar must re-export: {name}"


def test_b855_load_glm_config_parses_3lines(gc, tmp_path):
    cfg = tmp_path / "test.glm"
    cfg.write_text("# comment\nhttps://api.test\nmodel-x\nkey-y\n# tail\n")
    parsed = gc.load_glm_config(cfg)
    assert parsed == {
        "endpoint": "https://api.test",
        "model": "model-x",
        "api_key": "key-y",
    }


def test_b855_load_glm_config_raises_on_short(gc, tmp_path):
    cfg = tmp_path / "short.glm"
    cfg.write_text("endpoint\nmodel\n")
    with pytest.raises(ValueError, match="3 lines"):
        gc.load_glm_config(cfg)


@pytest.mark.parametrize(
    "model,expected",
    [
        ("glm-4v", True),
        ("glm-4.6v", True),
        ("glm-5v", True),
        ("glm-4.6", False),
        ("glm-5.1", False),
        ("Qwen3-VL-4B", False),
    ],
)
def test_b855_is_vision_model_heuristic(gc, model, expected):
    assert gc.is_vision_model(model) is expected


def test_b855_candidate_urls_appends_chat_completions(gc):
    assert gc.candidate_glm_urls("https://api.test") == ["https://api.test/chat/completions"]
    assert gc.candidate_glm_urls("https://api.test/") == ["https://api.test/chat/completions"]
    assert gc.candidate_glm_urls("https://api.test/chat/completions") == [
        "https://api.test/chat/completions"
    ]


def test_b855_extract_balanced_json_basic(gc):
    """B-847 logic survives extraction (greedy outer-balanced)."""
    assert gc.extract_balanced_json('{"a":1}') == '{"a":1}'
    assert gc.extract_balanced_json('prefix {"a":1} suffix') == '{"a":1}'


def test_b855_extract_balanced_json_nested_outer_wins(gc):
    """OUTERMOST balanced object — the bug B-847 fixed."""
    nested = '{"outer":[{"inner":1},{"inner":2}]}'
    assert gc.extract_balanced_json(nested) == nested


def test_b855_extract_balanced_json_string_with_braces(gc):
    assert gc.extract_balanced_json('{"msg":"has { in it"}') == '{"msg":"has { in it"}'


def test_b855_extract_balanced_json_unbalanced_returns_none(gc):
    assert gc.extract_balanced_json('{"a":1') is None


def test_b855_extract_balanced_json_no_braces_returns_none(gc):
    assert gc.extract_balanced_json("no braces here") is None


def test_b855_playbook_refresh_imports_from_glm_client():
    """B-855: glm_playbook_refresh.py:37 imports from glm_client (not sidecar)."""
    src = (REPO_ROOT / "scripts/maintenance/glm/glm_playbook_refresh.py").read_text()
    assert "from glm_client import _load_glm_config, _call_glm_chat" in src, (
        "playbook_refresh must import glm_client helpers directly"
    )
    # Confirm pre-fix import is gone
    assert "from glm_diagnosis_sidecar import _load_glm_config" not in src


def test_b855_batch_digest_imports_from_glm_client():
    """B-855: glm_batch_digest.py imports GLM helpers from glm_client."""
    src = (REPO_ROOT / "scripts/maintenance/glm/glm_batch_digest.py").read_text()
    assert "from glm_client import" in src, (
        "batch_digest must import glm_client helpers"
    )
    # Importlib boilerplate for these 3 specific helpers should be gone
    assert "_load_glm_config = sidecar._load_glm_config" not in src


# ---- B-856: atomic state writes + fail-loud load + mandatory locks ----

def test_b856_save_state_atomic_temp_rename(ds_d, tmp_path):
    """_save_state writes via temp+rename atomic pattern."""
    state_file = tmp_path / "state.json"
    ds_d._save_state(state_file, {"key": "value", "count": 42})
    # File exists
    assert state_file.exists()
    # Content correct
    data = json.loads(state_file.read_text())
    assert data["key"] == "value"
    assert data["count"] == 42
    assert "updated_at" in data
    # No leftover .tmp file
    assert not (tmp_path / "state.json.tmp").exists()


def test_b856_save_state_atomic_on_repeated_write(ds_d, tmp_path):
    """Multiple _save_state calls should never leave .tmp file."""
    state_file = tmp_path / "state.json"
    for i in range(5):
        ds_d._save_state(state_file, {"iter": i, "data": list(range(i + 1))})
    assert state_file.exists()
    assert json.loads(state_file.read_text())["iter"] == 4
    # No leftover tmp from any iteration
    assert not list(tmp_path.glob("*.tmp"))


def test_b856_load_state_returns_empty_when_missing(ds_d, tmp_path):
    """Missing file (first-run) still returns {} (legitimate)."""
    state_file = tmp_path / "no_such.json"
    assert ds_d._load_state(state_file) == {}


def test_b856_load_state_fails_loud_on_corrupt(ds_d, tmp_path):
    """B-856: corrupt state file MUST raise (was silent {} pre-fix)."""
    state_file = tmp_path / "corrupt.json"
    state_file.write_text("{ this is { not json")
    with pytest.raises(RuntimeError, match="state corrupt"):
        ds_d._load_state(state_file)


def test_b856_load_state_roundtrip_via_save(ds_d, tmp_path):
    """Save + load round-trip preserves keys."""
    state_file = tmp_path / "roundtrip.json"
    payload = {"contaminated_episodes": [1, 5, 7], "last_trigger_count": 42}
    ds_d._save_state(state_file, payload)
    loaded = ds_d._load_state(state_file)
    assert loaded["contaminated_episodes"] == [1, 5, 7]
    assert loaded["last_trigger_count"] == 42
    assert "updated_at" in loaded


def test_b856_append_jsonl_mandatory_flock_present():
    """B-856: glm_batch_digest._append_jsonl now uses fcntl.LOCK_EX."""
    src = (REPO_ROOT / "scripts/maintenance/glm/glm_batch_digest.py").read_text()
    # Locate _append_jsonl function body
    assert "def _append_jsonl" in src
    # Find lock pattern in function body (allow either fcntl or _fcntl alias)
    fn_start = src.index("def _append_jsonl")
    fn_end = src.index("\n\n\n", fn_start) if "\n\n\n" in src[fn_start:] else len(src)
    fn_body = src[fn_start:fn_end]
    assert "LOCK_EX" in fn_body, "_append_jsonl missing mandatory LOCK_EX"
    assert ".flock(" in fn_body, "_append_jsonl missing flock call"
