"""B-1996: a single out-of-table character must not be able to kill a condition.

Fire 2026-08-27, cls task 137: the agent emitted a TYPE payload carrying one
character outside VWA's key table. `browser_env.actions._keys2ids` maps an
unknown key to the *string* `" "` while declaring `-> list[int]`, so beartype
raised inside `create_keyboard_type_action`, the runner turned that into
`PaperGradeAbortError`, and Phase C died at 135/224 with three phantom cells
never launched. The observed payload was `[' ', 129982]` — one unknown
character plus the newline the wrapper appends to search queries.

The table is rebuilt here from VWA's own `constants.py` rather than faked, so
this test tracks the real upstream contract: if upstream widens the charset,
these assertions follow it instead of silently guarding a fiction.
"""
import re
from itertools import chain
from pathlib import Path

import pytest

from p79.envs.vwa_wrapper import _sanitize_keyboard_text

REPO = Path(__file__).resolve().parents[1]
CONSTANTS = REPO / "external/visualwebarena/browser_env/constants.py"


def _real_key2id():
    """Rebuild `_key2id` exactly as browser_env.actions does, without importing
    the package (which needs DATASET + live site URLs in the environment)."""
    src = CONSTANTS.read_text()
    m = re.search(r"SPECIAL_KEYS = \((.*?)\n\)", src, re.S)
    special = tuple(re.findall(r'"([^"]+)"', m.group(1)))
    ascii_set = "".join(chr(x) for x in range(32, 128))
    freq_set = "".join(chr(x) for x in range(129, 130000))
    return {k: i for i, k in enumerate(chain(special, ascii_set, freq_set, ["\n"]))}


@pytest.fixture
def key2id(monkeypatch):
    """Serve the real table through the import site the helper uses."""
    import sys
    import types

    table = _real_key2id()
    mod = types.ModuleType("browser_env.actions")
    mod._key2id = table
    pkg = sys.modules.get("browser_env") or types.ModuleType("browser_env")
    monkeypatch.setitem(sys.modules, "browser_env", pkg)
    monkeypatch.setitem(sys.modules, "browser_env.actions", mod)
    return table


def _keys2ids(keys, table):
    """The upstream mapping, verbatim (actions.py:429-436)."""
    from beartype.door import is_bearable

    return [
        table.get(str(k), table.get(k, " ")) if is_bearable(k, str) else int(k)
        for k in keys
    ]


# --- the payload that actually killed the fire -----------------------------


def test_the_task137_payload_no_longer_violates_the_int_contract(key2id):
    """One out-of-table char + the appended newline — the exact fire shape."""
    payload = "\x80\n"  # chr(128) is outside chr(32..127) and chr(129..129999)

    assert any(
        not isinstance(v, int) for v in _keys2ids(payload, key2id)
    ), "precondition: this payload must break the upstream contract, else the test is vacuous"

    cleaned, meta = _sanitize_keyboard_text(payload)

    assert all(isinstance(v, int) for v in _keys2ids(cleaned, key2id))
    assert meta["dropped_count"] == 1
    assert meta["dropped_codepoints"] == ["0x80"]


@pytest.mark.parametrize(
    "bad",
    [chr(0), chr(9), chr(13), chr(27), chr(31), chr(128), chr(130000), chr(0x20000)],
    ids=["nul", "tab", "cr", "esc", "us", "c128", "c130000", "cjk_ext_b"],
)
def test_every_out_of_table_class_is_removed(key2id, bad):
    cleaned, meta = _sanitize_keyboard_text(f"ab{bad}cd")
    assert cleaned == "abcd"
    assert meta["dropped_count"] == 1
    assert all(isinstance(v, int) for v in _keys2ids(cleaned, key2id))


# --- what must NOT be touched ----------------------------------------------


def test_newline_survives(key2id):
    """The wrapper appends '\\n' to fire a search. Dropping it would silently
    break every search task — a quieter failure than the crash being fixed."""
    cleaned, meta = _sanitize_keyboard_text("Bastien Piano Basics\n")
    assert cleaned == "Bastien Piano Basics\n"
    assert meta is None


@pytest.mark.parametrize(
    "text",
    ["", "plain ascii", "with  spaces", "punct!?.,-_@#$%", "café ünïcode", "日本語テキスト"],
)
def test_in_table_text_is_returned_unchanged_and_untagged(key2id, text):
    cleaned, meta = _sanitize_keyboard_text(text)
    assert cleaned == text
    assert meta is None, "no telemetry when nothing was dropped — None is the signal"


def test_list_payloads_are_supported(key2id):
    cleaned, meta = _sanitize_keyboard_text(["a", chr(128), "b"])
    assert cleaned == ["a", "b"]
    assert meta["dropped_count"] == 1


def test_int_keys_pass_through(key2id):
    """_keys2ids sends non-str keys down int(key); they cannot hit the default."""
    cleaned, meta = _sanitize_keyboard_text([65, 66])
    assert cleaned == [65, 66]
    assert meta is None


# --- degradation behaviour --------------------------------------------------


def test_fails_open_when_the_table_cannot_be_imported(monkeypatch):
    """No table → return the payload untouched rather than mangling it. The
    caller then behaves exactly as it did before this fix."""
    import builtins

    real_import = builtins.__import__

    def _boom(name, *a, **kw):
        if name == "browser_env.actions":
            raise ImportError("no VWA env")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _boom)
    text = f"ab{chr(128)}"
    cleaned, meta = _sanitize_keyboard_text(text)
    assert cleaned == text
    assert meta is None


def test_all_characters_dropped_yields_empty_not_crash(key2id):
    cleaned, meta = _sanitize_keyboard_text(chr(128) + chr(0))
    assert cleaned == ""
    assert meta["dropped_count"] == 2
    assert meta["kept_len"] == 0


def test_non_text_payload_is_left_alone(key2id):
    for payload in (None, 42, {"a": 1}):
        cleaned, meta = _sanitize_keyboard_text(payload)
        assert cleaned is payload
        assert meta is None
