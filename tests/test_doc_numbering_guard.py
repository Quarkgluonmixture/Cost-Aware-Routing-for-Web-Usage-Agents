"""Tests for the pre-push doc-numbering guard (scripts/maintenance/check_doc_numbering.py).

Covers the pure logic only (extract + analyze) — no git plumbing — so the
collision model is pinned independent of repo state.
"""
import importlib.util
import pathlib

_SCRIPT = (pathlib.Path(__file__).resolve().parents[1]
           / "scripts" / "maintenance" / "check_doc_numbering.py")
_spec = importlib.util.spec_from_file_location("check_doc_numbering", _SCRIPT)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_extract_sections_top_level_only():
    text = "## 262. a\n### 262.1 sub-header\n## 263. b\nprose mentioning 264 inline\n"
    assert mod.extract(text, "section") == [262, 263]


def test_extract_bugs_all_mentions():
    text = "**B-1828** new\nsee B-01 and B-1828 again\n| B-09 row |\n"
    assert sorted(mod.extract(text, "bug")) == [1, 9, 1828, 1828]


def test_cross_session_collision_flagged():
    base = "## 261. a\n"
    local = "## 261. a\n## 262. mine\n"
    remote = "## 261. a\n## 262. theirs\n"
    res = mod.analyze(local, base, remote, "section", check_selfdup=True)
    assert res["cross"] == [262]
    assert res["next_free"] == 263


def test_distinct_new_numbers_no_collision():
    base = "## 261. a\n"
    local = "## 261. a\n## 262. mine\n"
    remote = "## 261. a\n## 263. theirs\n"
    res = mod.analyze(local, base, remote, "section", check_selfdup=True)
    assert res["cross"] == []


def test_preexisting_duplicate_not_flagged():
    # §132 already duplicated in base (historical) → NOT a newly-introduced dup.
    base = "## 132. a\n## 132. b\n"
    local = "## 132. a\n## 132. b\n## 268. new\n"
    res = mod.analyze(local, base, "", "section", check_selfdup=True)
    assert res["selfdup"] == []


def test_new_self_duplicate_flagged():
    base = "## 261. a\n"
    local = "## 261. a\n## 268. x\n## 268. typo-dup\n"
    res = mod.analyze(local, base, "", "section", check_selfdup=True)
    assert res["selfdup"] == [268]


def test_bug_followup_multimention_not_flagged():
    # bug catalog: same B-number across original + followup is normal, never a dup.
    base = "**B-1830** orig\n"
    local = "**B-1830** orig\n**B-1830** (followup) more\n**B-1835** brand new\n"
    res = mod.analyze(local, base, "", "bug", check_selfdup=False)
    assert res["selfdup"] == []
    assert res["cross"] == []


def test_bug_cross_session_collision_flagged():
    base = "**B-1834** last\n"
    local = "**B-1834** last\n**B-1835** mine\n"
    remote = "**B-1834** last\n**B-1835** theirs\n"
    res = mod.analyze(local, base, remote, "bug", check_selfdup=False)
    assert res["cross"] == [1835]
    assert res["next_free"] == 1836
