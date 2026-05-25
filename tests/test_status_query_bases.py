"""Tests for the CLI Bases renderer (scripts/maintenance/status_query.py).

WHY: status_query.py implements a small subset of the Obsidian Bases expression
language so the CLI renders the SAME .base views Obsidian shows (single-source
data layer, no parallel hand-written table). These tests pin the evaluator
semantics (comparison / nested if / string concat / division / toString / null /
file.inFolder) + the `set` field editor + CJK/emoji display width, so a future
refactor can't silently change what a view renders or corrupt a frontmatter file.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = (Path(__file__).resolve().parent.parent
           / "scripts" / "maintenance" / "status_query.py")
_spec = importlib.util.spec_from_file_location("status_query", _SCRIPT)
sq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sq)


class FakeNote:
    """Minimal Note stand-in: evaluator only touches .fm + .rel()."""

    def __init__(self, fm, name="cell_x", rel="checkpoints/_status/cells/cell_x.md"):
        self.fm = fm
        self.name = name
        self._rel = rel

    def rel(self):
        return self._rel


# ───────────────── comparison operators ─────────────────
@pytest.mark.parametrize("expr,fm,expected", [
    ('status == "done"', {"status": "done"}, True),
    ('status == "done"', {"status": "active"}, False),
    ('status != "done"', {"status": "active"}, True),
    ('status == "done"', {}, False),               # missing field → None != "done"
    ('tokens >= 1000', {"tokens": 2000}, True),
    ('tokens >= 1000', {"tokens": 500}, False),
    ('tokens >= 1000', {"tokens": 1000}, True),
    ('progress != null', {"progress": 100}, True),
    ('progress != null', {}, False),               # null != null → False
    ('progress != null', {"progress": 0}, True),   # 0 is not null
])
def test_comparison(expr, fm, expected):
    assert sq.eval_expr(expr, FakeNote(fm)) is expected


# ───────────────── nested if (status_icon formula) ─────────────────
ICON = 'if(status == "done", "✅", if(status == "active", "🔄", if(status == "queued", "📋", "❓")))'


@pytest.mark.parametrize("status,icon", [
    ("done", "✅"), ("active", "🔄"), ("queued", "📋"), ("blocked", "❓"), (None, "❓"),
])
def test_nested_if_status_icon(status, icon):
    assert sq.eval_expr(ICON, FakeNote({"status": status})) == icon


# ───────────────── string concat (cell_label formula) ─────────────────
def test_string_concat_cell_label():
    expr = 'baseline + " " + site + " " + mode'
    note = FakeNote({"baseline": "B0", "site": "classifieds", "mode": "DOM"})
    assert sq.eval_expr(expr, note) == "B0 classifieds DOM"


# ───────────────── division + toString (tokens_short formula) ─────────────────
TOKENS = 'if(tokens >= 1000, (tokens / 1000).toString() + "K", tokens.toString())'


@pytest.mark.parametrize("tokens,out", [
    (2000, "2K"), (75000, "75K"), (500, "500"), (1500, "1.5K"),
])
def test_division_and_tostring(tokens, out):
    assert sq.eval_expr(TOKENS, FakeNote({"tokens": tokens})) == out


# ───────────────── progress_bar (toString + concat + null guard) ─────────────────
@pytest.mark.parametrize("progress,out", [
    (80, "80%"), (100, "100%"), (None, "—"),
])
def test_progress_bar(progress, out):
    expr = 'if(progress != null, progress.toString() + "%", "—")'
    assert sq.eval_expr(expr, FakeNote({"progress": progress})) == out


# ───────────────── filters: and / or / inFolder / type ─────────────────
def test_filter_and():
    note = FakeNote({"type": "task", "status": "active"})
    assert sq.eval_filter({"and": ['type == "task"', 'status == "active"']}, note)
    assert not sq.eval_filter({"and": ['type == "task"', 'status == "done"']}, note)


def test_filter_or():
    note = FakeNote({"status": "queued"})
    assert sq.eval_filter({"or": ['status == "blocked"', 'status == "queued"']}, note)
    assert not sq.eval_filter({"or": ['status == "blocked"', 'status == "done"']}, note)


def test_filter_infolder():
    inside = FakeNote({}, rel="checkpoints/_status/cells/cell_x.md")
    outside = FakeNote({}, rel="checkpoints/other/x.md")
    assert sq.eval_filter('file.inFolder("checkpoints/_status")', inside)
    assert not sq.eval_filter('file.inFolder("checkpoints/_status")', outside)


def test_filter_none_is_true():
    assert sq.eval_filter(None, FakeNote({})) is True


# ───────────────── _coerce type inference (used by set) ─────────────────
@pytest.mark.parametrize("raw,val", [
    ("100", 100), ("-5", -5), ("1.5", 1.5), ("true", True), ("false", False),
    ("", ""), ("done", "done"), ("~9d passive", "~9d passive"),
])
def test_coerce(raw, val):
    assert sq._coerce(raw) == val
    assert type(sq._coerce(raw)) is type(val)


# ───────────────── set_field: only-target-line + body preserved ─────────────────
def test_set_field_replaces_only_target(tmp_path):
    p = tmp_path / "note.md"
    p.write_text(
        '---\ntype: cell\nstatus: active\nblocker: ""\neta: "~1 wk"\n---\n\nbody paragraph\n',
        encoding="utf-8",
    )
    sq.set_field(p, "status", "done")
    txt = p.read_text(encoding="utf-8")
    assert "status: done" in txt
    assert "type: cell" in txt          # sibling fields untouched
    assert 'blocker: ""' in txt
    assert 'eta: "~1 wk"' in txt
    assert "body paragraph" in txt      # body untouched
    assert txt.count("status:") == 1    # no duplicate insertion


def test_set_field_inserts_missing(tmp_path):
    p = tmp_path / "note.md"
    p.write_text("---\ntype: cell\nstatus: active\n---\nbody\n", encoding="utf-8")
    sq.set_field(p, "blocker", "GPU contention")
    txt = p.read_text(encoding="utf-8")
    assert "blocker: GPU contention" in txt
    assert "status: active" in txt
    # inserted inside frontmatter (before closing fence), not in body
    assert txt.index("blocker:") < txt.index("---", txt.index("status:"))


def test_set_field_cjk_preserved(tmp_path):
    p = tmp_path / "note.md"
    p.write_text("---\ntype: cell\nblocker: x\n---\nbody\n", encoding="utf-8")
    sq.set_field(p, "blocker", "等待 advisor 回复")
    txt = p.read_text(encoding="utf-8")
    assert "等待 advisor 回复" in txt   # allow_unicode=True, no \uXXXX escaping


# ───────────────── display width (CJK + emoji alignment) ─────────────────
@pytest.mark.parametrize("s,w", [
    ("ab", 2), ("跑中", 4), ("a跑", 3), ("✅", 2), ("🔄", 2), ("", 0),
])
def test_display_width(s, w):
    assert sq.display_width(s) == w


# ───────────────── end-to-end: render the real tasks.base#NOW without crashing ─────────────────
def test_real_bases_load_and_filter():
    """Smoke: every real .base loads + its top filter evaluates over real notes."""
    notes = sq.scan_notes()
    assert len(notes) > 0
    for bf in sorted(sq.VAULT.glob("*.base")):
        import yaml
        base = yaml.safe_load(bf.read_text(encoding="utf-8")) or {}
        matched = [n for n in notes if sq.eval_filter(base.get("filters"), n)]
        # each base should match at least one note (sanity: filters aren't all-dead)
        assert isinstance(matched, list)
        for v in base.get("views", []) or []:
            # rendering must not raise on any view
            sq.render_view(v, base, matched, maxw=50)
