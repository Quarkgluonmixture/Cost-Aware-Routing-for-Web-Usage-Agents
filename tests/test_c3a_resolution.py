"""Fire-6 RCA Stage C3a (/stress 2026-05-20): evidence-gated, per-error-class
cross-fire-recurrence resolution. Clears Gate 8 Rule 2 ONLY when EVERY distinct
quarantine error_class has its own valid resolution (the eval-goto C1 resolution
must NOT clear the screenshot-timeout class). All checks fail-closed.

Uses REAL repo commits so the git-ancestry gate is genuinely exercised:
  e9875cc = RESOLUTION_COMMIT_FLOOR (C2 classification, contains C1)
  a211ec5 = C1b (descendant of floor, ancestor of HEAD) — screenshot fix
  3c767a2 = C1 (ANCESTOR of floor) → must be rejected (not floor-or-descendant)
  deadbeef… = bad object → rejected
"""
from __future__ import annotations

import json

import pytest

from scripts.maintenance import quarantine_registry as qr

EVAL_GOTO_EC = "EvaluatorUnavailableError (Page.goto: Timeout 30000ms)"
SCREENSHOT_EC = "Page.screenshot Timeout 30000ms"

FLOOR = "e9875cc"   # eval-goto valid (== floor)
C1B = "a211ec5"     # screenshot valid (descendant of floor, ancestor of HEAD)
PRE_FLOOR = "3c767a2"   # ancestor of floor → invalid
BAD = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"


@pytest.fixture
def reg(tmp_path, monkeypatch):
    r = tmp_path / "quarantine_registry.jsonl"
    monkeypatch.setattr(qr, "REGISTRY_PATH", r)
    return r


def _episode(tmp_path, name, producer=None, **over):
    """Write a diagnostic episode summary. If producer is given, place it under a
    run dir with an env_snapshot.json carrying git.commit=producer (so the P0-2
    screenshot producer-commit check can resolve it)."""
    base = {
        "task_id": 75, "benchmark_site": "classifieds",
        "diagnostic_replay": True, "sr_excluded": True,
        "needs_reevaluation": False, "eval_context_mode": "isolated_program_html_context",
        "eval_goto_timeout": False, "success": True,
    }
    base.update(over)
    if producer is not None:
        run_root = tmp_path / f"run_{name.replace('.', '_')}"
        ep_dir = run_root / "cond" / "episodes"
        ep_dir.mkdir(parents=True, exist_ok=True)
        (run_root / "env_snapshot.json").write_text(
            json.dumps({"git": {"commit": producer}}))
        p = ep_dir / name
    else:
        p = tmp_path / name
    p.write_text(json.dumps(base))
    return str(p)


def _quarantine_two_classes(reg):
    # mirror real cls-75: Fire-3 eval-goto + Fire-4 screenshot (2 distinct fires).
    qr.append_quarantine(site="classifieds", task_id=75, run_id="FIRE_A",
                         error_class=EVAL_GOTO_EC, callsite="agent_navigation")
    qr.append_quarantine(site="classifieds", task_id=75, run_id="FIRE_B",
                         error_class=SCREENSHOT_EC, callsite="agent_observation")
    # Classify both so unclassified_count = 0 → Rule 1 OFF, isolating Rule 2
    # (the recurrence gate that resolution clears). Matches real cls-75 state
    # (fully classified but still cross_fire_recurrence-blocked pre-resolution).
    for _ in range(2):
        qr.append_classification(site="classifieds", task_id=75,
                                 classification="evaluator", classified_by="test",
                                 rationale="matched-temporal-context", classified_via="x")


# ── append_resolution write-time verification ────────────────────────────────
def test_eval_goto_resolution_accepted(reg, tmp_path):
    ep = _episode(tmp_path, "good.json")
    qr.append_resolution(site="classifieds", task_id=75, error_class=EVAL_GOTO_EC,
                         resolved_by_commit=FLOOR, episode_summary_path=ep,
                         resolved_by="test", rationale="C2 isolated eval")
    ok, _ = qr.is_error_class_resolved("classifieds", 75, EVAL_GOTO_EC)
    assert ok


def test_screenshot_resolution_accepted_with_c1b(reg, tmp_path):
    ep = _episode(tmp_path, "clean.json", producer=C1B)  # episode produced AT C1b
    qr.append_resolution(site="classifieds", task_id=75, error_class=SCREENSHOT_EC,
                         resolved_by_commit=C1B, episode_summary_path=ep,
                         resolved_by="test", rationale="C1b architectural")
    ok, _ = qr.is_error_class_resolved("classifieds", 75, SCREENSHOT_EC)
    assert ok


# ── P1-2 (Q5=A): honest per-profile classified_via label ─────────────────────
def test_p1_2_screenshot_resolution_labeled_architectural_containment(reg, tmp_path):
    """Q5=A: a screenshot resolution must be labeled architectural_containment_c1b,
    NOT matched_temporal_context — the load-dependent Page.screenshot timeout
    cannot be reproduced in a small replay, so the resolution is C1b containment,
    not a temporal reproduction (OSF/prereg honesty)."""
    ep = _episode(tmp_path, "clean.json", producer=C1B)
    ev = qr.append_resolution(site="classifieds", task_id=75, error_class=SCREENSHOT_EC,
                              resolved_by_commit=C1B, episode_summary_path=ep,
                              resolved_by="test", rationale="C1b architectural")
    assert ev["classified_via"] == "architectural_containment_c1b"
    assert ev["classified_via"] != qr.RESOLUTION_REQUIRED_VIA


def test_p1_2_eval_goto_keeps_matched_temporal_label(reg, tmp_path):
    """eval_goto IS a matched-temporal-context reproduction (C1 isolation under
    real A100 load) → keeps that honest label, distinct from screenshot."""
    ep = _episode(tmp_path, "good.json")
    ev = qr.append_resolution(site="classifieds", task_id=75, error_class=EVAL_GOTO_EC,
                              resolved_by_commit=FLOOR, episode_summary_path=ep,
                              resolved_by="test", rationale="C1 isolated eval")
    assert ev["classified_via"] == "matched_temporal_context_diagnostic_replay"


def test_p1_2_mislabeled_screenshot_resolution_rejected(reg, tmp_path):
    """The pre-fix flat-constant label on a screenshot resolution is now REJECTED
    (exact registry state the audit caught for cls-75) — forces re-resolution
    with the honest architectural_containment_c1b label."""
    _quarantine_two_classes(reg)
    ep = _episode(tmp_path, "clean.json", producer=C1B)
    qr._append_event({
        "event_type": "resolution", "ts": "2026-05-20T13:00:00+00:00",
        "site": "classifieds", "task_id": 75, "error_class": SCREENSHOT_EC,
        "evidence_profile": "screenshot",
        "classified_via": "matched_temporal_context_diagnostic_replay",  # OLD mislabel
        "resolved_by_commit": C1B, "episode_summary_path": ep,
        "resolved_by": "legacy", "rationale": "mislabeled pre-Q5A",
    })
    ok, reason = qr.is_error_class_resolved("classifieds", 75, SCREENSHOT_EC)
    assert ok is False
    assert "classified_via" in reason


# ── P0-2: screenshot evidence must be produced AT C1b ────────────────────────
def test_p0_2_screenshot_rejects_pre_c1b_producer(reg, tmp_path):
    # episode produced at 9d46134 (C2, PRE-C1b) → screenshot resolution rejected
    ep = _episode(tmp_path, "prec1b.json", producer="9d46134")
    with pytest.raises(ValueError, match="producer|C1b-or-descendant"):
        qr.append_resolution(site="classifieds", task_id=75, error_class=SCREENSHOT_EC,
                             resolved_by_commit=C1B, episode_summary_path=ep,
                             resolved_by="test", rationale="pre-C1b episode")


def test_p0_2_screenshot_rejects_missing_env_snapshot(reg, tmp_path):
    # bare episode (no env_snapshot.json) → no producer commit → rejected
    ep = _episode(tmp_path, "noenv.json")  # producer=None
    with pytest.raises(ValueError, match="producer"):
        qr.append_resolution(site="classifieds", task_id=75, error_class=SCREENSHOT_EC,
                             resolved_by_commit=C1B, episode_summary_path=ep,
                             resolved_by="test", rationale="no env_snapshot")


# ── P1-3: reject mutable/symbolic refs ───────────────────────────────────────
@pytest.mark.parametrize("ref", ["HEAD", "master", "main", "origin/master"])
def test_p1_3_rejects_mutable_ref(reg, tmp_path, ref):
    ep = _episode(tmp_path, "g.json")
    with pytest.raises(ValueError, match="hex SHA|symbolic|mutable"):
        qr.append_resolution(site="classifieds", task_id=75, error_class=EVAL_GOTO_EC,
                             resolved_by_commit=ref, episode_summary_path=ep,
                             resolved_by="test", rationale="mutable ref")


# ── P0-1: preflight rejects a resolution pointing at another task's episode ───
def test_p0_1_preflight_rejects_wrong_task_episode(reg, tmp_path):
    _quarantine_two_classes(reg)
    # hand-write a resolution (bypassing append_resolution's write check) that
    # points at a task-4 clean episode — preflight re-check must reject it.
    wrong_ep = _episode(tmp_path, "task4.json", task_id=4)
    qr._append_event({
        "event_type": "resolution", "ts": "2026-05-20T10:00:00+00:00",
        "site": "classifieds", "task_id": 75, "error_class": EVAL_GOTO_EC,
        "evidence_profile": "eval_goto",
        "classified_via": qr.RESOLUTION_REQUIRED_VIA,
        "resolved_by_commit": FLOOR, "episode_summary_path": wrong_ep,
        "resolved_by": "attacker", "rationale": "wrong-task episode",
    })
    ok, reason = qr.is_error_class_resolved("classifieds", 75, EVAL_GOTO_EC)
    assert ok is False
    assert "task_id" in reason


def test_eval_goto_rejected_if_timeout_true(reg, tmp_path):
    ep = _episode(tmp_path, "bad.json", eval_goto_timeout=True)
    with pytest.raises(ValueError, match="eval_goto_timeout"):
        qr.append_resolution(site="classifieds", task_id=75, error_class=EVAL_GOTO_EC,
                             resolved_by_commit=FLOOR, episode_summary_path=ep,
                             resolved_by="test", rationale="x")


def test_rejected_if_not_diagnostic_replay(reg, tmp_path):
    ep = _episode(tmp_path, "nd.json", diagnostic_replay=False)
    with pytest.raises(ValueError, match="diagnostic_replay"):
        qr.append_resolution(site="classifieds", task_id=75, error_class=EVAL_GOTO_EC,
                             resolved_by_commit=FLOOR, episode_summary_path=ep,
                             resolved_by="test", rationale="x")


def test_rejected_if_not_sr_excluded(reg, tmp_path):
    ep = _episode(tmp_path, "nx.json", sr_excluded=False)
    with pytest.raises(ValueError, match="sr_excluded"):
        qr.append_resolution(site="classifieds", task_id=75, error_class=EVAL_GOTO_EC,
                             resolved_by_commit=FLOOR, episode_summary_path=ep,
                             resolved_by="test", rationale="x")


def test_rejected_commit_below_floor(reg, tmp_path):
    ep = _episode(tmp_path, "g.json")
    with pytest.raises(ValueError, match="floor|descendant"):
        qr.append_resolution(site="classifieds", task_id=75, error_class=EVAL_GOTO_EC,
                             resolved_by_commit=PRE_FLOOR, episode_summary_path=ep,
                             resolved_by="test", rationale="x")


def test_rejected_bad_commit(reg, tmp_path):
    ep = _episode(tmp_path, "g.json")
    with pytest.raises(ValueError):
        qr.append_resolution(site="classifieds", task_id=75, error_class=EVAL_GOTO_EC,
                             resolved_by_commit=BAD, episode_summary_path=ep,
                             resolved_by="test", rationale="x")


def test_rejected_task_id_mismatch(reg, tmp_path):
    ep = _episode(tmp_path, "g.json", task_id=4)
    with pytest.raises(ValueError, match="task_id"):
        qr.append_resolution(site="classifieds", task_id=75, error_class=EVAL_GOTO_EC,
                             resolved_by_commit=FLOOR, episode_summary_path=ep,
                             resolved_by="test", rationale="x")


def test_unknown_error_class_rejected(reg, tmp_path):
    ep = _episode(tmp_path, "g.json")
    with pytest.raises(ValueError, match="profile"):
        qr.append_resolution(site="classifieds", task_id=75, error_class="SomeOtherError",
                             resolved_by_commit=FLOOR, episode_summary_path=ep,
                             resolved_by="test", rationale="x")


# ── per-error-class: the KEY property ────────────────────────────────────────
def test_eval_goto_resolution_does_not_clear_screenshot(reg, tmp_path):
    _quarantine_two_classes(reg)
    ep = _episode(tmp_path, "eg.json")
    qr.append_resolution(site="classifieds", task_id=75, error_class=EVAL_GOTO_EC,
                         resolved_by_commit=FLOOR, episode_summary_path=ep,
                         resolved_by="test", rationale="eval-goto only")
    # eval-goto resolved, screenshot NOT → recurrence NOT resolved
    resolved, unresolved = qr.is_recurrence_resolved("classifieds", 75)
    assert resolved is False
    assert any("screenshot" in u.lower() for u in unresolved)


def test_both_classes_resolved_clears_recurrence(reg, tmp_path):
    _quarantine_two_classes(reg)
    qr.append_resolution(site="classifieds", task_id=75, error_class=EVAL_GOTO_EC,
                         resolved_by_commit=FLOOR, episode_summary_path=_episode(tmp_path, "eg.json"),
                         resolved_by="test", rationale="eval-goto")
    qr.append_resolution(site="classifieds", task_id=75, error_class=SCREENSHOT_EC,
                         resolved_by_commit=C1B,
                         episode_summary_path=_episode(tmp_path, "ss.json", producer=C1B),
                         resolved_by="test", rationale="screenshot")
    resolved, unresolved = qr.is_recurrence_resolved("classifieds", 75)
    assert resolved is True, unresolved


# ── preflight Rule 2 integration ─────────────────────────────────────────────
def test_preflight_halts_until_both_resolved(reg, tmp_path):
    _quarantine_two_classes(reg)
    # neither resolved → HALT
    halt, _ = qr.preflight_check("classifieds", [75])
    assert halt is True
    # only eval-goto resolved → still HALT (screenshot unresolved)
    qr.append_resolution(site="classifieds", task_id=75, error_class=EVAL_GOTO_EC,
                         resolved_by_commit=FLOOR, episode_summary_path=_episode(tmp_path, "eg.json"),
                         resolved_by="test", rationale="eg")
    halt, _ = qr.preflight_check("classifieds", [75])
    assert halt is True
    # both resolved → CLEARS
    qr.append_resolution(site="classifieds", task_id=75, error_class=SCREENSHOT_EC,
                         resolved_by_commit=C1B,
                         episode_summary_path=_episode(tmp_path, "ss.json", producer=C1B),
                         resolved_by="test", rationale="ss")
    halt, blocking = qr.preflight_check("classifieds", [75])
    assert halt is False, blocking


# ── re-verify defeats a hand-edited registry ─────────────────────────────────
def test_resolution_fails_closed_if_episode_deleted(reg, tmp_path):
    _quarantine_two_classes(reg)
    import os
    ep = _episode(tmp_path, "eg.json")
    qr.append_resolution(site="classifieds", task_id=75, error_class=EVAL_GOTO_EC,
                         resolved_by_commit=FLOOR, episode_summary_path=ep,
                         resolved_by="test", rationale="eg")
    assert qr.is_error_class_resolved("classifieds", 75, EVAL_GOTO_EC)[0] is True
    os.remove(ep)  # episode gone at "preflight" time
    ok, reason = qr.is_error_class_resolved("classifieds", 75, EVAL_GOTO_EC)
    assert ok is False
    assert "missing" in reason
