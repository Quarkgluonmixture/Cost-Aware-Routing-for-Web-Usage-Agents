"""Fire-4 RCA Wave 2 M6 test — quarantine registry investigation gate.

User decision 2026-05-19: cross-fire memory triggers INVESTIGATION (halt),
NOT auto-skip. Tests verify:
  1. append_quarantine creates correct event shape
  2. append_classification validates classification value
  3. count_unclassified accounts for both quarantine + classification events
  4. preflight_check returns (True, blocking_list) when threshold exceeded
  5. preflight_check returns (False, []) for clean tasks
  6. Most-recent classification timestamp determines latest state
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# Import test target with monkey-patched registry path so tests don't write
# to the canonical docs/checkpoints/quarantine_registry.jsonl.
from scripts.maintenance import quarantine_registry as qr


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    """Redirect REGISTRY_PATH to a tmp file for each test."""
    tmp_registry = tmp_path / "quarantine_registry.jsonl"
    monkeypatch.setattr(qr, "REGISTRY_PATH", tmp_registry)
    yield tmp_registry


class TestAppendQuarantine:
    def test_creates_correct_shape(self, isolated_registry):
        ev = qr.append_quarantine(
            site="classifieds",
            task_id=75,
            run_id="B0_dom_test_run",
            url="http://localhost:9980/index.php?page=item&id=84148",
            error_class="Page.screenshot Timeout",
            error_message="Page.screenshot: Timeout 30000ms exceeded.",
            callsite="agent_observation",
        )
        assert ev["event_type"] == "quarantine"
        assert ev["site"] == "classifieds"
        assert ev["task_id"] == 75
        assert ev["run_id"] == "B0_dom_test_run"
        assert ev["callsite"] == "agent_observation"
        assert ev["needs_reevaluation"] is True
        assert "ts" in ev
        # Verify written to disk
        events = qr._read_events()
        assert len(events) == 1
        assert events[0]["task_id"] == 75

    def test_error_message_capped(self, isolated_registry):
        """Long error messages capped at 500 chars to keep registry compact."""
        long_msg = "X" * 1000
        ev = qr.append_quarantine(
            site="classifieds", task_id=1, run_id="r", error_message=long_msg,
        )
        assert len(ev["error_message"]) == 500


class TestAppendClassification:
    def test_valid_classifications_accepted(self, isolated_registry):
        for cls in ("substrate", "agent_induced", "evaluator", "transient_drift", "undecided"):
            qr.append_classification(
                site="classifieds", task_id=1, classification=cls,
                classified_by="test", rationale="test",
            )
        events = qr._read_events()
        assert len(events) == 5

    def test_invalid_classification_rejected(self, isolated_registry):
        with pytest.raises(ValueError, match="invalid classification"):
            qr.append_classification(
                site="classifieds", task_id=1, classification="garbage",
                classified_by="test", rationale="test",
            )


class TestCountUnclassified:
    def test_zero_when_no_events(self, isolated_registry):
        assert qr.count_unclassified("classifieds", 75) == 0

    def test_counts_quarantine_minus_classifications(self, isolated_registry):
        # 2 quarantine events (Fire-3 + Fire-4 scenario)
        qr.append_quarantine(site="classifieds", task_id=75, run_id="fire-3")
        qr.append_quarantine(site="classifieds", task_id=75, run_id="fire-4")
        assert qr.count_unclassified("classifieds", 75) == 2

        # Classify one of them → unclassified count goes down to 1
        qr.append_classification(
            site="classifieds", task_id=75, classification="substrate",
            classified_by="op", rationale="docker drift",
        )
        assert qr.count_unclassified("classifieds", 75) == 1

        # Classify the other → 0
        qr.append_classification(
            site="classifieds", task_id=75, classification="agent_induced",
            classified_by="op", rationale="bad sequence",
        )
        assert qr.count_unclassified("classifieds", 75) == 0

    def test_other_tasks_isolated(self, isolated_registry):
        qr.append_quarantine(site="classifieds", task_id=75, run_id="r1")
        qr.append_quarantine(site="classifieds", task_id=99, run_id="r2")
        assert qr.count_unclassified("classifieds", 75) == 1
        assert qr.count_unclassified("classifieds", 99) == 1
        assert qr.count_unclassified("reddit", 75) == 0


class TestPreflightCheck:
    def test_clean_passes(self, isolated_registry):
        should_halt, blocking = qr.preflight_check("classifieds", [0, 1, 2, 75])
        assert should_halt is False
        assert blocking == []

    def test_unclassified_halts(self, isolated_registry):
        qr.append_quarantine(site="classifieds", task_id=75, run_id="r1")
        should_halt, blocking = qr.preflight_check("classifieds", list(range(0, 100)))
        assert should_halt is True
        assert len(blocking) == 1
        assert blocking[0]["task_id"] == 75
        assert blocking[0]["unclassified_count"] == 1

    def test_classified_passes(self, isolated_registry):
        qr.append_quarantine(site="classifieds", task_id=75, run_id="r1")
        qr.append_classification(
            site="classifieds", task_id=75, classification="agent_induced",
            classified_by="op", rationale="bad action",
        )
        should_halt, blocking = qr.preflight_check("classifieds", [75])
        assert should_halt is False
        assert blocking == []

    def test_higher_threshold(self, isolated_registry):
        """At threshold=3, single quarantine event does NOT halt."""
        qr.append_quarantine(site="classifieds", task_id=75, run_id="r1")
        qr.append_quarantine(site="classifieds", task_id=75, run_id="r2")
        # 2 unclassified events, threshold 3 → still ok
        should_halt, blocking = qr.preflight_check("classifieds", [75], halt_threshold=3)
        assert should_halt is False
        # 2 unclassified events, threshold 2 → halt
        should_halt, _ = qr.preflight_check("classifieds", [75], halt_threshold=2)
        assert should_halt is True


class TestLatestClassification:
    def test_none_when_no_classifications(self, isolated_registry):
        assert qr.latest_classification("classifieds", 75) is None

    def test_returns_most_recent(self, isolated_registry):
        qr.append_classification(
            site="classifieds", task_id=75, classification="undecided",
            classified_by="op1", rationale="initial guess",
        )
        qr.append_classification(
            site="classifieds", task_id=75, classification="agent_induced",
            classified_by="op2", rationale="reproduced and confirmed",
        )
        latest = qr.latest_classification("classifieds", 75)
        assert latest is not None
        assert latest["classification"] == "agent_induced"
        assert latest["classified_by"] == "op2"
