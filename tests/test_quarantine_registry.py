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
        """At threshold=3 (Rule 1), 2 unclassified events do NOT halt.

        /stress 2026-05-20 P0-A2-Hub regression: test must explicitly disable
        Rule 2 (cross-fire recurrence) via min_recurrent_fires=99 to isolate
        Rule 1 (unclassified threshold) semantics under test.
        """
        qr.append_quarantine(site="classifieds", task_id=75, run_id="r1")
        qr.append_quarantine(site="classifieds", task_id=75, run_id="r2")
        # 2 unclassified events, threshold 3, Rule 2 disabled → still ok
        should_halt, blocking = qr.preflight_check(
            "classifieds", [75], halt_threshold=3, min_recurrent_fires=99,
        )
        assert should_halt is False
        # 2 unclassified events, threshold 2, Rule 2 disabled → halt by Rule 1
        should_halt, _ = qr.preflight_check(
            "classifieds", [75], halt_threshold=2, min_recurrent_fires=99,
        )
        assert should_halt is True

    def test_cross_fire_recurrence_rule(self, isolated_registry):
        """/stress 2026-05-20 P0-A2-Hub: cross-fire recurrence (Rule 2)
        halts independent of classification status — different from Rule 1.
        """
        # 2 quarantine events across 2 distinct run_ids
        qr.append_quarantine(site="classifieds", task_id=75, run_id="r1")
        qr.append_quarantine(site="classifieds", task_id=75, run_id="r2")
        # Classify BOTH → unclassified_count=0; Rule 1 would pass
        qr.append_classification(
            site="classifieds", task_id=75, classification="transient_drift",
            classified_by="op", rationale="r1 isolated reproduce",
        )
        qr.append_classification(
            site="classifieds", task_id=75, classification="transient_drift",
            classified_by="op", rationale="r2 isolated reproduce",
        )
        # Rule 1 alone (high threshold, Rule 2 disabled) → passes
        should_halt_r1, _ = qr.preflight_check(
            "classifieds", [75], halt_threshold=99, min_recurrent_fires=99,
        )
        assert should_halt_r1 is False
        # Rule 2 with default min_recurrent_fires=2 → halts despite classified
        should_halt_r2, blocking_r2 = qr.preflight_check(
            "classifieds", [75], halt_threshold=99, min_recurrent_fires=2,
        )
        assert should_halt_r2 is True
        assert len(blocking_r2) == 1
        assert blocking_r2[0]["task_id"] == 75
        assert blocking_r2[0]["rule"] == "cross_fire_recurrence"
        assert blocking_r2[0]["recurrent_fires"] == 2

    def test_detect_recurrent_failures(self, isolated_registry):
        """/stress 2026-05-20 P0-A2-Hub: detect_recurrent_failures helper
        returns tasks with quarantine across ≥ min_fires distinct run_ids.
        """
        # task 1: only 1 fire → not recurrent
        qr.append_quarantine(site="classifieds", task_id=1, run_id="r1")
        # task 5: 2 fires same run_id (deduped) → still 1 distinct fire
        qr.append_quarantine(site="classifieds", task_id=5, run_id="r2")
        qr.append_quarantine(site="classifieds", task_id=5, run_id="r2")
        # task 75: 3 fires across 3 distinct run_ids → recurrent
        qr.append_quarantine(site="classifieds", task_id=75, run_id="r3")
        qr.append_quarantine(site="classifieds", task_id=75, run_id="r4")
        qr.append_quarantine(site="classifieds", task_id=75, run_id="r5")

        recurrent = qr.detect_recurrent_failures("classifieds", min_fires=2)
        recurrent_tids = {r["task_id"] for r in recurrent}
        assert 1 not in recurrent_tids
        assert 5 not in recurrent_tids  # same run_id deduplicated
        assert 75 in recurrent_tids

        rec_75 = next(r for r in recurrent if r["task_id"] == 75)
        assert rec_75["fire_count"] == 3
        assert sorted(rec_75["run_ids"]) == ["r3", "r4", "r5"]


class TestRecurrenceDedupInvariant:
    """Z-session decision A invariant (2026-05-20): recurrence MUST count unique
    fire events (run_ids), NOT raw registry rows. Confirmatory classification
    entries (e.g. Z empirical retier + parallel-session revised_tier on the
    same task) must NOT inflate the cross-fire recurrence count.
    """

    def test_classification_rows_do_not_inflate_recurrence(self, isolated_registry):
        # 2 distinct fires quarantine task 75
        qr.append_quarantine(site="classifieds", task_id=75, run_id="fire3")
        qr.append_quarantine(site="classifieds", task_id=75, run_id="fire4")
        # Pile on 6 classification rows (mirror Z retier + parallel revised_tier)
        for via in ("transient_drift", "transient_drift"):
            qr.append_classification(site="classifieds", task_id=75,
                                     classification=via, classified_by="a", rationale="x")
        for _ in range(4):
            qr.append_classification(site="classifieds", task_id=75,
                                     classification="unreproducible_in_isolation",
                                     classified_by="b", rationale="confirmatory")
        rec = qr.detect_recurrent_failures("classifieds", min_fires=2)
        rec_75 = next(r for r in rec if r["task_id"] == 75)
        # 8 total rows (2 quarantine + 6 classification) but fire_count MUST be 2
        assert rec_75["fire_count"] == 2, "classification rows must NOT inflate recurrence"
        assert sorted(rec_75["run_ids"]) == ["fire3", "fire4"]

    def test_duplicate_fire_rows_deduped(self, isolated_registry):
        # Same fire writes 3 quarantine rows for same task (e.g. retry within fire)
        for _ in range(3):
            qr.append_quarantine(site="classifieds", task_id=75, run_id="fire3")
        qr.append_quarantine(site="classifieds", task_id=75, run_id="fire4")
        rec = qr.detect_recurrent_failures("classifieds", min_fires=2)
        rec_75 = next(r for r in rec if r["task_id"] == 75)
        # 4 raw quarantine rows but only 2 unique fires
        assert rec_75["fire_count"] == 2, "same-fire duplicate rows must dedup by run_id"


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
