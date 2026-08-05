"""B-1957 — `benchmark_permanent`, the only classification that may downgrade
the paper-grade quarantine abort.

Motivating case: VWA shopping task 345's start_url references a Wikipedia image
absent from the deployed 2025-08 ZIM (404 on every path variant while the
article itself is 302). It can never produce the clean evidence episode
`append_resolution` demands, so pre-fix it aborted the condition every time —
the 2026-08-04 fire burned 320 episodes and left the host idle 20h.

The tests below pin the guards that keep this from becoming a blanket skip:
  1. the adjudication is per-error-class and the error_class is mandatory
  2. a non-matching error_class does NOT get waved through
  3. every OTHER classification (transient_drift &c.) still aborts
  4. the adjudication is revocable by appending a newer classification
  5. `count_unclassified` stops re-arming G8 once per condition for a defect
     already adjudicated — while keeping strict per-event accounting elsewhere
  6. the runner-side probe is fail-closed on every failure mode
"""
from __future__ import annotations

import pytest

from scripts.maintenance import quarantine_registry as qr

ERR = "error(start_url_content_error)"
OTHER_ERR = "error(timeout)"


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    tmp_registry = tmp_path / "quarantine_registry.jsonl"
    monkeypatch.setattr(qr, "REGISTRY_PATH", tmp_registry)
    yield tmp_registry


def _permanent(site="shopping", task_id=345, error_class=ERR):
    return qr.append_classification(
        site=site,
        task_id=task_id,
        classification="benchmark_permanent",
        classified_by="operator",
        rationale="upstream ZIM content drift; 404 on every path variant",
        classified_via="substrate_probe_2026-08-05",
        error_class=error_class,
    )


class TestErrorClassIsMandatory:
    def test_permanent_without_error_class_is_refused(self, isolated_registry):
        with pytest.raises(ValueError, match="REQUIRES --error-class"):
            qr.append_classification(
                site="shopping",
                task_id=345,
                classification="benchmark_permanent",
                classified_by="operator",
                rationale="no error class given",
            )

    def test_other_classifications_still_allow_omitting_it(self, isolated_registry):
        """Back-compat: the 24 pre-existing classification events carry no
        error_class, and re-classifying a transient must not suddenly require one."""
        ev = qr.append_classification(
            site="reddit",
            task_id=149,
            classification="transient_drift",
            classified_by="operator",
            rationale="proxy 503",
        )
        assert ev["classification"] == "transient_drift"
        assert ev["error_class"] is None


class TestGateScoping:
    def test_matching_error_class_allows_downgrade(self, isolated_registry):
        _permanent()
        ok, why = qr.is_permanently_classified("shopping", 345, ERR)
        assert ok is True
        assert "benchmark_permanent" in why

    def test_different_error_class_on_same_task_is_refused(self, isolated_registry):
        """The load-bearing guard: adjudicating the 404 must not wave through an
        unrelated OOM/timeout quarantine on the same task."""
        _permanent()
        ok, why = qr.is_permanently_classified("shopping", 345, OTHER_ERR)
        assert ok is False
        assert OTHER_ERR in why

    def test_different_task_is_refused(self, isolated_registry):
        _permanent()
        ok, _ = qr.is_permanently_classified("shopping", 346, ERR)
        assert ok is False

    def test_blank_error_class_is_refused(self, isolated_registry):
        _permanent()
        ok, why = qr.is_permanently_classified("shopping", 345, "")
        assert ok is False
        assert "no error_class" in why

    def test_non_permanent_classification_never_downgrades(self, isolated_registry):
        """transient_drift means 'might come back clean' — its correct handling is
        resume-rerun, NOT walking past the gate."""
        qr.append_classification(
            site="shopping", task_id=345, classification="transient_drift",
            classified_by="operator", rationale="looked transient",
            error_class=ERR,
        )
        ok, why = qr.is_permanently_classified("shopping", 345, ERR)
        assert ok is False
        assert "transient_drift" in why

    def test_adjudication_is_revocable_by_a_newer_event(self, isolated_registry):
        _permanent()
        assert qr.is_permanently_classified("shopping", 345, ERR)[0] is True
        qr.append_classification(
            site="shopping", task_id=345, classification="undecided",
            classified_by="operator", rationale="reopening — ZIM may be re-imported",
            error_class=ERR,
        )
        ok, _ = qr.is_permanently_classified("shopping", 345, ERR)
        assert ok is False, "most recent classification must win"


class TestUnclassifiedAccounting:
    def _quarantine(self, error_class=ERR, task_id=345):
        return qr.append_quarantine(
            site="shopping", task_id=task_id, run_id="B0_dom_shopping_test",
            url=None, error_class=error_class, error_message="404",
            callsite=None,
        )

    def test_permanent_stops_g8_rearming_every_condition(self, isolated_registry):
        """Pre-fix arithmetic (#quarantine - #classification) would read 6 after
        7 conditions each hit the same adjudicated defect, halting the NEXT fire
        on a question already answered."""
        _permanent()
        for _ in range(7):
            self._quarantine()
        assert qr.count_unclassified("shopping", 345) == 0

    def test_unadjudicated_error_class_still_counts(self, isolated_registry):
        """The exemption is scoped: a DIFFERENT failure on the same task must
        still demand its own investigation."""
        _permanent()
        for _ in range(3):
            self._quarantine()
        self._quarantine(error_class=OTHER_ERR)
        assert qr.count_unclassified("shopping", 345) == 1

    def test_strict_accounting_survives_for_other_tasks(self, isolated_registry):
        self._quarantine(task_id=77)
        self._quarantine(task_id=77)
        assert qr.count_unclassified("shopping", 77) == 2

    def test_preflight_clears_once_adjudicated(self, isolated_registry):
        _permanent()
        for _ in range(4):
            self._quarantine()
        halt, blocking = qr.preflight_check("shopping", [344, 345, 346])
        assert halt is False, f"expected clear, blocked by {blocking}"


class TestRunnerProbeFailsClosed:
    """The runner must keep aborting whenever the probe cannot give a clean YES."""

    def _runner(self):
        from p79.experiment.runner.main import ExperimentRunner
        return ExperimentRunner.__new__(ExperimentRunner)

    def test_no_adjudication_means_abort(self):
        ok, _ = self._runner()._quarantine_downgrade_allowed(
            "shopping", 345, "error(definitely_not_adjudicated_xyz)"
        )
        assert ok is False

    def test_missing_registry_script_means_abort(self, monkeypatch):
        import p79.experiment.runner.main as rm
        real_path = rm.Path

        class _Missing(type(real_path())):
            def exists(self):  # noqa: D102
                return False

        monkeypatch.setattr(
            rm, "Path", lambda *a, **k: _Missing(real_path(*a, **k))
        )
        ok, why = self._runner()._quarantine_downgrade_allowed("shopping", 345, ERR)
        assert ok is False
        assert "not found" in why

    def test_probe_exception_means_abort(self, monkeypatch):
        import subprocess

        def _boom(*a, **k):
            raise OSError("simulated fork failure")

        monkeypatch.setattr(subprocess, "run", _boom)
        ok, why = self._runner()._quarantine_downgrade_allowed("shopping", 345, ERR)
        assert ok is False
        assert "raised" in why
