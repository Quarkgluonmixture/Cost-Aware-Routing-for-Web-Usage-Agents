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

from pathlib import Path

import pytest

from scripts.maintenance import quarantine_registry as qr

REPO_ROOT = Path(__file__).resolve().parents[1]

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


class TestB1961ConditionCanActuallyFinalize:
    """The downgrade must let the condition REACH THE END, not just past task 345.

    codex Mode B P0-1 (2026-08-06): the original fix left `needs_reevaluation=True`
    on the downgraded episode. `runner/main.py` calls
    `aggregate_condition_metrics(episode_summaries)` WITHOUT `allow_quarantined`
    for any non-aborted condition, and that aggregator raises unconditionally on
    such an episode (`metrics.py`, B-784). Its own comment says the raise is
    "UNREACHABLE under paper_grade=True; if it fires here, M1 gate was bypassed"
    — and the B-1957 downgrade IS that bypass. Net effect pre-fix: the run would
    burn the remaining 115 episodes and then die at the finish line, unable to
    write a condition summary.

    Nothing in the original 15 tests exercised downgrade → finalize → load, which
    is exactly why the landmine survived a green suite.
    """

    @staticmethod
    def _episode(**over):
        """Build via the real dataclass so every canonical field is populated.

        A hand-rolled dict trips the aggregator's separate "no episode populates
        this field" guard, which would mask whether the B-784 quarantine check
        (the thing under test) passed or not.
        """
        import inspect
        from dataclasses import asdict
        from p79.experiment.types import EpisodeSummaryV2

        sig = inspect.signature(EpisodeSummaryV2)
        filled = {}
        for name, prm in sig.parameters.items():
            if prm.default is not inspect.Parameter.empty:
                continue                     # dataclass supplies it
            ann = str(prm.annotation)
            if "int" in ann and "float" not in ann:
                filled[name] = 0
            elif "float" in ann:
                filled[name] = 0.0
            elif "bool" in ann:
                filled[name] = False
            elif "Dict" in ann or "dict" in ann:
                filled[name] = {}
            elif "List" in ann or "list" in ann:
                filled[name] = []
            else:
                filled[name] = ""
        filled.update({k: v for k, v in over.items() if k in sig.parameters})
        ep = asdict(EpisodeSummaryV2(**filled))
        ep.update({k: v for k, v in over.items() if k not in sig.parameters})
        return ep

    def _downgraded_episode(self):
        """An episode shaped like one B-1961 just downgraded."""
        return self._episode(
            task_id=345, success=False, steps=0,
            error="start_url_content_error: tab title='Content not found'",
            benchmark_noise=True,
            benchmark_noise_category="start_url_content_error",
            needs_reevaluation=False,             # ← cleared by B-1961
            benchmark_permanent_adjudicated=True,
            # canonical latency field: the aggregator refuses a cohort where NO
            # episode populates it (unrelated to what this test asserts)
            total_latency_minus_retry_ms=0.0,
        )

    def _clean_episode(self, task_id):
        return self._episode(task_id=task_id, success=False, steps=12,
                             needs_reevaluation=False,
                             total_latency_minus_retry_ms=1000.0)

    def test_strict_aggregator_accepts_the_downgraded_episode(self):
        """The load-bearing one: strict aggregation (no allow_quarantined) must
        not raise, or the condition cannot finalize."""
        from p79.experiment.metrics import aggregate_condition_metrics
        eps = [self._clean_episode(1), self._downgraded_episode(), self._clean_episode(2)]
        agg = aggregate_condition_metrics(eps)      # strict, as the runner calls it
        assert agg["episodes"] == 3, "the adjudicated episode must still be counted"

    def test_strict_aggregator_still_rejects_an_unadjudicated_quarantine(self):
        """The B-784 invariant must survive: only ADJUDICATED episodes get through."""
        from p79.experiment.metrics import aggregate_condition_metrics
        ep = self._downgraded_episode()
        ep["needs_reevaluation"] = True            # un-adjudicated quarantine
        ep["benchmark_permanent_adjudicated"] = False
        with pytest.raises(ValueError, match="needs_reevaluation"):
            aggregate_condition_metrics([self._clean_episode(1), ep])

    def test_schema_registers_the_provenance_field(self):
        """An unregistered field would be dropped/rejected on write."""
        from p79.experiment import types as t
        assert hasattr(t.EpisodeSummaryV2, "__dataclass_fields__")
        assert "benchmark_permanent_adjudicated" in t.EpisodeSummaryV2.__dataclass_fields__
        src = (REPO_ROOT / "p79/experiment/types.py").read_text(encoding="utf-8")
        assert '"benchmark_permanent_adjudicated": (bool,),' in src, "type catalog entry missing"
        assert '"benchmark_permanent_adjudicated",' in src, "field-name list entry missing"

    def test_runner_clears_flag_and_stamps_provenance(self):
        """Both mutations must happen together — clearing the flag without the
        marker would make an adjudicated defect indistinguishable from a clean
        agent failure in every downstream analysis."""
        src = (REPO_ROOT / "p79/experiment/runner/main.py").read_text(encoding="utf-8")
        i_clear = src.index('summary["needs_reevaluation"] = False')
        i_stamp = src.index('summary["benchmark_permanent_adjudicated"] = True')
        assert abs(i_stamp - i_clear) < 400, "the two mutations must sit together"
        # and the episode must be re-persisted, else disk disagrees with memory
        tail = src[i_stamp:i_stamp + 2000]
        assert "write_episode_summary" in tail, "downgraded episode must be re-written to disk"


class TestB1962ResumeRefreshesAuth:
    """RESET_BEFORE=0 must still establish the logged-in precondition.

    codex Mode B P0-2 (2026-08-06): `reset_and_auth_gate` was the only caller of
    the auth gate and ran only under RESET_BEFORE=1, so a B-304-compliant resume
    launched with no auth refresh at all. Empirically the 2026-08-05 shopping
    resume spent its first 22 minutes on dead cookies (tasks 346/347/348 all hit
    max_steps with the agent on /customer/account/login/).
    """

    QUEUES = ["queue_baseline.sh", "queue_phantom_som.sh",
              "queue_phantom_text.sh", "queue_phantom_prompt.sh"]

    def test_lib_exposes_auth_only_gate(self):
        lib = (REPO_ROOT / "scripts/queues/_lib_paper_grade_gates.sh").read_text(encoding="utf-8")
        assert "auth_only_gate() {" in lib
        # must NOT reset anything — that is the whole point
        body = lib[lib.index("auth_only_gate() {"):lib.index("reset_and_auth_gate() {")]
        for forbidden in ("docker rm", "reset_vwa", "page=reset", "indexer:reindex"):
            assert forbidden not in body, f"auth_only_gate must not touch site state ({forbidden})"
        assert "auth_required_gate" in body, "must go through the B-224 gate"

    @pytest.mark.parametrize("script", QUEUES)
    def test_every_queue_refreshes_auth_on_resume(self, script):
        """Sibling propagation: the defect lived in a shared shape, so the fix
        must too. queue_phantom_dom.sh is a symlink to queue_phantom_text.sh."""
        src = (REPO_ROOT / "scripts/queues" / script).read_text(encoding="utf-8")
        assert "auth_only_gate" in src, f"{script} still launches a resume without auth"

    @pytest.mark.parametrize("script", QUEUES)
    def test_auth_failure_aborts_the_resume(self, script):
        """B-224 contract: never proceed NOT-LOGGED-IN under paper-grade."""
        src = (REPO_ROOT / "scripts/queues" / script).read_text(encoding="utf-8")
        i = src.index("auth_only_gate")
        assert "|| exit 1" in src[i:i + 400], f"{script}: auth_only_gate failure must abort"
