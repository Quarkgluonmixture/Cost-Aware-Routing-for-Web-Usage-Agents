"""B-1881 (reddit chain abort #3/#4, 2026-06-20): PRE-FLIGHT transient-substrate
episode-retry wrapper. Fire-4 RCA Wave 1 M1 made the FIRST quarantine event
fail-closed abort the whole condition — correct for non-transient compromise,
but TRANSIENT infra blips at the episode boundary (ZERO contamination) were
converting recoverable hiccups into total-condition loss (R28130 proxy-503 lost
55 ep; R26851 auth lost 137 ep).

3-AI /stress consensus (2026-06-20) narrowed the retry to PRE-FLIGHT (steps==0)
auth/network failures only — the master safety gate that makes the change
estimand-neutral (no site mutation, no stochastic-rollout redraw, not
agent-induced). These freeze that contract:
  1. pre-flight auth (steps==0) → bounded episode-retry → clean success.
  2. pre-flight network (steps==0) → retry.
  3. proxy_5xx → NEVER episode-retried (owned by B-1880 internal retry) → abort.
  4. mid-episode (steps>0) ANY class → NOT retryable (mutation/redraw risk) → abort.
  5. retry exhaustion → abort; non-transient → abort; diagnostic/dev → no retry.
  6. clean success after retry stamps transient_retry_count on the summary.
  7. failed canonical summary deleted before backoff (watchdog stale-ingest race).
  8. classifier conservatism.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from p79.experiment.environment import PaperGradeAbortError
from p79.experiment.runner.main import ExperimentRunner


_AUTH_MSG = (
    "first quarantine event ... (error=\"auth_required_gate('reddit') FAILED "
    "after 2 attempts: refresh_site_auth returned False\")"
)
_PROXY_MSG = "first quarantine ... (error='503 Server Error ... execute-api ... model-api/invoke')"
_NET_MSG = "first quarantine ... (error='HTTPConnectionPool: Max retries exceeded ... Connection reset')"
_TIMEOUT_MSG = "first quarantine ... (error='Page.screenshot: Timeout 30000ms exceeded')"


def _abort(transient_class, steps, msg="first quarantine ..."):
    return PaperGradeAbortError(msg, transient_class=transient_class, steps=steps)


def _make_runner(paper_grade=True, diagnostic_replay=False, max_retries=3):
    r = ExperimentRunner.__new__(ExperimentRunner)
    r.cfg = {"paper_grade": paper_grade, "transient_episode_max_retries": max_retries}
    r.diagnostic_replay = diagnostic_replay
    r._auth_last_refresh_ts = {}
    return r


def _task():
    return SimpleNamespace(site="reddit", task_id=140)


@pytest.fixture(autouse=True)
def _no_sleep_no_ntfy(monkeypatch):
    monkeypatch.setattr("p79.experiment.runner.main.time.sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "p79.experiment.runner.main._notify_transient_retry", lambda *_a, **_k: None
    )


def _invoke(runner, logger=None):
    logger = logger or MagicMock()
    return runner._run_and_record_episode(
        condition=MagicMock(), task=_task(), backend=MagicMock(),
        condition_logger=logger, condition_dir="/tmp/cd",
        effective_cid="phase1_dom_router_0", current_seed=42,
    ), logger


# --- 8. classifier conservatism (string → class) ---
@pytest.mark.parametrize("msg,expected", [
    (_AUTH_MSG, "auth"),
    (_PROXY_MSG, "proxy_5xx"),
    (_NET_MSG, "network"),
    (_TIMEOUT_MSG, None),
    ("error='403 Client Error ... model-api quota'", None),
    ("error='benchmark noise: api_rate_limit'", None),
])
def test_classifier(msg, expected):
    assert ExperimentRunner._classify_transient_substrate(msg) == expected


# --- 1. pre-flight auth (steps==0) → retry → success ---
def test_preflight_auth_retries_then_succeeds():
    runner = _make_runner()
    clean = {"success": True, "steps": 12}
    runner._run_and_record_episode_once = MagicMock(side_effect=[
        _abort("auth", 0), _abort("auth", 0), clean,
    ])
    (summary, logger) = _invoke(runner)
    assert summary is clean
    assert runner._run_and_record_episode_once.call_count == 3
    # lineage stamped + re-written to canonical
    assert summary["transient_retry_count"] == 2
    assert summary["transient_retry_classes"] == ["auth", "auth"]
    assert summary["is_retry_attempt"] is True
    assert logger.write_episode_summary.called
    # auth recovery forced re-auth (ts reset)
    assert runner._auth_last_refresh_ts["reddit"] == 0.0
    # 7. failed summary deleted before backoff (watchdog race) — 2 retries → 2 unlinks
    assert logger.summary_path.return_value.unlink.call_count == 2


# --- 2. pre-flight network (steps==0) → retry → success ---
def test_preflight_network_retries_then_succeeds():
    runner = _make_runner()
    clean = {"success": False, "steps": 30}
    runner._run_and_record_episode_once = MagicMock(side_effect=[_abort("network", 0), clean])
    (summary, _l) = _invoke(runner)
    assert summary is clean
    assert runner._run_and_record_episode_once.call_count == 2


# --- 3. proxy_5xx NEVER episode-retried (B-1880 owns it) → abort even at steps==0 ---
def test_proxy_5xx_not_retried_aborts():
    runner = _make_runner()
    runner._run_and_record_episode_once = MagicMock(side_effect=_abort("proxy_5xx", 0))
    with pytest.raises(PaperGradeAbortError):
        _invoke(runner)
    assert runner._run_and_record_episode_once.call_count == 1   # no retry


# --- 4. mid-episode (steps>0) NOT retryable for ANY class → abort ---
@pytest.mark.parametrize("tclass", ["auth", "network"])
def test_mid_episode_steps_gt0_not_retried(tclass):
    runner = _make_runner()
    runner._run_and_record_episode_once = MagicMock(side_effect=_abort(tclass, 7))
    with pytest.raises(PaperGradeAbortError):
        _invoke(runner)
    assert runner._run_and_record_episode_once.call_count == 1   # steps>0 → no retry


# --- 5a. retry exhaustion → abort ---
def test_exhaustion_reraises():
    runner = _make_runner(max_retries=3)
    runner._run_and_record_episode_once = MagicMock(side_effect=_abort("auth", 0))
    with pytest.raises(PaperGradeAbortError):
        _invoke(runner)
    assert runner._run_and_record_episode_once.call_count == 4   # 1 + 3 retries


# --- 5b. non-transient (class None) → immediate abort ---
def test_non_transient_reraises_immediately():
    runner = _make_runner()
    runner._run_and_record_episode_once = MagicMock(side_effect=_abort(None, 0))
    with pytest.raises(PaperGradeAbortError):
        _invoke(runner)
    assert runner._run_and_record_episode_once.call_count == 1


# --- 5c. diagnostic_replay / non-paper-grade / max=0 → no retry ---
@pytest.mark.parametrize("kw", [
    {"diagnostic_replay": True},
    {"paper_grade": False},
    {"max_retries": 0},
])
def test_bypass_no_retry(kw):
    runner = _make_runner(**kw)
    runner._run_and_record_episode_once = MagicMock(side_effect=_abort("auth", 0))
    with pytest.raises(PaperGradeAbortError):
        _invoke(runner)
    assert runner._run_and_record_episode_once.call_count == 1


# --- 6. clean first attempt → no retry machinery, no lineage stamp ---
def test_clean_first_attempt_no_retry():
    runner = _make_runner()
    clean = {"success": True, "steps": 5}
    runner._run_and_record_episode_once = MagicMock(return_value=clean)
    (summary, logger) = _invoke(runner)
    assert summary is clean
    assert "transient_retry_count" not in summary
    assert runner._run_and_record_episode_once.call_count == 1
    logger.log_trajectory_event.assert_not_called()
    logger.summary_path.return_value.unlink.assert_not_called()


# --- back-compat: a legacy PaperGradeAbortError with NO provenance → no retry (abort) ---
def test_legacy_abort_without_provenance_aborts():
    runner = _make_runner()
    runner._run_and_record_episode_once = MagicMock(
        side_effect=PaperGradeAbortError("legacy message no kwargs")
    )
    with pytest.raises(PaperGradeAbortError):
        _invoke(runner)
    assert runner._run_and_record_episode_once.call_count == 1
