"""Guards for the run-abort notification path (2026-08-09).

Context: the three fail-fast branches in `_run_and_record_episode` stop the run
on purpose — every subsequent task would fail identically — but until now none
of them notified anyone. The shop B0 vision run stopped on proxy quota at task
405 and sat dead six hours on a booked machine. These tests pin the two
properties that made the silence possible, so it cannot come back:

  1. every abort class reaches ntfy with urgent priority and a recovery hint
  2. a failing push NEVER raises — the caller is on its way to `raise`, and an
     exception from the notifier would replace a precise "quota exhausted"
     traceback with a misleading urllib one
"""
from __future__ import annotations

import urllib.request

import pytest

from p79.experiment.runner.helpers import _ABORT_RECOVERY_HINTS, push_run_abort_ntfy

ABORT_CLASSES = ["proxy_quota", "fatal_env", "evaluator_unavailable"]


class _Captured:
    def __init__(self):
        self.requests = []

    def __call__(self, req, timeout=None):  # mimics urlopen(req, timeout=...)
        self.requests.append(req)

        class _Ctx:
            def __enter__(self_inner):
                return None

            def __exit__(self_inner, *a):
                return False

        return _Ctx()


@pytest.fixture()
def captured(monkeypatch):
    cap = _Captured()
    monkeypatch.setattr(urllib.request, "urlopen", cap)
    monkeypatch.setenv("NTFY_TOPIC", "p79-test-topic")
    return cap


@pytest.mark.parametrize("abort_class", ABORT_CLASSES)
def test_each_abort_class_pushes_urgent_with_hint(captured, abort_class):
    push_run_abort_ntfy(
        "phase1_vision_router_0", "shopping", 405, abort_class, RuntimeError("boom")
    )
    assert len(captured.requests) == 1
    req = captured.requests[0]
    assert req.get_header("Priority") == "urgent", "run-stop must outrank routine pushes"
    assert abort_class in req.get_header("Title")
    body = req.data.decode("utf-8")
    assert "405" in body and "shopping" in body
    # The hint is the actionable half; a bare "it stopped" is what we already had.
    assert _ABORT_RECOVERY_HINTS[abort_class][:12] in body


def test_quota_hint_mentions_resume(captured):
    """The quota case is the recoverable one — the hint must say so.

    Getting this wrong is expensive in the other direction: an operator who
    believes the run is unrecoverable may wipe and re-fire 374 good episodes.
    """
    push_run_abort_ntfy("c", "shopping", 405, "proxy_quota", RuntimeError("403"))
    body = captured.requests[0].data.decode("utf-8")
    assert "resume" in body.lower()


def test_no_topic_is_a_noop(monkeypatch, captured):
    monkeypatch.setenv("NTFY_TOPIC", "")
    push_run_abort_ntfy("c", "shopping", 1, "proxy_quota", RuntimeError("x"))
    assert captured.requests == []


def test_push_failure_never_raises(monkeypatch):
    """Caller is one line away from `raise`; the notifier must not hijack it."""
    monkeypatch.setenv("NTFY_TOPIC", "p79-test-topic")

    def _boom(req, timeout=None):
        raise OSError("network down")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    push_run_abort_ntfy("c", "shopping", 405, "proxy_quota", RuntimeError("403"))


def test_unknown_abort_class_still_notifies(captured):
    """Unknown class loses the hint but must not lose the alert."""
    push_run_abort_ntfy("c", "shopping", 1, "brand_new_failure", RuntimeError("x"))
    assert len(captured.requests) == 1
    assert captured.requests[0].get_header("Priority") == "urgent"


def test_long_exception_is_truncated(captured):
    push_run_abort_ntfy("c", "shopping", 1, "proxy_quota", RuntimeError("x" * 5000))
    body = captured.requests[0].data.decode("utf-8")
    assert len(body) < 1200, "ntfy body must stay readable on a phone lock screen"


def test_all_three_runner_branches_call_the_notifier():
    """Source-level guard: the fix is three call sites, not one.

    A future refactor that keeps `raise` but drops a `push_run_abort_ntfy`
    call would silently restore the exact failure mode this fixes, and no
    behavioural test would catch it without standing up a whole runner.
    """
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "p79/experiment/runner/main.py"
    text = src.read_text()
    # 1 import + 3 call sites
    assert text.count("push_run_abort_ntfy(") == 3
    for cls in ABORT_CLASSES:
        assert f'"{cls}"' in text, f"{cls} branch lost its notifier argument"
