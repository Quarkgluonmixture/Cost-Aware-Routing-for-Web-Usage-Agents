"""B-1836 regression — evaluator-phase retry gate must fire on Playwright timeout.

Fire-5/6 unified root cause (实验笔记 §269, master_bug_catalog B-1836): the
pre-fix `is_nav_error` keyword list used "timed out" (spaced) and so NEVER
matched Playwright's "Timeout 30000ms exceeded" wording. Result: eval
Page.goto timeout fell straight through to abort with ZERO retries (forensic
dumps across Fire-5 task4 / Fire-6 task32 all show attempt=0), and B-1803's
per-retry fresh-context was dead code for timeouts.

This test pins the fix: eval_error_is_retryable() reuses the single-source
classify_timeout() so the retry gate recognises Playwright timeout wording,
and the local exponential backoff spans the transient cls-docker window. The
GLOBAL Page.goto timeout (30s) is intentionally NOT widened (would mask
substrate degradation — user directive 2026-05-22).
"""
from __future__ import annotations

import pytest

from p79.experiment.environment import (
    eval_error_is_retryable,
    _EVAL_MAX_RETRIES,
    _EVAL_RETRY_BACKOFF_BASE_S,
    _EVAL_RETRY_BACKOFF_CAP_S,
)


class TestEvalErrorIsRetryable:
    """The retry gate that B-1836 fixed."""

    @pytest.mark.parametrize("err", [
        # THE 3-fire killer — pre-B-1836 this returned False → zero retries.
        "Page.goto: Timeout 30000ms exceeded.",
        "Page.goto: Timeout 30000ms exceeded.\nCall log:\n  - navigating to ...",
        # Other Playwright timeout callsites.
        "Page.screenshot: Timeout 30000ms exceeded.",
        "Page.click: Timeout 30000ms exceeded.",
        "locator.click: Timeout 30000ms exceeded.",
        # Legacy spaced wording (old keyword) still covered.
        "operation timed out",
        "deadline exceeded after retry",
        # Non-timeout navigation errors (preserved from old inline list).
        "net::ERR_CONNECTION_REFUSED at http://localhost:9980",
        "navigation failed",
        "Target closed",
        "Page closed unexpectedly",
    ])
    def test_retryable_errors(self, err: str) -> None:
        assert eval_error_is_retryable(err) is True, (
            f"B-1836: eval should retry on {err!r} (Playwright timeout / nav error)"
        )

    @pytest.mark.parametrize("err", [
        None,
        "",
        "Element not found",
        "401 Unauthorized",
        "JavaScript error in page",
        "AssertionError: required_contents missing",
    ])
    def test_non_retryable_errors(self, err) -> None:
        assert eval_error_is_retryable(err) is False, (
            f"B-1836: non-nav/non-timeout error {err!r} must NOT trigger retry"
        )

    def test_b1836_anchor_unspaced_timeout_now_matches(self) -> None:
        """Explicit anchor for the EXACT pre-fix bug — 'timeout' (unspaced) in
        the Playwright message that the old 'timed out' (spaced) keyword missed."""
        msg = "Page.goto: Timeout 30000ms exceeded."
        assert "timed out" not in msg.lower(), "spaced keyword genuinely absent"
        assert "timeout" in msg.lower(), "unspaced keyword present"
        assert eval_error_is_retryable(msg) is True, (
            "the unspaced-'timeout' message must now route to retry (was the "
            "3-fire zero-retry root cause)"
        )


class TestEvalRetryBackoff:
    """B-1836 local backoff config — spans the window, global timeout untouched."""

    def test_max_retries_increased(self) -> None:
        assert _EVAL_MAX_RETRIES == 5, "B-1836 raised retries 3→5 to span the window"

    def test_backoff_sequence_spans_window(self) -> None:
        # attempt 0..3 produce the inter-retry sleeps; verify exponential w/ cap.
        seq = [
            min(_EVAL_RETRY_BACKOFF_BASE_S * (2 ** a), _EVAL_RETRY_BACKOFF_CAP_S)
            for a in range(_EVAL_MAX_RETRIES - 1)
        ]
        assert seq == [30.0, 60.0, 120.0, 180.0], f"unexpected backoff seq: {seq}"
        # Total retry span (sleeps + 5×30s goto timeouts) must exceed the
        # empirically-observed ~8min transient window (canary confirms
        # sufficiency empirically — this only asserts the design intent).
        total_span_s = sum(seq) + _EVAL_MAX_RETRIES * 30.0
        assert total_span_s >= 480.0, (
            f"retry span {total_span_s}s should cover the ~8min window"
        )
