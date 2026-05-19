"""Fire-4 RCA Wave 2 M5 test — `classify_timeout()` callsite taxonomy.

User A1=b decision 2026-05-19: timeout error messages must be decomposed
into (is_timeout, callsite) so:
  * `unverified_timeout_event=True` populates the summary
  * `timeout_callsite` carries the Playwright callsite for forensic review
  * `benchmark_noise=False` (backward-compat archive readers)

Test asserts:
  1. classify_timeout returns (False, None) for non-timeout strings
  2. classify_timeout returns (True, callsite) for timeout substrings
  3. callsite buckets are deterministic for known Playwright error patterns
  4. detect_benchmark_noise no longer auto-tags timeout substrings
     (verifies Wave 2 M5 downgrade per user A1=b)
"""
from __future__ import annotations

import pytest

from p79.experiment.metrics import classify_timeout, detect_benchmark_noise


class TestClassifyTimeoutCallsite:
    """Callsite taxonomy table (Playwright error messages → bucket)."""

    @pytest.mark.parametrize(
        "err,expected_callsite",
        [
            # Fire-3 task 75 EvaluatorUnavailableError-wrapped Page.goto timeout.
            # NOTE: in production the EvaluatorUnavailableError re-raises BEFORE
            # summary write (runner/main.py:1505), so classify_timeout is NOT
            # called for this case. Test included for taxonomy coverage.
            ("Page.goto: Timeout 30000ms exceeded.", "agent_navigation"),
            # Fire-4 task 75 Page.screenshot timeout (agent observation step).
            ("Page.screenshot: Timeout 30000ms exceeded.", "agent_observation"),
            # Agent action timeouts (click / fill / hover / type / select).
            ("Page.click: Timeout 30000ms exceeded.", "agent_action"),
            ("Page.fill: Timeout 30000ms exceeded.", "agent_action"),
            ("Page.type: Timeout 30000ms exceeded.", "agent_action"),
            ("Page.hover: Timeout 30000ms exceeded.", "agent_action"),
            ("Page.select_option: Timeout 30000ms exceeded.", "agent_action"),
            ("locator.click: Timeout 30000ms exceeded.", "agent_action"),
            # Network failures (ECONNREFUSED etc.).
            ("connect ECONNREFUSED 127.0.0.1:9980: Timeout", "network"),
            ("Connection refused; deadline exceeded", "network"),
            # Generic Playwright timeout without specific callsite.
            ("Playwright operation timed out at 30000ms", "agent_playwright_other"),
            # Unknown timeout pattern.
            ("operation timed out", "unknown"),
            ("deadline exceeded after retry", "unknown"),
        ],
    )
    def test_callsite_classification(self, err: str, expected_callsite: str) -> None:
        is_timeout, callsite = classify_timeout(err)
        assert is_timeout is True, f"Should detect timeout: {err}"
        assert callsite == expected_callsite, (
            f"Callsite mismatch for {err!r}: got {callsite!r}, want {expected_callsite!r}"
        )

    @pytest.mark.parametrize(
        "err",
        [
            None,
            "",
            "Element not found",
            "JavaScript error in page",
            "429 Too Many Requests",
            "401 Unauthorized",
        ],
    )
    def test_non_timeout_returns_false_none(self, err) -> None:
        is_timeout, callsite = classify_timeout(err)
        assert is_timeout is False, f"Should NOT detect timeout: {err!r}"
        assert callsite is None, f"Non-timeout callsite should be None, got {callsite!r}"


class TestDetectBenchmarkNoiseDowngrade:
    """Wave 2 M5 downgrade: detect_benchmark_noise no longer auto-tags timeouts."""

    @pytest.mark.parametrize(
        "err",
        [
            "Page.screenshot: Timeout 30000ms exceeded.",
            "Page.goto: Timeout 30000ms exceeded",
            "operation timed out",
            "deadline exceeded",
        ],
    )
    def test_timeout_no_longer_auto_tagged(self, err: str) -> None:
        """Pre-Wave-2 these returned (True, 'timeout'); post-Wave-2 → (False, None)."""
        is_noise, category = detect_benchmark_noise(err)
        assert is_noise is False, (
            f"M5 downgrade: timeout substring should no longer auto-tag benchmark_noise. "
            f"Error: {err!r}. Got is_noise={is_noise}"
        )
        assert category is None, (
            f"M5 downgrade: timeout category should be None. Got {category!r}"
        )

    @pytest.mark.parametrize(
        "err,expected_noise,expected_category",
        [
            ("429 Too Many Requests", True, "api_rate_limit"),
            ("rate limit exceeded", True, "api_rate_limit"),
            ("auth expired session", True, "auth_expired_or_session_invalid"),
            ("model-api invocation error", True, "api_infra"),
            ("execute-api timeout occurred", True, "api_infra"),
            ("captcha challenge required", True, "anti_bot_or_blocked"),
            ("geo-restricted region", True, "geo_restricted"),
        ],
    )
    def test_non_timeout_noise_categories_unchanged(
        self, err: str, expected_noise: bool, expected_category: str
    ) -> None:
        """Non-timeout noise taxonomy (api_rate_limit / auth / api_infra etc.) preserved."""
        is_noise, category = detect_benchmark_noise(err)
        assert is_noise is expected_noise, (
            f"Non-timeout noise category broken — error {err!r} got is_noise={is_noise}"
        )
        assert category == expected_category, (
            f"Non-timeout category mismatch — error {err!r} got {category!r}, "
            f"want {expected_category!r}"
        )
