"""B-993 (/stress A1.2-followup P1-3, 2026-05-17): proxy transport fixture
tests covering the OpenAI-style tool_calling + logprobs migration
(B-991 retire of GLM rescue). Verifies the new agent code paths against
mocked proxy responses so wire-format drift fails CI rather than mid-fire.

Coverage:
  F1. Success path — top-level `body.tool_calls[0].function.arguments`
      parses to valid action with `schema_valid=True`.
  F2. Malformed arguments — JSON decode failure falls back to Path-2 text
      parse (no crash, no GLM rescue path).
  F3. Missing logprobs — meta dict has 6 confidence keys all None when
      proxy omits `body.logprobs`.
  F4. Top-2 logprobs — 4 of 6 confidence fields populate (mean/min logprob
      + mean/min margin); entropy fields remain None per top-2 truncation.
  F5. Old Anthropic-format response shape — content-only string response
      (no tool_calls field) falls back to Path-2 text parse.

These tests freeze the contract that protects Phase 1a B0 fires from
silent proxy regressions / provider drift / response-shape changes.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def _proxy_agent(monkeypatch, tmp_path):
    """Construct a ProxyApiAgent with tool_calling enabled, mocking out
    PIL/network deps so we can exercise the parse paths in isolation."""
    monkeypatch.setenv("PROXY_API_KEY", "rp_test_dummy")
    from p79.agents.proxy_api_agent import ProxyApiAgent

    config = {
        "model": {
            "api_name": "qwen.qwen3-vl-235b-a22b",
            "base_url": "https://i5xpracyci.execute-api.eu-west-2.amazonaws.com/model-api/invoke",
            "use_tool_calling": True,
        },
        "agent": {"image_max_size": 256},
        "paper_grade": False,
    }
    agent = ProxyApiAgent(config)
    return agent


def _mock_obs():
    """Return a minimal obs object the agent accepts (no image; AXTree text)."""
    obs = MagicMock()
    obs.image = None
    obs.text = "[1] heading 'Results'\n[2] link 'cheapest blue kayak'"
    return obs


def _patch_requests_post(monkeypatch, response_body, status_code=200):
    """Patch `requests.post` inside proxy_api_agent module to return the
    given response body. Returns the resp_mock so the test can inspect
    call args if needed."""
    resp_mock = MagicMock()
    resp_mock.status_code = status_code
    resp_mock.json.return_value = response_body
    resp_mock.raise_for_status = MagicMock()
    resp_mock.elapsed.total_seconds = MagicMock(return_value=0.5)
    resp_mock.text = json.dumps(response_body) if isinstance(response_body, dict) else str(response_body)

    def _fake_post(url, json=None, headers=None, timeout=None, **kw):
        return resp_mock

    monkeypatch.setattr("p79.agents.proxy_api_agent.requests.post", _fake_post)
    return resp_mock


def test_f1_success_path_top_level_tool_calls(_proxy_agent, monkeypatch):
    """F1: Proxy returns top-level `tool_calls` → parse function.arguments
    JSON → validate_action succeeds → action_type populated."""
    response = {
        "content": "",
        "tool_calls": [
            {
                "id": "chatcmpl-tool-abc123",
                "type": "function",
                "function": {
                    "name": "web_action",
                    "arguments": json.dumps({
                        "action_type": "click",
                        "element_id": 2,
                        "thought": "Click the cheapest blue kayak link.",
                    }),
                },
            }
        ],
        "model": "qwen.qwen3-vl-235b-a22b",
        "usage": {"inputTokens": 200, "outputTokens": 50, "cost": 0.0007},
        "metadata": {"remaining_quota": 1000},
    }
    _patch_requests_post(monkeypatch, response)
    action, meta = _proxy_agent.step(
        instruction="Click the cheapest blue kayak",
        obs=_mock_obs(),
        history=[],
        observation_mode="dom",
    )
    assert action.get("action_type") == "click", f"action_type drift: {action}"
    assert action.get("element_id") == 2, f"element_id drift: {action}"
    assert meta["valid"] is True, "meta.valid should be True on success"
    assert meta["failure_reason"] is None or meta["failure_reason"] == "valid"
    assert meta["tool_calling"] is True


def test_f2_malformed_arguments_falls_back_to_text_parse(_proxy_agent, monkeypatch):
    """F2: function.arguments is not valid JSON → falls back to text parse
    path (Path-2). No crash. fail_reason should reflect the JSON decode
    failure or the text-parse outcome."""
    response = {
        "content": '{"action_type": "wait"}',  # fallback content is a valid wait action
        "tool_calls": [
            {
                "id": "chatcmpl-tool-bad",
                "type": "function",
                "function": {
                    "name": "web_action",
                    "arguments": "{this is not valid json}",
                },
            }
        ],
        "model": "qwen.qwen3-vl-235b-a22b",
        "usage": {"inputTokens": 100, "outputTokens": 10, "cost": 0.0002},
        "metadata": {},
    }
    _patch_requests_post(monkeypatch, response)
    # Should NOT raise; Path-1 fails JSON decode → fall through to Path-2.
    action, meta = _proxy_agent.step(
        instruction="test",
        obs=_mock_obs(),
        history=[],
        observation_mode="dom",
    )
    # Action should be a dict (text parse fallback or final wait default).
    assert isinstance(action, dict)
    assert "action_type" in action
    # GLM fallback fields preserved as zombie schema keys; existing
    # serialization is `attempted ? value : None` → both None when never
    # attempted (B-991 retire). The KEY presence is the contract; value
    # equality is None.
    assert meta["glm_fallback_used"] is False  # bool literal, hardcoded
    assert meta["glm_fallback_attempted"] is None  # serialized None when not attempted


def test_f3_missing_logprobs_fills_none(_proxy_agent, monkeypatch):
    """F3: Proxy response omits `logprobs` key → confidence helper returns
    6-field dict all None. Runner downstream skips writing
    `step_record["confidence"]` because mean_logprob is None."""
    response = {
        "content": "",
        "tool_calls": [
            {
                "id": "id1", "type": "function",
                "function": {
                    "name": "web_action",
                    "arguments": json.dumps({
                        "action_type": "click", "element_id": 2, "thought": "go",
                    }),
                },
            }
        ],
        "model": "qwen.qwen3-vl-235b-a22b",
        "usage": {"inputTokens": 100, "outputTokens": 20, "cost": 0.0003},
        "metadata": {},
        # NO logprobs key
    }
    _patch_requests_post(monkeypatch, response)
    action, meta = _proxy_agent.step(
        instruction="test", obs=_mock_obs(), history=[], observation_mode="dom",
    )
    assert action.get("action_type") == "click"
    # All 6 confidence fields exist in meta as None (zombie keys for cross-baseline schema parity).
    for key in ("mean_logprob", "min_logprob", "mean_margin", "min_margin",
                "mean_entropy", "max_entropy"):
        assert key in meta, f"meta missing confidence field {key}"
        assert meta[key] is None, f"meta.{key} should be None when no logprobs (got {meta[key]})"


def test_f4_top2_logprobs_fills_4_of_6_fields(_proxy_agent, monkeypatch):
    """F4: Proxy returns top-2 logprobs → 4 confidence fields populate
    (mean/min logprob + mean/min margin). Entropy fields remain None
    because full-vocab entropy is not recoverable from top-2 truncation."""
    logprobs_content = [
        {
            "token": "click",
            "logprob": -0.10,
            "top_logprobs": [
                {"token": "click", "logprob": -0.10},
                {"token": "type", "logprob": -2.50},
            ],
        },
        {
            "token": "_id",
            "logprob": -0.05,
            "top_logprobs": [
                {"token": "_id", "logprob": -0.05},
                {"token": "_label", "logprob": -3.00},
            ],
        },
    ]
    response = {
        "content": "",
        "tool_calls": [
            {
                "id": "id2", "type": "function",
                "function": {
                    "name": "web_action",
                    "arguments": json.dumps({
                        "action_type": "click", "element_id": 2, "thought": "go",
                    }),
                },
            }
        ],
        "model": "qwen.qwen3-vl-235b-a22b",
        "usage": {"inputTokens": 100, "outputTokens": 2, "cost": 0.0003},
        "metadata": {},
        "logprobs": {"content": logprobs_content},
    }
    _patch_requests_post(monkeypatch, response)
    action, meta = _proxy_agent.step(
        instruction="test", obs=_mock_obs(), history=[], observation_mode="dom",
    )
    # 4 populated fields
    assert meta["mean_logprob"] == pytest.approx((-0.10 + -0.05) / 2, abs=1e-6)
    assert meta["min_logprob"] == pytest.approx(-0.10, abs=1e-6)
    assert meta["mean_margin"] == pytest.approx(((-0.10 - -2.50) + (-0.05 - -3.00)) / 2, abs=1e-6)
    assert meta["min_margin"] == pytest.approx(min(2.40, 2.95), abs=1e-6)
    # 2 entropy fields None per top-2 truncation
    assert meta["mean_entropy"] is None, "entropy not recoverable from top-2"
    assert meta["max_entropy"] is None, "entropy not recoverable from top-2"


def test_f5_legacy_anthropic_text_only_response(_proxy_agent, monkeypatch):
    """F5: Provider drift / proxy regression returns Anthropic-style
    free-text response (no `tool_calls` field, just `content` string). Agent
    must fall back to Path-2 text parse instead of crashing or hanging."""
    response = {
        "content": '{"action_type": "click", "element_id": 2, "thought": "test"}',
        # No tool_calls field — simulates pre-migration response shape
        "model": "qwen.qwen3-vl-235b-a22b",
        "usage": {"inputTokens": 100, "outputTokens": 20, "cost": 0.0003},
        "metadata": {},
    }
    _patch_requests_post(monkeypatch, response)
    action, meta = _proxy_agent.step(
        instruction="test", obs=_mock_obs(), history=[], observation_mode="dom",
    )
    # Text parser should extract the JSON from content string.
    assert action.get("action_type") == "click", (
        f"text parse fallback failed on Anthropic shape: {action}"
    )
    assert action.get("element_id") == 2
    # No tool_calls means meta.tool_calling stays True (config flag), but
    # the actual route was Path-2.
    assert meta["valid"] is True
