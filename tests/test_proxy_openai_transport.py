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
    """F2: function.arguments is not valid JSON + content="" (production
    shape per B-1109 /stress A2.3b P1-2-A* — proxy returns empty content
    when tool_calls field is present). Falls back to text parse on empty
    raw_content → default wait action with failure_reason audit trail.
    No crash, no GLM rescue path."""
    response = {
        "content": "",  # B-1109: production shape — proxy emits content="" with tool_calls
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
    # Production failure mode: empty content + bad args → graceful wait
    # with failure_reason audit trail (parse_failed OR tool_arguments_json_decode).
    assert isinstance(action, dict)
    assert action.get("action_type") == "wait", (
        f"empty content + bad args should fall to wait; got {action}"
    )
    assert meta["valid"] is False
    assert meta["failure_reason"] is not None  # audit trail populated
    # GLM fallback zombie keys preserved per B-991 retire + B-1111 uniform-None
    # (/stress A2.3b P1-6-A 2026-05-18): all 4 keys serialize None uniformly
    # post-fix; pre-fix `used=False vs attempted=None` was semantic confusion
    # ("never tried" vs "never relevant" both mean GLM module non-existent).
    assert meta["glm_fallback_used"] is None  # B-1111 uniform-None zombie
    assert meta["glm_fallback_attempted"] is None
    assert meta["glm_fallback_latency_ms"] is None
    assert meta["glm_original_fail_reason"] is None


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


# ── P1-3-B (/stress GRL audit 2026-05-20, Q4=A): B0 action provenance ────────
_P13_LOGPROBS = {"content": [
    {"token": "x", "logprob": -0.1,
     "top_logprobs": [{"token": "x", "logprob": -0.1}, {"token": "y", "logprob": -2.0}]},
]}


def test_p1_3_paper_grade_invalid_tool_call_no_text_fallback(monkeypatch, tmp_path):
    """P1-3-B (Q4=A): paper-grade B0 with an EMITTED-but-invalid tool_call must
    NOT silently text-parse a DIFFERENT action (action provenance integrity +
    cross-baseline tool-call failure-rate honesty). Records action_source=
    'invalid' + valid=False; the content's (different) scroll action never runs."""
    monkeypatch.setenv("PROXY_API_KEY", "rp_test_dummy")
    from p79.agents.proxy_api_agent import ProxyApiAgent
    agent = ProxyApiAgent({
        "model": {"api_name": "qwen.qwen3-vl-235b-a22b",
                  "base_url": "https://i5xpracyci.execute-api.eu-west-2.amazonaws.com/model-api/invoke",
                  "use_tool_calling": True},
        "agent": {"image_max_size": 256}, "paper_grade": True,
    })
    response = {
        # content carries a DIFFERENT valid action — the silent-swap risk vector.
        "content": json.dumps({"action_type": "scroll", "scroll_direction": "down", "thought": "x"}),
        "tool_calls": [{"id": "bad", "type": "function",
                        "function": {"name": "web_action", "arguments": "{not valid json}"}}],
        "model": "qwen.qwen3-vl-235b-a22b",
        "usage": {"inputTokens": 100, "outputTokens": 10, "cost": 0.0002},
        "metadata": {}, "logprobs": _P13_LOGPROBS,
    }
    _patch_requests_post(monkeypatch, response)
    action, meta = agent.step(instruction="t", obs=_mock_obs(), history=[], observation_mode="dom")
    assert action.get("action_type") != "scroll", "must NOT execute content's different action"
    assert meta["action_source"] == "invalid"
    assert meta["valid"] is False
    assert meta["tool_call_valid"] is False
    assert meta["text_fallback_used"] is False


def test_p1_3_dev_mode_invalid_tool_call_still_text_falls_back(_proxy_agent, monkeypatch):
    """Contrast: non-paper-grade (dev) keeps the lenient fallback — the same
    invalid tool_call DOES text-parse the content action (action_source=
    'fallback'). Confirms the fix is scoped to paper_grade only."""
    response = {
        "content": json.dumps({"action_type": "scroll", "scroll_direction": "down", "thought": "x"}),
        "tool_calls": [{"id": "bad", "type": "function",
                        "function": {"name": "web_action", "arguments": "{not valid json}"}}],
        "model": "qwen.qwen3-vl-235b-a22b",
        "usage": {"inputTokens": 100, "outputTokens": 10, "cost": 0.0002}, "metadata": {},
    }
    _patch_requests_post(monkeypatch, response)
    action, meta = _proxy_agent.step(instruction="t", obs=_mock_obs(), history=[], observation_mode="dom")
    assert action.get("action_type") == "scroll", "dev mode text-parses the content action"
    assert meta["action_source"] == "fallback"
    assert meta["text_fallback_used"] is True


def test_p1_3_valid_tool_call_action_source(_proxy_agent, monkeypatch):
    """Valid native tool_call → action_source='tool_call', tool_call_valid=True,
    no text fallback."""
    response = {
        "content": "",
        "tool_calls": [{"id": "ok", "type": "function",
                        "function": {"name": "web_action",
                                     "arguments": json.dumps({"action_type": "click", "element_id": 2, "thought": "go"})}}],
        "model": "qwen.qwen3-vl-235b-a22b",
        "usage": {"inputTokens": 100, "outputTokens": 10, "cost": 0.0002}, "metadata": {},
    }
    _patch_requests_post(monkeypatch, response)
    action, meta = _proxy_agent.step(instruction="t", obs=_mock_obs(), history=[], observation_mode="dom")
    assert meta["action_source"] == "tool_call"
    assert meta["tool_call_valid"] is True
    assert meta["text_fallback_used"] is False


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


# B-1101 (/stress A2.3b P0-1-AC* OOB, 2026-05-18): proxy tool_calling
# empirical model emits `element_id: [37]` (1-element int list) under
# `tool_choice="auto"` despite schema declaring `type=integer`. AWS
# Bedrock proxy does NOT enforce tools schema on output. Pre-fix
# validate_action rejected list-typed _eid → Path-2 text parse on empty
# content → 30 wait/episode contamination. Coerce 1-element strict-int
# list → int with explicit `len==1` guard.
def test_f6_element_id_list_coerce_succeeds(_proxy_agent, monkeypatch):
    """F6: Model emits `element_id: [37]` (1-element int list) — validator
    coerces to int 37, action passes validation, dispatched correctly."""
    response = {
        "content": "",
        "tool_calls": [
            {
                "id": "id6", "type": "function",
                "function": {
                    "name": "web_action",
                    "arguments": json.dumps({
                        "action_type": "click",
                        "element_id": [37],  # list — model empirical emission
                        "thought": "click the kayak listing",
                    }),
                },
            }
        ],
        "model": "qwen.qwen3-vl-235b-a22b",
        "usage": {"inputTokens": 100, "outputTokens": 20, "cost": 0.0003},
        "metadata": {},
        "logprobs": {"content": [{"token": "click", "logprob": -0.1,
                                   "top_logprobs": [{"token": "click", "logprob": -0.1},
                                                    {"token": "type", "logprob": -2.5}]}]},
    }
    _patch_requests_post(monkeypatch, response)
    action, meta = _proxy_agent.step(
        instruction="test", obs=_mock_obs(), history=[], observation_mode="dom",
    )
    assert action.get("action_type") == "click", f"action_type drift: {action}"
    assert action.get("element_id") == 37, (
        f"1-element list [37] should coerce to int 37; got {action.get('element_id')!r}"
    )
    assert action.get("element_id_coerced_from_list") is True
    assert meta["valid"] is True


def test_f6b_element_id_multi_element_list_rejected(_proxy_agent, monkeypatch):
    """F6b: Multi-element list `[37, 38]` MUST reject (don't silent-pick
    first); otherwise model multi-target emit silently dispatches to
    arbitrary first id. Strict `len==1` guard."""
    response = {
        "content": "",
        "tool_calls": [
            {
                "id": "id6b", "type": "function",
                "function": {
                    "name": "web_action",
                    "arguments": json.dumps({
                        "action_type": "click",
                        "element_id": [37, 38],  # 2-element list — ambiguous
                        "thought": "click",
                    }),
                },
            }
        ],
        "model": "qwen.qwen3-vl-235b-a22b",
        "usage": {"inputTokens": 100, "outputTokens": 20, "cost": 0.0003},
        "metadata": {},
        "logprobs": {"content": [{"token": "x", "logprob": -0.1, "top_logprobs": [
            {"token": "x", "logprob": -0.1}, {"token": "y", "logprob": -2.0}]}]},
    }
    _patch_requests_post(monkeypatch, response)
    action, meta = _proxy_agent.step(
        instruction="test", obs=_mock_obs(), history=[], observation_mode="dom",
    )
    # Multi-element list does NOT coerce; validator rejects → wait fallback.
    assert action.get("action_type") == "wait", (
        f"multi-element [37,38] should reject and fall back to wait; got {action}"
    )
    assert "element_id_coerced_from_list" not in action


# B-1103 (/stress A2.3b P0-4-B* codex OOB, 2026-05-18): paper-grade B0
# missing-logprobs must fail-loud (advertised at launch but not invariant
# would survive otherwise).
def test_f7_paper_grade_missing_logprobs_raises(monkeypatch, tmp_path):
    """F7: paper_grade=True + use_tool_calling=True + proxy response
    missing `logprobs.content` → RuntimeError. Provider drift / quota
    mode change MUST surface, not silently produce zero-confidence rows."""
    monkeypatch.setenv("PROXY_API_KEY", "rp_test_dummy")
    from p79.agents.proxy_api_agent import ProxyApiAgent

    config = {
        "model": {
            "api_name": "qwen.qwen3-vl-235b-a22b",
            "base_url": "https://i5xpracyci.execute-api.eu-west-2.amazonaws.com/model-api/invoke",
            "use_tool_calling": True,
        },
        "agent": {"image_max_size": 256},
        "paper_grade": True,  # paper-grade run — fail-loud invariant
    }
    agent = ProxyApiAgent(config)
    response = {
        "content": "",
        "tool_calls": [
            {
                "id": "id7", "type": "function",
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
        # NO logprobs key — provider drift simulation
    }
    _patch_requests_post(monkeypatch, response)
    with pytest.raises(RuntimeError, match=r"missing logprobs|B-1103"):
        agent.step(instruction="test", obs=_mock_obs(), history=[], observation_mode="dom")


def test_f7b_dev_run_missing_logprobs_persists_confidence_error(_proxy_agent, monkeypatch):
    """F7b: Non-paper-grade dev run + missing logprobs → no raise; persist
    `meta["confidence_error"] = "missing_proxy_logprobs"` for audit."""
    response = {
        "content": "",
        "tool_calls": [
            {
                "id": "id7b", "type": "function",
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
        # NO logprobs — dev run, no raise
    }
    _patch_requests_post(monkeypatch, response)
    action, meta = _proxy_agent.step(
        instruction="test", obs=_mock_obs(), history=[], observation_mode="dom",
    )
    assert meta.get("confidence_error") == "missing_proxy_logprobs"
    assert action.get("action_type") == "click"


# B-1102 (/stress A2.3b P1-4-A, 2026-05-18): paper-grade B0 MUST set
# use_tool_calling=true post-B-991 (GLM rescue deleted).
# B-1105 (/stress A2.3b P0-5-A OOB, 2026-05-18): margin assertion when
# top[0] != chosen token (provider drift OR T>0 sampling).
def test_f9_margin_skipped_when_top_zero_not_chosen(_proxy_agent, monkeypatch):
    """F9: When top_logprobs[0].token != chosen entry.token (e.g. proxy
    drift returns top_logprobs sorted differently OR future T>0 sampling
    picks mid-list), margin computation must SKIP that token (not silently
    compute bogus margin from non-chosen alternatives)."""
    logprobs_content = [
        {  # tok 1: top[0]=chosen → margin valid
            "token": "click",
            "logprob": -0.10,
            "top_logprobs": [
                {"token": "click", "logprob": -0.10},
                {"token": "type", "logprob": -2.50},
            ],
        },
        {  # tok 2: top[0] != chosen → margin skipped (B-1105)
            "token": "select_option",
            "logprob": -1.50,
            "top_logprobs": [
                {"token": "click", "logprob": -0.05},  # chosen NOT top[0]
                {"token": "type", "logprob": -3.00},
            ],
        },
    ]
    response = {
        "content": "",
        "tool_calls": [{
            "id": "id9", "type": "function",
            "function": {"name": "web_action", "arguments": json.dumps({
                "action_type": "click", "element_id": 2, "thought": "go",
            })},
        }],
        "model": "qwen.qwen3-vl-235b-a22b",
        "usage": {"inputTokens": 100, "outputTokens": 2, "cost": 0.0003},
        "metadata": {},
        "logprobs": {"content": logprobs_content},
    }
    _patch_requests_post(monkeypatch, response)
    action, meta = _proxy_agent.step(
        instruction="test", obs=_mock_obs(), history=[], observation_mode="dom",
    )
    # mean_logprob averages BOTH chosen logprobs (-0.10 and -1.50) — chosen
    # logprob always recorded regardless of top[0] match.
    assert meta["mean_logprob"] == pytest.approx((-0.10 + -1.50) / 2, abs=1e-6)
    # margin SKIPS tok 2 (top[0]!=chosen) so margins_list only has tok1's 2.40
    assert meta["mean_margin"] == pytest.approx(2.40, abs=1e-6)
    assert meta["min_margin"] == pytest.approx(2.40, abs=1e-6)


def test_f8_paper_grade_without_tool_calling_raises_at_init(monkeypatch):
    """F8: paper_grade=True + use_tool_calling=False (or unset) → init
    RuntimeError. Misconfigured yaml surfaces at construction, NOT
    mid-fire after burning cell N tasks."""
    monkeypatch.setenv("PROXY_API_KEY", "rp_test_dummy")
    from p79.agents.proxy_api_agent import ProxyApiAgent

    config = {
        "model": {
            "api_name": "qwen.qwen3-vl-235b-a22b",
            "base_url": "https://i5xpracyci.execute-api.eu-west-2.amazonaws.com/model-api/invoke",
            "use_tool_calling": False,  # mis-config
        },
        "agent": {"image_max_size": 256},
        "paper_grade": True,
    }
    with pytest.raises(RuntimeError, match=r"paper-grade B0 requires use_tool_calling=true|B-1102"):
        ProxyApiAgent(config)


# ----- B-1880: capped exponential backoff (reddit chain abort #3, 2026-06-19) -----
# RCA: R28130 B0 dom reddit died at task 59/205 when a ~3min sustained AWS-proxy
# 503 window exhausted the 3-retry/70s budget -> first quarantine event ->
# PaperGradeAbortError -> whole 205-task condition lost at 58/205. Fix: thicken
# the retry budget (yaml max_retries up) but cap the exponential backoff so a
# single sleep cannot balloon (uncapped doubling at attempt 7 would sleep 1280s).
# These freeze the cap contract so a regression fails CI, not mid-fire.


def _build_proxy_agent(monkeypatch, model_extra):
    monkeypatch.setenv("PROXY_API_KEY", "rp_test_dummy")
    from p79.agents.proxy_api_agent import ProxyApiAgent

    config = {
        "model": {
            "api_name": "qwen.qwen3-vl-235b-a22b",
            "base_url": "https://i5xpracyci.execute-api.eu-west-2.amazonaws.com/model-api/invoke",
            "use_tool_calling": True,
            **model_extra,
        },
        "agent": {"image_max_size": 256},
        "paper_grade": False,
    }
    return ProxyApiAgent(config)


def _patch_503_capture_sleeps(monkeypatch):
    """All requests.post return HTTP 503 (raise_for_status raises HTTPError on
    the last attempt); capture each backoff sleep instead of sleeping. Returns
    the list that accumulates wait values in order."""
    import requests as _rq

    resp_mock = MagicMock()
    resp_mock.status_code = 503
    resp_mock.json.return_value = {}
    resp_mock.text = "Service Unavailable"

    def _raise_for_status():
        raise _rq.exceptions.HTTPError(
            "503 Server Error: Service Unavailable", response=resp_mock,
        )

    resp_mock.raise_for_status = _raise_for_status

    def _fake_post(url, json=None, headers=None, timeout=None, **kw):
        return resp_mock

    monkeypatch.setattr("p79.agents.proxy_api_agent.requests.post", _fake_post)

    sleeps: list = []
    monkeypatch.setattr(
        "p79.agents.proxy_api_agent.time.sleep", lambda s: sleeps.append(s),
    )
    return sleeps


def test_b1880_capped_exponential_backoff(monkeypatch):
    """B-1880: retry_backoff_max_s caps the doubling. max_retries=4, base=10,
    cap=60 -> sleeps [10, 20, 40, 60] (the 4th capped from 80)."""
    import requests as _rq

    agent = _build_proxy_agent(
        monkeypatch,
        {"max_retries": 4, "retry_backoff_s": 10, "retry_backoff_max_s": 60},
    )
    sleeps = _patch_503_capture_sleeps(monkeypatch)
    with pytest.raises(_rq.exceptions.HTTPError):
        agent.step(
            instruction="Find a kayak",
            obs=_mock_obs(),
            history=[],
            observation_mode="dom",
        )
    assert sleeps == [10, 20, 40, 60], f"capped backoff drift: {sleeps}"


def test_b1880_uncapped_backoff_back_compat(monkeypatch):
    """B-1880 back-compat: retry_backoff_max_s unset -> unbounded doubling
    (pre-B-1880 behavior preserved). max_retries=3, base=10 -> [10, 20, 40]."""
    import requests as _rq

    agent = _build_proxy_agent(
        monkeypatch, {"max_retries": 3, "retry_backoff_s": 10},
    )
    sleeps = _patch_503_capture_sleeps(monkeypatch)
    with pytest.raises(_rq.exceptions.HTTPError):
        agent.step(
            instruction="Find a kayak",
            obs=_mock_obs(),
            history=[],
            observation_mode="dom",
        )
    assert sleeps == [10, 20, 40], f"uncapped backoff drift: {sleeps}"


def test_b1880_base_yaml_thickened_budget():
    """B-1880: exp_v2_base.yaml api_strong carries the thickened budget so all
    B0 per-site configs inherit it (Hydra defaults + recursive merge)."""
    import yaml

    base = yaml.safe_load(
        (REPO_ROOT / "configs" / "exp_v2_base.yaml").read_text()
    )
    api_strong = base["backends"]["api_strong"]
    assert api_strong["max_retries"] == 8, api_strong.get("max_retries")
    assert api_strong["retry_backoff_s"] == 10, api_strong.get("retry_backoff_s")
    assert api_strong["retry_backoff_max_s"] == 60, api_strong.get("retry_backoff_max_s")
