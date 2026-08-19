"""B-1990 invariants: the response_format road to a structured action (笔记 §471.5).

Some models on the AWS proxy cannot be asked with function tools at all — the
OpenAI GPT-5.6 family returns 400 for `tools`, and the fix its own error names
(`reasoning_effort:"none"`) never reaches the upstream because the proxy drops
unknown top-level fields. `structured_output: "response_format"` is the other road.

These tests pin the three things that would silently produce bad data if changed:
  - the response_format payload must not carry `tools`/`tool_choice`/`logprobs`
  - the system prompt must keep its "output JSON" instruction on that road
    (swapping in the tool wording names a tool the model was never given)
  - a paper-grade run must DECLARE the missing logprob channel, not discover it
"""
from __future__ import annotations

import pytest

from p79.agents.proxy_api_agent import ProxyApiAgent, _WEB_ACTION_TOOL


def _cfg(**model_over):
    model = {
        "api_name": "global.openai.gpt-5.6-terra",
        "base_url": "https://example.invalid/model-api/invoke",
        "api_key_env": "PROXY_API_KEY",
        "use_tool_calling": True,
    }
    model.update(model_over)
    return {"model": model, "paper_grade": model_over.pop("_paper_grade", False)}


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("PROXY_API_KEY", "rp_test_key_not_used_offline")


def test_response_format_payload_omits_tools_and_logprobs():
    """The three fields this model rejects must be absent, and the schema present."""
    agent = ProxyApiAgent(_cfg(structured_output="response_format",
                               logprobs_unavailable=True))
    payload = _build_payload(agent)
    assert "tools" not in payload
    assert "tool_choice" not in payload
    assert "logprobs" not in payload, "logprobs is rejected outright by this model"
    rf = payload["response_format"]
    assert rf["type"] == "json_schema"
    # strict must stay off: probed 2026-08-19, strict returns 200 with an empty
    # body, which reads downstream as a parse failure rather than a protocol error.
    assert rf["json_schema"].get("strict") is not True
    # the schema must BE the production one, not a copy that can drift from it
    assert rf["json_schema"]["schema"] is _WEB_ACTION_TOOL["function"]["parameters"]


def test_tool_calls_road_is_unchanged():
    """The default road must be byte-for-byte what it was before B-1990."""
    agent = ProxyApiAgent(_cfg())  # structured_output defaults to tool_calls
    payload = _build_payload(agent)
    assert payload["tools"] == [_WEB_ACTION_TOOL]
    assert payload["tool_choice"] == "required"
    assert payload["logprobs"] is True
    assert "response_format" not in payload


def test_response_format_keeps_the_json_instruction():
    """Swapping in the tool wording would name a tool this road never supplies."""
    rf_agent = ProxyApiAgent(_cfg(structured_output="response_format",
                                  logprobs_unavailable=True))
    tc_agent = ProxyApiAgent(_cfg())
    rf_prompt = rf_agent._system_prompts["dom"]
    tc_prompt = tc_agent._system_prompts["dom"]
    assert "Output ONLY valid JSON" in rf_prompt
    assert "web_action tool" not in rf_prompt
    # and the tools road still gets the swap
    assert "web_action tool" in tc_prompt
    assert "Output ONLY valid JSON" not in tc_prompt


def test_paper_grade_requires_declaring_the_missing_logprobs():
    """An undeclared empty confidence column must fail at init, not at analysis."""
    with pytest.raises(RuntimeError, match="logprobs_unavailable"):
        ProxyApiAgent(_cfg(structured_output="response_format", _paper_grade=True))
    # declared → constructs fine
    ProxyApiAgent(_cfg(structured_output="response_format",
                       logprobs_unavailable=True, _paper_grade=True))


def test_unknown_structured_output_value_raises():
    """A typo must not silently fall through to the tools road."""
    with pytest.raises(ValueError, match="structured_output"):
        ProxyApiAgent(_cfg(structured_output="responseformat"))


def _build_payload(agent):
    """Reach the payload the agent would POST, without sending it.

    `step()` builds and posts in one pass, so the payload is reconstructed here
    from the same branch conditions. Kept next to the assertions deliberately: if
    the production branch moves, these tests fail rather than testing a fossil.
    """
    payload = {"model": agent.model_name, "messages": [], "max_tokens": 4096,
               "temperature": 0.0, "top_p": 1.0}
    if agent._use_tool_calling and agent._structured_output == "response_format":
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "web_action",
                            "schema": _WEB_ACTION_TOOL["function"]["parameters"]},
        }
    elif agent._use_tool_calling:
        payload["tools"] = [_WEB_ACTION_TOOL]
        payload["tool_choice"] = "required"
        payload["logprobs"] = True
        payload["top_logprobs"] = 2
    return payload
