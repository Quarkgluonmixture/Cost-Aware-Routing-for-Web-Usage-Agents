"""B-1986: the response_format road must ask for exactly one JSON object.

`response_format: json_schema` non-strict bounds the SHAPE of each object and not
how many the model emits. GPT-5.6-terra returned 6 / 3 / 5 top-level objects on three
identical production-shape calls; the parser's `multiple_actions` rule (B-409) then
correctly refused all of them, so every step scored zero valid actions while the model
was behaving well. The fix is an instruction, not a post-hoc take-first — see the
comment at the patch site for why that distinction matters for B0-comparability.

These tests pin the three things that can silently regress: the instruction is present
on the response_format road, absent on the tool_calls road, and its anchor sentence
still exists in the prompt (a missed anchor is a silent no-op, which is how B-1985
stayed invisible).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
AGENT_SRC = (REPO / "p79" / "agents" / "proxy_api_agent.py").read_text()
ANCHOR = "Output ONLY valid JSON. No markdown blocks, no explanations."


@pytest.fixture(autouse=True)
def _dummy_key(monkeypatch):
    """These tests never call the API; the key only has to exist for __init__.

    Self-contained on purpose: the fire-host verification gate runs this suite on a
    machine where the real key may not be exported, and a test that silently errors
    there is worse than no test (B-1985's lesson).
    """
    monkeypatch.setenv("PROXY_API_KEY", "dummy-not-used")


def _make_agent(structured_output: str):
    from p79.agents.proxy_api_agent import ProxyApiAgent
    return ProxyApiAgent({
        "model": {
            "api_name": "global.openai.gpt-5.6-terra",
            "base_url": "https://example.invalid/never-called",
            "api_key_env": "PROXY_API_KEY",
            "use_tool_calling": True,
            "structured_output": structured_output,
            "logprobs_unavailable": True,
        },
    })


def test_anchor_sentence_still_in_prompts():
    """If the prompt is reworded, the instruction silently stops installing."""
    agent = _make_agent("tool_calls")
    # tool_calls swaps the anchor out, so check the raw prompts instead.
    raw = agent._get_system_prompts()
    assert any(ANCHOR in p for p in raw.values()), (
        f"anchor {ANCHOR!r} no longer appears in any system prompt — "
        "the B-1986 instruction would install nowhere. Update the anchor."
    )


def test_response_format_road_asks_for_exactly_one_object():
    agent = _make_agent("response_format")
    for mode, prompt in agent._system_prompts.items():
        assert "Emit exactly ONE JSON object" in prompt, (
            f"{mode!r} prompt lacks the single-object instruction (B-1986)"
        )


def test_tool_calls_road_does_not_get_the_instruction():
    """tool_choice='required' already bounds it to one call; the line would be noise."""
    agent = _make_agent("tool_calls")
    for mode, prompt in agent._system_prompts.items():
        assert "Emit exactly ONE JSON object" not in prompt, (
            f"{mode!r} tool_calls prompt should not carry the response_format "
            "instruction (B-1986)"
        )


def test_missing_anchor_fails_loud_not_silent():
    """A no-op replace must raise, not quietly leave the prompt unchanged."""
    src = AGENT_SRC
    assert "could not install the" in src and "raise RuntimeError" in src, (
        "the response_format branch must fail loudly when the anchor is absent"
    )
    branch = src[src.index('elif self._structured_output == "response_format":'):]
    assert "_after == _before" in branch[:4000], (
        "the no-op guard is gone — a reworded prompt would silently disable B-1986"
    )
