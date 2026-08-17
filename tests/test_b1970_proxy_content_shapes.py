"""Response-shape fixtures for the AWS proxy contract (B-1970 / B-1979 / B-1980).

codex Mode B (/stress 2026-08-16) found the proxy test suite carried **zero**
list-shaped `content` fixtures, so every adversarial shape it probed lived in a test
blind spot — including the one that actually killed a 12-cell chain that morning.
These are those shapes, pinned.

Each test names the failure it prevents rather than the code path it covers.
"""
from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("PROXY_API_KEY", "rp_test_dummy")

from p79.agents.proxy_api_agent import ProxyApiAgent  # noqa: E402

BASE_CFG = {
    "model": {"api_name": "qwen.qwen3-vl-235b-a22b",
              "base_url": "https://example.invalid/model-api/invoke",
              "use_tool_calling": True},
    "agent": {"image_max_size": 256},
}

VALID_ARGS = json.dumps({"thought": "t", "confidence": 0.9,
                         "action_type": "click", "element_id": 7})


def _agent(paper_grade: bool = True) -> ProxyApiAgent:
    return ProxyApiAgent({**BASE_CFG, "paper_grade": paper_grade})


def _obs():
    o = MagicMock()
    o.image = None
    o.text = "[7] button 'Buy'"
    return o


def _step(agent, body: dict):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = body
    resp.raise_for_status = MagicMock()
    resp.elapsed.total_seconds = MagicMock(return_value=0.5)
    resp.text = json.dumps(body)
    with patch("p79.agents.proxy_api_agent.requests.post", return_value=resp):
        return agent.step(instruction="go", obs=_obs(), history=[], observation_mode="dom")


def _tool_call(name: str = "web_action", args: str = VALID_ARGS) -> dict:
    return {"type": "function", "function": {"name": name, "arguments": args}}


def _logprobs() -> dict:
    return {"content": [{"token": "x", "logprob": -0.1,
                         "top_logprobs": [{"token": "x", "logprob": -0.1},
                                          {"token": "y", "logprob": -2.0}]}]}


# ── B-1970: the drift that actually happened ────────────────────────────────────
def test_list_shaped_content_no_longer_aborts_a_recoverable_step():
    """The 2026-08-16 WA-shop chain died here with the action already parsed.

    `content` became an Anthropic block list while `tool_calls` stayed intact; the old
    blanket assertion fired anyway and turned a fully recoverable step into a
    PaperGradeAbortError that dropped 12 cells.
    """
    action, meta = _step(_agent(), {
        "content": [{"type": "text", "text": ""}],
        "text": "",
        "tool_calls": [_tool_call()],
        "logprobs": _logprobs(),
    })
    assert action["action_type"] == "click"
    assert meta["tool_call_emitted"] is True
    assert meta["tool_call_parse_path"] == "tool_calls", \
        "must come from tool_calls, not a text-parse fallback"


def test_tool_use_block_without_top_level_tool_calls_still_fails_loud():
    """The B-1110 tripwire's actual purpose, preserved through the B-1970 fix.

    This is the shape where falling through to text-parse WOULD lose a structured
    action, so it must still raise.
    """
    with pytest.raises(RuntimeError, match="tool_use"):
        _step(_agent(), {
            "content": [{"type": "tool_use", "name": "web_action", "input": {}}],
            "logprobs": _logprobs(),
        })


# ── B-1979: sibling `text` handling outside the list branch ─────────────────────
def test_action_in_sibling_text_is_not_dropped_when_content_is_empty_string():
    """codex probe: `content=""` + `text=<valid JSON>` used to return `wait`.

    The v1 fix put the sibling fallback inside the list branch, so a STRING-shaped
    empty content never consulted it.
    """
    payload = json.dumps({"thought": "t", "action_type": "click", "element_id": 7})
    action, _ = _step(_agent(), {"content": "", "text": payload, "logprobs": _logprobs()})
    assert action["action_type"] == "click", "sibling text carried the action and was ignored"


def test_disagreeing_content_and_sibling_fails_loud_under_paper_grade():
    """Two non-empty texts that differ: silently picking one changes what gets recorded.

    The four B-1970 probes found them byte-identical; that is one day's observation of a
    provider, not an invariant, and the provider had already drifted once that week.
    """
    with pytest.raises(RuntimeError, match="disagree"):
        _step(_agent(paper_grade=True), {
            "content": [{"type": "text", "text": "BLOCK"}],
            "text": "SIBLING-LONGER",
            "tool_calls": [_tool_call()],
            "logprobs": _logprobs(),
        })


def test_disagreeing_content_and_sibling_only_warns_in_dev():
    action, _ = _step(_agent(paper_grade=False), {
        "content": [{"type": "text", "text": "BLOCK"}],
        "text": "SIBLING-LONGER",
        "tool_calls": [_tool_call()],
        "logprobs": _logprobs(),
    })
    assert action["action_type"] == "click"


# ── B-1980: tool_calls scanned, not indexed at [0] ──────────────────────────────
def test_web_action_after_a_foreign_tool_is_still_recovered():
    """codex probe: `[wrong_tool, web_action]` used to return `wait`, valid=False.

    The B-1970 guard stayed quiet because its criterion was "top-level tool_calls is
    non-empty" rather than "an action was actually recovered".
    """
    action, meta = _step(_agent(), {
        "content": [{"type": "text", "text": ""}],
        "tool_calls": [_tool_call(name="some_other_tool", args="{}"), _tool_call()],
        "logprobs": _logprobs(),
    })
    assert action["action_type"] == "click"
    assert meta["tool_call_parse_path"] == "tool_calls"


def test_a_single_web_action_records_a_zero_drop_not_none():
    """0 vs None is what makes the parallel-emission RATE computable from disk.

    None would be indistinguishable from "this baseline has no tool-call channel",
    so §3.5.1 could only assert the shape never occurred rather than measure it.
    """
    _, meta = _step(_agent(), {
        "content": [{"type": "text", "text": ""}],
        "tool_calls": [_tool_call()],
        "logprobs": _logprobs(),
    })
    assert meta["parallel_web_action_dropped"] == 0
    assert meta["parallel_web_action_dropped_args"] is None, \
        "nothing was dropped, so there must be no payload implying otherwise"


def test_two_web_action_calls_take_the_first_and_record_the_drop():
    """The v1 B-1980 fix raised here and killed the 8-cell floor chain at task 0.

    It aborted on len(web_action) > 1, but B-1980's own stated failure is "an emitted
    action was LOST" — with two calls nothing is lost, there are two candidates. The
    runner is a one-action-per-step loop, so call 2+ is conditioned on state the agent
    has not observed; executing the first and re-observing is what the pre-2026-08-16
    code did, i.e. what every archived B0 episode already assumes. Raising was the
    behavioural change, not the drop.
    """
    second = _tool_call(args=json.dumps({"thought": "t", "confidence": 0.9,
                                         "action_type": "click", "element_id": 9}))
    action, meta = _step(_agent(paper_grade=True), {
        "content": [{"type": "text", "text": ""}],
        "tool_calls": [_tool_call(), second],
        "logprobs": _logprobs(),
    })
    assert action["element_id"] == 7, "must execute the FIRST call, not the last"
    assert meta["parallel_web_action_dropped"] == 1
    assert meta["tool_call_parse_path"] == "tool_calls", \
        "a dropped call must not demote the step to a text-parse fallback"
    # gemini F3: the count alone leaves "the drop was harmless" unfalsifiable — the
    # discarded payload has to survive to disk for a reviewer to check it.
    dropped = meta["parallel_web_action_dropped_args"]
    assert isinstance(dropped, list) and len(dropped) == 1
    assert json.loads(dropped[0])["element_id"] == 9, \
        "must retain the DISCARDED call's args, not the executed one's"


def test_foreign_tool_alongside_one_web_action_is_not_counted_as_a_drop():
    """`[wrong_tool, web_action]` discards a non-action, so the drop count stays 0.

    Conflating the two would inflate the disclosed parallel-emission rate with
    ordinary foreign-tool noise.
    """
    _, meta = _step(_agent(), {
        "content": [{"type": "text", "text": ""}],
        "tool_calls": [_tool_call(name="some_other_tool", args="{}"), _tool_call()],
        "logprobs": _logprobs(),
    })
    assert meta["parallel_web_action_dropped"] == 0
