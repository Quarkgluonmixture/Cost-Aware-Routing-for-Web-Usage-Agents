"""MiMo `/no_think` placement — the official card makes placement load-bearing.

"`/no_think` command must be the very last part of user message, which means after
`/no_think`, there shouldn't be any user content like image or video."

The inherited `Qwen3VLAgent.step()` appends the screenshot AFTER the text whenever
reference images are present, so a naive text-level append would leave an image after
the sentinel and silently void it — the model would keep thinking, the token budget
would keep being eaten, and nothing would report an error. These tests pin the
placement rather than the mechanism.
"""
from __future__ import annotations

from p79.agents.mimo_vl_agent import MiMoVLAgent

append = MiMoVLAgent._append_no_think


def _last_text(content: list) -> str | None:
    if not content or not isinstance(content[-1], dict):
        return None
    return content[-1].get("text") if content[-1].get("type") == "text" else None


def test_sentinel_is_last_element_when_image_trails_the_text():
    """The reference-image layout: [text, ..., "[Current screenshot]", image]."""
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "do the task"},
        {"type": "text", "text": "[Reference image 1]"},
        {"type": "image", "image": "<ref>"},
        {"type": "text", "text": "[Current screenshot]"},
        {"type": "image", "image": "<shot>"},
    ]}]
    out = append(messages)
    content = out[0]["content"]
    assert _last_text(content) == "/no_think"
    # the sentinel must come after EVERY image, not merely after the instruction
    last_image_idx = max(i for i, c in enumerate(content) if c.get("type") == "image")
    assert len(content) - 1 > last_image_idx


def test_sentinel_is_last_element_when_image_leads():
    """The no-reference-image layout: image is insert(0)'d ahead of the text."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": "<shot>"},
        {"type": "text", "text": "do the task"},
    ]}]
    content = append(messages)[0]["content"]
    assert _last_text(content) == "/no_think"


def test_original_messages_are_not_mutated():
    """step() reuses the same list for process_vision_info; in-place edits would
    make the two views disagree with what the caller wrote."""
    content = [{"type": "text", "text": "do the task"}]
    messages = [{"role": "user", "content": content}]
    out = append(messages)
    assert len(content) == 1, "caller's content list was mutated"
    assert messages[0]["content"] is content
    assert out[0]["content"] is not content


def test_string_content_gets_the_sentinel_appended():
    messages = [{"role": "user", "content": "do the task"}]
    assert append(messages)[0]["content"] == "do the task /no_think"


def test_targets_the_last_user_turn_not_the_first():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "first"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "reply"}]},
        {"role": "user", "content": [{"type": "text", "text": "second"}]},
    ]
    out = append(messages)
    assert _last_text(out[2]["content"]) == "/no_think"
    assert _last_text(out[0]["content"]) == "first"


def test_unrecognised_shapes_are_left_alone_rather_than_guessed():
    for messages in ([], [{"role": "system", "content": "s"}],
                     [{"role": "user", "content": 42}], "not-a-list"):
        assert append(messages) is messages
