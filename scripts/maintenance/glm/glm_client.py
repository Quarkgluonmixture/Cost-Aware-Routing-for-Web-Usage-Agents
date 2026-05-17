"""GLM API client — shared helpers for cron sidecars + standalone operator scripts.

B-855 (A1.15b Chunk δ P1-6, 2026-05-17): Extracted from
`glm_diagnosis_sidecar.py` (1996 LOC) to break tight coupling. Pre-fix,
`glm_playbook_refresh.py:37` had to import 2 helpers (`_load_glm_config`
+ `_call_glm_chat`) from glm_diagnosis_sidecar — pulling in a massive
module just to call the GLM API. `glm_batch_digest.py:38-53` did the
same via `importlib.spec_from_file_location` boilerplate.

This module owns the GLM HTTP transport + auth + JSON extraction layer
with NO project-specific dependencies (only stdlib). Callers import 5
public helpers:

- `load_glm_config(path)` — parse .auth/glm 3-line config (endpoint / model / api_key)
- `is_vision_model(model)` — heuristic for vision-model variants of GLM
- `candidate_glm_urls(endpoint)` — endpoint URL normalization
- `call_glm_chat(glm_cfg, messages, timeout_s=120)` — POST /chat/completions
  with thinking-model `reasoning_content` fallback handling
- `extract_balanced_json(text)` — outermost balanced `{...}` substring
  extraction (B-847 pattern, replaces broken `rfind`-based slicing for
  nested JSON outputs)

Underscored aliases (`_load_glm_config`, etc.) are kept for back-compat
with existing call sites that imported the private names.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def load_glm_config(cfg_path: Path) -> Dict[str, str]:
    """Parse `.auth/glm` 3-line config: endpoint / model / api_key.

    Comment lines (starting with `#`) and blank lines ignored.
    Raises ValueError if fewer than 3 non-comment lines present.
    """
    lines: List[str] = []
    for raw in cfg_path.read_text(encoding="utf-8").splitlines():
        t = raw.strip()
        if not t or t.startswith("#"):
            continue
        lines.append(t)
    if len(lines) < 3:
        raise ValueError(
            f"GLM config invalid: need 3 lines (endpoint/model/api_key), got {len(lines)}"
        )
    return {"endpoint": lines[0], "model": lines[1], "api_key": lines[2]}


def candidate_glm_urls(endpoint: str) -> List[str]:
    """Normalize GLM endpoint URL — append `/chat/completions` if missing."""
    ep = endpoint.rstrip("/")
    if ep.endswith("/chat/completions"):
        return [ep]
    return [f"{ep}/chat/completions"]


def is_vision_model(model: str) -> bool:
    """Heuristic for GLM vision-model variants (4v / 4.6v / 5v).

    Vision models cap `max_tokens` at 16K (smaller than text-only 131K)
    because image tokens consume the budget aggressively.
    """
    m = model.lower()
    return "4v" in m or "4.6v" in m or "5v" in m


def extract_balanced_json(text: str) -> Optional[str]:
    """Extract first balanced JSON object substring from `text`.

    Walks forward from first `{` tracking brace depth + string state,
    returns substring `text[start:end+1]` covering the outermost matched
    `{...}` pair. Returns None if no balanced object found.

    B-847 (A1.15b Chunk β P1-9): replaces `rfind("{") / rfind("}")`
    extraction in `_call_glm_chat`. The rfind approach returned the
    INNERMOST brace pair which silently corrupts nested JSON: for
    `{"failure_diagnosis":[{...}, {...}]}` it returned only the LAST
    inner `{...}` losing the wrapper key + all but one list element.

    Implementation notes:
    - Tracks `in_string` state via `"` (with backslash-escape handling)
      so braces inside strings don't shift depth.
    - First `{` outside any string = start. Matching `}` closing depth
      to zero = end.
    - Greedy (returns OUTERMOST balanced object first found).
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if in_string:
            if c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def call_glm_chat(
    glm_cfg: Dict[str, str],
    messages: Sequence[Dict[str, Any]],
    timeout_s: int = 120,
) -> str:
    """POST GLM chat/completions endpoint. Returns assistant text.

    Handles thinking-model (glm-4.6 / glm-5.1) output shapes:
    - Normal: `choices[0].message.content` is the answer.
    - Truncated (`finish_reason="length"`): content may be partial; prefer
      `reasoning_content` if present and extract balanced JSON via
      `extract_balanced_json`.
    - Pure thinking-model: `content=""` + answer in `reasoning_content`;
      extract balanced JSON or return raw reasoning text.

    Raises RuntimeError if response has no usable assistant content.
    """
    payload_variants = [
        {
            "model": glm_cfg["model"],
            "messages": list(messages),
            "temperature": 0.1,
            "max_tokens": 16384 if is_vision_model(glm_cfg["model"]) else 131072,
        },
    ]
    last_err: Optional[Exception] = None
    for url in candidate_glm_urls(glm_cfg["endpoint"]):
        for payload in payload_variants:
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=body,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {glm_cfg['api_key']}",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                choices = data.get("choices") or []
                if choices:
                    choice = choices[0]
                    msg_obj = choice.get("message") or {}
                    finish = choice.get("finish_reason", "")
                    msg = msg_obj.get("content")
                    reasoning = msg_obj.get("reasoning_content")
                    if isinstance(msg, str) and msg.strip():
                        if finish == "length" and isinstance(reasoning, str) and reasoning.strip():
                            # content truncated — try reasoning balanced-JSON first
                            balanced = extract_balanced_json(reasoning)
                            if balanced is not None:
                                return balanced
                        return msg.strip()
                    # Thinking models with content="" — fall through to reasoning
                    if isinstance(reasoning, str) and reasoning.strip():
                        balanced = extract_balanced_json(reasoning)
                        if balanced is not None:
                            return balanced
                        return reasoning.strip()
                text = data.get("output_text") or data.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()
                last_err = RuntimeError("response has no assistant content")
            except Exception as e:  # noqa: BLE001
                last_err = e
    raise RuntimeError(f"GLM request failed: {last_err}")


# Back-compat underscored aliases — existing callers import these names.
_load_glm_config = load_glm_config
_candidate_glm_urls = candidate_glm_urls
_is_vision_model = is_vision_model
_extract_balanced_json = extract_balanced_json
_call_glm_chat = call_glm_chat
