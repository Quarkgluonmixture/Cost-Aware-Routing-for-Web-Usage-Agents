#!/usr/bin/env python3
"""B-991 (/stress A1.2-followup P0-1, 2026-05-17): paper-grade probe artifact.

v1 (probe_proxy_capability.py) only saved FAILED Anthropic-format tool calls
(HTTP 400) — the OpenAI-format success was demonstrated in
probe_proxy_tool_format.py but went to stdout, not saved JSON. That left the
single saved ground truth (proxy_capability_230807.json) supporting the
OPPOSITE of the migration claim. v2 saves the full 4-variant matrix with
request/response hashes so paper-grade reviewer can independently verify the
proxy capability claim from immutable artifact.

Variants:
  V1. baseline_no_structured     — plain text, no tools, no logprobs
  V2. tool_choice_auto_openai    — OpenAI tools format, tool_choice="auto"
  V3. tool_choice_forced_openai  — OpenAI tools format, forced web_action
  V4. logprobs_only              — logprobs=True, top_logprobs=2, no tools
  V5. tool_forced_plus_logprobs  — both enabled together (production target)
  V6. response_format_json       — response_format json_object alternative

Output: docs/checkpoints/probes/proxy_capability_v2_<HHMMSS>.json
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "docs/checkpoints/probes"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_proxy_key_from_auth() -> str:
    auth_path = REPO_ROOT / ".auth" / "qwen_api"
    if not auth_path.exists():
        return ""
    for line in auth_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("rp_"):
            return line
    return ""


API_KEY = os.environ.get("PROXY_API_KEY", "") or _load_proxy_key_from_auth()
BASE_URL = os.environ.get(
    "PROXY_API_BASE",
    "https://i5xpracyci.execute-api.eu-west-2.amazonaws.com/model-api/invoke",
)
MODEL_NAME = os.environ.get("PROXY_MODEL_NAME", "qwen.qwen3-vl-235b-a22b")

if not API_KEY:
    print("ERROR: PROXY_API_KEY unset and .auth/qwen_api missing rp_ key", file=sys.stderr)
    sys.exit(1)

HEADERS = {"X-Api-Key": API_KEY, "Content-Type": "application/json"}
TIMEOUT = 60

WEB_ACTION_TOOL_OPENAI = {
    "type": "function",
    "function": {
        "name": "web_action",
        "description": "Execute a web navigation action on the current page.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {"type": "string", "description": "Brief reasoning."},
                "action_type": {
                    "type": "string",
                    "enum": ["click", "type", "scroll", "wait", "back", "forward",
                             "finish", "select_option", "tab_focus"],
                },
                "element_id": {"type": "integer"},
                "text": {"type": "string"},
                "scroll_direction": {"type": "string", "enum": ["up", "down"]},
                "answer": {"type": "string"},
            },
            "required": ["action_type", "thought"],
        },
    },
}

PROMPT_MSGS = [
    {
        "role": "user",
        "content": (
            "You are a web agent. Page elements:\n"
            "[1] heading 'Search results'\n"
            "[2] link '$320 Blue Inflatable Kayak'\n"
            "[3] link '$850 Blue Sea Kayak'\n"
            "Task: Click the cheapest blue kayak."
        ),
    }
]


def _hash_payload(p: Dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(p, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def _hash_body(b: Any) -> str:
    return hashlib.sha256(
        json.dumps(b, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()[:16]


def _post(payload: Dict[str, Any], label: str) -> Dict[str, Any]:
    try:
        resp = requests.post(BASE_URL, json=payload, headers=HEADERS, timeout=TIMEOUT)
    except requests.RequestException as exc:
        return {"label": label, "ok": False, "error": f"network: {exc}",
                "request_hash": _hash_payload(payload), "request_payload": payload}
    out: Dict[str, Any] = {
        "label": label,
        "status_code": resp.status_code,
        "ok": resp.status_code == 200,
        "request_hash": _hash_payload(payload),
        "request_payload": payload,
        "elapsed_ms": int(resp.elapsed.total_seconds() * 1000),
    }
    try:
        body = resp.json()
        out["body"] = body
        out["response_hash"] = _hash_body(body)
    except json.JSONDecodeError:
        out["body_text"] = resp.text[:1500]
    return out


def _analyze(r: Dict[str, Any]) -> Dict[str, Any]:
    """Extract verdict-relevant fields from response body."""
    analysis: Dict[str, Any] = {}
    if not r.get("ok"):
        return analysis
    body = r.get("body", {})
    # Tool calls — top-level (proxy's hybrid shape)
    if isinstance(body, dict):
        tc = body.get("tool_calls")
        if tc:
            analysis["has_tool_call"] = True
            analysis["tool_call_count"] = len(tc)
            first = tc[0]
            if isinstance(first, dict):
                fn = first.get("function", {})
                args_raw = fn.get("arguments", "")
                analysis["tool_function_name"] = fn.get("name")
                analysis["tool_arguments_raw"] = args_raw[:500]
                try:
                    parsed = json.loads(args_raw)
                    analysis["tool_arguments_parsed_ok"] = True
                    analysis["tool_arguments_keys"] = sorted(parsed.keys())
                except json.JSONDecodeError:
                    analysis["tool_arguments_parsed_ok"] = False
        else:
            analysis["has_tool_call"] = False
        # Logprobs — top-level (proxy's hybrid shape)
        lp = body.get("logprobs")
        if lp and isinstance(lp, dict) and "content" in lp:
            tokens = lp["content"]
            analysis["has_logprobs"] = True
            analysis["logprob_token_count"] = len(tokens)
            if tokens and isinstance(tokens[0], dict):
                analysis["logprob_first_token"] = tokens[0].get("token")
                analysis["logprob_first_value"] = tokens[0].get("logprob")
                analysis["logprob_top_n"] = len(tokens[0].get("top_logprobs", []))
        else:
            analysis["has_logprobs"] = False
        # Cost
        usage = body.get("usage", {})
        if isinstance(usage, dict):
            analysis["input_tokens"] = usage.get("inputTokens") or usage.get("input_tokens")
            analysis["output_tokens"] = usage.get("outputTokens") or usage.get("output_tokens")
            analysis["cost_usd"] = usage.get("cost")
        # Content (free text path)
        c = body.get("content")
        if isinstance(c, str):
            analysis["content_text_len"] = len(c)
            analysis["content_text_head"] = c[:200]
    return analysis


def v1_baseline() -> Dict[str, Any]:
    return _post({
        "model": MODEL_NAME, "messages": PROMPT_MSGS,
        "max_tokens": 256, "temperature": 0.0,
    }, "V1_baseline_no_structured")


def v2_tool_auto() -> Dict[str, Any]:
    return _post({
        "model": MODEL_NAME, "messages": PROMPT_MSGS,
        "max_tokens": 256, "temperature": 0.0,
        "tools": [WEB_ACTION_TOOL_OPENAI], "tool_choice": "auto",
    }, "V2_tool_choice_auto_openai")


def v3_tool_forced() -> Dict[str, Any]:
    return _post({
        "model": MODEL_NAME, "messages": PROMPT_MSGS,
        "max_tokens": 256, "temperature": 0.0,
        "tools": [WEB_ACTION_TOOL_OPENAI],
        "tool_choice": {"type": "function", "function": {"name": "web_action"}},
    }, "V3_tool_choice_forced_openai")


def v4_logprobs_only() -> Dict[str, Any]:
    return _post({
        "model": MODEL_NAME, "messages": PROMPT_MSGS,
        "max_tokens": 64, "temperature": 0.0,
        "logprobs": True, "top_logprobs": 2,
    }, "V4_logprobs_only")


def v5_tool_forced_plus_logprobs() -> Dict[str, Any]:
    return _post({
        "model": MODEL_NAME, "messages": PROMPT_MSGS,
        "max_tokens": 256, "temperature": 0.0,
        "tools": [WEB_ACTION_TOOL_OPENAI],
        "tool_choice": {"type": "function", "function": {"name": "web_action"}},
        "logprobs": True, "top_logprobs": 2,
    }, "V5_tool_forced_plus_logprobs")


def v6_response_format_json() -> Dict[str, Any]:
    msgs = [
        {
            "role": "user",
            "content": (
                "You MUST output ONLY a JSON object matching schema "
                "{action_type:str, element_id:int, thought:str}. "
                "No prose, no markdown, no preamble.\n\n"
                "Page: [1] heading 'results' [2] link 'cheapest kayak'.\n"
                "Task: click cheapest kayak."
            ),
        }
    ]
    return _post({
        "model": MODEL_NAME, "messages": msgs,
        "max_tokens": 256, "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }, "V6_response_format_json")


def main() -> int:
    print(f"[probe v2] endpoint={BASE_URL}")
    print(f"[probe v2] model={MODEL_NAME}")
    print(f"[probe v2] key_prefix={API_KEY[:6]}...\n")

    variants = [
        v1_baseline, v2_tool_auto, v3_tool_forced,
        v4_logprobs_only, v5_tool_forced_plus_logprobs, v6_response_format_json,
    ]
    results: List[Dict[str, Any]] = []
    for fn in variants:
        print(f"[probe v2] {fn.__name__}...")
        r = fn()
        r["analysis"] = _analyze(r)
        results.append(r)
        v = "OK" if r.get("ok") else f"HTTP {r.get('status_code')}"
        print(f"  → {v}  elapsed={r.get('elapsed_ms', '?')}ms")
        a = r["analysis"]
        if r.get("ok"):
            print(f"    has_tool_call={a.get('has_tool_call', '?')} "
                  f"has_logprobs={a.get('has_logprobs', '?')} "
                  f"cost=${a.get('cost_usd', '?')}")
            if a.get("has_tool_call"):
                print(f"    tool_arguments_keys={a.get('tool_arguments_keys')}")
                print(f"    tool_arguments_parsed_ok={a.get('tool_arguments_parsed_ok')}")
        else:
            err = r.get("body") or r.get("body_text") or r.get("error")
            print(f"    error={str(err)[:200]}")

    ts = _dt.datetime.now().strftime("%H%M%S")
    out_path = OUT_DIR / f"proxy_capability_v2_{ts}.json"
    out_path.write_text(json.dumps({
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "endpoint": BASE_URL,
        "model": MODEL_NAME,
        "key_prefix": API_KEY[:6],
        "purpose": "B-991 paper-grade probe artifact — full success+fail matrix for B0 migration audit trail",
        "variants_tested": 6,
        "results": results,
    }, indent=2, default=str))
    print(f"\n[probe v2] full JSON → {out_path}")

    # Verdict
    by_label = {r["label"]: r for r in results}
    print("\n=== VERDICT ===")
    v2 = by_label["V2_tool_choice_auto_openai"].get("analysis", {})
    v3 = by_label["V3_tool_choice_forced_openai"].get("analysis", {})
    v4 = by_label["V4_logprobs_only"].get("analysis", {})
    v5 = by_label["V5_tool_forced_plus_logprobs"].get("analysis", {})
    v6 = by_label["V6_response_format_json"].get("analysis", {})
    print(f"  V2 tool_auto:       tool_call={v2.get('has_tool_call')}  parsed_ok={v2.get('tool_arguments_parsed_ok')}")
    print(f"  V3 tool_forced:     tool_call={v3.get('has_tool_call')}  parsed_ok={v3.get('tool_arguments_parsed_ok')}")
    print(f"  V4 logprobs:        has_logprobs={v4.get('has_logprobs')}  top_n={v4.get('logprob_top_n')}")
    print(f"  V5 forced+logprobs: tool_call={v5.get('has_tool_call')}  has_logprobs={v5.get('has_logprobs')}")
    print(f"  V6 response_format: content_len={v6.get('content_text_len')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
