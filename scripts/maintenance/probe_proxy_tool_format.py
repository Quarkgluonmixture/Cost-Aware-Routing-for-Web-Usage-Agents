#!/usr/bin/env python3
"""Follow-up probe: find the tool_choice format the proxy accepts.

First probe (probe_proxy_capability.py) returned `Invalid 'tools':
missing field 'type'` for Anthropic-style tools on Qwen3-VL model.
This script tries 3 format variants to find the one the proxy validates.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_key() -> str:
    p = REPO_ROOT / ".auth" / "qwen_api"
    if not p.exists():
        return ""
    for line in p.read_text().splitlines():
        line = line.strip()
        if line.startswith("rp_"):
            return line
    return ""


API_KEY = os.environ.get("PROXY_API_KEY", "") or _load_key()
URL = "https://i5xpracyci.execute-api.eu-west-2.amazonaws.com/model-api/invoke"
MODEL = "qwen.qwen3-vl-235b-a22b"
HEAD = {"X-Api-Key": API_KEY, "Content-Type": "application/json"}

MSGS = [{"role": "user", "content": "Click the cheapest blue kayak: [1] $320 inflatable [2] $850 sea kayak"}]

VARIANTS = {
    "A_openai_function": {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "web_action",
                    "description": "Execute a web navigation action.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action_type": {"type": "string", "enum": ["click", "type", "scroll"]},
                            "element_id": {"type": "integer"},
                        },
                        "required": ["action_type"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
    },
    "B_anthropic_with_type_custom": {
        "tools": [
            {
                "type": "custom",
                "name": "web_action",
                "description": "Execute a web navigation action.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "action_type": {"type": "string", "enum": ["click", "type", "scroll"]},
                        "element_id": {"type": "integer"},
                    },
                    "required": ["action_type"],
                },
            }
        ],
        "tool_choice": "auto",
    },
    "C_openai_function_forced": {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "web_action",
                    "description": "Execute a web navigation action.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action_type": {"type": "string", "enum": ["click", "type", "scroll"]},
                            "element_id": {"type": "integer"},
                        },
                        "required": ["action_type"],
                    },
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "web_action"}},
    },
}


def run_variant(name: str, extra: dict) -> dict:
    payload = {
        "model": MODEL,
        "messages": MSGS,
        "max_tokens": 256,
        "temperature": 0.0,
        **extra,
    }
    try:
        r = requests.post(URL, json=payload, headers=HEAD, timeout=60)
    except Exception as e:
        return {"variant": name, "ok": False, "error": str(e)}
    out = {"variant": name, "status": r.status_code, "ok": r.status_code == 200}
    try:
        out["body"] = r.json()
    except Exception:
        out["body_text"] = r.text[:500]
    return out


def main():
    if not API_KEY:
        print("ERROR: no PROXY_API_KEY / rp_ key", file=sys.stderr)
        sys.exit(1)

    print(f"endpoint={URL} model={MODEL}\n")
    for name, extra in VARIANTS.items():
        print(f"--- {name} ---")
        r = run_variant(name, extra)
        if r.get("ok"):
            body = r.get("body", {})
            # Inspect for tool calls in known shapes
            content = body.get("content")
            has_tool_call = False
            tool_info = None
            # Shape 1: content is list with tool_use blocks (Anthropic)
            if isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict) and blk.get("type") in ("tool_use", "function_call"):
                        has_tool_call = True
                        tool_info = blk
                        break
            # Shape 2: choices[0].message.tool_calls (OpenAI)
            if not has_tool_call and "choices" in body:
                msg = body["choices"][0].get("message", {})
                if msg.get("tool_calls"):
                    has_tool_call = True
                    tool_info = msg["tool_calls"][0]
            print(f"  HTTP 200  has_tool_call={has_tool_call}")
            print(f"  content (head): {str(content)[:300]}")
            if has_tool_call:
                print(f"  tool_info: {json.dumps(tool_info)[:400]}")
            print(f"  body keys: {list(body.keys())}")
        else:
            err = r.get("body") or r.get("body_text") or r.get("error")
            print(f"  HTTP {r.get('status')}  err: {str(err)[:400]}")
        print()


if __name__ == "__main__":
    main()
