#!/usr/bin/env python3
"""Probe AWS proxy capability — tool_choice / logprobs / response_format.

Triggered 2026-05-17 to verify advisor "都搞定了" reply scope: does the
AWS API Gateway proxy now forward `tool_choice` + return `logprobs` for
B0 Qwen3-VL-235B-A22B, or did advisor only address quota+disk?

Tests 4 capabilities against the SAME endpoint+key+model used in paper-
grade B0 runs:
  T1. tool_choice="auto" + tools[] — does the response contain a
      `tool_use` block? (Plan A unblock signal.)
  T2. tool_choice={"type":"tool","name":"web_action"} (forced) — does
      the proxy honor the forced tool selection? (P79 actual usage.)
  T3. logprobs=true, top_logprobs=2 — does the response include token
      logprobs? (B-262 P0-7 cross-baseline confidence alignment.)
  T4. response_format={"type":"json_object"} — does JSON-mode work as
      an alternative to tool_choice for structured output?

Output: docs/checkpoints/probes/proxy_capability_<HHMMSS>.{json,md}
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

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
    print("ERROR: PROXY_API_KEY not set and .auth/qwen_api missing rp_ key", file=sys.stderr)
    sys.exit(1)

HEADERS = {"X-Api-Key": API_KEY, "Content-Type": "application/json"}
TIMEOUT = 60

WEB_ACTION_TOOL = {
    "name": "web_action",
    "description": "Execute a web navigation action on the current page.",
    "input_schema": {
        "type": "object",
        "properties": {
            "thought": {"type": "string"},
            "action_type": {
                "type": "string",
                "enum": ["click", "type", "scroll", "wait", "finish"],
            },
            "element_id": {"type": "integer"},
        },
        "required": ["action_type"],
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


def _post(payload: Dict[str, Any], label: str) -> Dict[str, Any]:
    try:
        resp = requests.post(BASE_URL, json=payload, headers=HEADERS, timeout=TIMEOUT)
    except requests.RequestException as exc:
        return {"label": label, "ok": False, "error": f"network: {exc}"}
    out: Dict[str, Any] = {
        "label": label,
        "status_code": resp.status_code,
        "ok": resp.status_code == 200,
    }
    try:
        out["body"] = resp.json()
    except json.JSONDecodeError:
        out["body_text"] = resp.text[:1000]
    return out


def t1_tool_auto() -> Dict[str, Any]:
    r = _post(
        {
            "model": MODEL_NAME,
            "messages": PROMPT_MSGS,
            "tools": [WEB_ACTION_TOOL],
            "tool_choice": "auto",
            "max_tokens": 512,
            "temperature": 0.0,
        },
        "T1_tool_choice_auto",
    )
    body = r.get("body", {})
    content = body.get("content", []) if isinstance(body, dict) else []
    types = [b.get("type") for b in content if isinstance(b, dict)]
    r["has_tool_use_block"] = "tool_use" in types
    r["content_block_types"] = types
    r["tool_use_input"] = next(
        (b.get("input") for b in content if isinstance(b, dict) and b.get("type") == "tool_use"),
        None,
    )
    return r


def t2_tool_forced() -> Dict[str, Any]:
    r = _post(
        {
            "model": MODEL_NAME,
            "messages": PROMPT_MSGS,
            "tools": [WEB_ACTION_TOOL],
            "tool_choice": {"type": "tool", "name": "web_action"},
            "max_tokens": 512,
            "temperature": 0.0,
        },
        "T2_tool_choice_forced",
    )
    body = r.get("body", {})
    content = body.get("content", []) if isinstance(body, dict) else []
    types = [b.get("type") for b in content if isinstance(b, dict)]
    r["has_tool_use_block"] = "tool_use" in types
    r["content_block_types"] = types
    r["tool_use_input"] = next(
        (b.get("input") for b in content if isinstance(b, dict) and b.get("type") == "tool_use"),
        None,
    )
    return r


def t3_logprobs() -> Dict[str, Any]:
    r = _post(
        {
            "model": MODEL_NAME,
            "messages": PROMPT_MSGS,
            "max_tokens": 64,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 2,
        },
        "T3_logprobs",
    )
    body = r.get("body", {})
    # Try multiple shapes — Anthropic-style vs OpenAI-style
    found = None
    if isinstance(body, dict):
        if "logprobs" in body:
            found = ("top-level", body["logprobs"])
        elif "choices" in body and body["choices"]:
            lp = body["choices"][0].get("logprobs")
            if lp:
                found = ("choices[0].logprobs", lp)
        elif "content" in body:
            for blk in body["content"]:
                if isinstance(blk, dict) and "logprobs" in blk:
                    found = ("content[].logprobs", blk["logprobs"])
                    break
    r["has_logprobs"] = found is not None
    if found:
        r["logprobs_path"] = found[0]
        r["logprobs_sample"] = (
            str(found[1])[:500] if not isinstance(found[1], (dict, list))
            else json.dumps(found[1])[:500]
        )
    return r


def t4_response_format_json() -> Dict[str, Any]:
    r = _post(
        {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Output ONLY a JSON object with keys action_type, element_id, thought. "
                        "Page: [1] heading 'results' [2] link 'cheapest kayak'. "
                        "Task: click cheapest kayak."
                    ),
                }
            ],
            "max_tokens": 256,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        },
        "T4_response_format_json",
    )
    body = r.get("body", {})
    content_raw = ""
    if isinstance(body, dict):
        if "content" in body:
            for blk in body["content"]:
                if isinstance(blk, dict) and blk.get("type") == "text":
                    content_raw = blk.get("text", "")
                    break
        elif "choices" in body and body["choices"]:
            content_raw = body["choices"][0].get("message", {}).get("content", "")
    r["raw_content"] = content_raw[:500]
    parsed_ok = False
    try:
        json.loads(content_raw)
        parsed_ok = True
    except Exception:
        pass
    r["raw_json_parses"] = parsed_ok
    return r


def main() -> int:
    print(f"[probe] endpoint={BASE_URL}")
    print(f"[probe] model={MODEL_NAME}")
    print(f"[probe] key_prefix={API_KEY[:6]}...")
    print()
    results = []
    for fn in (t1_tool_auto, t2_tool_forced, t3_logprobs, t4_response_format_json):
        print(f"[probe] running {fn.__name__}...")
        r = fn()
        results.append(r)
        verdict = "OK" if r.get("ok") else f"HTTP {r.get('status_code')}"
        print(f"  → {verdict}")
        if r.get("ok"):
            if "has_tool_use_block" in r:
                print(f"    has_tool_use_block: {r['has_tool_use_block']}  blocks: {r['content_block_types']}")
            if "has_logprobs" in r:
                print(f"    has_logprobs: {r['has_logprobs']}")
            if "raw_json_parses" in r:
                print(f"    raw_json_parses: {r['raw_json_parses']}")
        else:
            err = r.get("body") or r.get("body_text") or r.get("error")
            print(f"    error: {str(err)[:300]}")

    ts = _dt.datetime.now().strftime("%H%M%S")
    out_json = OUT_DIR / f"proxy_capability_{ts}.json"
    out_json.write_text(json.dumps({
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "endpoint": BASE_URL,
        "model": MODEL_NAME,
        "results": results,
    }, indent=2))
    print(f"\n[probe] full JSON → {out_json}")

    # Verdict summary
    t1 = next(r for r in results if r["label"] == "T1_tool_choice_auto")
    t2 = next(r for r in results if r["label"] == "T2_tool_choice_forced")
    t3 = next(r for r in results if r["label"] == "T3_logprobs")
    t4 = next(r for r in results if r["label"] == "T4_response_format_json")
    print("\n=== VERDICT ===")
    print(f"  tool_choice auto       : {'✓ tool_use returned' if t1.get('has_tool_use_block') else '✗ no tool_use block'}")
    print(f"  tool_choice forced     : {'✓ tool_use returned' if t2.get('has_tool_use_block') else '✗ no tool_use block'}")
    print(f"  logprobs               : {'✓ field present' if t3.get('has_logprobs') else '✗ not returned'}")
    print(f"  response_format=json   : {'✓ raw content valid JSON' if t4.get('raw_json_parses') else '✗ no / bad JSON'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
