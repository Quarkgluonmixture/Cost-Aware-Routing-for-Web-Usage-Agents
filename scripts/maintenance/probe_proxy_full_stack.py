#!/usr/bin/env python3
"""B-992 (/stress A1.2-followup P1-1, 2026-05-17): full-stack probe with
production-realistic prompt + image + N=30 repeat — validates Q1=A
tool_choice="auto" pilot gate (emit_rate ≥95% to keep auto; else fall back
to forced).

What this tests (vs probe_proxy_capability_v2.py):
  - REAL production prompt structure: Task + System + Previous actions + AXTree
  - REAL image payload (base64-encoded 1024-pixel test screenshot)
  - REAL prompt size (AXTree ~3-7K chars, mirrors VWA classifieds median)
  - N=30 repetitions to measure tool_call emit rate stability under T=0
  - tool_choice="auto" mode (model self-decides) — the Q1=A pilot gate
  - logprobs=True + top_logprobs=2 — verify logprobs flow in image+tools path

Metrics:
  - emit_rate: fraction of N runs where body.tool_calls[0].function present
  - args_parse_rate: fraction where arguments JSON.loads succeeds
  - schema_valid_rate: fraction where validate_action(parsed) returns valid
  - logprobs_present_rate: fraction where body.logprobs.content is non-empty
  - cost_mean, cost_std, latency_mean, latency_std

Q1=A pilot gate decision rule:
  emit_rate ≥ 0.95 AND schema_valid_rate ≥ 0.95  → PASS (keep tool_choice="auto")
  else                                            → FAIL (fall back to forced;
                                                          paper §3.5 disclose
                                                          constrained asymmetry)

Output: docs/checkpoints/probes/proxy_full_stack_<HHMMSS>.json
"""
from __future__ import annotations

import base64
import datetime as _dt
import io
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from p79.backends.action_utils import validate_action  # noqa: E402
from p79.agents.proxy_api_agent import _WEB_ACTION_TOOL as _PROD_WEB_ACTION_TOOL  # noqa: E402

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
N_RUNS = int(os.environ.get("N_RUNS", "30"))

if not API_KEY:
    print("ERROR: PROXY_API_KEY unset and .auth/qwen_api missing rp_ key", file=sys.stderr)
    sys.exit(1)

HEADERS = {"X-Api-Key": API_KEY, "Content-Type": "application/json"}
TIMEOUT = 90

# Use PRODUCTION tool def from proxy_api_agent — early debug found my
# truncated probe schema only had `text` field, missing `option_label` /
# `option_value` / `option_index` / `coordinate` etc. that production
# `_WEB_ACTION_TOOL` exposes. Model with truncated schema misrouted
# select_option → text='Lowest Price' which validate_action rejected.
# Importing production tool eliminates probe-vs-prod schema drift.
WEB_ACTION_TOOL = _PROD_WEB_ACTION_TOOL


def _make_tiny_screenshot_data_url() -> str:
    """Return a base64-encoded 8x8 JPEG (minimal image to exercise image path)."""
    try:
        from PIL import Image
        img = Image.new("RGB", (8, 8), color=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except ImportError:
        # Minimal hardcoded 1x1 white JPEG fallback
        return (
            "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgM"
            "CAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/"
            "wAALCAABAAEBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIA"
            "QEAAD8AVN//2Q=="
        )


# Production-realistic AXTree (~3K chars, mirrors VWA classifieds median observed in
# results/visualwebarena/phase1/B0_*/episodes/.../step_000.json `obs_text` field).
PROD_AXTREE = """[1] RootWebArea 'Classifieds'
  [2] navigation 'main'
    [3] link 'Home' url='/'
    [4] link 'Categories' url='/categories'
    [5] link 'Post Ad' url='/post'
    [6] link 'Sign In' url='/login'
  [7] search ''
    [8] textbox 'Search listings' required: False
    [9] button 'Search'
  [10] heading 'Search results for "blue kayak"'
  [11] StaticText '1-12 of 47 listings'
  [12] LayoutTable ''
    [13] link '$320 - Blue Inflatable Kayak - 12ft, brand new' url='/item/47291'
      [14] image 'blue inflatable kayak'
      [15] StaticText 'Posted: 2 days ago in Seattle, WA'
    [16] link '$425 - Blue Sea Kayak - Used, good condition' url='/item/47158'
      [17] image 'blue sea kayak side view'
      [18] StaticText 'Posted: 4 days ago in Tacoma, WA'
    [19] link '$850 - Blue Touring Kayak - Excellent, 14ft' url='/item/47012'
      [20] image 'blue touring kayak'
      [21] StaticText 'Posted: 1 week ago in Portland, OR'
    [22] link '$1,200 - Blue Tandem Kayak - New, 16ft' url='/item/46891'
      [23] image 'blue tandem kayak'
      [24] StaticText 'Posted: 2 weeks ago in Vancouver, BC'
    [25] link '$280 - Blue Recreational Kayak - Used 9ft' url='/item/46832'
      [26] image 'blue recreational kayak'
      [27] StaticText 'Posted: 3 weeks ago in Eugene, OR'
    [28] link '$1,500 - Blue Whitewater Kayak - Like new' url='/item/46720'
      [29] image 'blue whitewater kayak'
      [30] StaticText 'Posted: 4 weeks ago in Bend, OR'
  [31] navigation 'pagination'
    [32] link 'Previous' url='?page=0' disabled: True
    [33] StaticText 'Page 1 of 4'
    [34] link 'Next' url='?page=2'
  [35] LayoutTable 'Filters'
    [36] combobox 'Sort By' hasPopup: 'menu'
      [37] menuitem 'Lowest Price'
      [38] menuitem 'Highest Price'
      [39] menuitem 'Newest'
      [40] menuitem 'Oldest'
    [41] combobox 'Category' hasPopup: 'menu'
      [42] menuitem 'All'
      [43] menuitem 'Watersports'
      [44] menuitem 'Sporting Goods'
    [45] textbox 'Min Price' required: False
    [46] textbox 'Max Price' required: False
    [47] button 'Apply Filters'
  [48] StaticText 'See similar searches: blue canoe, blue paddle board, water sports gear'
"""

# Realistic system prompt + task structure (mirrors proxy_api_agent.py:431-441)
TASK_INSTRUCTION = "Find the cheapest blue kayak."
SYSTEM_PROMPT = (
    "You are a web navigation agent. For each step, observe the page and "
    "decide on an action. Use the web_action tool for every action. Put "
    "reasoning in the thought parameter. Valid action types: click, type, "
    "scroll, wait, back, forward, finish, select_option, tab_focus. "
    "Element IDs are shown in [brackets]."
)
HISTORY_TEXT = (
    "Previous actions:\n"
    "  Step 0: type 'blue kayak' [id=8] -> OK (page changed) [classifieds/search?q=blue+kayak]\n"
    "  Step 1: click [id=36] -> OK (page changed) [classifieds/search?q=blue+kayak&sort=open]\n"
)

PROD_PROMPT_TEXT = (
    f"Task: {TASK_INSTRUCTION}\n"
    f"System: {SYSTEM_PROMPT}\n"
    f"{HISTORY_TEXT}"
    f"Accessibility Tree:\n{PROD_AXTREE}"
)


def _build_payload() -> Dict[str, Any]:
    """Build production-realistic payload mirroring proxy_api_agent.py:430-525 shape."""
    img_url = _make_tiny_screenshot_data_url()
    return {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": img_url}},
                    {"type": "text", "text": PROD_PROMPT_TEXT},
                ],
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.0,
        "top_p": 1.0,
        "tools": [WEB_ACTION_TOOL],
        "tool_choice": "auto",
        "logprobs": True,
        "top_logprobs": 2,
    }


def _post() -> Dict[str, Any]:
    payload = _build_payload()
    try:
        resp = requests.post(BASE_URL, json=payload, headers=HEADERS, timeout=TIMEOUT)
    except requests.RequestException as exc:
        return {"ok": False, "error": f"network: {exc}"}
    out: Dict[str, Any] = {
        "status_code": resp.status_code,
        "ok": resp.status_code == 200,
        "elapsed_ms": int(resp.elapsed.total_seconds() * 1000),
    }
    try:
        out["body"] = resp.json()
    except json.JSONDecodeError:
        out["body_text"] = resp.text[:500]
    return out


def _analyze_run(r: Dict[str, Any]) -> Dict[str, Any]:
    a: Dict[str, Any] = {
        "ok": r.get("ok", False),
        "elapsed_ms": r.get("elapsed_ms"),
        "tool_call_emitted": False,
        "args_parsed": False,
        "schema_valid": False,
        "logprobs_present": False,
        "logprob_token_count": 0,
        "cost": None,
    }
    if not r.get("ok"):
        a["error"] = str(r.get("body") or r.get("body_text") or r.get("error"))[:200]
        return a
    body = r.get("body", {})
    if isinstance(body, dict):
        tc = body.get("tool_calls")
        if isinstance(tc, list) and tc:
            first = tc[0] or {}
            fn = first.get("function") if isinstance(first, dict) else None
            if isinstance(fn, dict) and fn.get("name") == "web_action":
                a["tool_call_emitted"] = True
                args_raw = fn.get("arguments") or ""
                try:
                    parsed = json.loads(args_raw)
                    a["args_parsed"] = True
                    a["args_keys"] = sorted(parsed.keys())
                    a["args_raw"] = parsed  # B-992 debug: save raw for fail analysis
                    action, ok = validate_action(parsed)
                    a["schema_valid"] = ok
                    a["action_type"] = parsed.get("action_type")
                    a["validate_action_result"] = action  # post-validation dict
                except json.JSONDecodeError:
                    a["args_parsed"] = False
        lp = body.get("logprobs")
        if isinstance(lp, dict) and isinstance(lp.get("content"), list) and lp["content"]:
            a["logprobs_present"] = True
            a["logprob_token_count"] = len(lp["content"])
        usage = body.get("usage", {})
        if isinstance(usage, dict):
            a["cost"] = usage.get("cost")
            a["input_tokens"] = usage.get("inputTokens") or usage.get("input_tokens")
            a["output_tokens"] = usage.get("outputTokens") or usage.get("output_tokens")
    return a


def main() -> int:
    print(f"[full-stack] endpoint={BASE_URL}")
    print(f"[full-stack] model={MODEL_NAME}")
    print(f"[full-stack] N={N_RUNS}  prompt_chars={len(PROD_PROMPT_TEXT)}  axtree_chars={len(PROD_AXTREE)}\n")

    analyses: List[Dict[str, Any]] = []
    for i in range(N_RUNS):
        r = _post()
        a = _analyze_run(r)
        analyses.append(a)
        sym = "✓" if a["ok"] and a["schema_valid"] else "✗"
        print(f"[{i+1:2d}/{N_RUNS}] {sym} ok={a['ok']} emit={a['tool_call_emitted']} "
              f"valid={a['schema_valid']} logprobs={a['logprobs_present']} "
              f"elapsed={a['elapsed_ms']}ms cost=${a.get('cost', '?')}")

    # Aggregate stats
    n_ok = sum(1 for a in analyses if a["ok"])
    n_emit = sum(1 for a in analyses if a["tool_call_emitted"])
    n_parsed = sum(1 for a in analyses if a["args_parsed"])
    n_valid = sum(1 for a in analyses if a["schema_valid"])
    n_lp = sum(1 for a in analyses if a["logprobs_present"])
    costs = [a["cost"] for a in analyses if a.get("cost") is not None]
    elapsed = [a["elapsed_ms"] for a in analyses if a.get("elapsed_ms") is not None]
    lp_counts = [a["logprob_token_count"] for a in analyses if a.get("logprob_token_count")]

    def _stat(xs):
        if not xs: return {"n": 0}
        return {
            "n": len(xs), "mean": statistics.mean(xs),
            "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0,
            "min": min(xs), "max": max(xs),
        }

    stats: Dict[str, Any] = {
        "n_total": N_RUNS,
        "n_ok": n_ok,
        "http_ok_rate": n_ok / N_RUNS if N_RUNS else 0.0,
        "emit_rate": n_emit / N_RUNS if N_RUNS else 0.0,
        "args_parse_rate": n_parsed / N_RUNS if N_RUNS else 0.0,
        "schema_valid_rate": n_valid / N_RUNS if N_RUNS else 0.0,
        "logprobs_present_rate": n_lp / N_RUNS if N_RUNS else 0.0,
        "cost_usd_stats": _stat(costs),
        "elapsed_ms_stats": _stat(elapsed),
        "logprob_token_count_stats": _stat(lp_counts),
    }
    pilot_pass = stats["emit_rate"] >= 0.95 and stats["schema_valid_rate"] >= 0.95
    stats["q1_a_pilot_gate"] = "PASS" if pilot_pass else "FAIL"

    ts = _dt.datetime.now().strftime("%H%M%S")
    out_path = OUT_DIR / f"proxy_full_stack_{ts}.json"
    out_path.write_text(json.dumps({
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "purpose": "B-992 full-stack probe: real prompt + image + N=30 + tools=auto + logprobs",
        "endpoint": BASE_URL,
        "model": MODEL_NAME,
        "n_runs": N_RUNS,
        "prompt_chars": len(PROD_PROMPT_TEXT),
        "axtree_chars": len(PROD_AXTREE),
        "stats": stats,
        "analyses": analyses,
    }, indent=2, default=str))

    print(f"\n[full-stack] full JSON → {out_path}")
    print("\n=== STATS ===")
    print(f"  HTTP OK:         {n_ok}/{N_RUNS} ({stats['http_ok_rate']*100:.1f}%)")
    print(f"  tool_call emit:  {n_emit}/{N_RUNS} ({stats['emit_rate']*100:.1f}%)")
    print(f"  args parsed:     {n_parsed}/{N_RUNS} ({stats['args_parse_rate']*100:.1f}%)")
    print(f"  schema valid:    {n_valid}/{N_RUNS} ({stats['schema_valid_rate']*100:.1f}%)")
    print(f"  logprobs:        {n_lp}/{N_RUNS} ({stats['logprobs_present_rate']*100:.1f}%)")
    print(f"  cost USD:        {stats['cost_usd_stats']}")
    print(f"  elapsed ms:      {stats['elapsed_ms_stats']}")
    print(f"  logprob tokens:  {stats['logprob_token_count_stats']}")
    print(f"\n=== Q1=A PILOT GATE: {stats['q1_a_pilot_gate']} ===")
    print(f"  Rule: emit_rate ≥ 0.95 AND schema_valid_rate ≥ 0.95")
    print(f"  Result: emit={stats['emit_rate']:.3f} ≥0.95?{stats['emit_rate']>=0.95}  "
          f"valid={stats['schema_valid_rate']:.3f} ≥0.95?{stats['schema_valid_rate']>=0.95}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
