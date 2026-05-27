#!/usr/bin/env python3
"""B-1867 / §302 cross-provider control: official DashScope API replay.

Same N=5 same-payload replay as probe_replay_step0_n5.py BUT against
Alibaba DashScope international (官方 Qwen API, OpenAI-compatible /chat/
completions) instead of AWS Bedrock proxy.

Discriminates:
  - If DashScope ALSO 5/5 unique  → Qwen3-VL-235B-A22B model-wide phenomenon
                                    (any provider has this floor)
  - If DashScope 5/5 identical    → AWS Bedrock implementation-specific
                                    (Bedrock proxy adds nondet on top)
  - Mixed                          → modality / API path partial culprit

Output: docs/checkpoints/probes/replay_step0_n5_official_<HHMMSS>.json
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from p79.agents.proxy_api_agent import _WEB_ACTION_TOOL  # noqa: E402
from p79.agents._shared_vl_utils import make_vision_prompt  # noqa: E402


def _load_key() -> str:
    k = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
    if k:
        return k
    auth = REPO_ROOT / ".auth/qwen_api_official"
    if auth.exists():
        for line in auth.read_text().splitlines():
            line = line.strip()
            # DashScope keys start with `sk-` (OpenAI-compatible)
            if line.startswith("sk-") or (line and not line.startswith("#")):
                return line
    return ""


API_KEY = _load_key()
BASE_URL = os.environ.get(
    "DASHSCOPE_API_BASE",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
)
MODEL_NAME = os.environ.get("DASHSCOPE_MODEL_NAME", "qwen3-vl-235b-a22b-instruct")
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}
TIMEOUT = 90

if not API_KEY:
    print("ERROR: DASHSCOPE_API_KEY unset and .auth/qwen_api_official empty/missing",
          file=sys.stderr)
    sys.exit(1)


R32024_DIR = REPO_ROOT / (
    "results/visualwebarena/phase1/"
    "B0_vision_classifieds_20260526_141916_610351680_689390_R32024/"
    "phase1_vision_router_0"
)
TASK_CFG_DIR = REPO_ROOT / "external/visualwebarena/config_files/vwa/test_classifieds"


def load_task_intent(task_id: int) -> str:
    p = TASK_CFG_DIR / f"{task_id}.json"
    if not p.exists():
        raise FileNotFoundError(f"task config not found: {p}")
    return json.loads(p.read_text()).get("intent", "")


def load_screenshot_b64(task_id: int) -> str:
    p = R32024_DIR / f"artifacts/classifieds_task_{task_id}/step_000/screenshot.png"
    if not p.exists():
        raise FileNotFoundError(f"screenshot not found: {p}")
    from PIL import Image
    img = Image.open(p).convert("RGB")
    max_size = 1024
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_payload(task_id: int) -> Dict[str, Any]:
    intent = load_task_intent(task_id)
    img_b64 = load_screenshot_b64(task_id)
    system_prompt = make_vision_prompt()
    user_text = (
        f"Task: {intent}\n"
        f"System: {system_prompt}\n"
    )
    return {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": user_text},
                ],
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.0,
        "top_p": 1.0,
        "tools": [_WEB_ACTION_TOOL],
        "tool_choice": "required",
        "logprobs": True,
        "top_logprobs": 2,
    }


def payload_hash(p: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()[:16]


def call_once(payload: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    try:
        r = requests.post(BASE_URL, json=payload, headers=HEADERS, timeout=TIMEOUT)
    except requests.RequestException as e:
        return {"ok": False, "error": f"network: {e}", "elapsed_ms": int((time.time() - t0) * 1000)}
    elapsed_ms = int((time.time() - t0) * 1000)
    out: Dict[str, Any] = {
        "status": r.status_code,
        "ok": r.status_code == 200,
        "elapsed_ms": elapsed_ms,
        "resp_headers": {
            k: v for k, v in r.headers.items()
            if k.lower() in (
                "x-request-id", "x-dashscope-trace-id", "x-dashscope-request-id",
                "x-ratelimit-remaining-requests", "x-ratelimit-remaining-tokens",
                "via", "server", "date", "content-length",
            )
        },
    }
    try:
        body = r.json()
    except json.JSONDecodeError:
        out["body_text"] = r.text[:500]
        return out
    out["resp_hash"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]
    if isinstance(body, dict):
        out["resp_model"] = body.get("model")
        out["resp_id"] = body.get("id")
        out["system_fingerprint"] = body.get("system_fingerprint")  # AUDIT GOLD (codex §4)
        out["usage"] = body.get("usage")
        choices = body.get("choices") or []
        msg = (choices[0] or {}).get("message") if choices else None
        if isinstance(msg, dict):
            # DashScope returns Qwen chat-template raw: tool call as JSON-in-text inside
            # message.content with optional </tool_call> trailing token. Fallback OpenAI
            # standard message.tool_calls (rare for DashScope but possible).
            tc = msg.get("tool_calls")
            args_obj = None
            if isinstance(tc, list) and tc:
                fn = (tc[0] or {}).get("function") if isinstance(tc[0], dict) else None
                if isinstance(fn, dict):
                    out["tool_name"] = fn.get("name")
                    args_raw = fn.get("arguments") or ""
                    out["args_hash"] = hashlib.sha256(args_raw.encode()).hexdigest()[:16]
                    out["parse_path"] = "tool_calls"
                    try:
                        args_obj = json.loads(args_raw)
                    except json.JSONDecodeError:
                        out["args_parse_fail"] = True
                        out["args_raw_snip"] = args_raw[:300]
            else:
                content = msg.get("content") or ""
                if isinstance(content, str) and content.strip():
                    # Strip Qwen internal <tool_call>...</tool_call> wrapper
                    raw = content
                    for marker in ("<tool_call>", "</tool_call>", "<|tool_call|>"):
                        raw = raw.replace(marker, "")
                    raw = raw.strip()
                    out["args_hash"] = hashlib.sha256(raw.encode()).hexdigest()[:16]
                    out["parse_path"] = "content_json_in_text"
                    try:
                        args_obj = json.loads(raw)
                    except json.JSONDecodeError:
                        # Try extracting first {...} block via brace matching
                        start = raw.find("{")
                        end = raw.rfind("}")
                        if start != -1 and end > start:
                            try:
                                args_obj = json.loads(raw[start:end+1])
                                out["parse_path"] = "content_brace_match"
                            except json.JSONDecodeError:
                                out["args_parse_fail"] = True
                                out["args_raw_snip"] = raw[:300]
                        else:
                            out["args_parse_fail"] = True
                            out["args_raw_snip"] = raw[:300]
                    out["content_raw_snip"] = content[:200]
                # reasoning_content (Qwen-specific, thinking-mode trace)
                rc = msg.get("reasoning_content")
                if isinstance(rc, str) and rc:
                    out["reasoning_snip"] = rc[:200]
            if isinstance(args_obj, dict):
                out["args"] = args_obj
                out["action_type"] = args_obj.get("action_type")
                out["element_id"] = args_obj.get("element_id")
                out["coord"] = args_obj.get("coordinate")
                out["thought_snip"] = (args_obj.get("thought") or "")[:200]
            # logprobs: DashScope puts under message.logprobs (not choices[0].logprobs)
            lp = msg.get("logprobs")
            if not isinstance(lp, dict):
                lp = (choices[0] or {}).get("logprobs")  # OpenAI standard fallback
            if isinstance(lp, dict):
                lp_content = lp.get("content") or []
                if lp_content and isinstance(lp_content[0], dict):
                    top = lp_content[0].get("top_logprobs", []) or []
                    if len(top) >= 2:
                        try:
                            out["margin_t0"] = float(top[0]["logprob"]) - float(top[1]["logprob"])
                            out["t0_tok1"] = top[0].get("token")
                            out["t0_tok2"] = top[1].get("token")
                        except (KeyError, TypeError, ValueError):
                            pass
        # Surface error body (DashScope returns {error: {code, message}} on 4xx)
        err = body.get("error")
        if err and not out["ok"]:
            out["error_body"] = err if isinstance(err, dict) else str(err)[:300]
    return out


def action_signature(call: Dict[str, Any]) -> str:
    if not call.get("ok"):
        return f"ERR/{call.get('status', 'net')}"
    at = call.get("action_type", "?")
    eid = call.get("element_id")
    coord = call.get("coord")
    target = str(eid) if eid is not None else (str(coord) if coord is not None else "?")
    return f"{at}/{target}"


def run_task(task_id: int, n: int, sleep_s: float = 0.0) -> Dict[str, Any]:
    payload = build_payload(task_id)
    p_hash = payload_hash(payload)
    intent_preview = payload["messages"][0]["content"][1]["text"].split("\n")[0][:80]
    print(f"\n[task {task_id}] payload_hash={p_hash}  {intent_preview}")
    calls: List[Dict[str, Any]] = []
    for i in range(n):
        c = call_once(payload)
        sig = action_signature(c)
        m = c.get("margin_t0", "n/a")
        rid = c.get("resp_id", "?")[:12] if c.get("resp_id") else "?"
        print(f"  [{i+1}/{n}] status={c.get('status')} ms={c.get('elapsed_ms')} action={sig} margin={m} respid={rid}")
        calls.append(c)
        if sleep_s > 0 and i < n - 1:
            time.sleep(sleep_s)
    sigs = [action_signature(c) for c in calls if c.get("ok")]
    unique = sorted(set(sigs))
    resp_hashes = sorted({c.get("resp_hash") for c in calls if c.get("ok") and c.get("resp_hash")})
    return {
        "task_id": task_id,
        "payload_hash": p_hash,
        "n_requested": n,
        "n_ok": sum(1 for c in calls if c.get("ok")),
        "unique_action_sigs": len(unique),
        "unique_action_list": unique,
        "unique_resp_hashes": len(resp_hashes),
        "calls": calls,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-ids", required=True, help="comma-sep classifieds task ids")
    ap.add_argument("--n", type=int, default=5, help="N replays per task")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between calls")
    ap.add_argument("--out-dir", default="docs/checkpoints/probes")
    ap.add_argument("--label", default="", help="optional label for output filename")
    args = ap.parse_args()

    task_ids = [int(x.strip()) for x in args.task_ids.split(",") if x.strip()]
    print(f"§302 cross-provider N=5 same-payload replay (DashScope intl)")
    print(f"  tasks={task_ids}  N={args.n}  sleep={args.sleep}s")
    print(f"  target={BASE_URL}")
    print(f"  model={MODEL_NAME}\n")

    results = []
    t_start = time.time()
    for tid in task_ids:
        try:
            r = run_task(tid, args.n, args.sleep)
            results.append(r)
        except Exception as e:
            print(f"[task {tid}] ERROR: {e}", file=sys.stderr)
            results.append({"task_id": tid, "error": str(e)})
    wall_s = int(time.time() - t_start)

    summary = {
        "ts_utc": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "purpose": "B-1867/§302 N=5 same-payload replay — DashScope intl official API (cross-provider control)",
        "provider": "dashscope-intl",
        "endpoint": BASE_URL,
        "model": MODEL_NAME,
        "n_per_task": args.n,
        "task_ids": task_ids,
        "wall_seconds": wall_s,
        "results": results,
        "stats": {
            "n_tasks_unique5": sum(1 for r in results if r.get("unique_action_sigs") == args.n),
            "n_tasks_unique1": sum(1 for r in results if r.get("unique_action_sigs") == 1),
            "n_tasks_mixed": sum(1 for r in results if 1 < (r.get("unique_action_sigs") or 0) < args.n),
            "n_tasks_error": sum(1 for r in results if r.get("error")),
        },
    }

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    label = f"_{args.label}" if args.label else ""
    hhmmss = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"replay_step0_n5_official_{hhmmss}{label}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n=== SUMMARY ===")
    print(f"  wall={wall_s}s  tasks={len(task_ids)}  N={args.n}")
    s = summary["stats"]
    print(f"  unique{args.n}/{args.n} (full diverge) : {s['n_tasks_unique5']}")
    print(f"  unique 1     (full det)      : {s['n_tasks_unique1']}")
    print(f"  mixed        (partial)       : {s['n_tasks_mixed']}")
    print(f"  errors                       : {s['n_tasks_error']}")
    print(f"\noutput: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
