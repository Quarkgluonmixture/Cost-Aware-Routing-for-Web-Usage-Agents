#!/usr/bin/env python3
"""B-37 verification probe — does temperature=0 give deterministic output?

Tests whether the proxy API (B0 235B Qwen3-VL) returns byte-identical responses
across N=5 calls with the same prompt + temperature=0 + same model. If responses
diverge, B-37 is fully manifest at the LLM API level (independent of any p79
seed propagation), and paper's reproducibility claim must be revised.

Methodology:
1. Construct a representative agent prompt (~400 tokens, AXTree-style)
2. Make 5 sequential calls with temperature=0, max_tokens=128
3. Hash + compare byte-level match across all 5 outputs
4. Report: full match / partial match / divergent

Output: docs/analysis/cross_sites/probe_b37_api_determinism.{json,md}
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "docs/analysis/cross_sites/probe_b37_api_determinism.json"
OUT_MD = ROOT / "docs/analysis/cross_sites/probe_b37_api_determinism.md"

# Read .env for API key
ENV_PATH = ROOT / ".env"
ENV_VARS = {}
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        ENV_VARS[k.strip()] = v.strip().strip("'\"")

API_KEY = ENV_VARS.get("PROXY_API_KEY") or os.environ.get("PROXY_API_KEY", "")
BASE_URL = ENV_VARS.get("PROXY_API_BASE") or os.environ.get("PROXY_API_BASE", "")
MODEL_NAME = "qwen.qwen3-vl-235b-a22b"

if not API_KEY or not BASE_URL:
    print("ERROR: PROXY_API_KEY / PROXY_API_BASE not set — cannot probe", file=sys.stderr)
    sys.exit(1)

# Representative prompt — short AXTree-style web task to mimic agent loop
PROMPT = """You are a web agent. Given the page below, output a single action as JSON.

Task: Find the cheapest blue kayak.
Current page (search results):
[1] heading "Search results for 'blue kayak'"
[2] link "Filter by category" url=...
[3] link "$1,200 - Blue Kayak 12ft - Excellent condition"
[4] link "$850 - Blue Sea Kayak - Used"
[5] link "$320 - Blue Inflatable Kayak"
[6] link "$1,500 - Blue Tandem Kayak - New"

Output JSON: {"action_type": "click", "element_id": <id>, "thought": "<short>"}"""


def call_api(prompt: str, temperature: float = 0.0, max_tokens: int = 128, seed: int | None = None) -> dict:
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if seed is not None:
        payload["seed"] = seed
    headers = {"X-Api-Key": API_KEY, "Content-Type": "application/json"}
    resp = requests.post(f"{BASE_URL}/v1/messages", json=payload, headers=headers, timeout=60)
    return {
        "status_code": resp.status_code,
        "body": resp.text,
        "elapsed_ms": resp.elapsed.total_seconds() * 1000,
    }


def extract_text(body: str) -> str:
    """Pull the assistant text from anthropic-format response."""
    try:
        d = json.loads(body)
        # Anthropic format: {"content": [{"type":"text","text":"..."}], ...}
        content = d.get("content")
        if isinstance(content, list):
            return "".join(c.get("text", "") for c in content if c.get("type") == "text")
        # OpenAI fallback: {"choices":[{"message":{"content":"..."}}]}
        choices = d.get("choices")
        if isinstance(choices, list) and choices:
            return choices[0].get("message", {}).get("content", "")
    except Exception:
        pass
    return body[:500]


def main():
    n_calls = 5
    print(f"Probing API determinism: {n_calls} calls × temperature=0 × {MODEL_NAME}", file=sys.stderr)
    print(f"Endpoint: {BASE_URL}/v1/messages", file=sys.stderr)

    results = []
    for i in range(n_calls):
        t0 = time.time()
        try:
            r = call_api(PROMPT, temperature=0.0, max_tokens=128)
            text = extract_text(r["body"])
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            results.append({
                "call_idx": i,
                "status_code": r["status_code"],
                "text": text,
                "text_digest": digest,
                "text_len": len(text),
                "elapsed_ms": r["elapsed_ms"],
            })
            print(f"[{i+1}/{n_calls}] status={r['status_code']} digest={digest} len={len(text)} elapsed={r['elapsed_ms']:.0f}ms", file=sys.stderr)
            print(f"  text: {text[:120]!r}", file=sys.stderr)
        except Exception as e:
            results.append({"call_idx": i, "error": f"{type(e).__name__}: {str(e)[:200]}"})
            print(f"[{i+1}/{n_calls}] ERROR: {e}", file=sys.stderr)
        time.sleep(1)  # rate limit gentle

    # Compare digests
    digests = [r.get("text_digest") for r in results if r.get("text_digest")]
    unique_digests = set(digests)
    all_match = len(unique_digests) == 1 and len(digests) == n_calls
    determinism_verdict = (
        "DETERMINISTIC (all calls byte-identical)" if all_match
        else f"NON-DETERMINISTIC ({len(unique_digests)} unique outputs across {len(digests)} calls)"
    )

    summary = {
        "audit_date": "2026-04-30",
        "purpose": "B-37 verification — does temperature=0 give byte-identical output across calls?",
        "model": MODEL_NAME,
        "endpoint": BASE_URL,
        "n_calls": n_calls,
        "n_successful": len(digests),
        "n_unique_digests": len(unique_digests),
        "all_match": all_match,
        "verdict": determinism_verdict,
        "results": results,
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2))

    md = [
        "# B-37 API Determinism Probe",
        "",
        f"**Audit date**: {summary['audit_date']}",
        f"**Model**: {MODEL_NAME}",
        f"**N calls**: {n_calls} × temperature=0",
        "",
        f"## Verdict: **{determinism_verdict}**",
        "",
        "## Per-call detail",
        "",
        "| Call | Status | Digest | Length | Elapsed |",
        "|---:|---:|:---|---:|---:|",
    ]
    for r in results:
        if r.get("error"):
            md.append(f"| {r['call_idx']+1} | ERROR | — | — | {r.get('error','')[:60]} |")
        else:
            md.append(f"| {r['call_idx']+1} | {r['status_code']} | `{r['text_digest']}` | {r['text_len']} | {r['elapsed_ms']:.0f}ms |")
    md.append("")
    md.append("## Output texts")
    md.append("")
    for r in results:
        if r.get("text"):
            md.append(f"### Call {r['call_idx']+1} (digest `{r.get('text_digest')}`)")
            md.append("```")
            md.append(r["text"][:500])
            md.append("```")
            md.append("")

    OUT_MD.write_text("\n".join(md))
    print(f"\nWrote {OUT_JSON}\n      {OUT_MD}", file=sys.stderr)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
