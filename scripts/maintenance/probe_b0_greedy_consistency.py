#!/usr/bin/env python3
"""B-138 (/stress A1.1 v8 Claude F4, 2026-05-15): T=0 greedy consistency
probe for B0 proxy. Measures whether `temperature=0.0, top_p=1.0` on the
remote proxy is *mechanically deterministic* (same prompt → same response
byte-for-byte across N calls) or merely *semantically greedy* (same
top-1 choice but residual non-determinism from fp8/int8 quant ties,
sampling kernel differences, top_k truncation interactions, or silent
seed-drop).

Paper-grade gap (Claude F4): comment in `proxy_api_agent.py:506-512`
claims T=0 + top_p=1.0 + seed forwarded ≡ HF `do_sample=False`, but
notes "some proxies ignore seed". The §1 hero claim's reproducibility
assertion depends on B0 step-level determinism. This probe quantifies it
cheaply (~30min) without VWA env dependency.

Methodology (cheap probe, no VWA):
  1. POST same payload (frozen prompt + image-free messages) N=10 times
     to B0 proxy.
  2. Hash response text byte-for-byte → unique response count.
  3. Report: byte-identical-rate, distinct-action-rate (parse + extract
     action_type to detect "semantic-greedy but byte-different" case).

Decision rule:
  • byte_identical_rate ≥ 99% → mechanical greedy confirmed, no
    paper §3.5 disclosure needed.
  • 90% ≤ byte_identical_rate < 99% → semantic-greedy with byte-noise;
    paper §3.5 should disclose "B0 proxy emulates greedy; reproducibility
    verified at X% step-level consistency over N runs".
  • byte_identical_rate < 90% → B0 truly non-deterministic at API
    layer; paper §1 reproducibility claim is invalid → escalate
    advisor for full T=0 audit (P0-9 parking lot).

Usage:
    PROXY_API_KEY=<key> python3 scripts/maintenance/probe_b0_greedy_consistency.py [N]

Default N=10. Output:
    JSON to docs/checkpoints/reproducibility/b0_greedy_probe_<HHMMSS>.json
    Console summary with verdict.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from p79.backends.action_utils import parse_action_text  # noqa: E402


PROMPT_TEXT = (
    "Task: Click the search button.\n"
    "System: You are a precise web navigation agent. Output ONLY valid JSON.\n"
    "Accessibility Tree:\n"
    "[1] button 'Search'\n"
    "[2] textbox 'Query'\n"
    "[3] link 'Home'\n"
)


def _build_payload(model_name: str) -> Dict[str, Any]:
    """Frozen payload — same exact bytes each call."""
    return {
        "model": model_name,
        "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT_TEXT}]}],
        "max_tokens": 256,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
    }


def _post_once(endpoint: str, api_key: str, payload: Dict[str, Any], timeout: int = 60) -> str:
    """One POST → return raw response text. Caller handles parsing."""
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "X-Api-Key": api_key,
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    parsed = json.loads(body)
    content = parsed.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text", "")
        return str(content)
    return str(content or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int, nargs="?", default=10,
                    help="Number of probe calls (default 10)")
    ap.add_argument("--endpoint", default=os.getenv(
        "PROXY_API_ENDPOINT",
        "https://i5xpracyci.execute-api.eu-west-2.amazonaws.com/model-api/invoke",
    ))
    ap.add_argument("--model", default="qwen.qwen3-vl-235b-a22b")
    args = ap.parse_args()

    api_key = os.getenv("PROXY_API_KEY")
    if not api_key:
        print("ERROR: PROXY_API_KEY env var not set", file=sys.stderr)
        return 2

    payload = _build_payload(args.model)
    responses = []
    print(f"Probing B0 ({args.model}) at T=0 N={args.n} times...")
    for i in range(args.n):
        t0 = time.time()
        try:
            text = _post_once(args.endpoint, api_key, payload)
        except Exception as exc:
            print(f"  [{i+1}/{args.n}] FAILED: {exc}")
            responses.append(None)
            continue
        elapsed = time.time() - t0
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        action, valid, fail_reason = parse_action_text(text)
        action_sig = (
            action.get("action_type"),
            action.get("element_id"),
            action.get("coordinate"),
        )
        responses.append({
            "text": text,
            "text_hash": text_hash,
            "elapsed_s": round(elapsed, 2),
            "action_sig": action_sig,
            "parse_valid": valid,
            "parse_fail_reason": fail_reason,
        })
        print(f"  [{i+1}/{args.n}] hash={text_hash} action={action_sig} valid={valid} ({elapsed:.1f}s)")

    # Analysis
    successful = [r for r in responses if r is not None]
    if not successful:
        print("\n❌ All probe calls failed; aborting")
        return 1

    n_total = len(successful)
    distinct_hashes = {r["text_hash"] for r in successful}
    distinct_actions = {r["action_sig"] for r in successful}
    byte_identical_rate = (n_total - len(distinct_hashes) + 1) / n_total  # count of mode-hash
    # Simpler: pct that match the most common response
    from collections import Counter
    hash_counts = Counter(r["text_hash"] for r in successful)
    top_hash, top_count = hash_counts.most_common(1)[0]
    byte_consistency = top_count / n_total
    action_counts = Counter(r["action_sig"] for r in successful)
    top_action, top_action_count = action_counts.most_common(1)[0]
    action_consistency = top_action_count / n_total

    # Verdict
    if byte_consistency >= 0.99:
        verdict = "MECHANICAL_GREEDY"
        disclose = "no paper §3.5 disclosure needed"
    elif byte_consistency >= 0.90:
        verdict = "SEMANTIC_GREEDY_WITH_NOISE"
        disclose = "paper §3.5 should disclose reproducibility rate"
    else:
        verdict = "NON_DETERMINISTIC"
        disclose = "❌ paper §1 reproducibility claim invalid → escalate advisor"

    summary = {
        "probe_meta": {
            "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            "model": args.model,
            "endpoint": args.endpoint,
            "n_calls": args.n,
            "n_successful": n_total,
            "n_failed": args.n - n_total,
            "payload_prompt_chars": len(PROMPT_TEXT),
        },
        "consistency": {
            "byte_consistency_rate": round(byte_consistency, 4),
            "action_consistency_rate": round(action_consistency, 4),
            "distinct_response_hashes": len(distinct_hashes),
            "distinct_action_signatures": len(distinct_actions),
            "top_response_hash": top_hash,
            "top_response_count": top_count,
            "top_action_signature": list(top_action) if top_action else None,
            "top_action_count": top_action_count,
        },
        "verdict": {
            "label": verdict,
            "byte_consistency_pct": round(byte_consistency * 100, 1),
            "paper_grade_action": disclose,
        },
        "responses": responses,
    }

    out_dir = REPO_ROOT / "docs" / "checkpoints" / "reproducibility"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
    out_path = out_dir / f"b0_greedy_probe_{stamp}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print()
    print("=" * 60)
    print(f"Verdict: {verdict} ({byte_consistency*100:.1f}% byte / {action_consistency*100:.1f}% action consistency)")
    print(f"Paper-grade action: {disclose}")
    print(f"Distinct response hashes: {len(distinct_hashes)} / {n_total}")
    print(f"Distinct action sigs: {len(distinct_actions)} / {n_total}")
    print(f"Output: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
