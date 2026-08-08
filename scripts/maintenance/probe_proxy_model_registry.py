#!/usr/bin/env python3
"""Proxy model-registry + B4 (Claude) feasibility probe — 2026-08-08.

Two questions, one artifact:

  Q1  Can the AWS proxy serve a Claude baseline (B4)?
  Q2  What else is on the registry, and at what price?

Answer to Q1 is yes and it costs one config line. `proxy_api_agent.py:704`
sends ``payload["model"] = model_cfg["api_name"]``; the proxy resolves that
against its own registry, so pointing `api_name` at an Anthropic modelId is the
entire change. The response shape differs (Anthropic-native
``content:[{type:"text",...}]`` vs the flat string qwen returns), but the
Anthropic branch already exists in the agent from the B-991 hybrid-shim work.

Q2 matters because the registry turned out to hold 57 models spanning
Anthropic / Google / Meta / Mistral / NVIDIA / DeepSeek / Qwen / Z.AI — i.e. the
model axis is limited by budget, not by availability.

⚠️ PRICING TRAP (cost me a wrong estimate on first read): B0 is
``qwen.qwen3-vl-235b-a22b`` at 0.001/0.005, NOT ``qwen.qwen3-235b-a22b-2507``
at 0.0008/0.004. Several Anthropic entries (Sonnet 5, Sonnet 4.6, Haiku 4.5,
Opus 4.7/4.8) are listed at the SAME 0.001/0.005 as B0 — so a Claude baseline
is not automatically the expensive option. Only the dated Sonnet/Opus 4.5
entries carry standard Bedrock list price (0.003/0.015, 0.015/0.075).

Usage:
    .venv/bin/python3 scripts/maintenance/probe_proxy_model_registry.py [--invoke]

Without --invoke it only does the free GET (no billing). With --invoke it also
sends one 1-token request per candidate in CANDIDATES to prove reachability
end-to-end (~$0.0001 total). Artifact lands in docs/checkpoints/probes/.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "docs/checkpoints/probes"
BASE = "https://i5xpracyci.execute-api.eu-west-2.amazonaws.com/model-api"

# Reachability candidates. The bogus first entry is deliberate: it is what
# revealed the registry endpoint at all ("Use GET /model-api/models for
# available models"), and it keeps the error shape in the artifact.
CANDIDATES = [
    "p79.definitely-not-a-real-model",
    "qwen.qwen3-vl-235b-a22b",                      # B0 control
    "eu.anthropic.claude-sonnet-5",                 # B4 candidate, B0-priced
    "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",  # B4 candidate, list-priced
]


def load_key() -> str:
    env = os.environ.get("PROXY_API_KEY", "")
    if env:
        return env
    auth = REPO_ROOT / ".auth" / "qwen_api"
    if not auth.exists():
        return ""
    for line in auth.read_text().splitlines():
        if line.strip().startswith("rp_"):
            return line.strip()
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--invoke", action="store_true", help="also send 1-token reachability calls (~$0.0001)")
    args = ap.parse_args()

    key = load_key()
    if not key:
        print("ERROR: PROXY_API_KEY unset and .auth/qwen_api has no rp_ key", file=sys.stderr)
        return 1
    headers = {"X-Api-Key": key, "Content-Type": "application/json"}

    reg = requests.get(f"{BASE}/models", headers=headers, timeout=60)
    registry = reg.json() if reg.status_code == 200 else None

    invocations = []
    quota = None
    if args.invoke:
        for name in CANDIDATES:
            r = requests.post(
                f"{BASE}/invoke",
                headers=headers,
                json={
                    "model": name,
                    "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
                    "max_tokens": 8,
                    "temperature": 0.0,
                    "top_p": 1.0,
                },
                timeout=90,
            )
            body = r.json() if r.headers.get("content-type", "").startswith("application/json") else None
            if isinstance(body, dict) and "metadata" in body:
                quota = body["metadata"].get("remaining_quota", quota)
            invocations.append(
                {
                    "requested": name,
                    "status": r.status_code,
                    # `model` echoed back may differ from `requested` — the proxy
                    # rewrites bare ids to a region-prefixed one (claude-sonnet-4-5
                    # -> eu.anthropic.claude-sonnet-4-5-...). Worth recording.
                    "resolved": (body or {}).get("model"),
                    "content_shape": type((body or {}).get("content")).__name__,
                    "usage": (body or {}).get("usage"),
                    "error": (body or {}).get("error"),
                }
            )

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact = {
        "probe": "proxy_model_registry_and_b4_feasibility",
        "generated": stamp,
        "registry_status": reg.status_code,
        "n_models": len(registry) if registry else None,
        "models": registry,
        "invocations": invocations,
        "remaining_quota": quota,
        "notes": {
            "b0_model": "qwen.qwen3-vl-235b-a22b",
            "b0_price": "in 0.001 / out 0.005",
            "b4_change_surface": "configs: model.api_name -> an Anthropic modelId; no new backend",
            "pricing_trap": "do not read B0's price off qwen.qwen3-235b-a22b-2507 (0.0008/0.004)",
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"proxy_model_registry_{stamp}.json"
    out.write_text(json.dumps(artifact, indent=1, ensure_ascii=False))

    if registry:
        vend: dict[str, int] = {}
        for m in registry:
            vend[m.get("provider", "?")] = vend.get(m.get("provider", "?"), 0) + 1
        print(f"registry: {len(registry)} models across {len(vend)} providers")
        for v, n in sorted(vend.items(), key=lambda kv: -kv[1]):
            print(f"  {v:14s} {n}")
    for inv in invocations:
        print(f"  {inv['requested']:48s} -> {inv['status']} resolved={inv['resolved']}")
    if quota:
        print(f"remaining_budget = ${quota.get('remaining_budget'):.4f} / {quota.get('budget_limit')}")
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
