#!/usr/bin/env python3
"""Minimal proxy liveness probe — ONE request, no tools, no output docs.

Built 2026-07-06 for the outage#4 recovery watch loop (5-min cadence needs
something far lighter than probe_proxy_capability.py's 4-request sweep).
Prints one line `HTTP <code>` and exits 0 iff 200 — designed to be called
from a bash probe loop that counts consecutive 200s.

Endpoint/key resolution mirrors probe_proxy_capability.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]


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
    sys.exit(2)

payload = {
    "model": MODEL_NAME,
    "max_tokens": 8,
    "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
}

try:
    resp = requests.post(
        BASE_URL,
        json=payload,
        headers={"X-Api-Key": API_KEY, "Content-Type": "application/json"},
        timeout=40,
    )
except requests.RequestException as exc:
    print(f"HTTP 000 network-error: {exc}")
    sys.exit(1)

print(f"HTTP {resp.status_code}")
sys.exit(0 if resp.status_code == 200 else 1)
