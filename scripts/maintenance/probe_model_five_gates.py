#!/usr/bin/env python3
"""Five-gate production-shape probe for any proxy modelId (笔记 §471).

WHY A NEW SCRIPT
----------------
The existing probes each hardcode B0. §468 ran the five-gate matrix for candidate
backbones as an ad-hoc pass, so re-running it for a newly authorised model meant
rebuilding the payload by hand. This script takes modelIds on the command line and
reuses the PRODUCTION artefacts (`_WEB_ACTION_TOOL`, `validate_action`) rather than
a hand-copied schema — a copy drifts, and a probe that passes on a stale schema is
worse than no probe.

THE FIVE GATES (§468). A model is usable as a baseline only if it clears all five:
  1. HTTP 200                     — reachable + authorised (403 = subscription, not capability)
  2. tool_call returned           — body.tool_calls[0].function present
  3. action is schema-valid       — validate_action() accepts the parsed arguments
  4. logprobs present             — body.logprobs.content non-empty (confidence schema)
  5. actually READ the image      — not merely "accepted an image field"

Gate 5 is the one that catches silent failures: §468 found `gemma-3-27b-it` returns
HTTP 200 on an image payload and still cannot see the picture, and §456 warns that
the most dangerous failure is "200 + a confident answer about an image it never saw".
So the image carries a token the model cannot guess: a random 3-digit mark id drawn
at call time and rendered large. The model is asked to click that mark. Guessing the
right 3-digit number by chance is 1/900.

⚠️ Prices in the registry DRIFT (§468 warned for display names; 2026-08-19 found it
true of prices too — GPT-5.6 was recorded as "strictly B0-priced 0.001/0.005" and is
actually three tiers spanning 25x). This probe re-reads price from the registry and
prints it next to the gates; do not quote a price from any document.

Usage:
  .venv/bin/python3 scripts/maintenance/probe_model_five_gates.py \
      global.openai.gpt-5.6-terra global.openai.gpt-5.6-luna
  # always include a known-good control (§444.3) — omit --no-control at your peril
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import io
import json
import os
import random
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from p79.agents.proxy_api_agent import _WEB_ACTION_TOOL  # noqa: E402  production schema
from p79.backends.action_utils import validate_action  # noqa: E402  production validator

BASE = "https://i5xpracyci.execute-api.eu-west-2.amazonaws.com/model-api"
CONTROL = "qwen.qwen3-vl-235b-a22b"  # B0 — known good; if THIS fails, the probe is broken


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


# Gate-5 payload. §456 established that COLOUR is the robust channel test: an
# uncommon colour cannot be guessed, and unlike a rendered number it does not
# depend on the model's OCR. The first version of this probe used a 3-digit mark
# alone and the B0 control FAILED it — B0 plainly saw the image (it described the
# background and the centred logo) but read `[326]` as `[360]`. That is a real
# finding about digit OCR (cf. §299.1 class C, where a model misread a colour off
# a compressed screenshot) but it makes digits a bad instrument. So: colour is
# primary, the mark is a secondary signal, and either one hitting counts as "saw it".
PALETTE = [("magenta", (255, 0, 255)), ("teal", (0, 128, 128)), ("olive", (128, 128, 0)),
           ("navy", (0, 0, 128)), ("maroon", (128, 0, 0)), ("lime", (0, 255, 0))]


def make_marked_image(mark: int, colour_rgb) -> str:
    """Screenshot-shaped: one big colour panel carrying one mark id."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (640, 360), (250, 250, 250))
    d = ImageDraw.Draw(img)
    d.rectangle([60, 60, 580, 300], fill=colour_rgb)
    try:
        font = ImageFont.load_default(size=140)
    except TypeError:
        font = ImageFont.load_default()
    d.text((150, 110), f"[{mark}]", fill="white", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def probe(model: str, key: str, price: dict | None) -> dict:
    mark = random.randint(100, 999)
    colour_name, colour_rgb = random.choice(PALETTE)
    b64 = make_marked_image(mark, colour_rgb)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": (
                "The screenshot shows one large coloured panel carrying a mark id. "
                "Call web_action to click it. In `thought`, FIRST state the panel's colour "
                "in one word, then the mark id you can read."
            )},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
        "tools": [_WEB_ACTION_TOOL],
        "tool_choice": "required",
        "max_tokens": 512,
        "temperature": 0.0,
        "top_p": 1.0,
        "logprobs": True,
        "top_logprobs": 2,
    }
    # 2026-08-19: `global.openai.gpt-5.6-*` reject function tools with a 400 that
    # names its own fix — "Function tools with reasoning_effort are not supported
    # ... set reasoning_effort to 'none'". We never sent that field, so the proxy
    # or the model defaults it; sending it explicitly is what clears the gate.
    # Harmless for models that ignore unknown keys, so it is sent unconditionally
    # and the control run proves it does not regress B0.
    if "openai" in model:
        payload["reasoning_effort"] = "none"
    out = {"model": model, "expected_mark": mark, "expected_colour": colour_name, "price": price,
           "g1_http200": False, "g2_tool_call": False, "g3_schema_valid": False,
           "g4_logprobs": False, "g5_saw_image": False}
    try:
        r = requests.post(f"{BASE}/invoke", headers={"X-Api-Key": key,
                          "Content-Type": "application/json"}, json=payload, timeout=180)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    out["status"] = r.status_code
    out["g1_http200"] = r.status_code == 200
    try:
        body = r.json()
    except Exception:  # noqa: BLE001
        out["error"] = r.text[:300]
        return out
    if r.status_code != 200:
        # 403 = subscription/permission; 400 = protocol. The distinction decides
        # whether this is something you can go ask for (§468).
        out["error"] = json.dumps(body)[:300]
        return out

    calls = body.get("tool_calls") or []
    if calls and isinstance(calls, list):
        fn = (calls[0] or {}).get("function") or {}
        out["g2_tool_call"] = bool(fn.get("name"))
        try:
            parsed = json.loads(fn.get("arguments") or "{}")
            out["action"] = parsed
            _, ok = validate_action(parsed)
            out["g3_schema_valid"] = bool(ok)
            blob = json.dumps(parsed, ensure_ascii=False).lower()
            out["g5_colour_hit"] = colour_name in blob
            out["g5_mark_hit"] = str(mark) in blob
            # Either channel counts; colour is the one that does not depend on OCR.
            out["g5_saw_image"] = out["g5_colour_hit"] or out["g5_mark_hit"]
        except Exception as e:  # noqa: BLE001
            out["parse_error"] = f"{type(e).__name__}: {e}"
    lp = (body.get("logprobs") or {}).get("content") or []
    out["g4_logprobs"] = len(lp) > 0
    meta = body.get("metadata") or {}
    out["cost_usd"] = meta.get("cost")
    out["remaining_quota"] = meta.get("remaining_quota")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--no-control", action="store_true",
                    help="skip the known-good control (NOT recommended — see §444.3)")
    args = ap.parse_args()

    key = load_key()
    if not key:
        print("ERROR: no PROXY_API_KEY and .auth/qwen_api has no rp_ key", file=sys.stderr)
        return 1
    headers = {"X-Api-Key": key, "Content-Type": "application/json"}

    prices = {}
    try:
        reg = requests.get(f"{BASE}/models", headers=headers, timeout=60).json()
        for m in (reg.get("models") if isinstance(reg, dict) else reg) or []:
            mid = m.get("modelId") or m.get("model_id") or m.get("id")
            prices[mid] = {"in": m.get("inputPrice"), "out": m.get("outputPrice")}
    except Exception as e:  # noqa: BLE001
        print(f"WARN: registry unreadable ({e}); prices will be null", file=sys.stderr)

    targets = list(args.models)
    if not args.no_control and CONTROL not in targets:
        targets.append(CONTROL)

    results = []
    for m in targets:
        res = probe(m, key, prices.get(m))
        results.append(res)
        gates = "".join("✓" if res[f"g{i}_{n}"] else "✗" for i, n in
                        [(1, "http200"), (2, "tool_call"), (3, "schema_valid"),
                         (4, "logprobs"), (5, "saw_image")])
        p = res.get("price") or {}
        tag = "  ← CONTROL" if m == CONTROL else ""
        print(f"{gates}  {m:42} status={res.get('status')} "
              f"in={p.get('in')} out={p.get('out')} cost={res.get('cost_usd')}{tag}")
        if res.get("error"):
            print(f"        error: {res['error'][:180]}")

    ctl = next((r for r in results if r["model"] == CONTROL), None)
    if ctl and not all(ctl[f"g{i}_{n}"] for i, n in
                       [(1, "http200"), (2, "tool_call"), (3, "schema_valid"),
                        (4, "logprobs"), (5, "saw_image")]):
        print("\n⚠️  CONTROL FAILED — the probe itself is broken. Do NOT read the other "
              "rows as evidence about those models (§470.8 / GOTCHAS §5).", file=sys.stderr)

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPO_ROOT / "docs" / "checkpoints" / "probes" / f"five_gates_{ts}.json"
    out.write_text(json.dumps({"generated": ts, "gates": results}, indent=1, ensure_ascii=False))
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
