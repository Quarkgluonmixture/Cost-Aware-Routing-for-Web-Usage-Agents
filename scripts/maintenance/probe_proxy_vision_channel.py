#!/usr/bin/env python3
"""Which proxy models can actually SEE an image, and what do they really cost?

Two gaps this closes, both left open by `probe_proxy_model_registry.py` (2026-08-08):

  Q1  §444.4 recorded reachability with a `max_tokens=8` TEXT probe and said so:
      "未验证图像通道 (VWA 4/6 mode 需吃图)". Four of six observation modes send a
      screenshot, so a text-reachable model that silently drops images is useless
      as a baseline -- and it fails in the worst possible way, by returning HTTP
      200 with a confident answer about an image it never saw.

  Q2  §444.2 chose sonnet-4-6 because it was "与 B0 完全同价 0.001/0.005". Between
      the 2026-08-09 and 2026-08-12 registry snapshots, ten entries moved off
      0.001/0.005 -- exactly the ones §444.4 had already flagged as placeholder
      priced. So the listed price is not a stable quantity, and the selection
      argument built on it is void. This probe reads the BILLED amount out of
      `metadata.remaining_quota` deltas instead of trusting any listed number.

The vision test is deliberately not an OCR task. The image is split left/right
into two uncommon colours and the model is asked for the LEFT half, so that:
  - a model that never received the image cannot land the answer by guessing a
    common colour (the failure mode we are hunting), and
  - "saw it but mixed up the halves" is distinguishable from "did not see it",
    which matters because the former is still a working image channel.

Usage:
    .venv/bin/python3 scripts/maintenance/probe_proxy_vision_channel.py            # text only, cheap
    .venv/bin/python3 scripts/maintenance/probe_proxy_vision_channel.py --vision   # + image probe

Cost: the text pass is ~$0.0001 total. The image pass sends a 224x224 PNG
(~67 image tokens by the w*h/750 rule) per candidate, so still well under a cent
overall -- but it DOES bill, and the billed delta is the point of the artifact.
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import io
import json
import re
import sys
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "docs/checkpoints/probes"
BASE = "https://i5xpracyci.execute-api.eu-west-2.amazonaws.com/model-api"

# (modelId, image_payload_format). The format follows proxy_api_agent.py:325 --
# anything whose name contains anthropic/claude takes Anthropic-native image
# blocks, everything else takes OpenAI image_url. Getting this wrong is itself a
# finding, so the probe records which format it sent.
CANDIDATES: list[tuple[str, str]] = [
    # --- control: the current B0, known to accept images -------------------------
    ("qwen.qwen3-vl-235b-a22b", "openai"),
    # --- B4 = Sonnet 5 (user decision 2026-08-12: at equal price take the newer
    #     model; sonnet-5 and sonnet-4-6 are both 0.003/0.015) -------------------
    ("eu.anthropic.claude-sonnet-5", "anthropic"),
    ("eu.anthropic.claude-sonnet-4-6", "anthropic"),   # the 08-09-verified fallback
    ("eu.anthropic.claude-opus-5", "anthropic"),       # 0.015/0.075 -- reachability only
    ("eu.anthropic.claude-haiku-4-5-20251001-v1:0", "anthropic"),
    # --- Google Gemma 4: appeared in the registry between 08-09 and 08-12, and is
    #     cheaper than B0 while being a generation above B2 (google.gemma-3-4b-it)
    ("google.gemma-4-31b", "openai"),
    ("google.gemma-4-26b-a4b", "openai"),
    ("google.gemma-3-27b-it", "openai"),               # same-family size control
    # --- cheap non-Anthropic frontier-ish, both repriced DOWN on 08-12 ----------
    ("zai.glm-5", "openai"),
]

LEFT_COLOUR = (255, 0, 255)    # magenta
RIGHT_COLOUR = (0, 128, 0)     # green
LEFT_WORDS = ("magenta", "pink", "purple", "violet", "fuchsia")
RIGHT_WORDS = ("green",)
VISION_PROMPT = ("This image is split into a left half and a right half, each a solid "
                 "colour. What colour is the LEFT half? Answer with one word.")


def load_key() -> str:
    import os
    if os.environ.get("PROXY_API_KEY"):
        return os.environ["PROXY_API_KEY"]
    auth = REPO / ".auth" / "qwen_api"
    if auth.exists():
        for line in auth.read_text().splitlines():
            if line.strip().startswith("rp_"):
                return line.strip()
    return ""


def make_split_image() -> str:
    """224x224 PNG, magenta left half / green right half, as a base64 payload."""
    from PIL import Image
    img = Image.new("RGB", (224, 224), RIGHT_COLOUR)
    for x in range(112):
        for y in range(224):
            img.putpixel((x, y), LEFT_COLOUR)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def image_block(fmt: str, b64: str) -> dict:
    if fmt == "anthropic":
        return {"type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": b64}}
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def call(key: str, model: str, content: list, max_tokens: int) -> dict:
    r = requests.post(
        f"{BASE}/invoke",
        headers={"X-Api-Key": key, "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "user", "content": content}],
              "max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0},
        timeout=120,
    )
    body = None
    if r.headers.get("content-type", "").startswith("application/json"):
        try:
            body = r.json()
        except ValueError:
            body = None
    return {"status": r.status_code, "body": body if isinstance(body, dict) else None,
            "raw": None if isinstance(body, dict) else r.text[:300]}


def extract_text(body: dict | None) -> str:
    """The proxy returns qwen as a flat string and Anthropic as a content list."""
    if not body:
        return ""
    c = body.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(b.get("text", "") for b in c if isinstance(b, dict))
    # tool_calls / choices shapes are not expected for a plain text ask, but record them
    return json.dumps(c)[:300] if c is not None else ""


def quota_of(body: dict | None):
    if not body:
        return None
    md = body.get("metadata")
    if isinstance(md, dict) and "remaining_quota" in md:
        return md["remaining_quota"]
    return body.get("remaining_quota")


def classify_vision(text: str, token_delta: int | None) -> str:
    """Colour answer AND the input-token delta, because either alone is ambiguous.

    The first run of this probe (2026-08-12) classified gemma-4-31b as a silent-drop
    suspect purely from its answer ("White"). Its input tokens told a different story:
    622 against 44 on the text call, so the image was very much received and billed.
    The colour answer alone cannot separate "never got the image" from "got it and
    misread it", and those two have opposite consequences -- the first disqualifies the
    endpoint, the second disqualifies the model. The token delta separates them.
    """
    t = text.lower()
    got_image = token_delta is not None and token_delta >= 50
    named_left = any(w in t for w in LEFT_WORDS)
    named_right = any(w in t for w in RIGHT_WORDS)
    if named_left:
        return "SAW_IT"                       # correct half, correct colour
    if named_right:
        return "SAW_IT_HALVES_SWAPPED"        # channel works; spatial answer wrong
    if not t.strip():
        return "EMPTY_REPLY"
    if got_image:
        return "IMAGE_BILLED_BUT_MISREAD"     # endpoint fine, model cannot read it
    return "SILENT_DROP_SUSPECT"              # 200, no colour, and no image tokens


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vision", action="store_true",
                    help="also send the image probe (bills a few tenths of a cent)")
    args = ap.parse_args()

    key = load_key()
    if not key:
        print("ERROR: no PROXY_API_KEY and no rp_ key in .auth/qwen_api", file=sys.stderr)
        return 1

    b64 = make_split_image() if args.vision else ""
    rows = []
    print(f"{'model':<46} {'text':>6} {'vision':>7}  {'verdict':<26} {'img tok':>8} billed")
    print("-" * 112)
    for model, fmt in CANDIDATES:
        row: dict = {"model": model, "image_format": fmt}

        t = call(key, model, [{"type": "text", "text": "hi"}], 8)
        row["text_status"] = t["status"]
        row["text_reply"] = extract_text(t["body"])[:80]
        row["text_error"] = (t["body"] or {}).get("error") or t["raw"]
        row["text_usage"] = (t["body"] or {}).get("usage")
        q_before = quota_of(t["body"])
        row["quota_after_text"] = q_before

        # The response's own `usage.cost` is the billed amount and it is authoritative:
        # every model probed on 2026-08-12 had cost == tokens x LISTED price exactly, so
        # the listed price is trustworthy ONCE it is a real price. What is not trustworthy
        # is assuming it stays put -- ten entries moved off the 0.001/0.005 placeholder
        # between 08-09 and 08-12, which is how §444.2's "same price as B0" argument for
        # sonnet-4-6 came to be built on a placeholder. Hence: re-derive, never remember.
        tu = row["text_usage"] or {}
        row["implied_unit_price"] = None
        vs, verdict, billed = "-", "text-only", (tu.get("cost") if tu else None)
        if args.vision and t["status"] == 200:
            v = call(key, model, [image_block(fmt, b64),
                                  {"type": "text", "text": VISION_PROMPT}], 24)
            row["vision_status"] = v["status"]
            row["vision_reply"] = extract_text(v["body"])[:120]
            row["vision_error"] = (v["body"] or {}).get("error") or v["raw"]
            vu = (v["body"] or {}).get("usage") or {}
            row["vision_usage"] = vu or None
            row["quota_after_vision"] = quota_of(v["body"])
            delta = (vu.get("inputTokens") - tu.get("inputTokens")
                     if vu.get("inputTokens") is not None
                     and tu.get("inputTokens") is not None else None)
            row["image_input_token_delta"] = delta
            vs = str(v["status"])
            verdict = (classify_vision(row["vision_reply"], delta) if v["status"] == 200
                       else f"HTTP_{v['status']}")
            billed = (tu.get("cost") or 0) + (vu.get("cost") or 0)
            row["billed_usd_both_calls"] = billed
        elif args.vision:
            verdict = f"skipped (text HTTP {t['status']})"
        row["verdict"] = verdict

        dtxt = row.get("image_input_token_delta")
        print(f"{model:<46} {t['status']:>6} {vs:>7}  {verdict:<26} "
              f"{'' if dtxt is None else f'+{dtxt:>4}tok'} "
              f"{'' if billed is None else f'${billed:.6f}'}")
        rows.append(row)

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"proxy_vision_channel_{stamp}.json"
    out.write_text(json.dumps({
        "probe": "proxy vision channel + billed-cost, closing §444.4's unverified image channel",
        "generated": stamp,
        "vision_pass_ran": bool(args.vision),
        "image_spec": {"size": "224x224 PNG", "left": LEFT_COLOUR, "right": RIGHT_COLOUR,
                       "prompt": VISION_PROMPT},
        "verdict_legend": {
            "SAW_IT": "named the left colour -- image channel works",
            "SAW_IT_HALVES_SWAPPED": "named the right colour -- channel works, spatial answer wrong",
            "IMAGE_BILLED_BUT_MISREAD": "input tokens jumped, so the image arrived and "
                                        "was billed -- the MODEL could not read it. "
                                        "Disqualifies the model, not the endpoint",
            "SILENT_DROP_SUSPECT": "200, no colour named, AND no image-sized token jump: "
                                   "the image likely never reached the model. This is the "
                                   "dangerous case -- it looks like a working call",
            "EMPTY_REPLY": "200 with empty content",
        },
        "results": rows,
    }, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")

    if args.vision:
        ok = [r["model"] for r in rows if r["verdict"].startswith("SAW_IT")]
        print(f"\nimage channel CONFIRMED on {len(ok)}/{len(rows)}: {ok}")
        misread = [r["model"] for r in rows if r["verdict"] == "IMAGE_BILLED_BUT_MISREAD"]
        if misread:
            print(f"⚠️  image arrived and was billed but the model misread it "
                  f"(model problem, not endpoint): {misread}")
        drop = [r["model"] for r in rows if r["verdict"] == "SILENT_DROP_SUSPECT"]
        if drop:
            print(f"🚨 silent-drop suspects (no colour AND no image tokens): {drop}")
        tot = sum(r.get("billed_usd_both_calls") or 0 for r in rows)
        print(f"\nthis probe billed ${tot:.6f} in total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
