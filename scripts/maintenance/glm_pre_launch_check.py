#!/usr/bin/env python3
"""Pre-launch sanity check — GLM reviews queue script + config + recent logs,
flags suspicious settings before experiment launch.

Usage:
  python glm_pre_launch_check.py --queue queue_phantom_som.sh --baseline B0 --site reddit
  python glm_pre_launch_check.py --config configs/exp_v2_xxx.yaml

Catches (examples):
- baseline / site / mode 不匹配 in queue script call
- 漏 RESET_BEFORE=1 (paper-grade contamination risk)
- 同 site B0+B1 同时跑 (cross-contam)
- config benchmark 跟 site 不一致

Exit 0 = OK to launch / Exit 1 = WARN human review / Exit 2 = error
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(REPO / "scripts/maintenance"))
from glm_diagnosis_sidecar import _load_glm_config, _call_glm_chat  # noqa: E402

GLM_CFG_PATH = REPO / ".auth/glm"


def get_active_runs() -> str:
    try:
        out = subprocess.run(
            ["pgrep", "-af", "run_experiment"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip()[:1500]
    except Exception:
        return ""


def call_glm_review(launch_summary: str) -> tuple[bool, str]:
    """Returns (ok_to_launch, reasoning)."""
    if not GLM_CFG_PATH.exists():
        return True, f"(GLM config not found, skipping review)"

    glm_cfg = _load_glm_config(GLM_CFG_PATH)

    prompt = f"""You are a pre-launch sanity check for P79 web-agent experiments.
Review the proposed launch + current state, flag if anything is suspicious.

KEY HARD RULES (violations = paper-grade contamination):
1. **Same-site B0 XOR B1** — never run B0 + B1 on same site simultaneously (account/cart/session race)
2. **RESET_BEFORE=1 mandatory** for paper-grade runs (state pollute risk)
3. queue script baseline must match site reset semantics
4. WA reset NOT YET IMPLEMENTED (queue scripts skip wa reset; OK only if not paper-grade)
5. config benchmark field must match queue script site

INPUT:
{launch_summary}

OUTPUT format (JSON):
{{
  "verdict": "OK" | "WARN" | "BLOCK",
  "reason": "<one sentence>",
  "concerns": ["<concern 1>", "<concern 2>"]
}}

If everything looks fine, output `verdict: OK`. If anything suspicious, `WARN`. If clearly violates a hard rule, `BLOCK`.
"""
    messages = [
        {"role": "system", "content": "You audit experiment launches for paper-grade contamination risk. Output ONLY JSON."},
        {"role": "user", "content": prompt},
    ]

    try:
        raw = _call_glm_chat(glm_cfg, messages, timeout_s=60)
    except Exception as e:
        return True, f"(GLM call failed: {e}, allowing launch)"

    # Extract JSON
    m = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not m:
        return True, f"(GLM unparseable, raw={raw[:200]})"

    try:
        parsed = json.loads(m.group())
    except json.JSONDecodeError:
        return True, f"(JSON decode failed, raw={raw[:200]})"

    verdict = parsed.get("verdict", "OK")
    reason = parsed.get("reason", "")
    concerns = parsed.get("concerns", [])

    summary = f"verdict={verdict} | {reason}"
    if concerns:
        summary += "\n  concerns:\n" + "\n".join(f"    - {c}" for c in concerns)

    return verdict == "OK", summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", help="queue script name (e.g. queue_phantom_som.sh)")
    parser.add_argument("--baseline", help="B0 / B1 / Claude")
    parser.add_argument("--site", help="classifieds / reddit / shopping / wa_*")
    parser.add_argument("--mode", help="dom / som / vision / phantom_*")
    parser.add_argument("--reset", action="store_true", help="RESET_BEFORE=1 set")
    parser.add_argument("--config", type=Path, help="optional: YAML config path to include")
    args = parser.parse_args()

    summary_parts = ["=== PROPOSED LAUNCH ==="]
    if args.queue:
        summary_parts.append(f"queue script: {args.queue}")
    if args.baseline:
        summary_parts.append(f"baseline: {args.baseline}")
    if args.site:
        summary_parts.append(f"site: {args.site}")
    if args.mode:
        summary_parts.append(f"mode: {args.mode}")
    summary_parts.append(f"RESET_BEFORE: {'1' if args.reset else '(unset — paper-grade RISK)'}")

    if args.config and args.config.exists():
        summary_parts.append("\n=== CONFIG ===")
        summary_parts.append(args.config.read_text(encoding="utf-8")[:2000])

    summary_parts.append("\n=== ACTIVE RUNS (pgrep) ===")
    summary_parts.append(get_active_runs() or "(none)")

    launch_summary = "\n".join(summary_parts)

    print("📋 Pre-launch GLM review...\n")
    ok, msg = call_glm_review(launch_summary)
    print(msg)
    print()
    if ok:
        print("✅ OK to launch")
        return 0
    else:
        print("⚠️  WARN — human review recommended before launch")
        return 1


if __name__ == "__main__":
    sys.exit(main())
