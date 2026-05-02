#!/usr/bin/env python3
"""PLAYBOOK §6 critical path snapshot daily refresh — GLM-synthesized.

Aggregates: active processes (`make active --json`) + open issues
(`_status/issues/`) + active/pending cells (`_status/cells/`),
GLM 5.1 synthesizes 1 paragraph + bullet list for §6, regex-replaces
in PLAYBOOK.md.

Run via cron @daily (`0 8 * * * .venv/bin/python ...`) or
`make glm-refresh-playbook` ad-hoc.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml

REPO = Path(__file__).resolve().parents[3]
PLAYBOOK = REPO / "docs/checkpoints/PLAYBOOK.md"
STATUS = REPO / "docs/checkpoints/_status"

# import GLM helpers from existing sidecar
sys.path.insert(0, str(REPO / "scripts/maintenance/glm"))
from glm_diagnosis_sidecar import _load_glm_config, _call_glm_chat  # noqa: E402

GLM_CFG_PATH = REPO / ".auth/glm"

SECTION_HEADER = "## §6 当前 critical path snapshot"
SECTION_END_RE = re.compile(r"\n##\s+§7", re.MULTILINE)


def read_status_dir(subdir: str) -> list[dict]:
    """Read all *.md frontmatter under _status/<subdir>/."""
    items = []
    d = STATUS / subdir
    if not d.exists():
        return items
    for p in sorted(d.glob("*.md")):
        text = p.read_text(encoding="utf-8")
        m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
        if not m:
            continue
        try:
            fm = yaml.safe_load(m.group(1)) or {}
        except yaml.YAMLError:
            continue
        fm["_file"] = p.name
        items.append(fm)
    return items


def get_active_processes() -> str:
    try:
        out = subprocess.run(
            ["make", "active", "--silent"],
            cwd=REPO, capture_output=True, text=True, timeout=30,
        )
        return out.stdout.strip()[:3000]  # cap to 3000 chars
    except Exception as e:
        return f"(make active failed: {e})"


def build_context() -> str:
    cells = read_status_dir("cells")
    issues = read_status_dir("issues")

    active_cells = [c for c in cells if c.get("status") == "active"]
    pending_cells = [c for c in cells if c.get("status") in ("pending", "queued", "blocked")]
    active_issues = [i for i in issues if i.get("status") == "active"]

    lines = [
        "=== ACTIVE PROCESSES (make active) ===",
        get_active_processes(),
        "",
        f"=== ACTIVE CELLS ({len(active_cells)}) ===",
    ]
    for c in active_cells:
        lines.append(f"- {c.get('baseline','?')} {c.get('site','?')} {c.get('mode','?')}: "
                     f"progress={c.get('progress','?')}%, blocker={c.get('blocker','')}, eta={c.get('eta','')}")

    lines.append(f"\n=== PENDING/QUEUED/BLOCKED CELLS ({len(pending_cells)}) ===")
    for c in pending_cells[:10]:  # cap
        lines.append(f"- {c.get('baseline','?')} {c.get('site','?')} {c.get('mode','?')} "
                     f"[{c.get('status','?')}]: blocker={c.get('blocker','')}")

    lines.append(f"\n=== ACTIVE ISSUES ({len(active_issues)}) ===")
    for i in active_issues:
        lines.append(f"- {i['_file']}: priority={i.get('priority','?')}, action={i.get('action','')}")

    return "\n".join(lines)


def call_glm_synthesize(context: str) -> Optional[str]:
    if not GLM_CFG_PATH.exists():
        print(f"⚠️  GLM config {GLM_CFG_PATH} not found; skipping GLM synth", file=sys.stderr)
        return None
    glm_cfg = _load_glm_config(GLM_CFG_PATH)
    prompt = f"""You synthesize a "current critical path snapshot" for a personal playbook.
Input is a structured dump of P79 paper experiment status.

Output ONLY the body for §6 (no header). Format:
- 4-6 lines, each with status emoji (✅/⏳/🚫/🔴) + brief cell or blocker description
- Last line: 1-sentence "today's bottleneck" summary

Be concise (under 120 words total). Use Chinese where natural.

INPUT:
{context}
"""
    messages = [
        {"role": "system", "content": "You are a concise project status synthesizer. Output Chinese-mixed bullet list, ~120 words."},
        {"role": "user", "content": prompt},
    ]
    try:
        return _call_glm_chat(glm_cfg, messages, timeout_s=60).strip()
    except Exception as e:
        print(f"⚠️  GLM call failed: {e}", file=sys.stderr)
        return None


def replace_section6(playbook_text: str, new_body: str) -> str:
    """Regex-replace §6 body block."""
    h6_match = re.search(r"^## §6 .*?$", playbook_text, re.MULTILINE)
    if not h6_match:
        raise ValueError("§6 header not found in PLAYBOOK")
    h7_match = SECTION_END_RE.search(playbook_text, pos=h6_match.end())
    if not h7_match:
        raise ValueError("§7 header not found in PLAYBOOK (boundary marker)")

    h6_end = h6_match.end()
    h7_start = h7_match.start()

    new_block = (
        f"\n\n> 自己 scratchpad. 用 ✅/⏳/🚫/🔴 标. 改这里, 不改 next_steps.\n"
        f"> *Last GLM refresh: {subprocess.run(['date','+%Y-%m-%d %H:%M'], capture_output=True, text=True).stdout.strip()}*\n\n"
        f"{new_body}\n"
    )
    return playbook_text[:h6_end] + new_block + playbook_text[h7_start:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="actually write (default dry-run)")
    parser.add_argument("--show-context", action="store_true", help="print aggregated context")
    args = parser.parse_args()

    context = build_context()
    if args.show_context:
        print(context)
        return 0

    print("📋 Synthesizing PLAYBOOK §6 via GLM...")
    new_body = call_glm_synthesize(context)
    if not new_body:
        print("❌ GLM synth failed, aborting")
        return 1

    print("\n=== GLM-generated §6 body ===")
    print(new_body)
    print("=" * 60)

    if not args.apply:
        print("\n(dry-run; pass --apply to write to PLAYBOOK.md)")
        return 0

    text = PLAYBOOK.read_text(encoding="utf-8")
    new_text = replace_section6(text, new_body)
    PLAYBOOK.write_text(new_text, encoding="utf-8")
    print(f"\n✏️  Updated {PLAYBOOK.relative_to(REPO)} §6")
    return 0


if __name__ == "__main__":
    sys.exit(main())
