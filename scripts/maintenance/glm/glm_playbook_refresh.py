#!/usr/bin/env python3
"""PLAYBOOK §1 (critical path) + §2 (automation status) daily dual-section
refresh — GLM-synthesized.

Single GLM 5.1 call aggregates:
  §1 critical path  ← `make active` + active/pending cells + active issues
  §2 automation     ← cron job last-run + cell_changelog tail + dead_links + ntfy fails

GLM emits both bodies in one response with markdown delimiter, parser splits
and regex-replaces each section in PLAYBOOK.md.

Run via cron @daily 08:00 BST (`0 8 * * *`) or `make glm-refresh-playbook`.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import yaml

REPO = Path(__file__).resolve().parents[3]
PLAYBOOK = REPO / "docs/checkpoints/PLAYBOOK.md"
STATUS = REPO / "docs/checkpoints/_status"
CRON_LOG_DIR = REPO / "logs/cron"
CHANGELOG = CRON_LOG_DIR / "cell_changelog.jsonl"

sys.path.insert(0, str(REPO / "scripts/maintenance/glm"))
from glm_diagnosis_sidecar import _load_glm_config, _call_glm_chat  # noqa: E402

GLM_CFG_PATH = REPO / ".auth/glm"

# §1 boundary: "## §1 ..." → "## §2 ..."
S1_END_RE = re.compile(r"\n##\s+§2\s", re.MULTILINE)
# §2 boundary: "## §2 ..." → "## §3 ..."
S2_END_RE = re.compile(r"\n##\s+§3\s", re.MULTILINE)


# ---- §1 context ----

def read_status_dir(subdir: str) -> list[dict]:
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
        return out.stdout.strip()[:3000]
    except Exception as e:
        return f"(make active failed: {e})"


# ---- §2 context ----

def get_cron_job_status() -> list[dict]:
    """For each cron log, return (name, last_modified, last_exit_marker, tail)."""
    jobs = []
    log_files = {
        "glm-update-cells": "glm_update_cells.log",
        "glm-refresh-playbook": "glm_playbook.log",
    }
    for name, fname in log_files.items():
        path = CRON_LOG_DIR / fname
        if not path.exists():
            jobs.append({"name": name, "last_run": "(no log yet)", "status": "?", "tail": ""})
            continue
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        text = path.read_text(encoding="utf-8", errors="ignore")[-3000:]
        # detect failure marker
        if "❌ P79 cron failed" in text or re.search(r"^Exit:\s*[1-9]", text, re.MULTILINE):
            status = "❌ recent failure"
        elif "Updated" in text or "no change" in text:
            status = "✅ ok"
        else:
            status = "🟡 unclear"
        jobs.append({
            "name": name,
            "last_run": mtime.isoformat(timespec="minutes"),
            "status": status,
            "tail": text[-500:],
        })
    # check-links: find latest dead_links_*.log
    dl_logs = sorted(CRON_LOG_DIR.glob("dead_links_*.log"), key=lambda p: p.stat().st_mtime)
    if dl_logs:
        latest = dl_logs[-1]
        mtime = datetime.fromtimestamp(latest.stat().st_mtime, tz=timezone.utc)
        text = latest.read_text(encoding="utf-8", errors="ignore")
        broken = text.count("BROKEN") + text.count("missing")
        jobs.append({
            "name": "check-links",
            "last_run": mtime.isoformat(timespec="minutes"),
            "status": f"🟡 {broken} warnings" if broken else "✅ clean",
            "tail": text[-1500:],
        })
    else:
        jobs.append({"name": "check-links", "last_run": "(never run)", "status": "—", "tail": ""})
    return jobs


def get_cell_changelog_tail(n: int = 20) -> list[dict]:
    """Read last N changelog entries (most recent first)."""
    if not CHANGELOG.exists():
        return []
    rows = []
    try:
        with CHANGELOG.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return []
    # Drop very-old entries (>3 days) for §2 freshness
    cutoff = datetime.now(timezone.utc) - timedelta(days=3)
    for line in lines[-200:]:  # cap parse
        try:
            row = json.loads(line)
            ts = datetime.fromisoformat(row["ts"])
            if ts >= cutoff:
                rows.append(row)
        except Exception:
            continue
    return rows[-n:][::-1]


def get_ntfy_recent_fails() -> str:
    """Best-effort poll ntfy server for last 24h failures (no auth required for public topic)."""
    try:
        import urllib.request
        topic = "p79-exp-dgx-spark"
        url = f"https://ntfy.sh/{topic}/json?poll=1&since=24h"
        with urllib.request.urlopen(url, timeout=8) as r:
            text = r.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return f"(ntfy poll failed: {e})"
    fails = []
    for line in text.splitlines():
        if "P79 cron fail" in line or '"priority":4' in line or '"priority":5' in line:
            try:
                msg = json.loads(line)
                title = msg.get("title", "")
                if "fail" in title.lower() or "error" in title.lower():
                    fails.append(f"  {datetime.fromtimestamp(msg.get('time', 0), tz=timezone.utc).isoformat(timespec='minutes')}: {title}")
            except Exception:
                pass
    return "\n".join(fails[-8:]) if fails else "(no fails in last 24h)"


# ---- context aggregation ----

def build_context() -> str:
    cells = read_status_dir("cells")
    issues = read_status_dir("issues")

    active_cells = [c for c in cells if c.get("status") == "active"]
    pending_cells = [c for c in cells if c.get("status") in ("pending", "queued", "blocked")]
    active_issues = [i for i in issues if i.get("status") == "active"]

    lines = [
        "=== §1 INPUT — ACTIVE PROCESSES (make active) ===",
        get_active_processes(),
        "",
        f"=== §1 INPUT — ACTIVE CELLS ({len(active_cells)}) ===",
    ]
    for c in active_cells:
        lines.append(
            f"- {c.get('baseline','?')} {c.get('site','?')} {c.get('mode','?')}: "
            f"progress={c.get('progress','?')}%, blocker={c.get('blocker','')}, eta={c.get('eta','')}"
        )

    lines.append(f"\n=== §1 INPUT — PENDING/QUEUED/BLOCKED CELLS ({len(pending_cells)}) ===")
    for c in pending_cells[:10]:
        lines.append(
            f"- {c.get('baseline','?')} {c.get('site','?')} {c.get('mode','?')} "
            f"[{c.get('status','?')}]: blocker={c.get('blocker','')}"
        )

    lines.append(f"\n=== §1 INPUT — ACTIVE ISSUES ({len(active_issues)}) ===")
    for i in active_issues:
        lines.append(f"- {i['_file']}: priority={i.get('priority','?')}, action={i.get('action','')}")

    # §2 inputs
    lines.append("\n=== §2 INPUT — CRON JOB HEALTH ===")
    for j in get_cron_job_status():
        lines.append(f"- {j['name']} | last={j['last_run']} | {j['status']}")

    lines.append("\n=== §2 INPUT — CELL CHANGELOG TAIL (last 3d, most recent first) ===")
    for row in get_cell_changelog_tail(15):
        ts = row['ts'][:16]  # strip seconds
        changes = ", ".join(row['changes'])[:120]
        lines.append(f"- {ts} {row['cell']}: {changes}")
    if not get_cell_changelog_tail(1):
        lines.append("(empty)")

    lines.append("\n=== §2 INPUT — DEAD LINK SCAN (latest) ===")
    dl_logs = sorted(CRON_LOG_DIR.glob("dead_links_*.log"), key=lambda p: p.stat().st_mtime)
    if dl_logs:
        text = dl_logs[-1].read_text(encoding="utf-8", errors="ignore")
        # extract first 1500 chars of warnings
        warn_lines = [l for l in text.splitlines() if "BROKEN" in l or "missing" in l or "WARN" in l]
        lines.append("\n".join(warn_lines[:20]) if warn_lines else "(clean — no broken links)")
    else:
        lines.append("(no scan run yet)")

    lines.append("\n=== §2 INPUT — NTFY FAIL HISTORY (last 24h) ===")
    lines.append(get_ntfy_recent_fails())

    return "\n".join(lines)


# ---- GLM call ----

def call_glm_dual(context: str) -> Optional[tuple[str, str]]:
    if not GLM_CFG_PATH.exists():
        print(f"⚠️  GLM config {GLM_CFG_PATH} not found", file=sys.stderr)
        return None
    glm_cfg = _load_glm_config(GLM_CFG_PATH)
    prompt = f"""你为一个 P79 实验项目的 personal playbook 同时合成 TWO 节内容。

## §1 — 当前 critical path snapshot (~120 词)
4-6 行 status emoji (✅/⏳/🚫/🔴) + 简短 cell 或 blocker 描述。
最后 1 行: "今日瓶颈: ..." 1 句总结。
中文为主.

## §2 — 自动化运行状态
四个 subsection (按此顺序输出):

### 2.1 Cron job 健康度 (last 24h)
3-row markdown table 列 cron jobs:
| Job | 上次 run | 状态 | 备注 |

### 2.2 Cell 状态变更近况 (changelog tail)
bullet list, 最近 5-8 条 cell frontmatter 变更, 每条 1 行: `时间(HH:MM) cell名: 变更字段`. 强调任何 `rerun_detected` / `status→active` / `status→done` 信号. 如 changelog 空则写"近 3 天无 cell 变更"。

### 2.3 Dead link warnings
若有 broken link 列前 5 条 (file:line — broken target). 否则: `✅ 无 broken link`。

### 2.4 Ntfy fail alerts 历史
列 last 24h 失败 (timestamp + title). 如无: `✅ 近 24h 无失败`。

输出格式 (严格按此分隔, 不要加额外引言):

=== SECTION 1 ===
<§1 body, no header line>

=== SECTION 2 ===
### 2.1 Cron job 健康度 (last 24h)

<table>

### 2.2 Cell 状态变更近况 (changelog tail)

<bullets>

### 2.3 Dead link warnings

<content>

### 2.4 Ntfy fail alerts 历史

<content>

INPUT (机器聚合数据, 据此合成):
{context}
"""
    messages = [
        {"role": "system", "content": "You synthesize personal-project playbook sections from machine-aggregated data. Output Chinese-mixed markdown, follow exact format."},
        {"role": "user", "content": prompt},
    ]
    try:
        raw = _call_glm_chat(glm_cfg, messages, timeout_s=90).strip()
    except Exception as e:
        print(f"⚠️  GLM call failed: {e}", file=sys.stderr)
        return None

    # Split on === SECTION 1 === / === SECTION 2 ===
    m1 = re.search(r"=== SECTION 1 ===\s*\n(.*?)(?:\n=== SECTION 2 ===\s*\n|\Z)", raw, re.DOTALL)
    m2 = re.search(r"=== SECTION 2 ===\s*\n(.*?)\Z", raw, re.DOTALL)
    if not (m1 and m2):
        print(f"⚠️  GLM output missing section delimiters; raw[:500]={raw[:500]}", file=sys.stderr)
        return None
    return m1.group(1).strip(), m2.group(1).strip()


# ---- replace ----

def replace_section1(text: str, body: str) -> str:
    h1 = re.search(r"^## §1\s+.*?$", text, re.MULTILINE)
    if not h1:
        raise ValueError("§1 header not found in PLAYBOOK")
    end = S1_END_RE.search(text, pos=h1.end())
    if not end:
        raise ValueError("§2 header (boundary) not found in PLAYBOOK")
    return text[:h1.end()] + f"\n\n{body}\n\n---\n" + text[end.start():]


def replace_section2(text: str, body: str) -> str:
    h2 = re.search(r"^## §2\s+.*?$", text, re.MULTILINE)
    if not h2:
        raise ValueError("§2 header not found in PLAYBOOK")
    end = S2_END_RE.search(text, pos=h2.end())
    if not end:
        raise ValueError("§3 header (boundary) not found in PLAYBOOK")
    return text[:h2.end()] + f"\n\n{body}\n\n---\n" + text[end.start():]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="actually write (default dry-run)")
    parser.add_argument("--show-context", action="store_true", help="print aggregated context")
    args = parser.parse_args()

    context = build_context()
    if args.show_context:
        print(context)
        return 0

    print("📋 Synthesizing PLAYBOOK §1 + §2 via GLM...")
    pair = call_glm_dual(context)
    if not pair:
        print("❌ GLM synth failed, aborting")
        return 1
    s1_body, s2_body = pair

    print("\n=== GLM-generated §1 body ===")
    print(s1_body)
    print("\n=== GLM-generated §2 body ===")
    print(s2_body)
    print("=" * 60)

    if not args.apply:
        print("\n(dry-run; pass --apply to write to PLAYBOOK.md)")
        return 0

    text = PLAYBOOK.read_text(encoding="utf-8")
    text = replace_section1(text, s1_body)
    text = replace_section2(text, s2_body)
    PLAYBOOK.write_text(text, encoding="utf-8")
    print(f"\n✏️  Updated {PLAYBOOK.relative_to(REPO)} §1 + §2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
