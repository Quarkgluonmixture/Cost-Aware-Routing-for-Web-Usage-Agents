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
import os
import re
import subprocess
import sys
import time
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
# B-855 (A1.15b Chunk δ P1-6): import from glm_client directly. Pre-fix
# imported from glm_diagnosis_sidecar (1996 LOC) just to call the GLM API
# — paid for the entire module load. Now: import from glm_client (~200
# LOC) directly. Decouples playbook refresh from diagnosis sidecar
# refactors.
from glm_client import _load_glm_config, _call_glm_chat  # noqa: E402

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
        "glm-refresh-playbook-s2": "glm_playbook_s2.log",
        "glm-refresh-playbook": "glm_playbook.log",
    }
    # Failure markers (in tail) — keep this list grep-able for future maintenance.
    # `notify_on_fail.sh` writes "❌ P79 cron failed" to ntfy not log, so we detect
    # the actual in-log markers from script aborts / make failures / GLM errors.
    fail_patterns = [
        r"❌ GLM synth failed, aborting",       # glm_playbook_refresh.py abort
        r"make: \*\*\* \[Makefile[^\]]+\] Error \d+",  # make rule failure
        r"⚠️\s+GLM call failed:",                # GLM upstream error (timeout/HTTP 5xx)
        r"^Traceback \(most recent call last\)", # Python exception
        r"❌ P79 cron failed",                   # legacy / external script
        r"^Exit:\s*[1-9]",                       # explicit exit code
    ]
    fail_re = re.compile("|".join(fail_patterns), re.MULTILINE)

    for name, fname in log_files.items():
        path = CRON_LOG_DIR / fname
        if not path.exists():
            jobs.append({"name": name, "last_run": "(no log yet)", "status": "?", "tail": ""})
            continue
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        text = path.read_text(encoding="utf-8", errors="ignore")[-3000:]
        # Detect status by looking at the LAST attempt in the tail.
        # Strategy: find positions of all fail markers + all success markers,
        # whichever is more recent (later in text) wins.
        last_fail_pos = -1
        for m in fail_re.finditer(text):
            last_fail_pos = max(last_fail_pos, m.start())
        last_ok_pos = -1
        for marker in ("Updated 0/", "Updated 1/", "Updated 2/", "✏️  Updated docs/checkpoints/PLAYBOOK.md"):
            idx = text.rfind(marker)
            if idx > last_ok_pos:
                last_ok_pos = idx
        if last_fail_pos == -1 and last_ok_pos == -1:
            status = "🟡 unclear"
        elif last_fail_pos > last_ok_pos:
            status = "❌ recent failure"
        else:
            status = "✅ ok"
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


def get_error_scan() -> str:
    """Read logs/cron/error_scan.json (produced by error_scan.py @5min cron)."""
    p = CRON_LOG_DIR / "error_scan.json"
    if not p.exists():
        return "(error_scan.json missing — error-scan cron 未运行)"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return f"(error_scan.json parse failed: {e})"
    n = d.get("n_errors", 0)
    n_files = d.get("n_files_scanned", "?")
    scanned = d.get("scanned_at", "?")
    if n == 0:
        return f"n_errors=0, n_files_scanned={n_files}, scanned_at={scanned}"
    lines = [f"n_errors={n}, n_files_scanned={n_files}, scanned_at={scanned}"]
    for e in d.get("errors", [])[:8]:
        lines.append(
            f"- [{e.get('severity', '?')}] {e.get('kind', '?')} | {e.get('file', '?')}:{e.get('line_no', '?')} "
            f"@ {e.get('mtime', '?')} | {e.get('snippet', '')[:200]}"
        )
    return "\n".join(lines)


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

def build_context(section: str = "both") -> str:
    """Aggregate context for §1 (critical path) and/or §2 (automation status).

    section ∈ {"1", "2", "both"} controls which inputs are gathered — saves
    `make active` subprocess + ntfy poll time when only one section needed.
    """
    lines: list[str] = []

    if section in ("1", "both"):
        cells = read_status_dir("cells")
        issues = read_status_dir("issues")

        active_cells = [c for c in cells if c.get("status") == "active"]
        pending_cells = [c for c in cells if c.get("status") in ("pending", "queued", "blocked")]
        active_issues = [i for i in issues if i.get("status") == "active"]

        lines += [
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

    if section in ("2", "both"):
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
            warn_lines = [l for l in text.splitlines() if "BROKEN" in l or "missing" in l or "WARN" in l]
            lines.append("\n".join(warn_lines[:20]) if warn_lines else "(clean — no broken links)")
        else:
            lines.append("(no scan run yet)")

        lines.append("\n=== §2 INPUT — NTFY FAIL HISTORY (last 24h) ===")
        lines.append(get_ntfy_recent_fails())

        lines.append("\n=== §2 INPUT — ERRORS scan (runner / watchdog / cron logs) ===")
        lines.append(get_error_scan())

    return "\n".join(lines)


# ---- GLM call ----

_S1_PROMPT = """## §1 — 今日 critical path 早报 (~150-200 词, narrative 风格)

写法: 像给一个忙碌的研究员讲今天该关心什么。**不要罗列**, 写**连续段落**:
1. 第 1 段 (~60 词): 主战场 — 当前正在跑的 runner / chain / 它的进度跟瓶颈, 用人话讲为什么慢/快, e.g. "B1 phantom_prompt classifieds 跑了 1.5 天, 才到 27%, 因为 seonglae 同时跑 8-seed sweep 抢 GPU, 吞吐降到 1 ep/h."
2. 第 2 段 (~50 词): 队列 + pending — 多少 cell 在排队, 卡在哪, 啥时候能动 (e.g. advisor align / RunPod approval / 前序 cell)
3. 第 3 段 (~50 词): Active issues 解读 — 不是抄 issue 标题, 是讲**为什么这些 issue 卡在这**, 跟主战场什么关系
4. 最后 1 行 **明确 next action 推荐**: "👉 建议: ..." — 1 句话, 具体动作 (e.g. "联系 seonglae 协调 GPU 或推 RunPod 经费走流程", "等 17:45 cron 自动同步即可", "立刻跑 make analysis 把新数据进 figures")

中文为主, 数字可以保留. 适度 emoji (🔴 ⏳ ▶️ ✅) 强调状态但不过量。
"""

_S2_PROMPT = """## §2 — 自动化运行状态
五个 subsection (按此顺序输出):

### 2.1 Cron job 健康度 (last 24h)
markdown table 列**所有** cron jobs (现 4 个: glm-update-cells / glm-refresh-playbook-s2 / glm-refresh-playbook / check-links):
| Job | 上次 run | 状态 | 备注 |

如果某 job 状态非 ✅, 备注列写人话原因 (e.g. "首次运行待 17:45 触发", "GLM API timeout 自动 retry").

### 2.2 Cell 状态变更近况 (changelog tail)
bullet list, 最近 5-8 条 cell frontmatter 变更, 每条 1 行: `时间(HH:MM) cell名: 变更字段`. 强调任何 `rerun_detected` / `status→active` / `status→done` / `pid_dead_cleared` 信号. 如 changelog 空则写"近 3 天无 cell 变更"。

### 2.3 Dead link warnings
若有 broken link 列前 5 条 (file:line — broken target). 否则: `✅ 无 broken link`。

### 2.4 Ntfy fail alerts 历史
列 last 24h 失败 (timestamp + title). 如无: `✅ 近 24h 无失败`。

### 2.5 🔴 Active errors / warnings (runner / watchdog log scan, last 24h)
基于 INPUT 里的 ERRORS scan, **用人话**总结每条:
- 若 `n_errors == 0`: 写 `✅ 近 24h 无 runner / watchdog 错误 (扫了 N 个 log 文件)`
- 否则: bullet list, 每条格式 `[severity] file (line N, time): 一句话讲发生啥` — 严重 (oom/traceback/not_logged_in) 用 🔴, 中度 (timeout/http5xx) 用 ⚠️, 轻度 (notify_fail) 用 ℹ️
- 重点: 不要把 stack trace 直接 paste, 用人话解释 — e.g. "CUDA OOM in episode 47, runner 自动重试" 不是 "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate ..."
"""


def call_glm(context: str, section: str = "both") -> Optional[tuple[Optional[str], Optional[str]]]:
    """Synthesize requested section(s). Returns (s1_body, s2_body) — None for unrequested."""
    if not GLM_CFG_PATH.exists():
        print(f"⚠️  GLM config {GLM_CFG_PATH} not found", file=sys.stderr)
        return None
    glm_cfg = _load_glm_config(GLM_CFG_PATH)

    parts = ["你为一个 P79 实验项目的 personal playbook 合成内容。\n"]
    output_format = ["输出格式 (严格按此分隔, 不要加额外引言):\n"]
    if section in ("1", "both"):
        parts.append(_S1_PROMPT)
        output_format.append("=== SECTION 1 ===\n<§1 body, no header line>\n")
    if section in ("2", "both"):
        parts.append(_S2_PROMPT)
        output_format.append("=== SECTION 2 ===\n### 2.1 Cron job 健康度 (last 24h)\n\n<table>\n\n### 2.2 Cell 状态变更近况 (changelog tail)\n\n<bullets>\n\n### 2.3 Dead link warnings\n\n<content>\n\n### 2.4 Ntfy fail alerts 历史\n\n<content>\n")

    prompt = "\n".join(parts) + "\n" + "\n".join(output_format) + f"\nINPUT (机器聚合数据, 据此合成):\n{context}\n"
    messages = [
        {"role": "system", "content": "You synthesize personal-project playbook sections from machine-aggregated data. Output Chinese-mixed markdown, follow exact format."},
        {"role": "user", "content": prompt},
    ]
    # Larger timeout for combined §1+§2 synthesis (more context, GLM thinking
    # model needs more time). §2-only is leaner and 90s usually suffices.
    timeout_s = 180 if section == "both" else 90
    # Retry on transient GLM upstream errors (timeout / HTTP 5xx). Most failures
    # at 21:00 BST window are GLM-side flakes; a single retry typically succeeds.
    raw = None
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            raw = _call_glm_chat(glm_cfg, messages, timeout_s=timeout_s).strip()
            if attempt > 0:
                print(f"✅ GLM call succeeded on retry #{attempt}", file=sys.stderr)
            break
        except Exception as e:
            last_err = e
            backoff = 5 * (3 ** attempt)  # 5s, 15s, 45s — total ~65s worst case
            print(f"⚠️  GLM call attempt {attempt+1}/3 failed: {e}; "
                  f"{'retrying in ' + str(backoff) + 's' if attempt < 2 else 'no more retries'}",
                  file=sys.stderr)
            if attempt < 2:
                time.sleep(backoff)
    # Audit (D) 2026-05-09: persist consecutive GLM-fail count and
    # ntfy at 3 consecutive failures (mirrors F36 SSH chain pattern).
    # File: logs/cron/glm_fail_count.json with {section: count}.
    GLM_FAIL_FILE = REPO / "logs" / "cron" / "glm_fail_count.json"
    try:
        prev = json.loads(GLM_FAIL_FILE.read_text()) if GLM_FAIL_FILE.exists() else {}
    except Exception:
        prev = {}
    if raw is None:
        prev[section] = int(prev.get(section, 0)) + 1
        try:
            GLM_FAIL_FILE.parent.mkdir(parents=True, exist_ok=True)
            GLM_FAIL_FILE.write_text(json.dumps(prev, indent=2))
        except Exception:
            pass
        if prev[section] >= 3:
            try:
                import urllib.request as _ureq
                ntfy_topic = os.environ.get("NTFY_TOPIC", "p79-exp-dgx-spark")
                _ureq.urlopen(_ureq.Request(
                    f"https://ntfy.sh/{ntfy_topic}",
                    data=(
                        f"GLM API failed {prev[section]}x consecutive (section={section}). "
                        f"PLAYBOOK §1+§2 stale; check .auth/glm key + 智谱 quota."
                    ).encode("utf-8"),
                    headers={"Title": "GLM API down", "Priority": "high"},
                ), timeout=10).read()
            except Exception:
                pass
        print(f"⚠️  GLM call failed after 3 attempts: {last_err} "
              f"(consecutive count: {prev[section]})", file=sys.stderr)
        return None
    # Reset failure count on success.
    if prev.get(section, 0) > 0:
        prev[section] = 0
        try:
            GLM_FAIL_FILE.write_text(json.dumps(prev, indent=2))
        except Exception:
            pass

    s1_body, s2_body = None, None
    if section in ("1", "both"):
        m1 = re.search(r"=== SECTION 1 ===\s*\n(.*?)(?:\n=== SECTION 2 ===\s*\n|\Z)", raw, re.DOTALL)
        if not m1:
            print(f"⚠️  GLM output missing SECTION 1 delimiter; raw[:500]={raw[:500]}", file=sys.stderr)
            return None
        s1_body = m1.group(1).strip()
    if section in ("2", "both"):
        m2 = re.search(r"=== SECTION 2 ===\s*\n(.*?)\Z", raw, re.DOTALL)
        if not m2:
            # If only §2 requested, GLM may have skipped the SECTION 2 marker — fall back to whole raw
            if section == "2":
                s2_body = raw.strip()
            else:
                print(f"⚠️  GLM output missing SECTION 2 delimiter; raw[:500]={raw[:500]}", file=sys.stderr)
                return None
        else:
            s2_body = m2.group(1).strip()
    return s1_body, s2_body


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
    parser.add_argument("--section", choices=["1", "2", "both"], default="both",
                        help="which section(s) to refresh (default both). "
                             "Use --section 2 for fast cron (avoids `make active` subprocess + cells/issues scan).")
    args = parser.parse_args()

    context = build_context(args.section)
    if args.show_context:
        print(context)
        return 0

    label = f"§{args.section}" if args.section != "both" else "§1 + §2"
    print(f"📋 Synthesizing PLAYBOOK {label} via GLM...")
    result = call_glm(context, args.section)
    if not result:
        print("❌ GLM synth failed, aborting")
        return 1
    s1_body, s2_body = result

    if s1_body is not None:
        print("\n=== GLM-generated §1 body ===")
        print(s1_body)
    if s2_body is not None:
        print("\n=== GLM-generated §2 body ===")
        print(s2_body)
    print("=" * 60)

    if not args.apply:
        print("\n(dry-run; pass --apply to write to PLAYBOOK.md)")
        return 0

    text = PLAYBOOK.read_text(encoding="utf-8")
    if s1_body is not None:
        text = replace_section1(text, s1_body)
    if s2_body is not None:
        text = replace_section2(text, s2_body)
    PLAYBOOK.write_text(text, encoding="utf-8")
    print(f"\n✏️  Updated {PLAYBOOK.relative_to(REPO)} {label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
