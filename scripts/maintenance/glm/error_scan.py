#!/usr/bin/env python3
"""Scan logs/ + logs/cron/ for runner / watchdog / cron errors in last 24h.

Output JSON to logs/cron/error_scan.json — consumed by glm_playbook_refresh
to populate PLAYBOOK §2.5 "Active errors / warnings".

Patterns scanned (priority order):
- Python Traceback (most-recent stack)
- NOT_LOGGED_IN / auth_refresh fail (VWA session race)
- CUDA OOM / OutOfMemoryError
- TimeoutError
- ❌ markers (notify_on_fail.sh emits these)
- watchdog ALERT / FAILURE patterns
- HTTP 5xx from proxy API

Cron @5min via Makefile target `error-scan` (or ad-hoc).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LOGS_DIR = REPO / "logs"
CRON_LOGS = REPO / "logs/cron"
OUTPUT = CRON_LOGS / "error_scan.json"

# Pattern priority — higher = more severe
PATTERNS = [
    ("oom",          re.compile(r"(?:CUDA out of memory|OutOfMemoryError|torch\.cuda\.OutOfMemoryError)", re.IGNORECASE), 90),
    ("traceback",    re.compile(r"^Traceback \(most recent call last\):", re.MULTILINE), 80),
    ("not_logged_in", re.compile(r"NOT[_ ]LOGGED[_ ]IN|auth_refresh.*(?:fail|error)|session.*expired", re.IGNORECASE), 75),
    ("watchdog_alert", re.compile(r"watchdog.*(?:ALERT|FAILURE|abort)", re.IGNORECASE), 70),
    ("notify_fail",  re.compile(r"❌ P79 cron (?:fail|error)|cron failed", re.IGNORECASE), 60),
    ("timeout",      re.compile(r"TimeoutError|asyncio\.TimeoutError|read timed out", re.IGNORECASE), 55),
    ("http5xx",      re.compile(r"HTTPError.*5\d{2}|5\d{2} (?:Server Error|Bad Gateway|Service Unavailable)"), 50),
    ("python_error", re.compile(r"^[A-Z]\w+Error: ", re.MULTILINE), 40),
]

MAX_TAIL_BYTES = 200_000  # 200KB tail per log
MAX_HITS_PER_FILE = 5
MAX_TOTAL_ERRORS = 50


def scan_file(path: Path, cutoff: datetime) -> list[dict]:
    """Return list of {kind, severity, file, line_no, snippet, ts} for recent matches."""
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return []
    if mtime < cutoff:
        return []

    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > MAX_TAIL_BYTES:
                f.seek(-MAX_TAIL_BYTES, 2)
                # discard partial first line
                f.readline()
            text = f.read().decode("utf-8", errors="ignore")
    except OSError:
        return []

    hits = []
    seen_kinds = {}  # dedup: at most 1 hit per kind per file
    for kind, pat, sev in PATTERNS:
        for m in pat.finditer(text):
            if seen_kinds.get(kind, 0) >= MAX_HITS_PER_FILE // 2:
                break
            # extract context (line containing match + 2 lines after for traceback)
            start = max(0, text.rfind("\n", 0, m.start()) + 1)
            end_lookahead = 3 if kind == "traceback" else 0
            end = m.end()
            for _ in range(end_lookahead):
                nl = text.find("\n", end)
                if nl == -1:
                    break
                end = nl + 1
            snippet = text[start:end].strip().replace("\n", " ⏎ ")[:240]
            line_no = text.count("\n", 0, m.start()) + 1
            hits.append({
                "kind": kind,
                "severity": sev,
                "file": str(path.relative_to(REPO)),
                "mtime": mtime.isoformat(timespec="minutes"),
                "line_no": line_no,
                "snippet": snippet,
            })
            seen_kinds[kind] = seen_kinds.get(kind, 0) + 1
            if len(hits) >= MAX_HITS_PER_FILE:
                return hits
    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=24, help="lookback window (default 24h)")
    parser.add_argument("--print", action="store_true", help="print JSON to stdout (default also writes to logs/cron/error_scan.json)")
    args = parser.parse_args()

    cutoff = datetime.now(timezone.utc) - timedelta(hours=args.hours)

    # Sources: logs/ top level (runner logs B*_*.log, queue chains), logs/cron/
    candidates = []
    for pattern in ("B0_*.log", "B1_*.log", "queue_*.log", "watchdog_*.log", "phantom_*.log"):
        candidates.extend(LOGS_DIR.glob(pattern))
    candidates.extend(CRON_LOGS.glob("*.log"))

    all_hits = []
    for f in candidates:
        if not f.is_file():
            continue
        hits = scan_file(f, cutoff)
        all_hits.extend(hits)
        if len(all_hits) >= MAX_TOTAL_ERRORS:
            break

    # Sort by severity desc, then mtime desc
    all_hits.sort(key=lambda h: (-h["severity"], h["mtime"]), reverse=False)
    all_hits = all_hits[:MAX_TOTAL_ERRORS]
    # Re-sort by mtime desc for output (most recent first)
    all_hits.sort(key=lambda h: h["mtime"], reverse=True)

    payload = {
        "scanned_at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "lookback_hours": args.hours,
        "n_files_scanned": len(candidates),
        "n_errors": len(all_hits),
        "errors": all_hits,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.print:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"📋 Scanned {len(candidates)} files in last {args.hours}h, found {len(all_hits)} errors → {OUTPUT.relative_to(REPO)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
