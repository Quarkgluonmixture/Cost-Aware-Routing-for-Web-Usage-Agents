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
import os
import re
import subprocess
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
    # Audit (G) 2026-05-09: P79-specific scaffold-bug patterns from
    # master_bug_catalog.md (B-22 / B-81h / B-81i / F23) — these would
    # otherwise hide as generic Tracebacks and not get prioritised.
    ("magento_redirect_loop", re.compile(r"302.*?metis|Magento.*?302.*?redirect|base_url.*?cycle", re.IGNORECASE), 88),
    ("cutlass_kernel_miss", re.compile(r"cutlassF: no kernel found|cutlass.*?launch.*?fail", re.IGNORECASE), 85),
    ("nvrtc_arch", re.compile(r"nvrtc:.*?invalid value for --gpu-architecture|sm_121.*?(?:not|invalid)", re.IGNORECASE), 85),
    ("fp_adjust_error", re.compile(r"fp_reason.*?adjustment_error|Failed to compute adjusted_success", re.IGNORECASE), 82),
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


# Audit (E) 2026-05-09: system-level health checks.
DISK_FREE_GB_THRESHOLD = 50  # ntfy if repo partition has < 50 GB free
TAILSCALE_FAIL_FILE = REPO / "logs" / "cron" / "tailscale_fail_count"
DISK_FAIL_FILE = REPO / "logs" / "cron" / "disk_fail_count"
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "p79-exp-dgx-spark")


def _push_ntfy(title: str, body: str, priority: str = "default") -> None:
    """Local ntfy push (avoids importing watchdog module)."""
    try:
        import urllib.request as _ureq
        _ureq.urlopen(_ureq.Request(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=body.encode("utf-8"),
            headers={"Title": title, "Priority": priority},
        ), timeout=10).read()
    except Exception:
        pass


def _check_disk() -> dict:
    """Return {free_gb, total_gb, pct_free, alert}. ntfy if below threshold."""
    import shutil
    usage = shutil.disk_usage(REPO)
    free_gb = usage.free / (1024 ** 3)
    total_gb = usage.total / (1024 ** 3)
    pct_free = 100.0 * usage.free / usage.total
    alert = free_gb < DISK_FREE_GB_THRESHOLD
    if alert:
        try:
            n_fail = int(DISK_FAIL_FILE.read_text().strip()) if DISK_FAIL_FILE.exists() else 0
        except Exception:
            n_fail = 0
        n_fail += 1
        try:
            DISK_FAIL_FILE.write_text(str(n_fail))
        except Exception:
            pass
        if n_fail >= 2:  # alert on 2 consecutive (5min × 2 = 10min) below
            _push_ntfy(
                "Disk free low",
                f"Repo partition free={free_gb:.1f}GB / total={total_gb:.1f}GB "
                f"({pct_free:.1f}%) — below {DISK_FREE_GB_THRESHOLD}GB threshold "
                f"({n_fail} consecutive ticks). Prune logs/ artifacts/.",
                priority="high",
            )
    elif DISK_FAIL_FILE.exists():
        try:
            DISK_FAIL_FILE.unlink()
        except Exception:
            pass
    return {"free_gb": round(free_gb, 1), "total_gb": round(total_gb, 1),
            "pct_free": round(pct_free, 1), "alert": alert}


def _check_tailscale() -> dict:
    """Return {status, peers_online, alert}. ntfy if tailscale appears down."""
    try:
        out = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True, text=True, timeout=8,
        )
        if out.returncode != 0:
            return {"status": "unreachable", "peers_online": 0, "alert": True}
        data = json.loads(out.stdout)
        # BackendState 'Running' = OK; 'Stopped' / 'NoState' = down
        backend = data.get("BackendState", "Unknown")
        peers = data.get("Peer", {}) or {}
        online = sum(1 for p in peers.values() if p.get("Online"))
        alert = backend != "Running"
        result = {"status": backend, "peers_online": online, "alert": alert}
    except FileNotFoundError:
        # tailscale CLI not installed — skip silently
        return {"status": "tailscale_cli_absent", "peers_online": 0, "alert": False}
    except Exception as e:
        result = {"status": f"probe_error:{type(e).__name__}", "peers_online": 0, "alert": True}

    if result["alert"]:
        try:
            n_fail = int(TAILSCALE_FAIL_FILE.read_text().strip()) if TAILSCALE_FAIL_FILE.exists() else 0
        except Exception:
            n_fail = 0
        n_fail += 1
        try:
            TAILSCALE_FAIL_FILE.write_text(str(n_fail))
        except Exception:
            pass
        if n_fail >= 3:  # 3 × 5min = 15min sustained
            _push_ntfy(
                "Tailscale state non-running",
                f"BackendState={result['status']} peers_online={result['peers_online']} "
                f"({n_fail} consecutive). Myriad SSH chain at risk.",
                priority="high",
            )
    elif TAILSCALE_FAIL_FILE.exists():
        try:
            TAILSCALE_FAIL_FILE.unlink()
        except Exception:
            pass
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=24, help="lookback window (default 24h)")
    parser.add_argument("--print", action="store_true", help="print JSON to stdout (default also writes to logs/cron/error_scan.json)")
    parser.add_argument("--skip-system-checks", action="store_true",
                        help="skip disk + tailscale probes (audit E)")
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

    # Audit (E) 2026-05-09: system-level health probes. Run once per
    # tick (5min cron); ntfy is rate-limited internally via consecutive-
    # fail counters in _check_disk / _check_tailscale.
    system_health = {}
    if not args.skip_system_checks:
        system_health = {
            "disk": _check_disk(),
            "tailscale": _check_tailscale(),
        }

    payload = {
        "scanned_at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "lookback_hours": args.hours,
        "n_files_scanned": len(candidates),
        "n_errors": len(all_hits),
        "errors": all_hits,
        "system_health": system_health,
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
