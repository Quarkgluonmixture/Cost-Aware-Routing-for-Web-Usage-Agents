#!/usr/bin/env python3
"""B-1957 — probe every task's start_url against the live substrate.

Answers one question with evidence instead of judgement: *which tasks name a
substrate resource that is not there?* Written for the AMENDMENT_10 candidate
rule ("start_url references a resource returning 404, not auth-induced, not
repairable"), whose whole warrant is that it was evaluated over EVERY
benchmark-site rather than hand-picked to catch one task.

Run it on a host that can reach the VWA/WA containers (the A100, not the DGX).

    python scripts/maintenance/probe_starturl_availability.py --out probe.tsv

TWO TRAPS THIS SCRIPT EXISTS TO AVOID — both were hit on 2026-08-05 and both
would have silently inflated the exclusion set by ~24 tasks:

  1. CONCURRENCY FALSE POSITIVES. A first pass at `-P 6` returned curl 000 for
     18 classifieds URLs; re-probed serially every one was 200 in 0.13-0.19s.
     The classifieds container is single-threaded PHP and queues under load.
     ⇒ this script is SERIAL by default. `--jobs` exists but warns.

  2. AUTH FALSE POSITIVES. Magento's admin returns 404 (not 302) for every URL
     when unauthenticated — including its own login page. A bare curl therefore
     reports "missing" for pages that exist and that the runner reaches fine
     with its `storage_state`. ⇒ every 404 host is re-probed with a control URL
     that must exist; if the control is also 404, the whole host's findings are
     flagged AUTH_SUSPECT rather than reported as missing resources.

A 404 only means "this task's start_url is unreachable" once its host's control
probe passes. Everything else needs a human.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Endpoints as the paper-grade host serves them (A100 self-hosted).
_DEFAULT_ENDPOINTS = {
    "REDDIT": "http://localhost:9999",
    "SHOPPING": "http://localhost:7770",
    "SHOPPING_ADMIN": "http://localhost:7780",
    "WIKIPEDIA": "http://localhost:8888",
    "CLASSIFIEDS": "http://localhost:9980",
    "HOMEPAGE": "http://localhost:4399",
}

CONFIGS = [
    ("visualwebarena", "classifieds", "config_files/vwa/test_classifieds.raw.json"),
    ("visualwebarena", "reddit", "config_files/vwa/test_reddit.raw.json"),
    ("visualwebarena", "shopping", "config_files/vwa/test_shopping.raw.json"),
    ("webarena", "reddit", "config_files/wa/test_reddit.raw.json"),
    ("webarena", "shopping", "config_files/wa/test_shopping.raw.json"),
    ("webarena", "shopping_admin", "config_files/wa/test_shopping_admin.raw.json"),
]

# Per-host URL that MUST resolve if the host is healthy and the probe is
# authorised. A 404 here means the probe cannot see this host properly (auth,
# routing) — so nothing on that host may be reported as a missing resource.
CONTROL_URLS = {
    "localhost:7770": "/",
    "localhost:9999": "/",
    "localhost:9980": "/",
    "localhost:8888": "/",
    # Magento admin 404s everything (even /admin) without a session — its
    # control is expected to FAIL, which is exactly the point: findings on
    # 7780 are auth-suspect by construction and must not drive exclusions.
    "localhost:7780": "/admin",
}


def _curl(url: str, timeout: int) -> str:
    try:
        proc = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
             "--max-time", str(timeout), url],
            capture_output=True, text=True, timeout=timeout + 5,
        )
        return (proc.stdout or "000").strip()
    except Exception:
        return "000"


def build_manifest():
    for key, val in _DEFAULT_ENDPOINTS.items():
        os.environ.setdefault(key, val)
    from p79.experiment.tasks import _placeholder_mapping

    mapping = _placeholder_mapping()

    def substitute(text: str) -> str:
        out = text
        for key, val in mapping.items():
            out = out.replace(key, val)
        return out

    rows = []
    for benchmark, site, rel in CONFIGS:
        path = REPO / "external/visualwebarena" / rel
        if not path.exists():
            print(f"# MISSING config {path}", file=sys.stderr)
            continue
        for t in json.load(open(path, encoding="utf-8")):
            raw = str(t.get("start_url", "") or "")
            if not raw:
                continue
            for idx, part in enumerate(substitute(raw).split("|AND|")):
                url = part.strip()
                if url.startswith("http"):
                    rows.append((benchmark, site, int(t["task_id"]), idx, url))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="starturl_probe.tsv")
    ap.add_argument("--timeout", type=int, default=15)
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel probes; >1 produces false 000s on the "
                         "single-threaded classifieds container (see docstring)")
    args = ap.parse_args()

    if args.jobs > 1:
        print(f"WARNING: --jobs={args.jobs} — concurrent probing produced 18 "
              f"spurious 000s on classifieds on 2026-08-05. Serial is the "
              f"trustworthy setting.", file=sys.stderr)

    rows = build_manifest()
    urls = sorted({r[4] for r in rows})
    print(f"[probe] {len(rows)} start_url references / {len(urls)} unique URLs",
          file=sys.stderr)

    codes = {}
    for i, url in enumerate(urls, 1):
        codes[url] = _curl(url, args.timeout)
        if i % 50 == 0:
            print(f"[probe] {i}/{len(urls)}", file=sys.stderr)

    # Control probes decide whether a 404 on a host is even interpretable.
    host_ok = {}
    for host, ctl in CONTROL_URLS.items():
        code = _curl(f"http://{host}{ctl}", args.timeout)
        host_ok[host] = code.startswith(("2", "3"))
        print(f"[probe] control {host}{ctl} = {code} "
              f"({'usable' if host_ok[host] else 'AUTH_SUSPECT'})", file=sys.stderr)

    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        f.write("benchmark\tsite\ttask_id\turl_index\thttp_code\tverdict\turl\n")
        for benchmark, site, task_id, idx, url in rows:
            code = codes.get(url, "000")
            host = url.split("/")[2] if "//" in url else ""
            if code.startswith(("2", "3")):
                verdict = "ok"
            elif not host_ok.get(host, True):
                verdict = "AUTH_SUSPECT"
            elif code == "000":
                verdict = "UNREACHABLE"
            else:
                verdict = "MISSING_RESOURCE"
            f.write(f"{benchmark}\t{site}\t{task_id}\t{idx}\t{code}\t{verdict}\t{url}\n")

    verdicts = Counter()
    flagged = defaultdict(list)
    for benchmark, site, task_id, idx, url in rows:
        code = codes.get(url, "000")
        host = url.split("/")[2] if "//" in url else ""
        if code.startswith(("2", "3")):
            verdicts["ok"] += 1
        elif not host_ok.get(host, True):
            verdicts["AUTH_SUSPECT"] += 1
            flagged["AUTH_SUSPECT"].append((benchmark, site, task_id, url))
        elif code == "000":
            verdicts["UNREACHABLE"] += 1
            flagged["UNREACHABLE"].append((benchmark, site, task_id, url))
        else:
            verdicts["MISSING_RESOURCE"] += 1
            flagged["MISSING_RESOURCE"].append((benchmark, site, task_id, url))

    print(f"\n[probe] verdicts: {dict(verdicts)}")
    for label in ("MISSING_RESOURCE", "UNREACHABLE", "AUTH_SUSPECT"):
        if flagged[label]:
            print(f"\n=== {label} ===")
            for benchmark, site, task_id, url in flagged[label]:
                print(f"  {benchmark}/{site}/{task_id}  {url}")
    print(f"\n[probe] wrote {out}")
    print("[probe] ONLY MISSING_RESOURCE rows are candidate exclusions. "
          "UNREACHABLE = re-probe serially before believing it. "
          "AUTH_SUSPECT = the probe cannot see that host; says nothing about "
          "whether the resource exists.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
