#!/usr/bin/env python3
"""Query the anti-redo ledger — "has this already been measured / adjudicated?"

The chronicle is ~20k lines over ~400 sections, append-only. grep only works when
you already guess the right keyword, which is precisely what you cannot do when
the question is "did someone already do this?". This is the lookup layer.

Built 2026-07-28 (REBUILD_PLAN Phase 0) after a session redid already-completed
work five times in one sitting.

    known.py oracle                     # anything mentioning "oracle"
    known.py --type MEASURED self_drop  # only measurements
    known.py --section 302              # everything from §302 and §302.x
    known.py --flagged                  # records a retraction later named
    known.py --absent                   # records whose artifact is really gone
    known.py --stats

Output ALWAYS carries `caveats` and any retraction flag. Dropping a caveat is
this ledger's most dangerous failure mode: a scope-free number invites exactly
the misuse the ledger exists to prevent.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

LEDGER = Path(__file__).resolve().parents[2] / "docs" / "reference" / "known" / "ledger.jsonl"

# per-type: which fields hold the "headline" and which hold supporting detail
HEAD = {
    "MEASURED": ("quantity", "value", "scope"),
    "ADJUDICATED": ("decision", "reasoning", "date"),
    "RETRACTED": ("former_claim", "why_dead", "replaced_by"),
    "DATA": ("what", "path", "grade"),
    "CLAIM_UNVERIFIED": ("claim", "why_unverified", None),
}
C = {"hdr": "\033[1m", "dim": "\033[2m", "warn": "\033[33m",
     "bad": "\033[31m", "ok": "\033[32m", "off": "\033[0m"}


def load() -> list[dict]:
    if not LEDGER.exists():
        sys.exit(f"ledger not found: {LEDGER}\n"
                 f"rebuild with docs/reference/known/rebuild_ledger.py")
    out = []
    for line in LEDGER.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def blob(r: dict) -> str:
    return " ".join(str(v) for k, v in r.items()
                    if k != "_chunk" and isinstance(v, (str, int, float)))


def show(r: dict, color: bool = True) -> None:
    c = C if color else {k: "" for k in C}
    t = r.get("type", "?")
    h = HEAD.get(t, ("quantity", "value", "scope"))
    sect = r.get("source_section", "?")

    print(f"{c['hdr']}[{t}] {sect}{c['off']}  {r.get(h[0]) or ''}")
    if h[1] and r.get(h[1]):
        print(f"    {c['ok']}{h[1]}{c['off']}: {r[h[1]]}")
    if h[2] and r.get(h[2]):
        print(f"    {h[2]}: {r[h[2]]}")

    # caveats are never optional
    if r.get("caveats"):
        print(f"    {c['warn']}caveats{c['off']}: {r['caveats']}")
    if r.get("superseded_by"):
        print(f"    {c['bad']}SUPERSEDED BY {r['superseded_by']}{c['off']}")
    for f in r.get("_cross_chunk_flags", []):
        print(f"    {c['bad']}⚑ {f}{c['off']}")

    src = r.get("source_artifact") or r.get("path") or r.get("recorded_where")
    if src:
        ex = r.get("artifact_exists")
        mark = {True: "✓", False: "✗ ABSENT"}.get(ex, "?")
        print(f"    {c['dim']}{mark} {src}{c['off']}")
    if r.get("_repair", "").startswith(("MOVED", "CONFIRMED")):
        print(f"    {c['dim']}{r['_repair']}{c['off']}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="*", help="terms; ALL must appear (case-insensitive)")
    ap.add_argument("--type", "-t", action="append",
                    choices=list(HEAD), help="filter by record type (repeatable)")
    ap.add_argument("--section", "-s", help="section number, e.g. 302 (covers §302.x)")
    ap.add_argument("--flagged", action="store_true",
                    help="only records a later retraction named")
    ap.add_argument("--absent", action="store_true",
                    help="only records whose artifact is confirmed gone")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--limit", "-n", type=int, default=25)
    ap.add_argument("--no-color", action="store_true")
    a = ap.parse_args()

    recs = load()

    if a.stats:
        from collections import Counter
        print(f"{len(recs)} records  ({LEDGER})")
        for k, v in sorted(Counter(r.get("type") for r in recs).items(),
                           key=lambda kv: -kv[1]):
            print(f"  {k:18s} {v}")
        print(f"  {'flagged':18s} {sum(1 for r in recs if r.get('_cross_chunk_flags'))}")
        print(f"  {'artifact absent':18s} "
              f"{sum(1 for r in recs if str(r.get('_repair','')).startswith('CONFIRMED'))}")
        return 0

    hits = recs
    if a.type:
        hits = [r for r in hits if r.get("type") in a.type]
    if a.section:
        n = a.section.lstrip("§")
        pat = re.compile(rf"§?\s*{re.escape(n)}(\.\d+)?\b")
        hits = [r for r in hits if pat.search(str(r.get("source_section", "")))]
    if a.flagged:
        hits = [r for r in hits if r.get("_cross_chunk_flags")]
    if a.absent:
        hits = [r for r in hits
                if str(r.get("_repair", "")).startswith("CONFIRMED")]
    for term in a.query:
        tl = term.lower()
        hits = [r for r in hits if tl in blob(r).lower()]

    # retracted and superseded first — if something is dead you want to see that
    # before you read the live records and start building on one of them
    order = {"RETRACTED": 0, "CLAIM_UNVERIFIED": 1}
    hits.sort(key=lambda r: (0 if r.get("superseded_by") or r.get("_cross_chunk_flags")
                             else 1, order.get(r.get("type"), 2)))

    print(f"{len(hits)} match(es)"
          f"{f', showing {a.limit}' if len(hits) > a.limit else ''}\n")
    for r in hits[:a.limit]:
        show(r, color=not a.no_color)
    return 0


if __name__ == "__main__":
    sys.exit(main())
