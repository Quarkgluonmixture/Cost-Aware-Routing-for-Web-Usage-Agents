#!/usr/bin/env python3
"""Verify the ledger against the chronicle it was extracted from.

The ledger is an LLM extraction of docs/checkpoints/实验笔记.md. It can be wrong
in three ways, in increasing order of danger:

  1. cites a § that does not exist            -> catchable, checked here
  2. reports a number absent from that §      -> catchable, checked here
  3. faithfully copies a number the chronicle got wrong -> NOT catchable here;
     needs recomputation from the run artifacts

This script does 1 and 2. It cannot do 3, and no amount of cross-checking the
ledger against the chronicle ever will — they share a source.

    verify_ledger.py                # summary
    verify_ledger.py --show-fail    # every record whose numbers are unsupported
    verify_ledger.py --section 302  # audit one section
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LEDGER = ROOT / "docs" / "reference" / "known" / "ledger.jsonl"
CHRONICLE = ROOT / "docs" / "checkpoints" / "实验笔记.md"

# a "checkable number": >=2 chars so we don't chase 0/1/2 noise, keeps decimals,
# percentages and pp values which is what the ledger's claims actually rest on
WHOLE = ""
ANALYSIS_DOCS = ""
NUM = re.compile(r"\d+\.\d+|\d{2,}")
# ledger fields that carry factual payload (not commentary)
VALUE_FIELDS = ("value", "quantity", "former_claim", "claim", "what", "path")


def load_sections() -> dict[tuple[int, int], str]:
    """(major, minor) -> that section's raw text. minor=0 means the '## N.' body."""
    lines = CHRONICLE.read_text(encoding="utf-8").splitlines()
    heads: list[tuple[int, tuple[int, int]]] = []
    for i, ln in enumerate(lines):
        m = re.match(r"^##\s+(\d+)\.", ln)
        if m:
            heads.append((i, (int(m.group(1)), 0)))
            continue
        m = re.match(r"^###\s+(\d+)\.(\d+)", ln)
        if m:
            heads.append((i, (int(m.group(1)), int(m.group(2)))))
    out: dict[tuple[int, int], str] = {}
    for idx, (start, key) in enumerate(heads):
        end = heads[idx + 1][0] if idx + 1 < len(heads) else len(lines)
        out[key] = "\n".join(lines[start:end])
    return out


def parse_refs(s: str) -> list[tuple[int, int]]:
    """'§302.1 §304' -> [(302,1),(304,0)]. Handles bare and multi refs."""
    if not isinstance(s, str):
        return []
    return [(int(a), int(b or 0))
            for a, b in re.findall(r"§?\s*(\d+)(?:\.(\d+))?", s)]


def section_text(secs: dict, ref: tuple[int, int]) -> str | None:
    """Text for a ref. A parent §N includes all its §N.x children."""
    if ref in secs:
        body = secs[ref]
        if ref[1] == 0:
            body += "\n" + "\n".join(v for k, v in secs.items()
                                     if k[0] == ref[0] and k[1] > 0)
        return body
    if ref[1] > 0 and (ref[0], 0) in secs:  # subsection missing -> try parent
        return "\n".join(v for k, v in secs.items() if k[0] == ref[0])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-fail", action="store_true")
    ap.add_argument("--section", help="audit one section number")
    ap.add_argument("--limit", type=int, default=15)
    a = ap.parse_args()

    global WHOLE, ANALYSIS_DOCS
    secs = load_sections()
    WHOLE = CHRONICLE.read_text(encoding="utf-8")
    # sections that are one-line pointers send the reader to these
    ANALYSIS_DOCS = "\n".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for p in sorted((ROOT / "docs" / "analysis").rglob("*.md"))
    )
    recs = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    print(f"chronicle: {len(secs)} sections, {len(WHOLE):,} chars    "
          f"analysis docs: {len(ANALYSIS_DOCS):,} chars    ledger: {len(recs)} records\n")

    if a.section:
        want = parse_refs(a.section)[0]
        recs = [r for r in recs
                if any(x[0] == want[0] for x in parse_refs(r.get("source_section", "")))]
        print(f"filtered to §{a.section}: {len(recs)} records\n")

    stat = Counter()
    failures = []
    for r in recs:
        refs = parse_refs(r.get("source_section", ""))
        if not refs:
            stat["no_section_cited"] += 1
            continue
        texts = [t for t in (section_text(secs, x) for x in refs) if t]
        if not texts:
            stat["section_NOT_FOUND"] += 1
            failures.append((r, "cited § does not exist in chronicle", []))
            continue
        pool = "\n".join(texts)

        nums = set()
        for f in VALUE_FIELDS:
            v = r.get(f)
            if isinstance(v, str):
                nums.update(NUM.findall(v))
        if not nums:
            stat["no_numbers_to_check"] += 1
            continue

        missing = [n for n in nums if n not in pool]
        if not missing:
            stat["in_cited_section"] += 1
            continue
        # Not in the cited § is NOT the same as fabricated. The chronicle keeps a
        # classification index at the top of the file and many [finding] sections
        # are one-line pointers by design (CLAUDE.md writing rules), so the number
        # legitimately lives elsewhere. Only absence from the WHOLE chronicle —
        # and from the analysis docs it points at — is evidence of invention.
        still = [n for n in missing if n not in WHOLE]
        if not still:
            stat["elsewhere_in_chronicle"] += 1
            failures.append((r, f"{len(missing)} number(s) outside cited § "
                                f"but present in chronicle (imprecise citation)", missing))
        else:
            outside = [n for n in still if n not in ANALYSIS_DOCS]
            if not outside:
                stat["in_analysis_docs"] += 1
                failures.append((r, f"{len(still)} number(s) only in analysis docs "
                                    f"(pointer section)", still))
            else:
                stat["NOT_FOUND_ANYWHERE"] += 1
                failures.append((r, "⚠ number absent from chronicle AND analysis docs",
                                 sorted(outside)))

    total = sum(stat.values())
    print("verification against cited sections:")
    for k in ("in_cited_section", "elsewhere_in_chronicle", "in_analysis_docs",
              "NOT_FOUND_ANYWHERE", "section_NOT_FOUND", "no_numbers_to_check",
              "no_section_cited"):
        if stat[k]:
            print(f"  {k:26s} {stat[k]:5d}   {stat[k]/total*100:5.1f}%")

    checked = (stat["in_cited_section"] + stat["elsewhere_in_chronicle"]
               + stat["in_analysis_docs"] + stat["NOT_FOUND_ANYWHERE"])
    if checked:
        traceable = checked - stat["NOT_FOUND_ANYWHERE"]
        print(f"\n  {checked} records carry checkable numbers")
        print(f"  {traceable} ({traceable/checked*100:.1f}%) trace to a real number in the source docs")
        print(f"  {stat['NOT_FOUND_ANYWHERE']} ({stat['NOT_FOUND_ANYWHERE']/checked*100:.1f}%) "
              f"UNTRACEABLE -> possible extraction error, review these first")

    if a.show_fail:
        print(f"\n--- {min(len(failures), a.limit)} of {len(failures)} failures ---")
        for r, why, missing in failures[:a.limit]:
            head = (r.get("quantity") or r.get("claim") or r.get("decision")
                    or r.get("former_claim") or r.get("what") or "")[:70]
            print(f"\n[{r.get('type')}] {r.get('source_section')}  {why}")
            print(f"    {head}")
            if r.get("value"):
                print(f"    value: {str(r['value'])[:110]}")
            if missing:
                print(f"    absent from §: {missing[:8]}")
    else:
        print(f"\n{len(failures)} records need review — rerun with --show-fail")
    return 0


if __name__ == "__main__":
    sys.exit(main())
