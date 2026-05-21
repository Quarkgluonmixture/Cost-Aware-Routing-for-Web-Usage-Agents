#!/usr/bin/env python3
"""Index, lint, and locate entries in master_bug_catalog.md.

The catalog is a single ~7500-line append-only doc organised by /stress cluster
(time/theme), while B-numbers are globally monotonic. The two orderings conflict,
so new entries occasionally land out of B-number order. This tool gives the catalog
a lint + lookup layer WITHOUT changing its narrative structure (the cheap "索引+约定"
option, not a data-layer migration).

Modes:
  --lint            report monotonicity inversions, duplicate definitions, gaps;
                    exit 1 if NEW (non-grandfathered) inversions appear.
  --write           regenerate docs/reference/master_bug_catalog_index.md
                    (lint summary + section -> B-number map). Idempotent.
  --find N [N ...]  print where B-N is defined (line + enclosing ## section).

Definition points — TWO tiers, which is the whole trick:
  * STRONG def = first occurrence as `### B-N.` (early format) or `**B-N**` (bold,
    table/list format). These are the canonical entries. Monotonicity + the section
    map use ONLY these.
  * Bare inline `B-N` (e.g. "closed via B-991") is a cross-REFERENCE, not a
    definition. Heavily cross-referenced numbers (B-991!) appear early as refs; if
    those counted as definitions they would poison running-max and flag every later
    normal number as an inversion. So inline hits are used only as a `--find`
    fallback for numbers that never get a strong entry, and for gap detection.
  `int()` normalises the early `B-01` zero-padding to `B-1`.

Maintenance convention (enforced by --lint going forward):
  * append new entries to the LAST section so B-numbers stay monotonic;
  * for a follow-up to an old bug, use a NEW number at the end + cross-link the
    old number in prose — never back-insert into the old section.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "docs/reference/master_bug_catalog.md"
INDEX = ROOT / "docs/reference/master_bug_catalog_index.md"

# B-1 … B-1821 predate the append-monotonic convention (this tool's introduction,
# 2026-05-21). Their out-of-order canonical defs are grandfathered: several /stress
# batches (A1.13/A1.14/A1.5 …) were written chunk-reverse (d→a) or back-filled, so
# 142 canonical entries are out of B-number order. Physically reordering them would
# churn git blame + break "nearby cross-link" reading, so --lint reports the count
# but won't fail. B-1822 onward MUST append monotonically (new inversions fail).
GRANDFATHERED: frozenset[int] = frozenset(range(1, 1822))

_STRONG_RE = re.compile(r"^###\s+B-0*(\d+)\b|\*\*B-0*(\d+)\*\*")
_HEADER_RE = re.compile(r"^##\s+(.*\S)\s*$")
_ANY_RE = re.compile(r"\bB-0*(\d+)\b")


def parse(lines: list[str]):
    """Return (strong, alld, sections).

    strong:   dict  B-number -> first STRONG definition line (canonical entries)
    alld:     dict  B-number -> first occurrence line (strong, else inline ref)
    sections: list  (line_idx, title) for every `## ` header
    """
    sections = [(i, m.group(1)) for i, ln in enumerate(lines)
                if (m := _HEADER_RE.match(ln))]
    strong: dict[int, int] = {}
    for i, ln in enumerate(lines):
        for m in _STRONG_RE.finditer(ln):
            strong.setdefault(int(m.group(1) or m.group(2)), i)
    alld = dict(strong)
    for i, ln in enumerate(lines):
        for m in _ANY_RE.finditer(ln):
            alld.setdefault(int(m.group(1)), i)
    return strong, alld, sections


def section_of(line_idx: int, sections: list[tuple[int, str]]) -> str:
    """Title of the nearest `## ` section at or above line_idx."""
    title = "(front matter)"
    for sline, stitle in sections:
        if sline <= line_idx:
            title = stitle
        else:
            break
    return title


def _bold_defs(lines: list[str]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for i, ln in enumerate(lines):
        for m in re.finditer(r"\*\*B-0*(\d+)\*\*", ln):
            out.setdefault(int(m.group(1)), []).append(i)
    return out


def _compact_ranges(nums: list[int]) -> str:
    """[1,2,3,7,8] -> 'B-1–B-3, B-7–B-8'."""
    if not nums:
        return "(none)"
    nums = sorted(nums)
    parts, lo, prev = [], nums[0], nums[0]
    for n in nums[1:] + [None]:
        if n is not None and n == prev + 1:
            prev = n
            continue
        parts.append(f"B-{lo}" if lo == prev else f"B-{lo}–B-{prev}")
        if n is not None:
            lo = prev = n
    return ", ".join(parts)


def analyse(lines: list[str]):
    strong, alld, sections = parse(lines)
    by_line = sorted(strong.items(), key=lambda kv: kv[1])  # canonical defs, file order
    inversions, rmax, rnum = [], 0, None
    for num, line in by_line:
        if num < rmax:
            inversions.append((num, line, rnum))
        else:
            rmax, rnum = num, num
    dups = {k: v for k, v in _bold_defs(lines).items() if len(v) > 1}
    gaps = [n for n in range(1, max(alld) + 1) if n not in alld] if alld else []
    return strong, alld, sections, inversions, dups, gaps


def cmd_lint(_args) -> int:
    lines = CATALOG.read_text().splitlines()
    strong, alld, sections, inversions, dups, gaps = analyse(lines)
    new_inv = [iv for iv in inversions if iv[0] not in GRANDFATHERED]
    print(f"catalog: {len(lines)} lines, {len(sections)} ## sections")
    print(f"B-numbers: {len(strong)} canonical entries (### / **bold**), "
          f"{len(alld)} total referenced, range B-{min(alld)}~B-{max(alld)}")
    print(f"\nmonotonicity inversions (canonical entries only): {len(inversions)} total, "
          f"{len(inversions) - len(new_inv)} grandfathered (≤B-1821 pre-convention), "
          f"{len(new_inv)} NEW")
    for num, line, after in new_inv:
        print(f"  ** NEW ** B-{num} @ L{line + 1} defined after B-{after} "
              f"— {section_of(line, sections)[:55]}")
    print(f"\nduplicate bold defs: {len(dups)} (summary-table echo + detail-list; not bugs)")
    for k in sorted(dups):
        print(f"  B-{k}: L{[l + 1 for l in dups[k]]}")
    print(f"\nnumber gaps: {len(gaps)}" + (f" — {_compact_ranges(gaps)}" if gaps else ""))
    if new_inv:
        print(f"\n✗ FAIL: {len(new_inv)} NEW inversion(s). Convention: append to the "
              f"LAST section + cross-link old numbers; never back-insert.")
        return 1
    print("\n✓ PASS: no new inversions (canonical B-number order monotonic going forward).")
    return 0


def cmd_find(args) -> int:
    lines = CATALOG.read_text().splitlines()
    strong, alld, sections = parse(lines)
    rc = 0
    for tok in args.find:
        n = int(re.sub(r"\D", "", str(tok)) or "0")
        if n in alld:
            line = alld[n]
            kind = "entry" if n in strong else "inline-ref-only"
            print(f"B-{n}  →  L{line + 1}  [{kind}]  §{section_of(line, sections)}")
            print(f"        {lines[line].strip()[:100]}")
        else:
            print(f"B-{n}  →  NOT FOUND")
            rc = 1
    return rc


def cmd_write(_args) -> int:
    lines = CATALOG.read_text().splitlines()
    strong, alld, sections, inversions, dups, gaps = analyse(lines)
    new_inv = [iv for iv in inversions if iv[0] not in GRANDFATHERED]
    today = _dt.date.today().isoformat()

    rows = []
    for sline, stitle in sections:
        nums = sorted(n for n, l in strong.items() if section_of(l, sections) == stitle)
        if nums:
            rows.append((sline, stitle, nums))

    out = [
        "# Master Bug Catalog — Index (auto-generated)",
        "",
        "> ⚠️ **Auto-generated** by `scripts/maintenance/index_bug_catalog.py --write`. "
        "勿手改 — regenerate after adding entries.",
        "> 🔎 Locate a number: `python3 scripts/maintenance/index_bug_catalog.py --find 1810`.",
        "> 🩺 Health check: `python3 scripts/maintenance/index_bug_catalog.py --lint` (before commit).",
        "",
        f"**Generated**: {today}  ·  **Catalog**: master_bug_catalog.md "
        f"({len(lines)} lines, {len(sections)} sections)  ·  "
        f"**B-numbers**: {len(strong)} canonical entries, {len(alld)} referenced, "
        f"range B-{min(alld)}~B-{max(alld)}",
        "",
        "## Health (lint)",
        "",
        f"- monotonicity inversions (canonical entries): **{len(inversions)}** total "
        f"({len(inversions) - len(new_inv)} grandfathered pre-2026-05-21, **{len(new_inv)} new**)",
        f"- duplicate bold defs: **{len(dups)}** (summary-table echo + detail-list; not bugs)",
        f"- number gaps: **{len(gaps)}**" + (f" — {_compact_ranges(gaps)}" if gaps else ""),
        "",
    ]
    if new_inv:
        out += ["<details><summary>new (non-grandfathered) inversion detail</summary>", ""]
        for num, line, after in new_inv:
            out.append(f"- B-{num} @ L{line + 1} after B-{after} "
                       f"— {section_of(line, sections)[:60]}")
        out += ["", "</details>", ""]

    out += [
        "## Section map (file order — find a number by its /stress cluster)",
        "",
        "| Line | Section | B-numbers (canonical entries) |",
        "|---|---|---|",
    ]
    for sline, stitle, nums in rows:
        title = stitle if len(stitle) <= 70 else stitle[:67] + "…"
        out.append(f"| L{sline + 1} | {title} | {_compact_ranges(nums)} |")
    out.append("")

    INDEX.write_text("\n".join(out))
    print(f"wrote {INDEX.relative_to(ROOT)} "
          f"({len(rows)} sections, {len(strong)} canonical entries, {len(new_inv)} new inversions)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--lint", action="store_true", help="report inversions / dups / gaps")
    g.add_argument("--write", action="store_true", help="regenerate the index file")
    g.add_argument("--find", nargs="+", metavar="N", help="locate B-number(s)")
    args = ap.parse_args()
    if args.lint:
        return cmd_lint(args)
    if args.write:
        return cmd_write(args)
    return cmd_find(args)


if __name__ == "__main__":
    sys.exit(main())
