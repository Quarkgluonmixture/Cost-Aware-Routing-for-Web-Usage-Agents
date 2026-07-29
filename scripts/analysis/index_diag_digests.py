#!/usr/bin/env python3
"""Index the /diag failure-attribution digests so the conclusion layer covers them.

`INDEX.md §2` records the gap: 41 per-condition digests under `docs/analysis/vwa_*/`,
far fewer distilled ledger records, and therefore the instruction that failure
analysis "must read the digests directly". Correct instruction, bad situation — a
corpus nothing indexes is a corpus that gets skipped under deadline.

WHAT THIS IS NOT
----------------
It is not an extraction of the digests' findings. Each carries 100-300 lines of
per-rule detail, Tier-2 deep dives and P-rule false-positive audits that only
matter while you are reading them. This builds the *navigation* layer: which
condition, how much was actually adjudicated, and where the non-clean ones are.

WHY THE COVERAGE FIELD IS THE POINT (B-1913, 2026-07-29)
--------------------------------------------------------
The first version of this script pattern-matched one table shape, parsed 2 of 41
digests, treated every unparsed digest as zeroes, and printed "**None.** Every
indexed condition attributes 100% of its failures to agent capability, with zero
scaffold bugs and zero evaluator false positives." That was false in both
directions: `B1_som_classifieds` states benchmark-FP ~1.5%, and
`B2_vision_reddit` contains a success-side benchmark-FP (task 160 / B-1889).

Worse, `B2_vision_reddit_diag_digest.md:60` had already written the warning the
script then violated:

    ⚠️ 本 digest 的三分类**不完整** —— 未深挖不等于「无 scaffold-bug /
    无 benchmark-FP」，只代表本轮没有查。请勿据此下「pipeline 干净」结论。

So the index distinguishes three states per digest — `parsed`, `unparsed`,
`self_declared_incomplete` — and refuses to emit a corpus-level "clean" verdict
unless every digest parsed. Absence of evidence is reported as absence of
evidence.

Usage:
  .venv/bin/python3 scripts/analysis/index_diag_digests.py \
      --out docs/analysis/cross_sites/diag_digest_index.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DIGEST_GLOB = "docs/analysis/vwa_*/*_diag_digest.md"

CLASSES = ("agent-limit", "scaffold-bug", "benchmark-fp", "unclear")

# The digests use two different §1 layouts, from different eras of the /diag skill.
#
# LAYOUT A — one table row per class:
#     | **agent-limit** | **175** | **100%** | ...
# LAYOUT B — the whole split compressed into a single cell:
#     | 三分类 (29 深挖) | **agent-limit 100%**（17 failed 深挖 = 17 agent-limit
#     + **0 scaffold-bug + 0 benchmark-FP**）· 12 success-hit = ... |
#
# The first version of this script only knew layout A, parsed 3 of 41, and called
# the other 38 unreadable. Most of them are layout B and perfectly readable.
ROW = re.compile(
    r"^\|\s*\*{0,2}(agent-limit|scaffold-bug|benchmark-FP|unclear)\*{0,2}\s*\|"
    r"\s*\*{0,2}([\d,]+)\*{0,2}\s*\|\s*\*{0,2}([\d.]+)\s*%",
    re.IGNORECASE | re.MULTILINE)
# layout B: "<N> agent-limit", "0 scaffold-bug", "0 benchmark-FP" anywhere in the
# 三分类 line. Counts only, no percentage — the denominator is stated separately.
INLINE = re.compile(
    r"\*{0,2}(\d+)\*{0,2}\s*\*{0,2}(agent-limit|scaffold-bug|benchmark-FP)\*{0,2}",
    re.IGNORECASE)
# LAYOUT C — same idea, class first: "三分类：agent-limit 5 · benchmark-FP 2 · scaffold-bug 0"
INLINE_REV = re.compile(
    r"\*{0,2}(agent-limit|scaffold-bug|benchmark-FP)\*{0,2}\s*[:：]?\s*\*{0,2}(\d+)\b",
    re.IGNORECASE)
TRICLASS_LINE = re.compile(r"^.*三分类.*$", re.MULTILINE)
# LAYOUT D — table row whose count column is prose:
#   | **agent-limit** | ~100% (221/221 failed) | ... |
#   | scaffold-bug    | 0                      | ... |
ROW_LOOSE = re.compile(
    r"^\|\s*\*{0,2}(agent-limit|scaffold-bug|benchmark-FP|unclear)\*{0,2}\s*\|"
    r"\s*([^|]{1,60})\|",
    re.IGNORECASE | re.MULTILINE)
FIRST_INT = re.compile(r"(\d[\d,]*)")

# A pointer digest carries no numbers ON PURPOSE and forwards to run-specific
# digests (dual-run preservation, 笔记 §297-298). Counting it as "unparsed"
# conflates "deliberately empty" with "failed to read".
POINTER = re.compile(r"指针文件|不含数字|digest pointer", re.IGNORECASE)
HEADLINE = re.compile(r"^>\s*\*{0,2}Headline\*{0,2}\s*[:：]?\s*(.+)$", re.MULTILINE)
FAILED_N = re.compile(r"failed\s*\((\d+)\)", re.IGNORECASE)

# A digest that says its own attribution is partial. Any of these means the
# absence of a scaffold-bug/benchmark-FP number is "not looked at", not "zero".
INCOMPLETE = re.compile(
    r"三分类\s*\*{0,2}不完整|未深挖不等于|情况未知|Tier-2\s*未做|"
    r"请勿据此下.*干净|未.*深挖|not\s+investigated", re.IGNORECASE)
# A digest that was later completed keeps its original "incomplete" wording as
# history — the Tier-2 addendum sits below it and says so. Without this, the
# index reports a resolved gap as still open forever, purely because the record
# of the gap is (correctly) not deleted.
RESOLVED = re.compile(r"Tier-2\s*补记|Tier-2\s+addendum|本轮补齐", re.IGNORECASE)

# Free-text signals that a non-agent-limit cause is present somewhere in the
# digest even when the structured table is absent. Deliberately over-inclusive:
# a false flag costs a human 30 seconds, a missed one costs a paper claim.
FP_SIGNAL = re.compile(
    r"benchmark-FP\s*[≈~=]?\s*[\d.]+\s*%|benchmark-FP\s+规则|"
    r"B-1889|边缘-FP|评测误判|deterministic FP", re.IGNORECASE)
SCAFFOLD_SIGNAL = re.compile(
    r"scaffold-bug\s*[≈~=]?\s*[1-9]|scaffold-adjacent|framework\s*bug", re.IGNORECASE)


def parse(path: Path) -> dict:
    txt = path.read_text(encoding="utf-8", errors="replace")
    name = path.name.replace("_diag_digest.md", "")
    parts = name.split("_")
    baseline = parts[0]
    site, run = None, None
    if parts[-1].startswith("R") and parts[-1][1:].isdigit():
        run, site = parts[-1], parts[-2]
    elif parts[-1] in ("classifieds", "reddit", "shopping"):
        site = parts[-1]
    mode = "_".join(parts[1:parts.index(site)]) if site and site in parts else None

    counts = {k.lower(): {"n": int(n.replace(",", "")), "pct": float(p)}
              for k, n, p in ROW.findall(txt)}
    layout = "A" if counts else None
    if not counts:
        # layouts B and C — scan the 三分类 line(s) in both orderings
        for line in TRICLASS_LINE.findall(txt):
            for n, k in INLINE.findall(line):
                counts.setdefault(k.lower(), {"n": int(n), "pct": None})
            for k, n in INLINE_REV.findall(line):
                counts.setdefault(k.lower(), {"n": int(n), "pct": None})
        if counts:
            layout = "B/C"
    if not counts:
        # layout D — table row whose count column is prose ("~100% (221/221 failed)")
        for k, col in ROW_LOOSE.findall(txt):
            m = FIRST_INT.search(col)
            if m:
                counts.setdefault(k.lower(),
                                  {"n": int(m.group(1).replace(",", "")), "pct": None,
                                   "raw": col.strip()})
        if counts:
            layout = "D"

    head = HEADLINE.search(txt)
    failed = FAILED_N.search(txt)
    incomplete = INCOMPLETE.search(txt)
    resolved = RESOLVED.search(txt)
    pointer = POINTER.search(txt)
    if resolved:
        incomplete = None  # the gap is recorded but has since been closed

    if pointer and not counts:
        coverage = "pointer"
    elif counts and not incomplete:
        coverage = "parsed"
    elif incomplete:
        coverage = "self_declared_incomplete"
    elif resolved:
        coverage = "parsed"
    else:
        coverage = "unparsed"

    return {
        "condition": name, "baseline": baseline, "mode": mode, "site": site,
        "run": run, "path": str(path.relative_to(REPO)),
        "n_lines": len(txt.splitlines()),
        "n_failed": int(failed.group(1)) if failed else None,
        "attribution": counts,
        "coverage": coverage, "table_layout": layout,
        "incomplete_quote": incomplete.group(0) if incomplete else None,
        "free_text_fp_signal": bool(FP_SIGNAL.search(txt)),
        "free_text_scaffold_signal": bool(SCAFFOLD_SIGNAL.search(txt)),
        "headline": head.group(1).strip() if head else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=REPO / "docs/analysis/cross_sites/diag_digest_index.md")
    ap.add_argument("--json-out", type=Path,
                    default=REPO / "docs/analysis/cross_sites/diag_digest_index.json")
    a = ap.parse_args()

    rows = sorted((parse(p) for p in REPO.glob(DIGEST_GLOB)),
                  key=lambda r: (r["baseline"], r["site"] or "", r["mode"] or ""))
    if not rows:
        raise SystemExit(f"no digests matched {DIGEST_GLOB}")

    n_parsed = sum(1 for r in rows if r["coverage"] == "parsed")
    n_inc = sum(1 for r in rows if r["coverage"] == "self_declared_incomplete")
    n_unp = sum(1 for r in rows if r["coverage"] == "unparsed")
    n_ptr = sum(1 for r in rows if r["coverage"] == "pointer")
    flagged = [r for r in rows
               if r["free_text_fp_signal"] or r["free_text_scaffold_signal"]
               or (r["attribution"].get("scaffold-bug", {}).get("n") or 0) > 0
               or (r["attribution"].get("benchmark-fp", {}).get("n") or 0) > 0]

    def cell(r, key):
        c = r["attribution"].get(key)
        if c:
            # layout B gives counts without a percentage; do not invent one
            return f"{c['n']} ({c['pct']:.0f}%)" if c["pct"] is not None else str(c["n"])
        if r["coverage"] == "pointer":
            return "↪"
        return "**?**" if r["coverage"] != "parsed" else "0"

    COV = {"parsed": "✅", "self_declared_incomplete": "⚠️ 自称不完整",
           "unparsed": "❔ 未解析", "pointer": "↪ 指针(数字在 run digest)"}

    L: list[str] = []
    L.append("# /diag digest index — failure attribution coverage")
    L.append("")
    L.append(f"- **{len(rows)} digests**: {n_parsed} with a readable three-way "
             f"split · {n_inc} that declare their own attribution incomplete · "
             f"{n_ptr} pointer files (numbers live in the run-specific digests "
             f"they forward to) · {n_unp} this script still cannot parse")
    L.append("- built by `scripts/analysis/index_diag_digests.py`")
    L.append("- **navigation layer only** — per-rule detail, Tier-2 deep dives and "
             "P-rule false-positive audits exist solely in the digests")
    L.append("")
    L.append("> ⚠️ **`?` means not looked at, not zero.** An earlier revision of "
             "this index treated unparsed digests as all-zero and concluded the "
             "corpus was clean; two digests contradict that "
             "(`B1_som_classifieds` benchmark-FP ≈1.5%, `B2_vision_reddit` "
             "task 160 / B-1889), and one had already written the warning the "
             "script violated. Coverage is now reported before content.")
    L.append("")
    L.append("Classes: **agent-limit** = model capability · **scaffold-bug** = our "
             "pipeline · **benchmark-FP** = evaluator misjudged.")
    L.append("")
    L.append("## Coverage and attribution")
    L.append("")
    L.append("| baseline | site | mode | coverage | failed | agent-limit | "
             "scaffold-bug | benchmark-FP | free-text flag |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["run"]:
            continue
        flag = []
        if r["free_text_scaffold_signal"]:
            flag.append("scaffold")
        if r["free_text_fp_signal"]:
            flag.append("FP")
        L.append(f"| {r['baseline']} | {r['site']} | {r['mode']} | "
                 f"{COV[r['coverage']]} | "
                 f"{r['n_failed'] if r['n_failed'] is not None else '—'} | "
                 f"{cell(r,'agent-limit')} | {cell(r,'scaffold-bug')} | "
                 f"{cell(r,'benchmark-fp')} | {', '.join(flag) or '—'} |")
    L.append("")

    L.append("## ⚠️ Digests carrying a non-agent-limit signal")
    L.append("")
    L.append("Either a non-zero structured count, or free text naming a "
             "benchmark-FP / scaffold issue. **A failure-analysis section must "
             "read these directly.**")
    L.append("")
    for r in flagged:
        bits = []
        if (r["attribution"].get("scaffold-bug", {}).get("n") or 0) > 0:
            bits.append(f"scaffold-bug {cell(r,'scaffold-bug')}")
        if (r["attribution"].get("benchmark-fp", {}).get("n") or 0) > 0:
            bits.append(f"benchmark-FP {cell(r,'benchmark-fp')}")
        if r["free_text_scaffold_signal"]:
            bits.append("free-text scaffold mention")
        if r["free_text_fp_signal"]:
            bits.append("free-text benchmark-FP mention")
        L.append(f"- **{r['condition']}** — {'; '.join(bits)} · `{r['path']}`")
    L.append("")

    inc = [r for r in rows if r["coverage"] == "self_declared_incomplete"]
    if inc:
        L.append("## ⚠️ Digests that declare their own attribution incomplete")
        L.append("")
        L.append("For these, a blank scaffold-bug / benchmark-FP cell means "
                 "**not investigated**. Citing them as evidence of a clean "
                 "pipeline is exactly the inference they warn against.")
        L.append("")
        for r in inc:
            L.append(f"- **{r['condition']}** — matched `{r['incomplete_quote']}` "
                     f"· `{r['path']}`")
        L.append("")

    L.append("## Corpus-level verdict")
    L.append("")
    resolved = n_parsed + n_ptr
    if resolved == len(rows):
        L.append(f"All {len(rows)} digests resolve ({n_parsed} parsed + {n_ptr} "
                 "pointer). A corpus-level statement is admissible **only if** the "
                 "self-declared-incomplete set is empty as well.")
    else:
        L.append(f"**Not admissible.** Only {n_parsed}/{len(rows)} digests expose a "
                 "machine-readable attribution table, so no statement of the form "
                 '"the pipeline is clean across all conditions" can be made from '
                 "this index. The per-condition rows above are the usable unit.")
    L.append("")

    run_specific = [r for r in rows if r["run"]]
    if run_specific:
        L.append("## Run-specific digests (replicates / ablation arms)")
        L.append("")
        for r in run_specific:
            L.append(f"- `{r['condition']}` ({r['run']}) — {COV[r['coverage']]} "
                     f"· `{r['path']}`")
        L.append("")

    L.append("## Headlines (where the digest states one)")
    L.append("")
    for r in rows:
        if r["headline"]:
            L.append(f"- **{r['condition']}** — {r['headline']}")
    L.append("")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L), encoding="utf-8")
    a.json_out.write_text(json.dumps(
        {"n_digests": len(rows), "n_parsed": n_parsed, "n_pointer": n_ptr,
         "n_self_declared_incomplete": n_inc, "n_unparsed": n_unp,
         "corpus_verdict_admissible": (n_parsed + n_ptr) == len(rows) and n_inc == 0,
         "digests": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {a.out}\nwrote {a.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
