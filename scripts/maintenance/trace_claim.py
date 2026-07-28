#!/usr/bin/env python3
"""Trace one claim down through all four layers, so it can be checked by hand.

    conclusions/*.md   what we now say it means   <- written by Claude
    ledger.jsonl       the structured record      <- extracted by Claude
    实验笔记.md         the original prose         <- written at the time
    artifact           the file on disk           <- the only non-Claude layer

Everything above the artifact line was written by a language model. Verifying a
number therefore means walking down to the artifact, not agreeing with the top
three layers — they share a source and cannot disconfirm each other.

    trace_claim.py 6.7            # anything mentioning 6.7
    trace_claim.py "drop-one"     # by phrase
    trace_claim.py --section 302  # everything from §302
    trace_claim.py 1.3528 --full  # don't truncate the prose
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "docs" / "reference" / "known" / "ledger.jsonl"
CONC = ROOT / "docs" / "reference" / "known" / "conclusions"
NOTE = ROOT / "docs" / "checkpoints" / "实验笔记.md"

C = {"h": "\033[1m", "d": "\033[2m", "w": "\033[33m", "r": "\033[31m",
     "g": "\033[32m", "b": "\033[36m", "0": "\033[0m"}


def layer(title: str, n: int, colour: str = "b") -> None:
    print(f"\n{C[colour]}{'─'*68}\n{title}  ({n}){C['0']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("term", nargs="?", help="number or phrase to trace")
    ap.add_argument("--section", "-s", help="trace a whole section instead")
    ap.add_argument("--full", action="store_true", help="do not truncate prose")
    ap.add_argument("--limit", "-n", type=int, default=6)
    a = ap.parse_args()
    if not a.term and not a.section:
        ap.error("give a term or --section")
    cut = 10_000 if a.full else 200

    key = a.term or a.section
    print(f"{C['h']}tracing: {key}{C['0']}")

    # ---- layer 1: what the conclusion layer says -------------------------
    hits = []
    for f in sorted(CONC.glob("*.md")):
        for i, ln in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if (a.term and a.term in ln) or (a.section and re.search(
                    rf"§\s*{re.escape(a.section.lstrip('§'))}(\.\d+)?\b", ln)):
                hits.append((f.name, i, ln.strip()))
    layer("① 结论层 — 我们现在说它是什么意思", len(hits))
    for name, i, ln in hits[:a.limit]:
        print(f"  {C['d']}{name}:{i}{C['0']}  {ln[:cut]}")
    if len(hits) > a.limit:
        print(f"  {C['d']}… 另 {len(hits)-a.limit} 处{C['0']}")

    # ---- layer 2: the ledger record --------------------------------------
    recs = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    lhits = []
    for r in recs:
        blob = json.dumps(r, ensure_ascii=False)
        if (a.term and a.term in blob) or (a.section and re.search(
                rf"{re.escape(a.section.lstrip('§'))}(\.\d+)?\b",
                str(r.get("source_section", "")))):
            lhits.append(r)
    layer("② 台账 — 结构化记录（含 caveats 与作废标记）", len(lhits))
    for r in lhits[:a.limit]:
        head = (r.get("quantity") or r.get("claim") or r.get("decision")
                or r.get("former_claim") or r.get("what") or "")
        print(f"  {C['h']}[{r['type']}] {r.get('source_section')}{C['0']} {head[:cut]}")
        for fld, col in (("value", "g"), ("scope", "b"), ("caveats", "w"),
                         ("why_dead", "r"), ("why_unverified", "w")):
            if r.get(fld):
                print(f"      {C[col]}{fld}{C['0']}: {str(r[fld])[:cut]}")
        if r.get("superseded_by"):
            print(f"      {C['r']}SUPERSEDED BY {r['superseded_by']}{C['0']}")
        for fl in r.get("_cross_chunk_flags", []):
            print(f"      {C['r']}⚑ {fl[:cut]}{C['0']}")
        src = r.get("source_artifact") or r.get("path") or r.get("recorded_where")
        if src:
            mark = {True: "✓", False: "✗ ABSENT"}.get(r.get("artifact_exists"), "?")
            print(f"      {C['d']}{mark} {src}{C['0']}")

    # ---- layer 3: the chronicle, as written at the time -------------------
    secs = {str(r.get("source_section")) for r in lhits if r.get("source_section")}
    if a.section:
        secs.add("§" + a.section.lstrip("§"))
    layer("③ 笔记原文 — 当时是怎么写的", len(secs))
    for s in sorted(secs)[:4]:
        n = re.search(r"(\d+)(?:\.(\d+))?", s)
        if not n:
            continue
        pat = (rf"^###\s+{n.group(1)}\.{n.group(2)}" if n.group(2)
               else rf"^##\s+{n.group(1)}\.")
        try:
            out = subprocess.run(["grep", "-nE", pat, str(NOTE)],
                                 capture_output=True, text=True, timeout=20).stdout
        except Exception:
            out = ""
        if out.strip():
            ln = out.strip().splitlines()[0]
            print(f"  {C['d']}{s}{C['0']} → 实验笔记.md:{ln.split(':')[0]}  "
                  f"{ln.split(':',1)[1][:120]}")
        else:
            print(f"  {C['d']}{s}{C['0']} → {C['w']}该 § 标题未在笔记中定位到{C['0']}")

    # ---- layer 4: the only layer Claude did not write --------------------
    arts = {r.get("source_artifact") or r.get("path") for r in lhits}
    arts = {a_ for a_ in arts if a_ and ("/" in a_ or "." in a_)}
    layer("④ artifact — 唯一不是 Claude 写的一层，核验必须走到这里", len(arts), "g")
    for p in sorted(arts)[:6]:
        tok = p.split()[0].split(":")[0].strip("`'\",;")
        exists = (ROOT / tok).exists()
        print(f"  {'✓' if exists else '✗'} {tok}")
    if not arts:
        print(f"  {C['w']}无 artifact 指针 —— 这条只有文字来源，无法复算{C['0']}")

    print(f"\n{C['w']}提醒：①②③ 同源（都源自笔记），互相印证不构成验证。"
          f"要证实一个数字，去 ④。{C['0']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
