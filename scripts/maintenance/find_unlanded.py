#!/usr/bin/env python3
"""Find adjudications that were decided but never landed downstream.

The ledger answers "was this decided?" and "was this measured?". It has no field
for **"was this decided and then never done?"** — an adjudication sits there
intact, unretracted, fully queryable, while nothing downstream ever implemented it.

That gap was found by hand on 2026-07-28: §108 fixed a four-dimension evidence
framework (Outcome / Macro / Micro / Efficiency), the code implements all four,
and neither paper draft contains a single one of the words. Nobody noticed for
three months; the user remembered it and asked.

This script automates that check for the class of adjudications that name a
framework as a slash-tuple, which is how such frameworks tend to be written.

    find_unlanded.py                 # frameworks absent from both drafts
    find_unlanded.py --all           # include ones that did land, for contrast
    find_unlanded.py --infra         # don't filter out path-like tuples

⚠️ Coverage is partial by construction. It sees slash-tuples. It does not see
single-word frameworks, implicit conventions, or anything a decision recorded in
prose rather than as a named structure. It lowers the odds of a silent omission;
it does not remove them.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "docs" / "reference" / "known" / "ledger.jsonl"
DRAFTS = [ROOT / "docs/checkpoints/paper_drafts/paperA",
          ROOT / "docs/checkpoints/paper_drafts/paperB"]
CODE = [ROOT / "scripts" / "analysis", ROOT / "p79"]

TUPLE = re.compile(r"\b([A-Za-z][A-Za-z\-]{2,}(?:\s*/\s*[A-Za-z][A-Za-z\-]{2,}){2,})\b")
# tuples that are filesystem paths or module chains, not conceptual frameworks
INFRA = re.compile(r"^(scripts|tools|p79|docs|logs|results|tests)/|"
                   r"(wrapper|watchdog|runner|unlink|rmtree|rename|maintenance)")

C = {"h": "\033[1m", "d": "\033[2m", "r": "\033[31m", "g": "\033[32m",
     "w": "\033[33m", "0": "\033[0m"}


def corpus(paths) -> str:
    out = []
    for p in paths:
        if not p.exists():
            continue
        for f in p.rglob("*"):
            if f.is_file() and f.suffix in {".md", ".py", ".tex"}:
                out.append(f.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="also show landed ones")
    ap.add_argument("--infra", action="store_true", help="keep path-like tuples")
    a = ap.parse_args()

    paper, code = corpus(DRAFTS), corpus(CODE)
    recs = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]

    seen = {}
    for r in recs:
        if r.get("type") != "ADJUDICATED":
            continue
        txt = " ".join(str(r.get(f, "")) for f in ("decision", "reasoning"))
        for m in TUPLE.findall(txt):
            name = re.sub(r"\s*/\s*", "/", m)
            if len(name) < 12:
                continue
            if not a.infra and INFRA.search(name):
                continue
            seen.setdefault(name, r.get("source_section"))

    rows = []
    for name, sec in seen.items():
        parts = [p.strip() for p in name.split("/")]
        rows.append((name, sec,
                     sum(1 for p in parts if p in paper),
                     sum(1 for p in parts if p in code), len(parts)))

    absent = [x for x in rows if x[2] == 0]
    absent.sort(key=lambda x: -x[3])

    print(f"{len(seen)} 个多元组框架名（已滤基建路径）"
          f"    {C['r']}{len(absent)} 个在两稿一个成分都没出现{C['0']}\n")
    print(f"{'§':12s} {'框架':52s} {'代码':>7s}  {'判读'}")
    print("-" * 92)
    for name, sec, inp, inc, n in absent:
        # in code but not in paper == decided, built, never written up
        verdict = (f"{C['r']}定了·建了·没写进稿{C['0']}" if inc >= n - 1
                   else f"{C['w']}定了·下游都没痕迹{C['0']}" if inc == 0
                   else f"{C['d']}部分实现{C['0']}")
        print(f"{str(sec):12s} {name[:52]:52s} {inc:>4d}/{n:<2d}  {verdict}")

    if a.all:
        print(f"\n{C['g']}—— 已落地的（对照）——{C['0']}")
        for name, sec, inp, inc, n in sorted(rows, key=lambda x: -x[2])[:10]:
            if inp:
                print(f"{str(sec):12s} {name[:52]:52s} paper {inp}/{n}")

    print(f"\n{C['w']}⚠️ 只覆盖『斜杠多元组』这一种框架写法。单名词框架、隐含约定、"
          f"以散文而非命名结构记录的裁定，本工具看不见。{C['0']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
