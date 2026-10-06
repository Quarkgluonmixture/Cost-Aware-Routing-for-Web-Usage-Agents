#!/usr/bin/env python3
"""Which of the 11 cells does each cross-site evidence product cover? (read-only)

Writes `docs/analysis/run_inventory/product_coverage.{md,json}` — the per-product half of the
run inventory (the per-run half is `run_matrix.md`). For every `docs/analysis/cross_sites/*`
product: its producer script, the date it was last committed, and which cells its own text
names.

Cell detection is textual plus structural (a product "covers" a cell if its JSON — or its md when there is no
JSON — names that baseline next to that site, in any of the spellings the producers use:
`cls_B0`, `B0/classifieds`, `B0.cls`, `classifieds·B0`, `B0_dom_classifieds_…`, `wa_red_B0`, …).
That is a coverage *index*, not a correctness check: a cell that is named may still be read
wrongly, and a product scoped by design to fewer cells is flagged here only as "not named".
Why it exists: the 2026-10-06 sweep (实验笔记 §531.2) found shopping in 0 of 55 products by
hand; this makes that count reproducible.

    python scripts/maintenance/evidence_layer_coverage.py
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CROSS = REPO / "docs/analysis/cross_sites"
OUT = REPO / "docs/analysis/run_inventory/product_coverage"
SCRIPTS = REPO / "scripts"

SITE = {
    "cls": r"(?:classifieds|cls)",
    "red": r"(?<![Ww][Aa][_·\- ])(?<![Ww][Aa])(?:reddit|red)(?!dit_wa)",
    "shop": r"(?:shopping|shop)",
    "wared": r"(?:wa[_·\- ]?red(?:dit)?|wared|WA[·_\- ]reddit|webarena[_·/ ]reddit)",
}
CELLS = [("cls", "B0"), ("cls", "B1"), ("cls", "B2"), ("cls", "B5"),
         ("red", "B0"), ("red", "B1"), ("red", "B2"),
         ("shop", "B0"), ("shop", "B1"),
         ("wared", "B0"), ("wared", "B1")]
SEP = r"[^A-Za-z0-9\n]{1,3}"
MODES = r"(?:dom|som|vision|phantom_[a-z]+|3mode|[a-z]+)"


def cell_regex(site: str, b: str) -> re.Pattern:
    s = SITE[site]
    pats = [rf"\b{b}{SEP}{s}", rf"{s}{SEP}{b}\b", rf"\b{b}_{MODES}_{s}"]
    if site == "wared":  # WA run ids: B0_dom_wa_reddit_…
        pats.append(rf"\b{b}_{MODES}_wa_reddit")
    return re.compile("|".join(f"(?:{p})" for p in pats), re.I)


REGEX = {c: cell_regex(*c) for c in CELLS}


BASE_TOKEN = re.compile(r"(?<![A-Za-z0-9])B([0-9])(?![0-9])")
SITE_TOKEN = [  # order matters: a WA mention is removed before VWA reddit is looked for
    ("wared", re.compile(r"(?<![Vv])wa[_\- ]?reddit|(?<![Vv])wa_red|wared|(?<![Vv])WA[·_\- ]red|webarena",
                         re.I)),
    ("cls", re.compile(r"classifieds|(?<![a-z])cls(?![a-z])|VWA-cla", re.I)),
    ("shop", re.compile(r"shopping|(?<![a-z])shop(?![a-z])", re.I)),
    ("red", re.compile(r"reddit|(?<![a-z])red(?![a-z])|VWA-red", re.I)),
]


def _cells_in_context(ctx: str) -> set[str]:
    bases = {f"B{d}" for d in BASE_TOKEN.findall(ctx)}
    sites, rest = set(), ctx
    for name, rx in SITE_TOKEN:
        if rx.search(rest):
            sites.add(name)
            rest = rx.sub(" ", rest)
    return {f"{st}_{bl}" for st in sites for bl in bases}


def json_cells(obj) -> set[str]:
    """Cells named structurally: a baseline and a site co-occurring on one node's key path plus
    that dict's own short string fields. Handles {"classifieds": {"B0": ...}},
    {"baseline": "B0", "site": ...}, "B0·VWA-cla" keys and escaped separators."""
    cells: set[str] = set()

    def visit(node, path: tuple[str, ...]):
        if isinstance(node, dict):
            own = tuple(v for v in node.values() if isinstance(v, str) and len(v) < 120)
            cells.update(_cells_in_context(" | ".join(path + own)))
            for k, v in node.items():
                visit(v, path + (str(k),))
        elif isinstance(node, list):
            for v in node:
                visit(v, path)
        elif isinstance(node, str) and len(node) < 200:
            cells.update(_cells_in_context(" | ".join(path + (node,))))

    visit(obj, ())
    return cells


def producer_of(stem: str, md_text: str) -> str:
    m = re.search(r"^producer:\s*(.+)$", md_text, re.M)
    if m:
        return m.group(1).strip().strip("`")
    hits = []
    for py in SCRIPTS.rglob("*.py"):
        if "__pycache__" in py.parts or "figures" in py.parts:
            continue
        t = py.read_text(encoding="utf-8", errors="replace")
        if f"{stem}.json" in t or f"{stem}.md" in t or f'"{stem}"' in t:
            hits.append(str(py.relative_to(REPO)).replace("\\", "/"))
    hits = [h for h in hits if "export_ablation_tables" not in h and "layered_status" not in h
            and "evidence_layer_coverage" not in h]
    return "; ".join(sorted(hits)[:2]) if hits else "?"


def last_commit(path: Path) -> str:
    out = subprocess.run(["git", "log", "-1", "--format=%cs", "--", str(path.relative_to(REPO))],
                         cwd=REPO, capture_output=True)
    return out.stdout.decode("utf-8", "replace").strip() or "untracked"


def main() -> None:
    stems = sorted({p.stem for p in CROSS.glob("*.json")} | {p.stem for p in CROSS.glob("*.md")})
    rows = []
    for stem in stems:
        js, md = CROSS / f"{stem}.json", CROSS / f"{stem}.md"
        md_text = md.read_text(encoding="utf-8", errors="replace") if md.exists() else ""
        body = js.read_text(encoding="utf-8", errors="replace") if js.exists() else md_text
        named = {f"{s}_{b}" for (s, b) in CELLS if REGEX[(s, b)].search(body)}
        if js.exists():
            try:
                named |= json_cells(json.loads(body))
            except json.JSONDecodeError:
                pass
        covered = [f"{s}_{b}" for (s, b) in CELLS if f"{s}_{b}" in named]
        src = js if js.exists() else md
        rows.append({"product": stem, "source": src.name, "producer": producer_of(stem, md_text),
                     "last_commit": last_commit(src), "cells": covered})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.with_suffix(".json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

    keys = [f"{s}_{b}" for (s, b) in CELLS]
    head = ["cls B0", "cls B1", "cls B2", "cls B5", "red B0", "red B1", "red B2",
            "shop B0", "shop B1", "WA B0", "WA B1"]
    L = ["# Product coverage — which cells each cross-site product names", "",
         "Generated by `scripts/maintenance/evidence_layer_coverage.py` (read-only; textual, see its "
         "docstring — ● = the product names that cell, · = it does not). Pairs with `run_matrix.md`.", "",
         "| product | producer | last commit | " + " | ".join(head) + " |",
         "|---|---|---|" + "---|" * len(head)]
    for r in rows:
        L.append(f"| `{r['product']}` | `{r['producer']}` | {r['last_commit']} | "
                 + " | ".join("●" if k in r["cells"] else "·" for k in keys) + " |")
    tot = {k: sum(k in r["cells"] for r in rows) for k in keys}
    L += ["", f"**Products naming each cell** (of {len(rows)}): "
          + " · ".join(f"{h} {tot[k]}" for h, k in zip(head, keys)), ""]
    OUT.with_suffix(".md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT.with_suffix('.md')}  ({len(rows)} products)")
    print(" · ".join(f"{h} {tot[k]}" for h, k in zip(head, keys)))


if __name__ == "__main__":
    main()
