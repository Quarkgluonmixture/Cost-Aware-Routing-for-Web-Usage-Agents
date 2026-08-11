#!/usr/bin/env python3
"""Shared parser for `router_objective_ordering.md`.

Three thesis figures (F5 design matrix, F6 SR heatmap, F7 cost-SR frontier) all
need the same thing: per (cell, mode) success rate and mean cost, for all eight
cells. That table exists once, in `router_objective_ordering.md`, and parsing it
three times in three scripts is how the six-cell/eight-cell drift in this
project's prose happened in the first place. So it is parsed once here.

The parser cross-checks every VWA value it extracts against `sr_per_mode.json`
(the canonical SR artefact) and raises on disagreement, so a future regeneration
of either file that silently changes a number fails loudly instead of producing
a figure that contradicts the table beside it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC_ORDER = ROOT / "docs/analysis/cross_sites/router_objective_ordering.md"
SRC_SR = ROOT / "docs/analysis/cross_sites/sr_per_mode.json"

MODES = ["DOM", "P-prompt", "P-text", "P-SoM", "SoM", "Vision"]

# Grouped by what is actually transmitted (TERMS.md §1.1); the four text-side
# modes all log image_payload_bytes == 0.
SIDES = [("text side  ·  no image", ["DOM", "P-prompt", "P-text", "P-SoM"]),
         ("combined  ·  text + image", ["SoM"]),
         ("visual  ·  image only", ["Vision"])]
C_SIDE = ["#0072B2", "#009E73", "#D55E00"]
SIDE_OF = {m: i for i, (_s, ms) in enumerate(SIDES) for m in ms}

# Display order: VWA cells first (primary), WA last (external validation).
CELL_ORDER = ["cls·B0", "red·B0", "cls·B1", "red·B1", "cls·B2", "red·B2",
              "wa_red·B0", "wa_red·B1"]
VWA_CELLS = CELL_ORDER[:6]
WA_CELLS = CELL_ORDER[6:]

_SITE_SHORT = {"classifieds": "cls", "reddit": "red", "wa_reddit": "wa_red"}
_RE_HEAD = re.compile(r"^## (\w+) · (B\d)\s+\(n=(\d+)\)", re.M)
_RE_ROW = re.compile(
    r"^\|\s*`single:([\w-]+)`\s*\|\s*fixed\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", re.M)


def load():
    """Return (cells, n_of, sr, cost).

    cells : list[str]  cell keys present, in CELL_ORDER
    n_of  : dict[cell] -> scored-set size
    sr    : dict[(cell, mode)] -> success rate in %
    cost  : dict[(cell, mode)] -> mean per-episode cost in the cell's own unit
    """
    text = SRC_ORDER.read_text(encoding="utf-8")
    heads = list(_RE_HEAD.finditer(text))
    if not heads:
        raise SystemExit(f"{SRC_ORDER}: no cell headers found")

    n_of, sr, cost = {}, {}, {}
    for i, h in enumerate(heads):
        site, base, n = h.group(1), h.group(2), int(h.group(3))
        cell = f"{_SITE_SHORT.get(site, site)}·{base}"
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        block = text[h.end():end]
        rows = _RE_ROW.findall(block)
        if len(rows) != len(MODES):
            raise SystemExit(f"{SRC_ORDER}: {cell} has {len(rows)} fixed rows, "
                             f"expected {len(MODES)} — refusing to plot")
        n_of[cell] = n
        for mode, s, c in rows:
            if mode not in MODES:
                raise SystemExit(f"{SRC_ORDER}: unknown mode {mode!r} in {cell}")
            sr[(cell, mode)] = float(s)
            cost[(cell, mode)] = float(c)

    _crosscheck(sr)
    cells = [c for c in CELL_ORDER if c in n_of]
    missing = set(n_of) - set(cells)
    if missing:
        raise SystemExit(f"{SRC_ORDER}: cells not in CELL_ORDER: {sorted(missing)}")
    return cells, n_of, sr, cost


def _crosscheck(sr: dict) -> None:
    """Fail loudly if the two artefacts disagree on any shared value."""
    canon = json.loads(SRC_SR.read_text(encoding="utf-8"))["cells"]
    checked = 0
    for v in canon.values():
        cell = f"{_SITE_SHORT.get(v['site'], v['site'])}·{v['baseline']}"
        key = (cell, v["mode"])
        if key not in sr:
            continue
        if abs(sr[key] - v["sr_pct"]) > 0.01:
            raise SystemExit(
                f"artefact disagreement on {key}: ordering.md says {sr[key]:.4f}, "
                f"sr_per_mode.json says {v['sr_pct']:.4f} — regenerate both")
        checked += 1
    if checked < 30:
        raise SystemExit(f"only {checked} values cross-checked against "
                         f"sr_per_mode.json; expected >= 30")
