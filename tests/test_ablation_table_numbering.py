"""Invariants for the evidence section's table numbering and cell resolution.

Two failures on 2026-08-03 motivated these, and neither was visible in the output:

1. The reading guide's table numbers were literals while its figures were injected. The
   registry grew from 29 tables to 35 and the guide kept describing "the 29 tables" with
   its groups mis-mapped from Table 5 onward — a reader following it to `Table 13` for the
   rerun band landed on the click-dispatch table instead.
2. `CELLS` spells the WebArena cells `wa_B0`/`wa_B1`; three products spell them
   `wa_red_B0`/`wa_red_B1`. `d["cells"].get(c)` returned None and the `continue` under it
   dropped both rows silently. The cascade table rendered six rows out of eight while its
   own caption discussed the WA result.

Both are the same shape: a lookup missed, nothing said so, and the artefact looked whole.

Loads the script via importlib (matches the A1.15 / A1.24 convention; tests/ has no
__init__.py so pytest prepend mode already has the dir on sys.path).
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ANALYSIS = REPO / "scripts/analysis"


def _load(name: str, rel: str):
    if str(ANALYSIS) not in sys.path:
        sys.path.insert(0, str(ANALYSIS))
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


EXPORT = _load("export_ablation_tables", "scripts/analysis/export_ablation_tables.py")


def _guide_table_numbers() -> set[int]:
    """Every table number the reading guide cites, expanding `Tables 4–7` runs."""
    text = EXPORT.build_guide()
    found: set[int] = set()
    for group in re.findall(r"Tables? ((?:\d+(?:–\d+)?)(?:, \d+(?:–\d+)?)*)", text):
        for part in group.split(", "):
            if "–" in part:
                lo, hi = (int(x) for x in part.split("–"))
                found.update(range(lo, hi + 1))
            else:
                found.add(int(part))
    return found


def test_table_numbers_are_derived_not_typed():
    """`T`/`TS` resolve from the registry, so a rename cannot silently renumber."""
    assert EXPORT.T(EXPORT.TABLES[0][0]) == "Table 1"
    assert EXPORT.T(EXPORT.TABLES[-1][0]) == f"Table {len(EXPORT.TABLES)}"
    with pytest.raises(KeyError):
        EXPORT.T("a-slug-that-does-not-exist")


def test_table_span_collapses_contiguous_runs():
    first_four = [slug for slug, _, _ in EXPORT.TABLES[:4]]
    assert EXPORT.TS(*first_four) == "Tables 1–4"
    assert EXPORT.TS(EXPORT.TABLES[0][0], EXPORT.TABLES[2][0]) == "Tables 1, 3"
    assert EXPORT.TS(EXPORT.TABLES[0][0]) == "Table 1"


def test_guide_covers_every_table():
    """Adding a table without placing it in a guide group leaves it undiscoverable.

    The guide is the only navigational text a prose author reads. A table absent from it
    is not merely unmentioned — it is invisible at the moment the prose gets written.
    """
    cited = _guide_table_numbers()
    expected = set(range(1, len(EXPORT.TABLES) + 1))
    assert cited == expected, (
        f"guide misses {sorted(expected - cited)}, invents {sorted(cited - expected)}")


def test_guide_states_the_live_table_count():
    assert f"the {len(EXPORT.TABLES)} tables" in EXPORT.build_guide()


@pytest.mark.parametrize("product", [
    "fusion_premium", "confidence_cascade_with_wa", "conditional_failure_attribution",
])
def test_every_product_cell_key_resolves(product):
    """A product cell that no `CELLS` entry resolves to is a row that silently vanishes."""
    data = EXPORT.load(product)
    cells = data.get("cells")
    if not isinstance(cells, dict):
        pytest.skip(f"{product} is not cell-keyed")
    assert EXPORT.unmatched_cells(cells) == []


def test_cell_get_accepts_both_wa_spellings():
    cells = {"wa_red_B0": {"marker": 1}, "cls_B0": {"marker": 2}}
    assert EXPORT.cell_get(cells, "wa_B0") == {"marker": 1}
    assert EXPORT.cell_get(cells, "cls_B0") == {"marker": 2}
    assert EXPORT.cell_get(cells, "red_B9") is None


def test_cell_label_is_pretty_for_both_spellings():
    assert EXPORT.cell_label("wa_red_B0") == EXPORT.cell_label("wa_B0") == "WA·B0"
    assert EXPORT.cell_label("cls_B0") == "cls·B0"
