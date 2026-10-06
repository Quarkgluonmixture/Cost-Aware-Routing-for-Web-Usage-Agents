"""`aggregate_failure_modes.output_section` — extension-registered cells stay out of `cells`.

The bug it guards (实验笔记 §533.1): shopping B0/B1 were registered under run_manifest
`extension:` on 2026-10-06 (§531.5). The aggregator decided `cells` vs `extension_cells`
by baseline alone, and B0/B1 are preregistered baselines, so the next recompute would have
moved 7 shopping conditions into the preregistered `cells` that the failure-mode figure and
the deployment profile read.
"""
from __future__ import annotations

from scripts.analysis.aggregate_failure_modes import output_section

PREREG = ["B0", "B1", "B2"]


def test_prereg_baseline_registered_as_extension_goes_to_extension():
    ext = {"B1_dom_shopping_20260809"}
    assert output_section(["B1_dom_shopping_20260809"], ext, "B1", PREREG) == "extension_cells"


def test_prereg_cell_stays_in_cells():
    ext = {"B1_dom_shopping_20260809"}
    assert output_section(["B1_dom_classifieds_X"], ext, "B1", PREREG) == "cells"


def test_extension_backbone_goes_to_extension():
    ext = {"B5_dom_classifieds_X"}
    assert output_section(["B5_dom_classifieds_X"], ext, "B5", PREREG) == "extension_cells"


def test_without_registry_falls_back_to_baseline_rule():
    assert output_section(["B5_x"], None, "B5", PREREG) == "extension_cells"
    assert output_section(["B1_x"], None, "B1", PREREG) == "cells"
