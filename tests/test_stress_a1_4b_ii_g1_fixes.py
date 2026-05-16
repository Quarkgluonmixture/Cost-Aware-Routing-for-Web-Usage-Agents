"""Invariant tests for /stress A1.4b-ii G1 dead-code cleanup (B-187/B-188/B-189).

- B-187: `compute_energy_step` deleted (was 0-caller dead code with wrong YAML key)
- B-188: `compute_waste_breakdown` deleted (was 0-caller dead code with math
         invariant violation: parts could exceed total); `compute_wasted_cost`
         no longer accepts dead `adjusted_success` kwarg
- B-189: paper §3.5 prose discloses seed=42 + B=1000 + which scripts share the seed
"""
from __future__ import annotations

from pathlib import Path

import inspect
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_PY = REPO_ROOT / "p79" / "experiment" / "metrics.py"
SECTION3 = REPO_ROOT / "docs" / "checkpoints" / "paper_drafts" / "section3_definition.md"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# ─── B-187 ──────────────────────────────────────────────────────────────────
def test_b187_compute_energy_step_deleted():
    src = _read(METRICS_PY)
    # The function definition is gone
    assert "def compute_energy_step(" not in src
    # The B-187 epitaph comment is present explaining why
    assert "B-187" in src
    # And there is no stale import surface
    import p79.experiment.metrics as m
    assert not hasattr(m, "compute_energy_step")


# ─── B-188 ──────────────────────────────────────────────────────────────────
def test_b188_compute_waste_breakdown_deleted():
    src = _read(METRICS_PY)
    assert "def compute_waste_breakdown(" not in src
    assert "B-188" in src
    import p79.experiment.metrics as m
    assert not hasattr(m, "compute_waste_breakdown")


def test_b188_compute_wasted_cost_no_adjusted_success_kwarg():
    from p79.experiment.metrics import compute_wasted_cost
    sig = inspect.signature(compute_wasted_cost)
    assert "adjusted_success" not in sig.parameters, (
        "adjusted_success was a §139.8-retired-layer kwarg; removed by B-188"
    )
    # The 2-arg API still works
    out = compute_wasted_cost([], success=True)
    assert out == {"wasted_cost_usd": 0.0, "wasted_energy_kwh": 0.0}
    out_fail = compute_wasted_cost(
        [{"cost_usd": {"total": 0.05}, "energy": {"kwh": 0.001}}],
        success=False,
    )
    assert out_fail["wasted_cost_usd"] == pytest.approx(0.05)
    assert out_fail["wasted_energy_kwh"] == pytest.approx(0.001)


# ─── B-189 ──────────────────────────────────────────────────────────────────
def test_b189_paper_seed_disclosure_in_section_3_5():
    """Paper §3.5 prose now discloses seed=42 + B=1000 + sharing-scripts."""
    src = _read(SECTION3)
    assert "seed=42" in src
    assert "B=1000" in src or "B=10000" in src or "1000 draws" in src
    # Wording mentions which scripts use the same seed
    assert "aggregate_phase1_prereg_gate.py" in src
    assert "aggregate_phantom_lift.py" in src
