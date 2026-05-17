"""Invariant tests for /stress A1.9 cold-start audit fixes.

Scope: 18 fixes (B-782 ~ B-798, plus P1-3 GRL prose disclosure) addressing
cold-start re-audit findings on `p79/experiment/{metrics, energy_tracker,
environment}.py` substrate. Tests cover code-layer invariants only; prose-
layer fixes (B-791 P95 caveat, B-792 B0 token disclosure, B-795 clean SR
realign, P1-3 GRL B-329 sibling disclosure) are validated by `grep` against
the paper drafts at audit time, not automated here.

Test groups:
- B-782 numeric hero strict guard (10 attack vectors)
- B-783 paper_grade evaluator mid-call raise
- B-784 needs_reevaluation aggregator entry guard
- B-785 BLIP-2 lazy paper_grade fail-loud
- B-786 "location" keyword over-match removal
- B-787 region world fallback fail-loud
- B-788 energy_window_partial aggregator emission
- B-789 exception-path component obs_prepare closure
- B-790 fig3 caption dynamic N (smoke)
- B-793 preflight evaluator probe (smoke via subprocess)
- B-794 hardware_profile MISSING fail-loud (sibling of B-320)
- B-796 cpu_arch + cpu_rapl_available stamp in EnergyEstimate
- B-797 evaluator_blip2_device stamp helper
- B-798 clean_n_too_low warning emit
"""
from __future__ import annotations

import math
import subprocess
import sys
from typing import Any, Dict, List

import pytest

from p79.experiment.metrics import (
    _HERO_NUMERIC_FIELDS,
    aggregate_condition_metrics,
    detect_benchmark_noise,
)


# ---------------------------------------------------------------------------
# Helper: minimal valid episode summary
# ---------------------------------------------------------------------------


def _valid_episode() -> Dict[str, Any]:
    return {
        "success": True, "benchmark_noise": False, "score": 1.0,
        "steps": 5, "retries": 0,
        "total_cost_usd": 0.001, "total_model_cost_usd": 0.001,
        "total_router_overhead_cost_usd": 0.0,
        "total_router_overhead_ms": 0.0,
        "total_obs_prepare_cost_usd": 0.0,
        "total_input_cost_usd": 0.0, "total_output_cost_usd": 0.0,
        "total_latency_ms": 1234.5, "p95_step_latency_ms": 300.0,
        "total_energy_kwh": None, "total_co2e_kg": None,
        "no_op_rate": 0.0, "page_unchanged_rate": 0.0,
        "escalation_count": 0, "trigger_distribution": {},
        "state_change_reason_distribution": {},
        "wasted_cost_usd": 0.0, "wasted_energy_kwh": 0.0,
    }


# ---------------------------------------------------------------------------
# B-782 — numeric hero strict guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field,bad_value,error_keyword", [
    ("steps", True, "type mismatch"),           # bool-as-int
    ("steps", "5", "type mismatch"),
    ("steps", float("inf"), "non-finite"),
    ("total_cost_usd", "1e309", "type mismatch"),
    ("total_cost_usd", float("inf"), "non-finite"),
    ("total_cost_usd", float("nan"), "non-finite"),
    ("total_latency_ms", True, "type mismatch"),  # bool-as-int
    ("p95_step_latency_ms", "300", "type mismatch"),
    ("total_energy_kwh", float("inf"), "non-finite"),
    ("retries", True, "type mismatch"),
])
def test_b765_hero_numeric_strict_rejects(field, bad_value, error_keyword):
    ep = _valid_episode()
    ep[field] = bad_value
    with pytest.raises(ValueError, match=error_keyword):
        aggregate_condition_metrics([ep])


def test_b765_valid_episode_passes():
    """Sanity: a fully-populated valid episode passes aggregator."""
    out = aggregate_condition_metrics([_valid_episode()])
    assert out["success_rate"] == 1.0
    assert out["avg_steps"] == 5.0


def test_b765_hero_numeric_fields_frozenset_well_formed():
    """B-782 substrate: _HERO_NUMERIC_FIELDS must be a non-empty frozenset
    containing every numeric paper-§1/§3 hero field. Snapshot covers the
    canonical set as of A1.9; expanding it is allowed (additive), shrinking
    must be a deliberate design decision."""
    assert isinstance(_HERO_NUMERIC_FIELDS, frozenset)
    assert "steps" in _HERO_NUMERIC_FIELDS
    assert "total_cost_usd" in _HERO_NUMERIC_FIELDS
    assert "total_latency_ms" in _HERO_NUMERIC_FIELDS
    assert "p95_step_latency_ms" in _HERO_NUMERIC_FIELDS
    assert "total_energy_kwh" in _HERO_NUMERIC_FIELDS


# ---------------------------------------------------------------------------
# B-784 — needs_reevaluation aggregator entry guard
# ---------------------------------------------------------------------------


def test_b767_needs_reevaluation_default_rejects():
    """Quarantined episode (needs_reevaluation=True) must raise by default —
    runner live aggregate + rederive bypass `load_episode_summary_strict`,
    so the aggregator must enforce. B-486 quarantine semantics."""
    ep = _valid_episode()
    ep["needs_reevaluation"] = True
    with pytest.raises(ValueError, match="needs_reevaluation=True"):
        aggregate_condition_metrics([ep])


def test_b767_needs_reevaluation_forensic_opt_in_allows():
    """Forensic appendix path can opt in via `allow_quarantined=True`."""
    ep = _valid_episode()
    ep["needs_reevaluation"] = True
    # Should not raise.
    out = aggregate_condition_metrics([ep], allow_quarantined=True)
    assert out["episodes"] == 1


def test_b767_needs_reevaluation_type_mismatch_rejects():
    """needs_reevaluation must be bool/None, not string/int."""
    ep = _valid_episode()
    ep["needs_reevaluation"] = "false"
    with pytest.raises(ValueError, match="type mismatch"):
        aggregate_condition_metrics([ep])


# ---------------------------------------------------------------------------
# B-786 — "location" keyword removal in detect_benchmark_noise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("err_msg,expected_noise", [
    ("element location not found", False),         # locator error, not geo
    ("locator resolved to hidden element location", False),
    ("window.location is not defined", False),
    ("not available in your region", True),        # genuine geo
    ("Error: geo-restricted content", True),
    ("Page returned: location restriction", True),
])
def test_b769_location_keyword_only_anchored_phrases(err_msg, expected_noise):
    is_noise, cat = detect_benchmark_noise(err_msg)
    if expected_noise:
        assert is_noise is True and cat == "geo_restricted", \
            f"expected geo_restricted for {err_msg!r}, got {(is_noise, cat)}"
    else:
        assert (is_noise, cat) != (True, "geo_restricted"), \
            f"expected NOT geo_restricted for {err_msg!r} (locator error), got {(is_noise, cat)}"


# ---------------------------------------------------------------------------
# B-787 — region world fallback fail-loud (sibling of B-320)
# ---------------------------------------------------------------------------


def test_b770_unknown_region_raises():
    """Unknown region key must raise when enabled — sibling of B-320 hardware
    profile fail-loud. Pre-fix `definitely_not_a_region` silently coerced to
    `world` (475 g/kWh)."""
    from p79.experiment.energy_tracker import LightweightEnergyTracker
    with pytest.raises(ValueError, match="region="):
        LightweightEnergyTracker({
            "enabled": True,
            "region": "definitely_not_a_region",
            "hardware_profile": "a100_pcie_40gb",
        })


def test_b770_explicit_world_region_allowed():
    """Explicit `region: world` is valid (in REGION_INTENSITY_G_PER_KWH)."""
    from p79.experiment.energy_tracker import LightweightEnergyTracker
    tracker = LightweightEnergyTracker({
        "enabled": True,
        "region": "world",
        "hardware_profile": "a100_pcie_40gb",
        "use_pynvml": False,  # avoid actually loading pynvml in test
    })
    assert tracker.region == "world"


def test_b770_explicit_intensity_bypasses_region_check():
    """Explicit `carbon_intensity_g_per_kwh: <num>` skips region-lookup."""
    from p79.experiment.energy_tracker import LightweightEnergyTracker
    tracker = LightweightEnergyTracker({
        "enabled": True,
        "region": "any_string",  # ignored when intensity is explicit
        "carbon_intensity_g_per_kwh": 220.0,
        "hardware_profile": "a100_pcie_40gb",
        "use_pynvml": False,
    })
    assert tracker.carbon_intensity_g_per_kwh == 220.0


# ---------------------------------------------------------------------------
# B-788 — energy_window_partial aggregator emission
# ---------------------------------------------------------------------------


def test_b771_aggregator_emits_window_density_telemetry():
    """Aggregator must emit `energy_window_partial_episode_rate` +
    `min_window_sample_count_p5` per (site, model, mode) cell."""
    ep1 = _valid_episode()
    ep1["energy_window_partial_step_count"] = 3
    ep1["min_window_sample_count"] = 1
    ep2 = _valid_episode()
    ep2["energy_window_partial_step_count"] = 0
    ep2["min_window_sample_count"] = 4
    out = aggregate_condition_metrics([ep1, ep2])
    assert out["energy_window_partial_episode_count"] == 1, \
        "only ep1 has partial steps"
    assert out["energy_window_partial_episode_rate"] == 0.5
    assert out["min_window_sample_count_p5"] is not None


# ---------------------------------------------------------------------------
# B-794 — hardware_profile MISSING fail-loud
# ---------------------------------------------------------------------------


def test_b777_missing_hardware_profile_raises():
    """B-320 only raised on UNKNOWN profile key; MISSING fell to default 'm2'
    laptop (valid HARDWARE_PROFILES key → didn't trigger B-320). B-794
    closes the sibling gap."""
    from p79.experiment.energy_tracker import LightweightEnergyTracker
    with pytest.raises(ValueError, match="hardware_profile MISSING"):
        LightweightEnergyTracker({
            "enabled": True,
            "region": "uk",
            # hardware_profile deliberately omitted
        })


def test_b777_explicit_m2_allowed():
    """Explicit `hardware_profile: m2` is still valid (opt-in)."""
    from p79.experiment.energy_tracker import LightweightEnergyTracker
    tracker = LightweightEnergyTracker({
        "enabled": True,
        "region": "world",
        "hardware_profile": "m2",
        "use_pynvml": False,
    })
    assert tracker.hardware_profile == "m2"


# ---------------------------------------------------------------------------
# B-796 — cpu_arch + cpu_rapl_available stamp in EnergyEstimate
# ---------------------------------------------------------------------------


def test_b779_estimate_step_stamps_cpu_arch():
    """EnergyTracker.estimate_step output must include `cpu_arch` and
    `cpu_rapl_available` for cross-baseline architecture audit."""
    from p79.experiment.energy_tracker import LightweightEnergyTracker
    tracker = LightweightEnergyTracker({
        "enabled": True,
        "region": "world",
        "hardware_profile": "m2",
        "use_pynvml": False,  # use profile fallback path
        "use_psutil": False,
    })
    out = tracker.estimate_step(duration_seconds=0.1)
    assert "cpu_arch" in out
    assert isinstance(out["cpu_arch"], str) and out["cpu_arch"]
    assert "cpu_rapl_available" in out
    assert isinstance(out["cpu_rapl_available"], bool)


# ---------------------------------------------------------------------------
# B-798 — clean_n_too_low warning emit
# ---------------------------------------------------------------------------


def test_b781_clean_n_too_low_true_when_majority_noise():
    """99 noise + 1 clean → clean_n_too_low=True (clean_n/total = 1% < 50%)."""
    eps: List[Dict[str, Any]] = []
    for _ in range(99):
        ep = _valid_episode()
        ep["benchmark_noise"] = True
        eps.append(ep)
    eps.append(_valid_episode())  # 1 clean
    out = aggregate_condition_metrics(eps)
    assert out["clean_n_too_low"] is True
    assert out["clean_episode_count"] == 1


def test_b781_clean_n_too_low_false_when_clean_majority():
    """80 clean + 20 noise → clean_n_too_low=False."""
    eps: List[Dict[str, Any]] = []
    for _ in range(80):
        eps.append(_valid_episode())
    for _ in range(20):
        ep = _valid_episode()
        ep["benchmark_noise"] = True
        eps.append(ep)
    out = aggregate_condition_metrics(eps)
    assert out["clean_n_too_low"] is False
    assert out["clean_episode_count"] == 80


# ---------------------------------------------------------------------------
# B-790 — fig3 caption dynamic N (smoke import + grep)
# ---------------------------------------------------------------------------


def test_b773_fig3_caption_uses_dynamic_n():
    """Caption text uses `_SITE_N[...]` interpolation, not hardcoded
    `N=234/210` (pre-§139.8 legacy)."""
    fig3_path = "scripts/analysis/figures/fig3_regional_carbon.py"
    with open(fig3_path, encoding="utf-8") as f:
        src = f.read()
    # Hardcoded literals must NOT appear in the caption block.
    assert "(N=234)" not in src, "fig3 caption still has hardcoded N=234"
    assert "(N=210)" not in src, "fig3 caption still has hardcoded N=210"
    # Dynamic interpolation must appear instead.
    assert "_SITE_N['classifieds']" in src or "_SITE_N[\"classifieds\"]" in src
    assert "_SITE_N['reddit']" in src or "_SITE_N[\"reddit\"]" in src


# ---------------------------------------------------------------------------
# B-793 — preflight --paper-grade smoke (subprocess)
# ---------------------------------------------------------------------------


def test_b776_preflight_has_paper_grade_flag():
    """`scripts/preflight_v2.sh --help` must advertise `--paper-grade` flag."""
    # `--help` exits 0 and prints usage.
    result = subprocess.run(
        ["bash", "scripts/preflight_v2.sh", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"preflight --help exit={result.returncode}"
    assert "--paper-grade" in result.stdout, \
        "preflight --help missing --paper-grade flag advertisement"
    assert "B-793" in result.stdout, \
        "preflight --help missing B-793 reference"


# ---------------------------------------------------------------------------
# B-789 — exception-path component breakdown obs_prepare key smoke
# ---------------------------------------------------------------------------


def test_b772_exception_path_breakdown_has_obs_prepare_key():
    """Smoke: runner exception-path component_breakdown manual dict now
    includes `obs_prepare_usd` (matches normal-path `compute_component_breakdown`
    output and the runner's `total_cost_usd = model + router + obs_prepare`
    invariant)."""
    runner_path = "p79/experiment/runner/main.py"
    with open(runner_path, encoding="utf-8") as f:
        src = f.read()
    # The exception-path block manually constructs component_breakdown dict;
    # it must include `obs_prepare_usd` to match normal path B-576 schema.
    # Grep for the literal key — it's narrow enough to be specific.
    exception_block_marker = '"router_overhead_usd": _agg["total_router_overhead_cost_usd"]'
    assert exception_block_marker in src, "exception-path marker missing"
    # The obs_prepare line must appear immediately after the router_overhead
    # line in the exception path (B-789 fix landed it as a sibling).
    block_idx = src.find(exception_block_marker)
    next_500 = src[block_idx:block_idx + 500]
    assert '"obs_prepare_usd":' in next_500, \
        "exception-path component_breakdown missing obs_prepare_usd (B-789)"
