"""Schema v2 field catalog — source of truth for "what fields exist + defaults".

This is purely declarative. Fields added retroactively (post-original-v2 ship)
are still listed here under their actual name + default, with a comment noting
the introducing audit / §.

When v3 lands, copy this dict to v3.py with the new fields added.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# Canonical episode-summary v2 field defaults. Used by:
#   1. `migrate("v2", "v3")` future migration to fill defaults.
#   2. validation / linting scripts to detect missing fields in old data.
EPISODE_SUMMARY_V2_DEFAULTS: Dict[str, Any] = {
    # --- core (required, no default) ---
    "schema_version": "v2",
    "run_id": "",
    "condition_id": "",
    "benchmark": "",
    "benchmark_site": "",
    "task_id": -1,
    "seed": 42,
    "success": False,
    "score": 0.0,
    "steps": 0,
    "retries": 0,
    "no_op_rate": 0.0,
    "page_unchanged_rate": 0.0,
    "total_latency_ms": 0.0,
    "p95_step_latency_ms": 0.0,
    "total_tokens": 0,
    "total_model_cost_usd": 0.0,
    "total_cost_usd": 0.0,
    "total_router_overhead_cost_usd": 0.0,
    "total_router_overhead_ms": 0.0,
    "total_energy_kwh": None,
    "total_co2e_kg": None,
    "escalation_count": 0,
    "trigger_distribution": {},
    "benchmark_noise": False,
    "benchmark_noise_category": None,
    "artifacts_dir": "",
    # --- additions (optional, default-filled) ---
    "state_change_reason_distribution": {},  # original v2
    "checklist_completion_rate": None,       # original v2
    "checklist_failed_items": None,          # original v2
    "error": None,                            # original v2
    "busy_wait_free_steps": 0,               # original v2
    "busy_wait_total_ms": 0.0,               # §97 RU-4
    "wasted_cost_usd": 0.0,                  # original v2
    "wasted_energy_kwh": 0.0,                # original v2
    "component_breakdown": None,             # original v2
    "total_input_cost_usd": 0.0,             # original v2
    "total_output_cost_usd": 0.0,            # original v2
    "total_obs_prepare_cost_usd": 0.0,       # original v2
    "agent_finished": None,                  # original v2 (§78)
    "energy_partial": False,                 # §97 RU-5
    "energy_step_complete_count": 0,         # §97 RU-5
    "adjusted_success": None,                # §97 Step-2
    "fp_reason": "",                         # §97 Step-2
    "has_effective_action": False,           # §97 Step-2
}


def fill_defaults(record: dict, defaults: Optional[Dict[str, Any]] = None) -> dict:
    """Return a copy of `record` with missing keys filled from `defaults`.

    Used by old-data readers that need a stable schema regardless of when
    the file was written. Does NOT overwrite existing keys.
    """
    d = defaults if defaults is not None else EPISODE_SUMMARY_V2_DEFAULTS
    out = dict(d)  # start with all defaults
    out.update(record)  # overlay actuals (existing values win)
    return out
