"""Schema v2 field catalog — source of truth for "what fields exist + defaults".

This is purely declarative. Fields added retroactively (post-original-v2 ship)
are still listed here under their actual name + default, with a comment noting
the introducing audit / §.

When v3 lands, copy this dict to v3.py with the new fields added.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# Import the canonical runtime constant so the `schema_version` default below
# matches what the runner actually writes (`types.SCHEMA_VERSION_V2 = "2.0"`).
# /stress A1.2 codex Mode B C1 fix: previously the default was the literal
# string "v2" while the runtime constant is "2.0", which would have caused
# `fill_defaults()` to silently mis-tag old episode records when the v3
# migration eventually lands. Aligning the source of truth here closes the
# latent schema-identity split before v3 work begins.
from p79.experiment.types import SCHEMA_VERSION_V2

# Canonical episode-summary v2 field defaults. Used by:
#   1. `migrate("v2", "v3")` future migration to fill defaults.
#   2. validation / linting scripts to detect missing fields in old data.
EPISODE_SUMMARY_V2_DEFAULTS: Dict[str, Any] = {
    # --- core (required, no default) ---
    "schema_version": SCHEMA_VERSION_V2,  # "2.0" — matches what runner writes
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
    # B-193 (/stress A1.4b-ii codex B-ii-2): paper §3.5 transparency
    # telemetry — runner stamps these in normal + exception paths
    # (`runner/main.py:1936-1953` + `:896-899`). Legacy summaries (pre-A1.4a
    # B-166/B-167/B-168) lack these; fill_defaults backfills them so
    # `aggregate_condition_metrics` always has a value to read.
    "trajectory_incomplete": False,
    "unknown_failure_reasons": {},
    "partial_recovery_step_count": 0,
    # B-403 (/stress A1.1 v8 Mode B P1-9, 2026-05-16): per-episode count of
    # image_encode_error steps (legacy summaries default 0 — assumed clean
    # absent telemetry; aggregator readers will see 0 and skip exclusion).
    "image_encode_error_step_count": 0,
    # adjusted_success / fp_reason / has_effective_action removed §139.8 —
    # post-hoc na_fp / eval_fp filter layer retired (fixed at the source:
    # B-91 evaluator guard + N/A exclusion at load). Archived pre-§139.8
    # summaries may still carry these keys (harmless, unread).
    # B-462 (/stress A1.5b Phase 1 P0-3-B codex OOB, 2026-05-17): Option K
    # covariate anchors. Legacy summaries lack these → aggregator falls back
    # to filesystem scan (B-389 robustness path); fresh runs stamp via
    # `_run_and_record_episode` entry/exit.
    "wallclock_start": None,
    "wallclock_end": None,
    # B-461 (/stress A1.5b Phase 1 P0-2-C gemini OOB, 2026-05-17): quarantine
    # flag — exception path sets True → resume gate (`runner/main.py:552-619`)
    # force re-runs instead of accepting summary. Legacy summaries default
    # False (assumed clean unless explicitly flagged).
    "needs_reevaluation": False,
}


# B-190 (/stress A1.4b-ii Claude D1 + codex B-ii-1, P0 OOB): mirror catalog
# for `StepRecordV2`. Previously only `EpisodeSummaryV2` had a defaults dict,
# so step-level JSONL rows lacked any `fill_defaults` path → readers either
# raised KeyError on missing optional fields OR silently saw None. The 5
# paper-grade-critical optionals (`parse_valid`, `parse_failure_reason`,
# `fallback_finish`, `image_meta`, `locator_route_meta`) MUST exist for
# §3 evidence layer claims (B-167 invalid_action taxonomy, B-156 locator
# route ON_TARGET rate, A1.1 codex C1 image_over_cap audit, B-165
# fallback_finish reward override guard). New step records will carry them;
# legacy JSONL is back-filled by `fill_defaults(record, STEP_RECORD_V2_DEFAULTS)`.
STEP_RECORD_V2_DEFAULTS: Dict[str, Any] = {
    # --- core (required, no default — fail if missing) ---
    "schema_version": SCHEMA_VERSION_V2,
    "run_id": "",
    "condition_id": "",
    "benchmark": "",
    "benchmark_site": "",
    "task_id": -1,
    "seed": 42,
    "step_idx": 0,
    "som": {},
    "observation_mode": "unknown",
    "router": {},
    "module_flags": {},
    "action_type": "",
    "action": {},
    "action_success": False,
    "page_changed": False,
    "latency_ms": {},
    "tokens": {},
    "cost_usd": {},
    "energy": {},
    "retry_count": 0,
    "error_category": None,
    "artifact_paths": {},
    "reward": 0.0,
    "done": False,
    # --- optional (default-filled when absent in old data) ---
    "page_change_reasons": [],
    "text_similarity": None,
    "checklist": None,
    "state_digest": None,
    "obs_url": None,
    # paper-grade critical optionals (B-167, A1.1 codex C1, B-156, B-09):
    "parse_valid": None,
    "parse_failure_reason": None,
    "fallback_finish": None,
    "confidence": None,
    "agent_visible_changed": None,
    "image_meta": None,
    # B-291 fix (2026-05-16, A1.8): separator between vision-mode-no-image
    # and old-data-missing-field. Old data gets False (archive lineage flag).
    "image_meta_recorded": False,
    "locator_route_meta": None,
    # B-440 (/stress A1.25 P0-2-B* codex OOB, 2026-05-17): retry-overwrite
    # split (primary + retry). Archive rows pre-A1.25 lack these → fill with
    # None so aggregators see consistent shape; B-440-aware aggregator
    # (e.g. aggregate_locator_route_metrics.py) falls back to legacy
    # `locator_route_meta` when `_primary` is None.
    "locator_route_meta_primary": None,
    "locator_route_meta_retry": None,
    # B-420 (/stress A1.3 v9, 2026-05-17): symmetric with locator_route_meta.
    "select_option_meta": None,
    # B-450 (/stress A1.4 P0-3-B codex OOB, 2026-05-17): select_option retry-
    # overwrite split, symmetric with locator_route_meta_primary/retry (B-440).
    # Archive rows pre-A1.4-P0-3 lack these → fill with None so aggregators
    # see consistent shape; B-450-aware aggregator falls back to legacy
    # `select_option_meta` when `_primary` is None.
    "select_option_meta_primary": None,
    "select_option_meta_retry": None,
    # B-284 fix (2026-05-16, A1.8): retire ghost-field status by registering
    # the 5 GLM/retry de-biasing fields in the schema catalog. Old data lacks
    # them → fill_step_defaults backfills None → downstream aggregators see
    # consistent shape regardless of when the JSONL was written.
    "retry_action_applied": None,
    "retry_action_type": None,
    # B-398 (/stress A1.1 v8 Mode A+B P0-3 overlap, 2026-05-16): explicit
    # attempted flag so attempted-but-failed cases are not silently merged
    # with never-tried. See `types.py:glm_fallback_attempted` for full
    # rationale.
    "glm_fallback_attempted": None,
    "glm_fallback_used": None,
    "glm_fallback_latency_ms": None,
    "glm_original_fail_reason": None,
}


def fill_defaults(record: dict, defaults: Optional[Dict[str, Any]] = None) -> dict:
    """Return a copy of `record` with missing keys filled from `defaults`.

    Used by old-data readers that need a stable schema regardless of when
    the file was written. Does NOT overwrite existing keys.

    Pass `defaults=EPISODE_SUMMARY_V2_DEFAULTS` for episode summaries (the
    backward-compat default when no `defaults` is supplied) or
    `defaults=STEP_RECORD_V2_DEFAULTS` for step-record JSONL rows.
    """
    d = defaults if defaults is not None else EPISODE_SUMMARY_V2_DEFAULTS
    out = dict(d)  # start with all defaults
    out.update(record)  # overlay actuals (existing values win)
    return out


def fill_step_defaults(record: dict) -> dict:
    """B-190 convenience: fill step-record defaults explicitly."""
    return fill_defaults(record, STEP_RECORD_V2_DEFAULTS)
