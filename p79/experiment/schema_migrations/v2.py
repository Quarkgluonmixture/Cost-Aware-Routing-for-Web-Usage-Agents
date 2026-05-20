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
    # B-1410 (/stress A2.7 P1-5-AB* 2-AI overlap A+B, 2026-05-18 + user 3-axis
    # canonical-estimand directive). Retry-adjusted total latency: end-to-end
    # episode latency minus B0 network-retry scaffold wait (10-70s × exponential
    # backoff on 429/5xx HTTP codes from proxy). B1/B2 have no equivalent
    # scaffold so the value equals `total_latency_ms` for local backends; B0
    # delta = `sum(step.network_retry_wait_ms)`. Canonical cross-baseline
    # latency axis per §3.5.1 B-1402. Legacy summaries default None →
    # aggregator falls back to raw `total_latency_ms` until runner-side rollup
    # write path lands (deferred to post-parallel-session-merge follow-up).
    "total_latency_minus_retry_ms": None,
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
    # Fire-4 RCA Wave 2 M5 timeout taxonomy defaults (additive — populated
    # by runner exception path when classify_timeout() detects timeout).
    "unverified_timeout_event": None,
    "timeout_callsite": None,
    "verified_substrate_noise": None,
    # Fire-6 RCA Stage C1 evaluator-context provenance defaults.
    "eval_context_mode": None,
    "eval_isolated_context_used": None,
    "eval_goto_latency_ms": None,
    "eval_goto_timeout": None,
    "eval_source_agent_url": None,
    "eval_target_url": None,
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
    # B-487 (/stress A1.5b Phase 1 P0-3-B codex OOB, 2026-05-17): Option K
    # covariate anchors. Legacy summaries lack these → aggregator falls back
    # to filesystem scan (B-389 robustness path); fresh runs stamp via
    # `_run_and_record_episode` entry/exit.
    "wallclock_start": None,
    "wallclock_end": None,
    # B-486 (/stress A1.5b Phase 1 P0-2-C gemini OOB, 2026-05-17): quarantine
    # flag — exception path sets True → resume gate (`runner/main.py:552-619`)
    # force re-runs instead of accepting summary. Legacy summaries default
    # False (assumed clean unless explicitly flagged).
    "needs_reevaluation": False,
    # B-485 (/stress A1.5b Phase 1 P0-1-ABC 3-AI overlap, 2026-05-17): resume
    # fingerprint (sha256[:16] of cfg.model.revision + backend.revision +
    # max_new_tokens + temperature + paper_grade + observation_mode +
    # transformers_version + prompt_hash). Legacy summaries default None →
    # identity gate sees `None != current_hash` mismatch → quarantine +
    # rerun, consistent with paper-grade rerun protocol.
    "resume_fingerprint": None,
    # B-554 (/stress A1.5 P1-4-AB* Claude+codex OOB, 2026-05-17): archive
    # cohort sentinel for reward-override retirement. Post-B-545 (A1.5b
    # Phase 2 commit `7832008`) episodes carry
    # `evaluator_authority_mode="post_B545_vwa_score_only"` + `reward_override
    # _applied=False` (mechanism retired); legacy archive summaries default
    # None for both. Aggregator stratification rule: mixed pre/post archives
    # must filter or annotate by these fields, otherwise SR estimand mixes
    # two semantically different `success` definitions. See
    # `types.py:EpisodeSummaryV2` for full docstring.
    "evaluator_authority_mode": None,
    "reward_override_applied": None,
    # P0-1-ABC* + P1-11-B* Phase 2 telemetry (/stress Phase 0 2026-05-19,
    # 3-AI overlap OOB): runner-intervention rollup. Always stamped ≥0.
    "runner_intervention_count": 0,
    "about_blank_recovery_count": 0,
    # P1-17-C* Phase 2 attempt-lineage (all None until checkpoint-restore
    # infrastructure lands; field reservation closes v2→v3 schema gap).
    "attempt_id": None,
    "attempt_index": None,
    "is_retry_attempt": None,
    "retry_trigger": None,
    "previous_attempt_error": None,
    "previous_attempt_effective_mutation_count": None,
    "substrate_restored_from_checkpoint": None,
    "checkpoint_id": None,
    "checkpoint_hash_before": None,
    "checkpoint_hash_after_restore": None,
    # P1-17-C* + Gemini F3 footprint telemetry (Appendix sensitivity column).
    # Default 0 for count fields; None for normalized score until aggregator
    # computes downstream.
    "effective_mutating_action_count": 0,
    "destructive_action_count": 0,
    "cart_mutation_count": 0,
    "submit_create_count": 0,
    "delete_remove_count": 0,
    "cycle_mutating_action_count": 0,
    "repeated_same_mutating_action_count": 0,
    "footprint_risk_score": None,
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
    # B-488 (/stress A1.25 GRL Chunk 3 P1-2-BC*, 2026-05-17): browser dialog
    # telemetry (per-step list of confirm/alert/beforeunload/prompt events).
    # Archive rows pre-B-488 lack the field → backfill None so aggregators
    # treat absence as "no dialog observed" (which is the safe interpretation
    # — pre-B-488 dialogs were handled but not recorded).
    "dialog_meta": None,
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
    # B-497 (/stress A1.5b Phase 1 P1-9-A, 2026-05-17): control-injected
    # action provenance — None when no control fired; dict when
    # `_anti_repeat_control` / `_no_early_finish_control` replaced the
    # agent's emitted action with a synthetic fallback. See
    # `types.py:control_intervention` for full schema.
    "control_intervention": None,
    # B-512 (/stress A1.5b Phase 2 P0-1-C gemini OOB, 2026-05-17): wrapper-
    # normalized canonical action form (post-`create_scroll_action(direction=)`
    # etc.). Pre-fix `step_record["action"]` was agent's raw emit → cross-
    # baseline evidence layer asymmetric (B0 enum vs B1/B2 free-form delta).
    # Now both raw + normalized recorded → reviewer can verify execution-
    # layer alignment from JSONL alone. None when wrapper did not emit
    # (mock env, exception path).
    "action_executed": None,
    # B-563/B-564/B-565 (/stress A1.22 P0-1/4/5 batch, 2026-05-17): cross-
    # baseline contract sealing. `cost_unit_basis` enum {"api_usd",
    # "electricity_usd_derived", "unknown"} declares the currency of
    # `cost_usd.{input,output,model}` for the baseline that produced this
    # row — aggregators MUST stratify before pooling. `cost_total_mixed
    # _unit_warn` flags B0 rows where `cost_usd.total` mixes API USD with
    # local-scaffold USD (router_overhead + obs_prepare) in a single
    # number. `element_bbox` was a runner-stamped ghost field; now
    # canonical schema (B-564). Pre-A1.22 archive rows lack these → fill
    # None; downstream `cost_unit_basis is None` ⟹ "archived lineage,
    # basis unknown" disposition.
    "cost_unit_basis": None,
    "cost_total_mixed_unit_warn": None,
    "element_bbox": None,
    # B-569 (/stress A1.22 P1-11-A, 2026-05-17): persist B0 network retry
    # telemetry. Pre-A1.22 archive rows lack these → fill None ("baseline
    # has no retry concept" or "archived lineage, telemetry unavailable").
    # B1/B2 always None even post-fix; B0 0 when no retries this step.
    "network_retry_count": None,
    "network_retry_wait_ms": None,
    # P0-1-ABC* Phase 2 telemetry (/stress Phase 0 2026-05-19, 3-AI overlap
    # OOB): about:blank recovery intervention attribution. None on normal
    # agent steps; runner stamps non-None on intervention steps.
    "intervention_type": None,
    "counted_as_agent_action": None,
    "intervention_from_url": None,
    "intervention_recovery_url": None,
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
