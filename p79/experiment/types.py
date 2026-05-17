from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict, MISSING
from typing import Any, Dict, List, Optional

# B-282 fix (2026-05-16, A1.8): canonical schema version uses semver.
# Aligned with `schema_migrations._CHAIN = ["2.0"]` so migrate() / current_version()
# operate on the same string the runner writes to disk. Pre-fix the version
# constant was "2.0" but `_CHAIN` was ["v2"], so any v2 → v3 migration call
# would have raised "Unknown schema version".
SCHEMA_VERSION_V2 = "2.0"


@dataclass
class ModuleFlags:
    m1_dom_select_fallback: bool = False
    m2_dom_first_input_fallback: bool = False
    m3_failure_trigger_retry: bool = False
    m4_two_stage_generation_grounding: bool = False

    def as_dict(self) -> Dict[str, bool]:
        return asdict(self)


@dataclass
class ConditionSpec:
    condition_id: str
    phase: str
    backend_id: str
    som_on: bool
    observation_mode: str
    router_on: bool
    modules: ModuleFlags = field(default_factory=ModuleFlags)
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["modules"] = self.modules.as_dict()
        return payload


@dataclass
class TaskSpec:
    benchmark: str
    site: str
    task_id: int
    intent: str
    config_file: str
    raw_task: Dict[str, Any]


@dataclass
class StepRecordV2:
    schema_version: str
    run_id: str
    condition_id: str
    benchmark: str
    benchmark_site: str
    task_id: int
    seed: int
    step_idx: int
    som: Dict[str, Any]
    observation_mode: str
    router: Dict[str, Any]
    module_flags: Dict[str, bool]
    action_type: str
    action: Dict[str, Any]
    action_success: bool
    page_changed: bool
    latency_ms: Dict[str, float]
    tokens: Dict[str, Optional[int]]
    cost_usd: Dict[str, float]
    energy: Dict[str, Optional[float]]
    retry_count: int
    error_category: Optional[str]
    artifact_paths: Dict[str, Optional[str]]
    reward: float
    done: bool
    page_change_reasons: List[str] = field(default_factory=list)
    text_similarity: Optional[float] = None
    checklist: Optional[Dict[str, Any]] = None
    state_digest: Optional[Dict[str, Any]] = None
    obs_url: Optional[str] = None
    parse_valid: Optional[bool] = None
    parse_failure_reason: Optional[str] = None
    fallback_finish: Optional[bool] = None
    confidence: Optional[Dict[str, Any]] = None
    # B-09 fix (Cluster 2 patch, 2026-04-30): split derivation of page_changed.
    # `page_changed` = bool(any of 12 reasons) — used internally for cycle/retry.
    # `agent_visible_changed` = bool(any AGENT_VISIBLE_REASONS reason) — used by
    # paper-grade SR derivation, fig0a metrics, search-loop detection. Excludes
    # form_value_changed / dom_complexity_changed / text_length_changed which
    # fire on form edits not visible in obs_text (B-09 root cause).
    agent_visible_changed: Optional[bool] = None
    # /stress A1.1 codex Mode B C1 fix: B0 image telemetry (over_cap, payload_bytes,
    # quality, compressed) lives in agent meta but was dropped in runner step_record
    # — Q5 audit of B0 image_over_cap fire rate was structurally impossible.
    # Image encode failure (C2) also surfaced here so cross-baseline missingness is
    # auditable from JSONL without grepping logs.
    image_meta: Optional[Dict[str, Any]] = None
    # B-291 fix (2026-05-16, A1.8): explicit "image_meta was recorded" flag
    # separating "vision-mode no image" (image_meta=None + recorded=True) from
    # "old data missing field" (image_meta=None + recorded=False). Pre-fix
    # `fill_step_defaults` filled None for both cases → paper §3 image_over_cap
    # claim could not distinguish archive missing-field from current-schema None.
    image_meta_recorded: bool = False
    # B-156 (/stress A1.3 v8 Claude F5 + codex P2-B7 dual catch, 2026-05-16):
    # locator-route dispatch result (Cluster 1 B-01/02/33 fix). Without this
    # field, paper §3 evidence layer cannot quantify locator-route ON_TARGET
    # rate from JSONL alone — the Tier 10 sweep 94.4% off-target → >80%
    # ON_TARGET goal becomes structurally unverifiable. Schema:
    #   {success, fallback_used, target_tag, error, action_kind}
    # action_kind ∈ {click, type, hover, upload, clear}. None when step did
    # not invoke locator-route (e.g. scroll / wait / coord-only click).
    locator_route_meta: Optional[Dict[str, Any]] = None
    # B-440 (/stress A1.25 P0-2-B* codex OOB, 2026-05-17): primary action's
    # locator-route dispatch telemetry, captured BEFORE the baseline retry
    # block in runner/main.py:1518-1611 can overwrite `next_info`. Pre-fix
    # the retry's `next_info` replaced the primary's → step_record.locator_
    # route_meta only ever showed retry meta (or None if retry was scroll/
    # wait), silently DELETING the walk-fail evidence layer for every step
    # that triggered baseline_retry_on_no_progress. Cross-baseline asymmetry
    # impact: B0/B1/Gemma3-VL have different retry-trigger rates → biased
    # ON_TARGET denominator. Post-fix: runner snapshots primary meta into
    # this field; the existing `locator_route_meta` retains "value at step
    # write time" semantics (= primary if no retry, else retry) for backward
    # compat with archive aggregators.
    locator_route_meta_primary: Optional[Dict[str, Any]] = None
    # B-440 companion field: retry's locator-route dispatch telemetry, None
    # when baseline_retry_on_no_progress did not fire. Aggregators can sum
    # retry hit rate per (site, model, mode) without re-scanning step JSONL.
    locator_route_meta_retry: Optional[Dict[str, Any]] = None
    # B-420 (/stress A1.3 v9 Mode B P1-5 OOB, 2026-05-17): select_option env
    # dispatch telemetry — distinguishes JS-exception / obs_nodes_info-missing
    # / dispatch-completed cases that previously collapsed under a bare
    # `logger.warning + create_none_action()` silent no-op. Empirical
    # 195/738 archive select_option rows had action_success=false + no
    # page_change with no taxonomy attribution. None when step did not
    # invoke select_option.
    select_option_meta: Optional[Dict[str, Any]] = None
    # B-450 (/stress A1.4 P0-3-B codex OOB, 2026-05-17): select_option_meta
    # retry-overwrite split, symmetric with locator_route_meta_primary/retry
    # (B-440). Pre-fix the runner wrote `select_option_meta_primary` to
    # step_record but the dataclass/defaults/PAPER_GRADE_STEP_OPTIONAL_KEYS
    # only listed `select_option_meta` — "ghost field" outside the canonical
    # schema. `fill_step_defaults` did not backfill; archive readers could
    # not produce per-step primary/retry split for paper §3.5 select_option
    # sub-taxonomy. Codex Mode B verified by grep: schema 0 / types 0 /
    # runner 1 mention. Aligns select_option half-landed retry-split with
    # the locator_route_meta full-landed pair.
    select_option_meta_primary: Optional[Dict[str, Any]] = None
    select_option_meta_retry: Optional[Dict[str, Any]] = None
    # B-284 fix (2026-05-16, A1.8): paper §3.5.1 cite these B0 proxy-specific
    # fields for GLM-rescue de-biasing audit. Pre-fix the runner wrote them to
    # JSONL (`runner/main.py:1663-1669`) but they were absent from this dataclass
    # AND from `schema_migrations/v2.py STEP_RECORD_V2_DEFAULTS` → "ghost
    # fields" outside the canonical schema. Reviewer reading paper §3.5.1 could
    # not find them in the dataclass. Adding them here brings the catalog in
    # sync with what runner actually writes (paper-grade reproducibility).
    # Status pending B-262 advisor decision on Qwen official API channel; if
    # GLM fallback retires, these stay as archive-read fields (runner stops
    # writing, paper §3.5.1 prose retires).
    retry_action_applied: Optional[bool] = None
    retry_action_type: Optional[str] = None
    # B-398 (/stress A1.1 v8 Mode A+B P0-3 overlap, 2026-05-16): explicit
    # `attempted` field so attempted-but-failed GLM cases are distinguishable
    # from never-tried (both used to collapse to `used=None`). Runner now
    # emits `attempted=True` whenever the proxy invoked GLM regardless of
    # success/failure outcome.
    glm_fallback_attempted: Optional[bool] = None
    glm_fallback_used: Optional[bool] = None
    glm_fallback_latency_ms: Optional[float] = None
    glm_original_fail_reason: Optional[str] = None
    # B-472 (/stress A1.5b Phase 1 P1-9-A, 2026-05-17): control-injected
    # action provenance. helpers.py `_anti_repeat_control` (L207) +
    # `_no_early_finish_control` (L228) return `(fallback_action, reason)`
    # — fallback REPLACES agent's emitted action. Pre-fix step record's
    # `action` field carried the synthetic fallback indistinguishably from
    # an agent-emitted action; paper §3 action taxonomy 把合成 scroll/type
    # 当 agent emission → taxonomy 静默污染. None when no control fired;
    # {"type": "anti_repeat"|"no_early_finish", "original_action": dict,
    # "reason": str} when fired. Runtime write path = Phase 2 audit slot
    # (`_run_episode` body L984+), this dataclass + defaults catalog
    # schema-only land first per B-280 paper-grade catalog discipline.
    control_intervention: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeSummaryV2:
    schema_version: str
    run_id: str
    condition_id: str
    benchmark: str
    benchmark_site: str
    task_id: int
    seed: int
    success: bool
    score: float
    steps: int
    retries: int
    no_op_rate: float
    page_unchanged_rate: float
    total_latency_ms: float
    p95_step_latency_ms: float
    total_tokens: int
    total_model_cost_usd: float
    total_cost_usd: float
    total_router_overhead_cost_usd: float
    total_router_overhead_ms: float
    total_energy_kwh: Optional[float]
    total_co2e_kg: Optional[float]
    escalation_count: int
    trigger_distribution: Dict[str, int]
    benchmark_noise: bool
    benchmark_noise_category: Optional[str]
    artifacts_dir: str
    state_change_reason_distribution: Dict[str, int] = field(default_factory=dict)
    checklist_completion_rate: Optional[float] = None
    checklist_failed_items: Optional[int] = None
    error: Optional[str] = None
    busy_wait_free_steps: int = 0
    # Total wall time spent in busy-wait stalls that did not consume a step
    # from the budget — added retroactively (RU-4); old data has 0 + audit warning.
    busy_wait_total_ms: float = 0.0
    wasted_cost_usd: float = 0.0
    wasted_energy_kwh: float = 0.0
    component_breakdown: Optional[Dict[str, float]] = None
    total_input_cost_usd: float = 0.0
    total_output_cost_usd: float = 0.0
    total_obs_prepare_cost_usd: float = 0.0
    agent_finished: Optional[bool] = None
    # Energy completeness diagnostics (RU-5): partial=True when any step in
    # the episode lacks an energy reading (NVML probe failed mid-episode).
    energy_partial: bool = False
    energy_step_complete_count: int = 0
    # B-193 (/stress A1.4b-ii codex B-ii-2, P1 OOB): runner stamps these
    # 3 paper §3.5 transparency fields in normal + exception paths
    # (`runner/main.py:1936-1953` + `:896-899`) per A1.4a B-166/B-167/B-168.
    # Pre-fix: the dataclass + `aggregate_condition_metrics` ignored them
    # entirely, so paper §3.5 claims `trajectory_incomplete_rate per cell`
    # but the aggregate layer could not produce that rate (transparency
    # metric structurally unproducible). Now they are part of the schema,
    # downstream aggregator (B-193 metrics.py) emits per-cell rates.
    trajectory_incomplete: bool = False
    unknown_failure_reasons: Dict[str, int] = field(default_factory=dict)
    partial_recovery_step_count: int = 0
    # B-403 (/stress A1.1 v8 Mode B P1-9, 2026-05-16): per-episode count of
    # steps with `image_meta.image_encode_error > 0`. Agent comments at
    # `qwen3vl_agent.py:355-363` + `gemma3vl_agent.py:330-336` mandated
    # "aggregate_*.py MUST symmetric-exclude steps with image_encode_error
    # > 0" for paper-grade cross-baseline SR comparability. Pre-fix the
    # contract existed in comments only — no aggregator implemented the
    # exclusion. Now: runner stamps the per-episode count here, episode-
    # level aggregators can either filter (drop infra-failed episodes) or
    # annotate (disclosure column) without re-reading step JSONL.
    image_encode_error_step_count: int = 0
    # §139.8: `adjusted_success` / `fp_reason` were removed — the post-hoc
    # na_fp / eval_fp filter layer is retired (fixed at the source: B-91
    # evaluator empty-pred guard + N/A task exclusion at load time). `success`
    # is now the canonical paper-grade outcome. Archived pre-§139.8 summaries
    # may still carry these keys on disk (harmless, unread).
    # B-462 (/stress A1.5b Phase 1 P0-3-B codex OOB, 2026-05-17): Option K
    # covariate substrate — `aggregate_trajectory_covariates.py:82-88` (B-389)
    # 期待 episode-level wallclock anchors to time-order reset_post_interrupt
    # events vs episode lifetime → `is_after_reset` / `prior_event_count`
    # covariates 才有数据 substrate. Pre-fix runner 不 stamp 任何 timestamp 到
    # summary, aggregator `ep_start_dt is None` fallback fires for every
    # episode → covariates 失效 across the board. ISO-8601 string per A1.19
    # P1-5-A* explicit-parse contract. Both Optional[str] — legacy summary
    # missing → aggregator falls through to filesystem scan (B-389 robustness).
    wallclock_start: Optional[str] = None
    wallclock_end: Optional[str] = None
    # B-461 (/stress A1.5b Phase 1 P0-2-C gemini OOB, 2026-05-17): exception-
    # path EpisodeSummaryV2 may complete summary write without evaluator
    # actually scoring task — `success=False` hardcode is conservative but
    # without a re-evaluation gate the false-negative is final (resume gate
    # accepts summary, skips re-run). Set True in exception path; resume
    # gate (`runner/main.py:552-619`) detects → force re-run instead of
    # skip. Quarantine-rerun pattern preferred over naive last-step success
    # inference (which conflates `action_success="stop"` with task-level
    # evaluator outcome; agent self-claim ≠ url_match / program_html / etc).
    needs_reevaluation: bool = False
    # B-460 (/stress A1.5b Phase 1 P0-1-ABC 3-AI overlap, 2026-05-17):
    # resume fingerprint = sha256[:16] of (cfg.model.revision +
    # cfg.backends[backend_id].revision + max_new_tokens + temperature +
    # paper_grade + observation_mode + transformers_version + prompt_hash).
    # B-169 quarantine 6-tuple (run_id/condition_id/seed/site/task_id/
    # schema_version) caught path identity but missed *experiment identity*:
    # restart with changed model SHA / runtime params / prompt template
    # silently ingested old summaries. Now identity gate compares loaded.
    # resume_fingerprint vs current state — mismatch → quarantine + rerun.
    # Pre-fix legacy summaries have `None` → mismatch triggers from-scratch
    # paper-grade rerun (Phase 1a 还没 fire, no real loss).
    resume_fingerprint: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunSummaryV2:
    schema_version: str
    run_id: str
    benchmark: str
    phase: str
    total_conditions: int
    total_episodes: int
    condition_metrics: List[Dict[str, Any]]
    assumptions: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# B-281 fix (2026-05-16, A1.8): derive REQUIRED set dynamically from dataclass
# fields with no default → dataclass is the single source of truth. Pre-fix the
# hand-maintained 25-string set drifted from the dataclass (24 required + 12
# optional + 1 schema-version-default), creating a latent path where a future
# required field would slip past `validate_step_record_v2` silently.
def _required_field_names(cls: type) -> frozenset:
    return frozenset(
        f.name
        for f in dataclasses.fields(cls)
        if f.default is MISSING and f.default_factory is MISSING
    )


# Computed at import time so callers (e.g. `validate_run.py`) can still import
# REQUIRED_STEP_FIELDS_V2 as a constant. Names match the dataclass exactly;
# add a required field there → it lands here without touching the validator.
REQUIRED_STEP_FIELDS_V2 = _required_field_names(StepRecordV2) | {"schema_version"}
REQUIRED_EPISODE_FIELDS_V2 = _required_field_names(EpisodeSummaryV2) | {"schema_version"}
REQUIRED_RUN_FIELDS_V2 = _required_field_names(RunSummaryV2) | {"schema_version"}


# B-280 fix (2026-05-16, A1.8): paper-grade critical OPTIONAL fields. Each MUST
# be present as a key (value MAY be None). Pre-fix `parse_valid` / `image_meta` /
# `locator_route_meta` etc. were typed Optional → validator did not require the
# KEY → silent omission was indistinguishable from explicit None. Paper §3
# evidence-layer claims (locator-route ON_TARGET rate, image_over_cap audit,
# parse-failure taxonomy) rely on the key being present even when value is None.
PAPER_GRADE_STEP_OPTIONAL_KEYS = frozenset({
    "parse_valid",
    "parse_failure_reason",
    "image_meta",
    "locator_route_meta",
    "locator_route_meta_primary",  # B-440 retry-overwrite split
    "locator_route_meta_retry",    # B-440 retry-overwrite split
    "select_option_meta",  # B-420
    "select_option_meta_primary",  # B-450 retry-overwrite split (symmetric w/ B-440)
    "select_option_meta_retry",    # B-450 retry-overwrite split (symmetric w/ B-440)
    "agent_visible_changed",
    "control_intervention",  # B-472 control-injected action provenance
})


# B-280 fix (2026-05-16, A1.8): per-field type expectations for the validator.
# Only "container-shape" types are checked here (dict / list / int / str / bool /
# float / None) — not deep nested key types. Goal: reject `{"som": null,
# "latency_ms": "0"}` at the write boundary, not type-check every nested key.
# Map: field name → tuple of acceptable types. `type(None)` means None is OK.
_STEP_FIELD_TYPES: Dict[str, tuple] = {
    "schema_version": (str,),
    "run_id": (str,),
    "condition_id": (str,),
    "benchmark": (str,),
    "benchmark_site": (str,),
    "task_id": (int,),
    "seed": (int,),
    "step_idx": (int,),
    "som": (dict,),
    "observation_mode": (str,),
    "router": (dict,),
    "module_flags": (dict,),
    "action_type": (str,),
    "action": (dict,),
    "action_success": (bool,),
    "page_changed": (bool,),
    "latency_ms": (dict,),
    "tokens": (dict,),
    "cost_usd": (dict,),
    "energy": (dict,),
    "retry_count": (int,),
    "error_category": (str, type(None)),
    "artifact_paths": (dict,),
    "reward": (int, float),
    "done": (bool,),
}


def _validate_required_and_version(
    record: Dict[str, Any], required: frozenset, label: str
) -> None:
    missing = sorted(required - set(record.keys()))
    if missing:
        raise ValueError(f"{label} missing required fields: {missing}")
    if record.get("schema_version") != SCHEMA_VERSION_V2:
        raise ValueError(
            f"{label} unexpected schema_version={record.get('schema_version')!r} "
            f"(expected {SCHEMA_VERSION_V2!r})"
        )


_COST_USD_REQUIRED_KEYS = frozenset({"input", "output", "model", "router_overhead", "total"})


def validate_step_record_v2(record: Dict[str, Any]) -> None:
    """Paper-grade step record validator (B-280 / B-281 fix 2026-05-16, A1.8).

    Checks:
      1. All REQUIRED fields (derived from `StepRecordV2` dataclass) present.
      2. `schema_version == SCHEMA_VERSION_V2`.
      3. Type shape for each REQUIRED field per `_STEP_FIELD_TYPES`.
      4. Paper-grade critical optional KEYS (PAPER_GRADE_STEP_OPTIONAL_KEYS) must
         be present even if value is None — explicit None vs missing distinction
         is paper §3 evidence-layer contract (B-280).
      5. B-338 (/stress A1.9 Mode B F7, 2026-05-16): `cost_usd` nested keys
         must include all of {input, output, model, router_overhead, total}
         (paper §3.5 cost breakdown disclosure depends on these). Pre-fix
         validator checked `cost_usd is dict` only, leaving silent drift
         possible (runner could rename "model" → "llm" → `compute_component_
         breakdown.get("model", 0)` silently zeros out paper §3 model
         cost number).

    Raises:
        ValueError on any contract violation. Callers (runner write boundary,
        post-hoc validate_run.py) are expected to fail-loud rather than recover.
    """
    _validate_required_and_version(record, REQUIRED_STEP_FIELDS_V2, "StepRecordV2")
    bad_types = []
    for fname, expected in _STEP_FIELD_TYPES.items():
        val = record.get(fname)
        if not isinstance(val, expected):
            bad_types.append(
                f"{fname}: expected {tuple(t.__name__ for t in expected)}, got {type(val).__name__}={val!r}"
            )
    if bad_types:
        raise ValueError(
            f"StepRecordV2 type mismatch on {len(bad_types)} field(s): "
            + "; ".join(bad_types)
        )
    missing_critical = sorted(PAPER_GRADE_STEP_OPTIONAL_KEYS - set(record.keys()))
    if missing_critical:
        raise ValueError(
            f"StepRecordV2 missing paper-grade critical optional keys (value may be None): "
            f"{missing_critical}. Paper §3 evidence-layer contract requires presence."
        )
    # B-444 (/stress A1.25 P1-8-B* codex OOB, 2026-05-17): nested telemetry
    # semantics validator. Pre-fix `locator_route_meta={}` or
    # `{"success": "false"}` (string instead of bool) passed silently —
    # downstream denominator logic later treated malformed records as
    # falsey/truthy depending on implementation, exactly the silent
    # pipeline corruption codex Mode B flagged. Fail-loud at write boundary.
    for meta_key in ("locator_route_meta", "locator_route_meta_primary",
                     "locator_route_meta_retry"):
        meta = record.get(meta_key)
        if meta is None:
            continue
        if not isinstance(meta, dict):
            raise ValueError(
                f"StepRecordV2.{meta_key} expected dict-or-None, got {type(meta).__name__}={meta!r}"
            )
        # Empty dict is suspicious (real dispatch always populates fields)
        # but legacy archive rows may have {} — accept but require at least
        # the success key when non-empty.
        if meta and "success" not in meta:
            raise ValueError(
                f"StepRecordV2.{meta_key} non-empty dict missing 'success' key: {meta!r}"
            )
        if "success" in meta and not isinstance(meta["success"], (bool, type(None))):
            raise ValueError(
                f"StepRecordV2.{meta_key}.success expected bool-or-None, got "
                f"{type(meta['success']).__name__}={meta['success']!r}"
            )
        if "action_kind" in meta and meta["action_kind"] not in {
            "click", "type", "type_coord", "hover", "upload", "clear",
            "select_option", None
        }:
            raise ValueError(
                f"StepRecordV2.{meta_key}.action_kind unexpected value: "
                f"{meta['action_kind']!r}"
            )
    # B-480 (/stress A1.25 GRL Chunk 2 P1-4-B*, 2026-05-17): nested validator
    # now loops over all three select_option_meta variants (legacy + primary
    # + retry) mirroring the locator_route_meta loop above. Pre-fix only
    # `select_option_meta` got the nested success-bool check — the `_primary`
    # / `_retry` fields could carry malformed payloads silently.
    # B-481 (/stress A1.25 GRL Chunk 2 P0-2-AB*, 2026-05-17): also validate
    # the new structured fields `matched` (bool), `match_stage` (enum),
    # `target_type` (enum). Closes the "success=True but actually no_match"
    # silent-pipeline-corruption that codex Mode B + Claude Mode A + parallel
    # A1.4 codex (B-453) all caught independently.
    _SELECT_MATCH_STAGES = {None, "exact", "ci", "fuzzy", "index", "none"}
    _SELECT_TARGET_TYPES = {None, "select", "css"}
    for sel_key in ("select_option_meta", "select_option_meta_primary",
                    "select_option_meta_retry"):
        sel_meta = record.get(sel_key)
        if sel_meta is None:
            continue
        if not isinstance(sel_meta, dict):
            raise ValueError(
                f"StepRecordV2.{sel_key} expected dict-or-None, got "
                f"{type(sel_meta).__name__}={sel_meta!r}"
            )
        if "success" in sel_meta and not isinstance(sel_meta["success"], (bool, type(None))):
            raise ValueError(
                f"StepRecordV2.{sel_key}.success expected bool-or-None, got "
                f"{type(sel_meta['success']).__name__}={sel_meta['success']!r}"
            )
        if "matched" in sel_meta and not isinstance(sel_meta["matched"], (bool, type(None))):
            raise ValueError(
                f"StepRecordV2.{sel_key}.matched expected bool-or-None, got "
                f"{type(sel_meta['matched']).__name__}={sel_meta['matched']!r}"
            )
        if "match_stage" in sel_meta and sel_meta["match_stage"] not in _SELECT_MATCH_STAGES:
            raise ValueError(
                f"StepRecordV2.{sel_key}.match_stage unexpected value: "
                f"{sel_meta['match_stage']!r} (expected {sorted(s for s in _SELECT_MATCH_STAGES if s is not None)})"
            )
        if "target_type" in sel_meta and sel_meta["target_type"] not in _SELECT_TARGET_TYPES:
            raise ValueError(
                f"StepRecordV2.{sel_key}.target_type unexpected value: "
                f"{sel_meta['target_type']!r} (expected {sorted(s for s in _SELECT_TARGET_TYPES if s is not None)})"
            )
    # B-338: nested cost_usd key validation.
    cost = record.get("cost_usd", {})
    if isinstance(cost, dict):
        missing_cost_keys = sorted(_COST_USD_REQUIRED_KEYS - set(cost.keys()))
        if missing_cost_keys:
            raise ValueError(
                f"StepRecordV2.cost_usd missing required nested keys "
                f"{missing_cost_keys}. Paper §3.5 cost breakdown depends on "
                "{input, output, model, router_overhead, total} all present."
            )


def validate_episode_summary_v2(record: Dict[str, Any]) -> None:
    """Paper-grade episode summary validator (B-285 fix 2026-05-16, A1.8).

    Mirrors `validate_step_record_v2`. Required-field set derived from
    `EpisodeSummaryV2` dataclass. Spot-checks the 3 paper §1 hero fields
    (`success` is bool, `score` is float, `steps` is int) explicitly — these
    are the type-coercion attack surface flagged by codex Mode B F3 (B-283).
    """
    _validate_required_and_version(record, REQUIRED_EPISODE_FIELDS_V2, "EpisodeSummaryV2")
    bad_types = []
    if not isinstance(record.get("success"), bool):
        bad_types.append(f"success: expected bool, got {type(record.get('success')).__name__}={record.get('success')!r}")
    if not isinstance(record.get("score"), (int, float)):
        bad_types.append(f"score: expected int|float, got {type(record.get('score')).__name__}")
    if not isinstance(record.get("steps"), int):
        bad_types.append(f"steps: expected int, got {type(record.get('steps')).__name__}")
    if not isinstance(record.get("task_id"), int):
        bad_types.append(f"task_id: expected int, got {type(record.get('task_id')).__name__}")
    if bad_types:
        raise ValueError(
            f"EpisodeSummaryV2 type mismatch on {len(bad_types)} field(s): "
            + "; ".join(bad_types)
        )


def validate_run_summary_v2(record: Dict[str, Any]) -> None:
    """Paper-grade run summary validator (B-296 fix 2026-05-16, A1.8).

    Required-field set derived from `RunSummaryV2` dataclass.
    `condition_metrics` must be a list; `assumptions` must be a dict.
    """
    _validate_required_and_version(record, REQUIRED_RUN_FIELDS_V2, "RunSummaryV2")
    if not isinstance(record.get("condition_metrics"), list):
        raise ValueError(
            f"RunSummaryV2.condition_metrics: expected list, "
            f"got {type(record.get('condition_metrics')).__name__}"
        )
    if not isinstance(record.get("assumptions"), dict):
        raise ValueError(
            f"RunSummaryV2.assumptions: expected dict, "
            f"got {type(record.get('assumptions')).__name__}"
        )
    if not isinstance(record.get("total_episodes"), int):
        raise ValueError(
            f"RunSummaryV2.total_episodes: expected int, "
            f"got {type(record.get('total_episodes')).__name__}"
        )
