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
    # B-488 (/stress A1.25 GRL Chunk 3 P1-2-BC* gemini + codex dual OOB,
    # 2026-05-17): per-step browser dialog events (confirm / alert /
    # beforeunload accepted; prompt dismissed). None when no dialog fired
    # during the step (most common); else a list of payloads. Paper §3.5.1
    # cross-baseline misclick blast-radius evidence layer — VWA shared-
    # account architecture (cls Blake / red Marvels) makes wrong delete /
    # submit cross-task contaminating; cross-baseline misclick rate
    # differs → asymmetric SR contamination via state-mutation
    # amplification. Reviewer can now grep `dialog_meta` per (site, model,
    # mode) to compute dialog_acceptance_rate.
    dialog_meta: Optional[List[Dict[str, Any]]] = None
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
    # B-497 (/stress A1.5b Phase 1 P1-9-A, 2026-05-17): control-injected
    # action provenance. helpers.py `_anti_repeat_control` (L207) +
    # `_no_early_finish_control` (L228) return `(fallback_action, reason)`
    # — fallback REPLACES agent's emitted action. Pre-fix step record's
    # `action` field carried the synthetic fallback indistinguishably from
    # an agent-emitted action; paper §3 action taxonomy 把合成 scroll/type
    # 当 agent emission → taxonomy 静默污染. None when no control fired;
    # {"type": "anti_repeat"|"no_early_finish", "original_action": dict,
    # "reason": str} when fired. Runtime write path landed at B-546
    # (/stress A1.5b Phase 2 P1-6-AB Claude F2 + codex B-541 cross-val).
    control_intervention: Optional[Dict[str, Any]] = None
    # B-512 (/stress A1.5b Phase 2 P0-1-C gemini OOB, 2026-05-17): post-wrapper
    # canonical action form. Pre-fix step_record["action"] carried the
    # agent's RAW emit (B0 `scroll_direction:"down"` enum; B1/B2 `delta:[dx,dy]`
    # free-form pixel) → evidence layer JSONL asymmetric on action vocabulary.
    # Gemini Mode C attack: "cross-baseline ablation 假设 B0/B1/B2 玩 same game,
    # action vocabulary 不同 → reviewer 直接 reject capability comparison". Real
    # surface: wrapper at `p79/envs/vwa_wrapper.py:395-414` already normalizes
    # both forms to `create_scroll_action(direction=...)` (execution layer
    # identical since paper §67 schema reform) but the normalized form was
    # never recorded. `action_executed` makes the wrapper-level alignment
    # visible in step JSONL.
    #
    # B-553 (click/type extension, /stress A1.5 P1-3-AB* Claude+codex OOB,
    # 2026-05-17): extended from scroll-only to click + type dispatch paths.
    # Shape varies by branch:
    #   - scroll:      {"action_type": "scroll", "direction": <up|down|noop>}
    #   - click_eid:   {"action_type": "click", "dispatch_path":
    #                   "element_id_locator_route" | "element_id_framework",
    #                   "fallback": bool}
    #   - click_coord: {"action_type": "click", "dispatch_path":
    #                   "coord_mouse_click", "fallback": False}
    #   - type_eid:    {"action_type": "type", "dispatch_path":
    #                   "element_id_locator_route" | "element_id_framework"
    #                   | "noop_invalid_element_id", "fallback": bool}
    #   - type_coord:  {"action_type": "type", "dispatch_path":
    #                   "coord_locator_route" | "coord_keyboard_fallback",
    #                   "fallback": bool}
    # `fallback=True` means the Cluster 1 locator-route walk-up FAILED and
    # the wrapper fell back to legacy framework path; reviewer can grep
    # `action_executed.fallback==True` to count cross-baseline fallback
    # rate (B0 235B rarely falls back; B1/B2 4B more often). None when
    # wrapper did not emit (mock env, exception path, or non-normalized
    # action types like back/forward/tab/finish/stop).
    action_executed: Optional[Dict[str, Any]] = None
    # B-563 (/stress A1.22 P0-1-ABC* 3-AI overlap, 2026-05-17): cross-baseline
    # cost-basis declaration. `cost_usd.{input,output,model}` units differ by
    # baseline: B0 reports commercial-API USD margin (provider margin + network
    # egress + infrastructure overhead, `configs/exp_v2_B0_*.yaml:cost_api`
    # input_cost_per_1k=0.001 output=0.005); B1+B2 (local) report
    # electricity-derived USD (`avg_total_energy_kwh × electricity_rate`, no
    # margin). Pre-fix any aggregator pooling `cost_usd.total` across baselines
    # mixed API-USD with electricity-USD (unit collision ~1000×) — paper §1
    # "4-fold drop-in property" hero claim was unit-collision artifact, not
    # scientific property. A1.21 P0-7 / B-527 added `cost_unit_basis_for(baseline)`
    # to `generate_per_task_sr.py` but the basis lived only as a CSV column,
    # never reaching `aggregate_phase1_full_prereg_decision.py` canonical
    # producer or 2 other `aggregate_*.py` cost consumers. Adding the basis
    # at the step_record schema layer makes the unit visible at the JSONL
    # boundary — cross-baseline aggregators MUST stratify (or assert single
    # basis) before pooling cost. Enum:
    #   - "api_usd"                  — B0 commercial proxy (Bedrock margin)
    #   - "electricity_usd_derived"  — B1/B2 local (energy × rate)
    #   - "unknown"                  — mock backend or unrecognized type
    # `validate_step_record_v2` requires the key be present (value may be None
    # for archived rows). NeurIPS area chair defuse: paper §1 cost claim
    # rewritten as "average of within-baseline normalized cost ratios"
    # (Gemini Mode C F1 5-paragraph attack defuse), and any future cross-
    # baseline absolute cost number must cite this basis explicitly.
    cost_unit_basis: Optional[str] = None
    # B-564 (/stress A1.22 P0-5-A* Claude OOB, 2026-05-17): close `element_bbox`
    # ghost-field hole. Pre-fix `runner/main.py:2451` stamped step_record[
    # "element_bbox"] = [...] when `obs.obs_nodes_info` provided a union_bound
    # for the target element, but the field was absent from this dataclass +
    # `STEP_RECORD_V2_DEFAULTS` + `validate_step_record_v2` ⟶ A1.8 "schema
    # = source of truth" contract (B-280/B-281) partial leak. Reviewer
    # grepping dataclass for `element_bbox` finds nothing; grepping JSONL
    # finds rows where the field exists. Now declared canonical at the
    # schema layer; runner write site (B-564 companion) keeps existing
    # behavior (only present when bbox was extractable from obs_nodes_info).
    # Type: 4-float list `[x, y, w, h]` (pixel coords, viewport-frame).
    # None when step did not produce a click target (no element_id, or
    # element_id not in obs_nodes_info union_bound).
    element_bbox: Optional[List[float]] = None
    # B-565 (/stress A1.22 P0-2-C* Gemini OOB, 2026-05-17): cross-baseline
    # mixed-unit ADD warn flag. Pre-fix `runner/main.py:2240-2247
    # cost_usd.total = token_cost.total + router_overhead + obs_prepare` for
    # B0 adds API-USD (token cost) + electricity-USD (router + obs_prepare
    # scaffold) into one number ⟶ mathematically incoherent even before
    # cross-baseline pooling. Set True when (`cost_unit_basis != "electricity
    # _usd_derived"` AND (`cost_usd.router_overhead != 0` OR `cost_usd.obs_prepare
    # != 0`)) — i.e. B0 row with non-zero local scaffold cost. Aggregators
    # can detect and either re-derive total from `cost_usd.model` alone OR
    # disclose the warn flag count per cell. False for B1/B2 (single basis)
    # and for B0 rows where local scaffold cost happens to be 0 (router off,
    # zero-cost obs_prepare). None for archived pre-A1.22 rows.
    cost_total_mixed_unit_warn: Optional[bool] = None
    # B-569 (/stress A1.22 P1-11-A Claude, 2026-05-17): network retry
    # telemetry persist. Pre-fix `proxy_api_agent.py:809-810` emitted
    # `network_retry_count` + `network_retry_wait_ms` into meta but runner
    # only consumed `network_retry_wait_ms` for the `total_minus_retry`
    # arithmetic — fields themselves were dropped at the runner→JSONL
    # boundary. Paper §3.5 "B0 network retry rate per cell" disclosure
    # column structurally unreproducible from raw JSONL. Now persisted
    # as discrete step_record fields; B1/B2 always None (no equivalent
    # network retry — Optional typed so None is honest contract, not
    # 0-cast that would suggest "retried 0 times" vs "no retry concept").
    network_retry_count: Optional[int] = None
    network_retry_wait_ms: Optional[float] = None

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
    # B-1600 (/stress 深入审 Mode A P0-1-A*, 2026-05-18): retry-adjusted total
    # latency canonical estimand rollup. Sum of step-level
    # `latency.total_minus_retry` (B0 = total - network_retry_wait_ms; B1/B2 =
    # total since meta.network_retry_wait_ms is None). Schema field was added
    # at A2.7 Chunk 4 (B-1410) + metrics.py rollup _avg + aggregator carry,
    # but runner write path was deferred — A2.7-followup §B sweep tracked as
    # B-1410-fu. Without this write step, paper §1 retry-adjusted canonical
    # latency claim (per memory `project_cost_latency_canonical_estimand`
    # 2026-05-18 user-locked 3-axis estimand + paper §3.5.1 disclosure) has
    # no data substrate post-Pass-1 fire — Pass-1 succeeds structurally but
    # produces zero rollup data.
    total_latency_minus_retry_ms: Optional[float] = None
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
    # B-487 (/stress A1.5b Phase 1 P0-3-B codex OOB, 2026-05-17): Option K
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
    # B-486 (/stress A1.5b Phase 1 P0-2-C gemini OOB, 2026-05-17): exception-
    # path EpisodeSummaryV2 may complete summary write without evaluator
    # actually scoring task — `success=False` hardcode is conservative but
    # without a re-evaluation gate the false-negative is final (resume gate
    # accepts summary, skips re-run). Set True in exception path; resume
    # gate (`runner/main.py:552-619`) detects → force re-run instead of
    # skip. Quarantine-rerun pattern preferred over naive last-step success
    # inference (which conflates `action_success="stop"` with task-level
    # evaluator outcome; agent self-claim ≠ url_match / program_html / etc).
    needs_reevaluation: bool = False
    # B-485 (/stress A1.5b Phase 1 P0-1-ABC 3-AI overlap, 2026-05-17):
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
    # B-554 (/stress A1.5 P1-4-AB* Claude+codex OOB, 2026-05-17): archive
    # cohort sentinel for reward-override retirement.
    #   - `evaluator_authority_mode`: enum {"post_B545_vwa_score_only",
    #     None}. Post-B-545 (A1.5b Phase 2 commit `7832008`) episodes carry
    #     "post_B545_vwa_score_only" semantic: `success = bool(score >= 1.0)`
    #     from VWA evaluator with NO post-hoc adjustment. Legacy archive
    #     summaries from before B-545 lack the field → None default →
    #     downstream consumer reads None as "pre-B545 legacy cohort"
    #     (success was potentially override-baked per B-165 narrowing).
    #   - `reward_override_applied`: enum {False, None}. Post-B-545 always
    #     False (mechanism retired). Legacy archive reads None.
    # Mixed pre/post-B-545 archive aggregation MUST stratify by these
    # fields. Codex Weak claim #6 + Claude F5 cross-validation; closes
    # paper §3 estimand mixed-cohort vulnerability (top-tier reviewer
    # would attack archive aggregation as estimand schizophrenia
    # otherwise).
    evaluator_authority_mode: Optional[str] = None
    reward_override_applied: Optional[bool] = None

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
    "control_intervention",  # B-497 control-injected action provenance
    "dialog_meta",  # B-488 browser dialog telemetry (misclick blast radius evidence layer)
    "action_executed",  # B-512 wrapper-normalized canonical action form
    # B-566 (/stress A1.22 P0-5-A* + P0-1-ABC* + P0-2-C* cross-baseline
    # contract sealing, 2026-05-17): close cross-baseline ghost-field
    # cluster — `fallback_finish` was IN dataclass + DEFAULTS but missed
    # validator KEY-presence enforcement (silent omission allowed);
    # `element_bbox` was previously a pure ghost (now declared by B-564);
    # `cost_unit_basis` (B-563) declares the unit of `cost_usd.{input,
    # output,model}` per baseline so aggregators MUST stratify before
    # pooling; `cost_total_mixed_unit_warn` (B-565) flags B0 rows where
    # `cost_usd.total` was constructed as a mixed-unit ADD (API USD +
    # local-scaffold USD) so consumers detect single-row incoherence
    # before any cross-baseline pooling. Validator now requires KEY
    # presence (value may be None for archived rows).
    "fallback_finish",
    "element_bbox",
    "cost_unit_basis",
    "cost_total_mixed_unit_warn",
    # B-569 (/stress A1.22 P1-11-A): persist B0 network retry telemetry
    # as discrete fields (was meta-only, dropped at JSONL boundary). B1/B2
    # always None; B0 0 when no retries, >0 with count + accumulated wait
    # ms. Validator KEY-presence enforced so reviewer grepping JSONL for
    # `network_retry_count` finds the field on every row (None ≡ baseline
    # has no retry concept, not "0 retries").
    "network_retry_count",
    "network_retry_wait_ms",
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


# B-740 fix (/stress A1.8 cold-start P1-6-C* Gemini OOB, 2026-05-17): import-time
# invariant that `_STEP_FIELD_TYPES` stays synced with auto-derived
# `REQUIRED_STEP_FIELDS_V2`. Pre-fix: B-281 derived REQUIRED from the dataclass
# (auto-sync) but `_STEP_FIELD_TYPES` was hand-maintained → adding a required
# dataclass field would auto-update REQUIRED but silently leave the new field
# with NO type check. Gemini Mode C attack: "schema 全覆盖虚假安全感". Now any
# drift fails loudly at module import (caller can't proceed without fixing).
_STEP_FIELD_TYPES_KEYS = frozenset(_STEP_FIELD_TYPES.keys())
assert _STEP_FIELD_TYPES_KEYS == REQUIRED_STEP_FIELDS_V2, (
    "B-740 invariant: _STEP_FIELD_TYPES drift detected. "
    f"In TYPES not in REQUIRED: {_STEP_FIELD_TYPES_KEYS - REQUIRED_STEP_FIELDS_V2}; "
    f"In REQUIRED not in TYPES: {REQUIRED_STEP_FIELDS_V2 - _STEP_FIELD_TYPES_KEYS}. "
    "Adding a required field to StepRecordV2 dataclass requires a matching entry in _STEP_FIELD_TYPES."
)
del _STEP_FIELD_TYPES_KEYS  # don't pollute module namespace


# B-731 fix (/stress A1.8 cold-start P0-1-AC* 2-AI OOB, 2026-05-17): paper-grade
# critical OPTIONAL field VALUE-type contract. Pre-fix `validate_step_record_v2`
# only enforced KEY presence on PAPER_GRADE_STEP_OPTIONAL_KEYS (B-280); VALUE
# could be any type (string `"false"` truthy attack, the B-283 attack vector
# extended to step-level paper-grade optionals). Now: any present key with a
# non-conforming VALUE type raises ValueError at write boundary. None is always
# accepted (semantic: "not measured this step"). Cold-start Claude F2 + Gemini
# C2 dual-catch. Empirical smoke confirmed pre-fix accepted `parse_valid="false"`
# (string) and 5+ siblings — sibling propagation of B-283 fix that B-280 closed
# the KEY-presence half but left VALUE-type half open.
_STEP_OPTIONAL_FIELD_TYPES: Dict[str, tuple] = {
    "parse_valid": (bool, type(None)),
    "parse_failure_reason": (str, type(None)),
    "image_meta": (dict, type(None)),
    "image_meta_recorded": (bool,),
    "locator_route_meta": (dict, type(None)),
    "locator_route_meta_primary": (dict, type(None)),
    "locator_route_meta_retry": (dict, type(None)),
    "select_option_meta": (dict, type(None)),
    "select_option_meta_primary": (dict, type(None)),
    "select_option_meta_retry": (dict, type(None)),
    "agent_visible_changed": (bool, type(None)),
    "control_intervention": (dict, type(None)),
    "dialog_meta": (list, type(None)),
    "action_executed": (dict, type(None)),
    "fallback_finish": (bool, type(None)),
    "element_bbox": (list, type(None)),
    "cost_unit_basis": (str, type(None)),
    "cost_total_mixed_unit_warn": (bool, type(None)),
    "network_retry_count": (int, type(None)),
    "network_retry_wait_ms": (int, float, type(None)),
}


# B-732 fix (/stress A1.8 cold-start P0-2-C* Gemini OOB, 2026-05-17): paper-
# grade critical Episode-level OPTIONAL keys. Pre-fix `validate_episode_summary_
# v2` had no equivalent of `PAPER_GRADE_STEP_OPTIONAL_KEYS` — runner exception
# path could silently drop `evaluator_authority_mode` (B-554) and `reward_
# override_applied` (B-554) → `fill_defaults` backfills to None → aggregator
# sees new data as "pre-B-545 legacy cohort" → A1.5b cohort isolation defense
# silently bypassed. User directive 2026-05-17 cold-start: "archive 不进 paper
# scope" → full 7-field enforce (no archive backward-compat hook). Fresh
# Pass-1+Pass-2 paper-grade runs MUST stamp all 7 sentinels.
PAPER_GRADE_EPISODE_OPTIONAL_KEYS = frozenset({
    # B-545 / B-554 archive cohort isolation sentinels
    "evaluator_authority_mode",
    "reward_override_applied",
    # B-487 Option K covariate anchors (paper §4 GLMM substrate)
    "wallclock_start",
    "wallclock_end",
    # B-485 resume identity gate (paper-grade rerun protocol)
    "resume_fingerprint",
    # B-486 quarantine flag (crash-before-evaluator distinguishing)
    "needs_reevaluation",
    # B-193 paper §3.5 transparency telemetry
    "trajectory_incomplete",
})


# B-732 companion + B-739 fix (/stress A1.8 cold-start P1-5-A Claude, 2026-05-17):
# per-field VALUE-type contract for Episode summary, both REQUIRED and OPTIONAL.
# Pre-fix `validate_episode_summary_v2` only type-checked 4 fields (success /
# score / steps / task_id); 30+ other fields including cost/latency/energy hero
# fields were pass-through. Cross-baseline paper §1 cost claim defensibility
# depends on these all being numeric — string-coercion attack on
# `total_cost_usd` etc. would slip past. Now: full type map covering the
# numeric hero fields + B-732 sentinels.
_EPISODE_FIELD_TYPES: Dict[str, tuple] = {
    # core typed hero fields
    "schema_version": (str,),
    "run_id": (str,),
    "condition_id": (str,),
    "benchmark": (str,),
    "benchmark_site": (str,),
    "task_id": (int,),
    "seed": (int,),
    "steps": (int,),
    "retries": (int,),
    "no_op_rate": (int, float),
    "page_unchanged_rate": (int, float),
    "total_latency_ms": (int, float),
    # B-1600 (/stress 深入审 Mode A P0-1-A*, 2026-05-18): rollup-write companion
    "total_latency_minus_retry_ms": (int, float, type(None)),
    "p95_step_latency_ms": (int, float),
    "total_tokens": (int,),
    "total_model_cost_usd": (int, float),
    "total_cost_usd": (int, float),
    "total_router_overhead_cost_usd": (int, float),
    "total_router_overhead_ms": (int, float),
    "total_energy_kwh": (int, float, type(None)),
    "total_co2e_kg": (int, float, type(None)),
    "escalation_count": (int,),
    "trigger_distribution": (dict,),
    "benchmark_noise": (bool,),
    "benchmark_noise_category": (str, type(None)),
    "artifacts_dir": (str,),
}


# Optional-field VALUE-types for Episode summary (B-732 companion). Sentinels +
# transparency telemetry covered. None is semantic "not measured / archive
# row".
_EPISODE_OPTIONAL_FIELD_TYPES: Dict[str, tuple] = {
    "evaluator_authority_mode": (str, type(None)),
    "reward_override_applied": (bool, type(None)),
    "wallclock_start": (str, type(None)),
    "wallclock_end": (str, type(None)),
    "resume_fingerprint": (str, type(None)),
    "needs_reevaluation": (bool,),
    "trajectory_incomplete": (bool,),
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
    # B-731 fix (/stress A1.8 cold-start P0-1-AC* 2-AI OOB, 2026-05-17): VALUE-type
    # check on paper-grade critical optionals. Pre-fix only KEY-presence enforced;
    # string `"false"` for `parse_valid` / `agent_visible_changed` / `fallback_finish`
    # would silently pass → downstream `bool(row.get(...))` truthy-cast → SR / FP
    # rate inflated. Empirical smoke (Claude F2) confirmed 6+ sibling fields
    # accepted poisoned types pre-fix. None always accepted (semantic: not
    # measured this step).
    bad_optional_types = []
    for fname, expected in _STEP_OPTIONAL_FIELD_TYPES.items():
        if fname not in record:
            continue  # presence check already enforced above
        val = record[fname]
        if not isinstance(val, expected):
            bad_optional_types.append(
                f"{fname}: expected {tuple(t.__name__ for t in expected)}, "
                f"got {type(val).__name__}={val!r}"
            )
    if bad_optional_types:
        raise ValueError(
            f"StepRecordV2 paper-grade optional field VALUE type mismatch on "
            f"{len(bad_optional_types)} field(s): " + "; ".join(bad_optional_types)
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
    # B-505 (/stress A1.25 GRL Chunk 2 P1-4-B*, 2026-05-17): nested validator
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
    """Paper-grade episode summary validator (B-285 fix 2026-05-16, A1.8;
    extended /stress A1.8 cold-start B-732/B-735/B-739, 2026-05-17).

    Mirrors `validate_step_record_v2`. Required-field set derived from
    `EpisodeSummaryV2` dataclass.

    Checks (in order):
      1. REQUIRED fields (derived from dataclass) present + `schema_version`
         matches `SCHEMA_VERSION_V2`.
      2. **Hero fields explicit type-check** (B-285 + B-735 cold-start):
         `success` must be bool (NOT bool-subclass-of-int loophole; the existing
         B-285 check is correct). `score` must be int|float **excluding bool**
         (B-735 fix: pre-fix `isinstance(True, (int, float)) == True` admitted
         `score=True` truthy bypass; Gemini Mode C C1 cold-start OOB attack).
         `steps`/`task_id` must be int.
      3. **All REQUIRED fields container-shape type-check** (B-739 cold-start):
         `_EPISODE_FIELD_TYPES` covers cost/latency/energy/escalation hero
         fields. Pre-fix only 4/40+ fields were checked → string-coercion
         attack on `total_cost_usd` etc. slipped past validator.
      4. **`PAPER_GRADE_EPISODE_OPTIONAL_KEYS` presence** (B-732 cold-start):
         B-545 cohort isolation sentinels (`evaluator_authority_mode`,
         `reward_override_applied`) + Option K covariate anchors
         (`wallclock_start`, `wallclock_end`) + resume identity gate
         (`resume_fingerprint`) + quarantine flag (`needs_reevaluation`) +
         transparency (`trajectory_incomplete`) MUST be present at write
         boundary (value may be None except `needs_reevaluation`/
         `trajectory_incomplete` which default False per dataclass). User
         directive 2026-05-17: "archive 不进 paper scope" → no backward-compat
         hook; fresh Pass-1+Pass-2 paper-grade runs MUST stamp all 7.
      5. **VALUE-type on each optional key** (B-732 companion): if present,
         the value must match `_EPISODE_OPTIONAL_FIELD_TYPES` (string `"false"`
         for `reward_override_applied` would otherwise be truthy-cast).

    Raises:
        ValueError on any contract violation. Runner write boundary expected
        to fail-loud rather than recover.
    """
    _validate_required_and_version(record, REQUIRED_EPISODE_FIELDS_V2, "EpisodeSummaryV2")
    bad_types = []
    # Hero field explicit checks (B-285 + B-735 cold-start).
    if not isinstance(record.get("success"), bool):
        bad_types.append(f"success: expected bool, got {type(record.get('success')).__name__}={record.get('success')!r}")
    # B-735 fix: exclude bool from score's int|float acceptance. Pre-fix
    # `isinstance(True, (int, float)) == True` because bool subclasses int;
    # `score=True` (Python literal) would silently pass → JSONL `{"score": true}`
    # → downstream strong-type Rust/Go consumer crash OR weak-type `mean()`
    # silently treats True → 1.0.
    score_val = record.get("score")
    if isinstance(score_val, bool) or not isinstance(score_val, (int, float)):
        bad_types.append(
            f"score: expected int|float (NOT bool), "
            f"got {type(score_val).__name__}={score_val!r}"
        )
    if not isinstance(record.get("steps"), int) or isinstance(record.get("steps"), bool):
        bad_types.append(f"steps: expected int (NOT bool), got {type(record.get('steps')).__name__}")
    if not isinstance(record.get("task_id"), int) or isinstance(record.get("task_id"), bool):
        bad_types.append(f"task_id: expected int (NOT bool), got {type(record.get('task_id')).__name__}")
    # B-739 cold-start: full REQUIRED type-shape check via _EPISODE_FIELD_TYPES.
    # Hero fields above are checked twice (once by name, once by map) — that's
    # OK; the explicit checks have richer error messages, the map ensures no
    # field falls through unchecked.
    for fname, expected in _EPISODE_FIELD_TYPES.items():
        if fname in {"success", "score", "steps", "task_id"}:
            continue  # already explicitly checked above
        if fname not in record:
            continue  # REQUIRED missing already caught by _validate_required_and_version
        val = record[fname]
        # Exclude bool from int|float acceptance for numeric hero fields
        # (B-735 lineage: bool subclasses int).
        if expected == (int, float) or expected == (int, float, type(None)):
            if isinstance(val, bool):
                bad_types.append(
                    f"{fname}: expected {tuple(t.__name__ for t in expected)} "
                    f"(NOT bool), got bool={val!r}"
                )
                continue
        if not isinstance(val, expected):
            bad_types.append(
                f"{fname}: expected {tuple(t.__name__ for t in expected)}, "
                f"got {type(val).__name__}={val!r}"
            )
    if bad_types:
        raise ValueError(
            f"EpisodeSummaryV2 type mismatch on {len(bad_types)} field(s): "
            + "; ".join(bad_types)
        )
    # B-732 cold-start: PAPER_GRADE_EPISODE_OPTIONAL_KEYS presence enforcement.
    # B-545 cohort isolation sentinels + Option K covariate anchors MUST be
    # present at write boundary even if value is None (consistent with
    # PAPER_GRADE_STEP_OPTIONAL_KEYS pattern).
    missing_critical = sorted(PAPER_GRADE_EPISODE_OPTIONAL_KEYS - set(record.keys()))
    if missing_critical:
        raise ValueError(
            f"EpisodeSummaryV2 missing paper-grade critical optional keys "
            f"(value may be None for sentinel fields; bool default False for "
            f"transparency fields): {missing_critical}. B-545 cohort isolation "
            f"+ Option K covariate + resume identity gate require presence at "
            f"write boundary."
        )
    # B-732 companion: VALUE-type check on each present optional key. Catches
    # string `"false"` truthy-cast attack on B-545 sentinels.
    bad_optional_types = []
    for fname, expected in _EPISODE_OPTIONAL_FIELD_TYPES.items():
        if fname not in record:
            continue
        val = record[fname]
        if not isinstance(val, expected):
            bad_optional_types.append(
                f"{fname}: expected {tuple(t.__name__ for t in expected)}, "
                f"got {type(val).__name__}={val!r}"
            )
    if bad_optional_types:
        raise ValueError(
            f"EpisodeSummaryV2 paper-grade optional VALUE type mismatch on "
            f"{len(bad_optional_types)} field(s): " + "; ".join(bad_optional_types)
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
