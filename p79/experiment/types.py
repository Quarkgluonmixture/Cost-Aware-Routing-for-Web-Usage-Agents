from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

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
    # B-156 (/stress A1.3 v8 Claude F5 + codex P2-B7 dual catch, 2026-05-16):
    # locator-route dispatch result (Cluster 1 B-01/02/33 fix). Without this
    # field, paper §3 evidence layer cannot quantify locator-route ON_TARGET
    # rate from JSONL alone — the Tier 10 sweep 94.4% off-target → >80%
    # ON_TARGET goal becomes structurally unverifiable. Schema:
    #   {success, fallback_used, target_tag, error, action_kind}
    # action_kind ∈ {click, type, hover, upload, clear}. None when step did
    # not invoke locator-route (e.g. scroll / wait / coord-only click).
    locator_route_meta: Optional[Dict[str, Any]] = None

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
    # §139.8: `adjusted_success` / `fp_reason` were removed — the post-hoc
    # na_fp / eval_fp filter layer is retired (fixed at the source: B-91
    # evaluator empty-pred guard + N/A task exclusion at load time). `success`
    # is now the canonical paper-grade outcome. Archived pre-§139.8 summaries
    # may still carry these keys on disk (harmless, unread).

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


REQUIRED_STEP_FIELDS_V2 = {
    "schema_version",
    "run_id",
    "condition_id",
    "benchmark",
    "benchmark_site",
    "task_id",
    "seed",
    "step_idx",
    "som",
    "observation_mode",
    "router",
    "module_flags",
    "action_type",
    "action",
    "action_success",
    "page_changed",
    "latency_ms",
    "tokens",
    "cost_usd",
    "energy",
    "retry_count",
    "error_category",
    "artifact_paths",
    "reward",
    "done",
}


def validate_step_record_v2(record: Dict[str, Any]) -> None:
    missing = sorted(REQUIRED_STEP_FIELDS_V2 - set(record.keys()))
    if missing:
        raise ValueError(f"StepRecordV2 missing required fields: {missing}")
    if record.get("schema_version") != SCHEMA_VERSION_V2:
        raise ValueError(
            f"Unexpected schema_version={record.get('schema_version')} (expected {SCHEMA_VERSION_V2})"
        )
