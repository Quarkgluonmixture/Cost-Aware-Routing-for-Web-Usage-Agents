from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


# B-782 (/stress A1.9 cold-start P0-1-AB* Claude+codex OOB, 2026-05-17):
# numeric hero fields that flow into paper §1 / §3 / §4 aggregates. Strict
# entry guard rejects bool/string/non-finite drift via _assert_strict_aggregator_types.
# Pre-fix B-322 only covered outcome bools (success / benchmark_noise / score);
# numeric hero fields were unguarded → `{"steps": True, "total_cost_usd": "1e309"}`
# silently passed `_avg(key)` (line 401-402 `float(x.get(key, 0.0))`) producing
# `avg_total_cost_usd=inf, cost_efficiency_ratio=nan`. rederive_episode_summary.py
# bypasses `load_episode_summary_strict` → no upstream defense; aggregator must
# enforce.
_HERO_NUMERIC_FIELDS = frozenset({
    "steps", "retries",
    "total_cost_usd", "total_model_cost_usd", "total_router_overhead_cost_usd",
    "total_obs_prepare_cost_usd", "total_input_cost_usd", "total_output_cost_usd",
    "total_latency_ms", "p95_step_latency_ms",
    "total_energy_kwh", "total_co2e_kg",
    "no_op_rate", "page_unchanged_rate",
    "wasted_cost_usd", "wasted_energy_kwh",
    "escalation_count",
    "busy_wait_total_ms",
    "checklist_completion_rate",
})


def compute_token_cost(
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    cost_cfg: Dict[str, Any],
) -> Dict[str, float]:
    input_t = float(input_tokens or 0)
    output_t = float(output_tokens or 0)
    in_rate = float(cost_cfg.get("input_cost_per_1k", 0.0))
    out_rate = float(cost_cfg.get("output_cost_per_1k", 0.0))

    input_cost = input_t / 1000.0 * in_rate
    output_cost = output_t / 1000.0 * out_rate
    total_cost = input_cost + output_cost
    return {
        "input": input_cost,
        "output": output_cost,
        "total": total_cost,
    }


def select_token_cost_cfg(metrics_cfg: Dict[str, Any], backend_type: Optional[str]) -> Dict[str, Any]:
    """Select token pricing config by backend type.

    - API backends (type starts with "api_") use metrics.cost_api when available.
    - Other backends use metrics.cost.
    """
    backend_type_norm = str(backend_type or "").lower().strip()
    prefer_api = backend_type_norm.startswith("api_")
    if prefer_api and isinstance(metrics_cfg.get("cost_api"), dict):
        return metrics_cfg.get("cost_api", {})
    return metrics_cfg.get("cost", {})


def compute_router_overhead_cost(router_overhead_ms: float, router_cfg: Dict[str, Any]) -> float:
    rate = float(router_cfg.get("overhead_cost_per_ms", 0.0))
    return float(router_overhead_ms) * rate


# B-187 (/stress A1.4b-ii gemini v5 #1 + codex B-ii-5, P2): deleted
# `compute_energy_step()` — 0 production callers. Real energy/CO2 pipeline
# is `p79/experiment/energy_tracker.py::LightweightEnergyTracker.estimate_step`
# (used by `runner/main.py:1472`). The deleted helper read the wrong YAML
# key (`co2e_kg_per_kwh: null`) instead of `carbon_intensity_g_per_kwh: 220`
# that EnergyTracker actually uses, which would have produced silent
# co2e_kg=None if any caller existed. Live production data verified
# CO2 telemetry is correct (UK grid 220 g/kWh = 0.22 kg/kWh ratio
# empirically across episode summaries).


def detect_benchmark_noise(error_message: Optional[str]) -> Tuple[bool, Optional[str]]:
    if not error_message:
        return False, None

    msg = error_message.lower()
    # B-199 (/stress A1.4b-ii codex B-ii-7, P2): added `api_rate_limit` and
    # `auth_expired_or_session_invalid` categories. Pre-fix these errors fell
    # through to "False, None" → counted as real agent failure in paper §3.4
    # noise denominator (forward-risk; current scan shows 0 production
    # contamination but Phase 1a may produce these).
    #
    # Order matters — more-specific patterns first so they don't get
    # shadowed by generic `navigation_error` etc.
    if any(k in msg for k in ("429", "rate limit", "too many requests")):
        return True, "api_rate_limit"
    if any(k in msg for k in ("auth expired", "login expired", "session expired",
                              "401 unauthorized", "403 forbidden auth")):
        return True, "auth_expired_or_session_invalid"
    # Proxy API transient errors (503/502/timeout). 403 quota exhaustion
    # is caught earlier by runner as fatal (re-raised), never reaches here.
    if any(k in msg for k in ("model-api", "execute-api")):
        return True, "api_infra"
    if any(k in msg for k in ("captcha", "anti-bot", "blocked", "forbidden", "access denied")):
        return True, "anti_bot_or_blocked"
    # B-786 (/stress A1.9 cold-start P1-1-AB* Claude+codex OOB, 2026-05-17):
    # removed bare `"location"` from geo-restriction keyword set. Pre-fix
    # any error containing `"location"` substring (e.g. `"element location
    # not found"`, `"locator resolved to hidden element location"`,
    # `"window.location is not defined"`) was classified as `geo_restricted`
    # → paper §3.4 noise taxonomy mis-categorized agent runtime errors as
    # deployment / site issues; clean_success_rate denominator wrongly
    # excluded real agent failures. Anchored phrases only: "not available
    # in your region", "geo-restricted", "location restriction" (semantic
    # geo restriction phrase, not bare location substring).
    if any(k in msg for k in (
        "geo-restricted",
        "not available in your region",
        "location restriction",
        "vpn detected",
    )):
        return True, "geo_restricted"
    if any(k in msg for k in ("timeout", "timed out", "deadline exceeded")):
        return True, "timeout"
    if any(k in msg for k in (
        "playwright", "browser has been closed", "target closed",
        "page closed", "context closed", "frame was detached",
    )):
        return True, "playwright_error"
    # B-199: `ERR_CONNECTION_REFUSED` is now matched here BEFORE the generic
    # navigation pattern so connection-class errors classify as `connection_error`
    # rather than `navigation_error` (codex B-ii-7 order-overlap concern).
    if any(k in msg for k in (
        "econnrefused", "econnreset", "epipe", "connection reset",
        "connection refused", "network error", "fetch failed",
        "err_connection_refused",
    )):
        return True, "connection_error"
    # B-335 (/stress A1.9 Mode A F5, 2026-05-16): split bare "502"/"503"
    # (short messages without container/proxy URL context) into
    # `unclassified_5xx`. Pre-fix bare HTTP 503 was uniformly bucketed to
    # docker_service_error even when error was from B0 proxy API short
    # response (no AWS gateway URL in error string) → paper §3.4 noise
    # breakdown mis-categorized API transient as docker container issue.
    # Specific container signatures still classify as docker_service_error;
    # "service unavailable" is a generic HTTP 503 phrase (not docker-
    # specific), so it now falls through to unclassified_5xx.
    if any(k in msg for k in ("docker", "container")):
        return True, "docker_service_error"
    if any(k in msg for k in ("502", "503", "service unavailable")):
        return True, "unclassified_5xx"
    if "start_url_content_error" in msg:
        return True, "start_url_content_error"
    if "site_infra_error" in msg:
        return True, "site_infra_error"
    if any(k in msg for k in ("navigation failed", "net::err_")):
        return True, "navigation_error"
    return False, None


def p95(values: List[float], *, strict: bool = False) -> float:
    """Linear-interp P95 (matches numpy default `method="linear"`).

    B-200 (/stress A1.4b-ii codex B-ii-6, P2): strict policy on None / NaN.
    Pre-fix `p95([None, None, 0, 0])` raised `TypeError` deep in `sorted`;
    `p95([NaN, 1, 2])` silently returned 1.9 (NaN ignored by sort but tail
    misleading). Now: filter None + NaN explicitly + return 0.0 on empty
    valid set (matches the existing empty-input contract). Callers that
    want strict mode should pass ``strict=True`` (see B-456).

    B-456 (/stress A1.4 P1-8-C gemini OOB, 2026-05-17): opt-in ``strict``
    mode for figure renderers / cross-arm aggregators that compute
    ``mean(p95)``. Pre-fix the catastrophic-empty case silently returned
    ``0.0`` — downstream ``mean`` then dragged the fleet average toward 0,
    falsely advantaging the most-failing arm. Renderers and per-arm tables
    should set ``strict=True`` so an empty-input p95 raises a ``ValueError``
    they can catch + display "N/A" instead of injecting a 0 into the math.
    Default ``strict=False`` keeps the legacy empty-input contract for the
    many callers that already disclose "p95=0.0 means catastrophic" inline.
    """
    import math as _math
    valid = [float(v) for v in values
             if v is not None and not (isinstance(v, float) and _math.isnan(v))]
    if not valid:
        if strict:
            raise ValueError(
                "p95 called with empty valid input set (no non-None / non-NaN "
                "values); set strict=False to receive the legacy 0.0 fallback "
                "or filter the empty case upstream before calling. B-456."
            )
        return 0.0
    if len(valid) == 1:
        return valid[0]
    ordered = sorted(valid)
    k = 0.95 * (len(ordered) - 1)
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    return float(ordered[f] + (k - f) * (ordered[c] - ordered[f]))


def _net_saving(baseline: float, routed: float, overhead: float) -> float:
    return float(baseline) - (float(routed) + float(overhead))


def net_saving(cost_baseline_total: float, cost_routed_model: float, cost_router_overhead: float) -> float:
    """
    Net saving for routed condition.

    Baseline is compared against routed total cost reconstructed from:
    routed_total = routed_model + routed_router_overhead
    """
    return _net_saving(cost_baseline_total, cost_routed_model, cost_router_overhead)


def net_saving_latency(
    latency_baseline_ms: float,
    latency_routed_ms: float,
    router_overhead_ms: float = 0.0,
) -> float:
    """Net latency saving between baseline and routed conditions.

    `latency_routed_ms` MUST be the *end-to-end* routed latency that already
    includes the router decision overhead (which is recorded inside each step's
    `latency_ms.total`). Therefore we DO NOT add `router_overhead_ms` again —
    it would double-count.

    `router_overhead_ms` is kept as a parameter for symmetry with the other
    net_saving_* functions and for diagnostic reporting (callers can pass 0).

    Semantics: saving = baseline - routed. Positive = routing is faster.
    """
    return float(latency_baseline_ms) - float(latency_routed_ms)


def net_saving_energy(
    energy_baseline_kwh: Optional[float],
    energy_routed_kwh: Optional[float],
    router_overhead_energy_kwh: Optional[float],
) -> Optional[float]:
    if energy_baseline_kwh is None or energy_routed_kwh is None:
        return None
    overhead = float(router_overhead_energy_kwh or 0.0)
    return _net_saving(energy_baseline_kwh, energy_routed_kwh, overhead)


# B-328 (/stress A1.9 Mode A F4 + Mode B F8 OOB, 2026-05-16): deleted
# `estimate_step_flops()` — 0 production callers (`grep -r estimate_step_flops
# p79 scripts tests | wc -l = 1` = self-definition only) AND formula was
# ~3× under-estimate of standard transformer FLOPs (`2*N*d²*L*4` =
# 8 N d² per layer vs Hoffmann standard 24 N d² for QKVO attention +
# FFN; SwiGLU pushes to ~28 N d²/layer). Paper §3 does not quote
# FLOPs/step numerically; if future work needs FLOPs, implement
# per-architecture formulas with attention(8 N d²) + FFN(depends on
# activation: ReLU/GeLU 16 N d², SwiGLU 24 N d²) split + per-model
# d_model / n_layers / FFN_mult from architecture config. Until then
# this dead-and-wrong helper is removed to prevent future citation
# of a broken formula.


def compute_wasted_cost(
    step_records: List[Dict[str, Any]],
    success: bool,
) -> Dict[str, float]:
    """For failed episodes, all step cost is wasted; for successful ones, wasted is 0.

    B-188 (/stress A1.4b-ii Claude D5, P2): removed legacy `adjusted_success`
    keyword arg. §139.8 retired the post-hoc na_fp / eval_fp filter layer
    (B-91 evaluator empty-pred guard at source + N/A task exclusion at load).
    `success` is now the canonical paper-grade outcome and the only argument
    this function needs. Empirical: 0 production callers passed
    `adjusted_success=`.
    """
    if success:
        return {"wasted_cost_usd": 0.0, "wasted_energy_kwh": 0.0}
    # Defensive: cost_usd / energy may be explicitly None on partial rows.
    total_cost = sum(float((s.get("cost_usd") or {}).get("total", 0)) for s in step_records)
    total_energy = sum(float((s.get("energy") or {}).get("kwh", 0) or 0) for s in step_records)
    return {"wasted_cost_usd": total_cost, "wasted_energy_kwh": total_energy}


# B-188 (/stress A1.4b-ii gemini v5 #4 + codex B-ii-8, P2): deleted
# `compute_waste_breakdown()` — 0 production callers (paper §3 fine-grained
# waste analysis lives in `scripts/analysis/analyze_reason_diagnostics.py:
# 764-794`, which has a richer impl with `loop_cost_usd` + `effective_cost_usd`
# not present here). The deleted helper had a math invariant violation
# (success episode → wasted_cost_usd=0 but no_op_cost / page_unchanged_cost
# unconditionally sum across all steps, so parts could exceed total). Per
# user decision 2026-05-16 "选 A": keep the analyzer impl as the
# fine-grained source of truth; binary `compute_wasted_cost` above is the
# canonical aggregator-level wasted metric.



def compute_component_breakdown(step_records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate cost by component type across all steps.

    B-576 (/stress A1.22 P2-19-B codex, 2026-05-17): include
    `obs_prepare_usd` in the breakdown so the parts close to `total`. Pre-fix
    `EpisodeSummaryV2.total_cost_usd` was constructed via runner as `model +
    router_overhead + obs_prepare` (`runner/main.py:2308-2315`) but
    `component_breakdown` only summed `model + router_overhead + energy`,
    omitting `obs_prepare`. Appendix component plots therefore systematically
    under-counted total cost source; cross-baseline B0/B1/B2 comparison did
    not close. Now the 3 cost components ({model, router_overhead, obs_
    prepare}) sum to the runner-emitted total; the test fixture in
    `tests/test_router_and_metrics.py` should be extended to assert this
    closure when next touched.
    """
    # Defensive: cost_usd / energy may be explicitly None on partial/error rows.
    model_cost = sum(float((s.get("cost_usd") or {}).get("model", 0)) for s in step_records)
    router_cost = sum(float((s.get("cost_usd") or {}).get("router_overhead", 0)) for s in step_records)
    obs_prepare_cost = sum(float((s.get("cost_usd") or {}).get("obs_prepare", 0)) for s in step_records)
    energy_kwh = sum(float((s.get("energy") or {}).get("kwh", 0) or 0) for s in step_records)
    return {
        "model_cost_usd": model_cost,
        "router_overhead_usd": router_cost,
        # B-576 (/stress A1.22 P2-19-B codex, 2026-05-17): obs_prepare cost
        # closure — runner's `cost_usd.total = model + router_overhead +
        # obs_prepare`, so the breakdown must include all 3 parts to be
        # additively consistent with the summary's `total_cost_usd`.
        "obs_prepare_usd": obs_prepare_cost,
        "total_energy_kwh": energy_kwh,
    }


def compute_wasted_energy(episode_summaries: List[Dict[str, Any]]) -> Optional[float]:
    """Total kWh spent on unsuccessful episodes (failed or hit max_steps).

    Returns None if no energy data is present in any failed episode.
    """
    vals = [
        float(x["total_energy_kwh"])
        for x in episode_summaries
        if not x.get("success") and x.get("total_energy_kwh") is not None
    ]
    return float(sum(vals)) if vals else None


def _compute_cost_efficiency_ratio(episode_summaries: List[Dict[str, Any]]) -> Optional[float]:
    """B-197: return None when no cost data, else `cost_on_success / total_cost`."""
    if not episode_summaries:
        return None
    total_cost = sum(float(x.get("total_cost_usd", 0.0)) for x in episode_summaries)
    if total_cost < 1e-9:
        # All-zero cost (B1 local, no API spend) → ratio undefined
        return None
    cost_on_success = sum(
        float(x.get("total_cost_usd", 0.0))
        for x in episode_summaries if x.get("success")
    )
    return cost_on_success / total_cost


def _assert_strict_aggregator_types(
    episode_summaries: List[Dict[str, Any]],
    *,
    allow_quarantined: bool = False,
) -> None:
    """B-322 (/stress A1.9 Mode A F3 + Mode B F5 OOB, 2026-05-16): aggregator
    entry strict-type-check on hero fields. A1.8 B-283 fixed string-truthy at
    `load_episode_summary_strict()`, but `aggregate_condition_metrics` was
    called from 3 sites (runner/main.py:636, rederive_episode_summary.py:280,
    analysis.py:200) passing raw dicts bypassing the strict loader.
    Defense-in-depth: enforce bool/numeric types at aggregator entry so any
    future schema regression (`"success": "false"` literal string,
    `"benchmark_noise": "True"`, `"score": "0.0"`) raises here rather than
    silently inflating paper §1 hero SR via Python's `bool('false') = True`.

    B-782 (/stress A1.9 cold-start P0-1-AB* Claude+codex OOB, 2026-05-17):
    extended to numeric hero fields. Pre-fix only outcome bools + score were
    type-checked; `_HERO_NUMERIC_FIELDS` (steps / total_cost_usd / latency /
    energy etc.) were unguarded → bool-as-int + string-coercion + inf/nan
    poisoned aggregates. Now per-field: `isinstance(v, (int, float)) and not
    isinstance(v, bool) and math.isfinite(v)`. `bool` excluded explicitly
    (bool is int subclass in Python).

    B-784 (/stress A1.9 cold-start P0-3-B* codex OOB, 2026-05-17): reject
    quarantined episodes (`needs_reevaluation=True`) by default. Pre-fix
    `aggregate_condition_metrics` accepted quarantined episodes silently;
    runner live aggregate + rederive bypass `load_episode_summary_strict`
    (which has `reject_needs_reevaluation=True` defense) → quarantined
    episodes counted as `success=False` in live `condition_summary_v2.json`,
    diverging from canonical `analysis.py` denominator post-fact. Forensic
    appendix may opt-in via `allow_quarantined=True`.
    """
    for idx, ep in enumerate(episode_summaries):
        # B-784: quarantine rejection (default-on; forensic appendix opt-in).
        nrv = ep.get("needs_reevaluation")
        if not allow_quarantined and nrv is True:
            raise ValueError(
                f"aggregate_condition_metrics episode[{idx}]: "
                f"needs_reevaluation=True (B-486 quarantine flag). "
                "Live aggregator / rederive must NOT count quarantined "
                "episodes as paper-grade outcomes — denominator would "
                "diverge from canonical analysis.py. Pass "
                "allow_quarantined=True only for forensic appendix paths."
            )
        if nrv is not None and not isinstance(nrv, bool):
            raise ValueError(
                f"aggregate_condition_metrics episode[{idx}]: "
                f"needs_reevaluation type mismatch — got "
                f"{type(nrv).__name__!s} (value={nrv!r}), expected bool/None."
            )

        if "success" in ep and not isinstance(ep["success"], bool):
            raise ValueError(
                f"aggregate_condition_metrics episode[{idx}]: success type "
                f"mismatch — got {type(ep['success']).__name__!s} "
                f"(value={ep['success']!r}), expected bool. JSON literal "
                "string-truthy attack vector → paper §1 hero SR inflation."
            )
        if "benchmark_noise" in ep and not isinstance(ep["benchmark_noise"], bool):
            raise ValueError(
                f"aggregate_condition_metrics episode[{idx}]: benchmark_noise "
                f"type mismatch — got {type(ep['benchmark_noise']).__name__!s} "
                f"(value={ep['benchmark_noise']!r}), expected bool."
            )
        score = ep.get("score")
        if score is not None:
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError(
                    f"aggregate_condition_metrics episode[{idx}]: score type "
                    f"mismatch — got {type(score).__name__!s} (value={score!r}), "
                    "expected int/float (bool excluded — bool is int subclass)."
                )
            if not math.isfinite(float(score)):
                raise ValueError(
                    f"aggregate_condition_metrics episode[{idx}]: score "
                    f"non-finite (value={score!r}); inf/nan would poison "
                    "downstream `success_rate / cost_efficiency_ratio`."
                )

        # B-782: numeric hero field strict check.
        for field in _HERO_NUMERIC_FIELDS:
            if field not in ep:
                continue
            v = ep[field]
            if v is None:
                # Tri-state field (e.g. total_energy_kwh None on energy-disabled
                # path) allowed; downstream aggregator skips Nones explicitly.
                continue
            if isinstance(v, bool):
                raise ValueError(
                    f"aggregate_condition_metrics episode[{idx}]: {field} "
                    f"type mismatch — got bool (value={v!r}), expected "
                    "int/float. bool-as-int is Python's int subclass "
                    "(`isinstance(True, int) == True`) → JSON literal "
                    "`true/false` would coerce silently to 1/0 and pollute "
                    "paper §1 / §3 aggregates."
                )
            if not isinstance(v, (int, float)):
                raise ValueError(
                    f"aggregate_condition_metrics episode[{idx}]: {field} "
                    f"type mismatch — got {type(v).__name__!s} (value={v!r}), "
                    "expected int/float."
                )
            if not math.isfinite(float(v)):
                raise ValueError(
                    f"aggregate_condition_metrics episode[{idx}]: {field} "
                    f"non-finite (value={v!r}). inf/nan would poison "
                    f"`avg_{field}` + `cost_efficiency_ratio` downstream."
                )


def aggregate_condition_metrics(
    episode_summaries: List[Dict[str, Any]],
    *,
    allow_quarantined: bool = False,
) -> Dict[str, Any]:
    """B-784 (/stress A1.9 cold-start P0-3-B*, 2026-05-17): `allow_quarantined`
    parameter (default False) routes quarantined episodes (`needs_reevaluation=True`)
    to a hard-fail at entry guard. Runner live aggregate (`runner/main.py:924`)
    + rederive (`scripts/maintenance/rederive_episode_summary.py:280`) +
    canonical analysis (`p79/experiment/analysis.py:278`) all pass default
    False; only forensic appendix paths should opt in.
    """
    if not episode_summaries:
        return {
            "episodes": 0,
            "success_rate": 0.0,
            "avg_steps": 0.0,
            "p95_step_latency_ms": 0.0,
            "avg_total_latency_ms": 0.0,
            # B-1410 (/stress A2.7 P1-5-AB*, 2026-05-18): canonical
            # cross-baseline latency = retry-adjusted (§3.5.1 B-1402 framework).
            # Empty-episode fallback = 0.0; populated runs that lack the
            # episode-summary field fall back to `avg_total_latency_ms` at
            # the rollup branch below (legacy data path).
            "avg_total_latency_minus_retry_ms": 0.0,
            "avg_total_model_cost_usd": 0.0,
            "avg_total_cost_usd": 0.0,
            "avg_router_overhead_cost_usd": 0.0,
            "avg_router_overhead_ms": 0.0,
            "avg_obs_prepare_cost_usd": 0.0,
            "avg_input_cost_usd": 0.0,
            "avg_output_cost_usd": 0.0,
            "avg_total_energy_kwh": None,
            "avg_total_co2e_kg": None,
            "avg_retries": 0.0,
            "avg_no_op_rate": 0.0,
            "avg_page_unchanged_rate": 0.0,
            "avg_escalation_count": 0.0,
            "trigger_distribution": {},
            "state_change_reason_distribution": {},
            "avg_checklist_completion_rate": None,
            "checklist_failure_episode_rate": None,
            "benchmark_noise_rate": 0.0,
            "wasted_energy_kwh": None,
            "avg_wasted_cost_usd": 0.0,
            "avg_wasted_energy_kwh": 0.0,
            "cost_efficiency_ratio": None,
            "avg_busy_wait_total_ms": 0.0,
            "energy_partial_episode_count": 0,
            "energy_partial_episode_rate": 0.0,
            # B-193 transparency telemetry defaults (empty-episode case):
            "trajectory_incomplete_episode_count": 0,
            "trajectory_incomplete_rate": 0.0,
            "partial_recovery_episode_count": 0,
            "partial_recovery_rate": 0.0,
            "unknown_failure_reason_distribution": {},
            # B-199 noise category distribution default:
            "benchmark_noise_category_distribution": {},
            # B-327 (/stress A1.9 Mode C F3 OOB, 2026-05-16): clean SR
            # excluding benchmark_noise episodes from numerator+denominator.
            "clean_success_rate": None,
            "clean_episode_count": 0,
        }

    # B-322 (/stress A1.9): defense-in-depth strict-type-check on entry.
    # B-784 (cold-start): thread `allow_quarantined` to entry guard.
    _assert_strict_aggregator_types(episode_summaries, allow_quarantined=allow_quarantined)

    success_rate = sum(1 for x in episode_summaries if x.get("success")) / len(episode_summaries)
    step_latencies = [float(x.get("p95_step_latency_ms", 0.0)) for x in episode_summaries]

    def _avg(key: str) -> float:
        return float(statistics.mean([float(x.get(key, 0.0)) for x in episode_summaries]))

    energy_vals = [x.get("total_energy_kwh") for x in episode_summaries if x.get("total_energy_kwh") is not None]
    co2_vals = [x.get("total_co2e_kg") for x in episode_summaries if x.get("total_co2e_kg") is not None]
    checklist_completion_vals = [
        float(x.get("checklist_completion_rate"))
        for x in episode_summaries
        if x.get("checklist_completion_rate") is not None
    ]
    checklist_failed_flags = [
        1 if int(x.get("checklist_failed_items", 0) or 0) > 0 else 0
        for x in episode_summaries
        if x.get("checklist_failed_items") is not None
    ]
    trigger_counter: Counter = Counter()
    reason_counter: Counter = Counter()
    benchmark_noise_flags: List[int] = []
    for ep in episode_summaries:
        trigger_dist = ep.get("trigger_distribution", {}) or {}
        if isinstance(trigger_dist, dict):
            for k, v in trigger_dist.items():
                try:
                    trigger_counter[str(k)] += int(v)
                except Exception:
                    continue
        dist = ep.get("state_change_reason_distribution", {}) or {}
        if isinstance(dist, dict):
            for k, v in dist.items():
                try:
                    reason_counter[str(k)] += int(v)
                except Exception:
                    continue
        benchmark_noise_flags.append(1 if bool(ep.get("benchmark_noise", False)) else 0)

    # Energy completeness aggregation (RU-5 / §97 audit).
    energy_partial_count = sum(
        1 for x in episode_summaries if bool(x.get("energy_partial", False))
    )

    # B-788 (/stress A1.9 cold-start P1-4-B* codex OOB, 2026-05-17):
    # step-level `energy_window_partial=True` (B-321 strict-window flag)
    # was stamped on individual step records but NEVER propagated to
    # episode/condition telemetry. EnergyTracker emits
    # `window_sample_count < 2` flag at step boundary; runner's per-episode
    # `energy_partial` only checks `kwh is None`, so `sample_count=1`
    # pynvml estimates passed silently → paper §3 energy comparison mixed
    # high-quality and low-density samples without operator visibility.
    # Now: episode summary is expected to surface
    # `energy_window_partial_step_count` and `min_window_sample_count` (see
    # runner B-788 partner change). Condition aggregator emits rate +
    # quantile for paper §3 disclosure.
    energy_window_partial_step_counts = [
        int(x.get("energy_window_partial_step_count", 0) or 0)
        for x in episode_summaries
    ]
    min_window_sample_counts = [
        int(x.get("min_window_sample_count", 0) or 0)
        for x in episode_summaries
        if x.get("min_window_sample_count") is not None
    ]
    energy_window_partial_episode_count = sum(
        1 for n in energy_window_partial_step_counts if n > 0
    )

    # B-193 (/stress A1.4b-ii codex B-ii-2, P1 OOB): aggregate the 3 paper §3.5
    # transparency telemetry fields stamped by runner per A1.4a B-166/B-167/B-168.
    # Pre-fix: these fields were written to per-episode summary JSON but never
    # consumed by aggregator → paper §3.5 `trajectory_incomplete_rate per cell`
    # claim was structurally unproducible. Now: per-cell counts + rates emitted
    # so paper §3.5 can cite them directly.
    trajectory_incomplete_count = sum(
        1 for x in episode_summaries if bool(x.get("trajectory_incomplete", False))
    )
    partial_recovery_episode_count = sum(
        1 for x in episode_summaries
        if int(x.get("partial_recovery_step_count", 0) or 0) > 0
    )
    unknown_failure_aggregated: Counter = Counter()
    for ep in episode_summaries:
        ufr = ep.get("unknown_failure_reasons", {}) or {}
        if isinstance(ufr, dict):
            for k, v in ufr.items():
                try:
                    unknown_failure_aggregated[str(k)] += int(v)
                except Exception:
                    continue
    return {
        "episodes": len(episode_summaries),
        "success_rate": success_rate,
        "avg_steps": _avg("steps"),
        # NOTE: This is P95 of per-episode P95s (approximate; not true global
        # step-latency P95). Per-episode end-to-end latency is reported
        # separately as avg_total_latency_ms — that is what should be used for
        # net_saving_latency comparisons (single-step P95 mixes step granularities).
        "p95_step_latency_ms": p95(step_latencies),
        "avg_total_latency_ms": _avg("total_latency_ms"),
        # B-1410 (/stress A2.7 P1-5-AB* 2-AI overlap Claude F7 + codex F2,
        # 2026-05-18 + user 3-axis canonical-estimand directive): canonical
        # cross-baseline latency axis is retry-adjusted (§3.5.1 B-1402 disclosure).
        # `total_latency_minus_retry_ms` may be None on legacy episode summaries
        # (pre-A2.7 runner write path); `_avg` falls back to skipping None values
        # in its existing missing-field logic. When all summaries lack the field,
        # the aggregated value will be 0.0 from the empty-input branch; downstream
        # consumers should check whether per-episode field is populated before
        # trusting the cross-baseline canonical-latency comparison.
        "avg_total_latency_minus_retry_ms": _avg("total_latency_minus_retry_ms"),
        "avg_total_model_cost_usd": _avg("total_model_cost_usd"),
        "avg_total_cost_usd": _avg("total_cost_usd"),
        "avg_router_overhead_cost_usd": _avg("total_router_overhead_cost_usd"),
        # End-to-end router overhead per episode (ms). Reported for diagnostics
        # only — net_saving_latency does NOT subtract this (already in routed total).
        "avg_router_overhead_ms": _avg("total_router_overhead_ms"),
        "avg_obs_prepare_cost_usd": _avg("total_obs_prepare_cost_usd"),
        # B-332 (/stress A1.9 Mode C F6 OOB, 2026-05-16): paper §3.2 quotes
        # "~30ms median obs-prepare latency" but pre-fix aggregator emitted
        # only USD aggregate (`avg_total_obs_prepare_cost_usd`), no latency
        # quantile → paper §3.2 number was structurally not producible from
        # this pipeline alone (required manual step_metrics.csv pivot).
        # Now: aggregate p50/p95 across each episode's
        # `obs_prepare_latency_ms_list` if present (runner emits per-step
        # obs_prepare latency in step record; episode summary may pre-aggregate).
        "p50_obs_prepare_ms": (
            float(statistics.median(
                [float(v) for x in episode_summaries
                 for v in (x.get("obs_prepare_latency_ms_list", []) or [])
                 if v is not None]
            )) if any(
                x.get("obs_prepare_latency_ms_list")
                for x in episode_summaries
            ) else None
        ),
        "p95_obs_prepare_ms": (
            p95([
                float(v) for x in episode_summaries
                for v in (x.get("obs_prepare_latency_ms_list", []) or [])
                if v is not None
            ]) if any(
                x.get("obs_prepare_latency_ms_list")
                for x in episode_summaries
            ) else None
        ),
        # B-195 (/stress A1.4b-ii gemini v1 G1, P1 OOB): per-cell avg of the
        # per-episode obs-prepare cost. Paper §3 currently cites
        # "~30 ms median obs-prepare latency" but pre-fix the aggregate layer
        # exposed only the SUM (`avg_total_obs_prepare_cost_usd`), not the
        # per-step distribution needed to verify a median latency claim. The
        # cost-USD aggregate is still emitted above; for the latency claim
        # paper §3 should now cite `avg_total_obs_prepare_cost_usd` per cell
        # + per-step latency must be sourced from `step_metrics.csv` (which
        # analysis.py already dumps from raw JSONL). Issue: B-195 closes
        # the cost-aggregate; B-195b (deferred) — emit median+p95 obs-prepare
        # ms separately if paper §3 wants that exact metric — pending
        # decision on whether step_metrics.csv pivot is sufficient.
        "avg_input_cost_usd": _avg("total_input_cost_usd"),
        "avg_output_cost_usd": _avg("total_output_cost_usd"),
        "avg_total_energy_kwh": (float(statistics.mean(energy_vals)) if energy_vals else None),
        "avg_total_co2e_kg": (float(statistics.mean(co2_vals)) if co2_vals else None),
        "avg_retries": _avg("retries"),
        "avg_no_op_rate": _avg("no_op_rate"),
        "avg_page_unchanged_rate": _avg("page_unchanged_rate"),
        "avg_escalation_count": _avg("escalation_count"),
        "trigger_distribution": dict(trigger_counter),
        "state_change_reason_distribution": dict(reason_counter),
        "avg_checklist_completion_rate": (
            float(statistics.mean(checklist_completion_vals)) if checklist_completion_vals else None
        ),
        "checklist_failure_episode_rate": (
            float(statistics.mean(checklist_failed_flags)) if checklist_failed_flags else None
        ),
        "benchmark_noise_rate": float(statistics.mean(benchmark_noise_flags)),
        # B-327 (/stress A1.9 Mode C F3 OOB, 2026-05-16) + B-600 (/stress A1.6a
        # P1-2-AC estimand decision, 2026-05-17): clean_success_rate =
        # SR over episodes excluding benchmark_noise=True (api_rate_limit /
        # auth_expired / playwright_crash / docker_service_error etc).
        #
        # ESTIMAND DECISION (Q2=(A), user directive 2026-05-17 + §139.8
        # alignment): Paper §1 hero reports **raw `success_rate`** —
        # `success` is canonical post-§139.8 (no adjusted SR), and the FP
        # framework is upstream-fixed (B-91 LLM judge `pred=""` guard +
        # `exclude_na_tasks: true` default). `clean_success_rate` is
        # **transparency appendix only** — it surfaces residual infra
        # noise (which is a different semantic from N/A FP / eval FP)
        # for paper §3 audit, NOT for §1 hero claim. Pre-Q2 the B-327
        # comment read "Paper §1 hero should report clean_SR" — that
        # framing was retracted because (a) §139.8 retired post-hoc
        # filtering as a class, and (b) watchdog auto-clean protocol
        # (`reference_watchdog_protocol.md`) cleans infra-noise episodes
        # forward via re-run, so clean vs raw SR delta should be small
        # for paper-grade fire (Phase 1a rerun = canonical).
        #
        # Returns None if all episodes are noise (denominator=0) so prose
        # can show "N/A — all episodes were infra noise" rather than
        # fake-zero.
        "clean_success_rate": (
            float(
                sum(
                    1 for x in episode_summaries
                    if x.get("success") and not bool(x.get("benchmark_noise", False))
                )
            ) / max(
                sum(
                    1 for x in episode_summaries
                    if not bool(x.get("benchmark_noise", False))
                ),
                1,
            )
            if any(not bool(x.get("benchmark_noise", False)) for x in episode_summaries)
            else None
        ),
        "clean_episode_count": sum(
            1 for x in episode_summaries
            if not bool(x.get("benchmark_noise", False))
        ),
        # B-798 (/stress A1.9 cold-start P2-5-C* gemini OOB, 2026-05-17):
        # clean SR small-n warning flag. When `clean_episode_count` is a small
        # fraction of total episodes, `clean_success_rate` becomes statistically
        # unstable (extreme: 99 noise + 1 clean success → clean_SR=100% misleading).
        # Flag emitted so paper §3.4 disclosure can warn / hide the metric.
        # Threshold = clean / total < 0.5 (more than half noise → unstable).
        "clean_n_too_low": (
            (sum(1 for x in episode_summaries if not bool(x.get("benchmark_noise", False)))
             / len(episode_summaries)) < 0.5
            if episode_summaries else False
        ),
        # B-199 (/stress A1.4b-ii gemini v1 G3): per-cell category distribution.
        # Pre-fix the 10-category breakdown produced by `detect_benchmark_noise`
        # was flattened to a single rate, losing site-specific infrastructure
        # insight (e.g., is reddit noise = captcha or rate_limit?). Paper §3.4
        # appendix can now cite this dict directly.
        "benchmark_noise_category_distribution": dict(Counter(
            str(x.get("benchmark_noise_category"))
            for x in episode_summaries
            if x.get("benchmark_noise") and x.get("benchmark_noise_category")
        )),
        # B-788 (/stress A1.9 cold-start P1-4-B*, 2026-05-17): energy window
        # density telemetry — fraction of episodes with at least one
        # `energy_window_partial` step (B-321 strict-window flag).
        "energy_window_partial_episode_count": energy_window_partial_episode_count,
        "energy_window_partial_episode_rate": (
            float(energy_window_partial_episode_count) / len(episode_summaries)
            if episode_summaries else 0.0
        ),
        "min_window_sample_count_p5": (
            float(statistics.quantiles(min_window_sample_counts, n=20)[0])
            if len(min_window_sample_counts) >= 20 else (
                float(min(min_window_sample_counts))
                if min_window_sample_counts else None
            )
        ),
        "wasted_energy_kwh": compute_wasted_energy(episode_summaries),
        "avg_wasted_cost_usd": float(statistics.mean(
            [float(x.get("wasted_cost_usd", 0.0)) for x in episode_summaries]
        )),
        "avg_wasted_energy_kwh": float(statistics.mean(
            [float(x.get("wasted_energy_kwh", 0.0)) for x in episode_summaries]
        )),
        # Fraction of total cost spent on successful episodes. §139.8: the
        # post-hoc adjusted_success layer is retired — `success` is now the
        # canonical paper-grade outcome (na_fp / eval_fp fixed at the source),
        # so this single ratio is the paper-grade ratio. No `*_adjusted`
        # counterpart is produced anymore.
        #
        # B-197 (/stress A1.4b-ii Claude D4 + gemini G4, P1): when ALL
        # episodes have cost=0 (B1 local model + no API), the previous
        # `max(..., 1e-12)` floor produced ratio=0.0 silently misleading
        # paper §3 readers as "0% cost efficiency". Now: return None when
        # no actual cost data is available, so downstream prose can show
        # "N/A — no cost data" rather than a fake-zero.
        "cost_efficiency_ratio": _compute_cost_efficiency_ratio(episode_summaries),
        # §97 audit additions:
        "avg_busy_wait_total_ms": _avg("busy_wait_total_ms"),
        "energy_partial_episode_count": energy_partial_count,
        "energy_partial_episode_rate": (
            float(energy_partial_count) / len(episode_summaries) if episode_summaries else 0.0
        ),
        # B-193 paper §3.5 transparency telemetry (A1.4a B-166/B-167/B-168 → aggregate):
        "trajectory_incomplete_episode_count": trajectory_incomplete_count,
        "trajectory_incomplete_rate": (
            float(trajectory_incomplete_count) / len(episode_summaries)
            if episode_summaries else 0.0
        ),
        "partial_recovery_episode_count": partial_recovery_episode_count,
        "partial_recovery_rate": (
            float(partial_recovery_episode_count) / len(episode_summaries)
            if episode_summaries else 0.0
        ),
        "unknown_failure_reason_distribution": dict(unknown_failure_aggregated),
    }
