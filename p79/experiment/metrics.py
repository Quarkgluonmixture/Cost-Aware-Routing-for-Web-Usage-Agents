from __future__ import annotations

import statistics
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


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


def compute_energy_step(metrics_cfg: Dict[str, Any]) -> Dict[str, Optional[float]]:
    energy_cfg = metrics_cfg.get("energy", {})
    if not energy_cfg.get("enabled", False):
        return {"kwh": None, "co2e_kg": None}

    kwh = energy_cfg.get("kwh_per_step")
    co2_per_kwh = energy_cfg.get("co2e_kg_per_kwh")
    if kwh is None:
        return {"kwh": None, "co2e_kg": None}
    co2 = None if co2_per_kwh is None else float(kwh) * float(co2_per_kwh)
    return {"kwh": float(kwh), "co2e_kg": co2}


def detect_benchmark_noise(error_message: Optional[str]) -> Tuple[bool, Optional[str]]:
    if not error_message:
        return False, None

    msg = error_message.lower()
    # Proxy API transient errors (503/502/timeout). 403 quota exhaustion
    # is caught earlier by runner as fatal (re-raised), never reaches here.
    if any(k in msg for k in ("model-api", "execute-api")):
        return True, "api_infra"
    if any(k in msg for k in ("captcha", "anti-bot", "blocked", "forbidden", "access denied")):
        return True, "anti_bot_or_blocked"
    if any(k in msg for k in ("geo-restricted", "not available in your region", "location")):
        return True, "geo_restricted"
    if any(k in msg for k in ("timeout", "timed out", "deadline exceeded")):
        return True, "timeout"
    if any(k in msg for k in (
        "playwright", "browser has been closed", "target closed",
        "page closed", "context closed", "frame was detached",
    )):
        return True, "playwright_error"
    if any(k in msg for k in (
        "econnrefused", "econnreset", "epipe", "connection reset",
        "connection refused", "network error", "fetch failed",
    )):
        return True, "connection_error"
    if any(k in msg for k in ("docker", "container", "service unavailable", "502", "503")):
        return True, "docker_service_error"
    if "start_url_content_error" in msg:
        return True, "start_url_content_error"
    if "site_infra_error" in msg:
        return True, "site_infra_error"
    if any(k in msg for k in ("navigation failed", "net::err_")):
        return True, "navigation_error"
    return False, None


def p95(values: List[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
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
    latency_baseline_ms: float, latency_routed_ms: float, router_overhead_ms: float
) -> float:
    return _net_saving(latency_baseline_ms, latency_routed_ms, router_overhead_ms)


def net_saving_energy(
    energy_baseline_kwh: Optional[float],
    energy_routed_kwh: Optional[float],
    router_overhead_energy_kwh: Optional[float],
) -> Optional[float]:
    if energy_baseline_kwh is None or energy_routed_kwh is None:
        return None
    overhead = float(router_overhead_energy_kwh or 0.0)
    return _net_saving(energy_baseline_kwh, energy_routed_kwh, overhead)


def estimate_step_flops(
    input_text_tokens: int,
    input_image_tokens: int,
    output_tokens: int,
    model_profile: str = "qwen3vl_4b",
) -> Dict[str, float]:
    """Estimate FLOPs per step based on token counts and model architecture.

    Uses the standard 2*N*d^2 approximation for transformer layers
    (covering Q/K/V/O projections + FFN ≈ 4x multiplier per layer).
    """
    profiles = {
        "qwen3vl_4b": {
            "d_model": 2560,
            "n_layers_llm": 36,
            "vit_d_model": 1280,
            "vit_layers": 32,
        },
    }
    p = profiles[model_profile]
    d, L = p["d_model"], p["n_layers_llm"]
    vit_d, vit_L = p["vit_d_model"], p["vit_layers"]

    # ViT encoder: 2 * tokens * d^2 * layers * 4 (attention + FFN)
    vit_flops = 2.0 * input_image_tokens * (vit_d ** 2) * vit_L * 4

    # LLM prefill: 2 * total_input * d^2 * layers * 4
    total_input = input_text_tokens + input_image_tokens
    llm_prefill_flops = 2.0 * total_input * (d ** 2) * L * 4

    # LLM decode: 2 * output * d^2 * layers * 4
    llm_decode_flops = 2.0 * output_tokens * (d ** 2) * L * 4

    return {
        "vit_encoder": vit_flops,
        "llm_prefill": llm_prefill_flops,
        "llm_decode": llm_decode_flops,
        "total": vit_flops + llm_prefill_flops + llm_decode_flops,
    }


def compute_wasted_cost(step_records: List[Dict[str, Any]], success: bool) -> Dict[str, float]:
    """For failed episodes, all step cost is wasted; for successful ones, wasted is 0."""
    if success:
        return {"wasted_cost_usd": 0.0, "wasted_energy_kwh": 0.0}
    total_cost = sum(float(s.get("cost_usd", {}).get("total", 0)) for s in step_records)
    total_energy = sum(float(s.get("energy", {}).get("kwh", 0) or 0) for s in step_records)
    return {"wasted_cost_usd": total_cost, "wasted_energy_kwh": total_energy}


def compute_waste_breakdown(step_records: List[Dict[str, Any]], success: bool) -> Dict[str, float]:
    """Break down cost into no-op, page-unchanged, and total components.

    Unlike compute_wasted_cost (binary: all-or-nothing), this provides
    fine-grained cost attribution useful for post-hoc analysis scripts.
    """
    total_cost = 0.0
    no_op_cost = 0.0
    page_unchanged_cost = 0.0
    for s in step_records:
        step_cost = float(s.get("cost_usd", {}).get("total", 0))
        total_cost += step_cost
        if s.get("action_success") is False:
            no_op_cost += step_cost
        action_type = str(s.get("action", {}).get("action_type", "") or "").lower()
        if s.get("page_changed") is False and action_type not in ("finish", "stop"):
            page_unchanged_cost += step_cost
    return {
        "total_cost_usd": total_cost,
        "no_op_cost_usd": no_op_cost,
        "page_unchanged_cost_usd": page_unchanged_cost,
        "wasted_cost_usd": total_cost if not success else 0.0,
    }


def compute_component_breakdown(step_records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate cost by component type across all steps."""
    model_cost = sum(float(s.get("cost_usd", {}).get("model", 0)) for s in step_records)
    router_cost = sum(float(s.get("cost_usd", {}).get("router_overhead", 0)) for s in step_records)
    energy_kwh = sum(float(s.get("energy", {}).get("kwh", 0) or 0) for s in step_records)
    return {"model_cost_usd": model_cost, "router_overhead_usd": router_cost, "total_energy_kwh": energy_kwh}


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


def aggregate_condition_metrics(episode_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not episode_summaries:
        return {
            "episodes": 0,
            "success_rate": 0.0,
            "avg_steps": 0.0,
            "p95_step_latency_ms": 0.0,
            "avg_total_model_cost_usd": 0.0,
            "avg_total_cost_usd": 0.0,
            "avg_router_overhead_cost_usd": 0.0,
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
        }

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

    return {
        "episodes": len(episode_summaries),
        "success_rate": success_rate,
        "avg_steps": _avg("steps"),
        # NOTE: This is P95 of per-episode P95s (approximate; not true global P95)
        "p95_step_latency_ms": p95(step_latencies),
        "avg_total_model_cost_usd": _avg("total_model_cost_usd"),
        "avg_total_cost_usd": _avg("total_cost_usd"),
        "avg_router_overhead_cost_usd": _avg("total_router_overhead_cost_usd"),
        "avg_obs_prepare_cost_usd": _avg("total_obs_prepare_cost_usd"),
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
        "wasted_energy_kwh": compute_wasted_energy(episode_summaries),
        "avg_wasted_cost_usd": float(statistics.mean(
            [float(x.get("wasted_cost_usd", 0.0)) for x in episode_summaries]
        )),
        "avg_wasted_energy_kwh": float(statistics.mean(
            [float(x.get("wasted_energy_kwh", 0.0)) for x in episode_summaries]
        )),
        # Fraction of total cost spent on successful episodes
        "cost_efficiency_ratio": (
            sum(float(x.get("total_cost_usd", 0.0)) for x in episode_summaries if x.get("success"))
            / max(sum(float(x.get("total_cost_usd", 0.0)) for x in episode_summaries), 1e-12)
        ),
    }
