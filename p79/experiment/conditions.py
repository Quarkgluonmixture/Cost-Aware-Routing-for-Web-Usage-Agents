from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from p79.experiment.types import ConditionSpec, ModuleFlags


def _module_flags_from_name(name: str) -> ModuleFlags:
    name = (name or "none").lower()
    if name in ("none", "off", "baseline"):
        return ModuleFlags()
    if name in ("m1", "m1_dom_select", "m1_dom_select_fallback"):
        return ModuleFlags(m1_dom_select_fallback=True)
    if name in ("m2", "m2_dom_input", "m2_dom_first_input_fallback"):
        return ModuleFlags(m2_dom_first_input_fallback=True)
    if name in ("m3", "m3_retry", "m3_failure_trigger_retry"):
        return ModuleFlags(m3_failure_trigger_retry=True)
    if name in ("m4", "m4_two_stage", "m4_two_stage_generation_grounding"):
        return ModuleFlags(m4_two_stage_generation_grounding=True)
    raise ValueError(f"Unknown module name: {name}")


def _load_best_condition_from_phase1(path: Path) -> Optional[Tuple[bool, str, str]]:
    """
    Returns tuple(som_on, observation_mode, condition_id) from phase1 run summary if available.
    """
    if not path.exists():
        return None

    # Accept either run root or direct run_summary.json path
    summary_path = path
    if path.is_dir():
        summary_path = path / "run_summary_v2.json"
    if not summary_path.exists():
        return None

    with open(summary_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    condition_metrics = payload.get("condition_metrics", [])
    if not condition_metrics:
        return None

    # Highest success, then lower cost, then lower latency.
    ranked = sorted(
        condition_metrics,
        key=lambda x: (
            float(x.get("success_rate", 0.0)),
            -float(x.get("avg_total_cost_usd", 0.0)),
            -float(x.get("p95_step_latency_ms", 0.0)),
        ),
        reverse=True,
    )
    top = ranked[0]
    obs_mode = str(top.get("observation_mode", "som"))
    som_on = obs_mode == "som"
    return som_on, obs_mode, str(top.get("condition_id", "phase1_best"))


def _default_backend_id(cfg: Dict[str, Any]) -> str:
    backends = cfg.get("backends", {})
    explicit = backends.get("default_backend")
    if explicit:
        return str(explicit)
    for k in backends:
        if k != "default_backend":
            return k
    return "local_4b"


def generate_conditions(cfg: Dict[str, Any]) -> List[ConditionSpec]:
    phase = str(cfg["experiment"]["phase"]).lower()
    backend_id = _default_backend_id(cfg)
    primary = cfg.get("variables", {}).get("primary", {})

    conditions: List[ConditionSpec] = []

    if phase == "phase1":
        # Flat 3-mode design: dom / som / vision.
        # "som" implies SOM_MARKS + marked image; "vision" implies raw screenshot only.
        # som_on is derived (True only when mode == "som").
        obs_values = [str(x) for x in primary.get("observation_mode", ["dom", "som", "vision"])]

        for obs_mode in obs_values:
            som_on = obs_mode == "som"
            cid = f"phase1_{obs_mode}_router_0"
            conditions.append(
                ConditionSpec(
                    condition_id=cid,
                    phase="phase1",
                    backend_id=backend_id,
                    som_on=som_on,
                    observation_mode=obs_mode,
                    router_on=False,
                    modules=ModuleFlags(),
                    label=f"Phase1 {obs_mode.upper()} mode",
                )
            )

    elif phase == "phase2":
        phase2_cfg = cfg.get("variables", {}).get("phase2", {})
        run_fixed_best = bool(phase2_cfg.get("run_fixed_best", True))
        run_routed = bool(phase2_cfg.get("run_routed", True))
        best_hint = phase2_cfg.get("best_from_phase1_run_dir")
        best = None
        if best_hint:
            best = _load_best_condition_from_phase1(Path(best_hint))

        if best is None:
            fixed_hint = phase2_cfg.get("fixed_condition", {})
            obs_mode = str(fixed_hint.get("observation_mode", primary.get("observation_mode", ["som"])[0]))
            som_on = obs_mode == "som"
            source_condition_id = "manual_phase2_fixed"
        else:
            som_on, obs_mode, source_condition_id = best

        fixed_id = "phase2_fixed_best"
        routed_id = "phase2_routed"

        if run_fixed_best:
            conditions.append(
                ConditionSpec(
                    condition_id=fixed_id,
                    phase="phase2",
                    backend_id=backend_id,
                    som_on=som_on,
                    observation_mode=obs_mode,
                    router_on=False,
                    modules=ModuleFlags(),
                    label="Phase2 fixed best representation",
                    metadata={"source_condition_id": source_condition_id},
                )
            )

        if run_routed:
            conditions.append(
                ConditionSpec(
                    condition_id=routed_id,
                    phase="phase2",
                    backend_id=backend_id,
                    som_on=som_on,
                    observation_mode=str(cfg.get("router", {}).get("cheap_default_mode", "dom")),
                    router_on=True,
                    modules=ModuleFlags(),
                    label="Phase2 routed (rule-based)",
                    metadata={
                        "source_condition_id": source_condition_id,
                        "fixed_reference": fixed_id,
                        "base_observation_mode": obs_mode,
                    },
                )
            )

        if not conditions:
            raise ValueError("phase2 has no enabled conditions; set run_fixed_best and/or run_routed to true")

    elif phase == "phase3":
        phase3_cfg = cfg.get("variables", {}).get("phase3", {})
        base = phase3_cfg.get("base_condition", {})

        base_obs = str(base.get("observation_mode", "dom"))
        base_som = base_obs == "som"
        base_router = bool(base.get("router_on", True))

        module_order = ["none", "m1", "m2", "m3", "m4"]
        for name in module_order:
            flags = _module_flags_from_name(name)
            cid = f"phase3_{name}"
            conditions.append(
                ConditionSpec(
                    condition_id=cid,
                    phase="phase3",
                    backend_id=backend_id,
                    som_on=base_som,
                    observation_mode=base_obs,
                    router_on=base_router,
                    modules=flags,
                    label=f"Phase3 module ablation: {name}",
                )
            )

    else:
        raise ValueError(f"Unsupported experiment.phase={phase}, expected one of phase1/phase2/phase3")

    baselines = cfg.get("baselines", {})
    if baselines.get("run_b0", False):
        b0_backend = baselines.get("b0_backend", "api_strong")
        conditions.append(
            ConditionSpec(
                condition_id="b0_strong_upper_bound",
                phase=phase,
                backend_id=str(b0_backend),
                observation_mode=str(baselines.get("b0_observation_mode", "som")),
                som_on=str(baselines.get("b0_observation_mode", "som")) == "som",
                router_on=False,
                modules=ModuleFlags(),
                label="B0 strong upper bound",
            )
        )

    return conditions
