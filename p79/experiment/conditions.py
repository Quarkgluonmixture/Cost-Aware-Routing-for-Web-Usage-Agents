from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from p79.experiment.types import ConditionSpec, ModuleFlags


# B-691 fix (/stress A1.7 cold-start P0-1-AC*, 2026-05-17): centralized
# observation_mode validation. Pre-fix the phase1 branch (L80-130) was the
# ONLY caller of the deprecation gate; phase2 + phase3 branches read
# obs_mode from yaml and passed straight into ConditionSpec without
# checking _DEPRECATED_OBS_MODES — so `configs/exp_v2_phase2.yaml`
# (observation_mode: "hybrid") + `configs/exp_v2_phase3.yaml`
# (observation_mode: "dom_only") would silently emit retired modes for
# any future Phase 2/3 fire. Extracting to a module-level helper means
# all 3 phase branches share one canonical gate.
_VALID_OBS_MODES = {
    "dom", "som", "vision",
    "phantom_som", "phantom_text", "phantom_prompt",
    "learned",  # v7 sentinel for learned router dispatch (runner main.py:1544)
}
_DEPRECATED_OBS_MODES = {
    "phantom_dom": "phantom_text",  # B-261 (2026-05-16): legacy alias retired
    "dom_only": None,  # B-263 (2026-05-16): Phase 1 v1 router design, never paper-1
    "hybrid": None,    # B-263 (2026-05-16): Phase 1 v1 router design, never paper-1
}


def _validate_obs_mode(mode: str, *, context: str = "observation_mode") -> str:
    """Validate that `mode` is in the canonical paper-1 6-mode universe.

    Raises ValueError on deprecated/retired modes (with replacement hint
    when available) or unknown modes (with valid-list hint). Returns the
    mode unchanged on success — callers can use it as a fluent passthrough:
    `obs_mode = _validate_obs_mode(yaml_value, context="phase2.fixed")`.
    """
    if mode in _DEPRECATED_OBS_MODES:
        replacement = _DEPRECATED_OBS_MODES[mode]
        hint = (
            f"use canonical '{replacement}' instead"
            if replacement
            else "this mode was retired and has no replacement in paper-1 scope"
        )
        raise ValueError(
            f"{context}='{mode}' is deprecated/retired in conditions.py; {hint}. "
            f"Valid modes: {sorted(_VALID_OBS_MODES)}"
        )
    if mode not in _VALID_OBS_MODES:
        raise ValueError(
            f"Unknown {context} '{mode}'. "
            f"Valid: {sorted(_VALID_OBS_MODES)}"
        )
    return mode


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
    # B-701 (/stress A1.7 cold-start P2-13-A, 2026-05-17): list valid options
    # in the error message for DX parity with `_validate_obs_mode` (L80+).
    # Pre-fix the error said "Unknown module name: <typed>" without hint —
    # config authors had to grep code to find aliases.
    _VALID_MODULE_NAMES = sorted([
        "none", "off", "baseline",
        "m1", "m1_dom_select", "m1_dom_select_fallback",
        "m2", "m2_dom_input", "m2_dom_first_input_fallback",
        "m3", "m3_retry", "m3_failure_trigger_retry",
        "m4", "m4_two_stage", "m4_two_stage_generation_grounding",
    ])
    raise ValueError(
        f"Unknown module name: {name}. Valid: {_VALID_MODULE_NAMES}"
    )


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

    # B-695 (/stress A1.7 cold-start P1-5-A, 2026-05-17): filter out
    # partial-data conditions BEFORE ranking. B-179/B-601/B-659 (chronicle
    # §183+) added `_synthesized=True` markers to condition_summary rows
    # whose underlying data is incomplete (cell crashed mid-run, archive
    # corrupt, etc.) and the downstream plot + Pareto paths honor the
    # flag. But `_load_best_condition_from_phase1` was never wired into
    # that gate — so phase2 "fixed best" selection could silently pick a
    # partial-data condition as the canonical fixed reference, biasing
    # the entire phase2 routed-vs-fixed comparison from the source.
    # Return None if all metrics are synthesized so the caller falls
    # back to the manual fixed_condition path (which is properly
    # validated by `_validate_obs_mode` via B-691).
    condition_metrics = [
        cm for cm in condition_metrics
        if not cm.get("_synthesized", False)
    ]
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
    # B-698 (/stress A1.7 cold-start P2-9-A, 2026-05-17): pre-fix this
    # function silently fell back to the first non-`default_backend` key
    # or to the literal string "local_4b" (= B1 local Qwen). If a yaml
    # forgot to set `backends.default_backend` while leaving the rest of
    # the fields hinting B0 (cost_api / experiment.name = "B0_*"), runs
    # would silently pick the wrong backend → run_meta records mixed
    # baseline state → reviewer audits sees inconsistency. The right
    # behavior is fail-loud so misconfigurations are caught at startup.
    # Note: `config.py:236 normalize_config` injects a default for the
    # standard yaml-merge path (`backends.setdefault("default_backend",
    # next(iter([...])))`) so this raise only fires when callers bypass
    # `normalize_config` (e.g. unit tests building bare cfgs) — in that
    # case the fail-loud catches missing test fixtures.
    non_default_keys = [k for k in backends if k != "default_backend"]
    if non_default_keys:
        return non_default_keys[0]
    raise ValueError(
        "cfg.backends has no `default_backend` and no concrete backend "
        "entries; yaml must explicitly set `backends.default_backend` "
        "(e.g. 'api_strong' for B0, 'local_4b' for B1, 'local_gemma' "
        "for B2). The legacy silent fallback to 'local_4b' was retired "
        "to prevent mixed-baseline run_meta contamination."
    )


def generate_conditions(cfg: Dict[str, Any]) -> List[ConditionSpec]:
    phase = str(cfg["experiment"]["phase"]).lower()
    backend_id = _default_backend_id(cfg)
    primary = cfg.get("variables", {}).get("primary", {})

    conditions: List[ConditionSpec] = []

    if phase == "phase1":
        # Flat 6-mode design: dom / som / vision / phantom_som / phantom_text / phantom_prompt.
        # "som" implies SOM_MARKS + marked image; "vision" implies raw screenshot only.
        # som_on is derived (True only when mode == "som"). phantom_* modes are non-SoM
        # variants probing the phantom routing space sibling-arm property (paper §3 hook).
        # B-261 fix (2026-05-16, A1.7): legacy alias "phantom_dom" deprecated → canonical
        # "phantom_text". Reading phantom_dom from yaml raises ValueError to fail-loud and
        # prevent resume:true × legacy alias silent overwrite of phantom_text data (the
        # cross-AI cycle attack vector). Existing archive run_dirs named
        # phase1_phantom_dom_router_0 stay historical; new fires use phantom_text canonical.
        obs_values = [str(x) for x in primary.get("observation_mode", ["dom", "som", "vision"])]
        # /stress A1.10 P0-3-A (2026-05-16): paper-1 6-mode canonical universe.
        # DEFAULT_CONFIG in config.py:23 historically shipped 3-mode
        # ["dom","som","vision"] which fell back silently when a yaml did NOT
        # override the field — risking incomplete Phase 1a runs missing
        # phantom cells. The fix is at DEFAULT_CONFIG (config.py:23) raising
        # the canonical fallback to the full 6-mode list. Per-condition yamls
        # still legitimately override to a 1-mode subset (e.g.
        # B0_dom_classifieds.yaml uses ["dom"] for a single-cell fire) — that
        # path is intended and not asserted here. The 6-mode discipline is
        # enforced at the Phase 1a launch-orchestrator layer
        # (queue_phase1_paper_grade.sh iterates all 6 modes); we leave
        # generate_conditions permissive so partial fires (e.g. resume one
        # mode) work without configuration gymnastics.
        # B-691 (/stress A1.7 cold-start P0-1-AC*, 2026-05-17): validation
        # delegated to module-level _validate_obs_mode helper (L80+) so
        # phase2 + phase3 branches can reuse the same gate.
        for _mode in obs_values:
            _validate_obs_mode(_mode, context="phase1.observation_mode")

        # Extract model_name from backend config for condition metadata (helps distinguish B0/B1/B2).
        backend_cfg = cfg.get("backends", {}).get(backend_id, {})
        model_name = backend_cfg.get("api_name") or backend_cfg.get("path") or backend_cfg.get("model_path", "unknown")

        # Phase 1a v6 (2026-05-16): `phase1.variant` enum controls which conditions to spawn,
        # enabling A100 sequential launch protocol (baseline pass first → router pass second)
        # per `proposals_v6.md` D3 + user-confirmed Phase 1a execution model 2026-05-16.
        # Values:
        #   "baseline" (default, backward-compat) — 6 baseline conditions/cell, no router
        #   "router"   — router-variant conditions/cell (count depends on router_kind), no baseline
        #   "both"     — baseline + router interleaved for single-launch
        # Legacy flag `include_router_variants` kept for backward-compat (True ≡ "both").
        #
        # v7 walk-back 2026-05-16 (Q3 drop cascade per user): `phase1.router_kind` subfield:
        #   "learned" (default v7+) — 1 router cond/cell with obs_mode="learned" sentinel;
        #                             LR predicts mode per task at runtime; no per-mode loop
        #   "cascade" — 6 router cond/cell (one per initial_mode) for v6 cascade L1+L2 design;
        #               DEFERRED to paper-2 per Q3 decision 2026-05-16
        phase1_cfg = cfg.get("variables", {}).get("phase1", {})
        if "variant" in phase1_cfg:
            variant_mode = str(phase1_cfg["variant"]).lower()
        elif phase1_cfg.get("include_router_variants", False):
            variant_mode = "both"
        else:
            variant_mode = "baseline"
        if variant_mode not in ("baseline", "router", "both"):
            raise ValueError(
                f"phase1.variant must be 'baseline'|'router'|'both', got: {variant_mode}"
            )
        emit_baseline = variant_mode in ("baseline", "both")
        emit_router = variant_mode in ("router", "both")
        router_kind = str(phase1_cfg.get("router_kind", "learned")).lower()
        if router_kind not in ("learned", "cascade"):
            raise ValueError(
                f"phase1.router_kind must be 'learned'|'cascade', got: {router_kind}"
            )

        for obs_mode in obs_values:
            som_on = obs_mode == "som"
            # B-692 (/stress A1.7 cold-start P0-2-A*, 2026-05-17): explicit
            # guard against emitting a "baseline" condition with the LR
            # sentinel mode. Pre-fix any yaml setting
            # observation_mode=["learned"] AND variant="both" (or missing
            # variant default "baseline") would produce a
            # `phase1_learned_router_0` condition with router_on=False —
            # but runner/main.py:1544 still triggers LR dispatch when
            # `condition.observation_mode == "learned"` regardless of
            # router_on, silently contaminating the "baseline" cell's SR
            # and cost data with LR-routed predictions. The cross-AI cycle
            # attack vector (Mode A self-audit + Mode C gemini overlap):
            # paper §1 hero claim "B0 baseline vs router" becomes "LR vs
            # LR" self-comparison. Fail-loud forces yaml authors to pick
            # phase1.variant explicitly when mixing learned with baseline
            # emit.
            if emit_baseline and obs_mode == "learned":
                raise ValueError(
                    f"observation_mode='learned' is a router-only sentinel; "
                    f"yaml mistakenly enabled it in the baseline emit pass "
                    f"(phase1.variant={variant_mode!r}). Set "
                    f"phase1.variant='router' to emit only the LR-dispatch "
                    f"condition, or remove 'learned' from observation_mode."
                )
            if emit_baseline:
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
                        metadata={"model_name": model_name, "router_variant": "baseline"},
                    )
                )
            if emit_router and router_kind == "cascade":
                cid_routed = f"phase1_{obs_mode}_router_v6"
                conditions.append(
                    ConditionSpec(
                        condition_id=cid_routed,
                        phase="phase1",
                        backend_id=backend_id,
                        som_on=som_on,
                        observation_mode=obs_mode,
                        router_on=True,
                        modules=ModuleFlags(),
                        label=f"Phase1 {obs_mode.upper()} mode + v6 router",
                        metadata={
                            "model_name": model_name,
                            "router_variant": "v6_pareto_cascade",
                            "initial_mode": obs_mode,
                        },
                    )
                )
        if emit_router and router_kind == "learned":
            # v7 learned-only: 1 router condition per cell (not per-mode); LR picks mode
            # per task at runtime. obs_mode="learned" sentinel signals runner to query LR.
            #
            # B-694 (/stress A1.7 cold-start P1-4-A, 2026-05-17): condition_id
            # now includes backend_id + site context. Pre-fix the id was a
            # bare constant "phase1_learned_router" — cross-cell aggregation
            # (analysis.py groups by condition_id) lumped all 6 router
            # conditions (B0/B1/B2 × cls/red) into a single bucket so
            # paper §6 router-pass results collapsed to one mean instead
            # of 6 cell-stratified rows. No backward-compat shim because
            # Pass-2 router has not fired yet (user confirmed 2026-05-17
            # /stress A1.7 fix scope decision Q1=B).
            #
            # B-696 (/stress A1.7 cold-start P1-6-AC, 2026-05-17): metadata
            # "mode_set" now reads `phase1.candidate_modes` from yaml
            # rather than echoing the sentinel `obs_values=["learned"]`.
            # The yaml field had zero code consumers (grep confirms 0
            # matches in p79/ + scripts/) — this is the consumer. The LR
            # actually predicts over the trained label space; the yaml
            # field encodes that label space so paper §6 Oracle ceiling
            # / drop-one analysis scripts can reconstruct the choice
            # space from condition metadata alone.
            phase1_cfg_local = cfg.get("variables", {}).get("phase1", {})
            candidate_modes = phase1_cfg_local.get("candidate_modes", obs_values)
            include_sites = cfg.get("task", {}).get("include_sites", [])
            if len(include_sites) != 1:
                raise ValueError(
                    f"phase1_learned_router condition requires exactly one "
                    f"site in task.include_sites (got {include_sites!r}); "
                    f"learned router is trained per-cell so single-site fires "
                    f"are the only supported launch pattern."
                )
            site_hint = str(include_sites[0])
            cid_learned = f"phase1_learned_router_{backend_id}_{site_hint}"
            conditions.append(
                ConditionSpec(
                    condition_id=cid_learned,
                    phase="phase1",
                    backend_id=backend_id,
                    som_on=False,  # LR may pick som per-task; per-condition som_on n/a
                    observation_mode="learned",
                    router_on=True,
                    modules=ModuleFlags(),
                    label=f"Phase1 learned router ({backend_id}/{site_hint})",
                    metadata={
                        "model_name": model_name,
                        "router_variant": "v7_learned",
                        "mode_set": list(candidate_modes),
                        "site": site_hint,
                    },
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
            # B-691 (/stress A1.7 cold-start P0-1-AC*, 2026-05-17): validate
            # phase2 fixed_condition obs_mode against the canonical 6-mode
            # universe. Pre-fix `exp_v2_phase2.yaml:13` declared
            # observation_mode="hybrid" (retired in `_DEPRECATED_OBS_MODES`
            # via B-263) but phase2 branch never called the gate — so a
            # future phase2 fire would silently emit a retired-mode
            # condition. Yaml post-fix points to "som" (paper-1 hero arm)
            # so this gate normally passes; raise here protects against
            # future config decay.
            _validate_obs_mode(obs_mode, context="phase2.fixed_condition.observation_mode")
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
        # B-691 (/stress A1.7 cold-start P0-1-AC*, 2026-05-17): validate
        # phase3 base_condition obs_mode against the canonical 6-mode
        # universe. Pre-fix `exp_v2_phase3.yaml:11` declared
        # observation_mode="dom_only" (retired) but phase3 branch never
        # called the gate. Yaml post-fix points to "dom" (paper §3 default
        # module-ablation baseline); gate protects against future drift.
        _validate_obs_mode(base_obs, context="phase3.base_condition.observation_mode")
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

    # B-269 fix (2026-05-16, A1.7): `baselines.run_b0` is dead code in paper-1
    # 3-baseline architecture (B0/B1/B2 as top-level cells). It was a Phase 1 v1
    # design artifact ("B1-only + B0 strong upper bound"). Raise on True to prevent
    # accidental fire from stale yaml or copy-paste error producing duplicate
    # b0_strong_upper_bound conditions alongside the regular B0 cell.
    baselines = cfg.get("baselines", {})
    if baselines.get("run_b0", False):
        if phase == "phase1":
            raise ValueError(
                "baselines.run_b0=True is retired for Phase 1a 3-baseline "
                "architecture (B0/B1/B2 as top-level cells). The flag was a "
                "Phase 1 v1 design artifact. Use per-condition yamls per "
                "baseline instead (configs/exp_v2_B0_*.yaml, B1_*, B2_*)."
            )
        # Phase 2/3 may still want the b0 upper-bound condition (paper-2 substrate).
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
