from __future__ import annotations

import json
import logging
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from p79.backends.base import BackendStepContext
from p79.backends.factory import create_backend
from p79.backends.action_utils import validate_action
from p79.envs.vwa_wrapper import P79Observation
from p79.experiment.checklist_module import ChecklistManagerLite
from p79.experiment.conditions import generate_conditions
from p79.experiment.config import resolve_output_root
from p79.experiment.energy_tracker import LightweightEnergyTracker
from p79.experiment.environment import create_environment, create_evaluator
from p79.experiment.logger_v2 import LoggerV2
from p79.experiment.metrics import (
    aggregate_condition_metrics,
    compute_router_overhead_cost,
    compute_token_cost,
    detect_benchmark_noise,
    net_saving,
    p95,
)
from p79.experiment.modules import apply_secondary_modules, m3_retry_action, should_trigger_m3_retry
from p79.experiment.router import RouterState, RuleBasedRouter
from p79.experiment.som import apply_som
from p79.experiment.state_change import build_page_state, detect_page_state_change
from p79.experiment.tasks import load_tasks
from p79.experiment.types import (
    SCHEMA_VERSION_V2,
    ConditionSpec,
    EpisodeSummaryV2,
    RunSummaryV2,
    StepRecordV2,
    validate_step_record_v2,
)

logger = logging.getLogger(__name__)


def _parse_seeds(seed_value: Any) -> List[int]:
    """Accept seed as int or list of ints."""
    if isinstance(seed_value, (list, tuple)):
        return [int(s) for s in seed_value]
    return [int(seed_value)]


def _action_signature(action: Dict[str, Any]) -> str:
    """Compact fingerprint of an action for cycle detection (strict: includes element_id)."""
    atype = str(action.get("action_type", "")).lower()
    eid = action.get("element_id", "")
    text = str(action.get("text", ""))[:60]
    coord = action.get("coordinate", "")
    delta = action.get("delta", "")
    return f"{atype}|eid={eid}|t={text}|c={coord}|d={delta}"


def _action_signature_soft(action: Dict[str, Any]) -> str:
    """Loose fingerprint ignoring element_id/coordinate (catches semantic loops
    where the same search query or click-type is repeated on re-rendered pages)."""
    atype = str(action.get("action_type", "")).lower()
    text = str(action.get("text", ""))[:60]
    delta = action.get("delta", "")
    return f"{atype}|t={text}|d={delta}"


def _detect_action_cycle(signatures: List[str], min_cycle: int = 1, max_cycle: int = 4,
                         min_reps: int = 3) -> int:
    """Return cycle length if the tail of *signatures* is a repeating cycle, else 0.

    Requires at least *min_reps* full repetitions of the cycle to trigger.
    E.g. [A,B,A,B,A,B] → cycle_len=2.  [A,A,A] → cycle_len=1.
    """
    n = len(signatures)
    for clen in range(min_cycle, max_cycle + 1):
        window = clen * min_reps
        if n < window:
            continue
        tail = signatures[-window:]
        pattern = tail[:clen]
        if all(tail[i] == pattern[i % clen] for i in range(window)):
            return clen
    return 0


class ExperimentRunner:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.output_root = resolve_output_root(cfg)
        self.phase = str(cfg["experiment"]["phase"]).lower()
        self.seeds = _parse_seeds(cfg["experiment"]["seed"])
        self.seed = self.seeds[0]
        self.max_steps = int(cfg.get("runtime", {}).get("max_steps", 40))
        self.resume = bool(cfg.get("runtime", {}).get("resume", True))

        self.conditions = generate_conditions(cfg)
        self.tasks = load_tasks(cfg, self.output_root)

        self.router = RuleBasedRouter(cfg)
        self.environment = create_environment(cfg.get("env", {}))
        self.evaluator = create_evaluator(cfg.get("env", {}))
        self.checklist_cfg = cfg.get("checklist", {})
        self.state_change_cfg = cfg.get("state_change", {})
        self.energy_tracker = LightweightEnergyTracker(cfg.get("metrics", {}).get("energy", {}))

        self._backends: Dict[str, Any] = {}

    def _get_backend(self, backend_id: str):
        if backend_id in self._backends:
            return self._backends[backend_id]

        backend_cfg = self.cfg.get("backends", {}).get(backend_id)
        if not backend_cfg:
            raise KeyError(f"Backend {backend_id} is not defined in config.backends")
        backend = create_backend(backend_id, backend_cfg)
        self._backends[backend_id] = backend
        return backend

    def _write_run_meta(self) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION_V2,
            "run_id": self.cfg["experiment"]["run_id"],
            "timestamp": time.time(),
            "log_path": self.cfg["experiment"].get("log_path"),
            "config": self.cfg,
            "conditions": [c.as_dict() for c in self.conditions],
            "task_count": len(self.tasks),
        }
        with open(self.output_root / "run_meta.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _cleanup_stale_runs(self) -> None:
        """Remove run dirs with 0 episode summaries that are older than 1 hour."""
        parent = self.output_root.parent  # e.g. results/{benchmark}/{phase}/
        if not parent.is_dir():
            return
        one_hour_ago = time.time() - 3600
        for run_dir in parent.iterdir():
            if not run_dir.is_dir() or run_dir == self.output_root:
                continue
            if run_dir.is_symlink():
                continue
            try:
                mtime = run_dir.stat().st_mtime
            except OSError:
                continue
            if mtime > one_hour_ago:
                continue
            # Check for any episode summary files.
            summaries = list(run_dir.glob("**/episode_summary_*.json"))
            if summaries:
                continue
            logger.info("Cleaning stale run dir (0 episodes, age>1h): %s", run_dir)
            try:
                shutil.rmtree(run_dir)
            except OSError as exc:
                logger.warning("Failed to remove stale run dir %s: %s", run_dir, exc)

    def _create_latest_symlink(self) -> None:
        """Create a latest_{site} symlink in the phase directory pointing to this run."""
        sites = self.cfg.get("task", {}).get("include_sites", [])
        if len(sites) == 1:
            site = sites[0]
        else:
            site = None
        link_name = f"latest_{site}" if site else "latest"
        latest_link = self.output_root.parent / link_name
        try:
            latest_link.unlink(missing_ok=True)
            latest_link.symlink_to(self.output_root.name)
            logger.info("Created symlink %s -> %s", latest_link, self.output_root.name)
        except OSError as exc:
            logger.warning("Failed to create latest symlink %s: %s", latest_link, exc)

    @staticmethod
    def _normalize_error_category(
        failure_reason: Optional[str],
        action_success: bool,
        page_changed: bool,
        env_error: Optional[str] = None,
    ) -> Optional[str]:
        if env_error:
            is_noise, _ = detect_benchmark_noise(env_error)
            return "benchmark_noise" if is_noise else "env_error"

        reason = str(failure_reason or "").strip().lower()
        if reason:
            if any(
                key in reason
                for key in (
                    "parse",
                    "json",
                    "keyword_",
                    "repaired_regex",
                )
            ):
                return "parse_error"
            if any(key in reason for key in ("invalid", "schema", "element_id", "action_type")):
                return "invalid_action"
            if any(key in reason for key in ("timeout", "playwright", "browser", "connection", "network", "env")):
                return "env_error"
            return "invalid_action"

        if not action_success and not page_changed:
            return "no_progress"
        return None

    def run(self) -> Path:
        self._cleanup_stale_runs()
        self._write_run_meta()

        run_condition_metrics: List[Dict[str, Any]] = []

        for condition in self.conditions:
            for current_seed in self.seeds:
                self.seed = current_seed
                seed_suffix = f"_seed{current_seed}" if len(self.seeds) > 1 else ""
                effective_cid = f"{condition.condition_id}{seed_suffix}"

                condition_dir = self.output_root / effective_cid
                condition_dir.mkdir(parents=True, exist_ok=True)
                condition_logger = LoggerV2(condition_dir)
                cond_meta = condition.as_dict()
                cond_meta["condition_id"] = effective_cid
                cond_meta["seed"] = current_seed
                condition_logger.write_condition_meta(cond_meta)

                episode_summaries: List[Dict[str, Any]] = []
                backend = self._get_backend(condition.backend_id)

                for task in self.tasks:
                    summary_file = condition_logger.summary_path(task.site, task.task_id)
                    if self.resume and summary_file.exists():
                        with open(summary_file, "r", encoding="utf-8") as f:
                            episode_summaries.append(json.load(f))
                        continue

                    logger.info(
                        "Running condition=%s seed=%d backend=%s site=%s task=%s",
                        effective_cid,
                        current_seed,
                        condition.backend_id,
                        task.site,
                        task.task_id,
                    )

                    try:
                        summary = self._run_episode(condition, task, backend, condition_logger, condition_dir)
                    except Exception as exc:
                        logger.warning(
                            "Episode failed at condition=%s seed=%d site=%s task=%s: %s",
                            effective_cid,
                            current_seed,
                            task.site,
                            task.task_id,
                            exc,
                        )
                        noise, noise_cat = detect_benchmark_noise(str(exc))
                        summary = EpisodeSummaryV2(
                            schema_version=SCHEMA_VERSION_V2,
                            run_id=self.cfg["experiment"]["run_id"],
                            condition_id=effective_cid,
                            benchmark=task.benchmark,
                            benchmark_site=task.site,
                            task_id=task.task_id,
                            seed=self.seed,
                            success=False,
                            score=0.0,
                            steps=0,
                            retries=0,
                            no_op_rate=0.0,
                            page_unchanged_rate=0.0,
                            total_latency_ms=0.0,
                            p95_step_latency_ms=0.0,
                            total_tokens=0,
                            total_model_cost_usd=0.0,
                            total_cost_usd=0.0,
                            total_router_overhead_cost_usd=0.0,
                            total_router_overhead_ms=0.0,
                            total_energy_kwh=None,
                            total_co2e_kg=None,
                            escalation_count=0,
                            trigger_distribution={},
                            benchmark_noise=noise,
                            benchmark_noise_category=noise_cat,
                            artifacts_dir=str(condition_dir),
                            error=str(exc),
                        ).as_dict()

                    condition_logger.write_episode_summary(task.site, task.task_id, summary)
                    episode_summaries.append(summary)

                aggregate = aggregate_condition_metrics(episode_summaries)
                aggregate.update(
                    {
                        "condition_id": effective_cid,
                        "seed": current_seed,
                        "phase": condition.phase,
                        "backend_id": condition.backend_id,
                        "som_on": condition.som_on,
                        "observation_mode": condition.observation_mode,
                        "router_on": condition.router_on,
                        "module_flags": condition.modules.as_dict(),
                    }
                )

                condition_logger.write_condition_summary(aggregate)
                run_condition_metrics.append(aggregate)

        assumptions = {
            "som_fallback": "degrade_to_text_som_with_flag",
            "mind2web": "deferred",
            "energy_missing_policy": "store_null_and_report_missing_ratio",
            "router_thresholds": self.cfg.get("router", {}).get("thresholds", {}),
        }

        if self.phase == "phase2":
            fixed = next((x for x in run_condition_metrics if x.get("condition_id") == "phase2_fixed_best"), None)
            routed = next((x for x in run_condition_metrics if x.get("condition_id") == "phase2_routed"), None)
            if fixed and routed:
                assumptions["phase2_net_saving"] = net_saving(
                    fixed.get("avg_total_cost_usd", 0.0),
                    routed.get("avg_total_model_cost_usd", 0.0),
                    routed.get("avg_router_overhead_cost_usd", 0.0),
                )

        run_trigger_dist: Counter = Counter()
        run_state_change_reason_dist: Counter = Counter()
        benchmark_noise_rates: List[float] = []
        for row in run_condition_metrics:
            trigger_dist = row.get("trigger_distribution", {}) or {}
            if isinstance(trigger_dist, dict):
                for reason, count in trigger_dist.items():
                    try:
                        run_trigger_dist[str(reason)] += int(count)
                    except Exception:
                        continue
            dist = row.get("state_change_reason_distribution", {}) or {}
            if isinstance(dist, dict):
                for reason, count in dist.items():
                    try:
                        run_state_change_reason_dist[str(reason)] += int(count)
                    except Exception:
                        continue
            if row.get("benchmark_noise_rate") is not None:
                benchmark_noise_rates.append(float(row.get("benchmark_noise_rate")))
        assumptions["trigger_distribution"] = dict(run_trigger_dist)
        assumptions["state_change_reason_distribution"] = dict(run_state_change_reason_dist)
        assumptions["benchmark_noise_rate_avg_across_conditions"] = (
            (sum(benchmark_noise_rates) / len(benchmark_noise_rates)) if benchmark_noise_rates else 0.0
        )

        run_summary = RunSummaryV2(
            schema_version=SCHEMA_VERSION_V2,
            run_id=self.cfg["experiment"]["run_id"],
            benchmark=self.cfg["experiment"]["benchmark"],
            phase=self.phase,
            total_conditions=len(self.conditions),
            total_episodes=sum(x.get("episodes", 0) for x in run_condition_metrics),
            condition_metrics=run_condition_metrics,
            assumptions=assumptions,
        ).as_dict()

        with open(self.output_root / "run_summary_v2.json", "w", encoding="utf-8") as f:
            json.dump(run_summary, f, indent=2, ensure_ascii=False)

        self._create_latest_symlink()
        self.environment.close()
        return self.output_root

    def _clone_observation_for_mode(self, obs: P79Observation, observation_text: str, mode: str) -> P79Observation:
        image = obs.image if mode == "hybrid" else None
        return P79Observation(text=observation_text, image=image, url=obs.url, raw=obs.raw)

    def _save_artifacts(self, episode_dir: Path, step_idx: int, obs: P79Observation) -> Dict[str, Optional[str]]:
        step_dir = episode_dir / f"step_{step_idx:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        screenshot_path = None
        if getattr(obs, "image", None) is not None:
            screenshot_path = str(step_dir / "screenshot.png")
            try:
                obs.image.save(screenshot_path)
            except Exception:
                screenshot_path = None

        dom_path = str(step_dir / "observation_dom.txt")
        with open(dom_path, "w", encoding="utf-8") as f:
            f.write(obs.text or "")

        return {
            "screenshot": screenshot_path,
            "dom": dom_path,
            "trace": None,
            "som_image": None,
        }

    def _run_episode(
        self,
        condition: ConditionSpec,
        task: Any,
        backend: Any,
        condition_logger: LoggerV2,
        condition_dir: Path,
    ) -> Dict[str, Any]:
        episode_dir = condition_dir / "artifacts" / f"{task.site}_task_{task.task_id}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        obs, info = self.environment.reset(task.config_file)
        current_info = info or {}
        trajectory: List[Any] = [{"observation": getattr(obs, "raw", None), "info": current_info}]

        router_state = RouterState()
        prev_action_success: Optional[bool] = None
        prev_page_changed: Optional[bool] = None

        step_records: List[Dict[str, Any]] = []
        trigger_distribution: Counter = Counter()
        state_change_reason_distribution: Counter = Counter()

        retry_total = 0
        escalation_count = 0
        action_signatures: List[str] = []
        action_signatures_soft: List[str] = []
        cycle_early_stop = False

        checklist_manager: Optional[ChecklistManagerLite] = None
        if bool(self.checklist_cfg.get("enabled", False)):
            checklist_manager = ChecklistManagerLite(
                task_description=task.intent,
                max_items=int(self.checklist_cfg.get("max_items", 4)),
            )
        latest_checklist_status = checklist_manager.get_status() if checklist_manager is not None else None

        similarity_threshold = float(self.state_change_cfg.get("similarity_threshold", 0.95))

        for step_idx in range(self.max_steps):
            step_start = time.time()

            artifacts = self._save_artifacts(episode_dir, step_idx, obs)
            som_result = apply_som(obs, condition.som_on, episode_dir, step_idx)
            artifacts["som_image"] = som_result.marked_image_path

            decision_mode, triggers, overhead, router_state = self.router.decide(
                router_enabled=condition.router_on,
                preferred_mode=condition.observation_mode,
                obs_text=som_result.som_text,
                state=router_state,
                prev_action_success=prev_action_success,
                prev_page_changed=prev_page_changed,
                checklist_status=latest_checklist_status,
            )
            trigger_distribution.update(triggers)

            if condition.router_on and decision_mode == "hybrid":
                escalation_count += 1

            screenshot_prep_start = time.time()
            obs_for_backend = self._clone_observation_for_mode(obs, som_result.som_text, decision_mode)
            if condition.router_on and decision_mode == "hybrid" and condition.observation_mode == "dom_only":
                overhead["extra_screenshot_ms"] = (time.time() - screenshot_prep_start) * 1000.0
            instruction = task.intent
            if checklist_manager and bool(self.checklist_cfg.get("inject_into_prompt", True)):
                instruction = f"{task.intent}\n\n{checklist_manager.format_for_prompt()}"

            context = BackendStepContext(
                observation_mode=decision_mode,
                som_enabled=condition.som_on,
                som_text=som_result.som_text,
                stage="single",
                history=step_records[-8:],
                module_flags=condition.modules.as_dict(),
            )

            planner_meta = {}
            planner_sub_goal: Optional[str] = None
            if condition.modules.m4_two_stage_generation_grounding:
                planner_context = BackendStepContext(
                    observation_mode=decision_mode,
                    som_enabled=condition.som_on,
                    som_text=som_result.som_text,
                    stage="planner",
                    history=step_records[-8:],
                    module_flags=condition.modules.as_dict(),
                )
                planner_action, planner_meta = backend.step(instruction, obs_for_backend, planner_context)
                overhead["extra_model_calls"] += float(planner_meta.get("model_calls", 1))
                planner_sub_goal = (
                    planner_action.get("thought")
                    or planner_meta.get("raw_text")
                    or str(planner_action)
                )

            backend_start = time.time()
            call_stage = "grounder" if condition.modules.m4_two_stage_generation_grounding else "single"
            context.stage = call_stage
            context.planner_sub_goal = planner_sub_goal
            action, meta = backend.step(instruction, obs_for_backend, context)
            backend_latency_ms = (time.time() - backend_start) * 1000.0

            action = validate_action(action)
            action = apply_secondary_modules(action, obs.text or "", condition.modules.as_dict())
            state_before = build_page_state(obs, current_info)

            env_step_start = time.time()
            next_obs, reward, terminated, truncated, next_info = self.environment.step(action)
            env_step_ms = (time.time() - env_step_start) * 1000.0

            state_after = build_page_state(next_obs, next_info)
            action_success, page_change_reasons, text_similarity = detect_page_state_change(
                state_before=state_before,
                state_after=state_after,
                action_type=str(action.get("action_type", "")).upper(),
                similarity_threshold=similarity_threshold,
            )
            page_changed = bool(page_change_reasons)
            if reward > 0 or terminated:
                action_success = True

            retry_count = 0
            retry_limit = int(self.cfg.get("router", {}).get("thresholds", {}).get("retry_limit", 1))
            if should_trigger_m3_retry(
                action_success=action_success,
                page_changed=page_changed,
                retry_count=retry_count,
                retry_limit=retry_limit,
                module_flags=condition.modules.as_dict(),
            ):
                retry_action = m3_retry_action(failed_action=action, obs_text=obs.text or "")
                retry_obs, retry_reward, retry_term, retry_trunc, retry_info = self.environment.step(retry_action)
                retry_count += 1
                retry_total += 1
                overhead["routing_retry_count"] += 1.0

                retry_state_after = build_page_state(retry_obs, retry_info)
                retry_success, retry_reasons, retry_similarity = detect_page_state_change(
                    state_before=state_before,
                    state_after=retry_state_after,
                    action_type=str(retry_action.get("action_type", "")).upper(),
                    similarity_threshold=similarity_threshold,
                )
                if retry_success or retry_reward > 0 or retry_term:
                    next_obs, reward, terminated, truncated, next_info = (
                        retry_obs,
                        retry_reward,
                        retry_term,
                        retry_trunc,
                        retry_info,
                    )
                    state_after = retry_state_after
                    page_change_reasons = list(dict.fromkeys(list(retry_reasons) + ["m3_retry_applied"]))
                    text_similarity = retry_similarity
                    page_changed = bool(retry_reasons)
                    action_success = True

            safe_next_info = next_info if isinstance(next_info, dict) else {}
            if "raw_action" in safe_next_info:
                trajectory.append(safe_next_info["raw_action"])
            trajectory.append({"observation": getattr(next_obs, "raw", None), "info": safe_next_info})

            input_tokens = int(meta.get("input_tokens") or 0)
            output_tokens = int(meta.get("output_tokens") or 0)
            token_total = input_tokens + output_tokens

            token_cost = compute_token_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_cfg=self.cfg.get("metrics", {}).get("cost", {}),
            )
            router_cfg = self.cfg.get("router", {})
            router_overhead_ms = (
                float(overhead.get("router_decision_ms", 0.0))
                + float(overhead.get("extra_dom_parse_ms", 0.0))
                + float(overhead.get("extra_screenshot_ms", 0.0))
            )
            router_overhead_cost = compute_router_overhead_cost(router_overhead_ms, router_cfg)
            router_overhead_cost += float(overhead.get("extra_model_calls", 0.0)) * float(
                router_cfg.get("extra_model_call_cost_usd", 0.0)
            )
            router_overhead_cost += float(overhead.get("routing_retry_count", 0.0)) * float(
                router_cfg.get("retry_cost_usd", 0.0)
            )
            step_total_cost = token_cost["total"] + router_overhead_cost
            failure_reason = meta.get("failure_reason")
            error_category = self._normalize_error_category(
                failure_reason=failure_reason if not meta.get("valid", True) else None,
                action_success=bool(action_success),
                page_changed=bool(page_changed),
                env_error=None,
            )

            total_latency_ms = (time.time() - step_start) * 1000.0
            energy = self.energy_tracker.estimate_step(duration_seconds=total_latency_ms / 1000.0)

            checklist_snapshot = None
            if checklist_manager is not None:
                checklist_manager.update_after_action(
                    action_success=bool(action_success),
                    error=(meta.get("failure_reason") if not meta.get("valid", True) else None),
                )
                checklist_snapshot = {
                    "items": [dict(item) for item in checklist_manager.task_checklist],
                    "status": checklist_manager.get_status(),
                }
                latest_checklist_status = checklist_snapshot["status"]

            step_record = StepRecordV2(
                schema_version=SCHEMA_VERSION_V2,
                run_id=self.cfg["experiment"]["run_id"],
                condition_id=condition.condition_id,
                benchmark=task.benchmark,
                benchmark_site=task.site,
                task_id=task.task_id,
                seed=self.seed,
                step_idx=step_idx,
                som={
                    "enabled": condition.som_on,
                    "degraded_som": bool(som_result.degraded_som),
                    "mark_count": som_result.mark_count,
                },
                observation_mode=decision_mode,
                router={
                    "enabled": condition.router_on,
                    "decision": decision_mode,
                    "trigger_reason": triggers,
                    "overhead_ms": overhead,
                },
                module_flags=condition.modules.as_dict(),
                action_type=str(action.get("action_type", "wait")),
                action=action,
                action_success=action_success,
                page_changed=page_changed,
                latency_ms={
                    "total": total_latency_ms,
                    "backend_infer": float(meta.get("infer_ms", backend_latency_ms)),
                    "env_step": env_step_ms,
                    "router_decision": float(overhead.get("router_decision_ms", 0.0)),
                },
                tokens={"input": input_tokens, "output": output_tokens, "total": token_total},
                cost_usd={
                    "input": token_cost["input"],
                    "output": token_cost["output"],
                    "model": token_cost["total"],
                    "router_overhead": router_overhead_cost,
                    "total": step_total_cost,
                },
                energy=energy,
                retry_count=retry_count,
                error_category=error_category,
                artifact_paths=artifacts,
                reward=float(reward),
                done=bool(terminated or truncated),
                page_change_reasons=page_change_reasons,
                text_similarity=text_similarity,
                checklist=checklist_snapshot,
                state_digest={
                    "url_before": state_before.get("url"),
                    "url_after": state_after.get("url"),
                    "title_before": state_before.get("title"),
                    "title_after": state_after.get("title"),
                },
            ).as_dict()

            validate_step_record_v2(step_record)
            condition_logger.write_step(task.site, task.task_id, step_record)
            step_records.append(step_record)
            state_change_reason_distribution.update(page_change_reasons)

            obs = next_obs
            current_info = safe_next_info
            prev_action_success = action_success
            prev_page_changed = page_changed

            if terminated or truncated:
                break

            # --- cycle detection (early stop, does not alter agent behaviour) ---
            action_signatures.append(_action_signature(action))
            action_signatures_soft.append(_action_signature_soft(action))
            cycle_len = _detect_action_cycle(action_signatures)
            # Soft check uses higher reps threshold to reduce false positives
            soft_cycle_len = _detect_action_cycle(action_signatures_soft, min_reps=3)
            if cycle_len > 0 or soft_cycle_len > 0:
                detected = cycle_len if cycle_len > 0 else soft_cycle_len
                mode = "strict" if cycle_len > 0 else "soft"
                logger.warning(
                    "Action cycle detected (%s, len=%d, reps>=3) at step %d for task %s/%d — early stop.",
                    mode, detected, step_idx, task.site, task.task_id,
                )
                cycle_early_stop = True
                break

        eval_result = self.evaluator.evaluate(trajectory=trajectory, config_file=task.config_file, env=self.environment)
        score = float(eval_result.score)

        # Override only when the agent issued a finish/stop and VWA reward
        # agrees — never override after cycle early-stop (agent did not finish).
        if (
            score == 0.0
            and step_records
            and step_records[-1].get("reward", 0.0) > 0
            and not cycle_early_stop
            and step_records[-1].get("action_type", "") in ("finish", "stop")
        ):
            score = 1.0

        success = bool(score >= 1.0)

        total_latency = sum(float(s["latency_ms"].get("total", 0.0)) for s in step_records)
        step_latencies = [float(s["latency_ms"].get("total", 0.0)) for s in step_records]
        total_tokens = sum(int(s["tokens"].get("total", 0)) for s in step_records)
        total_model_cost = sum(
            float(
                s["cost_usd"].get(
                    "model",
                    float(s["cost_usd"].get("input", 0.0)) + float(s["cost_usd"].get("output", 0.0)),
                )
            )
            for s in step_records
        )
        total_router_overhead_cost = sum(float(s["cost_usd"].get("router_overhead", 0.0)) for s in step_records)
        total_cost = total_model_cost + total_router_overhead_cost
        total_router_overhead_ms = sum(
            float(s["router"].get("overhead_ms", {}).get("router_decision_ms", 0.0))
            + float(s["router"].get("overhead_ms", {}).get("extra_dom_parse_ms", 0.0))
            + float(s["router"].get("overhead_ms", {}).get("extra_screenshot_ms", 0.0))
            for s in step_records
        )

        no_op_count = sum(1 for s in step_records if not bool(s.get("action_success", False)))
        unchanged_count = sum(1 for s in step_records if not bool(s.get("page_changed", False)))

        energy_vals = [s["energy"].get("kwh") for s in step_records if s["energy"].get("kwh") is not None]
        co2_vals = [s["energy"].get("co2e_kg") for s in step_records if s["energy"].get("co2e_kg") is not None]
        checklist_completion_rate = None
        checklist_failed_items = None
        if latest_checklist_status is not None:
            checklist_completion_rate = float(latest_checklist_status.get("completion_rate", 0.0))
            checklist_failed_items = int(latest_checklist_status.get("failed", 0) or 0)

        noise, noise_category = detect_benchmark_noise(eval_result.error)

        episode_summary = EpisodeSummaryV2(
            schema_version=SCHEMA_VERSION_V2,
            run_id=self.cfg["experiment"]["run_id"],
            condition_id=condition.condition_id,
            benchmark=task.benchmark,
            benchmark_site=task.site,
            task_id=task.task_id,
            seed=self.seed,
            success=success,
            score=score,
            steps=len(step_records),
            retries=retry_total,
            no_op_rate=(no_op_count / len(step_records) if step_records else 0.0),
            page_unchanged_rate=(unchanged_count / len(step_records) if step_records else 0.0),
            total_latency_ms=total_latency,
            p95_step_latency_ms=p95(step_latencies),
            total_tokens=total_tokens,
            total_model_cost_usd=total_model_cost,
            total_cost_usd=total_cost,
            total_router_overhead_cost_usd=total_router_overhead_cost,
            total_router_overhead_ms=total_router_overhead_ms,
            total_energy_kwh=(sum(float(x) for x in energy_vals) if energy_vals else None),
            total_co2e_kg=(sum(float(x) for x in co2_vals) if co2_vals else None),
            escalation_count=escalation_count,
            trigger_distribution=dict(trigger_distribution),
            benchmark_noise=noise,
            benchmark_noise_category=noise_category,
            artifacts_dir=str(episode_dir),
            state_change_reason_distribution=dict(state_change_reason_distribution),
            checklist_completion_rate=checklist_completion_rate,
            checklist_failed_items=checklist_failed_items,
            error=eval_result.error,
        ).as_dict()

        return episode_summary
