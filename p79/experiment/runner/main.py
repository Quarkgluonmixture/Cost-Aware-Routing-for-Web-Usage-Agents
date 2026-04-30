"""ExperimentRunner — main class extracted from runner.py during §97 Step-3 split.

Free helper functions live in `helpers.py`; this module hosts only the
ExperimentRunner class. The original `from p79.experiment.runner import
ExperimentRunner` import path is preserved via `runner/__init__.py`.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import shutil
import time
import urllib.request
import urllib.error
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from p79.backends.base import BackendStepContext
from p79.backends.factory import create_backend
from p79.backends.action_utils import extract_candidate_query, first_element_id_by_keyword, validate_action
from p79.envs.vwa_wrapper import P79Observation
from p79.experiment.checklist_module import ChecklistManagerLite
from p79.experiment.conditions import generate_conditions
from p79.experiment.config import resolve_output_root
from p79.experiment.energy_tracker import LightweightEnergyTracker
from p79.experiment.environment import create_environment, create_evaluator
from p79.experiment.logger_v2 import LoggerV2
from p79.experiment.metrics import (
    aggregate_condition_metrics,
    compute_component_breakdown,
    compute_router_overhead_cost,
    compute_token_cost,
    compute_wasted_cost,
    detect_benchmark_noise,
    net_saving,
    p95,
    select_token_cost_cfg,
)
from p79.experiment.modules import apply_secondary_modules, m3_retry_action, should_trigger_m3_retry
from p79.experiment.router import RouterState, RuleBasedRouter
from p79.experiment.som import prepare_observation_for_mode
from p79.experiment.state_change import build_page_state, detect_page_state_change, is_agent_visible_change
from p79.experiment.tasks import load_tasks
from p79.utils.auth_refresh import refresh_site_auth, should_refresh
from p79.experiment.types import (
    SCHEMA_VERSION_V2,
    ConditionSpec,
    EpisodeSummaryV2,
    RunSummaryV2,
    StepRecordV2,
    validate_step_record_v2,
)

# Helpers extracted into sibling module — re-imported here so any code inside
# the class that calls e.g. `_parse_seeds(...)` resolves correctly.
from p79.experiment.runner.helpers import (
    _parse_seeds,
    _action_signature,
    _action_signature_soft,
    _action_signature_fuzzy,
    _detect_action_cycle,
    _sanitize_query_text,
    _query_sanitization_control,
    _repeat_hits_same_target,
    _build_exploration_fallback_action,
    _anti_repeat_control,
    _no_early_finish_control,
    _notify_retry_pass,
)

logger = logging.getLogger(__name__)


def _seed_global_rng(seed: int) -> None:
    """B-37: propagate seed to Python/NumPy/torch RNG.

    Called at start of each (condition, seed) iteration. Without this, seed=42
    is metadata only — Python random.choice / np.random.shuffle / torch ops
    produce different results across runs. Paper-grade reproducibility claim
    requires this propagation.
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class ExperimentRunner:
    # Fatal environment errors that corrupt Playwright/asyncio state.
    # When caught, re-raise immediately so the process can exit cleanly.
    _FATAL_ENV_MARKERS = (
        "Sync API inside the asyncio",
        "asyncio loop",
        "Event loop is closed",
    )

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
        env_cfg = dict(cfg.get("env", {}))
        env_cfg.setdefault("benchmark", cfg["experiment"]["benchmark"])
        self.environment = create_environment(env_cfg)
        self.evaluator = create_evaluator(env_cfg)
        self.checklist_cfg = cfg.get("checklist", {})
        self.state_change_cfg = cfg.get("state_change", {})
        self.energy_tracker = LightweightEnergyTracker(cfg.get("metrics", {}).get("energy", {}))
        self.diagnostic_controls = cfg.get("diagnostic_controls", {}) or {}

        self._backends: Dict[str, Any] = {}
        self._auth_episode_counts: Dict[str, int] = {}  # per-site counter for auth refresh
        # Per-site N/A task IDs cache — used by §95 adjusted_success computation
        # in _run_episode. Pre-loaded once to avoid repeated config file reads.
        self._na_ids_cache: Dict[str, set] = {}
        try:
            from p79.experiment.analysis import _load_na_task_ids
            _benchmark = self.cfg.get("experiment", {}).get("benchmark", "visualwebarena")
            for _site in self.cfg.get("task", {}).get("include_sites", []):
                self._na_ids_cache[str(_site)] = _load_na_task_ids(str(_site), _benchmark)
        except Exception as _exc:
            logger.warning("Failed to pre-load N/A task IDs: %s", _exc)

    def _get_backend(self, backend_id: str):
        if backend_id in self._backends:
            return self._backends[backend_id]

        backend_cfg = self.cfg.get("backends", {}).get(backend_id)
        if not backend_cfg:
            raise KeyError(f"Backend {backend_id} is not defined in config.backends")
        # B-37 fix: inject experiment seed into backend cfg for downstream
        # propagation to LLM payload (proxy `seed` param) and torch generation.
        # Uses self.seed which was set per (condition, seed) pair in run().
        backend_cfg = dict(backend_cfg)  # shallow copy to avoid mutating self.cfg
        if backend_cfg.get("seed") is None:
            backend_cfg["seed"] = int(self.seed)
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
        """Conservative cleanup for clearly-aborted, empty run dirs only.

        Safety rules:
        - Never touch dirs containing run metadata/progress artifacts.
        - Use latest mtime in the whole run tree (not just top-level dir).
        - Allow opt-out via P79_DISABLE_STALE_CLEANUP=1.
        """
        if str(os.getenv("P79_DISABLE_STALE_CLEANUP", "")).strip().lower() in {"1", "true", "yes", "on"}:
            logger.info("Skipping stale run cleanup (P79_DISABLE_STALE_CLEANUP is enabled).")
            return

        def _latest_tree_mtime(path: Path) -> float:
            latest = 0.0
            try:
                latest = path.stat().st_mtime
            except OSError:
                return latest
            for child in path.rglob("*"):
                try:
                    child_mtime = child.stat().st_mtime
                except OSError:
                    continue
                if child_mtime > latest:
                    latest = child_mtime
            return latest

        parent = self.output_root.parent  # e.g. results/{benchmark}/{phase}/
        if not parent.is_dir():
            return
        one_hour_ago = time.time() - 3600
        # Analysis-output prefixes that share parent dir but are NOT run dirs
        # (do not delete these even if they look "empty" by the run-dir heuristic).
        _ANALYSIS_PREFIXES = (
            "b0_vs_b1",
            "cross_site",
            "cross_benchmark",
            "comparison_",
            "combined_",
            "latest_",  # symlinks already skipped, but be defensive
            "aggregate_",
            "B0_3mode/",  # gallery aggregate symlink targets
            "B1_3mode/",
        )
        for run_dir in parent.iterdir():
            if not run_dir.is_dir() or run_dir == self.output_root:
                continue
            if run_dir.is_symlink():
                continue
            # Skip non-run analysis output directories (b0_vs_b1, cross_site, etc.)
            if any(run_dir.name.startswith(p.rstrip("/")) for p in _ANALYSIS_PREFIXES):
                continue
            latest_mtime = _latest_tree_mtime(run_dir)
            if latest_mtime > one_hour_ago:
                continue

            has_run_meta = (run_dir / "run_meta.json").exists()
            has_progress = (
                next(run_dir.glob("**/*_summary_v2.json"), None) is not None
                or next(run_dir.glob("**/*_steps_v2.jsonl"), None) is not None
                or next(run_dir.glob("**/condition_meta.json"), None) is not None
            )
            if has_run_meta or has_progress:
                continue

            try:
                file_count = sum(1 for _ in run_dir.rglob("*") if _.is_file())
            except OSError:
                continue
            # Only remove tiny/empty trees to avoid deleting meaningful runs.
            if file_count > 5:
                continue
            logger.info("Cleaning stale empty run dir (files=%d, age>1h): %s", file_count, run_dir)
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
                # B-37 fix: propagate seed to Python/NumPy/torch RNG so seed=42 is
                # actually deterministic, not just metadata. Per (condition, seed)
                # pair so each condition gets fresh RNG state from the same seed.
                _seed_global_rng(current_seed)
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
                        try:
                            with open(summary_file, "r", encoding="utf-8") as f:
                                loaded = json.load(f)
                            has_steps = int(loaded.get("steps", 0)) > 0
                            has_error = bool(loaded.get("error"))

                            if has_steps or not has_error:
                                episode_summaries.append(loaded)
                                continue

                            # Zero-step error: skip — watchdog handles all retries
                            # with MAX_NOISE_RETRIES / MAX_CODE_BUG_RETRIES limits.
                            # Runner's retry pass (line 544) will re-run if watchdog
                            # has already deleted the summary.
                            logger.info(
                                "Skipping zero-step error episode site=%s task=%s (watchdog handles retry): %s",
                                task.site, task.task_id,
                                str(loaded.get("error", ""))[:120],
                            )
                            episode_summaries.append(loaded)
                            continue
                        except Exception:
                            pass  # Corrupted summary — fall through to re-run

                    summary = self._run_and_record_episode(
                        condition, task, backend, condition_logger,
                        condition_dir, effective_cid, current_seed,
                    )
                    episode_summaries.append(summary)

                # ── Retry pass: re-run tasks whose summaries were deleted ──
                # (e.g. by watchdog auto-cleanup of benchmark noise errors)
                retry_tasks = [
                    t for t in self.tasks
                    if not condition_logger.summary_path(t.site, t.task_id).exists()
                ]
                if retry_tasks:
                    retry_task_ids = [t.task_id for t in retry_tasks]
                    logger.info(
                        "Retry pass: %d tasks with missing summaries: %s",
                        len(retry_tasks), retry_task_ids,
                    )
                    # Remove stale entries from episode_summaries to avoid duplicates
                    retry_ids = {(t.site, t.task_id) for t in retry_tasks}
                    episode_summaries = [
                        s for s in episode_summaries
                        if (s.get("benchmark_site"), s.get("task_id")) not in retry_ids
                    ]
                    retry_ok, retry_fail = [], []
                    for task in retry_tasks:
                        summary = self._run_and_record_episode(
                            condition, task, backend, condition_logger,
                            condition_dir, effective_cid, current_seed,
                        )
                        episode_summaries.append(summary)
                        if summary.get("error"):
                            retry_fail.append(task.task_id)
                        else:
                            retry_ok.append(task.task_id)

                    # ── Retry pass summary ──
                    logger.info(
                        "Retry pass done: %d/%d succeeded, %d failed",
                        len(retry_ok), len(retry_tasks), len(retry_fail),
                    )
                    if retry_fail:
                        logger.warning(
                            "Retry pass still-failed tasks: %s", retry_fail,
                        )
                    _notify_retry_pass(
                        effective_cid, retry_task_ids, retry_ok, retry_fail,
                    )

                aggregate = aggregate_condition_metrics(episode_summaries)
                aggregate.update(
                    {
                        "condition_id": effective_cid,
                        "seed": current_seed,
                        "phase": condition.phase,
                        "backend_id": condition.backend_id,
                        "som_on": condition.som_on,  # derived: observation_mode == "som"
                        "observation_mode": condition.observation_mode,
                        "router_on": condition.router_on,
                        "module_flags": condition.modules.as_dict(),
                    }
                )

                condition_logger.write_condition_summary(aggregate)
                run_condition_metrics.append(aggregate)

                # Auto-run analysis after each condition completes
                self._run_post_condition_analysis(effective_cid)

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
        self.energy_tracker.close()
        return self.output_root

    def _run_post_condition_analysis(self, condition_id: str) -> None:
        """Run analyze_experiment.py in a subprocess after a condition completes."""
        import subprocess
        import sys
        # __file__ = p79/experiment/runner/main.py → parents[3] = repo root.
        # Bug fix 2026-04-26: was parents[2] (= p79/), pointing to p79/scripts/...
        script = Path(__file__).parents[3] / "scripts" / "analysis" / "analyze_experiment.py"
        if not script.exists():
            logging.warning("[runner] analyze_experiment.py not found, skipping post-condition analysis")
            return
        cmd = [sys.executable, str(script), "--run_dir", str(self.output_root)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                logging.info("[runner] Post-condition analysis completed for %s", condition_id)
            else:
                logging.warning(
                    "[runner] Post-condition analysis exited %d for %s: %s",
                    result.returncode, condition_id, result.stderr[-500:] if result.stderr else "",
                )
        except subprocess.TimeoutExpired:
            logging.warning("[runner] Post-condition analysis timed out for %s", condition_id)
        except Exception as exc:
            logging.warning("[runner] Post-condition analysis failed for %s: %s", condition_id, exc)

        # NOTE: confidence_calibration + cross_representation are triggered by watchdog,
        # not runner, to avoid duplicate runs and enable unified notification.

    def _clone_observation_for_mode(
        self,
        obs: P79Observation,
        mode: str,
        obs_prep: Any,  # SomResult from prepare_observation_for_mode
    ) -> P79Observation:
        """Build a P79Observation appropriate for the given observation mode.

        dom:    Full AXTree text, no image.
        som:    SOM_MARKS compressed text + marked image.
        vision: Empty text + raw screenshot.
        """
        return P79Observation(
            text=obs_prep.som_text,
            image=obs_prep.marked_image,
            url=obs.url,
            raw=obs.raw,
        )

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

    def _run_and_record_episode(
        self,
        condition: ConditionSpec,
        task: Any,
        backend: Any,
        condition_logger: LoggerV2,
        condition_dir: Path,
        effective_cid: str,
        current_seed: int,
    ) -> Dict[str, Any]:
        """Run one episode with error handling and write summary. Returns summary dict.

        Fatal env errors are re-raised; all other errors produce an error summary.
        """
        logger.info(
            "Running condition=%s seed=%d backend=%s site=%s task=%s",
            effective_cid, current_seed, condition.backend_id, task.site, task.task_id,
        )
        try:
            summary = self._run_episode(condition, task, backend, condition_logger, condition_dir)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            exc_str = str(exc)
            if any(marker in exc_str for marker in self._FATAL_ENV_MARKERS):
                logger.error(
                    "Fatal environment error at site=%s task=%s — "
                    "stopping run to allow clean restart: %s",
                    task.site, task.task_id, exc,
                )
                raise
            # Proxy API quota exhaustion — stop run (all subsequent tasks will fail)
            if "403" in exc_str and any(m in exc_str for m in ("model-api", "execute-api")):
                logger.error(
                    "Proxy API quota exhausted at site=%s task=%s — "
                    "stopping run: %s",
                    task.site, task.task_id, exc,
                )
                raise
            logger.warning(
                "Episode failed at condition=%s seed=%d site=%s task=%s: %s",
                effective_cid, current_seed, task.site, task.task_id, exc,
                exc_info=True,
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
            summary["wasted_cost_usd"] = 0.0
            summary["wasted_energy_kwh"] = 0.0
            summary["component_breakdown"] = {
                "model_cost_usd": 0.0,
                "router_overhead_usd": 0.0,
                "total_energy_kwh": 0.0,
            }

        try:
            condition_logger.write_episode_summary(task.site, task.task_id, summary)
        except Exception as write_exc:
            logger.error(
                "Failed to write episode summary for site=%s task=%s: %s",
                task.site, task.task_id, write_exc, exc_info=True,
            )
        logger.info(
            "Episode done site=%s task=%s success=%s steps=%s error=%s",
            task.site, task.task_id, summary.get("success"), summary.get("steps"),
            str(summary.get("error", ""))[:100] or "none",
        )
        return summary

    def _run_episode(
        self,
        condition: ConditionSpec,
        task: Any,
        backend: Any,
        condition_logger: LoggerV2,
        condition_dir: Path,
    ) -> Dict[str, Any]:
        # ── Auth refresh check (before browser context creation) ──
        site = task.site
        self._auth_episode_counts.setdefault(site, 0)
        self._auth_episode_counts[site] += 1
        if should_refresh(site, self._auth_episode_counts[site], self.cfg):
            # __file__ = p79/experiment/runner/main.py → repo root needs 4 .parent
            # (runner → experiment → p79 → REPO_ROOT). Bug fix 2026-04-26.
            auth_dir = Path(__file__).resolve().parent.parent.parent.parent / ".auth"
            benchmark = self.cfg.get("experiment", {}).get("benchmark", "")
            ok = refresh_site_auth(site, auth_dir, benchmark=benchmark)
            if ok:
                self._auth_episode_counts[site] = 0
                logger.info("Auth refreshed for %s", site)
            else:
                logger.warning("Auth refresh failed for %s — continuing with stale session", site)

        episode_dir = condition_dir / "artifacts" / f"{task.site}_task_{task.task_id}"
        if episode_dir.exists():
            shutil.rmtree(episode_dir)
            logger.info("Cleared stale artifacts for %s task %s", task.site, task.task_id)
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Clear stale JSONL from previous (interrupted) run
        stale_jsonl = condition_logger.step_log_path(task.site, task.task_id)
        if stale_jsonl.exists():
            stale_jsonl.unlink()
            logger.info("Cleared stale step JSONL for %s task %s", task.site, task.task_id)

        obs, info = self.environment.reset(task.config_file)
        current_info = info or {}

        # ── Start-URL tab health check ──────────────────────────────────
        _error_title_patterns = (
            "content not found", "not found", "404", "page not found",
            "osclass error",  # classifieds DB unavailable
            "500 internal server error",  # server-side crash (e.g. Postmill)
        )
        tab_titles = self.environment.get_all_tab_titles()
        for _tab_url, _tab_title in tab_titles:
            if any(pat in (_tab_title or "").lower() for pat in _error_title_patterns):
                raise RuntimeError(
                    f"start_url_content_error: tab title='{_tab_title}' url={_tab_url}"
                )

        trajectory: List[Any] = [{"observation": getattr(obs, "raw", None), "info": current_info}]

        # Load task reference images (e.g. "find this item" with a product photo).
        # All modes receive reference images — DOM mode encodes them as base64 in
        # the text prompt so the model can still reason about the target item.
        reference_images: list = []
        raw_image = task.raw_task.get("image")
        if raw_image:
            from PIL import Image as PILImage
            paths = [raw_image] if isinstance(raw_image, str) else list(raw_image)
            # __file__ now lives in p79/experiment/runner/main.py — go up 4 levels
            # to reach the repo root (was 3 levels in old runner.py).
            vwa_root = Path(__file__).resolve().parent.parent.parent.parent / "external" / "visualwebarena"
            # Pre-resize reference images once at episode load (Step-5 ref image
            # cache): saves N_steps × resize_per_step. Agent's per-step resize
            # check (`if max(ref_img.size) > max_size`) becomes a no-op when
            # incoming image is already within bounds.
            ref_max_size = int(self.cfg.get("agent", {}).get("image_max_size", 1024))
            for p in paths:
                img_path = vwa_root / p
                if img_path.exists():
                    try:
                        img = PILImage.open(str(img_path)).convert("RGB")
                        if max(img.size) > ref_max_size:
                            ratio = ref_max_size / max(img.size)
                            img = img.resize(
                                (int(img.size[0] * ratio), int(img.size[1] * ratio)),
                                PILImage.Resampling.LANCZOS,
                            )
                        reference_images.append(img)
                    except Exception as exc:
                        logger.warning("Failed to load reference image %s: %s", img_path, exc)
                else:
                    logger.warning("Reference image not found: %s", img_path)
            if reference_images:
                logger.info(
                    "Loaded %d reference image(s) for site=%s task=%s (pre-resized to <= %dpx)",
                    len(reference_images), task.site, task.task_id, ref_max_size,
                )

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
        # B-11/17/18 Cluster 3 fix: fuzzy signature track catches semantic loops
        # where strict/soft signatures miss because text varies (search-loop
        # with rephrased queries, click-loop on different element_ids of same
        # role on same URL).
        action_signatures_fuzzy: List[str] = []
        scroll_direction_history: List[str] = []
        url_stuck_streak = 0
        last_url = ""
        cycle_early_stop = False

        checklist_manager: Optional[ChecklistManagerLite] = None
        if bool(self.checklist_cfg.get("enabled", False)):
            checklist_manager = ChecklistManagerLite(
                task_description=task.intent,
                max_items=int(self.checklist_cfg.get("max_items", 4)),
            )
        latest_checklist_status = checklist_manager.get_status() if checklist_manager is not None else None

        similarity_threshold = float(self.state_change_cfg.get("similarity_threshold", 0.95))
        busy_wait_limit = int(self.cfg.get("runtime", {}).get("busy_wait_limit", 5))

        step_idx = 0
        consecutive_busy_waits = 0
        total_busy_waits = 0
        # Track total wall time spent in free busy-page waits (RU-4): these
        # don't consume a step but still count toward end-to-end episode time.
        busy_wait_total_ms = 0.0
        while step_idx < self.max_steps:
            step_start = time.time()

            # ── Early busy-page guard ─────────────────────────────────────
            # If the DOM is still loading (busy marker = 1), skip the LLM call
            # entirely and issue a free wait that does NOT consume a step from
            # the budget. Tolerate whitespace variations ("busy: 1" / "busy:1").
            obs_text_raw = getattr(obs, "text", "") or ""
            _busy_marker = bool(re.search(r"\bbusy\s*:\s*1\b", obs_text_raw))
            if _busy_marker and consecutive_busy_waits < busy_wait_limit:
                consecutive_busy_waits += 1
                total_busy_waits += 1
                _busy_start = time.time()
                wait_action = {"action_type": "wait"}
                next_obs, reward, terminated, truncated, next_info = self.environment.step(wait_action)
                _busy_elapsed_ms = (time.time() - _busy_start) * 1000.0
                busy_wait_total_ms += _busy_elapsed_ms
                logger.info(
                    "busy:1 free wait #%d (total %d, %.0fms) site=%s task=%s (step_idx=%d not consumed)",
                    consecutive_busy_waits, total_busy_waits, _busy_elapsed_ms,
                    task.site, task.task_id, step_idx,
                )
                obs = next_obs
                current_info = next_info if isinstance(next_info, dict) else {}
                if terminated or truncated:
                    break
                continue  # step_idx NOT incremented
            consecutive_busy_waits = 0  # reset consecutive counter on non-busy page

            artifacts = self._save_artifacts(episode_dir, step_idx, obs)

            # Use the preferred mode to compute router's size signal; on escalation
            # we re-prepare below with the actual decided mode.
            obs_prepare_start = time.time()
            _size_probe = prepare_observation_for_mode(obs, condition.observation_mode, episode_dir, step_idx)
            router_obs_text = _size_probe.som_text if condition.observation_mode != "vision" else (obs.text or "")

            decision_mode, triggers, overhead, router_state = self.router.decide(
                router_enabled=condition.router_on,
                preferred_mode=condition.observation_mode,
                obs_text=router_obs_text,
                state=router_state,
                prev_action_success=prev_action_success,
                prev_page_changed=prev_page_changed,
                checklist_status=latest_checklist_status,
            )
            trigger_distribution.update(triggers)

            if condition.router_on and decision_mode != condition.observation_mode:
                escalation_count += 1

            screenshot_prep_start = time.time()
            if decision_mode == condition.observation_mode:
                obs_prep = _size_probe
            else:
                obs_prep = prepare_observation_for_mode(obs, decision_mode, episode_dir, step_idx)
            artifacts["som_image"] = obs_prep.marked_image_path
            if decision_mode == "som" and obs_prep.som_text:
                _step_dir = episode_dir / f"step_{step_idx:03d}"
                _step_dir.mkdir(parents=True, exist_ok=True)
                try:
                    (_step_dir / "observation_som.txt").write_text(obs_prep.som_text, encoding="utf-8")
                except Exception:
                    pass
            obs_for_backend = self._clone_observation_for_mode(obs, decision_mode, obs_prep)
            obs_prepare_ms = (time.time() - obs_prepare_start) * 1000.0
            if condition.router_on and decision_mode != condition.observation_mode:
                overhead["extra_screenshot_ms"] = (time.time() - screenshot_prep_start) * 1000.0
            instruction = task.intent
            if checklist_manager and bool(self.checklist_cfg.get("inject_into_prompt", True)):
                instruction = f"{task.intent}\n\n{checklist_manager.format_for_prompt()}"

            # History window: configurable (cfg.agent.history_window),
            # default 8. Was hardcoded 8 pre-§97 Step-5.
            _history_window = int(self.cfg.get("agent", {}).get("history_window", 8))
            context = BackendStepContext(
                observation_mode=decision_mode,
                som_enabled=(decision_mode == "som"),
                som_text=obs_prep.som_text,
                stage="single",
                history=step_records[-_history_window:],
                module_flags=condition.modules.as_dict(),
                reference_images=reference_images,
            )

            planner_meta = {}
            planner_sub_goal: Optional[str] = None
            if condition.modules.m4_two_stage_generation_grounding:
                planner_context = BackendStepContext(
                    observation_mode=decision_mode,
                    som_enabled=(decision_mode == "som"),
                    som_text=obs_prep.som_text,
                    stage="planner",
                    history=step_records[-_history_window:],
                    module_flags=condition.modules.as_dict(),
                    reference_images=reference_images,
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

            action, _ = validate_action(action)
            action = apply_secondary_modules(action, obs.text or "", condition.modules.as_dict())
            if bool(self.diagnostic_controls.get("enabled", False)):
                diag_notes: List[str] = []
                query_cfg = self.diagnostic_controls.get("query_sanitization", {}) or {}
                anti_repeat_cfg = self.diagnostic_controls.get("anti_repeat", {}) or {}
                no_early_finish_cfg = self.diagnostic_controls.get("no_early_finish", {}) or {}

                if bool(query_cfg.get("enabled", False)):
                    action, note = _query_sanitization_control(action, query_cfg)
                    if note:
                        diag_notes.append(note)
                if bool(anti_repeat_cfg.get("enabled", False)):
                    action, note = _anti_repeat_control(
                        action=action,
                        step_records=step_records,
                        obs_text=obs.text or "",
                        instruction=task.intent,
                        cfg=anti_repeat_cfg,
                        query_cfg=query_cfg,
                    )
                    if note:
                        diag_notes.append(note)
                if bool(no_early_finish_cfg.get("enabled", False)):
                    action, note = _no_early_finish_control(
                        action=action,
                        step_records=step_records,
                        obs_text=obs.text or "",
                        instruction=task.intent,
                        cfg=no_early_finish_cfg,
                        query_cfg=query_cfg,
                    )
                    if note:
                        diag_notes.append(note)
                action, _ = validate_action(action)
                if diag_notes:
                    logger.info(
                        "Diagnostic controls applied site=%s task=%s step=%d notes=%s action=%s",
                        task.site,
                        task.task_id,
                        step_idx,
                        ";".join(diag_notes),
                        action,
                    )
            if self.state_change_cfg.get("form_snapshot_enabled", True):
                form_before = self.environment.snapshot_form_fields()
            else:
                form_before = None
            state_before = build_page_state(obs, current_info, form_snapshot=form_before)

            # Feed page complexity signals to router state for 3-way decisions
            router_state.dom_complexity_history.append(state_before.get("dom_complexity", 0))
            router_state.text_length_history.append(state_before.get("text_length", 0))
            if len(router_state.dom_complexity_history) > self.router.history_window:
                router_state.dom_complexity_history = router_state.dom_complexity_history[-self.router.history_window:]
                router_state.text_length_history = router_state.text_length_history[-self.router.history_window:]

            env_step_start = time.time()
            next_obs, reward, terminated, truncated, next_info = self.environment.step(action)
            env_step_ms = (time.time() - env_step_start) * 1000.0

            action_type_lower = str(action.get("action_type", "")).lower()
            if self.state_change_cfg.get("form_snapshot_enabled", True):
                form_after = self.environment.snapshot_form_fields()
            else:
                form_after = None
            state_after = build_page_state(next_obs, next_info, form_snapshot=form_after)

            # --- about:blank recovery ---
            about_blank_recovered = False
            post_url = state_after.get("url", "")
            if post_url.startswith("about:blank") or (not post_url and action_type_lower == "back"):
                recovery_url = task.raw_task.get("start_url", "")
                if recovery_url:
                    try:
                        next_obs, _, _, _, next_info = self.environment.navigate_to(recovery_url)
                        if self.state_change_cfg.get("form_snapshot_enabled", True):
                            form_after = self.environment.snapshot_form_fields()
                        else:
                            form_after = None
                        state_after = build_page_state(next_obs, next_info, form_snapshot=form_after)
                        about_blank_recovered = True
                        logger.warning(
                            "about:blank detected at step %d for task %s/%d — recovered to %s",
                            step_idx, task.site, task.task_id, recovery_url,
                        )
                    except Exception as exc:
                        logger.warning(
                            "about:blank recovery failed at step %d for task %s/%d: %s",
                            step_idx, task.site, task.task_id, exc,
                        )

            # ── Mid-episode site infrastructure check ──────────────────
            _INFRA_TITLE_PATTERNS = ("osclass error",)
            _title_after_lower = (state_after.get("title") or "").lower()
            if any(pat in _title_after_lower for pat in _INFRA_TITLE_PATTERNS):
                raise RuntimeError(
                    f"site_infra_error: title='{state_after.get('title')}' "
                    f"detected at step {step_idx} for task {task.site}/{task.task_id}"
                )

            action_success, page_change_reasons, text_similarity = detect_page_state_change(
                state_before=state_before,
                state_after=state_after,
                action_type=str(action.get("action_type", "")).upper(),
                similarity_threshold=similarity_threshold,
            )
            if about_blank_recovered:
                page_change_reasons.append("about_blank_recovery")
            page_changed = bool(page_change_reasons)
            # Do not use reward as action-success evidence: evaluator rewards can be noisy
            # and may mask real no-progress execution failures.
            if terminated and action_type_lower in ("finish", "stop", "done"):
                action_success = True

            retry_count = 0
            # NOTE: retry_limit only gates the trigger condition for the single
            # retry attempt below — there is no while-loop around the retry
            # block, so values >1 do NOT enable multiple consecutive retries.
            # If multi-retry is needed, the block at line ~1177 must become a loop.
            retry_limit = int(self.cfg.get("router", {}).get("thresholds", {}).get("retry_limit", 1))
            trigger_m3_retry = should_trigger_m3_retry(
                action_success=action_success,
                page_changed=page_changed,
                retry_count=retry_count,
                retry_limit=retry_limit,
                module_flags=condition.modules.as_dict(),
            )
            # Baseline robustness: if click/type made no progress, run one internal retry
            # even when optional M3 module is disabled.
            baseline_retry_on_no_progress = bool(
                self.cfg.get("runtime", {}).get("baseline_retry_on_no_progress", False)
            )
            trigger_baseline_retry = (
                baseline_retry_on_no_progress
                and (not trigger_m3_retry)
                and (not page_changed)
                and action_type_lower in ("click", "type")
                and retry_count < retry_limit
            )
            retry_was_applied = False
            retry_action_type_str: Optional[str] = None
            if trigger_m3_retry or trigger_baseline_retry:
                retry_action = m3_retry_action(failed_action=action, obs_text=obs.text or "")
                # Common bookkeeping (always incremented regardless of retry outcome)
                retry_count += 1
                retry_total += 1
                overhead["routing_retry_count"] += 1.0
                retry_was_applied = True
                retry_action_type_str = str(retry_action.get("action_type", "")).lower()
                try:
                    retry_obs, retry_reward, retry_term, retry_trunc, retry_info = self.environment.step(retry_action)
                except Exception as retry_exc:
                    # Retry env.step failed — preserve original obs/reward/terminated state
                    logger.warning(
                        "M3 retry env.step failed at step %d for task %s/%d: %s — keeping original state",
                        step_idx, task.site, task.task_id, retry_exc,
                    )
                else:
                    if self.state_change_cfg.get("form_snapshot_enabled", True):
                        retry_form = self.environment.snapshot_form_fields()
                    else:
                        retry_form = None
                    retry_state_after = build_page_state(retry_obs, retry_info, form_snapshot=retry_form)
                    retry_success, retry_reasons, retry_similarity = detect_page_state_change(
                        state_before=state_before,
                        state_after=retry_state_after,
                        action_type=str(retry_action.get("action_type", "")).upper(),
                        similarity_threshold=similarity_threshold,
                    )
                    retry_action_type = str(retry_action.get("action_type", "")).lower()
                    retry_action_type_str = retry_action_type
                    retry_success = bool(
                        retry_success or (retry_term and retry_action_type in ("finish", "stop", "done"))
                    )

                    # Retry step is executed in the real environment, so always adopt
                    # post-retry state to keep wrapper state and recorded observation aligned.
                    next_obs, reward, terminated, truncated, next_info = (
                        retry_obs,
                        retry_reward,
                        retry_term,
                        retry_trunc,
                        retry_info,
                    )
                    state_after = retry_state_after
                    text_similarity = retry_similarity
                    page_changed = bool(retry_reasons)
                    action_success = retry_success
                    if retry_reasons:
                        retry_tag = (
                            "m3_retry_applied" if trigger_m3_retry else "baseline_no_progress_retry_applied"
                        )
                        page_change_reasons = list(dict.fromkeys(list(retry_reasons) + [retry_tag]))
                    else:
                        page_change_reasons = []

            safe_next_info = next_info if isinstance(next_info, dict) else {}
            is_stop_action = str(action.get("action_type", "")).lower() in ("finish", "stop", "done")
            if "raw_action" in safe_next_info:
                trajectory.append(safe_next_info["raw_action"])
            # VWA evaluator expects trajectory to end with the stop action (not a trailing
            # observation).  Only append the post-step observation for non-terminal actions.
            if not is_stop_action:
                trajectory.append({"observation": getattr(next_obs, "raw", None), "info": safe_next_info})

            input_tokens = int(meta.get("input_tokens") or 0)
            output_tokens = int(meta.get("output_tokens") or 0)
            token_total = input_tokens + output_tokens

            token_cost = compute_token_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_cfg=select_token_cost_cfg(
                    metrics_cfg=self.cfg.get("metrics", {}),
                    backend_type=self.cfg.get("backends", {}).get(condition.backend_id, {}).get("type"),
                ),
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
            # Fold obs_prepare CPU cost into unified cost scalar (for router decisions)
            obs_prepare_cost = obs_prepare_ms * float(router_cfg.get("overhead_cost_per_ms", 0.0))
            step_total_cost = token_cost["total"] + router_overhead_cost + obs_prepare_cost
            failure_reason = meta.get("failure_reason")
            parse_valid = bool(meta.get("valid", True))
            fallback_finish = (
                action_type_lower == "finish"
                and (not parse_valid)
                and str(failure_reason or "").strip().lower() == "keyword_finish"
            )
            if fallback_finish:
                logger.warning(
                    "Fallback finish detected site=%s task=%s step=%d reason=%s",
                    task.site,
                    task.task_id,
                    step_idx,
                    failure_reason,
                )
            error_category = self._normalize_error_category(
                failure_reason=failure_reason if not parse_valid else None,
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
                    "enabled": (decision_mode == "som"),
                    "degraded_som": bool(obs_prep.degraded_som),
                    "mark_count": obs_prep.mark_count,
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
                    "obs_prepare": obs_prepare_ms,
                    "preprocessing": float(meta.get("preprocess_ms", 0.0)),
                    "generate": float(meta.get("generate_ms", 0.0)),
                    "backend_infer": float(meta.get("infer_ms", backend_latency_ms)),
                    "env_step": env_step_ms,
                    "router_decision": float(overhead.get("router_decision_ms", 0.0)),
                },
                tokens={
                    "input": input_tokens,
                    "input_text": int(meta.get("input_text_tokens", 0)),
                    "input_image": int(meta.get("input_image_tokens", 0)),
                    "output": output_tokens,
                    "total": token_total,
                    "thinking": meta.get("thinking_tokens"),
                },
                cost_usd={
                    "input": token_cost["input"],
                    "output": token_cost["output"],
                    "model": token_cost["total"],
                    "router_overhead": router_overhead_cost,
                    "obs_prepare": obs_prepare_cost,
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
                # B-09 Cluster 2 fix: agent_visible_changed excludes runner-
                # internal reasons (form_value/dom_complexity/text_length/
                # interactive_elements/form_fields). Use this for SR derivation
                # downstream; page_changed retains 12-union for cycle/retry.
                agent_visible_changed=is_agent_visible_change(page_change_reasons),
                state_digest={
                    "url_before": state_before.get("url"),
                    "url_after": state_after.get("url"),
                    "title_before": state_before.get("title"),
                    "title_after": state_after.get("title"),
                    "dom_complexity": state_before.get("dom_complexity"),
                    "text_length": state_before.get("text_length"),
                    "scroll_y_before": state_before.get("scroll_y"),
                    "scroll_y_after": state_after.get("scroll_y"),
                    "form_fields_changed": "form_value_changed" in page_change_reasons,
                },
            ).as_dict()
            # Convenience field for history rendering/debug.
            step_record["obs_url"] = state_after.get("url")
            # Optional parser/debug fields (non-required schema extras).
            step_record["parse_valid"] = parse_valid
            step_record["parse_failure_reason"] = (
                str(failure_reason) if failure_reason is not None else None
            )
            step_record["fallback_finish"] = fallback_finish
            step_record["retry_action_applied"] = retry_was_applied
            step_record["retry_action_type"] = retry_action_type_str
            # GLM fallback tracking (§67 Plan B)
            if meta.get("glm_fallback_used"):
                step_record["glm_fallback_used"] = True
                step_record["glm_fallback_latency_ms"] = meta.get("glm_fallback_latency_ms")
                step_record["glm_original_fail_reason"] = meta.get("glm_original_fail_reason")
            # Confidence metrics (optional, from logprobs extraction)
            if meta.get("mean_logprob") is not None:
                step_record["confidence"] = {
                    "mean_logprob": meta["mean_logprob"],
                    "min_logprob": meta["min_logprob"],
                    "mean_margin": meta["mean_margin"],
                    "min_margin": meta["min_margin"],
                }
                # Entropy metrics (added later, optional)
                if meta.get("mean_entropy") is not None:
                    step_record["confidence"]["mean_entropy"] = meta["mean_entropy"]
                    step_record["confidence"]["max_entropy"] = meta["max_entropy"]
            # Verbalized confidence (from agent JSON output)
            verbalized_conf = action.get("confidence")
            if verbalized_conf is not None:
                try:
                    verbalized_conf = float(verbalized_conf)
                    verbalized_conf = max(0.0, min(1.0, verbalized_conf))
                except (ValueError, TypeError):
                    verbalized_conf = None
                if verbalized_conf is not None:
                    if not isinstance(step_record.get("confidence"), dict):
                        step_record["confidence"] = {}
                    step_record["confidence"]["verbalized"] = verbalized_conf

            # Element bounding box for annotation overlay (from obs_nodes_info)
            eid = action.get("element_id")
            if eid is not None and hasattr(obs, "obs_nodes_info") and obs.obs_nodes_info:
                node_info = obs.obs_nodes_info.get(str(eid))
                if isinstance(node_info, dict):
                    ub = node_info.get("union_bound")
                    if ub and len(ub) == 4:
                        step_record["element_bbox"] = [float(v) for v in ub]

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
            # Skip strict signature for scroll when page actually changed —
            # scrolling down multiple times is normal browsing, not a cycle.
            is_scroll = str(action.get("action_type", "")).lower() == "scroll"
            if not (is_scroll and page_changed):
                action_signatures.append(_action_signature(action))
            # Only accumulate soft signatures when the page didn't change,
            # otherwise reset — clicking different elements that each cause
            # real navigation is not a cycle.
            if not page_changed:
                action_signatures_soft.append(_action_signature_soft(action))
            else:
                action_signatures_soft.clear()
            # B-11/17/18 fix: fuzzy signature track. Reset when page changes
            # (different URL/title/visible-text means agent navigated, not stuck).
            # Use agent_visible_changed if available; fallback to page_changed.
            avis_changed = is_agent_visible_change(page_change_reasons)
            if not avis_changed:
                action_signatures_fuzzy.append(
                    _action_signature_fuzzy(action, obs_url=str(state_after.get("url", "")))
                )
            else:
                action_signatures_fuzzy.clear()
            cycle_len = _detect_action_cycle(action_signatures)
            # Soft check uses higher reps threshold to reduce false positives
            soft_cycle_len = _detect_action_cycle(action_signatures_soft, min_reps=4)
            # Fuzzy check uses even higher threshold (5 reps) — most aggressive
            # collapse catches search-loop / click-loop with text-variation.
            fuzzy_cycle_len = _detect_action_cycle(action_signatures_fuzzy, min_reps=5)
            if cycle_len > 0 or soft_cycle_len > 0 or fuzzy_cycle_len > 0:
                if cycle_len > 0:
                    detected, mode, min_r = cycle_len, "strict", 3
                elif soft_cycle_len > 0:
                    detected, mode, min_r = soft_cycle_len, "soft", 4
                else:
                    detected, mode, min_r = fuzzy_cycle_len, "fuzzy", 5
                logger.warning(
                    "Action cycle detected (%s, len=%d, reps>=%d) at step %d for task %s/%d — early stop.",
                    mode, detected, min_r, step_idx, task.site, task.task_id,
                )
                cycle_early_stop = True
                break

            # --- scroll alternation detection ---
            if is_scroll:
                delta_y = 0.0
                raw_delta = action.get("delta")
                if isinstance(raw_delta, (list, tuple)) and len(raw_delta) >= 2:
                    delta_y = float(raw_delta[1])
                direction = "down" if delta_y >= 0 else "up"
                scroll_direction_history.append(direction)
                ALT_THRESHOLD = 6
                if len(scroll_direction_history) >= ALT_THRESHOLD:
                    tail = scroll_direction_history[-ALT_THRESHOLD:]
                    is_alternating = all(tail[i] != tail[i + 1] for i in range(ALT_THRESHOLD - 1))
                    if is_alternating:
                        logger.warning(
                            "Scroll alternation detected (len=%d) at step %d for task %s/%d — early stop.",
                            ALT_THRESHOLD, step_idx, task.site, task.task_id,
                        )
                        cycle_early_stop = True
                        break
            else:
                scroll_direction_history.clear()

            # --- URL stuck detection ---
            current_url = state_after.get("url", "")
            if current_url and current_url == last_url and action_type_lower == "click":
                url_stuck_streak += 1
            else:
                url_stuck_streak = 0
            last_url = current_url

            URL_STUCK_THRESHOLD = 5
            if url_stuck_streak >= URL_STUCK_THRESHOLD:
                logger.warning(
                    "URL stuck (%d consecutive clicks, url=%s) at step %d for task %s/%d — early stop.",
                    url_stuck_streak, current_url[:80], step_idx, task.site, task.task_id,
                )
                cycle_early_stop = True
                break

            step_idx += 1

        # VWA evaluator expects trajectory to end with an Action dict having "answer" key.
        # When the agent never stopped (max-steps / cycle), trajectory ends with an obs dict
        # which lacks "answer", causing KeyError: 'answer'.  Append a fake stop action.
        if not trajectory or not isinstance(trajectory[-1], dict) or "answer" not in trajectory[-1]:
            try:
                from browser_env.actions import create_stop_action  # type: ignore
                trajectory.append(create_stop_action(""))
            except Exception:
                import numpy as np
                trajectory.append({
                    "action_type": 6,  # ActionTypes.STOP
                    "coords": np.zeros(2, dtype=np.float32),
                    "element_role": 0, "element_name": "", "text": [],
                    "page_number": 0, "url": "", "nth": 0,
                    "pw_code": "", "element_id": "", "key_comb": "",
                    "direction": "", "answer": "", "raw_prediction": "",
                })

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
            logger.warning(
                "Reward override: evaluator=0 overridden to 1.0 (env reward>0, agent finished) site=%s task=%s",
                task.site, task.task_id,
            )

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
        total_obs_prepare_cost = sum(float(s["cost_usd"].get("obs_prepare", 0.0)) for s in step_records)
        total_cost = total_model_cost + total_router_overhead_cost + total_obs_prepare_cost
        total_router_overhead_ms = sum(
            float(s["router"].get("overhead_ms", {}).get("router_decision_ms", 0.0))
            + float(s["router"].get("overhead_ms", {}).get("extra_dom_parse_ms", 0.0))
            + float(s["router"].get("overhead_ms", {}).get("extra_screenshot_ms", 0.0))
            for s in step_records
        )

        no_op_count = sum(1 for s in step_records if not bool(s.get("action_success", False)))
        # Exclude finish/stop steps from unchanged_count: they intentionally
        # don't change the page (they end the episode), so counting them
        # inflates page_unchanged_rate. Aligns with reason_diagnostics
        # _compute_step_cost_breakdown which already excludes these.
        unchanged_count = sum(
            1 for s in step_records
            if not bool(s.get("page_changed", False))
            and str((s.get("action") or {}).get("action_type", "")).lower() not in ("finish", "stop")
        )

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

        # Enrich with wasted cost and component breakdown for cost-aware analysis
        wasted = compute_wasted_cost(step_records, success)
        episode_summary["wasted_cost_usd"] = wasted["wasted_cost_usd"]
        episode_summary["wasted_energy_kwh"] = wasted["wasted_energy_kwh"]
        breakdown = compute_component_breakdown(step_records)
        episode_summary["component_breakdown"] = breakdown
        # Track how many free busy-page waits were issued (not counted as steps)
        episode_summary["busy_wait_free_steps"] = total_busy_waits
        # Total wall time spent in busy-wait stalls (RU-4): not counted in
        # total_latency_ms (which sums step_records latencies), exposed
        # separately so end-to-end episode time = total_latency_ms + busy_wait_total_ms.
        episode_summary["busy_wait_total_ms"] = busy_wait_total_ms
        # Energy completeness diagnostics (RU-5): some NVML probes can fail
        # mid-episode; report partial flag + complete-step count so downstream
        # can decide whether to compare energy fairly across conditions.
        episode_summary["energy_step_complete_count"] = len(energy_vals)
        episode_summary["energy_partial"] = bool(
            step_records and len(energy_vals) < len(step_records)
        )
        # Input/output cost breakdown for fine-grained cost analysis
        episode_summary["total_input_cost_usd"] = sum(
            float(s["cost_usd"].get("input", 0.0)) for s in step_records
        )
        episode_summary["total_output_cost_usd"] = sum(
            float(s["cost_usd"].get("output", 0.0)) for s in step_records
        )
        episode_summary["total_obs_prepare_cost_usd"] = total_obs_prepare_cost

        # Agent-initiated finish flag (for N/A true positive detection)
        _last_at = str((step_records[-1].get("action") or {}).get("action_type", "")).lower() if step_records else ""
        _last_fb = bool(step_records[-1].get("fallback_finish", False)) if step_records else False
        _agent_finished = (_last_at in ("finish", "stop")) and not _last_fb
        episode_summary["agent_finished"] = _agent_finished

        # §95 adjusted_success — compute here as single source of truth.
        # Downstream analysis scripts read this field directly instead of
        # re-deriving it (was scattered across 5 locations pre-§97 audit).
        try:
            from p79.experiment.analysis import compute_adjusted_success
            _eval_types = (
                (task.raw_task.get("eval") or {}).get("eval_types") or []
                if hasattr(task, "raw_task") and isinstance(task.raw_task, dict)
                else []
            )
            _eval_type_str = "|".join(str(x) for x in _eval_types) if _eval_types else ""
            _has_eff = any(
                str((s.get("action") or {}).get("action_type", "")).lower() in ("type", "select_option")
                for s in step_records
            )
            _na_ids = self._na_ids_cache.get(task.site, set())
            _adj, _fp = compute_adjusted_success(
                task.task_id, task.site, success,
                na_task_ids=_na_ids,
                agent_finished=_agent_finished,
                eval_type=_eval_type_str,
                has_effective_action=_has_eff,
            )
            episode_summary["adjusted_success"] = bool(_adj)
            episode_summary["fp_reason"] = str(_fp)
            episode_summary["has_effective_action"] = bool(_has_eff)
        except Exception as _adj_exc:
            logger.warning(
                "Failed to compute adjusted_success for site=%s task=%s: %s",
                task.site, task.task_id, _adj_exc,
            )
            episode_summary.setdefault("adjusted_success", None)
            episode_summary.setdefault("fp_reason", "")
            episode_summary.setdefault("has_effective_action", False)

        return episode_summary
