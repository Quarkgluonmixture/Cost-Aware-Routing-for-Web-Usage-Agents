"""ExperimentRunner — main class extracted from runner.py during §97 Step-3 split.

Free helper functions live in `helpers.py`; this module hosts only the
ExperimentRunner class. The original `from p79.experiment.runner import
ExperimentRunner` import path is preserved via `runner/__init__.py`.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import random
import re
import shutil
import signal
import sys
import time
import urllib.request
import urllib.error
from collections import Counter
from datetime import datetime, timezone
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
from p79.experiment.environment import (
    create_environment,
    create_evaluator,
    PaperGradeAbortError,
)
from p79.experiment.io_utils import read_jsonl_dedup
from p79.experiment.logger_v2 import LoggerV2
from p79.experiment.metrics import (
    aggregate_condition_metrics,
    classify_step_accounting,
    compute_component_breakdown,
    compute_router_overhead_cost,
    compute_three_column_cost,
    compute_token_cost,
    compute_wasted_cost,
    detect_benchmark_noise,
    classify_timeout,
    net_saving,
    p95,
    select_token_cost_cfg,
)
# B-425 (/stress A1.3 v9 D1, 2026-05-17): p79/experiment/modules.py retired
# (M1/M2 select+input fallback + M3 retry + M4 two-stage 都 0/53924 archive
# usage). condition.modules.as_dict() still preserved on the schema for
# backward compat; runner just no longer dispatches the M1-M3 helpers.
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
    _action_intent_fulfilled,
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
    _notify_transient_retry,
    push_run_abort_ntfy,
)

logger = logging.getLogger(__name__)


def _compute_effective_paper_grade(cfg: Optional[Dict[str, Any]] = None) -> bool:
    """B-1602 (/stress 深入审 Mode A P1-3-A*, 2026-05-18): canonical
    paper_grade effective-bool source for the WHOLE runner.

    Pre-existing B-868 unification (/stress A1.23, 2026-05-17) created MAX-of
    (yaml ∨ env) logic at `_compute_resume_fingerprint` (L692-694) so resume
    fingerprint always records the strictest interpretation. BUT the sibling
    consumer `_seed_global_rng` (deterministic CUDA flags, L125 pre-fix) read
    `P79_PAPER_GRADE` env-only — yaml-only paper_grade users got LAX
    `torch.use_deterministic_algorithms(warn_only=True)` while evaluator
    + diagnostic_controls + backend_cfg propagation all went STRICT. Paper
    §3.5 "byte-identical hidden states" claim silently breaks under that
    misconfig. B-868's own comment block at L688-690 describes the exact bug
    but fix only touched fingerprint side.

    Now: canonical helper used by both `_seed_global_rng` AND
    `_compute_resume_fingerprint` so all paper_grade consumers share one
    source of truth.

    Returns True if EITHER (a) `cfg["paper_grade"]` truthy OR (b)
    `P79_PAPER_GRADE` env truthy ("1" / "true" / "yes" / "on" / case-insens).
    `cfg=None` (e.g., legacy callers without cfg context) → env-only.
    """
    _pg_env_raw = os.environ.get("P79_PAPER_GRADE", "").strip().lower()
    _pg_env = _pg_env_raw in ("1", "true", "yes", "on")
    _pg_yaml = bool((cfg or {}).get("paper_grade", False)) if cfg else False
    return _pg_yaml or _pg_env


def _seed_global_rng(seed: int, paper_grade_effective: Optional[bool] = None) -> None:
    """B-37 + B-827 (/stress A1.16 cold-start P1-1-BC*, 2026-05-17): propagate
    seed to Python/NumPy/torch RNG AND enforce deterministic CUDA flags.

    Called at start of each (condition, seed) iteration. Without this, seed=42
    is metadata only — Python random.choice / np.random.shuffle / torch ops
    produce different results across runs. Paper-grade reproducibility claim
    requires this propagation.

    B-827 enforcement (A1.16 cold-start P1-1-BC*):
    Pre-fix only `random/np/torch.manual_seed` were set. Paper §3.5 + prereg §7
    line 545 explicitly claim "byte-identical action traces, hidden states, and
    aggregate SR" — but A100 `nn.Embedding` backward, some `Conv2D`, and scatter
    ops are non-deterministic without `torch.use_deterministic_algorithms(True)`
    + `torch.backends.cudnn.deterministic=True` + `CUBLAS_WORKSPACE_CONFIG`.
    Cross-host hidden-state replay (paper-2 mechanism §5) was already failing
    silently; paper-1 within-run byte-identical claim equally untrue.

    Strict-vs-warn split:
      - P79_PAPER_GRADE=1 → `warn_only=False` (RuntimeError on non-deterministic
        op; surfaces real reproducibility violation rather than masking)
      - dev mode → `warn_only=True` (logs warning, continues — Phase 1a smoke
        without paper-grade gate)
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

        # B-827 P1-1-BC*: deterministic CUDA flags. Must be set BEFORE any
        # tensor op runs in this process (CUBLAS_WORKSPACE_CONFIG is read at
        # first CUBLAS call; deterministic_algorithms is checked per-op).
        # B-1602 (/stress 深入审 Mode A P1-3-A*, 2026-05-18): accept the
        # caller-resolved paper_grade_effective bool (yaml ∨ env unified via
        # `_compute_effective_paper_grade`). Pre-B-1602 this line read
        # env-only — yaml-true + env-false misconfig got LAX warn_only=True
        # silently breaking paper §3.5 reproducibility. Legacy callers
        # without paper_grade_effective fall back to env-only via the helper.
        if paper_grade_effective is None:
            paper_grade = _compute_effective_paper_grade(cfg=None)
        else:
            paper_grade = bool(paper_grade_effective)
        # Set CUBLAS env var first (idempotent — same value on every seed call)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=not paper_grade)
        except Exception as det_err:
            logger.warning(
                "torch.use_deterministic_algorithms(warn_only=%s) failed: %s — "
                "continuing without strict deterministic op gate",
                not paper_grade,
                det_err,
            )
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception as cudnn_err:
            logger.warning(
                "torch.backends.cudnn.deterministic=True failed: %s", cudnn_err
            )
    except ImportError:
        pass


class ExperimentRunner:
    # Fatal environment errors that corrupt Playwright/asyncio state.
    # When caught, re-raise immediately so the process can exit cleanly.
    # B-501 (/stress A1.5b Phase 1 P2-1-A, 2026-05-17): replace tuple of
    # substring patterns with explicit regex. Pre-fix `"asyncio loop"` as
    # plain substring would false-positive match benign log content like
    # "Reusing existing asyncio loop" → kill recoverable run. Now: explicit
    # phrases known to indicate fatal state.
    _FATAL_ENV_REGEX = re.compile(
        r"Sync API inside the asyncio|"
        r"asyncio loop is closed|"
        r"asyncio loop was destroyed|"
        r"Event loop is closed"
    )

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        # P0-1-AB (/stress GRL audit 2026-05-20, user Q1=A): paper_grade XOR
        # diagnostic_replay hard block — fires BEFORE any setup so a config
        # contradiction can never produce a single artifact (placed ahead of the
        # B-1407 mock-env check + the diagnostic_controls / baseline_retry
        # hard-fails below; same "paper_grade forbids X" family). diagnostic_replay
        # routes output to the non-canonical results/diagnostic_replay/ tree,
        # stamps every episode sr_excluded=True, AND suppresses the M1
        # PaperGradeAbortError (`and not self.diagnostic_replay`). A leaked
        # P79_DIAGNOSTIC_REPLAY=1 env would therefore silently turn a paper-grade
        # fire into zero-canonical-data + no fail-closed abort — operator believes
        # canonical, produces nothing in SR. Only queue_diagnostic_replay.sh
        # (no paper_grade) may enable diagnostic replay; the queue layer has a
        # mirror env reject for defense-in-depth.
        if bool(cfg.get("paper_grade", False)) and bool(cfg.get("diagnostic_replay", False)):
            raise RuntimeError(
                "paper_grade=True forbids diagnostic_replay=True. Diagnostic "
                "replay output is non-canonical (results/diagnostic_replay/) + "
                "sr_excluded=True on every episode + suppresses the M1 "
                "fail-closed quarantine abort — a paper-grade fire under this "
                "flag silently produces zero canonical SR data and loses the "
                "fail-closed guard. Unset P79_DIAGNOSTIC_REPLAY / "
                "diagnostic_replay for a canonical fire, or remove paper_grade "
                "and use queue_diagnostic_replay.sh for a diagnostic run."
            )
        self.output_root = resolve_output_root(cfg)
        self.phase = str(cfg["experiment"]["phase"]).lower()
        self.seeds = _parse_seeds(cfg["experiment"]["seed"])
        self.seed = self.seeds[0]
        self.max_steps = int(cfg.get("runtime", {}).get("max_steps", 40))
        # Protocol Reset #7 (§244 canonical, 2026-05-20): two-budget accounting.
        # PRIMARY budget = max_agent_actions (only valid, budget-consuming steps;
        # restores upstream "30 agent decisions" semantics). Inherits max_steps so
        # existing per-condition yamls (max_steps: 30) need no edit. SAFETY budget
        # bounds runaway episodes: max_model_attempts caps total LLM calls; the
        # parse-error caps terminate when the agent cannot produce usable actions.
        _rt = cfg.get("runtime", {})
        self.max_agent_actions = int(_rt.get("max_agent_actions", self.max_steps))
        self.max_total_parse_errors = int(_rt.get("max_total_parse_errors", 5))
        self.max_consecutive_parse_errors = int(_rt.get("max_consecutive_parse_errors", 3))
        self.max_model_attempts = int(
            _rt.get("max_model_attempts", self.max_agent_actions + self.max_total_parse_errors + 10)
        )
        self.resume = bool(cfg.get("runtime", {}).get("resume", True))

        self.conditions = generate_conditions(cfg)
        self.tasks = load_tasks(cfg, self.output_root)

        self.router = RuleBasedRouter(cfg)
        env_cfg = dict(cfg.get("env", {}))
        env_cfg.setdefault("benchmark", cfg["experiment"]["benchmark"])
        self.environment = create_environment(env_cfg)
        # B-544 (/stress A1.5b Phase 2 P0-4-B codex OOB, 2026-05-17): paper_grade
        # propagation so VwaEvaluator fail-loud on dep failure. Pre-fix evaluator
        # silently returned score=0 → entire batch SR zeroed under infra failure
        # (cross-baseline infra-fragility confound); now paper_grade=True raises
        # EvaluatorUnavailableError → caller writes needs_reevaluation=True
        # summary (B-486 quarantine semantics).
        self.evaluator = create_evaluator(
            env_cfg, paper_grade=bool(cfg.get("paper_grade", False)),
        )
        self.checklist_cfg = cfg.get("checklist", {})
        self.state_change_cfg = cfg.get("state_change", {})
        self.energy_tracker = LightweightEnergyTracker(cfg.get("metrics", {}).get("energy", {}))
        self.diagnostic_controls = cfg.get("diagnostic_controls", {}) or {}
        # Fire-6 RCA Stage C2 (/stress 2026-05-20): diagnostic-replay mode.
        # When True (set by --diagnostic-replay CLI flag / queue wrapper +
        # config.normalize), every episode is stamped sr_excluded=True (the
        # canonical SR firewall) and the M1 quarantine fail-closed abort
        # (~line 1821) is SUPPRESSED so a targeted replay runs ALL requested
        # tasks (e.g. 4 AND 75), capturing per-task forensics instead of
        # aborting at the first quarantine. NEVER set on a canonical fire —
        # output also routes to non-canonical results/diagnostic_replay/.
        self.diagnostic_replay = bool(cfg.get("diagnostic_replay", False))
        # B-486 (/stress A1.25 GRL Chunk 3 P0-2-C* gemini OOB, 2026-05-17):
        # paper_grade hard-block on diagnostic action controls (mirror B-340
        # GLM fallback hard-block pattern). Pre-fix `_anti_repeat_control` +
        # `_no_early_finish_control` could force search/scroll if agent
        # loops or finishes early; Phase 1a configs set
        # `diagnostic_controls.enabled: false` but any debug run accidentally
        # toggling would mix Runner-guided "rescue" successes with agent
        # autonomy → entire trajectory autonomy claim breaks under
        # peer review even on a single fire. Paper §3.5.1 also disclosed
        # post-A1.25 Chunk 3 (B-486 prose). Closes reviewer-attack vector
        # "code exists, why not disclose / why allow at all in paper-grade".
        if bool(cfg.get("paper_grade", False)) and bool(self.diagnostic_controls.get("enabled", False)):
            raise RuntimeError(
                "paper_grade=True forbids diagnostic_controls.enabled=True "
                "(anti_repeat + no_early_finish would force Runner-guided "
                "action injection — Paper §1 trajectory autonomy claim "
                "becomes unverifiable). Set diagnostic_controls.enabled: "
                "false in this run's yaml, or remove paper_grade from the "
                "queue script if this is intentionally a diagnostic run. "
                "See B-486 disclosure in paper §3.5.1."
            )
        # P0-6-AC (/stress Phase 0 unified bug list 2026-05-19, Claude+Gemini
        # 2-AI overlap): sibling pattern to diagnostic_controls hard-fail above.
        # `baseline_retry_on_no_progress=True` injects runner-side scroll/refresh
        # action after agent's failed click/type — same trajectory-autonomy
        # contamination as diagnostic_controls. Default False but a yaml
        # override / dev-iteration residue → paper-grade fire silently corrupted.
        # Cross-baseline asymmetry: B1 (faster steps, more attempts) triggers
        # no_progress more than B0 (slow steps) → asymmetric retry-injection
        # rate. Paper §3 cross-baseline step semantics requires this fail-closed.
        if bool(cfg.get("paper_grade", False)) and bool(
            cfg.get("runtime", {}).get("baseline_retry_on_no_progress", False)
        ):
            raise RuntimeError(
                "paper_grade=True forbids baseline_retry_on_no_progress=True "
                "(runner-side scroll/refresh injection after agent failed "
                "click/type — same trajectory autonomy violation as "
                "diagnostic_controls.enabled=True). Set "
                "runtime.baseline_retry_on_no_progress: false in this run's "
                "yaml, or remove paper_grade from the queue script for "
                "intentionally diagnostic runs. See B-486 disclosure path."
            )

        # B-144 (/stress A1.2 v8 codex B1, 2026-05-16): cache key is
        # ``(backend_id, seed)`` — previously a single ``backend_id`` key froze
        # the first seed into the agent cfg at construction, so seed-loop
        # updates to ``self.seed`` (main.py:333-334) never propagated to the
        # cached agent. Multi-seed runs (``experiment.seed: [42, 43, ...]``)
        # produced distinct ``condition_meta.seed`` rows but identical
        # model-side seed → mislabeled same-seed duplicates. Per-seed cache
        # means each seed switch reconstructs the agent with the correct
        # backend_cfg["seed"]; for local backends this re-imports the model
        # (one extra HF load per seed, paper-grade necessary).
        self._backends: Dict[Tuple[str, int], Any] = {}
        self._auth_episode_counts: Dict[str, int] = {}  # per-site counter for auth refresh
        # B-35 fix (笔记 §116.9): also track last refresh timestamp for time-based threshold
        self._auth_last_refresh_ts: Dict[str, float] = {}
        # §139.8: the per-site N/A task IDs cache was removed — it only fed the
        # retired post-hoc `compute_adjusted_success` call in `_run_episode`.
        # N/A tasks are now excluded at load time (`tasks.py::load_tasks`,
        # `task.exclude_na_tasks`), so the runner never sees them.

    def _get_backend(self, backend_id: str):
        # B-144 (/stress A1.2 v8 codex B1, 2026-05-16): cache key includes seed
        # so seed-loop reconstructs the agent on switch. See ``__init__`` for
        # the multi-seed reproducibility rationale.
        cache_key = (backend_id, int(self.seed))
        if cache_key in self._backends:
            return self._backends[cache_key]

        backend_cfg = self.cfg.get("backends", {}).get(backend_id)
        if not backend_cfg:
            raise KeyError(f"Backend {backend_id} is not defined in config.backends")
        # B-37 fix: inject experiment seed into backend cfg for downstream
        # propagation to LLM payload (proxy `seed` param) and torch generation.
        # Uses self.seed which was set per (condition, seed) pair in run().
        #
        # B-164 (/stress A1.4a v8 codex B5, 2026-05-16): deep copy not shallow.
        # Previously ``dict(backend_cfg)`` shared nested dicts/lists (e.g.
        # ``generation``, ``model_kwargs``, ``headers``, ``safety``) with
        # ``self.cfg``, so any agent constructor side effect that mutated a
        # nested key (defaults injection, header normalization) leaked into
        # subsequent (condition, seed) iterations through ``self.cfg``. Symptom:
        # seed=42 single-run ≠ seed=[42,43] runs[0] even though backend cache
        # key tuple was correct (B-144). ``copy.deepcopy`` isolates each
        # call.
        backend_cfg = copy.deepcopy(backend_cfg)
        if backend_cfg.get("seed") is None:
            backend_cfg["seed"] = int(self.seed)
        # B-83 fix: forward top-level `model.revision` into the backend cfg.
        # The runner passes ONLY the backend sub-config to create_backend, so
        # the `model:` block in exp_v2_base.yaml never reached the agent — the
        # local_qwen wrapper had no `revision` key and qwen3vl_agent silently
        # used its hard-coded default (merged config decoupled from loaded SHA).
        #
        # /stress A1.1 codex Mode B (2026-05-15) — B-131 fix: api_proxy
        # (B0 235B) MUST NOT inherit the top-level `model.revision` SHA,
        # which by historical convention pins B1's local Qwen 4B revision
        # (ebb281ec... — recorded in configs/exp_v2_base.yaml). Injecting a
        # 4B SHA into B0's `condition_meta.json` is paper-grade reproducibility
        # theatre: the proxy serves a 235B model whose provider snapshot is
        # unrelated to the HF 4B revision. Codex F2 attack (post-B-115 sibling
        # propagation): the prior fix dropped `local_gemma` from the injection
        # list, but missed `api_proxy` — same disease, same partial sweep.
        #
        # B0 provider provenance: TODO (separate `provider_snapshot_id` /
        # `api_model_version` field, requires proxy API support) — out of
        # B-131 scope, tracked in master_bug_catalog A1.1 F2 follow-up.
        _QWEN_CLASS_BACKEND_TYPES = {"local_qwen"}
        _model_revision = self.cfg.get("model", {}).get("revision")
        _backend_type = backend_cfg.get("type")
        if (
            _model_revision
            and backend_cfg.get("revision") is None
            and _backend_type in _QWEN_CLASS_BACKEND_TYPES
        ):
            backend_cfg["revision"] = _model_revision
        # B-340 (/stress A1.9 Mode C F4 defense-in-depth, 2026-05-16):
        # propagate top-level `paper_grade` flag into backend cfg so
        # api_proxy → ProxyApiAgent hard-block on use_glm_fallback can fire.
        backend_cfg.setdefault("paper_grade", bool(self.cfg.get("paper_grade", False)))
        backend = create_backend(backend_id, backend_cfg)
        self._backends[cache_key] = backend
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
        # B-489 (/stress A1.5b Phase 1 P1-1-AB 2-AI overlap, 2026-05-17):
        # atomic write via shared helper (B-331 lineage). Pre-fix plain
        # `open + json.dump` — sibling-propagation gap from B-331 batch
        # (B-331 added atomic for `run_summary_v2.json` last-write but
        # missed `run_meta.json` first-write). DGX crash mid-write →
        # truncated JSON → resume + analyze_experiment.py readers fail.
        from p79.experiment.logger_v2 import write_run_summary_atomic
        write_run_summary_atomic(self.output_root / "run_meta.json", payload)

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
            # B-502 (/stress A1.5b Phase 1 P2-2-A, 2026-05-17): preserve any
            # non-empty tree. Pre-fix `file_count > 5` threshold allowed dirs
            # with 1-5 files (e.g. resumed run with run_meta + condition_meta
            # + 2 summaries + 1 step jsonl = 5 files exactly) to be deleted
            # if has_run_meta check was bypassed by manual mv / rename. Now:
            # any non-zero file count → preserve. Only truly empty trees
            # become eligible.
            if file_count > 0:
                continue
            logger.info("Cleaning stale empty run dir (files=%d, age>1h): %s", file_count, run_dir)
            try:
                shutil.rmtree(run_dir)
            except OSError as exc:
                logger.warning("Failed to remove stale run dir %s: %s", run_dir, exc)

    def _create_latest_symlink(self) -> None:
        """Create a latest_{site} symlink in the phase directory pointing to this run.

        B-499 (/stress A1.5b Phase 1 P1-11-A, 2026-05-17): multi-site runs
        previously fell through to a no-suffix `latest` symlink — two
        concurrent multi-site runs would overwrite each other's symlink and
        readers couldn't tell which run was "latest". Now: multi-site uses
        sorted-joined site names (e.g. `latest_classifieds_reddit`), single-
        site preserved as `latest_{site}`. Zero-site raises explicitly
        (config bug, surface fast).
        """
        sites = self.cfg.get("task", {}).get("include_sites", [])
        if not sites:
            logger.warning(
                "B-499 _create_latest_symlink: cfg.task.include_sites is empty — "
                "skipping symlink creation (config bug, no site context to anchor)."
            )
            return
        if len(sites) == 1:
            link_name = f"latest_{sites[0]}"
        else:
            link_name = "latest_" + "_".join(sorted(str(s) for s in sites))
        latest_link = self.output_root.parent / link_name
        try:
            latest_link.unlink(missing_ok=True)
            latest_link.symlink_to(self.output_root.name)
            logger.info("Created symlink %s -> %s", latest_link, self.output_root.name)
        except OSError as exc:
            logger.warning("Failed to create latest symlink %s: %s", latest_link, exc)

    @staticmethod
    def _aggregate_partial_steps(
        partial_steps: List[Dict[str, Any]],
        condition_observation_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """B-168 (/stress A1.4a v8 codex B1, 2026-05-16) + B-490 (A1.5b
        Phase 1 P1-2-AC, 2026-05-17): aggregate metrics from partial JSONL
        rows so a mid-episode crash doesn't erase the tokens/cost/latency
        already incurred + correctly recompute escalation_count.

        Pre-B-168 the runner ``except`` path emitted ``steps=0, total_tokens=0,
        total_cost=0`` even after 12 step JSONL rows were already on disk,
        creating a JSONL-vs-summary divergence: same episode wrote two
        incompatible histories. ``read_jsonl_dedup`` already handles restart
        dedup and corrupt-line skipping; this helper computes the sums.

        B-490 (Claude A1.5b + gemini G3, A+C 2-AI overlap): pre-fix returned
        ``escalation_count: 0`` with cop-out comment "cannot reconstruct
        without state context". But each ``StepRecordV2.observation_mode``
        is a REQUIRED field per ``types.py:331`` schema; given
        ``condition.observation_mode`` (caller passes it) we can recompute
        escalation = sum(steps where mode != condition.mode). Pre-fix
        DGX-crash-rate × baseline-escalation-rate biased paper §4 router
        fire-rate headline downward — reviewer attack vector "your router
        fire-rate excludes crash-contaminated runs".

        Returns a dict of {steps, retries, total_tokens, total_*_cost_usd,
        total_latency_ms, p95_step_latency_ms, escalation_count, ...}.
        """
        if not partial_steps:
            return {
                "steps": 0, "retries": 0,
                "no_op_rate": 0.0, "page_unchanged_rate": 0.0,
                "total_latency_ms": 0.0, "p95_step_latency_ms": 0.0,
                # B-1600 (/stress 深入审 Mode A P0-1-A*, 2026-05-18): empty-partial
                # default mirrors `total_latency_ms` shape for consumer parity.
                "total_latency_minus_retry_ms": 0.0,
                "total_tokens": 0, "total_model_cost_usd": 0.0,
                "total_cost_usd": 0.0,
                "total_router_overhead_cost_usd": 0.0,
                "total_router_overhead_ms": 0.0,
                "escalation_count": 0,
                # P1-1 (/stress accounting audit 2026-05-21): two-budget counters +
                # three-column cost for the crashed-before-any-step case (0 steps ⇒
                # 0 cost is the honest value, mirrors compute_three_column_cost([])).
                "agent_action_step_count": 0,
                "valid_action_step_count": 0,
                "model_call_attempt_count": 0,
                "runner_iteration_count": 0,
                "parse_error_injected_wait_count": 0,
                "total_billed_cost_usd": 0.0,
                "canonical_action_cost_usd": 0.0,
                "protocol_wasted_cost_usd": 0.0,
            }
        from p79.experiment.metrics import p95 as _p95
        n = len(partial_steps)
        step_latencies = [float(s.get("latency_ms", {}).get("total", 0.0)) for s in partial_steps]
        total_latency = sum(step_latencies)
        # B-1600 (/stress 深入审 Mode A P0-1-A*, 2026-05-18): retry-adjusted
        # partial rollup mirrors success-path rollup at L3136-3145; falls back
        # to `total` when step record lacks `total_minus_retry`.
        total_latency_minus_retry_partial = sum(
            float(s.get("latency_ms", {}).get("total_minus_retry", s.get("latency_ms", {}).get("total", 0.0)))
            for s in partial_steps
        )
        total_tokens = sum(int(s.get("tokens", {}).get("total", 0)) for s in partial_steps)
        total_model_cost = sum(float(s.get("cost_usd", {}).get("model", 0.0)) for s in partial_steps)
        total_router_overhead_cost = sum(
            float(s.get("cost_usd", {}).get("router_overhead", 0.0)) for s in partial_steps
        )
        total_obs_prepare_cost = sum(
            float(s.get("cost_usd", {}).get("obs_prepare", 0.0)) for s in partial_steps
        )
        total_cost = total_model_cost + total_router_overhead_cost + total_obs_prepare_cost
        total_router_overhead_ms = sum(
            float(s.get("router", {}).get("overhead_ms", {}).get("router_decision_ms", 0.0))
            + float(s.get("router", {}).get("overhead_ms", {}).get("extra_dom_parse_ms", 0.0))
            + float(s.get("router", {}).get("overhead_ms", {}).get("extra_screenshot_ms", 0.0))
            for s in partial_steps
        )
        retries = sum(int(s.get("retry_count", 0)) for s in partial_steps)
        no_op_count = sum(1 for s in partial_steps if not bool(s.get("action_success", False)))
        unchanged_count = sum(
            1 for s in partial_steps
            if not bool(s.get("page_changed", False))
            and str((s.get("action") or {}).get("action_type", "")).lower() not in ("finish", "stop")
        )
        # B-490 (/stress A1.5b Phase 1 P1-2-AC, 2026-05-17): recompute
        # escalation_count from partial_steps using condition.observation_mode
        # as router-resting target. step.observation_mode is REQUIRED field
        # per StepRecordV2 schema (types.py:331). If caller didn't pass
        # condition_observation_mode (legacy callers), fall back to 0 with
        # WARN log so downstream paper §4 aggregator can distinguish from a
        # true-zero count.
        if condition_observation_mode is not None:
            escalation_count = sum(
                1 for s in partial_steps
                if str(s.get("observation_mode", "")) != condition_observation_mode
            )
        else:
            logger.warning(
                "B-490 _aggregate_partial_steps called without "
                "condition_observation_mode — escalation_count stays 0 (legacy fallback)"
            )
            escalation_count = 0
        # P1-1 (/stress accounting audit 2026-05-21, codex Mode B): recompute the
        # Protocol Reset two-budget counters + three-column cost from partial step
        # flags so a mid-episode crash does NOT emit steps>0 / total_model_cost>0
        # alongside counters=0 / cost=None (silent counter drift on the exact
        # failure path the audit cares about). runner_iteration_count = n here
        # (busy-waits aren't persisted as JSONL steps so the partial reader cannot
        # recover them; n is the recoverable floor).
        partial_agent_action = sum(1 for s in partial_steps if s.get("consumes_agent_action_budget") is True)
        partial_valid_action = sum(1 for s in partial_steps if s.get("valid_agent_action") is True)
        partial_parse_error_sink = sum(
            1 for s in partial_steps
            if s.get("valid_agent_action") is False and s.get("consumes_agent_action_budget") is False
        )
        partial_3col = compute_three_column_cost(partial_steps)
        # B-1798 (P1-1, /stress 2026-05-21): roll up cost-basis on the exception
        # path too so a crash episode carries the same step→episode basis chain as
        # the happy path (else the condition aggregator sees None on crash cohorts).
        _partial_bases = [s.get("cost_unit_basis") for s in partial_steps if s.get("cost_unit_basis")]
        partial_cost_unit_basis = (
            max(set(_partial_bases), key=_partial_bases.count) if _partial_bases else None
        )
        partial_mixed_warn = (
            any(bool(s.get("cost_total_mixed_unit_warn")) for s in partial_steps)
            if partial_steps else None
        )
        return {
            "steps": n,
            "retries": retries,
            "no_op_rate": (no_op_count / n) if n else 0.0,
            "page_unchanged_rate": (unchanged_count / n) if n else 0.0,
            "total_latency_ms": total_latency,
            "total_latency_minus_retry_ms": total_latency_minus_retry_partial,
            "p95_step_latency_ms": _p95(step_latencies),
            "total_tokens": total_tokens,
            "total_model_cost_usd": total_model_cost,
            "total_cost_usd": total_cost,
            "total_router_overhead_cost_usd": total_router_overhead_cost,
            # B-1799 (P1-5, /stress 2026-05-21 codex Mode B): return obs_prepare so
            # the exception-path component_breakdown (main.py ~1872 reads
            # _agg.get("total_obs_prepare_cost_usd")) closes to total_cost. Pre-fix
            # it was computed (L611-613) + folded into total_cost but NOT returned →
            # silent .get(...,0.0) → failed/partial episode breakdown under-counted
            # obs_prepare → total_cost > model+router+obs_prepare (component non-closure).
            "total_obs_prepare_cost_usd": total_obs_prepare_cost,
            "total_router_overhead_ms": total_router_overhead_ms,
            "escalation_count": escalation_count,
            "agent_action_step_count": partial_agent_action,
            "valid_action_step_count": partial_valid_action,
            "model_call_attempt_count": n,
            "runner_iteration_count": n,
            "parse_error_injected_wait_count": partial_parse_error_sink,
            "total_billed_cost_usd": partial_3col["total_billed_cost_usd"],
            "canonical_action_cost_usd": partial_3col["canonical_action_cost_usd"],
            "protocol_wasted_cost_usd": partial_3col["protocol_wasted_cost_usd"],
            "cost_unit_basis": partial_cost_unit_basis,
            "cost_total_mixed_unit_warn": partial_mixed_warn,
        }

    @staticmethod
    def _validate_resume_identity(
        loaded: Dict[str, Any],
        expected: Dict[str, Any],
    ) -> Optional[Dict[str, Tuple[Any, Any]]]:
        """B-169 (/stress A1.4a v8 codex B2, 2026-05-16): validate that a
        loaded summary on disk actually belongs to the current run.

        Pre-B-169 the resume gate accepted ANY file at the expected path —
        if output_root was reused with changed ``run_id``/``seed``/
        ``include_sites``/``max_tasks_per_site``, stale summaries silently
        ingested into the new aggregate. Identity tuple:
        ``(schema_version, run_id, condition_id, seed, benchmark_site, task_id)``.

        Returns:
            None if identity matches (accept resume), or
            dict of mismatched_field → (loaded_value, expected_value).
        """
        mismatches: Dict[str, Tuple[Any, Any]] = {}
        for field, expected_value in expected.items():
            loaded_value = loaded.get(field)
            if loaded_value != expected_value:
                mismatches[field] = (loaded_value, expected_value)
        return mismatches if mismatches else None

    @staticmethod
    def _normalize_error_category(
        failure_reason: Optional[str],
        action_success: bool,
        page_changed: bool,
        env_error: Optional[str] = None,
        agent_visible_changed: Optional[bool] = None,
    ) -> Optional[str]:
        """B-167 (/stress A1.4a v8 Claude F3 expanded scope, 2026-05-16):
        expanded from 5 categories to 10 + unknown_failure bucket.

        Pre-B-167, any failure_reason not matching parse/keyword OR
        timeout/network keywords fell through to "invalid_action" catch-all,
        silently contaminating paper §3.5 cross-baseline error taxonomy.
        Now distinguishes structural validation sub-categories emitted by
        ``validate_action_detailed`` (`p79/backends/action_utils.py`):

        - invalid_action_type: agent emitted unknown action_type
        - invalid_element_id: click/type/select missing element_id and coord
        - invalid_coord: malformed coord/delta (NaN, wrong shape, out of range)
        - invalid_select_option: select_option without option label/value/index
        - invalid_schema: structural dict-shape gap (non-dict, page_number, etc.)
        - runner_invalid_action: backend reported valid but runner rescued (B-134)
        - parse_error: backend JSON parse failed (with or without keyword rescue)
        - env_error: playwright/network/timeout failure
        - benchmark_noise: known noisy VWA error patterns
        - no_progress: action structurally succeeded but page didn't change
        - unknown_failure: future-proof catch-all (was silent invalid_action)

        Router-aware escalation policy (per-category target mode mapping)
        deferred to Phase 2 / paper-2 scope. Paper §3.5 disclosure added.
        """
        if env_error:
            is_noise, _ = detect_benchmark_noise(env_error)
            return "benchmark_noise" if is_noise else "env_error"

        reason = str(failure_reason or "").strip().lower()
        if reason:
            # Order: specific structural reasons before generic keyword tokens
            if "runner_invalid_action" in reason:
                return "runner_invalid_action"
            # Parse-layer failures (backend-side JSON / keyword rescue)
            if any(k in reason for k in (
                "parse_failed", "multiple_actions",
                "repaired_fenced", "repaired_raw_decode",
                "repaired_multiple_identical", "repaired_regex",
                "keyword_", "json",
            )):
                return "parse_error"
            # Env-layer failures
            if any(k in reason for k in (
                "timeout", "playwright", "browser",
                "connection", "network", "env_error",
            )):
                return "env_error"
            # Structural sub-categories from validate_action_detailed
            if reason == "invalid_action_type":
                return "invalid_action_type"
            if reason == "invalid_element_id":
                return "invalid_element_id"
            if reason == "invalid_coord":
                return "invalid_coord"
            if reason == "invalid_select_option":
                return "invalid_select_option"
            if reason in ("invalid_schema_dict", "invalid_schema"):
                return "invalid_schema"
            # Legacy catch (Path 2b raw_decode candidates all invalid)
            if reason == "invalid_action_repaired":
                return "invalid_schema"
            # Bare "invalid_action" string from legacy callers
            if reason == "invalid_action":
                return "invalid_schema"
            return "unknown_failure"

        # B-505 (/stress A1.5b Phase 1 P2-5-C gemini, 2026-05-17): prefer
        # agent_visible_changed signal over global page_changed when caller
        # provides it. Pre-fix `not page_changed` was polluted by B-09
        # invisible-change reasons (form_value_changed, dom_complexity_changed,
        # text_length_changed) that fire on background DOM mutation without
        # any user-visible state change. `no_progress` denominator was thus
        # diluted by silent successful form-edits. Now: when caller passes
        # agent_visible_changed (StepRecordV2.agent_visible_changed field per
        # B-09 fix), use it; legacy callers fall back to page_changed.
        _visible_changed = (
            agent_visible_changed if agent_visible_changed is not None
            else page_changed
        )
        if not action_success and not _visible_changed:
            return "no_progress"
        return None

    @staticmethod
    def _compute_resume_fingerprint(
        cfg: Dict[str, Any],
        condition: ConditionSpec,
    ) -> str:
        """B-485 (/stress A1.5b Phase 1 P0-1-ABC 3-AI overlap, 2026-05-17):
        compute sha256[:16] resume identity fingerprint.

        Pre-fix `_validate_resume_identity` 6-tuple (run_id/condition_id/
        seed/site/task_id/schema_version) was path identity only — didn't
        catch experiment-identity drift (model SHA changed, prompt template
        edited, max_new_tokens / temperature mutated, transformers upgraded).
        3-AI A1.5b audit converged on same finding from 3 angles (Claude:
        model_revision + prompt_hash; codex: backend_id + transformers +
        paper_grade + observation_mode; gemini: max_steps + max_new_tokens
        + temperature).

        Fingerprint embeds the union of those critical fields. Mismatch
        between current cfg's fingerprint and a loaded summary's stored
        fingerprint → quarantine + re-run (paper-grade rerun protocol).
        Legacy summaries lack `resume_fingerprint` → loaded.get returns
        None → mismatches as expected (Phase 1a 还没 fire; from-scratch
        rerun is the plan).
        """
        backend_id = condition.backend_id
        backend_cfg = cfg.get("backends", {}).get(backend_id, {}) or {}
        generation_cfg = backend_cfg.get("generation", {}) or {}
        # B-868 (/stress A1.23 P1-12 C, 2026-05-17): paper_grade UNIFICATION.
        # Pre-fix two sources of truth disagreed:
        #   (a) `cfg.get("paper_grade", False)` — yaml-driven, used here at 676
        #       and at line 185, 202, 1459 (env evaluator init + diagnostic
        #       controls + write_episode_summary fail-loud).
        #   (b) `os.environ.get("P79_PAPER_GRADE", "0") == "1"` — env-driven,
        #       used at line 124 (_seed_global_rng deterministic flags) +
        #       queue scripts default-on.
        # An operator could `P79_PAPER_GRADE=0 bash ...` (dirty/dev mode)
        # while the yaml still has `paper_grade: true` → fingerprint records
        # paper_grade=True but actual deterministic-RNG flags went warn_only.
        # Resume of that run with `P79_PAPER_GRADE=1` would match the
        # fingerprint (both yaml + env say True) and ingest the dirty steps
        # under paper-grade banner.
        # Fix: take MAX of yaml + env, i.e., paper_grade = True if EITHER
        # source asserts True. Mismatch between yaml and env in the original
        # run → fingerprint records True; resume requires env to assert True
        # too → if operator forgot to set env, fingerprint mismatch → forced
        # quarantine + rerun.
        _pg_yaml = bool(cfg.get("paper_grade", False))
        _pg_env = (os.environ.get("P79_PAPER_GRADE", "0") == "1")
        _pg_effective = _pg_yaml or _pg_env
        components: Dict[str, Any] = {
            "model_revision": cfg.get("model", {}).get("revision"),
            "backend_revision": backend_cfg.get("revision"),
            "max_new_tokens": generation_cfg.get("max_new_tokens"),
            "temperature": generation_cfg.get("temperature"),
            "paper_grade": _pg_effective,
            "paper_grade_yaml": _pg_yaml,
            "paper_grade_env": _pg_env,
            "observation_mode": condition.observation_mode,
            "max_steps": int(cfg.get("runtime", {}).get("max_steps", 40)),
            # P2-4 (/stress accounting audit 2026-05-21): Protocol Reset two-budget
            # knobs are experiment-identity-affecting (changing the agent-action
            # budget or parse caps = a different experiment). max_steps alone did
            # NOT capture an explicit max_agent_actions override that diverges from
            # max_steps → stale-resume could mix episodes run under different budgets.
            "max_agent_actions": int(cfg.get("runtime", {}).get("max_agent_actions", cfg.get("runtime", {}).get("max_steps", 40))),
            "max_model_attempts": cfg.get("runtime", {}).get("max_model_attempts"),
            "max_consecutive_parse_errors": cfg.get("runtime", {}).get("max_consecutive_parse_errors"),
            "max_total_parse_errors": cfg.get("runtime", {}).get("max_total_parse_errors"),
        }
        # transformers version — soft import; absent = best-effort sentinel
        try:
            import transformers  # type: ignore[import]
            components["transformers_version"] = str(getattr(transformers, "__version__", "unknown"))
        except Exception:
            components["transformers_version"] = "unavailable"
        # prompt_hash from canonical VL agent prompts (shared across B0/B1/B2)
        try:
            from p79.agents._shared_vl_utils import make_dom_prompt, make_som_prompt
            prompt_text = make_dom_prompt() + make_som_prompt()
            components["prompt_hash"] = hashlib.sha256(
                prompt_text.encode("utf-8")
            ).hexdigest()[:12]
        except Exception:
            components["prompt_hash"] = "unavailable"
        encoded = json.dumps(components, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def run(self) -> Path:
        # B-500 (/stress A1.5b Phase 1 P1-12-B codex OOB, 2026-05-17):
        # try/finally guard around the main loop. Pre-fix `environment.close()`
        # + `energy_tracker.close()` only fired on normal return — any fatal
        # exception inside the condition×seed loop leaked browser /
        # Playwright / energy probe state. Queue restart then inherited
        # dirty external site state (cart / auth / browser context). The
        # main loop body is extracted to `_run_main_loop` so the close
        # methods can fire in this method's finally block regardless of
        # exception path. close methods themselves wrapped in inner
        # try/except so close-failure doesn't mask the original exception.
        #
        # B-859 (/stress A1.23 P0-2 AB* OOB, 2026-05-17): SIGTERM handler to
        # honor B-500's finally: contract under OS-level signal abort. Pre-fix
        # runner had NO signal handlers → Python default SIGTERM disposition =
        # SIG_DFL = immediate process termination → finally: NEVER runs.
        # queue_chain.sh:102 `pkill -f "run_experiment.py.*${pattern}"` on
        # watchdog-death abort path sends default SIGTERM → environment.close()
        # + energy_tracker.close() NOT executed → Playwright browser + CDP
        # socket + Chromium subprocess leak → next chain iteration inherits
        # dirty external state. Now: SIGTERM raises KeyboardInterrupt →
        # Python exception machinery → finally: block fires → resources close.
        # SIGINT is already raised as KeyboardInterrupt by Python default.
        # SIGKILL / OOM cannot be intercepted (kernel-level); residual gap
        # disclosed in paper §3.5. User decision /stress A1.23 (Q2=A): SIGTERM
        # handler only; Option B runtime singleton lease dropped for Phase 1a
        # fire timing (paper-2 forward stub).
        def _on_sigterm(signum, _frame):  # pragma: no cover — signal-driven
            raise KeyboardInterrupt(
                f"SIGTERM received (signal={signum}) — converting for graceful finally (B-859)"
            )
        _prev_sigterm = signal.signal(signal.SIGTERM, _on_sigterm)

        self._cleanup_stale_runs()
        self._write_run_meta()
        try:
            return self._run_main_loop()
        finally:
            try:
                self.environment.close()
            except Exception as _env_close_exc:
                logger.warning("B-500 environment.close() failed: %s", _env_close_exc)
            try:
                self.energy_tracker.close()
            except Exception as _energy_close_exc:
                logger.warning("B-500 energy_tracker.close() failed: %s", _energy_close_exc)
            # B-859: restore prior SIGTERM disposition so callers (tests, REPL,
            # other ExperimentRunner instances in same process) aren't affected
            # by this run's handler. Best-effort — restoring fails only if
            # signal module is unavailable (which would have aborted setup).
            try:
                signal.signal(signal.SIGTERM, _prev_sigterm)
            except Exception:
                pass

    def _run_main_loop(self) -> Path:
        run_condition_metrics: List[Dict[str, Any]] = []

        for condition in self.conditions:
            for current_seed in self.seeds:
                self.seed = current_seed
                # B-37 fix: propagate seed to Python/NumPy/torch RNG so seed=42 is
                # actually deterministic, not just metadata. Per (condition, seed)
                # pair so each condition gets fresh RNG state from the same seed.
                # B-1602 (/stress 深入审 Mode A P1-3-A*, 2026-05-18): pass
                # caller-resolved paper_grade_effective so deterministic CUDA
                # flags strict-vs-warn split honors yaml ∨ env (mirror B-868
                # fingerprint unification at L741). Pre-B-1602 the function
                # read env-only — yaml-true + env-false misconfig silently
                # broke paper §3.5 "byte-identical hidden states" claim while
                # evaluator + diagnostic_controls + backend_cfg propagation
                # all went STRICT.
                _seed_global_rng(
                    current_seed,
                    paper_grade_effective=_compute_effective_paper_grade(self.cfg),
                )
                seed_suffix = f"_seed{current_seed}" if len(self.seeds) > 1 else ""
                effective_cid = f"{condition.condition_id}{seed_suffix}"

                condition_dir = self.output_root / effective_cid
                condition_dir.mkdir(parents=True, exist_ok=True)
                condition_logger = LoggerV2(condition_dir)

                # B-1645 (/stress A2.10 P1-6-A 2026-05-18): reset learned-router
                # diag state per (condition, seed) so cumulative `_lr_*` counters
                # don't leak across cells. Pre-fix `self._lr_fallback_count` was
                # initialized lazily on first fallback and never reset; in a
                # multi-condition runner.run() invocation, the second cell's
                # `learned_router_diag` would carry the prior cell's count.
                self._lr_dispatch_count = 0
                self._lr_fallback_count = 0
                self._lr_fallback_ntfy_fired = False
                self._lr_router_cache = {}  # cell-scoped artifact cache
                cond_meta = condition.as_dict()
                cond_meta["condition_id"] = effective_cid
                cond_meta["seed"] = current_seed
                condition_logger.write_condition_meta(cond_meta)

                # B-388 (A1.15 C2 Merge (i), 2026-05-16): Option K staging pickup.
                # Reset gate (`_lib_paper_grade_gates.sh:reset_and_auth_gate`)
                # writes `reset_post_interrupt` events to
                # `logs/trajectory_events_staging/RUN_${RUN_ID}.jsonl` because
                # condition_dir doesn't exist yet at reset time. Now that
                # condition_dir is created we pick up + merge any staging
                # events for this RUN_ID. Idempotent via "fresh dir only"
                # guard inside `merge_staging_trajectory_events` (resume case
                # keeps prior events). Best-effort: any merge failure
                # surfaces a warning but does not abort the runner.
                try:
                    from p79.experiment.logger_v2 import merge_staging_trajectory_events
                    _staging_run_id = str(self.cfg["experiment"].get("run_id", "")).strip()
                    if _staging_run_id:
                        _merged = merge_staging_trajectory_events(
                            condition_dir=condition_dir,
                            run_id=_staging_run_id,
                        )
                        if _merged:
                            print(
                                f"[runner][trajectory-events] merged {_merged} "
                                f"staging events into {effective_cid}/trajectory_events.jsonl"
                            )
                    elif bool(self.cfg.get("paper_grade", False)):
                        # B-503 (/stress A1.5b Phase 1 P2-3-A, 2026-05-17):
                        # paper-grade fail-loud on empty run_id. preregistration.md
                        # Appendix A claims trajectory_events.jsonl ALWAYS contains
                        # reset events — falsifiability requires non-empty run_id
                        # so staging file path is constructable.
                        raise ValueError(
                            "B-503: cfg.experiment.run_id is empty/blank — "
                            "paper-grade requires non-empty run_id for trajectory "
                            "events staging merge. preregistration.md Appendix A "
                            "trajectory_events claim falsifiability depends on it."
                        )
                    else:
                        # Dev mode: surface the silent no-op as a warning.
                        logger.warning(
                            "B-503 cfg.experiment.run_id is empty/blank — "
                            "skipping trajectory events staging merge."
                        )
                except Exception as _trajectory_merge_exc:
                    print(
                        f"[runner][trajectory-events][warn] staging merge "
                        f"failed for {effective_cid}: {_trajectory_merge_exc}"
                    )

                episode_summaries: List[Dict[str, Any]] = []
                # Fire-4 RCA Wave 1 M1: paper-grade strict fail-closed tracking.
                # On first PaperGradeAbortError caught from the task loop below,
                # these state vars are set so the aggregator + condition_summary
                # write know they need allow_quarantined=True + abort fields,
                # then the loop's outer scope re-raises to exit non-zero rc.
                _condition_aborted: bool = False
                _aborted_at_task: Optional[int] = None
                _abort_reason: Optional[str] = None
                _abort_message: Optional[str] = None
                backend = self._get_backend(condition.backend_id)

                # B-485 (/stress A1.5b Phase 1 P0-1-ABC, 2026-05-17): compute
                # resume identity fingerprint once per (condition, seed)
                # iteration; identity gate compares loaded summary's stored
                # fingerprint against this current-state hash.
                _resume_fingerprint = self._compute_resume_fingerprint(self.cfg, condition)

                # B-866 (/stress A1.23 P1-9 A, 2026-05-17): mid-run staging
                # re-merge. Pre-fix `merge_staging_trajectory_events` ran
                # exactly ONCE per condition_dir at line 754-766; if the
                # reset gate fires AGAIN mid-run (chain `--no-reset` not
                # passed + manual abort/retry on same RUN_ID), new
                # `reset_post_interrupt` events land in
                # `logs/trajectory_events_staging/RUN_${RUN_ID}.jsonl` but
                # never get merged into `condition_dir/trajectory_events.jsonl`
                # → paper §4 GLMM `is_after_reset` covariate systematically
                # FALSE-NEGATIVE on mid-run reset windows → covariate
                # adjustment biased. Counter-based re-merge every N=10 tasks
                # picks up any new staging events; fingerprint dedup
                # (B-491 in `merge_staging_trajectory_events`) prevents
                # double-merge so this is idempotent on resumes too.
                _staging_remerge_counter = 0

                for task in self.tasks:
                    # B-866: mid-run staging re-merge (every 10 tasks).
                    _staging_remerge_counter += 1
                    if _staging_remerge_counter % 10 == 0:
                        try:
                            from p79.experiment.logger_v2 import merge_staging_trajectory_events as _mid_merge
                            _staging_run_id_mid = str(self.cfg["experiment"].get("run_id", "")).strip()
                            if _staging_run_id_mid:
                                _mid_n = _mid_merge(
                                    condition_dir=condition_dir,
                                    run_id=_staging_run_id_mid,
                                )
                                if _mid_n:
                                    print(
                                        f"[runner][trajectory-events][mid-run] B-866 "
                                        f"merged {_mid_n} new staging events at "
                                        f"task {_staging_remerge_counter} of {effective_cid}"
                                    )
                        except Exception as _mid_exc:
                            # Best-effort — fingerprint dedup means worst case is
                            # we miss one mid-run event for paper §4 covariate;
                            # never block the runner hot path.
                            logger.warning(
                                "B-866 mid-run staging re-merge failed at task %d: %s",
                                _staging_remerge_counter, _mid_exc,
                            )
                    summary_file = condition_logger.summary_path(task.site, task.task_id)
                    if self.resume and summary_file.exists():
                        try:
                            with open(summary_file, "r", encoding="utf-8") as f:
                                loaded = json.load(f)

                            # B-169 (/stress A1.4a v8 codex B2, 2026-05-16) +
                            # B-485 (/stress A1.5b Phase 1 P0-1-ABC 3-AI
                            # overlap, 2026-05-17): identity tuple check.
                            # Pre-B-169 the resume gate accepted any file at
                            # the expected path; B-169 added 6-tuple path
                            # identity (run_id/condition_id/seed/site/task_id/
                            # schema_version); B-485 extends with
                            # `resume_fingerprint` (sha256[:16] embedding
                            # cfg.model.revision + backend.revision +
                            # max_new_tokens + temperature + paper_grade +
                            # observation_mode + transformers_version +
                            # prompt_hash) — restart with cfg drift now
                            # triggers quarantine + rerun instead of silent
                            # cross-condition contamination.
                            _expected = {
                                "schema_version": SCHEMA_VERSION_V2,
                                "run_id": self.cfg["experiment"]["run_id"],
                                "condition_id": effective_cid,
                                "seed": current_seed,
                                "benchmark_site": task.site,
                                "task_id": task.task_id,
                                "resume_fingerprint": _resume_fingerprint,
                            }
                            _mismatches = self._validate_resume_identity(loaded, _expected)
                            if _mismatches:
                                # Quarantine to <episodes>/quarantine/ for forensic audit
                                _quarantine_dir = condition_logger.episodes_dir / "quarantine"
                                _quarantine_dir.mkdir(parents=True, exist_ok=True)
                                _quarantine_path = (
                                    _quarantine_dir
                                    / f"{summary_file.stem}.{int(time.time())}.json"
                                )
                                try:
                                    shutil.move(str(summary_file), str(_quarantine_path))
                                    logger.warning(
                                        "B-169 resume identity mismatch site=%s task=%s — "
                                        "quarantined %s → %s. Mismatches: %s",
                                        task.site, task.task_id,
                                        summary_file.name, _quarantine_path.name,
                                        _mismatches,
                                    )
                                except OSError as _q_exc:
                                    logger.error(
                                        "B-169 quarantine move failed for %s: %s — re-running anyway",
                                        summary_file, _q_exc,
                                    )
                                # Fall through to re-run; do NOT continue
                            else:
                                # B-486 (/stress A1.5b Phase 1 P0-2-C gemini OOB,
                                # 2026-05-17): quarantine-rerun gate. Exception-
                                # path summaries (mid-evaluator crash, partial
                                # state) set `needs_reevaluation: True`. Pre-fix
                                # the resume gate accepted them as complete
                                # (has_steps=True from B-168 partial recovery →
                                # ingest into aggregate as success=False) → SR
                                # silent under-report in noisy environments.
                                # Now: detect flag → force re-run instead of
                                # ingest. True evaluator-based success/score
                                # only via re-run (agent self-claim of stop ≠
                                # task-level evaluator outcome; conflating both
                                # would over-report SR).
                                if bool(loaded.get("needs_reevaluation", False)):
                                    logger.info(
                                        "B-486 needs_reevaluation flag detected for "
                                        "site=%s task=%s — re-running episode "
                                        "(prior exception-path summary lacked "
                                        "evaluator score; quarantine + force re-run)",
                                        task.site, task.task_id,
                                    )
                                    # Move stale summary to quarantine for forensic;
                                    # next attempt will overwrite via normal path.
                                    _quarantine_dir = condition_logger.episodes_dir / "quarantine"
                                    _quarantine_dir.mkdir(parents=True, exist_ok=True)
                                    _quarantine_path = (
                                        _quarantine_dir
                                        / f"{summary_file.stem}.needs_reeval.{int(time.time())}.json"
                                    )
                                    try:
                                        shutil.move(str(summary_file), str(_quarantine_path))
                                    except OSError as _q_exc:
                                        logger.error(
                                            "B-486 quarantine move failed for %s: %s — "
                                            "re-running anyway",
                                            summary_file, _q_exc,
                                        )
                                    # Fall through to re-run; do NOT continue
                                else:
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

                    # Fire-4 RCA Wave 1 M1: catch PaperGradeAbortError so we
                    # can write condition_summary with abort fields BEFORE
                    # re-raising for process exit. Break exits this loop
                    # (`for task in self.tasks:`); the seed/condition loops
                    # above are exited via the outer re-raise after write.
                    try:
                        summary = self._run_and_record_episode(
                            condition, task, backend, condition_logger,
                            condition_dir, effective_cid, current_seed,
                        )
                        episode_summaries.append(summary)
                    except PaperGradeAbortError as _abort_exc:
                        logger.error(
                            "PaperGradeAbortError caught at site=%s task=%s — "
                            "aborting condition for paper-grade fail-closed: %s",
                            task.site, task.task_id, _abort_exc,
                        )
                        _condition_aborted = True
                        _aborted_at_task = int(task.task_id) if task.task_id is not None else None
                        _abort_reason = "quarantine"
                        _abort_message = str(_abort_exc)
                        break

                # ── Retry pass: re-run tasks whose summaries were deleted ──
                # (e.g. by watchdog auto-cleanup of benchmark noise errors)
                # Fire-4 RCA Wave 1 M1: skip retry pass when condition aborted
                # — we are in fail-closed mode, do NOT attempt re-runs on a
                # potentially-compromised substrate.
                retry_tasks = [] if _condition_aborted else [
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
                        # Fire-4 RCA Wave 1 M1: same fail-closed gate on retry
                        # pass — if a re-run produces a quarantine summary,
                        # abort identically (paper-grade does not retry on
                        # potentially-compromised substrate twice).
                        try:
                            summary = self._run_and_record_episode(
                                condition, task, backend, condition_logger,
                                condition_dir, effective_cid, current_seed,
                            )
                        except PaperGradeAbortError as _abort_exc:
                            logger.error(
                                "PaperGradeAbortError caught in retry pass at "
                                "site=%s task=%s — aborting condition: %s",
                                task.site, task.task_id, _abort_exc,
                            )
                            _condition_aborted = True
                            _aborted_at_task = int(task.task_id) if task.task_id is not None else None
                            _abort_reason = "quarantine_in_retry_pass"
                            _abort_message = str(_abort_exc)
                            break
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

                # Fire-4 RCA Wave 1 M1: aggregator must accept quarantined
                # episodes when the condition was aborted by paper-grade
                # fail-closed (`needs_reevaluation=True` in last episode is
                # the abort trigger itself, so default `allow_quarantined=False`
                # would raise inside aggregate). For non-aborted conditions
                # the strict default preserves B-784 quarantine rejection
                # invariant (live aggregate must not silently count
                # quarantined episodes as paper-grade outcomes).
                if _condition_aborted:
                    aggregate = aggregate_condition_metrics(
                        episode_summaries, allow_quarantined=True
                    )
                else:
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
                # Fire-4 RCA Wave 1 M1: stamp abort fields on condition_summary
                # so forensic + chain orchestration can distinguish "aborted by
                # paper-grade quarantine gate" from "completed all tasks" /
                # "killed by wallclock" / "killed by SIGTERM". Empty/None when
                # condition completed normally.
                aggregate["condition_aborted"] = _condition_aborted
                aggregate["aborted_at_task"] = _aborted_at_task
                aggregate["abort_reason"] = _abort_reason
                aggregate["abort_message"] = _abort_message
                if _condition_aborted:
                    # Episode count gives chain operator the "burned 0 tasks
                    # after first quarantine" claim that Fire-4 violated
                    # (burned 26 tasks). Always 1 in correct fail-closed flow.
                    aggregate["aborted_summary_count"] = len(episode_summaries)

                # B-1645 (/stress A2.10 P1-6-A 2026-05-18): surface learned-router
                # diagnostic block when observation_mode=="learned" so paper §6
                # H10 transparency disclosure can cite per-cell signal-strength
                # fallback rate. Only emitted for learned-mode conditions; absent
                # for baseline modes (DOM / SoM / Vision / P-text / P-prompt /
                # P-SoM) where the LR dispatch path is never invoked. The
                # `fallback_rate` denominator counts EVERY dispatch attempt
                # (including those that succeed); `fallback_count` is the subset
                # that fell back via candidate_modes filter / max_prob ≤ τ /
                # feature-extraction exception. Infrastructure-level failures
                # (LearnedRouterArtifactError per B-1640) do NOT reach this
                # accounting — they kill the cell run loudly before write.
                if condition.observation_mode == "learned":
                    dispatch_count = int(getattr(self, "_lr_dispatch_count", 0))
                    fallback_count = int(getattr(self, "_lr_fallback_count", 0))
                    fallback_rate = (
                        float(fallback_count / dispatch_count)
                        if dispatch_count > 0 else None
                    )
                    aggregate["learned_router_diag"] = {
                        "dispatch_count": dispatch_count,
                        "fallback_count": fallback_count,
                        "fallback_rate": fallback_rate,
                        "fallback_kind_note": (
                            "Signal-strength fallback only "
                            "(max_prob ≤ tau / candidate_modes filter / "
                            "non-infrastructure runtime exception). "
                            "Infrastructure-level failures "
                            "(LearnedRouterArtifactError) propagate and kill "
                            "the cell run; they are NOT counted here. "
                            "Per /stress A2.10 P0-3-B + P1-6-A user-mandate "
                            "hard-fail 2026-05-18."
                        ),
                    }

                # B-487 (/stress A1.5b Phase 1 P0-3-B codex OOB, 2026-05-17):
                # Option K covariate substrate — emit lightweight episode list
                # + ids into condition_summary so
                # `aggregate_trajectory_covariates.py:120-180` (B-389) can
                # correlate trajectory_events vs per-episode wallclock instead
                # of falling back to filesystem scan (which loses the
                # condition-authoritative episode set used for finalize-race
                # detection at B-385's aggregator intersection).
                aggregate["episode_summaries"] = [
                    {
                        "task_id": int(s.get("task_id", -1)),
                        "benchmark_site": str(s.get("benchmark_site", "")),
                        "success": bool(s.get("success", False)),
                        "score": float(s.get("score", 0.0)),
                        "wallclock_start": s.get("wallclock_start"),
                        "wallclock_end": s.get("wallclock_end"),
                        "needs_reevaluation": bool(s.get("needs_reevaluation", False)),
                        # B-485 propagation: carry fingerprint into the
                        # condition-summary episode list so post-hoc forensic
                        # readers can audit cfg-state-per-episode without
                        # re-reading every summary file.
                        "resume_fingerprint": s.get("resume_fingerprint"),
                    }
                    for s in episode_summaries
                    if isinstance(s, dict)
                ]
                aggregate["episode_ids"] = [
                    int(s.get("task_id", -1))
                    for s in episode_summaries
                    if isinstance(s, dict) and s.get("task_id") is not None
                ]

                condition_logger.write_condition_summary(aggregate)
                run_condition_metrics.append(aggregate)

                # Auto-run analysis after each condition completes
                # Fire-4 RCA Wave 1 M1: skip post-condition analysis on abort
                # — partial data + needs_reevaluation episode would break
                # rederive / cross_rep diagnostics; defer to manual re-fire.
                if not _condition_aborted:
                    self._run_post_condition_analysis(effective_cid)

                # Fire-4 RCA Wave 1 M1: re-raise PaperGradeAbortError now that
                # condition_summary has been written with abort fields.
                # Process exits non-zero rc → queue_chain rc≠0 → master
                # P0-2-B sentinel halt → red chain NOT launched. Forensic
                # evidence (aborted episode summary + condition summary with
                # aborted_at_task) preserved on disk for operator triage.
                if _condition_aborted:
                    # /stress 2026-08-09 Mode B P1-7: this is the paper-grade
                    # quarantine exit — condition_summary is on disk and the
                    # process is about to die non-zero, halting the chain. It
                    # had no notification, so a mid-chain abort was as silent
                    # as the quota stop that motivated push_run_abort_ntfy.
                    # Placed here rather than at the inner `except
                    # PaperGradeAbortError` because this is the single point
                    # every quarantine path funnels through, and only here are
                    # the task / reason fields settled.
                    push_run_abort_ntfy(
                        effective_cid,
                        str(getattr(self, "_last_site", "") or "?"),
                        _aborted_at_task,
                        "paper_grade_quarantine",
                        _abort_message or _abort_reason or "condition aborted",
                    )
                    raise PaperGradeAbortError(
                        _abort_message or
                        f"condition aborted at task={_aborted_at_task} "
                        f"(reason={_abort_reason}); see condition_summary.json"
                    )

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

        # B-331 (/stress A1.9 Mode B F6 OOB, 2026-05-16): atomic + fsync
        # write via shared helper. Pre-fix plain `json.dump` could truncate
        # on crash mid-write while condition_summary used atomic+fsync —
        # asymmetric durability across writers.
        from p79.experiment.logger_v2 import write_run_summary_atomic
        write_run_summary_atomic(
            self.output_root / "run_summary_v2.json", run_summary
        )

        self._create_latest_symlink()
        return self.output_root

    def _run_post_condition_analysis(self, condition_id: str) -> None:
        """Run analyze_experiment.py in a subprocess after a condition completes.

        B-498 (/stress A1.5b Phase 1 P1-10-A, 2026-05-17): timeout bumped from
        300s (5min) to 1800s (30min). Pre-fix shopping 466-task cross-condition
        aggregator on contested DGX CPU could exceed 5min → silent
        TimeoutExpired → stale paper aggregate. Bump + ntfy on timeout so
        operator sees the staleness.
        """
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
            # B-498: 30-min budget for large shop / multi-condition aggregators.
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if result.returncode == 0:
                logging.info("[runner] Post-condition analysis completed for %s", condition_id)
            else:
                logging.warning(
                    "[runner] Post-condition analysis exited %d for %s: %s",
                    result.returncode, condition_id, result.stderr[-500:] if result.stderr else "",
                )
        except subprocess.TimeoutExpired:
            logging.warning(
                "[runner] B-498 Post-condition analysis TIMED OUT after 1800s for %s — "
                "cross-condition aggregate will be stale until next manual `make analysis` run",
                condition_id,
            )
            # Best-effort ntfy push so operator sees the staleness; absent topic = no-op
            _topic = os.environ.get("NTFY_TOPIC", "").strip()
            if _topic:
                try:
                    _req = urllib.request.Request(
                        f"https://ntfy.sh/{_topic}",
                        data=(
                            f"P79 B-498: post-condition analyze TIMED OUT for "
                            f"{condition_id} after 30min — cross-condition aggregate stale"
                        ).encode("utf-8"),
                        method="POST",
                        headers={"Title": f"P79 timeout {condition_id}", "Priority": "high"},
                    )
                    urllib.request.urlopen(_req, timeout=15).close()
                except Exception:
                    pass  # best-effort, do not abort runner
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

        # B-495 (/stress A1.5b Phase 1 P1-7-B codex OOB, 2026-05-17): atomic
        # write for artifacts (tmp + fsync + replace + parent_fsync). Pre-fix
        # plain `obs.image.save()` + `open(...).write()` left artifact files
        # with no durability guarantee — step JSONL was fsync'd with
        # `artifact_paths` pointer, but the pointed-to file could be
        # half-written / missing post-crash. Result: paper §3 / §5 / gallery
        # evidence layer has dangling artifact references. B-225/B-331
        # atomicity chain extends here (last missing layer).
        # B-1830 (2026-05-22, B-1828 P0-1 sibling): only COMPUTE the screenshot
        # path here; the actual atomic PNG write is DEFERRED by the runner to
        # AFTER total_latency_ms + energy (vision's raw screenshot is consumed by
        # the model as the in-memory obs.image, but the disk write is inspection —
        # writing it inside the timed step window inflated vision latency canonical,
        # the same defect P0-1 fixed for som's marked-image save). The runner
        # writes obs.image atomically at this path post-timing (B-495 durability
        # preserved there). dom-text save below stays in-window UNCHANGED — it is
        # cheap + symmetric across all modes incl. dom (R9755 latency consistency).
        screenshot_path: Optional[str] = None
        if getattr(obs, "image", None) is not None:
            screenshot_path = str(step_dir / "screenshot.png")

        _dom_target = step_dir / "observation_dom.txt"
        _dom_tmp = step_dir / "observation_dom.txt.tmp"
        with open(_dom_tmp, "w", encoding="utf-8") as f:
            f.write(obs.text or "")
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(_dom_tmp), str(_dom_target))
        try:
            _pfd = os.open(str(step_dir), os.O_RDONLY)
            try:
                os.fsync(_pfd)
            finally:
                os.close(_pfd)
        except OSError:
            pass  # dir fsync best-effort
        dom_path = str(_dom_target)

        return {
            "screenshot": screenshot_path,
            "dom": dom_path,
            "trace": None,
            "som_image": None,
        }

    @staticmethod
    def _classify_transient_substrate(msg: str) -> Optional[str]:
        """Classify a quarantine error string as a transient-substrate class
        ("auth" / "proxy_5xx" / "network") or None (non-transient). Used for BOTH
        provenance/logging AND the retry gate, but classification alone does NOT
        authorize a retry — the wrapper (_run_and_record_episode) additionally
        requires steps==0 (pre-flight) AND class ∈ {auth, network} (proxy_5xx is
        owned by B-1880's internal retry, so it aborts here).

        B-1881 (reddit chain abort #3/#4, 2026-06-20). CONSERVATIVE by design —
        only unambiguous infra signals match. Agent-induced Playwright timeouts
        (M5 taxonomy:
        agent_observation / agent_navigation / agent_action), benchmark noise,
        evaluator-init failures, and proxy 403 (quota) all return None so they
        still abort — misclassifying a real agent/benchmark failure as transient
        would mask it by retrying. Matches on the (≤200-char-truncated) error
        embedded in the PaperGradeAbortError message; the discriminating tokens
        (auth_required_gate / 503 / execute-api / connection reset) all appear at
        the head of those error strings."""
        s = msg or ""
        low = s.lower()
        # auth: site session login failure (AuthRefreshFailure from auth gate).
        if (
            "authrefreshfailure" in low
            or "auth_required_gate" in low
            or "refresh_site_auth returned false" in low
            or "login_failed" in low
        ):
            return "auth"
        # proxy 5xx: AWS proxy (API Gateway → Bedrock) transient unavailability.
        # 403 (quota) is raised upstream before reaching the quarantine path, so
        # only genuine 5xx transients reach here.
        if ("execute-api" in low or "model-api" in low) and (
            "503" in s or "502" in s or "504" in s
            or "service unavailable" in low or "bad gateway" in low
            or "gateway timeout" in low
        ):
            return "proxy_5xx"
        # network: connection-level transients (proxy or VWA site).
        if (
            "connection aborted" in low
            or "connection reset" in low
            or "connection refused" in low
            or "max retries exceeded" in low
            or "remote end closed connection" in low
            or "temporary failure in name resolution" in low
        ):
            return "network"
        return None

    def _recover_transient_substrate(self, tclass: str, task: Any, attempt: int) -> None:
        """Recover from a transient-substrate quarantine before an episode retry
        (B-1881). Bounded capped backoff lets the transient window pass; the auth
        class additionally forces the next episode's auth gate to re-authenticate.

        The next `_run_episode` call cleans the prior attempt's partial artifacts
        on entry (B-488 stale-archive / wipe), so a retry runs on fresh substrate."""
        backoff = min(30 * (2 ** (attempt - 1)), 120)  # 30, 60, 120 (cap 120)
        logger.info(
            "B-1881 transient recovery (class=%s attempt=%d): backoff %ds before "
            "episode retry on fresh substrate",
            tclass, attempt, backoff,
        )
        time.sleep(backoff)
        if tclass == "auth":
            # Force the next _run_episode auth gate to re-authenticate: set the
            # last-refresh ts to epoch so should_refresh() sees a huge
            # seconds_since and re-runs auth_required_gate. (The failed gate did
            # not update the ts, but a recent prior success could otherwise
            # suppress the re-gate — set explicitly.)
            self._auth_last_refresh_ts[task.site] = 0.0

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
        """B-1881 transient-substrate retry wrapper around _run_and_record_episode_once.

        Fire-4 RCA Wave 1 M1 made the FIRST quarantine event (needs_reevaluation
        =True) fail-closed abort the whole condition — correct for non-transient
        substrate compromise, but for TRANSIENT infra blips (AWS-proxy 5xx,
        reddit/shop session-auth login failure, connection resets) it converts a
        recoverable hiccup into total-condition loss (reddit chain abort #3
        R28130 proxy-503 lost 55 ep; #4 R26851 auth lost 137 ep — both failed at
        episode boundaries with ZERO contamination).

        Now: a PRE-FLIGHT (steps==0) transient quarantine of class auth/network
        triggers a BOUNDED episode-level retry on fresh substrate (re-auth /
        capped backoff) instead of a condition abort. The steps==0 gate is the
        master safety (3-AI /stress consensus 2026-06-20): the agent took no
        browser action, so there is NO site mutation (codex P0-1), NO stochastic-
        rollout to redraw (P0-2), and the failure CANNOT be agent-induced (P1-3)
        — the retry is "the episode finally starting on valid substrate", which
        leaves the SR estimand UNCHANGED (hence PROTOCOL_NOTE_02, not an OSF
        amendment). Mid-episode failures (steps>0), proxy_5xx (owned by B-1880's
        ~11min internal retry), non-transient quarantines (agent / benchmark /
        evaluator), and retry exhaustion all still re-raise PaperGradeAbortError
        for operator intervention — the fail-closed safety net is preserved.
        Retries are logged to trajectory_events.jsonl + stamped on the canonical
        episode summary (transient_retry_count) + ntfy for transparency. Bound via
        yaml `transient_episode_max_retries` (default 3); diagnostic_replay + dev
        mode keep the legacy single-attempt fail-closed.
        """
        _paper_grade = bool(self.cfg.get("paper_grade", False))
        _max = int(self.cfg.get("transient_episode_max_retries", 3))
        # B-1881 (3-AI /stress consensus 2026-06-20): retry ONLY pre-flight
        # (steps==0) transient infra failures, where the agent took no browser
        # action ⇒ no site mutation (codex P0-1), no stochastic-rollout redraw
        # (3-AI P0-2), no agent-induced masking (P1-3). proxy_5xx is EXCLUDED:
        # the proxy agent already retries 5xx internally ~11min (B-1880); an
        # episode-level proxy_5xx reaching here = that budget exhausted =
        # sustained outage ⇒ legitimate abort (also avoids the B-1880×B-1881
        # ~4×11min worst-case time-to-abort, P1-5).
        _RETRYABLE_CLASSES = {"auth", "network"}
        _attempt = 0
        _retry_classes: List[str] = []
        while True:
            try:
                summary = self._run_and_record_episode_once(
                    condition, task, backend, condition_logger,
                    condition_dir, effective_cid, current_seed,
                )
                # B-1881 P1-6 transparency: stamp retry lineage on the (clean)
                # returned summary so the CANONICAL data layer — not just
                # trajectory_events.jsonl — records that this outcome was
                # transient-retry-rescued (reviewer can audit retry frequency
                # from episode summaries; populates the EpisodeSummaryV2
                # attempt-lineage reservation in types.py).
                if _attempt > 0 and isinstance(summary, dict):
                    summary["transient_retry_count"] = _attempt
                    summary["transient_retry_classes"] = list(_retry_classes)
                    summary["is_retry_attempt"] = True
                    summary["attempt_index"] = _attempt
                    summary["retry_trigger"] = _retry_classes[-1]
                    try:
                        condition_logger.write_episode_summary(
                            task.site, task.task_id, summary,
                        )
                    except Exception:  # canonical write already done in once(); best-effort stamp
                        pass
                return summary
            except PaperGradeAbortError as _abort:
                _tclass = getattr(_abort, "transient_class", None)
                _steps = int(getattr(_abort, "steps", 0) or 0)
                if (
                    not _paper_grade
                    or self.diagnostic_replay
                    or _tclass not in _RETRYABLE_CLASSES
                    or _steps != 0            # mid-episode (steps>0) ⇒ NOT retryable
                    or _attempt >= _max
                ):
                    # non-transient / mid-episode / exhausted / dev → fail-closed abort
                    raise
                _attempt += 1
                _retry_classes.append(_tclass)
                logger.warning(
                    "B-1881 PRE-FLIGHT transient quarantine (class=%s steps=0) at "
                    "site=%s task=%s — episode-level retry %d/%d on fresh substrate "
                    "(NOT condition abort): %s",
                    _tclass, task.site, task.task_id, _attempt, _max,
                    str(_abort)[:200],
                )
                # P1-4: delete the failed canonical summary BEFORE the backoff so
                # the watchdog (tracks by episode-key, not mtime —
                # experiment_watchdog.py:2151) cannot ingest the needs_reevaluation
                # summary during the sleep window, false-alert, and key-lock
                # against the clean retry overwrite. steps==0 ⇒ no forensic step
                # data is lost (the episode never ran); the trajectory event below
                # preserves the retry record.
                try:
                    condition_logger.summary_path(
                        task.site, task.task_id,
                    ).unlink(missing_ok=True)
                except Exception:
                    pass
                try:
                    condition_logger.log_trajectory_event(
                        "transient_substrate_retry",
                        task_index=(
                            int(task.task_id) if task.task_id is not None else None
                        ),
                        metadata={
                            "transient_class": _tclass,
                            "retry_attempt": _attempt,
                            "max_retries": _max,
                            "steps_at_failure": _steps,
                            "site": task.site,
                            "condition_id": effective_cid,
                        },
                    )
                except Exception:  # transparency best-effort; never block recovery
                    pass
                _notify_transient_retry(
                    effective_cid, task.site, task.task_id, _tclass, _attempt, _max,
                )
                self._recover_transient_substrate(_tclass, task, _attempt)
                continue

    def _run_and_record_episode_once(
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
        A first quarantine event (needs_reevaluation=True) under paper_grade raises
        PaperGradeAbortError — the transient-retry wrapper (_run_and_record_episode)
        decides whether to retry (transient) or propagate (abort).
        """
        logger.info(
            "Running condition=%s seed=%d backend=%s site=%s task=%s",
            effective_cid, current_seed, condition.backend_id, task.site, task.task_id,
        )
        # B-487 (/stress A1.5b Phase 1 P0-3-B codex OOB, 2026-05-17): Option K
        # covariate anchor. Stamp wallclock_start pre-try so both normal AND
        # exception paths can attach to summary — aggregate_trajectory_covariates
        # (B-389) uses it to time-order reset_post_interrupt events vs episode
        # lifetime (`is_after_reset` / `prior_event_count` covariates).
        _wallclock_start = datetime.now(timezone.utc).isoformat()
        # B-1800 (P1-6, /stress 2026-05-21 codex Mode B OOB): crash-atomic billed-
        # cost ledger. The billed model call (_run_episode: backend.step) happens
        # ~960 lines before the step JSONL is written; an exception in that window
        # loses the in-flight step's billed model spend + its model-call attempt
        # (the except block below recovers ONLY written JSONL via
        # _aggregate_partial_steps). Reset the instance ledger here (per episode);
        # _run_episode increments it right after each billed call; the except block
        # corrects the recovered _agg so §1 total_billed_cost / model_call_attempt
        # are not under-reported on a mid-step crash.
        self._episode_inflight_billed_usd = 0.0
        self._episode_inflight_model_attempts = 0
        # B-485 (/stress A1.5b Phase 1 P0-1-ABC, 2026-05-17): compute resume
        # fingerprint per episode so summary write carries it for later
        # restart's identity gate. Per-episode compute is OK (microsecond
        # hash work) and avoids passing the value down through the call
        # chain from run().
        _resume_fingerprint = self._compute_resume_fingerprint(self.cfg, condition)
        try:
            summary = self._run_episode(
                condition, task, backend, condition_logger, condition_dir,
                effective_cid=effective_cid,
            )
            # B-487: stamp anchors on success path. _run_episode returns
            # the dict; we inject Option K covariate substrate fields here
            # so all consumers (write_episode_summary + aggregator) see them.
            summary["wallclock_start"] = _wallclock_start
            summary["wallclock_end"] = datetime.now(timezone.utc).isoformat()
            # B-485: stamp resume fingerprint for next restart's identity gate.
            summary["resume_fingerprint"] = _resume_fingerprint
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            exc_str = str(exc)
            # B-1662 (/stress A2.11 P0-3-BC 2026-05-18, user Q2=A): evaluator
            # infrastructure unavailable = fail-fast condition abort, NOT write
            # needs_reevaluation=True summary to canonical episodes dir. Pre-fix
            # B-486 quarantine path wrote half-failed summary that downstream
            # metrics aggregator hard-fails on, but only AFTER condition runs to
            # completion = hours wasted before operator sees env bug. Plus
            # canonical dir gets polluted with quarantine summaries that next
            # attempt must clean. Now: env infra error → immediate raise →
            # condition aborts → operator fixes env (NLTK download / OpenAI
            # key / VWA submodule) + re-fires from clean state. Empirical
            # 2026-05-18 13:28:06 fire: cls runner burned 4 tasks on missing
            # NLTK punkt; with this guard, runner would abort at task 0.
            from p79.experiment.environment import EvaluatorUnavailableError
            # All three branches below STOP the run rather than quarantine the
            # episode, because every subsequent task would fail identically.
            # Each also pushes an urgent ntfy: the stop is by design, but the
            # silence was not — 2026-08-09 the shop B0 vision run stopped on
            # quota at task 405 and sat dead six hours on a booked machine
            # before anyone noticed. push_run_abort_ntfy never raises.
            if isinstance(exc, EvaluatorUnavailableError):
                logger.error(
                    "Evaluator infrastructure unavailable at site=%s task=%s — "
                    "ABORTING condition (do NOT write needs_reevaluation summary; "
                    "operator must fix env + re-fire from clean state): %s",
                    task.site, task.task_id, exc,
                )
                push_run_abort_ntfy(
                    effective_cid, task.site, task.task_id, "evaluator_unavailable", exc,
                )
                raise
            if self._FATAL_ENV_REGEX.search(exc_str):
                logger.error(
                    "Fatal environment error at site=%s task=%s — "
                    "stopping run to allow clean restart: %s",
                    task.site, task.task_id, exc,
                )
                push_run_abort_ntfy(
                    effective_cid, task.site, task.task_id, "fatal_env", exc,
                )
                raise
            # Proxy API quota exhaustion — stop run (all subsequent tasks will fail)
            if "403" in exc_str and any(m in exc_str for m in ("model-api", "execute-api")):
                logger.error(
                    "Proxy API quota exhausted at site=%s task=%s — "
                    "stopping run: %s",
                    task.site, task.task_id, exc,
                )
                push_run_abort_ntfy(
                    effective_cid, task.site, task.task_id, "proxy_quota", exc,
                )
                raise
            logger.warning(
                "Episode failed at condition=%s seed=%d site=%s task=%s: %s",
                effective_cid, current_seed, task.site, task.task_id, exc,
                exc_info=True,
            )
            noise, noise_cat = detect_benchmark_noise(str(exc))
            # Fire-4 RCA Wave 2 M5 (/stress 3-AI 2026-05-19, user A1=b):
            # classify timeout separately so the summary distinguishes
            # "unverified timeout pending manual review" (M5 callsite
            # taxonomy: agent_observation / agent_navigation / agent_action /
            # network / agent_playwright_other / unknown) from auto-tagged
            # benchmark noise (api_rate_limit / auth_expired / api_infra /
            # anti_bot / geo_restricted). Pre-Wave-2: any timeout substring
            # set `benchmark_noise=True, category="timeout"` → silently
            # excluded from clean SR transparency metric → reviewer
            # cherry-picking lever (Fire-4 task 75 auto-noise classification
            # could've masked agent-induced DOM deadlock).
            _is_timeout, _timeout_callsite = classify_timeout(str(exc))

            # B-168 (/stress A1.4a v8 codex B1, 2026-05-16): partial-step
            # crash recovery. Try to read any JSONL rows already written
            # before the exception fired, so the summary's
            # steps/tokens/cost/latency reflect what actually happened
            # rather than zero-step erasure. Pre-fix the except path
            # emitted ``steps=0,total_cost=0`` for crashes that occurred
            # after 12+ step JSONL writes, creating same-episode JSONL
            # vs summary divergence (paper-grade evidence layer split).
            _partial_steps: List[Dict[str, Any]] = []
            try:
                _jsonl_path = condition_logger.step_log_path(task.site, task.task_id)
                if _jsonl_path.exists():
                    # B-493 (/stress A1.5b Phase 1 P1-5-B codex OOB, 2026-05-17):
                    # pass `summary_path` to enable B-180 identity guard. Pre-fix
                    # corrupt summary fall-through to rerun read raw JSONL via
                    # plain `read_jsonl_dedup(_jsonl_path)`; if 2nd attempt
                    # crashed before step_idx=0 was written, exception path
                    # would read prior-attempt's JSONL and emit error summary
                    # with stale cost/steps from prior attempt. B-180 guard
                    # (io_utils.py:131-178) rejects segments whose identity
                    # doesn't match the summary path → returns empty list, so
                    # exception path correctly emits zero-step error summary
                    # for this attempt instead of cross-attempt contamination.
                    _summary_path = condition_logger.summary_path(task.site, task.task_id)
                    _partial_steps = read_jsonl_dedup(_jsonl_path, summary_path=_summary_path)
            except Exception as _read_exc:
                logger.warning(
                    "B-168 partial-JSONL read failed for site=%s task=%s: %s — "
                    "falling back to zero-step error summary",
                    task.site, task.task_id, _read_exc,
                )
                _partial_steps = []
            # B-490 (A1.5b Phase 1 P1-2-AC, 2026-05-17): pass condition's
            # observation_mode so escalation_count can be recomputed from
            # partial steps (not hardcoded 0 = silent paper §4 bias).
            _agg = self._aggregate_partial_steps(_partial_steps, condition.observation_mode)
            # B-1800 (P1-6): correct the JSONL-recovered billed total + model-call
            # count with the crash-atomic ledger. `_agg` sums only WRITTEN steps; the
            # ledger also includes the in-flight (unwritten) billed model call lost to
            # the crash. delta = ledger − recovered; attribute it to billed + wasted
            # (the unwritten step produced no valid action) so canonical+wasted≡billed
            # holds and §1 total_billed is not under-reported. getattr-guarded for
            # any path that bypassed the per-episode reset.
            _ledger_billed = float(getattr(self, "_episode_inflight_billed_usd", 0.0) or 0.0)
            _ledger_attempts = int(getattr(self, "_episode_inflight_model_attempts", 0) or 0)
            _recovered_billed = float(_agg.get("total_billed_cost_usd") or 0.0)
            _billed_delta = max(0.0, _ledger_billed - _recovered_billed)
            if _billed_delta > 0.0:
                _agg["total_billed_cost_usd"] = _recovered_billed + _billed_delta
                _agg["total_model_cost_usd"] = float(_agg.get("total_model_cost_usd") or 0.0) + _billed_delta
                _agg["total_cost_usd"] = float(_agg.get("total_cost_usd") or 0.0) + _billed_delta
                _agg["protocol_wasted_cost_usd"] = float(_agg.get("protocol_wasted_cost_usd") or 0.0) + _billed_delta
            if _ledger_attempts > int(_agg.get("model_call_attempt_count") or 0):
                _agg["model_call_attempt_count"] = _ledger_attempts

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
                steps=_agg["steps"],
                retries=_agg["retries"],
                no_op_rate=_agg["no_op_rate"],
                page_unchanged_rate=_agg["page_unchanged_rate"],
                total_latency_ms=_agg["total_latency_ms"],
                total_latency_minus_retry_ms=_agg["total_latency_minus_retry_ms"],
                p95_step_latency_ms=_agg["p95_step_latency_ms"],
                total_tokens=_agg["total_tokens"],
                total_model_cost_usd=_agg["total_model_cost_usd"],
                total_cost_usd=_agg["total_cost_usd"],
                total_router_overhead_cost_usd=_agg["total_router_overhead_cost_usd"],
                total_router_overhead_ms=_agg["total_router_overhead_ms"],
                total_energy_kwh=None,
                total_co2e_kg=None,
                escalation_count=_agg["escalation_count"],
                # P1-1 (/stress accounting audit 2026-05-21, codex Mode B): two-budget
                # counters + three-column cost on the exception path so a crash episode
                # carries consistent accounting (not steps>0 with counters=0/cost=None).
                agent_action_step_count=_agg["agent_action_step_count"],
                valid_action_step_count=_agg["valid_action_step_count"],
                model_call_attempt_count=_agg["model_call_attempt_count"],
                runner_iteration_count=_agg["runner_iteration_count"],
                parse_error_injected_wait_count=_agg["parse_error_injected_wait_count"],
                total_billed_cost_usd=_agg["total_billed_cost_usd"],
                canonical_action_cost_usd=_agg["canonical_action_cost_usd"],
                protocol_wasted_cost_usd=_agg["protocol_wasted_cost_usd"],
                trigger_distribution={},
                benchmark_noise=noise,
                benchmark_noise_category=noise_cat,
                # Fire-4 RCA Wave 2 M5 timeout taxonomy fields. Defaults None
                # for non-timeout episodes; bool/string when classify_timeout
                # detects timeout. verified_substrate_noise stays None until
                # manual review tooling (Wave 4 / Phase 2 follow-up) flips
                # to True (legitimate substrate) — until then the episode
                # stays in agent-failure denominator (counted in SR).
                unverified_timeout_event=_is_timeout,
                timeout_callsite=_timeout_callsite,
                verified_substrate_noise=None,
                # Fire-6 RCA Stage C2: diagnostic-replay provenance (error path).
                diagnostic_replay=self.diagnostic_replay,
                sr_excluded=self.diagnostic_replay,
                artifacts_dir=str(condition_dir),
                error=str(exc),
                # B-487 (/stress A1.5b Phase 1 P0-3-B): Option K anchors —
                # propagate captured start ts + stamp end-of-attempt ts so
                # aggregator's time-ordering covariates have data substrate
                # even on crash path.
                wallclock_start=_wallclock_start,
                wallclock_end=datetime.now(timezone.utc).isoformat(),
                # B-485 (/stress A1.5b Phase 1 P0-1-ABC): stamp resume
                # fingerprint so quarantined error summaries also carry
                # identity invariant (downstream forensic audit may still
                # want to reason about which cfg-state produced the crash).
                resume_fingerprint=_resume_fingerprint,
                # B-486 (/stress A1.5b Phase 1 P0-2-C gemini OOB): flag
                # exception-path summary for resume-gate force re-run.
                # B-168 partial recovery preserves steps/cost/latency but
                # evaluator did NOT score (mid-eval crash). Marking this
                # flag forces re-run on next restart; without it, resume
                # gate ingests success=False as "complete" → task never
                # re-evaluated → SR systematic under-report.
                needs_reevaluation=True,
                # B-554 (/stress A1.5 P1-4-AB* Claude+codex OOB, 2026-05-17):
                # cohort sentinel — even exception-path summaries stamp
                # the post-B-545 authority semantic so consumers can
                # distinguish "legacy archive (None)" from "post-B545
                # exception path (post_B545_vwa_score_only)".
                evaluator_authority_mode="post_B545_vwa_score_only",
                reward_override_applied=False,
            ).as_dict()
            # B-194 (/stress A1.4b-ii codex B-ii-3, P1 OOB): exception-path
            # MUST mirror canonical `compute_wasted_cost(steps, success=False)`
            # semantics — failed episode → wasted = total. Pre-fix this path
            # force-zeroed `wasted_cost_usd` even though `_agg["total_cost_usd"]`
            # was already recovered from JSONL (B-168), creating a systematic
            # under-report: `success=False AND total_cost>0 AND wasted_cost=0`
            # contradicts the paper §3.6 definition that all cost on a failed
            # episode is wasted. Now: wasted = total (recovered from partial
            # JSONL), wasted_energy carries forward from the runner-side total
            # if available (else 0.0; energy not always populated on crash).
            summary["wasted_cost_usd"] = float(_agg.get("total_cost_usd", 0.0))
            summary["wasted_energy_kwh"] = 0.0
            # B-789 (/stress A1.9 cold-start P1-5-B* codex OOB, 2026-05-17):
            # exception-path component breakdown must include `obs_prepare_usd`
            # to match the normal path (B-576) and the runner's
            # `total_cost_usd = model + router_overhead + obs_prepare` invariant.
            # Pre-fix manual dict here only had model/router/energy → cross-baseline
            # component plots on failed/quarantine cohorts (paper §3.6) systematically
            # under-counted obs_prepare overhead, breaking schema closure with the
            # normal-path output of `compute_component_breakdown`.
            summary["component_breakdown"] = {
                "model_cost_usd": _agg["total_model_cost_usd"],
                "router_overhead_usd": _agg["total_router_overhead_cost_usd"],
                # B-1799 (P1-5): _aggregate_partial_steps now returns this key, so
                # the .get() resolves to the real obs_prepare sum (was silent 0.0).
                "obs_prepare_usd": float(_agg.get("total_obs_prepare_cost_usd", 0.0)),
                "total_energy_kwh": 0.0,
            }
            # B-1798 (P1-1): propagate the cost-basis rollup to the crash summary so
            # the condition aggregator can stratify crash cohorts too (None ≡ no
            # recoverable step carried a basis).
            summary["cost_unit_basis"] = _agg.get("cost_unit_basis")
            summary["cost_total_mixed_unit_warn"] = _agg.get("cost_total_mixed_unit_warn")
            # B-166 propagation: error summaries also flagged incomplete
            summary["trajectory_incomplete"] = True
            summary["unknown_failure_reasons"] = {}
            summary["partial_recovery_step_count"] = _agg["steps"]

        try:
            condition_logger.write_episode_summary(task.site, task.task_id, summary)
        except Exception as write_exc:
            # B-323 (/stress A1.9 Mode B F1 OOB, 2026-05-16): paper-grade mode
            # raises on episode summary disk-write failure. Pre-fix the bare
            # `try/except + log` swallowed NFS / disk-full / permission errors
            # → in-memory aggregate counted the episode while
            # `episodes/*_summary_v2.json` was missing on disk → `analyze_run()`
            # re-scan path produced different denominators than runner's live
            # path → paper §1 / §3 disk-vs-memory split-brain. Now: paper-grade
            # mode (`paper_grade: true` in yaml) fails loud; dev mode still
            # swallows + logs for backwards compat.
            paper_grade = bool(self.cfg.get("paper_grade", False))
            if paper_grade:
                raise RuntimeError(
                    f"paper-grade write_episode_summary FAILED site={task.site} "
                    f"task={task.task_id}: {write_exc!r}. Fail-loud per B-323; "
                    "in-memory aggregate vs disk evidence split-brain unacceptable "
                    "for paper-grade fire."
                ) from write_exc
            logger.error(
                "Failed to write episode summary for site=%s task=%s: %s",
                task.site, task.task_id, write_exc, exc_info=True,
            )
        logger.info(
            "Episode done site=%s task=%s success=%s steps=%s error=%s",
            task.site, task.task_id, summary.get("success"), summary.get("steps"),
            str(summary.get("error", ""))[:100] or "none",
        )
        # Fire-4 RCA Wave 1 M1 (/stress 3-AI Fire-4 RCA 2026-05-19, A+B+C overlap):
        # paper-grade strict fail-closed on first quarantine event. Pre-fix R1
        # P1-10-B "preserve" semantic was pure tagging — Fire-4 task 75
        # quarantined at 20:48 but runner burned 26 more tasks (76-101) until
        # 4h wallclock kill at 21:46. Now: first `needs_reevaluation=True`
        # summary written under paper_grade=True triggers PaperGradeAbortError
        # → outer task-loop catch writes condition_summary with
        # `condition_aborted=True, aborted_at_task=N, abort_reason="quarantine"`
        # + re-raises so process exits non-zero rc → chain sentinel rc≠0 →
        # master P0-2-B halt. User core complaint "并没有在 75 的时候自动暂停"
        # answered by this gate. Couples with M5 (timeout_callsite taxonomy,
        # Wave 2) to also flag agent-step Page.screenshot timeouts as
        # quarantine — Wave 1 alone catches B-486 evaluator-path quarantines
        # AND the existing exception-path `needs_reevaluation=True` set at
        # the EpisodeSummaryV2 constructor (line 1605, B-486).
        paper_grade = bool(self.cfg.get("paper_grade", False))
        # Fire-6 RCA Stage C2 (/stress 2026-05-20): diagnostic-replay suppresses
        # the M1 fail-closed abort. A targeted `--diagnostic-replay --tasks 4,75`
        # must run BOTH tasks and capture per-task forensics (C1 isolation
        # success/failure proof + `_dump_eval_timeout_forensic`); aborting at
        # task 4's first quarantine would never reach task 75. The error summary
        # is still written with needs_reevaluation=True (forensic preserved) and
        # sr_excluded=True (never enters SR). This is NOT a canonical Gate
        # bypass — it only fires when diagnostic_replay=True, which forces
        # non-canonical output + sr_excluded on every episode.
        if (
            paper_grade
            and not self.diagnostic_replay
            and bool(summary.get("needs_reevaluation", False))
        ):
            # B-1957 (2026-08-05): a standing `benchmark_permanent` adjudication
            # downgrades this abort to an ordinary episode failure. Motivating
            # case: VWA shopping task 345's start_url references a Wikipedia
            # image absent from the deployed ZIM (probed 404 across every path
            # variant while the article itself is 302 — upstream content drift,
            # already adjudicated in §81 follow-up 2026-04-29). Such a task can
            # NEVER produce the clean evidence episode `append_resolution`
            # demands, so without this it aborts every condition forever: the
            # 2026-08-04 fire burned 320 episodes + 20h of idle host before
            # anyone looked.
            #
            # TWO independent conditions must hold, sourced from different
            # places, so neither alone can wave an episode through:
            #   1. the episode is benchmark-noise classified (runner's own
            #      fields) — an evaluator/code_bug/substrate quarantine never
            #      reaches the probe at all;
            #   2. the registry carries a `benchmark_permanent` classification
            #      for THAT exact error_class (operator adjudication).
            # Anything else — probe error, missing registry, timeout, unknown
            # classification — leaves the abort intact (fail-closed).
            _bn_category = str(summary.get("benchmark_noise_category") or "")
            if bool(summary.get("benchmark_noise", False)) and _bn_category:
                _err_class = f"error({_bn_category})"
                _allowed, _why = self._quarantine_downgrade_allowed(
                    str(task.site), int(task.task_id), _err_class
                )
                if _allowed:
                    # B-1961 (/stress Mode B P0-1, 2026-08-06): the flag MUST be
                    # settled here, not left set. Leaving `needs_reevaluation=True`
                    # on a NON-aborted condition means the run reaches episode 435
                    # and then dies inside `aggregate_condition_metrics`
                    # (metrics.py B-784 raise), whose own comment says it is
                    # "UNREACHABLE under paper_grade=True; if it fires here, M1
                    # gate was bypassed" — and B-1957's downgrade is exactly such
                    # a bypass. Pre-fix this converted "abort at task 345" into
                    # "burn the remaining 115 episodes, THEN abort", and the
                    # canonical loader (`analysis.py` reject_needs_reevaluation)
                    # would silently drop the episode, moving the denominator to
                    # 434 outside PROTOCOL_EXCLUSIONS' control — an estimand
                    # change nobody declared.
                    #
                    # Forensics survive: `error` keeps the full message, the
                    # registry holds the adjudication event, and the dedicated
                    # field below marks the episode as adjudicated rather than
                    # clean, so analysis can tell the two apart without parsing
                    # error strings.
                    summary["needs_reevaluation"] = False
                    summary["benchmark_permanent_adjudicated"] = True
                    logger.warning(
                        "B-1957 quarantine DOWNGRADED to ordinary failure "
                        "site=%s task=%s error_class=%s — %s. "
                        "needs_reevaluation CLEARED + "
                        "benchmark_permanent_adjudicated=True (B-1961) so the "
                        "condition can finalize; scored-set treatment stays an "
                        "analysis-layer decision (PROTOCOL_EXCLUSIONS), not a "
                        "runner one.",
                        task.site, task.task_id, _err_class, _why,
                    )
                    try:
                        condition_logger.write_episode_summary(
                            task.site, task.task_id, summary
                        )
                    except Exception as _rewrite_exc:  # noqa: BLE001
                        raise RuntimeError(
                            f"B-1961 failed to re-persist downgraded episode "
                            f"site={task.site} task={task.task_id}: "
                            f"{_rewrite_exc!r}. The in-memory summary and the "
                            "on-disk one would disagree on needs_reevaluation, "
                            "which is the exact disk-vs-memory split-brain B-323 "
                            "fails loud on."
                        ) from _rewrite_exc
                    return summary

            from p79.experiment.environment import PaperGradeAbortError
            # B-1881: attach structured provenance so the transient-retry wrapper
            # gates on (class, steps) without re-parsing the truncated message.
            # steps==0 ⇒ pre-flight (agent took no browser action ⇒ no mutation,
            # no rollout to redraw); >0 ⇒ mid-episode (NOT retryable — possible
            # site mutation / stochastic-rollout redraw per 3-AI B-1881 audit).
            _abort_steps = int(summary.get("steps", 0) or 0)
            _abort_tclass = self._classify_transient_substrate(
                str(summary.get("error", ""))
            )
            raise PaperGradeAbortError(
                f"first quarantine event under paper_grade=True at "
                f"site={task.site} task={task.task_id} condition={effective_cid}: "
                f"summary carries needs_reevaluation=True "
                f"(error={str(summary.get('error', ''))[:200]!r}). "
                f"Condition aborts to prevent burning compute on potentially-"
                f"compromised substrate. Operator: reproduce the failure, then "
                f"`python scripts/maintenance/quarantine_registry.py classify "
                f"--site={task.site} --task-id={task.task_id} --as=<one of "
                f"substrate|agent_induced|evaluator|transient_drift|"
                f"unreproducible_in_isolation|benchmark_permanent> --rationale=...` "
                f"(Wave 2 M6 investigation gate, NOT an auto skip-list), then "
                f"re-fire. Only `benchmark_permanent` (+ its required "
                f"--error-class) lets a future fire past this task without a "
                f"substrate fix — reserve it for defects proven unrepairable.",
                transient_class=_abort_tclass,
                steps=_abort_steps,
            )
        return summary

    def _quarantine_downgrade_allowed(
        self, site: str, task_id: int, error_class: str
    ) -> tuple:
        """B-1957 — does a standing `benchmark_permanent` adjudication cover this?

        Probes the registry through its CLI rather than importing it, keeping
        `scripts/maintenance/` out of the fire's import path (the boundary the
        watchdog already uses). Fail-closed by construction: a missing script,
        a non-zero exit, a timeout, or any exception all return False, so the
        quarantine abort survives every failure mode of this check.
        """
        import subprocess

        registry = (
            Path(__file__).resolve().parents[3]
            / "scripts" / "maintenance" / "quarantine_registry.py"
        )
        if not registry.exists():
            return False, f"registry CLI not found at {registry}"
        try:
            proc = subprocess.run(
                [
                    sys.executable, str(registry), "gate-check",
                    "--site", site,
                    "--task-id", str(task_id),
                    "--error-class", error_class,
                ],
                capture_output=True, text=True, timeout=20,
            )
        except Exception as exc:  # noqa: BLE001 — fail-closed on ANY probe failure
            return False, f"gate-check probe raised: {exc!r}"
        if proc.returncode != 0:
            return False, (
                (proc.stdout or proc.stderr or "no standing adjudication").strip()[:300]
            )
        return True, (proc.stdout or "").strip()[:300]

    def _run_episode(
        self,
        condition: ConditionSpec,
        task: Any,
        backend: Any,
        condition_logger: LoggerV2,
        condition_dir: Path,
        effective_cid: Optional[str] = None,
    ) -> Dict[str, Any]:
        # B-132 (codex F1 fix 2026-05-15): multi-seed schema drift —
        # `condition_dir` and `condition_meta.json` use `effective_cid`
        # (= condition.condition_id + "_seedN" when len(seeds) > 1), but
        # step JSONL + episode summary previously wrote `condition.condition_id`
        # unchanged, causing silent join/aggregate collision when multi-seed
        # mode activates. Default to `condition.condition_id` so single-seed
        # mode (current Phase 1a) is byte-identical; explicit effective_cid
        # threading via `_run_and_record_episode` for multi-seed correctness.
        if effective_cid is None:
            effective_cid = condition.condition_id
        # ── B-1884 / Fix 4: reddit shared-account identity restore ──
        # reddit task 138 ("change my username") renames the shared test
        # account; the username IS the login credential, so the periodic fresh
        # re-login below would then fail-closed (root cause of the 2026-06
        # reddit abort saga). Heal it idempotently HERE — BEFORE the auth-refresh
        # check — so a renamed account is restored before the next re-login.
        # Setup phase (pre-trajectory) → not measured execution; mirrors the
        # per-task require_reset classifieds already gets. Estimand = clean
        # per-task (user decision 2026-06-25). See p79/utils/reddit_identity.py
        # + PROTOCOL_NOTE_04 + 笔记 §357.
        if task.site == "reddit":
            from p79.utils.reddit_identity import restore_reddit_identity
            restore_reddit_identity(self.cfg, logger=logger)
        # ── B-1936 / PROTOCOL_NOTE_07: shopping per-task cart isolation ──
        # Upstream's `require_reset` is a no-op on shopping (envs.py:172 is
        # classifieds-only), so the cart accumulates across a condition and the
        # 104 cart-page `program_html` evaluators drift toward false success.
        # Enabled per user decision 2026-08-03 (PROTOCOL_NOTE_07) — shopping is
        # pre-data, so this defines its estimand rather than changing a measured
        # one. Setup phase, before the measured trajectory, like the reddit
        # restore above.
        #
        # B-1943 (codex Mode B F7): the return value is CHECKED. Discarding it
        # meant a failed clear (missing container, timeout, schema drift, wrong
        # customer identity) degraded to a log line while the condition carried
        # on over an accumulating cart — indistinguishable from a clean run in
        # every summary. Under paper-grade the helper is forced fail-closed so
        # the episode aborts loudly instead.
        if task.site in ("shopping", "shopping_admin"):
            from p79.utils.shopping_cart_reset import clear_shopping_cart
            if not clear_shopping_cart(self.cfg, logger=logger):
                logger.error(
                    "shopping_cart_reset FAILED before task %s — cart isolation is "
                    "part of the shopping protocol (PROTOCOL_NOTE_07); this episode "
                    "would run on an unknown cart state",
                    getattr(task, "task_id", "?"),
                )
        # ── Auth refresh check (before browser context creation) ──
        site = task.site
        self._auth_episode_counts.setdefault(site, 0)
        self._auth_episode_counts[site] += 1
        # B-35 fix: pass time-since-last-refresh so refresh fires before PHP
        # session.gc_maxlifetime (~1440s) expires mid-long-episode.
        _now = time.time()
        _last = self._auth_last_refresh_ts.get(site, _now)
        _seconds_since = _now - _last
        if should_refresh(
            site,
            self._auth_episode_counts[site],
            self.cfg,
            seconds_since_refresh=_seconds_since,
        ):
            # __file__ = p79/experiment/runner/main.py → repo root needs 4 .parent
            # (runner → experiment → p79 → REPO_ROOT). Bug fix 2026-04-26.
            auth_dir = Path(__file__).resolve().parent.parent.parent.parent / ".auth"
            benchmark = self.cfg.get("experiment", {}).get("benchmark", "")
            # B-220 fix (2026-05-16, A1.5 Item 19): replace warning-only with
            # auth_required_gate. Pre-fix: refresh failure → logger.warning +
            # continue → NOT-LOGGED-IN session lands in step_record + condition_summary.
            # Post-fix: AuthRefreshFailure raised → episode aborts; outer
            # `_run_episode_safe` catches it (sentinel-style raise per
            # PaperGradeAbortError contract below) → condition records the
            # episode as auth-aborted, watchdog picks up via state + retries.
            try:
                from p79.utils.auth_refresh import auth_required_gate, AuthRefreshFailure
                auth_required_gate(site, auth_dir, benchmark=benchmark)
                self._auth_episode_counts[site] = 0
                self._auth_last_refresh_ts[site] = _now
                logger.info("Auth gate passed for %s (seconds_since=%.0f)", site, _seconds_since)
            except AuthRefreshFailure as _exc:
                # Record + propagate so outer episode-safe wrapper logs +
                # surfaces to condition_summary. Watchdog state will see the
                # pattern + retry the condition after backoff.
                logger.error(
                    "Auth gate FAILED for %s — NOT proceeding with episode "
                    "(paper-grade contamination prevention): %s",
                    site, _exc,
                )
                raise

        episode_dir = condition_dir / "artifacts" / f"{task.site}_task_{task.task_id}"
        stale_jsonl = condition_logger.step_log_path(task.site, task.task_id)

        # B-488 (/stress A1.5b Phase 1 P0-4-C gemini OOB, 2026-05-17):
        # in-progress-aware archive. Pre-fix unconditional `shutil.rmtree`
        # at entry destroyed B-222 watchdog-preserved forensic — when runner
        # crashes mid-episode the `.in_progress` marker survives, watchdog
        # orphan cleanup correctly skips, but the next `_run_episode` entry
        # would wipe everything. Now: if marker present (runner-crash recovery
        # path) archive episode_dir + step JSONL to `.stale_<ts>` sibling
        # preserving forensic for B-168 partial recovery + paper §3
        # "Restart-resilient trajectory logging" claim. If marker absent
        # (watchdog auto-clean retry path already wiped everything, or fresh
        # task) wipe as before. Watchdog (`experiment_watchdog.py:1397+1413`)
        # MUST also skip `.stale_*` archives from orphan cleanup (B-488
        # companion patch).
        _has_in_progress = (
            episode_dir.exists() and (episode_dir / ".in_progress").exists()
        )
        if _has_in_progress:
            _ts = int(time.time())
            _stale_archive = episode_dir.parent / f"{episode_dir.name}.stale_{_ts}"
            try:
                episode_dir.rename(_stale_archive)
                if stale_jsonl.exists():
                    _jsonl_stale = stale_jsonl.with_name(
                        stale_jsonl.stem + f".stale_{_ts}" + stale_jsonl.suffix
                    )
                    stale_jsonl.rename(_jsonl_stale)
                logger.info(
                    "B-488 archived stale episode forensic for %s task %s → "
                    "%s (.in_progress marker present, runner-crash recovery path)",
                    task.site, task.task_id, _stale_archive.name,
                )
            except OSError as _archive_exc:
                logger.warning(
                    "B-488 archive rename FAILED for %s task %s (falling back "
                    "to wipe so runner can proceed): %s",
                    task.site, task.task_id, _archive_exc,
                )
                if episode_dir.exists():
                    shutil.rmtree(episode_dir, ignore_errors=True)
                if stale_jsonl.exists():
                    try:
                        stale_jsonl.unlink()
                    except OSError:
                        pass
        else:
            # Watchdog auto-clean already wiped OR fresh task — original
            # behavior preserved.
            if episode_dir.exists():
                shutil.rmtree(episode_dir)
                logger.info(
                    "Cleared stale artifacts for %s task %s",
                    task.site, task.task_id,
                )
            if stale_jsonl.exists():
                stale_jsonl.unlink()
                logger.info(
                    "Cleared stale step JSONL for %s task %s",
                    task.site, task.task_id,
                )

        episode_dir.mkdir(parents=True, exist_ok=True)
        # B-222 (2026-05-16, A1.5 Item 6): in-progress marker for watchdog
        # orphan-cleanup safety. Watchdog will skip pruning this artifact dir
        # while the marker file is present (see experiment_watchdog.py
        # orphan-cleanup block). Marker removed by post-episode cleanup below.
        try:
            (episode_dir / ".in_progress").touch()
        except OSError:
            pass  # filesystem ro / mkdir race; runner can still proceed

        obs, info = self.environment.reset(task.config_file)
        current_info = info or {}

        # Fire-6 RCA Stage C1b (/stress 2026-05-20): episode-start default for
        # the screenshot-timeout fatality gate (VwaWrapper._gate_screenshot_timeout).
        # condition.observation_mode covers all FIXED modes (dom→non-fatal blank
        # recovery; som/vision/phantom_*→fatal) + the wait/reset paths. For
        # router_on / learned the per-step refinement below overrides to the
        # effective decision_mode. setattr-guarded for MockEnv (no attr in tests).
        if hasattr(self.environment, "observation_mode"):
            self.environment.observation_mode = condition.observation_mode

        # ── v7 learned router runtime dispatch (paper-1 §6 LR predictor) ─
        # When condition.observation_mode == "learned" (sentinel from
        # conditions.py v7 phase1.router_kind=learned), load the trained LR
        # pickle once per condition + predict per-task mode from features.
        # The predicted mode replaces condition.observation_mode for THIS
        # episode only; downstream step JSONL records the predicted mode for
        # paper-grade tracking. Fallback to safe_fallback_target on any error.
        #
        # B-693 (/stress A1.7 cold-start P0-3-C, 2026-05-17): the pre-fix
        # block lacked a try/except wrapper. `load_lr_pipeline()` was the
        # raw call point — a corrupt .pkl / numpy version mismatch / file
        # permission error would propagate all the way to runner.run() and
        # nuke the entire Pass-2 router cell (6 conditions × cls + red ×
        # 3 baselines = 36 condition fires depending on phase1.variant
        # config). `predict_mode()` itself had an internal try/except for
        # the model.predict() call (learned_router.py:120) but NOT for
        # feature extraction failures (load_task_image_field reading
        # malformed VWA task JSON). Mode C gemini caught this as P0-3
        # ("LR 运行时分发缺乏防御性异常处理") with the cross-AI defuse:
        # wrap the entire LR dispatch block; log.error + count fallbacks
        # so reviewer can audit fallback rate from runner logs per cell.
        # Single-fire ntfy push on first fallback per condition gives
        # user real-time alarm without spamming.
        if condition.observation_mode == "learned":
            # ── A2.5 Chunk C: fold-aware learned router dispatch ──
            # Q1=C + (E''') + (b) design: within-cell 5-fold CV deployment.
            # For each task, lookup the fold that held it out at training,
            # apply fold-k vectorizer + selected_idx + LR + τ_{C,k} threshold.
            # See p79/policies/learned_router.py:predict_mode_fold_aware.
            #
            # B-1645 (/stress A2.10 P1-6-A 2026-05-18): infrastructure-level
            # errors (missing artifact / corrupt pickle / dim mismatch / no
            # fold_assignment entry) now raise LearnedRouterArtifactError per
            # B-1640 hard-fail mandate; this wrapper PROPAGATES that error
            # (no silent phantom_som fallback for infrastructure failures).
            # Only task-level signal-strength fallback (max_prob ≤ τ +
            # candidate_modes filter) stays silent + counted.
            from p79.policies.learned_router import LearnedRouterArtifactError

            router_cfg = self.cfg.get("router", {})
            safe_fallback = str(router_cfg.get("safe_fallback_target", "phantom_som"))
            cell_id = str(router_cfg.get("cell_id", ""))
            artifacts_dir = router_cfg.get(
                "artifacts_dir", "results/phantom_paper/l1_router"
            )
            candidate_modes = router_cfg.get("candidate_modes", [])

            # B-1645: hard-fail at config-validation time before any dispatch.
            # Empty cell_id → all artifact filenames missing prefix → silent
            # fallback to phantom_som per pre-B-1640 behavior.
            if not cell_id:
                raise LearnedRouterArtifactError(
                    "[learned router] router.cell_id config field is empty — "
                    "Pass-2 paper-grade fire requires explicit per-cell config "
                    "(e.g. cell_id: \"B0_classifieds\" in "
                    "configs/exp_v2_<baseline>_router_learned_<site>.yaml). "
                    "Hard-fail per /stress A2.10 P0-3-B / P1-6-A user-mandate "
                    "2026-05-18 (NO silent phantom_som fallback for "
                    "infrastructure errors)."
                )

            predicted_mode: Optional[str] = safe_fallback
            fold_diag: dict = {"fallback_fired": False, "fallback_reason": "not_dispatched"}

            # Increment dispatch counter (every attempt counts, regardless of
            # signal-strength outcome). Pairs with `_lr_fallback_count` for
            # per-cell `learned_router_diag.fallback_rate` disclosure.
            self._lr_dispatch_count = getattr(self, "_lr_dispatch_count", 0) + 1

            try:
                from p79.policies.learned_router import (
                    difficulty_to_int,
                    estimate_input_tokens,
                    extract_raw_features,
                    load_task_image_field,
                    predict_mode_fold_aware,
                )

                # Extract task-config features
                task_intent = (
                    task.raw_task.get("intent", "")
                    if hasattr(task, "raw_task") else ""
                )
                task_has_image = load_task_image_field(task.config_file)
                # F1 (B-1805): VWA stores reasoning_difficulty as an ordinal string
                # ("easy"/"medium"/"hard"). The old `int(...)` raised ValueError and
                # the bare `except: pass` silently zeroed it → serve saw 0 while train
                # saw the real value (train/serve skew). difficulty_to_int is the same
                # mapping the extractor uses; only file/JSON errors fall through now.
                reasoning_difficulty = 0
                try:
                    with open(task.config_file) as _cfg_f:
                        _cfg = json.load(_cfg_f)
                    reasoning_difficulty = difficulty_to_int(
                        _cfg.get("reasoning_difficulty")
                    )
                except (OSError, json.JSONDecodeError) as _diff_err:
                    logger.warning(
                        "[learned router] could not read reasoning_difficulty "
                        "from %s: %s; using 0", task.config_file, _diff_err
                    )

                # Step-0 obs features (mode-agnostic DOM-style at env.reset return)
                dom_complexity = (obs.text or "").count("\n") + 1 if obs.text else 0
                text_length = len(obs.text) if obs.text else 0
                # F4 (B-1817): char//4 estimate — the SAME estimate_input_tokens the
                # train extractor uses, so this feature is identical train ≡ serve.
                tokens_input_text = estimate_input_tokens(text_length)

                raw_features = extract_raw_features(
                    intent=task_intent,
                    has_reference_image=task_has_image,
                    dom_complexity=dom_complexity,
                    text_length=text_length,
                    tokens_input_text=tokens_input_text,
                    reasoning_difficulty=reasoning_difficulty,
                )

                predicted_mode, fold_diag = predict_mode_fold_aware(
                    cell_id=cell_id,
                    task_id=task.task_id,
                    artifacts_dir=artifacts_dir,
                    cache=self._lr_router_cache,
                    raw_features=raw_features,
                    fallback_mode=safe_fallback,
                )

                # B-696 (preserved from pre-Chunk-C) candidate_modes sanity check.
                # This is a runtime filter (not infrastructure), so silent-
                # fallback-with-counter is the correct semantic.
                if candidate_modes and predicted_mode not in candidate_modes:
                    logger.warning(
                        "[learned router Chunk C] predicted mode=%s NOT in "
                        "candidate_modes=%s; falling back to %s",
                        predicted_mode, candidate_modes, safe_fallback,
                    )
                    predicted_mode = safe_fallback
                    fold_diag["candidate_modes_filter_fired"] = True
                    self._lr_fallback_count = (
                        getattr(self, "_lr_fallback_count", 0) + 1
                    )

                if fold_diag.get("fallback_fired"):
                    self._lr_fallback_count = (
                        getattr(self, "_lr_fallback_count", 0) + 1
                    )

                logger.info(
                    "[learned router Chunk C] cell=%s task_id=%s fold_k=%s "
                    "tau=%s max_prob=%s predicted=%s fallback_fired=%s reason=%s",
                    cell_id, task.task_id,
                    fold_diag.get("fold_k_used"),
                    fold_diag.get("tau_used"),
                    fold_diag.get("max_prob"),
                    predicted_mode,
                    fold_diag.get("fallback_fired"),
                    fold_diag.get("fallback_reason"),
                )
            except LearnedRouterArtifactError:
                # B-1645 (/stress A2.10 P1-6-A 2026-05-18): re-raise
                # infrastructure-level errors per user-mandate hard-fail.
                # NO silent phantom_som fallback for missing/corrupt artifact —
                # the entire cell run dies loudly so user diagnoses immediately
                # rather than at §6 aggregator weeks later.
                if not getattr(self, "_lr_fallback_ntfy_fired", False):
                    self._lr_fallback_ntfy_fired = True
                    try:
                        # B-1835 (2026-05-22): NO local `import os` here — module-level
                        # `import os` (line 13) already provides it. A local import made
                        # `os` function-local for ALL of _run_episode → the deferred
                        # image-save `os.open` (B-1832, line ~3151) hit
                        # UnboundLocalError whenever this rarely-run LR-fallback block
                        # did NOT execute (the common case). B-1832's .tmp fix unmasked
                        # it (PIL save now succeeds → reaches os.open → crash → caught
                        # as warning → images STILL lost). Only urllib.request stays local.
                        import urllib.request
                        topic = os.environ.get("NTFY_TOPIC", "")
                        if topic:
                            msg = (
                                f"[A2.10 LR HARD-FAIL] LearnedRouterArtifactError "
                                f"condition={condition.condition_id} "
                                f"task={task.site}/{task.task_id} cell={cell_id}"
                            )
                            urllib.request.urlopen(
                                f"https://ntfy.sh/{topic}",
                                data=msg.encode("utf-8"),
                                timeout=3,
                            )
                    except Exception:
                        pass  # ntfy is best-effort; never let it break the cell
                raise
            except Exception as exc:
                # Non-infrastructure runtime exception (e.g., feature extraction
                # failed on malformed task config). Silent fallback retained per
                # B-693 robustness; counted for transparency.
                self._lr_fallback_count = (
                    getattr(self, "_lr_fallback_count", 0) + 1
                )
                logger.error(
                    "[learned router Chunk C] non-infrastructure fallback "
                    "for task=%s/%s — exception=%s; using safe_fallback=%s; "
                    "lr_fallback_count_so_far=%d",
                    task.site, task.task_id, repr(exc), safe_fallback,
                    self._lr_fallback_count,
                )
                if not getattr(self, "_lr_fallback_ntfy_fired", False):
                    self._lr_fallback_ntfy_fired = True
                    try:
                        # B-1835 (2026-05-22): NO local `import os` here — module-level
                        # `import os` (line 13) already provides it. A local import made
                        # `os` function-local for ALL of _run_episode → the deferred
                        # image-save `os.open` (B-1832, line ~3151) hit
                        # UnboundLocalError whenever this rarely-run LR-fallback block
                        # did NOT execute (the common case). B-1832's .tmp fix unmasked
                        # it (PIL save now succeeds → reaches os.open → crash → caught
                        # as warning → images STILL lost). Only urllib.request stays local.
                        import urllib.request
                        topic = os.environ.get("NTFY_TOPIC", "")
                        if topic:
                            msg = (
                                f"[A2.5 Chunk C] LR fallback fired "
                                f"condition={condition.condition_id} "
                                f"task={task.site}/{task.task_id} "
                                f"exc={repr(exc)[:200]}"
                            )
                            urllib.request.urlopen(
                                f"https://ntfy.sh/{topic}",
                                data=msg.encode("utf-8"),
                                timeout=3,
                            )
                    except Exception:
                        pass  # ntfy is best-effort; never let it break the cell
                predicted_mode = safe_fallback
            # Derive per-task condition with predicted observation_mode. Keep
            # condition_id stable (analysis groups by condition_id, not mode)
            # so per-task mode predictions are recorded via step JSONL
            # observation_mode field (set inside the router decision below).
            from dataclasses import replace as _dc_replace
            condition = _dc_replace(
                condition,
                observation_mode=predicted_mode,
                som_on=(predicted_mode == "som"),
            )

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
        prev_action_intent_fulfilled: Optional[bool] = None   # B-1891

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
        # Advisor 5/5 sync (笔记 §110, advisor_sync_5_5_outcomes.md §A.1):
        # Option A — early-stop CANCELLED entirely. Detection logic kept for
        # paper-grade diagnostic logging but trajectory NOT terminated.
        # Default False (per advisor cancel); re-enable only with explicit
        # config opt-in for ablation studies.
        _early_stop_enabled = bool(
            self.cfg.get("runtime", {}).get("early_stop_enabled", False)
        )

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
        # B-1832 P2-1b artifact gate (3-AI /stress 2026-05-22, codex F4): per-episode
        # image-save outcome counters. The B-1828/B-1830 deferred save (post-timing)
        # increments these; a som/vision episode that ATTEMPTED saves but landed 0
        # (fail>0, ok==0) is the B-1832 total-loss signature → loud once-per-episode
        # ERROR after the step loop (replaces the silent per-step warning flood that
        # let B-1832 run 105 ep with zero artifacts undetected).
        _img_save_ok = 0
        _img_save_fail = 0
        # Track total wall time spent in free busy-page waits (RU-4): these
        # don't consume a step but still count toward end-to-end episode time.
        busy_wait_total_ms = 0.0
        # Protocol Reset #6/#7 (§244 canonical, 2026-05-20): two-budget loop
        # counters. PRIMARY = agent_action_count (only valid/consuming steps);
        # SAFETY = step_idx (model-call ceiling) + parse-error caps. step_idx
        # counts persisted steps (= model calls; busy-waits `continue` without
        # incrementing), so `step_idx < max_model_attempts` bounds LLM calls.
        agent_action_count = 0
        valid_action_count = 0
        consecutive_parse_errors = 0
        total_parse_errors = 0
        # B-1800 (P1-6): hoist the per-episode token-cost config (constant — depends
        # only on backend type) so the crash-atomic billed-cost ledger can value the
        # in-flight model spend immediately after backend.step, before the
        # crash-prone env.step / obs / energy / evaluator processing downstream.
        _token_cost_cfg = select_token_cost_cfg(
            metrics_cfg=self.cfg.get("metrics", {}),
            backend_type=self.cfg.get("backends", {}).get(condition.backend_id, {}).get("type"),
        )
        # B-1860: Qwen 0-1000 contract. The parse-error caps below
        # (consecutive_parse_errors / total_parse_errors) MUST count only
        # genuinely-malformed steps, NOT a 0-1000 coordinate format mismatch.
        # This invariant is enforced upstream, not here: a 0-1000 coord now
        # passes `validate_action` (action_utils._is_valid_coordinate_pair
        # by-value check) → runner_valid_post_backend=True → parse_valid=True →
        # classify_step_accounting returns is_injected_wait_sink=False → the
        # counters at L3107-3113 do NOT increment. Only a coord that is STILL
        # malformed after that by-value judgment (NaN / inf / wrong shape /
        # negative) trips parse_valid=False and feeds these caps. Pre-fix, a
        # 0-1000 coord declared "normalized" was hard-rejected → 13.6% of
        # vision steps counted as parse errors → 48% of episodes cap-killed.
        while (
            agent_action_count < self.max_agent_actions
            and step_idx < self.max_model_attempts
            and consecutive_parse_errors < self.max_consecutive_parse_errors
            and total_parse_errors < self.max_total_parse_errors
        ):
            step_start = time.time()
            # B-321 (/stress A1.9 Mode A F2 OOB, 2026-05-16): capture monotonic
            # step boundary so EnergyTracker can strictly bound pynvml sample
            # window to inference period. Wall-clock `step_start` above can
            # drift on NTP sync; monotonic is the only reliable boundary for
            # window arithmetic.
            step_start_monotonic = time.monotonic()

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

            # /stress A1.10 P0-4-B* (2026-05-16): learned-router cells skip
            # the rule-based router decision step. Pre-fix: learned router
            # `_dc_replace`d `condition.observation_mode` with the LR-predicted
            # mode at episode start (lines ~1082-1087 above), but the
            # rule-based `RuleBasedRouter.decide()` then still ran with
            # `router_enabled=condition.router_on=True` and could *re-pick*
            # the mode mid-episode via streak/threshold escalation. Result:
            # paper §6 learned-router cells reported a hybrid (LR + rule)
            # policy rather than pure LR-on-task oracle validation, breaking
            # the H10 Pareto non-dominance attribution. Post-fix: when the
            # condition is the v7 learned-router cell, mark routing as
            # already-decided (decision_mode = condition.observation_mode =
            # the LR prediction landed at episode start), emit a single
            # learned-route trigger, and zero out router overhead.
            _is_learned_cell = (
                condition.metadata.get("router_variant") == "v7_learned"
                if condition.metadata
                else False
            )
            if _is_learned_cell:
                decision_mode = condition.observation_mode
                triggers = ["v7_learned_route"]
                overhead = {
                    "router_decision_ms": 0.0,
                    "extra_dom_parse_ms": 0.0,
                    "extra_screenshot_ms": 0.0,
                    "extra_model_calls": 0.0,
                    "routing_retry_count": 0.0,
                    "rule_router_skipped": 1.0,
                }
            else:
                decision_mode, triggers, overhead, router_state = self.router.decide(
                    router_enabled=condition.router_on,
                    preferred_mode=condition.observation_mode,
                    obs_text=router_obs_text,
                    state=router_state,
                    prev_action_success=prev_action_success,
                    prev_page_changed=prev_page_changed,
                    prev_action_intent_fulfilled=prev_action_intent_fulfilled,
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
            # Sequential SoM identifier contract (2026-05-25): push the seq-keyed
            # dispatch map to the env for SoM-family modes (som / phantom_som /
            # phantom_text) so that click/type/hover [seq] resolve to the right
            # element. Use `is not None` (NOT truthiness): {} means "SoM-family,
            # zero marks" → override with empty so a stray id-click fails closed.
            # AXTree modes (dom / p-prompt / vision) leave obs_nodes_info_seq=None
            # → the env keeps its native nodeId-keyed map. Router-correct: this is
            # keyed on the per-step decision_mode's obs_prep.
            if obs_prep.obs_nodes_info_seq is not None:
                self.environment.set_dispatch_obs_nodes_info(obs_prep.obs_nodes_info_seq)
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
            # B-1828 P0-1: SoM marked-image disk save is DEFERRED to after
            # total_latency_ms + energy are computed (see post-energy block below)
            # so the inspection write never enters the latency/carbon window.
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
            # B-1800 (P1-6, /stress 2026-05-21 codex Mode B OOB): the model call
            # above is already billed. Stamp its spend into the instance ledger NOW
            # — the step record is not written until ~960 lines below, and any
            # exception in between (env.step / screenshot / energy / evaluator) would
            # otherwise lose this billed cost + model attempt from the crash-recovery
            # summary (which reads only written JSONL). The ledger is the
            # authoritative §1 billed total on the except path; on the happy path it
            # is ignored (the summary uses the written step records).
            self._episode_inflight_model_attempts += 1
            self._episode_inflight_billed_usd += compute_token_cost(
                input_tokens=int(meta.get("input_tokens") or 0),
                output_tokens=int(meta.get("output_tokens") or 0),
                cost_cfg=_token_cost_cfg,
            )["total"]

            # B-134 (/stress A1.1 v8 codex F3, 2026-05-15): save runner-side
            # validate_action result instead of discarding the bool. When
            # backend returns a malformed action (e.g. unknown action_type
            # like "clik"), validate_action rescues to {"action_type":"wait"}
            # but previously runner threw away the bool, leaving JSONL with
            # parse_valid=True / parse_failure_reason=None. That split the
            # schema source-of-truth between "agent's self-reported validity"
            # and "what runner actually executed", silently inflating
            # parse_success rate across baselines. Now record both and emit
            # a dedicated failure_reason so failure taxonomy can distinguish
            # agent-side invalid from runner-rescued no-ops.
            #
            # B-552 (/stress A1.5 P1-2-AB* Claude+codex OOB, 2026-05-17):
            # snapshot the agent's RAW action emit BEFORE validate_action
            # mutates it (rescue path turns "clik" → wait). Pre-fix the only
            # snapshot of pre-mutation action was `_control_original_action`
            # at L1865, which captured action POST-validate_action and
            # PRE-diagnostic-control — so "agent self-emission" in paper §3
            # taxonomy was muddled (rescued actions read as agent-emit).
            # Recording `_agent_raw_action` separately lets paper §3
            # taxonomy properly distinguish:
            #   - raw_action: literal backend emit (subject to rescue)
            #   - control_intervention.original_action: post-validate +
            #     pre-control (what backend got after rescue)
            #   - action: what executed (post-validate + post-control)
            _agent_raw_action = dict(action) if isinstance(action, dict) else None
            action, runner_valid_post_backend = validate_action(action)
            # B-425: apply_secondary_modules (M1/M2) call retired — 0/53924
            # archive rows had m1_dom_select_fallback or m2_dom_first_input
            # set to True (codex Mode B numeric receipts). The functions are
            # restorable from git history when paper-2 module ablation
            # resumes.
            # B-546 (/stress A1.5b Phase 2 P1-6-AB Claude F2 + codex B-541
            # cross-validation, 2026-05-17): control_intervention write path.
            # Pre-fix `_anti_repeat_control` / `_no_early_finish_control` /
            # `_query_sanitization_control` fired `diag_notes` strings to
            # logger.info but NEVER wrote to step_record. Phase 1 B-497
            # added the schema (`types.py:control_intervention` +
            # `STEP_RECORD_V2_DEFAULTS`) + PAPER_GRADE_STEP_OPTIONAL_KEYS
            # but left runtime path as "Phase 2 audit slot". codex Mode B
            # empirical spot-check verified: corrupted-archive grep over
            # `results/visualwebarena/phase1/*/*/episodes/*.jsonl` found
            # `hits=0` for `control_intervention`. Schema fake → paper §3
            # disclosure unsubstantiated.
            # Snapshot the action AFTER validate_action rescue but BEFORE
            # diagnostic controls fire. paper §3 action taxonomy can then
            # distinguish 3 layers via step_record:
            #   (a) raw_action: literal backend emit (B-552, may be invalid)
            #   (b) control_intervention.original_action: post-validate +
            #       pre-control (what backend got after rescue, but before
            #       any diagnostic-control mutation)
            #   (c) action: post-validate + post-control (what executed)
            # B-552 clarification 2026-05-17: comment previously claimed
            # `original_action` = "pre-control agent self-emitted", but
            # validate_action already mutated by here. Three-layer model
            # restores faithful semantic.
            _control_original_action = dict(action) if isinstance(action, dict) else None
            _control_fires: List[Dict[str, Any]] = []
            if bool(self.diagnostic_controls.get("enabled", False)):
                diag_notes: List[str] = []
                query_cfg = self.diagnostic_controls.get("query_sanitization", {}) or {}
                anti_repeat_cfg = self.diagnostic_controls.get("anti_repeat", {}) or {}
                no_early_finish_cfg = self.diagnostic_controls.get("no_early_finish", {}) or {}

                if bool(query_cfg.get("enabled", False)):
                    action, note = _query_sanitization_control(action, query_cfg)
                    if note:
                        diag_notes.append(note)
                        _control_fires.append({"type": "query_sanitization", "reason": note})
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
                        _control_fires.append({"type": "anti_repeat", "reason": note})
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
                        _control_fires.append({"type": "no_early_finish", "reason": note})
                # B-134: same bool-save pattern after diagnostic-controls
                # mutation. If diagnostic controls mutate to an invalid
                # action, the runner-rescue must be visible in failure
                # taxonomy.
                action, runner_valid_post_diag = validate_action(action)
                runner_valid_post_backend = runner_valid_post_backend and runner_valid_post_diag
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

            # Fire-6 RCA Stage C1b: refine the screenshot-timeout fatality gate
            # to the EFFECTIVE per-step mode (decision_mode = predicted mode for
            # learned, escalated mode for rule-based router, else condition mode)
            # so the dom-only non-fatal recovery tracks the actual mode the
            # observation built by this env.step() will be consumed as.
            if hasattr(self.environment, "observation_mode"):
                self.environment.observation_mode = decision_mode
            env_step_start = time.time()
            next_obs, reward, terminated, truncated, next_info = self.environment.step(action)
            env_step_ms = (time.time() - env_step_start) * 1000.0
            # B-440 (/stress A1.25 P0-2-B* codex OOB, 2026-05-17): snapshot
            # primary action's locator-route + select_option meta BEFORE the
            # baseline retry block can overwrite `next_info`. Without this,
            # any step that triggered baseline_retry_on_no_progress had its
            # primary dispatch evidence silently deleted from the step
            # record. Paper §3 ON_TARGET denominator was biased; cross-
            # baseline asymmetry contaminated by per-baseline retry-trigger
            # rate. Companion fields written into step_record below.
            _primary_locator_route_meta = (
                next_info.get("locator_route_meta") if isinstance(next_info, dict) else None
            )
            _primary_select_option_meta = (
                next_info.get("select_option_meta") if isinstance(next_info, dict) else None
            )
            # B-547 (/stress A1.5b Phase 2 P1-7-AB Claude F3 + codex B-542
            # cross-validation, 2026-05-17): same retry-overwrite hole closure
            # for `dialog_meta` (B-509 misclick blast-radius evidence) and
            # `runtime_sleep_ms` (B-510 wrapper-level settle-tax). Pre-fix:
            # primary action triggered a confirm dialog (e.g. click delete)
            # but retry was scroll (no dialog) → primary's dialog signal
            # silently dropped from JSONL. Cross-baseline confound: B0 235B
            # rarely fires baseline_retry → cleaner dialog/sleep trail;
            # B1/B2 4B frequently fires retry → systematically biased trail.
            # Snapshot primary now; step_record below writes primary +
            # retry + backward-compat (read post-retry) for paper §3.5.1.
            _primary_dialog_meta = (
                next_info.get("dialog_meta") if isinstance(next_info, dict) else None
            )
            _primary_runtime_sleep_ms = (
                int(next_info.get("runtime_sleep_ms", 0) or 0)
                if isinstance(next_info, dict) else 0
            )
            # B-512 (/stress A1.5b Phase 2 P0-1-C gemini OOB, 2026-05-17):
            # wrapper-normalized action form snapshot. Same retry-overwrite
            # caution — retry's action_executed (if retry was scroll) would
            # overwrite primary's. Currently retry actions are scroll/click/
            # wait so action_executed is only set on scroll-retry paths;
            # snapshot for symmetry with B-440/B-547 hole-closure pattern.
            _primary_action_executed = (
                next_info.get("action_executed") if isinstance(next_info, dict) else None
            )

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
            # P0-1-ABC* (/stress Phase 0 unified bug list 2026-05-19, 3-AI overlap):
            # about:blank recovery is a RUNNER intervention (navigate_to(start_url)),
            # NOT agent action. Pre-fix append "about_blank_recovery" to
            # page_change_reasons → page_changed=True silently credits runner's
            # navigation as agent progress, polluting paper §3 step semantics
            # (no_progress denominator, action_success terminate path,
            # baseline_retry trigger gate, paper §1/§3 SR-step aggregate).
            # Post-fix: page_changed boolean reflects AGENT action only;
            # about_blank_recovery tracked separately as runner intervention.
            # Phase 2 telemetry adds step-level intervention_type + episode-level
            # runner_intervention_count + about_blank_recovery_count.
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
            # B-425 (/stress A1.3 v9 D1, 2026-05-17): M3 module retired (0/53924
            # archive rows had m3_failure_trigger_retry True). The baseline
            # retry-on-no-progress path is paper-grade preserved; trigger
            # condition simplified accordingly.
            baseline_retry_on_no_progress = bool(
                self.cfg.get("runtime", {}).get("baseline_retry_on_no_progress", False)
            )
            trigger_baseline_retry = (
                baseline_retry_on_no_progress
                and (not action_success)
                and (not page_changed)
                and action_type_lower in ("click", "type")
                and retry_count < retry_limit
            )
            retry_was_applied = False
            retry_action_type_str: Optional[str] = None
            if trigger_baseline_retry:
                # B-425: inline retry action generator (was: m3_retry_action in
                # p79/experiment/modules.py, file deleted).
                _failed_type = action_type_lower
                if _failed_type == "click":
                    retry_action = {
                        "action_type": "scroll",
                        "delta": [0, 0.5],
                        "coordinate_type": "normalized",
                        "thought": "Baseline retry: click failed, scroll down to reveal target.",
                    }
                elif _failed_type == "type":
                    _eid = first_element_id_by_keyword(
                        obs.text or "", ("textbox", "input", "search", "edit")
                    )
                    if _eid is not None:
                        retry_action = {
                            "action_type": "click",
                            "element_id": int(_eid),
                            "thought": "Baseline retry: type failed, click input field to focus.",
                        }
                    else:
                        retry_action = {
                            "action_type": "scroll",
                            "delta": [0, 0.3],
                            "coordinate_type": "normalized",
                            "thought": "Baseline retry: type failed, no input found, scroll to reveal.",
                        }
                else:
                    retry_action = {
                        "action_type": "wait",
                        "thought": "Baseline retry: brief wait.",
                    }
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
                        # B-425: M3 retry tag dropped (M3 module retired); only
                        # baseline retry path remains.
                        retry_tag = "baseline_no_progress_retry_applied"
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
                cost_cfg=_token_cost_cfg,  # B-1800: hoisted above (was inline select)
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
            # B-134 (/stress A1.1 v8 codex F3, 2026-05-15): parse_valid is
            # AND of agent self-report AND runner-side validate_action bool.
            # If backend reported valid=True but runner had to rescue to
            # wait (action_type unknown / coordinate missing / etc), this
            # is a "runner_invalid_action" failure mode — explicit in the
            # taxonomy, not silently absorbed into "valid wait action".
            agent_parse_valid = bool(meta.get("valid", True))
            parse_valid = agent_parse_valid and runner_valid_post_backend
            failure_reason = meta.get("failure_reason")
            if not runner_valid_post_backend and agent_parse_valid:
                # Backend self-reported valid but runner found malformed
                # action; surface as its own failure mode.
                failure_reason = "runner_invalid_action"
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

            # Protocol Reset #6/#7 (§244 canonical, 2026-05-20): two-budget
            # classification (see metrics.classify_step_accounting for the rule).
            _goto_blocked = (safe_next_info.get("goto_meta") or {}).get("allowed") is False
            if _goto_blocked:
                # B-1782: off-site goto refused by the VWA-origin whitelist →
                # explicit policy category, not generic no_progress/invalid_action.
                error_category = "policy_blocked_offsite"
            _acct = classify_step_accounting(
                parse_valid=parse_valid,
                action_type=action_type_lower,
                goto_blocked=_goto_blocked,
            )
            valid_agent_action = _acct["valid_agent_action"]
            consumes_agent_action_budget = _acct["consumes_agent_action_budget"]
            counts_as_runner_iteration = _acct["counts_as_runner_iteration"]
            if consumes_agent_action_budget:
                agent_action_count += 1
            if valid_agent_action:
                valid_action_count += 1
            if _acct["is_injected_wait_sink"]:
                total_parse_errors += 1
                consecutive_parse_errors += 1
            else:
                # Genuine action OR policy-blocked goto = agent produced parseable
                # output → reset the consecutive-unparseable safety streak.
                consecutive_parse_errors = 0

            total_latency_ms = (time.time() - step_start) * 1000.0
            # B-321: pass step_start_monotonic for strict pynvml sample window.
            energy = self.energy_tracker.estimate_step(
                duration_seconds=total_latency_ms / 1000.0,
                step_start_monotonic=step_start_monotonic,
            )

            # B-1828 P0-1 + B-1830 (2026-05-22, 3-AI /stress): write inspection
            # images to disk HERE — AFTER total_latency_ms AND energy are computed
            # — so the PNG writes enter NEITHER the latency canonical
            # (total_minus_retry) NOR the energy/carbon window. Two images:
            #  (1) som's marked image (P0-1): bbox DRAW stays in obs_prepare
            #      (model-essential, already cloned into obs_for_backend); only the
            #      SAVE is excluded. phantom has marked_image=None → no save
            #      (symmetric: B-1828 removed phantom's draw entirely).
            #  (2) vision's raw screenshot (B-1830): _save_artifacts now only
            #      computes artifacts["screenshot"] path; the write is deferred
            #      here (the model already consumed the in-memory obs.image).
            # The earlier in-window saves (now removed) biased the latency
            # estimand. Atomic (tmp + fsync + replace, B-495 durability preserved).
            # Non-fatal (artifact-only; gallery/spot-check, not model input).
            for _img, _path in (
                (getattr(obs_prep, "marked_image", None), obs_prep.marked_image_path),
                (getattr(obs, "image", None), artifacts.get("screenshot")),
            ):
                if _img is None or not _path:
                    continue
                try:
                    _ip = Path(_path)
                    _ip.parent.mkdir(parents=True, exist_ok=True)
                    _itmp = _ip.with_suffix(_ip.suffix + ".tmp")
                    # B-1832 (2026-05-22): PIL infers the encoder from the file
                    # extension; the ".tmp" atomic-temp suffix is NOT a registered
                    # extension → `KeyError: '.tmp'` → EVERY som marked-image and
                    # vision raw-screenshot save silently failed (swallowed by the
                    # except below as a per-step warning). The B-1828/B-1830 fix
                    # repaired the latency/energy estimand but lost all visual
                    # artifacts. Pass the encoder explicitly from the FINAL path's
                    # real extension so the deferred save actually writes the PNG.
                    _fmt = _ip.suffix.lstrip(".").upper()
                    if _fmt == "JPG":
                        _fmt = "JPEG"
                    if not _fmt:
                        # B-1832 P2-2 (3-AI /stress, gemini G4 + Claude F2): an
                        # extensionless path would give _fmt="" → save(format="")
                        # → ValueError → same silent artifact loss as B-1832, just a
                        # different trigger. Default to PNG (all VWA artifact paths
                        # are .png anyway).
                        _fmt = "PNG"
                    _img.save(str(_itmp), format=_fmt)
                    _ifd = os.open(str(_itmp), os.O_RDONLY)
                    try:
                        os.fsync(_ifd)
                    finally:
                        os.close(_ifd)
                    os.replace(str(_itmp), str(_ip))
                    # B-1832 P2-1a (3-AI /stress, gemini G2 + codex F4): fsync the
                    # PARENT DIR after os.replace so the rename (name→inode link)
                    # is durable — the B-495 contract _save_artifacts honors
                    # (main.py:1615-1622) but this deferred loop had dropped, so the
                    # earlier "B-495 durability preserved" comment was only half-true
                    # (file synced, directory entry not).
                    try:
                        _pfd = os.open(str(_ip.parent), os.O_RDONLY)
                        try:
                            os.fsync(_pfd)
                        finally:
                            os.close(_pfd)
                    except OSError:
                        pass  # dir fsync best-effort (matches _save_artifacts)
                    _img_save_ok += 1  # B-1832 P2-1b artifact gate
                except Exception:
                    _img_save_fail += 1  # B-1832 P2-1b artifact gate
                    logger.warning("B-1828/B-1830 deferred image save failed: %s", _path, exc_info=True)

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
                condition_id=effective_cid,
                benchmark=task.benchmark,
                benchmark_site=task.site,
                task_id=task.task_id,
                seed=self.seed,
                step_idx=step_idx,
                som={
                    "enabled": (decision_mode == "som"),
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
                    # B-401 (/stress A1.1 v8 Mode A P1-3, 2026-05-16): preserve
                    # None semantics for B0 (proxy API has no preprocess/
                    # generate boundary exposable). Default-0.0 made B0 rows
                    # look like "preprocessing=0, generate=0" which was
                    # dishonest schema-level — None is correct.
                    "preprocessing": (
                        float(meta["preprocess_ms"])
                        if meta.get("preprocess_ms") is not None
                        else None
                    ),
                    "generate": (
                        float(meta["generate_ms"])
                        if meta.get("generate_ms") is not None
                        else None
                    ),
                    "backend_infer": float(meta.get("infer_ms", backend_latency_ms)),
                    # B-567 (/stress A1.22 P1-8-AC* 2-AI overlap, 2026-05-17,
                    # Claude F4 + Gemini F6): cross-baseline-fair backend_infer.
                    # `backend_infer` above measures `time.time()` around the
                    # entire `agent.step()` call which on B0 includes the inner
                    # `_max_retries × _backoff` retry+sleep loop (the network
                    # retry is internal to `proxy_api_agent.py:585-635`). B1/B2
                    # have no equivalent retry → their `backend_infer` is pure
                    # inference. Pre-fix: paper §3 mean step latency 跨 baseline
                    # 跑会因 B0 transient retry spike inflate by 10-70s/step
                    # while B1/B2 unchanged — apples-to-apples violated. B-143
                    # `total_minus_retry` only corrected `latency.total`, not
                    # `backend_infer`. Now emit both side-by-side: legacy
                    # `backend_infer` retained for archive compat, new
                    # `backend_infer_minus_retry` is the cross-baseline-fair
                    # source field for paper §3 latency table consumers.
                    "backend_infer_minus_retry": (
                        float(meta.get("infer_ms", backend_latency_ms))
                        - float(meta.get("network_retry_wait_ms") or 0.0)
                    ),
                    "env_step": env_step_ms,
                    "router_decision": float(overhead.get("router_decision_ms", 0.0)),
                    # B-143 (/stress A1.1 v8 Claude F7, 2026-05-15): B0
                    # proxy network retry adds 10-70s scaffold overhead;
                    # subtract for cross-baseline fair latency comparison
                    # (B1/B2 have no equivalent). Always emitted; 0 when
                    # no retries fired or for B1/B2 local backends.
                    "total_minus_retry": total_latency_ms - float(meta.get("network_retry_wait_ms") or 0.0),
                    # B-489 (/stress A1.25 GRL Chunk 3 P1-3-AB, 2026-05-17):
                    # wrapper-level wait_for_timeout settle-tax accumulator
                    # (sleep_after_execution * fired_branches per step,
                    # excludes locator-dispatch internal sleeps). Paper §4
                    # latency reports both `total` and `total - runtime_sleep`
                    # so mode-delta is not confounded with settle-tax
                    # composition (P-SoM 减少 TYPE/SELECT → less settle-tax
                    # → apparent latency gain partially from runtime, not
                    # representation efficiency).
                    # B-547 (/stress A1.5b Phase 2 P1-7-AB): backward-compat
                    # field reads post-retry; `runtime_sleep_primary` /
                    # `runtime_sleep_retry` (below at step_record extras
                    # block) preserves primary-vs-retry split per B-440
                    # hole-closure pattern. Paper §4 latency consumer can
                    # choose pure-primary view for cross-baseline parity.
                    "runtime_sleep": float(next_info.get("runtime_sleep_ms", 0) or 0),
                    "runtime_sleep_primary": float(_primary_runtime_sleep_ms),
                    "runtime_sleep_retry": (
                        float(next_info.get("runtime_sleep_ms", 0) or 0)
                        if retry_was_applied else 0.0
                    ),
                },
                # B-562 (/stress A1.22 P0-4-A* Claude OOB, 2026-05-17):
                # preserve None semantics for B0 input_text / input_image
                # tokens. Pre-fix `int(meta.get("input_text_tokens", 0))`
                # cast B0's missing key (proxy `usage.input_tokens` does
                # not break down by modality) to literal 0 ⟶ every B0
                # JSONL row read "input=2400, input_text=0, input_image=0"
                # which is a silent positive lie: cross-baseline paper §1
                # cost-per-modality sum would average B0's 0 with B1+B2
                # real ints, biasing the pool toward 0 (B0 = zero-variance
                # artifact). Matches B-401 latency split semantic (None,
                # not 0, for "backend cannot expose"). Aggregators that
                # care about token breakdown must None-safe sum (drop NaN
                # explicitly, not pandas implicit). Default None semantics
                # match `latency_ms.preprocessing/generate` (`runner/main.py
                # :2259-2268`).
                tokens={
                    "input": input_tokens,
                    "input_text": (
                        int(meta["input_text_tokens"])
                        if meta.get("input_text_tokens") is not None
                        else None
                    ),
                    "input_image": (
                        int(meta["input_image_tokens"])
                        if meta.get("input_image_tokens") is not None
                        else None
                    ),
                    "output": output_tokens,
                    "total": token_total,
                    "thinking": meta.get("thinking_tokens"),
                },
                # B-565 (/stress A1.22 P0-2-C* Gemini OOB, 2026-05-17):
                # `cost_usd.total = model + router_overhead + obs_prepare`
                # is preserved for backward compat with existing aggregators
                # but is mathematically incoherent on B0 when local scaffold
                # cost is non-zero (API USD + local-amortized USD). The
                # `cost_total_mixed_unit_warn` companion field (stamped
                # below at step_record write block) flags such rows so
                # consumers can either re-derive `total` from `model` alone
                # under api_usd basis, or disclose the warn count per cell.
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
            # Protocol Reset #6/#7 (§244, 2026-05-20): two-budget accounting flags.
            step_record["valid_agent_action"] = valid_agent_action
            step_record["consumes_agent_action_budget"] = consumes_agent_action_budget
            step_record["counts_as_runner_iteration"] = counts_as_runner_iteration
            step_record["parse_failure_reason"] = (
                str(failure_reason) if failure_reason is not None else None
            )
            step_record["fallback_finish"] = fallback_finish
            step_record["retry_action_applied"] = retry_was_applied
            step_record["retry_action_type"] = retry_action_type_str
            # B-563/B-565 (/stress A1.22 P0-1-ABC* + P0-2-C*, 2026-05-17):
            # cross-baseline cost-basis disambiguation. `_cost_unit_basis`
            # mirrors `_image_pipeline` pattern (~30 lines below) — single
            # backend-type → enum mapping so cross-baseline aggregators
            # can stratify before pooling `cost_usd.{input,output,model}`.
            # Pre-A1.22 archive rows lack the field → fill_step_defaults
            # backfills None ("archived lineage, basis unknown"). The
            # `cost_total_mixed_unit_warn` flag fires on B0 rows where the
            # `cost_usd.total` sum mixes API-USD (token cost) with
            # local-scaffold USD (router_overhead + obs_prepare). False
            # for B1/B2 (single-basis local) and for B0 rows where local
            # scaffold cost happens to be 0 (router off, zero-cost obs).
            _backend_type_for_cost = self.cfg.get("backends", {}).get(
                condition.backend_id, {}
            ).get("type", "unknown")
            _cost_unit_basis_map = {
                "api_proxy": "api_usd",
                "local_qwen": "electricity_usd_derived",
                "local_gemma": "electricity_usd_derived",
                "mock": "unknown",  # mock backends emit zero cost
            }
            step_record["cost_unit_basis"] = _cost_unit_basis_map.get(
                _backend_type_for_cost, "unknown",
            )
            step_record["cost_total_mixed_unit_warn"] = bool(
                step_record["cost_unit_basis"] == "api_usd"
                and (
                    float(router_overhead_cost or 0) > 0
                    or float(obs_prepare_cost or 0) > 0
                )
            )
            # B-569 (/stress A1.22 P1-11-A Claude, 2026-05-17): persist B0
            # network retry telemetry as discrete step_record fields. Pre-fix
            # `meta["network_retry_count"]` + `meta["network_retry_wait_ms"]`
            # were dropped at runner→JSONL boundary (only consumed for
            # `latency.total_minus_retry` arithmetic). B1/B2 always None
            # (no network retry equivalent — Optional typed so absence is
            # honest "baseline does not retry" not 0-cast "retried 0 times").
            step_record["network_retry_count"] = meta.get("network_retry_count")
            step_record["network_retry_wait_ms"] = meta.get("network_retry_wait_ms")
            # P1-3-B (/stress GRL audit 2026-05-20, user Q4=A): persist B0 action
            # provenance as discrete step fields (action-channel analogue of the
            # network_retry block above). action_source ∈ {tool_call, text_json,
            # fallback, invalid}; tool_call_valid (None if no emit);
            # text_fallback_used. Under paper_grade an emitted-but-invalid
            # tool_call → action_source="invalid" + valid=False, NO text fallback
            # (no silent action swap). Also persists the B-1588 emission-audit
            # trio (tool_call_emitted / parse_path / fallback_reason) that was
            # previously meta-only + dropped at the JSONL boundary, so paper
            # §3.5.1 cross-baseline action-channel disclosure + tool_call_emit_rate
            # gate are reproducible from disk. B1/B2 None (text-JSON backend).
            step_record["action_source"] = meta.get("action_source")
            step_record["tool_call_valid"] = meta.get("tool_call_valid")
            step_record["text_fallback_used"] = meta.get("text_fallback_used")
            step_record["tool_call_emitted"] = meta.get("tool_call_emitted")
            step_record["tool_call_parse_path"] = meta.get("tool_call_parse_path")
            step_record["tool_call_fallback_reason"] = meta.get("tool_call_fallback_reason")
            # B-1980-followup (2026-08-17): parallel-emission accounting. The v1
            # B-1980 fix aborted the condition when the proxy emitted >1
            # `web_action` in one response; that criterion was shape rather than
            # loss and killed the 8-cell floor chain at task 0. We now execute the
            # first call (the loop re-observes before the next step, so call 2+ is
            # speculative) and persist how many were dropped, so §3.5.1 discloses
            # a measured rate instead of asserting the shape never occurs.
            step_record["parallel_web_action_dropped"] = meta.get("parallel_web_action_dropped")
            step_record["parallel_web_action_dropped_args"] = meta.get(
                "parallel_web_action_dropped_args"
            )
            # B-1797 (P1-7/P1-8, /stress 2026-05-21 codex Mode B): B1/B2 text-JSON
            # repair-path provenance (analogue of tool_call_parse_path). None for
            # B0 (tool-call backend) + clean Path-1 JSON; "repaired_fenced" /
            # "repaired_multiple_identical" on repaired text-JSON steps. Keeps the
            # provenance OUT of parse_failure_reason (which is now invalid-only).
            step_record["text_parse_path"] = meta.get("text_parse_path")
            # P0-1-ABC* Phase 2 telemetry (/stress Phase 0 2026-05-19, 3-AI):
            # about:blank recovery intervention attribution. Stamp non-None
            # values only when runner intervention fired this step;
            # legacy None on normal agent steps (already default from
            # StepRecordV2 dataclass).
            if about_blank_recovered:
                step_record["intervention_type"] = "about_blank_recovery"
                step_record["counted_as_agent_action"] = False
                step_record["intervention_from_url"] = state_before.get("url") or ""
                step_record["intervention_recovery_url"] = task.raw_task.get("start_url", "")
            # GLM fallback tracking (§67 Plan B)
            # B-398 (/stress A1.1 v8 Mode A+B P0-3 overlap, 2026-05-16):
            # persist ALL attempted-fallback steps, not only the succeeded
            # ones. Pre-fix `if meta.get("glm_fallback_used"):` (truthy
            # check) meant failed-attempt steps (`attempted=True, used=
            # False`) collapsed to the same JSONL shape as "never tried"
            # (all 4 fields default None per `STEP_RECORD_V2_DEFAULTS`).
            # Downstream could not compute true GLM hit-rate from JSONL —
            # e.g. 20 success / 10 fail / 70 no-attempt rendered as
            # 20/100=20% instead of 20/30=67% (off by ~47%). Paper §3
            # GLM-disclosure audit-trail structurally unrecoverable.
            # Post-fix: emit on attempted, capture used + reason + latency
            # regardless of outcome. Combined with B-395 paper_grade hard-
            # block + B-396 yaml flip, paper-grade fire should hit zero
            # attempts (use_glm_fallback=false), but if any non-paper-
            # grade pilot/dev run uses GLM, the audit-trail is intact.
            if meta.get("glm_fallback_attempted"):
                step_record["glm_fallback_attempted"] = True
                step_record["glm_fallback_used"] = bool(meta.get("glm_fallback_used"))
                step_record["glm_fallback_latency_ms"] = meta.get("glm_fallback_latency_ms")
                step_record["glm_original_fail_reason"] = meta.get("glm_original_fail_reason")
            # /stress A1.1 codex Mode B C1 fix: persist B0 image telemetry into the
            # step record so paper-grade audit can recover image-encode behaviour
            # from JSONL alone (previously meta had over_cap / payload_bytes /
            # quality / compressed but runner dropped them — Q5 was structurally
            # impossible). C2 fix piggybacks via `encode_error`.
            # B-140 (/stress A1.1 v8 codex F5, 2026-05-15): image_meta is now
            # MANDATORY (always present in step_record), not optional. This
            # closes the "unsupported vs missing vs failed extraction"
            # ambiguity — analysis can no longer distinguish "no image"
            # from "image but no telemetry" from "image but failed encode"
            # if missingness is just absence. Now: image_meta always emitted
            # with `pipeline` label (proxy_jpeg_data_url vs hf_processor_pil)
            # + all 5 telemetry fields (None when N/A). Schema fixed.
            # B-575 (/stress A1.22 P2-18-A Claude, 2026-05-17): map "mock"
            # backend type to "no_image" so test_runner_smoke + mock-mode
            # downstream aggregators don't write "unknown" pipeline label
            # (any aggregator filter `pipeline != "unknown"` would drop
            # mock data → mock fixture verification silently broken).
            _backend_type = self.cfg.get("backends", {}).get(condition.backend_id, {}).get("type", "unknown")
            _image_pipeline = {
                "api_proxy": "proxy_jpeg_data_url",
                "local_qwen": "hf_processor_pil",
                "local_gemma": "hf_processor_pil",
                "mock": "no_image",  # B-575: mock smoke fixture rather than unknown
            }.get(_backend_type, "unknown")
            _image_meta_payload: Dict[str, Any] = {
                "pipeline": _image_pipeline,
                "image_over_cap": meta.get("image_over_cap"),
                "image_payload_bytes": meta.get("image_payload_bytes"),
                # B-400 (/stress A1.1 v8 Mode A+C overlap P1-2, 2026-05-16):
                # ref + total payload bytes piggyback. B0 emits the new
                # fields; B1/B2 leave them None (HF processor path has no
                # JPEG payload concept). Aggregator should prefer
                # `image_payload_bytes_total` for cross-task cost claim.
                "image_payload_bytes_screenshot": meta.get("image_payload_bytes_screenshot"),
                "image_payload_bytes_ref": meta.get("image_payload_bytes_ref"),
                "image_payload_bytes_total": meta.get("image_payload_bytes_total"),
                "image_quality": meta.get("image_quality"),
                "image_compressed": meta.get("image_compressed"),
                "image_encode_error": meta.get("image_encode_error"),
                # B-139 piggyback: image_token_count_method (B2 only emits
                # the field; None for B0/B1 where pipeline ≠ Gemma3 token id)
                "image_token_count_method": meta.get("image_token_count_method"),
            }
            step_record["image_meta"] = _image_meta_payload
            # B-324 (/stress A1.9 Mode B F2 OOB, 2026-05-16): image_meta_recorded
            # separator flag. A1.8 B-291 added `image_meta_recorded: bool` to
            # `StepRecordV2` + `STEP_RECORD_V2_DEFAULTS` but runner never wrote
            # it → A1.8 schema separator was structurally inert. Tag whether
            # image_meta payload reflects a real image step (mode declares
            # image AND image actually rendered AND encoding OK) vs no-image
            # step where image_meta is uniformly None by design.
            #
            # B-397 (/stress A1.1 v8 Mode A+B P0-2 overlap, 2026-05-16):
            # backend-aware truth source fix. Pre-fix:
            #   (a) image-mode set was {"som", "vision", "phantom_som"}, but
            #       per `p79/experiment/som.py:322-323` phantom_som strips
            #       image (P-SoM = SoM-prompt + [SOM_MARKS] text + NO image).
            #       Including it inflated the "image-expected" denominator.
            #   (b) truth source was `image_payload_bytes is not None`, but
            #       only B0 (`proxy_api_agent.py:741`) emits that field — it
            #       comes from B0's base64 JPEG pipeline. B1/B2 (HF processor
            #       PIL path) have no `image_payload_bytes` key → meta.get()
            #       returns None → image_meta_recorded permanently False on
            #       all B1/B2 SoM/vision steps. Any downstream aggregator
            #       filter on `image_meta_recorded == True` would silently
            #       exclude all B1/B2 image-axis data.
            # Post-fix: image-mode set = {"som", "vision"} only; truth source
            # is OR over backend-aware signals (B1/B2 via input_image_tokens,
            # B0 via image_payload_bytes). All 3 baselines now consistent.
            _image_mode = decision_mode in {"som", "vision"}
            _encode_ok = _image_meta_payload.get("image_encode_error") is None
            _image_sent = (
                int(meta.get("input_image_tokens") or 0) > 0
                or _image_meta_payload.get("image_payload_bytes") is not None
            )
            step_record["image_meta_recorded"] = bool(
                _image_mode and _encode_ok and _image_sent
            )
            # B-156 (/stress A1.3 v8 Claude F5 + codex P2-B7 dual catch, 2026-05-16):
            # locator-route dispatch telemetry from VWA wrapper info dict.
            # None when step did not invoke locator-route (scroll / wait / coord-only
            # click); otherwise {success, fallback_used, target_tag, error, action_kind}
            # so Phase 1a clean run can be audited for Cluster 1 ON_TARGET rate (paper
            # §3 evidence layer for B-01/02/33 fix).
            # B-440 (/stress A1.25 P0-2-B* codex OOB, 2026-05-17): split into
            # `_primary` + `_retry` to close the retry-overwrite hole. The
            # backward-compat `locator_route_meta` field keeps "value at step
            # write time" semantics — equals primary when no retry, else equals
            # the retry's meta (existing aggregator behavior preserved). The
            # `_primary` field is the canonical evidence layer for paper §3.
            step_record["locator_route_meta"] = next_info.get("locator_route_meta")
            step_record["locator_route_meta_primary"] = _primary_locator_route_meta
            # B-1891: `action_success` answers "did the framework avoid raising",
            # which is not the same question as "was the agent's intent carried
            # out". When the locator reports walk_fail — the referenced element
            # has no actionable ancestor — a degraded fallback still runs and
            # `action_success` comes back True. This field answers the second
            # question, so a stuck episode is visible without having to reach
            # into `locator_route_meta` (which is what made P36 the only /diag
            # rule able to see this class of failure).
            #
            # ADDITIVE ONLY. `action_success` keeps its exact meaning and its
            # exact values, so the agent-facing FAILED feedback in
            # `format_history()`, every landed step record, and every SR number
            # are untouched. `_ACTION_INTENT_FAILURE_MARKERS` is deliberately a
            # locator-error allowlist rather than "any error string" — a new
            # locator error class should have to be classified on purpose, not
            # silently start suppressing this signal.
            step_record["action_intent_fulfilled"] = _action_intent_fulfilled(
                bool(action_success), next_info.get("locator_route_meta")
            )
            step_record["locator_route_meta_retry"] = (
                next_info.get("locator_route_meta") if retry_was_applied else None
            )
            # B-420 (/stress A1.3 v9 Mode B P1-5 OOB, 2026-05-17): persist
            # select_option dispatch telemetry (None when step did not
            # invoke select_option; otherwise dict with action_kind /
            # dispatch_path / success / error).
            # B-440 companion: same retry-overwrite hole closure for select_option.
            # Primary captures pre-retry state; backward-compat field retains
            # post-retry semantics.
            step_record["select_option_meta"] = next_info.get("select_option_meta")
            step_record["select_option_meta_primary"] = _primary_select_option_meta
            # B-488 (/stress A1.25 GRL Chunk 3 P1-2-BC*, 2026-05-17): browser
            # dialog telemetry — per-step list of dialog events (None when
            # no dialog fired). Paper §3.5.1 misclick blast-radius evidence
            # layer. Wrapper drains its accumulator into info at end of step.
            # B-547 (/stress A1.5b Phase 2 P1-7-AB Claude F3 + codex B-542):
            # primary/retry split (mirror B-440/B-450 pattern). Backward-
            # compat `dialog_meta` field retains post-retry semantics; new
            # `dialog_meta_primary` is the canonical evidence layer for
            # paper §3.5.1 cross-baseline misclick blast-radius rate.
            step_record["dialog_meta"] = next_info.get("dialog_meta")
            step_record["dialog_meta_primary"] = _primary_dialog_meta
            step_record["dialog_meta_retry"] = (
                next_info.get("dialog_meta") if retry_was_applied else None
            )
            # B-512 (/stress A1.5b Phase 2 P0-1-C gemini OOB): wrapper-
            # normalized canonical action form. Reads from `next_info` so
            # if retry fired and overwrote, post-retry's normalized form is
            # captured backward-compat; `_primary_action_executed` snapshot
            # preserves primary's normalized form for paper §4.X.6 audit.
            # Pre-fix step_record["action"] was agent's raw emit only →
            # cross-baseline action-vocab asymmetry (B0 enum vs B1/B2 delta)
            # visible in JSONL; now wrapper-level execution-layer alignment
            # also auditable from disk alone.
            step_record["action_executed"] = next_info.get("action_executed")
            step_record["action_executed_primary"] = _primary_action_executed
            # Fire-6 C1b /stress P1-5: stamp recovered Page.screenshot timeout
            # (dom-mode blank placeholder) so paper §4 can disclose recovered-
            # timeout steps + exclude their ~+30s wait from canonical latency.
            step_record["screenshot_timeout_recovered"] = bool(
                next_info.get("screenshot_timeout_recovered")
            ) if isinstance(next_info, dict) else False
            # Fire-6 C1b /stress B-1780 (GRL audit Q3=A, B-1773 follow-up): the
            # per-step recovered screenshot-timeout wait (~the 30s Playwright
            # ceiling that fired) so canonical latency can EXCLUDE it. B-1773
            # added the bool flag + count "so paper §4 can ... exclude their
            # ~+30s wait" but never wired the subtraction (write-only telemetry).
            # Capped at env_step_ms so a step can never subtract more wall time
            # than it actually took; 0.0 when not recovered. The dom artifact
            # screenshot is NOT a decision input, so this infra wait must not
            # inflate dom canonical latency vs SoM/Vision (efficiency claim).
            step_record["screenshot_timeout_recovered_ms"] = (
                min(30000.0, float(env_step_ms))
                if (isinstance(next_info, dict) and next_info.get("screenshot_timeout_recovered"))
                else 0.0
            )
            # B-552 (/stress A1.5 P1-2-AB* Claude+codex OOB, 2026-05-17):
            # agent's RAW pre-validate action emit. Paper §3 taxonomy 3-layer
            # model (raw_action / control_intervention.original_action /
            # action) — see L1842-1860 + L1875 comments. Reviewer can grep
            # `raw_action.action_type` differing from `action.action_type`
            # to count cross-baseline rescue rate (B0 235B → B1/B2 4B
            # differential is exactly the B-134 contamination vector).
            step_record["raw_action"] = _agent_raw_action
            # B-546 (/stress A1.5b Phase 2 P1-6-AB Claude F2 + codex B-541):
            # control_intervention write path. None when no control fired or
            # diagnostic_controls.enabled=False (Phase 1a default). Dict
            # carries `original_action` (pre-control agent self-emit) +
            # `fires` (list of {type, reason} per control fired in order).
            # Phase 1 B-497 declared schema; Phase 2 makes it visible in
            # JSONL so paper §3 diagnostic-exploration disclosure is
            # reproducible from disk.
            step_record["control_intervention"] = (
                {
                    "original_action": _control_original_action,
                    "fires": _control_fires,
                }
                if _control_fires else None
            )
            # B-505 (/stress A1.25 GRL Chunk 2 P1-4-B* codex OOB, 2026-05-17):
            # close `select_option_meta_retry` ghost-field hole — schema /
            # dataclass / defaults (B-450) all declared the field but the
            # runner never wrote it. Symmetric to `locator_route_meta_retry`
            # at line ~1929; only populated when retry actually fired (so
            # archive aggregators can distinguish "no retry" from
            # "retry-overwrite reconstructable").
            step_record["select_option_meta_retry"] = (
                next_info.get("select_option_meta") if retry_was_applied else None
            )
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
            # B-564 (/stress A1.22 P0-5-A* Claude OOB, 2026-05-17): always
            # initialize the key (value None when no bbox extractable) so
            # `validate_step_record_v2` PAPER_GRADE_STEP_OPTIONAL_KEYS
            # presence check passes. Pre-fix the key was conditionally
            # absent on steps without `obs_nodes_info` ⟹ reviewer grep of
            # JSONL for `element_bbox` had to handle KeyError vs None as
            # two paths. Now: None ≡ "no bbox extractable this step" (no
            # element_id, no obs_nodes_info, or no union_bound), list ≡
            # bbox present. Single canonical absence path.
            step_record["element_bbox"] = None
            eid = action.get("element_id")
            # Sequential SoM identifier contract (2026-05-25): for SoM-family modes
            # the agent's element_id is a seq id, so resolve the bbox via the
            # seq-keyed map (obs_prep.obs_nodes_info_seq carries union_bound under
            # seq keys), not the native obs.obs_nodes_info (codex review P2).
            _bbox_src = (
                obs_prep.obs_nodes_info_seq
                if getattr(obs_prep, "obs_nodes_info_seq", None) is not None
                else (getattr(obs, "obs_nodes_info", None) or None)
            )
            if eid is not None and _bbox_src:
                node_info = _bbox_src.get(str(eid))
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
            prev_action_intent_fulfilled = step_record.get("action_intent_fulfilled")  # B-1891
            # /stress A1.10 P0-2-AB* (2026-05-16): router input now consumes
            # the **agent-visible** page-change signal, not raw runner-internal
            # page_changed (any-reason). Pre-fix the B-09 split landed only at
            # SR derivation but routing decision still saw form_value_changed
            # / dom_complexity_changed / text_length_changed / form_fields_changed
            # / interactive_elements_changed → router reset unchanged_streak on
            # signals the agent cannot perceive, suppressing legitimate
            # escalation. AGENT_VISIBLE_REASONS = {url_changed, title_changed,
            # content_changed, scroll_changed, modal_state_changed}. The
            # `page_changed` runner-internal field is retained on the step
            # record for unchanged retry/cycle-detection paths.
            # (`is_agent_visible_change` is module-imported at the top of this file.)
            prev_page_changed = is_agent_visible_change(page_change_reasons)

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
                    "Action cycle detected (%s, len=%d, reps>=%d) at step %d for task %s/%d — %s.",
                    mode, detected, min_r, step_idx, task.site, task.task_id,
                    "early stop" if _early_stop_enabled else "diagnostic only (early-stop disabled per advisor 5/5)",
                )
                if _early_stop_enabled:
                    cycle_early_stop = True
                    break

            # --- scroll alternation detection ---
            if is_scroll:
                # /stress A2.4b Chunk α (2026-05-18): prefer canonical
                # `scroll_direction` enum (post-prompt-unification all baselines
                # emit `up`/`down` directly). Archive runs pre-unification still
                # carry `delta` — read as fallback so backward-compat preserved.
                sd = action.get("scroll_direction")
                if sd in {"up", "down"}:
                    direction = sd
                else:
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
                            "Scroll alternation detected (len=%d) at step %d for task %s/%d — %s.",
                            ALT_THRESHOLD, step_idx, task.site, task.task_id,
                            "early stop" if _early_stop_enabled else "diagnostic only (early-stop disabled per advisor 5/5)",
                        )
                        if _early_stop_enabled:
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
                    "URL stuck (%d consecutive clicks, url=%s) at step %d for task %s/%d — %s.",
                    url_stuck_streak, current_url[:80], step_idx, task.site, task.task_id,
                    "early stop" if _early_stop_enabled else "diagnostic only (early-stop disabled per advisor 5/5)",
                )
                if _early_stop_enabled:
                    cycle_early_stop = True
                    break

            step_idx += 1

        # VWA evaluator expects trajectory to end with an Action dict having "answer" key.
        # When the agent never stopped (max-steps / cycle), trajectory ends with an obs dict
        # which lacks "answer", causing KeyError: 'answer'.  Append a fake stop action.
        #
        # B-166 (/stress A1.4a v8 Claude F4, 2026-05-16): trajectory_incomplete
        # telemetry. The fake stop action's empty answer is fed to VWA's
        # string_match evaluator, which compares "" vs ground-truth ("$19.99",
        # etc.) → score=0 for all max-steps-timeout episodes regardless of
        # actual capability. B1/B2 4B baselines time out far more than B0 235B
        # → cross-baseline SR rank contains a non-capability timeout-rate
        # confound. Disclosure path (Path A): SR remains canonical (no
        # adjustment); `trajectory_incomplete=True` recorded as a transparency
        # metric for paper §3.5 + cross-cell aggregation.
        trajectory_incomplete = False
        if not trajectory or not isinstance(trajectory[-1], dict) or "answer" not in trajectory[-1]:
            trajectory_incomplete = True
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

        # B-545 (/stress A1.5b Phase 2 P0-5-AC Claude F12 + gemini B-513
        # cross-validation, 2026-05-17): RETIRED reward-override block.
        # Pre-fix paper §3/§4 prose claimed "canonical evaluator success,
        # no post-hoc adjustment" but main.py L2614-2630 secretly overrode
        # `score=0 → score=1` when agent self-reported `action_type=="finish"`
        # + parse_valid + env_reward>0. Cross-AI 2-AI catch (Claude F12 +
        # gemini B-513 cross-val): the override is exactly the kind of
        # estimand schizophrenia top-tier reviewers attack — "you claim
        # raw evaluator authority but secretly bake agent self-report into
        # SR; paper §3 estimand definition is a false claim".
        #
        # B-165 (A1.4a) had already restricted the override to real-finish
        # (excluding fallback_finish + parse_invalid) to close the
        # cross-baseline B0-vs-B1/B2 differential. Phase 2 reframes:
        # rather than narrow override conditions further, eliminate the
        # mechanism. `success` is now strictly `score >= 1.0` from the
        # VWA evaluator output, full stop.
        #
        # Disclosure: paper §3.5 "evaluator authority" disclosure paragraph
        # to be updated to (a) state pure evaluator authority post-B-545
        # and (b) historical note that B-165 + B-545 retired the override
        # mechanism in two stages. Disclosure prose is DEFERRED to next
        # paper round (parallel session has uncommitted `section3_*.md` /
        # `section4_*.md` edits — touching here would collide).
        success = bool(score >= 1.0)

        total_latency = sum(float(s["latency_ms"].get("total", 0.0)) for s in step_records)
        # B-1600 (/stress 深入审 Mode A P0-1-A*, 2026-05-18): retry-adjusted
        # episode rollup. B0 step records have `latency_ms.total_minus_retry =
        # total - network_retry_wait_ms`; B1/B2 have `total_minus_retry = total`
        # (meta.network_retry_wait_ms is None → 0.0 fallback per L2579). Falls
        # back to `total` if `total_minus_retry` absent for backward compat
        # with legacy step records (pre-B-143/B-1600).
        total_latency_minus_retry = sum(
            float(s["latency_ms"].get("total_minus_retry", s["latency_ms"].get("total", 0.0)))
            for s in step_records
        )
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
        # Fire-4 RCA Wave 2 M5 (mirror of exception-path stamping): runner
        # success path also passes eval_result.error through both noise
        # taxonomies. eval_result.error is typically empty/None on success;
        # populated when the evaluator returned a non-fatal error message
        # that didn't trigger EvaluatorUnavailableError (e.g., partial
        # eval failure). classify_timeout handles None gracefully (returns
        # False, None) so this is safe on the happy path.
        _is_timeout, _timeout_callsite = classify_timeout(eval_result.error)

        episode_summary = EpisodeSummaryV2(
            schema_version=SCHEMA_VERSION_V2,
            run_id=self.cfg["experiment"]["run_id"],
            condition_id=effective_cid,
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
            total_latency_minus_retry_ms=total_latency_minus_retry,
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
            # Fire-4 RCA Wave 2 M5 timeout taxonomy fields (success path).
            unverified_timeout_event=_is_timeout,
            timeout_callsite=_timeout_callsite,
            verified_substrate_noise=None,
            artifacts_dir=str(episode_dir),
            state_change_reason_distribution=dict(state_change_reason_distribution),
            checklist_completion_rate=checklist_completion_rate,
            checklist_failed_items=checklist_failed_items,
            error=eval_result.error,
            # B-554 (/stress A1.5 P1-4-AB* Claude+codex OOB, 2026-05-17):
            # archive cohort sentinel. Post-B-545 (A1.5b Phase 2 commit
            # `7832008`) episodes carry pure-evaluator semantic.
            # `evaluator_authority_mode` enum allows future B-545-style
            # estimand migrations to add new tags without breaking legacy
            # consumers. `reward_override_applied=False` makes the absence
            # of override explicit in JSONL (vs missing-field ambiguity).
            evaluator_authority_mode="post_B545_vwa_score_only",
            reward_override_applied=False,
            # Fire-6 RCA Stage C1 (/stress 2026-05-20): evaluator-context
            # provenance stamped from eval_result. getattr fallback to None
            # keeps NullEvaluator / dev-mode / pre-C1 paths schema-valid.
            eval_context_mode=getattr(eval_result, "eval_context_mode", None),
            eval_isolated_context_used=getattr(eval_result, "eval_isolated_context_used", None),
            eval_goto_latency_ms=getattr(eval_result, "eval_goto_latency_ms", None),
            eval_goto_timeout=getattr(eval_result, "eval_goto_timeout", None),
            eval_source_agent_url=getattr(eval_result, "eval_source_agent_url", None),
            eval_target_url=getattr(eval_result, "eval_target_url", None),
            # Fire-6 RCA Stage C2: diagnostic-replay provenance (main path).
            diagnostic_replay=self.diagnostic_replay,
            sr_excluded=self.diagnostic_replay,
        ).as_dict()

        # Enrich with wasted cost and component breakdown for cost-aware analysis
        wasted = compute_wasted_cost(step_records, success)
        episode_summary["wasted_cost_usd"] = wasted["wasted_cost_usd"]
        episode_summary["wasted_energy_kwh"] = wasted["wasted_energy_kwh"]
        breakdown = compute_component_breakdown(step_records)
        episode_summary["component_breakdown"] = breakdown
        # Track how many free busy-page waits were issued (not counted as steps)
        episode_summary["busy_wait_free_steps"] = total_busy_waits
        # Protocol Reset #6/#7/#8 (§244 canonical, 2026-05-20): two-budget
        # accounting rollup + three-column cost. Counters: agent_action = budget-
        # consuming steps (max_agent_actions denominator); valid_action = genuine
        # parse-valid decisions (excludes policy-blocked goto); model_call_attempt
        # = persisted steps (= LLM calls); runner_iteration = + free busy-waits;
        # parse_error_injected_wait = unparseable/structural sinks (safety numerator).
        episode_summary["agent_action_step_count"] = agent_action_count
        episode_summary["valid_action_step_count"] = valid_action_count
        episode_summary["model_call_attempt_count"] = len(step_records)
        # B-1832 P2-1b artifact gate: record per-episode image-save outcomes +
        # loud once-per-episode ERROR on the total-loss signature (attempts>0, all
        # failed). dom/phantom never attempt image saves (counters stay 0 → no
        # false alarm); som/vision do. Condition-level paper-grade gate consumes
        # these fields downstream (queue sentinel artifact check).
        episode_summary["artifact_image_saves_ok"] = _img_save_ok
        episode_summary["artifact_image_saves_fail"] = _img_save_fail
        if _img_save_fail > 0 and _img_save_ok == 0:
            logger.error(
                "B-1832 ARTIFACT GATE: episode task=%s mode=%s saved 0/%d images "
                "(ALL deferred saves failed) — visual artifacts lost this episode; "
                "investigate PIL/format/path BEFORE trusting this condition's gallery.",
                getattr(task, "task_id", "?"), condition.observation_mode, _img_save_fail,
            )
        episode_summary["runner_iteration_count"] = len(step_records) + total_busy_waits
        episode_summary["parse_error_injected_wait_count"] = total_parse_errors
        # Three-column cost (#8, see metrics.compute_three_column_cost). billed =
        # Σ all model cost; canonical = Σ valid_agent_action cost; wasted = residual
        # (parse-error + policy-blocked + recovery). canonical + wasted ≡ billed.
        _3col = compute_three_column_cost(step_records)
        episode_summary["total_billed_cost_usd"] = _3col["total_billed_cost_usd"]
        episode_summary["canonical_action_cost_usd"] = _3col["canonical_action_cost_usd"]
        episode_summary["protocol_wasted_cost_usd"] = _3col["protocol_wasted_cost_usd"]
        # B-1798 (P1-1, /stress 2026-05-21 Claude A2 + codex B4 2-AI overlap):
        # roll up cost_unit_basis (modal) + cost_total_mixed_unit_warn (any) from
        # step records to the EPISODE summary. Pre-fix these were stamped on STEP
        # records only; the condition aggregator (metrics.py reads
        # ep.get("cost_unit_basis")) saw all None → cross_site CSV "unknown" for
        # every paper-grade row → §1 cross-baseline cost stratification
        # unverifiable from artifact (re-opened B-1559). Episode summary is the
        # strict accounting boundary for the basis chain.
        _ep_bases = [s.get("cost_unit_basis") for s in step_records if s.get("cost_unit_basis")]
        episode_summary["cost_unit_basis"] = (
            max(set(_ep_bases), key=_ep_bases.count) if _ep_bases else None
        )
        episode_summary["cost_total_mixed_unit_warn"] = (
            any(bool(s.get("cost_total_mixed_unit_warn")) for s in step_records)
            if step_records else None
        )
        # Fire-6 C1b /stress P1-5: episode rollup of recovered Page.screenshot
        # timeouts (dom-mode). Paper §4 disclosure + latency-confound count.
        episode_summary["screenshot_timeout_recovered_count"] = sum(
            1 for s in step_records if s.get("screenshot_timeout_recovered")
        )
        # Fire-6 C1b /stress B-1780 (GRL audit Q3=A, B-1773 follow-up): two-layer
        # latency. raw = total_latency_ms (includes the recovered ~30s timeout —
        # operational truth). canonical = retry-adjusted MINUS the recovered
        # artifact screenshot-timeout (dom artifact-only is not a decision input,
        # so its infra wait must not contaminate the dom-vs-SoM/Vision efficiency
        # claim). recovered_total_ms = the subtracted delta; the aggregator emits
        # the per-cell rate. Clamp canonical ≥ 0.
        episode_summary["screenshot_timeout_recovered_total_ms"] = sum(
            float(s.get("screenshot_timeout_recovered_ms") or 0.0) for s in step_records
        )
        _minus_retry = episode_summary.get("total_latency_minus_retry_ms")
        episode_summary["total_latency_canonical_ms"] = (
            max(0.0, float(_minus_retry) - episode_summary["screenshot_timeout_recovered_total_ms"])
            if _minus_retry is not None else None
        )
        # B-1831 (/stress 2026-05-22, user A): episode-level reset Page.goto retry
        # telemetry (pre-episode infra recovery; NOT benchmark_noise, NOT in the
        # SR denominator — a recovered reset just means a transient browser/docker
        # hiccup at env.reset was retried before the agent run started). Sourced
        # from VwaWrapper._reset_goto_telemetry; getattr-guarded for MockEnv.
        _reset_tele = getattr(self.environment, "_reset_goto_telemetry", None) or {}
        episode_summary["reset_goto_timeout_count"] = int(_reset_tele.get("reset_goto_timeout_count", 0))
        episode_summary["reset_goto_retry_count"] = int(_reset_tele.get("reset_goto_retry_count", 0))
        episode_summary["reset_goto_recovered"] = bool(_reset_tele.get("reset_goto_recovered", False))
        episode_summary["reset_goto_latency_ms_per_attempt"] = _reset_tele.get("reset_goto_latency_ms_per_attempt", [])
        episode_summary["reset_retry_reason"] = _reset_tele.get("reset_retry_reason")
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
        # B-788 (/stress A1.9 cold-start P1-4-B* codex OOB, 2026-05-17):
        # surface step-level `energy_window_partial` (B-321 strict-window flag)
        # to episode + condition aggregator. Pre-fix flag stamped at step
        # boundary by EnergyTracker but never aggregated → paper §3 energy
        # comparison mixed high-quality and low-density samples.
        _energy_partial_steps = sum(
            1 for s in step_records
            if isinstance(s.get("energy"), dict)
            and bool(s["energy"].get("energy_window_partial", False))
        )
        episode_summary["energy_window_partial_step_count"] = _energy_partial_steps
        _window_counts = [
            int(s["energy"].get("window_sample_count", 0) or 0)
            for s in step_records
            if isinstance(s.get("energy"), dict)
            and s["energy"].get("source") == "pynvml"
        ]
        episode_summary["min_window_sample_count"] = (
            min(_window_counts) if _window_counts else None
        )
        # B-797 (/stress A1.9 cold-start P2-4-C gemini, 2026-05-17): BLIP-2
        # device telemetry — surface evaluator-side captioning device (cuda /
        # cpu / None) into episode summary for cross-baseline latency audit.
        # `self.evaluator` is `VwaEvaluator` (or NullEvaluator with no attr).
        _blip2_dev = getattr(self.evaluator, "_blip2_device", None)
        episode_summary["evaluator_blip2_device"] = _blip2_dev
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

        # B-166 (/stress A1.4a v8 Claude F4, 2026-05-16): trajectory_incomplete
        # — see the fake stop-action block above. Recorded here so per-cell
        # aggregation can report `trajectory_incomplete_rate` as a transparency
        # metric (paper §3.5). Always emitted to keep schema invariant.
        episode_summary["trajectory_incomplete"] = trajectory_incomplete

        # B-441 (/stress A1.25 P0-6-B* codex OOB, 2026-05-17): image_encode_error
        # per-episode count. Schema declared this field at A1.1 (B-403) for
        # cross-baseline (B0 proxy JPEG vs B1/B2 HF processor) symmetric
        # exclusion in aggregators. Pre-fix the runner never stamped the
        # field → downstream `scripts/analysis/aggregate_sr_fp_per_mode.py:
        # 112-116` defaulted missing → 0 → every episode looked "clean" →
        # cross-baseline filter was structurally fake. Codex Mode B B4 catch:
        # paper §3 image_meta-based filtering claim was unsupported by data
        # layer. Aggregator can now drop infra-failed episodes OR annotate
        # disclosure column without re-scanning step JSONL.
        episode_summary["image_encode_error_step_count"] = sum(
            1 for _s in step_records
            if isinstance(_s.get("image_meta"), dict)
            and _s["image_meta"].get("image_encode_error") is not None
        )

        # P0-1-ABC* + P1-11-B* Phase 2 telemetry (/stress Phase 0 2026-05-19,
        # 3-AI overlap OOB): runner-intervention rollup. Counts step-records
        # where intervention_type is set (about:blank recovery + future
        # intervention types). Paper §3 disclosure Appendix column
        # `runner_intervention_step_rate = runner_intervention_count / steps`
        # per cell — cross-baseline asymmetric scaffold-effect transparency.
        episode_summary["runner_intervention_count"] = sum(
            1 for _s in step_records if _s.get("intervention_type") is not None
        )
        episode_summary["about_blank_recovery_count"] = sum(
            1 for _s in step_records
            if _s.get("intervention_type") == "about_blank_recovery"
        )
        # P1-17-C* Phase 2 attempt-lineage: defer runner stamping until
        # checkpoint-restore infrastructure lands. Field reservation via
        # EpisodeSummaryV2 dataclass defaults (all None); paper §3.X.6
        # covariate trail consumer reads None as "primary attempt, no
        # restore" semantic. v2.py defaults already backfill.
        # P1-17-C* + Gemini F3 footprint telemetry: aggregator-side compute
        # deferred to paper-2 round (requires action_executed dispatch-path
        # heuristic spec lock). Field reservation via EpisodeSummaryV2
        # dataclass defaults (count 0, score None). Stamp here when heuristic
        # spec is locked; for now defaults pass through.

        # B-167 (/stress A1.4a v8 Claude F3 expanded scope, 2026-05-16):
        # unknown_failure_reasons Counter exposes any failure_reason that
        # fell through to ``unknown_failure`` category. Acts as a paper-grade
        # tripwire — if a previously-unseen backend error string appears
        # frequently, Counter surfaces it for catalog inclusion in the next
        # taxonomy bump. Empty dict when no unknown failures.
        _unknown_reasons = Counter(
            s.get("parse_failure_reason") for s in step_records
            if s.get("error_category") == "unknown_failure"
            and s.get("parse_failure_reason")
        )
        episode_summary["unknown_failure_reasons"] = dict(_unknown_reasons)

        # §139.8: the runner no longer computes `adjusted_success` / `fp_reason`.
        # The post-hoc na_fp / eval_fp filter layer is retired — those FPs are
        # fixed at the source now (empty-pred guard in the VWA evaluator,
        # master bug B-91, + N/A task exclusion at load time). `success` above
        # is already the canonical paper-grade outcome. `agent_finished` is
        # still recorded above as a standalone diagnostic.

        # B-222 (2026-05-16, A1.5 Item 6): episode 完成 — 移除 .in_progress
        # marker so watchdog orphan-cleanup may safely prune this artifact dir
        # if its summary doesn't land (e.g. summary-write crash). The marker
        # was touched after episode_dir.mkdir(parents=True, exist_ok=True).
        try:
            _marker = episode_dir / ".in_progress"
            if _marker.exists():
                _marker.unlink()
        except OSError:
            pass  # filesystem ro / race; downstream gallery handles missing marker

        return episode_summary
