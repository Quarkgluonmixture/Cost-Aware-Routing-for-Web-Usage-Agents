from __future__ import annotations

import copy
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "experiment": {
        "name": "p79_experiment",
        "benchmark": "visualwebarena",
        "phase": "phase1",
        "seed": 42,
        "output_root": "results",
    },
    "variables": {
        "primary": {
            # /stress A1.10 P0-3-A (2026-05-16): canonical fallback is the
            # paper-1 6-mode universe per paper §1 hero claim. Per-condition
            # yamls may override to a subset (e.g. 1 mode for resume-one-cell
            # workflow) but the unspecified-fallback is the full canonical
            # set so a yaml omitting this field generates the complete
            # phantom routing space rather than a silent 3-mode subset.
            "observation_mode": [
                "dom", "som", "vision",
                "phantom_som", "phantom_text", "phantom_prompt",
            ],
            "router": [False, True],
        },
        "secondary": {
            "modules": ["none"],
        },
        "deferred": {
            "enable_mind2web": False,
            "enable_learning_router": False,
        },
    },
    "router": {
        "cheap_default_mode": "dom",
        "rich_escalation_mode": "som",
        "thresholds": {
            "dom_size_threshold": 12000,
            "unchanged_steps_trigger": 2,
            "no_progress_steps_trigger": 2,
            "retry_limit": 1,
        },
        "checklist_trigger": {
            "enabled": False,
            "stalled_steps_trigger": 2,
            "failed_item_trigger": True,
        },
        "overhead_cost_per_ms": 0.0,
    },
    "metrics": {
        "cost": {
            "input_cost_per_1k": 0.0,
            "output_cost_per_1k": 0.0,
        },
        "energy": {
            "enabled": False,
            "kwh_per_step": None,
            "co2e_kg_per_kwh": None,
            "hardware_profile": "m2",
            "region": "world",
            "use_psutil": True,
            "fixed_power_watts": None,
            "use_pynvml": True,
            "sample_interval_s": 0.5,
            # `track_model_load` / `model_load_amortize_over` removed in §97
            # ET-12 (record_model_load was dead code).
        },
    },
    "checklist": {
        "enabled": False,
        "inject_into_prompt": True,
        "max_items": 4,
    },
    "state_change": {
        "similarity_threshold": 0.95,
        "form_snapshot_enabled": True,
    },
    "auth_refresh": {
        "enabled": True,
        "interval": 5,
        # B-35 fix (笔记 §116.9): time-based threshold prevents PHP session
        # gc_maxlifetime (1440s on cls/shopping) expiring mid-long-episode
        "time_interval_seconds": 1200,
        "sites": ["classifieds", "reddit", "shopping", "shopping_admin"],
    },
    "analysis": {
        "outputs": {
            "save_plots": True,
            "save_csv": True,
        }
    },
}


# B-574 (/stress A1.22 P1-16-B codex, 2026-05-17): explicit-clear sentinel
# for yaml-driven config override. Pre-fix `_merge_dict` skipped any
# `v is None` key during merge, which meant yaml `revision: null` (a
# legitimate "clear inherited value" expression) silently fell through →
# inherited value retained → reviewer reading the normalized config sees
# "inherited revision SHA" believing override applied. Especially
# dangerous for cross-baseline asymmetric overrides (one yaml clears
# `use_glm_fallback` for B0, expects null but gets inherited true →
# paper-grade GLM rescue contamination). Sentinel `{"__delete__": true}`
# is the explicit clear; yaml writes `revision: {__delete__: true}` to
# wipe an inherited key.
_DELETE_SENTINEL = "__delete__"


def _merge_dict(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for k, v in (update or {}).items():
        # B-574: explicit delete sentinel — yaml `key: {__delete__: true}`
        # removes the inherited key entirely. Without this, yaml authors
        # have no way to "unset" a base-config default short of forking the
        # whole config.
        if isinstance(v, dict) and v.get(_DELETE_SENTINEL) is True:
            merged.pop(k, None)
            continue
        if v is None:
            # B-574: still skip plain `None` to preserve legacy semantics
            # (most yamls use `key:` for "no override" rather than "delete").
            # Authors who truly want delete must use the explicit sentinel.
            continue
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _merge_dict(merged[k], v)
        else:
            merged[k] = copy.deepcopy(v)
    return merged


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    defaults = cfg.get("defaults", [])
    # Tolerate single-string form (`defaults: "base.yaml"`) — wrap in list to
    # avoid iterating characters of the string.
    if isinstance(defaults, str):
        defaults = [defaults]
    elif not isinstance(defaults, (list, tuple)):
        defaults = []
    merged = copy.deepcopy(DEFAULT_CONFIG)
    for default_cfg in defaults:
        default_path = Path(default_cfg)
        if not default_path.exists():
            candidate = path.parent / default_cfg
            if candidate.exists():
                default_path = candidate
        if not default_path.exists():
            raise FileNotFoundError(f"Referenced default config not found: {default_cfg}")
        with open(default_path, "r", encoding="utf-8") as f:
            base_piece = yaml.safe_load(f) or {}
        merged = _merge_dict(merged, base_piece)

    cfg_no_defaults = dict(cfg)
    cfg_no_defaults.pop("defaults", None)
    merged = _merge_dict(merged, cfg_no_defaults)
    return normalize_config(merged)


def normalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)

    # B-395 (/stress A1.1 v8 3-AI overlap P0-1, 2026-05-16): paper_grade flag
    # wires top-level → backends → agent so the B-340 GLM hard-block at
    # `proxy_api_agent.py:179-186` is reachable. Source priority:
    #   1. env var P79_PAPER_GRADE=1 (paper-grade queue scripts export this)
    #   2. yaml top-level `paper_grade: true` (explicit override)
    #   3. default False (dev session / mock runs)
    # Without this wire B-340 raise is unreachable → GLM scaffold can
    # silently enable on B0 paper-grade fire and contaminate paper §1
    # cost-fairness. Codex Mode B F1 + Gemini Mode C P0-5 + Claude Mode A
    # P0-1 three-AI overlap (highest confidence finding in A1.1 batch).
    _paper_grade_env = os.environ.get("P79_PAPER_GRADE", "").strip().lower()
    if _paper_grade_env in ("1", "true", "yes", "on"):
        cfg["paper_grade"] = True
    else:
        cfg.setdefault("paper_grade", False)

    # Fire-6 RCA Stage C2 (/stress 2026-05-20): diagnostic-replay mode flag.
    # Mirrors the paper_grade env→yaml→default precedence. Source priority:
    #   1. env P79_DIAGNOSTIC_REPLAY=1 (queue_diagnostic_replay.sh wrapper)
    #   2. yaml/CLI top-level `diagnostic_replay: true`
    #   3. default False (canonical fire / dev / mock)
    # When True: resolve_output_root routes to results/diagnostic_replay/
    # (non-canonical), the runner stamps every episode sr_excluded=True
    # (canonical SR firewall) + suppresses the M1 quarantine abort. This is a
    # task-scoped reproduction harness, NEVER a canonical fire path.
    _diag_env = os.environ.get("P79_DIAGNOSTIC_REPLAY", "").strip().lower()
    if _diag_env in ("1", "true", "yes", "on"):
        cfg["diagnostic_replay"] = True
    else:
        cfg.setdefault("diagnostic_replay", False)

    experiment = cfg.setdefault("experiment", {})
    experiment.setdefault("name", "p79_experiment")
    experiment.setdefault("benchmark", "visualwebarena")
    experiment.setdefault("phase", "phase1")
    experiment.setdefault("seed", 42)
    experiment.setdefault("output_root", "results")
    sites = cfg.get("task", {}).get("include_sites", [])
    _uid = uuid.uuid4().hex[:6]
    if len(sites) == 1:
        experiment.setdefault("run_id", f"run_{sites[0]}_{int(time.time())}_{_uid}")
    else:
        experiment.setdefault("run_id", f"run_{int(time.time())}_{_uid}")

    env_cfg = cfg.setdefault("env", {})
    env_cfg.setdefault("type", "vwa")
    env_cfg.setdefault("headless", True)
    env_cfg.setdefault("observation_type", "accessibility_tree")
    env_cfg.setdefault("dry_run", False)
    env_cfg.setdefault("viewport_width", 1280)
    env_cfg.setdefault("viewport_height", 720)

    task_cfg = cfg.setdefault("task", {})
    task_cfg.setdefault("include_sites", ["shopping", "reddit", "classifieds"])
    task_cfg.setdefault("max_tasks_per_site", None)
    task_cfg.setdefault("task_ids", {})
    # §139.8: N/A (unanswerable) tasks excluded from the scored set by default —
    # pre-registered scope decision (see preregistration.md). Set False only for
    # a dedicated N/A-capability study.
    task_cfg.setdefault("exclude_na_tasks", True)

    backends = cfg.setdefault("backends", {})
    if not backends:
        backends["local_4b"] = {
            "type": "local_qwen",
            "path": "Qwen/Qwen3-VL-4B-Instruct",
            "quantization": "4bit",
            "device": "cuda",
            "max_new_tokens": 512,
            "temperature": 0.1,
            "top_p": 0.9,
            "dom_mode": "llm",
            "mock_mode": True,
        }
    backends.setdefault("default_backend", next(iter([k for k in backends.keys() if k != "default_backend"]), "local_4b"))

    cfg.setdefault("baselines", {})
    cfg["baselines"].setdefault("run_b0", False)
    cfg["baselines"].setdefault("b0_backend", "api_strong")
    cfg["baselines"].setdefault("b0_observation_mode", "som")

    thresholds = cfg.setdefault("router", {}).setdefault("thresholds", {})
    thresholds.setdefault("dom_size_threshold", 12000)
    thresholds.setdefault("unchanged_steps_trigger", 2)
    thresholds.setdefault("no_progress_steps_trigger", 2)
    thresholds.setdefault("retry_limit", 1)
    cfg["router"].setdefault("cheap_default_mode", "dom")
    cfg["router"].setdefault("rich_escalation_mode", "som")
    cfg["router"].setdefault("overhead_cost_per_ms", 0.0)
    cfg["router"].setdefault("extra_model_call_cost_usd", 0.0)
    cfg["router"].setdefault("retry_cost_usd", 0.0)
    checklist_trigger_cfg = cfg["router"].setdefault("checklist_trigger", {})
    checklist_trigger_cfg.setdefault("enabled", False)
    checklist_trigger_cfg.setdefault("stalled_steps_trigger", 2)
    checklist_trigger_cfg.setdefault("failed_item_trigger", True)

    cfg.setdefault("runtime", {})
    cfg["runtime"].setdefault("max_steps", 40)
    cfg["runtime"].setdefault("resume", True)
    cfg["runtime"].setdefault("busy_wait_limit", 5)
    cfg["runtime"].setdefault("baseline_retry_on_no_progress", False)
    # Protocol Reset #7 (§244 canonical, 2026-05-20): two-budget accounting.
    # `max_agent_actions` is the PRIMARY budget (only valid, budget-consuming
    # steps decrement it — restores upstream "30 agent decisions" semantics).
    # It defaults to `max_steps` so existing per-condition yamls (max_steps: 30)
    # inherit the right budget with zero yaml churn; `max_steps` itself is now
    # only a resume-fingerprint / telemetry input, no longer the loop cap.
    # The safety budget bounds runaway episodes (all-parse-error / pathological
    # loops): `max_model_attempts` caps total LLM calls; the parse-error caps
    # terminate when the agent cannot produce parseable actions. `max_model_
    # attempts` derives from the budget so it scales if max_agent_actions changes
    # and can never cut before the primary budget is exhausted.
    cfg["runtime"].setdefault("max_agent_actions", cfg["runtime"]["max_steps"])
    cfg["runtime"].setdefault("max_consecutive_parse_errors", 3)
    cfg["runtime"].setdefault("max_total_parse_errors", 5)
    cfg["runtime"].setdefault(
        "max_model_attempts",
        int(cfg["runtime"]["max_agent_actions"]) + int(cfg["runtime"]["max_total_parse_errors"]) + 10,
    )

    cfg.setdefault("checklist", {})
    cfg["checklist"].setdefault("enabled", False)
    cfg["checklist"].setdefault("inject_into_prompt", True)
    cfg["checklist"].setdefault("max_items", 4)

    cfg.setdefault("state_change", {})
    cfg["state_change"].setdefault("similarity_threshold", 0.95)
    cfg["state_change"].setdefault("form_snapshot_enabled", True)

    cfg.setdefault("auth_refresh", {})
    cfg["auth_refresh"].setdefault("enabled", True)
    cfg["auth_refresh"].setdefault("interval", 5)
    cfg["auth_refresh"].setdefault("time_interval_seconds", 1200)  # B-35 fix
    cfg["auth_refresh"].setdefault("sites", ["classifieds", "reddit", "shopping", "shopping_admin"])

    cfg.setdefault("metrics", {}).setdefault("energy", {})
    cfg["metrics"]["energy"].setdefault("hardware_profile", "m2")
    cfg["metrics"]["energy"].setdefault("region", "world")
    cfg["metrics"]["energy"].setdefault("use_psutil", True)
    cfg["metrics"]["energy"].setdefault("fixed_power_watts", None)
    cfg["metrics"]["energy"].setdefault("use_pynvml", True)
    cfg["metrics"]["energy"].setdefault("sample_interval_s", 0.5)
    # `track_model_load` / `model_load_amortize_over` removed in §97 ET-12.

    return cfg


def resolve_output_root(cfg: Dict[str, Any]) -> Path:
    experiment = cfg["experiment"]
    # Fire-6 RCA Stage C2 (/stress 2026-05-20): diagnostic-replay output is
    # NON-CANONICAL. It must NOT land under results/{benchmark}/{phase}/ where
    # canonical aggregators glob (results/visualwebarena/phase1/...) — diagnostic
    # episodes would otherwise be discoverable by paper §1 SR producers. Isolate
    # to results/diagnostic_replay/<run_id>/. This is the FIRST line of defense;
    # sr_excluded=True + load_episode_summary_strict(reject_sr_excluded=True) is
    # the second (catches accidental dir merge / explicit mis-pointing).
    if cfg.get("diagnostic_replay"):
        root = Path(experiment["output_root"]) / "diagnostic_replay" / experiment["run_id"]
    else:
        root = Path(experiment["output_root"]) / experiment["benchmark"] / experiment["phase"] / experiment["run_id"]
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_task_file(path_or_env: str) -> str:
    if path_or_env.startswith("$"):
        env_name = path_or_env[1:]
        value = os.environ.get(env_name)
        if not value:
            raise ValueError(f"Task config env var not set: {env_name}")
        return value
    return path_or_env
