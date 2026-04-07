from __future__ import annotations

import copy
import os
import time
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
            "observation_mode": ["dom", "som", "vision"],
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
            "track_model_load": False,
            "model_load_amortize_over": 0,
        },
    },
    "checklist": {
        "enabled": False,
        "inject_into_prompt": True,
        "max_items": 4,
    },
    "state_change": {
        "similarity_threshold": 0.95,
    },
    "analysis": {
        "outputs": {
            "save_plots": True,
            "save_csv": True,
        }
    },
}


def _merge_dict(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for k, v in (update or {}).items():
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

    experiment = cfg.setdefault("experiment", {})
    experiment.setdefault("name", "p79_experiment")
    experiment.setdefault("benchmark", "visualwebarena")
    experiment.setdefault("phase", "phase1")
    experiment.setdefault("seed", 42)
    experiment.setdefault("output_root", "results")
    sites = cfg.get("task", {}).get("include_sites", [])
    if len(sites) == 1:
        experiment.setdefault("run_id", f"run_{sites[0]}_{int(time.time())}")
    else:
        experiment.setdefault("run_id", f"run_{int(time.time())}")

    env_cfg = cfg.setdefault("env", {})
    env_cfg.setdefault("type", "vwa")
    env_cfg.setdefault("headless", True)
    env_cfg.setdefault("observation_type", "accessibility_tree")
    env_cfg.setdefault("dry_run", False)
    env_cfg.setdefault("viewport_width", 1280)
    env_cfg.setdefault("viewport_height", 720)

    task_cfg = cfg.setdefault("task", {})
    task_cfg.setdefault("include_sites", ["shopping", "reddit", "wikipedia", "classifieds"])
    task_cfg.setdefault("max_tasks_per_site", None)
    task_cfg.setdefault("task_ids", {})

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

    cfg.setdefault("checklist", {})
    cfg["checklist"].setdefault("enabled", False)
    cfg["checklist"].setdefault("inject_into_prompt", True)
    cfg["checklist"].setdefault("max_items", 4)

    cfg.setdefault("state_change", {})
    cfg["state_change"].setdefault("similarity_threshold", 0.95)

    cfg.setdefault("metrics", {}).setdefault("energy", {})
    cfg["metrics"]["energy"].setdefault("hardware_profile", "m2")
    cfg["metrics"]["energy"].setdefault("region", "world")
    cfg["metrics"]["energy"].setdefault("use_psutil", True)
    cfg["metrics"]["energy"].setdefault("fixed_power_watts", None)
    cfg["metrics"]["energy"].setdefault("use_pynvml", True)
    cfg["metrics"]["energy"].setdefault("sample_interval_s", 0.5)
    cfg["metrics"]["energy"].setdefault("track_model_load", False)
    cfg["metrics"]["energy"].setdefault("model_load_amortize_over", 0)

    return cfg


def resolve_output_root(cfg: Dict[str, Any]) -> Path:
    experiment = cfg["experiment"]
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
