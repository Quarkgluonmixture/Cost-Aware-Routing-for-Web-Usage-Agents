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
    # B-1884 / Fix 4 (2026-06-25): reddit task 138 ("change my username")
    # renames the shared test account; username IS the login credential, so the
    # periodic fresh re-login above would fail-closed. Restore it idempotently
    # at the START of every reddit task (before auth_refresh) — mirrors the
    # per-task require_reset classifieds gets from upstream VWA (which never
    # implemented a reddit reset). Verified DB path (笔记 §354): postmill DB
    # inside the vwa-reddit container, reached as `su - postgres` (peer auth),
    # table users / id=13915. Defaults overridable here. See
    # p79/utils/reddit_identity.py + PROTOCOL_NOTE_04.
    "reddit_identity_reset": {
        "enabled": True,
        "container": "vwa-reddit",
        "db": "postmill",
        "db_os_user": "postgres",
        "table": "users",
        "username_column": "username",
        # postmill login matches the lowercase canonical column, NOT username —
        # both must be restored (A100-verified 2026-06-25). "" to skip.
        "normalized_username_column": "normalized_username",
        "seed_normalized_username": "",  # "" → auto = seed_username.lower()
        "user_id": 13915,
        "seed_username": "MarvelsGrantMan136",
        "sql_override": "",
        "timeout_s": 30,
        "fail_closed": False,
    },
    # B-1936 / PROTOCOL_NOTE_07 (2026-08-03): shopping per-task cart isolation
    # — **ENABLED** per user decision after the /stress 3-AI round.
    # Upstream's `require_reset` is a no-op on shopping (envs.py:172 implements
    # classifieds only); 108 of 466 VWA shopping tasks mutate the cart and 104
    # are graded by loading the cart page, so within a condition the cart
    # accumulates and those substring evaluators drift toward false success.
    #
    # Why enabled rather than disclosed (reverses the 2026-08-03 morning call):
    #   * the "cross-site estimand consistency" argument for leaving it was
    #     FALSE — classifieds already gets 22 per-task full-site resets from
    #     upstream while reddit/shopping get none, so there is no consistency to
    #     protect (only heterogeneity to choose between);
    #   * shopping is PRE-DATA (zero VWA shopping runs on disk), so this defines
    #     shopping's estimand instead of changing a measured one, and the same
    #     fix after firing would cost 18 condition re-runs;
    #   * meta-analysis handles documented protocol heterogeneity; it cannot
    #     launder unidirectional measurement error.
    # reddit keeps its accumulation (already bound; §402.7 disclose-only stands).
    #
    # Cleared before EVERY shopping task, not just the 19 upstream flags: those
    # flags hit only 1 of the 10 tasks actually at risk, and they cannot see the
    # diffuse channel (an agent adding to the cart while exploring any task).
    # Idempotent — on an empty cart the DELETE matches 0 rows.
    #
    # fail_closed = "auto" (B-1943): hard-fail under P79_PAPER_GRADE=1, warn on
    # dev boxes. A silent clear failure under fire would let a condition run over
    # an unknown cart state, indistinguishable from clean in every summary.
    "shopping_cart_reset": {
        "enabled": True,
        "container": "vwa-shopping",
        "db": "magentodb",
        "db_user": "magentouser",
        "db_password": "MyPassword",
        "quote_table": "quote",
        "quote_item_table": "quote_item",
        # "" → $VWA_SHOPPING_USER, then the WebArena seed account emma.lopez.
        "customer_email": "",
        "sql_override": "",
        "timeout_s": 30,
        # "auto" = hard-fail under P79_PAPER_GRADE=1, warn otherwise (B-1943).
        # Unconditional True broke every dev/smoke run on a box without the
        # shopping container; unconditional False was the silent-skip codex F7
        # flagged. "auto" is the same shape as VWA_RESET_MODE.
        "fail_closed": "auto",
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


def _normalize_defaults_field(cfg: Dict[str, Any]) -> list:
    """Read a config's `defaults:` field as a list.

    Tolerates the single-string form (`defaults: "base.yaml"`) — wrapping it
    avoids iterating the characters of the string.
    """
    defaults = cfg.get("defaults", [])
    if isinstance(defaults, str):
        return [defaults]
    if not isinstance(defaults, (list, tuple)):
        return []
    return list(defaults)


def _resolve_default_path(default_cfg: str, referrer: Path) -> Path:
    """Resolve a `defaults:` entry: try as given (cwd-relative), then relative
    to the file that referenced it."""
    default_path = Path(default_cfg)
    if not default_path.exists():
        candidate = referrer.parent / default_cfg
        if candidate.exists():
            default_path = candidate
    if not default_path.exists():
        raise FileNotFoundError(f"Referenced default config not found: {default_cfg}")
    return default_path


def _load_defaults_chain(path: Path, _seen: frozenset = frozenset()) -> Dict[str, Any]:
    """Load one config file with its OWN `defaults:` chain already applied.

    Returns the raw merged body (DEFAULT_CONFIG is NOT folded in here — that
    stays the caller's job, so the outer fold order is unchanged).

    B-1888 (2026-07-27): this function is why multi-level inheritance works at
    all. `load_experiment_config` used to read each `defaults:` entry with a
    bare `yaml.safe_load`, so a base config's own `defaults:` was never
    followed — it was merged in as an inert `defaults` KEY. Single-level chains
    (every VWA per-condition config -> exp_v2_base.yaml) were unaffected and so
    the gap stayed invisible for two months. Two-level chains were not: all 55
    WA configs inherit exp_v2_wa_base.yaml, which in turn declares
    exp_v2_base.yaml — meaning every WA config silently resolved WITHOUT the
    base layer. That drops `backends.local_4b.type` (the crash that surfaced
    this), and with it the model path, the OOM guard, token pricing, carbon
    intensity, and the tool-calling defaults. It only surfaced now because WA
    had never been fired.

    For a base file that declares no `defaults:` of its own, the return value
    equals the old `yaml.safe_load` result minus the absent `defaults` key —
    so single-level chains parse byte-identically to before (asserted by
    `test_defaults_chain_vwa_single_level_unchanged`).
    """
    resolved = path.resolve()
    if resolved in _seen:
        cycle = " -> ".join(str(p) for p in (*_seen, resolved))
        raise ValueError(f"Circular `defaults:` reference in config chain: {cycle}")
    _seen = _seen | {resolved}

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    acc: Dict[str, Any] = {}
    for default_cfg in _normalize_defaults_field(cfg):
        acc = _merge_dict(acc, _load_defaults_chain(
            _resolve_default_path(default_cfg, path), _seen))

    body = dict(cfg)
    body.pop("defaults", None)
    return _merge_dict(acc, body)


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    merged = copy.deepcopy(DEFAULT_CONFIG)
    for default_cfg in _normalize_defaults_field(cfg):
        # B-1888: recurse, so a base config's own `defaults:` is honoured.
        base_piece = _load_defaults_chain(_resolve_default_path(default_cfg, path))
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

    # B-1881 (reddit chain abort #3/#4, 2026-06-20): bounded episode-level retry
    # budget for TRANSIENT-substrate quarantines (auth / proxy_5xx / network).
    # The runner's transient-retry wrapper (_run_and_record_episode) retries the
    # episode on fresh substrate up to this many times instead of fail-closed
    # aborting the whole condition; non-transient quarantines + exhaustion still
    # abort. yaml-exposed for reproducibility (reviewer can set 0 to restore the
    # legacy single-attempt fail-closed). Applies only under paper_grade=True
    # (and never under diagnostic_replay).
    cfg.setdefault("transient_episode_max_retries", 3)

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
    # P2-5 (/stress accounting audit 2026-05-21): guard against an explicit yaml
    # override setting max_model_attempts BELOW the primary budget + parse cap,
    # which would let the safety ceiling truncate an episode before the agent has
    # spent its action budget (silent capability under-measurement). Clamp up.
    _budget_floor = int(cfg["runtime"]["max_agent_actions"]) + int(cfg["runtime"]["max_total_parse_errors"])
    if int(cfg["runtime"]["max_model_attempts"]) < _budget_floor:
        cfg["runtime"]["max_model_attempts"] = _budget_floor

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
