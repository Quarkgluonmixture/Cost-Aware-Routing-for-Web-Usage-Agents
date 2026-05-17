from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from p79.experiment.types import TaskSpec

logger = logging.getLogger(__name__)


def _is_na_task(task: Dict[str, Any]) -> bool:
    """True if the task's reference answer is N/A (unanswerable task) — §139.8.

    N/A tasks (all `string_match` + `reference_answers.fuzzy_match == "N/A"`
    across all VWA + WA sites) are excluded from the scored set: under the
    no-N/A-exit agent prompt they are un-passable (zero discriminative signal),
    and the VWA evaluator cannot distinguish a reasoned N/A judgement from an
    early exit (WebArena-Verified). Pre-registered exclusion.
    """
    ref = (task.get("eval") or {}).get("reference_answers") or {}
    return isinstance(ref, dict) and ref.get("fuzzy_match") == "N/A"


PLACEHOLDER_DEFAULTS = {
    "__REDDIT__": "http://localhost:9999",
    "__SHOPPING__": "http://localhost:7770",
    "__SHOPPING_ADMIN__": "http://localhost:7780/admin",
    "__GITLAB__": "http://localhost:8023",
    "__WIKIPEDIA__": "http://localhost:8888",
    "__MAP__": "http://localhost:3000",
    "__HOMEPAGE__": "http://localhost:4399",
    "__CLASSIFIEDS__": "http://localhost:9980",
}


def _replace_placeholders(obj: Any, mapping: Dict[str, str]) -> Any:
    if isinstance(obj, str):
        out = obj
        for k, v in mapping.items():
            out = out.replace(k, v)
        return out
    if isinstance(obj, list):
        return [_replace_placeholders(x, mapping) for x in obj]
    if isinstance(obj, dict):
        return {k: _replace_placeholders(v, mapping) for k, v in obj.items()}
    return obj


def _placeholder_mapping() -> Dict[str, str]:
    mapping = dict(PLACEHOLDER_DEFAULTS)
    mapping["__REDDIT__"] = os.environ.get("REDDIT", mapping["__REDDIT__"])
    mapping["__SHOPPING__"] = os.environ.get("SHOPPING", mapping["__SHOPPING__"])
    mapping["__SHOPPING_ADMIN__"] = os.environ.get("SHOPPING_ADMIN", mapping["__SHOPPING_ADMIN__"])
    mapping["__GITLAB__"] = os.environ.get("GITLAB", mapping["__GITLAB__"])
    mapping["__WIKIPEDIA__"] = os.environ.get("WIKIPEDIA", mapping["__WIKIPEDIA__"])
    mapping["__MAP__"] = os.environ.get("MAP", mapping["__MAP__"])
    mapping["__HOMEPAGE__"] = os.environ.get("HOMEPAGE", mapping["__HOMEPAGE__"])
    mapping["__CLASSIFIEDS__"] = os.environ.get("CLASSIFIEDS", mapping["__CLASSIFIEDS__"])

    # Wikipedia ZIM version rewrite — user's Kiwix loads `2025-08`, but VWA raw
    # configs hardcode `2022-05` (§81). Default to the version actually mounted
    # on quark; env override lets future ZIM upgrades change this without code.
    #
    # /stress A1.18-re (B-602 P2-11-C* Gemini OOB, 2026-05-17): pre-fix only
    # mapped the single `2022-05` date string, so if upstream VWA updated its
    # raw task configs to a different historical ZIM date (e.g. `2024-01`), the
    # key-based replace silently no-op'd and tasks would point at a missing ZIM
    # at runtime. Now: (a) expose `WIKIPEDIA_ZIM_LEGACY_VERSIONS` env var as
    # comma-separated list of all legacy date strings to rewrite (default
    # covers known VWA upstream dates 2022-05 + 2023-04 + 2024-01); (b) the
    # rewriter maps every listed legacy version to the current target. Future
    # VWA upstream merges can extend this list via env var without code edit.
    zim_override = os.environ.get(
        "WIKIPEDIA_ZIM_VERSION", "wikipedia_en_all_maxi_2025-08"
    )
    legacy_versions_csv = os.environ.get(
        "WIKIPEDIA_ZIM_LEGACY_VERSIONS",
        "2022-05,2023-04,2024-01",
    )
    for legacy_date in (v.strip() for v in legacy_versions_csv.split(",") if v.strip()):
        legacy_key = f"wikipedia_en_all_maxi_{legacy_date}"
        if legacy_key != zim_override:  # never self-map
            mapping[legacy_key] = zim_override

    return mapping


def _site_matches(site: str, task: Dict[str, Any]) -> bool:
    sites = [str(s).lower() for s in task.get("sites", [])]
    if site in sites:
        return True
    # Exact placeholder match avoids substring collision (e.g. shopping vs shopping_admin)
    placeholder = f"__{site}__"
    start_url = str(task.get("start_url", "")).lower()
    if placeholder in start_url:
        return True
    return False


def _load_json_tasks(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unexpected task config payload type for {path}: {type(payload)}")


def load_tasks(cfg: Dict[str, Any], output_dir: Path) -> List[TaskSpec]:
    benchmark = cfg["experiment"]["benchmark"]
    task_cfg = cfg.get("task", {})

    include_sites = [s.lower() for s in task_cfg.get("include_sites", [])]
    task_ids_map = task_cfg.get("task_ids", {}) or {}
    max_tasks_per_site = task_cfg.get("max_tasks_per_site")
    # §139.8: exclude N/A (unanswerable) tasks from the scored set at load time.
    # Pre-registered scope decision — see preregistration.md. Set False only for
    # a dedicated N/A-capability study.
    exclude_na_tasks = bool(task_cfg.get("exclude_na_tasks", True))

    site_configs = task_cfg.get("site_configs", {})
    if not site_configs and task_cfg.get("config_file"):
        site_configs = {site: task_cfg["config_file"] for site in include_sites}

    if not site_configs:
        raise ValueError("task.site_configs is required for the unified experiment runner")

    task_config_dir = output_dir / "task_configs"
    task_config_dir.mkdir(parents=True, exist_ok=True)

    mapping = _placeholder_mapping()
    all_specs: List[TaskSpec] = []

    for site in include_sites:
        site_path = site_configs.get(site)
        if not site_path:
            continue

        tasks = _load_json_tasks(site_path)
        selected_ids = set(task_ids_map.get(site, []))

        selected: List[Dict[str, Any]] = []
        n_na_excluded = 0
        for t in tasks:
            if selected_ids and int(t.get("task_id", -1)) not in selected_ids:
                continue
            if not _site_matches(site, t):
                continue
            if exclude_na_tasks and _is_na_task(t):
                n_na_excluded += 1
                continue
            selected.append(t)
        if n_na_excluded:
            logger.info(
                "load_tasks[%s]: excluded %d N/A task(s) from scored set (§139.8)",
                site, n_na_excluded,
            )

        if max_tasks_per_site is not None:
            selected = selected[: int(max_tasks_per_site)]

        for task in selected:
            task_copy = copy.deepcopy(task)
            resolved_task = _replace_placeholders(task_copy, mapping)

            task_id = int(resolved_task["task_id"])
            config_path = task_config_dir / f"{site}_task_{task_id}.json"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(resolved_task, f, indent=2, ensure_ascii=False)

            all_specs.append(
                TaskSpec(
                    benchmark=benchmark,
                    site=site,
                    task_id=task_id,
                    intent=str(resolved_task.get("intent", "")),
                    config_file=str(config_path),
                    raw_task=resolved_task,
                )
            )

    return all_specs
