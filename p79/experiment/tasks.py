from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from p79.experiment.types import TaskSpec


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
    return mapping


def _site_matches(site: str, task: Dict[str, Any]) -> bool:
    sites = [str(s).lower() for s in task.get("sites", [])]
    start_url = str(task.get("start_url", "")).lower()

    if site in sites:
        return True
    if site in start_url:
        return True

    if site == "wikipedia":
        return ("wikipedia" in sites) or ("__wikipedia__" in start_url)
    if site == "reddit":
        return ("reddit" in sites) or ("__reddit__" in start_url)
    if site == "shopping":
        return ("shopping" in sites) or ("__shopping__" in start_url)
    if site == "classifieds":
        return ("classifieds" in sites) or ("__classifieds__" in start_url)

    return True


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
        for t in tasks:
            if selected_ids and int(t.get("task_id", -1)) not in selected_ids:
                continue
            if not _site_matches(site, t):
                continue
            selected.append(t)

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
