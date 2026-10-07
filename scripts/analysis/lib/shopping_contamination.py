"""Which shopping episodes are touched by a known harness defect — one reader, per episode.

B-2002 (master_bug_catalog, 笔记 §531.9): on VWA shopping, a TYPE into a search box that already
holds a query submits old + new (`chair` + `chair with wheels` -> `q=chairchair with wheels`).
Criterion, as the catalog defines it: a TYPE step with non-empty text whose URL BEFORE the action
already carries `q=`, and whose URL AFTER carries q == q_before + text. The URL after a step is
that step's `obs_url` (笔记 §292); the URL before is the previous step's `obs_url`, or the task's
`start_url` for step 0. The catalog's counts (e.g. B1 P-text 385 of 756 such TYPE steps, 112
episodes) are the check that this reader matches the definition (笔记 §547).

Also exported: the task ids a cross-mode reading on shopping must drop for reasons that are not
about the agent — B-2003 (wishlist not reset, tasks 108/159/160/163) and the 42 grid-order
sensitive tasks of `vwa_shopping/grid_order_drift_20261007.json` (笔记 §534).
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from p79.experiment.io_utils import read_jsonl_dedup

REPO = Path(__file__).resolve().parents[3]
GRID_DRIFT = REPO / "docs/analysis/vwa_shopping/grid_order_drift_20261007.json"
B2003_TASKS = frozenset({108, 159, 160, 163})


def _q(url: str | None) -> str | None:
    if not url:
        return None
    v = parse_qs(urlparse(url).query).get("q")
    return v[0] if v else None


def episode_b2002(steps_path: Path, start_url: str | None) -> dict:
    """{'typed_into_filled': n TYPE steps whose before-URL had q=, 'appended': n of those matching
    the defect}. A step with no parseable URL is skipped, never counted as clean evidence."""
    recs = sorted(read_jsonl_dedup(steps_path), key=lambda r: r.get("step_idx", 0))
    before = start_url
    filled = appended = 0
    for r in recs:
        a = r.get("action") or {}
        after = r.get("obs_url")
        if str(a.get("action_type", "")).lower() == "type":
            text = str(a.get("text") or "").replace("\n", "").strip()
            qb, qa = _q(before), _q(after)
            if text and qb:
                filled += 1
                if qa == qb + text:
                    appended += 1
        if after:
            before = after
    return {"typed_into_filled": filled, "appended": appended}


def start_urls(task_config_dir: Path) -> dict[int, str]:
    out = {}
    for f in task_config_dir.glob("shopping_task_*.json"):
        j = json.loads(f.read_text(encoding="utf-8"))
        if j.get("task_id") is not None:
            out[int(j["task_id"])] = j.get("start_url")
    return out


def grid_order_sensitive_tasks() -> frozenset[int]:
    d = json.loads(GRID_DRIFT.read_text(encoding="utf-8"))
    for key in ("order_sensitive_tasks", "position_dependent_tasks", "tasks"):
        if key in d:
            ids = d[key]
            return frozenset(int(t if not isinstance(t, dict) else t["task_id"]) for t in ids)
    raise KeyError(f"{GRID_DRIFT}: no task list under a known key ({sorted(d)})")
