from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


class LoggerV2:
    def __init__(self, condition_dir: Path):
        self.condition_dir = condition_dir
        self.episodes_dir = self.condition_dir / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)

    def write_condition_meta(self, meta: Dict[str, Any]) -> None:
        self.condition_dir.mkdir(parents=True, exist_ok=True)
        with open(self.condition_dir / "condition_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

    def step_log_path(self, site: str, task_id: int) -> Path:
        return self.episodes_dir / f"{site}_task_{task_id}_steps_v2.jsonl"

    def summary_path(self, site: str, task_id: int) -> Path:
        return self.episodes_dir / f"{site}_task_{task_id}_summary_v2.json"

    def write_step(self, site: str, task_id: int, record: Dict[str, Any]) -> None:
        path = self.step_log_path(site, task_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def write_episode_summary(self, site: str, task_id: int, summary: Dict[str, Any]) -> None:
        path = self.summary_path(site, task_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

    def condition_summary_path(self) -> Path:
        return self.condition_dir / "condition_summary_v2.json"

    def write_condition_summary(self, payload: Dict[str, Any]) -> None:
        self.condition_dir.mkdir(parents=True, exist_ok=True)
        with open(self.condition_summary_path(), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
