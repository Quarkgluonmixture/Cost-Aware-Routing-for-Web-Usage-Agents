from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


# Adapted from external_code/checklist.py + checklist_manager.py (Aiden Yiliu Li, Apache-2.0)
@dataclass
class ChecklistItem:
    id: str
    description: str
    status: str = "pending"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_checklist_status(task_checklist: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not task_checklist:
        return {
            "total": 0,
            "completed": 0,
            "in_progress": 0,
            "pending": 0,
            "failed": 0,
            "progress": 0.0,
            "completion_rate": 0.0,
            "success_rate": 0.0,
        }

    status_counts = {"pending": 0, "in_progress": 0, "completed": 0, "failed": 0}
    for item in task_checklist:
        status = str(item.get("status", "pending") or "pending").strip().lower().replace(" ", "_")
        if status not in status_counts:
            status = "pending"
        status_counts[status] += 1

    total = len(task_checklist)
    completed = status_counts["completed"]
    failed = status_counts["failed"]

    completion_rate = completed / total if total > 0 else 0.0
    success_rate = completed / (completed + failed) if (completed + failed) > 0 else 0.0

    return {
        "total": total,
        "completed": completed,
        "in_progress": status_counts["in_progress"],
        "pending": status_counts["pending"],
        "failed": failed,
        "progress": completion_rate * 100.0,
        "completion_rate": completion_rate,
        "success_rate": success_rate,
    }


def format_checklist_for_prompt(task_checklist: List[Dict[str, Any]]) -> str:
    if not task_checklist:
        return "No checklist available"

    status = get_checklist_status(task_checklist)
    lines = ["TASK CHECKLIST:"]
    for i, item in enumerate(task_checklist, 1):
        st = str(item.get("status", "pending")).lower()
        mark = {
            "pending": "[PENDING]",
            "in_progress": "[IN_PROGRESS]",
            "completed": "[COMPLETED]",
            "failed": "[FAILED]",
        }.get(st, "[PENDING]")
        lines.append(f"{i:2d}. {mark} {item.get('id', f'requirement_{i}')}: {item.get('description', '')}")

    lines.append(
        f"Progress {status['completed']}/{status['total']} ({status['progress']:.1f}%), failed={status['failed']}"
    )
    return "\n".join(lines)


class ChecklistManagerLite:
    """
    Lightweight checklist manager for deferred module integration.

    No extra model call by default. It atomizes task description and updates status
    based on action outcomes.
    """

    def __init__(self, task_description: str, max_items: int = 4):
        self.task_description = task_description or ""
        self.max_items = max_items
        self.task_checklist: List[Dict[str, Any]] = self._generate_atomic_checklist(self.task_description)

    def _generate_atomic_checklist(self, task_description: str) -> List[Dict[str, Any]]:
        raw = task_description.strip()
        if not raw:
            return [ChecklistItem(id="requirement_1", description="Complete task", status="pending").as_dict()]

        normalized = re.sub(r"\s+", " ", raw)
        normalized = normalized.replace(";", ",")
        normalized = re.sub(r"\b(and|then|,|->)\b", "|", normalized, flags=re.IGNORECASE)
        parts = [p.strip(" .") for p in normalized.split("|") if p.strip(" .")]
        if not parts:
            parts = [raw]

        dedup = []
        seen = set()
        for part in parts:
            short = " ".join(part.split()[:12])
            key = short.lower()
            if key in seen:
                continue
            seen.add(key)
            dedup.append(short)
            if len(dedup) >= self.max_items:
                break

        items = []
        for i, desc in enumerate(dedup, 1):
            items.append(ChecklistItem(id=f"requirement_{i}", description=desc, status="pending").as_dict())
        return items

    def get_status(self) -> Dict[str, Any]:
        return get_checklist_status(self.task_checklist)

    def format_for_prompt(self) -> str:
        return format_checklist_for_prompt(self.task_checklist)

    def _update_item(self, item_id: str, status: str) -> bool:
        normalized = (status or "pending").strip().lower().replace(" ", "_")
        if normalized not in {"pending", "in_progress", "completed", "failed"}:
            return False

        for item in self.task_checklist:
            if item.get("id") == item_id:
                item["status"] = normalized
                return True
        return False

    def update_after_action(self, action_success: bool, error: Optional[str]) -> None:
        target_idx = None
        for i, item in enumerate(self.task_checklist):
            st = item.get("status", "pending")
            if st in ("in_progress", "pending"):
                target_idx = i
                if st == "in_progress":
                    break

        if target_idx is None:
            return

        item = self.task_checklist[target_idx]
        old_status = item.get("status", "pending")

        if action_success and not error:
            new_status = "completed" if old_status == "in_progress" else "in_progress"
        else:
            new_status = "failed"

        self._update_item(str(item.get("id")), new_status)
