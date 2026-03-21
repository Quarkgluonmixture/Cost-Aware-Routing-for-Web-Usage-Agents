from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RouterState:
    unchanged_streak: int = 0
    no_progress_streak: int = 0
    checklist_stall_streak: int = 0
    prev_checklist_completed: Optional[int] = None


class RuleBasedRouter:
    def __init__(self, cfg: Dict[str, Any]):
        router_cfg = cfg.get("router", {})
        thresholds = router_cfg.get("thresholds", {})
        checklist_cfg = router_cfg.get("checklist_trigger", {})

        self.cheap_default_mode = str(router_cfg.get("cheap_default_mode", "dom_only"))
        self.dom_size_threshold = int(thresholds.get("dom_size_threshold", 12000))
        self.unchanged_steps_trigger = int(thresholds.get("unchanged_steps_trigger", 2))
        self.no_progress_steps_trigger = int(thresholds.get("no_progress_steps_trigger", 2))
        self.checklist_trigger_enabled = bool(checklist_cfg.get("enabled", False))
        self.checklist_stalled_steps_trigger = int(checklist_cfg.get("stalled_steps_trigger", 2))
        self.checklist_failed_item_trigger = bool(checklist_cfg.get("failed_item_trigger", True))

    def decide(
        self,
        router_enabled: bool,
        preferred_mode: str,
        obs_text: str,
        state: RouterState,
        prev_action_success: Optional[bool],
        prev_page_changed: Optional[bool],
        checklist_status: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, List[str], Dict[str, float], RouterState]:
        start = time.time()

        if prev_page_changed is False:
            state.unchanged_streak += 1
        elif prev_page_changed is True:
            state.unchanged_streak = 0

        if prev_action_success is False:
            state.no_progress_streak += 1
        elif prev_action_success is True:
            state.no_progress_streak = 0

        triggers: List[str] = []

        dom_parse_start = time.time()
        dom_size = len(obs_text or "")
        dom_parse_ms = (time.time() - dom_parse_start) * 1000.0

        if dom_size > self.dom_size_threshold:
            triggers.append("dom_size_exceeds_threshold")
        if prev_action_success is False:
            triggers.append("action_failed")
        if state.unchanged_streak >= self.unchanged_steps_trigger:
            triggers.append("page_unchanged_streak")
        if state.no_progress_streak >= self.no_progress_steps_trigger:
            triggers.append("no_progress_streak")

        if self.checklist_trigger_enabled and checklist_status:
            total = int(checklist_status.get("total", 0) or 0)
            completed = int(checklist_status.get("completed", 0) or 0)
            failed = int(checklist_status.get("failed", 0) or 0)

            if total > 0:
                if state.prev_checklist_completed is None:
                    state.prev_checklist_completed = completed
                    state.checklist_stall_streak = 0
                else:
                    if completed <= int(state.prev_checklist_completed):
                        state.checklist_stall_streak += 1
                    else:
                        state.checklist_stall_streak = 0
                    state.prev_checklist_completed = completed

                if state.checklist_stall_streak >= self.checklist_stalled_steps_trigger:
                    triggers.append("checklist_progress_stalled")

            if self.checklist_failed_item_trigger and failed > 0:
                triggers.append("checklist_has_failed_items")

        if not router_enabled:
            decision = preferred_mode
        else:
            decision = "hybrid" if triggers else self.cheap_default_mode

        router_decision_ms = (time.time() - start) * 1000.0
        overhead = {
            "router_decision_ms": router_decision_ms,
            "extra_dom_parse_ms": dom_parse_ms,
            "extra_screenshot_ms": 0.0,
            "extra_model_calls": 0.0,
            "routing_retry_count": 0.0,
        }

        return decision, triggers, overhead, state
