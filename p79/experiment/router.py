from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RouterState:
    unchanged_streak: int = 0
    no_progress_streak: int = 0
    checklist_stall_streak: int = 0
    prev_checklist_completed: Optional[int] = None
    # 3-way routing state
    dom_complexity_history: List[int] = field(default_factory=list)
    text_length_history: List[int] = field(default_factory=list)
    current_mode: str = "dom"
    success_streak: int = 0


class RuleBasedRouter:
    def __init__(self, cfg: Dict[str, Any]):
        router_cfg = cfg.get("router", {})
        thresholds = router_cfg.get("thresholds", {})
        checklist_cfg = router_cfg.get("checklist_trigger", {})

        self.cheap_default_mode = str(router_cfg.get("cheap_default_mode", "dom"))
        self.rich_escalation_mode = str(router_cfg.get("rich_escalation_mode", "som"))
        self.dom_size_threshold = int(thresholds.get("dom_size_threshold", 12000))
        self.unchanged_steps_trigger = int(thresholds.get("unchanged_steps_trigger", 2))
        self.no_progress_steps_trigger = int(thresholds.get("no_progress_steps_trigger", 2))
        self.checklist_trigger_enabled = bool(checklist_cfg.get("enabled", False))
        self.checklist_stalled_steps_trigger = int(checklist_cfg.get("stalled_steps_trigger", 2))
        self.checklist_failed_item_trigger = bool(checklist_cfg.get("failed_item_trigger", True))

        # 3-way routing config
        self.modes: List[str] = list(
            router_cfg.get("modes", [self.cheap_default_mode, self.rich_escalation_mode])
        )
        self.dom_complexity_trigger = int(thresholds.get("dom_complexity_trigger", 500))
        self.text_length_trigger = int(thresholds.get("text_length_trigger", 12000))
        self.deescalation_streak = int(thresholds.get("deescalation_streak", 3))
        self.history_window = int(thresholds.get("history_window", 5))

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
            state.success_streak = 0
        elif prev_action_success is True:
            state.no_progress_streak = 0
            state.success_streak += 1

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

        # DOM complexity / text length triggers (from state_change enrichment)
        if state.dom_complexity_history:
            latest_dc = state.dom_complexity_history[-1]
            if latest_dc > self.dom_complexity_trigger:
                triggers.append("dom_complexity_high")
        if state.text_length_history:
            latest_tl = state.text_length_history[-1]
            if latest_tl > self.text_length_trigger:
                triggers.append("text_length_high")

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
            if triggers:
                # Escalate: pick the next more expensive mode from modes list
                current_idx = (
                    self.modes.index(state.current_mode)
                    if state.current_mode in self.modes
                    else 0
                )
                decision = self.modes[min(current_idx + 1, len(self.modes) - 1)]
            elif state.success_streak >= self.deescalation_streak and state.current_mode != self.modes[0]:
                # De-escalate after sustained success
                current_idx = (
                    self.modes.index(state.current_mode)
                    if state.current_mode in self.modes
                    else 0
                )
                decision = self.modes[max(current_idx - 1, 0)]
                state.success_streak = 0  # reset after de-escalation
            else:
                decision = self.cheap_default_mode
        state.current_mode = decision

        router_decision_ms = (time.time() - start) * 1000.0
        overhead = {
            "router_decision_ms": router_decision_ms,
            "extra_dom_parse_ms": dom_parse_ms,
            "extra_screenshot_ms": 0.0,
            "extra_model_calls": 0.0,
            "routing_retry_count": 0.0,
        }

        return decision, triggers, overhead, state
