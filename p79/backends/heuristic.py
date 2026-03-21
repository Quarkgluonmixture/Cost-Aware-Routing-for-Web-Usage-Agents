from __future__ import annotations

from typing import Any, Dict, Tuple

from p79.backends.action_utils import extract_candidate_query, first_element_id_by_keyword
from p79.backends.base import BackendStepContext


class HeuristicDomBackend:
    """Cheap fallback backend for DOM-only steps without model calls."""

    backend_id = "heuristic_dom"

    def step(self, instruction: str, obs: Any, context: BackendStepContext) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        obs_text = getattr(obs, "text", "") or ""

        input_id = first_element_id_by_keyword(obs_text, ("textbox", "input", "search", "edit"))
        button_id = first_element_id_by_keyword(obs_text, ("button", "link", "menu item", "option"))

        lower_instruction = (instruction or "").lower()
        if any(k in lower_instruction for k in ("search", "find", "look for", "buy", "add")) and input_id is not None:
            query = extract_candidate_query(instruction)
            action = {
                "thought": "Use a low-cost DOM strategy: type query into a likely input field.",
                "action_type": "type",
                "element_id": input_id,
                "text": f"{query}\n",
            }
            reason = "dom_query_input"
        elif button_id is not None:
            action = {
                "thought": "Use a low-cost DOM strategy: click a likely actionable element.",
                "action_type": "click",
                "element_id": button_id,
            }
            reason = "dom_click_candidate"
        else:
            action = {
                "thought": "No reliable DOM anchor found; move viewport to gather more context.",
                "action_type": "scroll",
                "delta": [0, 0.8],
                "coordinate_type": "normalized",
            }
            reason = "dom_scroll_fallback"

        meta = {
            "raw_output": action,
            "valid": True,
            "failure_reason": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "model_calls": 0,
            "backend_type": "heuristic_dom",
            "dom_reason": reason,
        }
        return action, meta
