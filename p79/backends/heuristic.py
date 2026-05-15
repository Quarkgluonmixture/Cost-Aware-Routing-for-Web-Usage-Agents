from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from p79.backends.action_utils import extract_candidate_query, first_element_id_by_keyword
from p79.backends.base import BackendStepContext


class HeuristicDomBackend:
    """Cheap fallback backend for DOM-only steps without model calls.

    Constructor accepts ``(backend_id, config)`` to match the other backends
    (api_proxy / local_qwen / local_gemma / MockBackend) — previously the
    factory's heuristic_dom branch silently dropped the entire cfg dict and
    mutated `backend_id` as an instance attribute after construction (/stress
    A1.2 F3). The cfg is stored even though this backend doesn't currently
    consume any keys, so any future config-driven behavior (e.g. mock_mode)
    flows through the normal dispatch path.
    """

    def __init__(self, backend_id: str = "heuristic_dom", config: Optional[Dict[str, Any]] = None):
        self.backend_id = backend_id
        self.config = config or {}

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
