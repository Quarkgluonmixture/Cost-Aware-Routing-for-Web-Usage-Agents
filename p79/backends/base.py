from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple


@dataclass
class BackendStepContext:
    observation_mode: str
    som_enabled: bool
    som_text: str
    stage: str = "single"
    planner_sub_goal: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    module_flags: Dict[str, bool] = field(default_factory=dict)
    reference_images: List[Any] = field(default_factory=list)  # PIL Images from task config


class AgentBackend(Protocol):
    backend_id: str

    def step(
        self,
        instruction: str,
        obs: Any,
        context: BackendStepContext,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...


class BackendError(RuntimeError):
    pass
