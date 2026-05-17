from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple


_VALID_STAGES = frozenset({"single", "planner", "grounder"})


@dataclass
class BackendStepContext:
    """Per-step context passed from runner to backend wrappers.

    B-810 (/stress A1.2 cold-start P1-3-A* Claude OOB, 2026-05-17):
    ``reference_images`` migrated from ``List[Any]`` (mutable + pass-by-
    reference) to ``Tuple[Any, ...]`` (immutable). Pre-fix any agent
    introducing ``reference_images.append(...)`` / ``.pop()`` semantics
    would silently mutate the same list across episode steps and across
    M4 planner/grounder paired contexts, violating paper §3.4 step-
    independence assumption. Type-system enforces invariant; agents can
    still iterate via ``for ref_img in context.reference_images: ...``.

    B-811 (/stress A1.2 cold-start P1-6-B* codex OOB, 2026-05-17): stage
    enum strict-validated at construction. Pre-fix typos like
    ``stage="planer"`` (missing `n`) silently fell through to the
    single-stage prompt path in all 3 wrappers, contaminating any future
    Phase 2 planner/grounder routing ablation with mislabeled stage
    records.
    """

    observation_mode: str
    som_enabled: bool
    som_text: str
    stage: str = "single"
    planner_sub_goal: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    module_flags: Dict[str, bool] = field(default_factory=dict)
    reference_images: Tuple[Any, ...] = field(default_factory=tuple)  # PIL Images, immutable

    def __post_init__(self) -> None:
        # B-811: enforce stage enum at the boundary.
        if self.stage not in _VALID_STAGES:
            raise ValueError(
                f"BackendStepContext.stage={self.stage!r} not in {_VALID_STAGES} — "
                f"check yaml config / Phase 2 routing dispatch for typos."
            )
        # B-810: tolerate list at the boundary by freezing, so the runner can
        # still pass `reference_images=reference_images` where the local var
        # was built as a mutable list during episode load (current call sites
        # at runner/main.py:1889/1902). Defensive freeze preserves the
        # immutability contract without breaking callers mid-cutover.
        if not isinstance(self.reference_images, tuple):
            self.reference_images = tuple(self.reference_images)


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
