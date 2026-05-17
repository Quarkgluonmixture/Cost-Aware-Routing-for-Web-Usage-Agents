"""Single-source stage_prefix builder for cross-baseline planner/grounder.

B-812 (/stress A1.2 cold-start P1-1-A* Claude OOB, 2026-05-17): pre-fix the
identical stage_prefix block was copy-pasted byte-for-byte across the three
backend wrappers (`api_proxy.py:147-158`, `local_qwen.py:112-124`,
`local_gemma.py:117-129`). Paper §3.4 planner/grounder ablation claim assumes
all three baselines see byte-identical prefix tokens for the same `stage`;
any future prose tuning that touched one wrapper would silently drift the
others, breaking cross-baseline asymmetry. A1.4 B-451 consolidated the
`_shared_vl_utils` for vision-mode wrappers under the same rationale but
stopped short of stage-prefix sibling — this module closes that gap.

Invariant lock test lives in tests/test_stage_prefix_cross_baseline.py.
"""

from __future__ import annotations

from typing import Optional


def build_stage_prefix(stage: str, planner_sub_goal: Optional[str] = None) -> str:
    """Build the canonical stage prefix string for a given pipeline stage.

    Args:
        stage: one of "single" / "planner" / "grounder" (caller is responsible
            for validating; `BackendStepContext.__post_init__` enforces).
        planner_sub_goal: sub-goal text from the prior planner pass; only
            consumed when ``stage == "grounder"``.

    Returns:
        Prefix string to prepend to the per-step instruction. Empty string
        for the default single-stage path so concatenation is a no-op.
    """
    if stage == "planner":
        return (
            "[Stage: planner] Based on the task and interaction history, "
            "identify the immediate sub-goal for this step. Output ONLY a "
            "short sub-goal description (one sentence), not an action.\n\n"
        )
    if stage == "grounder":
        sub_goal = planner_sub_goal or ""
        return (
            f"[Stage: grounder] Sub-goal: {sub_goal}\n"
            "Based on the sub-goal above and the current page state, "
            "produce a concrete action JSON.\n\n"
        )
    return ""
