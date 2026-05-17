"""Invariant tests for /stress A1.4a v8 Commit G1 quick-wins (B-164 ~ B-166).

Each test pins one specific contract that, once it regresses, would break
paper §3 evidence layer (cross-baseline SR fairness) or multi-seed
reproducibility.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# B-164 — backend cfg deep copy (multi-seed isolation, /stress codex B5)
# ---------------------------------------------------------------------------


def test_backend_cfg_uses_deepcopy_not_shallow_dict():
    """``_get_backend`` must ``copy.deepcopy`` backend_cfg to prevent nested
    mutation leakage across (condition, seed) iterations through self.cfg.
    Pre-B-164 ``dict(backend_cfg)`` shared nested ``generation`` /
    ``model_kwargs`` / ``headers`` references; constructor side effects
    poisoned subsequent seed runs even with cache key tuple (B-144) correct.
    """
    src = (REPO_ROOT / "p79/experiment/runner/main.py").read_text(encoding="utf-8")
    assert "import copy" in src, "B-164 missing `import copy` at module top"
    # Inside _get_backend the shallow dict() must be retired
    get_backend_start = src.find("def _get_backend(self")
    assert get_backend_start >= 0, "_get_backend not found"
    get_backend_block = src[get_backend_start:get_backend_start + 3000]
    code_lines = "\n".join(
        line for line in get_backend_block.splitlines()
        if not line.strip().startswith("#")
    )
    assert "backend_cfg = copy.deepcopy(backend_cfg)" in code_lines, (
        "B-164 _get_backend must use copy.deepcopy(backend_cfg)"
    )
    assert "backend_cfg = dict(backend_cfg)" not in code_lines, (
        "B-164 regression: shallow dict(backend_cfg) still in code"
    )


# ---------------------------------------------------------------------------
# B-165 / B-545 — reward override mechanism RETIRED entirely
# ---------------------------------------------------------------------------
# Lineage:
#   B-165 (A1.4a, commit `a1b04e8`, 2026-05-16): narrowed override conditions
#     to require parse_valid AND not fallback_finish (`_real_finish` guard)
#     to close cross-baseline B0-vs-B1/B2 fallback_finish differential.
#   B-545 (A1.5b Phase 2, commit `7832008`, 2026-05-17): eliminated the
#     override mechanism entirely. `success = bool(score >= 1.0)` from VWA
#     evaluator output, no post-hoc adjustment.
#
# This test now pins the **B-545 retirement** invariant rather than the
# B-165 narrowing invariant (the latter is implied by the former). Pre-fix
# /stress A1.5 P0-3-A (Claude F2, 2026-05-17): test still asserted
# `_real_finish` GUARD VARIABLE presence after B-545 retired the entire
# block, so pytest went RED. B-550 inverts the assertions.


def test_reward_override_mechanism_retired_post_B545():
    """B-545 (A1.5b Phase 2, 2026-05-17) retired the reward-override mechanism
    entirely. `success` derives strictly from `score >= 1.0` where `score` is
    the VWA evaluator output. Pre-B-545 the runner secretly overrode `score=0`
    → `score=1` when agent self-reported finish + env_reward>0, contradicting
    paper §3 estimand claim "canonical evaluator success, no post-hoc
    adjustment" (top-tier reviewer estimand-schizophrenia attack vector).

    This test pins the post-B-545 invariant: the override block must NOT
    exist in `runner/main.py`. Companion B-165 narrowing (real-finish guard)
    is subsumed because the entire mechanism is gone.
    """
    src = (REPO_ROOT / "p79/experiment/runner/main.py").read_text(encoding="utf-8")
    # B-545: the override mechanism is gone. None of its identifying tokens
    # should appear in active (non-comment) code.
    code_only = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#")
    )
    # The _real_finish guard variable (B-165 era) must be retired.
    assert "_real_finish" not in code_only, (
        "B-545 regression: `_real_finish` guard variable resurrected — "
        "override mechanism is supposed to be retired entirely, not gated."
    )
    # The score-override assignment must not appear in active code.
    assert "score = 1.0" not in code_only, (
        "B-545 regression: `score = 1.0` override assignment present in "
        "active code — override mechanism must be removed, not narrowed."
    )
    # The logger.warning string used by the old override branch must be gone.
    assert "Reward override" not in code_only, (
        "B-545 regression: `Reward override` log string still in active "
        "code — override mechanism residue."
    )
    # Positive contract: success derives from `score >= 1.0` (the canonical
    # line at `_run_episode` close).
    assert "success = bool(score >= 1.0)" in code_only, (
        "B-545 contract: `success = bool(score >= 1.0)` line must be "
        "the canonical success derivation post-override-retirement."
    )


# ---------------------------------------------------------------------------
# B-166 — trajectory_incomplete telemetry (Claude F4, paper §3.5 disclosure)
# ---------------------------------------------------------------------------


def test_trajectory_incomplete_field_set_when_no_explicit_finish():
    """When the trajectory does not end with a finish action, the runner
    appends a fake stop action with empty answer. This must be recorded as
    ``trajectory_incomplete=True`` in the episode summary so paper §3.5 can
    report timeout rate as a transparency metric. SR remains canonical.
    """
    src = (REPO_ROOT / "p79/experiment/runner/main.py").read_text(encoding="utf-8")
    # Flag must be defined before the fake-stop block
    assert "trajectory_incomplete = False" in src, (
        "B-166 missing trajectory_incomplete initialization to False"
    )
    # Inside the "no answer at trajectory tail" branch, flag must be set True
    assert "trajectory_incomplete = True" in src, (
        "B-166 missing trajectory_incomplete=True assignment in fake-stop branch"
    )
    # Must propagate to episode_summary
    assert 'episode_summary["trajectory_incomplete"] = trajectory_incomplete' in src, (
        "B-166 episode_summary must stamp trajectory_incomplete telemetry"
    )


def test_trajectory_incomplete_complementary_to_agent_finished():
    """`trajectory_incomplete` and `agent_finished` are complementary
    telemetry: incomplete=True implies agent never issued real finish
    (because fake-stop fires only when last trajectory item lacks 'answer').
    Test that both fields are recorded so analysis can cross-check rather
    than infer one from absence of the other.
    """
    src = (REPO_ROOT / "p79/experiment/runner/main.py").read_text(encoding="utf-8")
    assert 'episode_summary["agent_finished"]' in src
    assert 'episode_summary["trajectory_incomplete"]' in src


# ---------------------------------------------------------------------------
# Cross-AI memory infra (AGENTS.md + GEMINI.md pointer files)
# ---------------------------------------------------------------------------


def test_agents_md_pointer_to_claude_md():
    """AGENTS.md must exist and point at .claude/CLAUDE.md as canonical
    project context. Confirms cross-AI memory infra is wired so any codex /
    other AI CLI invocation outside /codex-stress workflow still picks up
    P79 project facts.
    """
    agents_md = REPO_ROOT / "AGENTS.md"
    assert agents_md.exists(), "AGENTS.md not in repo root"
    content = agents_md.read_text(encoding="utf-8")
    assert ".claude/CLAUDE.md" in content, (
        "AGENTS.md must reference .claude/CLAUDE.md as single source of truth"
    )


def test_gemini_md_pointer_to_claude_md():
    """GEMINI.md (parallel to AGENTS.md, gemini CLI convention) must exist
    and point at .claude/CLAUDE.md. Same rationale as AGENTS.md.
    """
    gemini_md = REPO_ROOT / "GEMINI.md"
    assert gemini_md.exists(), "GEMINI.md not in repo root"
    content = gemini_md.read_text(encoding="utf-8")
    assert ".claude/CLAUDE.md" in content, (
        "GEMINI.md must reference .claude/CLAUDE.md as single source of truth"
    )
