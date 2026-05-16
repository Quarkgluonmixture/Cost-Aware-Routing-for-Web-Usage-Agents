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
# B-165 — fallback_finish reward override guard (Claude F2 + codex B3 dual)
# ---------------------------------------------------------------------------


def test_reward_override_requires_real_finish_not_fallback():
    """Reward override (score 0→1 when env reward>0) must require a REAL
    agent finish (parse_valid AND not fallback_finish). Pre-B-165 it only
    checked action_type==finish, so keyword-rescue fallbacks (B1/B2 frequent,
    B0 rare) silently triggered SR inflation differentially → paper-grade
    cross-baseline contamination.
    """
    src = (REPO_ROOT / "p79/experiment/runner/main.py").read_text(encoding="utf-8")
    # Must define _real_finish guard
    assert "_real_finish" in src, "B-165 missing _real_finish guard variable"
    # The guard must check both fallback_finish AND parse_valid
    assert "fallback_finish" in src and "parse_valid" in src, (
        "B-165 guard must verify both fallback_finish flag and parse_valid"
    )
    # Reward override block must reference _real_finish, not direct action_type check
    override_idx = src.find("score = 1.0")
    assert override_idx >= 0
    # Find the if-condition just before this override (preceding 30 lines)
    pre_override = src[max(0, override_idx - 1500):override_idx]
    assert "_real_finish" in pre_override, (
        "B-165 reward override must use _real_finish guard"
    )
    # Confirm the old loose check pattern is gone from the guard block
    assert 'step_records[-1].get("action_type", "") in ("finish", "stop")' not in pre_override, (
        "B-165 regression: old action_type-only check still gates reward override"
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
