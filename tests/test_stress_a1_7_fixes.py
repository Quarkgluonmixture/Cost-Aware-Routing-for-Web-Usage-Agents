"""Invariant tests for /stress A1.7 cold-start fixes (B-691~B-701).

Audit context: 2026-05-17 cold-start A1.7 of `p79/experiment/conditions.py`
+ `configs/exp_v2_*.yaml` after B-261~B-269 (2026-05-16) initial landing.
The cold-start cycle found 13 unique residual gaps (8 Claude unique + 3
Gemini unique + 2 overlap; codex Mode B FAIL 2-retry budget); user
confirmed fix scope Q1=B (no aliasing shim for P1-4 — Pass-2 not yet
fired) + Q2=A (LR try-except + log/ntfy/fallback_count) + Q3=A (phase2
routed-mode sentinel deferred to paper-2). 11 P0+P1+P2 fixes landed code-
side: B-691 phase2/3 dead-mode helper extract / B-692 learned-baseline
contamination guard / B-693 LR dispatch try-except / B-694 router id
collision (backend+site qualified) / B-695 _load_best partial-cell
filter / B-696 mode_set metadata reads candidate_modes / B-697
RuleBasedRouter unlisted-mode raise / B-698 default_backend silent
fallback raise / B-699 phase2.yaml dead som_on field removed / B-700
base.yaml max_steps 40→30 / B-701 module ValueError lists valid names.
"""

from __future__ import annotations

import pytest
import yaml
from pathlib import Path

from p79.experiment.conditions import (
    _default_backend_id,
    _module_flags_from_name,
    _validate_obs_mode,
    generate_conditions,
)
from p79.experiment.types import ConditionSpec, ModuleFlags


REPO_ROOT = Path(__file__).resolve().parent.parent


# ───────────────────────────────────────────────────────────────────────
# B-691 P0-1: phase2/phase3 yaml retired-mode validation
# ───────────────────────────────────────────────────────────────────────

def test_b680_validate_obs_mode_rejects_deprecated_hybrid():
    """`hybrid` was retired in B-263 — must raise across all phases."""
    with pytest.raises(ValueError, match="deprecated/retired"):
        _validate_obs_mode("hybrid", context="phase2.fixed")


def test_b680_validate_obs_mode_rejects_deprecated_dom_only():
    """`dom_only` was retired in B-263 — must raise."""
    with pytest.raises(ValueError, match="deprecated/retired"):
        _validate_obs_mode("dom_only", context="phase3.base")


def test_b680_validate_obs_mode_rejects_legacy_alias_phantom_dom():
    """`phantom_dom` legacy alias (B-261) — must raise with replacement hint."""
    with pytest.raises(ValueError, match="phantom_text"):
        _validate_obs_mode("phantom_dom", context="phase1.observation_mode")


def test_b680_validate_obs_mode_passes_canonical_six_modes():
    """All 6 paper-1 canonical modes + learned sentinel must pass."""
    for mode in ["dom", "som", "vision", "phantom_som", "phantom_text",
                 "phantom_prompt", "learned"]:
        assert _validate_obs_mode(mode) == mode


def test_b680_phase2_yaml_post_fix_obs_mode():
    """Verify the post-fix phase2.yaml no longer references retired modes."""
    cfg_path = REPO_ROOT / "configs" / "exp_v2_phase2.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    obs_mode = cfg["variables"]["phase2"]["fixed_condition"]["observation_mode"]
    assert obs_mode not in ("hybrid", "dom_only", "phantom_dom")
    assert obs_mode == "som"  # paper-1 hero arm


def test_b680_phase3_yaml_post_fix_obs_mode():
    """Verify the post-fix phase3.yaml no longer references retired modes."""
    cfg_path = REPO_ROOT / "configs" / "exp_v2_phase3.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    obs_mode = cfg["variables"]["phase3"]["base_condition"]["observation_mode"]
    assert obs_mode not in ("hybrid", "dom_only", "phantom_dom")
    assert obs_mode == "dom"  # paper §3 default ablation baseline


def test_b680_generate_conditions_phase2_rejects_injected_hybrid():
    """If a future yaml regression re-injects hybrid, generate_conditions raises."""
    cfg = {
        "experiment": {"phase": "phase2"},
        "variables": {
            "primary": {},
            "phase2": {
                "fixed_condition": {"observation_mode": "hybrid"},
                "run_fixed_best": True,
                "run_routed": False,
            },
        },
        "backends": {"default_backend": "api_strong", "api_strong": {}},
    }
    with pytest.raises(ValueError, match="deprecated/retired"):
        generate_conditions(cfg)


def test_b680_generate_conditions_phase3_rejects_injected_dom_only():
    """If a future yaml regression re-injects dom_only, phase3 branch raises."""
    cfg = {
        "experiment": {"phase": "phase3"},
        "variables": {
            "phase3": {
                "base_condition": {"observation_mode": "dom_only", "router_on": True}
            },
        },
        "backends": {"default_backend": "api_strong", "api_strong": {}},
    }
    with pytest.raises(ValueError, match="deprecated/retired"):
        generate_conditions(cfg)


# ───────────────────────────────────────────────────────────────────────
# B-692 P0-2: learned-baseline contamination guard
# ───────────────────────────────────────────────────────────────────────

def test_b681_learned_mode_in_baseline_emit_raises():
    """obs_mode='learned' + variant='both' (or default baseline) must raise."""
    cfg = {
        "experiment": {"phase": "phase1"},
        "task": {"include_sites": ["classifieds"]},
        "variables": {
            "primary": {"observation_mode": ["learned"]},
            "phase1": {"variant": "both"},
        },
        "backends": {"default_backend": "local_4b", "local_4b": {}},
    }
    with pytest.raises(ValueError, match="router-only sentinel"):
        generate_conditions(cfg)


def test_b681_learned_mode_with_router_only_variant_passes():
    """variant='router' is the canonical path for learned mode — must NOT raise."""
    cfg = {
        "experiment": {"phase": "phase1"},
        "task": {"include_sites": ["classifieds"]},
        "variables": {
            "primary": {"observation_mode": ["learned"]},
            "phase1": {"variant": "router", "router_kind": "learned"},
        },
        "backends": {"default_backend": "local_4b", "local_4b": {}},
    }
    conditions = generate_conditions(cfg)
    assert len(conditions) == 1
    assert conditions[0].observation_mode == "learned"
    assert conditions[0].router_on is True


# ───────────────────────────────────────────────────────────────────────
# B-693 P0-3: LR try-except wrapper — source-level invariant check
# ───────────────────────────────────────────────────────────────────────

def test_b682_runner_main_has_try_except_around_lr_dispatch():
    """LR dispatch block in runner/main.py must be wrapped in try/except."""
    src = (REPO_ROOT / "p79" / "experiment" / "runner" / "main.py").read_text()
    # Locate the LR dispatch comment block
    assert 'if condition.observation_mode == "learned":' in src
    # The fix wraps the imports + lazy-load + predict_mode inside try/except
    # We look for the canonical pattern landed in B-693.
    assert "B-693" in src
    assert "_lr_fallback_count" in src
    assert "safe_fallback_target" in src
    # Confirm try/except actually wraps load_lr_pipeline (the catastrophic path).
    learned_idx = src.index('if condition.observation_mode == "learned":')
    try_idx = src.index("try:", learned_idx)
    except_idx = src.index("except Exception as exc:", try_idx)
    load_idx = src.index("load_lr_pipeline", learned_idx)
    assert try_idx < load_idx < except_idx, (
        "load_lr_pipeline must be inside the try block, not outside"
    )


# ───────────────────────────────────────────────────────────────────────
# B-694 P1-4: learned router condition_id includes backend + site
# ───────────────────────────────────────────────────────────────────────

def test_b683_learned_router_condition_id_includes_backend_and_site():
    """phase1_learned_router id must include backend_id + site_hint (no shim)."""
    cfg = {
        "experiment": {"phase": "phase1"},
        "task": {"include_sites": ["classifieds"]},
        "variables": {
            "primary": {"observation_mode": ["learned"]},
            "phase1": {"variant": "router", "router_kind": "learned"},
        },
        "backends": {"default_backend": "api_strong", "api_strong": {}},
    }
    conditions = generate_conditions(cfg)
    assert len(conditions) == 1
    cid = conditions[0].condition_id
    assert "api_strong" in cid
    assert "classifieds" in cid
    assert cid == "phase1_learned_router_api_strong_classifieds"


def test_b683_learned_router_multi_site_raises():
    """Multi-site include_sites is not supported for learned router."""
    cfg = {
        "experiment": {"phase": "phase1"},
        "task": {"include_sites": ["classifieds", "reddit"]},
        "variables": {
            "primary": {"observation_mode": ["learned"]},
            "phase1": {"variant": "router", "router_kind": "learned"},
        },
        "backends": {"default_backend": "api_strong", "api_strong": {}},
    }
    with pytest.raises(ValueError, match="exactly one site"):
        generate_conditions(cfg)


# ───────────────────────────────────────────────────────────────────────
# B-695 P1-5: _load_best filters _synthesized partial-cell rows
# ───────────────────────────────────────────────────────────────────────

def test_b684_load_best_skips_synthesized_partial_rows(tmp_path):
    """_load_best should filter out _synthesized=True rows before ranking."""
    import json
    from p79.experiment.conditions import _load_best_condition_from_phase1

    run_dir = tmp_path / "fake_phase1_run"
    run_dir.mkdir()
    summary_payload = {
        "condition_metrics": [
            # Partial cell with the highest SR — must be skipped
            {
                "condition_id": "phase1_som_router_0",
                "observation_mode": "som",
                "success_rate": 0.95,
                "avg_total_cost_usd": 0.10,
                "p95_step_latency_ms": 500.0,
                "_synthesized": True,  # partial — should be filtered
            },
            # Real cell with lower SR — should win after filter
            {
                "condition_id": "phase1_dom_router_0",
                "observation_mode": "dom",
                "success_rate": 0.50,
                "avg_total_cost_usd": 0.05,
                "p95_step_latency_ms": 300.0,
            },
        ]
    }
    (run_dir / "run_summary_v2.json").write_text(json.dumps(summary_payload))

    result = _load_best_condition_from_phase1(run_dir)
    assert result is not None
    _som_on, obs_mode, source_id = result
    assert obs_mode == "dom"  # NOT "som" despite "som" having higher SR
    assert source_id == "phase1_dom_router_0"


def test_b684_load_best_returns_none_when_all_synthesized(tmp_path):
    """If every condition is partial-data, _load_best returns None."""
    import json
    from p79.experiment.conditions import _load_best_condition_from_phase1

    run_dir = tmp_path / "fake_phase1_run"
    run_dir.mkdir()
    summary_payload = {
        "condition_metrics": [
            {
                "condition_id": "phase1_dom_router_0",
                "observation_mode": "dom",
                "success_rate": 0.5,
                "avg_total_cost_usd": 0.05,
                "p95_step_latency_ms": 300.0,
                "_synthesized": True,
            },
            {
                "condition_id": "phase1_som_router_0",
                "observation_mode": "som",
                "success_rate": 0.6,
                "avg_total_cost_usd": 0.08,
                "p95_step_latency_ms": 400.0,
                "_synthesized": True,
            },
        ]
    }
    (run_dir / "run_summary_v2.json").write_text(json.dumps(summary_payload))

    result = _load_best_condition_from_phase1(run_dir)
    assert result is None


# ───────────────────────────────────────────────────────────────────────
# B-696 P1-6: mode_set metadata reads phase1.candidate_modes
# ───────────────────────────────────────────────────────────────────────

def test_b685_mode_set_reads_candidate_modes_from_yaml():
    """Learned router condition metadata 'mode_set' must reflect candidate_modes."""
    candidates = ["dom", "som", "phantom_som", "phantom_text"]
    cfg = {
        "experiment": {"phase": "phase1"},
        "task": {"include_sites": ["classifieds"]},
        "variables": {
            "primary": {"observation_mode": ["learned"]},
            "phase1": {
                "variant": "router",
                "router_kind": "learned",
                "candidate_modes": candidates,
            },
        },
        "backends": {"default_backend": "api_strong", "api_strong": {}},
    }
    conditions = generate_conditions(cfg)
    assert conditions[0].metadata["mode_set"] == candidates


# ───────────────────────────────────────────────────────────────────────
# B-697 P1-8: RuleBasedRouter raises on unlisted current_mode
# ───────────────────────────────────────────────────────────────────────

def test_b686_rule_based_router_unlisted_mode_raises():
    """If current_mode not in router.modes during escalation, must raise."""
    from p79.experiment.router import RuleBasedRouter, RouterState
    cfg = {
        "router": {
            "cheap_default_mode": "dom",
            "rich_escalation_mode": "som",
            "modes": ["dom", "som"],  # default 2-mode list
            "thresholds": {
                "dom_size_threshold": 100,  # low threshold to easy-trigger
                "unchanged_steps_trigger": 1,
                "no_progress_steps_trigger": 1,
                "retry_limit": 1,
            },
        }
    }
    router = RuleBasedRouter(cfg)
    state = RouterState(current_mode="vision")  # NOT in modes list
    # Trigger escalation path: large obs_text exceeds dom_size_threshold=100
    big_obs = "x" * 5000
    with pytest.raises(ValueError, match="does not"):
        router.decide(
            router_enabled=True,
            preferred_mode="vision",
            obs_text=big_obs,
            state=state,
            prev_action_success=False,
            prev_page_changed=False,
        )


# ───────────────────────────────────────────────────────────────────────
# B-698 P2-9: _default_backend_id raises on empty backends
# ───────────────────────────────────────────────────────────────────────

def test_b687_default_backend_id_raises_on_empty_backends():
    """Empty backends dict must raise (no more silent local_4b fallback)."""
    cfg = {"backends": {}}
    with pytest.raises(ValueError, match="no `default_backend`"):
        _default_backend_id(cfg)


def test_b687_default_backend_id_picks_explicit_default():
    """Explicit default_backend takes priority."""
    cfg = {"backends": {"default_backend": "api_strong", "api_strong": {}}}
    assert _default_backend_id(cfg) == "api_strong"


def test_b687_default_backend_id_picks_first_concrete_when_no_default():
    """When no default_backend, pick first concrete entry (not the magic string)."""
    cfg = {"backends": {"local_gemma": {}, "api_strong": {}}}
    backend = _default_backend_id(cfg)
    assert backend in ("local_gemma", "api_strong")
    assert backend != "default_backend"


# ───────────────────────────────────────────────────────────────────────
# B-699 P2-10: phase2.yaml dead som_on field removed
# ───────────────────────────────────────────────────────────────────────

def test_b688_phase2_yaml_no_dead_som_on_field():
    """phase2.yaml fixed_condition must not contain the misleading som_on field."""
    cfg = yaml.safe_load(
        (REPO_ROOT / "configs" / "exp_v2_phase2.yaml").read_text()
    )
    fixed = cfg["variables"]["phase2"]["fixed_condition"]
    assert "som_on" not in fixed, (
        "som_on was a dead yaml field — conditions.py derives it from "
        "obs_mode and never reads the yaml value"
    )


# ───────────────────────────────────────────────────────────────────────
# B-700 P2-11: base.yaml max_steps 40 → 30
# ───────────────────────────────────────────────────────────────────────

def test_b689_base_yaml_max_steps_aligned_to_30():
    """exp_v2_base.yaml max_steps must match the per-condition yaml override."""
    cfg = yaml.safe_load(
        (REPO_ROOT / "configs" / "exp_v2_base.yaml").read_text()
    )
    assert cfg["runtime"]["max_steps"] == 30


# ───────────────────────────────────────────────────────────────────────
# B-701 P2-13: module ValueError lists valid names
# ───────────────────────────────────────────────────────────────────────

def test_b690_module_flags_error_lists_valid_names():
    """_module_flags_from_name error msg must include the valid alias list."""
    with pytest.raises(ValueError, match=r"Valid: \[.*'m1'.*'m2'.*'m3'.*'m4'.*\]"):
        _module_flags_from_name("m1_typo_xyz")
