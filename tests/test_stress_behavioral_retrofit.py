"""Behavior-level retrofit pairs for source-grep-only stress tests.

/stress A1.12 P1-2 (2026-05-16, T1-3 = B 渐进 3 文件): existing
`test_stress_a1_1_fixes.py` / `test_stress_a1_2_fixes.py` / `test_stress_a1_4a_g3_fixes.py`
contain ~80 source-grep assertions that pass on string presence rather
than runtime behavior. Codex Mode B P1-4 caught the failure mode: a
comment containing `DeprecationWarning` would pass without any
`warnings.warn()` ever firing.

Rather than rewrite the 3 large files in-place (risk of conflict with
parallel sessions + audit history), this file ADDS behavior-level
companion tests for the most paper-grade-critical invariants:

- B-145 GLM fallback DeprecationWarning: real `warnings.catch_warnings`
  exercise on `ProxyApiAgent.__init__` with `use_glm_fallback=True`
- B-146 Gemma agent decoupled from Qwen: import time runtime check that
  `gemma3vl_agent` module load does NOT pull `qwen3vl_agent` into
  `sys.modules` (cross-family decoupling contract)
- B-92 prompt static-method: runtime callability on the agent class
  without instance (companion to source-grep `staticmethod` check)
- B-144 backend cache (seed, backend_id) key: behavior test that two
  different seeds yield distinct backend instances on the same backend_id

These are PAIR tests — they live alongside the source-grep tests, not
replace them. Source-grep tests catch refactor-time string drift;
behavior tests catch runtime semantic regressions.
"""
from __future__ import annotations

import os
import sys
import warnings

import pytest


# ─── B-145: GLM fallback DeprecationWarning (real runtime emission) ─────────
def test_b145_proxy_agent_emits_deprecation_warning_at_runtime(monkeypatch):
    """Companion to test_stress_a1_2_fixes:test_proxy_agent_emits_deprecation_*.

    The source-grep version only asserted strings exist in source. This
    actually constructs `ProxyApiAgent` with `use_glm_fallback=True` and
    catches the warning at runtime — proves the `warnings.warn(...)` call
    on line 165 actually fires, not just that the string is present.
    """
    monkeypatch.setenv("PROXY_API_KEY", "dummy_for_warning_test")
    from p79.agents.proxy_api_agent import ProxyApiAgent

    config = {
        "model": {
            "api_name": "qwen.qwen3-vl-235b-a22b",
            "base_url": "http://localhost:9999",  # bogus, never called in __init__
            "use_glm_fallback": True,
            "glm_config": "/nonexistent/glm_config_for_test",  # avoid real file load
        },
        # paper_grade=False so B-340 hard-raise does not block (we WANT the
        # warning path to run; the hard-raise path is separately tested below).
        "paper_grade": False,
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            ProxyApiAgent(config)
        except Exception:
            # `_load_glm_config` may raise on bogus path AFTER warning fired —
            # the warning is what we're testing, not the load completion.
            pass
        relevant = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert relevant, (
            "B-145 GLM fallback DeprecationWarning did NOT fire at runtime. "
            "Source-grep test would still pass — this catches the real regression. "
            f"All warnings caught: {[(w.category.__name__, str(w.message)[:60]) for w in caught]}"
        )
        msg = str(relevant[0].message)
        assert "GLM fallback" in msg, f"Warning message drift: {msg!r}"
        assert "deprecated" in msg.lower() or "retire" in msg.lower(), (
            f"Warning missing 'deprecated'/'retire' keyword: {msg!r}"
        )


def test_b340_paper_grade_mode_hard_blocks_glm_fallback(monkeypatch):
    """B-340 paper-grade hard-raise companion to B-145.

    With `paper_grade: true` + `use_glm_fallback: true`, agent construction
    must raise RuntimeError — not silently warn-then-proceed (which would
    let GLM contaminate paper-grade cost-fairness comparison).
    """
    monkeypatch.setenv("PROXY_API_KEY", "dummy_for_paper_grade_test")
    from p79.agents.proxy_api_agent import ProxyApiAgent

    config = {
        "model": {
            "api_name": "qwen.qwen3-vl-235b-a22b",
            "base_url": "http://localhost:9999",
            "use_glm_fallback": True,
        },
        "paper_grade": True,
    }
    with pytest.raises(RuntimeError, match=r"paper-grade|B-340|cost-fairness"):
        ProxyApiAgent(config)


# ─── B-146: Gemma agent decoupled from Qwen (runtime sys.modules check) ─────
def test_b146_gemma_agent_module_does_not_pull_qwen_into_sys_modules():
    """Companion to test_stress_a1_2_fixes:test_gemma_agent_does_not_import_qwen_*.

    The source-grep version only checks the literal import line is absent.
    This runtime test imports `gemma3vl_agent` in a clean sub-state and
    verifies `qwen3vl_agent` is NOT pulled in transitively.
    """
    # Drop any cached Qwen/Gemma agent module so we measure fresh load.
    for mod in ("p79.agents.qwen3vl_agent", "p79.agents.gemma3vl_agent"):
        sys.modules.pop(mod, None)

    import p79.agents.gemma3vl_agent  # noqa: F401 — measure module load side effects

    assert "p79.agents.qwen3vl_agent" not in sys.modules, (
        "B-146 decoupling regression: importing gemma3vl_agent now pulls "
        "qwen3vl_agent into sys.modules — this re-introduces transitive deps "
        "(qwen_vl_utils, Qwen3VLForConditionalGeneration) that cross-family "
        "should not need."
    )


# ─── B-92: @staticmethod prompt builders (runtime callability) ──────────────
def test_b92_qwen_prompt_static_methods_callable_without_instance():
    """Companion to test_agents_prompt_parity:test_qwen_prompt_methods_are_staticmethod.

    The parity file uses `isinstance(descriptor, staticmethod)` (descriptor-
    level inspection). This pair test calls the method bare on the class —
    proves runtime invocation works, not just that the decorator is in
    place. Pre-fix `_make_dom_prompt(self)` would TypeError here.
    """
    from p79.agents.qwen3vl_agent import Qwen3VLAgent

    # The signature is `_make_dom_prompt()` (no args after self-removal).
    # Calling on the class without instance must work.
    dom = Qwen3VLAgent._make_dom_prompt()
    som = Qwen3VLAgent._make_som_prompt()
    vision = Qwen3VLAgent._make_vision_prompt()
    # Non-trivial output (B0 prompt parity asserts > 100 chars; this is the
    # behavior proof, not the parity check).
    for name, out in (("dom", dom), ("som", som), ("vision", vision)):
        assert isinstance(out, str), f"{name} prompt non-str: {type(out)}"
        assert len(out) > 100, f"{name} prompt suspiciously short: {len(out)} chars"


# ─── B-144: backend cache key (real runner build, not source pattern) ───────
def test_b144_runner_backend_cache_keys_by_seed_at_runtime(tmp_path):
    """Companion to test_stress_a1_2_fixes:test_runner_backend_cache_keyed_by_*.

    The source-grep version regex-matches the cache-key declaration line.
    This runtime test exercises `ExperimentRunner._get_backend` twice with
    distinct seeds + same backend_id and verifies different backend objects
    come back (not the same cached instance with stale seed).
    """
    import json
    from p79.experiment.runner import ExperimentRunner

    # Minimal site_configs so load_tasks() does not raise.
    site_cfg = tmp_path / "classifieds.json"
    site_cfg.write_text(json.dumps([{
        "task_id": 0, "intent": "x", "sites": ["classifieds"],
        "start_url": "__CLASSIFIEDS__/",
    }]))

    cfg = {
        "experiment": {
            "name": "b144_cache_test",
            "benchmark": "visualwebarena",
            "phase": "phase1",
            "seed": 42,
            "output_root": str(tmp_path),
            "run_id": "b144_run",
        },
        "task": {
            "include_sites": ["classifieds"],
            "max_tasks_per_site": 1,
            "task_ids": {},
            "site_configs": {"classifieds": str(site_cfg)},
        },
        "env": {"type": "mock"},
        "runtime": {"max_steps": 1, "resume": False},
        "variables": {"primary": {"observation_mode": ["dom"]}},
        "router": {
            "cheap_default_mode": "dom",
            "rich_escalation_mode": "som",
            "thresholds": {
                "dom_size_threshold": 12000,
                "unchanged_steps_trigger": 2,
                "no_progress_steps_trigger": 2,
                "retry_limit": 1,
            },
            "overhead_cost_per_ms": 0.0,
        },
        "metrics": {
            "cost": {"input_cost_per_1k": 0.0, "output_cost_per_1k": 0.0},
            "energy": {"enabled": False, "kwh_per_step": None, "co2e_kg_per_kwh": None},
        },
        "checklist": {"enabled": False},
        "state_change": {"similarity_threshold": 0.95},
        "backends": {
            "default_backend": "local_4b",
            # B-425 (/stress A1.3 v9 D1, 2026-05-17): heuristic family retired
            "local_4b": {"type": "local_qwen", "mock_mode": True, "dom_mode": "llm"},
        },
        "baselines": {"run_b0": False},
    }
    runner = ExperimentRunner(cfg)
    # Seed the cache via _get_backend at two different self.seed values.
    runner.seed = 42
    b_seed42 = runner._get_backend("local_4b")
    runner.seed = 7
    b_seed7 = runner._get_backend("local_4b")

    assert b_seed42 is not b_seed7, (
        "B-144 cache key regression: backend with same backend_id but different "
        "seeds returned identical cached instance. Pre-fix bug: cache keyed on "
        "backend_id only → first seed frozen into all subsequent backend reuse."
    )
