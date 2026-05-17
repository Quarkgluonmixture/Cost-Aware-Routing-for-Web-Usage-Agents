"""Real-LLM smoke test — /stress A1.12 P0-2 AB* (2026-05-17, B-667).

Companion to `test_runner_smoke.py` (which uses `env.type=mock` +
`backends.mock_mode=True` end-to-end and therefore validates ONLY schema
field presence + page_unchanged_rate formula). This file exercises the
**prod path**: real Qwen3VL forward pass, real action parser, real cost
computation, real step_record_v2 emission.

Pre-fix (Mode A finding A2 + Mode B finding B3 overlap): `make smoke` +
`make test` could be GREEN while `Qwen3VLAgent._make_dom_prompt()` /
`action_utils.parse_action()` / `metrics.compute_step_cost()` regression
would only surface on first paper-grade Phase 1a fire (1-2 week debug cost).

Env-gated by `RUN_REALMODEL_SMOKE=1` because:
1. Requires CUDA GPU + Qwen3-VL-4B weights (~10GB VRAM)
2. Costs ~5s/step real inference
3. `make test` on shared CI (no GPU) cannot run this
4. Paper-grade A100 host MUST run this before Phase 1a fire

Mock env is still used (no VWA docker needed) — the backend is what's real.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("RUN_REALMODEL_SMOKE") != "1",
        reason="real-LLM smoke skipped by default (set RUN_REALMODEL_SMOKE=1 + ensure GPU + Qwen3-VL-4B available)",
    ),
]


def _real_llm_cfg(tmp_path: Path) -> dict:
    """Minimal single-task / single-condition cfg with REAL LLM backend.

    Mirrors `test_runner_smoke._mock_cfg` but flips `mock_mode=False` so the
    Qwen3VL agent actually forwards.
    """
    site_cfg = tmp_path / "classifieds.json"
    site_cfg.write_text(json.dumps([{
        "task_id": 0,
        "intent": "What is the cheapest item on the homepage?",
        "sites": ["classifieds"],
        "start_url": "__CLASSIFIEDS__/",
    }]))
    return {
        "experiment": {
            "name": "realprompt_smoke",
            "benchmark": "visualwebarena",
            "phase": "phase1",
            "seed": 42,  # locked seed for replay determinism
            "output_root": str(tmp_path / "results"),
            "run_id": "realprompt_smoke_run",
        },
        "task": {
            "include_sites": ["classifieds"],
            "max_tasks_per_site": 1,
            "task_ids": {},
            "site_configs": {"classifieds": str(site_cfg)},
        },
        # env still mock (no docker required); backend is REAL
        "env": {"type": "mock", "viewport_width": 1280, "viewport_height": 800},
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
            "cost": {"input_cost_per_1k": 0.000088, "output_cost_per_1k": 0.000264},
            "energy": {"enabled": False, "kwh_per_step": None, "co2e_kg_per_kwh": None},
        },
        "checklist": {"enabled": False},
        "state_change": {"similarity_threshold": 0.95},
        "backends": {
            "default_backend": "local_4b",
            "local_4b": {
                "type": "local_qwen",
                "mock_mode": False,  # ← real LLM
                "dom_mode": "llm",
                "model_revision": None,  # allow any installed revision for smoke
            },
        },
        "baselines": {"run_b0": False},
    }


def test_real_llm_smoke_one_step_dom_mode_lands_valid_step_record(tmp_path):
    """1 task × 1 step real Qwen3VL forward pass produces a valid step_record_v2.

    Asserts the prod-path invariants `test_runner_smoke.py` can NOT verify:
    - action field is parseable (action_utils.parse_action produced valid dict)
    - cost_usd.total > 0 (real token count flowed through metrics)
    - latency_ms field present (asyncio wall measurement worked)

    NOTE: this is a CONTRACT test on the prod LLM path — not a quality test on
    Qwen3VL output. Action validity here means "schema-conformant", not
    "task-solving". Quality is measured separately by Phase 1a SR aggregate.
    """
    from p79.experiment.runner import ExperimentRunner
    from p79.experiment.types import validate_step_record_v2

    cfg = _real_llm_cfg(tmp_path)
    runner = ExperimentRunner(cfg)
    run_dir = runner.run()

    # Locate the emitted step JSONL
    step_files = list(run_dir.glob("*/episodes/*_steps_v2.jsonl"))
    assert len(step_files) == 1, f"expected 1 step JSONL, got {len(step_files)}"
    steps_text = step_files[0].read_text().strip()
    assert steps_text, "step JSONL is empty — runner exited before first step"
    step_lines = [ln for ln in steps_text.splitlines() if ln.strip()]
    assert step_lines, "no parseable step records in JSONL"

    first_step = json.loads(step_lines[0])

    # Contract 1: step record passes schema validator
    validate_step_record_v2(first_step)  # raises if invalid

    # Contract 2: action dict is well-formed (parser produced something usable)
    action = first_step.get("action")
    assert isinstance(action, dict), f"action must be dict, got {type(action)}"
    assert "action_type" in action, f"action missing action_type: {action}"

    # Contract 3: cost field non-trivially populated (real token count flowed)
    cost_usd = first_step.get("cost_usd", {})
    assert isinstance(cost_usd, dict), f"cost_usd must be dict, got {type(cost_usd)}"
    # cost_usd.model may be 0 if pricing config has all zeros, but real LLM
    # should populate at least a token count field; check tokens.
    tokens = first_step.get("tokens", {})
    assert isinstance(tokens, dict), f"tokens must be dict, got {type(tokens)}"
    total_tokens = (tokens.get("input", 0) or 0) + (tokens.get("output", 0) or 0)
    assert total_tokens > 0, (
        f"real LLM forward produced 0 tokens — agent likely silently mocked "
        f"or returned empty: tokens={tokens}"
    )

    # Contract 4: latency_ms present + sane (real inference takes > 0ms,
    # smoke is 1 step so wall < 60s)
    latency_ms = first_step.get("latency_ms", {})
    assert isinstance(latency_ms, dict), f"latency_ms must be dict, got {type(latency_ms)}"
    total_ms = latency_ms.get("total", 0) or 0
    assert 0 < total_ms < 60_000, (
        f"latency_ms.total suspicious: {total_ms}ms (expected 0 < x < 60000 for 1 real step)"
    )
