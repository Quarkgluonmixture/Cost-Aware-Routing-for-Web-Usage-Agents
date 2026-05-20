"""Fire-6 RCA Stage C1 test — VwaEvaluator._classify_eval_context.

User-approved (2026-05-20): L2 program_html-only evaluator isolation. The
classifier decides whether a task's eval can run in an isolated fresh
browser context (page.goto to explicit target URL — agent DOM discarded
by the goto anyway) vs MUST reuse the agent's page (url_match needs live
page.url; program_html with `__last_url__`/`"last"`/`func`-url derives the
target from the agent's final navigation).

Inspection 2026-05-20 (evaluators.py:368): program_html does
`page.goto(target_url)` then reads from the server-rendered page — for
explicit-URL targets this is semantically identical to a fresh isolated
page AND avoids the runner-context cumulative-state 30s timeout (Fire-3/4/5
task 75/4 pattern).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from p79.experiment.environment import VwaEvaluator


def _write_cfg(tmp_path: Path, eval_block: dict) -> str:
    p = tmp_path / "task.json"
    p.write_text(json.dumps({"eval": eval_block}), encoding="utf-8")
    return str(p)


class TestClassifyEvalContext:
    def test_explicit_url_program_html_isolates(self, tmp_path):
        cfg = _write_cfg(tmp_path, {
            "eval_types": ["program_html"],
            "program_html": [
                {"url": "http://localhost:9980/index.php?page=item&id=84144",
                 "locator": "func:get_query_text(__page__, '.price')",
                 "required_contents": {"must_include": ["25000.00"]}},
            ],
        })
        mode, target = VwaEvaluator._classify_eval_context(cfg)
        assert mode == "isolated_program_html_context"
        assert target == "http://localhost:9980/index.php?page=item&id=84144"

    def test_last_literal_url_keeps_agent_page(self, tmp_path):
        cfg = _write_cfg(tmp_path, {
            "eval_types": ["program_html"],
            "program_html": [
                {"url": "last", "locator": "func:get_query_text(__page__, '.x')",
                 "required_contents": {"must_include": ["y"]}},
            ],
        })
        mode, _ = VwaEvaluator._classify_eval_context(cfg)
        assert mode == "agent_page"

    def test_last_url_substitution_keeps_agent_page(self, tmp_path):
        cfg = _write_cfg(tmp_path, {
            "eval_types": ["program_html"],
            "program_html": [
                {"url": "func:'http://x/' + __last_url__.split('=')[-1]",
                 "locator": "", "required_contents": {"must_include": ["z"]}},
            ],
        })
        mode, _ = VwaEvaluator._classify_eval_context(cfg)
        assert mode == "agent_page"

    def test_func_url_keeps_agent_page(self, tmp_path):
        cfg = _write_cfg(tmp_path, {
            "eval_types": ["program_html"],
            "program_html": [
                {"url": "func:something()", "locator": "",
                 "required_contents": {"must_include": ["z"]}},
            ],
        })
        mode, _ = VwaEvaluator._classify_eval_context(cfg)
        assert mode == "agent_page"

    def test_url_match_keeps_agent_page(self, tmp_path):
        cfg = _write_cfg(tmp_path, {"eval_types": ["url_match"], "reference_url": "http://x"})
        mode, _ = VwaEvaluator._classify_eval_context(cfg)
        assert mode == "agent_page"

    def test_program_html_plus_url_match_keeps_agent_page(self, tmp_path):
        cfg = _write_cfg(tmp_path, {
            "eval_types": ["program_html", "url_match"],
            "program_html": [
                {"url": "http://localhost:9980/index.php?page=item&id=1",
                 "locator": "", "required_contents": {"must_include": ["x"]}},
            ],
        })
        mode, _ = VwaEvaluator._classify_eval_context(cfg)
        assert mode == "agent_page"

    def test_string_match_no_browser(self, tmp_path):
        cfg = _write_cfg(tmp_path, {"eval_types": ["string_match"]})
        mode, _ = VwaEvaluator._classify_eval_context(cfg)
        assert mode == "no_browser_required"

    def test_missing_config_fail_safe_agent_page(self):
        # Non-existent config → conservative agent_page (never isolate when unsure)
        mode, _ = VwaEvaluator._classify_eval_context("/nonexistent/path/task.json")
        assert mode == "agent_page"

    def test_real_task_4_and_75_isolate(self):
        """Empirical anchor: cls task 4 + 75 (3-fire pattern) must isolate."""
        base = "external/visualwebarena/config_files/vwa/test_classifieds"
        for tid in (4, 75):
            cfg = f"{base}/{tid}.json"
            if not Path(cfg).exists():
                pytest.skip(f"task config {cfg} not present")
            mode, target = VwaEvaluator._classify_eval_context(cfg)
            assert mode == "isolated_program_html_context", f"task {tid} should isolate"
            assert "id=" in (target or ""), f"task {tid} target should be explicit item URL"


class TestEvalContextSchemaFields:
    """Fire-6 C1 6 eval-context metadata fields land 4-place sync."""

    def test_eval_context_fields_4place(self):
        import dataclasses
        from p79.experiment.schema_migrations.v2 import EPISODE_SUMMARY_V2_DEFAULTS
        from p79.experiment.types import (
            EpisodeSummaryV2,
            PAPER_GRADE_EPISODE_OPTIONAL_KEYS,
            _EPISODE_OPTIONAL_FIELD_TYPES,
        )
        dc_fields = {f.name for f in dataclasses.fields(EpisodeSummaryV2)}
        fields = {
            "eval_context_mode", "eval_isolated_context_used",
            "eval_goto_latency_ms", "eval_goto_timeout",
            "eval_source_agent_url", "eval_target_url",
        }
        for f in fields:
            assert f in dc_fields, f"{f} missing in EpisodeSummaryV2 dataclass"
            assert f in EPISODE_SUMMARY_V2_DEFAULTS, f"{f} missing in v2 defaults"
            assert f in _EPISODE_OPTIONAL_FIELD_TYPES, f"{f} missing in type map"
            assert f in PAPER_GRADE_EPISODE_OPTIONAL_KEYS, f"{f} missing in paper-grade keys"

    def test_episode_eval_result_carries_fields(self):
        from p79.experiment.environment import EpisodeEvalResult
        r = EpisodeEvalResult(
            score=1.0, error=None,
            eval_context_mode="isolated_program_html_context",
            eval_isolated_context_used=True,
            eval_goto_latency_ms=639.0,
            eval_goto_timeout=False,
            eval_source_agent_url="http://localhost:9980/edit",
            eval_target_url="http://localhost:9980/item?id=1",
        )
        assert r.eval_context_mode == "isolated_program_html_context"
        assert r.eval_isolated_context_used is True
        assert r.eval_goto_latency_ms == 639.0
        assert r.eval_goto_timeout is False
