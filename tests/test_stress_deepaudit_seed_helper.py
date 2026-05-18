"""Regression tests for /stress 深入审 Mode A — Chunk 3 B-1602 P1-3-A*.

Asserts `_compute_effective_paper_grade(cfg)` helper unifies yaml ∨ env
paper_grade sources + `_seed_global_rng(seed, paper_grade_effective=...)`
honors the unified bool. Pre-fix: `_seed_global_rng` read env-only at
`runner/main.py:125`, while sibling consumer `_compute_resume_fingerprint`
(L741) already did yaml ∨ env unification per B-868. Asymmetric defense:
yaml-only paper_grade users got LAX `torch.use_deterministic_algorithms
(warn_only=True)` (paper §3.5 "byte-identical hidden states" silently
broken) while evaluator + diagnostic_controls + backend_cfg propagation
all went STRICT. See `docs/checkpoints/master_bug_catalog.md ##
/stress 深入审` (B-1602) + chronicle §220.
"""

from __future__ import annotations

import os

import pytest

from p79.experiment.runner.main import _compute_effective_paper_grade


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Each test starts with P79_PAPER_GRADE env unset to isolate states."""
    monkeypatch.delenv("P79_PAPER_GRADE", raising=False)


def test_helper_returns_false_when_no_config_no_env():
    """cfg=None + env unset → False (dev/dirty mode)."""
    assert _compute_effective_paper_grade(cfg=None) is False
    assert _compute_effective_paper_grade(cfg={}) is False


def test_helper_returns_true_when_yaml_only(monkeypatch):
    """cfg paper_grade=True + env unset → True (B-1602 fix path: pre-fix this
    case got LAX warn_only=True in _seed_global_rng despite STRICT elsewhere)."""
    monkeypatch.delenv("P79_PAPER_GRADE", raising=False)
    assert _compute_effective_paper_grade(cfg={"paper_grade": True}) is True


def test_helper_returns_true_when_env_only(monkeypatch):
    """env P79_PAPER_GRADE=1 + cfg paper_grade absent → True (default queue
    script pathway: queue_phase1_paper_grade.sh:87 exports env)."""
    monkeypatch.setenv("P79_PAPER_GRADE", "1")
    assert _compute_effective_paper_grade(cfg={}) is True
    assert _compute_effective_paper_grade(cfg=None) is True


def test_helper_returns_true_when_both_set(monkeypatch):
    """Both yaml + env True → True (no conflict)."""
    monkeypatch.setenv("P79_PAPER_GRADE", "1")
    assert _compute_effective_paper_grade(cfg={"paper_grade": True}) is True


def test_helper_env_truthy_string_variants(monkeypatch):
    """env accepts 1 / true / yes / on (case-insensitive) per config.py:188 contract."""
    for raw in ("1", "true", "TRUE", "True", "yes", "YES", "on", "ON"):
        monkeypatch.setenv("P79_PAPER_GRADE", raw)
        assert _compute_effective_paper_grade(cfg={}) is True, (
            f"env={raw!r} should be truthy"
        )


def test_helper_env_falsy_string_variants(monkeypatch):
    """env empty / 0 / off / random-string → False."""
    for raw in ("", "0", "false", "no", "off", "random_garbage"):
        monkeypatch.setenv("P79_PAPER_GRADE", raw)
        assert _compute_effective_paper_grade(cfg={}) is False, (
            f"env={raw!r} should be falsy"
        )


def test_helper_yaml_truthy_variants(monkeypatch):
    """cfg paper_grade accepts python truthy values (bool True / 1)."""
    monkeypatch.delenv("P79_PAPER_GRADE", raising=False)
    assert _compute_effective_paper_grade(cfg={"paper_grade": True}) is True
    assert _compute_effective_paper_grade(cfg={"paper_grade": 1}) is True
    # Falsy
    assert _compute_effective_paper_grade(cfg={"paper_grade": False}) is False
    assert _compute_effective_paper_grade(cfg={"paper_grade": 0}) is False
    assert _compute_effective_paper_grade(cfg={"paper_grade": None}) is False


def test_seed_global_rng_accepts_paper_grade_effective_kwarg():
    """Signature accepts new kwarg without breaking legacy single-arg callers."""
    from p79.experiment.runner.main import _seed_global_rng
    import inspect

    sig = inspect.signature(_seed_global_rng)
    params = list(sig.parameters.values())
    assert any(p.name == "paper_grade_effective" for p in params), (
        "B-1602: function must accept paper_grade_effective kwarg"
    )
    # paper_grade_effective param has default None for back-compat
    pg_param = next(p for p in params if p.name == "paper_grade_effective")
    assert pg_param.default is None


def test_seed_global_rng_legacy_caller_falls_back_to_env(monkeypatch):
    """Legacy caller `_seed_global_rng(seed)` without kwarg falls back to
    env-only via helper (preserves pre-B-1602 behaviour for unrefactored callers).
    """
    from p79.experiment.runner.main import _seed_global_rng
    # Should not raise — just exercises the back-compat path
    monkeypatch.setenv("P79_PAPER_GRADE", "0")
    _seed_global_rng(42)
    # With paper_grade=False (default), no deterministic op enforcement assertion
    # is reachable from outside (relies on torch internals); just verify call
    # doesn't crash.
