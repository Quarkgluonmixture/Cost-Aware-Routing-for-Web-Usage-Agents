"""Fire-6 RCA Stage C2 (/stress 2026-05-20): diagnostic-replay mode + Gate 8
diagnostic-scoped override invariants.

Covers:
  - schema: diagnostic_replay + sr_excluded present in all 4 sync places
  - config: non-canonical output_root redirect + env-driven normalize
  - strict loader: sr_excluded firewall (default-reject, opt-out, archive-safe,
    type-guard)
  - Gate 8 override: 4-guard fail-closed scoping (the security property — a
    leaked env var ALONE must NOT bypass canonical Gate 8)
  - CLI _parse_tasks range parsing
"""
from __future__ import annotations

import argparse
import dataclasses
import json

import pytest

from scripts.maintenance import quarantine_registry as qr


# ─── schema 4-place sync ──────────────────────────────────────────────────────
def test_schema_fields_in_all_four_places():
    from p79.experiment.types import (
        EpisodeSummaryV2,
        PAPER_GRADE_EPISODE_OPTIONAL_KEYS,
        _EPISODE_OPTIONAL_FIELD_TYPES,
    )
    from p79.experiment.schema_migrations.v2 import EPISODE_SUMMARY_V2_DEFAULTS

    dc = {f.name for f in dataclasses.fields(EpisodeSummaryV2)}
    for k in ("diagnostic_replay", "sr_excluded"):
        assert k in dc, f"{k} missing from EpisodeSummaryV2 dataclass"
        assert k in EPISODE_SUMMARY_V2_DEFAULTS, f"{k} missing from v2 defaults"
        assert EPISODE_SUMMARY_V2_DEFAULTS[k] is False, f"{k} canonical default must be False"
        assert k in PAPER_GRADE_EPISODE_OPTIONAL_KEYS, f"{k} missing from paper-grade keys"
        assert k in _EPISODE_OPTIONAL_FIELD_TYPES, f"{k} missing from type map"
        assert _EPISODE_OPTIONAL_FIELD_TYPES[k] == (bool,), f"{k} must be always-bool"


# ─── config: output_root redirect + env normalize ─────────────────────────────
def test_output_root_canonical_vs_diagnostic(tmp_path):
    from p79.experiment.config import normalize_config, resolve_output_root

    canon = normalize_config({
        "experiment": {"benchmark": "visualwebarena", "phase": "phase1",
                       "run_id": "r1", "output_root": str(tmp_path)},
    })
    assert canon["diagnostic_replay"] is False
    p = resolve_output_root(canon)
    assert p.parts[-3:] == ("visualwebarena", "phase1", "r1")

    diag = normalize_config({
        "diagnostic_replay": True,
        "experiment": {"benchmark": "visualwebarena", "phase": "phase1",
                       "run_id": "diag_x", "output_root": str(tmp_path)},
    })
    pd = resolve_output_root(diag)
    assert pd.parts[-2:] == ("diagnostic_replay", "diag_x")
    # CRITICAL: diagnostic output must NOT live under the canonical phase tree
    assert "phase1" not in pd.parts


def test_normalize_env_flag(monkeypatch, tmp_path):
    from p79.experiment.config import normalize_config

    monkeypatch.setenv("P79_DIAGNOSTIC_REPLAY", "1")
    c = normalize_config({
        "experiment": {"benchmark": "visualwebarena", "phase": "phase1",
                       "run_id": "r", "output_root": str(tmp_path)},
    })
    assert c["diagnostic_replay"] is True


# ─── strict loader: sr_excluded firewall ──────────────────────────────────────
def _write_ep(path, **over):
    base = {"schema_version": "2.0", "success": True, "task_id": 4}
    base.update(over)
    path.write_text(json.dumps(base))


def test_strict_loader_default_rejects_sr_excluded(tmp_path):
    from p79.experiment.io_utils import load_episode_summary_strict

    p = tmp_path / "1_summary_v2.json"
    _write_ep(p, sr_excluded=True)
    # default reject_sr_excluded=True → lenient returns None, strict raises
    assert load_episode_summary_strict(p, mode="lenient") is None
    with pytest.raises(ValueError):
        load_episode_summary_strict(p, mode="strict")
    # explicit opt-out loads the diagnostic episode (the diagnostic analysis path)
    got = load_episode_summary_strict(p, mode="strict", reject_sr_excluded=False)
    assert got is not None and got["sr_excluded"] is True


def test_strict_loader_archive_row_safe(tmp_path):
    """Pre-C2 archive rows (no sr_excluded key) still load (key absent → False)."""
    from p79.experiment.io_utils import load_episode_summary_strict

    p = tmp_path / "1_summary_v2.json"
    _write_ep(p)  # no sr_excluded key
    assert load_episode_summary_strict(p, mode="strict") is not None


def test_strict_loader_sr_excluded_type_guard(tmp_path):
    """String 'false' is Python-truthy — must fail loud, not slip a diagnostic
    episode INTO canonical SR (B-734 type-guard lineage)."""
    from p79.experiment.io_utils import load_episode_summary_strict

    p = tmp_path / "1_summary_v2.json"
    _write_ep(p, sr_excluded="false")
    with pytest.raises(ValueError):
        load_episode_summary_strict(p, mode="strict")


# ─── Gate 8 override: 4-guard fail-closed scoping (UNIT) ──────────────────────
def _ns(**kw):
    d = dict(diagnostic_replay=True, output_path="results/diagnostic_replay/x")
    d.update(kw)
    return argparse.Namespace(**d)


def test_override_active_all_guards(monkeypatch):
    monkeypatch.setenv("QUARANTINE_DIAGNOSTIC_REPLAY", "1")
    active, reasons = qr._diagnostic_override_active(_ns(), [4, 75])
    assert active is True, reasons


def test_override_inactive_without_env(monkeypatch):
    monkeypatch.delenv("QUARANTINE_DIAGNOSTIC_REPLAY", raising=False)
    active, reasons = qr._diagnostic_override_active(_ns(), [4, 75])
    assert active is False
    assert any("env" in r for r in reasons)


def test_override_inactive_without_flag(monkeypatch):
    monkeypatch.setenv("QUARANTINE_DIAGNOSTIC_REPLAY", "1")
    active, _ = qr._diagnostic_override_active(_ns(diagnostic_replay=False), [4, 75])
    assert active is False


def test_override_inactive_canonical_path(monkeypatch):
    monkeypatch.setenv("QUARANTINE_DIAGNOSTIC_REPLAY", "1")
    active, reasons = qr._diagnostic_override_active(
        _ns(output_path="results/visualwebarena/phase1/x"), [4, 75])
    assert active is False
    assert any("non-canonical" in r for r in reasons)


def test_override_inactive_too_many_tasks(monkeypatch):
    monkeypatch.setenv("QUARANTINE_DIAGNOSTIC_REPLAY", "1")
    active, reasons = qr._diagnostic_override_active(_ns(), list(range(0, 234)))
    assert active is False
    assert any("too large" in r for r in reasons)


def test_override_inactive_empty_tasks(monkeypatch):
    monkeypatch.setenv("QUARANTINE_DIAGNOSTIC_REPLAY", "1")
    active, _ = qr._diagnostic_override_active(_ns(), [])
    assert active is False


# ─── Gate 8 override: end-to-end (preflight stays pure; leaked env still halts) ─
@pytest.fixture
def recurrent_registry(tmp_path, monkeypatch):
    """task 75 quarantined across 2 distinct fires + FULLY classified
    (unclassified_count=0) — still a PURE cross_fire_recurrence (Rule 2) HALT.
    Mirrors the real cls-75 state and exercises the key property: classification
    does NOT unblock a cross-fire-recurrent task."""
    reg = tmp_path / "quarantine_registry.jsonl"
    rows = [
        {"event_type": "quarantine", "site": "classifieds", "task_id": 75,
         "run_id": "FIRE_A", "error_class": "EvaluatorUnavailableError"},
        {"event_type": "quarantine", "site": "classifieds", "task_id": 75,
         "run_id": "FIRE_B", "error_class": "Page.screenshot Timeout"},
        # two classifications → unclassified_count = max(0, 2-2) = 0 → Rule 1
        # OFF, so the HALT is purely Rule 2 (cross_fire_recurrence).
        {"event_type": "classification", "site": "classifieds", "task_id": 75,
         "classification": "unreproducible_in_isolation",
         "ts": "2026-05-20T00:00:00+00:00"},
        {"event_type": "classification", "site": "classifieds", "task_id": 75,
         "classification": "unreproducible_in_isolation",
         "ts": "2026-05-20T00:01:00+00:00"},
    ]
    reg.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr(qr, "REGISTRY_PATH", reg)
    return reg


def test_preflight_check_stays_pure_halts(recurrent_registry):
    """Core gate returns the REAL halt regardless of any override layer."""
    should_halt, blocking = qr.preflight_check("classifieds", [75])
    assert should_halt is True
    assert any(b["task_id"] == 75 and "recurrence" in b["rule"] for b in blocking)


def test_cmd_preflight_canonical_halts(recurrent_registry, monkeypatch):
    monkeypatch.delenv("QUARANTINE_DIAGNOSTIC_REPLAY", raising=False)
    ns = argparse.Namespace(site="classifieds", tasks="75", halt_threshold=1,
                            diagnostic_replay=False, output_path=None)
    assert qr._cmd_preflight(ns) == 1


def test_cmd_preflight_diagnostic_override_passes(recurrent_registry, monkeypatch):
    monkeypatch.setenv("QUARANTINE_DIAGNOSTIC_REPLAY", "1")
    ns = argparse.Namespace(site="classifieds", tasks="4,75", halt_threshold=1,
                            diagnostic_replay=True,
                            output_path="results/diagnostic_replay/diag_x")
    assert qr._cmd_preflight(ns) == 0


def test_cmd_preflight_leaked_env_alone_still_halts(recurrent_registry, monkeypatch):
    """SECURITY: env set but a canonical preflight call (no --diagnostic-replay
    flag, no non-canonical output path) MUST still HALT."""
    monkeypatch.setenv("QUARANTINE_DIAGNOSTIC_REPLAY", "1")
    ns = argparse.Namespace(site="classifieds", tasks="75", halt_threshold=1,
                            diagnostic_replay=False, output_path=None)
    assert qr._cmd_preflight(ns) == 1


# ─── CLI _parse_tasks ─────────────────────────────────────────────────────────
def test_parse_tasks_range_and_csv():
    from p79.cli.run_experiment import _parse_tasks

    assert _parse_tasks("4,75") == [4, 75]
    assert _parse_tasks("0-3") == [0, 1, 2, 3]
    assert _parse_tasks("7") == [7]
    assert _parse_tasks("5,5,3,1-2") == [1, 2, 3, 5]
