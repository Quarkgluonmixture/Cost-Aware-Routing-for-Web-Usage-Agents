"""P1-14-B schema 4-place sync invariant test (/stress Phase 0 2026-05-19).

Codex Mode B F8 finding: Phase 2 telemetry fields can land additively, BUT
each new field MUST sync 4 places or silent drift occurs:

  1. Dataclass default (`p79/experiment/types.py:StepRecordV2 / EpisodeSummaryV2`)
  2. v2 defaults dict (`p79/experiment/schema_migrations/v2.py:STEP_RECORD_V2_DEFAULTS / EPISODE_SUMMARY_V2_DEFAULTS`)
  3. Optional type map (`_STEP_OPTIONAL_FIELD_TYPES / _EPISODE_OPTIONAL_FIELD_TYPES`)
  4. Paper-grade optional-keys set (`PAPER_GRADE_STEP_OPTIONAL_KEYS / PAPER_GRADE_EPISODE_OPTIONAL_KEYS`)
     — only when runner always stamps (else field stays optional dataclass-default-only)

Drift patterns this test catches:
- field added to dataclass but NOT to v2 defaults dict
  → legacy JSONL readers see KeyError on filldefaults
- field added to v2 defaults but NOT to dataclass
  → new writes silently miss the field
- field in PAPER_GRADE_*_OPTIONAL_KEYS but runner doesn't stamp + dataclass
  has no default → every paper-grade write fails validation
- type map missing field → silent VALUE-type contract violation

The test is a CONTAINMENT check: every field present in dataclass MUST appear in
v2 defaults. Type-map coverage is checked for fields in PAPER_GRADE_*_OPTIONAL_KEYS.
Runner-stamp verification is NOT done here (would require grep p79/experiment/runner/);
that's tracked per-field at code review time.
"""
from __future__ import annotations

import dataclasses

import pytest

from p79.experiment.schema_migrations.v2 import (
    EPISODE_SUMMARY_V2_DEFAULTS,
    STEP_RECORD_V2_DEFAULTS,
)
from p79.experiment.types import (
    EpisodeSummaryV2,
    PAPER_GRADE_EPISODE_OPTIONAL_KEYS,
    PAPER_GRADE_STEP_OPTIONAL_KEYS,
    StepRecordV2,
    _EPISODE_OPTIONAL_FIELD_TYPES,
    _STEP_OPTIONAL_FIELD_TYPES,
)


def _dataclass_field_names(cls) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)}


def test_step_record_dataclass_synced_with_v2_defaults():
    """Every StepRecordV2 dataclass field must appear in STEP_RECORD_V2_DEFAULTS."""
    dc_fields = _dataclass_field_names(StepRecordV2)
    defaults_keys = set(STEP_RECORD_V2_DEFAULTS.keys())
    missing_in_defaults = dc_fields - defaults_keys
    extra_in_defaults = defaults_keys - dc_fields
    assert not missing_in_defaults, (
        f"P1-14-B drift: StepRecordV2 dataclass fields NOT in v2 defaults: "
        f"{sorted(missing_in_defaults)}. Add to "
        f"schema_migrations/v2.py:STEP_RECORD_V2_DEFAULTS."
    )
    assert not extra_in_defaults, (
        f"P1-14-B drift: v2 defaults has keys NOT in StepRecordV2 dataclass: "
        f"{sorted(extra_in_defaults)}. Remove stale keys OR add to dataclass."
    )


def test_episode_summary_dataclass_synced_with_v2_defaults():
    """Every EpisodeSummaryV2 dataclass field must appear in EPISODE_SUMMARY_V2_DEFAULTS."""
    dc_fields = _dataclass_field_names(EpisodeSummaryV2)
    defaults_keys = set(EPISODE_SUMMARY_V2_DEFAULTS.keys())
    missing_in_defaults = dc_fields - defaults_keys
    extra_in_defaults = defaults_keys - dc_fields
    assert not missing_in_defaults, (
        f"P1-14-B drift: EpisodeSummaryV2 dataclass fields NOT in v2 defaults: "
        f"{sorted(missing_in_defaults)}. Add to "
        f"schema_migrations/v2.py:EPISODE_SUMMARY_V2_DEFAULTS."
    )
    assert not extra_in_defaults, (
        f"P1-14-B drift: v2 defaults has keys NOT in EpisodeSummaryV2: "
        f"{sorted(extra_in_defaults)}. Remove stale keys OR add to dataclass."
    )


def test_paper_grade_step_optional_keys_have_type_map_entry():
    """Every key in PAPER_GRADE_STEP_OPTIONAL_KEYS must have a type-map entry."""
    type_map_keys = set(_STEP_OPTIONAL_FIELD_TYPES.keys())
    missing = PAPER_GRADE_STEP_OPTIONAL_KEYS - type_map_keys
    assert not missing, (
        f"P1-14-B drift: PAPER_GRADE_STEP_OPTIONAL_KEYS without type-map entry: "
        f"{sorted(missing)}. Add to types.py:_STEP_OPTIONAL_FIELD_TYPES."
    )


def test_paper_grade_episode_optional_keys_have_type_map_entry():
    """Every key in PAPER_GRADE_EPISODE_OPTIONAL_KEYS must have a type-map entry."""
    type_map_keys = set(_EPISODE_OPTIONAL_FIELD_TYPES.keys())
    missing = PAPER_GRADE_EPISODE_OPTIONAL_KEYS - type_map_keys
    assert not missing, (
        f"P1-14-B drift: PAPER_GRADE_EPISODE_OPTIONAL_KEYS without type-map entry: "
        f"{sorted(missing)}. Add to types.py:_EPISODE_OPTIONAL_FIELD_TYPES."
    )


def test_paper_grade_step_optional_keys_have_dataclass_field():
    """Every key in PAPER_GRADE_STEP_OPTIONAL_KEYS must exist as dataclass field."""
    dc_fields = _dataclass_field_names(StepRecordV2)
    missing = PAPER_GRADE_STEP_OPTIONAL_KEYS - dc_fields
    assert not missing, (
        f"P1-14-B drift: PAPER_GRADE_STEP_OPTIONAL_KEYS keys NOT in StepRecordV2 "
        f"dataclass: {sorted(missing)}. Add field OR remove from OPTIONAL_KEYS."
    )


def test_paper_grade_episode_optional_keys_have_dataclass_field():
    """Every key in PAPER_GRADE_EPISODE_OPTIONAL_KEYS must exist as dataclass field."""
    dc_fields = _dataclass_field_names(EpisodeSummaryV2)
    missing = PAPER_GRADE_EPISODE_OPTIONAL_KEYS - dc_fields
    assert not missing, (
        f"P1-14-B drift: PAPER_GRADE_EPISODE_OPTIONAL_KEYS keys NOT in "
        f"EpisodeSummaryV2 dataclass: {sorted(missing)}. Add field OR remove."
    )


def test_phase2_intervention_fields_present():
    """P0-1-ABC* Phase 2 intervention fields land in all 4 places."""
    step_dc = _dataclass_field_names(StepRecordV2)
    episode_dc = _dataclass_field_names(EpisodeSummaryV2)
    step_defaults = set(STEP_RECORD_V2_DEFAULTS.keys())
    episode_defaults = set(EPISODE_SUMMARY_V2_DEFAULTS.keys())

    intervention_step_fields = {
        "intervention_type",
        "counted_as_agent_action",
        "intervention_from_url",
        "intervention_recovery_url",
    }
    intervention_episode_fields = {
        "runner_intervention_count",
        "about_blank_recovery_count",
    }

    for f in intervention_step_fields:
        assert f in step_dc, f"P0-1-ABC* intervention step field missing in dataclass: {f}"
        assert f in step_defaults, f"P0-1-ABC* intervention step field missing in v2 defaults: {f}"
        assert f in _STEP_OPTIONAL_FIELD_TYPES, (
            f"P0-1-ABC* intervention step field missing in _STEP_OPTIONAL_FIELD_TYPES: {f}"
        )
        assert f in PAPER_GRADE_STEP_OPTIONAL_KEYS, (
            f"P0-1-ABC* intervention step field missing in PAPER_GRADE_STEP_OPTIONAL_KEYS: {f}"
        )

    for f in intervention_episode_fields:
        assert f in episode_dc, f"P0-1-ABC* intervention episode field missing in dataclass: {f}"
        assert f in episode_defaults, f"P0-1-ABC* intervention episode field missing in v2 defaults: {f}"
        assert f in _EPISODE_OPTIONAL_FIELD_TYPES, (
            f"P0-1-ABC* intervention episode field missing in _EPISODE_OPTIONAL_FIELD_TYPES: {f}"
        )
        assert f in PAPER_GRADE_EPISODE_OPTIONAL_KEYS, (
            f"P0-1-ABC* intervention episode field missing in PAPER_GRADE_EPISODE_OPTIONAL_KEYS: {f}"
        )
