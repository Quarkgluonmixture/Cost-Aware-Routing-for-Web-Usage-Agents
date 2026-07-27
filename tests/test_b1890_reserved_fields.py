"""B-1890 — reserved schema fields must be declared, and declarations must be true.

A count field that the runner never assigns still serializes as `0`, which is a
legal observation. Nothing in a landed JSON tells the two apart. On 2026-07-27
that produced a retracted headline ("114 of 256 successes were zero-mutation
mutation tasks") derived from `effective_mutating_action_count == 0` across 558
episodes that all carried the dataclass default.

Two directions are locked:
  * a field on the NOT_POPULATED list must not quietly start being written
    (that means the list is stale and analysts are avoiding a usable field);
  * a field that is constant-at-default across the whole library must be on the
    list (that is the trap, unlabelled).
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import pytest

from p79.experiment.schema_migrations.v2 import (
    EPISODE_SUMMARY_V2_DEFAULTS,
    NOT_POPULATED_BY_RUNNER,
)

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "visualwebarena" / "phase1"

# Fields that ARE computed by the runner but happen to be constant across the
# current library — a real measurement that legitimately reads zero, not the
# B-1890 trap. Each was checked against its assignment site before being listed;
# a bare name here would turn this test into a rubber stamp.
CONSTANT_BY_NATURE = {
    "session_lost_preserved": "PROTOCOL_NOTE_01 covariate; fires only on session loss",
    "needs_reevaluation": "B-486 quarantine flag; fires only on runner exception",
    "benchmark_noise": "B-600 transparency flag; rare by construction",
    # verified 2026-07-27 — all seven assigned by the runner, all genuinely 0:
    "seed": "config echo, not a measurement; schema default 42 equals the Phase 1 seed",
    "escalation_count": "runner/main.py:638 — router escalations; Phase 1 conditions are router_0, so structurally 0",
    "retries": "runner/main.py:623 — sums step-level retry_count; distinct from the B-1881 episode-level transient retry",
    "image_encode_error_step_count": "runner/main.py:4448 — B-403 transparency column; 0 means no encode failures occurred",
    "partial_recovery_step_count": "runner/main.py:2124 — written only on the partial-recovery path",
    "screenshot_timeout_recovered_count": "runner/main.py:4351 — screenshot-timeout recovery is rare",
    "screenshot_timeout_recovered_total_ms": "runner/main.py:4361 — companion to the count above",
}


def test_every_reserved_field_exists_in_the_schema():
    unknown = sorted(set(NOT_POPULATED_BY_RUNNER) - set(EPISODE_SUMMARY_V2_DEFAULTS))
    assert not unknown, (
        f"NOT_POPULATED_BY_RUNNER names fields absent from the v2 schema: {unknown}. "
        "Either the field was renamed or the entry is stale."
    )


def test_every_reserved_field_carries_a_reason():
    for name, reason in NOT_POPULATED_BY_RUNNER.items():
        assert len(reason) > 20, f"{name}: reason too thin to act on"
        assert "B-1890" in reason, f"{name}: reason should point at the bug record"


def test_constant_by_nature_entries_carry_a_reason():
    """Same bar for the other list — an unexplained name here silently exempts a
    field from the scan, which is the failure mode this whole file exists to
    prevent."""
    for name, reason in CONSTANT_BY_NATURE.items():
        assert len(reason) > 25, f"{name}: reason too thin — say why 0 is real"
    assert not (set(CONSTANT_BY_NATURE) & set(NOT_POPULATED_BY_RUNNER)), (
        "a field cannot be both computed-and-constant and never-computed"
    )


def test_runner_does_not_assign_reserved_fields():
    """The declaration has to stay true. If an implementation lands, its entry
    must be deleted in the same commit — otherwise analysts keep steering around
    a field that is now real."""
    runner_src = (REPO / "p79" / "experiment" / "runner" / "main.py").read_text()
    offenders = []
    for name in NOT_POPULATED_BY_RUNNER:
        # An assignment or dict-write, not a mere mention in a comment.
        for pat in (f'"{name}":', f"{name} =", f'["{name}"] ='):
            for line in runner_src.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if pat in stripped:
                    offenders.append((name, stripped[:90]))
    assert not offenders, (
        "reserved field is now being written by the runner — delete its "
        f"NOT_POPULATED_BY_RUNNER entry: {offenders}"
    )


@pytest.mark.skipif(
    os.environ.get("RUN_LOCAL_DATA_TESTS") != "1",
    reason="library-wide scan skipped by default — set RUN_LOCAL_DATA_TESTS=1",
)
def test_no_unlabelled_field_is_constant_at_default_across_the_library():
    """The other direction: catch the NEXT reserved-looking field before someone
    builds a claim on it. Samples landed episode summaries and flags any numeric
    schema field that never once deviates from its default."""
    summaries = sorted(RESULTS.glob("*/*/episodes/*_summary_v2.json"))
    if len(summaries) < 200:
        pytest.skip(f"only {len(summaries)} landed summaries; scan needs a real library")
    rng = random.Random(42)
    sample = rng.sample(summaries, min(600, len(summaries)))

    numeric_defaults = {
        k: v for k, v in EPISODE_SUMMARY_V2_DEFAULTS.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    ever_deviated = set()
    seen = set()
    for path in sample:
        try:
            row = json.load(open(path))
        except Exception:
            continue
        for k, default in numeric_defaults.items():
            if k not in row:
                continue
            seen.add(k)
            if row[k] != default:
                ever_deviated.add(k)

    suspect = sorted(seen - ever_deviated - set(NOT_POPULATED_BY_RUNNER) - set(CONSTANT_BY_NATURE))
    assert not suspect, (
        f"these numeric fields never deviate from their default across {len(sample)} "
        f"landed episodes: {suspect}\n"
        "Either the runner does not populate them — add to NOT_POPULATED_BY_RUNNER "
        "so nobody builds an analysis on them (B-1890) — or the constancy is real "
        "and rare, in which case add to CONSTANT_BY_NATURE with a reason."
    )
