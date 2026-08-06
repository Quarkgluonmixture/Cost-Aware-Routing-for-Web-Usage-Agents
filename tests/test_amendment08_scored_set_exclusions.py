"""AMENDMENT_08 — scored-set protocol exclusions (reddit tasks 160 + 58).

The amendment separates two denominators that were previously one number:

    collection denominator   how many episodes a run must produce   (red 205)
    scoring denominator      what a success RATE divides by         (red 203)

Most of what can go wrong here is a caller reaching for the wrong one, so the
tests below pin the split at each layer rather than just the two integers.
"""

from __future__ import annotations

import pytest

from p79.experiment.analysis import paper_scored_task_count, scored_task_count
from p79.experiment.tasks import (
    PROTOCOL_EXCLUSIONS,
    ProtocolExclusion,
    load_tasks,
    protocol_excluded_task_ids,
)
from scripts.analysis.lib.canonical_task_universe import (
    collected_task_ids,
    expected_scored_ids,
    protocol_excluded_in_universe,
)


# --------------------------------------------------------------------------- #
# registry shape
# --------------------------------------------------------------------------- #

def test_registry_holds_exactly_the_two_amended_reddit_tasks():
    entries = PROTOCOL_EXCLUSIONS[("visualwebarena", "reddit")]
    assert {e.task_id for e in entries} == {58, 160}
    assert {e.tier for e in entries} == {"A", "B"}
    assert all(e.amendment == "AMENDMENT_08" for e in entries)


def test_every_entry_states_a_rule_and_a_reason():
    """A bare task-ID list is the thing a reviewer cannot check. Each entry has to
    carry the uniform criterion it instantiates plus why this task meets it."""
    for entries in PROTOCOL_EXCLUSIONS.values():
        for e in entries:
            assert isinstance(e, ProtocolExclusion)
            assert len(e.rule) > 40, f"task {e.task_id}: rule too thin to audit"
            assert len(e.reason) > 80, f"task {e.task_id}: reason too thin to audit"


def test_tier_selection_drives_the_sensitivity_arms():
    assert protocol_excluded_task_ids("reddit", tiers=()) == frozenset()
    assert protocol_excluded_task_ids("reddit", tiers=("A",)) == frozenset({160})
    assert protocol_excluded_task_ids("reddit") == frozenset({58, 160})


def test_sites_without_exclusions_are_untouched():
    # AMENDMENT_09 (2026-08-03) gave shopping its own tier-A entries, so the
    # "no exclusions" set is classifieds + both WA sites. Reddit and shopping
    # are asserted by their own tests.
    assert protocol_excluded_task_ids("classifieds") == frozenset()
    assert protocol_excluded_task_ids("reddit", "webarena") == frozenset()
    assert protocol_excluded_task_ids("shopping", "webarena") == frozenset()
    assert protocol_excluded_task_ids("shopping_admin", "webarena") == frozenset()


def test_amendment09_shopping_exclusions():
    """AMENDMENT_09: the SAME tier-A rule, applied to shopping, pre-data.

    463/465 are conditional "add X to cart IF ..." tasks whose condition is
    false, so their program_html carries only `must_exclude` — an agent that
    never opens the page scores 1. Verified as a uniform rule, not a pick: the
    predicate selects exactly these 2 of 466 VWA shopping tasks and 0 of the 192
    WA shopping / 182 WA shopping_admin tasks.
    """
    assert protocol_excluded_task_ids("shopping", tiers=("A",)) == frozenset({463, 465})
    # tier B is empty for shopping — nothing here needed trajectories to warrant
    assert protocol_excluded_task_ids("shopping", tiers=("B",)) == frozenset()


def test_amendment10_shopping_substrate_exclusion():
    """AMENDMENT_10: task 345, tier E (substrate 404, environment-probe-confirmed).

    A DIFFERENT rule from AMENDMENT_08/09 — not "the eval cannot discriminate"
    but "the start_url resource does not exist". Rule was evaluated over 1466
    start_url references / 431 unique URLs across all 6 benchmark-sites and
    selects exactly this one; the two false-positive groups (session-less admin
    probe, 6-way concurrency) are documented in the amendment §2.

    Tier E is in the DEFAULT tiers. That is the load-bearing assertion here: if
    it were dropped, the amendment doc would say 432 while the code returned
    433 — a silent doc/code fork of exactly the kind that makes a scored-set
    number untrustworthy.
    """
    assert protocol_excluded_task_ids("shopping", tiers=("E",)) == frozenset({345})
    assert protocol_excluded_task_ids("shopping") == frozenset({345, 463, 465})
    # tier E must not leak into the sites that never had a substrate defect
    assert protocol_excluded_task_ids("reddit", tiers=("E",)) == frozenset()
    assert protocol_excluded_task_ids("classifieds", tiers=("E",)) == frozenset()
    assert protocol_excluded_task_ids("shopping", "webarena", tiers=("E",)) == frozenset()


# --------------------------------------------------------------------------- #
# the two denominators
# --------------------------------------------------------------------------- #

def test_collection_denominator_is_unchanged():
    """Fire-completeness contract: a landed run still owes 205 reddit episodes.
    If this drifts, `validate_run` / `paper_grade_check` start reporting real
    runs as contaminated."""
    assert scored_task_count("reddit", "visualwebarena") == 205
    assert scored_task_count("classifieds", "visualwebarena") == 224


def test_scoring_denominator_drops_only_the_excluded_tasks():
    assert paper_scored_task_count("reddit", "visualwebarena") == 203
    assert paper_scored_task_count("classifieds", "visualwebarena") == 224
    assert paper_scored_task_count("reddit", "visualwebarena", tiers=()) == 205
    assert paper_scored_task_count("reddit", "visualwebarena", tiers=("A",)) == 204
    # AMENDMENT_09 → 433, AMENDMENT_10 → 432. The collection denominator stays
    # 435 (asserted separately below), which is what keeps the fire-completeness
    # check and both sensitivity arms computable from any landed run.
    assert paper_scored_task_count("shopping", "visualwebarena", tiers=()) == 435
    assert paper_scored_task_count("shopping", "visualwebarena", tiers=("A",)) == 433
    assert paper_scored_task_count("shopping", "visualwebarena") == 432
    assert scored_task_count("shopping", "visualwebarena") == 435


def test_scoring_denominator_cannot_double_subtract_an_na_task(monkeypatch):
    """An exclusion naming an already-N/A-dropped task must be a no-op, not a
    second subtraction — otherwise the denominator silently under-counts."""
    import p79.experiment.tasks as tasks_mod

    na_ids = sorted(
        __import__("p79.experiment.analysis", fromlist=["x"])._load_na_task_ids(
            "reddit", "visualwebarena"
        )
    )
    assert na_ids, "reddit should have N/A tasks; fixture assumption broken"
    monkeypatch.setitem(
        tasks_mod.PROTOCOL_EXCLUSIONS,
        ("visualwebarena", "reddit"),
        (
            ProtocolExclusion(
                task_id=na_ids[0], tier="A", rule="x" * 50, reason="y" * 100,
                amendment="TEST",
            ),
        ),
    )
    assert paper_scored_task_count("reddit", "visualwebarena") == 205


# --------------------------------------------------------------------------- #
# the canonical universe + its SHA
# --------------------------------------------------------------------------- #

def test_universe_split_matches_the_counts():
    assert len(collected_task_ids("reddit")) == 205
    assert len(expected_scored_ids("reddit")[0]) == 203
    assert protocol_excluded_in_universe("reddit") == frozenset({58, 160})
    assert protocol_excluded_in_universe("classifieds") == frozenset()


def test_excluded_ids_are_absent_from_the_scored_universe():
    scored, _ = expected_scored_ids("reddit")
    assert 58 not in scored and 160 not in scored
    assert {58, 160} <= collected_task_ids("reddit")


def test_changing_the_universe_changes_the_sha():
    """The SHA is what stale artifacts are caught by — if it did not move, a
    pre-amendment figure would silently pass the provenance cross-check."""
    pre = expected_scored_ids("reddit", "visualwebarena", ())[1]
    post = expected_scored_ids("reddit", "visualwebarena", ("A", "B"))[1]
    assert pre != post


# --------------------------------------------------------------------------- #
# the runner must NOT apply these
# --------------------------------------------------------------------------- #

def test_load_tasks_still_yields_the_excluded_tasks(tmp_path):
    """Deliberate: keeping collection at 205 is what makes both sensitivity arms
    computable from any run, old or new, and keeps the completeness contract
    identical across the amendment boundary."""
    from p79.experiment.analysis import _resolve_site_config

    cfg = {
        "experiment": {"benchmark": "visualwebarena"},
        "task": {
            "include_sites": ["reddit"],
            "site_configs": {
                "reddit": str(_resolve_site_config("reddit", "visualwebarena"))
            },
            "exclude_na_tasks": True,
        },
    }
    ids = {int(s.task_id) for s in load_tasks(cfg, tmp_path)}
    assert len(ids) == 205
    assert {58, 160} <= ids


def test_load_tasks_does_not_import_the_exclusion_helper():
    """Guard against a later edit wiring the registry into the fire path, which
    would make new runs produce 203 episodes and break the exact-count check
    against every landed run."""
    import inspect

    src = inspect.getsource(load_tasks)
    assert "protocol_excluded" not in src
    assert "PROTOCOL_EXCLUSIONS" not in src


# --------------------------------------------------------------------------- #
# aggregator completeness must tolerate the collected-but-unscored extras
# --------------------------------------------------------------------------- #

def test_aggregator_treats_excluded_episodes_as_expected_not_contamination(tmp_path):
    """205 landed episodes against a 203-task scored set must still read as a
    complete cell; only genuinely unexpected IDs may set `extra_ids`."""
    import json

    from scripts.analysis.aggregate_sr_fp_per_mode import aggregate_cell

    scored, _ = expected_scored_ids("reddit")
    ep = tmp_path / "episodes"
    ep.mkdir()
    for tid in sorted(collected_task_ids("reddit")):
        (ep / f"reddit_task_{tid}_summary_v2.json").write_text(
            json.dumps(
                {
                    "schema_version": "2.0",
                    "task_id": tid,
                    "success": tid == 160,   # only the excluded task "succeeds"
                    "benchmark_site": "reddit",
                    "obs_mode": "dom",
                }
            )
        )

    row = aggregate_cell("B0", "reddit", "dom", ep)
    assert row["expected_n"] == 203
    assert row["extra_ids"] == []
    assert row["protocol_excluded_observed"] == [58, 160]
    assert row["complete"] is True
    assert row["completeness_ratio"] == 1.0
    # the one "success" sat on an excluded task, so the scored SR is zero
    assert row["n_success"] == 0
    assert row["sr_pct"] == 0.0


def test_aggregator_still_flags_a_genuinely_unexpected_task_id(tmp_path):
    import json

    from scripts.analysis.aggregate_sr_fp_per_mode import aggregate_cell

    ep = tmp_path / "episodes"
    ep.mkdir()
    for tid in sorted(collected_task_ids("reddit")) + [9999]:
        (ep / f"reddit_task_{tid}_summary_v2.json").write_text(
            json.dumps(
                {"schema_version": "2.0", "task_id": tid, "success": False,
                 "benchmark_site": "reddit", "obs_mode": "dom"}
            )
        )
    row = aggregate_cell("B0", "reddit", "dom", ep)
    assert row["extra_ids"] == [9999]
    assert row["complete"] is False


# --------------------------------------------------------------------------- #
# the H1 gate aggregator must not treat the excluded episodes as contamination
# --------------------------------------------------------------------------- #

def test_gate_aggregator_does_not_skip_reddit_cells():
    """The near-miss this test exists for: `aggregate_phase1_prereg_gate` runs its
    OWN completeness check, separate from `aggregate_sr_fp_per_mode`. Fixing only
    the latter left all three reddit cells failing `complete_exact`, so the gate
    silently ran at k=3 on classifieds alone and reported framing=R5 — a paper-death
    verdict produced entirely by dropped data."""
    from scripts.analysis.aggregate_phantom_lift import CELLS
    from scripts.analysis.aggregate_phase1_prereg_gate import _cell_drop_one_theta_se

    reddit_cells = [c for c in CELLS if c["site"] == "reddit"]
    if not reddit_cells:
        pytest.skip("no reddit cells registered in this checkout")
    for cell in reddit_cells:
        r = _cell_drop_one_theta_se(cell)
        if r.get("expected_n") != 203:
            pytest.skip("landed reddit data absent on this host")
        assert r["complete_exact"] is True, (
            f"{cell['baseline']}/reddit skipped: extra_ids={r.get('extra_ids')}"
        )
        for mode, extras in r["protocol_excluded_observed"].items():
            assert extras == [58, 160], f"{mode}: {extras}"


def test_gate_sensitivity_arms_are_reproducible():
    """AMENDMENT_08 §5 promises a reviewer can recompute each arm. That needs the
    gate to accept a narrower universe AND the landed-but-unscored ids together —
    without `tolerate_extra_ids` every arm but the pre-amendment one fails
    `complete_exact` and the promised comparison cannot be run."""
    from scripts.analysis.aggregate_phantom_lift import CELLS
    from scripts.analysis.aggregate_phase1_prereg_gate import _cell_drop_one_theta_se

    cell = next((c for c in CELLS if c["site"] == "reddit"), None)
    if cell is None:
        pytest.skip("no reddit cell registered")
    for tiers in ((), ("A",), ("A", "B")):
        ids, _ = expected_scored_ids("reddit", "visualwebarena", tiers)
        tol = collected_task_ids("reddit") - ids
        r = _cell_drop_one_theta_se(cell, expected_ids=ids, tolerate_extra_ids=tol)
        if "n_tasks" not in r:
            pytest.skip("landed reddit data absent on this host")
        assert r["complete_exact"] is True, f"tiers={tiers}: {r.get('extra_ids')}"
        assert r["n_tasks"] == len(ids)
