"""B-1906: turn the "grep census" lesson into a standing lint.

/stress Mode B (codex) 2026-07-27 found three universe leaks that all three
prior universe fixes had missed, and the横切 lesson recorded in 实验笔记 §388.7.1
was "修横切关注点必须 grep 普查, 不能顺着调用链走".  A lesson that depends on
somebody remembering to run a census is not a defense.  These tests ARE the
census, and they run on every ``make test``.

Two mechanically-decidable anti-patterns are banned outright:

1. ``expected_scored_ids(...)[1]`` — takes only the digest.  The producer then
   writes a correct ``canonical_task_universe_sha256`` while its rows stay on
   the wider COLLECTED set, so the cross-artifact SHA check that exists to catch
   universe drift sees a valid digest and passes.  Worse than no SHA at all.
2. ``_, sha = expected_scored_ids(...)`` — same failure, tuple-unpack spelling.

Plus a default-deny sweep: any analysis producer that reads episode summaries
straight off disk must either consume the canonical universe or be listed in
``EPISODE_READER_EXEMPT`` with a reason.  New scripts fail until triaged, which
is the point — the three leaks were all in scripts nobody thought to re-check.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ANALYSIS_DIRS = (REPO / "scripts" / "analysis",)
CANONICAL_NAMES = {"expected_scored_ids", "collected_task_ids", "restrict_to_scored"}

# --------------------------------------------------------------------------
# Two lists, deliberately kept apart.  Collapsing them is how "we'll get to it"
# turns into "that one is fine".
# --------------------------------------------------------------------------

# (1) The COLLECTED set is the correct denominator here.  Each entry says WHY.
EPISODE_READER_EXEMPT: dict[str, str] = {
    "lib/episode_rows.py": "the shared row loader itself; every caller owns its "
    "own universe decision downstream of it",
    "lib/run_registry.py": "maps runs to directories; reports COLLECTION "
    "progress, which is measured against the collection contract not the "
    "scoring set",
    "diag_pattern_match.py": "per-episode failure-pattern scan; a "
    "protocol-excluded episode still exhibits a failure mode worth counting",
    "write_digests.py": "/diag failure narratives over the COLLECTED set. Its "
    "embedded scored-RATE strings are a separate defect tracked as B-1913",
    "validate_run.py": "run completeness validator — comparing the collected "
    "set against the collection contract is precisely its job",
    "validate_run_manifest.py": "manifest integrity check over collected "
    "episodes, pre-scoring by construction",
    "validate_fire_manifest.py": "pre-fire manifest check; runs before any "
    "outcome exists, so no scored universe applies",
    "analyze_reason_diagnostics.py": "per-episode reason extraction for failure "
    "analysis; denominators are never quoted as scored rates",
    "analyze_cross_representation.py": "cross-mode diagnostic over collected "
    "episodes; produces no paper-facing rate",
    "compare_cross_run_same_condition.py": "cross-replicate identity check; "
    "compares two collections against each other, not against the scored set",
}

# (2) Paper-facing producers whose universe is still the COLLECTED set.
# B-1906 umbrella.  This list is a RATCHET: `test_triage_list_is_a_ratchet`
# fails if a new offender appears, so the族 cannot grow while it is being
# drained.  Remove an entry only when the script actually restricts.
UNIVERSE_TRIAGE_PENDING: dict[str, str] = {
    # B-1905 (aggregate_phantom_lift.py) was drained 2026-07-27: it now
    # intersects every per-comparison universe with the scored IDs. Removing it
    # from this list is REQUIRED by the ratchet, and is what keeps a later
    # regression from hiding behind a stale entry.
    "aggregate_cost_electricity.py": "cost aggregates over collected episodes",
    "aggregate_cross_site.py": "cross-site pooling over collected episodes",
    "aggregate_trajectory_covariates.py": "covariate table over collected set",
    "axis1_microbehavior.py": "scored count as denominator, rows unfiltered",
    "axis_effect_size.py": "scored count as denominator, rows unfiltered",
    "mechanism_per_task.py": "scored count as denominator, rows unfiltered",
    "hero_claim_bootstrap.py": "bootstraps a hero claim over the collected set",
    "generate_per_task_sr.py": "per-task SR table over the collected set",
    "collect_analysis_summary.py": "summary roll-up inherits collected rows",
    "layered_status.py": "status roll-up inherits collected rows",
    "compare_b0_b1.py": "B0/B1 contrast over collected episodes",
    "compare_pilot_t0_vs_paper_grade.py": "pilot contrast over collected set",
    "cross_mode_failure_taxonomy.py": "cross-mode taxonomy over collected set",
    "cross_mode_routable_deepdive.py": "routability deep-dive, collected set",
    "router_pareto_analysis.py": "router Pareto over collected set",
    "router_prior_baselines.py": "prior baselines over collected set",
    "router_archive_diagnostic.py": "archive diagnostic over collected set",
    "l2_partial_trajectory_auroc.py": "AUROC over collected set",
    "p1_archive_simulation.py": "archive simulation over collected set",
    "tier_a_id_perturbation.py": "perturbation replay over collected set",
    "b0_paired_idperturb_replay.py": "paired replay over collected set",
    "analyze_confidence_calibration.py": "calibration over collected set",
    "analyze_comment_selflink_loop.py": "loop analysis over collected set",
    "analyze_comment_selflink_loop_v2.py": "loop analysis over collected set",
    "analyze_noninteractive_click_earlystop.py": "collected set",
    "analyze_reddit_selflink_cycle.py": "collected set",
    "analyze_search_over_browse.py": "collected set",
    "figures/fig0d_taskpool_jaccard.py": "task-pool Jaccard over collected set",
    "figures/fig0e_category_mode_heatmap.py": "heatmap over collected set",
    "figures/fig0f_overlap_stacked_bar.py": "overlap bars over collected set",
    "figures/fig1ab_cascade_diamond.py": "cascade over collected set",
    "figures/fig2f_first_divergence.py": "divergence over collected set",
    "figures/fig3a_token_cost_intra_baseline.py": "token cost, collected set",
    "figures/fig3c_latency_per_step.py": "latency, collected set",
    "figures/fig3d_cost_sr_frontier.py": "cost-SR frontier, collected set",
}


def _py_files() -> list[Path]:
    out: list[Path] = []
    for d in ANALYSIS_DIRS:
        out.extend(p for p in d.rglob("*.py") if "__pycache__" not in str(p))
    return sorted(out)


def _rel(p: Path) -> str:
    return str(p.relative_to(ANALYSIS_DIRS[0]))


def _alias_map(tree: ast.AST) -> dict[str, str]:
    """local name -> canonical name, for aliased imports (`as _sct` etc.)."""
    alias: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name in CANONICAL_NAMES:
                    alias[a.asname or a.name] = a.name
    return alias


def _calls_to(tree: ast.AST, alias: dict[str, str], canonical: str) -> list[ast.Call]:
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        local = f.id if isinstance(f, ast.Name) else (
            f.attr if isinstance(f, ast.Attribute) else None
        )
        if local and (alias.get(local) or local) == canonical:
            out.append(node)
    return out


def test_no_sha_only_subscript_consumption() -> None:
    """Ban ``expected_scored_ids(...)[1]``: digest without restriction."""
    offenders: list[str] = []
    for path in _py_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover
            continue
        alias = _alias_map(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Subscript):
                continue
            inner = node.value
            if not isinstance(inner, ast.Call):
                continue
            f = inner.func
            local = f.id if isinstance(f, ast.Name) else (
                f.attr if isinstance(f, ast.Attribute) else None
            )
            if not local or (alias.get(local) or local) != "expected_scored_ids":
                continue
            idx = node.slice
            if isinstance(idx, ast.Constant) and idx.value == 1:
                offenders.append(f"{_rel(path)}:{node.lineno}")
    assert not offenders, (
        "expected_scored_ids(...)[1] takes the canonical digest but leaves the "
        "rows unrestricted — the artifact then carries a correct "
        "canonical_task_universe_sha256 over COLLECTED-set data, which is "
        "exactly what the cross-artifact SHA check cannot catch (B-1906). "
        "Use restrict_to_scored(), or consume [0] and filter.\n  "
        + "\n  ".join(offenders)
    )


def test_no_ids_discarding_tuple_unpack() -> None:
    """Ban ``_, sha = expected_scored_ids(...)``: same leak, other spelling."""
    offenders: list[str] = []
    for path in _py_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover
            continue
        alias = _alias_map(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            f = node.value.func
            local = f.id if isinstance(f, ast.Name) else (
                f.attr if isinstance(f, ast.Attribute) else None
            )
            if not local or (alias.get(local) or local) != "expected_scored_ids":
                continue
            tgt = node.targets[0]
            if not isinstance(tgt, ast.Tuple) or not tgt.elts:
                continue
            first = tgt.elts[0]
            if isinstance(first, ast.Name) and first.id == "_":
                offenders.append(f"{_rel(path)}:{node.lineno}")
    assert not offenders, (
        "discarding the ID set while keeping the digest reproduces B-1906.\n  "
        + "\n  ".join(offenders)
    )


# Reading episodes THROUGH a helper never names `*_summary_v2.json`, so a literal
# grep for the filename misses it.  Found 2026-07-28 when
# `aggregate_cross_mode_failure_signatures.py` reached every episode via
# `_discover_episodes` and sailed past this lint with reddit denominators of 205.
# Any new helper that hands out episode paths belongs in this set.
EPISODE_READER_MARKERS = ("_summary_v2.json", "_discover_episodes")


def _unrestricted_episode_readers() -> list[str]:
    """Analysis producers that read summaries without touching the universe.

    Detects both direct readers (the filename literal) and transitive ones (a
    helper that yields episode paths).
    """
    out: list[str] = []
    for path in _py_files():
        src = path.read_text(encoding="utf-8")
        if not any(marker in src for marker in EPISODE_READER_MARKERS):
            continue
        if any(name in src for name in CANONICAL_NAMES):
            continue
        out.append(_rel(path))
    return sorted(out)


def test_no_new_unrestricted_episode_readers() -> None:
    """Default-deny for anything not already triaged.

    A NEW script that reads episode summaries without restricting to the scored
    universe fails here.  That is the whole point: all three leaks codex found
    on 2026-07-27 (B-1901/1904/1905) were in scripts nobody thought to re-check,
    and a census that only runs when someone remembers is not a defense.
    """
    known = set(EPISODE_READER_EXEMPT) | set(UNIVERSE_TRIAGE_PENDING)
    new = [rel for rel in _unrestricted_episode_readers() if rel not in known]
    assert not new, (
        "new producer(s) read episode summaries but never reference the "
        "canonical scored universe, so their numbers silently include the "
        "AMENDMENT_08 protocol-excluded tasks (reddit 58/160).\n"
        "  → restrict via restrict_to_scored(), or\n"
        "  → add to EPISODE_READER_EXEMPT with a reason the COLLECTED set is "
        "the right denominator.\n  " + "\n  ".join(new)
    )


def test_triage_list_is_a_ratchet() -> None:
    """B-1906 backlog may shrink, never grow.

    An entry that no longer offends (because it now restricts) must be REMOVED
    from ``UNIVERSE_TRIAGE_PENDING``; leaving it there would let a later
    regression reintroduce the leak silently under cover of a stale entry.
    """
    offending = set(_unrestricted_episode_readers())
    resolved = sorted(set(UNIVERSE_TRIAGE_PENDING) - offending)
    assert not resolved, (
        "these scripts now consume the canonical universe — delete them from "
        f"UNIVERSE_TRIAGE_PENDING so the ratchet keeps holding: {resolved}"
    )


def test_no_stale_entries_in_either_list() -> None:
    """An entry for a deleted/renamed file hides the next real offender."""
    stale = sorted(
        rel
        for rel in (*EPISODE_READER_EXEMPT, *UNIVERSE_TRIAGE_PENDING)
        if not (ANALYSIS_DIRS[0] / rel).exists()
    )
    assert not stale, f"lists name missing files: {stale}"


def test_exempt_reasons_are_substantive() -> None:
    thin = [rel for rel, why in EPISODE_READER_EXEMPT.items() if len(why) < 30]
    assert not thin, f"exemption reasons too thin to audit: {thin}"


def test_lists_are_disjoint() -> None:
    """A script is either legitimately collected-set, or awaiting a fix."""
    both = sorted(set(EPISODE_READER_EXEMPT) & set(UNIVERSE_TRIAGE_PENDING))
    assert not both, f"claimed both exempt and pending: {both}"


@pytest.mark.parametrize("site,expected_excluded", [("reddit", [58, 160]), ("classifieds", [])])
def test_restrict_to_scored_digest_reflects_contents(site, expected_excluded) -> None:
    """The second digest must come from the rows, not from the canonical set."""
    import sys

    sys.path.insert(0, str(REPO / "scripts" / "analysis"))
    from lib.canonical_task_universe import (  # noqa: PLC0415
        expected_scored_ids,
        restrict_to_scored,
        task_id_set_sha256,
    )

    scored, canonical_sha = expected_scored_ids(site)

    # Complete container: both digests agree.
    rows = {t: {"success": False} for t in scored}
    kept, prov = restrict_to_scored(rows, site)
    assert prov["universe_complete"] is True
    assert prov["content_task_ids_sha256"] == canonical_sha
    assert prov["dropped_protocol_excluded"] == []

    # Collected container (scored + protocol-excluded): extras dropped, digests
    # still agree because what SURVIVED is exactly the scored set.
    rows_collected = dict(rows) | {t: {"success": True} for t in expected_excluded}
    kept, prov = restrict_to_scored(rows_collected, site)
    assert sorted(prov["dropped_protocol_excluded"]) == expected_excluded
    assert set(kept) == set(scored)
    assert prov["content_task_ids_sha256"] == canonical_sha

    # Incomplete container: canonical digest unchanged, content digest DIVERGES.
    # This is the property that makes B-1906 unexpressible.
    short = dict(rows)
    dropped = sorted(scored)[0]
    del short[dropped]
    kept, prov = restrict_to_scored(short, site)
    assert prov["universe_complete"] is False
    assert prov["canonical_task_universe_sha256"] == canonical_sha
    assert prov["content_task_ids_sha256"] != canonical_sha
    assert prov["content_task_ids_sha256"] == task_id_set_sha256(
        frozenset(scored) - {dropped}
    )
    assert prov["missing_from_universe"] == [dropped]


def test_restrict_to_scored_rejects_foreign_ids() -> None:
    """IDs that are neither scored nor protocol-excluded are contamination."""
    import sys

    sys.path.insert(0, str(REPO / "scripts" / "analysis"))
    from lib.canonical_task_universe import (  # noqa: PLC0415
        expected_scored_ids,
        restrict_to_scored,
    )

    scored, _ = expected_scored_ids("reddit")
    rows = {t: {"success": False} for t in scored}
    rows[9999] = {"success": True}
    with pytest.raises(ValueError, match="contamination"):
        restrict_to_scored(rows, "reddit", label="synthetic")


def test_restrict_to_scored_require_complete_fails_closed() -> None:
    import sys

    sys.path.insert(0, str(REPO / "scripts" / "analysis"))
    from lib.canonical_task_universe import (  # noqa: PLC0415
        expected_scored_ids,
        restrict_to_scored,
    )

    scored, _ = expected_scored_ids("reddit")
    short = {t: {"success": False} for t in sorted(scored)[:-1]}
    with pytest.raises(ValueError, match="scored tasks; missing"):
        restrict_to_scored(short, "reddit", require_complete=True)
