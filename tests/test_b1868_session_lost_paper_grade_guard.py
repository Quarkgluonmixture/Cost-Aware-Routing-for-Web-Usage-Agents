"""B-1868 (AMENDMENT_08) invariant tests — watchdog session-cleanup paper_grade guard.

Pre-B-1868 (R14849 2026-05-27): watchdog unconditionally deleted contaminated
session-lost episodes + RESUME_MISSING re-ran under different server state =
denominator surgery (sibling of B-1777 on error-retry path; same paper-killer
class).

Post-B-1868 (this patch): paper_grade=True PRESERVES contaminated episodes as
canonical denominator failures with structured covariate event for paper §3.5
disclosure + §4 GLMM analysis. Dev-mode (P79_PAPER_GRADE!=1) cleanup path
unchanged for non-paper-grade iteration.

Mirrors B-1777 / `_can_auto_retry` extraction pattern (tests/test_grl_audit_2026_05_20.py).

Critical invariant under test (user 2026-05-27 §5.4): metadata builders MUST
NOT emit generic `is_noise` field — risks downstream strict loader silently
excluding the episode = re-creates the denominator surgery this patch removes.
"""

import json
from pathlib import Path

from scripts.maintenance.experiment_watchdog import (
    _append_watchdog_audit_fallback,
    _build_session_lost_detected_metadata,
    _build_session_lost_event_key,
    _build_session_lost_preserved_metadata,
    _mark_episode_infra_covariate,
)


def _detected_kw():
    return dict(
        site="classifieds",
        condition_id="phase1_phantom_som_router_0",
        task_id=143,
        condition_key="phase1_phantom_som_router_0/classifieds_task_143",
        streak=3,
        run_id="B0_phantom_som_classifieds_20260527_R14849",
    )


def _preserved_kw():
    return dict(
        site="classifieds",
        condition_id="phase1_phantom_som_router_0",
        task_id=143,
        condition_key="phase1_phantom_som_router_0/classifieds_task_143",
        wave_size=3,
        wave_task_index=1,
        run_id="B0_phantom_som_classifieds_20260527_R14849",
    )


# ---------------------------------------------------------------------------
# Test 1 — preserved metadata: structural invariants under paper_grade
# ---------------------------------------------------------------------------

def test_b1868_preserved_metadata_has_required_non_exclusionary_keys():
    """Preserved metadata MUST carry the policy + covariate fields paper §3.5 /
    §4 GLMM aggregator binds to. If any of these go missing, downstream cannot
    distinguish a preserved infra-contaminated ep from an ordinary failure."""
    m = _build_session_lost_preserved_metadata(**_preserved_kw())
    required = {
        "site",
        "condition_id",
        "task_id",
        "condition_key",
        "session_loss",
        "auth_lost",
        "paper_grade_guard",
        "primary_denominator_policy",
        "exclusion_policy",
        "infra_covariate",
        "wave_size",
        "wave_task_index",
        "event_key",
    }
    missing = required - set(m.keys())
    assert not missing, f"preserved metadata missing required keys: {missing}"

    # Policy values must bind aggregator to non-exclusionary semantics.
    assert m["primary_denominator_policy"] == "preserve_as_observed_failure"
    assert m["exclusion_policy"] == "do_not_exclude_from_primary"
    assert m["paper_grade_guard"] == "B-1868"
    assert m["infra_covariate"] == "session_lost_preserved"


def test_b1868_preserved_metadata_forbids_is_noise():
    """CRITICAL (user 2026-05-27 §5.4): `is_noise=True` must NEVER appear in
    B-1868 covariate metadata. Pre-patch session-cleanup path emitted it via
    `clear_task_files extra_metadata`; some downstream consumer (strict loader,
    noise filter, GLMM exclusion path) could be tempted to treat it as an
    exclusion trigger and silently shrink the denominator. The whole point of
    B-1868 is to PRESERVE these episodes — `is_noise` would re-create the
    denominator surgery this patch removes.

    If you're adding a noise marker for diagnostic / GLMM use, name it
    something specific and non-exclusionary like `diagnostic_covariate` or
    `infra_covariate` (the latter is already in this metadata)."""
    m = _build_session_lost_preserved_metadata(**_preserved_kw())
    assert "is_noise" not in m, (
        "B-1868 preserved metadata MUST NOT contain `is_noise` — risks "
        "downstream strict loader silently excluding the preserved episode. "
        "Use specific non-exclusionary field (e.g. `infra_covariate`) instead."
    )


def test_b1868_detected_metadata_forbids_is_noise():
    """Same invariant applies to detection-time event (line ~2219)."""
    m = _build_session_lost_detected_metadata(**_detected_kw())
    assert "is_noise" not in m, (
        "B-1868 detected metadata MUST NOT contain `is_noise` — same reasoning "
        "as preserved metadata (see test docstring)."
    )


# ---------------------------------------------------------------------------
# Test 2 — event_key idempotency for aggregator dedup
# ---------------------------------------------------------------------------

def test_b1868_event_key_deterministic_per_phase():
    """Same (run_id, condition_id, task_id, condition_key, phase) tuple must
    produce identical event_key across calls. Downstream aggregator dedups on
    event_key so a retried covariate log (e.g. watchdog restart, fallback audit
    jsonl replay) does NOT double-count the same episode in paper §4 GLMM."""
    k1 = _build_session_lost_event_key(
        run_id="r1", condition_id="c1", task_id=143,
        condition_key="c1/cls_task_143", phase="preserved",
    )
    k2 = _build_session_lost_event_key(
        run_id="r1", condition_id="c1", task_id=143,
        condition_key="c1/cls_task_143", phase="preserved",
    )
    assert k1 == k2, "event_key must be deterministic across calls"
    assert "B-1868" in k1, "event_key must carry the B-### prefix for grep-ability"


def test_b1868_event_key_distinct_per_phase():
    """Detection-time and restore-time events for the SAME episode must have
    distinct event_keys so they don't dedup-collapse into a single covariate
    row — both phases are independently relevant to paper §4 GLMM (detection
    timing, restore latency)."""
    detected = _build_session_lost_event_key(
        run_id="r1", condition_id="c1", task_id=143,
        condition_key="c1/cls_task_143", phase="detected",
    )
    preserved = _build_session_lost_event_key(
        run_id="r1", condition_id="c1", task_id=143,
        condition_key="c1/cls_task_143", phase="preserved",
    )
    assert detected != preserved, "phase suffix must produce distinct keys"


def test_b1868_event_key_embedded_in_preserved_metadata():
    """Preserved-metadata's `event_key` field must equal the builder output for
    the corresponding tuple — ensures inline `cov_metadata['event_key']` is
    NOT silently drifted from the canonical builder."""
    kw = _preserved_kw()
    m = _build_session_lost_preserved_metadata(**kw)
    expected = _build_session_lost_event_key(
        run_id=kw["run_id"],
        condition_id=kw["condition_id"],
        task_id=kw["task_id"],
        condition_key=kw["condition_key"],
        phase="preserved",
    )
    assert m["event_key"] == expected


# ---------------------------------------------------------------------------
# Test 3 — wave context preserved for condition-level integrity threshold
# ---------------------------------------------------------------------------

def test_b1868_preserved_metadata_wave_context_passes_through():
    """Condition-level integrity threshold (per PROTOCOL_NOTE_01 catalog entry:
    >2% / >5 tasks → `infra_contaminated_high`; >10-20% → archive non-canonical)
    requires aggregator to see `wave_size` per preserved episode + the per-wave
    index. Both must round-trip through the builder verbatim."""
    m = _build_session_lost_preserved_metadata(
        site="classifieds", condition_id="c1", task_id=143,
        condition_key="c1/cls_task_143",
        wave_size=7, wave_task_index=3,
        run_id="r1",
    )
    assert m["wave_size"] == 7
    assert m["wave_task_index"] == 3


# ---------------------------------------------------------------------------
# Test 4 — inline call-site source-grep (Claude P1-4-A forward guard)
# ---------------------------------------------------------------------------

def test_b1868_inline_call_sites_present():
    """Forward-guard: future refactor / merge-conflict resolution must NOT
    bypass the metadata helpers (e.g. re-inline a dict literal with `is_noise=
    True` or guard the paper_grade branch behind `if False:`). Pure metadata
    tests above protect the helpers themselves; this test protects the
    helpers' callsites. Cheap source-grep — fails the moment inline code
    stops calling the builders.

    Driver: codex Mode B + gemini Mode C cross-AI audit 2026-05-27 explicitly
    flagged "Tests don't cover behavioral coverage; bypass is possible". This
    test closes that gap without needing heavy subprocess integration.
    """
    src_path = Path(__file__).resolve().parents[1] / "scripts" / "maintenance" / "experiment_watchdog.py"
    src = src_path.read_text()
    assert "if _watchdog_paper_grade:" in src, (
        "paper_grade branch missing — preserve vs clean routing broken"
    )
    assert "_build_session_lost_preserved_metadata(" in src, (
        "restore-time inline code stopped calling preserved-metadata helper"
    )
    assert "_build_session_lost_detected_metadata(" in src, (
        "detection-time inline code stopped calling detected-metadata helper"
    )
    assert "session_lost_paper_grade_preserved" in src, (
        "event_type literal missing from inline restore-time emission"
    )
    assert "session_lost_contaminated_detected" in src, (
        "event_type literal missing from inline detection-time emission"
    )
    assert "_mark_episode_infra_covariate(" in src, (
        "canonical-summary infra_covariates marker (P1-3-B*) not invoked"
    )
    assert "_append_watchdog_audit_fallback(" in src, (
        "shared fallback helper (P1-1-B / P1-2-B*) not invoked"
    )


# ---------------------------------------------------------------------------
# Test 5 — fallback audit jsonl atomic write + flock pattern
# ---------------------------------------------------------------------------

def test_b1868_fallback_audit_jsonl_round_trips(tmp_path: Path):
    """B-1868 §5.1 fallback helper round-trip: write entry → file exists →
    contents parse as valid jsonl → re-write appends (no overwrite)."""
    cond_dir = tmp_path / "cond_x"
    entry_1 = {
        "wallclock_ts": "2026-05-27T19:00:00+00:00",
        "exception": "OSError: disk full",
        "intended_event_type": "session_lost_paper_grade_preserved",
        "metadata": {
            "site": "classifieds", "task_id": 143,
            "event_key": "B-1868:r1:c1:143:k1:session_lost_preserved",
        },
    }
    ok_1 = _append_watchdog_audit_fallback(cond_dir, entry_1)
    assert ok_1 is True
    fb_path = cond_dir / "watchdog_session_preserved_failures.jsonl"
    assert fb_path.exists()
    lines_1 = fb_path.read_text().strip().split("\n")
    assert len(lines_1) == 1
    parsed_1 = json.loads(lines_1[0])
    assert parsed_1["intended_event_type"] == "session_lost_paper_grade_preserved"
    assert parsed_1["metadata"]["task_id"] == 143

    # Second call appends, does not overwrite.
    entry_2 = dict(entry_1)
    entry_2["metadata"] = dict(entry_1["metadata"])
    entry_2["metadata"]["task_id"] = 144
    entry_2["metadata"]["event_key"] = "B-1868:r1:c1:144:k1:session_lost_preserved"
    ok_2 = _append_watchdog_audit_fallback(cond_dir, entry_2)
    assert ok_2 is True
    lines_2 = fb_path.read_text().strip().split("\n")
    assert len(lines_2) == 2
    parsed_2 = json.loads(lines_2[1])
    assert parsed_2["metadata"]["task_id"] == 144


# ---------------------------------------------------------------------------
# Test 6 — _mark_episode_infra_covariate idempotency + atomic write
# ---------------------------------------------------------------------------

def test_b1868_mark_episode_infra_covariate_idempotent(tmp_path: Path):
    """B-1868 P1-3-B* canonical-summary marker is the dual-path defense for
    paper §4 GLMM covariate visibility. Must be idempotent (re-emit on
    watchdog restart safe) and atomic (no torn write on crash mid-rename)."""
    cond_dir = tmp_path / "cond_x"
    ep_dir = cond_dir / "episodes"
    ep_dir.mkdir(parents=True)
    summary_path = ep_dir / "classifieds_task_143_summary_v2.json"
    initial = {"task_id": 143, "site": "classifieds", "success": False, "score": 0.0}
    summary_path.write_text(json.dumps(initial))

    # First call: appends marker, persists.
    ok_1 = _mark_episode_infra_covariate(
        condition_dir=cond_dir, task_id=143, site="classifieds",
        covariate="session_lost_preserved",
    )
    assert ok_1 is True
    written = json.loads(summary_path.read_text())
    assert "infra_covariates" in written
    assert written["infra_covariates"] == ["session_lost_preserved"]

    # Second call: idempotent — already present, no duplication.
    ok_2 = _mark_episode_infra_covariate(
        condition_dir=cond_dir, task_id=143, site="classifieds",
        covariate="session_lost_preserved",
    )
    assert ok_2 is True
    written_2 = json.loads(summary_path.read_text())
    assert written_2["infra_covariates"] == ["session_lost_preserved"]

    # Original fields preserved.
    assert written_2["task_id"] == 143
    assert written_2["site"] == "classifieds"
    assert written_2["success"] is False


def test_b1868_mark_episode_infra_covariate_no_summary_file(tmp_path: Path):
    """If summary file does not exist (race with runner mid-write or condition
    aborted before summary land), return False best-effort. Event-log + fallback
    are the other two redundancy channels per catalog L8156-8160."""
    cond_dir = tmp_path / "cond_x"
    (cond_dir / "episodes").mkdir(parents=True)
    ok = _mark_episode_infra_covariate(
        condition_dir=cond_dir, task_id=999, site="classifieds",
        covariate="session_lost_preserved",
    )
    assert ok is False


# ---------------------------------------------------------------------------
# Test 7 — aggregator dual-path covariate emission (P0-1 + P1-3 round-trip)
# ---------------------------------------------------------------------------

def test_b1868_aggregator_reads_summary_infra_covariates(tmp_path: Path):
    """Codex Mode B Finding 5 / P1-3-B* dual-path defense: aggregator MUST
    populate `session_lost_preserved=True` from summary `infra_covariates` list
    even when trajectory_events.jsonl has NO session_lost_paper_grade_preserved
    event (e.g. logger failed AND fallback file also missing — schema-level
    dual-path is the last line of defense for paper §4 GLMM covariate
    visibility)."""
    from scripts.analysis.aggregate_trajectory_covariates import compute_episode_covariates

    cond_dir = tmp_path / "phase1_phantom_som_router_0"
    (cond_dir / "episodes").mkdir(parents=True)
    # No trajectory_events.jsonl — simulate full logger + fallback failure.
    cs = {
        "episode_summaries": [
            {"task_id": 143, "site": "classifieds", "success": False,
             "score": 0.0, "infra_covariates": ["session_lost_preserved"]},
            {"task_id": 144, "site": "classifieds", "success": False,
             "score": 0.0, "infra_covariates": []},
        ]
    }
    (cond_dir / "condition_summary_v2.json").write_text(json.dumps(cs))

    rows = compute_episode_covariates(cond_dir)
    assert len(rows) == 2
    row_143 = next(r for r in rows if r["task_id"] == 143)
    row_144 = next(r for r in rows if r["task_id"] == 144)
    assert row_143["session_lost_preserved"] is True, (
        "summary infra_covariates dual-path NOT consumed — Mode B Finding 5 "
        "regression: aggregator only reads event-log, not canonical summary"
    )
    assert row_144["session_lost_preserved"] is False


def test_b1868_aggregator_reads_event_log(tmp_path: Path):
    """Aggregator MUST populate `session_lost_preserved=True` from event-log
    when summary lacks the marker (e.g. older summary written before
    watchdog patch but event landed)."""
    from scripts.analysis.aggregate_trajectory_covariates import compute_episode_covariates

    cond_dir = tmp_path / "phase1_phantom_som_router_0"
    (cond_dir / "episodes").mkdir(parents=True)
    cs = {
        "episode_summaries": [
            # No infra_covariates field at all → defaults to []
            {"task_id": 143, "site": "classifieds", "success": False, "score": 0.0},
        ]
    }
    (cond_dir / "condition_summary_v2.json").write_text(json.dumps(cs))
    # Event log carries the preserved event
    ev = {
        "event_type": "session_lost_paper_grade_preserved",
        "task_index": 143,
        "wallclock_ts": "2026-05-27T19:00:00+00:00",
        "metadata": {
            "site": "classifieds", "task_id": 143,
            "wave_size": 3, "wave_task_index": 1,
            "event_key": "B-1868:r1:c1:143:k1:session_lost_preserved",
            "infra_covariate": "session_lost_preserved",
        },
    }
    (cond_dir / "trajectory_events.jsonl").write_text(json.dumps(ev) + "\n")

    rows = compute_episode_covariates(cond_dir)
    row_143 = next(r for r in rows if r["task_id"] == 143)
    assert row_143["session_lost_preserved"] is True
    assert row_143["session_lost_preserved_wave_size"] == 3


def test_b1868_aggregator_event_key_dedup(tmp_path: Path):
    """Codex Mode B Finding 1 / P0-2-B*: aggregator MUST dedup by event_key
    so watchdog restart replay does NOT double-count `prior_event_count` or
    `n_task_events` or session-loss covariates. Test: same event_key emitted
    twice → aggregator sees one row's n_task_events == 1 not 2."""
    from scripts.analysis.aggregate_trajectory_covariates import compute_episode_covariates

    cond_dir = tmp_path / "phase1_phantom_som_router_0"
    (cond_dir / "episodes").mkdir(parents=True)
    cs = {
        "episode_summaries": [
            {"task_id": 143, "site": "classifieds", "success": False, "score": 0.0},
        ]
    }
    (cond_dir / "condition_summary_v2.json").write_text(json.dumps(cs))
    ev_1 = {
        "event_type": "session_lost_paper_grade_preserved",
        "task_index": 143,
        "wallclock_ts": "2026-05-27T19:00:00+00:00",
        "metadata": {
            "site": "classifieds", "task_id": 143, "wave_size": 3,
            "event_key": "B-1868:r1:c1:143:k1:session_lost_preserved",
        },
    }
    # Same event written twice (simulating watchdog restart replay)
    (cond_dir / "trajectory_events.jsonl").write_text(
        json.dumps(ev_1) + "\n" + json.dumps(ev_1) + "\n"
    )

    rows = compute_episode_covariates(cond_dir)
    row_143 = next(r for r in rows if r["task_id"] == 143)
    assert row_143["n_task_events"] == 1, (
        "event_key dedup broken — duplicate events double-counted"
    )
    assert row_143["session_lost_preserved"] is True
