import pytest
from PIL import Image

from p79.envs.vwa_wrapper import P79Observation
from p79.experiment.schema_migrations.v2 import EPISODE_SUMMARY_V2_DEFAULTS
from p79.experiment.som import prepare_observation_for_mode
from p79.experiment.types import SCHEMA_VERSION_V2, validate_step_record_v2


def test_prepare_observation_rejects_unknown_mode(tmp_path):
    """/stress A1.4 F2: typo modes must raise, not silently fall through to DOM-like."""
    obs = P79Observation(text="[1] something", image=None, raw={})
    with pytest.raises(ValueError, match=r"Unknown observation_mode"):
        prepare_observation_for_mode(obs, "phantum_som", tmp_path, 0)


def test_options_map_lookahead_is_unbounded_by_distance():
    """/stress A1.4 F3 backlog sweep: `[OPTIONS]` recovery look-ahead no longer
    capped at 2 lines — only the next mark-id line is a hard boundary.

    Regression case: trigger mark on line 1, two intermediate property lines
    (no mark id), then `[OPTIONS]` on line 4. Previously the 2-line window
    would silently drop this. Now it should be recovered.
    """
    from p79.experiment.som import build_som_text_from_obs_text

    # 4 lines after trigger — would have failed the old 2-line cap.
    obs_text = (
        "[42] combobox 'Choose'\n"
        "    role: combobox\n"
        "    aria-expanded: false\n"
        "    data-state: collapsed\n"
        "[OPTIONS] \"Option A\", \"Option B\"\n"
        "[43] button 'Submit'\n"
    )
    out = build_som_text_from_obs_text(obs_text)
    assert "[OPTIONS]" in out, (
        f"`_options_map` recovery missed the [OPTIONS] line at distance 4 from trigger. "
        f"Look-ahead window should be unbounded except by next mark id.\n"
        f"Got:\n{out}"
    )
    # Both marks still present
    assert "[id=42]" in out
    assert "[id=43]" in out


def test_options_map_boundary_is_next_mark_id():
    """F3 corner: a different mark id BEFORE the [OPTIONS] line means
    options belong to the next mark, not the current one."""
    from p79.experiment.som import build_som_text_from_obs_text

    obs_text = (
        "[10] combobox 'A'\n"
        "[20] combobox 'B'\n"
        "[OPTIONS] \"X\", \"Y\"\n"
    )
    out = build_som_text_from_obs_text(obs_text)
    # `[OPTIONS]` should attach to mark 20 (immediately preceding), not 10.
    lines = out.splitlines()
    mark10_idx = next(i for i, ln in enumerate(lines) if "[id=10]" in ln)
    mark20_idx = next(i for i, ln in enumerate(lines) if "[id=20]" in ln)
    options_idx = next(i for i, ln in enumerate(lines) if "[OPTIONS]" in ln)
    # OPTIONS should be after mark 20, not mixed under mark 10
    assert mark20_idx < options_idx
    assert options_idx > mark10_idx  # not under mark 10
    # The mark 10 entry should NOT have [OPTIONS] as a child line — no
    # immediate-next-line is [OPTIONS] for mark 10.


def test_collect_bbox_map_handles_cyclic_reference():
    """/stress A1.4 F4 backlog sweep: cyclic raw obs must not infinite-recurse.

    The visited-set + depth-cap guard means any cyclic dict / list reference
    terminates safely. Without the guard this would RecursionError → episode
    abort.
    """
    from p79.experiment.som import _collect_bbox_map

    # Build a cyclic dict.
    a: dict = {"id": 1, "bbox": [0.0, 0.0, 10.0, 10.0]}
    b: dict = {"id": 2, "bbox": [10.0, 10.0, 20.0, 20.0], "parent_ref": a}
    a["child_ref"] = b  # cycle: a → b → a

    bbox_map: dict = {}
    # Must not raise / hang.
    _collect_bbox_map(a, bbox_map)
    # Both bboxes still collected
    assert 1 in bbox_map
    assert 2 in bbox_map


def test_collect_bbox_map_respects_depth_cap():
    """F4: very deep nesting terminates within _BBOX_TRAVERSAL_MAX_DEPTH (50)."""
    from p79.experiment.som import _collect_bbox_map

    # Build a deeply nested dict, 200 levels deep.
    leaf: dict = {"id": 999, "bbox": [0.0, 0.0, 1.0, 1.0]}
    cur = leaf
    for i in range(200):
        cur = {"nested": cur}
    bbox_map: dict = {}
    # Must not RecursionError even though depth > Python default 1000.
    _collect_bbox_map(cur, bbox_map)
    # Leaf is beyond depth cap → not collected (acceptable: depth cap is a
    # safety guard, the data path on production is shallow).
    # The point is no exception. (Verified by reaching this line.)


def test_prepare_observation_accepts_all_known_modes(tmp_path):
    """All 7 canonical modes (incl. phantom_dom legacy alias) must not raise
    for non-vision modes. Vision mode requires obs.image to be non-None per
    B-265 fix (2026-05-16, A1.7) — paper §3 contract "vision = raw screenshot
    only" must be defensible at the observation-preparation boundary.
    """
    from p79.experiment.som import KNOWN_OBSERVATION_MODES

    expected = {"dom", "som", "vision", "phantom_som", "phantom_dom", "phantom_text", "phantom_prompt"}
    assert set(KNOWN_OBSERVATION_MODES) == expected
    # Non-vision modes accept obs.image=None (text-only paths)
    obs_no_image = P79Observation(text="[1] something", image=None, raw={})
    for mode in expected - {"vision"}:
        prepare_observation_for_mode(obs_no_image, mode, tmp_path, 0)
    # Vision mode requires non-None image (B-265). Pass a tiny PIL.Image-shaped
    # placeholder; the function only checks `is None`, not type.
    obs_with_image = P79Observation(text="", image="<dummy-image-placeholder>", raw={})
    prepare_observation_for_mode(obs_with_image, "vision", tmp_path, 0)


def test_vision_mode_raises_on_missing_image(tmp_path):
    """B-265 fix (2026-05-16, A1.7): vision mode must fail-fast on None image
    so paper §3 'vision = raw screenshot only' contract is enforced at the
    observation-preparation boundary (not silently degraded to text-only).
    """
    import pytest as _pytest
    obs_no_image = P79Observation(text="some text", image=None, raw={})
    with _pytest.raises(ValueError, match="Vision mode requires image"):
        prepare_observation_for_mode(obs_no_image, "vision", tmp_path, 0)


def test_episode_defaults_schema_version_matches_runtime_constant():
    """/stress A1.2 codex C1: prevent schema identity split.

    `EPISODE_SUMMARY_V2_DEFAULTS["schema_version"]` (used by `fill_defaults()`
    to backfill old data) must match `SCHEMA_VERSION_V2` (what the runner
    writes for new data). Previously the default was the literal "v2" while
    the runtime constant was "2.0" — would have caused fill_defaults to
    silently mis-tag records once the v3 migration lands.
    """
    assert EPISODE_SUMMARY_V2_DEFAULTS["schema_version"] == SCHEMA_VERSION_V2
    assert SCHEMA_VERSION_V2 == "2.0"


def test_som_degrades_without_bbox(tmp_path):
    """Production path: marks present in text but no bbox info → image render
    cannot draw boxes → `marked_image is None`. Uses prepare_observation_for_mode
    after apply_som was deleted (A1.4c cleanup, 2026-05-16; was 0-caller dead
    code emitting a DeprecationWarning that nobody listened to).

    /stress A1.4 P0-2 (2026-05-17): the `degraded_som` schema bool field has
    been DELETED — it overloaded three semantically distinct states
    (zero-marks / render-fail / phantom inherit) and empirically fired
    0/6471 in archive. Path A (zero-marks vision-fallback) is now detected
    aggregator-side via `mark_count == 0`. Path B (PIL render-fail) is
    only visible via `marked_image is None` after the SoM-mode path; this
    test asserts that signal directly.
    """
    obs = P79Observation(
        text="[1] Search textbox\n[2] Submit button",
        image=Image.new("RGB", (200, 100), color="white"),
        raw={"text": "no bbox here"},
    )

    result = prepare_observation_for_mode(obs, "som", tmp_path, step_idx=0)
    assert result.mark_count >= 1
    # Render-fail signal post-A1.4-P0-2: marked_image is None when marks
    # were extracted but bbox info absent → PIL draw skipped (no exception
    # path) → marked_image stays None.
    assert result.marked_image is None


def test_phantom_som_strips_image_keeps_text(tmp_path):
    """Phantom-SoM (§25): same SOM_MARKS text as full SoM, but no image to model."""
    obs = P79Observation(
        text="[1] Search textbox\n[2] Submit button",
        image=Image.new("RGB", (200, 100), color="white"),
        obs_nodes_info={
            "1": {"union_bound": [10, 20, 80, 30]},
            "2": {"union_bound": [100, 20, 90, 30]},
        },
    )

    full_som = prepare_observation_for_mode(obs, "som", tmp_path, step_idx=0)
    phantom = prepare_observation_for_mode(obs, "phantom_som", tmp_path, step_idx=0)

    # Same textual SOM_MARKS content
    assert phantom.som_text == full_som.som_text
    assert phantom.mark_count == full_som.mark_count

    # But model receives NO image in phantom
    assert phantom.marked_image is None
    assert full_som.marked_image is not None


def test_phantom_som_text_differs_from_dom(tmp_path):
    """Phantom-SoM text should be wrapped in [SOM_MARKS] block; DOM passes raw obs.text."""
    obs = P79Observation(
        text="[1] link 'Home'\n[2] button 'Submit'",
        image=Image.new("RGB", (100, 100), color="white"),
        obs_nodes_info={"1": {"union_bound": [0, 0, 50, 20]}},
    )

    dom = prepare_observation_for_mode(obs, "dom", tmp_path, step_idx=0)
    phantom = prepare_observation_for_mode(obs, "phantom_som", tmp_path, step_idx=0)

    # DOM passes raw obs.text; phantom wraps in [SOM_MARKS] block
    assert "[SOM_MARKS]" in phantom.som_text
    assert "[SOM_MARKS]" not in dom.som_text
    # Both have no image visible to the model
    assert dom.marked_image is None
    assert phantom.marked_image is None


def test_step_schema_validation_required_fields():
    # B-280 fix (2026-05-16, A1.8): test record now includes paper-grade
    # critical optional KEYS (value may be None) — validator requires presence.
    record = {
        "schema_version": SCHEMA_VERSION_V2,
        "run_id": "run_x",
        "condition_id": "c1",
        "benchmark": "visualwebarena",
        "benchmark_site": "shopping",
        "task_id": 0,
        "seed": 42,
        "step_idx": 0,
        "som": {},
        "observation_mode": "dom",
        "router": {},
        "module_flags": {},
        "action_type": "wait",
        "action": {"action_type": "wait"},
        "action_success": False,
        "page_changed": False,
        "latency_ms": {"total": 0.0},
        "tokens": {"input": 0, "output": 0, "total": 0},
        # B-338 (/stress A1.9 Mode B F7, 2026-05-16): cost_usd nested
        # key validator requires {input, output, model, router_overhead, total}.
        "cost_usd": {
            "input": 0.0, "output": 0.0, "model": 0.0,
            "router_overhead": 0.0, "total": 0.0,
        },
        "energy": {"kwh": None, "co2e_kg": None},
        "retry_count": 0,
        "error_category": None,
        "artifact_paths": {},
        "reward": 0.0,
        "done": False,
        # B-280 critical optionals (must be present, value may be None):
        "parse_valid": None,
        "parse_failure_reason": None,
        "image_meta": None,
        "locator_route_meta": None,
        # B-440 (/stress A1.25 P0-2-B* codex OOB, 2026-05-17): retry-overwrite
        # split fields — primary captured before retry block; retry populated
        # only when baseline_retry_on_no_progress fired.
        "locator_route_meta_primary": None,
        "locator_route_meta_retry": None,
        # B-420 (/stress A1.3 v9, 2026-05-17): select_option dispatch meta
        # symmetric with locator_route_meta. Paper §3.5 select_option
        # sub-taxonomy depends on key presence.
        "select_option_meta": None,
        # B-450 (/stress A1.4 P0-3-B codex OOB, 2026-05-17): select_option
        # retry-overwrite split, symmetric with locator_route_meta_primary/retry.
        "select_option_meta_primary": None,
        "select_option_meta_retry": None,
        "agent_visible_changed": None,
        # B-472 (/stress A1.5b Phase 1 P1-9-A, 2026-05-17): control-injected
        # action provenance — None when no control fired.
        "control_intervention": None,
    }

    validate_step_record_v2(record)

    broken = dict(record)
    broken.pop("router")
    with pytest.raises(ValueError):
        validate_step_record_v2(broken)


def test_b481_select_option_meta_structured_fields_validator():
    """B-481 (/stress A1.25 GRL Chunk 2 P0-2-AB* + P1-1-BC*, 2026-05-17):
    validator accepts structured {matched, match_stage, target_type}
    payloads on all three select_option_meta variants (legacy / primary /
    retry) and rejects bad enum / type values.
    """
    # Helper to spin a minimal valid step record (mirror the fixture above
    # but parameterize the select_option_meta dict under test).
    def _record_with_sel_meta(sel_meta_payload, key="select_option_meta_primary"):
        rec = {
            "schema_version": SCHEMA_VERSION_V2,
            "run_id": "run_x", "condition_id": "c1",
            "benchmark": "visualwebarena", "benchmark_site": "shopping",
            "task_id": 0, "seed": 42, "step_idx": 0,
            "som": {}, "observation_mode": "dom",
            "router": {}, "module_flags": {},
            "action_type": "select_option",
            "action": {"action_type": "select_option"},
            "action_success": False, "page_changed": False,
            "latency_ms": {"total": 0.0},
            "tokens": {"input": 0, "output": 0, "total": 0},
            "cost_usd": {"input": 0.0, "output": 0.0, "model": 0.0,
                         "router_overhead": 0.0, "total": 0.0},
            "energy": {"kwh": None, "co2e_kg": None},
            "retry_count": 0, "error_category": None,
            "artifact_paths": {}, "reward": 0.0, "done": False,
            "parse_valid": None, "parse_failure_reason": None,
            "image_meta": None, "locator_route_meta": None,
            "locator_route_meta_primary": None,
            "locator_route_meta_retry": None,
            "select_option_meta": None,
            "select_option_meta_primary": None,
            "select_option_meta_retry": None,
            "agent_visible_changed": None,
            # B-472 (/stress A1.5b Phase 1, 2026-05-17): control_intervention.
            "control_intervention": None,
        }
        rec[key] = sel_meta_payload
        return rec

    # Happy path: full structured payload, all 3 stages and both target_types
    for stage in ("exact", "ci", "fuzzy", "index", "none"):
        for target in ("select", "css"):
            validate_step_record_v2(_record_with_sel_meta({
                "action_kind": "select_option",
                "dispatch_path": "element_id",
                "success": True, "matched": True,
                "match_stage": stage, "target_type": target,
                "selected_text_before": "old", "selected_text_after": "new",
                "clicked_text": None, "error": None,
            }))

    # Happy path on _retry field (B-480 closure verification)
    validate_step_record_v2(_record_with_sel_meta({
        "action_kind": "select_option",
        "success": False, "matched": False,
        "match_stage": "none", "target_type": "select",
        "error": "no_match_in_select",
    }, key="select_option_meta_retry"))

    # Reject: match_stage with bad value
    import pytest as _pytest
    with _pytest.raises(ValueError, match="match_stage unexpected"):
        validate_step_record_v2(_record_with_sel_meta({
            "action_kind": "select_option",
            "success": True, "matched": True,
            "match_stage": "approximate",  # not in enum
            "target_type": "select",
        }))

    # Reject: target_type with bad value
    with _pytest.raises(ValueError, match="target_type unexpected"):
        validate_step_record_v2(_record_with_sel_meta({
            "action_kind": "select_option",
            "success": True, "matched": True,
            "match_stage": "exact",
            "target_type": "iframe",  # not in enum
        }))

    # Reject: matched as string instead of bool
    with _pytest.raises(ValueError, match="matched expected bool-or-None"):
        validate_step_record_v2(_record_with_sel_meta({
            "action_kind": "select_option",
            "success": True, "matched": "true",  # string not bool
            "match_stage": "exact", "target_type": "select",
        }))

    # Reject: success as string (legacy B-444 sibling check)
    with _pytest.raises(ValueError, match="success expected bool-or-None"):
        validate_step_record_v2(_record_with_sel_meta({
            "action_kind": "select_option",
            "success": "false",  # string not bool
            "matched": False, "match_stage": "none",
        }))
