"""Invariant tests for /stress A1.3 v8 Commit F fixes (B-156 ~ B-161).

Each test pins one specific contract that, once it regresses, would
break paper §3 evidence layer, multi-tab handling, or security.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# B-156 — locator-route telemetry threaded into StepRecordV2 via info dict
# ---------------------------------------------------------------------------


def test_step_record_v2_has_locator_route_meta_field():
    """``StepRecordV2.locator_route_meta`` field is the paper §3 evidence-layer
    handle for Cluster 1 (B-01/02/33) ON_TARGET-rate audit. Without it, the
    Tier 10 94.4% off-target → >80% ON_TARGET goal cannot be verified from
    JSONL."""
    src = (REPO_ROOT / "p79/experiment/types.py").read_text(encoding="utf-8")
    assert re.search(
        r"locator_route_meta:\s*Optional\[Dict\[str,\s*Any\]\]\s*=\s*None",
        src,
    ), "StepRecordV2 missing `locator_route_meta` Optional[Dict] field"


def test_vwa_wrapper_emits_locator_route_meta_in_info():
    """vwa_wrapper.step() must surface ``_locator_route_meta`` into the
    returned ``info`` dict so the runner can persist it."""
    src = (REPO_ROOT / "p79/envs/vwa_wrapper.py").read_text(encoding="utf-8")
    assert 'info["locator_route_meta"] = _locator_route_meta' in src, (
        "vwa_wrapper.step() must stamp _locator_route_meta into info dict"
    )
    # Click + type branches populate _locator_route_meta with action_kind
    assert '_locator_route_meta["action_kind"] = "click"' in src
    assert '_locator_route_meta["action_kind"] = "type"' in src


def test_runner_persists_locator_route_meta_into_step_record():
    """Runner step_record must read `next_info.get("locator_route_meta")`."""
    src = (REPO_ROOT / "p79/experiment/runner/main.py").read_text(encoding="utf-8")
    assert 'step_record["locator_route_meta"] = next_info.get("locator_route_meta")' in src, (
        "Runner must persist locator_route_meta from env info into step_record"
    )


# ---------------------------------------------------------------------------
# B-157 — locator-route new-tab handling
# ---------------------------------------------------------------------------


def test_locator_route_click_captures_tabs_before_count():
    """vwa_wrapper.step() click branch must snapshot context.pages length
    BEFORE locator-route dispatch so it can mimic VWA's tab-switch logic
    after a successful click that opened ``target=_blank``/``window.open``."""
    src = (REPO_ROOT / "p79/envs/vwa_wrapper.py").read_text(encoding="utf-8")
    # The snapshot uses _num_tabs_before
    assert "_num_tabs_before" in src, (
        "B-157 missing _num_tabs_before capture before locator-route click"
    )
    # The success branch switches to last opened page when count grew
    assert re.search(
        r"len\(_pages_now\)\s*>\s*_num_tabs_before",
        src,
    ), "B-157 missing new-tab detection (len(pages) > num_tabs_before)"
    assert "self._env.page = _new_page" in src
    assert "bring_to_front()" in src


# ---------------------------------------------------------------------------
# B-158 — dialog handler registered at BrowserContext level
# ---------------------------------------------------------------------------


def test_dialog_handler_attached_at_context_level():
    """B-158 swapped page-level dialog registration to context-level so every
    new Page (including window.open / target=_blank tabs from B-157) inherits
    the auto-handler. Identity check on context (not page) prevents listener
    accumulation across episode resets."""
    src = (REPO_ROOT / "p79/envs/vwa_wrapper.py").read_text(encoding="utf-8")
    assert "_dialog_registered_context" in src, (
        "B-158 missing _dialog_registered_context tracker"
    )
    assert 'ctx.on("page"' in src, (
        "B-158 missing ctx.on(\"page\", ...) future-page hookup"
    )
    # Page-level tracker must be gone from actual code (comment refs OK).
    # Strip docstrings/comments before checking — pre-fix attribute name was
    # ``self._dialog_registered_page`` (instance attribute), modern refs only
    # in comments referencing the historical fix.
    code_lines = [
        line for line in src.splitlines()
        if not line.strip().startswith("#") and not line.strip().startswith('"""')
    ]
    code_only = "\n".join(code_lines)
    assert "self._dialog_registered_page" not in code_only, (
        "Pre-fix self._dialog_registered_page attribute still in code — B-158 regression"
    )


# ---------------------------------------------------------------------------
# B-159 — asyncio loop running → RuntimeError loud
# ---------------------------------------------------------------------------


def test_lazy_init_raises_on_running_asyncio_loop():
    """B-159: silent passthrough (pre-fix) caused cryptic "Sync API inside the
    asyncio loop" RuntimeError mid-init. Fail loud with actionable message."""
    src = (REPO_ROOT / "p79/envs/vwa_wrapper.py").read_text(encoding="utf-8")
    # Two spots: _lazy_init + reset
    assert src.count("detected an active asyncio loop") >= 2, (
        "B-159 fail-loud raise missing in either _lazy_init or reset()"
    )
    # The raise message guides the caller to subprocess isolation
    assert "subprocess" in src, (
        "B-159 raise message should suggest subprocess isolation"
    )


# ---------------------------------------------------------------------------
# B-160 — navigate_to URL escape via json.dumps
# ---------------------------------------------------------------------------


def test_navigate_to_url_uses_json_dumps_escape():
    """B-160 closed the f-string injection vector. Check the navigate_to
    function body specifically — not the whole file (comments may reference
    the historical pattern)."""
    src = (REPO_ROOT / "p79/envs/vwa_wrapper.py").read_text(encoding="utf-8")
    # Extract just the navigate_to function body
    nav_start = src.find("def navigate_to(self")
    assert nav_start >= 0, "navigate_to function not found"
    nav_block = src[nav_start:nav_start + 2000]
    # Strip comment lines
    nav_code = "\n".join(
        line for line in nav_block.splitlines()
        if not line.strip().startswith("#")
    )
    # f-string pattern with raw "{url}" must NOT appear in code
    assert 'f\'page.goto("{url}")\'' not in nav_code, (
        "B-160 regression: f-string url interpolation still in navigate_to code"
    )
    # json.dumps(url) (or _json.dumps(url) alias) must be used
    assert "json.dumps(url)" in nav_code, (
        "B-160 navigate_to must escape url via json.dumps"
    )


# ---------------------------------------------------------------------------
# B-161 — Shadow DOM elementFromPoint penetration
# ---------------------------------------------------------------------------


def test_locator_dispatch_pierces_shadow_dom():
    """B-161: ``_JS_RESOLVE_CLICK``/``_INPUT``/``_UPLOAD`` must use the
    shadow-DOM-piercing helper instead of bare ``document.elementFromPoint``.
    Without piercing, Reddit redesign / modern SPA / web components shadow
    hosts intercept the click → walk-fail → framework bbox-center fallback
    (the B-33 buggy path Cluster 1 was meant to retire)."""
    src = (REPO_ROOT / "p79/envs/locator_dispatch.py").read_text(encoding="utf-8")
    assert "_pierceElementFromPoint" in src, (
        "B-161 missing _pierceElementFromPoint shadow-descent helper"
    )
    assert "shadowRoot" in src, "B-161 missing shadowRoot traversal"
    # All 3 resolvers use the helper (search for pattern occurrences)
    pierce_calls = src.count("_pierceElementFromPoint(cx, cy)")
    assert pierce_calls >= 3, (
        f"B-161 expects >=3 _pierceElementFromPoint calls (3 resolvers), found {pierce_calls}"
    )
    # Walk-up uses getRootNode().host to escape shadow boundaries
    assert "getRootNode" in src, (
        "B-161 walk-up missing getRootNode().host shadow-boundary escape"
    )
