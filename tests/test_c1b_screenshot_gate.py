"""Fire-6 RCA Stage C1b (/stress 2026-05-20): dom-only screenshot-timeout
fatality gate (VWAWrapper._gate_screenshot_timeout).

The submodule (async_envs.astep) recovers a Page.screenshot timeout to a blank
placeholder + info['screenshot_timeout_recovered']=True, mode-agnostically.
The wrapper is the SINGLE mode-gating chokepoint:
  - dom            → artifact-only → non-fatal (no raise)
  - som / vision / phantom_* / None → (potential) decision-input → re-raise
    (fail-safe-fatal; None = fatal)
"""
from __future__ import annotations

import pytest

from p79.envs.vwa_wrapper import VWAWrapper


def _wrapper(mode):
    w = VWAWrapper(dry_run=True)
    w.observation_mode = mode
    return w


def test_dom_recovery_is_nonfatal():
    # dom: artifact-only blank recovery must NOT raise
    _wrapper("dom")._gate_screenshot_timeout({"screenshot_timeout_recovered": True})


@pytest.mark.parametrize("mode", ["som", "vision"])
def test_decision_input_modes_are_fatal(mode):
    with pytest.raises(TimeoutError, match="Page.screenshot"):
        _wrapper(mode)._gate_screenshot_timeout({"screenshot_timeout_recovered": True})


@pytest.mark.parametrize("mode", ["phantom_som", "phantom_dom", "phantom_text", "phantom_prompt"])
def test_phantom_modes_are_fatal(mode):
    # Per user directive: C1b non-fatal recovery is restricted to dom ONLY.
    with pytest.raises(TimeoutError):
        _wrapper(mode)._gate_screenshot_timeout({"screenshot_timeout_recovered": True})


def test_none_mode_is_failsafe_fatal():
    # Unset mode must be treated as decision-input (fail-safe fatal).
    with pytest.raises(TimeoutError):
        _wrapper(None)._gate_screenshot_timeout({"screenshot_timeout_recovered": True})


def test_no_recovery_flag_is_noop():
    # No screenshot timeout → gate does nothing, even in a decision-input mode.
    _wrapper("som")._gate_screenshot_timeout({})
    _wrapper("som")._gate_screenshot_timeout({"screenshot_timeout_recovered": False})


def test_dom_no_flag_noop():
    _wrapper("dom")._gate_screenshot_timeout({"screenshot_timeout_recovered": False})
