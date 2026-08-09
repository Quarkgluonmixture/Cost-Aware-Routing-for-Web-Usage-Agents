"""Guards for the proxy-budget watcher's alert state machine (2026-08-09).

The watcher got its alert rules wrong twice in one night, in opposite
directions, which is why this file exists:

  1. **False positive** — v1 treated any non-200 as "budget exhausted" and
     alerted on a `503`, i.e. the KNOWN B-1880 proxy outage. A monitor that
     cries wolf on routine infra noise gets muted, and then it is worth less
     than nothing.
  2. **False negative** — the rewrite gated TOPPED_UP on `rem > lowest + 10`.
     Started against an already-empty pool (exactly the situation it gets
     started in), the first successful read becomes its own baseline and the
     single most important alert never fires at all.

Both are pinned below. The rule that matters: TOPPED_UP keys off the STATE
TRANSITION out of quota rejection, not off a numeric jump.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "proxy_budget_watch",
    Path(__file__).resolve().parents[1] / "scripts/maintenance/proxy_budget_watch.py",
)
pbw = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pbw)

FRESH = {"fired": set(), "lowest": None, "consecutive_quota": 0,
         "seen_quota_rejection": False}


def run(seq, low_threshold=1.0):
    """Feed a sequence of (remaining, outcome) and collect alerts in order."""
    state, fired = dict(FRESH), []
    for rem, outcome in seq:
        state, alerts = pbw.decide_alerts(state, rem, outcome, low_threshold)
        fired.extend(alerts)
    return state, fired


# --- failure 1: the false positive -----------------------------------------

def test_single_503_never_alerts():
    """B-1880 proxy outage is not a budget signal."""
    _, fired = run([(None, "proxy_outage:503")])
    assert fired == []


def test_sustained_503_cluster_never_alerts():
    """8-10min 503 clusters are documented behaviour; 6 polls must stay quiet."""
    _, fired = run([(None, "proxy_outage:503")] * 6)
    assert fired == []


def test_network_errors_never_alert():
    _, fired = run([(None, "network:ConnectionError")] * 4)
    assert fired == []


def test_single_quota_rejection_is_not_enough():
    """One 403 could be a blip; exhaustion needs the second confirmation."""
    _, fired = run([(None, "quota:403")])
    assert fired == []


def test_two_consecutive_quota_rejections_alert():
    _, fired = run([(None, "quota:403"), (None, "quota:403")])
    assert fired == ["EXHAUSTED"]


def test_quota_then_ok_then_quota_does_not_alert():
    """The counter must RESET on success, not accumulate across recoveries."""
    _, fired = run([(None, "quota:403"), (50.0, "ok"), (None, "quota:403")])
    assert "EXHAUSTED" not in fired


# --- failure 2: the false negative -----------------------------------------

def test_topped_up_fires_when_watcher_starts_on_empty_pool():
    """THE regression. Watcher starts empty-pool, money arrives, must alert.

    Under the `rem > lowest + 10` rule this returned no TOPPED_UP at all.
    """
    _, fired = run([
        (None, "quota:403"),
        (None, "quota:403"),
        (500.0, "ok"),          # top-up lands
    ])
    assert "TOPPED_UP" in fired, "top-up after an empty start must alert"


def test_topped_up_fires_on_first_read_after_rejection():
    """Even a single rejection then success counts — no need to reach EXHAUSTED."""
    _, fired = run([(None, "quota:403"), (250.0, "ok")])
    assert fired == ["TOPPED_UP"]


def test_topped_up_also_fires_on_large_jump_without_rejection():
    """Second path: balance jumps while still healthy (top-up before empty)."""
    _, fired = run([(5.0, "ok"), (500.0, "ok")])
    assert "TOPPED_UP" in fired


def test_topped_up_does_not_fire_on_ordinary_drift():
    """Consumption must never look like a top-up."""
    _, fired = run([(500.0, "ok"), (480.0, "ok"), (475.0, "ok")])
    assert "TOPPED_UP" not in fired


def test_each_alert_fires_at_most_once():
    seq = [(None, "quota:403"), (None, "quota:403"), (None, "quota:403"),
           (500.0, "ok"), (0.5, "ok"), (0.4, "ok")]
    _, fired = run(seq)
    for alert in ("EXHAUSTED", "TOPPED_UP", "LOW"):
        assert fired.count(alert) <= 1, f"{alert} fired {fired.count(alert)}x"


# --- LOW threshold ----------------------------------------------------------

def test_low_fires_below_threshold():
    _, fired = run([(0.42, "ok")])
    assert fired == ["LOW"]


def test_low_does_not_fire_above_threshold():
    _, fired = run([(1.5, "ok")])
    assert fired == []


def test_low_threshold_is_well_above_one_real_step():
    """A $0.06/episode workload needs headroom, not a $0.01 tripwire.

    The default exists so the alert arrives while there is still time to act.
    """
    import inspect

    src = inspect.getsource(pbw.main)
    assert '"--low-threshold", type=float, default=1.00' in src


# --- probe shape ------------------------------------------------------------

def test_probe_uses_production_max_tokens():
    """A probe that does not have production's shape only proves the probe runs.

    Empirically 2026-08-09: a max_tokens=1 probe reported healthy for 74 minutes
    after the real run had already been rejected.
    """
    assert pbw.PROD_MAX_TOKENS == 4096
    import inspect

    sig = inspect.signature(pbw.probe)
    assert sig.parameters["max_tokens"].default == pbw.PROD_MAX_TOKENS
