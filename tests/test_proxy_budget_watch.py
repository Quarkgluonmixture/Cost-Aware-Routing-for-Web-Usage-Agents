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
    """A rejection then a successful read on a pool we had never measured.

    `lowest is None` means the watcher has no baseline, so the money is
    necessarily new — this is the drained-pool-restart case.
    """
    _, fired = run([(None, "quota:403"), (250.0, "ok")])
    assert fired == ["TOPPED_UP"]


def test_topped_up_does_NOT_fire_on_transient_rejection_at_same_balance():
    """THE Mode B P1-8 regression. A blip must not announce credit.

    `ok($100) → quota → ok($100)` used to fire TOPPED_UP with the balance
    unchanged, because the rule keyed on "was rejected at some point" alone.
    Acting on that alert means restarting a chain against an empty pool.
    """
    _, fired = run([(100.0, "ok"), (None, "quota:403"), (100.0, "ok")])
    assert "TOPPED_UP" not in fired, "unchanged balance must never read as a top-up"


def test_topped_up_fires_when_balance_really_rose_after_rejection():
    _, fired = run([(0.5, "ok"), (None, "quota:403"), (500.0, "ok")])
    assert "TOPPED_UP" in fired


def test_auth_or_rate_never_triggers_exhausted():
    """Two consecutive credential failures are not a budget signal."""
    _, fired = run([(None, "auth_or_rate:403"), (None, "auth_or_rate:403")])
    assert fired == []


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

def test_probe_actually_sends_production_max_tokens(monkeypatch):
    """Assert the WIRE payload, not the constant.

    The previous version checked `PROD_MAX_TOKENS == 4096` and the signature
    default — both of which stay true if someone hardcodes `"max_tokens": 1`
    into the request body, which is exactly the regression the test exists to
    prevent (/stress 2026-08-09 Mode B P2-1).

    Note the *reason* for 4096 deliberately does NOT cite the "74-minute blind
    window" this file was originally written around: that number came from
    comparing A100 (UTC) and DGX (BST) timestamps without converting, and the
    real gap is ~14 minutes against a 10-minute poll interval (笔记 §446.7).
    The choice stands on the cheap argument: same shape as production.
    """
    seen = {}

    class _Resp:
        status_code = 200
        text = ""

        def json(self):
            return {"metadata": {"remaining_quota": {"remaining_budget": 42.0}}}

    def _capture(url, headers=None, json=None, timeout=None):
        seen.update(json or {})
        return _Resp()

    monkeypatch.setattr(pbw.requests, "post", _capture)
    rem, outcome = pbw.probe("fake-key")
    assert outcome == "ok" and rem == 42.0
    assert seen["max_tokens"] == pbw.PROD_MAX_TOKENS == 4096, (
        f"probe sent max_tokens={seen.get('max_tokens')} — the wire payload must "
        "match production, not just the module constant"
    )


# --- probe outcome classification (Mode B P1-9) -----------------------------

class _R:
    def __init__(self, code, text=""):
        self.status_code, self.text = code, text

    def json(self):
        import json as _j

        return _j.loads(self.text)


@pytest.mark.parametrize(
    "code,body,expected",
    [
        # The REAL exhaustion body, measured 2026-08-09 on a drained pool.
        (403, '{"error":"Budget exceeded","usedUsd":999.99,"budgetUsd":1000}', "quota:403"),
        # Same status code, entirely different cause — must NOT read as budget.
        (403, '{"error":"invalid api key"}', "auth_or_rate:403"),
        (429, '{"error":"rate limit: retry later"}', "auth_or_rate:429"),
        # B-1880 proxy outage stays its own thing.
        (503, "Service Unavailable", "proxy_outage:503"),
        (500, "oops", "proxy_outage:500"),
    ],
)
def test_probe_classifies_by_body_not_status_code(monkeypatch, code, body, expected):
    """An expired credential must never be reported as 'out of money'.

    Pre-fix, `status_code in (402,403,429)` alone meant quota — so a rotated
    API key would have fired the urgent budget-exhausted alert.
    """
    monkeypatch.setattr(
        pbw.requests, "post", lambda *a, **k: _R(code, body)
    )
    rem, outcome = pbw.probe("fake-key")
    assert rem is None
    assert outcome == expected


def test_exhausted_body_still_yields_a_balance():
    """The 403 body carries usedUsd/budgetUsd — readable when 200 is not."""
    got = pbw.parse_exhausted_balance(
        '{"error":"Budget exceeded","usedUsd":999.5,"budgetUsd":1000}'
    )
    assert got is not None and abs(got - 0.5) < 1e-6
    assert pbw.parse_exhausted_balance("not json") is None
