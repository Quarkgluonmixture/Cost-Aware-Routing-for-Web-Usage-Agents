"""B-1936: shopping per-task cart isolation unit tests.

The module is **ENABLED** per PROTOCOL_NOTE_07 (user decision 2026-08-03) after
a /stress 3-AI round knocked out two of the three supports for the earlier
disclose-only call. `test_enabled_by_default_under_protocol_note_07` locks that,
and `test_zero_rows_is_not_success` locks the failure mode that makes an enabled
reset dangerous: a clear that silently matched nothing.

Covers SQL construction (idempotent guard + injection-safe quoting + override),
the docker-exec command shape (password via MYSQL_PWD env, never argv, per
B-747), the enabled gate, the customer-email resolution order, and the
soft-fail / fail-closed subprocess contract. No live DB / docker needed.

Also asserts the two facts about the TASK SET that document the defect and
constrain the shape of any future fix, read straight out of the upstream config
files:
  * shopping's `require_reset` flags cover only a fraction of the cart-mutating
    tasks, so keying a reset off the flag would reproduce upstream's blind spot;
  * no task depends on pre-existing cart contents, so emptying would be safe
    if it were ever authorised.
"""
import json
from pathlib import Path
from unittest import mock

import pytest

from p79.utils.shopping_cart_reset import (
    build_clear_cart_sql,
    build_docker_argv,
    clear_shopping_cart,
    _resolve,
    _DEFAULTS,
    _SEED_CUSTOMER_EMAIL,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
VWA_SHOPPING_TASKS = REPO_ROOT / "external/visualwebarena/config_files/vwa/test_shopping.json"

_CART_MUTATION_PHRASES = (
    "to my cart", "into my cart", "in my cart", "to my shopping cart", "add to cart",
)


def _load_shopping_tasks():
    if not VWA_SHOPPING_TASKS.exists():
        pytest.skip("VWA submodule task configs not checked out")
    return json.loads(VWA_SHOPPING_TASKS.read_text())


def test_default_sql_clears_items_and_zeroes_totals():
    sql = build_clear_cart_sql(_resolve(None))
    assert "DELETE qi FROM quote_item qi" in sql
    assert "INNER JOIN quote q ON qi.quote_id = q.entity_id" in sql
    assert f"q.customer_email = '{_SEED_CUSTOMER_EMAIL}'" in sql
    # Denormalised totals must be zeroed too, or the header cart badge keeps
    # showing a count for an empty cart — which the agent can see.
    for col in ("items_count=0", "items_qty=0", "grand_total=0", "base_grand_total=0"):
        assert col in sql


def test_sql_is_a_noop_on_already_empty_cart():
    """The UPDATE must carry a guard so a clean cart matches 0 rows."""
    sql = build_clear_cart_sql(_resolve(None))
    assert "AND (items_count<>0 OR items_qty<>0 OR grand_total<>0)" in sql


def test_quote_row_is_not_deleted():
    """Deleting the quote row risks dangling the session's cached quote_id.

    An emptied quote is equivalent to the storefront and strictly safer.
    """
    sql = build_clear_cart_sql(_resolve(None))
    assert "DELETE FROM quote " not in sql
    assert "DELETE q FROM quote" not in sql


def test_customer_email_resolution_order():
    # explicit config wins
    c = _resolve({"shopping_cart_reset": {"customer_email": "explicit@x.com"}})
    assert c["customer_email"] == "explicit@x.com"
    # else env
    with mock.patch.dict("os.environ", {"VWA_SHOPPING_USER": "envuser@x.com"}):
        assert _resolve(None)["customer_email"] == "envuser@x.com"
    # else seed account
    with mock.patch.dict("os.environ", {}, clear=True):
        assert _resolve(None)["customer_email"] == _SEED_CUSTOMER_EMAIL


def test_email_single_quote_is_doubled():
    c = _resolve({"shopping_cart_reset": {"customer_email": "o'brien@x.com"}})
    assert "'o''brien@x.com'" in build_clear_cart_sql(c)


def test_sql_override_wins_verbatim():
    c = _resolve({"shopping_cart_reset": {"sql_override": "DELETE FROM quote_item"}})
    assert build_clear_cart_sql(c) == "DELETE FROM quote_item"


def test_docker_argv_passes_password_via_env_not_argv():
    """B-747: the DB password must not appear in argv (`ps auxe` on a shared VM)."""
    argv = build_docker_argv(_resolve(None))
    assert argv[:2] == ["docker", "exec"]
    assert "-e" in argv and any(a.startswith("MYSQL_PWD=") for a in argv)
    assert not any(a.startswith("-p") and len(a) > 2 for a in argv), (
        "password must travel via MYSQL_PWD env, never as a -p<pass> argv"
    )
    assert "vwa-shopping" in argv
    assert "magentodb" in argv


_ENABLED = {"shopping_cart_reset": {"enabled": True}}
_OK = mock.Mock(returncode=0, stdout="1\t0\n", stderr="")   # quotes=1, items=0


def test_enabled_by_default_under_protocol_note_07():
    """PROTOCOL_NOTE_07 (2026-08-03): shopping gets per-task cart isolation.

    Flipping this back OFF is also an estimand change and needs the same paper
    trail — shopping is pre-data today, so the default here DEFINES its estimand
    rather than changing a measured one.
    """
    from p79.experiment.config import DEFAULT_CONFIG
    blk = DEFAULT_CONFIG["shopping_cart_reset"]
    assert blk["enabled"] is True, "PROTOCOL_NOTE_07 enables per-task cart isolation"
    assert blk["fail_closed"] == "auto", (
        "B-1943: fail_closed resolves per environment — hard-fail under "
        "P79_PAPER_GRADE=1 (a silent clear failure would leave the condition on "
        "an unknown cart state, indistinguishable from clean), warn on dev boxes "
        "that have no shopping container so smoke runs still work"
    )
    assert _DEFAULTS["enabled"] is True

    # the resolution itself, both ways
    with mock.patch.dict("os.environ", {"P79_PAPER_GRADE": "1"}):
        assert _resolve(None)["fail_closed"] is True
    with mock.patch.dict("os.environ", {}, clear=True):
        assert _resolve(None)["fail_closed"] is False
    # explicit setting still wins over the auto resolution
    with mock.patch.dict("os.environ", {"P79_PAPER_GRADE": "1"}):
        assert _resolve({"shopping_cart_reset": {"fail_closed": False}})["fail_closed"] is False


def test_zero_rows_is_not_success():
    """B-1942 (codex Mode B F8): rc==0 proves the SQL parsed, not that it matched.

    MySQL exits 0 when a valid DELETE/UPDATE matches zero rows. A wrong
    `customer_email` — a username where an email belongs is the realistic case —
    therefore produced a fully "successful" run that touched nothing. Believing
    you are clean when you are not is the worst of the three states.
    """
    # `_SOFT_LOCAL` opts out of fail-closed so the FALSE return is observable;
    # `test_customer_not_found_fails_closed_loudly` covers the raising path.
    _SOFT_LOCAL = {"shopping_cart_reset": {"enabled": True, "fail_closed": False}}

    # verify reports quotes=0 → customer never found → must NOT be success
    seq = [mock.Mock(returncode=0, stdout="", stderr=""),          # the clear
           mock.Mock(returncode=0, stdout="0\t0\n", stderr="")]   # verify: no quote
    with mock.patch("p79.utils.shopping_cart_reset.subprocess.run", side_effect=seq):
        assert clear_shopping_cart(_SOFT_LOCAL) is False

    # items survive the clear → also not success
    seq = [mock.Mock(returncode=0, stdout="", stderr=""),
           mock.Mock(returncode=0, stdout="1\t3\n", stderr="")]
    with mock.patch("p79.utils.shopping_cart_reset.subprocess.run", side_effect=seq):
        assert clear_shopping_cart(_SOFT_LOCAL) is False

    # quote exists and is empty → the only success
    seq = [mock.Mock(returncode=0, stdout="", stderr=""),
           mock.Mock(returncode=0, stdout="1\t0\n", stderr="")]
    with mock.patch("p79.utils.shopping_cart_reset.subprocess.run", side_effect=seq):
        assert clear_shopping_cart(_ENABLED) is True


def test_customer_not_found_fails_closed_loudly():
    """The misconfiguration must abort, not warn, when fail_closed is set."""
    seq = [mock.Mock(returncode=0, stdout="", stderr=""),
           mock.Mock(returncode=0, stdout="0\t0\n", stderr="")]
    cfg = {"shopping_cart_reset": {"enabled": True, "fail_closed": True}}
    with mock.patch("p79.utils.shopping_cart_reset.subprocess.run", side_effect=seq):
        with pytest.raises(RuntimeError, match="no quote row"):
            clear_shopping_cart(cfg)


def test_runner_checks_the_return_value():
    """B-1943 (codex Mode B F7): the runner must not discard the result."""
    src = (REPO_ROOT / "p79/experiment/runner/main.py").read_text(encoding="utf-8")
    i = src.index("clear_shopping_cart")
    window = src[i:i + 400]
    assert "if not clear_shopping_cart" in window, (
        "runner discards clear_shopping_cart()'s result — a failed clear would "
        "degrade to a log line while the condition ran on an accumulating cart"
    )


def test_explicitly_disabled_returns_true_without_subprocess():
    with mock.patch("p79.utils.shopping_cart_reset.subprocess.run") as m:
        ok = clear_shopping_cart({"shopping_cart_reset": {"enabled": False}})
    assert ok is True
    m.assert_not_called()


def test_success_returns_true():
    """Two round-trips now: the clear, then the B-1942 verify."""
    seq = [mock.Mock(returncode=0, stdout="", stderr=""),
           mock.Mock(returncode=0, stdout="1\t0\n", stderr="")]
    with mock.patch("p79.utils.shopping_cart_reset.subprocess.run", side_effect=seq):
        assert clear_shopping_cart(_ENABLED) is True


# ── failure contract ──────────────────────────────────────────────────────────
# B-1943 inverted the default: `fail_closed` is now True, so failures RAISE.
# Soft-fail is the explicit opt-out, for dev boxes with no shopping container.
_SOFT = {"shopping_cart_reset": {"enabled": True, "fail_closed": False}}


def test_nonzero_rc_raises_under_paper_grade():
    fake = mock.Mock(returncode=1, stdout="", stderr="ERROR 1146 unknown table")
    with mock.patch.dict("os.environ", {"P79_PAPER_GRADE": "1"}), \
         mock.patch("p79.utils.shopping_cart_reset.subprocess.run", return_value=fake):
        with pytest.raises(RuntimeError, match="clear rc=1"):
            clear_shopping_cart(_ENABLED)


def test_docker_missing_raises_under_paper_grade_but_warns_on_dev():
    """The dev path is what keeps smoke tests runnable on a box with no container."""
    with mock.patch.dict("os.environ", {"P79_PAPER_GRADE": "1"}), \
         mock.patch("p79.utils.shopping_cart_reset.subprocess.run",
                    side_effect=FileNotFoundError("docker not found")):
        with pytest.raises(FileNotFoundError):
            clear_shopping_cart(_ENABLED)
    with mock.patch.dict("os.environ", {}, clear=True), \
         mock.patch("p79.utils.shopping_cart_reset.subprocess.run",
                    side_effect=FileNotFoundError("docker not found")):
        assert clear_shopping_cart(_ENABLED) is False


def test_soft_fail_is_the_explicit_opt_out():
    """Dev boxes without the container opt out; paper-grade must not."""
    fake = mock.Mock(returncode=1, stdout="", stderr="boom")
    with mock.patch("p79.utils.shopping_cart_reset.subprocess.run", return_value=fake):
        assert clear_shopping_cart(_SOFT) is False
    with mock.patch(
        "p79.utils.shopping_cart_reset.subprocess.run",
        side_effect=FileNotFoundError("docker not found"),
    ):
        assert clear_shopping_cart(_SOFT) is False


def test_config_block_present_in_defaults():
    # CLAUDE.md: new config fields must have a DEFAULT_CONFIG default
    from p79.experiment.config import DEFAULT_CONFIG
    assert "shopping_cart_reset" in DEFAULT_CONFIG
    assert DEFAULT_CONFIG["shopping_cart_reset"]["container"] == "vwa-shopping"
    assert _DEFAULTS["quote_table"] == "quote"


def test_runner_invokes_cart_reset_for_shopping_sites():
    """The hook must sit alongside the reddit restore, in the setup phase."""
    src = (REPO_ROOT / "p79/experiment/runner/main.py").read_text(encoding="utf-8")
    assert "clear_shopping_cart" in src, "runner must call clear_shopping_cart"
    assert 'task.site in ("shopping", "shopping_admin")' in src, (
        "both Magento sites must get cart isolation"
    )
    # It has to run BEFORE the auth-refresh/browser-context work, like the
    # reddit restore — i.e. in setup, outside the measured trajectory.
    assert src.index("clear_shopping_cart") < src.index("Auth refresh check"), (
        "cart reset must run in the setup phase, before measured execution"
    )


def test_require_reset_flags_do_not_cover_the_cart_mutating_tasks():
    """Why the reset is unconditional rather than keyed off `require_reset`.

    Upstream flags 19 shopping tasks; far more than that mutate the cart. If we
    gated on the flag we would inherit exactly the coverage hole that makes the
    flag useless here (and it is a no-op on shopping anyway — envs.py:172).
    """
    tasks = _load_shopping_tasks()
    flagged = {t["task_id"] for t in tasks if t.get("require_reset")}
    mutators = {
        t["task_id"] for t in tasks
        if any(p in t["intent"].lower() for p in _CART_MUTATION_PHRASES)
    }
    assert len(mutators) > 3 * len(flagged), (
        f"expected cart mutators ({len(mutators)}) to far exceed require_reset "
        f"flags ({len(flagged)}) — if upstream ever fixes the flags, revisit "
        f"whether the unconditional reset is still the right shape"
    )
    assert mutators - flagged, "unflagged cart mutators are the reason for this fix"


def test_no_shopping_task_depends_on_preexisting_cart_contents():
    """Why emptying is safe: every cart-referencing task ADDS to it.

    If a task ever ships that reads a pre-seeded cart, this guard fails and the
    unconditional reset has to become conditional.
    """
    tasks = _load_shopping_tasks()
    suspicious = []
    for t in tasks:
        intent = t["intent"].lower()
        if "cart" not in intent:
            continue
        if not any(
            verb in intent
            for verb in ("add", "put", "place", "order", "buy", "purchase", "checkout")
        ):
            suspicious.append((t["task_id"], t["intent"]))
    assert not suspicious, (
        f"task(s) may depend on pre-existing cart state, which the per-task "
        f"cart reset would destroy: {suspicious}"
    )
