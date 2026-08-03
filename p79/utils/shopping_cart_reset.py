"""Shopping per-task cart isolation (B-1936, 2026-08-03).

WHY THIS EXISTS
---------------
VWA's ``require_reset`` is implemented for classifieds ONLY
(``external/visualwebarena/browser_env/envs.py:172``::

    # Reset site if needed. Currently only supported for Classifieds.
    # TODO(jykoh): Add reset functionality for Shopping/Reddit.
    if instance_config.get("require_reset", False):
        if "classifieds" in instance_config["sites"]:

so on shopping the flag parses and then does nothing. reddit got a P79
compensation in B-1884 (``p79/utils/reddit_identity.py``); shopping had none.

The exposure is not marginal. Measured over
``config_files/vwa/test_shopping.json`` (466 tasks):

  * **108 tasks mutate the cart** ("add ... to my cart" / "into my shopping cart")
  * **104 tasks' evaluators READ the cart page** (``program_html`` against
    ``__SHOPPING__/checkout/cart``)
  * **19 tasks are flagged ``require_reset``** — so 89 cart-mutating tasks are
    not even flagged, and the 19 that are get a no-op anyway.

A condition runs these sequentially against one Magento instance, so the cart
accumulates monotonically for the whole condition.

An exhaustive static scan (2026-08-03, after a first pass that scanned only
``must_include`` and had to be redone) puts the identifiable exposure at
**6-10 tasks**, all in the **false-success** direction:

  * 3 task PAIRS whose ``must_include`` target string is IDENTICAL — (86, 87),
    (223, 224), (348, 349). Tasks run in ascending id order, so the second of
    each pair can pass on the first one's residue without acting.
  * 2 tasks whose target is a bare colour word — 453 (``"Green"``), 455
    (``"Gray"``) — satisfiable by any leftover product containing that word.
  * 2 tasks (463, 465) that pass on inaction regardless of cart state. Those are
    an evaluator defect, not this module's business — see AMENDMENT_09 below.

The **false-failure** direction, initially assumed, does not materialise: the 8
JS-locator exact-quantity tasks (289 / 320 / 321 and siblings) target products
no other task targets, so quantities do not accumulate across tasks.

Denominator matters here: 6-10 out of the **104 cart-graded** tasks (≈6-10%), not
out of 435 scored (≈2%). The errors are unidirectional and correlate with
position in the run, so they are a systematic order-dependent bias on the
cart-manipulation task cluster, not noise that averages out. A diffuse channel
(an agent adding to the cart while exploring any other task) is not statically
enumerable and is only closed by clearing unconditionally.

STATUS: ENABLED (PROTOCOL_NOTE_07, user decision 2026-08-03)
------------------------------------------------------------
Empties the shopping account's cart at the START of every shopping task, in the
setup phase before the measured trajectory — the same integration point as
``restore_reddit_identity``. Idempotent: on an already-empty cart the DELETE
matches 0 rows and the UPDATE matches 0 rows.

This reverses an earlier same-day call to disclose-and-not-fix. That call rested
on three supports and a ``/stress`` 3-AI round knocked out two of them:

  * **"cross-site estimand consistency" — FALSE.** classifieds already receives
    22 per-task FULL-SITE resets (upstream implements ``require_reset`` for that
    site alone) while reddit and shopping receive none. There was never a
    consistency to preserve, only a choice of which heterogeneity to carry. And
    note what this implies: clearing a cart does NOT make shopping match
    classifieds either — matching would mean a full container rebuild at 19 task
    points, ~4.75h per condition. Alignment is not on the menu; disclosure of a
    per-site protocol table is.
  * **"comparable in scale to reddit's ~6 tasks" — wrong denominator.** The
    exposure is 6-10 of the **104** cart-graded tasks (≈6-10%), not of 435
    (≈2%), and it is unidirectional (false successes only).
  * **"disclosure quality is higher here" — still true**, and it is what makes
    the sensitivity analysis in ``next_steps`` possible.

What decided it: shopping is **pre-data**. Zero VWA shopping runs exist on disk,
so this DEFINES shopping's estimand rather than changing a measured one, and the
identical fix after firing would cost 18 condition re-runs. reddit is not
reopened — its accumulation is bound in landed data and §402.7's disclose-only
ruling stands. Meta-analysis can carry documented protocol heterogeneity; it
cannot launder unidirectional measurement error.

WHY EVERY TASK RATHER THAN THE FLAGGED ONES
-------------------------------------------
Upstream flags 19 shopping tasks ``require_reset``. Those flags hit **1 of the
10** tasks actually at risk of a cart-residue collision — restricting the reset
to them would fix ~10% of the problem. Restricting it instead to the 10 tasks
this project enumerated would tune the protocol to a list that was already
demonstrated incomplete once (the first pass scanned only ``must_include`` and
missed the ``must_exclude`` channel entirely), and would still not cover the
diffuse channel: an agent adding to the cart while exploring any of the other
456 tasks. Unconditional-and-idempotent is the only form that closes all three.

NOT COVERED BY THIS MODULE — see AMENDMENT_09
---------------------------------------------
Tasks **463** and **465** stay broken with a perfectly clean cart: their
``program_html`` evals carry only ``must_exclude``, so doing nothing scores 1.
That is an evaluator defect, not a state-leakage one, and it is handled by the
scored-set exclusion in ``p79/experiment/tasks.py`` (AMENDMENT_09, the same
tier-A uniform rule AMENDMENT_08 applied to reddit 160).

Emptying is safe for the whole task set: no VWA/WA shopping task depends on
pre-existing cart contents. Every task mentioning the cart puts something INTO
it (verified across all 466 VWA shopping + 192 WA shopping + 182 WA
shopping_admin tasks, 2026-08-03).

SCOPE / LIMITS
--------------
  * Clears the CART (``quote`` / ``quote_item``). It does not roll back placed
    orders (``sales_order``): tasks 272/273 check out, and un-placing an order
    is not a cart operation. Full order rollback needs the container rebuild in
    ``_reset_vwa_local_shopping``, which runs per condition, not per task.
  * Targets the logged-in customer's quote. A cart built while the session had
    silently dropped to guest (``customer_email IS NULL``) is not matched —
    that state is itself an auth failure and is caught by ``auth_required_gate``.
  * **A100 live verification 2026-08-03 — EXERCISED end-to-end.** Against the
    running ``vwa-shopping`` container:

      - all 8 assumed ``quote`` columns exist; ``quote_item.quote_id`` exists;
        ``quote_address_item`` / ``quote_item`` / ``quote_item_option`` all
        declare ``ON DELETE CASCADE`` off ``quote_item``, so the item delete
        carries its dependents out;
      - ``clear_shopping_cart({})`` runs and returns True —
        *"ok for emma.lopez@gmail.com (customer resolved, items=0)"*;
      - the B-1942 guard fires correctly on a wrong identity: a username where
        an email belongs returns False with the diagnostic, and under
        ``P79_PAPER_GRADE=1`` it raises;
      - **the seed state has quote=0 / quote_item=0 rows** (customer_entity=27,
        sales_order=189). This is what forced B-1949.

    Still not exercised: a NON-TRIVIAL clear (put an item in, clear it, confirm
    it is gone). The runs above were all against an empty cart, so the DELETE
    matched zero rows every time — the statements are proven to parse and the
    verification logic is proven to discriminate, but the delete has not yet been
    observed removing anything. Every identifier is config-overridable, so a
    schema surprise is a config edit, not a code change.

Runs WHERE THE RUNNER RUNS (the A100 self-hosted VWA host), shelling to
``docker exec vwa-shopping ...``. ``fail_closed`` defaults to **"auto"**
(B-1943) = hard-fail when ``P79_PAPER_GRADE=1``, warn otherwise. Under fire, a
clear that silently did nothing would leave the condition on an unknown cart
state, indistinguishable from a clean one in every summary; on a dev box with no
container the same event is a logged warning so smoke runs still work. Explicit
``fail_closed: true/false`` overrides the resolution.
"""
from __future__ import annotations

import logging
import os
import shlex
import subprocess
from typing import Optional

_logger = logging.getLogger(__name__)

# Every value is overridable via the ``shopping_cart_reset`` config block, so a
# schema surprise found at A100 verification time is a one-line config change.
_DEFAULTS = {
    # ENABLED per PROTOCOL_NOTE_07 (user decision 2026-08-03) — see the module
    # docstring for what decided it. Turning this OFF again is also an estimand
    # change and needs the same paper trail.
    "enabled": True,
    "container": "vwa-shopping",
    "db": "magentodb",
    "db_user": "magentouser",
    # B-747: password travels via the MYSQL_PWD env injected into `docker exec`,
    # never as a `-p...` argv, so it stays out of `ps auxe` on the shared VM.
    "db_password": "MyPassword",
    "quote_table": "quote",
    "quote_item_table": "quote_item",
    # B-1949: identity is probed here, not in `quote` — Magento creates the
    # quote row lazily, so a clean cart legitimately has none.
    "customer_table": "customer_entity",
    # "" → falls back to $VWA_SHOPPING_USER, then to the WebArena seed account.
    # CLAUDE.md hard rule #1 records shop's shared account as emma.lopez.
    "customer_email": "",
    "sql_override": "",       # if set, used verbatim (overrides all table logic)
    "timeout_s": 30,
    # "auto" = hard-fail under P79_PAPER_GRADE=1, warn on dev boxes (B-1943).
    "fail_closed": "auto",
}

_SEED_CUSTOMER_EMAIL = "emma.lopez@gmail.com"


def _resolve(cfg: Optional[dict]) -> dict:
    out = dict(_DEFAULTS)
    out.update((cfg or {}).get("shopping_cart_reset", {}) or {})
    if not out.get("customer_email"):
        out["customer_email"] = os.environ.get("VWA_SHOPPING_USER") or _SEED_CUSTOMER_EMAIL
    # B-1943 (cont): `fail_closed: "auto"` — hard-fail under paper-grade, warn
    # elsewhere. codex Mode B F7 asked for fail-closed *when enabled under
    # paper-grade*; making it unconditional instead broke every dev/smoke run on
    # a box with no shopping container (the reset raised, the episode was
    # quarantined). Mirrors `VWA_RESET_MODE=auto` and the AUTH_GATE_BYPASS
    # paper-grade hard-block. Explicit True/False still wins.
    if out.get("fail_closed") == "auto":
        out["fail_closed"] = os.environ.get("P79_PAPER_GRADE", "0") == "1"
    return out


def _sqlq(s: str) -> str:
    """SQL string literal escaping (double single-quotes)."""
    return str(s).replace("'", "''")


def build_clear_cart_sql(c: dict) -> str:
    """Idempotent cart-empty for one customer (no-op on an already-empty cart).

    Two statements, in this order:

    1. ``DELETE`` the quote's line items. Magento's ``quote_item_option`` and
       friends cascade off ``quote_item`` by foreign key, so the item rows carry
       their dependents out with them.
    2. ``UPDATE`` the quote's cached totals to zero. Magento denormalises
       ``items_count`` / ``items_qty`` / the totals onto ``quote``; leaving them
       stale would keep the header cart badge showing a count for a cart with no
       contents, which the agent can see. The ``AND (...)`` guard keeps the
       statement a genuine no-op when the values are already zero.

    The quote ROW is deliberately kept (and kept active) rather than deleted:
    the browser session may already hold this ``quote_id``, and deleting the row
    out from under it risks a dangling reference. An empty quote is the same
    thing to the storefront and is the safer of the two.
    """
    if c.get("sql_override"):
        return str(c["sql_override"])
    q = str(c["quote_table"])
    qi = str(c["quote_item_table"])
    email = _sqlq(c["customer_email"])
    delete_items = (
        f"DELETE qi FROM {qi} qi "
        f"INNER JOIN {q} q ON qi.quote_id = q.entity_id "
        f"WHERE q.customer_email = '{email}';"
    )
    zero_totals = (
        f"UPDATE {q} SET items_count=0, items_qty=0, "
        f"subtotal=0, base_subtotal=0, grand_total=0, base_grand_total=0 "
        f"WHERE customer_email='{email}' "
        f"AND (items_count<>0 OR items_qty<>0 OR grand_total<>0);"
    )
    return f"{delete_items} {zero_totals}"


def build_verify_sql(c: dict) -> str:
    """Post-clear assertion: the identity RESOLVES and no line items survive.

    B-1942 (codex Mode B F8, 2026-08-03): ``rc == 0`` does not mean anything was
    cleared. MySQL exits 0 when a syntactically valid DELETE/UPDATE matches zero
    rows, so a wrong ``customer_email`` — a username where an email was expected,
    a drifted seed account, a typo in the env — produces a perfectly successful
    run that touches nothing, and the caller records "cart cleared". Believing
    you are clean when you are not is worse than knowing you are dirty: the
    former is invisible in every summary.

    B-1949 (A100 live verification, 2026-08-03): the identity probe must hit
    ``customer_entity``, NOT ``quote``. The first version asserted ``quotes >= 1``
    and would have aborted the first task of every shopping condition, because
    **Magento creates the quote row lazily** — measured on the live A100 seed
    container: ``quote`` = 0 rows and ``quote_item`` = 0 rows, while
    ``customer_entity`` = 27 rows with ``emma.lopez@gmail.com`` present and
    ``sales_order`` = 189. A cart that has never been used simply has no quote.
    Conflating "no quote row" with "customer not found" turned the clean state
    into a hard failure. The two zero-states are:

      * ``customer == 0`` → the identity is wrong. Real misconfiguration.
      * ``items == 0`` with no quote row → the cart is empty. Normal, and the
        state every task should start from.
    """
    q = str(c["quote_table"])
    qi = str(c["quote_item_table"])
    ce = str(c["customer_table"])
    email = _sqlq(c["customer_email"])
    return (
        f"SELECT (SELECT COUNT(*) FROM {ce} WHERE email='{email}') AS customer, "
        f"(SELECT COUNT(*) FROM {qi} qi INNER JOIN {q} q ON qi.quote_id=q.entity_id "
        f"WHERE q.customer_email='{email}') AS items;"
    )


def build_docker_argv(c: dict, sql: str | None = None) -> list[str]:
    """``docker exec -e MYSQL_PWD=... <container> mysql -u <user> <db> -e "<sql>"``."""
    if sql is None:
        sql = build_clear_cart_sql(c)
    return [
        "docker", "exec",
        "-e", "MYSQL_PWD={}".format(c["db_password"]),
        str(c["container"]),
        "mysql", "-u", str(c["db_user"]), "-N", "-B", str(c["db"]), "-e", sql,
    ]


def clear_shopping_cart(
    cfg: Optional[dict], *, logger: Optional[logging.Logger] = None
) -> bool:
    """Idempotently empty the shopping test account's cart.

    Returns True on success (including the no-op case where the cart was
    already empty), False on failure. Never raises unless
    ``shopping_cart_reset.fail_closed`` is set.
    """
    log = logger or _logger
    c = _resolve(cfg)
    if not c.get("enabled", True):
        return True

    def _fail(msg: str, exc: Exception | None = None) -> bool:
        log.warning("shopping_cart_reset: %s", msg)
        if c.get("fail_closed"):
            if exc is not None:
                raise exc
            raise RuntimeError(f"shopping_cart_reset: {msg}")
        return False

    def _run(sql: str):
        return subprocess.run(
            build_docker_argv(c, sql), capture_output=True, text=True,
            timeout=int(c["timeout_s"]),
        )

    try:
        r = _run(build_clear_cart_sql(c))
    except Exception as exc:  # docker missing (dev box), timeout, etc.
        return _fail(f"could not clear cart ({type(exc).__name__}): {exc}", exc)
    if r.returncode != 0:
        return _fail(f"clear rc={r.returncode} stderr={(r.stderr or '')[-300:]}")

    # B-1942: rc==0 proves the statements PARSED, not that they matched anything.
    # Verify the customer's quote exists and is now empty before reporting success.
    try:
        v = _run(build_verify_sql(c))
    except Exception as exc:
        return _fail(f"verify query failed ({type(exc).__name__}): {exc}", exc)
    if v.returncode != 0:
        return _fail(f"verify rc={v.returncode} stderr={(v.stderr or '')[-300:]}")
    fields = (v.stdout or "").split()
    if len(fields) < 2 or not all(f.isdigit() for f in fields[:2]):
        return _fail(f"verify returned unparseable output {(v.stdout or '')!r:.120}")
    customer, items = int(fields[0]), int(fields[1])
    if customer == 0:
        # The misconfiguration that would otherwise pass silently: the identity
        # does not resolve, so every DELETE matched zero rows for a reason that
        # has nothing to do with the cart being clean. (B-1949: probed against
        # customer_entity — an absent QUOTE row is normal, an absent CUSTOMER is
        # not.)
        return _fail(
            f"customer_email={c['customer_email']!r} not found in "
            f"{c['customer_table']} — the cart was NOT verified clean, the identity "
            f"never resolved (check VWA_SHOPPING_USER: an account username where an "
            f"email is expected produces exactly this)"
        )
    if items != 0:
        return _fail(f"cart still holds {items} item(s) after clear")
    log.info(
        "shopping_cart_reset: ok for %s (customer resolved, items=0)",
        c["customer_email"],
    )
    return True
