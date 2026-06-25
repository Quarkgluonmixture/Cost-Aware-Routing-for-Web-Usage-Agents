"""Reddit shared-account identity restore (B-1884 / Fix 4, 2026-06-25).

VWA reddit task 138 ("Change my username to the first name of the recipient
in the image") is a *destructive* task: a capable model successfully renames
the shared test account (postmill user ``id=13915``,
``MarvelsGrantMan136`` → e.g. "Patrick"). The username IS the login
credential, so after the rename P79's periodic fresh re-login
(``p79/utils/auth_refresh.py``, every ~5 episodes, to survive the 24-min PHP
session expiry) fails with "Invalid credentials" → ``auth_required_gate``
raises ``AuthRefreshFailure`` → ``PaperGradeAbortError`` → the whole reddit
condition fail-closes. This was the root cause of the 2026-06 reddit abort
saga (see ``docs/reference/master_bug_catalog.md`` B-1884 + 笔记 §354/§355/§357).

**Fix 4** (estimand = clean per-task capability; the option recommended +
chosen by the user 2026-06-25): restore the username to its seed value at the
START of every reddit task, BEFORE the auth-refresh check — mirroring the
per-task ``require_reset`` that classifieds already gets from the upstream VWA
harness (which never implemented a reddit reset:
``external/visualwebarena/browser_env/envs.py:172`` ``TODO(jykoh)``). The
restore is idempotent (no-op when already correct) and runs in the SETUP
phase *before* the measured trajectory, so it does NOT touch measured
execution and leaves already-bound classifieds data unaffected.

DB access path is **verified** (笔记 §354 cross-system audit, codex_outputs/
cross_system_docker_audit_codex_003729.md): the postmill database lives INSIDE
the ``vwa-reddit`` container and is reached as the ``postgres`` superuser via
unix-socket peer auth (``su - postgres``) — the password-based ``-U`` path is
not provisioned. Database ``postmill``, table ``users``, column ``username``,
target row ``id=13915``.

Runs WHERE THE RUNNER RUNS (the A100 self-hosted VWA host), shelling to
``docker exec vwa-reddit ...``. On a dev box without the container it fails
soft (logged, non-fatal) unless ``fail_closed=True``; set
``reddit_identity_reset.enabled=false`` to silence on dev.

A100 live-verified 2026-06-25 (scripts/maintenance/verify_reddit_identity_fix.sh):
postmill's ``users`` table DOES carry a lowercase canonical ``normalized_username``
column and login matches against IT — so a ``username``-only restore does NOT
restore login. This helper therefore restores BOTH columns. With the correct
two-column simulation (rename username→Patrick AND normalized_username→patrick)
the deployed restore yields a fresh login as MarvelsGrantMan136 = LOGIN_OK.
"""
from __future__ import annotations

import logging
import shlex
import subprocess
from typing import Optional

_logger = logging.getLogger(__name__)

# Verified defaults (笔记 §354). Every value is overridable via the
# ``reddit_identity_reset`` config block so a schema surprise found at A100
# verification time is a one-line config change, not a code change.
_DEFAULTS = {
    "enabled": True,
    "container": "vwa-reddit",
    "db": "postmill",
    "db_os_user": "postgres",      # su - <user> for unix-socket peer auth
    "table": "users",
    "username_column": "username",
    # Postmill stores a lowercase canonical form in `normalized_username` and
    # matches login against IT (User::setUsername sets both; the loader queries
    # WHERE normalized_username = lower(:input)). Verified on A100 2026-06-25:
    # restoring `username` ALONE does NOT restore login — `normalized_username`
    # MUST be restored too, else fresh login still fails. Set the column name to
    # "" to skip (non-postmill schema).
    "normalized_username_column": "normalized_username",
    "seed_normalized_username": "",   # "" → auto = seed_username.lower()
    "user_id": 13915,
    "seed_username": "MarvelsGrantMan136",
    "sql_override": "",            # if set, used verbatim (overrides all column logic)
    "timeout_s": 30,
    "fail_closed": False,          # a transient docker hiccup must not kill the run by default
}


def _resolve(cfg: Optional[dict]) -> dict:
    out = dict(_DEFAULTS)
    out.update((cfg or {}).get("reddit_identity_reset", {}) or {})
    return out


def _sqlq(s: str) -> str:
    """SQL string literal escaping (double single-quotes)."""
    return str(s).replace("'", "''")


def build_restore_sql(c: dict) -> str:
    """Idempotent UPDATE that restores the seed identity (no-op when correct).

    Restores BOTH ``username`` and (when configured) postmill's lowercase
    canonical ``normalized_username`` — login matches the latter, so a
    username-only restore does NOT fix login (verified A100 2026-06-25). The
    idempotency guard fires if EITHER column has drifted. SQL-literal single
    quotes are doubled; identifiers come from config (not user input). An
    explicit ``sql_override`` wins verbatim.
    """
    if c.get("sql_override"):
        return str(c["sql_override"])
    table = str(c["table"])
    uid = int(c["user_id"])
    col = str(c["username_column"])
    seed = _sqlq(c["seed_username"])
    sets = [f"{col}='{seed}'"]
    guard = [f"{col}<>'{seed}'"]
    ncol = str(c.get("normalized_username_column") or "")
    if ncol:
        nseed = _sqlq(c.get("seed_normalized_username") or str(c["seed_username"]).lower())
        sets.append(f"{ncol}='{nseed}'")
        guard.append(f"{ncol}<>'{nseed}'")
    return (
        f"UPDATE {table} SET {', '.join(sets)} "
        f"WHERE id={uid} AND ({' OR '.join(guard)})"
    )


def build_docker_argv(c: dict) -> list[str]:
    """``docker exec <container> bash -lc "su - postgres -c 'psql -d postmill -c <sql>'"``."""
    sql = build_restore_sql(c)
    inner = "psql -d {db} -v ON_ERROR_STOP=1 -c {sql}".format(
        db=shlex.quote(str(c["db"])), sql=shlex.quote(sql),
    )
    su_cmd = "su - {user} -c {inner}".format(
        user=shlex.quote(str(c["db_os_user"])), inner=shlex.quote(inner),
    )
    return ["docker", "exec", str(c["container"]), "bash", "-lc", su_cmd]


def restore_reddit_identity(
    cfg: Optional[dict], *, logger: Optional[logging.Logger] = None
) -> bool:
    """Idempotently restore the reddit test account username to its seed value.

    Returns True on success (including the idempotent ``UPDATE 0`` no-op),
    False on failure. Never raises unless ``reddit_identity_reset.fail_closed``
    is set (then the subprocess error / non-zero rc is re-raised).
    """
    log = logger or _logger
    c = _resolve(cfg)
    if not c.get("enabled", True):
        return True
    argv = build_docker_argv(c)
    try:
        r = subprocess.run(
            argv, capture_output=True, text=True, timeout=int(c["timeout_s"]),
        )
    except Exception as exc:  # docker missing (dev box), timeout, etc.
        log.warning(
            "reddit_identity_reset: could not run restore (%s): %s",
            type(exc).__name__, exc,
        )
        if c.get("fail_closed"):
            raise
        return False
    if r.returncode == 0:
        # psql prints "UPDATE 1" (was renamed → healed) or "UPDATE 0" (already correct).
        log.info("reddit_identity_reset: ok (%s)", (r.stdout or "").strip() or "no output")
        return True
    log.warning(
        "reddit_identity_reset: restore rc=%d stderr=%s",
        r.returncode, (r.stderr or "")[-300:],
    )
    if c.get("fail_closed"):
        raise RuntimeError(
            f"reddit_identity_reset failed rc={r.returncode}: {(r.stderr or '')[-300:]}"
        )
    return False
