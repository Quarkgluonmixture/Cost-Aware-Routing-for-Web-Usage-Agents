"""B-1884 / Fix 4: reddit shared-account identity restore unit tests.

Covers SQL construction (idempotent + injection-safe quoting + override),
the verified docker-exec command shape, the enabled gate, and the
soft-fail / fail-closed subprocess contract. No live DB / docker needed.
"""
from unittest import mock

import pytest

from p79.utils.reddit_identity import (
    build_restore_sql,
    build_docker_argv,
    restore_reddit_identity,
    _resolve,
    _DEFAULTS,
)


def test_default_sql_restores_both_columns_idempotently():
    sql = build_restore_sql(_resolve(None))
    # both username AND the postmill canonical column must be restored
    # (A100-verified: login matches normalized_username, not username)
    assert "username='MarvelsGrantMan136'" in sql
    assert "normalized_username='marvelsgrantman136'" in sql  # auto-lowercased
    assert "WHERE id=13915" in sql
    # idempotent guard fires if EITHER column drifted
    assert "username<>'MarvelsGrantMan136'" in sql
    assert "normalized_username<>'marvelsgrantman136'" in sql
    assert " OR " in sql


def test_normalized_column_disabled_falls_back_to_username_only():
    c = _resolve({"reddit_identity_reset": {"normalized_username_column": ""}})
    sql = build_restore_sql(c)
    assert "normalized_username" not in sql
    assert "SET username='MarvelsGrantMan136' WHERE id=13915" in sql


def test_explicit_seed_normalized_overrides_autolower():
    c = _resolve({"reddit_identity_reset": {"seed_normalized_username": "CUSTOM"}})
    assert "normalized_username='CUSTOM'" in build_restore_sql(c)


def test_sql_override_wins_verbatim():
    c = _resolve({"reddit_identity_reset": {"sql_override": "UPDATE users SET x=1"}})
    assert build_restore_sql(c) == "UPDATE users SET x=1"


def test_seed_username_single_quote_is_doubled():
    c = _resolve({"reddit_identity_reset": {"seed_username": "O'Brien"}})
    sql = build_restore_sql(c)
    # SQL-literal escaping: ' -> '' (no dangling/odd quote that breaks the statement)
    assert "username='O''Brien'" in sql


def test_docker_argv_matches_verified_path():
    argv = build_docker_argv(_resolve(None))
    # verified (笔记 §354): docker exec vwa-reddit bash -lc "su - postgres -c '...'"
    assert argv[:5] == ["docker", "exec", "vwa-reddit", "bash", "-lc"]
    joined = argv[5]
    assert joined.startswith("su - postgres -c ")
    assert "psql -d postmill" in joined
    assert "ON_ERROR_STOP=1" in joined


def test_disabled_returns_true_without_subprocess():
    with mock.patch("p79.utils.reddit_identity.subprocess.run") as m:
        ok = restore_reddit_identity({"reddit_identity_reset": {"enabled": False}})
    assert ok is True
    m.assert_not_called()


def test_success_returns_true():
    fake = mock.Mock(returncode=0, stdout="UPDATE 1\n", stderr="")
    with mock.patch("p79.utils.reddit_identity.subprocess.run", return_value=fake):
        assert restore_reddit_identity({}) is True


def test_idempotent_noop_update0_is_success():
    fake = mock.Mock(returncode=0, stdout="UPDATE 0\n", stderr="")
    with mock.patch("p79.utils.reddit_identity.subprocess.run", return_value=fake):
        assert restore_reddit_identity({}) is True


def test_nonzero_rc_soft_fails_by_default():
    fake = mock.Mock(returncode=1, stdout="", stderr="psql: error")
    with mock.patch("p79.utils.reddit_identity.subprocess.run", return_value=fake):
        assert restore_reddit_identity({}) is False


def test_docker_missing_soft_fails_by_default():
    with mock.patch(
        "p79.utils.reddit_identity.subprocess.run",
        side_effect=FileNotFoundError("docker not found"),
    ):
        assert restore_reddit_identity({}) is False


def test_fail_closed_reraises_on_nonzero_rc():
    fake = mock.Mock(returncode=1, stdout="", stderr="boom")
    with mock.patch("p79.utils.reddit_identity.subprocess.run", return_value=fake):
        with pytest.raises(RuntimeError):
            restore_reddit_identity({"reddit_identity_reset": {"fail_closed": True}})


def test_fail_closed_reraises_on_subprocess_error():
    with mock.patch(
        "p79.utils.reddit_identity.subprocess.run",
        side_effect=TimeoutError("timeout"),
    ):
        with pytest.raises(TimeoutError):
            restore_reddit_identity({"reddit_identity_reset": {"fail_closed": True}})


def test_config_block_present_in_defaults():
    # CLAUDE.md: new config fields must have a DEFAULT_CONFIG default
    from p79.experiment.config import DEFAULT_CONFIG
    assert "reddit_identity_reset" in DEFAULT_CONFIG
    assert DEFAULT_CONFIG["reddit_identity_reset"]["user_id"] == 13915
    assert _DEFAULTS["seed_username"] == "MarvelsGrantMan136"
