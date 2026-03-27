from __future__ import annotations

from p79.utils.asyncio_workarounds import should_downgrade_asyncio_context


class TargetClosedError(Exception):
    pass


def test_should_downgrade_target_closed_future_exception() -> None:
    ctx = {
        "message": "Future exception was never retrieved",
        "exception": TargetClosedError("Target page, context or browser has been closed"),
    }
    assert should_downgrade_asyncio_context(ctx) is True


def test_should_not_downgrade_other_future_exception() -> None:
    ctx = {
        "message": "Future exception was never retrieved",
        "exception": RuntimeError("database is locked"),
    }
    assert should_downgrade_asyncio_context(ctx) is False


def test_should_not_downgrade_non_future_message() -> None:
    ctx = {
        "message": "Task exception was never retrieved",
        "exception": TargetClosedError("Target page, context or browser has been closed"),
    }
    assert should_downgrade_asyncio_context(ctx) is False

