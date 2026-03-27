from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_PATCH_INSTALLED_ATTR = "_p79_target_closed_handler_installed"


def _is_target_closed_exception(exc: Optional[BaseException]) -> bool:
    if exc is None:
        return False

    name = exc.__class__.__name__.lower()
    text = str(exc).lower()
    module = exc.__class__.__module__.lower()
    return (
        "targetclosederror" in name
        or "target page, context or browser has been closed" in text
        or ("playwright" in module and "closed" in text)
    )


def should_downgrade_asyncio_context(context: Dict[str, Any]) -> bool:
    """
    Return True only for Playwright cleanup noise:
    "Future exception was never retrieved" + TargetClosedError.
    """
    message = str(context.get("message", "")).lower()
    if "future exception was never retrieved" not in message:
        return False

    exc = context.get("exception")
    if isinstance(exc, BaseException) and _is_target_closed_exception(exc):
        return True

    future = context.get("future")
    if future is not None and "targetclosederror" in repr(future).lower():
        return True

    return False


def install_asyncio_target_closed_warning_filter() -> None:
    """
    Patch asyncio loop exception handler globally to downgrade noisy
    Playwright TargetClosedError cleanup logs from ERROR to WARNING.
    """
    if getattr(asyncio.BaseEventLoop, _PATCH_INSTALLED_ATTR, False):
        return

    original_call_exception_handler = asyncio.BaseEventLoop.call_exception_handler

    def _patched_call_exception_handler(self: asyncio.BaseEventLoop, context: Dict[str, Any]) -> None:
        if should_downgrade_asyncio_context(context):
            exc = context.get("exception")
            if isinstance(exc, BaseException):
                logger.warning("Suppressed asyncio TargetClosedError cleanup noise: %s", exc)
            else:
                logger.warning("Suppressed asyncio TargetClosedError cleanup noise.")
            return

        original_call_exception_handler(self, context)

    asyncio.BaseEventLoop.call_exception_handler = _patched_call_exception_handler  # type: ignore[assignment]
    setattr(asyncio.BaseEventLoop, _PATCH_INSTALLED_ATTR, True)

