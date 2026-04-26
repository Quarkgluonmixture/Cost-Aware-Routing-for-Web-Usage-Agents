"""Backward-compatible package for `p79.experiment.runner`.

§97 Step-3 split: the original `runner.py` (1637L) was split into:
  - `helpers.py` — free functions (cycle detection, diagnostic controls, ntfy)
  - `main.py`    — `ExperimentRunner` class

External code that imported `from p79.experiment.runner import ExperimentRunner`
or any of the helper functions continues to work via the re-exports below.
"""
from p79.experiment.runner.main import ExperimentRunner

# Re-export helpers for callers that imported them directly from the original module.
from p79.experiment.runner.helpers import (  # noqa: F401
    _parse_seeds,
    _action_signature,
    _action_signature_soft,
    _detect_action_cycle,
    _sanitize_query_text,
    _query_sanitization_control,
    _repeat_hits_same_target,
    _build_exploration_fallback_action,
    _anti_repeat_control,
    _no_early_finish_control,
    _notify_retry_pass,
)

__all__ = ["ExperimentRunner"]
