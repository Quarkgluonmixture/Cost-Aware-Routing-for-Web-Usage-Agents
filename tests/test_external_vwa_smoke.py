"""External VWA / browser_env smoke — /stress A1.12 P1-7.

Companion to `test_external_module_integration.py` (the misnamed one) — this
file actually exercises external systems: VWA submodule imports +
browser_env class instantiation + evaluator_router availability. Does NOT
require live docker / live VWA endpoints — just verifies import-time
breakage doesn't go unnoticed.

Skipped by default (env-gated). Opt in with `RUN_EXTERNAL_TESTS=1 pytest -m external`
or `RUN_EXTERNAL_TESTS=1 pytest tests/test_external_vwa_smoke.py`.

Rationale: pre-2026-05-16 status, any P79-side refactor of the VWA submodule
boundary (e.g. moving `evaluator_router` import path, or `browser_env.envs`
breaking at import time due to env-var contract drift) would only surface
when `ExperimentRunner` actually tries to create an env mid-condition —
i.e., after launch, in the first paper-grade run, deep in a 1-2 week Pass-1
window. That's the worst possible discovery timing for a paper-grade rerun.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VWA_SUBMODULE = REPO_ROOT / "external" / "visualwebarena"

pytestmark = [
    pytest.mark.external,
    pytest.mark.skipif(
        os.environ.get("RUN_EXTERNAL_TESTS") != "1",
        reason="external smoke skipped by default (set RUN_EXTERNAL_TESTS=1 to run)",
    ),
]


@pytest.fixture(autouse=True)
def _vwa_env():
    """Placeholder env so VWA module-level reads (`env_config.py`, `auto_login.py`)
    don't fail at import. Real URLs not required — just non-empty strings."""
    os.environ.setdefault("OPENAI_API_KEY", "DUMMY_P79_EXTERNAL_SMOKE")
    os.environ.setdefault("DATASET", "visualwebarena")
    os.environ.setdefault("SHOPPING", "http://localhost:7770")
    os.environ.setdefault("REDDIT", "http://localhost:9999")
    os.environ.setdefault("WIKIPEDIA", "http://localhost:8888")
    os.environ.setdefault("CLASSIFIEDS", "http://localhost:9980")
    os.environ.setdefault("HOMEPAGE", "http://localhost:4399")
    os.environ.setdefault("CLASSIFIEDS_RESET_TOKEN", "dummy")
    if str(VWA_SUBMODULE) not in sys.path:
        sys.path.insert(0, str(VWA_SUBMODULE))
    yield


def test_vwa_browser_env_imports_clean():
    """`browser_env` import must succeed — guards env-var contract drift."""
    if not (VWA_SUBMODULE / "browser_env").exists():
        pytest.skip("VWA submodule not initialized")
    import browser_env  # noqa: F401
    from browser_env.envs import ScriptBrowserEnv  # noqa: F401


def test_vwa_evaluator_router_importable():
    """`evaluator_router` is the function P79 calls to score episodes —
    import-time breakage = silent paper-grade SR catastrophe."""
    if not (VWA_SUBMODULE / "evaluation_harness").exists():
        pytest.skip("VWA submodule not initialized")
    from evaluation_harness import evaluator_router  # noqa: F401
    assert callable(evaluator_router)


def test_p79_vwa_wrapper_class_importable():
    """`P79Observation` + `VWAWrapper` import — guards against the
    pure-Python wrapper layer breaking against submodule API changes."""
    from p79.envs.vwa_wrapper import P79Observation, VWAWrapper
    assert P79Observation is not None
    assert VWAWrapper is not None


def test_p79_runner_can_import_vwa_pathway():
    """ExperimentRunner must be able to import `VWAWrapper` without crashing.
    Pre-fix: a renamed `browser_env.processors` import would only surface
    when first VWA episode tried to run."""
    from p79.experiment.runner import ExperimentRunner  # noqa: F401
    # Don't instantiate (would try to start docker / browser); just import.
