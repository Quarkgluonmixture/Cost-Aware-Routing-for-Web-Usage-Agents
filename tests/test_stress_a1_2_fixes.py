"""Invariant tests for /stress A1.2 v8 Commit D fixes (B-144 ~ B-155).

Each test pins one specific contract that, once it regresses, would
silently break cross-baseline parity or paper-grade reproducibility.
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# B-144 — multi-seed backend cache uses (backend_id, seed) tuple key
# ---------------------------------------------------------------------------


def test_runner_backend_cache_keyed_by_backend_id_and_seed():
    """The ``_backends`` cache key must be ``(backend_id, int(self.seed))`` so a
    seed switch in the run-loop forces backend reconstruction with the new
    seed propagated to agent_cfg. Pre-fix: ``self._backends`` was keyed on
    ``backend_id`` only, freezing the first seed into the cached agent."""
    src = (REPO_ROOT / "p79/experiment/runner/main.py").read_text(encoding="utf-8")

    # Cache declaration uses Tuple[str, int] type hint
    assert re.search(
        r"self\._backends:\s*Dict\[Tuple\[str,\s*int\],\s*Any\]\s*=\s*\{\}",
        src,
    ), "self._backends must be Dict[Tuple[str, int], Any] keyed by (backend_id, seed)"

    # _get_backend builds the cache_key from (backend_id, int(self.seed))
    assert re.search(
        r"cache_key\s*=\s*\(backend_id,\s*int\(self\.seed\)\)",
        src,
    ), "_get_backend must derive cache_key from (backend_id, int(self.seed))"

    # Cache store + lookup both use cache_key
    assert "self._backends[cache_key] = backend" in src
    assert "if cache_key in self._backends" in src


# ---------------------------------------------------------------------------
# B-145 — GLM fallback default disabled; deprecation surfaced
# ---------------------------------------------------------------------------


def test_glm_fallback_default_disabled_in_base_yaml():
    """``configs/exp_v2_base.yaml`` MUST default ``use_glm_fallback: false`` so
    paper-grade runs cannot accidentally pick up B0's cross-baseline-asymmetric
    parse-error recovery model. Explicit yaml override is still permitted for
    proxy parse-rate investigation only."""
    yaml = (REPO_ROOT / "configs/exp_v2_base.yaml").read_text(encoding="utf-8")
    # Find the api_strong block's use_glm_fallback line (defensively allow
    # arbitrary whitespace + comment lines between).
    m = re.search(r"^\s*use_glm_fallback:\s*(true|false)\s*$", yaml, flags=re.MULTILINE)
    assert m is not None, "use_glm_fallback line not found in exp_v2_base.yaml"
    assert m.group(1) == "false", (
        f"use_glm_fallback default must be false (got {m.group(1)}); "
        "this is the cross-baseline cost-fairness contract."
    )


def test_proxy_agent_raises_on_glm_fallback_enabled():
    """B-991 (2026-05-17): GLM fallback fully retired. Enabling
    ``use_glm_fallback: true`` MUST raise RuntimeError at construction
    (no more DeprecationWarning path — the legacy code is deleted).
    AWS proxy now supports native tool_calling so parse-error rescue
    is unnecessary."""
    src = (REPO_ROOT / "p79/agents/proxy_api_agent.py").read_text(encoding="utf-8")
    # Source contract: explicit raise on use_glm_fallback=true
    assert "use_glm_fallback=true is no longer supported" in src, (
        "B-991 retire contract missing — agent must reject use_glm_fallback=true."
    )
    # The legacy GLM call method must be deleted
    assert "_call_glm_extract" not in src or "RETIRED" in src, (
        "Legacy _call_glm_extract should be deleted (B-991 retire)."
    )


# ---------------------------------------------------------------------------
# B-146 — Gemma agent decoupled from Qwen heavy imports
# ---------------------------------------------------------------------------


def test_gemma_agent_does_not_import_qwen_agent_class():
    """``p79/agents/gemma3vl_agent.py`` must NOT import ``Qwen3VLAgent`` —
    that pulls in ``transformers.Qwen3VLForConditionalGeneration`` +
    ``qwen_vl_utils`` transitively, defeating the cross-family decoupling.
    Shared helpers live in ``p79/agents/_shared_vl_utils.py`` instead."""
    src = (REPO_ROOT / "p79/agents/gemma3vl_agent.py").read_text(encoding="utf-8")
    # The import statement must be gone (comment references are fine)
    assert not re.search(r"^\s*from\s+p79\.agents\.qwen3vl_agent\s+import",
                         src, flags=re.MULTILINE), (
        "Gemma agent still imports from qwen3vl_agent — this re-introduces the "
        "qwen_vl_utils transitive dep that B-146 fixed."
    )
    assert not re.search(r"Qwen3VLAgent\._(make_dom|make_som|make_vision|format_history|compute_confidence)",
                         src), (
        "Gemma agent must call shared helpers, not Qwen3VLAgent classmethods."
    )

    # Positive check: it imports from the shared module
    assert "from p79.agents._shared_vl_utils import" in src


def test_shared_vl_utils_module_exists_and_exports_six_helpers():
    """The single source of truth for cross-baseline VL helpers."""
    src_path = REPO_ROOT / "p79/agents/_shared_vl_utils.py"
    assert src_path.exists(), "_shared_vl_utils.py missing"
    src = src_path.read_text(encoding="utf-8")
    for fn in ("make_dom_prompt", "make_som_prompt", "make_vision_prompt",
               "format_history", "compute_confidence", "wait_for_vram"):
        assert re.search(rf"^def\s+{fn}\b", src, flags=re.MULTILINE), (
            f"Shared module must export top-level def {fn}()"
        )


def test_shared_module_has_no_heavy_qwen_deps():
    """The shared helpers module must keep dependencies minimal so Gemma /
    cross-family scripts / future B3/B4 can consume it without pulling in
    Qwen-specific packages. Check actual import statements (not docstring
    mentions of the deprecated dep)."""
    src = (REPO_ROOT / "p79/agents/_shared_vl_utils.py").read_text(encoding="utf-8")
    # Forbidden: import lines that pull qwen_vl_utils or Qwen3 model class
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith(("#", '"""', "'''")):
            continue  # comment / docstring line
        if stripped.startswith(("import ", "from ")):
            assert "qwen_vl_utils" not in stripped, (
                f"shared module imports qwen_vl_utils: {stripped!r}"
            )
            assert "Qwen3VLForConditionalGeneration" not in stripped, (
                f"shared module imports Qwen3VLForConditionalGeneration: {stripped!r}"
            )


def test_qwen_classmethods_byte_identical_to_shared_module():
    """Backward-compat invariant: legacy ``Qwen3VLAgent._make_*_prompt()``
    callers (proxy_api_agent, mechanistic scripts, test_agents_prompt_parity)
    must still get the same bytes after the refactor."""
    from p79.agents import _shared_vl_utils
    from p79.agents.qwen3vl_agent import Qwen3VLAgent

    assert Qwen3VLAgent._make_dom_prompt() == _shared_vl_utils.make_dom_prompt()
    assert Qwen3VLAgent._make_som_prompt() == _shared_vl_utils.make_som_prompt()
    assert Qwen3VLAgent._make_vision_prompt() == _shared_vl_utils.make_vision_prompt()
    # _format_history is also byte-equivalent on the empty-history case
    assert Qwen3VLAgent._format_history([]) == _shared_vl_utils.format_history([])


def test_gemma_prompts_byte_identical_to_qwen_via_shared_source():
    """Cross-baseline byte-identical prompts is the paper-grade contract.
    The shared module is the single source — Gemma and Qwen must both
    resolve to the same bytes."""
    from p79.agents import gemma3vl_agent
    from p79.agents.qwen3vl_agent import Qwen3VLAgent

    assert gemma3vl_agent._DOM_PROMPT == Qwen3VLAgent._make_dom_prompt()
    assert gemma3vl_agent._SOM_PROMPT == Qwen3VLAgent._make_som_prompt()
    assert gemma3vl_agent._VISION_PROMPT == Qwen3VLAgent._make_vision_prompt()


# ---------------------------------------------------------------------------
# B-147 — max_new_tokens default aligned 4096 across 3 backend wrappers
# ---------------------------------------------------------------------------


def test_three_backend_wrappers_max_new_tokens_default_aligned():
    """Defense-in-depth: all 3 backend wrappers must default ``max_new_tokens``
    to 4096 so a future config refactor that drops the explicit yaml setting
    cannot silently regress to 512 (which truncates the ~400-1500 tok thought
    + JSON envelope → silent parse_failed)."""
    paths = [
        REPO_ROOT / "p79/backends/api_proxy.py",
        REPO_ROOT / "p79/backends/local_qwen.py",
        REPO_ROOT / "p79/backends/local_gemma.py",
    ]
    defaults = []
    for p in paths:
        src = p.read_text(encoding="utf-8")
        m = re.search(r'config\.get\(\s*"max_new_tokens"\s*,\s*(\d+)\s*\)', src)
        assert m is not None, f"{p.name}: max_new_tokens default not found"
        defaults.append((p.name, int(m.group(1))))
    distinct = set(d for _, d in defaults)
    assert distinct == {4096}, (
        f"Backend wrapper max_new_tokens defaults must all equal 4096; got {defaults}"
    )


# ---------------------------------------------------------------------------
# B-148 — api_key_env allowlist guard
# ---------------------------------------------------------------------------


def test_api_proxy_backend_rejects_arbitrary_api_key_env():
    """Allowlist guard rejects a yaml that tries to redirect ``api_key_env``
    to, e.g., ``AWS_SECRET_ACCESS_KEY``. Without this guard the proxy agent
    would read whatever env var the config requested → secret exfiltration
    vector via verbose error tracing / config logging."""
    from p79.backends.api_proxy import ApiProxyBackend, _ALLOWED_API_KEY_ENVS

    # Sanity: defaults pass
    assert ApiProxyBackend._validate_api_key_env("PROXY_API_KEY") == "PROXY_API_KEY"
    # Adversarial: non-allowlist raises
    with pytest.raises(ValueError, match="not in the allowlist"):
        ApiProxyBackend._validate_api_key_env("AWS_SECRET_ACCESS_KEY")
    with pytest.raises(ValueError, match="not in the allowlist"):
        ApiProxyBackend._validate_api_key_env("OPENAI_API_KEY")

    # Documented allowlist members are accepted
    for allowed in ("PROXY_API_KEY", "DASHSCOPE_API_KEY", "GLM_API_KEY"):
        assert ApiProxyBackend._validate_api_key_env(allowed) == allowed
    # Allowlist surface (in case future test extends it) is the documented set
    assert _ALLOWED_API_KEY_ENVS >= {
        "PROXY_API_KEY", "DASHSCOPE_API_KEY", "GLM_API_KEY",
    }


# ---------------------------------------------------------------------------
# B-149 — image_over_cap threaded via image_meta dict (verified contract)
# ---------------------------------------------------------------------------


def test_image_meta_dict_carries_over_cap_field():
    """Already landed via B-140 (Commit B); this test pins the contract so
    a future refactor cannot accidentally drop ``image_over_cap`` from the
    step_record ``image_meta`` dict — paper §3.5 audit of B0 vision-mode
    over-budget triggers depends on this field being present."""
    src = (REPO_ROOT / "p79/experiment/runner/main.py").read_text(encoding="utf-8")
    assert '"image_over_cap": meta.get("image_over_cap")' in src, (
        "Runner step_record image_meta dict must carry image_over_cap field "
        "for B0 vision-mode payload-budget audit."
    )


# ---------------------------------------------------------------------------
# B-153 — stale comment local_qwen.py:27-28 realigned with B-136 strict mode
# ---------------------------------------------------------------------------


def test_local_qwen_revision_comment_reflects_strict_mode():
    """Comment must no longer claim "falls back to default + warns"; B-136
    made the agent RAISE on missing revision. A reader trusting the stale
    comment could submit a yaml without ``model.revision`` thinking it would
    soft-fail; instead first load would crash."""
    src = (REPO_ROOT / "p79/backends/local_qwen.py").read_text(encoding="utf-8")
    assert "falls back to its default + warns" not in src, (
        "Stale comment still present — should reference B-136 strict mode"
    )
    assert "B-136" in src, "Updated comment should cross-link B-136 strict-mode fix"


# ---------------------------------------------------------------------------
# B-149-mock — api_proxy mock action aligned with local backends (scroll)
# ---------------------------------------------------------------------------


def test_api_proxy_mock_action_aligned_with_local_mocks_scroll():
    """Cross-baseline mock parity: api_proxy mock_mode must emit
    ``scroll [0, 0.8]`` like local_qwen / local_gemma / MockBackend. Pre-fix
    api_proxy emitted ``click element_id=1`` — tests asserting mock parity
    silently failed only on the proxy path.

    B-808 (/stress A1.2 cold-start P2-2-AC, 2026-05-17): removed dead
    ``coordinate_type`` field from mock — scroll uses ``delta``, not
    ``coordinate``. Updated this test to match the new canonical mock
    signature.
    """
    src = (REPO_ROOT / "p79/backends/api_proxy.py").read_text(encoding="utf-8")
    # The mock_mode block must contain scroll + delta (canonical scroll
    # action shape — no coordinate_type which is a dead field for scroll).
    mock_block = re.search(
        r"if\s+self\.mock_mode:\s*\n(?:.*\n){0,30}?\s*return\s+action,",
        src,
    )
    assert mock_block is not None, "mock_mode block not found in api_proxy.py"
    block = mock_block.group(0)
    assert '"action_type": "scroll"' in block, (
        "api_proxy mock should emit scroll action (was click)"
    )
    assert '"delta": [0, 0.8]' in block
    # B-808: coordinate_type was a dead field on scroll (scroll uses delta).
    # Updated mock signature does NOT include coordinate_type.
    assert '"coordinate_type"' not in block, (
        "P2-2-AC: coordinate_type is dead on scroll mock — should have been removed"
    )


# ---------------------------------------------------------------------------
# B-154 — MockBackend backend_type tagged with backend_id
# ---------------------------------------------------------------------------


def test_factory_mockbackend_backend_type_includes_backend_id():
    """factory.MockBackend.backend_type must be ``mock_<backend_id>`` not
    bare ``mock``, so tests dispatching ``type: mock`` directly can still
    tell which baseline is being exercised."""
    from p79.backends.factory import MockBackend
    from p79.backends.base import BackendStepContext

    for bid in ("test_b0", "test_b1", "test_b2"):
        m = MockBackend(bid, {})
        _, meta = m.step("instr", None, BackendStepContext(
            observation_mode="dom", som_enabled=False, som_text=""
        ))
        assert meta["backend_type"] == f"mock_{bid}", (
            f"MockBackend({bid}) backend_type should be 'mock_{bid}', "
            f"got {meta['backend_type']!r}"
        )


# ---------------------------------------------------------------------------
# B-155 — PIL version pinned >=10 in pyproject + import-time guard
# ---------------------------------------------------------------------------


def test_pyproject_pillow_pinned_for_resampling_enum():
    """pyproject must pin ``pillow>=10.0,<12.0`` so ``Image.Resampling.LANCZOS``
    is available + future major bumps cannot silently change resize behaviour
    across DGX / Condenser / Myriad."""
    pyproj = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    # Find pillow line — allow either >=10.0 or ~=10 forms; reject bare "pillow"
    m = re.search(r'"pillow([>=<~,\s\d.]*)"', pyproj)
    assert m is not None, "pillow dependency not found in pyproject.toml"
    spec = m.group(1).strip()
    assert spec, (
        "pillow must have a version specifier (>=10.0,<12.0); bare 'pillow' "
        "leaves cross-host bit-identical reproducibility undefined"
    )
    assert ">=10" in spec.replace(" ", ""), (
        f"pillow pin must require >=10 for Resampling enum; got {spec!r}"
    )


def test_image_utils_asserts_pil_version_at_import():
    """image_utils must guard against a manually-overridden PIL by asserting
    version at import time. Defense-in-depth alongside the pyproject pin.

    B-819 (/stress A1.2 cold-start codex F4 honest-gap, 2026-05-17): replaced
    bare ``assert`` with explicit ``raise RuntimeError`` so ``python -O``
    cannot strip the lower-bound guard. Upper bound warns rather than raises
    until dev env catches up to the pyproject pin (currently Pillow 12.1.1
    is installed, pyproject says ``<12.0``).
    """
    src = (REPO_ROOT / "p79/backends/image_utils.py").read_text(encoding="utf-8")
    assert "_PIL_VERSION_PARTS" in src, (
        "image_utils.py missing PIL version sentinel — see B-155"
    )
    assert "PIL.__version__" in src
    # B-819: lower bound MUST raise (not assert) so python -O cannot strip.
    assert "< (10, 0)" in src or "<(10, 0)" in src, (
        "B-819: lower bound guard must use explicit raise on < (10, 0) — "
        "not bare assert which python -O strips"
    )
    assert "raise RuntimeError" in src, (
        "B-819: lower bound enforced via explicit raise, not assert"
    )
