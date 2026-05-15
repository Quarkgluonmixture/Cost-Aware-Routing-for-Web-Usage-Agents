"""Cross-baseline prompt parity invariants for B0 / B1 / Gemma3-VL.

Paper-grade requirement: all three baselines must see byte-identical system
prompts per observation mode for cross-model comparison validity.

These tests guard the structural mechanism (NOT just the snapshot value) so
future regressions surface as CI failures instead of silent paper-grade drift:

1. Qwen3VLAgent._make_*_prompt are @staticmethod — they do NOT depend on a
   bound `self` to evaluate. Regressing to instance methods would re-introduce
   the `_make_dom_prompt(None)` bound-method-on-None pattern in Gemma's module
   load, which silently breaks the moment any future maintainer adds a
   `self.X` reference inside the prompt body.

2. Gemma3VLAgent's module-level _DOM/_SOM/_VISION_PROMPT constants are
   byte-identical to Qwen3VLAgent's @staticmethod outputs.

3. ProxyApiAgent (B0) reuses Qwen3VLAgent's @staticmethod prompt builders
   directly (/stress A1.1 F1 fix, 2026-05-15) — B0/B1/B2 are now byte-identical
   prompt-for-prompt across all 7 modes. The legacy B0 prompt had shopping-
   specific examples ("Blankets & Throws" / "Home & Kitchen" / "Electronics" /
   "Jewelry & Watches") that leaked a shopping-domain prior into the cross-
   baseline comparison on classifieds + reddit. The fix removes the leak by
   making B0 share the same source-of-truth as B1/B2.
"""

from __future__ import annotations


def test_qwen_prompt_methods_are_staticmethod():
    """Regression guard: _make_*_prompt must remain @staticmethod.

    If someone reverts to `def _make_dom_prompt(self)`, the Gemma module-load
    snapshot pattern silently re-introduces the bound-method-on-None fragility.
    """
    from p79.agents.qwen3vl_agent import Qwen3VLAgent

    for name in ("_make_dom_prompt", "_make_som_prompt", "_make_vision_prompt"):
        descriptor = Qwen3VLAgent.__dict__.get(name)
        assert isinstance(descriptor, staticmethod), (
            f"{name} must be @staticmethod so Gemma's module-load reuse does "
            f"not depend on self=None happening to work; got {type(descriptor)!r}"
        )


def test_qwen_prompts_callable_without_self():
    """Smoke: @staticmethod descriptors are callable without an instance."""
    from p79.agents.qwen3vl_agent import Qwen3VLAgent

    dom = Qwen3VLAgent._make_dom_prompt()
    som = Qwen3VLAgent._make_som_prompt()
    vis = Qwen3VLAgent._make_vision_prompt()
    assert isinstance(dom, str) and len(dom) > 100
    assert isinstance(som, str) and len(som) > 100
    assert isinstance(vis, str) and len(vis) > 100
    # The three prompts are not the same — guard against silent merging.
    assert dom != som != vis


def test_gemma_module_constants_match_qwen_staticmethod_output():
    """B1 / Gemma3-VL byte-identical prompt invariant.

    Gemma's module-level _DOM/_SOM/_VISION_PROMPT are evaluated once at module
    load. This test re-evaluates the Qwen staticmethods at test time and
    compares — catches drift if anyone edits one side without the other.
    """
    from p79.agents.qwen3vl_agent import Qwen3VLAgent
    from p79.agents import gemma3vl_agent

    assert gemma3vl_agent._DOM_PROMPT == Qwen3VLAgent._make_dom_prompt()
    assert gemma3vl_agent._SOM_PROMPT == Qwen3VLAgent._make_som_prompt()
    assert gemma3vl_agent._VISION_PROMPT == Qwen3VLAgent._make_vision_prompt()


def test_no_legacy_self_or_none_prompt_callsites():
    """/stress A1.1 B-92 propagation: no caller may pass `self` or `None` to
    the @staticmethod prompt methods anywhere in p79/ or scripts/.

    The original A1.1 B-92 fix swept agents but missed mechanistic files —
    `extract_hidden_states.py:73-82` + cross-family scripts at lines
    `run_stage4_h1_qwen2vl.py:153-154` and `run_stage4_h1_phi35.py:179-180`
    still passed `self` / `None`, which would now TypeError on instantiation
    (Mechanism §5 is paused so this stayed latent, but
    /stress A1.4 caught it).
    """
    import re
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    pattern = re.compile(r"_make_(dom|som|vision)_prompt\((self|None)\)")
    hits: list[str] = []
    for root in [repo / "p79", repo / "scripts"]:
        for path in root.rglob("*.py"):
            for i, line in enumerate(open(path), start=1):
                if pattern.search(line):
                    hits.append(f"{path.relative_to(repo)}:{i}: {line.rstrip()}")
    assert not hits, (
        "Legacy `_make_*_prompt(self|None)` callsites must be removed — these "
        "would TypeError against the @staticmethod descriptor:\n  "
        + "\n  ".join(hits)
    )


def test_mechanistic_build_user_text_has_accessibility_tree_prefix():
    """/stress A1.4 B-103 fix: mechanistic / cross-family `_build_user_text`
    must prepend `Accessibility Tree:\\n` for DOM-style modes so NPZ input
    is byte-identical to production agent (qwen3vl_agent.py:441-450).
    """
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    files = [
        repo / "p79" / "mechanistic" / "extract_hidden_states.py",
        repo / "scripts" / "mechanistic" / "run_stage4_h1_qwen2vl.py",
        repo / "scripts" / "mechanistic" / "run_stage4_h1_phi35.py",
    ]
    for f in files:
        src = f.read_text()
        assert "Accessibility Tree:" in src, (
            f"{f.relative_to(repo)}: missing 'Accessibility Tree:' prefix — "
            f"NPZ input would not be byte-identical to production agent "
            f"for `dom` / `phantom_prompt` modes (B-103 regression)."
        )


def test_agent_layer_strict_rejects_unknown_mode():
    """/stress A1.4 F2 defense-in-depth: agent step() must raise on typo mode.

    Layer 1 (som.py.prepare_observation_for_mode) catches typo modes first,
    but if any code path bypasses som.py and calls agent.step directly with
    a typo'd observation_mode, the agent must also raise rather than silently
    fall through to the DOM prompt. This test verifies the strict-mode check
    is in the agent step body for all 3 baselines by inspecting source.
    """
    import p79.agents.qwen3vl_agent as qwen
    import p79.agents.proxy_api_agent as proxy
    import p79.agents.gemma3vl_agent as gemma

    for mod, name in [(qwen, "qwen3vl_agent"), (proxy, "proxy_api_agent"), (gemma, "gemma3vl_agent")]:
        src = open(mod.__file__).read()
        # Must contain the explicit raise on unknown mode
        assert "Unknown observation_mode" in src, (
            f"{name}: missing strict mode-validation raise. "
            f"silent .get(mode, dom_default) is paper-grade defense gap (/stress A1.4 F2)."
        )
        # Must NOT still have the silent .get(observation_mode, ..."dom"]) pattern
        assert '_system_prompts.get(observation_mode' not in src, (
            f"{name}: still uses silent .get fallback for system_prompts dispatch — "
            f"replace with strict raise on unknown mode."
        )


def test_b0_b1_b2_mode_dispatch_keys_identical():
    """All three baselines must dispatch the same 7 observation modes."""
    from p79.agents.proxy_api_agent import ProxyApiAgent

    expected_modes = {
        "dom", "som", "vision",
        "phantom_som", "phantom_dom", "phantom_text", "phantom_prompt",
    }

    # B0 — _get_system_prompts is @staticmethod after F1 fix, callable directly.
    proxy_prompts = ProxyApiAgent._get_system_prompts()
    assert set(proxy_prompts.keys()) == expected_modes, (
        f"B0 mode dispatch missing keys: {expected_modes - set(proxy_prompts.keys())}"
    )

    # B1 / B2 — their _system_prompts dict is built in __init__ which loads a
    # model, so we cannot instantiate here. Instead verify the dict-build
    # source code statically by reading the module — the mode-key list is the
    # invariant we guard against silent removal.
    import p79.agents.qwen3vl_agent as qwen_mod
    import p79.agents.gemma3vl_agent as gemma_mod

    qwen_src = open(qwen_mod.__file__).read()
    gemma_src = open(gemma_mod.__file__).read()
    for mode in expected_modes:
        assert f'"{mode}":' in qwen_src, f"B1 (Qwen) source missing mode key {mode!r}"
        assert f'"{mode}":' in gemma_src, f"B2 (Gemma) source missing mode key {mode!r}"


def test_b0_prompts_byte_identical_to_b1_b2():
    """/stress A1.1 F1 fix invariant: B0/B1/B2 system prompts must be byte-
    identical per observation mode after 2026-05-15.

    Previously B0 had its own shopping-specific f-string-templated prompt body
    (cf. ``proxy_api_agent.py:_get_system_prompts`` legacy implementation)
    with leaked examples like "Blankets & Throws" / "Home & Kitchen" /
    "Electronics" / "Jewelry & Watches". On the Phase 1a classifieds + reddit
    workload these injected a shopping-domain prior into the B0 baseline
    alone, breaking the cross-baseline prompt-parity invariant asserted at
    ``gemma3vl_agent.py:23-27``.

    The fix collapses B0's prompts to ``Qwen3VLAgent._make_*_prompt()`` so
    all three baselines see the same source-of-truth prompt body. This test
    guards against silent regression — if anyone re-introduces a B0-only
    prompt branch, CI fails before paper-grade data lands.
    """
    from p79.agents.proxy_api_agent import ProxyApiAgent
    from p79.agents.qwen3vl_agent import Qwen3VLAgent

    proxy_prompts = ProxyApiAgent._get_system_prompts()

    # DOM, SoM, Vision must be byte-identical to the Qwen3 source-of-truth.
    assert proxy_prompts["dom"] == Qwen3VLAgent._make_dom_prompt(), (
        "B0 DOM prompt drifted from B1 — re-check ProxyApiAgent._get_system_prompts"
    )
    assert proxy_prompts["som"] == Qwen3VLAgent._make_som_prompt(), (
        "B0 SoM prompt drifted from B1 — re-check ProxyApiAgent._get_system_prompts"
    )
    assert proxy_prompts["vision"] == Qwen3VLAgent._make_vision_prompt(), (
        "B0 vision prompt drifted from B1 — re-check ProxyApiAgent._get_system_prompts"
    )

    # Phantom mode dispatch must mirror B1's choices: phantom_dom/phantom_text
    # share the DOM prompt, phantom_som/phantom_prompt share the SoM prompt.
    assert proxy_prompts["phantom_dom"] == proxy_prompts["dom"]
    assert proxy_prompts["phantom_text"] == proxy_prompts["dom"]
    assert proxy_prompts["phantom_som"] == proxy_prompts["som"]
    assert proxy_prompts["phantom_prompt"] == proxy_prompts["som"]


def test_b0_prompts_have_no_shopping_specific_examples():
    """/stress A1.1 F1 fix invariant: B0 prompts must not contain the legacy
    shopping-domain examples that leaked a baseline-specific prior on the
    classifieds + reddit Phase 1a workload.
    """
    from p79.agents.proxy_api_agent import ProxyApiAgent

    proxy_prompts = ProxyApiAgent._get_system_prompts()
    legacy_leaks = (
        "Blankets & Throws",
        "Home & Kitchen",
        "Jewelry & Watches",
    )
    for mode, prompt in proxy_prompts.items():
        for leak in legacy_leaks:
            assert leak not in prompt, (
                f"B0 prompt for mode {mode!r} still contains legacy shopping "
                f"example {leak!r} — F1 leak re-introduced. See "
                f"/stress A1.1 cross-AI diff 2026-05-15."
            )
