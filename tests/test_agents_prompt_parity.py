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

3. ProxyApiAgent (B0) is structurally independent (its _get_system_prompts
   builds its own copies) — we only check the *high-level shape* across B0
   vs B1/B2: the same set of mode keys exists in each. Byte-equality on the
   prompt text between B0 and B1/B2 does NOT hold (B0 prompt was authored
   separately, with documented differences — see paper §3 disclosure footnote)
   so we only assert the mode dispatch keys match, not the prompt bodies.
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


def test_b0_b1_b2_mode_dispatch_keys_identical():
    """All three baselines must dispatch the same 7 observation modes.

    Byte-equality of prompt TEXT does not hold between B0 (proxy) and B1/B2
    (local) — B0 prompts are authored separately (see paper §3 footnote).
    But the mode dispatch table keys must agree: any baseline missing a mode
    key silently degrades to the DOM fallback.
    """
    from p79.agents.proxy_api_agent import ProxyApiAgent

    expected_modes = {
        "dom", "som", "vision",
        "phantom_som", "phantom_dom", "phantom_text", "phantom_prompt",
    }

    # B0 — instantiate ProxyApiAgent's static _get_system_prompts via a minimal
    # config that won't trigger any network / model load. _get_system_prompts is
    # an instance method but uses no instance state; we call it via __get__.
    proxy_prompts = ProxyApiAgent._get_system_prompts.__get__(
        type("_StubProxy", (), {})()
    )()
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
