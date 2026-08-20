"""B-1985: every key the agent reads from `model_cfg` must be forwarded by the backend.

`ApiProxyBackend` hands the agent an explicitly-built `agent_cfg["model"]` dict. That
is an allowlist, and an allowlist silently drops whatever it does not name: a yaml key
that no one forwards is obeyed by no one, and nothing fails — the run just behaves as
though the operator had never written the line.

This has now happened twice. B-340 fixed it for `paper_grade`; B-1985 found
`structured_output`, `logprobs_unavailable` and `image_format` still dropped, which
sent every B5 episode down the `tools` road it was configured to avoid and killed the
smoke at step 0 with HTTP 400.

Rather than trust a hand-maintained list, derive BOTH sides from the source: the keys
the agent asks for, and the keys the backend supplies. Adding a `model_cfg.get("x")`
to the agent without forwarding `x` fails here instead of mid-fire.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AGENT = REPO / "p79" / "agents" / "proxy_api_agent.py"
BACKEND = REPO / "p79" / "backends" / "api_proxy.py"


def _keys_agent_reads() -> set[str]:
    return set(re.findall(r'model_cfg\.get\(\s*"([a-z0-9_]+)"', AGENT.read_text()))


def _keys_backend_forwards() -> set[str]:
    src = BACKEND.read_text()
    start = src.index('agent_cfg = {')
    block = src[start:src.index('"agent": {', start)]
    return set(re.findall(r'^\s*"([a-z0-9_]+)"\s*:', block, re.M))


def test_no_agent_model_key_is_silently_dropped():
    reads, forwards = _keys_agent_reads(), _keys_backend_forwards()
    assert reads, "parser found no model_cfg.get() calls — the regex has drifted"
    assert forwards, "parser found no forwarded keys — the block markers have drifted"
    dropped = sorted(reads - forwards)
    assert not dropped, (
        "ApiProxyBackend does not forward these keys, so a yaml setting them is "
        f"silently ignored: {dropped}. Add them to agent_cfg['model'] in "
        "p79/backends/api_proxy.py (B-1985; B-340 is the same shape)."
    )


def test_b5_structured_output_reaches_the_agent():
    """The specific key whose absence cost the 2026-08-20 B5 smoke."""
    forwards = _keys_backend_forwards()
    for key in ("structured_output", "logprobs_unavailable"):
        assert key in forwards, f"{key} must be forwarded — see B-1985"
