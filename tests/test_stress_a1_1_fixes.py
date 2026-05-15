"""/stress A1.1 cross-AI diff fix invariants (2026-05-15).

This test file collects the regression guards for the five fixes that
landed in response to the /stress A1.1 + codex Mode B + gemini Mode C
cross-AI audit:

  F1 — B0 prompt parity (delete shopping-domain prior leak)
  C1 — Persist B0 image telemetry into step records
  C2 — Record image-encode failure count in agent meta
  C3 — Drop scroll/back keyword fallback in parse_action_text
  C4 — Stop runner from forwarding top-level Qwen revision into
       non-Qwen-class backends (local_gemma in particular)

F1 + C3 have their own targeted unit tests in test_agents_prompt_parity.py
and test_action_utils.py respectively; this file holds the cross-cutting
schema + source-level invariants for C1 / C2 / C4.
"""
from __future__ import annotations

import re
from pathlib import Path

from p79.experiment.types import StepRecordV2


REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# C1 — StepRecordV2 carries the image_meta optional field
# ---------------------------------------------------------------------------


def test_step_record_v2_has_image_meta_field():
    """C1: StepRecordV2 must expose `image_meta` so the runner can persist
    B0 image telemetry (over_cap / payload_bytes / quality / compressed /
    encode_error). Previously these lived in agent meta but the runner
    dropped them before writing the step record → archived JSONL had no
    way to audit B0 image encode behaviour."""
    fields = {f for f in StepRecordV2.__dataclass_fields__}
    assert "image_meta" in fields, (
        "StepRecordV2 missing `image_meta` field — /stress A1.1 codex C1 "
        "regression. Archived B0 runs cannot be image-audited from JSONL."
    )
    # Optional + default None so existing test fixtures still validate.
    default = StepRecordV2.__dataclass_fields__["image_meta"].default
    assert default is None, "image_meta must default to None (optional schema)"


# ---------------------------------------------------------------------------
# C1 / C2 — runner step_record builder forwards image_meta from agent meta
# ---------------------------------------------------------------------------


def test_runner_persists_image_meta_from_agent_meta():
    """C1: runner.main must lift the image_* keys from agent meta into
    step_record.image_meta. We assert by source-grep rather than by
    instantiating ExperimentRunner (which needs full cfg + site configs +
    a live or mock env) so this stays a fast unit test."""
    src = (REPO_ROOT / "p79" / "experiment" / "runner" / "main.py").read_text()
    # The fix block writes the image_meta payload dict.
    assert 'step_record["image_meta"] = _image_meta_payload' in src, (
        "runner.main no longer writes step_record['image_meta'] — C1 fix "
        "reverted. Image telemetry would be dropped from JSONL again."
    )
    # The block must consider all five image meta keys.
    for key in (
        "image_over_cap",
        "image_payload_bytes",
        "image_quality",
        "image_compressed",
        "image_encode_error",
    ):
        assert f'"{key}"' in src, (
            f"runner.main image_meta block missing key {key!r} — partial "
            f"persistence will leave that audit dimension blind."
        )


def test_proxy_agent_meta_carries_image_encode_error_field():
    """C2: ProxyApiAgent.step() must write `image_encode_error` into meta
    so the runner can persist it and downstream audit can detect silent
    text-only episodes (encode-fail path)."""
    src = (REPO_ROOT / "p79" / "agents" / "proxy_api_agent.py").read_text()
    # Counter is incremented in both reference-image and screenshot catch.
    assert "_image_encode_error_count += 1" in src, (
        "ProxyApiAgent no longer increments _image_encode_error_count — "
        "C2 fix reverted. Silent text-only episodes become invisible again."
    )
    # The meta dict surfaces it under the documented key.
    assert '"image_encode_error":' in src, (
        "ProxyApiAgent meta missing `image_encode_error` field — C2 fix "
        "reverted. Encode-failure rate becomes unauditable from JSONL."
    )


# ---------------------------------------------------------------------------
# C4 — runner only forwards top-level revision into Qwen-class backends
# ---------------------------------------------------------------------------


def test_runner_revision_forward_is_qwen_class_gated():
    """C4: top-level cfg.model.revision is a Qwen-specific historical
    convention (paper-grade lock manifest pins the Qwen SHA there).
    Forwarding into a Gemma3 backend with revision=None would silently
    overwrite the B2 base model SHA — codex Mode B Q3 caught this leak.
    """
    src = (REPO_ROOT / "p79" / "experiment" / "runner" / "main.py").read_text()
    # The gating set must be present.
    assert "_QWEN_CLASS_BACKEND_TYPES" in src, (
        "runner.main missing _QWEN_CLASS_BACKEND_TYPES gate — C4 fix "
        "reverted. Top-level Qwen revision could leak into local_gemma."
    )
    # The set must contain exactly the two Qwen-class backends.
    set_decl = re.search(
        r"_QWEN_CLASS_BACKEND_TYPES\s*=\s*\{([^}]*)\}", src
    )
    assert set_decl is not None, "Could not locate _QWEN_CLASS_BACKEND_TYPES literal"
    types_seen = {
        s.strip().strip('"').strip("'")
        for s in set_decl.group(1).split(",")
        if s.strip()
    }
    assert types_seen == {"local_qwen", "api_proxy"}, (
        f"_QWEN_CLASS_BACKEND_TYPES drifted to {types_seen!r}; expected "
        f"{{'local_qwen', 'api_proxy'}}. Adding more types here re-opens "
        f"the cross-backend revision leak surface."
    )
    # The conditional must include the backend-type check.
    assert "_backend_type in _QWEN_CLASS_BACKEND_TYPES" in src, (
        "runner.main revision-forward conditional no longer checks backend "
        "type — C4 fix partially reverted; leak path re-opened."
    )
