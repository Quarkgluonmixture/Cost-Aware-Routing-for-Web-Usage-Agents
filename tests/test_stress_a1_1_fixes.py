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
    """B-115 (Gemma3 leak) + B-131 (api_proxy leak): top-level
    cfg.model.revision is a Qwen-local-4B-specific historical convention
    (paper-grade lock manifest pins the B1 local Qwen SHA there).
    Forwarding into Gemma3 (B-115) or api_proxy (B-131) backends would
    silently inject the wrong base-model SHA — both leak vectors closed
    by gating injection to {"local_qwen"} only. B-131 attack: codex
    Mode B F2 caught that the B-115 fix only dropped local_gemma but
    left api_proxy in the injection set, leaving B0 235B's
    condition_meta.json reporting a B1 4B SHA.
    """
    src = (REPO_ROOT / "p79" / "experiment" / "runner" / "main.py").read_text()
    # The gating set must be present.
    assert "_QWEN_CLASS_BACKEND_TYPES" in src, (
        "runner.main missing _QWEN_CLASS_BACKEND_TYPES gate — B-115/B-131 "
        "fixes reverted. Top-level Qwen revision could leak into other backends."
    )
    # The set must contain ONLY local_qwen (B-131 closed the api_proxy leak;
    # B-115 had previously closed the local_gemma leak).
    set_decl = re.search(
        r"_QWEN_CLASS_BACKEND_TYPES\s*=\s*\{([^}]*)\}", src
    )
    assert set_decl is not None, "Could not locate _QWEN_CLASS_BACKEND_TYPES literal"
    types_seen = {
        s.strip().strip('"').strip("'")
        for s in set_decl.group(1).split(",")
        if s.strip()
    }
    assert types_seen == {"local_qwen"}, (
        f"_QWEN_CLASS_BACKEND_TYPES drifted to {types_seen!r}; expected "
        f"{{'local_qwen'}} (B-131 fix). Adding api_proxy back re-opens the "
        f"B0 fake-revision-pin leak; adding local_gemma back re-opens "
        f"B-115 leak."
    )
    # The conditional must include the backend-type check.
    assert "_backend_type in _QWEN_CLASS_BACKEND_TYPES" in src, (
        "runner.main revision-forward conditional no longer checks backend "
        "type — B-115/B-131 fix partially reverted; leak path re-opened."
    )


# ---------------------------------------------------------------------------
# B-116 — cross-baseline max_new_tokens parity (2026-05-15 evening)
# ---------------------------------------------------------------------------


def test_configs_b0_b1_b2_max_new_tokens_parity():
    """B-116: All `exp_v2_*.yaml` configs must declare max_new_tokens=4096.

    Previously B0=4096 vs B1/B2=384 — a 12x cap asymmetry that codex Mode B
    Q2 quantified at 0.017% silent truncation on B1, but a paper-grade
    cross-baseline parse_fail comparison should not have ANY truncation
    delta. The retired guard rail "B0/B1 设计不对称是已知论文披露即可代码不改"
    no longer applies after the 学长 logprob / official API negotiation
    began (Class 1 inherent vs Class 2 historical code sloppy split — see
    实验笔记 §142). All configs now run at the same cap.

    If anyone re-introduces a per-baseline cap delta, paper §3.5.1 owes a
    fresh disclosure block + this test should be the first canary to fail.
    """
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    configs = sorted((repo / "configs").glob("exp_v2_*.yaml"))
    assert len(configs) > 0, "No exp_v2_*.yaml configs found — path wrong?"

    drift: list[str] = []
    for cfg in configs:
        for i, line in enumerate(cfg.read_text().splitlines(), start=1):
            # match lines like `    max_new_tokens: 384` (or any non-4096)
            stripped = line.strip()
            if stripped.startswith("max_new_tokens:"):
                # parse the value (strip inline comments)
                val_str = stripped.split(":", 1)[1].split("#", 1)[0].strip()
                try:
                    val = int(val_str)
                except ValueError:
                    drift.append(f"{cfg.name}:{i}: cannot parse value {val_str!r}")
                    continue
                if val != 4096:
                    drift.append(f"{cfg.name}:{i}: max_new_tokens={val} (expected 4096)")

    assert not drift, (
        "Cross-baseline max_new_tokens parity broken — at least one config "
        "drifted from the unified 4096 cap. /stress A1.1 F3 fix B-116 (§142) "
        "regression. Drift list:\n  " + "\n  ".join(drift)
    )



# ---------------------------------------------------------------------------
# /stress A1.1 v8 — Commit A round (B-133 / B-134 / B-135 / B-136 / B-137)
# ---------------------------------------------------------------------------


def test_b1_b2_image_encode_lenient_cross_baseline_align():
    """B-133 (/stress A1.1 v8 3-AI overlap P0-5, 2026-05-15): all 3 agents
    must have lenient try/except around image encoding + emit
    image_encode_error count to meta. Previously B0 was lenient but B1/B2
    raised, producing asymmetric episode outcomes for the same root cause.
    """
    b1_src = (REPO_ROOT / "p79" / "agents" / "qwen3vl_agent.py").read_text()
    b2_src = (REPO_ROOT / "p79" / "agents" / "gemma3vl_agent.py").read_text()
    b0_src = (REPO_ROOT / "p79" / "agents" / "proxy_api_agent.py").read_text()

    for label, src in [("B0 (proxy)", b0_src), ("B1 (qwen3vl)", b1_src), ("B2 (gemma3vl)", b2_src)]:
        assert "_image_encode_error_count" in src, (
            f"{label} missing _image_encode_error_count counter — B-133 "
            f"cross-baseline lenient alignment regressed."
        )
        assert '"image_encode_error"' in src, (
            f"{label} meta dict missing image_encode_error field — "
            f"runner.image_meta payload would be asymmetric across baselines."
        )


def test_runner_validate_action_bool_saved_into_parse_valid():
    """B-134 (/stress A1.1 v8 codex F3, 2026-05-15): runner must save the
    validate_action bool and combine it with agent meta.valid into
    parse_valid. Previously the bool was discarded (action, _ =
    validate_action(action)), splitting schema source-of-truth between
    agent self-report and runner-rescued action.
    """
    src = (REPO_ROOT / "p79" / "experiment" / "runner" / "main.py").read_text()
    assert "runner_valid_post_backend" in src, (
        "runner.main missing runner_valid_post_backend — B-134 bool-save "
        "regressed; runner is back to discarding validate_action result."
    )
    assert '"runner_invalid_action"' in src, (
        "runner.main missing 'runner_invalid_action' failure_reason — "
        "B-134 failure taxonomy expansion regressed."
    )
    assert "parse_valid = agent_parse_valid and runner_valid_post_backend" in src, (
        "runner.main parse_valid computation no longer ANDs agent + runner "
        "validity — B-134 contract regressed."
    )


def test_qwen_gemma_revision_strict_mode():
    """B-136 (/stress A1.1 v8 Claude F5, 2026-05-15): qwen3vl + gemma3vl
    must raise RuntimeError on missing model.revision (paper-grade strict
    mode). Previously B1 fell back to a hard-coded SHA (silent provenance
    lie) and B2 warned-and-loaded-HF-HEAD. Single cross-baseline policy:
    revision MUST be explicit in yaml.
    """
    qwen_src = (REPO_ROOT / "p79" / "agents" / "qwen3vl_agent.py").read_text()
    gemma_src = (REPO_ROOT / "p79" / "agents" / "gemma3vl_agent.py").read_text()

    # Hardcoded fallback constant assignment must be gone. (Comments
    # mentioning the historical name are OK; we look for the active
    # ASSIGNMENT pattern at column 0/indented-block level.)
    import re as _re
    assignment_pattern = _re.compile(
        r"^\s+_DEFAULT_REVISION\s*=", _re.MULTILINE
    )
    assert assignment_pattern.search(qwen_src) is None, (
        "qwen3vl_agent still has an active `_DEFAULT_REVISION = ...` "
        "assignment — B-136 strict mode regressed; silent provenance "
        "fallback re-opened."
    )

    # Both agents must raise on missing revision.
    for label, src in [("qwen3vl_agent", qwen_src), ("gemma3vl_agent", gemma_src)]:
        assert "model.revision must be pinned" in src, (
            f"{label} no longer raises strict-mode message on missing "
            f"revision — B-136 regression."
        )
        assert "raise RuntimeError" in src, (
            f"{label} missing RuntimeError raise for unpinned revision — "
            f"B-136 strict-mode regression."
        )


def test_base_yaml_three_baselines_temperature_uniform_zero():
    """B-137 (/stress A1.1 v8 codex F8, 2026-05-15): base yaml temperature
    must be 0.0 for all 3 baseline backend blocks (local_4b / local_gemma /
    api_strong). Code uses do_sample=False so the value never reaches
    generate(), but run_meta records it verbatim — reviewer reading the
    metadata sees cross-baseline decoding config asymmetry if any drift.
    """
    cfg_text = (REPO_ROOT / "configs" / "exp_v2_base.yaml").read_text()

    import re as _re
    # Find every backend block's temperature line by walking backend names.
    for backend in ("local_4b", "local_gemma", "api_strong"):
        # Match e.g. `  local_4b:\n ... temperature: <value>` (allow any
        # interleaved lines until next backend block or top-level key).
        pattern = _re.compile(
            rf"^\s{{2}}{backend}:\s*\n"  # backend header (2-space indent)
            r"((?:\s{4,}.*\n)+?)"          # inner block (indented lines)
            r"^\s{4}temperature:\s*([\d.]+)",
            _re.MULTILINE,
        )
        m = pattern.search(cfg_text)
        assert m is not None, (
            f"exp_v2_base.yaml: could not locate temperature line under "
            f"backend `{backend}:` — block layout drifted from B-137 fix."
        )
        temp_val = float(m.group(2))
        assert temp_val == 0.0, (
            f"exp_v2_base.yaml backend `{backend}` temperature drifted to "
            f"{temp_val} (expected 0.0). B-137 cross-baseline T=0 metadata "
            f"parity regression. Code uses do_sample=False but yaml is the "
            f"reviewer-visible run_meta source."
        )


# ---------------------------------------------------------------------------
# /stress A1.1 v8 — Commit B round (B-138 / B-139 / B-140 / B-143)
# ---------------------------------------------------------------------------


def test_b2_image_token_count_method_emitted_in_meta():
    """B-139 (/stress A1.1 v8 Claude F6, 2026-05-15): gemma3vl_agent must
    emit image_token_count_method enum ("exact_id_match" vs
    "estimate_256_per_image") so silent method switch on transformers
    version upgrade doesn't silently change cost-accounting semantics.
    """
    src = (REPO_ROOT / "p79" / "agents" / "gemma3vl_agent.py").read_text()
    assert "image_token_count_method" in src, (
        "gemma3vl_agent missing image_token_count_method meta field — "
        "B-139 audit-trail regression."
    )
    assert '"exact_id_match"' in src, (
        "gemma3vl_agent missing exact_id_match enum value — B-139 regression."
    )
    assert '"estimate_256_per_image"' in src, (
        "gemma3vl_agent missing estimate_256_per_image enum value — B-139 regression."
    )


def test_runner_image_meta_is_mandatory_dict():
    """B-140 (/stress A1.1 v8 codex F5, 2026-05-15): image_meta is now
    MANDATORY in step_record (always emitted, fields=None when N/A).
    Schema change: previous conditional `if _image_meta_payload:` is
    replaced with unconditional assignment + `pipeline` label.
    """
    src = (REPO_ROOT / "p79" / "experiment" / "runner" / "main.py").read_text()
    # Old conditional gone (or doesn't gate the assignment)
    assert 'step_record["image_meta"] = _image_meta_payload' in src, (
        "runner.main no longer writes image_meta unconditionally — B-140 "
        "mandatory schema regressed."
    )
    # New pipeline enum
    assert '"pipeline":' in src or "pipeline\": _image_pipeline" in src, (
        "runner.main image_meta missing `pipeline` enum field — B-140 "
        "regression. proxy_jpeg_data_url vs hf_processor_pil label needed."
    )
    # Pipeline mapping exists for all 3 backend types
    for backend_type in ("api_proxy", "local_qwen", "local_gemma"):
        assert f'"{backend_type}"' in src, (
            f"runner.main image_meta pipeline mapping missing `{backend_type}` "
            f"key — B-140 regression."
        )


def test_proxy_agent_emits_network_retry_meta():
    """B-143 (/stress A1.1 v8 Claude F7, 2026-05-15): proxy_api_agent must
    emit network_retry_count + network_retry_wait_ms in meta so runner can
    compute latency_ms_minus_retry. Without these fields the B0 retry
    overhead (10-70s scaffold) is conflated with model inference latency
    in cross-baseline comparisons.
    """
    src = (REPO_ROOT / "p79" / "agents" / "proxy_api_agent.py").read_text()
    assert '"network_retry_count"' in src, (
        "ProxyApiAgent meta missing network_retry_count — B-143 regression."
    )
    assert '"network_retry_wait_ms"' in src, (
        "ProxyApiAgent meta missing network_retry_wait_ms — B-143 regression."
    )
    # Counter increment must happen in both retry paths (network exc + status code)
    import re as _re
    increments = _re.findall(r"_retry_count \+= 1", src)
    assert len(increments) >= 2, (
        f"ProxyApiAgent _retry_count incremented in {len(increments)} place(s) "
        f"— expected 2 (network exception path + HTTP status retryable path). "
        f"B-143 partial regression: one retry path silently drops the count."
    )


def test_runner_emits_latency_ms_minus_retry():
    """B-143: runner step_record latency_ms dict must include
    `total_minus_retry` for cross-baseline-fair latency comparison.
    """
    src = (REPO_ROOT / "p79" / "experiment" / "runner" / "main.py").read_text()
    assert '"total_minus_retry"' in src, (
        "runner.main latency_ms missing total_minus_retry field — B-143 "
        "cross-baseline fair-latency contract regressed."
    )


def test_b138_probe_script_exists_and_compiles():
    """B-138 (/stress A1.1 v8 Claude F4, 2026-05-15): T=0 greedy
    consistency probe script must exist as a runnable cheap probe (no
    VWA dep) for paper-§3.5 reproducibility audit. Full audit deferred
    to advisor; this probe is the quick-win lightweight verification.
    """
    probe = REPO_ROOT / "scripts" / "maintenance" / "probe_b0_greedy_consistency.py"
    assert probe.exists(), (
        "B-138 probe script missing — paper §3.5 reproducibility audit "
        "lightweight verification cannot be performed."
    )
    import py_compile
    try:
        py_compile.compile(str(probe), doraise=True)
    except py_compile.PyCompileError as exc:
        raise AssertionError(f"B-138 probe script does not compile: {exc}")
    # Sanity: script declares its 3-tier verdict labels (mechanical / semantic / nondet)
    src = probe.read_text()
    for label in ("MECHANICAL_GREEDY", "SEMANTIC_GREEDY_WITH_NOISE", "NON_DETERMINISTIC"):
        assert label in src, (
            f"B-138 probe verdict label `{label}` missing — paper §3.5 "
            f"disclosure decision tree depends on all 3 labels being emitted."
        )
