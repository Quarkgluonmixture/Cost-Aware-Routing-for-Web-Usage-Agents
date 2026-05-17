# Section 4 — Known Evaluator Limitations & Disclosure (Draft)

**Status**: 🟡 Draft prose for paper §4 / §3 limitation table. Each subsection cites the
master_bug_catalog.md entry by ID. Reviewer-defensible: bugs are CONFIRMED but blast radius
bounded; mitigations or paper-§3 disclosure rather than retraction.

**Source**: `docs/reference/master_bug_catalog.md` B-15 / B-20 / B-21 / B-22 / B-26 / B-28 +
`docs/checkpoints/pre_run/preregistration.md` §A1/A3 design asymmetries.

---

## §4.X.1 ua_match GPT-judge drift (B-20)

VWA's `ua_match` evaluator uses a GPT-4o-mini judge to rate the agent's terminating answer
against the task's reference answer. The judge prompt template is fixed in
`evaluation_harness/helper_functions.py` (`llm_fuzzy_match`) and not modified in this work.
However, GPT-4o-mini is a stochastic API: the judge's output drifts across re-evaluations
in 4 distinct modes (semantic equivalence vs strict literal match, spurious "partial credit",
hallucinated rationale, and length-dependent confidence). Static audit of 87 N/A-task FP
episodes (笔记 §95) showed the judge's binary verdict varies on ~12% of borderline cases when
re-queried with temperature ≥0.

**Mitigation in this work (revised 2026-05-14 §139.8 FP-architecture restructure)**: The root
cause — GPT-4o-mini scoring an empty prediction as correct — is now fixed at the **VWA evaluator**
itself (B-91 patch in submodule branch `p79-patches` commit `f0c835b`: empty-prediction guard
on `llm_fuzzy_match` / `llm_ua_match`). With this upstream guard, `na_fp` no longer needs a
post-hoc filter class; raw `success` from the (fixed) evaluator is correct at the boundary.
Additionally, N/A tasks are excluded at task-load time (`exclude_na_tasks: true` default; 73
N/A tasks across VWA+WA, 5.3% of 1390), eliminating the un-passable-task FP vector entirely.
Judge temperature is pinned at `temperature=0`. The prior post-hoc `compute_adjusted_success`
layer and 3-variant sensitivity ladder (raw / +na_fp / +na_fp+eval_fp) are retired; canonical
metric is raw `success`. Appendix D retains the pre-§139.8 ladder for archive contamination
disclosure only.

**Residual concern**: If a future reviewer re-runs the evaluator with a newer GPT-4o-mini
snapshot, single-task labels may flip. The aggregate per-cell SR is robust to this within
±2pp by simulation. We make this explicit in our reproducibility statement (§3.X) rather
than retract the SR claim.

---

## §4.X.2 string_match fuzzy_threshold misnomer (B-21)

VWA's `string_match` evaluator exposes a `fuzzy_threshold` parameter that suggests a
numerical similarity cutoff for string matching. In practice (catalog B-21 static audit),
the parameter is **only honored when fuzzy_threshold=1.0** — under which the evaluator
falls through to the same GPT-4o-mini fuzzy_match judge as `ua_match`. Threshold values
strictly below 1.0 trigger a brittle exact-token-overlap path with no judge involvement.
This is effectively binary GPT-judged matching, not a tunable similarity metric.

**Mitigation**: We use `fuzzy_threshold=1.0` consistently across all conditions (verified
via condition_meta.json `evaluator_config.fuzzy_threshold`), so the variability source is
the same as B-20 ua_match drift and is jointly bounded by the same FP filter robustness.
The mis-naming does not affect our results, but we flag it for readers attempting to
interpret raw VWA evaluator parameters.

---

## §4.X.3 program_html selector brittleness (B-22)

VWA's `program_html` evaluator scores tasks by goto'ing a target URL and querying DOM with
CSS/XPath selectors authored in each task's reference config. Static audit (笔记 §107
Tier 5) found 562 of 1598 (35.2%) selectors are class-only or attribute-only patterns
(e.g., Magento's `.order-details-items.ordered`, classifieds' `.price` / `.desc`) that
match site-skin-dependent layout. When the site's CSS skin updates between evaluator
authoring time (2024) and our experimental deployment (2026), selectors can match the
wrong DOM node or miss the intended element entirely.

**Per-cell quantification + mitigation (revised 2026-05-14 §139.8 FP-architecture restructure)**:
The post-hoc `eval_fp` filter class is retired. The dominant program_html FP root cause was
site-state contamination across cells (cart accumulation, posted listings persisting across
modes), which is now prevented upstream by the `RESET_BEFORE=1` per-cell reset protocol —
contamination is not an after-the-fact filtering problem, it is prevented by clean per-cell
start state. The remaining selector-skin-drift residual is bounded: per-cell selector-hit-rate
parity is monitored in `validate_run.py --strict` (target pre/post ratio 0.95-1.05); flagged
selectors are listed in Appendix D for transparency but no longer trigger a separate FP class
in the primary metric.

**Cannot-fix scope**: Patching all 562 brittle selectors requires authoring a parallel
evaluator harness, which is out of scope for this paper. We retain VWA's evaluator with the
B-91 source-level patch only; selector-skin-drift impact is disclosed in Appendix D rather
than absorbed into a post-hoc FP filter.

---

## §4.X.4 finish_wrong_state — agent error not scaffold (B-15)

In Tier 2 silent-failure analysis (笔记 §107), 1552 of 4501 episodes (34.5%) had the agent
emit `finish` while the page state did not match the task goal. Initial framing classified
this as a scaffold bug; subsequent self-replay (笔记 §95 reform) showed it is an **agent
reasoning error** — the agent decides to terminate prematurely or with partial completion,
not a runner / dispatch / observation failure.

**Treatment (revised 2026-05-14 §139.8 FP-architecture restructure)**: The post-hoc `eval_fp`
filter rule is retired. The `agent_finished=True with no effective action` pattern is captured
behaviorally in the agent's trajectory data (`finish` step + zero `click` / `type` / `scroll`
between page-load and finish), not as an FP-filter exclusion. Reviewers can audit this pattern
directly in released JSONL traces. The agent error itself is not a paper limitation — different
baselines and modes can succeed or fail at terminating decisions, and our paired-design comparison
absorbs this into per-task variance.

---

## §4.X.5 in_viewport_ratio operator precedence (B-26, **FIXED 2026-04-19**)

In upstream VWA `external/visualwebarena/browser_env/processors.py:218`, the
`in_viewport_ratio` calculation `overlap_w * overlap_h / w * h` is parsed by Python as
`((overlap_w * overlap_h) / w) * h` — multiplication-first then division — instead of the
intended ratio `(overlap_w * overlap_h) / (w * h)`. With the upstream formula the 0.6
viewport-overlap threshold (`current_viewport_only=True`) is effectively bypassed, allowing
partially-visible elements to remain in the AXTree with their full text content even when
they are visually truncated.

**Fix applied** (2026-04-19, commit `3f9ceca` on VWA submodule branch `p79-patches`):
we parenthesise the ratio as `(overlap_w * overlap_h) / (width * height)`. After the fix
the 0.6 threshold is mathematically meaningful — `ratio ≥ 0.6` implies the element centre
lies inside the 1280 × 720 viewport (`center_y ≤ 720 − 0.1h < 720`), so partially-visible
elements whose centre falls outside the viewport are excluded from the AXTree rather than
exposed with their full text. All B0+B1 DOM and SoM conditions across all sites were re-run
after the fix (decision recorded in 实验笔记 §80; Vision modes are unaffected by viewport
filtering and were not re-run). Paper-grade Phase 1a uses the fixed scaffold throughout.

**Implication for our claims** (post-fix): the previously-feared "DOM has structural
text-exposure advantage from a no-op viewport filter" pathway is closed by the fix. DOM mode
no longer receives full text for visually-truncated elements whose centre falls outside the
viewport; SoM and Phantom-SoM read from the same fixed AXTree, so paired comparisons
(P-SoM ↔ DOM, P-SoM ↔ SoM) remain mutually consistent at the text-payload layer. The §1
hero claim (P-SoM ≥ best of DOM/SoM/Vision) is supported under the fixed scaffold and does
not rely on the upstream operator-precedence bug as a confound source. The pre-fix archive
(`docs/archive/analysis_pre_2026-05-15/`) is retained for sensitivity reference only and
must not be mixed with Phase 1a clean-run numbers.

**Downstream consequence — rule-based router numeric thresholds rendered dead by the
viewport fix (A1.10 P0-1-ABC* cross-reference 2026-05-16).** The `in_viewport_ratio`
correction collapses typical AXTree text length from the pre-fix ~12-20k char regime
(full-DOM intuition) to the post-fix ~3 k char regime (cleaned viewport-only AXTree).
The router constants `dom_size_threshold = 12000`, `dom_complexity_trigger = 500`,
`text_length_trigger = 12000` in `p79/experiment/router.py` were inherited from the
pre-fix regime and were **not** recalibrated to the post-fix distribution; they fire
< 0.5 % empirically on the Phase 1a clean-run archive (full disclosure in §3.5 router
thresholds paragraph). Paper-1 reports the streak-driven routing decision and defers
threshold recalibration to paper-2; §4 rollups should be read accordingly when the
trigger-distribution table shows ~0 fires for the three numeric triggers.

---

## §4.X.6 scroll direction confusion (B-28)

Early experiments (B0 cls/red, 笔记 §50) revealed inconsistent agent behavior for scroll
direction conventions: Web CSS uses `dy>0 = scroll DOWN` (content moves up), but Win32 and
macOS natural scrolling invert this convention. The 235B model occasionally chose the wrong
direction sign, producing scroll-up-when-needed-down patterns counted as no-progress.

**Mitigation**: §67 schema reform replaced `delta: [dx, dy]` with explicit
`scroll_direction: enum("up", "down")` in the action schema (B0 only via tool-calling
schema; B1 still uses delta in greedy decoding). This eliminates the symbol convention
confound for B0 going forward but does not retroactively fix archived B0 data. We disclose
this asymmetry in §3 evaluator-side fairness discussion.

---

## §4.X.7 A1/A3 baseline-design asymmetries (B-56)

This work compares B0 (Qwen3-VL-235B-A22B via proxy API) against B1 (Qwen3-VL-4B-Instruct
local). Two configuration asymmetries are intentional and documented:

**A1 — Decoding strategy**: B0 uses `temperature=0.0` with `top_p=1.0` (B-37 fix
post-§107); B1 uses `do_sample=False` (greedy top-1). Both target deterministic outputs,
but B0 still inherits proxy-side stochasticity for which the API has no `seed` parameter.
Cross-run trajectory variance for B0 is bounded by single-step branching at ties; aggregate
SR is stable (laughs at our N=234+210+466 sampling).

**A3 — Token budget (RESOLVED 2026-05-15 per commit `9f70b4e` / B-116 fix)**: Previously B0 used `max_new_tokens=4096` while B1 used `max_new_tokens=384`. The 12× asymmetry was identified by codex /stress F3 as a paper-grade contamination vector — under the agent's thought+JSON envelope (~400-1500 tokens typical), the 384 cap caused silent truncation on B1/B2 and parse failures that B0's GLM rescue scaffold could mask. **B-116 unified B1 and B2 to `max_new_tokens=4096` to match B0** (commit `9f70b4e fix(configs): B-116 unify B1/B2 max_new_tokens 384→4096 — §142 F3 close`), eliminating the asymmetry. The A100 canonical rerun runs all 3 baselines at 4096; no truncation-rate sensitivity check needed. **Note**: a parse-error recovery scaffold asymmetry (B0 GLM rescue, B1/B2 none) remains an open item; see §3.5.1 cross-baseline disclosure + master_bug_catalog B-86.

---

## §4.X.8 Cross-machine numerical drift (笔记 §114 Gap 5)

Our work runs across three GPU architectures: DGX Spark (NVIDIA GB10, sm_121), UCL Condense
A100 (sm_80), and UCL Myriad (sm_70 V100 / sm_80 A100). Mechanistic Stage 2B/2C activation
patching outputs are sensitive to floating-point matmul precision differences across CUDA
generations (sm_70 vs sm_80 vs sm_121). We run `numerical_determinism_check.py` to quantify
maximum absolute hidden-state drift |Δh| across machines on a fixed input.

**Reproducibility statement**: Cross-machine numerical agreement on Qwen3-VL-4B between
{DGX, A100, Myriad} layers L0-L35: max |Δh| < [TBD post-rerun, target <1e-2] at L11 (the
mirage causal layer per §5). This bounds inter-machine reproducibility drift to a level that
does not flip top-1 logit comparisons; aggregate SR claims are unaffected.

---

## §4.X.9 Pre-Phase-A vs post-Phase-A asymmetry (B-01 to B-37 family)

The 36-condition / 6-cell Phase 1a rerun (preregistration.md §4 cell inclusion) uses post-Phase-A code only
(commit ≥ `3c15cd7`, dispatch + page_changed + cycle + RNG fixes deployed). Pre-Phase-A
data is retained as Appendix D robustness check (preregistration.md `Cell inclusion (Appendix D)`).
For mechanistic Stage 2B/2C input artifacts, we use pre-Phase-A archived observations
(`results/mechanistic/archive_subset_b1_cls/`); per 笔记 §116 user-prompt analysis, agent
trajectory bugs (Phase A scaffold issues) do **not** affect the model's forward-pass
input→output mapping at any frozen step. Mechanism findings (L11 causal layer, forward-vs-reverse
asymmetry) are therefore unaffected by Phase A vintage; we make this independence explicit
in §5.

---

## §4.X.10 Stage 2B input vintage independence (笔记 §116 user Q)

Mechanistic Stage 2B (forward L11 mirage causal layer) and Stage 2C (reverse direction
asymmetry) use frozen `observation_dom.txt` + `screenshot_annotated.png` artifacts from
`B1_phantom_som_classifieds_20260428` archive (pre-Phase-A). Per 笔记 §116 user analysis:
the mechanistic claim is about model forward-pass behavior given a fixed input, not about
agent trajectory soundness. Phase A bugs in dispatch / cycle / RNG affect *which step* the
agent reaches, not *what the model thinks* given a frozen step's observation. The L11
mirage finding is therefore Phase-A-vintage-independent.

For full robustness, we pre-specify a post-Phase-A spot-check (5-10 tasks from a clean
post-`3c15cd7` cell) where we re-run Stage 2B and verify L11 causal layer holds. This
sensitivity check is in §5 Appendix and does not gate the main mechanism claim.

---

## §4.X.11 VWA submodule `p79-patches` branch — full disclosure table

We run paper-grade experiments against a forked VisualWebArena submodule pinned to branch
`p79-patches`, HEAD `2f9b0b47175a1bffa01e13100e3075e212161a89`. Upstream base is
`89f5af29305c3d1e9f97ce4421462060a70c9a03` on `main`. The full set of behavioural patches
between the upstream base and our pinned HEAD is reproduced below for OSF reproducibility
review and cross-paper comparability. Per **B-607 P1-1-AC\*** (2026-05-17 /stress A1.18-re Claude+Gemini OOB), the patch-bundle integrity witness has been migrated from the prior `git diff base..HEAD | sha256sum` recipe (environment-dependent on `diff.algorithm` / `core.autocrlf` / git version) to a **tree-hash chain** of git-canonical commit and tree SHAs (`git rev-list base..HEAD --format=tformat:'%H %T' | sha256sum`), which is byte-deterministic across all git versions and OS environments. The current tree-hash-chain SHA-256 is
`5c6c5f625f44ca1b2155b9cad280b5aecb3e6939cf0599540fcef0900028fb0f`.

**Archive vintage disclosure (added 2026-05-17 /stress A1.18-re B-617 P1-11-C Gemini)**: The Phase 1a pre-fix archive (B0/B1 only, collected pre-2026-05-13 on DGX→quark Tailscale stack) was produced under a much earlier submodule HEAD (`f0c835b` or earlier — predating the `eb5cbd8` A1.18 sweep + `c1765ee` / `1c3a615` / `2f9b0b4` A1.25-and-onward sweeps). The canonical paper-grade rerun (post-§139.8 FP architecture + B2 = Gemma3-VL included) runs on A100 self-hosted VWA Docker against the current pinned HEAD `2f9b0b4`. **Paper §1 hero numbers cite ONLY the canonical-rerun-at-`2f9b0b4` data** — archive data is retained as Appendix D "pre-§139.8 contamination reference" only, never folded into §1 4-fold-drop-in claim. Within-rerun comparisons (across modes / baselines / sites on the same A100 stack at the same HEAD) are unaffected by this version-rift; cross-vintage comparison (archive ↔ canonical) is explicitly disclaimed as a confound vector — not used for paper-grade claims.

| Commit (short) | Subject | Behavioural impact | Affected files | Paper §-disclosure |
|---|---|---|---|---|
| `e9c63b7` | wait for networkidle before screenshot | Lazy-loaded images settle before screenshot; superseded by `eb5cbd8` single-barrier fix (wait moved to `ObservationHandler.get_observation` shared barrier) | `browser_env/processors.py` (`ImageObservationProcessor.process` local wait, now removed) | §3.5 LLM-judge / observation-timing paragraph |
| `3f9ceca` | composite runtime patches | (a) viewport-ratio operator-precedence fix (B-26); (b) Chromium `--host-resolver-rules` for Tailscale DNS (later replaced by `eb5cbd8` env-driven `VWA_CHROMIUM_LAUNCH_ARGS`); (c) `float()` cast on Playwright mouse coords for NumPy 2.0 compatibility (sync sites only — async sites cast added by `eb5cbd8`); (d) `VWA_EVAL_MODEL` env var (default `gpt-4o-mini`, upstream default was deprecated `gpt-4-1106-preview`); (e) lazy OpenAI client init (later wrapped in `threading.Lock` + env-fingerprint check by `eb5cbd8`); (f) `Meta+A`+`Backspace` clear-before-type added to all 5 `execute_type` / `aexecute_type` dispatch paths (commit body omitted (f); disclosed retroactively here and superseded by P79 wrapper `vwa_wrapper.py::locator.fill()` for canonical paper-grade path) | `browser_env/processors.py`, `browser_env/envs.py`, `browser_env/actions.py`, `evaluation_harness/helper_functions.py`, `llms/providers/openai_utils.py` | §4.X.5 (viewport, FIXED), §3.5 (judge model + clear-before-type), §4.X.12 (host-resolver) |
| `16b60d7` | setup script + extra task configs | `prepare.sh` defensive Python resolver (later extended with Windows `py -3` fallback by `eb5cbd8`); WebArena homepage placeholder replaced with `localhost`; VWA shopping task pool committed (later substituted with VWA-canonical `__SHOPPING__` / `__CLASSIFIEDS__` / `__REDDIT__` / `__WIKIPEDIA__` placeholders by `eb5cbd8`); WA non-visual task configs committed | `prepare.sh`, `environment_docker/webarena-homepage/templates/index.html`, `config_files/vwa/test_shopping.json`, `config_files/wa/test_*.raw.json` | §3.5 (task pool source); §4.X.12 (private IP propagation into task configs, now closed) |
| `832f037` | `.gitignore` runtime data | Wikipedia ZIM dump and classifieds compose state excluded from git tracking; data fetched separately per host | `.gitignore` only | none required (no behavioural impact on agent / evaluator) |
| `f0c835b` | B-91 empty-prediction guard | `llm_fuzzy_match` / `llm_ua_match` return `0.0` deterministically on empty / whitespace-only `pred`; closes the dominant FP source for string_match and N/A tasks. Match-after-LLM-judge logic tightened in `eb5cbd8` to also log unexpected judge responses for audit | `evaluation_harness/helper_functions.py` | §3.5 (FP source-level fix); see also `reference_fp_architecture_2026-05-14.md` |
| `eb5cbd8` | /stress A1.18 full sweep (15 findings) | (a) 913 VWA task config files rewritten to canonical `__SHOPPING__` / `__CLASSIFIEDS__` / `__REDDIT__` / `__WIKIPEDIA__` placeholders, closing the private-IP-in-config-data propagation (793-hit baseline reduced to 0 tracked-file hits); (b) `envs.py` chromium launch args env-driven via `VWA_CHROMIUM_LAUNCH_ARGS` (no hardcoded private IP); (c) `processors.py` networkidle wait moved to single shared barrier in `ObservationHandler.get_observation`, removing the asymmetric pre-fix where text observation never waited; (d) `helper_functions.py` softened-assert tightened to log unexpected judge responses to `evaluator_unexpected_response_log.csv` (gitignored); (e) `openai_utils.py` lazy client wrapped in `threading.Lock` with sha256(api_key + base_url) env-fingerprint check; (f) async OpenAI throttlers return `str` directly (caller dict-indexing path normalized); (g) `aexecute_action` signature now includes `obseration_processor` param, CLEAR/UPLOAD branches use truly async primitives; (h) `create_upload_action` sets `ActionTypes.UPLOAD` (was `TYPE`, making the UPLOAD branch unreachable); (i) async `aexecute_mouse_hover` + `aexecute_upload` wrap coords in `float()` (sibling propagation completion); (j) `prepare.sh` adds Windows `py -3` fallback | `browser_env/{actions.py, envs.py, processors.py}`, `evaluation_harness/helper_functions.py`, `llms/providers/openai_utils.py`, `prepare.sh`, `config_files/vwa/test_shopping.json` + 912 gitignored per-task config files | §4.X.5 (viewport stale-doc closure), §3.5 (judge / clear-before-type / observation timing), §4.X.12 (IP propagation closed), this §4.X.11 row |
| `c1765ee` | /stress A1.25 GRL Chunk 1 (3 fixes) | (a) **B-445** `create_mouse_click_action` truthiness fix at `actions.py:657-672` — pre-fix `if left and top:` rejected legitimate `(0.0, y)` / `(x, 0.0)` boundary clicks → vision-mode coord clicks at viewport edges silently dropped; post-fix `if left is not None and top is not None:` preserves boundary values; (b) **B-446** sync `SELECT_OPTION` at `actions.py:1410-1428` extracts `pw_action_args` / `pw_action_kwargs` from parsed final call and passes forward — pre-fix `parsed_code[-1]["arguments"]` was parsed but discarded → `locator.select_option()` called with no option (silent no-op); (c) **B-447** sync UPLOAD parser+factory at `actions.py:1717` + `:1807` — pre-fix factory used `ActionTypes.TYPE` (UPLOAD branch unreachable) and id-based regex matched `type` not `upload`, so upload was doubly dead | `browser_env/actions.py` | §3.5.3 P79 GRL action-layer disclosure (catalog entries B-445/446/447 in `master_bug_catalog.md`) |
| `1c3a615` | /stress A1.25 GRL Chunk 4 (4 fixes) | (a) **B-535** `llm_fuzzy_match` polarity inversion at `helper_functions.py:626-634` — pre-fix `if "correct" in response: return 1.0` substring-matched `"incorrect"` / `"partially correct"` / `"not correct"` all as 1.0 (long-standing upstream VWA bug, monkeypatch-verified); post-fix check negative phrases FIRST then positive then fail-closed. Sibling fix for `llm_ua_match` extends negative-first to cover `"not the same"` / `"not same"`. Paper §1 SR no longer systematically inflated by evaluator polarity. **Cross-paper SR comparisons against VWA / WebArena-Verified / PAE are NOT directly comparable** under this fix (those papers use upstream-buggy parser); within-paper paired comparisons (B0/B1/B2, baseline ↔ phantom) remain valid because every cell judged by same `gpt-4o-mini` post-fix; (b) **B-538** async `SELECT_OPTION` mirror sync fix at `actions.py:1593-1597` (sibling-propagation of B-446); (c) **B-539** UPLOAD field decouple from `_keys2ids` encoding at `actions.py:704-728` — pre-fix `text` was key-encoded list of int but `set_files()` expects path str → type-corrupted runtime; post-fix `text` and new `file_path` field hold raw path; also remove `\n` enter-flag suffix at line 1830 (file-chooser has no submit-Enter semantic); (d) **B-540** `VWA_CHROMIUM_LAUNCH_ARGS` use `shlex.split` not `.split()` so quoted multi-word args (e.g. `--host-resolver-rules=MAP host IP`) stay as one argv item | `evaluation_harness/helper_functions.py`, `browser_env/{actions.py, envs.py}` | §3.5 evaluator-patch policy, §3.5.3 P79 GRL action-layer disclosure (catalog entries B-535/538/539/540) |
| `2f9b0b4` | /stress A1.18-re Chunk 1 (11 fixes) | (a) **B-604** `scripts/generate_test_data.py` idempotency — `rmtree(output_dir)` before regen + emit `generation_manifest.json` with raw_sha256 + per-file SHAs; pre-fix never deleted stale split files → version-rift pollution of task universe on replayer rerun; (b) **B-615** generated JSON byte determinism — explicit `encoding="utf-8" newline="\n"` + `sort_keys=True` + `ensure_ascii=False` + trailing newline (pre-fix locale + Windows-CRLF sensitive); (c) **B-609** UPLOAD `action2create_function` dispatch — uses raw text string (post-B-447 schema) instead of `_id2key` on int-list; backward-compat list path retained for pre-B-447 archive traces (fixes A1.25 GRL Chunk 1 sibling-propagation gap in round-trip serializer); (d) **B-610** `_log_unexpected_judge_response` uses module-level `_AUDIT_LOG_LOCK` (replaces TOCTOU `getattr/setattr` lazy-init that race-windowed concurrent first calls); (e) **B-611** `_throttled_openai_completion_acreate` + `_throttled_openai_chat_completion_acreate` fail-loud (`raise RuntimeError(...) from e`) instead of silent empty-string fallback — per user direction 2026-05-17, surface infrastructure failure rather than mask as model output; (f) **B-612** `retry_with_exponential_backoff` drops `BadRequestError` from retryable (non-transient) + preserves original exception via `raise RuntimeError(...) from e`; (g) **B-613** `ImageObservationProcessor.process` bounded retry — `page.wait_for_load_state("load", timeout=5000)` replaces unbounded `wait_for_event("load")`, `PlaywrightTimeoutError` caught + `meta_data["screenshot_retry_timeout"]` flagged; (h) **B-614** `ObservationHandler.get_observation` records networkidle barrier outcome (`networkidle_ok` / `networkidle_elapsed_ms` / `networkidle_exception_type`) into text/image processor meta_data so upstream runner can mark `needs_reevaluation` on repeated barrier misses; (i) **B-618** `llm_fuzzy_match` + `llm_ua_match` polarity check tightened from `"X" in response` substring to `resp.startswith(...)` exact prefix; (j) **B-623** `create_mouse_hover_action` mirrors B-445 click contract (`is not None` for both coords); (k) **B-625** `prepare.sh resolve_python_argv` emits NUL-separated tokens for proper bash array argv (Windows `py -3` fallback now actually executable) | `evaluation_harness/helper_functions.py`, `llms/providers/openai_utils.py`, `browser_env/{actions.py, processors.py}`, `scripts/generate_test_data.py`, `prepare.sh` | §3.5 evaluator-patch policy + LLM-judge model disclosure (`startswith` tighten, `llm_ua_match` scope-of-patch), §4.X.11 row, §4.X.12 (per-task config idempotency), prereg §7 (SBOM tree-hash chain) |

**Per-task config materialization disclosure (B-605 P0-1-AB Claude+codex OOB, 2026-05-17)**: The 912 per-task config files (`config_files/vwa/test_{classifieds,reddit,shopping}/{0..N}.json`) are gitignored derived artifacts deterministically regenerated from the tracked `config_files/vwa/test_{site}.raw.json` templates via `scripts/generate_test_data.py`. Both source files (templates + script) ARE tracked in the submodule and therefore covered by the SBOM tree-hash chain (3) in prereg §7. OSF replayers materialize them via `make vwa-generate-configs` (sets `DATASET=visualwebarena` + per-site env vars + invokes the generation script). Post-A1.18-re (B-604 + B-615) the generator is **clean-idempotent** (rmtree before regen) and **byte-deterministic** (explicit UTF-8 + LF + sort_keys), with `config_files/generation_manifest.json` (per-template raw_sha256 + first/last split sha256) as the verification artifact. Pre-B-604 the gitignored split was vulnerable to stale-file pollution if a replayer regenerated against a shortened template — addressed by the rmtree-before-regen invariant.

**OSF reproducibility verification commands** (run inside the cloned P79 repo):

```bash
cd external/visualwebarena
git rev-parse HEAD                                 # must match 2f9b0b47175a1bffa01e13100e3075e212161a89
git rev-parse origin/main                          # upstream base; if not present, fetch
# Tree-hash chain — env-independent SBOM witness (B-607 P1-1-AC* 2026-05-17, replaces
# legacy `git diff base..HEAD | sha256sum` which was sensitive to local git config):
git rev-list 89f5af29305c3d1e9f97ce4421462060a70c9a03..HEAD --format=tformat:'%H %T' | sha256sum
# must match 5c6c5f625f44ca1b2155b9cad280b5aecb3e6939cf0599540fcef0900028fb0f
```

These three hashes are also locked in `docs/checkpoints/pre_run/osf_lock_manifest.md` and
re-stated in `docs/checkpoints/pre_run/locked_versions.md`; any divergence indicates the
submodule has drifted from the paper-grade pin.

**Composite commit caveat**: commits `3f9ceca` (six independent fixes a-f) and `eb5cbd8`
(ten independent fixes a-j) and `2f9b0b4` (eleven independent fixes a-k, this A1.18-re
sweep) are all composite. We retain composite form for `3f9ceca` because its content
already produced the Phase 1a archive disclosed in Appendix D; for `eb5cbd8` because the
15-finding /stress A1.18 sweep required atomic substrate coverage; for `2f9b0b4` because
the 25-finding /stress A1.18-re sweep similarly required substrate coverage across
multiple files for which a single CI/test pass was the verification gate. Subsequent
single-fix flow is restored at `c1765ee` (B-445/446/447 isolated) and `1c3a615`
(B-535/538/539/540 isolated). The full behavioural list (a–k for `2f9b0b4`) above closes
the disclosure gap that bundled-commit format would otherwise hide. **Bundling is
documented, not concealed (B-619 P2-1-A 2026-05-17 paper §4.X.11 caveat expansion)**.

---

## §4.X.12 Hardcoded Tailscale IP in VWA submodule + task configs

For Phase 1a we run the VWA Docker container set on a Windows host inside our private
Tailscale network (IP `100.95.81.103`, hostname `quark`). To make the Chromium browser
launched by the VWA scaffold resolve the upstream CMU seed URLs to that host, commit
`3f9ceca` adds `--host-resolver-rules=MAP metis.lti.cs.cmu.edu 100.95.81.103` to
`browser_env/envs.py`. In addition, the committed `config_files/vwa/test_*.json` task
configs (added by `16b60d7`) inline `http://100.95.81.103:{9980,9999,7770,8888}/...` URLs
into individual task `start_url` fields, propagating the private IP into the task pool
itself rather than into a single chromium launch flag. We disclose the propagation rather
than rewrite history because the Phase 1a archive was produced under these configs.

**Reproducer impact**: a third party cloning the repo outside our Tailscale network cannot
reach `100.95.81.103`. To reproduce, the reproducer must (a) bring up the VWA Docker
container set on their own host, (b) rewrite `100.95.81.103` to that host's address in
both `external/visualwebarena/config_files/vwa/test_*.json` and the chromium launch arg in
`browser_env/envs.py`, or set `VWA_HOST_RESOLVER_RULES` plus the per-site env vars
**`REDDIT` / `SHOPPING` / `SHOPPING_ADMIN` / `CLASSIFIEDS` / `GITLAB` / `WIKIPEDIA` / `MAP` / `HOMEPAGE`**
read by `p79/experiment/tasks.py::_placeholder_mapping` (lines 55-62) — note these are
**bare site names without a `_BASE_URL` suffix** (corrected via A1.10 P1-10-C* prose fix
2026-05-16, replacing the earlier incorrect `*_BASE_URL` hint that would have silently
fallen back to localhost defaults rather than redirecting to the reproducer's host).
The P79 wrapper `scripts/vwa_env_remote.sh` exports these same names. The P79 self-host
fire on the A100 host applies the same substitution before launch.

This is an OSF lock-time disclosure rather than a code fix: the Phase 1a archive cannot be
re-keyed without re-running, so we document the propagation here and recommend reproducers
treat the IP as a `VWA_HOST` placeholder.

**`.auth/` runtime artifact disclosure (B-606 P0-2-C\* Gemini OOB, 2026-05-17)**: Playwright
`storage_state` files at `.auth/{classifieds,reddit,shopping}_state.json` are gitignored
runtime artifacts captured during agent login. They contain domain-bound cookies tied to
the Tailscale IP `100.95.81.103` used during pre-§eb5cbd8 archive collection. **Reproducers
MUST re-capture authentication state against their own VWA Docker stack**; directly using
repo-provided `.auth/` files on a replayer host triggers silent cookie-domain mismatch in
Playwright (`storage_state` is bound to the origin from which cookies were issued), causing
all subsequent navigation to fail authentication despite "successful" auth-file load. The
re-capture path is: `bash scripts/vwa/setup_vwa.sh` on the replayer host (invokes
`external/visualwebarena/browser_env/auto_login.py` against the replayer's local VWA
Docker stack with replayer's `REDDIT` / `SHOPPING` / `CLASSIFIEDS` env vars set). Phase 1a
B0 prereq in `phase1_plan.md §B0` now lists this re-capture as a launch gate. **The IP-
contamination claim in §4.X.12 above ("propagation closed") applies to tracked code +
task configs; `.auth/` runtime artifacts are out of SBOM scope and must be re-captured per
replayer host.**

---

## §4.X.13 Trajectory event log — best-effort enrichment, race-window event drop

**Stub (B-386 A1.15 C1, 2026-05-16) — full prose post-data; current placeholder for reviewer audit trail.**

Phase 1a fire writes `trajectory_events.jsonl` (Option K, B-313~B-384) per condition_dir,
recording all auto-clean / auto-refresh / reset perturbation events. Paper §4 uses this
log to emit per-episode covariates (`is_after_reset` / `had_auth_clear` /
`had_finalize_race_clear`) for GLMM fixed-effect adjustment. Event writes are
**best-effort post-hoc enrichment** rather than 2-phase committed transactional audit
trail: in `scripts/maintenance/experiment_watchdog.py` the event write happens AFTER the
destructive op (`unlink` + `rmtree` + `_purge_digest_records`), creating a ~2-3s race
window in which SIGKILL/OOM/`restart_watchdog.sh kill -9` drops the event while the
filesystem mutation persists.

**Bias direction**: dropped events → covariate column undercounts true perturbation rate
→ GLMM fixed effect underestimates. The drop mechanism (SIGKILL during a fixed-size race
window) is mode-symmetric so direction of P-SoM drop-one effect is preserved but
magnitude estimate is conservative. Supplementary Table S-trajectory-loss (planned
post-data) bounds event-drop rate by intersection of `condition_summary_v2.json` episode
list with `trajectory_events.jsonl` event list.

**Why not 2-phase commit**: see decision T2'=(a) in A1.15 audit (Pre-fire 闭环 effort
budget ~8.5h; 2-phase commit adds +1.5h with marginal ROI given mode-symmetric drop).
Future tightening to 2-phase available as Tier 2 hardening (schema B-313 already supports
`task_auto_clear_intent` + `task_auto_cleared` 2-event pair).

---

## §4.X.14 Outcome-dependent auto-retry policy — SR upward bias disclosure

**Stub (T5=(b) short disclose, A1.15 C1, 2026-05-16) — full prose + Supp Table S-retry
post-data; current placeholder.**

Episodes whose initial outcome is classified as benchmark-noise (`benchmark_noise=True`)
or transient runtime error (`error(session|auth|connection|timeout|noise)` etc.) are
automatically retried up to **N=3** times by `experiment_watchdog.py`
(`MAX_NOISE_RETRIES`); episodes failing as `error(code_bug)` retry up to **N=2**
(`MAX_CODE_BUG_RETRIES`). Episodes classified as `fail` or `max_steps` are **NOT
retried**. This is **outcome-dependent retry**: retry probability conditional on initial
outcome → P(success | noise+retry) > P(success | first-try-only) for the noise subset,
pulling overall SR estimate upward by the retry-eligible subset's per-mode retry
conversion rate.

**Rationale**: episodes that fail evaluator gates (`error(evaluator)`) or exceed step
budget (`max_steps`) represent genuine model/task incapability, not infrastructure noise;
retrying them measures a different quantity (persistence under fixed prompt) and is
excluded to keep SR an estimate of single-attempt task-solving rate conditional on clean
infrastructure.

**Bias direction per mode**: P-SoM uses a regex-extracted SOM_MARKS payload that may
elicit different `error(*)` rates than baseline modes (DOM/SoM/Vision) — if P-SoM noise
rate is higher → P-SoM SR is retry-inflated relative to baseline. The 1.7-3.3 pp drop-one
effect may be partly retry artifact rather than phantom-routing signal. Supplementary
Table S-retry (planned post-data) reports per-mode retry rates `(noise_retries,
code_bug_retries, clean_first_try)` so reviewers can assess retry-bias direction.

**Why disclose vs change policy**: changing retry policy mid-fire violates data integrity;
disclosing + reporting per-mode retry rates lets reviewers compute a conservative
drop-one bound. Full prose draft + Supp Table generation deferred to post-fire data land
(T5=(b) decision, A1.15 C1).

---

## References

- `docs/reference/master_bug_catalog.md` — full bug catalog (~80 entries)
- `docs/checkpoints/pre_run/preregistration.md` §3-§4 — locked analysis choices including FP filter
- `docs/checkpoints/pre_run/evaluator_change_protocol.md` — Protocol A Tier classification
- 笔记 §95 (FP reform) / §107 (Phase A wave) / §114 (provenance) / §116 (audit) / §116.X user prompts
- 笔记 §163.3 + §163.4 (Option K trajectory event schema + cross-talk insight 2026-05-16)
- 笔记 §165 (A1.15 C1 chronicle, planned post-merge)


---

## §4.X.18 Artifact filename and diagnostic control disclosures (A1.5b Phase 1, B-494 + B-496)

### `observation_dom.txt` content is mode-conditional (B-494)

Per-step observation artifacts are written to `<run_dir>/<condition_id>/artifacts/<site>_task_<task_id>/step_NNN/observation_dom.txt` regardless of `condition.observation_mode`. The filename is historical (originally added when the runner only supported DOM mode) and content semantics differ by mode:

- **DOM mode**: file contains the full AXTree text (canonical "DOM observation").
- **SoM mode**: file contains the `[SOM_MARKS]` block (semantically a compressed mark list, not a DOM tree).
- **Vision mode**: file contains an empty or near-empty observation string (vision mode renders observation through screenshot, not text).
- **Phantom modes** (`phantom_dom` / `phantom_som` / `phantom_text` / `phantom_prompt`): content depends on the specific phantom variant — see paper §3.5 phantom mode taxonomy.

Readers of `observation_dom.txt` for parsing or replay **must** consult the canonical `condition.observation_mode` field (in `condition_summary_v2.json` and per-step `StepRecordV2.observation_mode`) for the actual content semantics. A mode-aware rename (e.g. `observation_som.txt` / `observation_vision.txt`) is deferred to a future schema-bump release due to ~14 consumer scripts (mechinterp tools / digest pipeline / gallery / watchdog session-health check) currently hardcoding the legacy filename. Cross-link: master_bug_catalog B-494.

### Diagnostic exploration controls use site-uniform thresholds (B-496)

The runner's diagnostic exploration controls (`_no_early_finish_control`, `_anti_repeat_control`; defined in `p79/experiment/runner/helpers.py`) use site-uniform threshold defaults:

- `min_exploration_steps = 5`
- `min_page_changes = 2`
- `min_search_attempts = 2`

Different VWA sites have systematically different natural per-step page-change rates (classifieds is form-heavy with high per-click page-changes; reddit is scroll-heavy with low per-click page-changes; shopping is mixed). A site-uniform threshold therefore produces site-correlated control fire rates — classifieds rarely triggers `_no_early_finish_control`; reddit triggers it frequently.

**Paper-grade impact**: paper-grade runs disable these controls (`cfg.diagnostic_controls.enabled = False` is the default). Any analysis citing diagnostic-control fire-rate data across sites (e.g. §6 supporting evidence using diagnostic-mode runs) must disclose this systematic bias. Per-site threshold calibration based on empirical per-site page-change rates is deferred. Cross-link: master_bug_catalog B-496.
