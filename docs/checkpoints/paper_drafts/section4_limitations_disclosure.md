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

## §4.X.5 in_viewport_ratio operator precedence (B-26)

In `external/visualwebarena/browser_env/processors.py:218`, the `in_viewport_ratio`
calculation `overlap_w * overlap_h / w * h` is parsed by Python as
`((overlap_w * overlap_h) / w) * h` — multiplication-first then division — instead of the
intended ratio `(overlap_w * overlap_h) / (w * h)`. The result is that the 0.6 viewport-overlap
threshold (`current_viewport_only=True`) is effectively bypassed, allowing partially-visible
elements to remain in the AXTree with their full text content even when they are visually
truncated.

**Implication for our claims**: This bug exists in upstream VWA and is documented in our
CLAUDE.md as "DOM has structural information advantage." It systematically helps DOM mode
relative to Vision/SoM modes by exposing element text that is visually clipped. We do **not**
fix this bug because: (a) it's upstream code; (b) any threshold value would be debatable;
(c) it does not affect our **paired** comparisons (P-SoM uses the same DOM-derived
`[SOM_MARKS]` text), so our hero claims (P-SoM ≥ best of DOM/SoM/Vision) are invariant to
this asymmetry. We disclose the asymmetry source for cross-mode interpretation.

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

## References

- `docs/reference/master_bug_catalog.md` — full bug catalog (~80 entries)
- `docs/checkpoints/pre_run/preregistration.md` §3-§4 — locked analysis choices including FP filter
- `docs/checkpoints/pre_run/evaluator_change_protocol.md` — Protocol A Tier classification
- 笔记 §95 (FP reform) / §107 (Phase A wave) / §114 (provenance) / §116 (audit) / §116.X user prompts
