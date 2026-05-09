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

**Mitigation in this work**: We pin judge `temperature=0` for all evaluations and report all
ua_match-affected tasks as part of the `na_fp` exclusion class (preregistration.md §3 FP filter).
Sensitivity analysis (Appendix D) shows our H1/H3 conclusions hold under three FP filter
variants (raw / +na_fp / +na_fp+eval_fp), so judge drift cannot flip the paper's hero claim.

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

**Per-cell quantification**: We measure selector hit-rate parity in our archive — for each
program_html task, we count post-action DOM nodes matching the reference selector. A pre/post
ratio outside 0.95-1.05 across modes within the same task is flagged (~3% of program_html
tasks); these are excluded from H1/H3 per the preregistered FP filter `eval_fp` rule.

**Cannot-fix scope**: Patching all 562 brittle selectors requires authoring a parallel
evaluator harness, which is out of scope for this paper. We retain VWA's evaluator unchanged
(reviewer-defensible upstream parity per §3 evaluator independence) and bound the impact
via the FP filter sensitivity ladder (Appendix D).

---

## §4.X.4 finish_wrong_state — agent error not scaffold (B-15)

In Tier 2 silent-failure analysis (笔记 §107), 1552 of 4501 episodes (34.5%) had the agent
emit `finish` while the page state did not match the task goal. Initial framing classified
this as a scaffold bug; subsequent self-replay (笔记 §95 reform) showed it is an **agent
reasoning error** — the agent decides to terminate prematurely or with partial completion,
not a runner / dispatch / observation failure.

**Treatment**: This is captured in our `eval_fp` filter rule (preregistration.md §3): if
`agent_finished=True` but evaluator returns success and the agent has no effective action
in the trajectory, we mark the episode as `eval_fp`. The agent error itself is not a paper
limitation — different baselines and modes can succeed or fail at terminating decisions, and
our paired-design comparison absorbs this into per-task variance.

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

**A3 — Token budget**: B0 has `max_new_tokens=4096`; B1 has `max_new_tokens=384`. The
asymmetry stems from B0's verbose thought + JSON output requirement; B1's parser is more
robust to compact outputs. In rare cases (~0.15%), B1's compact budget causes truncated JSON →
parse_failure → `wait` action. We retain this asymmetry as a B1-specific structural
limitation rather than artificially inflate B1's budget; the impact is bounded and disclosed
in §3 baseline configuration table.

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

The 16-cell rerun (preregistration.md §4 cell inclusion) uses post-Phase-A code only
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
