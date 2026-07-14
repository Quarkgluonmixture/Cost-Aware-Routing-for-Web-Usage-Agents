---
title: "A Polarity Bug in the VisualWebArena LLM Judge: Implications for Cross-Paper SR Comparability"
status: draft-v0
date: 2026-07-02
target: evaluation-systems workshop TBD
track: B
origin: workshop_subpaper_plan.md §2 + master_bug_catalog B-91 + B-535
---

## Abstract

VisualWebArena (VWA) evaluates success on free-form string tasks via an LLM judge
(`llm_fuzzy_match`) and on not-applicable tasks via a sibling function (`llm_ua_match`).
We identify two long-standing defects in the upstream evaluation harness that together inflate
Success Rate (SR): (i) the judge is never guarded against empty-answer predictions, so agents
that exit without submitting a real answer may be incorrectly credited; and (ii) the
positive-verdict parsing is vulnerable to negation confusion — the upstream harness blocks
`"incorrect"` and `"partially correct"` but scores negations such as `"not correct"` as
positive (1.0) through an `assert "correct" in response` fallthrough, and *crashes* rather
than failing closed on responses matching neither phrase.  (An intermediate rewrite in our own
derived harness briefly widened this hole to a bare positive-substring test before both
variants were closed; we disclose the attribution split explicitly in §2.2.)  The upstream
pair propagates unmodified to every derived study that inherits the benchmark harness.  We describe the
mechanism, document minimal source-level patches, and discuss implications for cross-paper SR
comparability in the WebArena/VWA benchmark family.

---

## 1  Introduction

LLM-as-judge evaluation has become the default mechanism for scoring free-text and
task-completion outcomes in web-agent benchmarks.  VisualWebArena \citep{koh2024visualwebarena}
and its WebArena predecessor \citep{zhou2024webarena} delegate fuzzy string matching and
not-applicable (N/A) task scoring to GPT-4-class models: the agent's final answer is passed to
a judge prompt, and the judge responds with a natural-language verdict that the harness parses
into a binary 0/1 success signal.

This pipeline places correctness of the evaluation squarely on two properties of the judge
integration code: (a) that the judge is called only when the agent has produced a meaningful
answer, and (b) that the parsing of the judge's natural-language response correctly maps
positive verdicts to 1 and negative verdicts to 0.  A failure in either property silently
inflates SR for every study that uses the upstream harness without modification.

Recent work has raised broader concerns about evaluator reliability in web-agent benchmarks.
\citet{xue2025illusion} argue that reported SR improvements in the WebArena family may
partially reflect evaluator artefacts rather than genuine capability gains.
\citet{wu2026mobilegym} measure VLM-judge error at 10.2\% on a controlled mobile-GUI task
set.  \citet{bai2026webgym} report that rubric revision for a large-scale web-training judge
trades precision (73\%$\to$93\%) against recall (96\%$\to$85\%), describing residual judge error
as "almost impossible to eliminate."  Even purpose-built deterministic evaluators show
$\sim$5\% error under human audit \citep{zhang2026infiniteweb}.

Against this backdrop, a polarity-level bug in the VWA judge integration---one that inverts
the sign of the verdict for an identifiable class of judge responses---constitutes a
systematic, non-random inflation of SR that affects every study in the benchmark family.  We
argue this warrants a dedicated disclosure note, complementing the one-sentence mention in a
companion controlled study, because: (1) the fix is non-trivial for downstream reproducers who
do not know the bug exists; (2) the magnitude of the effect on cross-paper comparability is
unknown and potentially large; and (3) a minimal, backward-compatible patch is available and
can be adopted by any VWA-derived study without changing the judge model or task definitions.

---

## 2  The Bug: Mechanism and Discovery

### 2.1  Empty-prediction false positive (B-91)

**Location**: `external/visualwebarena/evaluation_harness/helper_functions.py`,
functions `llm_fuzzy_match` and `llm_ua_match` (the B-91 guards were inserted at lines 589
and 634 of fix commit `f0c835b`; in the current patched file the two functions sit at lines
588 and 678 with guards at 596 and 685).

**Mechanism**: When a web agent exhausts its step budget or encounters an unrecoverable error,
the VWA runner appends a synthetic stop action with `answer=""`.  This empty string is then
passed to the LLM judge as the predicted answer.  Neither `llm_fuzzy_match` nor `llm_ua_match`
guarded against an empty or whitespace-only prediction prior to the fix.  Two failure paths
follow.  The judge may occasionally emit a positive verdict outright on an empty prediction;
more commonly it *correctly* emits a negative verdict — and if that verdict lands in the
negation class of §2.2 (`"not correct"`), the polarity-vulnerable parsing flips it to a
positive score.  The two defects therefore compound rather than act independently: B-91
supplies the degenerate input, and the §2.2 parsing hole converts the judge's correct
rejection of it into a credited success.  In either path the harness credits a task the
agent never answered.

The deterministic evaluation paths (`exact_match`, `must_include`, `one_of`,
`required_values`) already return 0 on empty predictions by construction.  The failure is
specific to the two LLM-judge paths.

**Discovery**: Identified by code inspection during an audit of the VWA evaluation harness
(2026-05-14).  Literature context: WebArena-Verified classifies N/A scoring as an "evaluation
mechanism issue"; \citet{lu2025agentrewardbench} provide a systematic framework for
categorising evaluator failure modes in web-agent benchmarks.
(来源: `docs/reference/master_bug_catalog.md` B-91)

**Fix (B-91, commit `f0c835b`, submodule branch `p79-patches`)**: A deterministic guard is
inserted at the top of both `llm_fuzzy_match` and `llm_ua_match`:

```python
if not pred or not pred.strip():
    return 0.0
```

This returns 0.0 before any judge API call, eliminating the false-positive path without
changing behaviour for non-empty predictions.

### 2.2  Polarity inversion false positive (B-535)

**Location**: `helper_functions.py`, response parsing in `llm_fuzzy_match`; sibling
vulnerability in `llm_ua_match`.

**Mechanism — upstream (baseline commit `89f5af2`)**: the upstream parser is

```python
if "partially correct" in response or "incorrect" in response:
    return 0.0
else:
    assert "correct" in response, response
    return 1.0
```

Two defects.  (a) *Negation fallthrough*: negative verdicts that avoid the two blocked
phrases — `"not correct"`, `"not exactly correct"` — fall through to the `assert`, which
passes (the substring `"correct"` is present) and returns 1.0.  (b) *Crash instead of
fail-closed*: any response containing none of the phrases raises `AssertionError`,
aborting evaluation rather than returning 0.

**Attribution split — intermediate widening (our derived branch only)**: an early local
rewrite in our derived harness (state `1c3a615^`) replaced the assert block with a bare
positive-substring test (`if "correct" in response: return 1.0`), which *widened* the hole
to score `"incorrect"` and `"partially correct"` as positive — strictly worse than upstream.
We state this explicitly so the cross-paper claim is not overreached: the widest variant
existed only in our derived branch; **the upstream defect inherited by other studies is the
narrower negation-fallthrough + assert-crash pair**.

**Discovery**: adversarial code audit (monkeypatch-verified, 2026-05-17): mock responses
`"incorrect"` / `"partially correct"` / `"not correct"` each returned 1.0 under the
then-current derived harness; `"not correct"` also returns 1.0 under upstream `89f5af2`
semantics.  (来源: `docs/reference/master_bug_catalog.md` B-535)

**Fix (B-535, commit `1c3a615`, submodule branch `p79-patches`)**: negative-first, anchored,
fail-closed (actual patched code):

```python
resp = response.strip()
if resp.startswith(("incorrect", "partially correct", "not correct")):
    return 0.0
if resp.startswith("correct"):
    return 1.0
_log_unexpected_judge_response(...)   # fail-closed default
return 0.0
```

The same pattern is applied to `llm_ua_match`.  Anchored `startswith` on the stripped
response (rather than substring containment) removes the entire "negation containing the
positive token" class; responses matching neither pattern return 0.0 and are logged for
audit.

---

## 3  Measured Impact

The magnitude of the SR inflation from these bugs depends on (a) how frequently the judge
produces the affected response strings, and (b) how large the fraction of tasks evaluated by
the LLM-judge paths is.

**Fraction of tasks affected**: In VWA, the `llm_fuzzy_match` path covers string-match tasks
(tasks where `eval.fuzzy_match` is a human-readable reference answer).  The `llm_ua_match`
path covers N/A tasks (tasks where `eval.fuzzy_match == "N/A"`).  On the VWA classifieds
site, 10 of 234 tasks (4.3\%) are N/A tasks; on reddit, 5 of 210 tasks (2.4\%); on shopping,
31 of 466 tasks (6.6\%).  Non-N/A `fuzzy_match` tasks---the branch that calls
`llm_fuzzy_match` on a human-readable reference answer---constitute **0 of 234** classifieds
tasks, **0 of 210** reddit tasks, and **0 of 466** shopping tasks in the checked-in VWA
configs.  All `fuzzy_match` entries in these three release splits are instead exactly `"N/A"`
(10/5/31 respectively) and therefore enter the sibling `llm_ua_match` path if exact N/A
matching fails.  Thus B-535's fuzzy-parser branch has zero direct task-config exposure in
these splits, while its sibling UA-parser polarity bug is exposed through the N/A tasks.
(来源 [V]: `external/visualwebarena/config_files/vwa/test_{classifieds,reddit,shopping}.raw.json`,
counted 2026-07-14; command and output archived in
`docs/checkpoints/codex_outputs/trackb_tbd_2026-07-14.md`)

**Quantitative SR delta (polarity inversion, B-535)**: **Blocked on a paired evaluator-replay
run; no numerical delta is recoverable from the current repository.**  The bug catalog records
no pre/post-fix SR, and the local episode archive retains final `score`/`success` but not every
raw LLM-judge response needed to apply both parsers to identical verdict text.  A valid estimate
requires a frozen shared trajectory/output archive for an explicitly identified study, one
instrumented judge pass that persists each raw response (with task, prediction, judge model and
version), and offline replay of those same responses through the upstream `89f5af2` parser and
the fixed parser; the result should report site-stratified paired
$\mathrm{SR}_{\text{upstream}}-\mathrm{SR}_{\text{fixed}}$ with uncertainty.  A genuinely
cross-paper estimate additionally requires the prior paper's raw outputs, which are not present
here.
(来源 [V]: `docs/reference/master_bug_catalog.md` B-91/B-535 + repository archive-schema
inspection, 2026-07-14; evidence commands in the Track B TBD report)

**Qualitative scope**: Every VWA-derived paper that uses the upstream `helper_functions.py`
without modification is exposed to both bugs.  This includes the original VWA paper
\citep{koh2024visualwebarena}, WebArena-Verified (non-arXiv venue; needs manual bib entry),
PAE \citep{zhou2025pae}---noted in the literature review as reporting that around half of its
successful WebArena trajectories were evaluator false positives, independently of this
polarity bug---and other WebArena-family studies.

**Within-paper comparisons are attenuated, not invalidated**: the polarity fallthrough is
*not* additive noise that cancels in a paired difference.  It fires precisely when the judge
emits a negative verdict — disproportionately on the *weaker* arm of any comparison — so on
the LLM-judged task subset it awards more spurious successes to the weaker condition and
compresses the measured margin toward zero (a differential compression of effect sizes, not a
cancellation).  Within-paper rankings computed on a uniformly buggy harness therefore tend to
*understate* true differences on that subset; sign reversals are not expected under this
mechanism, but effect-size estimates are biased toward zero.  The cross-paper problem is
sharper still: a study using the patched harness will report lower absolute SR than a study
using the upstream harness on the same task set, with no clean way to bridge the gap without
re-running baselines.

**Cross-paper SR gap introduced by the patch**: After applying B-535, absolute SR numbers
from the fixed harness are not directly comparable to published SR numbers from any prior VWA
or WebArena paper.  The direction of the gap is deterministic (patched SR ≤ upstream SR on
any task where the judge emitted a polarity-inverted response), but the magnitude is
unknown without re-evaluation.
(来源: `docs/reference/master_bug_catalog.md` B-535 cross-paper SR comparability note)

---

## 4  Broader Pattern

The two bugs identified here fit a recurring pattern in the evaluator-reliability literature.
LLM judges are typically integrated via a single natural-language API call, and the response
parsing step is a simple string operation assumed to be correct by construction.  Because the
judge itself is non-deterministic, evaluation errors are easy to attribute to randomness and
hard to detect without systematic adversarial testing.

\citet{wu2026mobilegym} address a related failure mode in their AnswerSheet protocol: their
deterministic judge avoids both false-reject (paraphrase-miss) and false-accept (gold-answer
leakage substring) failures precisely because they avoid natural-language response parsing
altogether.  The negation fallthrough in the upstream VWA parser (and the bare-substring
variant in our derived branch) is an instance of the false-accept class: a token intended to
identify positive verdicts also matches negative verdicts that contain it.

\citet{bai2026webgym} observe that rubric revision for their GPT-4o-based judge trades
precision against recall, concluding that residual judge error is "almost impossible to
eliminate."  The polarity bug studied here is qualitatively different: it is a deterministic
code error that is entirely eliminable by inverting the check order.  It does not require
judge fine-tuning, rubric redesign, or human annotation to fix.

\citet{xue2025illusion} raise the concern that apparent SR progress in the WebArena family
may be artefactual.  A polarity-level evaluator bug that assigns score 1.0 to negative
judge responses provides a concrete mechanism by which this could occur: an agent that
performs better on tasks where the judge tends to emit polarity-inverted responses will appear
to outperform a weaker agent by more than its true margin.

The deterministic-evaluator literature \citep{zhang2026infiniteweb} suggests that even
purpose-built deterministic evaluators retain $\sim$5\% error under human audit.  This
baseline noise figure contextualises the polarity bug: if the buggy judge emits
polarity-inverted responses on even a small fraction of tasks, it can contribute noise
comparable to the irreducible baseline error of a purpose-built deterministic evaluator.

\citet{lu2025agentrewardbench} propose a structured framework for categorising web-agent
reward-model failures by type (format error, semantic error, hallucination, etc.).  The
polarity bug fits their "parsing error" category: the judge produces semantically correct
output, but the harness misinterprets it.

---

## 5  Recommendations

### 5.1  Polarity and negation test vectors

Evaluator integration code should include unit tests that directly exercise the response
parsing logic with polarity-inverted strings.  A minimal test battery for a binary
correct/incorrect judge should include at minimum: `"correct"`, `"incorrect"`,
`"partially correct"`, `"not correct"`, `"the answer is correct"`, `"the answer is incorrect"`,
an empty string, and a whitespace-only string.  For N/A tasks, the equivalent battery should
cover `"same"`, `"not the same"`, `"different"`, and `"not different"`.  These test vectors
should be run against the actual parsing function, not just the judge prompt.

### 5.2  Evaluator version pinning and disclosure

Papers using LLM-as-judge evaluation should pin the evaluator harness to a specific commit
SHA and disclose: (a) the judge model and version, (b) the harness commit, and (c) any
patches applied relative to the upstream baseline.  This enables reproducers to identify
whether their setup is comparable to the reported numbers.  The companion study to this note
records the evaluator SHA in a provenance manifest captured before each experimental run, so
that OSF reproducibility auditors can verify the canonical patched harness was active.

### 5.3  Deterministic-first design

Where possible, evaluation design should prefer deterministic evaluation paths over
LLM-judge paths, as these eliminate the class of bugs described here entirely.  When LLM
judges are required, the response parsing step should be treated as a security-critical
surface: negative phrases must be tested *before* positive ones (negative-first — the
reverse order re-opens the `"not correct"` class), matching should be anchored
(`startswith` on the stripped response, or word-boundary regex `\bcorrect\b`) or replaced
by structured output rather than substring containment, and the parser should fail-closed
(return 0.0 and log) on any response matching neither pattern.  The fix in B-535
demonstrates that this redesign requires fewer than ten lines of code.

---

## 6  Limitations

This note characterises two bugs and their patch, but does not quantify the empirical SR
inflation across the VWA benchmark family.  A full cross-paper re-evaluation would require
access to raw model outputs from prior VWA studies; these are not publicly available for all
papers in scope.  The note therefore provides a qualitative scope assessment rather than a
numerical SR-delta estimate.  The companion controlled study (a companion controlled study
using the patched harness) provides within-study evidence that the patch is correctness-
improving; cross-study comparability remains an open empirical question.

---

## References

\bibliography{paper}
