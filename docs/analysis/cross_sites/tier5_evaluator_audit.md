# Tier 5 Evaluator-Side Static Audit

Audit date: 2026-04-30  
Scope: `external/visualwebarena/evaluation_harness/*`, VWA/WA task configs under `external/visualwebarena/config_files`, and our FP-filter code in `p79/experiment/analysis.py` plus `scripts/analysis/*`. The requested `external/webarena/evaluation_harness/` and `external/visualwebarena/evaluation_harness/webarena_utils.py` paths are not present in this checkout; WA task configs are evaluated through the same `external/visualwebarena/evaluation_harness` fork.

## Executive Summary

Tier 5 finds evaluator-side bias in all three audited surfaces. The largest already-known issue is the `ua_match` N/A accept bias: for `reference_answers.fuzzy_match == "N/A"`, `StringEvaluator` first exact-matches `N/A`, then falls back to an LLM reason-equivalence judge. The prompt tells GPT that the task is inherently unachievable before it sees the reported reason, which makes blank or generic failure reports too easy to rationalize as "same". Our `na_fp` filter covers the common no-finish case, but not active `finish` actions with blank/generic answers.

`string_match` is not Levenshtein/token-Jaccard fuzzy matching. It is a mixture of exact lowercase comparison, token/substr containment, AND-over-list / OR-within-entry logic, and an LLM fuzzy judge. There is no normalized edit-distance threshold. In runner semantics, success requires total evaluator score `>= 1.0`; for the LLM fuzzy path the effective threshold is binary `1.0` if the completion contains `correct` and not `incorrect` / `partially correct`.

`program_html` is the most brittle static layer. It navigates to target pages, sleeps for three seconds only when the URL is not `"last"`, then evaluates raw JS/helper locators. There is no `wait_for_selector`, no causal state delta, and broad exception swallowing maps selector failures to empty strings. Across raw task configs, I counted 1,068 program-html tasks and 1,598 target checks; target locator categories are 1,100 raw `document.*` JS, 233 full-page scans, 143 CSS-through-`get_query_text`, 63 shopping helper calls, 31 reddit helper calls, 16 lambda JS, and 12 other helper calls. A simple static brittleness flag (`querySelectorAll`, child/indexed selectors, direct-child `>`, or array indexing) hits 562 target checks.

## Audit A: `ua_match` GPT-Judge Prompt

Entry points:

- `StringEvaluator.__call__`: `external/visualwebarena/evaluation_harness/evaluators.py:203`
- N/A branch: `external/visualwebarena/evaluation_harness/evaluators.py:257-270`
- `llm_ua_match`: `external/visualwebarena/evaluation_harness/helper_functions.py:619`

Full prompt template from `helper_functions.py:623-633`:

```python
message = ""
message += f"task: {question}\n"
message += f"actual unachievable reason: {reference}\n"
message += f"reported unachievable reason: {pred}\n"
message += (
    "The task described above is inherently unachievable due to the reason specified under 'actual unachievable reason'. "
    "An individual previously attempted this task and was unable to complete it. They provided a reason for their failure, "
    "which is listed under 'reported unachievable reason'. Your role is to review both the actual and reported reasons. "
    "Determine if the reported reason aligns with the actual reason, even if implicitly. "
    "If the stated reason is in line with the actual reason, respond with 'same'. Otherwise, respond with 'different'."
)
```

The call uses `generate_from_openai_chat_completion(..., temperature=0, max_tokens=768, top_p=1.0, context_length=0)` at `helper_functions.py:645-652`. The model is not fully fixed: it defaults to `gpt-4o-mini`, but `VWA_EVAL_MODEL` or `OPENAI_EVAL_MODEL` can override it at `helper_functions.py:639-643`.

Known drift modes:

1. N/A accept bias. The prompt pre-commits that the task is impossible and asks whether the reported reason aligns "even if implicitly". For blank/generic reported reasons, GPT can infer rather than verify. This matches checkpoint §78a/§27: N/A tasks were systematically over-accepted when the agent never actually reported a valid answer.

2. Active-finish N/A gap. Our filter preserves N/A successes when `agent_finished=True` (`p79/experiment/analysis.py:76-79`). In a 20-row spot check from non-archived `episode_reason_rows.csv`, the false positives removed by adjusted success were all non-finish N/A cases, but an adjusted-true N/A example exists with `final_action_type=finish` and empty `final_answer` for reddit task 31. Static implication: if the model actively emits a finish action but leaves the answer blank or generic, post-hoc `na_fp` does not challenge the GPT judge.

3. Ambiguous-instruction inconsistency. Temperature is zero, but OpenAI chat completions are still not a deterministic proof system, no seed is passed, and model deployment can change behind the fixed model alias. The prompt has no calibrated rubric for "implicitly", no examples, and no abstain path.

4. Response parser brittleness. `llm_ua_match` returns `0.0` if `"different"` appears anywhere before checking `"same"` (`helper_functions.py:653-656`). A noncompliant response such as "not different" would be marked wrong, while a longer explanation containing "same" can be marked correct despite violating "only output" instructions.

Existing coverage:

- Covered: `na_fp` removes raw successes on N/A tasks when the agent did not actively finish. See `compute_adjusted_success` lines `76-79`, and cross-representation mirror lines `438-450`.
- Not fully covered: active finishes with empty/generic wrong N/A answers; LLM nondeterminism/model override drift; false negatives from overly strict or parser-violating responses.

Recommended fixes:

- Replace `ua_match` for N/A with deterministic gates before GPT: require non-empty answer, reject generic "impossible" without site-specific evidence, and require mention of the configured `string_note` entities or a typed impossibility class.
- Rewrite the prompt to withhold the gold reason until after extracting the reported reason claim, and ask for JSON with `same: true|false`, `evidence_span`, and `missing_required_fact`.
- Run majority vote only for residual ambiguous cases. Use three samples with distinct seeds if the provider supports seed; keep temperature low but nonzero only if sampling is intentional.
- Prefer deterministic task-level N/A rules for known unreachable task families instead of GPT when the reference is exactly `"N/A"`.

## Audit B: `string_match` Fuzzy Threshold

Main function: `StringEvaluator.__call__`, `external/visualwebarena/evaluation_harness/evaluators.py:203-277`. Fuzzy helper: `llm_fuzzy_match`, `external/visualwebarena/evaluation_harness/helper_functions.py:581-617`.

Relevant function text:

```python
def clean_answer(answer: str) -> str:
    if answer.startswith("'") and answer.endswith("'"):
        answer = answer[1:-1]
    elif answer.startswith('"') and answer.endswith('"'):
        answer = answer[1:-1]
    return answer.lower()

def exact_match(ref: str, pred: Union[str, int]) -> float:
    if isinstance(pred, int):
        pred = str(pred)
    return float(StringEvaluator.clean_answer(pred) == StringEvaluator.clean_answer(ref))

def must_include(ref: str, pred: str) -> float:
    clean_ref = StringEvaluator.clean_answer(ref)
    clean_pred = StringEvaluator.clean_answer(pred)
    if len(word_tokenize(clean_ref)) == 1:
        tok_pred = word_tokenize(clean_pred)
        return float(clean_ref in tok_pred)
    else:
        return float(clean_ref in clean_pred)

def fuzzy_match(ref: str, pred: str, intent: str) -> float:
    return llm_fuzzy_match(pred, ref, intent)
```

`llm_fuzzy_match` asks GPT whether the student answer is semantically equivalent to the reference answer. It uses the same model override chain as `ua_match`, temperature `0`, and returns `0.0` for `"partially correct"` or `"incorrect"`, `1.0` for `"correct"`, otherwise `0.0`.

Threshold and normalization:

- Effective fuzzy threshold: `1.0` binary label. There is no Levenshtein, edit distance, token Jaccard, embedding similarity, or numeric threshold.
- Case sensitivity: all string evaluator paths call `clean_answer`, so exact/must_include/must_exclude/one_of are case-insensitive.
- Whitespace: no `.strip()` is applied in `clean_answer`; leading/trailing spaces can break exact matching.
- Punctuation: no punctuation normalization except removing one pair of surrounding single or double quotes. Tokenization in single-word `must_include` can incidentally handle some punctuation.
- Multi-answer behavior: the outer list is conjunctive. `must_include: ["A", "B"]` requires both. Alternatives must be encoded inside a single string as `"A |OR| B"`. `one_of` is substring-based over cleaned values.

Edge cases:

- Pred `"$5.99"` vs target `"5.99"`: `must_include` likely matches because NLTK tokenizes `"$5.99"` as `["$", "5.99"]`; `exact_match` fails; `required_values` fails because `str_2_int` cannot parse decimals/dollar signs.
- Pred `"5.99"` vs target `"$5.99"`: `must_include` likely misses because the target tokenization has two tokens and falls back to substring search for `"$5.99"` inside `"5.99"`.
- Pred `"five point nine nine dollars"` vs target `"5.99"`: deterministic paths miss. LLM fuzzy may accept, depending on task context.
- Empty answer: deterministic exact/must_include usually fails, but the N/A GPT fallback can accept empty/generic reports as discussed above.
- Numeric decimals: `required_values` casts via `str_2_int`, so decimal answers are not supported in that branch.

False positive / false negative estimate from available paper-grade diagnostics:

- Non-archived reason diagnostics contain 1,316 `string_match` rows.
- Raw `string_match` successes: 228. Adjusted successes: 141. Removed: 87, all `na_fp`.
- 20-row spot check: 12 raw-true/adjusted-false samples were N/A tasks with non-finish final actions and null final answers; 4 raw-true/adjusted-true samples included normal exact matches and active N/A finishes; 4 raw-false samples were normal numeric/count misses.

Interpretation: current filters catch the largest observed `string_match` FP family in these diagnostics, but static code still has false-negative risk for exact whitespace/punctuation and false-positive risk for broad `one_of` substring matching and active-finish N/A GPT acceptance.

## Audit C: `program_html` DOM Selector Brittleness

Entry point: `HTMLContentExactEvaluator.__call__`, `external/visualwebarena/evaluation_harness/evaluators.py:345-480`.

Selector pattern:

- Target URL can be literal, `"last"`, or `func:*` evaluated through Python `eval`.
- For non-`last` URLs, evaluator calls `page.goto(target_url)` then `time.sleep(3)` (`evaluators.py:367-369`).
- Empty locator uses `page.content()` full-page text/html.
- Locators starting `document.` or `[...document.` are evaluated as JS.
- Locators starting `lambda:` are evaluated as JS function strings.
- Locators starting `func:` are Python helper calls through `eval`.
- Required content uses exact/must_include/must_exclude/required_values/fuzzy_match with the same string evaluator semantics.

Brittleness concerns:

- No `wait_for_selector` or network-idle wait. Async DOM after the fixed 3-second sleep can produce false negatives; `"last"` targets get no sleep at all.
- Raw JS selectors are often positional: `querySelectorAll(...)[0]`, direct class names, direct child chains, and Magento/Reddit class contracts.
- Helper functions swallow exceptions and return `""` / `0` / `{}`. That converts selector breakage into ordinary evaluator failure, hiding root cause.
- `reddit_get_latest_comment_content_by_username()` checks the latest comment by username, not causal provenance. Old comments by the same account can create false positives; this is exactly the §78c reddit 69/72 failure mode.
- Full-page scans can accept stale or incidental text anywhere in the HTML, especially after deletes (`404`) or carts/wishlists where old state persists.

Task pool examples:

| Benchmark | Site | Task | Locator | Brittleness |
|---|---|---:|---|---|
| VWA | classifieds | 4 | `func:get_query_text(__page__, '.price')`, `.desc` | Class-only selectors; no wait; price formatting exactness. |
| VWA | classifieds | 5 | empty locator / full page must include `404` | Full-page scan; delete is click-only and can be overfiltered as no effective action. |
| VWA | shopping | 0 | `document.querySelector(".order-details-items.ordered").outerText` | Magento class contract; async order page render. |
| VWA | shopping | 37 | `document.querySelector('.products-grid.wishlist').textContent` | Wishlist page async render; stale wishlist state possible. |
| VWA | reddit | 19 | `document.querySelectorAll('div.submission__vote')[0]...class` | Positional vote form; class contract; click-only state. |
| VWA | reddit | 69 | `func:reddit_get_latest_comment_content_by_username(..., 'MarvelsGrantMan136')` | Latest-comment-by-user is non-causal; old matching comment can pass. |
| VWA | reddit | 80 | eight repeated `querySelectorAll(...)[0]` vote checks | Positional selector repeated across pages; exact class equality. |
| WA | shopping | 118 | empty locator / full page must include `jaw bruxism`, `mouth guard` | Full-page text can accept incidental content. |
| WA | reddit | 399 | `document.querySelector(".user-bio__biography").outerText` | Single class contract; no wait. |
| WA | webarena | 620 | `document.querySelector('.submission__inner').outerText` | Broad container text; URL plus program_html, but no causal post creation check. |
| WA | shopping_admin | 423 | `document.querySelector('input[name="product[sale]"]').value` | Attribute selector is stable-ish, but admin async form load can race fixed sleep. |

Recommended fixes:

- Add selector-level waits: `page.wait_for_selector` for CSS-derived selectors and `wait_for_load_state("networkidle")` where stable.
- Convert helper checks for mutating tasks to backend/API state-delta checks keyed by action timestamp or episode id, not "latest by username".
- For click-only state changes, define effective action as task-specific state delta, not only `type/select_option`.
- Log selector exceptions separately from content mismatch so evaluator breakage can be audited.

## Audit D: FP Filter Cross-Reference

Checkpoint §78a introduced `agent_finished = final_action_type in {finish, stop} and not fallback_finish` and preserves active N/A finishes while removing non-finish N/A successes. §78b added `string_match + success + !agent_finished -> eval_fp`. §78c extended this to `program_html`. §95 removed `visual_fp` and simplified eval_fp to:

```text
string_match:  agent_finished=False -> E-FP
program_html:  agent_finished=False and not has_effective_action -> E-FP
url_match:     no E-FP
```

Canonical implementation is `p79/experiment/analysis.py:52-87`; cross-representation mirror is `scripts/analysis/analyze_cross_representation.py:364-462`; reason diagnostics computes `has_effective_action` from only `type` and `select_option` at `scripts/analysis/analyze_reason_diagnostics.py:1978-1982`.

Covered evaluator bugs:

- N/A no-finish accept bias: covered by `na_fp`.
- Empty/fallback string answer after no active finish: covered for `string_match` when `agent_finished=False`.
- Program-html old-state/no-action false positives: partially covered when no active finish and no `type/select_option`.

Gaps:

- Active N/A finish with blank/generic answer is not covered.
- GPT judge drift/model override is not covered.
- Deterministic string false negatives from whitespace/punctuation/numeric words are not covered because filters only remove FPs.
- `program_html` selector brittleness false negatives are not covered.
- Click-only state changes are ambiguous under current `has_effective_action`; the simplified rule can remove real click-caused successes, while stale click-only successes remain hard to separate without causal state deltas.

Recommended Section 4 framing:

Use Tier 5 as an evaluator-side limitation and mitigation audit: "We report raw and adjusted success. Adjusted success removes two empirically observed evaluator false-positive families: N/A acceptance without an active answer and non-causal evaluator successes for string/program-html tasks. A static evaluator audit further identifies residual evaluator-side biases not eliminated by post-hoc filtering, including active-but-empty N/A finishes, LLM judge drift, deterministic string normalization false negatives, and brittle DOM selectors. These motivate causal state-delta evaluators or Verified-style backend monitors in future VWA evaluations."
