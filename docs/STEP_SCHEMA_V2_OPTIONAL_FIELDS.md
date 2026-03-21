# StepRecordV2 Optional Field Semantics

This document defines optional `StepRecordV2` fields used for RQ3 interpretability.

## `page_change_reasons`
- Type: `List[str]`
- Meaning: structured reasons why the page is considered changed between adjacent steps.
- Typical values:
  - `url_changed`
  - `title_changed`
  - `content_changed`
  - `form_fields_changed`
  - `m3_retry_applied`

## `text_similarity`
- Type: `float | null`
- Meaning: text similarity score used in state-change detection (higher means more similar).
- Used with `state_change.similarity_threshold`.

## `checklist`
- Type: `object | null`
- Shape:
  - `items`: checklist item array with `id`, `description`, `status`
  - `status`: aggregate status (`total`, `completed`, `in_progress`, `pending`, `failed`, `completion_rate`)
- Lifecycle:
  - Updated after each step when `checklist.enabled=true`.
  - `null` when checklist module is disabled.

## `state_digest`
- Type: `object | null`
- Meaning: lightweight before/after page digest to support debugging and explainability.
- Current keys:
  - `url_before`, `url_after`
  - `title_before`, `title_after`

## `error_category`
- Type: `str | null`
- Normalized categories (do not rely on raw backend failure strings):
  - `parse_error`
  - `invalid_action`
  - `no_progress`
  - `env_error`
  - `benchmark_noise`
- `null` means no step-level error signal was detected.
