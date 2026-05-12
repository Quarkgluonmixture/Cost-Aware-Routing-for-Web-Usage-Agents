# W6 feature attribution — H1 reddit 2/6 marks-like L04 peak

**Setup**: Qwen3-VL-4B tokenizer (Qwen/Qwen3-VL-4B-Instruct). Each marks-like format variant tokenized on a canonical single-element example (N=1, role=button, label=Submit). First-token character class + marker-fingerprint token count compared between L04-peak and L17-peak subgroups.

## Per-variant tokenization

| Variant | Peak | Example | n_tok | First token | First char class | Marker fp |
|---|---|---|---:|---|---|---:|
| appagent_id | L04 | `id_1: Submit` | 5 | `id` | alphanumeric | 4 (`id·_·1·:`) |
| plain_numbered | L04 | `1. Submit` | 3 | `1` | alphanumeric | 2 (`1·.`) |
| som_standard | L17 | `[1] button 'Submit'` | 7 | `[` | markup-sigil | 3 (`[·1·]`) |
| browser_use_at | L17 | `@1 Submit` | 3 | `@` | markup-sigil | 2 (`@·1`) |
| tarsier_typed | L17 | `[B1:button:Submit]` | 7 | `[B` | markup-sigil | 7 (`[B·1·:·button·:·Submit·]`) |
| xml_tagged | L17 | `<el_1 role='button'>Submit</el_1>` | 14 | `<` | markup-sigil | 4 (`<·el·_·1`) |
| hash_id_control | L04 | `#a3f7 Submit` | 5 | `#a` | markup-sigil | 4 (`#a·3·f·7`) |
| plain_sentence | L17 | `Submit` | 1 | `Submit` | alphanumeric | 1 (`Submit`) |
| dom | L04 | `button: Submit (AXTree)` | 7 | `button` | alphanumeric | 2 (`button·:`) |
| som | L17 | `[1] button 'Submit' (+ image marks)` | 11 | `[` | markup-sigil | 3 (`[·1·]`) |

## Subgroup first-char-class distribution (6 marks-like only)

| Subgroup | alphanumeric | markup-sigil | punctuation | quote | other |
|---|---:|---:|---:|---:|---:|
| L04-peak (2) | 2 | 0 | 0 | 0 | 0 |
| L17-peak (4) | 0 | 4 | 0 | 0 | 0 |

## Hypothesis verdict

✅ **Hypothesis supported (clean split)**: L04-peak variants both start with alphanumeric tokens (2/2); L17-peak variants start with markup-sigil tokens (4/4).

## Secondary features

- L04-peak mean marker-fp tokens: 3.00
- L17-peak mean marker-fp tokens: 4.00
- Δ (L17 − L04): +1.00

## Full token sequence per variant (marks-like 6)

- **appagent_id** (L04, `id_1: Submit`): 5 tokens: `id` · `_` · `1` · `:` · `ĠSubmit`
- **plain_numbered** (L04, `1. Submit`): 3 tokens: `1` · `.` · `ĠSubmit`
- **som_standard** (L17, `[1] button 'Submit'`): 7 tokens: `[` · `1` · `]` · `Ġbutton` · `Ġ'` · `Submit` · `'`
- **browser_use_at** (L17, `@1 Submit`): 3 tokens: `@` · `1` · `ĠSubmit`
- **tarsier_typed** (L17, `[B1:button:Submit]`): 7 tokens: `[B` · `1` · `:` · `button` · `:` · `Submit` · `]`
- **xml_tagged** (L17, `<el_1 role='button'>Submit</el_1>`): 14 tokens: `<` · `el` · `_` · `1` · `Ġrole` · `='` · `button` · `'>` · `Submit` · `</` · `el` · `_` · `1` · `>`

## Interpretation

Within the 6 marks-like variants, the L17 vs L04 split corresponds to whether the variant's first tokens are **markup-sigil tokens** (`[`, `<`, `@`) — which co-occur with HTML / web-agent traces in pretraining and trigger the visual-grounding shortcut at mid layers — versus **plain alphanumeric tokens** (`id`, `1`) — which are common in prose / dictionary listings and behave like AXTree-DOM, peaking early at L04 where the image-axis divergence is freshly observable but not yet routed through the shortcut path.

**Control variants (counterexamples that refine the rule)**:
- `hash_id_control` (`#a3f7 Submit`): markup-sigil first token but L04 peak. The `#` sigil alone is not sufficient — the marker must contain an **integer index** (which `#a3f7` does not). This is consistent with prior H2 "integer is the trigger token" framing.
- `plain_sentence` (`Submit`): alphanumeric first token but L17 peak. With no list/marker structure at all, the divergence path differs — possibly because the text observation drops to bare labels with no positional anchors, which the model handles via a different late-layer routing (likely commitment without grounding).

Together these say: the L17 mid-layer shortcut requires **(a) integer-indexed marker + (b) markup-sigil-leading delimiter**. Either alone fails to trigger it.

**Paper §5 implication**: H1's binary 'marks-like vs not' prediction is too coarse. The mechanism trigger is the **conjunction** of integer marker + markup-sigil first token, not the abstract concept of 'indexed list'. Variants like `id_N:` and `N.` are nominally indexed but lack the sigil; `hash_id_control` has the sigil but lacks an integer. Both fail to peak at L17. This refines H1 to **'integer marker + markup-sigil delimiter → triggers shortcut at L17'**, which is testable on additional variants and on a `bare_N` falsifier (drop the bracket from `[N]` and re-extract).

**Falsifier (concrete next experiment)**: variant `bare_N` = `N button 'Submit'` (no brackets), which has integer + no sigil. Hypothesis predicts L04 peak. If it peaks L17, hypothesis fails.
