# Stage 4 H1 test: indexed-list format variation

Test refined H1 hypothesis (pretraining co-occurrence shortcut):
*"input contains mark-like indexed region list → activates visual-grounding pathway"*

**Method**: For each variant V (= different text format applied to same observation), compute per-layer cosine gap between V hidden state mean and SoM (marks+image) baseline hidden state mean. Peak layer indicates **when image-axis divergence emerges**:
- Peak L04: image-presence detected freshly early → variant does NOT trigger marks-shortcut (behaves like AXTree-DOM)
- Peak L17+: image-axis divergence delayed → variant DOES trigger marks-shortcut

## Result table (sorted by peak layer)

| Variant | Format example | H1 class | Peak layer | Peak cosine gap |
|---|---|---|---|---|
| appagent_id | `id_N: label (AppAgent)` | marks-like | **L04** | 0.0488 |
| plain_numbered | `N. label (numbered)` | marks-like | **L04** | 0.0505 |
| hash_id_control | `#hash label (no integer)` | control (no integer) | **L04** | 0.0508 |
| dom | `AXTree (baseline DOM)` | AXTree-baseline | **L04** | 0.0495 |
| som_standard | `[N] role 'label' (SoM)` | marks-like | **L17** | 0.0429 |
| browser_use_at | `@N label (Browser Use)` | marks-like | **L17** | 0.0515 |
| tarsier_typed | `[BN:role:label] (Tarsier)` | marks-like | **L17** | 0.0457 |
| xml_tagged | `<el_N role='..'>label</el_N> (XML)` | marks-like | **L17** | 0.0431 |
| plain_sentence | `'a, b, c, ...' (no list)` | control (no list) | **L17** | 0.0521 |

## Grouped by H1 prediction

### marks-like  (mean peak L13)

- `[N] role 'label' (SoM)`: peak **L17** = 0.0429
- `@N label (Browser Use)`: peak **L17** = 0.0515
- `id_N: label (AppAgent)`: peak **L04** = 0.0488
- `[BN:role:label] (Tarsier)`: peak **L17** = 0.0457
- `N. label (numbered)`: peak **L04** = 0.0505
- `<el_N role='..'>label</el_N> (XML)`: peak **L17** = 0.0431

### control (no integer)  (mean peak L4)

- `#hash label (no integer)`: peak **L04** = 0.0508

### control (no list)  (mean peak L17)

- `'a, b, c, ...' (no list)`: peak **L17** = 0.0521

### AXTree-baseline  (mean peak L4)

- `AXTree (baseline DOM)`: peak **L04** = 0.0495

## H1 verdict

- **6 marks-like variants**: mean peak layer = 13, range L04-L17
- **2 control variants** (no integer / no list): mean peak layer = 10, range L04-L17
- **AXTree-DOM baseline**: peak L04

→ **H1 MIXED**: peak distribution doesn't fit simple binary prediction. Needs deeper analysis.
