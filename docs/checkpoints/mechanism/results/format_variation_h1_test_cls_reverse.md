# Stage 4 H1 test: indexed-list format variation

Test refined H1 hypothesis (pretraining co-occurrence shortcut):
*"input contains mark-like indexed region list → activates visual-grounding pathway"*

**Method**: For each variant V (= different text format applied to same observation), compute per-layer cosine gap between V hidden state mean and SoM (marks+image) baseline hidden state mean. Peak layer indicates **when image-axis divergence emerges**:
- Peak L04: image-presence detected freshly early → variant does NOT trigger marks-shortcut (behaves like AXTree-DOM)
- Peak L17+: image-axis divergence delayed → variant DOES trigger marks-shortcut

## Result table (sorted by peak layer)

| Variant | Format example | H1 class | Peak layer | Peak cosine gap |
|---|---|---|---|---|
| dom | `AXTree (baseline DOM)` | AXTree-baseline | **L04** | 0.0434 |
| plain_sentence | `'a, b, c, ...' (no list)` | control (no list) | **L22** | 0.0529 |
| som_standard | `[N] role 'label' (SoM)` | marks-like | **L36** | 0.0429 |
| browser_use_at | `@N label (Browser Use)` | marks-like | **L36** | 0.0520 |
| appagent_id | `id_N: label (AppAgent)` | marks-like | **L36** | 0.0526 |
| tarsier_typed | `[BN:role:label] (Tarsier)` | marks-like | **L36** | 0.0475 |
| plain_numbered | `N. label (numbered)` | marks-like | **L36** | 0.0518 |
| xml_tagged | `<el_N role='..'>label</el_N> (XML)` | marks-like | **L36** | 0.0439 |
| hash_id_control | `#hash label (no integer)` | control (no integer) | **L36** | 0.0516 |

## Grouped by H1 prediction

### marks-like  (mean peak L36)

- `[N] role 'label' (SoM)`: peak **L36** = 0.0429
- `@N label (Browser Use)`: peak **L36** = 0.0520
- `id_N: label (AppAgent)`: peak **L36** = 0.0526
- `[BN:role:label] (Tarsier)`: peak **L36** = 0.0475
- `N. label (numbered)`: peak **L36** = 0.0518
- `<el_N role='..'>label</el_N> (XML)`: peak **L36** = 0.0439

### control (no integer)  (mean peak L36)

- `#hash label (no integer)`: peak **L36** = 0.0516

### control (no list)  (mean peak L22)

- `'a, b, c, ...' (no list)`: peak **L22** = 0.0529

### AXTree-baseline  (mean peak L4)

- `AXTree (baseline DOM)`: peak **L04** = 0.0434

## H1 verdict

- **6 marks-like variants**: mean peak layer = 36, range L36-L36
- **2 control variants** (no integer / no list): mean peak layer = 29, range L22-L36
- **AXTree-DOM baseline**: peak L04

→ **H1 PARTIAL**: marks-like AND controls all peak late — finding is broader than 'indexed list' (any text payload triggers).
