# Pipeline Audit — Phase 1 (script-level) + Phase 2 (macro methodology)

**Context**: User pivot 2026-05-13 afternoon — stop prose-chasing, audit pipeline before next data extraction. Goal: make analysis scripts paper-grade so new data (matched-N v2 NPZ, cross-family Phi-3.5 + Qwen2-VL-7B, reverse-tier) one-shot produces conclusions.

**Method**: 10-question methodology checklist (Phase 1) + 8-dimension framework audit (Phase 2). Claude + codex parallel, scope-split.

---

## Phase 1: script-level findings (Claude + codex, 10 scripts, 21 findings)

### 10-question methodology checklist

| # | Question | Rationale |
|---|---|---|
| Q1 | Per-task vs of-means averaging (Jensen check)? | OOB attack today |
| Q2 | dtype precision (fp32 vs bf16)? | sub-permille precision |
| Q3 | Layer-index convention (block-input vs -output)? | cross-pipeline drift |
| Q4 | Sample size / tier / step config consistent? | v1/v2 confound |
| Q5 | Silent failure handling (raise vs continue)? | unbalanced NPZ |
| Q6 | Statistical procedure (paired/unpaired, held-out, MC)? | reviewer-3 target |
| Q7 | Bootstrap CI on reported magnitudes? | standard demand |
| Q8 | Control variants (random / task-shuffle / reverse)? | causal defense |
| Q9 | Provenance (git_dirty, formatter hash, revision pin)? | reproducibility |
| Q10 | Code↔docstring consistency? | trust-but-verify |

### P0 findings (8 total) — fix BEFORE next data extraction

| # | File:Line | Bug | Impact | Effort | Source |
|---|---|---|---|---|---|
| **P0-1** | `stage4_logit_lens_axis2.py:114-127` | Applies `lm_head` to `mode-mean hidden state` → KL of decoded-means, not per-task KL averaged. Jensen + softmax non-linearity makes the "amplification 7-10×" framing terminology-sensitive | Hero claim §1.2 "cosine-causal disjoint" — confirmed 1.1-3.9× per-task vs of-means via `stage4_logit_lens_per_task.py` defuse fire; reddit axis-1 understated 2.9-3.9× | 2-3h (rerun + prose) | Claude |
| **P0-2** | `run_stage4_multimode_extract.py:227-228` + `run_stage4_format_variation_extract.py:181-196` | Silent per-mode extraction failures → partial NPZ. `logger.error` not raise; only ALL-tasks-fail → SystemExit | matched-N v2 NPZ could be ragged (some modes 23/24, others 24/24) without warning; cosine/logit/steering downstream uninterpretable | 1h | Codex |
| **P0-3** | `run_stage4_format_variation_extract.py:8-11 / 169-171` | Docstring claims variants cluster with baseline P-text, but `ALL_MODES` has 8 variants + dom + som, **no phantom_text / phantom_som** | H1 test main anchor missing — claim "marks-like variants cluster with P-text" cannot be verified because P-text wasn't extracted | 45-90min | Codex (OOB) |
| **P0-4** | `run_stage4_method44_v2_sweep.py:104-179` | Direction fit on full NPZ then evaluated on same tasks — in-sample causal steering, not held-out | §5.3 "L33 α=10 H-mean 0.33 sweet spot" claim potentially leakage; cohort-memorized direction, not transferable mechanism | 2-3h | Codex |
| **P0-5** | `run_stage4_method44_v2_sweep.py:55` | Defaults to `hidden_states.npz` (v1 buggy NPZ), while other Stage 4 scripts use `hidden_states_v2_fixed.npz` | Method 4.4 steering direction computed on v1 buggy data; downstream §5.3 incoherent with cosine/logit-lens §1.2 (different NPZ) | 20min | Codex |
| **P0-6** | `stage4_robustness.py:25` | Hardcoded `NPZ = hidden_states.npz` (v1 buggy). The 5/5 robustness pass paper §5 cites is still on v1 data | §5 "robustness 5/5 pass" claim is based on v1 buggy NPZ; needs re-run on v2 to validate v2 results pass same robustness suite | 30min rerun + verify | Claude |
| **P0-7** | `stage4_axis2_layer_profile.py:60-72` | `mean → cosine` over all examples, not `(task_id, step)` matched-pairs cosine averaged | §7.3.0 / §5.7 axis-2 layer profile claim could be aggregate artifact (mixing task content differences into layer profile) | 1-2h | Codex |
| **P0-8** | `stage4_logit_lens_axis2.py:86` | `.to(lm_head.weight.dtype)` forces fp32 hidden → bf16 before lm_head; bf16 has ~3 decimal precision; KL 4th decimal unreliable for sub-permille mean-diff | "KL 8-44×" claim's precision boundary; pair magnitudes 0.05-0.09 in bf16 have ±5e-4 noise floor | 1h refactor + rerun | Claude |

### P1 findings (8 total) — defendable but should fix

| # | File:Line | Bug | Effort | Source |
|---|---|---|---|---|
| P1-1 | `stage4_pca_cosine_gap.py:88` | AUROC direction = (c1-c2)/||c1-c2|| same as cosine direction; AUROC=1.0 + cos=0.005 doesn't mean "well separated" geometrically | 1-2h prose + perturbation AUROC | Claude |
| P1-2 | `stage4_axis2_per_task_fragility.py:73-83` | Doesn't assert complete (task × step × mode) grid | 30-45min | Codex |
| P1-3 | `stage4_axis2_per_task_fragility.py:89-111` | No bootstrap CI / MC correction on per-task fragility | 1h | Codex |
| P1-4 | `run_stage4_method44_v2_sweep.py:87-89, 197-211` | `is_json_valid` = `startswith('{' or '"')` heuristic; markdown says "JSON valid" but it's first-char only | 15min (rename + json.loads) | Codex |
| P1-5 | `run_stage2b_continuation_pilot.py:237-259` | Provenance snapshot failure non-fatal | 20min | Codex |
| P1-6 | All Stage 4 analysis | No bootstrap CI on cosine peak layer location | 1h sweep | Claude |
| P1-7 | All Stage 4 + Stage 2/3 | Layer-index convention not disambiguated in docs; Method 4.4 uses `patcher.layers[L] ↔ npz[:, L+1, :]` (offset+1) while layer-profile uses `H[:, L, :]` directly — potential off-by-one cross-pipeline | 30min documentation + assertion | Codex (cross-pipeline flag) |
| P1-8 | `stage4_logit_lens_axis2.py` | Vanilla logit lens has known mid-layer artifact (Belrose 2023 tuned lens); not cited in paper §5 | 1h prose + caveat | Claude |

### P2 findings (5 total) — cosmetic / doc updates

| # | File:Line | Bug | Effort |
|---|---|---|---|
| P2-1 | `stage4_axis2_layer_profile.py:101, 114` | Hardcoded "288 ex" in markdown (actual N=144) | 15min dynamic |
| P2-2 | `stage4_layer_axis_emergence.py:14, 130-140` | Docstring + result MD prose still v1 framing ("AXTree → L04 vs flat → L17-L36") — v2 data shows cls/reddit divergence; result MD inherits stale prose | 30min |
| P2-3 | `run_stage4_format_variation_extract.py:210-217` | No dtype normalization (`np.asarray(h, dtype=np.float32)`) or shape assertion | 20min |
| P2-4 | `run_stage2b_continuation_pilot.py:441-451` | Plot uses mean±std not bootstrap CI band | 45min |
| P2-5 | All extraction scripts | No `git_dirty=false` enforcement; current v2 NPZs have `git_dirty: true` in provenance | 30min CLI flag |

### Phase 1 source attribution

| Audit Source | P0 | P1 | P2 | Total |
|---|---:|---:|---:|---:|
| Claude (5 scripts: cosine_gap / logit_lens_axis2 / robustness / layer_axis_emergence / run_stage4_multimode_extract) | 4 | 3 | 2 | 9 |
| Codex (5 scripts: axis2_layer_profile / axis2_per_task_fragility / format_variation_extract / run_stage2b / method44_v2_sweep) | 4 | 5 | 3 | 12 |
| **Merged** | **8** | **8** | **5** | **21** |

### Phase 1 cross-pipeline coherence risks (urgent)

1. **Layer-index off-by-one** (codex flag): Method 4.4 uses `patcher.layers[L] ↔ npz[:, L+1, :]`, but analysis scripts use `H[:, L, :]` directly with `L0 = embedding, L36 = final block`. If PCA / logit-lens uses different convention, **L17 / L23 claim could be off by 1 layer**.
2. **NPZ source mismatch**: Method 4.4 sweep reads `hidden_states.npz` (v1 buggy), but cosine_gap / logit_lens / layer_profile use `hidden_states_v2_fixed.npz`. Cross-pipeline comparison invalid.
3. **Robustness suite on v1 NPZ**: §5 cites "5/5 robustness pass" but `stage4_robustness.py:25` reads v1 hidden_states.npz; v2 robustness not yet validated.

---

## Phase 2: macro methodology audit (in flight via codex)

Status: codex fire `docs/checkpoints/codex_prompts/macro_methodology_audit_2026-05-13.md` 2026-05-13 09:50 BST. Output → `docs/checkpoints/codex_outputs/macro_methodology_audit_FINAL_2026-05-13.md`. Monitor armed (30min timeout).

8 audit dimensions:
1. Identification protocol (Lin & Liu 5-step disclosure)
2. Causal claim framework consistency
3. Theoretical framework Zoom 1-4 logical chain
4. Cross-pipeline coherence (this echoes Phase 1 flag)
5. Statistical framework (MC correction, held-out boundaries)
6. Falsifiability + counter-claims
7. Generalization argument
8. Lit anchor strength (load-bearing vs decorative)

(Phase 2 findings appended below when codex output lands.)

---

## Fix priority for next data extraction

**Pre-extraction P0 (must fix BEFORE matched-N v2 / cross-family / reverse-tier re-runs)**:

1. **P0-2** Fail-closed extraction grid → 1h (BLOCKS extraction re-runs)
2. **P0-3** Add phantom_text + phantom_som to format_variation baselines → 45-90min (BLOCKS H1 re-runs)
3. **P0-5** Update Method 4.4 default NPZ to v2_fixed → 20min (BLOCKS Method 4.4 re-runs)
4. **P0-6** Update robustness suite default NPZ to v2 → 30min (BLOCKS §5 robustness validation)
5. **P0-4** Held-out task split in Method 4.4 → 2-3h (REQUIRED for paper-grade steering)

Total **~5-7h pre-extraction work**. After these 5 fixes, next data extraction one-shot produces paper-grade Method 4.2 + 4.4 + format variation results.

**Post-extraction (can fix after new data lands)**:
- P0-1 logit lens per-task refactor (already partial via `stage4_logit_lens_per_task.py` defuse)
- P0-7 axis-2 layer profile per-task refactor
- P0-8 fp32 throughout for logit lens
- All P1 + P2

---

## Implementation order recommendation

| Order | Item | Time | Reason |
|---|---|---|---|
| 1 | P0-5 (Method 4.4 NPZ default to v2) | 20min | Smallest LOC, biggest pipeline coherence win |
| 2 | P0-6 (robustness NPZ default to v2) | 30min | Validates v2 retraction claims still hold |
| 3 | P0-2 (fail-closed extraction grid) | 1h | Blocks next extraction; design fix |
| 4 | P0-3 (add phantom_text + som baselines) | 45-90min | Blocks H1 re-run |
| 5 | P0-4 (held-out task split in Method 4.4) | 2-3h | Most LOC but design-critical |

Total: 5-7h to fix all P0 BEFORE next data extraction.

---

## Audit completeness

- Top 10 scripts audited (Claude 5 + codex 5)
- 21 findings ranked
- Cross-pipeline coherence flagged
- Phase 2 macro audit fired (in flight)

Next: P0 implementation per order above, then re-fire data extraction with confidence.
