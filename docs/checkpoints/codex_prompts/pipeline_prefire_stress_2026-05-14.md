# Pipeline pre-fire /stress (Mode B codex scope)

You are an implementer-reviewer who has built activation patching, mean-difference steering, logit lens, HDMI reliability scoring, PCA cosine gap, and per-task paired NPZ analysis pipelines. You debug your own pipeline 50+ times. You catch the bugs YOU would catch in YOUR code.

## Context (read FIRST, do not re-explore)

The user's mechanism pipeline (paper §5 of P79 / Cost-Aware-Routing) has just landed Commits A-E (`9f3f516..38449be`) which fixed 27 audit findings. Method 4.4 v2 split sweep landed on Myriad 366792 with held-out 0.12 vs in-sample 0.29 — A5 counter-claim succeeded, §5.3 prose downgraded.

User now wants a **pre-fire pipeline audit** before the NEXT data extraction. Two extractors are candidates for next fire: cross-family P2/P3 (Phi-3.5-Vision + Qwen2-VL-7B). Claude is auditing those.

Your assigned scope (DIFFERENT from Claude — complement, don't duplicate):
1. `scripts/mechanistic/run_stage4_multimode_extract.py` — Commit C added P0-2 fail-closed grid here. Find what Commit C MISSED.
2. `scripts/mechanistic/run_stage4_format_variation_extract.py` — Commit C added P0-2 + P0-3 (phantom_text, phantom_som BASELINES) here. Find what's still wrong.
3. `scripts/mechanistic/run_stage2b_continuation_pilot.py` — the patching pipeline that produced §5.4 hero numbers. Cross-pipeline coherence with the new Stage 4 v2 NPZ schema?
4. (Brief skim only) `p79/mechanistic/activation_patching.py` — Method 4.4 substrate. Off-by-one between patcher.layers[L] and H[:, L+1, :] documented in plan.md §1.4.

DO NOT re-read cross-family scripts (`run_stage4_h1_phi35.py`, `run_stage4_h1_qwen2vl.py`) — Claude has those.

## What to look for (principled, code-level, OOB)

1. **Commit C said it added fail-closed grid check + layer-index assertions. Find what's STILL silent**:
   - Does the grid check actually fire under realistic partial-failure scenarios (one task all-modes fail, vs one mode all-tasks fail)?
   - Does the layer-index assertion catch the right error mode?
   - Provenance JSON: does it record the formatter HASH so v1/v2/v3 are mechanically separable, or just the version string?

2. **Coherence with v2 NPZ schema**:
   - `_paired_npz_helpers.py` expects `hidden_states`, `mode_labels_str`, `task_ids`, `step_indices`. Do all 3 of your assigned scripts produce that exact schema?
   - Layer-index convention: extract stores `H[:, 0, :] = embedding`, `H[:, L+1, :] = block L output`. Stage 2/3 patching uses `patcher.layers[L]` which corresponds to `H[:, L+1, :]` — but Stage 2 patching scripts likely use `H[:, L, :]` (block L-1 output). Where does this off-by-one bite cross-pipeline?

3. **Bug 1 tier filter, Bug 2 SOM_MARKS regex, Bug 5 model revision propagation**:
   - All 3 were applied to `run_stage4_multimode_extract.py` (per `qsub_stage4_multimode_extract_cls_v2.sh` comment). Did `run_stage4_format_variation_extract.py` get the same fixes? Did `run_stage2b_continuation_pilot.py` get them?

4. **Method 4.4 v2 split (Commit D) introduced concepts that may NOT propagate**:
   - The `--also-report-in-sample` flag. Other downstream analyses (e.g., `reaggregate_method44_v2_hmean.py` — Claude's viz audit touched this) — do they read `per_task_eval` correctly, OR fall back to `results` key (which is now an ALIAS for eval, but used to be all-tasks union)?
   - Direction fit on train-mask: is it preserved through any aggregate JSON the script writes, or does the downstream see "all 24 tasks" inflated direction?

5. **Out-of-box thinking**:
   - The `run_stage2b_continuation_pilot.py` produces `patching_continuation_results.json` consumed by paper §5.4 prose. Did the v2 NPZ migration break ANY downstream consumer? E.g., if Stage 2 uses v1 NPZ for direction but Stage 4 reports v2 numbers, the §5.4 table mixes vintages.
   - Provenance JSON cross-pipeline: Stage 4 has `provenance.json`. Stage 2 has anything? If Stage 2 used a DIFFERENT model revision than the pinned Stage 4 SHA, the §5.4 "all from same Qwen3-VL-4B" claim is misleading.

## Output format (≤ 600 words findings prose, then inline fixes if confident)

For each finding:
```
### Finding N — <title> [P0|P1|P2]
**Claim** — what prose/protocol expects, file:line
**Code reality** — what actually happens, file:line + function
**Attack** — principled error in 1-3 sentences
**Defuse** — specific fix
**Effort** — minutes / hours
**Confidence** — high / med / low
```

Then list inline fixes you applied (if high confidence + low effort + non-paper-altering).

## Constraints

- ≥ 3 findings, ≥ 1 OOB
- Apply fixes inline ONLY IF high-confidence + low-effort + you're sure they don't conflict with Claude's findings in cross-family scripts (which you haven't read)
- For data-altering fixes (e.g., re-extract), document only — do NOT patch (user will decide)
- Voice: implementer, hostile-but-principled, no hedging
