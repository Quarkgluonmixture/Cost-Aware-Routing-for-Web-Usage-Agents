# Codex focused audit — P79 v2 retraction (3 specific claims)

Independent hostile reviewer mode. **Do NOT** read prior codex outputs or Claude reviews. Read **only** the files listed below — don't go exploring.

## Context

P79 paper-1, Qwen3-VL-4B B1, mechanism §5. v2 NPZ migration just landed (Bug 1 tier filter + Bug 2 SOM_MARKS regex + Bug 5 model revision). Author rewrote `docs/checkpoints/mechanism/plan.md` with v2 numbers. User is **skeptical** of the new framing and wants a brutal cross-AI sanity check.

## Read these 4 files only

1. `docs/checkpoints/mechanism/plan.md` §0 + §1.2 + §1.3 (the 3 most-changed sections)
2. `docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md`
3. `scripts/analysis/stage4_pca_cosine_gap.py` (lines 40-180 — cosine_gap function + main loop)
4. `docs/checkpoints/mechanism/results/axis2_logit_lens_v2.md`

## Attack these 3 claims (don't explore beyond)

### Claim A: "Cosine-causal disjoint" hero (plan.md §1.2)

Author argues:
- cosine gap 0.5-1% (residual stream geometric)
- KL ~0.05-0.09 (logit lens output)
- patching Δoverlap 20-30% (causal behavior)
- → "geometry underestimates causal by orders of magnitude"

Attack: are these comparable? Cosine measures mean-distance, KL measures distribution divergence, Δoverlap measures behavior change after intervention. Calling this a "disjoint" — is this defensible or category error?

### Claim B: V1 magnitude → v2 magnitude collapse is "artifact correction"

`method42_v1_vs_v2_comparison.md` shows axis-1 text-format cosine collapsed -81% (0.025 → 0.005). Author calls v1 "buggy artifact" and v2 "correct".

Attack: alternative interpretation — v2's properly-included `[SOM_MARKS]` HOMOGENIZES text across `som / phantom_som / phantom_text / phantom_prompt` modes, all carrying same `[id=N] {label}` block. V1 may have **accidentally exposed real signal** by selectively dropping marks differently per mode. How would author distinguish "v2 corrects bug" from "v2 masks real signal"?

### Claim C: Logit lens "amplification 8-44×"

`axis2_logit_lens_v2.md` line 3: "Apply Qwen3-VL-4B's final_norm + lm_head to per-layer **per-mode mean** hidden states." Then computes KL between modes.

Attack: per-mode means are averaged representations that don't correspond to any actual forward pass. Reviewer-3 will say "decoded an average that doesn't exist; KL between two averaged distributions is not a meaningful 'amplification' of the underlying signal — it could be averaging artifact". Is there per-task KL computation (decode each task's actual hidden state through lm_head then average task KLs) instead?

## Deliverable

Output **only**:

```markdown
## Verdict line
[one sentence: do the 3 claims survive hostile reading?]

## Claim A: Cosine-causal disjoint
Strength: [strong | weak | broken]
Attack: [specific quote + what's wrong]
Defuse: [what evidence/reframing would save it]
Effort: [hours]

## Claim B: v1→v2 magnitude collapse
Strength: [strong | weak | broken]
Attack: ...
Defuse: ...
Effort: ...

## Claim C: Logit lens amplification
Strength: [strong | weak | broken]
Attack: ...
Defuse: ...
Effort: ...

## Single highest-leverage move tonight
[1-2 sentences]
```

Brutal but fair, ~500-800 words total. **Do NOT** explore other files. **Do NOT** re-read full plan.md (just the 3 sections cited). Output goes to stdout — the wrapper captures it.

End your response with the literal token: `=== AUDIT COMPLETE ===`
