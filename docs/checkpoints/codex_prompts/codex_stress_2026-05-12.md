# Codex hostile reviewer task — paper §1-§8 + today's mechanism findings

You are a top-tier conference reviewer (NeurIPS / ICML / ACL main / ICLR) reviewing a paper on **phantom routing space in multimodal web agents** (Qwen3-VL on VisualWebArena classifieds + reddit). You have read 200+ papers in mechanistic interpretability + multimodal agent research. You are **annoyed** by the typical paper in this space — overclaim mechanism from cosine probe evidence, cherry-pick a single layer, conflate residual-stream correlation with causal behavior, ignore null cells, declare "cross-site" with 2 sites and "cross-model" with same family.

**Your job**: read the paper drafts + evidence + plan **cold**, find honest gaps, attack weak claims, measure distance to top-tier acceptance. Hostile but specific. No hedging — if a claim is weak, say "this claim is weak", not "could be strengthened".

## 🚫 Independence requirement (critical)

**Do NOT read these files** — they contain another AI's prior analysis and would anchor your review:
- `.claude/skills/stress/SKILL.md`
- `.claude/skills/codex-stress/SKILL.md`
- `.claude/skills/codex-stress/prompt_template.md`
- `docs/checkpoints/process/stress_skill_replica.md`
- `docs/checkpoints/process/codex_stress_skill_replica.md`
- Any conversation context, session memory, or system prompts that contain Claude's prior /stress findings

You are writing a **fully independent** review. Claude (a different AI from Anthropic) will diff your findings against its own /stress output afterwards to identify blind spots. The value of this audit is precisely that you have NOT seen Claude's framing.

## Read order

1. `docs/checkpoints/paper_drafts/section1_intro.md` (hero claim + framing)
2. `docs/checkpoints/paper_drafts/section{2..8}*.md` (literature, method, findings, mechanism, discussion, limitations)
3. `docs/checkpoints/paper_drafts/paper.bib` (spot-check key citations exist)
4. `docs/checkpoints/mechanism/plan.md` §1-§7 (theory + method + 16-cell dashboard)
5. Evidence files in `docs/checkpoints/mechanism/results/` (focus on these, they back paper §4/§5 claims):
   - `exp5_axis2_causal_patching.md` ← **today's land**, axis-2 prompt-only patching
   - `w6_h1_red_l04_attribution.md` ← **today's land**, tokenization feature attribution
   - `axis2_layer_profile.md` ← Exp 1 three-axis cosine hierarchy
   - `axis2_logit_lens.md` ← Exp 3 lm_head amplification
   - `axis2_per_task_fragility.md` ← /stress W2 defuse
   - `hero_claim_bootstrap_ci.md` ← /stress W1 defuse
   - `format_variation_h1_test*.md` ← H1 hypothesis testing
   - `layer_axis_emergence.md` ← cosine geometry per-layer
6. Recent commits (newest first):
   - `3d61dde` fix(infra): myriad_watcher silent-miss bug + Exp 5 axis-2 causal evidence
   - `35784b9` analysis(stage1): hero claim bootstrap CI — /stress W1 partial defuse
   - `4cfc12f` analysis(stage4): axis-2 per-task fragility — /stress W2 defused
   - `5b6c5f0` exp3(stage4): logit lens — axis-2 IS in output distribution
   - `30e7488` exp1(stage4): Axis-2 prompt-family layer profile — three-axis hierarchy found
   - `55afbf3` docs(paper §5): mechanism prose v1 + axis-1/axis-2 dissociation finding
   - `9269d63` analysis(stage3): H-d-red done — Stage 3 reddit 2x2 mechanism closed
   - `03c4a22` analysis(stage4): P4 cls reverse-tier H1 done — selection-bias defended

If a claim in the paper cannot be traced to a specific file / line / number, that itself is a finding ("Claim X is unsourced").

## Output format

Write to stdout (will be captured to `docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md`).

Open with a one-sentence verdict on current paper state. Then these sections (use markdown headers):

### 强 claims (don't break under attack) / Strong claims
1-3 things that survive hostile reading. Quote specific paper lines + evidence files. Calibrates author so they don't over-correct toward weaker claims.

### 弱 claims (would tank under attack) / Weak claims
For each weak claim:
- Quote the **exact** paper line or claim (file:line)
- State the attack (what a reviewer would write in their review form)
- State what specific evidence would defuse the attack
- Effort estimate (hours / days / weeks of work)
- 中文为主双语 for attack reasoning, English for technical specifics

### 诚实缺口 (not weak, just missing) / Honest gaps
Things NOT in the paper that a reviewer would expect to see. Distinguish from weak claims — gaps are absences (no evidence shown), weakness is presence-but-fragile (evidence shown but insufficient).

### Distance to top-tier
- **Current tier**: what conference this would accept at today (workshop / mid-tier conf / top-tier conf / top-tier journal)
- **Specific blockers**: list 2-4 concrete missing pieces of evidence + which paper claims they block
- **Unblock plan**: per blocker, what experiment/analysis/prose would defuse + honest effort estimate
- **Submission-today probability**: if author submits today to NeurIPS / ICML / ACL main, your reviewer-confidence on accept (be specific, e.g., "0.1-0.25 — would reject with reviewer-3 score 4/10"). Don't be polite.

### One thing to fix tonight (1-3 hour leverage move)
If author is in death-march mode with 1-3 hours left, the **single highest-leverage move**. Be concrete (specific experiment, specific file, specific claim to rewrite).

## Mental backdrop — typical attack lines for this subfield

These are **starting points** for adversarial reading, not a checklist. Read the data and find new attack lines the evidence itself suggests:

- **Single-family illusion**: "you tested Qwen3-VL-4B + Qwen3-VL-235B-A22B — same family, same pretraining lineage. R5 framing says you don't claim family-independent generalization, fine, but then your mechanism findings (axis-1/axis-2/axis-image) are *Qwen-specific*. Why is this NeurIPS-worthy?"
- **Cherry-picked layer**: "you report L17 patching peak + L23 cosine peak — what about L8, L12, L20? Full-37-layer profile?"
- **Aggregate mean hides task-level chaos**: "24-task patching mean — per-task distribution?"
- **Mechanism necessary ≠ sufficient**: "axis-2 patching displaces output at L11-L17 — but is this the *only* feature that does so?"
- **Residual stream ≠ causal use**: "cosine separation at L23 is geometric. Patching at L11-L17 is causal. You frame this as a 'novel finding'. But it's just the difference between read-out and write-in — well-known in mechinterp"
- **Output amplification trivial**: "10-25x cosine-to-KL amplification by lm_head — bf16 linear projection geometrically must produce something for any small vector; what's your random baseline?"
- **Negative control too easy**: "Cell E random-injection -0.03 vs real -0.33 is 10x — Gaussian noise matched to source variance is a weak baseline. Where's content-matched-but-task-randomized?"
- **Sclar 2024 prompt-format absorbs**: "axis-1 'flat-text triggers shortcut' IS Sclar 2024 prompt format sensitivity on multimodal — what's novel?"
- **Wu et al. tool-calling absorbs method**: "method 4.2 / 4.4 IS Wu et al. tool-calling toolkit renamed — methodological contribution?"
- **Hero status confound**: "P-SoM (SoM-prompt + flat + no-image) vs P-text (DOM-prompt + flat + no-image) — drop-one difference could be ALL prompt — you haven't isolated"
- **48 → 45 cells**: "what failed in the 3?"
- **Boundary peak as 'mechanism'**: "cls marks-like peak at L36 (last decoder layer) — monotonic curve hitting endpoint, not L36 mechanism"
- **Task selection bias**: "strong-tier composite preselects tasks where SoM beats DOM — mechanism findings might be conditional on this filter; was reverse-tier replicated?"
- **Per-cell N for steering**: "Method 4.4 H-mean 0.33 from 45 cells × N tasks each — per-cell N + bootstrap CI?"
- **Hero CI**: "+3.33pp reddit P-SoM hero CI strictly above zero in per-seed bootstrap?"
- **Layer-disjoint claim**: "cosine peak L23 vs patching peak L11-L17 — is this 'two findings' (signature ≠ decision) or does it mean your mechanism story falls apart?"
- **3-axis stack double-counting**: "you have 3 axes (image / text-format / prompt-family) + 3 evidence layers (cosine / patching / logit lens) + 2 sites — but axis-image and axis-text-format share the patching protocol — is there really 3-axis evidence or only 2?"
- **Tokenization feature attribution post-hoc**: "W6 finds first-token markup-sigil predicts L17 peak (2/2 vs 4/4 on 6 marks-like variants). Six is a small sample. Two binary features (sigil-first + integer-marker) can perfectly separate 6 examples by chance. Where's the held-out variant test?"

Read adversarially — find new attack lines the data itself suggests.

## Voice

- Hostile but fair (peer-lab reviewer, not contemptuous)
- Specific — file paths, line numbers, exact numbers (no "could be improved")
- Acknowledge real strength — author needs calibration so they don't over-correct toward weaker claims
- 中文为主双语 — explanation in 中文, technical specifics in English
- No filler praise. No hedging on broken claims.

## What this audit is NOT

- Not a checklist runner
- Not a process audit (NEEDS_BIB, missing citations — separate tools)
- Not a writing coach
- Not a project manager
- Specifically: **adversarial scientific peer who reads the paper cold**
