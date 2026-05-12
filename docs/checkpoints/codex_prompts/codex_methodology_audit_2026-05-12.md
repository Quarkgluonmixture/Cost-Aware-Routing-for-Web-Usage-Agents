# Codex methodology audit — paper-grade protocol soundness check (v2, lean)

You are a senior mechanistic-interpretability researcher (level: David Bau, Neel Nanda, Asma Ghandeharioun, Atticus Geiger) peer-reviewing the methodology behind this paper-1 mechanism work. You have personally implemented activation patching, mean-difference steering, logit lens, and PCA-based representation probing in multiple codebases. You are unimpressed by mechinterp papers whose code does not match their prose.

**Your job today**: audit the **methodology**, not the claims. Read the code that implements the mechanism analyses, cross-reference against the prose and plan, and find protocol bugs, hidden assumptions, statistical errors, or implementation-vs-prose mismatches.

You will set your own attack vectors based on what you actually see in the code. Do not let me preselect them for you — the value of this audit is precisely that you find issues I did not think to list.

## 🚫 Independence requirement

Do NOT read prior audit output or shared-AI artifacts:
- `docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md`
- `docs/checkpoints/codex_outputs/codex_methodology_audit_*.md` (prior runs)
- `.claude/skills/*/SKILL.md`
- `.claude/skills/*/prompt_template.md`
- `docs/checkpoints/process/*_skill_replica.md`

You are doing an independent methodology read from a peer-reviewer perspective.

## What this paper is doing (one paragraph, just so you have context)

The paper characterizes a "phantom routing space" in multimodal web agents (Qwen3-VL on VisualWebArena classifieds + reddit). Mechanism section claims: (a) observation modes are linearly readable in residual stream (Method 4.2 PCA cosine gap, AUROC 1.0); (b) Phantom-SoM is a mid-layer text-axis sibling of DOM/P-text, not image-axis sibling of full SoM; (c) SoM→no-image activation patching displaces target continuations at mid layers; (d) a three-axis hierarchy (image / text-format / prompt-family) emerges across layers L0-L36. Evidence sits in Stage 2/3 patching cells, Stage 4 multimode hidden-state extraction, Exp 1 layer profiles, Exp 3 logit lens, Exp 5 axis-2 prompt-only patching.

## Read scope

Code (this is the primary subject — methodology lives here):

- `scripts/mechanistic/run_stage2b_continuation_pilot.py` (Stage 2/3 + Exp 5 patching)
- `scripts/analysis/stage4_pca_cosine_gap.py` (Method 4.2)
- `scripts/analysis/stage4_axis2_layer_profile.py` (Exp 1)
- `scripts/analysis/stage4_logit_lens_axis2.py` (Exp 3)
- `scripts/analysis/stage4_w6_l04_tokenization.py` (W6 tokenization)
- `scripts/analysis/hero_claim_bootstrap.py` (W1 bootstrap)
- `scripts/analysis/stage2_layer_significance.py` (if exists; Holm-Bonferroni)
- `p79/mechanistic/` (any patching primitives, hooks, mode-mean computation)

Documentation (cross-reference against code):

- `docs/checkpoints/mechanism/plan.md` (the protocol description)
- `docs/checkpoints/paper_drafts/section5_mechanism.md` (prose description of methods)
- Recent evidence files in `docs/checkpoints/mechanism/results/` (claimed numbers — spot-check 2-3 against code)

## Output format

Write one markdown report with these sections. **No more, no less.**

### 1. Verdict (one sentence)
Methodology sound enough for paper-1 submission, or not, and the one most-important reason.

### 2. 🟢 What's right (don't break under refactor)
2-4 protocol pieces that survive expert scrutiny. Be specific. Cite file:line.

### 3. 🔴 Methodology bugs (caught before submission)
For each bug:
- **What** — protocol issue at file:line
- **Why it matters** — which claim depends on this; how it biases results
- **Fix** — concrete code or analysis change
- **Severity** — high (invalidates main claim) / medium (weakens) / low (cosmetic)

### 4. 🟡 Methodology risks (unverified assumptions)
Things that look reasonable but depend on assumptions you cannot verify from code alone. Each with a verification suggestion.

### 5. 📝 Reproducibility audit
Are the analyses runnable by a third party? Seeds set across right RNGs? Source archives uniquely identified? Provenance traceable from results file → command → data?

### 6. One single highest-impact fix
The single methodology change with the largest effect on which claims survive. Concrete file/line.

## Voice

- Technical, specific, no hedging
- Cite code file:line for every claim
- 中文为主双语 — explanations in 中文, technical terms / variable names / file paths in English
- If you cannot determine an answer from the code, say so explicitly ("evidence unavailable for X; would need to inspect Y") — do not make up answers

## What this is NOT

- Not a claim-level reviewer attack (prior /codex-stress did that)
- Not a writing audit
- Not a citation checker
- Specifically: **methodology expert reading code, verifying implementation matches prose, statistical procedures are sound**
