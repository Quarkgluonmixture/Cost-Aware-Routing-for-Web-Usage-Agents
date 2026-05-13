# Codex macro methodology audit (Phase 2)

You are a top-tier methodology reviewer (NeurIPS / ICML / ACL main / ICLR) who has **personally implemented** activation patching / mean-diff steering / logit lens / HDMI. You debug your own grad student's pipeline regularly.

## Context

P79 paper-1 (毕设) §5 mechanism. Phase 1 (code-level) just landed 21 findings (8 P0 / 8 P1 / 5 P2). Phase 1 caught script bugs. **Phase 2 is DESIGN / METHODOLOGY review** — at the framework level, in a zero-code-bug assumption.

Audit 8 macro dimensions:

1. **Identification protocol** (Lin & Liu 2026 5-step disclosure norm) — does paper §5 actually satisfy? Stress-test each step.
2. **Causal claim framework** — current claim (L11-L17 patching window + cosine-causal disjoint hero + lm_head amplification + Stage 2/3 additivity). Is the claim internally consistent across pieces?
3. **Theoretical framework Zoom 1-4** — logical chain from architectural → behavioral axes → named phenomena → model-internal. Where does chain break? Which links are evidence-backed vs assumed?
4. **Cross-pipeline coherence** — Stage 2/3 patching + Stage 4 NPZ + Method 4.4 steering + Format variation. Layer-index / sample size / NPZ provenance / model revision consistency across pipelines.
5. **Statistical framework** — family of hypotheses (primary vs secondary), multiple-comparison correction strategy (Holm/Bonferroni/FDR — scope?), held-out vs in-sample boundary, paired vs unpaired test choices.
6. **Falsifiability + counter-claims** — for each hero claim (drop-one oracle / cosine-causal disjoint / mid-layer fusion locus / image-axis peak dichotomy), is there an explicit counter-claim? How would the counter-claim be rejected by evidence?
7. **Generalization argument** — paper claims cross-site (cls + reddit) + cross-family extension. What's the principled argument that 2 sites + 1 family extrapolates? Selection-bias defense (reverse-tier) properly framed?
8. **Lit anchor strength** — 5 anchors (Wu et al. 2026 tool calling / Ma&Rui 2026 probe-vs-causal / HDMI Khorasani 2026 / Lin&Liu 2026 disclosure / Peale 2026 routing). For each: load-bearing or decorative? What specific load does each anchor carry?

## Read scope

Read these (cold, NOT prior reviews):

1. `docs/checkpoints/mechanism/plan.md` (mechanism workspace, full)
2. `docs/checkpoints/paper_drafts/section5_mechanism.md` (paper §5 prose)
3. `docs/checkpoints/paper_drafts/section1_intro.md` (hero claim setup)
4. `docs/checkpoints/paper_planning.md` §2 only (theory framework canonical, optional)

Hard limit: 5 files. STOP and write when you hit 5.

## 🚫 Do NOT read

- Any `docs/checkpoints/codex_outputs/*` (Phase 1 codex outputs, prior reviews)
- Any `.claude/skills/*` (skill replicas / Claude analysis)
- Any `docs/checkpoints/process/stress_skill_replica.md`

## Write-FIRST explore-SECOND

Step 1 (~300 words): write preliminary methodology audit based on FIRST 2 files (plan.md + section5_mechanism.md). Flag 3-4 worst design-level findings immediately.
Step 2: expand findings with 1-2 supplementary file reads.
Step 3: ranked design fixes.

If you find yourself reading more than 4 files without writing, STOP and write.

## Output format (≤ 700 words, atomic to -o file)

```markdown
## Verdict (one sentence)
Macro methodology grade of paper §5 framework — what tier reviewer would assess this at.

## Critical methodology gaps (P0 — framework-level errors)
2-4 findings. For each:
- **Dimension**: <which of the 8 audit dimensions>
- **Issue**: <design / framework error>
- **Specific quote**: <paper §5 or plan.md line citation>
- **Why this kills paper-grade**: <reviewer-3 perspective>
- **Fix**: <design-level change, not code change>
- **Effort**: <prose hours / new experiment days>

## Medium gaps (P1 — defendable but should fix)
2-4 findings, same format.

## Out-of-box callout
The single design issue that 95% of reviewers would miss first-read. Quote file:line.

## Theory framework structural risk
Examine Zoom 1-4 logical chain. Where does it break? Which links are evidence-backed (cite line) vs theoretically-assumed (no evidence)? Specifically: how does §5 connect to §1 hero claim?

## Distance to top-tier
- Tier today: <workshop / mid-tier / top-tier / journal>
- Top blocker: <one sentence>
- Submission-today probability: <specific number 0.0-1.0>

## One thing to fix tonight (1-3h)
Specific design-level change. Not code. Not prose polish.
```

End with literal `=== DONE ===`.

## Voice

- Top-tier reviewer who has implemented these methods
- Specific quotes from paper §5 / plan.md, with line numbers
- 中文为主双语 — design critique + framework 用 中文, exact quotes + bibkeys + variable names 保留 English
- No hedging on broken framework
- Acknowledge what survives so author doesn't over-correct

## What this audit is NOT

- ❌ Not code review (Phase 1 covered)
- ❌ Not fact-checking
- ❌ Not prose editing
- ❌ Not "consider adding X" — must be principled critique

The goal: surface DESIGN errors that no code fix would resolve.
