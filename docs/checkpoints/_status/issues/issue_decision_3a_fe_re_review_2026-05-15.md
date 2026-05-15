---
type: issue
category: decision
status: open
priority: P0
action: advisor confirm or rollback Decision 3A (FE estimand → RE+Knapp-Hartung at k=6) BEFORE OSF lock email
created: 2026-05-15
updated: 2026-05-15
gates: OSF lock email + paper §1 generalization claim defensibility
---

# Decision 3A FE→RE rollback escalation (post-Batch-1-5 audit)

**B-130 escalation, NOT auto-fix.** Decision 3A was witness-locked by you-advisor 2026-05-14;
single-AI Claude /stress cannot unilaterally roll it back. Gemini Mode C audit 2026-05-15 (P0-2)
independently challenged the FE choice from a *generalization-claim-coupling* angle that prior
Claude+codex 7-day audit history missed.

## What got locked 2026-05-14 (Decision 3A)

Per `preregistration.md §2.4` + §4 row "Pooling estimator + heterogeneity pre-spec" +
Appendix A 2026-05-14 entry:

- **Estimand**: fixed-effects inverse-variance pooled average θ_FE over the 4 (now 6 per B2)
  *planned* (site, model) cells. The cells are the design, not a population sample → no
  between-cell variance τ² in the estimand → no DerSimonian-Laird, no REML.
- **Rationale (Claude v6 + codex cross-think)**: avoids DL τ² downward bias + RE Wald
  anti-conservatism at k<10 (Veroniki et al. 2016 / IntHout et al. 2014). FE is sound at
  any k under CLT on per-cell θ_i.
- **Witness**: paper_planning §19 + 实验笔记 §142 — advisor email pending lock.

## Gemini Mode C P0-2 attack 2026-05-15 (cold cross-AI, prose-anchored)

> Quote: `osf_lock_manifest.md` "Fixed-effects inverse-variance pooled average... (decision '3A' 2026-05-14 — NOT DerSimonian-Laird; the cells are the design not a population, so no τ²)."
>
> Attack: 这是一个致命的统计学自杀 (statistical trap). 放弃 Random Effects (REML+HK/DL) 转而使用 Fixed-Effects (FE) 意味着你假设所有 6 个 cells 存在**唯一真实的 effect size**. 这在统计学上直接剥夺了论文 generalizability 的合法性. Reviewer 会攻击: FE 只能证明 "在这 3 个特定模型和 2 个特定网站上有效", 无法泛化到 "Web Agents" 这一 broader population.
>
> Defuse: 立即推翻 Decision 3A,回滚到 Random Effects meta-analysis. 即便 k=6 较小,RE 配合 Knapp-Hartung 调整也远比强制假设同质性的 FE 更符合顶级会议的统计严谨性标准.
>
> Severity: P0.

## Why this is real (not just a model-disagreement)

The FE-vs-RE choice is **estimand-definition**, not just estimator-tuning:

- **FE estimand**: "average drop-one over EXACTLY these 6 planned cells (cls/red × B0/B1/B2)". Inference scope = these 6 cells, period.
- **RE estimand**: "average drop-one over a HYPOTHETICAL population of (site, model) cells, with the 6 observed cells as a sample". Inference scope = "Web Agents on VWA-style tasks" broadly.

Paper §1 hero hook currently says:
> "We characterize the **phantom routing space**: configurations on the 'skip annotated image' boundary..."

The implicit-generalization framing ("phantom routing space exists for Qwen+Gemma agents on VWA-style tasks") is RE-shaped, not FE-shaped. If the estimator says "we only learned about these 6 cells", the hook overpromises.

Veroniki/IntHout fragility is real at k<10, but **it does not vanish by switching to FE**:
- DL τ² downward-biased at k=4 → eases (but not gone) at k=6
- RE Wald anti-conservative at k=4 → eases (but not gone) at k=6
- Knapp-Hartung adjustment uses t-distribution at k-1 df (5 at k=6), restoring conservativeness

Gemini's recommendation (RE+Knapp-Hartung at k=6) is **statistically more conservative** than the 2026-05-14 FE choice while preserving generalization-claim language.

## Tradeoff matrix

| Estimator | Pros | Cons | Reviewer attack |
|---|---|---|---|
| **FE (Decision 3A, 2026-05-14)** | Sound at any k; no τ² estimation needed; clean | Inference limited to 6 cells; generalization claim 受限 | "你只测了 3 个模型 × 2 站点,凭什么 claim 'phantom routing space is generalizable property'?" |
| **RE+Knapp-Hartung (Gemini P0-2 recommend)** | Restores generalization claim; HK adjustment fixes anti-conservativeness at k<10 | Still has DL τ² downward bias at k=6; FE vs RE point estimate may diverge if I² > 25% | "你用 RE 在 k=6 上 pool — IntHout 2014 / Veroniki 2016 都说 k<10 时 RE 不稳" |
| **DL random-effects** (current script implementation) | What aggregate_phantom_meta.py actually computes today; matches archive | Most fragile at k<10; both biases active | "你 prereg 说 FE,代码跑 DL — code↔prose mismatch" |

The current state has a **third bug**: prereg says FE (Decision 3A), but `aggregate_phantom_meta.py` + `preregistration_decision_test.py` actually compute DL — code↔prose drift.

## Recommended advisor question

**Email/sync agenda for advisor**:

> Post-Batch-1-5 cross-AI audit surfaced a Decision 3A challenge from an angle our 2026-05-14 lock didn't cover:
>
> Gemini Mode C (independent prose audit): "FE estimand 把 paper §1 generalization claim 阉割了 — RE+Knapp-Hartung at k=6 才能保留 'phantom routing space is generalizable' 这种顶会 framing"
>
> Three options:
>   (a) Keep Decision 3A FE (我们 2026-05-14 选的) + paper §1 hook 软化 wording 到 "characterizes phantom routing space on these 6 cells"
>   (b) Roll back to RE+Knapp-Hartung at k=6 (Gemini recommendation) — restore generalization-claim framing
>   (c) Report both as primary + sensitivity — FE in main text, RE in appendix or vice versa
>
> Need advisor decision before OSF DOI lock email goes out.

## Affected gates

- 🔴 OSF lock email — locks estimand choice into DOI artifact; cannot send until decision
- 🔴 `aggregate_phantom_meta.py` — currently runs DL (matches no prereg version cleanly)
- 🔴 `scripts/analysis/preregistration_decision_test.py` — same DL-vs-FE-vs-RE inconsistency
- 🟠 `osf_lock_manifest.md §2.2` — was updated 2026-05-15 Batch 4 to FE wording; will need rewrite if (b) or (c)
- 🟠 Paper §1 hero claim language — generalization-claim coupling

## Cross-references

- `docs/checkpoints/pre_run/preregistration.md` §2 estimand + §2.4 power + §4 pooling row + Appendix A 2026-05-14 Decision 3A entry
- `docs/checkpoints/gemini_outputs/post_batch5_2026-05-15_201813.md` Mode C P0-2 (full attack)
- `docs/checkpoints/codex_outputs/post_batch5_FINAL_2026-05-15_201813.md` Mode B (10 findings, none on FE/RE — code-anchored audit didn't reach methodology layer)
- `docs/checkpoints/实验笔记.md` §142 (advisor 2026-05-14 sync + Decision 3A lock chronicle)
- `docs/checkpoints/实验笔记.md` §143 (this audit + cross-AI Mode C P0-2 chronicle, to be appended)

## Status

⏳ **Open** — pending advisor sync / email lock decision (a)/(b)/(c).

**Claude will NOT modify aggregate_phantom_meta.py / preregistration_decision_test.py estimator
implementation unilaterally** — current DL implementation flagged in docstrings as
"FE per Decision 3A vs DL current impl vs RE Gemini-recommended → all 3 pending advisor lock".

After advisor decision lands → update preregistration §2.4 + §4 + Appendix A + osf_lock_manifest
§2.2 + the 2 affected analysis scripts + this issue → status: decided/closed.
