## Verdict (one sentence)
§5 现在是 **workshop / borderline mid-tier** 的机制框架：实验素材很强，但 causal estimand、claim triangulation、generalization boundary 尚未达到 NeurIPS/ICML/ACL main 的 methodology bar。

## Critical methodology gaps (P0 — framework-level errors)

- **Dimension**: Causal claim framework / §1-§5 coherence  
  **Issue**: §5 的因果对象是 50-token continuation patching，不是 §1 hero 的 task-level drop-one oracle / exploration-commit behavior。  
  **Specific quote**: “The bridge from patching displacement to behavioral SR remains open.” `section5_mechanism.md:133`; §1 says “principal hero metric is therefore the drop-one oracle” `section1_intro.md:7`.  
  **Why this kills paper-grade**: Reviewer-3 会说：你证明了 hidden state 改 token continuation，但没有证明它解释 routing value。  
  **Fix**: 明确把 §5 claim 降级为 “causal continuation mechanism”；另加一张 design bridge：patched continuations → action type / target id / trajectory outcome proxy，或声明 behavioral bridge open。  
  **Effort**: prose 2h；new experiment 2-4 days.

- **Dimension**: Cross-pipeline coherence / causal claim framework  
  **Issue**: plan 的 hero 依赖 `lm_head amplification`，但 §5 正文还说 v2 logit lens pending。  
  **Specific quote**: plan: “KL 0.05-0.09… anchor ‘geometry underestimates causal’” `plan.md:64-67`; §5: “Re-run pending on v2 NPZ” `section5_mechanism.md:22`, “needs re-running” `section5_mechanism.md:137`.  
  **Why this kills paper-grade**: 三角论证缺一个角；不能把 pending pipeline 当 load-bearing evidence。  
  **Fix**: submission version 二选一：删除 KL amplification from core claim，或把 v2 logit lens 完整转入 §5 evidence table with provenance + tests。  
  **Effort**: prose 1h；analysis/report 0.5-1 day.

- **Dimension**: Identification protocol / falsifiability  
  **Issue**: Lin & Liu 5-step 被格式化满足，但 assumptions 没有对称 stress-test。A3/A4 未证，A5 control 还弱。  
  **Specific quote**: “Reverse-tier 15 tasks pending” `plan.md:208`; “Not tested… unknown” `plan.md:209`; §5 admits Gaussian “is weak baseline” `section5_mechanism.md:20`.  
  **Why this kills paper-grade**: disclosure norm 不是列 assumption，而是给每个 causal identification threat 一个 credible falsifier。  
  **Fix**: 每个 hero claim 配 explicit counter-claim + rejection test；Exp5 必须用 task-shuffled/content-matched source control，不靠 Gaussian。  
  **Effort**: prose 2h；experiment 1-2 days.

## Medium gaps (P1 — defendable but should fix)

- **Dimension**: Theoretical framework Zoom 1-4  
  **Issue**: canonical §2 原本说 “Paper 不 self-probe Zoom 4” `paper_planning.md:358-380`，但现在 §5 已经 self-probe；theory layer 未同步。  
  **Specific quote**: “Zoom 4 标 future work, 不 self-probe” `paper_planning.md:373-380`; §5 says “Zoom-4 layer” `section5_mechanism.md:7`.  
  **Why this kills paper-grade**: theory framework 和 paper execution 相互否认。  
  **Fix**: rewrite Zoom 4 as “B1 local self-probe only; B0/B235B internal mechanism future work.”  
  **Effort**: prose 2-3h.

- **Dimension**: Statistical framework  
  **Issue**: Holm scope 不清。Layer-wise patching 有 Holm，但 AUROC 15 pairs × 37 layers、format variants、peak-layer searches 没有统一 hypothesis family。  
  **Specific quote**: “Holm-Bonferroni correction across the canonical grid” `section5_mechanism.md:60`; “all 15 mode pairs × all 37 layers” `section5_mechanism.md:17`.  
  **Why this kills paper-grade**: peak-layer / hero-table selection 看起来像 garden of forking paths。  
  **Fix**: declare primary endpoints: L11-L17 patching, held-out AUROC, image-axis magnitude rank. Everything else FDR/exploratory.  
  **Effort**: prose 2h.

- **Dimension**: Generalization argument  
  **Issue**: 2 sites + 1 model family 被写得比证据更广。  
  **Specific quote**: “cross-family Phi… larger Qwen… deferred” `section5_mechanism.md:107`; plan says “not paper-critical” `plan.md:410`.  
  **Why this kills paper-grade**: cross-site yes；cross-family no。不要暗示 training-prior universality。  
  **Fix**: claim boundary = Qwen3-VL-4B, cls+reddit; family generalization future work。  
  **Effort**: prose 1h.

## Out-of-box callout
最容易漏掉的问题：H1 trigger attribution 已从 “flat-list” 改成 post-hoc “integer-marker + markup-sigil”，但 plan 仍写 “flat-list… AXTree hierarchical is the unique format” `plan.md:115-117`，§5 又承认 “post-hoc… held-out falsifiers… not yet run” `section5_mechanism.md:95`.

## Theory framework structural risk
Zoom 1 architectural boundary 有证据/定义支撑：phantom space locks image off `paper_planning.md:99-105`。Zoom 2 behavioral axes partly supported by §1 search-loop/FP claims `section1_intro.md:11`。Zoom 3 named phenomena mostly lit-assumed, not directly identified in this model. Zoom 4 now has real B1 evidence, but it connects to §1 hero only weakly because §5 itself says behavioral SR bridge open `section5_mechanism.md:133`.

## Distance to top-tier
- Tier today: workshop / borderline mid-tier
- Top blocker: causal continuation evidence has not been linked to drop-one oracle behavior.
- Submission-today probability: 0.18

## One thing to fix tonight (1-3h)
重写 §5.1 evidence table into **primary / secondary / exploratory** claims, and remove `lm_head amplification` + H1 trigger attribution from primary causal claim unless v2 provenance and held-out falsifiers are included.

=== DONE ===