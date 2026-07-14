# §1 / Abstract 双分支预写 (队列③, 2026-07-01)

> ⚠️ **字数警告（2026-07-14 实测）**：下列旧“内联句”原样 splice 会把 250 词 abstract 推到 Branch A 278 词、Branch B 291 词；不得用于 CMT。请改用 §6 的完整替换 abstract。
>
> **用途**: verdict day 按 realized gate JSON 选一支 splice 进 `aaai27_main.md`，删另一支。
> 两分支 = 当前真实可能的两个世界: **Branch A** = **H1-PASS ∧ H3 两轴 PASS** (R-tier 由 B2 claim-tier
> gate 定 R1/R2；⚠️ 触发条件含 H3 — A 支 abstract 句断言了双轴 structural gate 通过, H1-pass 但任一
> H3 轴 fail 时 **A 支不适用**, 属未预写角落, 现写并把 H3 断言句替换为该轴 realized verdict)；
> **Branch B** = H1-FAIL + H3 两轴 PASS (= prereg Amendment 02 **Route C'-S**: lower-claim
> phantom-space-structure paper；不是 R5→R3 rescue，是 pre-registered reporting scope)。
> 依据: H1 interim k=3 (cls-only) θ_FE=+0.98pp [−0.05,2.00] 恰低于 +1.0pp 门槛，verdict 押 reddit
> (archive reddit = P-SoM 最强 site) → 两分支都真可能用上。
> **未预写的低概率分支**: Route C'-R (H3 fail + H10 pass, router-only paper) / Route F (双 fail,
> Track B note) — H3 interim 两轴 CI 均排除 0，这两支概率低；若真落入，verdict day 现写 (量小)。
>
> **Splice 纪律**: ① 数字槽 `«...»` 全部从 gate JSON / `phase1_full_prereg_decision.csv` 填，禁手抄
> interim；② splice 后必跑 checklist item 9 banned grep (0 hits)；③ 重数词数 (item 7)；④ H3/H10
> 各自 verdict 独立于本文件——H3 意外 fail 时本文件两支都不适用，回 Amendment 02 ladder。

---

## 槽位约定

| 槽 | 来源 (verdict day) |
|---|---|
| «THETA» | `pooled_h1_fe.theta_FE_pp` (FE 点估计; bootstrap median 只作 sheet 交叉核对) |
| «CI_LO» | `pooled_h1_bootstrap.ci95_lo_pp_bootstrap` (percentile CI = PRIMARY 报告 CI per B-1009) |
| «CI_HI» | `pooled_h1_bootstrap.ci95_hi_pp_bootstrap` |
| «P_BOOT» | `pooled_h1_bootstrap.p_one_sided_bootstrap` (primary gate 统计量; normal-Z 只进 appendix) |
| «K» | 实际入池完整 cell 数 (=6 或 k<6 透明披露 per advisor 预案(a)) |
| «AX1» / «AX1_CI_LO» / «AX1_CI_HI» | H3 axis-1 FE pooled unique-contribution及其 primary bootstrap CI |
| «AX2» / «AX2_CI_LO» / «AX2_CI_HI» | H3 axis-2 FE pooled unique-contribution及其 primary bootstrap CI |
| «UNIQ_CLS» «UNIQ_RED» | P-SoM per-site unique-pass counts (canonical [A], 非 archive 7+6) |

---

## 1. Abstract 内联句 (现稿 L45 "On landed cells each arm opens ... claim tier." 整句替换)

### Branch A (H1-PASS)

> Each arm opens a low-overlap success pool with non-empty unique coverage, and the pre-registered fixed-effects pooled drop-one gate over the «K» (site, model) cells **passes**: removing P-SoM from the portfolio costs «THETA»pp of oracle coverage (95% CI [«CI_LO», «CI_HI»], task-paired per-cell bootstrap; one-sided p = «P_BOOT» against the +1.0pp substantive threshold), and the per-axis structural gates confirm that neither a prompt trick nor a format swap alone reproduces the space.
>
> （⚠️ 末句 = H3 双轴断言, 由本支触发条件 H1∧H3 保证; 若 splice 时任一 H3 轴 fail → 本支不适用, 见文件头警告）

### Branch B (H1-FAIL + H3-PASS)

> Each arm opens a low-overlap success pool with non-empty unique coverage, but the flagship arm's pooled drop-one contribution does **not** clear the pre-registered +1.0pp substantive threshold («THETA»pp, 95% CI [«CI_LO», «CI_HI»], task-paired per-cell bootstrap); both per-axis structural gates pass («AX1»pp and «AX2»pp unique contribution, CIs excluding 0). Following our pre-registered framing ladder, we therefore report a *structured* finding at a lower claim tier: the phantom arms form a real, non-redundant region of the representation space whose per-arm irreplaceable coverage is positive but small.

（Branch B 同时把 abstract 末句 "We do not claim ... DOM-equivalent cost." 保留不动——complementarity
框架在两支下都成立；禁止在 B 支加任何 "nearly passes" / "trending" 措辞。）

---

## 2. §1 ¶3 (R-CONDITIONAL 段, 现稿 L55 "At the start ... **Its value is complementarity.**" 整段替换)

### Branch A

> At the start of this project, P-SoM looked like a broken ablation: a prompt that promises an image the model never sees, over a legend meant to index that image. The data reject that expectation. On both sites and within each backbone, the phantom arms behave as *distinct routing arms*: they solve tasks that DOM, full SoM, and Vision all miss («UNIQ_RED» Reddit + «UNIQ_CLS» Classifieds tasks uniquely solved by P-SoM alone [A]), and they fail characteristic task sets of their own. Cross-mode success-pool overlap is far from complete (same-task Jaccard 0.29–0.49 on the archive substrate [V] — *above* the independence baseline of E[J] ≈ 0.06–0.10, as shared task difficulty predicts, yet leaving a non-empty unique-pass residue per arm); the unique-pass sets survive per-task inspection rather than collapsing into a single task family. We emphasize what we do **not** claim: P-SoM is not the best single arm on every site, and we do not claim it replaces full SoM. **Its value is complementarity.**

### Branch B

> At the start of this project, P-SoM looked like a broken ablation: a prompt that promises an image the model never sees, over a legend meant to index that image. The data reject that expectation *in structure*, though not at the flagship magnitude we pre-registered. On both sites and within each backbone, the phantom arms behave as *distinct routing arms*: they solve tasks that DOM, full SoM, and Vision all miss, and they fail characteristic task sets of their own; the two structural axes of the space carry independently significant unique contribution (§5.3). What does not survive the pre-registered test is the flagship arm's *pooled magnitude*: P-SoM's drop-one contribution is positive but below the +1.0pp substantive threshold («THETA»pp, 95% CI [«CI_LO», «CI_HI»]). We report this verdict as pre-registered rather than re-framing around a post-hoc metric. We emphasize what we do **not** claim: P-SoM is not the best single arm on every site, we do not claim it replaces full SoM, and under this verdict we do not claim its unique coverage is substantively irreplaceable — the space's documented value is its *structure* and its cost profile.

---

## 3. §1 ¶4 (hero-gate 段, 现稿 L57 "Because ... motivates the preregistered oracle-vs-realized comparison in §6." 末两句替换)

### Branch A

> The pooled P-SoM contribution is the pre-registered hero gate of the paper (H1: one-sided superiority over a +1.0pp substantive threshold), and it **passes**: «THETA»pp (95% CI [«CI_LO», «CI_HI»], p = «P_BOOT»). The drop-one number is an *oracle ceiling*, not a realized router gain; it motivates the preregistered oracle-vs-realized comparison in §6.

### Branch B

> The pooled P-SoM contribution was the pre-registered hero gate of the paper (H1: one-sided superiority over a +1.0pp substantive threshold), and it **does not pass**: «THETA»pp (95% CI [«CI_LO», «CI_HI»], p = «P_BOOT»). Per our pre-registered framing ladder, the paper's claim therefore rests on the *structural* gates (§5.3) and the by-construction cost profile — not on flagship-arm superiority. The drop-one estimate remains reported in full as characterization; §6 evaluates the learned router on the same menu regardless, since routing value depends on the menu's joint structure rather than any single arm clearing a threshold.

---

## 4. Contributions 清单 (§1 L65-67) 分支差异

- **#1 (characterization)**: 两支不动 (verdict-neutral 措辞已就位, gates 句自带 <H3-VERDICT> 槽照常填)。
- **#2 (behavioural observation)**: 两支不动。
- **#3 (router)**: 不随 H1 分支; 随 **H10** 分支 (aaai27_main §6 已双分支预写, B-1753 解耦)。
- **Branch B 额外一处**: §1 L61 "asks whether the screenshot-free remainder constitutes independent
  routing structure" — B 支后接一句: "our answer is *structurally yes, at sub-threshold flagship
  magnitude* — a characterization we argue is exactly what a routing-menu designer needs to know."

## 5. 词数账

- Branch A 旧 splice 实测：abstract **+28** 词；§1 **+21** 词。
- Branch B 旧 splice 实测：abstract **+41** 词；§1 三处基础替换 **+85** 词，另加规定的 extra sentence **+23** 词。

## 6. ≤250 词完整替换 Abstract（verdict day 直接整段替换）

以下计数仅计 abstract 正文，不含 `# Abstract` 标题；计数时把每个命名槽替换为最长合理的单-token 数字（如 `−100.00`、`+100.00`、`<0.0001`、`224`），因此不会低估数字落地后的 `wc -w`。

### Branch A 完整 abstract（H1-PASS ∧ H3 双轴 PASS）

<!-- ABSTRACT_A_BEGIN -->
Multimodal web agents built on Set-of-Marks (SoM) receive a bundled observation: a marked screenshot, its textual legend, and an image-grounding prompt. We study the boundary that omits the annotated screenshot while holding the model, action space, evaluator, and task-supplied reference images fixed. Crossing hierarchical accessibility-tree versus flattened `[SOM_MARKS]` text with DOM-style versus SoM-style prompts yields three screenshot-free arms—P-text, P-prompt, and P-SoM—which we call the **phantom routing space**. Across VisualWebArena Classifieds and Reddit and three backbones, we compare this menu with DOM, full SoM, and Vision. The pre-registered fixed-effects pooled drop-one gate passes over «K» complete (site, model) cells: removing P-SoM costs «THETA»pp of oracle coverage (95% CI [«CI_LO», «CI_HI»], one-sided p=«P_BOOT» against +1.0pp). Both structural axes also pass: «AX1»pp (95% CI [«AX1_CI_LO», «AX1_CI_HI»]) and «AX2»pp (95% CI [«AX2_CI_LO», «AX2_CI_HI»]); P-SoM uniquely solves «UNIQ_CLS» Classifieds and «UNIQ_RED» Reddit task IDs under the registered site-level union estimand. By construction, the arms retain DOM's text pipeline while skipping screenshot encoding; a pre-registered cost-ratio check tests that property. A per-cell learned router selects among the six representations and is evaluated under a Pareto deployment gate (<H10-VERDICT>). We do not claim P-SoM replaces full SoM: its value is complementary, irreplaceable routing coverage.
<!-- ABSTRACT_A_END -->

实测正文词数（最长合理数字替换）：**197**。

### Branch B 完整 abstract（H1-FAIL ∧ H3 双轴 PASS）

<!-- ABSTRACT_B_BEGIN -->
Multimodal web agents built on Set-of-Marks (SoM) receive a bundled observation: a marked screenshot, its textual legend, and an image-grounding prompt. We study the boundary that omits the annotated screenshot while holding the model, action space, evaluator, and task-supplied reference images fixed. Crossing hierarchical accessibility-tree versus flattened `[SOM_MARKS]` text with DOM-style versus SoM-style prompts yields three screenshot-free arms—P-text, P-prompt, and P-SoM—which we call the **phantom routing space**. Across VisualWebArena Classifieds and Reddit and three backbones, we compare this menu with DOM, full SoM, and Vision. P-SoM's pooled drop-one contribution does not clear the pre-registered +1.0pp substantive threshold over «K» complete cells («THETA»pp, 95% CI [«CI_LO», «CI_HI»], one-sided p=«P_BOOT»). Both structural axes pass: «AX1»pp (95% CI [«AX1_CI_LO», «AX1_CI_HI»]) and «AX2»pp (95% CI [«AX2_CI_LO», «AX2_CI_HI»]); P-SoM uniquely solves «UNIQ_CLS» Classifieds and «UNIQ_RED» Reddit task IDs under the registered site-level union estimand. Following the pre-registered framing ladder, we therefore claim a non-redundant representation region with small per-arm irreplaceable coverage, not flagship-arm superiority. By construction, the arms retain DOM's text pipeline while skipping screenshot encoding; a pre-registered cost-ratio check tests that property. A per-cell learned router is evaluated under a Pareto deployment gate (<H10-VERDICT>). We do not claim P-SoM replaces full SoM: complementarity remains the primary finding.
<!-- ABSTRACT_B_END -->

实测正文词数（最长合理数字替换）：**203**。

## 7. 与 runbook (队列④) 的接口

verdict day 流程: 读 gate JSON → 本文件选支 → splice → 填 «槽» → 删 (R-CONDITIONAL) 标记与另一支
→ banned grep → 词数 → /stress。本文件是 runbook 第 2-4 步的输入。
