# Codex Section 5 prose — reddit mechanism (axis × LLM-explanation × evidence wiring)

## 任务

写 paper Section 5 关于 **reddit site** 的 mechanism prose 草稿 (~2500 words)。Section 5 的整体结构是 **Site × Axis × LLM-mechanism**，每个 site 是一节，reddit 是其中一节。

不写 classifieds 和 shopping (数据 not ready / sparse)。专注 reddit 部分能写到 paper-quality。

仓库：`/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents`，必须用 `.venv/bin/python3`。**不要 commit**。

## Section 5 总体定位（写之前理解）

Section 4 (Empirical Findings) 给的是 **evidence catalog** —— 4 dimensions (Outcome / Macro / Micro / Efficiency) 每条 evidence 用 sub-codes (0a-0g, 1a-1c, 2a-2e, 3a-3d) 标记。Section 5 是**解释层**，回答 "为什么这些 evidence 在 axis swap 下产生这种 shift"。

**Section 5 的 narrative 不重复 Section 4 数字**——cite 数字时用 `(0c +3.81pp drop-one)` 这种引用形式，主笔墨花在 LLM-mechanism explanation 上。

## Reddit 子节内部结构（target output）

```
## 5.X Reddit: text-dominated substrate, axis 1 primary

### 5.X.1 Substrate
  - Reddit (Postmill clone) 的信息结构: forum hierarchy → posts → comments
  - Navigation affordance: sidebar f/<forum> link + search + post/comment link
  - Image 角色: content clue (post attachments, ref images) NOT 导航 affordance
  - Intrinsic search: false — search-loop 是失败 basin
  - URL routing: path-based /f/<forum>/<post>/<comment>

### 5.X.2 Axis 1 (text payload structure) — PRIMARY
  - LLM mechanism: AXTree hierarchy 让 sidebar forum link 埋在树深处 (in-context attention 衰减) →
    agent 优先 search box → search-loop 死循环。
    [SOM_MARKS] flat indexed list 让 forum link 在 attention pattern 中显著 (flat structure 减 attention dilution) →
    agent 直接 click forum → trajectory 短 + decision 准 + SR up
  - Cross-dimension evidence:
    * Outcome 0c: +P-text adds +3.81pp drop-one oracle lift over 3-mode
    * Macro 1c: search-loop% DOM 51.9% → P-SoM 35.7% → SoM 31.4% (N=210 全数据, FRESH 04-29)
    * Micro 2a: URL Jaccard 0.573 axis-1-alone (path-only signature)
    * Micro 2b: target-page hit rate +3.47pp axis 1
    * Micro 2c: keyword-repeat -0.633 (axis 1 减少死循环)
  - Case study (cite mechanism_case_studies.md):
    Reddit task #81 DOM vs P-text — DOM combinatorial loops on upvote button via repeated stale click;
    P-text matched DOM through 2 actions then collapsed onto same upvote control with no_progress.
    The divergence: action-state serialization issue (DOM saw button-state change "Upvote→Retract upvote",
    P-text didn't).

### 5.X.3 Axis 2 (prompt) — secondary
  - LLM mechanism: SoM-prompt 让 agent commit 更 honest (prompt 描述 "point at marked element" → 
    decision prior 更 conservative) → search-phrasing 减 / backtracking 增 / FP 降
  - Cross-dimension evidence:
    * Outcome 0b: P-SoM FP rate 0.48% (5-mode lowest, 'most honest commit' marker)
    * Outcome 0c: +P-SoM (axis 2 on top of P-text) adds another +3.33pp drop-one
    * Outcome 0d: P-text↔P-SoM Jaccard 0.571 (≤0.7 sentinel safe — 不同 task pool)
    * Macro 1b: axis 2 cascade 3/8 dominant on red strategy metrics (search/type/scroll)
  - Case study:
    Reddit task #7 DOM vs Phantom-SoM — DOM overfit visual description into long exact query, 
    failed search 30 steps. P-SoM treated task as "find OP recipe comment", broader query "cake recipe",
    succeeded in 5 steps with comment permalink as terminal object.
    Mechanism: prompt-induced query breadth + marked-comment affordance.

### 5.X.4 Axis 3 (image) — weak (text-dominated substrate)
  - LLM mechanism: 图主要是 content clue (post attachments) not 导航 affordance.
    + image 让 agent visual-hijack 到 image token (attention dilution from sparse text into dense pixels)
    → SoM red adj_SR 10.48% < P-SoM 13.81% — 加图反 hurt
  - Cross-dimension evidence:
    * Outcome 0a: SoM red 10.48% < P-SoM red 13.81% (-3.33pp regression with image)
    * Macro 1b: cls 5/8 axis 3 dominant vs red 3/8 (差异显著)
    * Efficiency 3b: red token cost +733 image tokens, no SR benefit
  - Case study:
    Reddit task #0 Phantom-SoM vs SoM — both correctly identified target sushi platter image.
    P-SoM clicked sushi image URL three times then recovered to actual post URL "/i-ate-sushi-platter"
    at step 6. SoM stuck in image-anchor trap for all 30 steps, alternated between sushi image URL
    and /f/food, never broke through.
    This is counterexample to naive "image always helps" — image disambiguation 在 reddit 上 
    over-anchored marked image element vs actual post link.

### 5.X.5 Compound axis 1+2 (P-SoM vs DOM)
  - Aggregate SR delta: red P-SoM 13.81% > DOM 9.52% (+4.29pp, statistically modest within 2σ noise)
  - Task-pool Jaccard 0.571 — 同 SR 不同 routing pool (4-fold drop-in property (d))
  - Cross-dimension: routing arm 价值在 task-pool complementarity 而非 aggregate SR

### 5.X.6 Capability tier consideration (B0 vs B1)
  - B1 reddit 5-mode 数据: DOM 9.5%, SoM (TBD), Vision (TBD), P-text (TBD pending), P-SoM (TBD pending)
  - Layer 1c strategy gradient B1: search-loop 同样存在 (rates TBD)
  - Important: B1 reddit phantom 数据 pending; capability-tier scaling claim 留 Section 7 generalization
  - 现在写: B0 reddit mechanism 已 saturated, B1 cross-tier 论证延后

### 5.X.7 Acknowledged silent-failure noise
  - Tier 1-5 audit + click probe 量化的 silent-failure modes (~76% of failed trajectories
    have signature) does NOT specifically penalize cross-mode comparison on reddit because:
    * §106 union_bound mode asymmetry 1.7× DOM:SoM (Tier 3 estimate range 1.5-6.5×, reddit 
      处于低端)
    * Agent click-loop pathology (35% of click-loop fail) — search-box-no-type, heading-as-link —
      affects all DOM-bearing modes similarly (DOM/SoM/P-text/P-SoM/P-prompt 都看到 search/heading 
      节点), so cross-mode SR gap 部分自带 cancellation
    * Vision (-element_id path) 受影响最小 (Tier 2 mode concentration 57%) — 这本身是 paper finding 
      支持 axis 3 image 的 capability-immunity 而非 contamination
  - Recommendation: Section 4 限制条款 cite Tier 3 5-category taxonomy, Section 5 mechanism 论证 
    建立在 silent-failure-aware-but-not-silent-failure-driven evidence (Outcome Jaccard, Macro 
    cascade) 上
```

## 输入资源（必读）

```
docs/analysis/cross_sites/site_mechanism_dictionary.md   ← reddit substrate claim (5.X.1) 主参考
docs/analysis/cross_sites/site_mechanism_dictionary.json
docs/analysis/cross_sites/sr_fp_per_mode.{json,md}        ← Outcome 0a/0b 数字
docs/analysis/cross_sites/phantom_lift.csv (-> results/phantom_paper/)  ← Outcome 0c oracle lift
docs/analysis/cross_sites/axis_effect_size_report.md      ← Macro 1a/1b cascade
docs/analysis/cross_sites/axis1_microbehavior_report.md   ← Micro 2a-2e
docs/analysis/cross_sites/mechanism_per_task.json         ← E1/E2/E3/E4 metrics
docs/analysis/cross_sites/mechanism_case_studies.md       ← 8 case studies (4 reddit)
docs/checkpoints/paper_planning.md §3 (4-dimension framework)
docs/checkpoints/paper_planning.md §2 (3-axis theory)
docs/analysis/paper_drafts/section3_definition.md         ← 已写, 知道 phantom 设计 framing
docs/analysis/layered_evidence_status.md                  ← live evidence 现状

# 已 done 的 reference (B0 reddit 6-mode 全齐):
results/visualwebarena/phase1/B0_3mode_reddit_20260422/             # DOM/SoM/Vision
results/visualwebarena/phase1/B0_phantom_text_reddit_20260427/      # P-text
results/visualwebarena/phase1/B0_phantom_som_reddit_20260428/       # P-SoM
results/visualwebarena/phase1/B0_phantom_prompt_reddit_20260429/    # P-prompt 210/210 ✅
```

## 输出

`docs/analysis/paper_drafts/section5_mechanism_reddit.md` (~2500 words)

格式约束：
- 学术 prose（NOT bullet-heavy；要写成 paper-readable 段落）
- 每条 evidence cite 时用 `(Outcome 0c +3.81pp drop-one oracle lift)` 或 `(Macro 1c, search-loop 51.9%→35.7%)` 这种内联格式
- 不复 Section 4 全部数字 — 只 cite 论证需要的最关键 ~10-15 个数字
- Cross-axis interaction 用一段 prose 串起 5.X.5
- Case study 引用 mechanism_case_studies.md 任务 ID + 1-2 sentence summary，不重复全文
- Acknowledge silent-failure noise (5.X.7) — 但用 Tier 3 taxonomy + cross-mode bias 1.5-6.5× lit-cited 数字 而非 over-claim
- 结尾留 forward reference: "Section 6 routing implementation will leverage these mechanism insights to..."

风格 hint：参考 ACL/EMNLP main track 论文的 mechanism analysis section。Avoid:
- "We discovered" / "Our novel" overreach
- Bullet lists (academic prose 用段落)
- "It is interesting that" / "Surprisingly" 这种 filler
- Speculation 没 evidence backing

## 不要做的事

- 不要写 classifieds / shopping (数据 not ready)
- 不要写 capability-tier B0 vs B1 (留 Section 7)
- 不要写 router design (留 Section 6)
- 不要 commit
- 不要 modify 任何 evidence file
- 不要 cite 数字超过 source artifact 实际值（live 数字以 `sr_fp_per_mode.json` 为准）
- 不要把 silent-failure audit 当 Section 5 主线 — 仅作 5.X.7 acknowledgement (~150 words)

## 验证

完成后跑：
```bash
wc -w docs/analysis/paper_drafts/section5_mechanism_reddit.md   # ~2500 期望
grep -c "Outcome\|Macro\|Micro\|Efficiency" docs/analysis/paper_drafts/section5_mechanism_reddit.md  # ≥ 10 (evidence cite)
grep -c "task #" docs/analysis/paper_drafts/section5_mechanism_reddit.md  # ≥ 3 (case study refs)
```

## token 预算

~30K (read references + write 2500 word prose)

## 触发命令

```bash
~/.npm-global/bin/codex exec --skip-git-repo-check \
  -C /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents \
  -s danger-full-access \
  --output-last-message logs/codex_section5_reddit.last.md \
  - < docs/checkpoints/codex_prompts/section5_reddit_prose.md \
  > logs/codex_section5_reddit.run.log 2>&1
```
