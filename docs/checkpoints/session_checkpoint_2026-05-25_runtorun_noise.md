# Session Checkpoint 2026-05-25 — Run-to-run noise 威胁 H1/H3 + SoM nodeId 发现 + fire 停

> ✅ **EXECUTED + SUPERSEDED 2026-05-25 PM**(本 checkpoint 的"待深挖"已全部执行):SoM 编号决策 = **deterministic sequential**(§4.1 答案 = design-change 非 bug,但 production SoM 是 sequential → 改之);落码 + 4 轮 codex 验证 + **AMENDMENT_07 / B-1862**(witnessed git tag + OSF kv9sf)+ Phase 1a **全 36-cond fire 重新起飞**(B0 cls dom R21557)。当前状态以 **笔记 §295/§296 + next_steps §0 + AMENDMENT_07** 为准;本文档保留作当时(fire 停)的 chronicle。

> **给新 session 的 self-contained handoff**。今天从一个 ptext sanity check 滚雪球到 paper-grade
> 核心威胁 + 一个 game-changing 的 SoM 实现发现,经 GPT 三轮 audit + 多次自我修正。对话 context 会丢,
> 这个文档是唯一入口。新 session 任务:**复盘 + 深挖**(尤其 SoM 编号决策)。
>
> ⚠️ **所有今天的文档改动未 commit**(§6),因为含多个中途被推翻的判断(§3),需新 session reconcile 后再 commit。

---

## 0. TL;DR(30 秒)

1. **起点**:验证 B-1860 坐标修复对 P-text(无图无坐标)有无副作用 → ptext archive(R19776,pre-fix)↔ current(R2647,post-fix)同 task 群对比。
2. **结论链**:B-1860 对 P-text 无副作用 ✅ → 但挖出 **run-to-run noise 威胁 H1/H3 hero(drop-one / set-difference)** → GPT 三轮 audit 确认 + 收敛 → 发现 **P79 的 SoM/phantom 用 CDP nodeId(偏离标准 sequential SoM),正是 element-ID churn(noise 主源)的实现根源**。
3. **fire 已停**(用户决定):在 SoM 编号决策 + noise mitigation 定之前不烧可能要重跑的 nodeId 数据。
4. **新 session 核心深挖**:**SoM 编号改不改 sequential?**(决定要不要重跑全 fire)+ replicate plan + visual-anchor/format confound。

---

## 1. 完整 arc(发生顺序)

1. **ptext repro 对比**(脚本 `scripts/analysis/compare_cross_run_same_condition.py`,今天新建):archive R19776 ↔ current R2647 同 task 群 SR 28.4%↔25.4%(Δ-3pp 噪声内),6 个 flip。
2. **flip 根因(3 次自我修正)**:① 初判 "model 非确定 thought 不同" ❌ → ② 用户提示 element_id flip ✅ 但我误判 harmless → ③ 误判 task16 "起始污染" ❌(实为我把 `obs_url`=action后 当起始)→ 定论:**零起始污染,B-1860 无副作用**;flip = run-to-run noise。
3. **威胁升级**(用户问 "noise 超 floor 威胁 phantom?"):查 prereg → **H1 PRIMARY = P-SoM 6-mode drop-one,task-level bootstrap(prereg L96)漏 run-to-run 方差 → anti-conservative(偏假阳)**。
4. **GPT 三轮 cross-AI audit**(briefs+outputs 见 §7):
   - round-1:确认 H1 脆;纠正我 3 点(H10 非免疫 / drop-one 正偏非必然 / self-oracle=diagnostic 非 bias)。
   - round-2(给 prereg 全文):H3 也脆(operator-level)· H10 中→中-高(training-label noise)· latency noisy 非 robust · noise 5→7 layer · hero 降级=claim-strength rubric 非 gate。
   - round-3(prune 验证):**"prune cleaner not safer"——核心威胁不变,我没过度乐观**;但 prune 过头 3 处(denominator/page-load/judge 降级不删);新增 **visual-anchor 双刃剑** + **semantic-unique vs trajectory-unique**。
5. **element-ID 纠正**(用户坚持):我之前说 element_id flip harmless → **错,矛盾 §282**。§282 实证 element-ID churn 是轨迹分叉 **dominant 源,影响行为**(模型对 id token 敏感)。
6. **noise 源 prune**(用户项目知识逐条 prune,我查证):7 层 → 2 主源 + 1 残留(§2)。
7. **SoM nodeId 发现(今天最 game-changing)**:用户质疑 "SOM_MARKS 是 sequential 吗" → 查代码 + 真实 obs definitive:**P79 SOM_MARKS/AXTree 用 CDP `getFullAXTree` 原生 nodeId(乱序不连续,如 `[5] Logout` 夹在 `[115][118]` 间),不是 sequential**。标准 SoM(Yang 2023)+ VWA 原生 image_som(`draw_bounding_boxes` L947 `index+1`)都是 sequential,**但 P79 偏离了**。
8. **fire 停**:latency alert(10s,非致命)+ SoM 决策未定 → 用户决定立刻全停(kill queue_chain + R2647 runner + watchdog)。

---

## 2. 关键定论(已确认,新 session 可直接用)

**威胁分级**(GPT 三轮收敛):
| Hypothesis | 暴露 | 机制 |
|---|---|---|
| H1(P-SoM 6-mode drop-one) | **高** | binary success 的 uniqueness operator + task-level bootstrap 漏 run-to-run |
| H3(\|P-text∖P-SoM\| set difference) | **高**(operator-level,verdict 取决 margin) | 2-mode set diff,单 flip blast radius 大;prereg L181 自承 I²=71% LOO-fragile(注:此 I² 是旧 P-text-drop-in metric,与当前 H3 gate 有混淆,GPT round-2 指出) |
| H10(realized router Pareto) | **中-高** | training-label noise(Pass-1 success matrix)+ realized SR flip;取决 cost-driven vs SR-driven margin |
| H2(a) cost | 低-中 | by-construction,不碰 binary success |
| H2(b) latency | 中-高但非 gating | systems noise 重灾区 |

**noise 源(prune 后)**:
- **主源 1:element-ID churn**(layer 2)= P79 用 CDP nodeId(per-snapshot 重分配)→ 模型对 id token 敏感 → 轨迹分叉(§282 dominant,**影响行为**)。**phantom 无图可能更敏感**(失视觉锚,§282 撤回③)。
- **主源 2:B0 MoE**(layer 3)= Qwen3-VL-235B-A22B MoE,Bedrock batch serving expert routing + GPU FP 非确定,temp=0+seed=42 仍翻(§242 实测 12% per-task flip,不可消除;§282 拆分 provider 仅 ~16%,element-ID 主导)。
- **小残留:fuzzy string_match judge nondeterm**(layer 5)= gpt-4o-mini judge `temperature=0` 但 **no seed**,label-direct,deterministic-isolation 隔离不到;N/A FP 已 B-91 patch + exclude_na;非空 fuzzy 未实测(catalog 估 <5% but real, would need probe)。
- **已 prune(项目机制处理)**:layer 0 denominator(common-mode + outcome-independent,但 GPT 要求保留为 integrity monitor)· layer 1 reset(per-condition RESET_BEFORE + B-1839 docker restart + §282 实测无污染)· page-load(networkidle+busy:1+settle 缓解,GPT 要求保留为 covariate)· agent retry(paper_grade forbids baseline_retry)· systematic drift(canonical 单 code version 无影响,但 replicate 必须 same-code pair)。

**实测 noise floor**:~12% per-task flip(dom §242)/ ~9%(ptext §292)。⚠️ **per-task flip ≠ SR-level pp**(SR 双向抵消很稳,§242 matched SR 差异在噪声内;drop-one 不抵消才是威胁)。GPT 换算:N≈210 时 1pp≈2 task / 3pp≈6-7 task / 12% flip≈25 label 不稳。

**axis 1 干净(好消息)**:DOM(nested AXTree)↔ P-text(flat SOM_MARKS)= **纯拍扁(去 `\t` nesting)**,id 都 CDP nodeId(同一套),几乎所有节点都有 `[N]`(连 StaticText/heading)→ 信息量 ≈1.00×。**不混 id 方案,不 confound**(用户曾担心,已排除)。

**mitigation framework**(DOI 锁不加 gate):witnessed **non-gating** sensitivity 层 = self-oracle symmetric discordance(脚本已实现)+ replicate-calibrated MC(H1+H3)+ evaluator-only judge probe + H10 label-perturbation;hero 降级 = **claim-strength rubric**(非 gate,3 红线:replicate fail≠H1 fail / floor≈effect≠R1→R3 / replicate 不替换 canonical label)。

**GPT bottom line**:H1 strict 若只过 1-2pp 且 floor 也 1-2pp+ → 可诚实报 prereg pass,但 hero 措辞降级 "P-SoM stable unique task-solving contribution" → "single-run oracle evidence with reproducibility caveat"。

---

## 3. 中途被推翻的判断(诚实账 — 新 session 别重蹈,且这些文档处待修)

| 我曾说 | 实际 | 谁纠正 |
|---|---|---|
| flip step-0 thought 不同 = model 非确定 | element_id flip + 措辞抖动,decision 同 | 用户 |
| task16 = archive 起始污染 | `obs_url`=action后,起始 url_before 两 run 一致,零污染;model 非确定 step-0 action | 自查代码 |
| element_id flip = decision-harmless(印证§282) | **矛盾§282**:element-ID churn 是 dominant 源,**影响行为** | 用户 |
| 6 flip 全归 model 非确定 | element-ID churn(主)+ model(次) | 用户+§282 |
| H10 只降 power 不假阳 | 非免疫(false Pareto pass + training-label noise) | GPT r1+r2 |
| latency robust(4-fold 3 腿) | latency noisy(2.5 腿) | GPT r2 |
| noise 7 层结构完备 | prune 到 2 主源,但 denominator/page-load/judge 降级不删 | 用户+GPT r3 |
| `[N]` = enumerate idx | **CDP getFullAXTree 原生 nodeId** | 用户坚持重查 |
| 视觉锚是确证机制 | confounded(§282 SoM>DOM 混了 image 轴 + format 轴,未 disentangle) | 自查 |

**⚠️ 文档状态**:笔记 **§292**(不是 §293)标题/正文原写 "flip 全归 model-nondeterm + element_id flip harmless(印证§282)" = **错的**(应为 element-ID churn 主源影响行为)—— **已加纠正 banner 指向 §294**(正文保留作 chronicle)。§293(H1 脆弱性)build on §292 但本身无 harmless claim。新 session reconcile 时确认 banner 充分即可。

---

## 4. 待深挖(新 session 核心)

### 4.1 ⭐ SoM 编号决策(最大,决定要不要重跑全 fire)
- **事实**:P79 SoM/phantom 用 CDP nodeId(偏离标准 sequential SoM)= element-ID churn 实现根源。
- **mitigation 候选**:改用 sequential-over-interactable(= VWA 原生 visid / 标准 SoM)→ 可能**减 churn + 对齐 SoM 标准定义**,比 patch element_id 干净(是向标准靠拢非发明非标准方案)。
- **待查**:① git 历史 + `docs/literature/...Set-of-Mark Text...` — P79 是否曾 sequential 后改 nodeId(用户印象 sequential)?是 bug-fix(回归原设计)还是 design-change?② sequential 是否真更稳(它仍依赖 interactable 集合稳定,需 replicate 实测 nodeId vs sequential churn 率)。③ 改 = substrate change = 影响所有 SOM_MARKS mode + witness + **重跑全 fire**。
- **决策影响**:改 → 全 fire 重跑(用新编号);不改 → disclose churn + sensitivity layer。**这是 fire 重启前必须定的**。
- **paper terminology 风险**:paper 叫 "Set-of-Mark text" 但实现是 CDP-nodeId-marks,非标准 sequential SoM → disclose 或改实现。

### 4.2 replicate plan(fire 停了,现在可规划;P0 原本=不碰 running fire,现已停)
- 独立 namespace `results/repro_replicates/`,绝不污染 canonical。same-code same-substrate pair only(避 systematic drift)。
- Probe:① same-code end-to-end replicate(B0 cls/red × {P-SoM, SoM, P-text, DOM})测总 floor;② observation-diff attribution(raw obs hash / id-stripped hash / screenshot hash → 分 element-ID vs MoE);③ byte-identical obs replay(隔离 MoE);④ evaluator-only repeat(fuzzy/ua judge,隔离 layer 5);⑤ H1+H3 replicate-calibrated MC;(⑥ H10 label-perturbation)。
- **H_robust 可证伪预测(方向待定)**:P-SoM self-discordance ≤ {dom,som,vision}? — §282 逻辑预测**反方向**(P-SoM 无图失视觉锚 → 可能更脆)。赢=phantom robust 卖点;输=phantom 更脆(hero 坏消息)。

### 4.3 visual-anchor vs format confound
- §282 SoM(90% step-0 稳)> DOM(73%)的差异**混了**:① 截图视觉锚(image 轴)② 文本格式(SOM_MARKS flat vs AXTree nested)。**编号机制不是因素**(都 nodeId)。需 P-SoM(flat+无图)对照 disentangle:P-SoM vs SoM=纯截图;P-SoM vs DOM=纯 format。
- GPT theory 洞察:**image-presence axis 不只是 cost/semantics 轴,也是 trajectory-stability 轴**(双刃剑:去图省 cost 但失稳定锚)。决策:进 paper_planning Risk 6 + 标 §2 候选(待 replicate 确认 P-SoM 是否真更脆再定是否改 §2 正文)。

### 4.4 semantic-unique vs trajectory-unique(GPT r3)
- H1/H3 现观测 **trajectory-unique**(这次跑碰巧 P-SoM 解了别的没解)。hero 想 claim **semantic-unique**(表征真 enable task class)。run-to-run noise 大 → 观测 unique 可能只是 trajectory-unique。replicate audit 本质=区分这两者。

---

## 5. Fire 状态(已停 2026-05-25 ~16:2x BST)

- **kill**:queue_chain(386567)+ R2647 runner(462381)+ watchdog(462431),verify 全死。
- **已完成 conditions(数据保留,但若改 SoM 编号要重跑)**:B0 cls dom(R31194)/ som(R9725)/ vision(R24792)。
- **中断**:R2647 B0 phantom_text(partial,~ep 50-70 / 234)。
- **未跑**:B0 psom/pprompt + B1×6 + B2×6(链未到)。
- **重启条件**:① SoM 编号决策定(改 sequential→全重跑;不改→续跑)② noise mitigation 框架定。重启用 `queue_chain.sh`(手动 16-cell,见 next_steps §0③)或 `queue_phase1_paper_grade.sh launch`(FORCE_NEW 全重跑)。
- ⚠️ **重启前必读** CLAUDE.md 实验启动 hard rules(同 site 单 baseline / RESET_BEFORE / 单 site chain)。

---

## 6. 文档落地状态(全部未 commit — 新 session reconcile 后 commit)

| 文档 | 状态 |
|---|---|
| `scripts/analysis/compare_cross_run_same_condition.py` | ✅ 新建(SR 对比 + symmetric self-oracle discordance + decision-trace + reset-goto scan),py_compile 过,ptext 实测验证 |
| 笔记 **§292** | ⚠️ **有 element-ID harmless 错,已加纠正 banner 指向 §294**(§3) |
| 笔记 **§293** | ✅ H1 脆弱性 + GPT round-1(无 harmless claim) |
| 笔记 **§294** | ✅ stub 已写(指向本 checkpoint;含 GPT round-2/3 + element-ID 纠正 + SoM 发现 + fire 停精要;新 session 可扩展) |
| `AMENDMENT_06`(prereg_amendments/) | 🟡 草稿已写(non-gating sensitivity);待整合 round-2/3(7-layer / claim-strength rubric / 5 probe / 红线)+ 命名加 "claim-strength disclosure rubric" |
| paper_planning **Risk 6** | 🟡 已更新(element-ID + latency + √6 common-mode + GPT 3 纠正);待加 visual-anchor 双刃剑 + semantic-vs-trajectory + SoM nodeId |
| next_steps §0④/⑤ | 🟡 已加 ptext repro + H1 sensitivity bullet + §293 锚;待加 fire 停 + SoM 决策 + §294 锚(本 session 末更新) |
| phase1_plan **§D4** | 🟡 已加 reproducibility sensitivity layer;待加 5 probe 扩展 + SoM 决策 |
| GPT briefs | `codex_prompts/gpt_runtorun_noise_full_audit_2026-05-25.md` + `gpt_runtorun_noise_round3_prune_2026-05-25.md` |
| GPT outputs | `codex_outputs/gpt_audit_05_25.md`(round-2)+ `gpt_round_3.md`(round-3) |

**reconcile 顺序建议**(新 session):① 确认笔记 §292 element-ID 纠正 banner(已加) → ② 写 §294(全貌)→ ③ 整合 AMENDMENT_06(round-2/3)→ ④ paper_planning/phase1_plan 补 → ⑤ 一次 commit(中途错误判断不进 git)。

---

## 7. 关键文件/锚点指针

- **代码机制**:`p79/experiment/som.py`(SOM_MARKS builder = nodeId,L64 extract_mark_id / L96 build_som_text)· `external/visualwebarena/browser_env/processors.py`(L412 getFullAXTree / L532+597 obs_node_id=nodeId / L865+947 draw_bounding_boxes sequential visid)。
- **笔记前序**:§242(SR variance RCA = MoE 12% + eval-isolation)· §282(element-ID churn dominant + 视觉锚 + 3 撤回)· §292(ptext repro,**element-ID 归因已加纠正 banner**)· §293(H1 脆弱性 + GPT round-1)· §294(本 session 全貌 stub)。
- **catalog**:B-1858(element-ID churn)· B-37(seed not propagated)· B-91(judge empty-guard)· B-1839(per-condition docker restart)· B-1860(coord contract,已 apply)。
- **prereg**:`pre_run/preregistration.md` L96-98(H1 task-level bootstrap)· L177-181(H3)· L103-111(SE-floor B-1003)· L234(H10 realized)。
- **数据**:archive `results/visualwebarena/phase1/_archive_b1860coord_R19776_ptext_partial180_20260525` · current R2647(同 phase1 目录)。

---

## 8. 新 session 第一步建议

1. 读本 checkpoint + GPT outputs(`gpt_audit_05_25.md` + `gpt_round_3.md`)+ 笔记 §242/§282/§292/§293 → 复盘。
2. 深挖优先级:**SoM 编号决策**(§4.1,查 git/literature)→ 它决定 fire 重启策略(重跑 vs 续跑)。
3. 然后 reconcile 文档(§6 顺序)+ commit。
4. fire 不要急着重启 —— 等 SoM 决策 + mitigation 定。
