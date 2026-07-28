# 作废与待验（B 批：156 条 RETRACTED + 92 条 CLAIM_UNVERIFIED，§1–§397.10）

Claude 主 session 逐条通读 248 条产出，2026-07-28。聚合非转写：逐条索引见 `ledger.jsonl`。

---

## 一、错误模式（最重要的一节）

156 条作废里，**内容各不相同，犯法高度重复**。这一节按"犯法"归类，因为查内容
可以查台账，查犯法只能靠这里。每条都注明它实际造成过的损失。

### M1. 从代码文本/commit diff 推断，不落实证

- **§297.5** dom archive 是否被污染，三轮 flip-flop（纯净→污染→干净）。原文自己写下根因：
  *"从 commit message/diff 文本推断某改动影响哪个 mode 而不落实证"*。真相靠三条实证定案：
  prompt 函数分线 + 0 坐标动作实测率 + 时间戳。
- **§250.3 → §251** B0 缺 element_id 归因 schema-optional，据此改了 6 处 tool schema 描述
  （commit 8668a84）。re-run 3/3 仍失败 → 真根因是 `tool_choice="required"` 的强制本身。
- **§342** "MiMo 的 thinking 关不掉"（grep chat_template 2471 字符无相关逻辑）→ 官方 HF card
  反转：`/no_think` 能干净关（放 user msg 末尾，99.84% 成功率），控制在 RL 训练进权重、
  不在 template 层。连带作废"需换 SFT checkpoint"。
- **§323 CORRECTION** 我"修正"了一条本来正确的结论（pprompt 是否暴露 img-src），
  原因是 grep 错了文件/step。**修正本身才是误判。**

### M2. 查代理量，不查真对象

- **§397.10(4)** 报 sweep 完成，查的是 worker pid 38617，真对象是 sweep 驱动 pid 38603。
  一个子进程退出不说明任务结束。
- **§265** 断言 running fire 含某 fix，依据是磁盘上的当前代码。实证：**进程加载的是
  spawn-time 磁盘版本**，必须 `reflog × ps-lstart` 才能确定。R12265 实际锁的是 e00a54c，
  两个"已修好"的 commit 从没被加载过。
- **§214** probe artifact 报 `schema_valid_rate=1.0`，实测同一 validator 当场返回 False —
  **probe-time 与 production 代码状态 drift** ⇒ 假 PASS。

### M3. 分子改了，分母没改

同型犯两次，第二次在第一次被记录之后：

- **§387.9** 扣两个 FP 后报 6.94%→6.37%，分母仍是 205×18 ⇒ 修正为 6.40%（234/3654）
- **§393.1** `write_digests` 报 som 7.80% vs dom 6.34%，分子扣了 task 160 分母仍 205；
  **且括号里的计数 17/14 与百分比自相矛盾**（7.80% = 16/205）——
  原文结论：*"说明那句话从没被人对着数据核过"*

### M4. 子串匹配冒充结构化提取

- **§387.15.3** 判 wikipedia 取证用 `:8888` 子串，命中一个 host 其实是 9999 的 URL ⇒
  "9 个成功里 8 个未取证"实为 **9/9 全部未取证**
- **§231** witness 文件用 `episodes/*_summary.json` 判零计数，canonical 是 `_summary_v2.json` ⇒
  零计数是假的，文件当时确实存在。原文自评：
  ***"这是 memory feedback-spotcheck-length-claims 的实证再现 —— 我读过该 memory 仍犯"***

### M5. 小样本给出假确定性

- **§246** `n=3` smoke 3/3 一致 ⇒ 判 torch.compile 保数值。全量实测 90%/72.5% 分叉，
  与 vLLM 无差别。原文标注：*"本 session 第二次小样本误导，第一次是 n=1 spike 判 benign"*
- **§111.5b → §117.2** N=1 得出"反向 patching 任何层都无 effect"；N=15 后 reverse 的
  mid-layer 幅度与 forward 相当（Welch p=0.535）
- **§117.4** task-0 的 93% flip 当作代表性发现；24-task 均值 Δ=-0.093，task-0 是 outlier
- **§387.12** 4 个 sub-agent 集体断言某分支"一次都没出现"——**在它们各自抽的 6-8 个样本里成立**，
  总体上 B2 出现 374 次(psom)/895 次(dom)

### M6. 单臂测量制造幽灵 confound

- **§316** "全节点编号给弱模型多约 12pp 上吊绳"⇒ 补 dom 对照臂后基本拆掉：dom 也是全节点
  编号（纯 StaticText 点击率 dom 甚至略高），差分里抵消。原文教训：
  ***"单臂测量制造幽灵 confound"***
- **§387.10** "页面内嵌视觉的 64 任务在无图 mode 下结构性不可解" ⇒ 受控 dom→som 对比增益 ≈ 0
  （+0.00/+1.56/+0.00pp）。若真如此 som 必须显著更高。

### M7. in-sample 估计冠以推断性名字

- **§397.7**（三家 AI 同时收敛，我自审漏了）：`Bayes ceiling` 实为 modal agreement ·
  `interaction` 实为排序的代数后果 · `mode-invariant` 实为池化后的残差。
  原文诊断为什么自审抓不到：
  ***"我自审问的是『这个数换个分母还成立吗』（计量问题），三家问的是『这个数配得上这个名字吗』
  （估计量问题）——换口径复核抓不出估计量层的错。"***
- **§397.5** 按特征向量分组的天花板 83.9/89.1：reddit 78 组里 **69 组是单例**（占 75% 的行），
  单例按定义 100% 正确 ⇒ 主要在量"向量有多独特"

### M8. 指标的判定基准在各臂不是同一个

- **§397.9 → §397.10(1)** 幻觉率定义为 `element_id ∉ obs_nodes_info`，而该集合按 text payload
  分成两套键空间 ⇒ 跨臂比较等于比两个灵敏度不同的探测器。一般形状（原文）：
  ***"一个指标定义为『落在集合 S 外』，而 S 本身随臂变化 ⇒ 该比率不是同一个量，
  它是 行为 × S 覆盖度 的乘积。要么固定 S 比，要么别比。"***

### M9. 照着别人报的表面问题修

- **§397.6**：*"跨 AI 报的那条往往不是最深的那条。#9 报『分组错』真问题是『前提假』；
  #11 报『次数少』真问题是『裁定由 B 决定』；#3 报『标签互换』真问题是『证据类型选错』。
  照着报的修 = 修一半。"*
- **§397.2** codex 报"按 task_id 分组而非特征向量"，真问题是**前提假**：5 个数值特征里 3 个
  读自该 cell 自己的 step-0 观测 ⇒ 跨 backbone 同一 task 的特征向量本就不同。
  前提假 ⇒ 分组错 ⇒ 数字错，**三层一起修才算修完**。

### M10. 自己刚立的防线自己第一个绕过

- **§397.5(c)** B-1906 棘轮昨天立的，今天新写的脚本就从 grep 看不见的路径进来
  （经 `_discover_episodes` 读 episode，源码零次出现 `_summary_v2.json`）。
  教训：**防线要按能力布（谁读了 episode），不按字面布（谁写了那个文件名）**。

### M11. 看到"资源残留"就判泄漏

- **§391.3** 残留的 `.lock` 文件被判为 site lock 泄漏 ⇒ 那是真 flock（`exec 9>` + `flock -n 9`），
  内核在进程死亡时自动释放，残留文件只是空 marker。当时被拦是因为 pilot chain **真的活着**。
  教训：***"flock / lockfile / pidfile 三种语义对残留文件的容忍度完全不同，先读锁的实现"***

---

## 二、Paper 层：已死的 claim

| 死掉的说法 | 为什么 | 现在是什么 | § |
|---|---|---|---|
| **hero = drop-one oracle**，引 archive 3.33/2.56pp | k=6 三个数（0.7897/1.3528/2.0877）**一处都不在 Paper A 里**；drop-one 在 6-mode k=6 下 0.0–1.3pp，**H1 FAIL p=0.807**。稿子的 hero 建在被自己数据推翻的量上 | H1 FAIL 作为主结果**前置**（不作 caveat），立论换 H3 双轴 | §395.6 |
| **4-fold drop-in property** headline（读作"丢图性能几乎不减"/"更便宜的 SoM"） | 把 PRIMARY（独立路由空间）与 SECONDARY（部署红利）层级压平。**paper §1 正文本来就精确**，误读源在 CLAUDE.md hook + memory | 修 4 处对齐 §1；paper 未动 | §340 |
| Paper B §3「天花板是成本天花板，oracle SR 逐格等于最强单模」 | 三家 stress 独立收敛同一 P0：稿件 oracle 列 6/6 精确匹配产物的 **triage_only**，不是散文定义的 oracle。真 oracle SR 增益 +3.45~+16.07pp；**同表 solvable 列自己就否证**（union 43.3% vs best single 27.23%） | 换 `oracle_sr_cost` 行 + 两半天花板结构 | §396.2 |
| 两篇稿把 post-hoc 排除写成 **preregistered exclusions** | `preregistration.md:742` 自己明写 POST-HOC / OUTCOME-VISIBLE。**对一篇卖预注册严谨性的 paper 是最贵的一种错**（codex 独有，无人预见） | 如实标 post-hoc | §396.2 |
| Paper B §2.1「两个带图 mode 更贵」+ §6.4「cost tier」 | cheapest 行六格**全部等于 Vision** —— 丢掉整棵 AXTree 省的 token 比图像 token 贵得多（cls·B1 Vision 0.04316 vs DOM 0.05951，−27.5%）。那一档同时装着最贵的 SoM 和最便宜的 Vision ⇒ 是 **modality tier 不是 cost tier** | §6.4 更名 screenshot tier | §396.2 |
| §4.2「若 compact id 是主机制，axis-1 应更大」 | axis-1 两臂**都用紧凑 id**，该轴上 compact-id 是常量 ⇒ 推断物理上不成立。变化的是 axis-2，而 axis-2 更大 ⇒ **正确推断与稿子写的相反** | 见 §397.10(2) 的 id-regime 假说（**待验**） | §396.2 |
| 「5–25× 幻觉引用率下降」(per-cell) | 底层三值 9.75×/24.83×/2.32× ⇒ 真区间 **2.3–24.8×**，下界被抬高两倍多；且是 per-backbone 写成 per-cell。根在 `write_digests.py:170` 自己算错，paper 照抄继承 | 后被 §397.9/§397.10 整体判为**跨 namespace 无效** | §396.2 |
| abstract「portfolio irreplaceability」 | estimand 偷换：H3 测的是 **pairwise 集合差**，不是 portfolio 不可替代性 | 显式 pairwise 表述 | §380 |
| Table 3 表头「Wald p (Holm, m=2)」 | axis-2 的 7.52e-07 **未乘 2**，报的是 raw p | — | §396.2 |
| 稿件 header「28 conditions」 | 实为 30 | 30 | §380 |

---

## 三、统计与估计量：裁定的演化链

**每条都注明"现在算数的是哪个"，因为这些最容易被重新提起。**

- **pooling estimand**：DL random-effects → 争议（gemini 反攻 FE）→ **FE inverse-variance
  (Decision 3A)**，理由是 design grounds（6 cells = 完整决策族，不是总体抽样），不是
  小 k 文献。⚠️ §215 撤回过一次误引：Veroniki 2016 实际推荐 REML/Paule-Mandel，
  IntHout 2014 的 HKSJ 正是为 k≤10 —— **两篇都推荐 HKSJ-adjusted RE，不是 FE 替代**。
  DL/HKSJ 降为 Appendix-only。
- **SE floor = 0.68pp**（Agresti-Coull anchor, B-1003），prereg L718 早锁；AMENDMENT_03
  把代码从 `<=0` 对齐到 `<0.68` 并标 "implementation alignment, NOT an estimand change"。
  ⚠️ §389.8：照"实现对齐 prereg"去改**会把 estimand 倒退回 05-18 之前**。
  ⚠️ §388.3：超额 flooring 的方向是**偏向 H1**（θ_FE 0.653→0.790），不是压制 H1。
- **FE-pool power**：48.3%（理论 1-sample SE 上界）→ **81% (k=3 archive) / 97% (k=6 projected)**
  （paired-bootstrap SE 小 2.2 倍）。整条"power crisis → 战略三选一"作废。
- **Holm family key**：`(test, metric, cell_key)` → revert 回 **`(test, metric)`**
- **H10 CV protocol**：LOCO 6-fold → **task-held-out 5-fold within fixed cells**（LOCO 降 Appendix）。
  LOCO 与 per-cell LR-head 架构不兼容，且 cells 共享 task pool 不是真分布外。
- **H10 DEFER 触发**：archive 上评估 → **必须在 Phase 1a fresh data 上评估**。
  archive 与 Phase 1a 用同一批 task ID ⇒ 用 archive 锁 DEFER 等价 peeking at test set。
- **AUROC ⇏ 标签可预测**：red·B2 是直接反例（**AUROC 0.483** 却是唯一显著的那格）。
  *"全局判别与尾部可用性是两个性质，本数据上解耦；base SR 2–27% 的 regime 里
  『AUROC 高』既不必要也不充分。"*

---

## 四、element-ID / id-namespace：完整演化（这条线最长且反转最多）

```
element_id 不 patch（改 substrate = 改 estimand，正确做法是量化而非消除）
  └─ §295 证伪前提：production/标准 SoM 全是 sequential 重编号
     （WebVoyager / SeeAct-Choice / AndroidControl / browser-use，VWA 自己的 image_som 也是）
  └─ AMENDMENT_07：SoM-family 改 deterministic sequential 1..K
     └─ §298.4 "element_id 是 red herring，真机制是 MoE"（我自己中途说的）→ 被 user 两次拽回
     └─ §298.3 线性拆解 12.1% ≈ 10.5% + 1-2pp
        └─ §302 RETRACT：跨 model/modality/serving/perturbation 四维不可比 = category error
           vision 反例直接击穿（无 element ID + step-0 截图字节相同，仍 14.3% discordance）
        └─ §302.8.2 双层 framing：Layer 1 model 内在（cross-provider ~75% partial-nondet）
           + Layer 2 provider 外在（AWS Bedrock 再加 4–16×）。DashScope 4/20 bit-exact 反驳
           了"任何 provider 都一样"
  └─ §397.9 id-namespace 表 → §397.10(1) 修正：
     compact = {som, phantom_som, phantom_text} **三个**（不是两个）
     native  = {dom, p-prompt}
     vision  = **零 element_id**，幻觉率 0.000 是结构性不适用（不是 native 测量值）
```

**待验**（§397.10(2)，原文两处标"尚未验证"）：axis-1 两臂同 regime、axis-2 跨 regime，
是否就是 axis-2 (2.09pp) > axis-1 (1.35pp) 的原因。

---

## 五、基础设施：真根因链（每条都推翻过一个看似合理的中间归因）

- **eval timeout**：`async_envs.py` asyncio 泄漏假说 → 三层证据推翻（P79 从不实例化
  `AsyncScriptBrowserEnv`，那是死代码；146 次 stale-loop 是 B-1581 patch **检测到自己上一集
  装的 loop**，自指 artifact）→ §241 evaluator 复用 agent 累积态 page → **§253 退化的
  BrowserContext**（同-context 新 page 继承退化；全新 context 加载同页 ~170ms）
- **reddit auth abort**：账号 MarvelsGrantMan136 不存在（在**被污染的旧容器**上查完就归因）
  → 真 reset 后 fresh 容器里存在（id=13915）⇒ 旧容器 0 行是**账号被改名** →
  **§355 task 138 的 intent 就是"改用户名"**，B0 成功执行（改成 Patrick）× auth_refresh
  每 5ep 用旧用户名 fresh login × fail-closed abort
- **local 4B 慢**：max_new_tokens KV 开销（证伪：HF DynamicCache 增量不预分配）+
  dom 仍 encode 图像（证伪：`_clone_observation_for_mode` 对 dom 设 `image=None`）→
  **eager kernel-launch overhead**（77ms/token vs 5ms/token 带宽下限）
- **WA config**：以为 55 个 WA config 正常继承 base → `load_experiment_config` 展开
  `defaults:` 用裸 `yaml.safe_load` **不递归**，VWA 全是一级继承所以旧代码碰巧正确，
  **WA 全是二级 ⇒ 整个 base 层缺失**。WA 自 2026-05-15 生成起一次都没 fire 过，无人踩到；
  *"即使当初 fire 成功，产出的成本/碳数字也会是错的"*
- **WA/VWA 容器关系**：以为 WA 跑自己一套 docker stack ⇒ 5 个 queue 脚本硬 guard exit 1 →
  实际 **A100 上 WA reddit 就是 vwa-reddit 那个 postmill 容器**，同 image 同端口，
  两 benchmark 的 reddit task `storage_state` 字段**逐字节相同**

---

## 六、待验清单（92 条 CLAIM_UNVERIFIED 里，仍在影响当前决策的）

| 待验 | 卡在哪 | § |
|---|---|---|
| axis-2 > axis-1 是否因跨 id-regime 边界 | 需专门设计（现有数字推不出，且部分是定义性的） | §397.10 |
| DGX 24-cell sweep 会在 08-01 于第 7-8 cell 截断 | 外推预测，截断尚未发生 | §397.10 |
| 若 MiMo/GLM 也地板 ⇒ Qwen3-VL 是小模型里的异类（"finding 非 failure"） | B3 floor pilot 至今无有效数字 | §339 §342 |
| 模型轴 headroom 弱（4.02pp vs 16.07pp）是否因 menu 能力偏斜 | 自标 menu-specific caveat，"B3 8 月可检验" | §377 |
| B2(4B) 的 format/信息可分假设是否比 12B 更弱 | Judge Circuits 只测到 12B（Gemma-3-12B entangled，zero-ablate 砍半 MMLU 81→19%） | §310 |
| site flock 粒度错（per-benchmark 而非 per-container）理论上允许 WA/VWA reddit 并发互毁 | 原文标"遗留风险(未修,记录在案)"，本次因 VWA 侧无 run 未触发 | §387.3 |
| comment/reply 任务难的合取结构解释 | 原文明写"尚未验证，标记为待查" | §387.8 |
| latency ~50%↓ | 标 canonical 重算 pending | §382 |

---

## ⚠️ 七、需要 user 判的（我无从验证）

台账里大量 ADJUDICATED/RETRACTED 写的是"user 拍板 X"或"advisor 收口 Y"。
**这些对不对只有 user 知道**，我能验的只是"笔记确实这么记了"。B 批里涉及的：

- §221.1 user 即时 retract「archive 不是 prereg substrate 也不是 demote 依据」
- §223 Q5=A（Track A 真独立）/ Q9=A（Track B 所有 R-tier 可独立发表）
- §208.3 user 实测 OpenAI-style 拿 200 OK，override 了 gemini 的 REJECT MIGRATION
- §384.2 user 拍板整套 mechanism 重跑基于 A100 + GPU-type 统一到 DGX 单卡
- §293/§297/§298 user 两次把"element_id 是 red herring"拽回

---

## 八、本批的 meta 观察

**M1–M11 里，有 6 条（M2/M3/M4/M5/M9/M10）的原文自己就写下了教训**，而它们仍然复发。
§231 是最锋利的一例：*"这是 memory feedback-spotcheck-length-claims 的实证再现 ——
我读过该 memory 仍犯。"*

这与 REBUILD_PLAN 对第五类错误的诊断一致：**不是缺记录，是推理时没回读已有规则**。
台账挡得住"这事测过吗/定过吗"，挡不住这个。
