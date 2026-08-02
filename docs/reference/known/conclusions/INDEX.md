---
type: reference
status: complete
created: 2026-07-28
purpose: Phase 1 结论层总索引 — 台账按主题聚合后的入口 (聚合覆盖 §1–§397.10; 台账本身已到 §408)
---

# 结论层索引

台账（`../ledger.jsonl`，**2131** 条，覆盖 §1–§408）回答「这个量测过吗」。本目录的主题聚合只到 §397.10 —— 见下方覆盖性闭合处的 ⚠️。
**结论层回答「这件事现在算什么」** —— 把散落几十个 § 讲同一件事的记录聚成一个主题，
给出当前值 / 演变 / 已作废 / caveats / 证据。

查单条事实 → `scripts/maintenance/known.py <keyword>`
查一件事的全貌 → 本目录

## 五批产出

| 文件 | 覆盖 | 条数 | 节数 | 谁做 |
|---|---|---|---|---|
| `adjudicated_A1.md` | §5–§119 工程建设 + framing 成形 | 219 | 16 | Claude 亲读 |
| `adjudicated_A2.md` | §121–§164 pre-fire 审计密集 | 177 | 15 | Claude 亲读 |
| `adjudicated_A3.md` | §165–§240 fire 前冲刺 + Fire-1~6 | 229 | 15 | Claude 亲读 |
| `adjudicated_A4.md` | §241–§397 Protocol Reset + 治理 + 投稿 | 206 | 11 | Claude 亲读 |
| `retracted.md` | 全程 · 作废与待验 | 248 | 8 | Claude 亲读 |
| `measured_qualitative.md` | 全程 · 无数字实测 | 30 | 10 | Claude 亲读 |
| `data_inventory.md` | 全程 · 数据资产 | 49 | 6 | Claude 亲读 |
| `measured_D1.md` | §1–§128.5 | 219 | 66 | subagent |
| `measured_D2.md` | §128.6–§207.6 | 219 | 41 | subagent |
| `measured_D3.md` | §207.4–§311 | 219 | 27 | subagent |
| `measured_D4.md` | §312.2–§397.10 | 217 | 58 | subagent |

覆盖性实测闭合：`831 + 875 + 248 + 30 + 49 = 2033`，覆盖 **§1–§397.10**。

> ⚠️ **主题聚合落后于台账 98 条（2026-08-02）。** ledger 现有 **2131** 条，覆盖到 **§408**；
> 上面这 8 个主题文件只聚合到 §397.10。未进聚合的是 chunk 8/9/10 = **24 + 25 + 49 = 98** 条
> （§398–§408：AAAI 撤出与 REALM 双投、噪声 blast-radius、论文三分转向、证据层九产物、
> 六项验证协议）。**这段区间只能用 `known.py <keyword>` 查，不要以为这里的主题文件覆盖了它。**
> 已就地标注的 supersede 见 `measured_D1.md` §21（axis 两产物 paper-grade 终值）、
> `measured_D1.md` C3（参考图占比口径）、`measured_D3.md`（碳仪器 NVML 回落）。

A 批 `219+177+229+206 = 831` = ledger 中 §1–§397.10 区间的 ADJUDICATED 总数，无裁定漏到别批。

## 按问题找

| 你想问 | 去哪 |
|---|---|
| 这个决定定过吗？为什么这么定？ | `adjudicated_A1..A4.md`（按时期） |
| 这个说法还算数吗？ | **`retracted.md`** —— 156 条作废的全量清单 |
| 这个数字是多少？什么 scope？ | `measured_D1..D4.md`（按 § 时期） |
| 这份数据还在吗？能支撑什么？ | `data_inventory.md` —— **第一节列已不在磁盘上的** |
| 代码到底怎么实现的？ | `measured_qualitative.md` |
| **我又要犯哪种错？** | `retracted.md` **§一** —— 11 类错误模式 |

## `retracted.md` §一 是最该先读的一节

156 条作废里，**内容各不相同，犯法高度重复**。归成 M1–M11：

| | 犯法 | 典型损失 |
|---|---|---|
| M1 | 从 commit diff / 代码片段推断，不落实证 | dom 是否被污染，三轮 flip-flop |
| M2 | 查代理量不查真对象 | worker pid 冒充 sweep 驱动 |
| M3 | 分子改了分母没改 | 同型犯两次，第二次在第一次被记录之后 |
| M4 | 子串匹配冒充结构化提取 | `:8888` 命中 host 9999 |
| M5 | 小样本假确定性 | n=3 smoke 3/3 一致 → 全量 90%/72.5% 分叉 |
| M6 | 单臂测量制造幽灵 confound | "12pp 上吊绳" 补对照后拆掉 |
| M7 | in-sample 估计冠推断性名字 | Bayes ceiling / interaction / mode-invariant 三名全撤 |
| M8 | 指标判定基准跨臂不等 | 幻觉率的 S 随臂变化 |
| M9 | 照着别人报的表面问题修 = 修一半 | |
| M10 | 自己刚立的防线自己第一个绕过 | B-1906 棘轮 |
| M11 | 看到资源残留就判泄漏 | 真 flock 内核自动释放 |

**11 类里 6 类的原文自己就写下过教训，然后复发了。** 建这套东西的过程里
又发作了三次（详见下方"已知限制"第 3 条）。

## ⚠️ 已知限制（比能力更重要）

### 1. 结论层来自台账，台账来自笔记 —— 三层同源

「结论层说 X」= 「笔记这么记过 X」。**没有任何一层回到 run artifact 复算。**
台账层验过 99.6% 数字可追溯（`../verify_ledger.py`），但那只证明忠实转录，
不证明笔记本身对。要用进论文的数字，回 artifact 复算。

### 2. `_cross_chunk_flags` 是线索，不是判定

标 `named by RETRACTED §X` 只表示「某条作废记录提到了这个 §」，**不等于该记录被作废**。
2026-07-28 修正后的三分：

| | 数量 | 含义 |
|---|---|---|
| 删除 | 30 | **paper 章节号污染** —— 笔记 §1–§8 与 paper §1–§8 撞号，"paper §2" 被解析成"笔记 §2" |
| 反转 | 17 | **点名者自身已作废** —— 被它点名多半说明本条是它的**反证**（如 §397.4 错说"没有 replicate"，被点名的正是那些 replicate 记录） |
| 保留 | 117 | 待人判 |

### 3. 跨批作废回标 —— 试过，做不到，未做

按 type 切批导致同一主题的 MEASURED 与 RETRACTED 分家（D1 实证撞上：§103 那条
「少 ~50%」它找不到，因为是 RETRACTED 在 B 批）。尝试用特征数字自动回标，**失败**：
77 处命中里绝大多数是假阳性 —— `2026` 是年份、`4096` 撤的是「造成 KV 开销」这个假设
而非配置值本身、§120 原文明说「两数各自正确只是表述混淆」。

根因：**一直在用符号匹配承载语义关系**。同一根因在建这套东西的过程里发作三次
（downstream_scan 的 `0.05` → flag 的 § 号撞号 → 这次的数字）。前两次改参数，这次认输。

⇒ **用结论层里的数字前，先 `grep` 一遍 `retracted.md`。** 那是全量作废清单，人写的、有语义。

### 4. 矛盾未调和（这是设计，不是欠债）

各批的矛盾清单留在各自文件末尾：D1 13 条 · D2 10 条 · D3 6 条 · D4 13 条 ·
A1–A4 共 51 条待核。**一律两侧并列，不选边** —— 在没有新证据时制造确定性，
正是这次重建要修的病。

### 5. 有一类只有 user 能判

台账里大量裁定写的是「user 拍板 X」「advisor 2026-05-14 收口 Y」。
**这些对不对只有 user 知道**，我能验的只是「笔记确实这么记了」。
`retracted.md §七` 列了 B 批涉及的 5 条；A 批的 51 条待核里也有一部分属此类。

## 主 session 已裁的悬案（不必重裁）

- **§397.9「符号相反 = 真交互」仍然成立。** 台账给它挂了 `named by RETRACTED §397.10`，
  D4 保守并列。裁定依据：§397.10(1) 修正的是"compact namespace 只有两个 mode"这个隐含说法
  （实为三个，SoM 也在内），而该论证用的两组比较（DOM/P-prompt 都 native、
  P-text/P-SoM 都 1..K）**恰被 §397.10 确认**；另有实测支持（模型输出 element_id：
  p-som 1/12/68 · p-text 1/13/72 vs p-prompt 139/4074/26235 · dom 2/3606/61833）。
- **`preregistration_decision_test.py` 的注释不需要修。** 曾被 PROGRESS 记为"stale 坑"，
  实测推翻：第 34 行明写 `⚠️ REWRITTEN 2026-05-13 (historical):`，46-47 行成对且忠实记录了
  §143.6 的未决状态，`Makefile:471` 明写该文件已 retired。**原判断是 M1 的实例。**
- **§299.4 的 Δ−3.2pp 没丢** —— 归在 §397.10 名下（因 §397.10 引用了它），非遗漏。

## 演化链（同一主题被切在多片里，按此拼读）

- **FP 体系**：§78 → §83 → §88 → §95（简化为两条规则）→ §139.8（上游根因）→ §158.6（hard-delete）
- **pooling estimand**：K-of-N → DL+TOST → 单侧优效 → FE Decision 3A → gemini 反攻 →
  bootstrap percentile → DL/HKSJ 四层退役
- **scope**：16 → 24/4 → 36/6 → **42 conditions / 6 cells**
- **element-ID namespace**：不 patch → §295 证伪前提 → AMENDMENT_07 sequential →
  §298.4 "red herring" 被拽回 → §302 线性分解 RETRACT → §397.9 表 → §397.10(1) 修正为三个 mode
- **router**：v1–v6 → v7 walk-back（learned-only）→ 离线负结果链（0/6 Pareto 胜出）
  → §399 最有利角落复测（同族池 × cost-tier）**0/26 支配 · 0/26 在前沿上**，加严不改判


---

# 结论层**之外**还有什么（2026-07-28 user 点出，台账不覆盖）

台账只抽 `实验笔记.md`。以下都是独立产物，**查台账查不到**，新 session 必须单独看。

## 0. ⚠️ 常驻事实 —— 只写在 `CLAUDE.md` / memory 里，三道防线全都看不见

**2026-07-29 实证的盲区。** 台账的数据源是实验笔记，结论层的数据源是台账，
`find_unlanded.py` 只认斜杠多元组 —— 所以**凡是以散文形式写在 `CLAUDE.md` 或 memory 里的
常驻配置，三者都扫不到**。当天的实例：讨论噪声通道时列了「跨 GPU greedy ±3–5pp」，
而 paper-grade fire **全部在单一 A100 型号上跑**，那条根本不适用；这条信息在
`CLAUDE.md:124` 和 `CLAUDE.md:144-146` 明写着，且加载在 context 里。
REBUILD_PLAN 开头把这类错误单列为**第五类**——「不是缺一次查询，是没有应用一条已经
加载在 context 里的规则」——这是它第三次复发。

**下面这些只在 `CLAUDE.md` / memory 有，动手前必须直接去读，不要指望台账**：

| 常驻事实 | 位置 | 为什么要紧 |
|---|---|---|
| **paper-grade fire host = Condenser A100 VM**（A100-PCIE-40GB，VWA docker self-hosted），**不是** DGX→quark Tailscale；standing decision 2026-05-14 | `CLAUDE.md:124` + memory `project_paper_grade_target_host` | 决定跨 GPU / 环境类 limitation 适不适用。可用 step record 的 `energy.cpu_arch` 验证：**A100 = x86_64，DGX Spark = aarch64**（2026-07-29 扫 24 condition，96/96 全 x86_64） |
| **三层算力的角色划分**（DGX 共享争抢 = dev/mechanistic · A100 独占 = paper-grade fire · Myriad = 批处理，CGNAT 连不了 VWA） | `CLAUDE.md:144-146` | 任何「这个跑在哪 / 挡没挡别的」的推理都必须先读它 |
| **实验启动 hard rules**（同 site 只能跑一个 baseline；必须 reset；禁裸跑 runner，五个 queue script 二选一；同一物理 host 只跑一条 site chain） | `CLAUDE.md:170-185` | 违反会产生 paper-grade 污染数据 |
| **B0 proxy 协议 shim**（Anthropic URL + OpenAI tools schema + 顶层 `tool_calls`/`logprobs`；`tool_choice="required"`；GLM rescue 已物理删除） | `CLAUDE.md` B0 段 + memory `reference_aws_proxy_hybrid_shim` | 影响 B0 的可复现性叙述与 confidence 字段可用性 |
| **cost/latency canonical estimand**、**FP 体系已退役**、**condition vs cell 术语** | `CLAUDE.md` + 对应 memory | 写方法学章节时的口径来源 |

> **checklist 规则（REBUILD_PLAN Phase 5 要落的那条）**：
> **凡涉及「哪台机器 / 什么硬件 / 跑在哪 / 挡没挡别的」的推理，先回读上表，再往下想。**
> §396.7 已经把这条教训记过一次而它仍复发 —— 所以它必须是一个步骤，不是一条决心。

## 1. Canvas —— 可视化框架，四层证据的原始形态在这

`docs/checkpoints/canvas/`（Obsidian canvas，非 markdown，grep 不到内容）

| 文件 | 节点 | 装的什么 |
|---|---|---|
| `paper_section2_framework.canvas` | 42 | **Evidence ⫨ Explanation 双层 + Zoom 1-4 + 四维证据**（§108 的可视化） |
| `dual_track_taxonomy.canvas` | 19 | **3×3 干预分类学**：(i) Bug fix / (ii) Affordance synthesis / (iii) Channel addition × L1 Server-side / L2 Agent-pipeline / L3 LLM-internal |
| `experiment_matrix.canvas` | 33 | paper architecture + 六个 mode 的定义（text/prompt/image/cost 四属性） |

⚠️ `CLAUDE.md` 的 canvas 清单写的是「phantom_space + paper_section2_framework +
experiment_matrix」，但 **`phantom_space.canvas` 不存在**（2026-07-29 实证），实际第三个是
`dual_track_taxonomy.canvas`。

读法：`python3 -c "import json;d=json.load(open(...));
[print(n.get('text','')[:200]) for n in d['nodes']]"`

### 2026-07-29 逐个核完的裁定

**① `dual_track_taxonomy.canvas`（19 节点）— 两块必进，其余不进 8 页**

装的是**论文的定位坐标系**：3 类干预（(i) Bug fix / (ii) Affordance synthesis /
(iii) Channel addition）× 3 层（L1 Server-side / L2 Agent-pipeline / L3 LLM-internal）。

| 内容 | 裁定 |
|---|---|
| **paper-1 niche = (ii)×L3 inference-time only**，4 个 sub-tier 里的最后一格：Pretraining-time = **Magma**(MS, Qwen3-VL backbone) · Fine-tune-time = **ScribeAgent**(CMU, Qwen 7B, WA 51.3%) · RAG offline-explore = **AppAgent-v2**(Tencent) · **inference-time only = 我们，无人占** | ✅ **必进 §2 related work** —— 这是回答「你和谁不一样」的最短路径，且与 §109.17 的 novelty 防御同源 |
| **~34 条 environment fix**（(i)×L1 ~6 § + (i)×L2 ~28 §，笔记 §1–§108）；canvas 自注「→ paper §3 footnote disclose + Appendix」 | ✅ **必进 §3 footnote** —— 不写就是 reproducibility 漏洞（"你改了 benchmark 吗"） |
| 完整 3×3 矩阵本体 | ❌ 内部 framing 工具，不是结论，8 页放不下 |
| **GRL layer ~28 §**（flagship: walk-up click **94.4% off-target → >80%**） | ❌ 归**独立 bug paper**（CLAUDE.md 已有该规划），不占主 paper 篇幅 |
| **NLWeb**(MS, 已部署 Tripadvisor+Shopify) · **A2A**(Google, 0 done) env-side pilot | ❌ canvas 自标 paper-2 / future |
| **Gap ~7 § self-perception**（agent 不知页面状态）| ⏸ 原指向 §5 mechanism，随 mechanism 暂搁 |

⚠️ **canvas 末节写着「学长 5/5 sync 想 lock 的 3 件」——2026-05-05 提出，至今（07-29）
近三个月未 lock**，其中「Env-side pilot 进 paper-1 §7.x 还是 paper-2」在 8 页 workshop
scope 下需重新裁。**这三件应进和学长的对账清单。**

**② `paper_section2_framework.canvas`（42 节点）— 四维已跑完，但数字不可引用**

框架本体（Evidence ⫨ Explanation 双层 + Zoom 1-4 + 4×4 = 16 sub-cells）有效；
cross-mode 那一列已于 2026-07-28/29 全部实测（见上 §7）。
⚠️ **canvas 单元格里的数字是 2026-05-03 快照且多条已 retract**（仍写 `drop-one 1.7-3.8pp`、
`B0 red P-text +3.81 CI sig`，而 k=6 后 **H1 已 FAIL**）——**框架可引用，数字一律重算**。

**③ `experiment_matrix.canvas`（33 节点）— instructional，注意 scope 已收窄**

自标 *"Instructional / 指导性 (NOT status tracking)"*。含 VWA(910 task) + WA(480 task)
= 6 站 ~1390 task/condition 的全景。⚠️ **paper-1 现 scope 只有 VWA cls+red**，
WA 与 shopping 部分是更早期规划，读时不要当作当前 scope。

## 2. /diag 失败归因 —— 41 个 per-condition digest（2026-07-29 已建索引）

`docs/analysis/vwa_{classifieds,reddit}/<model>_<mode>_<site>_diag_digest.md`
三分类：**scaffold-bug / agent-limit / benchmark-FP**，含 P-rule presence-vs-causation 的逐条 caveat。

**索引已建** → `docs/analysis/cross_sites/diag_digest_index.{md,json}`
（producer `scripts/analysis/index_diag_digests.py`）。它是**导航层不是替代**——
per-rule 细节、Tier-2 深挖、P-rule 误报审计只存在于 digest 本体。

**覆盖度实况（这才是索引的主要产出）**：

| 状态 | 数量（2026-07-29 补完 B2·reddit 后）|
|---|---|
| 三分类可读 | **37 / 41** |
| **digest 自己声明归因不完整** | **3 / 41**（补前 9）|
| 指针文件（故意不含数字，转 run-specific digest）| 1 / 41 |
| 无法解析 | **0**（补前 29）|

**condition 覆盖是全的**：41 = 36 paper-grade condition（6 cell × 6 mode）+ 5 run-specific。

✅ **B2·reddit 六个 mode 的 Tier-2 已于 2026-07-29 补齐**（§402）：14 个 no-hit failed
**全部 agent-limit**，scaffold-bug 0 · benchmark-FP 0 ⇒ 该格 ~1–4% SR 是真能力地板。

⚠️ **仍自称不完整的 3 个**：`B0_som_classifieds` · `B0_som_reddit` ·
`B0_dom_classifieds_R31194`（replicate）。这三格的 scaffold-bug / benchmark-FP
仍是**未知不是零**。

⇒ **corpus 级结论仍不可作**（索引标 `not admissible`）——因为那 3 个。
**实质结论已进结论层** → `measured_D4.md` 附录 B（Y1–Y5）。

**为什么之前 29 个「解析不了」**：不是 digest 有问题，是解析器只认一种表格形状。
digest 实际有 **4 种 layout**（/diag skill 不同时期演进）：A 每类一行的表（3 个）·
B/C 三分类压在单行、数字与类名顺序两种（14 个）· D 三列表但计数列是散文
（`~100% (221/221 failed)`，17 个）· 指针文件（1 个）。四种全认后 **未解析归零**。

> ⚠️ **B-1913 教训（2026-07-29）**：本索引第一版只匹配一种表格形状，解析中 2/41，
> 把未解析当成零，于是打印出「**None.** 每个 condition 都是 100% agent-limit，
> 零 scaffold bug、零 benchmark-FP，pipeline 没有制造低成功率」。**两个方向都是假的**——
> `B1_som_classifieds` 写着 benchmark-FP ≈1.5%，`B2_vision_reddit` 含 success 侧
> benchmark-FP（task 160 / B-1889）。而后者第 60 行**早就写下了那条被脚本违反的警告**：
>
> > ⚠️ 本 digest 的三分类**不完整** —— 未深挖不等于「无 scaffold-bug / 无 benchmark-FP」，
> > 只代表本轮没有查。**请勿据此下「pipeline 干净」结论。**
>
> 现在脚本区分 `parsed` / `self_declared_incomplete` / `unparsed` 三态，
> **空白一律显示 `?`（= 没查，不是零）**，且只有全部解析成功才允许 corpus 级陈述。
> 这是 M2「查代理量不查真对象」的同型：**把解析失败当成测量结果**。

## 3. 正在跑的（状态会变，别信这里的快照，去查）

| 什么 | 在哪 | 查法 |
|---|---|---|
| mechanistic canonical sweep 24 cell | DGX，**驱动 pid 在 `.sweep.pid`**（不是 worker pid） | `cat logs/mechanistic_canonical/.sweep.pid && ps -p $(cat ...)` |
| WA reddit 全量 6 模式 | A100 | `_status/tasks/*.md` frontmatter |

⚠️ §397.10(4) 的教训：**查驱动 pid，不查 worker pid** —— 一个子进程退出不说明任务结束。

## 4. 未来实验（尚无数据，别当证据）

- **B3 = MiMo-VL** 跨族第三模型：DGX 适配 → **A100 fire**（12 conditions ≈ 2-2.5 周）
- **Phase 1b shop 扩展**：shop × 3 models × 6 modes，2026-09+ 期
- **WA 其他站点**：shopping / shopping_admin **故意不开**（无 reset 实现 ⇒ 不可能 paper-grade）

## 5. Router 的真实状态（2026-07-28 查证）

**live router 一次都没跑过。** `task_pass2_router` = `SUPERSEDED 2026-07-16`：
H10 结构 fail-closed（≤3/6 可训）+ 会议拍板不打 live Pass-2 ⇒ live router 推 paper-2。
现有全部是 **offline replay**（`results/phantom_paper/l1_router_offline_20260715/` 等，
产物自带 OFFLINE / NON-GATE 大字标注）。

**offline 为什么全负 —— 五层各自独立失败**：

| 层 | 失败点 | 证据 |
|---|---|---|
| ① 标签供给 | solvable 7-43% ⇒ 每 cell 仅 15-97 标签 ⇒ **4/6 cell 训不出**；B0·red **0/5 folds** | §379 |
| ② 打不过白送的 | **0/6 Pareto 支配 always-cheapest**；cls·B2 把 212/224 送去最便宜 = 重新发现"永远用 Vision"，−20.8% 还不如固定策略 | §387.16.4 |
| ③ 标签不是它声称的 | **12.5-54.64% 的标签**上 MODES 顺序返回了**严格更贵**的成功 mode，而 docstring 声称 ascending prior cost | §395.2 |
| ④ best_mode 不稳 | red·B0 五折选 **DOM/DOM/SoM/SoM/DOM** —— "连自己的重采样都复现不出来" | §392.2 |
| ⑤ AUROC ⇏ 可用 | red·B2 是唯一显著格，AUROC **0.483**（低于随机） | §394 |

② 在**四重**加严下均成立（诚实阈值 §388.4 / bundle 置换 §388.7.2 / 真嵌套 §392.2 /
**最有利角落 §399**）。§399 用独立 producer 把 ② 放到它最有利的配置下复测 —— 同族池
（B0+B1，which-mode 冲突 45-48% vs 跨族 81.8%）× cost-tier 标签（结构上免疫 ③ 的缺陷，
plug-in 天花板 97.5%）：**严格支配 0/26 arm×cell**，**相对六固定 mode 菜单非支配 0/26**
（router 从未落在经验 Pareto 前沿上）。唯一过锁定判据（95% 非支配 vs always-cheapest）的
reddit·B0 在 7 臂里过 5 臂 —— 含跨族、含 which-mode、**含 per-cell 训练** ⇒ 与同族/粗粒度/
池化三者都无关，真因是该格 always-cheapest（Vision 7.39% SR）本身弱。
⚠️ 引用 pass 率必须说明是哪一档：**非支配 = admissibility，支配 = superiority**（§399.3）。

⚠️ **§401 两条修正**：(a) 那句「同族/粗粒度/池化都不是原因」是从「两 arm 都过 0.95」推的，
**该推理无效**——反例就在同表：reddit·B0 which-mode **15.27%@0.10415** 优于 cost-tier
**14.29%@0.10803** 的两个轴，却因都及格被判等价；且 reddit tier 标签 **63/14**（少数类 18%）
严重不平衡，粒度与标签变异在唯一通过的 cell 上混淆。现改报共现 + 点估计。
(b) 「H-pool 数学上不可能（Vision 是成本下界）」的反驳**已被证伪**——Vision 只是 per-mode
mean 最低，在 **47–71% 的 task 上不是最便宜**，per-task cost oracle 便宜 **22–46%**。
那张 headroom 表是 cost-routing 上界，可独立进论文。

## 6. 已知未落地的裁定（`find_unlanded.py` 首跑）

- **§108 四维证据框架** —— 代码 4/4，两稿 0
- **§135.2 HKSJ** —— §215 承诺"新增 Appendix-D-bis HKSJ-adjusted RE sensitivity 行"，两稿 0 次
- **§155.3 `Raw/Adjusted/Same-task`** —— SR 三口径，代码 2/3，稿中 0
- **§178.5 / §211.2 / §109.17** —— benchmark / 模型族 / industry 定位在稿中全 0 ⇒ related work 偏薄

## 7. per-mode 四维画像 —— ✅ 已跑完（2026-07-28，§400）

历史：2×2 的目的是 **disentangle 两个效应**（§103），做完归因就停了；
**Vision 结构性地不在 2×2 网格上**（无 AXTree text）⇒ 连顺带算到的机会都没有。
此前只有 Macro 维按 mode 跑过一次。

**现状**：四维 × 6 mode × 6 cell 全部跑完 →
`docs/analysis/cross_sites/per_mode_four_dimension_profile.{md,json}`。

18 个指标里 **7 个有 6/6 cell 一致的极值 mode，全部是 Vision** ——
但**经验发现是 0 个**（§401.2 修正了 §400.2 的二分类）：

| 类 | 指标 | 幅度 |
|---|---|---|
| **经验** | —— | **0 条** |
| ◆ 架构下游 | `scroll_frac` Vision 6/6 最高 | **1.25–6.77×** 次高 mode |
| ◆ 架构下游 | `action-execution failure rate` Vision 6/6 最高 | 1.06–1.60× |
| ◆ 架构下游 | `page-unchanged (no-op) rate` Vision 6/6 最高 | 1.07–1.58× |
| ⚙️ 构造必然 | `locator fallback rate` Vision 6/6 最低 | Vision 发坐标零 element id，**几乎不进 locator 路径** |
| ⚙️ 构造必然 | `tokens` / `cost` / `cost_rel_dom` Vision 6/6 最低 | 不带 AXTree 文本 |

⚠️ **◆ 与 ⚙️ 都不得当行为发现引用。** ⚙️ 是设计的重述；**◆ 三条同属一条机械链**
——坐标寻址 → 点不准 → 页面不变 → 被迫滚动重定位，幅度真实但**方向可从设计预测**
（Gemini cross-AI 2026-07-29 抓，§401.2 落地）。升格 ◆ 为行为发现需先建立
「坐标寻址系统应得多少」的基线，再证明 Vision 超出它——本画像没做。
机制解释属 Explanation 层，canvas 自己的 reviewer caveat「Evidence ≠ Explanation」适用。

⚠️ 另三处已修（§401）：**tie 曾被 display order 静默打破**（B2·cls 的 SoM 与 Vision
都是 2.2321%/5 solves；修正后 `unique solves` 的最高 mode 由 SoM 3/6 变 **Vision 2.5/6**）·
**跨 mode 曾未配对**（B0·red 拿 P-SoM 201 task 对比其余 203）· **step 级比率的 estimand
未声明**（现并列报 task-macro 与 pooled-step，两者差距可观：B1·cls Vision action-fail
**0.4540 vs 0.6386**）。

⚠️ **canvas 数字已 stale，别照抄**：`Efficiency × mode` 仍写 `drop-one 1.7-3.8pp`、
`Outcome × task` 仍写 `B0 red P-text +3.81 CI sig` —— 那是 2026-05-03 k=4/5 快照，
k=6 后 **H1 已 FAIL**（θ_FE 0.7897, p=0.807, §395.6）。产物里没有一个数字取自 canvas。

✅ `B0·red · P-SoM` 的 `read_jsonl_dedup: summary identity mismatch` **已定根因**（§400.1）：
quarantine→resume rerun 写了新 summary 但没换 steps JSONL，只影响 **task 87 / 149 两个 episode**。
全库审计 `audit_steps_summary_identity.py`：36 组合 / 7686 scored episode 仅此 2 个 = **0.03%**，
其余 4 个带 stale/quarantine 的 condition 全部通过 ⇒ 个案非流程缺陷。
