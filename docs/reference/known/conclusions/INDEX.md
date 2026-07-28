---
type: reference
status: complete
created: 2026-07-28
purpose: Phase 1 结论层总索引 — 2033 条台账按主题聚合后的入口
---

# 结论层索引

台账（`../ledger.jsonl`，2033 条）回答「这个量测过吗」。
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

覆盖性实测闭合：`831 + 875 + 248 + 30 + 49 = 2033` = ledger 全量。
A 批 `219+177+229+206 = 831` = ledger 中 ADJUDICATED 总数，无裁定漏到别批。

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


---

# 结论层**之外**还有什么（2026-07-28 user 点出，台账不覆盖）

台账只抽 `实验笔记.md`。以下都是独立产物，**查台账查不到**，新 session 必须单独看。

## 1. Canvas —— 可视化框架，四层证据的原始形态在这

`docs/checkpoints/canvas/`（Obsidian canvas，非 markdown，grep 不到内容）

| 文件 | 节点 | 装的什么 |
|---|---|---|
| `paper_section2_framework.canvas` | 42 | **Evidence ⫨ Explanation 双层 + Zoom 1-4 + 四维证据**（§108 的可视化） |
| `dual_track_taxonomy.canvas` | 19 | **3×3 干预分类学**：(i) Bug fix / (ii) Affordance synthesis / (iii) Channel addition × L1 Server-side / L2 Agent-pipeline / L3 LLM-internal |
| `experiment_matrix.canvas` | 33 | paper architecture + 六个 mode 的定义（text/prompt/image/cost 四属性） |

`dual_track_taxonomy` 与 MAG 论文（agent-environment co-design）直接对得上，
而它**从未进过任何 paper draft**。读法：`python3 -c "import json;d=json.load(open(...));
[print(n.get('text','')[:200]) for n in d['nodes']]"`

## 2. /diag 失败归因 —— 41 个 per-condition digest

`docs/analysis/vwa_{classifieds,reddit}/<model>_<mode>_<site>_diag_digest.md`

三分类：**scaffold-bug / agent-limit / benchmark-FP**，含 P-rule presence-vs-causation 的
逐条 caveat。台账有 20 条相关记录，**远少于 41 个 digest** ⇒ 覆盖不全，
写 failure analysis 必须直接读 digest。

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

② 在三重加严下均成立（诚实阈值 §388.4 / bundle 置换 §388.7.2 / 真嵌套 §392.2）。

## 6. 已知未落地的裁定（`find_unlanded.py` 首跑）

- **§108 四维证据框架** —— 代码 4/4，两稿 0
- **§135.2 HKSJ** —— §215 承诺"新增 Appendix-D-bis HKSJ-adjusted RE sensitivity 行"，两稿 0 次
- **§155.3 `Raw/Adjusted/Same-task`** —— SR 三口径，代码 2/3，稿中 0
- **§178.5 / §211.2 / §109.17** —— benchmark / 模型族 / industry 定位在稿中全 0 ⇒ related work 偏薄

## 7. per-mode 四维画像 —— 定了框架但从没按 mode 算过

2×2 的目的是 **disentangle 两个效应**（§103），做完归因就停了；
**Vision 结构性地不在 2×2 网格上**（无 AXTree text）⇒ 连顺带算到的机会都没有。

2026-07-28 首次按 mode 跑 Macro 维，Vision 画像极强：
`scroll_frac` **6/6 cell 最高，是其他 mode 的 2.6–10 倍**；`type_frac` 最低
（B1·cla 0.0713 vs DOM 0.3449）。机制假设（**未验**）：viewport-only ⇒ 只能靠滚动探索。
`per_task_metrics(baseline, site, mode)` 已是 per-mode 签名，补齐是纯分析不需重跑。

⚠️ `B0·red · P-SoM` 抛 `read_jsonl_dedup: summary identity mismatch`，36 个组合唯一报错，待查。
