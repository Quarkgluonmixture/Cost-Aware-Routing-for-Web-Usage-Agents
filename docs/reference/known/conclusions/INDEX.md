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
