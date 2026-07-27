# /diag digest — B2 × `som` × reddit

*生成 2026-07-27（Tier-1 全量 + Tier-2 深挖）*

> **定位声明**：本 digest 是**单 condition** 的失败归因，不下 cross-mode / cross-model 结论。
> 跨 mode 定量比较须等 reddit 规则批（R1–R8 + H2）落地、`RULESET_VERSION` 升到 `8-reddit-*`
> 并全量重扫后再做（/diag skill「discover-then-freeze」硬纪律）。


## 1. Header

| 字段 | 值 |
|---|---|
| **Run** | `B2_som_reddit_20260717` |
| **Condition** | `phase1_som_router_0` |
| **Site / Mode / Model** | reddit / `som` / B2 = Gemma3-VL `google/gemma-3-4b-it` (local) |
| **Episodes** | 205 |
| **SR** | **1.46%** (3 success / 202 failed) |
| **ruleset_version** | `7-p6p16clsgate-b1860coord` |
| **Tier-1 三子集** | failed+hit 193 · **failed-NO-hit 9** · success+hit 0 |

## 2. Tier-1 规则分布（failed 侧）

| 规则 | 含义 | step-level hits | 命中 episode 数 |
|---|---|---|---|
| `P36` | WALK_FAIL_DEGENERATE | 1257 | 131 |
| `P5` | 感知缺失循环 | 205 | 112 |
| `P31` | budget 耗尽未完成 | 169 | 169 |
| `P4` | 根节点误操作 | 115 | 29 |
| `P14` | URL 自环 | 88 | 70 |
| `P12` | 从不翻页 | 50 | 50 |
| `P25` | 跨站任务跳过其中一站 | 35 | 35 |
| `P10` | 跨步数值记忆失败 | 6 | 6 |
| `P13` | 搜索代替浏览 | 2 | 2 |

**success 侧 fire 的规则**: 无（success 侧 0 命中）

**failed-NO-hit episode（deterministic 盲区）**: [1, 13, 64, 89, 104, 113, 153, 154, 171]

**success episode**: [130, 160, 170]


## 3. Tier-2 深挖

**覆盖范围**：10 ep（no-hit 7 + success 全 3）· 1 sonnet sub-agent

**三分类**：agent-limit 7 · **benchmark-FP 3（即本 condition 仅有的 3 个 success 全部可疑）** · scaffold-bug 0 · unclear 0

### 具体发现

- **3 个 success 无一被证实为真**：task 130（全程从未确认点中 Subscribe，在 /f/memes ↔ /forums 间震荡耗尽 30 步）· task 160（must_exclude-only eval，见 B-1889）· task 170（唯一一次语义正确的 Subscribe 点击 `action_success=false`，此后再无第二次尝试）。三者共同点：**全部 `agent_finished=false`**。
- ⚠️ **跨 episode 状态泄漏嫌疑**：`require_reset` 在 reddit 上是 no-op（`external/visualwebarena/browser_env/envs.py:172` 只有 classifieds 分支真 reset），205 个 episode 顺序跑下来订阅列表是累积态 —— 对「检查 sidebar 订阅状态」这类 eval 构成泄漏风险。task 130 的 'memes' / task 170 的 'sports' 都可能是前序 episode 的残留。**本轮未独立证实，列为待查**。
- **task 1 / 13 / 153** — reference-image→目标帖子的视觉匹配失败：选中语义无关的 forum（AskReddit 而非 Newark、personalfinance 而非 dataisbeautiful）。
- **task 89 / 104** — 把「发评论」任务当纯文本 QA：1 步 finish 交答案，零 type 动作，评论从未真正发布。
- **task 113** — create-post 多图任务完全跑偏，30 步 thought 从未出现任何目标 forum 名或 create/post 关键词。

### 为什么这个 cell 是 1.46%

(a) 视觉-参考图匹配能力显著弱于 B0/B1；(b) 状态自我监控缺陷 —— 对同一 element_id 连点 5–7 次却意识不到在原地打转；(c) 长 instruction + 多模态下容易丢失任务目标。**SoM 的标注图对 Gemma3 未构成明显帮助** —— `element_id_coerced_from_string` 几乎每步出现（模型常把 element_id 输出成字符串需 runner 强转），点击失败率高，提示它对「编号↔元素」的对应关系把握也不稳。

## 4. 🔁 Self-evolving — 提议规则

- `success==true AND agent_finished==false` → 强制人工复核，不直接计入 SR（本 condition 3/3 命中，全场性价比最高）
- `program_html.required_contents` 只含 must_exclude → 标记「可被纯 inaction 平凡满足」（→ B-1889）
- `intent` 含 'leave a comment' 且全 episode type 次数=0 → `premature-finish-no-comment-post`
- 同一 element_id 连点 ≥5 次且 url 不变（不论 content_changed 真假）→ `stuck-click-loop`

> 这些提议**尚未落码**。按 discover-then-freeze 纪律，reddit 六 mode × 三 model 的 discover 产物应合并成一批（R1–R8 + H2）后统一 bump `RULESET_VERSION` 到 `8-reddit-*` 并全量重扫，而不是逐条落码逐次重扫。

## 5. Actionable

- ⚠️ **本 cell 的 success 含 task 160（B-1889 benchmark-FP）**。若排除，SR 1.46% → 0.98%。排除与否属 prereg 级改动，**待 user / advisor 决策**，本 digest 不自行调整数字。
- 未发现需要开 B-number 的 scaffold-bug（本轮范围内）。

---

**Cross-link**: 笔记 §387.6 / §387.7 · master_bug_catalog B-1889 (task 160 passive-FP) / B-1890 (footprint 字段恒 0，勿用作判据) · `/tmp/diag_red/` Tier-1 原始扫描产物
