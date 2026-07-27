# /diag digest — B0 dom reddit

| 字段 | 值 |
|---|---|
| **Run** | `B0_dom_reddit_20260625_154833_928747130_2827521_R11344` (manifest-bound authoritative; reddit 史上首条 paper-grade 干净 condition) |
| **Condition** | `phase1_dom_router_0` |
| **Site / Model / Mode** | reddit / B0 (Qwen3-VL-235B-A22B) / dom (仅 AXTree 文本) |
| **Episodes** | 205 |
| **SR** | **14.6%** (30 success / 175 failed) |
| **ruleset_version** | 分析跑于 `6-b12clsfull-b1860coord` (v6); **H1 site-gate 已落 → `7-p6p16clsgate-b1860coord` (2026-06-27)**: reddit P6/P16 命中现归 0 (§4/§5/§7), cls 不变 (105/52/1/1 v6==v7 实测)。本 digest 三子集数字为 v6-scan (= H1 的动机依据); v7 下 reddit success_with_hits 10→1, failed_NO_HIT 38→45 |
| **Tier-2 深挖** | 38 no-hit failed (全覆盖) + 10 success-with-hits (FP 审计) = 48 ep / 8 sonnet sub-agents |
| **生成** | /diag 3-tier, 2026-06-27 |

> ⚠️ **单 condition digest，不下 cross-mode 结论**。reddit 是该 ruleset 首个非-classifieds 站 → 本 digest 既是错因归因，也是 **discover 产物**（暴露 cls 标定规则库的 reddit 覆盖缺口 + 跨站误报）。cross-mode 定量比较须等 reddit 6-mode 齐 + ruleset 扩 `7-*` 全量重扫后才做。

---

## 0b. v8 freeze 补记（2026-07-27）

`RULESET_VERSION` 已升至 **`8-reddit-p41p46-b1890fix`**（reddit 规则批 P41–P46 + B-1890 修复 + P33 reddit
路径扩展）。全部 36 个 canonical condition 已在该版本下重扫 → **cross-mode 聚合解锁**。
本 condition 的 v8 数字如下，**跨 condition 聚合请用这一组**：

| 指标 | v8 |
|---|---|
| SR | **14.29%** (29/203) |
| failed + hit | 158 |
| **failed NO-hit** | **17** |
| success + hit | 10 |

> ⚠️ **B-1913 (2026-07-27)**: 上表 SR 已对齐 AMENDMENT_08 计分集 (reddit **203**)。此前写作 collected 分母 205 —— 那是 scored rate 配 collected 分母。分子同样按计分集重取 (被排除的 task 58/160 若曾计为成功则一并移出)。**权威来源 `docs/analysis/cross_sites/sr_per_mode.json`**；本文件其余 `/205` 计数是 episode 级覆盖率, collected 分母正确, 未改。


v8 新规则在本 condition 的 failed 侧命中: {'P44': 9, 'P45': 67, 'P46': 6, 'P43': 56}；
success 侧: {'P42': 1}。

正文各节的 Tier-2 定性结论不受版本变更影响（新规则只是把此前 no-hit 的 episode 归了类，
未推翻任何已有归因）。**若本 condition 的 success 含 task 160 / task 58，请一并读
master_bug_catalog 的 B-1889 / B-1892** —— 那两个是评测器缺陷而非模型表现。

---

## 0. v7 重扫补记（2026-07-27）

本 digest 正文的三子集与 per-rule 数字是 **v6 扫描**（H1 site-gate 的动机依据，保留原样不改）。
reddit 六 mode × 三 model 现已全部在 **`7-p6p16clsgate-b1860coord`** 下重扫完成，
本 condition 的 v7 实测如下 —— **跨 condition 聚合请用这一组**（skill 硬纪律：聚合前 verify 全 digest 同版本）：

| 指标 | v6（正文） | **v7（聚合用）** |
|---|---|---|
| SR | 14.6% (30/205) | **14.63% (30/205)** — 不变 (ruleset 版本对比, collected 分母口径一致故可比; paper SR 见 `sr_per_mode.json` = 29/203) |
| failed + hit | 137 | **130** |
| **failed NO-hit** | 38 | **45** |
| success + hit | 10 | **1** |

v7 per-rule（failed 侧，step-level）：
`P36`=585 · `P5`=96 · `P31`=94 · `P14`=18 · `P12`=15 · `P25`=15 · `P27`=3 · `P10`=2 · `P4`=1，
**`P6`=0 · `P16`=0**（H1 site-gate 生效，正文 §4/§5/§7 预告的跨站误报清零已兑现）。

正文 §3 的 Tier-2 结论（38 个 no-hit 全覆盖深挖 → 7 个 reddit 特有失败族、0 scaffold-bug、0 benchmark-FP）
**仍然成立**：v7 新增的 7 个 no-hit 是原先被 P6/P16 跨站误报"遮住"的 episode，属同一 agent-limit 家族。
⚠️ 但有一处需按新证据修正：正文断言"success 侧 10/10 真成功、0 benchmark-FP"——
本 cell 的 success **不含** task 160，故 **B-1889 不影响本 cell**；该断言在本 cell 维持有效。

**Cross-link**: 笔记 §387.6 / §387.7 · master_bug_catalog B-1889 / B-1890 · `/tmp/diag_red/B0_dom.json`

---

## 1. 三分类统计

| 类别 | failed (175) | 占比 | 说明 |
|---|---|---|---|
| **agent-limit** | **175** | **100%** | Tier-1 hit 137 + Tier-2 no-hit 38，**全部**模型能力局限 |
| **scaffold-bug** | 0 | 0% | Tier-2 38 no-hit 深挖 0 框架 bug；pipeline 干净 |
| **benchmark-FP** | 0 | 0% | failed 侧 0 评测误判 |
| **unclear** | 0 | 0% | — |

**success 侧 (30)**: Tier-2 审计 10 个 success-with-hits → **10/10 真成功，0 benchmark-FP，0 hit_causal**。即触发的规则全是 presence-only 误报（见 §4）。

> **Headline**: B0 dom reddit 的 14.6% SR **纯能力地板** —— 无 pipeline bug、无评测 FP（两侧皆 0）。失败全部是 DOM 模式 + 4B/235B 能力在 reddit 任务族上的真实局限。这对 paper 是干净的 agent-limit 证据，不需要修代码。

---

## 2. Tier-1 规则分布 (failed-only, occurrence 计数)

| Rule | 含义 | failed 出现次数 | failed episodes |
|---|---|---|---|
| **P36** | walk_fail 退化引用 (mode-robust) | 585 | 86 |
| **P31** | budget 耗尽未完成 (trajectory_incomplete) | 94 | 94 |
| **P5** | 感知缺失循环 | 96 | 61 |
| P14 | URL 自环 | 18 | 17 |
| P6 | 视觉任务 DOM 必败 ⚠️ | 17 | 17 |
| P12 | 从不翻页 | 15 | 15 |
| P25 | 跨站任务跳过其中一站 ⚠️ | 15 | — |
| P16 | 视觉图像内容 DOM 不可读 ⚠️ | 14 | 14 |
| P27 / P10 / P4 | 放弃 / 数值记忆 / 根节点 | 3 / 2 / 1 | — |

**读法**: 表层失败签名 = **P31 budget-incomplete (94) + P36 walk_fail-degenerate (86) + P5 perception-loop (61)** 三足鼎立 = "跑满 30 步没干成 / 退化引用 / 感知死循环" 的能力地板三连。但这些是**症状签名**，真正的 reddit 特有死因藏在 no-hit 子集 (§3)。⚠️ P6/P16/P25 标记 = 跨站误报 (§4)，不可当 reddit 视觉失败计数。

---

## 3. Tier-2 新发现 — reddit 特有失败族 (no-hit 38, 全 agent-limit)

cls 标定的 ruleset 对这 38 个 no-hit 完全盲。深挖揭示 **7 个 reddit 特有失败族**（classifieds 不存在），按频次/可检测性排：

### 3.1 DOM 图像盲区 — 页面内嵌图 (最主导)
任务需读「帖子页面内的图」或做 page-image VQA，dom 模式拿不到页面截图 → 系统性失败。
- **page_image_query 评测** (t78/99/167): eval 直接 VQA 截图，dom 必败。
- **图像计数/颜色/动物识别** (t89/90/91/92/95): "How many X in this picture" / "what color" / "what animal" → 猜 (kitten vs dog / white vs purple)。
- **代表**: t92 dom 看不到图→猜动物"kitten"，正确"dog"；t95 猜雪色"white"，正确"purple/pink"(彩色处理图)。

### 3.2 submission_images URL 陷阱 (高可检测)
agent 点帖子里的图 → 跳到裸图片 URL (`/submission_images/<hash>.jpg`) → 丢失帖子 URL → url_match 失败。
- **代表**: t116/t125/t135 — 都先找对了帖子，点图后停在 `submission_images/...jpg`，eval_source_agent_url 不再是 `/f/.../<id>`。t120 = start_url 本身就是 submission_images，dom 看不到，2 步放弃。

### 3.3 多目标私信早停 (高可检测)
任务要私信 ≥5 个用户，agent 发 1 条即 finish。
- **代表**: t202 (1/11), t203 (1/7) — eval program_html `must_include` 含 ≥5 用户名，agent 只访问 1 个 profile。

### 3.4 子版块导航 vs 全局搜索混淆
agent 用全局搜索代替直接导航 `/f/<subreddit>/`，落地错误子版块还 hallucinate 成对的。
- **代表**: t172 (落 f/washingtondc 而非 f/jerseycity，错版块发评论), t197 (落 f/baltimore 自称 f/food，且从不试 "sort by all time")。

### 3.5 用户主页导航语义混淆
`/user/<name>/comments` 聚合标签页被当成目的地（应去 Submissions 或具体帖评论区）。
- **代表**: t30/t32 (intent_template_id=12 "find user who posted this and navigate to comments")。

### 3.6 导航循环 (reddit 三层链接)
连续点 "N comments" 链接 / Subscribe↔Unsubscribe 反复切换，URL 不变。
- **代表**: t105 (Subscribe/Unsubscribe 29 步，page_unchanged_streak=22), t145/t69 (连续点 comments 链接 url 不变)。

### 3.7 金额数值幻觉 (image-only 金融帖)
wallstreetbets/dataisbeautiful 帖金额只在图里，agent 从评论文字/先验 hallucinate。
- **代表**: t144 ("several thousand"→编"$5000"), t146 (被评论"-400k"污染→答"-400000"，正确 $209,783.15), t123 (图表读不了，从评论读到错误总收入)。

---

## 4. Tier-2 success-with-hits FP 审计 — P6/P16/P25 跨站误报 (重要 rule-hygiene)

**10/10 success-with-hits 真成功，触发规则全 hit_causal=false。** 根因精确定位：

> **P6/P16 (视觉规则) 在 reddit 是 presence-only 跨站 FP。** 它们按 classifieds 的"图 = 页面内嵌 listing photo（dom 拿不到）"标定。但 reddit 任务的"图"是 **task reference image**，走独立 ref-image 通道送进模型 (`image_payload_bytes_ref > 0`)，**dom 模式也可见**（仅页面截图 `image_payload_bytes_screenshot=None`）。→ agent 实际看得到参考图并成功 → 规则"视觉必败"标记与成功直接矛盾。

- **代表**: t40 (2 步成功还报 P6)、t129/130/131 (intent_template_id=51 "subscribe to forum related to the image"，ref-image 可见，正确订阅)。
- **task 138 确认 (B-1884 核心)**: **真成功，非 FP**。`image_payload_bytes_ref=123288`（信封图含收件人 "Patrick"）经 ref 通道送达 → B0 改名 MarvelsGrantMan136→Patrick，`eval_source_agent_url=/user/Patrick/account`，program_html `.site-nav` 含 "patrick" PASS。**坐实 B-1884 自毁级联根因 = 真实成功改名，不是误判。**
- **P25 (t58)**: agent 确实跳过 wikipedia 只搜 reddit，但凭先验给对答案 → 成功 episode 上非因果。

⚠️ **连带影响**: failed 侧 P6(17)/P16(14) 命中也部分错配（agent 看得到 ref 图，失败另有原因）。真正的 reddit 视觉失败应由 §3.1 的 `eval_type=page_image_query` 信号捕获，而非 presence-only 的 P6/P16。

---

## 5. 🔁 Self-evolving — 提议 reddit P-rule (扩 `7-*`)

按 discover-then-freeze: reddit 落地 → 字典扩并集 → bump version → 全量重扫。27/38 no-hit 标 deterministic_candidate。**去重为 8 条候选 + 2 条 hygiene fix**:

| # | 规则草案 | signal (0-token) | 类型 | 优先级 | 来源 task |
|---|---|---|---|---|---|
| **R1** | `P-RED-PAGE-IMAGE-QUERY` | `eval_type==page_image_query AND mode==dom` → fail | agent-limit (mode-specific, **可 route**) | ⭐ 最高 | 78/99/167 |
| **R2** | `P-RED-SUBMISSION-IMG-ENDSTATE` | `eval_source_agent_url 含 '/submission_images/' AND eval 含 url_match` | agent-limit | ⭐ 高 | 116/125/135 |
| **R3** | `P-RED-IMGSTART-DOM` | `start_url 含 'submission_images/' AND mode==dom AND steps<=2` | agent-limit (可 route) | 高 | 120 |
| **R4** | `P-RED-MULTI-USER-DM-INCOMPLETE` | `program_html must_include ≥5 用户名 但 agent 只访问 1 profile` | agent-limit | 高 | 202/203 |
| **R5** | `P-RED-WRONG-SUBREDDIT` | `eval reference_url 的 /f/<X> ≠ eval_source_agent_url 的 /f/<Y>` | agent-limit | 中 | 172/197 |
| **R6** | `P-RED-USER-COMMENTS-TAB` | `final_url 匹配 ^.*/user/[^/]+/comments/?$` (停聚合页) | agent-limit | 中 | 30/32 |
| **R7** | `P-RED-NAV-LOOP` | `page_unchanged_streak>=10 AND steps>=15` 或 `≥3 连续 click url 不变` | 行为缺陷 (cross-mode 通用, **需 module**) | 中 | 105/145/69 |
| **R8** | `P-RED-SEARCH-EXHAUSTION` | `answer 含 'No post found'/'no results' AND eval=string_match` | agent-limit (cross-site 通用) | 低 | 124 |

**Hygiene fix (落 v7 必做，否则 cross-mode 聚合失真)**:
- **H1 — P6/P16 site-gate → classifieds — ✅ LANDED 2026-06-27 (`7-p6p16clsgate-b1860coord`)**: 选 site-gate 而非 ref-image carve-out — 理由 (a) `image_payload_bytes_ref` 不在 summary (在 steps), 且 `config.image` 既是 P6 触发又无法区分 FP/TP; (b) **实证 P6/P16 在 reddit 根本没抓到真视觉失败** (genuine page-image-blind 全在 no-hit), reddit 真视觉失败由 R1=page_image_query 接管 → site-gate 不丢真信号。实现 = `_benchmark_site(summary,steps)!="classifieds": return []` (benchmark_site 实测 reddit 205/205 / cls 224/224 可靠)。**验证**: reddit P6 17→0 / P16 14→0 (failed) + success-FP {P6:6,P16:8}→0; **cls 字节级不变** (105/52/1/1 v6==v7); DGX+A100 md5 `aea8c026` 一致。
- **H2 — P25 加 success-gate**: `success is not True` 才计 hit_causal（已是 success-side 收窄惯例）。

> **router 论点连带**: R1/R3 = mode-specific (换 mode 能救 → 可 route)；R7 = cross-mode 通用行为缺陷 (换 mode 救不了 → 需 retry/memory module)。reddit 失败族同样落「通用 vs mode-specific」二分，与 cls 一致 → 支撑 paper router 证据栈。但**单 condition 不下定论**，待 reddit 6-mode 齐验证哪些 mode-specific。

---

## 6. 代表 episode (按类)

| task | 类别 | 死因 | 证据 |
|---|---|---|---|
| t92 | agent-limit (图像盲) | dom 看不到图→猜动物 kitten | finish="kitten" vs ref="dog" (string_match) |
| t135 | agent-limit (img-URL 陷阱) | 滚动 11 次点图直链，停 submission_images | eval_source_agent_url 含 `/submission_images/b1b08d5f….jpg` |
| t202 | agent-limit (多目标早停) | 11 用户私信只发 1 条即 finish | must_include 11 名，agent 访 1 profile |
| t197 | agent-limit (子版块混淆) | 全局搜索落 f/baltimore 自称 f/food | eval_source `/f/baltimore` vs intent "in f/food" |
| t138 | **success (真)** | B0 真改名成功 = B-1884 根因 | ref-img "Patrick" 可见→改名→`/user/Patrick/account` PASS |
| t40 | success + P6 误报 | 2 步成功，P6 误标视觉必败 | url_match `/f/pittsburgh` PASS, hit_causal=false |

---

## 7. Actionable

- **[infra] A100 diag 版本偏移 — ✅ 已修 (2026-06-27)**: A100 部署 `diag_pattern_match.py` 曾停在 **v5 (`5-domsomvispsom-b1860coord`, Jun 9)**，缺 v6 的 P34-P40 + success-safe 收窄。本 digest 用 DGX v6 重扫（rsync run 数据到 DGX 本地）。**已 rsync v6 脚本到 A100** (md5 `a63f7488` 两端一致 + py_compile OK + A100 重扫 R11344 = 147 with-hits 行为一致) → A100 端 cron/autorun 现出 v6 数字。
- **[diag self-evolve] H1 ✅ 已落 (`7-p6p16clsgate-b1860coord`, 2026-06-27)**; **剩 R1-R8 + H2 (P25 success-gate) 待落 → 后续 bump `8-reddit-*`**（按 discover-then-freeze，reddit 6-mode 齐后一起 freeze 比较）。**cls 全量重扫-relabel 暂缓**: H1 对 cls 是 no-op (105/52/1/1 v6==v7 实测) → 18 cls digest 数字仍有效, 仅 version label 停 v6, 待 freeze 点统一 relabel (behavior-invariant, 非 stale 数据)。
- **[paper finding]** B0 dom reddit = 干净 agent-limit 地板 (0 scaffold/0 FP 双侧)，reddit 引入 cls 无的新失败族（页面图盲 / submission-img 陷阱 / 多目标 DM / 子版块导航）。→ 进 paper 失败分析 + router 证据 (mode-specific vs 通用二分成立)。
- **[非 bug] P36/P31/P5 主导 = 能力地板签名**，无需修。
- **[cross-ref]** B-1884 task-138 真成功改名复核 = 本 digest §4 独立确认（非照搬笔记）。

---

*Tier-1 deterministic (P-rules 公开可复现) + Tier-2 Claude sonnet 定性 (8 sub-agents, 48 ep)。success-side FP 审计 = P6/P16/P25 presence-only，已标。run 数据本地镜像 (无 artifacts) 在 scratchpad，可复跑。*
