# Cross-mode Routing Failure Taxonomy — B0 classifieds

> Modes: dom, som, vision, phantom_text, phantom_som, phantom_prompt | common tasks N=224 | deterministic taxonomy (no sub-agent) | EARLY_STEPS=5
> ⚠️ PROVISIONAL — 6/6 mode, single (model,site), presence-light (NAV/IMG from step log). NOT paper-grade.

## 0. 方法 — 为什么 task-centric 而非 pairwise

6 mode 两两比较 = C(6,2)=15 对; exclusive 子集 = 2^6 爆炸。本框架给每个 (task,mode)
失败打**统一 taxonomy 标签**, 所有视图是标签聚合 → 核心是「失败类型 × mode」矩阵
(行=失败类型固定, 列=mode), **O(mode) 不是 O(mode²)**。6 mode 时只多 3 列, 不重写
叙事。脚本: `scripts/analysis/cross_mode_failure_taxonomy.py` (确定性, 无 sub-agent)。

## 1. Task classes (success matrix)

| class | N | meaning |
|---|---:|---|
| universal-solve | 9 | all modes solve (easy) |
| universal-fail | 127 | no mode solves (hard / benchmark-FP) |
| **routable** | 88 | partial — **routing value lives here** |

## 2. Failure-type × mode matrix  (← O(mode), 6-mode just adds columns)

| failure type | dom | som | vision | phantom_text | phantom_som | phantom_prompt | meaning |
|---|---:|---:|---:|---:|---:|---:|---|
| SEARCH-NAV | 6 | 15 | 10 | 10 | 8 | 3 | had category/pattern sig but never reached it (nav miss, behavioral) |
| THUMBNAIL | 37 | 24 | 26 | 35 | 35 | 32 | reached correct list page, wrong listing picked (thumbnail recog = IMG upstream) |
| UNCLEAR-NAV | 70 | 58 | 62 | 68 | 67 | 68 | task has NO sig (~65% cls search/on-this-page) — nav-vs-thumbnail UNDECIDABLE |
| IMG | 16 | 8 | 6 | 12 | 10 | 15 | reached correct listing detail but wrong answer (perception/reasoning) |
| BUDGET | 56 | 58 | 64 | 64 | 69 | 62 | trajectory_incomplete (no valid finish) |
| (of which EARLY) | 18 | 23 | 14 | 21 | 22 | 26 | ...gave up < 5 steps |

> **NAV 三分** (列表页 sig=sCategory+sPattern 判 '是否到达正确列表页'): **SEARCH-NAV**=有 sig 但
> 没到 (导航失败, 行为层) · **THUMBNAIL**=到了正确列表页没点对缩略图 (= **图像识别上游**) ·
> **UNCLEAR-NAV**=task 无任何 sig (~65% cls 是 search/on-this-page 无 sCategory) → 当前判据**判不了**
> nav-vs-thumbnail (诚实标注不强分)。**可靠的只有 THUMBNAIL+IMG (图像识别全谱, dom>som>vision 梯度)
> + BUDGET**; SEARCH-NAV 真实但小 (~6-11); UNCLEAR-NAV 大 = **判据天花板**, 拆它需 listing-level
> observation (correct item link 出现在 agent 哪个列表页) — 要 sync DOM/som obs 文本 (next refinement)。

## 3. Routing value per mode (exclusive solves + how others failed)

| mode | SR | exclusive-solve | (others' failure on those tasks) |
|---|---:|---:|---|
| dom | 17.4% | 4 | BUDGET:7, THUMBNAIL:7, SEARCH-NAV:5, IMG:1 |
| som | 27.2% | 6 | THUMBNAIL:17, BUDGET:9, IMG:3, SEARCH-NAV:1 |
| vision | 25.0% | 9 | THUMBNAIL:28, SEARCH-NAV:8, BUDGET:5, IMG:4 |
| phantom_text | 15.6% | 2 | SEARCH-NAV:3, BUDGET:3, IMG:2, THUMBNAIL:2 |
| phantom_som | 15.6% | 2 | IMG:8, BUDGET:2 |
| phantom_prompt | 19.6% | 6 | THUMBNAIL:14, BUDGET:9, SEARCH-NAV:7 |

full 6-mode oracle SR = 43.3%

## 4. Exclusive task ids (drill-down)

- **dom**: [23, 105, 132, 146]
- **som**: [49, 118, 148, 160, 171, 192]
- **vision**: [90, 97, 106, 123, 124, 125, 163, 193, 203]
- **phantom_text**: [12, 137]
- **phantom_som**: [93, 217]
- **phantom_prompt**: [1, 56, 64, 68, 117, 142]
