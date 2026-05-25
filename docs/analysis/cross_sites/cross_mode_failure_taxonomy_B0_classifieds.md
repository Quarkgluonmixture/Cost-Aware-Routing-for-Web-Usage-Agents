# Cross-mode Routing Failure Taxonomy — B0 classifieds

> Modes: dom, som, vision | common tasks N=224 | deterministic taxonomy (no sub-agent) | EARLY_STEPS=5
> ⚠️ PROVISIONAL — 3/6 mode, single (model,site), presence-light (NAV/IMG from step log). NOT paper-grade.

## 0. 方法 — 为什么 task-centric 而非 pairwise

6 mode 两两比较 = C(6,2)=15 对; exclusive 子集 = 2^6 爆炸。本框架给每个 (task,mode)
失败打**统一 taxonomy 标签**, 所有视图是标签聚合 → 核心是「失败类型 × mode」矩阵
(行=失败类型固定, 列=mode), **O(mode) 不是 O(mode²)**。6 mode 时只多 3 列, 不重写
叙事。脚本: `scripts/analysis/cross_mode_failure_taxonomy.py` (确定性, 无 sub-agent)。

## 1. Task classes (success matrix)

| class | N | meaning |
|---|---:|---|
| universal-solve | 15 | all modes solve (easy) |
| universal-fail | 138 | no mode solves (hard / benchmark-FP) |
| **routable** | 71 | partial — **routing value lives here** |

## 2. Failure-type × mode matrix  (← O(mode), 6-mode just adds columns)

| failure type | dom | som | vision | meaning |
|---|---:|---:|---:|---|
| SEARCH-NAV | 8 | 7 | 11 | had category/pattern sig but never reached it (nav miss, behavioral) |
| THUMBNAIL | 39 | 26 | 24 | reached correct list page, wrong listing picked (thumbnail recog = IMG upstream) |
| UNCLEAR-NAV | 73 | 70 | 59 | task has NO sig (~65% cls search/on-this-page) — nav-vs-thumbnail UNDECIDABLE |
| IMG | 11 | 4 | 3 | reached correct listing detail but wrong answer (perception/reasoning) |
| BUDGET | 59 | 49 | 73 | trajectory_incomplete (no valid finish) |
| (of which EARLY) | 31 | 24 | 19 | ...gave up < 5 steps |

> **NAV 三分** (列表页 sig=sCategory+sPattern 判 '是否到达正确列表页'): **SEARCH-NAV**=有 sig 但
> 没到 (导航失败, 行为层) · **THUMBNAIL**=到了正确列表页没点对缩略图 (= **图像识别上游**) ·
> **UNCLEAR-NAV**=task 无任何 sig (~65% cls 是 search/on-this-page 无 sCategory) → 当前判据**判不了**
> nav-vs-thumbnail (诚实标注不强分)。**可靠的只有 THUMBNAIL+IMG (图像识别全谱, dom>som>vision 梯度)
> + BUDGET**; SEARCH-NAV 真实但小 (~6-11); UNCLEAR-NAV 大 = **判据天花板**, 拆它需 listing-level
> observation (correct item link 出现在 agent 哪个列表页) — 要 sync DOM/som obs 文本 (next refinement)。

## 3. Routing value per mode (exclusive solves + how others failed)

| mode | SR | exclusive-solve | (others' failure on those tasks) |
|---|---:|---:|---|
| dom | 15.2% | 3 | THUMBNAIL:2, SEARCH-NAV:2, IMG:1, BUDGET:1 |
| som | 30.4% | 18 | THUMBNAIL:18, BUDGET:11, SEARCH-NAV:4, IMG:3 |
| vision | 24.1% | 10 | THUMBNAIL:13, BUDGET:4, SEARCH-NAV:2, IMG:1 |

full 3-mode oracle SR = 38.4%

## 4. Exclusive task ids (drill-down)

- **dom**: [93, 201, 214]
- **som**: [1, 4, 12, 22, 45, 47, 49, 58, 61, 79, 102, 113, 145, 179, 184, 186, 187, 206]
- **vision**: [64, 83, 101, 106, 112, 118, 125, 163, 165, 192]
