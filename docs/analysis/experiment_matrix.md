# Phase 1 实验矩阵

> 最后更新：2026-04-25
> 本文件追踪 Phase 1 baseline 实验（B0/B1 × 三模式）的运行进度与分析状态。
> 口径：Raw SR 来自 `condition_summary_v2.json` / episode summaries；Adjusted SR 仅扣除 N/A FP + eval FP（§95 废弃 visual_fp）。

---

## 一、VWA (VisualWebArena) — 三站

### Classifieds (234 tasks)

| Baseline | Mode | 进度 | Raw SR | Adjusted SR | FP | 状态 |
|----------|------|------|--------|-------------|----|----|
| **B0** (235B API) | dom | **234/234** | 14.96% (35) | 12.95% (29/224) | NA 6 | ✅ 完成 |
| | som | **234/234** | 23.08% (54) | 20.54% (46/224) | NA 8 | ✅ 完成 |
| | vision | **234/234** | 15.81% (37) | 12.05% (27/224) | NA 10 | ✅ 完成 |
| **B1** (4B local) | dom | **234/234** | 11.11% (26) | 7.59% (17/224) | NA 9 | ✅ 完成 |
| | som | **234/234** | 17.52% (41) | 13.84% (31/224) | NA 10, eval 1 | ✅ 完成 |
| | vision | **234/234** | 11.11% (26) | 7.14% (16/224) | NA 10 | ✅ 完成 |

### Reddit (210 tasks)

| Baseline | Mode | 进度 | Raw SR | Adjusted SR | FP | 状态 |
|----------|------|------|--------|-------------|----|----|
| **B0** (235B API) | dom | **210/210** | 11.43% (24) | 8.78% (18/205) | NA 5, Vis 1 | ✅ 完成 |
| | som | **210/210** | 13.81% (29) | 11.71% (24/205) | NA 5 | ✅ 完成 |
| | vision | **210/210** | 8.57% (18) | 6.34% (13/205) | NA 5 | ✅ 完成 |
| **B1** (4B local) | dom | **210/210** | 10.00% (21) | 6.83% (14/205) | NA 5, Vis 2 | ✅ 完成 |
| | som | **210/210** | 8.10% (17) | 5.85% (12/205) | NA 5 | ✅ 完成 |
| | vision | **210/210** | 4.76% (10) | 2.44% (5/205) | NA 5 | ✅ 完成 |

### Shopping (466 tasks)

| Baseline | Mode | 进度 | Raw SR | Adjusted SR | FP | 状态 |
|----------|------|------|--------|-------------|----|----|
| **B0** (235B API) | dom | **466/466** | 11.80% (55) | 4.37% (19/435) | NA 30, Vis 12 | ✅ 完成 |
| | som | 0/466 | — | — | — | ❌ 未开始 |
| | vision | 0/466 | — | — | — | ❌ 未开始 |
| **B1** (4B local) | dom | **206/466** | 9.22% (19/206) | — | — | 🔄 进行中（resume 已启动） |
| | som | 0/466 | — | — | — | ⏳ 等 DOM |
| | vision | 0/466 | — | — | — | ⏳ 等 SoM |

---

## 二、WA (WebArena) — 三站

### Shopping (192 tasks)

| Baseline | Mode | 进度 | Raw SR | 状态 |
|----------|------|------|--------|------|
| **B0** | dom | 0/192 | — | ❌ 空壳（目录已创建，0 episodes） |
| | som | 0/192 | — | ❌ 空壳 |
| | vision | 0/192 | — | ❌ 未跑 |
| **B1** | 全部 | — | — | ❌ 未开始 |

### Shopping Admin (182 tasks) / Reddit (106 tasks)

| Baseline | Mode | 状态 |
|----------|------|------|
| B0 / B1 | 全部 | ❌ 未开始 |

---

## 三、分析管线状态

### 3.1 GLM Digest（逐 episode 根因）

> `analysis/digest/digest_{mode}.jsonl`。部分 digest 正在被 GLM rerun 覆盖，记录数会继续变化。

| Run | 站点 | DOM | SoM | Vision | 状态 |
|-----|------|-----|-----|--------|------|
| B0 classifieds | VWA | 170 records | 69 records | 124 records | 🔄 GLM rerun 中（latest log: 21/233） |
| B1 classifieds | VWA | 175 records | 0 records | 101 records | 🔄 GLM rerun 中（latest log: 27/358） |
| B0 reddit | VWA | 186 records | 176 records | 191 records | ✅ 已生成 |
| B1 reddit | VWA | 189 records | 193 records | 195 records | ✅ 已生成 |
| B0 shopping | VWA | 212 records | — | — | ✅ DOM only |
| B1 shopping | VWA | 164 records | — | — | 🔄 watchdog digest sidecar 正在跑 |

### 3.2 Cross-Representation（跨模式交叉分析）

| Run | 站点 | 状态 | 说明 |
|-----|------|------|------|
| B0 classifieds | VWA | ✅ 已跑 | 3 模式完整，`analysis/results/cross_representation/` |
| B0 reddit | VWA | ✅ 已跑 | 3 模式完整，vision 已由 104/210 补到 210/210 |
| B1 classifieds | VWA | ✅ 已跑 | 3 模式完整，含 adjusted labels |
| B1 reddit | VWA | ✅ 已跑 | 3 模式完整，watchdog 自动分析已完成 |
| B0 shopping | VWA | ❌ 无法跑 | 仅 DOM 1 个条件，需 ≥2 模式 |
| B1 shopping | VWA | ❌ 未跑 | DOM 仍在进行中 |

### 3.3 Gallery（可视化画廊）

| Gallery | 站点 | 状态 |
|---------|------|------|
| B0 combined | VWA classifieds / reddit / shopping DOM | ✅ `results/B0_3mode/gallery.html` |
| B1 combined | VWA classifieds / reddit / shopping partial | ✅ `results/B1_3mode/gallery.html` |
| B0 classifieds | VWA | ✅ 单 run gallery |
| B0 reddit | VWA | ✅ 单 run gallery |
| B0 shopping | VWA shopping DOM | ✅ 单 run gallery |
| B1 classifieds | VWA | ✅ 单 run gallery |
| B1 reddit | VWA | ✅ 单 run gallery |
| B1 shopping | VWA shopping DOM partial | 🔄 watchdog 持续刷新 |

### 3.4 Signals（置信度/校准分析）

| Run | 状态 | 说明 |
|-----|------|------|
| B1 classifieds | ✅ 21 table + 20 plot | token-level + verbalized + behavioral |
| B1 reddit | ✅ 21 table + 20 plot | token-level + verbalized + behavioral；vision analysis 已完成 |
| B0 classifieds | ✅ 11 table + 8 plot | verbalized + behavioral only（API 无 logprobs） |
| B0 reddit | ✅ 11 table + 8 plot | verbalized + behavioral only |
| B0 shopping | ✅ 10 table + 7 plot | verbalized + behavioral only（仅 DOM） |
| B1 shopping | ❌ 未跑 | DOM 未完成 |

### 3.5 Digest 文档（人工分析报告）

| 站点 | B0 Digests | B1 Digests | Findings | 跨模型 |
|------|-----------|-----------|----------|--------|
| VWA Classifieds | ✅ DOM / SoM / Vision | ✅ DOM / SoM / Vision | ✅ B0 + B1 | ✅ B0_B1_findings |
| VWA Reddit | ✅ DOM / SoM / Vision | ✅ DOM / SoM / Vision | ✅ B0 + B1 | ✅ B0_B1_findings |
| VWA Shopping | ✅ DOM | 🔄 DOM partial | ✅ B0 (DOM only) | ❌ |
| Cross-sites | ✅ B0 跨站汇总 | ❌ | — | — |
| WA 全站 | ❌ | ❌ | ❌ | ❌ |

---

## 四、汇总

### 4.1 运行进度

|  | 完成 | 进行中/部分 | 未开始 | 总 cells |
|--|------|-----------|--------|---------|
| **VWA** (2×3×3=18) | **13** | **1** | **4** | 18 |
| **WA** (2×3×3=18) | 0 | 0 | **18** | 18 |
| **合计** | **13** | **1** | **22** | **36** |

完成的 13 个 VWA cell：

- B0 classifieds ×3
- B0 reddit ×3
- B0 shopping dom
- B1 classifieds ×3
- B1 reddit ×3

进行中的 1 个 VWA cell：

- B1 shopping dom：206/466，当前队列已 resume，PID=4022239

### 4.2 当前运行编排

```text
GPU/local 4B:
  ✅ B1 classifieds   234/234 ×3
  ✅ B1 reddit        210/210 ×3
  🔄 B1 shopping DOM  206/466, queue resumed at 2026-04-24 02:55:58

API/235B:
  ✅ B0 classifieds   234/234 ×3
  ✅ B0 reddit        210/210 ×3
  ✅ B0 shopping DOM  466/466

Sidecars:
  🔄 GLM rerun B0 classifieds digest: latest log 21/233
  🔄 GLM rerun B1 classifieds digest: latest log 27/358
  🔄 B1 shopping watchdog digest sidecar active
```

当前队列日志：

- `logs/queue_b1_resume_20260424_020903.log`
- `logs/b1_3mode_shopping_dom_B1_3mode_shopping_20260413.log`
- `logs/experiment_watchdog_b1_shopping_B1_3mode_shopping_20260413.log`

### 4.3 关键缺口

1. **B1 Shopping DOM**：206/466，正在 resume；DOM 完成后会 reset → SoM → reset → Vision。
2. **B0 Shopping SoM/Vision**：尚未启动；B0 shopping 目前只有 DOM，因此 cross-representation 不能跑。
3. **B1 Shopping SoM/Vision**：等待 DOM 完成。
4. **WA 全站**：仍未真正开始；B0 WA Shopping 只有空壳目录。
5. **Digest rerun**：B0/B1 classifieds digest 正在被 GLM rerun 更新，文档级人工分析可能需要等 rerun 完成后再锁数。
6. **跨模型文档**：Reddit `B0_B1_findings.md` 已生成（2026-04-24），三模式完整。

### 4.4 本轮重要状态变化

| 变化 | 影响 |
|------|------|
| B0 reddit vision 从 104/210 补到 210/210 | B0 reddit 三模式完整，可做完整 cross-rep |
| B1 reddit SoM/Vision 补齐到 210/210 | B1 reddit 三模式完整，signals/cross-rep/watchdog analysis 已完成 |
| B1 shopping 从未开始推进到 DOM 206/466 并已 resume | 当前主队列已进入 shopping |
| B1 shopping initial reset health check 显示 `shopping: FAIL`，但 auth refresh 成功 | runner 已启动，继续观察是否产生新 episode |
| B0/B1 classifieds digest 正在 rerun | digest records 暂时是动态值 |
