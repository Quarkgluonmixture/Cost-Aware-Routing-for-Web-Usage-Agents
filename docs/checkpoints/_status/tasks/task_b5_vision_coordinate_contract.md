---
type: task
status: planned
priority: P1
horizon: next
order: 4
blocker: "改动在 fire import 路径上 (action_utils.normalize_coordinate_pair + proxy_api_agent tool schema) ⇒ 先留 witness tag 再改; 重跑 ~$50 走 B0 同一 AWS proxy"
eta: "新 session 接手; 顺序 = ① 补 B5 六 condition 进 _status/cells + 失败桶 + diag (零 GPU) → ② coordinate_contract 改 + witness → ③ B5 vision cls 重跑 (224 题 × $0.22) → ④ 复核 §505.10 / §505.18 / §505.19 引 B5 vision 的三处"
detail: docs/reference/master_bug_catalog.md#B-1997
created: 2026-09-11
updated: 2026-09-11
---

# B5 vision 坐标契约修复 + 重跑 (B-1997)

**背景** (笔记 §508.1): GPT-5.6 不守 schema 的 0–1000 坐标契约, 步间换制 (纯像素 / x 像素·y 千分 / 纯千分);
`normalize_coordinate_pair` 按数值判档把像素 x 当千分制 ⇒ click 85% 落空, B5 vision SR 7.4 / 12.1% 是 harness 产物不是模型属性。
三层证据: click 后页面变化率 (B5 13–15% vs B0 72%) · A100 截图单例 (task 0 step 7–14) · 分页按钮像素位群体检验 (174 vs 30)。

## 顺序 (user 2026-09-11: 补的开新 session, 重跑放后续 todo)

1. **补登记 (零 GPU)**: B5 六个 condition → `_status/cells/cell_*.md` frontmatter + `failure_modes_per_cell.md` + `/diag` Tier-1 digest。
   B5 vision 两个 run 要标 `B-1997 contaminated`, 不进任何跨 backbone 比较。
2. **修契约 (fire 路径, 需 witness)**: 按 backend 声明 `coordinate_contract` —— 默认 `qwen_0_1000_by_value` (现状, B0/B1/B2 一行不动);
   B5 = `pixel`: prompt 报实际图像尺寸并要求像素坐标, 归一化 ÷W / ÷H, 保留 true-OOB 不 clamp 原则。改前 `git tag` witness (feedback_pre_fire_protocol_witness)。
3. **重跑**: `RESET_BEFORE=1 bash scripts/queues/queue_baseline.sh B5 vision classifieds` (224 题, ≈$50)。跑完先做分页像素位检验 (§508.1 的群体检验) 再看 SR。
4. **复核下游**: §505.10 (B5 union / route-away 含 vision 臂) · §505.18 (R1 视觉谓词→vision 在 cls_B5 −17.4) · §505.19 (「mode 契合度是 backbone 属性」机制句)。

## 顺带的行为阶梯 (§508.2–508.5, 零 GPU, 未进稿, 方向由 user 定)
空转率 B2→B5 降 10× 而步数在 B0 后不降 · 失败从「跑不完」迁到「答错」· thought 逐字重复 11%→0 · harness history 8 步不含 thought (记忆消融候选)。
