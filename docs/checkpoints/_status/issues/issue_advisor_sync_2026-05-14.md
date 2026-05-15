---
type: issue
category: decision
status: active
priority: high
action: Phase 1 critical path — 审查 bug+pipeline → cls+red baseline 干净 clean run (含 Gemma3-VL) → 同步做 router 双路线
created: 2026-05-14
updated: 2026-05-14
---

# Advisor sync + discussion 收口 2026-05-14 — scope 重定

学长 advisor sync (§137) + 后续讨论收口 (§138). 详见 实验笔记 §137-§138 + paper_planning §19 decision log + ADVISOR_SYNC.md post-sync block.

## ⭐ Key priority signal — 论文写作交 advisor

学长: **论文写作不用管, 如果能跑出来他可以和我一起写**.
→ 学生 focus = **experiment execution producing results**; paper prose = advisor-side.

## Final decisions (§138 收口)

| # | 决定 | 状态 |
|---|---|---|
| 1 | **Gemma3-VL 正式纳入 baseline** — 现 3 模型: B0 Qwen3-VL-235B-A22B / B1 Qwen3-VL-4B / **Gemma3-VL = `google/gemma-3-4b-it`** (4B 量级对齐 B1 = matched-capability cross-family control; bf16 unquant 装 A100 40GB) | ✅ locked (model id 用户 2026-05-14 确认) |
| 2 | **Mechanism 部分暂搁** — §5 (patching / layer probe / logit lens / SAE) 整个先不管; §133/§136 mechanism v2 冻结存档. §137 "SAE 为重点" 作废 | ✅ locked |
| 3 | **Venue cascade**: 主 paper = EMNLP (ARR 5/25, 11 天冲) → workshop → NeurIPS | ✅ locked (用户 2026-05-14 确认) |
| 4 | **独立 bug 研究 paper** — cross-benchmark bug 聚合研究 (e.g. agisdk), 可单独投 workshop; **不替换**主 paper 的 workshop 节点 | ✅ 方向 locked |
| 5 | **Router = Phase 1 并行核心线** — 双路线: (a) rule-based (task 属性区分) (b) learned classifier routing; 未来按 mode 行为模式 route | ✅ locked |

## Phase 0 infra prereq — A100 bring-up (blocks ALL Phase 1 runs)

**重要决定 (2026-05-14)**: Phase 1 实验跑在 **A100 VM 的独立 dockerized VWA stack**, 不再用 quark docker (实验笔记 §138.8 + memory `reference_compute_resources.md`). → 解决了 A100-can't-reach-quark-VWA 的 blocker.

1. **A100 venv dep install** — `~/venvs/p79` 当前只有 torch (实测 2026-05-14 SSH), 缺 transformers / qwen_vl_utils → `pip install -e ".[analysis,dev]"`
2. **A100 VM VWA docker bring-up** — cls / red (+ shop) dockerized on VM
3. **shopping base_url config** — 旧 quirk "非 localhost", local docker 化后需重核

## Phase 1 critical path (current)

1. **审查 bug + pipeline** — 先做
2. **cls + red baseline 完全干净 clean run** — Phase 1a 干净重跑, 现含 Gemma3-VL → condition count 需重算 (cls+red × 3 模型 × 6 modes = 36 conditions / 6 cells, 待 planning confirm)
3. **同步做 router** — rule-based + learned classifier 两条路线并行

## 🟡 Pending (留 user / follow-up)

1. **§137 "SL" (推测 MLSys) 是否仍在 cascade** — venue cascade 现确认 EMNLP → workshop → NeurIPS; "SL"/MLSys 是否并存或被取代待确认
2. **Phase 1a condition matrix 为 Gemma3-VL 重设计** — queue 脚本 + condition 生成扩展 (24 → ~36)
3. **Gemma3-VL pipeline 接入** — 新 `gemma3vl_agent.py` + `local_gemma` backend + factory 注册; codex+claude cross-review 发现 4 个隐藏污染向量 (GLM parse-repair / model.revision 转发断链 / queue_chain collision B0-vs-B1-only / StepRecordV2 无 model_family) — 详 `docs/checkpoints/codex_outputs/gemma3vl_integration_crossreview_2026-05-14.md`
4. **paper §1 prose + preregistration H7/H8 banner** — router un-defer 下游 (advisor-side, 学生不 gate)

## Refs

- `docs/checkpoints/实验笔记.md §137-§138` (sync + discussion 收口 chronicle)
- `docs/checkpoints/paper_planning.md §19` (decision log)
- `docs/checkpoints/ADVISOR_SYNC.md` (post-sync block 2026-05-14)
- `issue_advisor_sync_preregistration.md` (前次 5/5 sync + preregistration lock — 独立 issue)
- agisdk: https://github.com/agi-inc/agisdk (独立 bug 研究 paper 的 cross-benchmark 参考)
