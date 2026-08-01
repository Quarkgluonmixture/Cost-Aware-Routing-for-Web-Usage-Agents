---
type: audit
ref: S8-pareto
title: Multi-metric Pareto (cost + lat + carbon)
status: done
priority: P1
effort: ~2h
phase: section8
blocker: ''
---

# S8 · Multi-metric Pareto

Multi-metric Pareto (cost + lat + carbon)。Section 8 前置 (~2h)。

## 已完成 2026-08-02

产物 `scripts/analysis/aggregate_multimetric_pareto.py` → `cross_sites/multimetric_pareto.{md,json}`。
单源自 `per_mode_four_dimension_profile.md`, 两者不可能不一致。

结论: latency **是**独立轴 (跨度 1.12–1.40× vs cost 1.12–1.63×; 3/6 格最便宜 ≠ 最快, 且恰好是
三个 classifieds 格)。加进去后前沿在 3/6 格变宽, 其中 B2·cls 从 1 个模式变 5 个。
**双刃**: 三轴上支配更难达成 ⇒ §5.3 的负结论 a fortiori 成立; 但非支配也更容易满足 ⇒
论文凡是把非支配当信息的地方都要对着这个更宽的前沿读。已焊进 §5.3 + Appendix A.8 (Table 15)。

碳/能耗轴仍未做 (`energy_tracker.py` 采着但 kwh/co2e 字段为 None)。
