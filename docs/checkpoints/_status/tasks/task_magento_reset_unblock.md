---
type: task
status: todo
priority: P2
horizon: backlog
order: 60
blocker: ""
eta: "post-submission (08-05 之后); 一次实现解锁两个缺口"
detail: scripts/maintenance/reset_wa_sites.sh + reset_vwa_sites.sh
created: 2026-08-03
updated: 2026-08-03
---

# Magento DB restore — 一个实现解锁「第三 workload」+「第二应用」

## 为什么值得单列

证据层现在有两条结构性缺口，**它们卡在同一件事上**：

1. **第三个 workload** —— `results/visualwebarena/phase1/*shop*` = **0 个目录**（不是零 episode，
   是零目录）。两个 workload 看到符号翻转却无法刻画它绕哪个轴转，claim 4 因此永远停在
   「在这两个站点之间反转」。
2. **第二个应用** —— WA reddit **就是** `vwa-reddit` 容器（同 image / 同端口 / 同账号，
   `storage_state` 字节相同，见 `_lib_paper_grade_gates.sh:462-467`）。所以「跨两个 benchmark
   成立」在 reddit 轴上实际是**一个应用、两套 task set**。

两者都需要 shopping 站点跑起来，而 shopping 跑不起来只因为 **Magento DB restore 没实现**：
`wa_reset_supported()` 对 reddit 之外一律返回 1（硬失败，故意不静默跳过，B-647）。

## roadmap 已经过时，真实工作量比它写的小

`reset_wa_sites.sh` 的 header roadmap 写于 2026-05-17，第 1 步是「确认 WA 有自己的 docker
stack（`wa_shopping` / `wa_reddit` 容器）」，第 3 步是「WA 专属账号 + `WA_*_USER/PASS`」。

**这两步的前提已被 §387.3 RETRACTED** —— A100 上 WA 与 VWA 共用一套容器，reddit 已实证走
`_reset_vwa_local_reddit`。所以剩下的真实工作是**一件事**：Magento DB SQL restore
（VWA shopping 的 `_reset_vwa_local_shopping` 也是同一个 rc=78 stub）。

## 先做的一步（估工作量，不是实现）

看 VWA 官方 setup（`scripts/vwa/`）里有没有现成的 DB dump / restore 路径可以直接接 ——
Magento 镜像通常自带初始 dump。**估准了才好排 08-05 之后的顺序**；如果官方有现成 dump，
这可能是小时级而不是天级。

## 排序说明

§381（2026-07-16）把 shop 18-cond 排到期刊版长线，理由是「电商类与 cls 站点泛化边际低 +
6 条 B0 重挂 proxy」。⚠️ 那个理由针对的是 **18 conditions 的算力**，不是 reset 实现本身 ——
实现是一次性的，且它同时给两个缺口开路。所以本 task 与 §381 的排序**不冲突**：
先实现（便宜、一次性），跑不跑 18 conditions 另算。

## Cross-link

- `EVIDENCE_LAYER_SUMMARY.md` §4a 两条 ⚠️ NEW 条目
- [[实验笔记]] §419（B3 接线当日一并核出）· §387.3（WA/VWA 共用容器的 retraction）
- `scripts/queues/_lib_paper_grade_gates.sh` `wa_reset_supported()` (line 475)
