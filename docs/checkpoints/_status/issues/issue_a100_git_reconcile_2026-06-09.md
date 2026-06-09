---
type: issue
category: maintenance
status: open
priority: medium
action: RECONCILE 主体✅(A100→master dc34d3f, md5 全对齐). 残留 2 follow-up — ①评估 2 router 孤儿(diag snapshot 50fa462)是否回流 master ②工作流根治停用 scp 改 git pull
created: 2026-06-09
updated: 2026-06-09
---

# A100 git reconcile — 切回 master + 根治 scp 漂移工作流

> **状态 2026-06-09**: reconcile 主体 ✅ 完成。A100 现 `branch=master HEAD=dc34d3f`, tracked 改动 0, 7 脚本 md5 逐字节对齐 DGX。"两个分支" 根除。残留 2 个 follow-up (见末尾)。

## 背景 (笔记 §327)
DGX 与 A100 两独立 clone, scp 单文件部署绕过 git → A100 HEAD 永冻 `diag-discover-then-freeze` (5/27), 本地 master 落后 origin 152。每次热修只活 working tree, git 不认。= 用户 "两个分支" 痛点 + 定时炸弹。

## 已完成
**Step 1 — 拆引线 (DGX 侧)**: 孤儿 `queue_baseline.sh` 20 行 (BUG-6 QUARK_TZ + BUG-2 local-URL preflight) 回流 → commit **dc34d3f** + push 3 commit → origin/master。

**Step 2 — 停 fire + 删污染 som 数据**: kill -9 整条 chain 树 (orchestrator 951871 + queue_chain 952568 + runner 953530 + watchdog 953561, PID-based kill -0 验证全 DEAD)。删未完成污染 run **R23029** (439M, 35/234, 无 aggregate 引用) + 6 logs + applylock。

**Step 3 — reconcile (A100)**: `git add -A && commit`(working-tree 全改动固化到 diag 分支 snapshot **50fa462**)→ `git checkout -B master origin/master`。结果: branch=master, HEAD=dc34d3f, tracked 改动 0, untracked 0, master 正确 track origin/master。三方 blob 比对: 15 改动文件中 13 aligned (scp 内容已 == master, 仅 HEAD 记账滞后) + 2 orphan。7 脚本 A100↔DGX md5 逐字节对齐。

## 残留 follow-up
1. **2 个 router 孤儿评估** — `queue_phase1_router_paper_grade.sh` (33 行: B-1841 空数组 set-u guard + Pass-1 die→WARN) + `reset_and_launch.sh` (58 行) 现固化于 **diag snapshot `50fa462`** (A100), master 版是 origin 旧版。判断是「该回流的 router 热修」还是「某并行 session 半成品」→ 若回流: `git show 50fa462:<path>` 取内容 → DGX commit → push (像 queue_baseline 那样带 witness)。**Pass-2 router 尚未 fire, 不阻塞 Pass-1**。
2. **工作流根治** — 停用 scp 单文件部署, 热修改走 DGX `commit→push` → A100 `git pull` (A100 master 现已 track origin, `git pull` 直接可用)。杜绝漂移再生。

## 风险
明天 2026-06-10 ARC Rancher 升级若 reboot A100 (reboot 前须 detach p-79, KubeVirt #17417, 见 [[reference_condenser_a100_infra]])。Pass-1 fire 已停 (本次主动 kill), 重启后需重新 launch。
