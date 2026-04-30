# 学长 sync 草稿（2026-04-30）

**目的**: 简略 sync 进展 + 约会议讨论细节
**长度**: ~250 字

---

## 学长好，

想跟您简略同步一下，具体细节希望能开个会聊一下。

**1. Phantom-SoM 主线**：finding 出来了，4-fold drop-in property（cost ≈ DOM / latency ~50% / signal AUROC ≥ baseline / drop-one oracle 1.7-3.3pp）。Section 1-3 paper 草稿已写完，evidence 95% ready。

**2. VWA framework bugs 发现**：今天系统 audit 发现了 37 个 scaffold-level bugs（5-tier audit + 文献支撑 / Gemini DR 综述对照），其中最严重的 dispatch bug 让 94.4% 的失败 click 落在错误 DOM target 上。今天已经修完（Phase A 4-cluster patch），pilot 验证 PASS。这个 bug 影响 absolute SR 数字，但不影响 cross-mode 比较（symmetric contamination + Vision counter-evidence），所以 Phantom finding 仍 valid。这本身可能也是一个值得披露的副 contribution。

**3. 资源障碍**：UCL Myriad 申请下来了我也试着用了，但 UCL 防火墙 drop Tailscale CGNAT 段，**Myriad 不能 reach 我家里 quark Windows 的 VWA docker**，物理级 blocked，没法绕过（详见 `docs/reference/MYRIAD_SMOKE_REPORT.md`）。同时 DGX GPU 争抢非常严重，B1 baseline 跑 234 ep 需要 20+ 小时，Phase A 14-cell 全跑需要 150-200h wallclock。

**4. 提议**：调研 RunPod 4090 dedicated 只要 $0.6/h，14-cell 全跑预计 ~$150-200 总预算。想申请课题经费走 RunPod。

**问您几件事**：
- RunPod 经费可不可以走？走什么流程？
- Phantom finding 方向您觉得 OK 吗？
- VWA bug 这一发现要不要单独成文？
- **paper-strategy 4 个 framing 决定** (我自己 audit 出来的, 想听您意见):
  1. **Early-stop bias on micro metrics**: agent cycle-detect 早停 truncate trajectory, 让 micro 2a (URL Jaccard) / 2b (target hit rate) / 2c (keyword repeat) 三个指标 cross-mode 不可比 (短 trajectory 分母小). 选 A 关掉早停 (cost +$1300) / B 算 length-normalized / C demote 这 3 个到 secondary, 主用 first-divergence (uncensored) ?
  2. **B0 pre/post Phase A 数据 sampling 不对称**: Phase A 之前 B0 用 T=0.1 stochastic, 之后 T=0 greedy. archived data + 14-cell re-run 是不同 sampling regime. 直接合 vs 只用 post-Phase-A 重跑 数据?
  3. **Single seed=42, 没 replication study**: SR delta 数字没有 across-seed variance estimate. bootstrap CI 是 task-level binomial, 不 capture sampling variance. 加 N=3 replication (3× cost) 还是 disclose limitation?
  4. **Cross-site SR 不直接可比**: cls/red/shop 任务池不同, agent capability 不同. Section 5 已经 site-modulated framing, 但是否需要更明确 "我们不 claim cross-site dominance"?

详 `docs/reference/PAPER_STRATEGY_OPEN_QUESTIONS.md` (我整理的 open audit)

具体细节希望能开个会聊聊（30-45 分钟应该够），您方便的时候定个时间。

详细文档我都整理在 repo 里了：
- `docs/reference/VWA_FRAMEWORK_BUGS_AND_PHASE_A_FIXES.md`
- `docs/reference/MYRIAD_SMOKE_REPORT.md`
- `docs/checkpoints/paper_planning.md`

谢谢学长！

—— 嘉名
