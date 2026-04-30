# 学长 sync 草稿（2026-04-30）

**目的**: 简略 sync 进展 + 约会议讨论细节
**长度**: ~250 字

---

## 学长好，

想跟您简略同步一下，具体细节希望能开个会聊一下。

**1. Phantom-SoM 主线**：finding 出来了，4-fold drop-in property（cost ≈ DOM / latency ~50% / signal AUROC ≥ baseline / drop-one oracle 1.7-3.3pp）。Section 1-3 paper 草稿已写完，evidence 95% ready。

**2. VWA framework bugs 发现**：今天系统 audit 发现了 37 个 scaffold-level bugs（5-tier audit + 文献支撑 / Gemini DR 综述对照），其中最严重的 dispatch bug 让 94.4% 的失败 click 落在错误 DOM target 上。今天已经修完（Phase A 4-cluster patch），pilot 验证 PASS。这个 bug 影响 absolute SR 数字，但不影响 cross-mode 比较（symmetric contamination + Vision counter-evidence），所以 Phantom finding 仍 valid。这本身可能也是一个值得披露的副 contribution。

**3. 资源障碍**：UCL Myriad 申请下来了我也试着用了，但 UCL 防火墙 drop Tailscale CGNAT 段，**Myriad 不能 reach 我家里 quark Windows 的 VWA docker**，物理级 blocked，没法绕过（详见 `docs/reference/MYRIAD_SMOKE_REPORT.md`）。

DGX GPU 争抢的实测数据：B1 P-text cls 现在跑 26h 完成 198/234 ep，**~8 min/ep average**（peak 时段更慢）。对比 B0 proxy（无 GPU 争抢）~3.5 min/ep。即 **DGX shared 比 dedicated 资源慢 2-3× average，peak 时段 5-10×**。Phase A 14-cell × 234 ep 全跑：DGX shared 需 ~437h（~18 天 24/7 wallclock），独占 4090 估 ~87-145h（3-6 天）。

**4. 提议**：调研 RunPod 4090 dedicated 只要 $0.6/h。基于实测吞吐:
- 4090 dedicated 估 ~87-145 GPU hours × $0.6/h = **~$52-87 actual**
- + 30% buffer (crash/retry/idle): **~$70-115 reasonable estimate**
- 申请 **$200 budget** 留 head-room for additional probes (Q3 B0 multi-call extended verify ~$10, Tier 5 evaluator probe ~$20, P-prompt diamond shop ~$30, Section 5 ad-hoc query ~$20).

**Wallclock 影响是 deal-breaker**: paper data ready 时间从 ~3 周 (DGX) 缩到 ~1 周 (4090). paper writing + 学长 review 时间从此 unblocked.

**问您几件事**：
- RunPod 经费可不可以走？走什么流程？
- Phantom finding 方向您觉得 OK 吗？
- VWA bug 这一发现要不要单独成文？
- **paper-strategy 2 个 framing 决定** (我 audit 出来 + 自己 sanity check 后, 这两个真 ask 您):
  1. **Early-stop bias on micro metrics**: agent cycle-detect 早停 truncate trajectory, 让 micro 2a (URL Jaccard) / 2b (target hit rate) / 2c (keyword repeat) 三个指标 cross-mode 不可比 (短 trajectory 分母小). 选 A 关掉早停 (cost +$1300) / B 算 length-normalized / C demote 这 3 个到 secondary, 主用 first-divergence (uncensored)?
  2. **B0 pre/post Phase A 数据 sampling 不对称**: Phase A 之前 B0 用 T=0.1 stochastic, 之后 T=0 greedy. archived data + 14-cell re-run 是不同 sampling regime. 直接合 vs 只用 post-Phase-A 重跑 数据 (弃 archived)?

**🆕 实证 anchor for B0/B1 reproducibility 不对称** (我跑了 cheap probe 验证):
- Probe: 5 calls × T=0 + top_p=1.0 + seed=42 forwarded × 同 prompt × proxy API
- Result: 5/5 byte-level distinct outputs **但** 5/5 same action (`click element_id=5`)
- **Conclusion**: B0 token-level non-deterministic, decision-level convergent
- Section 4 disclosure paragraph drafted in `docs/analysis/cross_sites/probe_b37_api_determinism.md`
- Cost: $0.005, 给 paper 实证 anchor 而不是 "trust the proxy" claim

详 `docs/reference/PAPER_STRATEGY_OPEN_QUESTIONS.md` (audit 9 个 issues, 7 个我自决 disclose, 上面 2 个真 ask 您)

具体细节希望能开个会聊聊（30-45 分钟应该够），您方便的时候定个时间。

详细文档我都整理在 repo 里了：
- `docs/reference/VWA_FRAMEWORK_BUGS_AND_PHASE_A_FIXES.md`
- `docs/reference/MYRIAD_SMOKE_REPORT.md`
- `docs/checkpoints/paper_planning.md`

谢谢学长！

—— 嘉名
