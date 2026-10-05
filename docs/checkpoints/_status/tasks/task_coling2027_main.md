---
type: task
status: active
priority: P0
horizon: now
order: 3
blocker: "A100 不可达 (UCL VPN 未连 + condenser 证书 2026-09-28 已过期); 09-22 之后 A100 上新写的结果本地没有"
eta: "**ARR submission 2026-10-12** (AoE) → commit COLING **2026-12-23** → notif 2027-02-10。已核 2027.coling-iccl.org 2026-10-06"
detail: docs/checkpoints/paper_planning.md
created: 2026-08-08
updated: 2026-10-06
---

# COLING 2027 — 完整版目标（2026-10-06 由 NAACL 2027 改投）

**2026-10-06 user 定: 目标会议由 NAACL 2027 改为 COLING 2027, archival, 是唯一的主会目标。**
本文件原名 `task_naacl2027_main.md`; 下文 08-08 ~ 09-09 各段保留原貌, 其中的「NAACL 稿」一律读作「COLING 稿」。
8 月 REALM 表单勾的「Plan to submit to ACL ARR 2026 August」**实际没投** (user 2026-10-06 确认) ⇒ 手上没有可直接 commit 的 ARR review。

## 硬事实 (已核 `https://2027.coling-iccl.org/`, 2026-10-06)

| | |
|---|---|
| **ARR submission** | **2026-10-12** (Mon), 23:59 AoE |
| commitment (meta-review 后) | **2026-12-23** |
| 录取通知 | 2027-02-10 |
| 会议 | 2027-05-09 ~ 05-14, **澳门**（线上部分 05-06~07） |

**改投不改 ARR 周期**: COLING 2027 是第一届走 ACL Rolling Review 的 COLING, 收的正是 10 月 ARR 周期 ——
和原 NAACL 计划**同一个 10-12 截稿**。改的只是 12-23 commit 到哪。
⚠️ 页数上限 / 模板官网首页未列, 投前去 CFP 页核, ⛔ 别按 NAACL 的记忆写。

## 已到手的外部反馈（两个 workshop 均为 non-archival poster, 不占投稿权）

- **REALM @ EMNLP 2026** #192 接受 (09-08), 意见 → `_status/issues/issue_realm_reviews_2026-09-09.md`
- **VLM4RWD @ NeurIPS 2026** #18 Accept (Poster) (09-29), 意见 → `_status/issues/issue_vlm4rwd_reviews_2026-09-29.md`
  - 两边交集仍是: rerun control 是最强贡献; 「路由学不到」的**解释**(label supply) 被认为还没立住;
    rerun band 覆盖不足; 只比「两轴同时胜过固定策略」不够, 要看 SR–cost 折中本身

## 真实时间窗口 ⚠️（2026-10-06 重算）

```
10-06 今天 ──6天── 10-12 ARR 截稿 ──~10周── 12-23 COLING commit ──7周── 02-10 notif
```

- **只剩 6 天**: 10-12 前能进稿的只有**已有数据 + 0-compute 分析**; 新 GPU 实验赶不上。
- A100 不可达 (2026-10-06 实测): 跳板机 `ssh.condenser.arc.ucl.ac.uk` 连接超时 (UCL VPN 未连), 证书 09-28 已过期。
  A100 上本项目数据 09-22~23 全量拉到 `E:100-condenser-backup\`; 但 local replicate chain (09-15 发车, 预计 ~09-25 收尾)
  **在拉取之后还在写** —— `workspace` 远端比本地多 3,978 个文件, 很可能就是这批。要用这批 replicate 必须先恢复 A100 访问。

## 要跨过去的是「证据强度」, 不是 polish

当前 REALM 稿 = 一篇很完整的 MSc measurement + routing study。workshop 靠清楚的问题 + 扎实实验 + 有意思的结果就能成立; COLING reviewer 会继续追下去。**七个最可能的攻击面**（学长/user 2026-08-08 列）:

- [ ] 1. routing 的**泛化**到底怎么样
- [ ] 2. 是否**跨 site / benchmark** — 手上有 WA pilot + shop Phase 1b (pre-fix) 可用
- [ ] 3. **baseline 是否足够强**
- [ ] 4. router 是否真正 outperform **简单 heuristic** — ⚠️ 这条项目内已有硬结论: `§387.16.4` 的两道控制 (always-cheapest 固定策略 + label-shuffle 零分布) 显示**路由的两半都失败且败因不同**; NAACL 稿必须正面处理, 不能绕
- [ ] 5. **cost-accuracy trade-off 是否稳定**
- [ ] 6. DOM / SoM / Vision 的观察能否形成**更一般化的结论**
- [ ] 7. 近乎完美的 **AUROC** 是 task 易区分, 还是 **leakage / construction artifact**

### 第 7 条: 已拆过的雷, 风险在「只报一半」

台账里这条已经被自己诊断并裁定过了 —— **结论正是 artifact**:
- `§111.2` Stage-1 linear probe 三个 setup 全 `L1+ AUROC=1.0`, 裁定 **trivial**: 根因是 probe 在 last input token position **永远 trivially 编码 input 差异**（text 内容/长度/image tokens 本就不同）⇒ **linear probe 对该 contrastive setup 是 wrong tool**, mirage signature 必须用 patching (causal) 测
- `§127.1` 另一处 AUROC 1.0 已标 **in-sample、非 held-out**
- `§394` **RETRACTED** router 的 "AUROC 0.65-0.72 in 5/6 cells" 叙述: 第 6 格 red·B2 是 **0.483（低于随机）**, 而它偏偏是**唯一显著的那格**。替代表述 = 「全局判别 (AUROC) 与尾部可用性是两个性质, 本数据上二者解耦; base SR 2-27% 的 regime 里 AUROC 高既不必要也不充分」

⇒ 风险不是"被 reviewer 发现", 而是**稿子里只写好看的那一半**。写作时逐条带上 caveat，反而是加分项（negative finding 是 ACL 系列明确接受的类型）。
⚠️ 另注: mechanism 线 (§5) 自 2026-05-14 起 shelved。若 NAACL 稿不含 mechanism, 第 7 条的 probe 部分不适用; 若含, **必须带上"linear probe 是 wrong tool"的说明**。

## 输入与顺序

1. **REALM 审稿意见 (2026-09-07, user 2026-08-19 再更正; 08-21 已作废)** — 免费的一轮 top-tier 反馈, **落在毕设交付之前 11 天**（旧文档写"之后", 已随日期一并更正）。决定「按原框架投还是重构」的主要依据, **等它到手再定稿件形态**。
2. **REALM 稿本体** (#192, 正文 8 页) + 四步论证结构
3. **毕设全稿 (09-05)** — 为 rubric 补的文献图谱 / benchmark EDA / 形式化公式, 部分可反哺会议稿 appendix

## 与毕设的关系

**不是同一份东西**。毕设 = problem-first / concept-first / 文献综述与 EDA 齐备的长文; NAACL 稿 = 8-9 页单一论证线。**共享数据与图, 不共享结构。**

## 为什么这不是瞎抬目标

不是"没有论文然后幻想冲 NAACL", 而是已有 8 页主文 + 大量 appendix + 完整实验 pipeline。接下来两个月针对上面七条补实验 + 重新 framing, **NAACL 是合理的 stretch target**。若成 (MSc 一作、从 dissertation 长出), CV 信号与"硕士有篇 workshop"不是一个量级。

## 2026-09-09 晚 · router 重想之后的稿件形态（笔记 §505）

七个攻击面在今晚之后的状态：1 泛化 → §505.19（难度跨 backbone/site 迁，mode 契合度不迁）；2 跨 site/benchmark → 11 格全用，但 6/11 无 band（fire 清单 #1/#2）；3 baseline 强度 → B5 (GPT-5.6) 在册，§505.10「更强不反转」；4 router vs heuristic → §505.18 正反向规则 + §387.16.4；5 cost-accuracy 稳定性 → §505.22 前沿两轴一致；6 更一般化结论 → 预算路由 6/6 mode；7 AUROC artifact → §505.3 增量 AUROC ≈ 0 与 §457 的区别写清。

**主张改写**（候选题）：*Route the budget, not the representation: what a web-agent router can learn.*
- 可学的是 task 难度（稳定、可迁），不可学的是 task×mode 契合（交互/噪声 0.12–0.34，标签 70–80% 是硬币，step 0 不可见，更强 backbone 不反转）。
- 十一种 router 构造收敛在 +1–2pp over random；难度支撑的杠杆是预算：省 41% 且过重跑（14/18）。
- 实用附带：选臂 pilot 样本量（臂差在 band 内的格跑满也选不出）；跨臂一致性 verifier（温和）。
**风险**：early-stop/预算的先例文献未核（写作前必做）；6/11 格无 band；learned 对 fixed 的 1.4pp 是 steady-state；全离线（variant D 点火可部分补）。
