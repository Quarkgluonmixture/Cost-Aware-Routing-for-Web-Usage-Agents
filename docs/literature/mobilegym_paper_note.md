# Paper Reading Note — *MobileGym: A Verifiable and Highly Parallel Simulation Platform for Mobile GUI Agent Research*

**Source:** Wu, Hao, Wang, Wu, Xiao, Li, Zhou, Ju, Liu, Fan, Zhang (中科院自动化所 + PKU + CUHK, Zhaoxiang Zhang 组), arXiv:2605.26114v2, 2026-05. Code: github.com/Purewhiter/mobilegym (Apache 2.0 / data CC BY-NC).
**Found via:** GitHub AI 每日精选 cron 2026-06-10; repo + arXiv id 已实测核验 (GitHub API ⭐586, arXiv API 命中)。
**Use case for our project:** **bug 研究 paper (cross-benchmark) 素材为主**; paper-1 仅 related-work 级 (eval-reliability 收敛趋势佐证)。不是 P79 可直接采用的实验平台 (mobile 域, 非 web)。

---

## 0. One-sentence summary

浏览器内 React 实现的"类 Android"手机模拟环境 (非模拟器/真机), 全环境状态 = 结构化 JSON (world data ⊕ runtime overlay ⊕ OS runtime), 由此得到三件 VWA 类 benchmark 结构性做不到的事: **确定性 state-diff judge**、**全环境意外副作用检测 (USE metric)**、**亚秒级 fork/reset 支撑 256 并行 online RL**。

## 1. 核心机制 (精确)

- **28 apps (12 daily + 16 system) / 416 参数化 task templates (256 test + 160 train, 严格不相交)**; 每 template 经 instruction variation + parameter sampling + environment configuration 三路实例化, >27k distinct instances。
- **Judge = 任务自带 Python check function 读 state diff**, 非 VLM、非字符串相似度; 亚毫秒出 verdict。
- **AnswerSheet protocol**: query 类任务的答案提交从 free-text 改为 GUI 内打字段类型的表单 + 类型化 matcher (exact / numeric tolerance / format / choice) — 显式针对 free-text 匹配的两类失效: false reject (等价表述) + false accept (CoT 泄漏 gold answer 被 substring 误判)。
- **Metrics**: SR (primary) / PR / **FC (false complete)** / **USE (unexpected side effects, 全环境 state diff 检出任务外状态变化)** / OT (overdue)。
- **难度分层 L1-L4 = post-hoc 8-model empirical calibration** (BBH 式), 显式排除 Qwen3-VL-4B 及其 fine-tune 防泄漏。
- 资源: ~400MB RAM / ~50MB disk / ~3s cold-start per instance; 256 并行单机 <10% CPU, 全 256-task 评测 ~6min。

## 2. 与 P79 直接相关的数字

| 数字 | 精确含义 | 引用注意 |
|---|---|---|
| **VLM judge 10.2% 误判** | 人工复核 118 条 signal-bucket 真机轨迹, Qwen3.6-Plus judge 错 12/118 (base 5/59 + trained 7/59); GPT-5.4 重判同样 12/118 (部分不同子集) — Appendix J | n=118, CI 宽; 且这是 **VLM-as-judge** 误差, ≠ 我们的**确定性 evaluator 实现 bug** FP 类 (ua_match N/A / string_match 等)。引用时不可错位: 两者是 judge 不可靠性的两个不同机制 |
| **Qwen3-VL-4B-Instruct SR 9.4±0.6%** (PR 20.1) | MobileGym-Bench 256 test, 4 trials | B1 同款模型的跨域 capability anchor: 4B generalist 在 mobile GUI 也是弱 agent, 支撑 B1 = low-capability robustness check 框架 |
| GRPO +12.8pt (256-task) / signal 子集真机 +40.7pt, 95.1% retention | 10 GRPO steps, 单节点 3×RTX Pro 6000 + 96 并行实例 | sim-to-real 自称 existence proof, 59-task 分层 signal 子集, 非全集 |

## 3. Mapping to P79

1. **Bug paper (primary)**: (a) AnswerSheet 动机段 = free-text 匹配 FP/FN 失效目录, 与我们 VWA string_match FP 类一一对应, 可作 cross-benchmark 旁证; (b) FC/USE/OT 诊断指标命名是现成的 taxonomy 参照; (c) MobileGym 本身是新鲜 audit 候选 (新 codebase, bug surface 在 task judge 函数与 FSM spec 层)。
2. **USE metric 概念可迁移**: 我们的 site-state contamination 关切 (cart/comment cross-pollution → reset hard rule) 是 ad-hoc 防御; MobileGym 把它做成了 principled 全环境 diff。**可落地 idea (暂存, 防 scope creep)**: A100 VWA docker 的 Postgres/MySQL 做 per-episode DB dump diff → web 域的 side-effect 诊断, 现有 benchmark 无此 signal。
3. **Paper-1 related work (secondary)**: 他们把 VWA 列为 web 域 verifiable environment; 反向引用 = "verdict reliability 已成 mobile 域 benchmark 的一阶设计轴" 收敛趋势佐证。仅当 paper-1 有 eval-FP 段落时才入 bib。
4. **Bibliography 可挖**: WebGym (2601.02439) / AutoWebWorld (2602.14296, FSM 合成无限 verifiable web env — 与 MobileGym declarative navigation FSM 同思路且在 web 域) / InfiniteWeb (2601.04126) / OpenApps (2511.20766) — 2026 web 域可验证环境合成线。**Sweep 已完成 2026-06-10 → 见 [[verifiable_web_envs_2026_sweep]]**。

## 4. Hostile-reader caveats (引用前自检)

- 10.2% 来自单一 bucket 118 条轨迹; 两个 judge 模型都错 12/118 但子集部分不同 — 数字稳健性中等, 作 motivation 可, 作精确常数不可。
- Leaderboard proprietary 模型基本单 run (API 成本), 仅 Gemini 3.1 Pro 估了 variance。
- 整个 interaction-fidelity 主张押在一个 59-task 真机 transfer study 上; surrogate app ≠ 真后端 (作者自己 Limitations 承认: 不建模 server-side 动态)。

## 5. Disposition

- **Parked against bug paper (primary) + paper-1 related-work (conditional)**。
- Action: bug paper 动笔时引 AnswerSheet 失效目录 + FC/USE taxonomy; 若做 cross-benchmark audit, MobileGym 入候选列表 (与 agisdk 并列)。
- DB-diff side-effect 诊断 idea 暂存本 note, 不进 Phase 1 scope。
- 未入 `paper.bib`。
