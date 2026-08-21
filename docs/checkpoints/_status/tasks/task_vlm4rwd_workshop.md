---
type: task
status: done
priority: P2
horizon: now
order: 45
blocker: ""
eta: "**已提交 2026-08-21**。notif **2026-09-29**, camera-ready 2026-10 月。非归档 ⇒ 不占 NAACL 2027 ARR (10-12) 投稿权。下一个动作在 09-29, 此前无事。"
detail: docs/checkpoints/实验笔记.md §473
created: 2026-08-21
updated: 2026-08-21
---

# VLM4RWD @ NeurIPS 2026 workshop 投稿 (non-archival) — 已提交

**Venue**: [Grounded and Faithful Vision-Language Models for Real-World Deployment](https://vlm4rwd.github.io/),
NeurIPS 2026 Workshop, Sydney, Dec 11 2026 · OpenReview `NeurIPS.cc/2026/Workshop/VLM4RWD`

**内容** = REALM 在审稿 (Overleaf `6a59017b04233a73ed5ec570` @ `a456bff`) 的 NeurIPS 模板移植
+ 一段 workshop 主题对位。产物 `deliverables/vlm4rwd/`。

## 关键时间

| | |
|---|---|
| 提交 | ✅ 2026-08-21（deadline 08-31 05:00, 以 OpenReview 表单为准 —— 官网 CFP 写 08-30 少一天） |
| Notification | **2026-09-29** |
| Camera-ready | 2026-10 月 |
| Workshop | 2026-12-11 |

## 与其它三条线的关系

- **REALM #192**（notif 09-07 / camera-ready 09-14）—— 同为非归档, CFP 明写已投递工作可投
  （只是不参评 workshop 奖项）⇒ **不冲突**
- **毕设 09-05 硬截止** —— 本投稿不消耗毕设资源, 已完成
- **NAACL 2027 ARR 10-12** —— 非归档 ⇒ **不占投稿权**

⚠️ **本副本与 REALM 在审稿已分叉**: 多一段 workshop 对位; 且修了三处 REALM 那边**仍存在**的
原稿 bug（`tab02` 引 `tab:t04` 论证 non-separability 而 `tab05` 明写不是 / `paper-B` 源项目残留
/ Multiplicity 段落被归进 Limitations）。REALM camera-ready 09-14 是修那三处的窗口 —— **未决定**。

## 提交后如果要改（camera-ready 阶段）

- `\usepackage[dblblindworkshop, final]{neurips_2026}` 加 `final` ⇒ 页脚才显示 workshop 名,
  作者块解匿名（当前 review 模式下页脚硬编码主会字样, 是官方行为不是配置错）
- **禁用 `wrapfigure`**（会静默裁 caption, 笔记 §473.1）
- caption 完整性自查见 `deliverables/vlm4rwd/README.md`

## 已知风险（等 09-29 揭晓）

topic fit 是结构性的: 该 workshop 要 grounded/faithful VLM 的**方法**, 本稿是**方法学否定**。
gemini 3.1 Pro 独立判 reject / <15%。已把对位重心移到 contribution (i) 的 rerun-matched null
（对该 workshop 所有 grounding 方法通用）—— 能搭的最结实的桥, 再往上只能加实验。
非归档 + 判罚成本低 ⇒ 价值在曝光与反馈。台账 `CLAIM_UNVERIFIED §473.7`。
