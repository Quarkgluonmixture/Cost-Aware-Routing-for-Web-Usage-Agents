---
type: issue
category: decision
status: active
priority: high
action: NOTE_06 两轨制已激活 → B0 pprompt land 即 k=5 verdict; B3=MiMo 8月启动; router-baseline efficiency 对比 (offline) 排队; 作者名单+提名待学长回复
created: 2026-07-15
updated: 2026-07-15
---

# Advisor sync 2026-07-15 — 两轨制签字 + B3 路线 + router-baseline 新指令

周会提前一天举行 (原计划 07-16)。学长总体评价 "很 nice"。详见 实验笔记 §373 + paper_planning §19 decision log。

## Decisions

1. **PROTOCOL_NOTE_06 两轨制 — 口头 APPROVE** ✅
   - k=5 提交基线 (五格固定集) + B2_red 提交前齐则无条件升 k=6
   - 激活变更集 commit `765d31a` + witness tag `protocol-note-06-k5-early-verdict-signed-20260715`
   - 四项披露已落 (§4 句 / §8 段 / prereg log 行 / OSF 知会排队)
   - ⚠️ 待办: 学长一行书面确认 (chat) 归档 + §6 决定块补全名; OSF 知会发布后回填 URL

2. **B3 路线 — MiMo-VL 先行, 之后扩展其他模型** ✅
   - 8 月 (投稿后) 启动; MiMo-VL-7B = cross-family floor 攻击的回应
   - "之后是其他的模型" = B3 不是单点, 是 post-submission 扩展序列的第一步

3. **新指令: 找 router baseline (prior work) 对比 efficiency** 🆕
   - 用既有 offline replay 基座实现文献 router 算法作 baseline (RouteLLM-style / FrugalGPT cascade / kNN 等), 同 OOF 协议、同 cost-SR 平面, OFFLINE/NON-GATE
   - 顺带采纳 router 文献的 efficiency 指标惯例 (如 PGR/APGR 类 "每美元收回多少 oracle gap")
   - 定位: §6/supplement 增强 + 审稿人 "为何不比较既有 router" 的预防; deadline 前 offline 可行, live 版 paper-2

## Open (user 行动)

- [ ] 作者名单 + reciprocal reviewer 提名: 已给学长发问题清单 (资格判定 ≥2 一作 / ≥5 合作 archival; OpenReview profile; COI) — Jul 21 AoE 冻结
- [ ] 学长一行书面确认两轨制 (归档进 NOTE_06 §6)
- [ ] OSF 知会发布 (`deliverables/osf_notice_protocol_note_06_2026-07-15.md`)
- [ ] D7 story 版章节交付 (读 §2.0/§2.8 后拍板哪版)
