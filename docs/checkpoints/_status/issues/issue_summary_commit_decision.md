---
type: issue
category: decision
status: pending-evaluation
priority: low
action: 评估 git LFS / 直 git for paper-grade archive
---

# Tier A summary commit decision

是否 commit `condition_summary_v2.json` + `run_meta.json` 入 git LFS / 直 git？
size: 10 cond × ~50KB = ~500KB. 好处: paper repro reviewer 不需 hub access. 坏处: 实验未冻结前每次 rederive 改动多。
