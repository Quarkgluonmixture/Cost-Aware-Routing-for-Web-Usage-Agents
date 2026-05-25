---
type: audit
ref: AMENDMENT_04-fu
title: analysis-alignment 收尾 (3 项)
status: deferred
priority: P2
effort: (a) paper-finalize · (b) 随 fire · (c) data-land 后
phase: paper-finalize
blocker: ''
---

# AMENDMENT_04-fu · analysis-alignment 收尾 (3 项, 均 deferred 非阻塞)

AMENDMENT_04 (B-1849~1857) tag + OSF kv9sf 已 witness; 剩:
- **(a) prereg prose sync** — `AMENDMENT_04 §3` supersession table 记 prereg §4 latency row →
  scaffold-adjusted (B-1854) + §H10 LR `class_weight balanced→None` (B-995/P1-7), 与 A03-fu +
  R2-P2-10-C 同批 fold at paper-finalize (DOI anchor 不改, estimand 值不变)
- **(b) stale artifacts regenerate** (P1-5) — fire 后 `make analysis` 重生 `h10_pareto_verdict.json`
  (旧 h0_rejected schema) + `cross_site_*` (缺 baseline/billed/basis 列), 可加 CI reject-old-schema
- **(c) /stress retro** — data land 后对照实际 θ_FE/H10 verdict 回看 B-1849~1857 哪些真拦 paper-grade 错
