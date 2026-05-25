---
type: audit
ref: A03-fu
title: prereg L103-111 honesty-surface sync
status: deferred
priority: P2
effort: 5 min
phase: paper-finalize
blocker: ''
---

# A03-fu · prereg L103-111 honesty-surface sync

degenerate-SE-floor 段仍是 pre-B-1003 "SE = 0 exactly" 措辞 + 指针指
`aggregate_phase1_prereg_gate.py:185-187`, 与 L98/L718 锁定的 0.68pp Agresti-Coull
threshold 矛盾。Fix: "SE = 0 exactly" → "SE < 0.68pp threshold" + repoint canonical
`aggregate_phase1_full_prereg_decision`。纯措辞 sync, 0.68 estimand 值不变 (已 recorded
AMENDMENT_03 §3 + tag prereg-amendment-03-implementation-alignment-20260524)。与 R2-P2-10-C 同批。
