#!/usr/bin/env bash
# 单一真相源 — mechanistic canonical sweep 的 24 个 cell 定义。
#
# WHY 它被抽出来: 这份清单现在有两个消费者 —
#   * scripts/queues/queue_mechanistic_canonical.sh   (DGX 串行 sweep)
#   * scripts/queues/sbatch_mechanistic_b1966_rerun.sh (Sparks Slurm array, 2 GPU)
# 复制一份到 Slurm 脚本里, 就等于给自己埋一个「两边参数悄悄分叉」的坑 ——
# 而 B-1966 正是同一个形状: 契约在两处各写一遍, 其中一处漂了, 结果看起来完全正常。
# 所以这里不复制, 两边一律 source 本文件。
#
# 格式: "<cell 名>|<site>|<CLS|RED 子集键>|<传给 run_stage2b_continuation_pilot.py 的额外参数>"
# 顺序有意义: P1 主效应在前, P4 探索臂最后 (最先可弃)。
# shellcheck disable=SC2034

CELLS=(
  # ── P1 main effect ────────────────────────────────────────────────────────
  "p1_fwd_strong_cls|classifieds|CLS|--n-tasks 24 --tier strong"
  "p1_fwd_strong_red|reddit|RED|--n-tasks 24 --tier strong"
  "p1_rev_reverse_cls|classifieds|CLS|--n-tasks 15 --reverse"
  "p1_rev_reverse_red|reddit|RED|--n-tasks 15 --reverse --tier reverse"
  # ── P2 effect: the phantom_som → phantom_text pair ────────────────────────
  "p2_psom_ptext_cls|classifieds|CLS|--n-tasks 24 --source-mode phantom_som --target-mode phantom_text"
  "p2_psom_ptext_red|reddit|RED|--n-tasks 24 --source-mode phantom_som --target-mode phantom_text"
  # ── its content control (the strong one, codex 2026-05-12) ────────────────
  "p2_taskshuf_cls|classifieds|CLS|--n-tasks 24 --source-mode phantom_som --target-mode phantom_text --task-shuffle --task-shuffle-seed 42"
  "p2_taskshuf_red|reddit|RED|--n-tasks 24 --source-mode phantom_som --target-mode phantom_text --task-shuffle --task-shuffle-seed 42"
  # ── P1's missing control: random-inject against the already-finished main ──
  #    effect arms. Promoted out of P5 (2026-07-30) — P1 is the headline claim
  #    and this is the only control it still lacks, so it comes before any new
  #    effect arm. Nothing else closes an already-computed claim this cheaply.
  "p5_rand_cls|classifieds|CLS|--n-tasks 24 --tier strong --random-inject --random-seed 42"
  "p5_rand_red|reddit|RED|--n-tasks 24 --random-inject --random-seed 42"
  # ── remaining controls for the P2 pair: direction, then random ────────────
  "p5_psom_ptext_rev_cls|classifieds|CLS|--n-tasks 24 --source-mode phantom_som --target-mode phantom_text --reverse"
  "p5_psom_ptext_rev_red|reddit|RED|--n-tasks 24 --source-mode phantom_som --target-mode phantom_text --reverse"
  "p5_psom_ptext_rand_cls|classifieds|CLS|--n-tasks 24 --source-mode phantom_som --target-mode phantom_text --random-inject --random-seed 42"
  "p5_psom_ptext_rand_red|reddit|RED|--n-tasks 24 --source-mode phantom_som --target-mode phantom_text --random-inject --random-seed 42"
  # ── P3 selection-bias 2x2 cross-subsets (robustness on P1; all 4 or none) ──
  "p3_fwd_revtier_cls|classifieds|CLS|--n-tasks 15 --tier reverse"
  "p3_rev_strongtier_cls|classifieds|CLS|--n-tasks 24 --reverse --tier strong"
  "p3_fwd_revtier_red|reddit|RED|--n-tasks 15 --tier reverse"
  "p3_rev_strongtier_red|reddit|RED|--n-tasks 24 --reverse --tier strong"
  # ── P4 cross-mode spectrum (som → X) — exploratory, safest to lose, last ───
  "p4_som_dom_cls|classifieds|CLS|--n-tasks 24 --target-mode dom"
  "p4_som_dom_red|reddit|RED|--n-tasks 24 --target-mode dom"
  "p4_som_ptext_cls|classifieds|CLS|--n-tasks 24 --target-mode phantom_text"
  "p4_som_ptext_red|reddit|RED|--n-tasks 24 --target-mode phantom_text"
  "p4_som_pprompt_cls|classifieds|CLS|--n-tasks 24 --target-mode phantom_prompt"
  "p4_som_pprompt_red|reddit|RED|--n-tasks 24 --target-mode phantom_prompt"
)
