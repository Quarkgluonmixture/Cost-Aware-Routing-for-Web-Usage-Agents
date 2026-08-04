#!/usr/bin/env bash
# Canonical mechanistic re-run — priority-ordered, resumable, deadline-bounded.
#
# WHY THIS EXISTS (2026-07-23, 笔记 §384):
# The §5 mechanism cells were all computed off the 2026-04 pre-fix archive
# (B1_phantom_som_cls_20260428 / B1_3mode_reddit_20260413) on a mix of Myriad
# V100 + A100 — while the paper body uses the 2026-06 canonical A100 fire. Two
# data sources in one paper, plus §300.2's cross-GPU greedy nondeterminism
# (±3-5pp). This re-runs every cell against canonical artifacts on one GPU.
#
# ORDERING: cells are ordered by WHICH CLAIM THEY SUPPORT, not by stage number,
# because the run is deadline-bounded and will be truncated mid-list. Whatever
# finishes is a self-contained evidence set; the tail becomes future work.
#   P1 main effect (som→phantom_som, both sites, both directions)
#   P2 content-specificity (task-shuffle — the strong control, codex 2026-05-12)
#   P3 selection-bias 2x2 (direction × tier cross-subsets)
#   P4 cross-mode spectrum (som→dom / →phantom_text / →phantom_prompt)
#   P5 weaker controls (random-inject) + remaining phantom-pair arms
#
# SPEC (calibrated 2026-07-23 on canonical strong tier, 3 tasks × both lengths):
#   --max-new-tokens 50 — 15 preserves peak LOCATION but compresses amplitude
#   2.3-11.7x and erases small effects (task 180: overlap→src 0.04 → 0.00) and
#   misplaces secondary-metric peaks (L15 → L12). Only buys 1.5x speed because
#   generation is the minority of per-layer cost (hooks + re-forward dominate).
#   --n-tasks 24 strong / 15 reverse — matches historical cell sizes. Canonical
#   strong tier is larger (cls 57 / red 41 vs 24 / 47) and manifest-sorted by
#   composite desc, so 24 = top-24 by score. NOTE: composite does NOT predict
#   patching effect (calibration task 70 scored 9.20, showed zero effect), so
#   expect ~1/3 of tasks to be null — that heterogeneity is itself the §300.3
#   finding, not a defect.
#
# Usage:
#   bash scripts/queues/queue_mechanistic_canonical.sh            # run
#   DEADLINE=2026-08-01 bash .../queue_mechanistic_canonical.sh   # custom stop
#   DRY_RUN=1 bash .../queue_mechanistic_canonical.sh             # list only
# Resumable: a cell whose output dir already has pilot_summary.md is skipped,
# so kill/restart at any point takes over where it left off.
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1

OUT_ROOT="$REPO/results/mechanistic/canonical"
SUB_CLS="$REPO/results/mechanistic/archive_subset_b1_cls_canonical"
SUB_RED="$REPO/results/mechanistic/archive_subset_b1_red_canonical"
DEADLINE="${DEADLINE:-2026-08-01}"
DRY_RUN="${DRY_RUN:-0}"
NTFY_TOPIC="${NTFY_TOPIC:-p79-exp-dgx-spark}"
LOGDIR="$REPO/logs/mechanistic_canonical"
SCRIPT=scripts/mechanistic/run_stage2b_continuation_pilot.py

export PYTORCH_NVML_BASED_CUDA_CHECK=1 CUDA_MPS_PIPE_DIRECTORY="" CUDA_MPS_LOG_DIRECTORY=""
mkdir -p "$OUT_ROOT" "$LOGDIR"

DEADLINE_TS=$(date -d "$DEADLINE" +%s 2>/dev/null) || { echo "bad DEADLINE: $DEADLINE"; exit 1; }

for d in "$SUB_CLS" "$SUB_RED"; do
  [ -f "$d/manifest.json" ] || { echo "FATAL: missing canonical subset $d — run curate chain first"; exit 1; }
done

# name | site | subset | extra args
# (source/target default som → phantom_som; --tier defaults from --reverse)
CELLS=(
  # ── P1 main effect ────────────────────────────────────────────────────────
  "p1_fwd_strong_cls|classifieds|CLS|--n-tasks 24 --tier strong"
  "p1_fwd_strong_red|reddit|RED|--n-tasks 24 --tier strong"
  "p1_rev_reverse_cls|classifieds|CLS|--n-tasks 15 --reverse"
  "p1_rev_reverse_red|reddit|RED|--n-tasks 15 --reverse --tier reverse"
  # ── P2 content-specificity (phantom_som → phantom_text pair + task-shuffle) ─
  "p2_psom_ptext_cls|classifieds|CLS|--n-tasks 24 --source-mode phantom_som --target-mode phantom_text"
  "p2_psom_ptext_red|reddit|RED|--n-tasks 24 --source-mode phantom_som --target-mode phantom_text"
  "p2_taskshuf_cls|classifieds|CLS|--n-tasks 24 --source-mode phantom_som --target-mode phantom_text --task-shuffle --task-shuffle-seed 42"
  "p2_taskshuf_red|reddit|RED|--n-tasks 24 --source-mode phantom_som --target-mode phantom_text --task-shuffle --task-shuffle-seed 42"
  # ── P3 selection-bias 2x2 cross-subsets ───────────────────────────────────
  "p3_fwd_revtier_cls|classifieds|CLS|--n-tasks 15 --tier reverse"
  "p3_rev_strongtier_cls|classifieds|CLS|--n-tasks 24 --reverse --tier strong"
  "p3_fwd_revtier_red|reddit|RED|--n-tasks 15 --tier reverse"
  "p3_rev_strongtier_red|reddit|RED|--n-tasks 24 --reverse --tier strong"
  # ── P4 cross-mode spectrum (som → X) ──────────────────────────────────────
  "p4_som_dom_cls|classifieds|CLS|--n-tasks 24 --target-mode dom"
  "p4_som_dom_red|reddit|RED|--n-tasks 24 --target-mode dom"
  "p4_som_ptext_cls|classifieds|CLS|--n-tasks 24 --target-mode phantom_text"
  "p4_som_ptext_red|reddit|RED|--n-tasks 24 --target-mode phantom_text"
  "p4_som_pprompt_cls|classifieds|CLS|--n-tasks 24 --target-mode phantom_prompt"
  "p4_som_pprompt_red|reddit|RED|--n-tasks 24 --target-mode phantom_prompt"
  # ── P5 weaker controls + remaining phantom-pair arms ──────────────────────
  "p5_rand_cls|classifieds|CLS|--n-tasks 24 --tier strong --random-inject --random-seed 42"
  "p5_rand_red|reddit|RED|--n-tasks 24 --random-inject --random-seed 42"
  "p5_psom_ptext_rand_cls|classifieds|CLS|--n-tasks 24 --source-mode phantom_som --target-mode phantom_text --random-inject --random-seed 42"
  "p5_psom_ptext_rand_red|reddit|RED|--n-tasks 24 --source-mode phantom_som --target-mode phantom_text --random-inject --random-seed 42"
  "p5_psom_ptext_rev_cls|classifieds|CLS|--n-tasks 24 --source-mode phantom_som --target-mode phantom_text --reverse"
  "p5_psom_ptext_rev_red|reddit|RED|--n-tasks 24 --source-mode phantom_som --target-mode phantom_text --reverse"
)

notify () {
  curl -s -H "Title: $1" -d "$2" "https://ntfy.sh/${NTFY_TOPIC}" > /dev/null 2>&1 || true
}

TOTAL=${#CELLS[@]}
IDX=0; RAN=0; SKIPPED=0
# Clear any stale marker from a previous sweep so this run's done-monitor can
# only fire on THIS run's completion (resume runs re-touch it at the end).
[ "$DRY_RUN" = "1" ] || rm -f "$OUT_ROOT/.SWEEP_DONE"
echo "[$(date '+%F %H:%M:%S')] mechanistic canonical sweep — $TOTAL cells, deadline $DEADLINE"

for entry in "${CELLS[@]}"; do
  IDX=$((IDX+1))
  NAME="${entry%%|*}"; rest="${entry#*|}"
  SITE="${rest%%|*}"; rest="${rest#*|}"
  SUBKEY="${rest%%|*}"; EXTRA="${rest#*|}"
  [ "$SUBKEY" = "CLS" ] && SUBSET="$SUB_CLS" || SUBSET="$SUB_RED"
  OUT="$OUT_ROOT/$NAME"

  if [ -f "$OUT/pilot_summary.md" ]; then
    echo "[$(date '+%H:%M:%S')] ($IDX/$TOTAL) $NAME — already done, skip"
    SKIPPED=$((SKIPPED+1)); continue
  fi

  NOW=$(date +%s)
  if [ "$NOW" -ge "$DEADLINE_TS" ]; then
    echo "[$(date '+%H:%M:%S')] DEADLINE $DEADLINE reached — stopping before $NAME"
    notify "P79 mechanistic sweep 到点截断" "$RAN 个 cell 完成 ($SKIPPED skipped)，停在 $NAME ($IDX/$TOTAL)"
    break
  fi

  if [ "$DRY_RUN" = "1" ]; then
    echo "  ($IDX/$TOTAL) $NAME [$SITE] $EXTRA"
    continue
  fi

  mkdir -p "$OUT"
  echo "[$(date '+%H:%M:%S')] ($IDX/$TOTAL) $NAME [$SITE] launching — $EXTRA"
  START=$(date +%s)
  # shellcheck disable=SC2086
  .venv/bin/python3 "$SCRIPT" \
    --site "$SITE" --step 2 --max-new-tokens 50 \
    --output-dir "$OUT" --archived-run-dir "$SUBSET" \
    $EXTRA > "$LOGDIR/${NAME}.log" 2>&1
  RC=$?
  MIN=$(( ($(date +%s) - START) / 60 ))
  if [ $RC -eq 0 ]; then
    RAN=$((RAN+1))
    echo "[$(date '+%H:%M:%S')] ($IDX/$TOTAL) $NAME DONE in ${MIN}min"
  else
    echo "[$(date '+%H:%M:%S')] ($IDX/$TOTAL) $NAME FAILED rc=$RC after ${MIN}min — see $LOGDIR/${NAME}.log"
    notify "P79 mechanistic cell FAILED" "$NAME rc=$RC — sweep 继续下一个"
  fi
done

# DRY_RUN must never touch the completion marker or notify: a dry run that
# leaves .SWEEP_DONE behind makes the real sweep indistinguishable from a
# finished one (and fires a false "DONE" push). Empirically bit us 2026-07-23:
# the dry run's marker tripped the real run's done-monitor 60s after launch.
if [ "$DRY_RUN" = "1" ]; then
  echo "[$(date '+%F %H:%M:%S')] dry run finished — $TOTAL cells listed, nothing executed"
  exit 0
fi

echo "[$(date '+%F %H:%M:%S')] sweep finished — $RAN ran, $SKIPPED skipped, of $TOTAL"
touch "$OUT_ROOT/.SWEEP_DONE"
notify "P79 mechanistic sweep DONE" "$RAN 个 cell 新跑完 ($SKIPPED 已存在) / 共 $TOTAL → results/mechanistic/canonical/"
