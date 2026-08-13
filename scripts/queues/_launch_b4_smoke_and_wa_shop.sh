#!/usr/bin/env bash
# _launch_b4_smoke_and_wa_shop.sh — runs AFTER the 2026-08-09 shopping chain finishes.
#
# Three steps, strictly serial (hard rule #3: one site chain per host):
#   1. B4 dom  smoke (1 task) — does tool_choice=required work on an Anthropic modelId
#                               through this hybrid proxy? 台账 §243 says the object form
#                               is silently swallowed and Anthropic-native {type:tool}
#                               returns 400, so this is NOT implied by the image-channel
#                               probe of 笔记 §456.1.
#   2. B4 som  smoke (1 task) — can it read a real 1280x720 SoM-ANNOTATED screenshot?
#                               §456.1 only proved the channel with a 224x224 colour block,
#                               and 台账 §100/§101 show SoM marks are destructive to OCR.
#   3. B1 x VWA-classifieds SoM replicate — a same-condition rerun of an already-run
#                               condition, to widen the rerun-floor coverage.
#
# STEP 3 WAS CHANGED on 2026-08-13 (笔记 §464). It used to be "WA-shop B0 dom", bought to
# make the winner-reversal moderator identifiable. Three independent zero-preset framings
# (codex gpt-5.6-sol / DeepSeek V4 Pro / Kimi K3, 笔记 §463) all reframed the paper around
# the rerun floor as the instrument, and NONE of the three kept winner-reversal as a main
# line -- so that $17 was buying an artifact of the old frame. Meanwhile all three name
# floor coverage as the methodological centre, and DeepSeek's single strongest objection is
# that the floor exists in ONE full cell ("a two-cell anecdote"). B1 is a local model, so
# the replicate costs $0. B1 x cls is picked over the B2 cells because B2 sits near the
# floor (1-8 successes per arm), where too few flips occur for a band to be measurable.
#
# WHY THIS FILE EXISTS instead of just editing the queue scripts in place: the chain was
# still running when the B4 work landed, and bash reads a long script incrementally --
# overwriting queue_baseline.sh mid-run can make the live shell read a misaligned offset.
# So the edited scripts were shipped as `*.sh.b4` and are moved into place HERE, i.e. only
# once nothing is executing them.
#
# FORCE_NEW=1 is exported at the top level on purpose: queue_chain.sh:394 reads
# `FORCE_NEW="${FORCE_NEW:-0}"` while a comment 20 lines above claims the master chain
# exports 1. B-1916 was exactly this, and for a smoke a resume-glob would silently attach
# to an existing run instead of minting a fresh one.

set -uo pipefail

REPO="/home/ubuntu/workspace/p79"
cd "$REPO" || exit 1
NTFY="${NTFY_TOPIC:-p79-exp-dgx-spark}"
LOG_TS="$(date -u +%Y%m%d_%H%M%S)"
export FORCE_NEW=1
export RESET_BEFORE=1

say() { echo "[b4-wa $(date -u '+%H:%M:%S')] $*"; }
push() { curl -s -m 20 -H "Title: $1" -d "$2" "https://ntfy.sh/${NTFY}" >/dev/null 2>&1 || true; }

# ---------- 0. refuse to run while anything is still executing ----------------------
if pgrep -f "queue_chain\.sh" >/dev/null || pgrep -f "run_experiment\.py" >/dev/null; then
  say "REFUSING: a chain or runner is still alive. This script must run only after the"
  say "           2026-08-09 shopping chain has finished (it moves queue_*.sh into place)."
  push "b4 launch REFUSED" "chain/runner still alive; nothing was changed"
  exit 3
fi

# ---------- 1. move the edited queue scripts into place -----------------------------
moved=0
for f in "$REPO"/scripts/queues/*.sh.b4; do
  [ -e "$f" ] || continue
  tgt="${f%.b4}"
  cp -p "$tgt" "${tgt}.pre_b4_$(date -u +%Y%m%d)" 2>/dev/null || true
  mv "$f" "$tgt"
  say "installed $(basename "$tgt") (backup: $(basename "$tgt").pre_b4_*)"
  moved=$((moved + 1))
done
say "queue scripts installed: ${moved}"

# B4 must be accepted by the whitelist now, or every step below dies at argument parsing.
if ! grep -q '"B4"' "$REPO/scripts/queues/queue_baseline.sh"; then
  say "FATAL: queue_baseline.sh still has no B4 in its whitelist."
  push "b4 launch FAILED" "queue_baseline.sh lacks B4 whitelist; nothing launched"
  exit 4
fi

# ---------- 2. the two smokes ------------------------------------------------------
for mode in dom som; do
  cfg="configs/exp_v2_B4_${mode}_classifieds_smoke.yaml"
  if [ ! -f "$cfg" ]; then
    say "FATAL: missing $cfg"; push "b4 smoke FAILED" "missing $cfg"; exit 5
  fi
  say "=== B4 ${mode} smoke (1 task) ==="
  SMOKE_CONFIG="$cfg" bash scripts/queues/queue_baseline.sh B4 "$mode" classifieds \
    > "logs/b4_${mode}_smoke_${LOG_TS}.log" 2>&1
  rc=$?
  say "B4 ${mode} smoke launcher rc=${rc}"
  # The launcher backgrounds a runner; wait for it rather than racing the next step.
  for _ in $(seq 1 120); do
    pgrep -f "run_experiment\.py.*B4_${mode}_classifieds_smoke" >/dev/null || break
    sleep 30
  done
  say "B4 ${mode} smoke runner finished"
done

# One line per smoke so the ntfy body is readable on a phone.
summary=""
for mode in dom som; do
  d=$(ls -dt "$REPO"/results/visualwebarena/phase1/B4_${mode}_classifieds_smoke_* 2>/dev/null | head -1)
  if [ -n "$d" ]; then
    n=$(ls "$d"/*/episodes/*summary*.json 2>/dev/null | wc -l)
    summary="${summary}${mode}: ${n} ep in $(basename "$d"); "
  else
    summary="${summary}${mode}: NO RUN DIR; "
  fi
done
say "smoke summary: ${summary}"
push "B4 smoke done" "${summary} — check tool-call emit rate + parse_error_rate before the 6-condition fire"

# ---------- 3. B1 x VWA-cls SoM replicate (widens the rerun floor) ------------------
# FORCE_NEW=1 is exported at the top of this script and is LOAD-BEARING here: without it
# mint_run_id resume-globs the existing canonical run and this produces no second
# observation at all, while looking like it ran (B-1916).
say "=== B1 som classifieds — same-condition replicate (224 scored tasks, local, \$0) ==="
setsid nohup bash scripts/queues/queue_chain.sh \
  "queue_baseline.sh B1 som classifieds" \
  > "logs/queue_chain_b1_cls_som_replicate_${LOG_TS}.log" 2>&1 < /dev/null &
say "B1 cls som replicate chain launched (pid $!)"
push "B1 cls som replicate launched" \
  "widens rerun-floor coverage beyond the single fully-floored cell (DeepSeek: two-cell anecdote). Local model, no API spend."
