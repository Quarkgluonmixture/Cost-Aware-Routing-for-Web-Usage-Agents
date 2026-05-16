#!/bin/bash
# launch.sh — one-shot wrapper: cell note → pre-launch check → nohup queue script.
#
# Usage:
#   bash scripts/maintenance/launch.sh BASELINE SITE MODE [TARGET_SECTION] [PRIORITY]
#
# Env (override defaults):
#   RESET=1                 RESET_BEFORE before launch (default: 1)
#   FORCE_NO_CHECK=1        skip pre-launch sanity (NOT recommended)
#   DRY=1                   show what would run, don't execute
#
# What it does (in order):
#   1. Resolve queue script from MODE
#   2. Auto-create _status/cells/cell_<lc>_<site3>_<mode>.md if missing
#   3. Run glm-pre-launch-check (BLOCK exit 2 = abort, WARN exit 1 = pause+confirm)
#   4. nohup-launch queue script in background, print PID + log path
#
# Examples:
#   bash scripts/maintenance/launch.sh B0 reddit phantom_text
#   RESET=0 bash scripts/maintenance/launch.sh B1 classifieds som  # rerun without reset
#   DRY=1 bash scripts/maintenance/launch.sh B0 shopping vision    # dry-run

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 BASELINE SITE MODE [TARGET_SECTION] [PRIORITY]" >&2
  # B-305 (A1.17 P2-2): help text updated to reflect 3-baseline reality
  # (B2=Gemma3-VL nominated 2026-05-14, advisor discussion §138).
  echo "  BASELINE: B0 | B1 | B2" >&2
  echo "  SITE:     classifieds | reddit | shopping | wa_shopping | wa_shopping_admin | wa_reddit" >&2
  echo "  MODE:     dom | som | vision | phantom_text | phantom_som | phantom_prompt" >&2
  exit 64
fi

BASELINE="$1"
SITE="$2"
MODE="$3"
TARGET_SECTION="${4:-4}"
PRIORITY="${5:-tier1}"
RESET="${RESET:-1}"
FORCE_NO_CHECK="${FORCE_NO_CHECK:-0}"
DRY="${DRY:-0}"

# ---- Mode → queue script ----
case "$MODE" in
  dom|som|vision)        QUEUE="queue_baseline.sh"; QUEUE_ARGS="$BASELINE $MODE $SITE" ;;
  phantom_text)          QUEUE="queue_phantom_text.sh"; QUEUE_ARGS="$BASELINE $SITE" ;;
  phantom_som)           QUEUE="queue_phantom_som.sh"; QUEUE_ARGS="$BASELINE $SITE" ;;
  phantom_prompt)        QUEUE="queue_phantom_prompt.sh"; QUEUE_ARGS="$BASELINE $SITE" ;;
  *) echo "❌ unknown MODE: $MODE" >&2; exit 65 ;;
esac

# ---- Mode → paper-facing label + cell filename ----
declare -A MODE_LABEL=(
  [dom]="DOM" [som]="SoM" [vision]="Vision"
  [phantom_text]="P-text" [phantom_som]="P-SoM" [phantom_prompt]="P-prompt"
)
declare -A MODE_FILE=(
  [dom]="dom" [som]="som" [vision]="vision"
  [phantom_text]="ptext" [phantom_som]="psom" [phantom_prompt]="pprompt"
)
declare -A SITE_3=(
  [classifieds]="cls" [reddit]="red" [shopping]="shop"
  [wa_shopping]="wa_shop" [wa_shopping_admin]="wa_admin" [wa_reddit]="wa_red"
)
declare -A SITE_N=(
  [classifieds]=234 [reddit]=210 [shopping]=466
  [wa_shopping]=192 [wa_shopping_admin]=182 [wa_reddit]=106
)

LABEL="${MODE_LABEL[$MODE]}"
FILE_MODE="${MODE_FILE[$MODE]}"
SITE3="${SITE_3[$SITE]:-$SITE}"
N="${SITE_N[$SITE]:-234}"
BASELINE_LC=$(echo "$BASELINE" | tr '[:upper:]' '[:lower:]')

CELL_FILE="$REPO/docs/checkpoints/_status/cells/cell_${BASELINE_LC}_${SITE3}_${FILE_MODE}.md"

# ---- Step 1: auto-create cell note if missing ----
if [ ! -f "$CELL_FILE" ]; then
  echo "📝 Creating cell note: $(basename "$CELL_FILE")"
  if [ "$DRY" = "1" ]; then
    echo "  (DRY — would write $CELL_FILE)"
  else
    cat > "$CELL_FILE" <<EOF
---
type: cell
baseline: $BASELINE
site: $SITE
mode: $LABEL
status: pending
progress: 0
target_section: $TARGET_SECTION
priority: $PRIORITY
phase_a: post-fix
n: $N
blocker: ""
eta: ""
---

# $BASELINE $SITE3 $LABEL (pending — auto-created by launch.sh $(date +%Y-%m-%d))

Cell note auto-generated. Cron \`glm-update-cells\` 会在 launch 后开始填
\`status\` / \`progress\` / \`sr_raw\` / \`pid\` / \`last_run_id\`. 自己只需补
\`blocker\` / \`eta\` 等语义字段.
EOF
  fi
else
  echo "✓ Cell note exists: $(basename "$CELL_FILE")"
fi

# ---- Step 2: pre-launch sanity (B-306 A1.17 P1-3+P1-10+P1-11 absorbed) ----
# B-306 (A1.17 2026-05-16): replaced glm_pre_launch_check.py with deterministic
# shell asserts. Rationale:
#   - 4/5 hard rules (same-site collision / RESET / queue-script match / WA reset)
#     already deterministically enforced in queue_chain.sh + queue_baseline.sh +
#     _lib_paper_grade_gates.sh:reset_and_auth_gate; GLM layer redundant.
#   - Only rule unique to glm_pre_launch_check was config-↔-site benchmark match;
#     now covered by the YAML-grep below.
#   - GLM dependency removed (LLM variance / API outage / non-deterministic gate
#     in paper-grade launch path = anti-pattern). Per codex Mode B C-8 defuse
#     "GLM should be advisory only".
# Pre-fix bugs absorbed: P1-3 BLOCK→WARN exit-code collapse / P1-10 non-greedy
# regex / P1-11 config-missing fail-open asymmetric — all obsolete because file
# itself deleted.
if [ "$FORCE_NO_CHECK" != "1" ]; then
  echo ""
  echo "🔍 Running deterministic pre-launch sanity..."

  if [ "$DRY" = "1" ]; then
    echo "  (DRY — deterministic checks skipped)"
  else
    # Rule #1 — Same-site single baseline (3-way collision, paper-grade hard rule)
    for OTHER in B0 B1 B2; do
      [ "$OTHER" = "$BASELINE" ] && continue
      if pgrep -f "run_experiment.*${OTHER}_.*_${SITE}_" >/dev/null 2>&1; then
        echo "❌ BLOCK: ${OTHER} already running on site=${SITE} (paper-grade hard rule §106)" >&2
        echo "  shared docker container + user account → cross-contamination" >&2
        exit 2
      fi
    done

    # Rule #2 — RESET_BEFORE for paper-grade (allow override via env)
    if [ "$RESET" != "1" ] && [ "${P79_ALLOW_NO_RESET:-0}" != "1" ]; then
      echo "❌ BLOCK: RESET=0 + paper-grade default" >&2
      echo "  set P79_ALLOW_NO_RESET=1 for dev rerun (NOT paper-grade)" >&2
      exit 2
    fi

    # Rule #5 — config ↔ site benchmark match (was glm-unique catch)
    CFG_CANDIDATES=(
      "configs/exp_v2_${BASELINE}_${MODE}_${SITE}.yaml"
      "configs/exp_v2_${BASELINE}_${SITE}_${MODE}.yaml"
    )
    for cfg in "${CFG_CANDIDATES[@]}"; do
      if [ -f "$cfg" ]; then
        EXPECTED_BENCH="visualwebarena"
        [[ "$SITE" == wa_* ]] && EXPECTED_BENCH="webarena"
        if ! grep -q "benchmark:[[:space:]]*${EXPECTED_BENCH}" "$cfg"; then
          echo "❌ BLOCK: ${cfg} benchmark mismatch (expected ${EXPECTED_BENCH})" >&2
          exit 2
        fi
        break
      fi
    done

    echo "✓ Deterministic pre-launch sanity passed"
  fi
fi

# ---- Step 3: nohup launch ----
LOG="logs/launch_${BASELINE_LC}_${SITE3}_${FILE_MODE}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

if [ "$DRY" = "1" ]; then
  echo ""
  echo "✓ DRY summary:"
  echo "  cell:  $CELL_FILE"
  echo "  queue: scripts/queues/$QUEUE $QUEUE_ARGS"
  echo "  reset: RESET_BEFORE=$RESET"
  echo "  log:   $LOG"
  exit 0
fi

echo ""
echo "🚀 Launching: scripts/queues/$QUEUE $QUEUE_ARGS"
echo "   log: $LOG"
RESET_BEFORE="$RESET" nohup bash "scripts/queues/$QUEUE" $QUEUE_ARGS \
  > "$LOG" 2>&1 < /dev/null &
PID=$!
disown
echo "✓ Launched PID=$PID"

# Post-launch hook: fire PLAYBOOK §1+§2 refresh in background so the new run
# shows up immediately (don't wait for next 2h cron tick). Best-effort, never
# blocks launch on GLM API hiccup.
nohup bash -c "sleep 30 && cd '$REPO' && make glm-update-cells APPLY=1 && make glm-refresh-playbook APPLY=1" \
  >> logs/cron/glm_playbook.log 2>&1 < /dev/null &
disown
echo "✓ Triggered PLAYBOOK refresh in background (30s delay for cell autodetect)"
echo ""
echo "Monitor:"
echo "  tail -f $LOG"
echo "  make active                                # 实时扫"
echo "  cat docs/checkpoints/_status/cells/$(basename "$CELL_FILE")  # cell frontmatter"
