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
  echo "  BASELINE: B0 | B1 | Claude" >&2
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

# ---- Step 2: pre-launch sanity (paper-grade contamination defense) ----
if [ "$FORCE_NO_CHECK" != "1" ]; then
  echo ""
  echo "🔍 Running pre-launch sanity check..."
  PRECHECK_ARGS="--queue $QUEUE --baseline $BASELINE --site $SITE --mode $MODE"
  if [ "$RESET" = "1" ]; then PRECHECK_ARGS="$PRECHECK_ARGS --reset"; fi

  if [ "$DRY" = "1" ]; then
    echo "  (DRY — would run: .venv/bin/python scripts/maintenance/glm/glm_pre_launch_check.py $PRECHECK_ARGS)"
    PRECHECK_RC=0
  else
    set +e
    .venv/bin/python scripts/maintenance/glm/glm_pre_launch_check.py $PRECHECK_ARGS
    PRECHECK_RC=$?
    set -e
  fi

  if [ "$PRECHECK_RC" = "2" ]; then
    echo "❌ Pre-launch BLOCK (hard rule violation). Aborting." >&2
    exit 2
  elif [ "$PRECHECK_RC" = "1" ]; then
    echo ""
    echo "⚠️  Pre-launch WARN — review concerns above. Proceed? [y/N]" >&2
    read -r ans
    if [ "$ans" != "y" ] && [ "$ans" != "Y" ]; then
      echo "Aborted by user." >&2
      exit 1
    fi
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
echo ""
echo "Monitor:"
echo "  tail -f $LOG"
echo "  make active                                # 实时扫"
echo "  cat docs/checkpoints/_status/cells/$(basename "$CELL_FILE")  # cell frontmatter"
