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

# B-904 (/stress A2.2 P0-3-A* OOB, 2026-05-17): source paper-grade lib. Pre-fix
# launch.sh was the ONLY user-facing manual entry skipping `_lib_paper_grade_gates.sh`,
# leaving init_paper_grade_env (P79_PAPER_GRADE=1 default) + acquire_site_lock
# + cross-mode collision API all dormant on `make launch` / manual rescue path.
# Defense stack (B-548/B-754/B-755/B-756/B-639) bypassed entirely under
# `P79_ALLOW_NO_RESET=1 bash launch.sh ...` workflow.
# Note: `assert_a100_url_locality` deliberately NOT called here — that gate is
# already enforced inside the downstream queue script (queue_baseline.sh:97 +
# queue_phantom_*.sh:84) where it fails fast just before actual fire. Calling it
# here would break DGX dev-session DRY-run + sanity-check workflows (cwd != A100,
# Tailscale URLs not local). Operators firing paper-grade fire still hit the gate
# via the queue leaf, no defense gap.
# shellcheck disable=SC1091
source "$REPO/scripts/queues/_lib_paper_grade_gates.sh"
init_paper_grade_env "$REPO"

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
    # B-903 (/stress A2.2 P0-2-A* OOB, 2026-05-17): site-collision check uses lib
    # API instead of inline `pgrep -f "_${OTHER}_.*_${SITE}_"`. Pre-fix substring
    # pattern missed `_[0-9]{8}_` date anchor → False-Positive BLOCK on
    # `B0_dom_wa_shopping_admin_<date>_` when user `bash launch.sh B1 dom shopping`
    # (substring match `_shopping_`); reverse direction False-Negative MISS for
    # `bash launch.sh B0 dom wa_shopping` vs running B1 VWA shopping (pattern
    # `B1_.*_wa_shopping_` not matching VWA `_dom_shopping_<date>_` line). Lib
    # `assert_no_cross_mode_collision` (B-858) + queue_chain.sh:208 (B-637) both
    # already use anchored `_${site}_[0-9]{8}_` + WA exclusion; this propagates
    # to user-facing manual entry.
    # The lib helper is structured as same-baseline + cross-mode check; for the
    # launch.sh "cross-baseline" semantics we wrap a loop over 2 other baselines.
    _BENCHMARK="vwa"
    [[ "$SITE" == wa_* ]] && _BENCHMARK="wa"
    _SITE_FOR_LIB="$SITE"
    [[ "$_BENCHMARK" == "wa" ]] && _SITE_FOR_LIB="${SITE#wa_}"
    for OTHER in B0 B1 B2; do
      [ "$OTHER" = "$BASELINE" ] && continue
      # Fake RUN_ID for the lib helper's self-exclusion — pattern `_OTHER_*` won't match
      # since RUN_ID we pass has `BASELINE` prefix.
      _FAKE_RID="${BASELINE}_launch_sentinel_$$"
      # Call lib with OTHER baseline + same site → returns rc=1 (FATAL via exit 1)
      # if OTHER is running on same (site, benchmark). Use subshell to capture exit.
      if ( assert_no_cross_mode_collision "$OTHER" "$_SITE_FOR_LIB" "$_BENCHMARK" "$_FAKE_RID" "launch_collision_check" ) >/dev/null 2>&1; then
        : # no collision
      else
        echo "❌ BLOCK: ${OTHER} already running on site=${SITE} benchmark=${_BENCHMARK} (paper-grade hard rule §106)" >&2
        echo "  shared docker container + user account → cross-contamination" >&2
        echo "  (date-anchored detection via lib assert_no_cross_mode_collision, B-903 propagation)" >&2
        exit 2
      fi
    done

    # B-904 (P0-3-A* cont): RESET gate now uses P79_PAPER_GRADE env naming
    # (set by init_paper_grade_env above, default 1). Pre-fix `P79_ALLOW_NO_RESET`
    # was orphan naming not honored by lib hard-blocks (B-639/B-754). Now: under
    # P79_PAPER_GRADE=1 (default) RESET=0 explicit FATAL; under P79_PAPER_GRADE=0
    # (dev opt-out) RESET=0 allowed. Legacy P79_ALLOW_NO_RESET=1 still honored as
    # back-compat shim that implies P79_PAPER_GRADE=0 (warns operator).
    if [ "$RESET" != "1" ]; then
      if [ "${P79_ALLOW_NO_RESET:-0}" = "1" ]; then
        echo "[launch][warn] P79_ALLOW_NO_RESET=1 legacy shim active; treating as P79_PAPER_GRADE=0 dev opt-out (B-904)." >&2
        export P79_PAPER_GRADE=0
      fi
      if [ "${P79_PAPER_GRADE:-1}" = "1" ]; then
        echo "❌ BLOCK: RESET=0 + P79_PAPER_GRADE=1 (paper-grade default)" >&2
        echo "  set P79_PAPER_GRADE=0 for dev rerun (NOT paper-grade)" >&2
        exit 2
      fi
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

# B-906 (/stress A2.2 P1-8-AB, 2026-05-17): post-launch GLM PLAYBOOK refresh hook
# delay 30s → 300s. Pre-fix 30s fires while reddit reset_and_auth_gate is in the
# postmill cold-start polling window (60 iters × 3s = up to 180s postmill warm-up
# per `reset_vwa_sites.sh::_reset_vwa_local_reddit`) + 15s settle + 60s auth gate
# = up to 255s before runner spawn. GLM cron at 30s sees no runner pid → cell
# frontmatter writes "pending with no pid" 5min false-active window. Manual rescue
# / master orchestrator may misread "idle" and trigger same-site collision.
# 300s covers reddit worst-case. Option C sentinel infra (lib `logs/launching/`
# per-iteration JSON write + active_processes recognition) reserved for Tier 3
# (B-912+) — see 笔记 §208.
nohup bash -c "sleep 300 && cd '$REPO' && make glm-update-cells APPLY=1 && make glm-refresh-playbook APPLY=1" \
  >> logs/cron/glm_playbook.log 2>&1 < /dev/null &
disown
echo "✓ Triggered PLAYBOOK refresh in background (300s delay; covers reddit cold-start worst-case 255s, B-906)"
echo ""
echo "Monitor:"
echo "  tail -f $LOG"
echo "  make active                                # 实时扫"
echo "  cat docs/checkpoints/_status/cells/$(basename "$CELL_FILE")  # cell frontmatter"
