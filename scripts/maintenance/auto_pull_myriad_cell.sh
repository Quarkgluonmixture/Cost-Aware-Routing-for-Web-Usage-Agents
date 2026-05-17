#!/usr/bin/env bash
# auto_pull_myriad_cell.sh — pulled by myriad_watcher.py GONE_HOOKS dispatch.
#
# Audit (A)+(C) 2026-05-09:
#   When a Myriad job disappears from qstat (state went r → gone), the
#   watcher cron looks up the job's name in GONE_HOOKS and fires this
#   script to:
#     0. (NEW 2026-05-10) Probe remote for done-sentinel (pilot_summary.md
#        OR condition_summary_v2.json). If missing → abort + low-priority
#        ntfy (job qdel'd / crashed; do NOT pull partial incremental writes).
#        Override: P79_SKIP_SENTINEL=1.
#     1. SCP results from Myriad to DGX (via DGX → quark → Myriad chain)
#     2. Run validate_run.py --strict if the cell has a paper-grade
#        Phase 1 condition_summary_v2.json (paper hygiene gate)
#     3. Update the matching _status/cells/cell_*.md frontmatter
#     4. ntfy push with cell summary + validation verdict
#
# Usage (called by myriad_watcher only; not interactive):
#   bash scripts/maintenance/auto_pull_myriad_cell.sh \
#     <job_id> <job_name> <remote_dir_basename> [<cell_md_relpath>]
#
# Example invocation:
#   bash scripts/maintenance/auto_pull_myriad_cell.sh \
#     336424 cellg_rev_ stage2c_cellg_rev_reddit_reverse_myriad
#
# Env (override defaults):
#   QUARK_KEY=~/.ssh/vwa_windows
#   QUARK_HOST=Quark@YOUR_HOST_IP
#   MYRIAD_USER=ucab352
#   MYRIAD_REMOTE_BASE=/home/ucab352/Scratch/p79/results/mechanistic
#   NTFY_TOPIC=p79-exp-dgx-spark
#   P79_SKIP_ANALYSIS=1     skip make analysis trigger (B chain)
#   P79_SKIP_VALIDATE=1     skip validate-strict (C chain)
#   P79_SKIP_SENTINEL=1     skip Phase 0 done-sentinel check (paper-grade
#                           rerun forensic edge cases; default OFF)

set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <job_id> <job_name> <remote_dir_basename> [<cell_md_relpath>]" >&2
    exit 64
fi

JOB_ID="$1"
JOB_NAME="$2"
REMOTE_BASENAME="$3"
CELL_MD="${4:-}"

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

LOG="logs/cron/auto_pull_${JOB_ID}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs/cron
exec > >(tee -a "$LOG") 2>&1
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] auto_pull_myriad_cell job=$JOB_ID name=$JOB_NAME remote=$REMOTE_BASENAME"

QUARK_KEY="${QUARK_KEY:-$HOME/.ssh/vwa_windows}"
QUARK_HOST="${QUARK_HOST:?QUARK_HOST must be set, e.g. Quark@YOUR_HOST_IP}"
MYRIAD_USER="${MYRIAD_USER:-ucab352}"
MYRIAD_REMOTE_BASE="${MYRIAD_REMOTE_BASE:-/home/ucab352/Scratch/p79/results/mechanistic}"
NTFY_TOPIC="${NTFY_TOPIC:-p79-exp-dgx-spark}"
LOCAL_DIR="$REPO/results/mechanistic/$REMOTE_BASENAME"

mkdir -p "$LOCAL_DIR"

push_ntfy() {
    local title="$1"
    local body="$2"
    local prio="${3:-default}"
    curl -sS --max-time 10 -d "$body" \
        -H "Title: $title" -H "Priority: $prio" -H "Tags: gear" \
        "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true
}

# Phase 0 (added 2026-05-10 after qdel pollution incident): require done-sentinel
# on remote before pulling. Stage 2/3 mech runs write pilot_summary.md as the
# LAST step; Phase 1 paper-grade cells write condition_summary_v2.json. If
# neither exists, the job either qdel'd mid-run or crashed → partial incremental
# writes (24-tasks JSON written task-by-task per F18+F19 audit) would otherwise
# get SCP'd and pollute the local dir, masquerading as valid data.
#
# Bypass: P79_SKIP_SENTINEL=1 (paper-grade rerun edge cases where pull-anyway
# is preferred for forensics; default OFF).
if [ "${P79_SKIP_SENTINEL:-0}" != "1" ]; then
    echo "Phase 0: probing remote for done-sentinel (pilot_summary.md OR condition_summary_v2.json OR hidden_states.npz)"
    SENTINEL_CHECK=$(ssh -i "$QUARK_KEY" -o BatchMode=yes -o ConnectTimeout=30 "$QUARK_HOST" \
        "ssh -i \$env:USERPROFILE\\.ssh\\id_rsa_myriad ${MYRIAD_USER}@myriad.rc.ucl.ac.uk \"\
            test -s '$MYRIAD_REMOTE_BASE/$REMOTE_BASENAME/pilot_summary.md' && echo SENTINEL_OK_PILOT && exit 0; \
            test -s '$MYRIAD_REMOTE_BASE/$REMOTE_BASENAME/condition_summary_v2.json' && echo SENTINEL_OK_CONDITION && exit 0; \
            test -s '$MYRIAD_REMOTE_BASE/$REMOTE_BASENAME/hidden_states.npz' && echo SENTINEL_OK_HIDDEN_STATES && exit 0; \
            echo SENTINEL_MISSING\"" \
        2>/dev/null | tail -1 | tr -d '[:space:]')
    case "$SENTINEL_CHECK" in
        SENTINEL_OK_PILOT|SENTINEL_OK_CONDITION|SENTINEL_OK_HIDDEN_STATES)
            echo "  $SENTINEL_CHECK — proceeding with pull"
            ;;
        SENTINEL_MISSING)
            push_ntfy "auto_pull SKIP (no sentinel): $JOB_NAME" \
                "job=$JOB_ID remote=$REMOTE_BASENAME → no pilot_summary.md / condition_summary_v2.json / hidden_states.npz on remote. Likely qdel'd / crashed. Skipping SCP to avoid polluting local dir with partial data." \
                "low"
            echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] auto_pull SKIP: no done-sentinel on remote (job likely qdel'd or crashed). Set P79_SKIP_SENTINEL=1 to override."
            exit 0
            ;;
        *)
            # SSH probe failed — fall through to Phase 1 (existing pull-anyway behavior preserves backward compatibility on transient SSH blips)
            echo "  WARN: sentinel probe inconclusive ('$SENTINEL_CHECK') — falling through to Phase 1 attempt"
            ;;
    esac
fi

# Phase 1: SCP via SSH chain (cat | base64 because direct scp DGX→Myriad
# doesn't go through Tailscale)
echo "Phase 1: pulling artifacts via DGX → quark → Myriad chain"
PULLED=0
FAILED=()
for FILE in env_snapshot.json patching_continuation_results.json \
            pilot_summary.md run_manifest.json patching_continuation_curves.png \
            condition_summary_v2.json hidden_states.npz \
            hidden_states_v2_fixed.npz hidden_states_v2_fixed.provenance.json; do
    REMOTE_PATH="$MYRIAD_REMOTE_BASE/$REMOTE_BASENAME/$FILE"
    LOCAL_PATH="$LOCAL_DIR/$FILE"
    ssh -i "$QUARK_KEY" -o BatchMode=yes -o ConnectTimeout=30 "$QUARK_HOST" \
        "ssh -i \$env:USERPROFILE\\.ssh\\id_rsa_myriad ${MYRIAD_USER}@myriad.rc.ucl.ac.uk \"cat $REMOTE_PATH 2>/dev/null | base64 -w0\"" \
        2>/dev/null | tail -1 | base64 -d > "$LOCAL_PATH" 2>/dev/null || true
    SIZE=$(stat -c%s "$LOCAL_PATH" 2>/dev/null || echo "0")
    if [ "$SIZE" -gt 0 ]; then
        echo "  $FILE → $SIZE bytes"
        PULLED=$((PULLED + 1))
    else
        # Not all cells write all files; missing is OK if at least summary exists
        rm -f "$LOCAL_PATH"
        FAILED+=("$FILE")
    fi
done

if [ "$PULLED" -eq 0 ]; then
    push_ntfy "auto_pull FAIL: $JOB_NAME" \
        "job=$JOB_ID remote=$REMOTE_BASENAME → 0 files pulled. SSH chain or remote path issue." \
        "high"
    echo "ERROR: 0 files pulled, abort"
    exit 2
fi

# B-830 (/stress A1.16 cold-start P1-4-B*, 2026-05-17): post-pull JSON schema
# validation. Pre-fix: size-only check (line 125-133) accepted truncated /
# corrupt JSON / wrong git revision / wrong cell snapshot silently; Phase 4
# `make analysis FAST=1` then ate the contaminated data into paper §5 figures.
# Now: each .json file must parse + env_snapshot.json must have schema_version
# field + condition_summary_v2.json must have schema_version=v2 OR be quarantined.
echo "Phase 1.5: post-pull JSON schema validation (B-830)"
JSON_VALIDATION_FAILED=()
for JSON_FILE in env_snapshot.json run_manifest.json condition_summary_v2.json \
                 patching_continuation_results.json hidden_states_v2_fixed.provenance.json; do
    JSON_PATH="$LOCAL_DIR/$JSON_FILE"
    if [ ! -f "$JSON_PATH" ]; then
        continue  # missing is OK per existing tolerance
    fi
    # Parse JSON; record corruption
    if ! .venv/bin/python3 -c "import json,sys; json.load(open('$JSON_PATH'))" 2>/dev/null; then
        JSON_VALIDATION_FAILED+=("$JSON_FILE:corrupt-json")
        continue
    fi
    # Schema-specific minimum-field checks
    case "$JSON_FILE" in
        env_snapshot.json)
            if ! .venv/bin/python3 -c "
import json,sys
d=json.load(open('$JSON_PATH'))
assert d.get('schema_version'), 'missing schema_version'
assert isinstance(d.get('git'),dict), 'missing git block'
assert isinstance(d.get('models'),dict), 'missing models block'
" 2>/dev/null; then
                JSON_VALIDATION_FAILED+=("$JSON_FILE:schema-fields-missing")
            fi
            ;;
        condition_summary_v2.json)
            if ! .venv/bin/python3 -c "
import json,sys
d=json.load(open('$JSON_PATH'))
sv=str(d.get('schema_version',''))
assert sv.startswith('v2'), f'schema_version={sv!r} not v2'
" 2>/dev/null; then
                JSON_VALIDATION_FAILED+=("$JSON_FILE:not-schema-v2")
            fi
            ;;
    esac
done

if [ ${#JSON_VALIDATION_FAILED[@]} -gt 0 ]; then
    echo "  ⚠️  ${#JSON_VALIDATION_FAILED[@]} JSON file(s) failed schema validation:"
    for f in "${JSON_VALIDATION_FAILED[@]}"; do
        echo "    - $f"
    done
    if [ "${P79_PAPER_GRADE:-0}" = "1" ]; then
        push_ntfy "auto_pull JSON-SCHEMA-FAIL: $JOB_NAME" \
            "job=$JOB_ID — ${#JSON_VALIDATION_FAILED[@]} json file(s) failed schema validation: ${JSON_VALIDATION_FAILED[*]}" \
            "high"
        echo "ERROR: P79_PAPER_GRADE=1 + JSON validation failed → abort"
        exit 3
    fi
    # Dev mode: warn but continue (Phase 2 validate_run.py may still pass for condition_summary)
    echo "  (P79_PAPER_GRADE != 1, continuing with warnings)"
else
    echo "  ✓ all pulled JSON files passed schema validation"
fi

# Phase 2: validate-strict gate (audit C)
VALIDATE_VERDICT="skipped"
if [ "${P79_SKIP_VALIDATE:-0}" != "1" ] && [ -f "$LOCAL_DIR/condition_summary_v2.json" ]; then
    echo "Phase 2: validate-strict gate"
    set +e
    .venv/bin/python3 scripts/analysis/validate_run.py --run-dir "$LOCAL_DIR" --strict \
        --output "$LOCAL_DIR/validation_report.json" 2>&1 | tail -10
    VALIDATE_RC=$?
    set -e
    if [ "$VALIDATE_RC" -eq 0 ]; then
        VALIDATE_VERDICT="✅ pass"
    else
        VALIDATE_VERDICT="❌ FAIL (quarantine)"
    fi
fi

# Phase 3: cell frontmatter status update + analysis trigger (audit B)
if [ -n "$CELL_MD" ] && [ -f "$REPO/$CELL_MD" ]; then
    echo "Phase 3: updating cell note $CELL_MD"
    if [[ "$VALIDATE_VERDICT" == *"FAIL"* ]]; then
        # Add quarantined flag (manual review required before paper-grade promotion)
        .venv/bin/python3 -c "
import sys, yaml
from pathlib import Path
p = Path('$REPO/$CELL_MD')
text = p.read_text()
parts = text.split('---', 2)
if len(parts) >= 3:
    fm = yaml.safe_load(parts[1]) or {}
    fm['status'] = 'quarantined'
    fm['quarantine_reason'] = 'validate-strict failed post-pull'
    fm['quarantine_at'] = '$(date -u +%Y-%m-%dT%H:%M:%SZ)'
    body = parts[2]
    new_fm = '\n'.join(f'{k}: {v!r}' if isinstance(v, str) and (':' in v or v == '') else f'{k}: {v}' for k, v in fm.items())
    p.write_text('---\n' + new_fm + '\n---' + body)
    print(f'Marked $CELL_MD quarantined')
" 2>&1 || echo "  (cell md update failed, skip)"
    fi
fi

# Phase 4: trigger make analysis (audit B chain)
if [ "${P79_SKIP_ANALYSIS:-0}" != "1" ]; then
    echo "Phase 4: trigger make analysis FAST=1 (background)"
    nohup bash -c "cd '$REPO' && make analysis FAST=1 > logs/cron/post_pull_analysis_${JOB_ID}.log 2>&1" \
        >/dev/null 2>&1 &
    disown
    echo "  triggered analysis pipeline in background"
fi

# Phase 5: notify
SUMMARY_LINE=""
if [ -f "$LOCAL_DIR/pilot_summary.md" ]; then
    SUMMARY_LINE=$(grep -m1 "Best layer\|Holm\|p_Holm\|Significance" "$LOCAL_DIR/pilot_summary.md" 2>/dev/null | head -1 | tr '|' ' ' | cut -c-120)
fi
push_ntfy "Cell pulled: $JOB_NAME" \
    "job=$JOB_ID files=$PULLED validate=$VALIDATE_VERDICT $SUMMARY_LINE" \
    "default"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] auto_pull DONE pulled=$PULLED validate=$VALIDATE_VERDICT"
exit 0
