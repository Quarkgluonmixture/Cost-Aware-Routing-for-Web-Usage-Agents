#!/usr/bin/env bash
# capture_doi1_witness.sh — Reusable OSF DOI 1 pre-canonical-outcome-creation witness capture
#
# Stage 2 of two-DOI workflow per docs/checkpoints/pre_run/osf_lock_manifest.md §3a.
# Replaces the one-shot bash heredoc that caused B-1670 schema-mismatch pattern bug
# (used `episodes/*_summary.json` instead of canonical `*_summary_v2.json` per
# p79/experiment/logger_v2.py:111 + p79/experiment/analysis.py:209).
#
# Canonical schema sources (cited as comments; replicators can git-grep verify):
#   - episodes/<site>_task_<N>_summary_v2.json  (per p79/experiment/logger_v2.py:114)
#   - episodes/<site>_task_<N>_steps_v2.jsonl   (per p79/experiment/logger_v2.py:111)
#   - condition_summary_v2.json                 (per p79/experiment/runner/main.py)
#
# Includes B-1675 P1-4 known-positive sanity probe: alongside target-pattern grep,
# runs an any-file grep on the same scope. If target=0 AND any-file>0 = schema-
# mismatch detected → abort + report. If both=0 = genuinely empty. If target>0 =
# tier auto-downgrade (pre-outcome-creation → pre-outcome-inspection → pre-analysis).
#
# Usage:
#   capture_doi1_witness.sh [--remote condense-a100] [--run-dir-pattern PATTERN]
#                            [--output OUTFILE] [--label LABEL]
#                            [--bundle-regen] [--dry-run]
#
# Defaults:
#   --remote condense-a100 (Phase 1a paper-grade canonical host)
#   --run-dir-pattern "results/visualwebarena/phase1/B?_*_2026*"
#   --output docs/checkpoints/pre_run/artifact_existence_check_doi1_canonical_<UTC-TS>.txt
#   --label canonical (alt: interim, retraction)
#
# Examples:
#   # Interim scan (post-cleanup, pre-fire):
#   ./capture_doi1_witness.sh --label interim --remote condense-a100
#
#   # Canonical at fire-3 PID-alive:
#   ./capture_doi1_witness.sh --label canonical --remote condense-a100 \
#     --run-dir-pattern "results/visualwebarena/phase1/B?_*_<fire3-TS>*" \
#     --bundle-regen
#
#   # Local pytest invocation (no remote):
#   ./capture_doi1_witness.sh --run-dir-pattern "tests/fixtures/witness_test/B?_*"
#
# Exit codes:
#   0  capture successful
#   1  argument error
#   2  remote SSH failure
#   3  schema-mismatch detected (target=0 AND any-file>0) — pattern bug suspected
#   4  bundle regen failed
#
# Audit trail:
#   - B-1670: original witness pattern bug (Mode A+B+C cross-AI 2026-05-18)
#   - B-1675: this script (P1-6 deferred from Stage 1 commit)
#   - tests/test_doi1_witness_pattern.py: regression test (mandatory before fire-3)
#
# See: docs/checkpoints/pre_run/DOI_1_README.md
#      docs/reference/master_bug_catalog.md ## /stress witness pattern bug retraction
#      docs/checkpoints/实验笔记.md §231

set -euo pipefail

# -----------------------------------------------------------------------------
# Canonical schema patterns (DO NOT MODIFY without updating logger_v2.py / analysis.py)
# -----------------------------------------------------------------------------
readonly EPISODE_SUMMARY_PATTERN="*_summary_v2.json"        # per logger_v2.py:114
readonly EPISODE_STEPS_PATTERN="*_steps_v2.jsonl"           # per logger_v2.py:111
readonly CONDITION_SUMMARY_NAME="condition_summary_v2.json" # per runner/main.py
readonly KNOWN_POSITIVE_PATTERN="*.json"                    # any-JSON sanity probe

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
REMOTE_HOST=""
RUN_DIR_PATTERN="results/visualwebarena/phase1/B?_*_2026*"
LABEL="canonical"
OUTFILE=""
BUNDLE_REGEN=0
DRY_RUN=0

# -----------------------------------------------------------------------------
# Arg parse
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --remote)        REMOTE_HOST="$2"; shift 2 ;;
        --run-dir-pattern) RUN_DIR_PATTERN="$2"; shift 2 ;;
        --output)        OUTFILE="$2"; shift 2 ;;
        --label)         LABEL="$2"; shift 2 ;;
        --bundle-regen)  BUNDLE_REGEN=1; shift ;;
        --dry-run)       DRY_RUN=1; shift ;;
        -h|--help)
            sed -n '/^#/p' "$0" | sed 's/^# \?//' | head -60
            exit 0 ;;
        *)
            echo "Unknown arg: $1" >&2
            echo "Try $0 --help" >&2
            exit 1 ;;
    esac
done

# -----------------------------------------------------------------------------
# Compute UTC timestamp + default outfile
# -----------------------------------------------------------------------------
UTC_TS=$(date -u +%Y%m%dT%H%M%SZ)
UTC_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)

if [[ -z "$OUTFILE" ]]; then
    OUTFILE="docs/checkpoints/pre_run/artifact_existence_check_doi1_${LABEL}_${UTC_TS}.txt"
fi

# -----------------------------------------------------------------------------
# Construct probe script (run on local or remote)
# -----------------------------------------------------------------------------
read -r -d '' PROBE_SCRIPT <<'PROBE_EOF' || true
# Probe script: diagnostic-only, tolerate find errors on no-match patterns
# (don't use -e so missing-dir patterns become "0 count" not script-abort)
set -uo pipefail
PATTERN="$1"
CAPTURED_AT_UTC="$2"
LABEL="$3"

# Expand pattern via nullglob; if no match, set empty array
shopt -s nullglob
DIRS=( $PATTERN )
shopt -u nullglob

echo "=== Phase 1a outcome artifact existence check (DOI 1 ${LABEL} witness) ==="
echo ""
echo "Captured-at-UTC: $CAPTURED_AT_UTC"
echo "Host: $(hostname)"
echo "Uptime-since: $(uptime -s 2>/dev/null || echo unknown)"
echo "Run-dir-pattern: $PATTERN"
echo ""
echo "## Canonical patterns (per p79/experiment/logger_v2.py:111 + :114 + analysis.py:209)"
echo ""
echo "  episodes/<site>_task_<N>_summary_v2.json  ← per-episode outcome (earliest tier)"
echo "  episodes/<site>_task_<N>_steps_v2.jsonl   ← step-level intermediate"
echo "  condition_summary_v2.json                 ← condition-level aggregate"
echo ""
echo "## Outcome artifact counts (canonical schema)"
echo ""

# Use nullglob-expanded array; if no run dirs match → all counts 0
if [ ${#DIRS[@]} -eq 0 ]; then
    SUM_FILES=0
    STP_FILES=0
    CS_FILES=0
    ANY_JSON=0
    ANY_FILES=0
    echo "  (no run dirs match pattern — substrate genuinely empty)"
else
    SUM_FILES=$(find "${DIRS[@]}" -path "*/episodes/*_summary_v2.json" 2>/dev/null | wc -l)
    STP_FILES=$(find "${DIRS[@]}" -path "*/episodes/*_steps_v2.jsonl" 2>/dev/null | wc -l)
    CS_FILES=$(find "${DIRS[@]}" -name "condition_summary_v2.json" 2>/dev/null | wc -l)
    ANY_JSON=$(find "${DIRS[@]}" -name "*.json" 2>/dev/null | wc -l)
    ANY_FILES=$(find "${DIRS[@]}" -type f 2>/dev/null | wc -l)
fi

echo "episode_summary_v2_count: $SUM_FILES"
echo "episode_steps_v2_count:   $STP_FILES"
echo "condition_summary_v2_count: $CS_FILES"
echo ""

# B-1675 known-positive sanity probe
echo "## Known-positive sanity probe (B-1675 P1-4 schema-mismatch detection)"
echo ""
echo "any_json_in_run_dirs: $ANY_JSON"
echo "any_files_in_run_dirs: $ANY_FILES"
echo ""

# Status determination
if [ "$SUM_FILES" = "0" ] && [ "$STP_FILES" = "0" ] && [ "$CS_FILES" = "0" ]; then
    if [ "$ANY_JSON" -gt "0" ] || [ "$ANY_FILES" -gt "5" ]; then
        echo "STATUS: SCHEMA-MISMATCH-SUSPECTED — target=0 but any-files>0"
        echo "  → pattern bug pattern, abort + investigate"
        echo "  → see B-1670 retraction history for pattern-bug class"
        exit 3
    else
        echo "STATUS: pre-outcome-creation (canonical patterns return 0, any-files also ≤5 = scaffold-only)"
    fi
elif [ "$CS_FILES" -gt "0" ]; then
    echo "STATUS: post-outcome-aggregation (condition_summary_v2 present)"
    echo "  → DOI 1 tier downgrade to pre-analysis"
elif [ "$SUM_FILES" -gt "0" ] || [ "$STP_FILES" -gt "0" ]; then
    echo "STATUS: post-outcome-creation (per-episode outcome present, no condition aggregate)"
    echo "  → DOI 1 tier downgrade to pre-outcome-inspection"
fi
echo ""

# Provenance
echo "## Provenance"
echo ""
if [ -d ~/workspace/p79/.git ]; then
    cd ~/workspace/p79
    echo "Git HEAD: $(git rev-parse HEAD)"
    echo "preregistration-locked tag: $(git rev-parse preregistration-locked 2>/dev/null || echo NOT_LOCAL)"
    if [ -d external/visualwebarena ]; then
        echo "VWA submodule HEAD: $(git -C external/visualwebarena rev-parse HEAD)"
    fi
fi
PROBE_EOF

# -----------------------------------------------------------------------------
# Execute probe
# -----------------------------------------------------------------------------
if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] Would capture witness to: $OUTFILE"
    echo "[DRY-RUN] Run-dir-pattern: $RUN_DIR_PATTERN"
    echo "[DRY-RUN] Label: $LABEL"
    echo "[DRY-RUN] Remote: ${REMOTE_HOST:-local}"
    exit 0
fi

mkdir -p "$(dirname "$OUTFILE")"

if [[ -n "$REMOTE_HOST" ]]; then
    # Remote execution via SSH (banner-stripped output)
    SSH_OUT=$(ssh -q -o ConnectTimeout=15 "$REMOTE_HOST" \
        "cd ~/workspace/p79 && bash -s '$RUN_DIR_PATTERN' '$UTC_ISO' '$LABEL'" \
        <<<"$PROBE_SCRIPT" 2>&1) || {
        RC=$?
        echo "ERROR: SSH probe failed (rc=$RC)" >&2
        echo "$SSH_OUT" >&2
        exit 2
    }
    # Strip SSH jumphost banner (preserves probe content from === marker onwards)
    echo "$SSH_OUT" | sed -n '/^=== Phase 1a outcome/,$p' > "$OUTFILE"
else
    # Local execution
    bash -c "$PROBE_SCRIPT" -- "$RUN_DIR_PATTERN" "$UTC_ISO" "$LABEL" > "$OUTFILE"
fi

# -----------------------------------------------------------------------------
# Cross-link footer + SHA-256 self-doc
# -----------------------------------------------------------------------------
cat >> "$OUTFILE" <<FOOTER

## Cross-reference

- DOI 1 README: docs/checkpoints/pre_run/DOI_1_README.md
- Capture script: scripts/maintenance/capture_doi1_witness.sh (B-1675)
- Catalog entry: docs/reference/master_bug_catalog.md ## /stress witness pattern bug retraction (B-1670~B-1679)
- Chronicle: docs/checkpoints/实验笔记.md §231
- Retracted predecessor: docs/checkpoints/pre_run/artifact_existence_check_doi1_20260518T135722Z.txt (VOID)
- Interim corrected: docs/checkpoints/pre_run/artifact_existence_check_doi1_interim_20260518T144258Z.txt

## SHA-256 self-doc
FOOTER

# Compute SHA-256 (excluding the SHA line itself, which will follow)
SHA=$(sha256sum "$OUTFILE" | cut -d' ' -f1)
echo "$SHA  $(basename "$OUTFILE") (full file including this line's SHA reference)" >> "$OUTFILE"

# -----------------------------------------------------------------------------
# Report + optional bundle regen
# -----------------------------------------------------------------------------
echo "✓ Witness captured: $OUTFILE"
echo "  Label: $LABEL"
echo "  UTC timestamp: $UTC_ISO"
echo "  SHA-256: $SHA"
echo "  Size: $(wc -c < "$OUTFILE") bytes"
echo ""
echo "  Status: $(grep '^STATUS:' "$OUTFILE" | head -1)"

if [[ $BUNDLE_REGEN -eq 1 ]]; then
    BUNDLE_DIR="docs/checkpoints/pre_run/osf_deposit_DOI1_${UTC_TS}"
    echo ""
    echo "[bundle-regen] target dir: $BUNDLE_DIR"

    if [[ -d "$BUNDLE_DIR" ]]; then
        echo "[bundle-regen] removing existing dir"
        rm -rf "$BUNDLE_DIR"
    fi
    mkdir -p "$BUNDLE_DIR/pre_run" "$BUNDLE_DIR/paper_drafts"

    # Copy pre_run/ contents (excluding DOI_2_* placeholder + other deposit dirs)
    cd docs/checkpoints/pre_run
    for f in *.md *.txt; do
        [[ -f "$f" && ! "$f" =~ ^DOI_2 && ! -d "$f" ]] && cp "$f" "../../../$BUNDLE_DIR/pre_run/"
    done
    cd - >/dev/null

    # Copy paper drafts
    cp docs/checkpoints/paper_drafts/section*.md docs/checkpoints/paper_drafts/paper.bib \
        "$BUNDLE_DIR/paper_drafts/" 2>/dev/null

    # Generate manifest
    cd "$BUNDLE_DIR"
    find . -type f -not -name "MANIFEST_SHA256.txt" | sort | xargs sha256sum > MANIFEST_SHA256.txt
    cd - >/dev/null

    FILES=$(find "$BUNDLE_DIR" -type f | wc -l)
    SIZE=$(du -sh "$BUNDLE_DIR" | cut -f1)
    echo "[bundle-regen] ✓ $FILES files, $SIZE"
fi
