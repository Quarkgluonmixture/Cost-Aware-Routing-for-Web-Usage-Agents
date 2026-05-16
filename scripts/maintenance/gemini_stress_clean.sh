#!/usr/bin/env bash
# gemini_stress_clean.sh — chatter-free wrapper around `gemini --prompt` for
# /gemini-stress Mode C cross-AI audit dispatch.
#
# Why this exists
# ---------------
# Gemini CLI `-p`/`--prompt` mode tends to inject conversational chatter
# ("I have completed the audit...", "A detailed report has been saved...")
# even when the prompt explicitly requests structured output. When the
# response is captured via stdout redirect, this chatter blob gets
# interleaved with the structured markdown — and in the worst case
# (observed 2026-05-16 A1.4b-i + A1.4b-ii gemini outputs), the chatter
# replaces individual table rows mid-render, causing structural data loss.
#
# Fix path (Path C): use `--output-format json`. Gemini puts the response
# inside a JSON envelope's `.response` field; the JSON validity constraint
# suppresses chatter naturally (a chatter blob would break JSON). Then we
# jq-extract `.response` to recover clean markdown.
#
# Usage
# -----
#   scripts/maintenance/gemini_stress_clean.sh \
#       <prompt_file_path> \
#       <output_md_path>
#
# Or via stdin:
#   gemini_stress_clean.sh - <output_md_path> < prompt.md
#
# Optional env:
#   GEMINI_MODEL          — pin model (default: gemini lets CLI pick;
#                           plan mode auto-routes to gemini-3.1-pro-preview)
#   GEMINI_APPROVAL_MODE  — override approval mode (default: "plan"). Set to
#                           "yolo" only if the audit needs write access (rare
#                           — audit should be read-only by design).
#   GEMINI_KEEP_RAW       — if set, also write raw JSON to <output>.raw.json
#                           for audit trail / debugging structural issues.
#
# Exit codes:
#   0  — success
#   1  — usage error / missing input
#   2  — gemini CLI failed
#   3  — JSON parse failed (gemini returned something other than JSON)
#   4  — empty response field

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <prompt_file|-> <output_md_path>" >&2
  exit 1
fi

PROMPT_PATH="$1"
OUTPUT_PATH="$2"
RAW_JSON_PATH="${OUTPUT_PATH%.md}.raw.json"

if [[ "$PROMPT_PATH" == "-" ]]; then
  PROMPT_BODY=$(cat)
else
  if [[ ! -f "$PROMPT_PATH" ]]; then
    echo "ERROR: prompt file not found: $PROMPT_PATH" >&2
    exit 1
  fi
  PROMPT_BODY=$(<"$PROMPT_PATH")
fi

MODEL_FLAG=()
if [[ -n "${GEMINI_MODEL:-}" ]]; then
  MODEL_FLAG=(--model "$GEMINI_MODEL")
fi

mkdir -p "$(dirname "$OUTPUT_PATH")"

# IMPORTANT — choice of approval mode (`--yolo` vs `--approval-mode plan`):
#   yolo  → full shell + read/write tools. DEFAULT (set 2026-05-16 per
#           memory feedback_cross_ai_audit.md retract). Empirically required
#           when the prompt asks gemini to read multiple paper-drafts or
#           handoff artifacts: plan-mode silently blocks tool calls and the
#           wrapper extracts a hallucinated meta-summary from `.response`
#           (observed 2026-05-16 A1.4c first run: 1488 B vs 6699 B with --yolo
#           on the same prompt; Trial C Q4 empirical: 4940 tokens real call
#           vs Trial A plan-mode `326 ms / 0 tokens` suspicious cache hit).
#           This matches the memory retract: "gemini --yolo -p ≡ codex
#           --sandbox danger-full-access = full shell".
#   plan  → read-only (no write_file / edit_file). Retained as OPT-IN via
#           GEMINI_APPROVAL_MODE=plan env var for prompts that fully inline
#           context AND want gemini to write structured output via the
#           `.response` payload only. Auto-routes to `gemini-3.1-pro-preview`
#           in plan mode per Google CLI defaults, so the option survives if
#           a future audit prefers Pro-tier routing with inline-only context.
#
# IMPORTANT (cont) — yargs argument-parsing order:
#   --output-format BEFORE --prompt. yargs consumes the next token after
#   -p/--prompt as its value, so the long flag must come earlier or it
#   silently collides (observed 2026-05-16).
TMP_RAW=$(mktemp /tmp/gemini_stress_clean.XXXXXX.json)
trap 'rm -f "$TMP_RAW"' EXIT

APPROVAL_MODE="${GEMINI_APPROVAL_MODE:-yolo}"
if [ "$APPROVAL_MODE" = "yolo" ]; then
  GEMINI_FLAGS=(--yolo)
else
  GEMINI_FLAGS=(--approval-mode "$APPROVAL_MODE")
fi
if ! gemini "${GEMINI_FLAGS[@]}" --output-format json "${MODEL_FLAG[@]}" \
       --prompt "$PROMPT_BODY" > "$TMP_RAW" 2>&1; then
  echo "ERROR: gemini CLI exit code != 0" >&2
  tail -30 "$TMP_RAW" >&2
  cp "$TMP_RAW" "$RAW_JSON_PATH" 2>/dev/null || true
  exit 2
fi

# Gemini prints "YOLO mode is enabled..." preamble lines BEFORE the JSON
# object — they break jq parsing. Find the first '{' and start there.
JSON_START=$(grep -n '^{' "$TMP_RAW" | head -1 | cut -d: -f1 || true)
if [[ -z "$JSON_START" ]]; then
  echo "ERROR: no JSON object found in gemini output" >&2
  tail -30 "$TMP_RAW" >&2
  cp "$TMP_RAW" "$RAW_JSON_PATH" 2>/dev/null || true
  exit 3
fi

TMP_JSON=$(mktemp /tmp/gemini_stress_clean.XXXXXX.json)
trap 'rm -f "$TMP_RAW" "$TMP_JSON"' EXIT
tail -n +"$JSON_START" "$TMP_RAW" > "$TMP_JSON"

if ! jq -e '.response' "$TMP_JSON" > /dev/null 2>&1; then
  echo "ERROR: gemini JSON envelope missing .response field" >&2
  jq -r 'keys[]' "$TMP_JSON" 2>&1 | sed 's/^/    /' >&2
  cp "$TMP_JSON" "$RAW_JSON_PATH" 2>/dev/null || true
  exit 3
fi

jq -r '.response' "$TMP_JSON" > "$OUTPUT_PATH"

SIZE=$(wc -c < "$OUTPUT_PATH")
if [[ "$SIZE" -lt 100 ]]; then
  echo "ERROR: gemini .response field is empty or near-empty ($SIZE bytes)" >&2
  exit 4
fi

if [[ -n "${GEMINI_KEEP_RAW:-}" ]]; then
  cp "$TMP_JSON" "$RAW_JSON_PATH"
  echo "  raw envelope kept at: $RAW_JSON_PATH" >&2
fi

# Footer with usage stats from JSON envelope (latency + tokens) — useful
# for post-flight verification per Skill v7.4 Phase 3 runtime check.
USAGE=$(jq -r '
  .stats.models // {} | to_entries | .[0].value // {} |
  "\(.api.totalLatencyMs // 0)ms / \(.tokens.total // 0) tokens"
' "$TMP_JSON" 2>/dev/null || echo "—")
{
  echo
  echo "---"
  echo "_Wrapper: \`scripts/maintenance/gemini_stress_clean.sh\`. Stats: $USAGE._"
} >> "$OUTPUT_PATH"

echo "✓ gemini clean dispatch: $OUTPUT_PATH ($SIZE bytes)" >&2
exit 0
