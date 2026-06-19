#!/usr/bin/env bash
# agy_stress_clean.sh — dispatch wrapper around Antigravity CLI `agy --print`
# for /gemini-stress Mode C cross-AI audit.
#
# Replaces gemini_stress_clean.sh (RETIRED 2026-06-19)
# ----------------------------------------------------
# Google retired the Gemini CLI free/Pro OAuth tier on 2026-06-18 (individual
# users get `IneligibleTierError / UNSUPPORTED_CLIENT`) and migrated everyone
# to the Antigravity CLI (`agy`, Go rewrite, installs to ~/.local/bin). Two
# consequences:
#   1. agy has NO `--output-format json` → the old JSON-envelope + jq
#      `.response` extraction machinery is gone.
#   2. agy's `-p`/`--print` mode does NOT exhibit gemini's chatter-injection
#      problem (the ONLY reason the JSON wrapper existed). Verified 2026-06-19:
#      raw `agy -p` returns clean structured markdown directly.
# → This wrapper is now a thin model-lock + timeout + cold-read-isolation shell
#   around a plain `agy -p` call.
#
# COLD-READ ISOLATION (why no --add-dir by default, why neutral cwd)
# -----------------------------------------------------------------
# Mode C = INDEPENDENT cold-read reviewer. The repo root contains AGENTS.md +
# GEMINI.md, both of which point agy at .claude/CLAUDE.md + handoff docs. If agy
# runs with cwd=repo or `--add-dir <repo-root>`, it auto-loads those context
# files and (a) breaks the cold-read premise (it would read CLAUDE.md, the exact
# thing it must NOT see) and (b) triggers a file-read cascade / agent tool-loop.
# Empirically 2026-06-19: `--add-dir "$PWD"` on a generation prompt TIMED OUT at
# 120s; the same prompt with no --add-dir from a neutral cwd returned in 15s.
# → DEFAULT: no --add-dir, run from a scratch cwd. The audit prompt must be
#   SELF-CONTAINED — the orchestrator (Claude) inlines paper-draft contents into
#   the prompt rather than asking agy to open files. This is also more
#   deterministic and matches the "self-contained prompt" rule from
#   [[reference-gemini-cli]] ("in-repo bare -p 会跑偏").
#
# If you DO need agy to read files itself, set AGY_ADD_DIR to a MINIMAL subtree
# (e.g. docs/checkpoints/paper_drafts) — NEVER the repo root — so the root-level
# AGENTS.md/GEMINI.md are outside scope.
#
# Read-only by design: do NOT use --dangerously-skip-permissions (agy's yolo
# flag). It forces agent tool-loop mode (slow: 120s timeout vs 13s plain -p) and
# is unnecessary for read-only audit.
#
# Lineage lock: agy's palette ALSO has Claude Opus/Sonnet + GPT-OSS. Mode C is a
# Google-lineage reviewer — AGY_MODEL MUST stay a Gemini model. Default =
# "Gemini 3.1 Pro (Low)".
#
# Why (Low) not (High): empirical 2026-06-19. "(High)" reasoning effort does NOT
# terminate in print mode within 240s on a substantive audit prompt (timed out,
# empty output). "(Low)" returns a strong hostile audit (3 findings, P0/P1, OOB
# attacks quoting claims) in ~22s. The reviewer-quality bottleneck is independent
# perspective, not thinking budget — Low effort on Gemini 3.1 Pro is already
# submission-grade. Use (High) only as an explicit opt-in with a large AGY_TIMEOUT.
#
# Usage
# -----
#   scripts/maintenance/agy_stress_clean.sh <prompt_file|-> <output_md_path>
#   agy_stress_clean.sh - <output_md_path> < prompt.md
#
# Env:
#   AGY_MODEL    — model display name (default: "Gemini 3.1 Pro (Low)"). Keep Gemini.
#   AGY_TIMEOUT  — print-mode wall cap seconds (default 300).
#   AGY_ADD_DIR  — optional minimal read subtree (default: NONE — self-contained prompt).
#                  NEVER set to the repo root (AGENTS.md injection — see above).
#
# Exit codes: 0 ok · 1 usage/missing input · 2 agy failed (incl. timeout / not on PATH) · 4 empty response

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <prompt_file|-> <output_md_path>" >&2
  exit 1
fi

PROMPT_PATH="$1"
OUTPUT_PATH="$2"
# Absolute output path — we run agy from a scratch cwd below.
mkdir -p "$(dirname "$OUTPUT_PATH")"
OUTPUT_PATH="$(cd "$(dirname "$OUTPUT_PATH")" && pwd)/$(basename "$OUTPUT_PATH")"

export PATH="$HOME/.local/bin:$PATH"
if ! command -v agy >/dev/null 2>&1; then
  echo "ERROR: 'agy' (Antigravity CLI) not found on PATH. Install: curl -fsSL https://antigravity.google/cli/install.sh | bash" >&2
  exit 2
fi

if [[ "$PROMPT_PATH" == "-" ]]; then
  PROMPT_BODY=$(cat)
else
  if [[ ! -f "$PROMPT_PATH" ]]; then
    echo "ERROR: prompt file not found: $PROMPT_PATH" >&2
    exit 1
  fi
  PROMPT_BODY=$(<"$PROMPT_PATH")
fi

AGY_MODEL="${AGY_MODEL:-Gemini 3.1 Pro (Low)}"
AGY_TIMEOUT="${AGY_TIMEOUT:-300}"

ADD_DIR_FLAG=()
if [[ -n "${AGY_ADD_DIR:-}" ]]; then
  # Resolve to absolute (we cd to scratch before invoking agy).
  AGY_ADD_DIR_ABS="$(cd "$AGY_ADD_DIR" && pwd)"
  ADD_DIR_FLAG=(--add-dir "$AGY_ADD_DIR_ABS")
fi

# Neutral scratch cwd so agy does NOT auto-load the repo's AGENTS.md/GEMINI.md.
SCRATCH=$(mktemp -d /tmp/agy_stress.XXXXXX)
trap 'rm -rf "$SCRATCH"' EXIT

set +e
( cd "$SCRATCH" && timeout "$AGY_TIMEOUT" agy \
    "${ADD_DIR_FLAG[@]}" \
    --model "$AGY_MODEL" \
    --print-timeout "${AGY_TIMEOUT}s" \
    -p "$PROMPT_BODY" ) > "$OUTPUT_PATH" 2>&1
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "ERROR: agy CLI failed (exit $RC; 124=timeout after ${AGY_TIMEOUT}s)" >&2
  tail -20 "$OUTPUT_PATH" >&2 || true
  exit 2
fi

SIZE=$(wc -c < "$OUTPUT_PATH")
if [[ "$SIZE" -lt 100 ]]; then
  echo "ERROR: agy response empty or near-empty ($SIZE bytes)" >&2
  exit 4
fi

{
  echo
  echo "---"
  echo "_Wrapper: \`scripts/maintenance/agy_stress_clean.sh\` · model: \`${AGY_MODEL}\` · ${SIZE} bytes._"
} >> "$OUTPUT_PATH"

echo "✓ agy clean dispatch: $OUTPUT_PATH ($SIZE bytes, model: $AGY_MODEL)" >&2
exit 0
