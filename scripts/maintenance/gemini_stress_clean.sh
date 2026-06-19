#!/usr/bin/env bash
# gemini_stress_clean.sh — RETIRED 2026-06-19. Deprecation shim.
#
# Google retired the Gemini CLI free/Pro OAuth tier on 2026-06-18 (individual
# users get `IneligibleTierError / UNSUPPORTED_CLIENT`). The /gemini-stress
# Mode C dispatch now runs on the Antigravity CLI (`agy`) via
# scripts/maintenance/agy_stress_clean.sh. The old gemini `--output-format json`
# + jq `.response` machinery is obsolete (agy has no JSON output and does not
# inject chatter, so no envelope is needed).
#
# This shim forwards to the new wrapper so any stale caller keeps working while
# loudly announcing the migration. Remove once no caller references this name.

set -euo pipefail
NEW="$(dirname "$0")/agy_stress_clean.sh"
echo "⚠️  gemini_stress_clean.sh is RETIRED (gemini CLI OAuth tier killed 2026-06-18)." >&2
echo "    Forwarding to: $NEW" >&2
echo "    Update your caller to invoke agy_stress_clean.sh directly." >&2
exec "$NEW" "$@"
