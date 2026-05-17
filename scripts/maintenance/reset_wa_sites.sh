#!/usr/bin/env bash
# reset_wa_sites.sh — WA (WebArena) site reset gate
#
# Mirrors reset_vwa_sites.sh interface but targets WA Docker stack (separate
# from VWA upstream). Phase 1b WA shopping/shopping_admin extension was R3 →
# R1 framing decision pending advisor sync 2026-05-14; user 2026-05-17 picked
# "长期完整实现 reset_wa_sites.sh" per /stress A1.13 chunk (d) Q&A.
#
# CURRENT STATE (B-647 scaffold, 2026-05-17):
#   - **scaffold only** — per-site reset bodies are NOT implemented yet.
#   - rc=78 (matches existing VWA `_reset_vwa_local_shopping` "not implemented"
#     sentinel convention used by `_lib_paper_grade_gates.sh:212`).
#   - queue scripts (queue_baseline + 3 phantom) catch rc=78 + emit specific
#     "WA reset NOT IMPLEMENTED" FATAL → no silent skip (closes A1.13 P1-4-BC
#     codex F7 + gemini G6 silent-skip bug).
#
# FUTURE FULL IMPL (per chunk d planned scope):
#   1. Confirm WA Docker stack present on paper-grade target host
#      (a100-jiaming-test or remote). `docker ps | grep -iE 'webarena|wa_'`
#      should show wa_shopping / wa_reddit / wa_shopping_admin containers
#      OR confirm separate WA docker-compose project exists.
#   2. Implement per-site reset semantics:
#      • wa_reddit: `docker compose -p wa_reddit down + up -d` (Postmill DB reset)
#      • wa_shopping: Magento DB SQL restore (similar to VWA shop stub)
#      • wa_shopping_admin: Magento admin session + cart restore
#   3. Extend `p79.utils.auth_refresh` to support `benchmark="wa"`:
#      • Playwright login flows for WA reddit/shopping/shopping_admin user accounts
#      • Credentials env: WA_REDDIT_USER / WA_REDDIT_PASS / WA_SHOPPING_USER / ...
#      • `auth_required_gate('wa_shopping', ..., benchmark='wa')` semantics
#   4. Remove the `BENCHMARK=wa` hard-fail branch in queue scripts;
#      reset_and_auth_gate gains `--benchmark` arg routing to this script vs
#      reset_vwa_sites.sh.
#
# Usage (when full-impl lands):
#   bash scripts/maintenance/reset_wa_sites.sh <site>
#   - site: wa_shopping | wa_reddit | wa_shopping_admin
#
# Env (when full-impl lands):
#   VWA_RESET_MODE       — auto | local | remote (mirrors reset_vwa_sites.sh)
#   VWA_RESET_ENABLE     — set 0 to dry-run/disable
#   WA_DOCKER_PROJECT    — docker compose project name (default `wa`)
#   WA_<SITE>_USER/PASS  — Playwright login credentials per site

set -euo pipefail

# Export the function so callers can source this file and invoke directly,
# mirroring reset_vwa_sites.sh contract: callers `source reset_wa_sites.sh`
# then call `reset_wa_sites <site> [label]`.
reset_wa_sites() {
  local site="${1:-}"
  local label="${2:-wa_reset}"
  case "${site}" in
    wa_shopping|wa_reddit|wa_shopping_admin)
      ;;
    *)
      echo "[reset_wa_sites][error] unknown site: ${site:-<empty>}" >&2
      echo "[reset_wa_sites][error] expected: wa_shopping | wa_reddit | wa_shopping_admin" >&2
      return 2
      ;;
  esac

  # B-647 scaffold (A1.13 P1-4 chunk d, 2026-05-17): full reset body TBD.
  # Returns rc=78 "not implemented" sentinel — `_lib_paper_grade_gates.sh:212`
  # catches this rc + emits site-specific FATAL surface ("Implement reset_wa_${site}
  # body before paper-grade Phase 1b launch.").
  echo "[reset_wa_sites][error] reset for site=${site} NOT IMPLEMENTED (B-647 scaffold)" >&2
  echo "[reset_wa_sites][error] WA Docker stack reset semantics + auth_required_gate WA support are pending Phase 1b advisor sync." >&2
  echo "[reset_wa_sites][error] See header comment for full-impl roadmap; fill the per-site case branch below." >&2
  return 78
}

# Allow standalone invocation (not just sourced):
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  reset_wa_sites "$@"
fi
