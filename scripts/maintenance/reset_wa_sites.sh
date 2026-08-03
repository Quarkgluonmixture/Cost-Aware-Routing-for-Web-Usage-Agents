#!/usr/bin/env bash
# reset_wa_sites.sh — WA (WebArena) site reset, delegating to the VWA resets.
#
# HISTORY / WHY THIS IS A DELEGATION AND NOT ITS OWN IMPLEMENTATION
#
# The B-647 scaffold (2026-05-17) was written on the assumption that WA would
# need its own docker stack (`wa_reddit` / `wa_shopping` containers) and its own
# credentials, and it hard-failed every WA reset with rc=78 until that stack
# existed. The assumption was wrong, and it was retracted in two steps:
#
#   §387.3 (2026-07-27) — for reddit: WA reddit IS the `vwa-reddit` postmill
#     container. Same image, same port, and the two benchmarks' reddit task
#     files name a byte-identical `storage_state` under the same account.
#   B-1930 (2026-08-03) — for shopping: likewise. WA `test_shopping.raw.json`
#     and VWA `test_shopping.json` both declare `sites: ["shopping"]`, both
#     resolve `__SHOPPING__` to the same endpoint, and both use
#     `./.auth/shopping_state.json`. shopping_admin (7780) is that same
#     `vwa-shopping` container on its second port.
#
# VWA is a fork of WebArena, so for these sites the reset semantics are
# identical rather than merely analogous — there is no second stack to reset.
# Writing WA-specific reset bodies would mean maintaining a second copy of
# logic that mutates the same containers, and the copies would drift.
#
# The paper-grade launch path does NOT come through this file: queue scripts
# call `wa_reset_supported` + `reset_and_auth_gate`, which routes straight to
# `reset_vwa_sites`. This wrapper exists for operators invoking a WA reset by
# hand, and for anything still referencing the old `wa_*` site names.
#
# Usage:
#   bash scripts/maintenance/reset_wa_sites.sh <site>
#   - site: wa_shopping | wa_reddit | wa_shopping_admin
#           (bare shopping / reddit / shopping_admin also accepted)
#
# Env: identical to reset_vwa_sites.sh (VWA_RESET_MODE / VWA_RESET_ENABLE / ...).

set -euo pipefail

# Export the function so callers can source this file and invoke directly,
# mirroring reset_vwa_sites.sh contract: callers `source reset_wa_sites.sh`
# then call `reset_wa_sites <site> [label]`.
reset_wa_sites() {
  local site="${1:-}"
  local label="${2:-wa_reset}"

  # Normalise `wa_<site>` → `<site>`: the container is the same object either
  # way, and reset_vwa_sites keys on the bare site name.
  local vwa_site
  case "${site}" in
    wa_shopping|shopping)             vwa_site="shopping" ;;
    wa_shopping_admin|shopping_admin) vwa_site="shopping_admin" ;;
    wa_reddit|reddit)                 vwa_site="reddit" ;;
    *)
      echo "[reset_wa_sites][error] unknown site: ${site:-<empty>}" >&2
      echo "[reset_wa_sites][error] expected: wa_shopping | wa_reddit | wa_shopping_admin" >&2
      return 2
      ;;
  esac

  local _repo_root
  _repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  # shellcheck disable=SC1091
  source "${_repo_root}/scripts/maintenance/reset_vwa_sites.sh"

  echo "[${label}] WA ${site} → VWA ${vwa_site} reset (shared container; B-1930 / §387.3)"
  reset_vwa_sites "${vwa_site}" "${label}"
}

# Allow standalone invocation (not just sourced):
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  reset_wa_sites "$@"
fi
