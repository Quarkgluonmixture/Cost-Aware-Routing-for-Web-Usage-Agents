#!/usr/bin/env bash
# run_mimo_b3_pilot.sh — DEV-ONLY B3 (MiMo-VL-7B-RL) classifieds FLOOR PILOT (§340, 2026-06-16).
# NOT paper-grade. Answers the one question the literature can't: does MiMo floor on OUR
# fixed-JSON scaffold (≫1% = viable cross-family B3; ~1% = floors → strategic §8 finding)?
#
# Reuses the tested queue_baseline.sh machinery via its SMOKE_CONFIG dev path — env sourcing
# (vwa_env_remote.sh → quark CLASSIFIEDS), quark cls reset + classifieds auth, runner, watchdog —
# in DEV mode (P79_PAPER_GRADE=0). Writes NO new launch logic.
#
# ⚠️ The BASELINE positional arg to queue_baseline is "B1" = a VALIDATION-ONLY LABEL
# (queue_baseline's allowlist is {B0,B1,B2}; it rejects "B3"). The ACTUAL model is
# local_mimo (MiMoVLAgent), set by the SMOKE_CONFIG pilot yaml's backend block — the
# label only touches cosmetics (watchdog aggregate-prefix) + the B1-cls collision check
# (none on DGX). The run dir is named from the SMOKE_CONFIG basename: B3_som_classifieds_pilot.
#
# Runs on DGX; hits quark VWA via Tailscale; does NOT touch the A100 paper-grade fire
# (different host + different VWA instance + different GPU).
#
# Usage:  bash scripts/maintenance/run_mimo_b3_pilot.sh
#   RESET_BEFORE=0  skip the quark cls reset (default 1 = clean start + fresh auth)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

PILOT_CFG="configs/exp_v2_B3_som_classifieds_pilot.yaml"
QUARK_CLS="http://100.95.81.103:9980"

if [[ ! -f "${PILOT_CFG}" ]]; then
  echo "[mimo-pilot][abort] pilot config not found: ${PILOT_CFG}" >&2; exit 1
fi

# Preflight: quark classifieds reachable (the DGX→quark dev VWA path must be up).
code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 8 "${QUARK_CLS}/" 2>/dev/null || echo "000")
if ! [[ "${code}" =~ ^(200|301|302)$ ]]; then
  echo "[mimo-pilot][abort] quark classifieds ${QUARK_CLS} not reachable (HTTP ${code})." >&2
  echo "  Start the VWA Docker stack on quark (Windows) — at least classifieds:9980 — and" >&2
  echo "  verify the port is published on the Tailscale interface + firewall allows inbound." >&2
  exit 1
fi
echo "[mimo-pilot] quark classifieds reachable (HTTP ${code}); launching B3 (MiMo-VL) som floor pilot ..."
echo "[mimo-pilot] model=local_mimo (MiMoVLAgent); 'B1' below = validation-only label, NOT the model."

# DEV mode + SMOKE_CONFIG (non-canonical run dir) + RESET_BEFORE (clean quark cls + fresh auth).
P79_PAPER_GRADE=0 \
RESET_BEFORE="${RESET_BEFORE:-1}" \
SMOKE_CONFIG="${PILOT_CFG}" \
  bash scripts/queues/queue_baseline.sh B1 som classifieds

echo
echo "[mimo-pilot] launched (runner detached + watchdog). Monitor:"
echo "  tail -f logs/B3_som_classifieds_pilot_*_runner.log"
echo "  floor SR = scored successes / 25 cls tasks."
