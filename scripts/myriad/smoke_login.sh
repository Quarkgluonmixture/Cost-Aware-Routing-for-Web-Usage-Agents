#!/usr/bin/env bash
# P79 Myriad onboarding smoke test — LOGIN node side
#
# Run on Myriad after `ssh ucab352@myriad.rc.ucl.ac.uk`.
# Verifies onboarding steps 1-3 (README §139) from login-node perspective,
# plus module availability, quota, queue health, hub-spoke ssh.
#
# Output:  ~/p79_myriad_smoke_login_<timestamp>.log
# Next:    qsub scripts/myriad/smoke_compute.qsub

set -uo pipefail   # NOT -e — we want to keep going past failures

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${HOME}/p79_myriad_smoke_login_${TS}.log"

VWA_HOST="100.95.81.103"           # quark Tailscale IP
VWA_PORTS=(9980 7770 9999 4399)    # cls / shop / reddit / homepage
DGX_HOST="spark-9ea3"              # README §181 hub host

PASS=0; FAIL=0; WARN=0
ok()   { echo "  [PASS] $*"; PASS=$((PASS+1)); }
bad()  { echo "  [FAIL] $*"; FAIL=$((FAIL+1)); }
warn() { echo "  [WARN] $*"; WARN=$((WARN+1)); }

probe_tcp() {
  local host="$1" port="$2"
  timeout 5 bash -c ">/dev/tcp/${host}/${port}" 2>/dev/null
}

exec > >(tee "${LOG}") 2>&1

cat <<HDR
===== P79 Myriad smoke (LOGIN node) =====
Time:     $(date -Iseconds)
Host:     $(hostname)
User:     $(whoami)
Repo:     ${REPO_ROOT}
Log:      ${LOG}
HDR

echo
echo "----- [1] Filesystem & quota -----"
df -h "${HOME}" 2>/dev/null | head -3 || true
if command -v gquota >/dev/null 2>&1; then
  gquota || warn "gquota returned non-zero"
else
  warn "gquota not in PATH (Myriad-specific cmd)"
fi

echo
echo "----- [2] Module system -----"
if command -v module >/dev/null 2>&1; then
  ok "module command available"
  echo "  CUDA modules:"
  module avail cuda 2>&1 | grep -E "cuda/[0-9]" | head -10 || warn "no cuda modules listed"
  echo "  PyTorch modules:"
  module avail pytorch 2>&1 | grep -E "pytorch/[0-9]" | head -10 || warn "no pytorch modules listed"
  echo "  Python3 modules:"
  module avail python3 2>&1 | grep -E "python3/[0-9]" | head -10 || warn "no python3 modules listed"
else
  bad "module command missing — Lmod not loaded?"
fi

echo
echo "----- [3] Onboarding step 1 — .env -----"
if [[ -f "${REPO_ROOT}/.env" ]]; then
  ok ".env present at ${REPO_ROOT}/.env"
  echo "    keys defined:"
  grep -oE "^[A-Z_]+=" "${REPO_ROOT}/.env" | sed 's/=$//' | sed 's/^/      - /' || true
else
  bad ".env missing — copy from DGX or recreate (OPENAI_API_KEY / DASHSCOPE_API_KEY / P79_GLM_KEY)"
fi

echo
echo "----- [4] Onboarding step 2 — scripts/vwa_env_remote.sh -----"
if [[ -f "${REPO_ROOT}/scripts/vwa_env_remote.sh" ]]; then
  ok "vwa_env_remote.sh present"
  if grep -q "VWA_REMOTE_HOST=${VWA_HOST}" "${REPO_ROOT}/scripts/vwa_env_remote.sh" 2>/dev/null; then
    ok "VWA_REMOTE_HOST=${VWA_HOST} configured"
  else
    warn "VWA_REMOTE_HOST not pointing to ${VWA_HOST} — check if you want different routing"
  fi
else
  bad "scripts/vwa_env_remote.sh missing — copy template from DGX_SPARK_MACHINE_QUIRKS.md"
fi

echo
echo "----- [5] Onboarding step 3 — VWA reach (LOGIN side) -----"
echo "  General internet sanity:"
if curl -sS -o /dev/null -w "    https://www.google.com -> HTTP %{http_code} %{time_total}s\n" \
    --connect-timeout 5 --max-time 10 https://www.google.com 2>/dev/null; then
  ok "outbound HTTPS OK"
else
  bad "outbound HTTPS blocked (login node should not be this restricted)"
fi

echo "  Tailscale ${VWA_HOST} TCP probe:"
TS_OK=0
for port in "${VWA_PORTS[@]}"; do
  if probe_tcp "${VWA_HOST}" "${port}"; then
    ok "${VWA_HOST}:${port} reachable"
    TS_OK=$((TS_OK+1))
  else
    warn "${VWA_HOST}:${port} unreachable from login node"
  fi
done
echo "  -> ${TS_OK}/${#VWA_PORTS[@]} VWA ports reachable from login"

echo
echo "----- [6] Onboarding step 5 — torch CUDA wheel (login probe) -----"
if [[ -d "${REPO_ROOT}/.venv" ]]; then
  ok ".venv present at ${REPO_ROOT}/.venv (will reuse)"
  if "${REPO_ROOT}/.venv/bin/python3" -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())" 2>/dev/null; then
    ok "torch importable in .venv (real cuda check requires GPU node — see compute smoke)"
  else
    warn ".venv torch import failed — install torch+cuXXX wheel"
  fi
else
  warn ".venv not yet created — run 'pip install -e .' + torch wheel after smoke passes"
fi

echo
echo "----- [7] Hub-spoke — ssh to DGX (${DGX_HOST}) -----"
echo "  TCP probe ${DGX_HOST}:22:"
if probe_tcp "${DGX_HOST}" 22; then
  ok "${DGX_HOST}:22 reachable — rsync-to-hub viable"
else
  warn "${DGX_HOST}:22 unreachable from login node — Tier B push will need bastion or VPN"
fi
echo "  ssh handshake (BatchMode, ConnectTimeout=5):"
if timeout 8 ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new \
     "${DGX_HOST}" "echo SSH_OK_\$(hostname)" 2>/dev/null; then
  ok "ssh ${DGX_HOST} succeeded — hub-spoke rsync ready"
else
  warn "ssh ${DGX_HOST} failed — need to copy ~/.ssh/id_* + add Myriad pubkey to DGX authorized_keys"
fi

echo
echo "----- [8] Job scheduler health -----"
echo "  qstat (your jobs):"
qstat 2>&1 | head -8 || true
echo "  Queue summary (qstat -g c, GPU rows only):"
qstat -g c 2>&1 | grep -iE "gpu|^queue|---" | head -20 || warn "qstat -g c gave no GPU info"

echo
echo "===== SUMMARY (LOGIN) ====="
echo "  PASS=${PASS}  FAIL=${FAIL}  WARN=${WARN}"
echo "  Log: ${LOG}"
echo
if [[ "${TS_OK}" -ge 3 ]]; then
  echo "  Verdict: VWA reachable from login. Now submit compute-node probe:"
  echo "      qsub scripts/myriad/smoke_compute.qsub"
elif [[ "${TS_OK}" -gt 0 ]]; then
  echo "  Verdict: VWA partially reachable from login (${TS_OK}/${#VWA_PORTS[@]} ports)."
  echo "  Investigate which ports are blocked before submitting compute job."
else
  echo "  Verdict: VWA UNREACHABLE from login node."
  echo "  Tailscale path is blocked at UCL firewall. Options:"
  echo "    (a) Install tailscaled in user-space (rare — usually no admin)"
  echo "    (b) Reverse SSH tunnel from quark into Myriad (keep-alive needed)"
  echo "    (c) Run VWA docker on Myriad itself (different baseline — paper-grade impact)"
  echo "    (d) Drop B1-on-Myriad plan; use Myriad only for analysis tasks"
fi
