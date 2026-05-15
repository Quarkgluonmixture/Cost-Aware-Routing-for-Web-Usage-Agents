#!/usr/bin/env bash
# A100 self-host VWA setup runbook (one-time, ~1-2h wallclock)
#
# Run on UCL Condense A100 VM (Ubuntu 22.04) AFTER:
#   - SSH cert generated + ssh condense-a100 works (quark side)
#   - p79 repo cloned at ~/workspace/p79
#   - venv + pip install done
#   - NVIDIA driver installed + nvidia-smi works
#
# Result: cls + red + shop VWA Docker stack running on A100, agent reaches via
# localhost. NO Tailscale to quark needed. paper §3 disclose this deployment.
#
# Usage:
#   bash scripts/setup/a100_self_host_vwa.sh           # full setup, all 3 sites
#   bash scripts/setup/a100_self_host_vwa.sh classifieds  # only cls
#
# Idempotent: re-running detects existing containers + skips already-done steps.
#
# ============================================================================
# KNOWN ISSUES (discovered 2026-05-15, chronicled in 实验笔记 §140.11):
#
# 1. VWA HF dataset 404
#    `webarena/{Shopping,Reddit,Wikipedia,Classifieds}` HF repos no longer
#    exist. setup_vwa.sh now pulls from CMU metis + archive.org mirrors:
#      - shopping_final_0712.tar  (~67GB, http://metis.lti.cs.cmu.edu/...)
#      - postmill-...withimg.tar  (~53GB, http://metis.lti.cs.cmu.edu/...)
#      - wikipedia_en_all_maxi_2025-08.zim  (~95GB, kiwix.org —
#        NOT VWA upstream 2022-05 because P79 queue scripts hardcode
#        WIKIPEDIA_ZIM_VERSION=2025-08 (笔记 §81 Kiwix-VWA version drift fix).
#        Prod quark Windows docker also runs 2025-08.)
#      - classifieds_docker_compose.zip     (~25MB, archive.org)
#
# 2. Docker 29.x + containerd-snapshotter pulls jykoh/classifieds DEADLOCK
#    Symptom: `docker compose up` (or `docker pull jykoh/classifieds:latest`)
#    hangs forever at layer ec57d9f250af extraction, dockerd logs
#    `failed to cleanup extract-XXX: NotFound`. Stale lease 持久化 8h.
#    Workaround: DGX `docker save jykoh/classifieds:latest | ssh condense-a100
#    'docker load'` (streams over SSH, bypasses A100's containerd extraction
#    pipeline). Requires DGX-side image cache.
#    DO NOT run `docker rm` on partial-extracted container — leaves locked
#    snapshot in containerd meta that survives `systemctl restart docker`.
#
# 3. `pip install -r external/visualwebarena/requirements.txt` is destructive
#    VWA pinned versions (torch==2.0.1+cu117, transformers==4.34.0) downgrade
#    the p79 venv. MUST use `pip install -e external/visualwebarena/ --no-deps`
#    so only the webarena package metadata installs; let p79 pyproject control
#    deps.
#
# 4. Reset wrapper: VWA_RESET_MODE=local auto-detected from SSH key absence
#    scripts/maintenance/reset_vwa_sites.sh now supports A100-local reset
#    (cls = curl reset endpoint ~0.1s, reddit = docker rm + run ~58s,
#    shop = stub pending Phase 1b SQL-restore). RESET_BEFORE=1 works.
# ============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VWA_DIR="${REPO_ROOT}/external/visualwebarena/environment_docker"

SITES_TO_DEPLOY=("${@:-classifieds reddit shopping}")

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ---------------------------------------------------------------------------
# Step 1: Verify prerequisites
# ---------------------------------------------------------------------------

log "=== Step 1: Verify prereqs ==="

if ! command -v docker &>/dev/null; then
  log "Docker not found. Install:"
  log "  sudo apt update && sudo apt install -y docker.io docker-compose-v2"
  log "  sudo systemctl enable --now docker"
  log "  sudo usermod -aG docker \$USER && newgrp docker"
  exit 1
fi

if ! docker ps &>/dev/null; then
  log "Cannot connect to Docker daemon (current user lacks docker group?)."
  log "  sudo usermod -aG docker \$USER && newgrp docker"
  exit 1
fi

log "Docker version: $(docker version --format '{{.Server.Version}}' 2>/dev/null)"

if ! command -v docker-compose &>/dev/null && ! docker compose version &>/dev/null; then
  log "Docker Compose v2 not available. Install: sudo apt install -y docker-compose-v2"
  exit 1
fi

DOCKER_COMPOSE="docker compose"
if ! docker compose version &>/dev/null; then
  DOCKER_COMPOSE="docker-compose"
fi

if [ ! -d "${VWA_DIR}" ]; then
  log "VWA repo not found at ${VWA_DIR}"
  log "  cd ${REPO_ROOT} && git submodule update --init --recursive"
  exit 1
fi

log "VWA repo: ${VWA_DIR}"

# ---------------------------------------------------------------------------
# Step 2: Disk space check
# ---------------------------------------------------------------------------

log "=== Step 2: Disk space check ==="

AVAIL_GB=$(df --output=avail -BG / | tail -1 | tr -d ' G')
log "Free disk: ${AVAIL_GB} GB"

REQ_GB=130
if [ "${AVAIL_GB}" -lt "${REQ_GB}" ]; then
  log "WARN: Free disk ${AVAIL_GB} GB < required ${REQ_GB} GB for full VWA stack"
  log "      Continuing — may fail mid-setup. Free space first if possible."
fi

# ---------------------------------------------------------------------------
# Step 3: Deploy each site
# ---------------------------------------------------------------------------

deploy_classifieds() {
  log "  Deploying classifieds (port 9980, OSClass + MySQL)..."
  cd "${VWA_DIR}/classifieds_docker_compose"
  if [ ! -d "./mysql" ]; then
    log "  WARN: ./mysql/ DB init dir not found. Check VWA repo state."
  fi
  ${DOCKER_COMPOSE} up -d
  cd "${REPO_ROOT}"
}

deploy_reddit() {
  log "  Deploying reddit (port 9999, Postmill)..."
  # NOTE: reddit/postmill compose path varies by VWA version; adapt as needed
  REDDIT_DIR="${VWA_DIR}/reddit"  # may need adjustment
  if [ ! -d "${REDDIT_DIR}" ]; then
    log "  Reddit compose dir not found at ${REDDIT_DIR}"
    log "  Manual: docker run --name vwa-reddit -p 9999:80 -d <postmill-pop-image>"
    log "  See Phase 1 quark setup; image used was 'postmill-pop:latest'"
    return 1
  fi
  cd "${REDDIT_DIR}"
  ${DOCKER_COMPOSE} up -d
  cd "${REPO_ROOT}"
}

deploy_shopping() {
  log "  Deploying shopping (port 7770, Magento + DB)..."
  SHOP_DIR="${VWA_DIR}/shopping"  # may need adjustment
  if [ ! -d "${SHOP_DIR}" ]; then
    log "  Shopping compose dir not found at ${SHOP_DIR}"
    log "  Manual: docker run --name vwa-shopping -p 7770:80 -d <magento-image>"
    log "  See Phase 1 quark setup. NOTE: Magento needs base_url config matching"
    log "  A100 VM IP / 127.0.0.1 — different from quark Phase 1 (was 10.x.x.x)"
    return 1
  fi
  cd "${SHOP_DIR}"
  ${DOCKER_COMPOSE} up -d
  cd "${REPO_ROOT}"
}

log "=== Step 3: Deploy VWA sites ==="

for site in ${SITES_TO_DEPLOY[@]}; do
  log "Site: $site"
  case "$site" in
    classifieds) deploy_classifieds ;;
    reddit)      deploy_reddit ;;
    shopping)    deploy_shopping ;;
    *)           log "Unknown site: $site"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Step 4: Wait for sites to be ready + smoke check
# ---------------------------------------------------------------------------

log "=== Step 4: Smoke check (wait up to 120s) ==="

check_url() {
  local url=$1
  local name=$2
  for i in {1..24}; do
    if curl -sS --max-time 3 -I "$url" 2>/dev/null | head -1 | grep -qE "200|302|301"; then
      log "  $name OK: $url responds"
      return 0
    fi
    sleep 5
  done
  log "  $name FAILED: $url did not respond after 120s"
  return 1
}

for site in ${SITES_TO_DEPLOY[@]}; do
  case "$site" in
    classifieds) check_url "http://localhost:9980" "classifieds" || true ;;
    reddit)      check_url "http://localhost:9999" "reddit" || true ;;
    shopping)    check_url "http://localhost:7770" "shopping" || true ;;
  esac
done

# ---------------------------------------------------------------------------
# Step 5: Magento base_url adjustment (shopping only)
# ---------------------------------------------------------------------------

if [[ " ${SITES_TO_DEPLOY[@]} " =~ " shopping " ]]; then
  log "=== Step 5: Magento base_url note ==="
  log "  Magento internal base_url MUST match how the agent will access it."
  log "  Phase 1 used http://100.95.81.103:7770 (quark Tailscale IP)."
  log "  A100 self-host: agent uses http://localhost:7770 OR http://127.0.0.1:7770."
  log "  If 302 redirect loops occur, run inside container:"
  log "    docker exec -it vwa-shopping bash -c \"php /var/www/magento2/bin/magento setup:store-config:set --base-url=http://127.0.0.1:7770/\""
  log "    docker exec -it vwa-shopping bash -c \"php /var/www/magento2/bin/magento setup:store-config:set --base-url-secure=http://127.0.0.1:7770/\""
  log "    docker exec -it vwa-shopping bash -c \"php /var/www/magento2/bin/magento cache:flush\""
fi

# ---------------------------------------------------------------------------
# Step 6: Auth file generation (Playwright session state, paper-grade)
# ---------------------------------------------------------------------------

log "=== Step 6: Playwright auth file generation ==="
log "  Phase 1 stored auth state in DGX:.auth/ (gitignored)."
log "  A100 self-host: regenerate auth files locally."
log "  cd \$(REPO_ROOT) && python3 p79/utils/auth_refresh.py --site classifieds"
log "  Repeat for each site. Auth files saved to .auth/ on A100."

# ---------------------------------------------------------------------------
# Step 7: Reproducibility verification (paper §3 disclosure prep)
# ---------------------------------------------------------------------------

log "=== Step 7: Cross-environment HTML byte-equivalence check ==="
log "  To verify A100 deployment matches Phase 1 quark deployment:"
log "  1. From DGX: curl http://100.95.81.103:9980 | sha256sum"
log "  2. From A100: curl http://localhost:9980 | sha256sum"
log "  3. Hashes should match (modulo timestamp / session-cookie deltas)."
log "  Document results in paper §3 / Appendix D."

log ""
log "=== A100 self-host VWA setup DONE ==="
log "Sites deployed: ${SITES_TO_DEPLOY[@]}"
log ""
log "Next:"
log "  1. Verify each port responds: curl -I http://localhost:9980 / 9999 / 7770"
log "  2. Generate auth files: python3 p79/utils/auth_refresh.py --site <site>"
log "  3. Launch 16-cell rerun (post-advisor email + threshold lock)"
log "  4. Mechanistic Stage 2B can run NOW (doesn't need VWA, uses archived data)"
