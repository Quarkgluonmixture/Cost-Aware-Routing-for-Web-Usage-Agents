#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VWA_DIR="${PROJECT_DIR}/external/visualwebarena"
ENV_DIR="${VWA_DIR}/environment_docker"

SITES="all"
AUTO_SETUP=0
HOSTNAME_VALUE="localhost"
CHECK_ONLY=0

usage() {
  cat <<USAGE
Usage: bash scripts/vwa/start_vwa_docker.sh [options]

Options:
  --sites <list>       Comma-separated sites: all|shopping|shopping_admin|reddit|wikipedia|classifieds|homepage
                       (shopping_admin shares the shopping container, adds host port 7780 → same Magento)
  --auto-setup         Run scripts/vwa/setup_vwa.sh automatically when required assets are missing
  --hostname <value>   Hostname used to patch VWA templates (default: localhost)
  --check-only         Only validate prerequisites, do not start services
  -h, --help           Show this help

Examples:
  bash scripts/vwa/start_vwa_docker.sh --sites all
  bash scripts/vwa/start_vwa_docker.sh --sites shopping,reddit --auto-setup
  bash scripts/vwa/start_vwa_docker.sh --check-only
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sites)
      SITES="${2:-all}"
      shift 2
      ;;
    --auto-setup)
      AUTO_SETUP=1
      shift
      ;;
    --hostname)
      HOSTNAME_VALUE="${2:-localhost}"
      shift 2
      ;;
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

contains_site() {
  local needle="$1"
  if [[ "${SITES}" == "all" ]]; then
    return 0
  fi
  IFS=',' read -r -a selected <<< "${SITES}"
  for item in "${selected[@]}"; do
    if [[ "$(echo "${item}" | xargs)" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

find_running_container() {
  local name
  for name in "$@"; do
    if docker ps --format '{{.Names}}' | grep -q "^${name}$"; then
      echo "${name}"
      return 0
    fi
  done
  return 1
}

find_existing_container() {
  local name
  for name in "$@"; do
    if docker ps -a --format '{{.Names}}' | grep -q "^${name}$"; then
      echo "${name}"
      return 0
    fi
  done
  return 1
}

check_prerequisites() {
  local missing=0

  if ! command -v docker >/dev/null 2>&1; then
    echo "[MISSING] docker command not found" >&2
    missing=1
  fi

  if [[ ! -d "${VWA_DIR}" ]] || [[ -z "$(ls -A "${VWA_DIR}" 2>/dev/null || true)" ]]; then
    echo "[MISSING] external/visualwebarena repository" >&2
    missing=1
  fi

  if contains_site "shopping" || contains_site "shopping_admin"; then
    if ! docker images --format '{{.Repository}}:{{.Tag}}' | grep -q '^shopping_final_0712:latest$'; then
      echo "[MISSING] shopping_final_0712:latest image" >&2
      missing=1
    fi
  fi

  if contains_site "reddit"; then
    if ! docker images --format '{{.Repository}}:{{.Tag}}' | grep -q '^postmill-populated-exposed-withimg:latest$'; then
      echo "[MISSING] postmill-populated-exposed-withimg:latest image" >&2
      missing=1
    fi
  fi

  if contains_site "wikipedia"; then
    if [[ ! -f "${ENV_DIR}/data/wikipedia_en_all_maxi_2025-08.zim" ]]; then
      echo "[MISSING] wikipedia ZIM file (2025-08; P79 queue scripts hardcode this version, see 笔记 §81)" >&2
      missing=1
    fi
  fi

  if contains_site "classifieds"; then
    if [[ ! -d "${ENV_DIR}/classifieds_docker_compose" ]]; then
      echo "[MISSING] classifieds_docker_compose directory" >&2
      missing=1
    fi
  fi

  if (( missing == 1 )); then
    if (( AUTO_SETUP == 1 )); then
      echo "Missing prerequisites detected, running setup..."
      local setup_target="all"
      if [[ "${SITES}" != "all" ]]; then
        # setup_vwa expects dataset names; homepage has no dataset package
        setup_target="${SITES//homepage/}" 
        setup_target="${setup_target//,,/,}"
        setup_target="${setup_target#,}"
        setup_target="${setup_target%,}"
        [[ -z "${setup_target}" ]] && setup_target="all"
      fi
      bash "${PROJECT_DIR}/scripts/vwa/setup_vwa.sh" --target-dataset "${setup_target}"
      return 0
    fi
    echo "Prerequisite check failed. Re-run with --auto-setup or run scripts/vwa/setup_vwa.sh first." >&2
    return 1
  fi

  echo "Prerequisite check passed."
  return 0
}

start_shopping() {
  local want_admin=0
  contains_site "shopping_admin" && want_admin=1
  if (( want_admin == 1 )); then
    echo "[START] shopping (http://${HOSTNAME_VALUE}:7770) + shopping_admin (http://${HOSTNAME_VALUE}:7780 → same container)"
  else
    echo "[START] shopping (http://${HOSTNAME_VALUE}:7770)"
  fi
  local container_name=""
  container_name="$(find_running_container vwa-shopping shopping || true)"
  if [[ -n "${container_name}" ]]; then
    echo "${container_name} already running; reconfiguring base URL"
  else
    container_name="$(find_existing_container vwa-shopping shopping || true)"
    if [[ -n "${container_name}" ]]; then
      docker start "${container_name}" >/dev/null 2>&1
    else
      container_name="vwa-shopping"
      local port_args="-p 7770:80"
      (( want_admin == 1 )) && port_args="${port_args} -p 7780:80"
      # B-753 (/stress A1.17 cold-start P1-10 C* OOB, 2026-05-17): P79_VWA_TZ first;
      # legacy QUARK_TZ as fallback for transition. See init_paper_grade_env.
      docker run --name "${container_name}" -e TZ="${P79_VWA_TZ:-${QUARK_TZ:-Europe/London}}" ${port_args} -d shopping_final_0712 >/dev/null
    fi
    # B-1952 (2026-08-03, first live shopping fire): wait for mysqld to be
    # QUERY-READY, don't sleep a constant.
    #
    # `sleep 10` was enough for `docker start` on an existing container and is
    # nowhere near enough for a fresh `docker run` off the 141 GB
    # shopping_final_0712 image. Observed on the first real shopping launch:
    # 10s after `docker run`, `pgrep mysqld` inside the container was still 0,
    # so both base_url patches below (config:set AND the authoritative SQL
    # UPDATE) executed against a dead socket, were swallowed by their
    # `>/dev/null 2>&1`, and the verify SELECT that follows read back the
    # image-baked `http://metis.lti.cs.cmu.edu:7770/`. `start_vwa_docker.sh`
    # then `return 1`, `_reset_vwa_local_shopping` reported "rebuild FAILED",
    # and the chain fail-closed at condition [1/7] — 90 seconds in, having used
    # none of B-1931's 900s budget. **The reset failed by being too FAST, not
    # too slow**, which is why raising the outer timeout did not touch it.
    #
    # Failing this way also resurrects §103: the metis base_url is what the
    # IMAGE ships. Every `docker rm -f` + rebuild restores it, and the only
    # thing that removes it is the patch below. A patch that silently no-ops is
    # therefore indistinguishable from the original bug — the site 302s to a
    # hostname the A100 cannot resolve.
    #
    # Poll the same way `_reset_vwa_local_classifieds` waits on classifieds_db:
    # `SELECT 1` proves grant tables loaded AND the DB is queryable, which
    # `mysqladmin ping` does not (it returns as soon as the server accepts a
    # connection, before it can answer). 60 × 5s = 5 min ceiling; measured cold
    # start is well under that, and the enclosing reset budget is 900s.
    local _my_ok=0 _i
    for _i in $(seq 1 60); do
      if docker exec -e MYSQL_PWD=MyPassword "${container_name}" \
           mysql -u magentouser magentodb -sN -e "SELECT 1" >/dev/null 2>&1; then
        _my_ok=1
        echo "[START] Magento mysqld query-ready after $((_i*5))s"
        break
      fi
      sleep 5
    done
    if (( _my_ok == 0 )); then
      echo "[START] ✗ Magento mysqld NOT query-ready after 300s — refusing to patch base_url" >&2
      echo "[START]   (patching now would silently no-op and leave the image-baked" >&2
      echo "[START]    metis.lti.cs.cmu.edu base_url active — see B-1952 / 笔记 §103)" >&2
      return 1
    fi
  fi
  # BUG-5 fix (2026-05-16, codex Attack 2): strip `|| true` from Magento patches.
  # Silently swallowing failures was masking patch-not-applied bugs → image-baked
  # metis URL stays active → BUG-4 hang cascade for Phase 1b shop.
  # B-1952: keep stderr instead of discarding it. BUG-5 removed `|| true` so a
  # failure could no longer be *ignored*, but `>/dev/null 2>&1` still meant a
  # failure could not be *read*: the first live shopping fire produced exactly
  # one line — "rebuild FAILED (start_vwa_docker.sh non-zero)" — with the actual
  # cause (mysqld not up, both patches hitting a dead socket) nowhere on disk.
  # Non-fatal here because the authoritative check is the DB verify below; this
  # is diagnosis, not control flow.
  local _cfgset_err
  if ! _cfgset_err=$(docker exec "${container_name}" \
        /var/www/magento2/bin/magento setup:store-config:set \
        --base-url="http://${HOSTNAME_VALUE}:7770" 2>&1 >/dev/null); then
    echo "[START] ⚠️  setup:store-config:set non-zero: ${_cfgset_err:0:200}" >&2
  fi
  # B-747 (/stress A1.17 cold-start P1-4 AB* OOB, 2026-05-17, B-717 sibling):
  # MYSQL_PWD env replaces `-pMyPassword` argv. `docker exec -e MYSQL_PWD ...`
  # propagates env into container; mysql client reads MYSQL_PWD via libmysqlclient.
  # `ps auxe` on A100 VM (UCL Condense multi-user surface) no longer leaks
  # plaintext Magento DB password.
  local _sql_err
  if ! _sql_err=$(docker exec -e MYSQL_PWD=MyPassword "${container_name}" mysql -u magentouser magentodb -e "UPDATE core_config_data SET value='http://${HOSTNAME_VALUE}:7770/' WHERE path IN ('web/unsecure/base_url', 'web/secure/base_url');" 2>&1 >/dev/null); then
    echo "[START] ⚠️  base_url SQL UPDATE non-zero: ${_sql_err:0:200}" >&2
  fi
  # Verify DB-side base_url actually patched (config:set caches stale via app/etc; SQL UPDATE is authoritative)
  local actual_url
  actual_url=$(docker exec -e MYSQL_PWD=MyPassword "${container_name}" mysql -u magentouser magentodb -sN -e \
               "SELECT value FROM core_config_data WHERE path='web/unsecure/base_url';" 2>/dev/null || echo "?")
  if [[ "${actual_url}" != "http://${HOSTNAME_VALUE}:7770/" ]]; then
    echo "[START] ✗ Magento base_url patch FAILED: got '${actual_url}' want 'http://${HOSTNAME_VALUE}:7770/'" >&2
    # B-1952: name the §103 failure mode explicitly. `metis.lti.cs.cmu.edu` is
    # what the IMAGE ships, so reading it back means the patch no-opped and the
    # storefront will 302 to a host the A100 cannot resolve — the exact state
    # §103 diagnosed in 2026-04 and that returns on every rebuild.
    if [[ "${actual_url}" == *metis* ]]; then
      echo "[START]   ↳ this is the image-baked value (§103): the patch did not take." >&2
      echo "[START]   ↳ usual cause = mysqld not query-ready yet; the poll above should" >&2
      echo "[START]     have prevented it, so check whether the container is healthy." >&2
    fi
    return 1
  fi
  docker exec "${container_name}" /var/www/magento2/bin/magento cache:flush >/dev/null 2>&1
  # BUG-13 fix (Claude NEW4): ES indexes baked with metis URLs; need reindex post base_url change.
  # B-311 (A1.17 P1-8, gemini OOB unique): pre-fix `indexer:reindex` was fire-and-forget;
  # reindex takes 5-10min on 68GB shopping image. Script returned before completion;
  # if agent runner started immediately, search-autocomplete / category-update tasks
  # would silently fail (no model error, just empty results) → SR confounded by
  # infra not model. Now poll `indexer:status` until all "Ready" or 10min timeout.
  docker exec "${container_name}" /var/www/magento2/bin/magento indexer:reindex >/dev/null 2>&1 || \
    echo "[START] ⚠️  Magento indexer:reindex command returned non-zero (continuing to poll status)" >&2

  # Poll indexer:status — each row format "Indexer_Name: Status" where Status ∈ {Ready, Reindex required, Processing}
  # Done = ALL non-empty rows say "Ready".
  # B-1953 (2026-08-03, measured on the first real shopping reset): bound the
  # poll by WALL CLOCK, not by iteration count. `60 iterations x sleep 10` was
  # read as "10 min max" — including by B-1931, which sized the enclosing reset
  # timeout at 900s on that assumption. But each iteration ALSO runs
  # `docker exec ... magento indexer:status`, and on a container whose IO is
  # saturated by its own reindex that call takes many seconds. Real ceiling was
  # `60 x (10s + status_duration)` = unbounded. Measured: a reset still running
  # at 24.5 min, i.e. it would have been killed by B-1931's 900s budget mid-way,
  # right after `docker rm -f` had already destroyed the container.
  local poll_deadline=$(( $(date +%s) + ${MAGENTO_REINDEX_MAX_S:-4200} ))
  local poll_i=0
  while (( $(date +%s) < poll_deadline )); do
    poll_i=$(( poll_i + 1 ))
    # B-1955 (2026-08-04): 主判据改走 `indexer_state` 表, 不再解析 CLI 输出。
    # 本镜像的 `magento indexer:status` 输出是**表格**:
    #     +------------------------+----------------+--------+
    #     | ID                     | Title          | Status |
    #     | catalogsearch_fulltext | Catalog Search | Ready  |
    # 行内**没有冒号** → 下方旧判据 `awk -F: '/:/ && NF'` 匹配 0 行 → total_rows=0 →
    # 完成条件 `total_rows>0 && non_ready==0` **永远为假** → 每个 condition 的 reset
    # 都空转到 deadline, 并打印一句假的 "did NOT reach all-Ready"。
    # 实测 2026-08-03 fire: reindex 23:17:57→23:57 (40min) 就 11/11 valid, 脚本却仍在轮询。
    # 这也是 B-1953 (900→2400s) / B-1954 (2400→6000s) 两次加 timeout 的真实病根 ——
    # 用坏掉的仪器测量, 测出的是仪器故障, 不是被测对象的属性。
    # 表里 status ∈ {valid, invalid, working} ↔ CLI {Ready, Reindex required, Processing}。
    local total_rows=0 non_ready=0 sql_out
    sql_out=$(docker exec -e MYSQL_PWD=MyPassword "${container_name}" \
      mysql -u magentouser magentodb -sN \
      -e "SELECT COUNT(*), SUM(status<>'valid') FROM indexer_state;" 2>/dev/null | head -1)
    if [[ "${sql_out}" =~ ^([0-9]+)[[:space:]]+([0-9]+)$ ]]; then
      total_rows="${BASH_REMATCH[1]}"
      non_ready="${BASH_REMATCH[2]}"
    else
      # Fallback: mysqld 不可达 / 表结构变更 → 退回 CLI。表格与冒号两种格式都试,
      # 因为 B-748 记录过 `Name: Status` 形态 (可能来自别的 Magento 版本)。
      local idx_status
      idx_status=$(docker exec "${container_name}" /var/www/magento2/bin/magento indexer:status 2>/dev/null || echo "")
      total_rows=$(echo "${idx_status}" | awk -F'|' '/^\|/ && NF>=4 && $4 !~ /^ *Status *$/ {n++} END {print n+0}')
      non_ready=$(echo "${idx_status}" | awk -F'|' '/^\|/ && NF>=4 && $4 !~ /^ *Status *$/ && $4 !~ /Ready/ {n++} END {print n+0}')
      if (( total_rows == 0 )); then
        total_rows=$(echo "${idx_status}" | awk -F: '/:/ && NF {n++} END {print n+0}')
        non_ready=$(echo "${idx_status}" | awk -F: '/:/ && NF && $2 !~ /Ready/ {n++} END {print n+0}')
      fi
      if (( total_rows == 0 )); then
        echo "[START] ⚠️  Magento indexer status unreadable (SQL+CLI both) at poll ${poll_i}" >&2
        sleep 10
        continue
      fi
    fi
    # B-748 (/stress A1.17 cold-start P1-5+18 B* OOB, 2026-05-17) 的两条护栏在
    # B-1955 改判据后**依然生效**, 不要删:
    #   (a) 不用 `grep | grep -v | wc -l` 管线 —— 在 `set -euo pipefail` 下 all-Ready
    #       成功路径会让 `grep -v` rc=1 传播出去, 脚本在健康路径上退出;
    #   (b) `total_rows > 0` 必须保留 —— 读不到状态时 non_ready 天然为 0, 若无此条件
    #       会被误判成 "all Ready" 提前 break, reindex 未完就放行 → 搜索/autocomplete
    #       任务返回空结果 → 静默 SR 污染。SQL 判据下同理: 表为空时 SUM() 返回 NULL,
    #       正则不匹配 → 落 fallback, 而非当作 0 个 non-ready。
    if (( total_rows > 0 && non_ready == 0 )); then
      echo "[START] Magento indexer all Ready after $(( $(date +%s) - (poll_deadline - ${MAGENTO_REINDEX_MAX_S:-4200}) ))s / ${poll_i} polls (${total_rows} indexers)"
      break
    fi
    sleep 10
  done
  if (( $(date +%s) >= poll_deadline )); then
    echo "[START] ⚠️  Magento indexer:reindex did NOT reach all-Ready within ${MAGENTO_REINDEX_MAX_S:-4200}s — shop search/autocomplete tasks may fail" >&2
  fi
}

start_reddit() {
  echo "[START] reddit/forum (http://${HOSTNAME_VALUE}:9999)"
  local container_name=""
  container_name="$(find_running_container vwa-reddit forum || true)"
  if [[ -n "${container_name}" ]]; then
    echo "${container_name} already running"
  else
    container_name="$(find_existing_container vwa-reddit forum || true)"
    if [[ -n "${container_name}" ]]; then
      docker start "${container_name}" >/dev/null 2>&1
    else
      # B-753 (/stress A1.17 cold-start P1-10 C* OOB, 2026-05-17): P79_VWA_TZ first.
      docker run --name vwa-reddit -e TZ="${P79_VWA_TZ:-${QUARK_TZ:-Europe/London}}" -p 9999:80 -d postmill-populated-exposed-withimg >/dev/null
    fi
  fi
}

start_wikipedia() {
  echo "[START] wikipedia (http://${HOSTNAME_VALUE}:8888)"
  # BUG-16 fix (2026-05-16, Claude NEW7): refuse start while wget still writing
  # zim file. kiwix-serve RW-mounts data dir; concurrent wget append + kiwix
  # read = potential fs race / corrupt response. Sequence: wget exit → start.
  if pgrep -af "wget.*wikipedia_en_all_maxi.*\.zim" >/dev/null 2>&1; then
    echo "[START] ✗ refusing kiwix-serve start: wget still downloading ZIM (race risk)" >&2
    return 1
  fi
  local container_name=""
  container_name="$(find_running_container vwa-wikipedia wikipedia || true)"
  if [[ -n "${container_name}" ]]; then
    echo "${container_name} already running"
  else
    container_name="$(find_existing_container vwa-wikipedia wikipedia || true)"
    if [[ -n "${container_name}" ]]; then
      docker start "${container_name}" >/dev/null 2>&1
    else
      # Symlink-trick support: if host data dir contains symlinks (e.g. zim file →
       # /mnt/scratch/... when host /home filled up), container needs the target
      # path bind-mounted at the same location for symlink to resolve. Else
      # kiwix-serve sees symlink, follows to /mnt/scratch/... inside container
      # namespace, fails to find the file, Exits(0). Auto-detect + add mount.
      local extra_mounts=""
      if [[ -L "${ENV_DIR}/data/wikipedia_en_all_maxi_2025-08.zim" ]]; then
        local symtarget
        symtarget=$(readlink -f "${ENV_DIR}/data/wikipedia_en_all_maxi_2025-08.zim")
        local symdir
        symdir=$(dirname "${symtarget}")
        if [[ "${symdir}" != "${ENV_DIR}/data" ]]; then
          echo "[START] wikipedia: symlink → ${symtarget}; bind-mounting ${symdir} into container"
          extra_mounts="--volume=${symdir}:${symdir}:ro"
        fi
      fi
      # B-753 (/stress A1.17 cold-start P1-10 C* OOB, 2026-05-17): P79_VWA_TZ first.
      docker run -d --name vwa-wikipedia -e TZ="${P79_VWA_TZ:-${QUARK_TZ:-Europe/London}}" --volume="${ENV_DIR}/data/:/data" ${extra_mounts} -p 8888:80 ghcr.io/kiwix/kiwix-serve:3.3.0 wikipedia_en_all_maxi_2025-08.zim >/dev/null
    fi
  fi
}

start_classifieds() {
  echo "[START] classifieds (http://${HOSTNAME_VALUE}:9980)"
  local classifieds_running=0
  # B-752 (/stress A1.17 cold-start P1-15 A, 2026-05-17): broader regex matches
  # both compose v1 (`classifieds_db_1`) and compose v2 (`classifieds-app-1`) +
  # legacy bare `classifieds`. Pre-fix `^classifieds$` matched only the app
  # container if it was named bare (uncommon under compose); when project_name=
  # classifieds compose names app `classifieds-app-1` → grep miss → classifieds_running=0
  # → fall into `docker compose up --build` + DB seed retry → SEEDING ON ALREADY-
  # POPULATED DB = silent wipe + reseed = paper-grade data destruction on 2nd
  # invoke. Broader detect prevents accidental re-seed.
  if docker ps --format '{{.Names}}' | grep -qE '^classifieds(-app-1|_app_1|_db_1|-db-1|$)'; then
    classifieds_running=1
    echo "classifieds already running; reconfiguring compose hostname"
  fi

  local compose_dir="${ENV_DIR}/classifieds_docker_compose"
  if [[ ! -d "${compose_dir}" ]]; then
    echo "Classifieds compose directory missing: ${compose_dir}" >&2
    return 1
  fi

  # B-751 (/stress A1.17 cold-start P1-8 AB* OOB, 2026-05-17): template render
  # pattern replaces in-place sed. Pre-fix `sed -i "s|<your-server-hostname>|...|g"`
  # was non-idempotent on HOSTNAME_VALUE change — first run replaces placeholder
  # with host A; second run with host B can't find placeholder (already gone),
  # only the `CLASSIFIEDS=http://[^:]+:9980/` regex catches host A. Pristine
  # template + cp-then-sed ensures any HOSTNAME_VALUE rewrites cleanly.
  local compose_yml="${compose_dir}/docker-compose.yml"
  local compose_template="${compose_dir}/docker-compose.yml.template"
  if [[ ! -f "${compose_template}" ]]; then
    # First-time migration: save current as template (assumes pristine on first run).
    cp "${compose_yml}" "${compose_template}"
    echo "[START] cls compose: saved pristine template at ${compose_template} (B-751 first-time migration)"
  fi
  cp -f "${compose_template}" "${compose_yml}"
  sed -i "s|<your-server-hostname>|${HOSTNAME_VALUE}|g" "${compose_yml}"
  sed -i -E "s|CLASSIFIEDS=http://[^:]+:9980/|CLASSIFIEDS=http://${HOSTNAME_VALUE}:9980/|g" "${compose_yml}"
  (cd "${compose_dir}" && docker compose up --build -d)
  if (( classifieds_running == 0 )); then
    sleep 15
    # B-312 (A1.17 P1-12, BUG-5 sibling-propagation): pre-fix had `|| true` swallowing
    # SQL load failure → empty cls DB → all cls tasks 0% SR. BUG-5 fix stripped
    # `|| true` from shopping Magento patches (line 190-191) but cls DB seed missed
    # propagation. Now: retry up to 3 times with 5s sleep (DB warming race), then
    # FATAL return 1 if all retries fail. Aborts startup loudly rather than silently
    # leaving cls site broken.
    local seed_rc=0 seed_retry
    for seed_retry in 1 2 3; do
      # B-747 cont (P1-4 sibling): MYSQL_PWD env injection (cls DB seed path,
      # 3rd callsite). Same vulnerability class as start_shopping mysql calls;
      # `ps auxe` audit no longer reveals root password.
      docker exec -e MYSQL_PWD=password classifieds_db mysql -u root osclass \
        -e 'source docker-entrypoint-initdb.d/osclass_craigslist.sql' >/dev/null 2>&1
      seed_rc=$?
      if (( seed_rc == 0 )); then
        echo "[START] classifieds DB seed OK (attempt ${seed_retry})"
        break
      fi
      echo "[START] classifieds DB seed attempt ${seed_retry} failed (rc=${seed_rc}), retrying in 5s..." >&2
      sleep 5
    done
    if (( seed_rc != 0 )); then
      echo "[START] ✗ FATAL: classifieds DB seed failed after 3 retries; cls site will be empty (all cls tasks would 0% SR)" >&2
      return 1
    fi
  fi
}

start_homepage() {
  echo "[START] homepage (http://${HOSTNAME_VALUE}:4399)"
  # B-751 cont (P1-8 AB* OOB): template render pattern for homepage. Pre-fix
  # `perl -pi -e` was idempotent on FIRST hostname change (replace placeholder
  # → host A) but BROKEN on HOSTNAME_VALUE change to host B — only `<your-server-
  # hostname>` and `localhost:*` matched, NOT old `host_A:*`. Migrating to
  # template + cp + render ensures clean re-render on any HOSTNAME change.
  local _homepage_html="${ENV_DIR}/webarena-homepage/templates/index.html"
  local _homepage_template="${ENV_DIR}/webarena-homepage/templates/index.html.template"
  if [[ ! -f "${_homepage_template}" ]]; then
    cp "${_homepage_html}" "${_homepage_template}"
    echo "[START] homepage: saved pristine template at ${_homepage_template} (B-751 first-time migration)"
  fi
  cp -f "${_homepage_template}" "${_homepage_html}"
  perl -pi -e "s|<your-server-hostname>|${HOSTNAME_VALUE}|g" "${_homepage_html}"
  perl -pi -e "s|localhost:9980|${HOSTNAME_VALUE}:9980|g; s|localhost:7770|${HOSTNAME_VALUE}:7770|g; s|localhost:9999|${HOSTNAME_VALUE}:9999|g; s|localhost:8888|${HOSTNAME_VALUE}:8888|g" "${_homepage_html}"

  if pgrep -f 'flask run.*4399' >/dev/null 2>&1; then
    echo "homepage already running"
    return
  fi
  (cd "${ENV_DIR}/webarena-homepage" && nohup flask run --host=0.0.0.0 --port=4399 >/tmp/vwa_homepage.log 2>&1 &)
}

main() {
  echo "=== VWA Docker Startup ==="
  echo "project_dir=${PROJECT_DIR}"
  echo "sites=${SITES}"
  echo "hostname=${HOSTNAME_VALUE}"

  check_prerequisites

  if (( CHECK_ONLY == 1 )); then
    echo "Check-only mode complete."
    exit 0
  fi

  { contains_site "shopping" || contains_site "shopping_admin"; } && start_shopping
  contains_site "reddit" && start_reddit
  contains_site "wikipedia" && start_wikipedia
  contains_site "classifieds" && start_classifieds
  contains_site "homepage" && start_homepage

  echo ""
  echo "=== Running containers ==="
  docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | grep -E 'shopping|forum|wikipedia|classifieds|db|redis|chrome|NAMES' || true
}

main "$@"
