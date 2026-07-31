#!/usr/bin/env bash
# reset_vwa_sites.sh — VWA 站点 reset 工具（通用，可 source）
#
# 用法（source 后调用）：
#   source scripts/maintenance/reset_vwa_sites.sh
#   reset_vwa_sites all              # reset classifieds + reddit + shopping + shopping_admin
#   reset_vwa_sites classifieds      # 只 reset classifieds
#   reset_vwa_sites reddit
#   reset_vwa_sites shopping
#   reset_vwa_sites shopping_admin   # WA shopping_admin (Magento admin)
#
# 部署模式（VWA_RESET_MODE，默认 auto）：
#   remote  — DGX→quark→Windows PowerShell（默认 SSH 私钥存在时）
#   local   — A100 自托管 docker，本机操作（SSH 私钥不存在时 fallback）
#   auto    — 按 SSH 私钥存在与否自动选择
#
# 环境变量覆盖：
#   VWA_RESET_MODE      remote | local | auto
#   VWA_RESET_SSH_KEY   私钥路径（默认 ~/.ssh/vwa_windows）— remote 模式用
#   VWA_RESET_SSH_HOST  目标主机（默认 必须设置 e.g., quark@YOUR_HOST_IP）— remote 模式用
#   VWA_RESET_SCRIPT    Windows PowerShell 脚本路径（默认 C:\vwa\reset_vwa.ps1）— remote 模式用
#   VWA_RESET_ENABLE    设为 0 可禁用 reset（dry-run/本地调试用）
#   CLASSIFIEDS_RESET_TOKEN  cls reset endpoint token — local 模式用

VWA_RESET_MODE="${VWA_RESET_MODE:-auto}"
VWA_RESET_SSH_KEY="${VWA_RESET_SSH_KEY:-${HOME}/.ssh/vwa_windows}"
# B-744 (/stress A1.17 cold-start P0-1 AB* OOB, 2026-05-17): `:?` removed at
# source time. Pre-fix `${VWA_RESET_SSH_HOST:?...}` forced every source path
# (including A100 self-host local mode where SSH host is irrelevant) to set
# this env or `set -euo pipefail` would kill the chain at line 27. Validation
# now lives in the remote branch (line ~210) where the env is actually used.
VWA_RESET_SSH_HOST="${VWA_RESET_SSH_HOST:-}"
VWA_RESET_SCRIPT="${VWA_RESET_SCRIPT:-C:\\vwa\\reset_vwa.ps1}"
VWA_RESET_ENABLE="${VWA_RESET_ENABLE:-1}"

# --- A100 self-host local-mode helpers ----------------------------------------

# reset_vwa_local_classifieds — HTTP reset endpoint（OSClass / jykoh image）
# BUG-3 fix (2026-05-16, codex Attack 6 + gemini NEW-OOB-2): OSClass controller
# at /usr/src/myapp/oc-includes/osclass/controller/reset.php expects POST + page=reset
# (NOT GET page=reset_database — that has no controller, OSClass silently swallows
# returning homepage 200 → wrapper false-positive; 0 SQL executed across entire
# Phase 1a run; cls DB state contaminated cross-episode). Codex verified live via
# `docker exec mysql query oc_t_item_comment` — 2 stale comments persisted post-"reset".
# 3-5pp drift (gemini estimate) / 0.2-0.8pp bounded (codex on require_reset subset).
_reset_vwa_local_classifieds() {
    local label="$1"
    # ── Gate 3 (2026-05-23): per-condition docker restart (VWA_RESTART_DOCKER=1) ──
    # WHY: reddit reset already does `docker rm+run` (fresh container each
    # condition); classifieds reset was HTTP `page=reset` + SQL sentinel +
    # PHP-cache clear ONLY — app + db containers persisted across ALL conditions
    # (canary R11315 ran on a 6-7 day-old classifieds container). That accretes
    # PHP-FPM worker memory + MySQL connection-pool/buffer state → (a) substrate
    # decay (the ~7-10min latency-degradation windows behind Fire-5/6 eval-timeout
    # aborts), (b) cross-condition latency confound (a condition on a warm 6-day
    # container is not comparable to one on a fresh container). Restarting both
    # containers per condition makes cls symmetric with reddit's rm+run. The
    # B-1836 retry safety-net stays in place but should now rarely fire. Flag-gated
    # so dev runs (flag unset) skip the ~30-60s cost.
    if [[ "${VWA_RESTART_DOCKER:-0}" == "1" ]]; then
        echo "[${label}][reset_vwa][local] Gate3 per-condition docker restart: classifieds_db + classifieds"
        local _restart_out _ntfy="${NTFY_TOPIC:-p79-exp-dgx-spark}"
        # B-1839 P2-4: capture restart output (was >/dev/null) so per-container errors surface.
        if ! _restart_out=$(docker restart classifieds_db classifieds 2>&1); then
            echo "[${label}][reset_vwa][local] classifieds docker restart FAILED (fail-closed): ${_restart_out}" >&2
            curl -sf -d "🔴 B-1839 cls docker restart FAILED: ${_restart_out}" "ntfy.sh/${_ntfy}" >/dev/null 2>&1 || true
            return 1
        fi
        # wait MySQL query-ready (db restart cold ~5-15s). B-1839 P2-1: `SELECT 1` on
        # osclass DB confirms grant-tables + DB queryable — stricter than `mysqladmin
        # ping` which returns once the server accepts connections (before query-ready).
        local _i _db_ok=0
        for _i in $(seq 1 30); do
            if docker exec -e MYSQL_PWD=password classifieds_db mysql -uroot osclass -sN -e "SELECT 1" >/dev/null 2>&1; then
                _db_ok=1; break
            fi
            sleep 2
        done
        if [[ "${_db_ok}" != "1" ]]; then
            echo "[${label}][reset_vwa][local] classifieds_db not query-ready 60s post-restart (fail-closed)" >&2
            curl -sf -d "🔴 B-1839 cls db not query-ready 60s post-restart" "ntfy.sh/${_ntfy}" >/dev/null 2>&1 || true
            return 1
        fi
        # wait HTTP 200 (OSClass cold ~10-15s) — also warms the FPC so the head-of-
        # chain task playwright wait_until=load (30s) won't trip a cold path.
        local _http_ok=0 _code
        for _i in $(seq 1 30); do
            _code=$(curl -sS -o /dev/null --max-time 10 -w "%{http_code}" \
                    "http://localhost:9980/index.php?page=login" 2>/dev/null || echo "000")
            if [[ "${_code}" == "200" ]]; then _http_ok=1; break; fi
            sleep 2
        done
        if [[ "${_http_ok}" != "1" ]]; then
            echo "[${label}][reset_vwa][local] classifieds HTTP not ready 60s post-restart (last=${_code}, fail-closed)" >&2
            curl -sf -d "🔴 B-1839 cls HTTP not 200 60s post-restart (last=${_code})" "ntfy.sh/${_ntfy}" >/dev/null 2>&1 || true
            return 1
        fi
        echo "[${label}][reset_vwa][local] classifieds containers fresh + warm (db query OK, http 200)"
    fi
    # B-757 (/stress A1.17 cold-start P1-14 A, 2026-05-17): token from env or
    # .auth/cls_reset_token (gitignored). Hardcoded literal removed —
    # committed-in-source secrets fail OSF audit. Migration (1-time): operator
    # writes existing token to .auth/cls_reset_token, then `git rm -f` any
    # historical occurrences via `git filter-repo`.
    local token="${CLASSIFIEDS_RESET_TOKEN:-}"
    local _repo_root
    _repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    if [[ -z "${token}" && -f "${_repo_root}/.auth/cls_reset_token" ]]; then
        token="$(cat "${_repo_root}/.auth/cls_reset_token" 2>/dev/null | tr -d '[:space:]')"
    fi
    if [[ -z "${token}" ]]; then
        echo "[${label}][reset_vwa][local] CLASSIFIEDS_RESET_TOKEN missing (env or .auth/cls_reset_token); aborting cls reset" >&2
        return 1
    fi
    # B-1870 (cls reset POST transient resilience, 2026-06-09): the reset POST
    # (page=reset) is a heavy op (DB DROP/seed + PHP cache/session clear) that can
    # transiently fail with http=000000 (TCP connect refused) when the container is
    # still settling right after the L87 login-200 health probe passed. B-1839's
    # warm-up retry covered the health probe (lightweight page=login) but NOT this
    # actual reset call — an asymmetric gap. Observed 2026-06-09 R32472 (B2 som cls):
    # a single 000000 here cascade-halted the whole cls chain (fail-closed working as
    # designed, but on a recoverable transient). Fix: symmetric retry+backoff. The
    # reset is idempotent (000=request never reached server → no partial exec; even a
    # 500 is DROP+seed-idempotent), so re-POST is safe. Fail-closed PRESERVED — return
    # 1 only after all 4 attempts exhaust. See master_bug_catalog B-1870.
    local code _ri _reset_ok=0
    for _ri in 1 2 3 4; do
        code=$(curl -sS -o /dev/null --max-time 60 -w "%{http_code}" \
               -X POST -d "token=${token}" \
               "http://localhost:9980/index.php?page=reset" 2>/dev/null || echo "000")
        if [[ "${code}" == "200" ]]; then _reset_ok=1; break; fi
        if [[ "${_ri}" -lt 4 ]]; then
            echo "[${label}][reset_vwa][local] classifieds reset POST attempt ${_ri}/4 failed (http=${code}); backoff $(( _ri * 5 ))s" >&2
            sleep $(( _ri * 5 ))
        fi
    done
    if [[ "${_reset_ok}" != "1" ]]; then
        echo "[${label}][reset_vwa][local] classifieds HTTP FAIL (http=${code}) after 4 attempts" >&2
        return 1
    fi
    # B-746 (/stress A1.17 cold-start P0-3 C* OOB, 2026-05-17, Q2=D'):
    # gemini "sentinel theater" attack — pre-fix only checked 3 tables; OSClass
    # has dozens. Two-layer defense:
    #   (a) Sentinel scope expanded 3→5 tables (added oc_t_alerts user-subscriptions
    #       + oc_t_latest_searches search-history). Full DROP+seed restore (gemini
    #       Option A) deferred to A1.17b — needs OSClass install-state flag verify +
    #       app-cache/session interaction test (笔记 §194 risk analysis 1-3).
    #   (b) App-layer hygiene: clear PHP filesystem cache + session files. This is
    #       'whichever-path-chosen' belt-and-suspenders — even if sentinel passes,
    #       stale PHP cache/session can leak prior-episode state through app layer
    #       (clean DB ≠ clean app). Failures are non-fatal logged (cleanup is
    #       best-effort hardening, not paper-grade gate).
    #
    # B-747 (/stress A1.17 cold-start P1-4 AB* OOB, 2026-05-17, B-717 sibling):
    # MYSQL_PWD env injection replaces `-ppassword` argv on all 5 mysql calls.
    # `docker exec -e MYSQL_PWD ...` propagates env into container; mysql client
    # reads MYSQL_PWD via libmysqlclient. `ps auxe` on A100 VM no longer leaks
    # plaintext password (UCL Condense VM admin/sidecar surface).
    local comments_count items_count user_count alerts_count searches_count
    comments_count=$(docker exec -e MYSQL_PWD=password classifieds_db mysql -uroot osclass -sN -e \
            "SELECT COUNT(*) FROM oc_t_item_comment WHERE b_active=1;" 2>/dev/null || echo "?")
    # B-1571 / B-1572 / B-1573 / B-1576 (/stress A1.24 P0-1-ABC + P0-2-B* + P0-3-C*
    # + B-1576 follow-up hot-fix 2026-05-18 ~13:00 BST): MAX-seed-pk_i_id boundary
    # assertion replaces both the original `fk_i_user_id > 0` filter AND the initial
    # B-1571 canonical-12-ID-set assertion. Empirical evidence from live cls DB
    # post-fresh-reset on A100 condense (2026-05-18):
    #
    #   docker exec classifieds_db mysql -e 'SELECT MIN(pk_i_id), MAX(pk_i_id),
    #     COUNT(*) FROM oc_t_item WHERE b_active=1;'
    #   → 1 | 84154 | 84149   (84149 active seed items spanning pk_i_id 1..84154)
    #
    # The initial B-1571 fix wrongly restricted "canonical seed" to the 12 IDs
    # (84143..84154) from classifieds_restore.sql:53-65 — codex F2 OOB attack relied
    # on `grep -c '^(841[0-9][0-9],1,' classifieds_restore.sql = 12` which only
    # matched fk_i_user_id=1 INSERT lines (Blake Sullivan's 12 listings). The
    # remaining ~84137 seed items live in the sibling osclass_craigslist.sql with
    # fk_i_user_id=0 (system-seeded guest posts). Initial B-1571 fix would have
    # false-FAILed against the full seed image (84137 items "outside the 12-ID
    # canonical set"). B-1576 corrects the scope: use the MAX(pk_i_id) post-reset
    # as the boundary; experiment-posted items auto-increment to pk_i_id >= 84155.
    #
    # Three failures the unified fix still catches (B-1571/B-1572/B-1573 OOB
    # attacks remain defused):
    #   (a) P0-1-ABC: 84149 seed items now PASS (all ≤ 84154); pre-fix always FAILed.
    #   (b) P0-2-B* (codex F2 OOB): blake.sullivan (user_id=1) experiment posts
    #       still caught — their pk_i_id auto-increments past 84154 regardless of
    #       which user posted them. JOIN-on-username approach would have silently
    #       hidden them; pk_i_id > MAX_SEED catches all post-seed inserts.
    #   (c) P0-3-C* (gemini F3 OOB): guest posts (fk_i_user_id NULL/0) still caught
    #       — they too get auto-increment pk_i_id past 84154.
    #
    # Defense in depth: OR with dt_pub_date > 2024-01-01 catches edge case where
    # AUTO_INCREMENT might be manually reset (e.g. by ALTER TABLE) and an experiment
    # post overwrites a deactivated seed pk_i_id. All seed items have dt_pub_date
    # in 2023-09 to 2023-11 (verified live).
    items_count=$(docker exec -e MYSQL_PWD=password classifieds_db mysql -uroot osclass -sN -e \
            "SELECT COUNT(*) FROM oc_t_item WHERE b_active=1 AND (pk_i_id > 84154 OR dt_pub_date > '2024-01-01');" 2>/dev/null || echo "?")
    # Exclude seed/admin users (`s_username IN ('admin','user_seed')` etc. — keep
    # broad exclusion via `b_active=1 AND s_username NOT LIKE '%admin%'`).
    user_count=$(docker exec -e MYSQL_PWD=password classifieds_db mysql -uroot osclass -sN -e \
            "SELECT COUNT(*) FROM oc_t_user WHERE b_active=1 AND s_username NOT LIKE '%admin%';" 2>/dev/null || echo "?")
    # B-746a: 2 new sentinel tables (gemini "sentinel theater" defuse, scope D' = 5 tables).
    # oc_t_alerts = user subscriptions (search alerts); oc_t_latest_searches = recent
    # search history. Both mutate per-episode via user actions; both invisible to
    # the 3-table pre-fix sentinel.
    alerts_count=$(docker exec -e MYSQL_PWD=password classifieds_db mysql -uroot osclass -sN -e \
            "SELECT COUNT(*) FROM oc_t_alerts;" 2>/dev/null || echo "?")
    searches_count=$(docker exec -e MYSQL_PWD=password classifieds_db mysql -uroot osclass -sN -e \
            "SELECT COUNT(*) FROM oc_t_latest_searches;" 2>/dev/null || echo "?")

    # Each table independently asserted; report all failures in one pass.
    local failed=0
    if [[ "${comments_count}" != "0" ]]; then
        echo "[${label}][reset_vwa][local] cls sentinel FAIL: oc_t_item_comment=${comments_count} (expected 0)" >&2
        failed=1
    fi
    if [[ "${items_count}" != "0" ]]; then
        echo "[${label}][reset_vwa][local] cls sentinel FAIL: oc_t_item (pk_i_id > 84154 OR dt_pub_date > '2024-01-01') = ${items_count} (expected 0 — experiment-posted contamination)" >&2
        failed=1
    fi
    # User table: VWA classifieds seeds a fixed set of seed users; reset should
    # leave only those. Empirically the seed user list is small (~5-10); >20 user
    # accounts post-reset = previous episode's user creates persisted. Heuristic
    # threshold 20 (conservative; tighten when seed user count known).
    if [[ "${user_count}" =~ ^[0-9]+$ ]] && (( user_count > 20 )); then
        echo "[${label}][reset_vwa][local] cls sentinel FAIL: oc_t_user non-admin count=${user_count} > 20 (seed expected ~5-10)" >&2
        failed=1
    fi
    # B-746a: new sentinels — both should be 0 post-reset (no seed alerts/searches).
    if [[ "${alerts_count}" != "0" ]]; then
        echo "[${label}][reset_vwa][local] cls sentinel FAIL: oc_t_alerts=${alerts_count} (expected 0)" >&2
        failed=1
    fi
    if [[ "${searches_count}" != "0" ]]; then
        echo "[${label}][reset_vwa][local] cls sentinel FAIL: oc_t_latest_searches=${searches_count} (expected 0)" >&2
        failed=1
    fi
    if (( failed == 1 )); then
        echo "[${label}][reset_vwa][local] classifieds reset SQL incomplete (5-table sentinel rejected)" >&2
        return 1
    fi
    # B-746b: PHP app-layer hygiene — file cache + session cleanup.
    # OSClass caches: oc-content/cache/ (template + data caches), oc-content/runtime/
    # (compiled templates + plugins). Session: PHP default tmpfs + custom session
    # paths. Container name 'classifieds' = OSClass app (compose project default).
    # Non-fatal: file-system cleanup is best-effort defense-in-depth; sentinel
    # already verified DB clean.
    docker exec classifieds sh -c 'rm -rf /usr/src/myapp/oc-content/cache/* /usr/src/myapp/oc-content/runtime/* 2>/dev/null; find /tmp -maxdepth 2 -name "sess_*" -delete 2>/dev/null; find /usr/src/myapp/oc-content -maxdepth 3 -name "sessions" -type d -exec sh -c "rm -rf \"\$1\"/*" _ {} \; 2>/dev/null; exit 0' 2>/dev/null || \
        echo "[${label}][reset_vwa][local] WARN: cls PHP cache/session cleanup non-fatal failure (DB sentinel already PASSed)" >&2
    echo "[${label}][reset_vwa][local] classifieds OK (http=200, 5-table sentinel: comments=${comments_count}, items_contam=${items_count}, users=${user_count}, alerts=${alerts_count}, searches=${searches_count}; canonical seed pk_i_id ≤ 84154 + dt_pub_date ≤ 2023-11-16 preserved; PHP cache+session cleared)"
    return 0
}

# reset_vwa_local_reddit — docker rm + run（postmill image seeds itself）
# Why: postmill-populated-exposed-withimg has the seed DB built in; recreate
# = fresh state. ~30-60s warm-up after run.
_reset_vwa_local_reddit() {
    local label="$1"
    docker rm -f vwa-reddit 2>/dev/null || true
    # B-309 (A1.17 P1-6, gemini OOB unique): docker run must include -e TZ to
    # match start_vwa_docker.sh:217 (initial container start). Pre-fix reset
    # produced container with TZ=UTC (Docker default); initial start used
    # Europe/London. Tasks with relative-time semantics ("within the last hour"
    # types in reddit) saw different system time before vs after reset → systematic
    # noise in ablation. VWA_REDDIT_TZ env override defaulting to Europe/London
    # (was QUARK_TZ legacy name; harmless on A100 self-host but rename improves clarity).
    # B-753 (/stress A1.17 cold-start P1-10 C* OOB, 2026-05-17): P79_VWA_TZ unified
    # var (was VWA_REDDIT_TZ vs QUARK_TZ asymmetry with start_vwa_docker.sh:247).
    # Pre-fix reset-time TZ env name differed from initial-start TZ env → reddit
    # container could see different TZ across reset → "posts from today" relative-
    # time tasks saw systematic ablation noise. init_paper_grade_env exports
    # P79_VWA_TZ=Europe/London default; legacy VWA_REDDIT_TZ / QUARK_TZ kept as
    # fallback for transition.
    if ! docker run -d --name vwa-reddit \
            -e TZ="${P79_VWA_TZ:-${VWA_REDDIT_TZ:-${QUARK_TZ:-Europe/London}}}" \
            -p 9999:80 postmill-populated-exposed-withimg >/dev/null; then
        echo "[${label}][reset_vwa][local] reddit docker run FAILED" >&2
        return 1
    fi
    # warm-up wait — postmill cold-start ~60-120s (DB init slower than expected).
    # Poll for HTTP 200 (not just connection-up; 500 = still initializing).
    local i code
    for i in $(seq 1 60); do
        code=$(curl -sS -o /dev/null --max-time 5 -w "%{http_code}" \
            http://localhost:9999/ 2>/dev/null || echo "000")
        if [[ "${code}" == "200" ]]; then
            echo "[${label}][reset_vwa][local] reddit OK (warm-up=$((i*3))s)"
            return 0
        fi
        sleep 3
    done
    echo "[${label}][reset_vwa][local] reddit warm-up TIMEOUT after 180s (last http=${code})" >&2
    return 1
}

# reset_vwa_local_shopping — docker rm + rebuild（shopping_final_0712 self-seeds）
# B-299 (A1.17 2026-05-16 cross-AI A+B P0) left this as a `return 78` placeholder
# that hard-failed the gate; implemented 2026-07-31 to unblock Phase 1b.
#
# Why rebuild rather than the SQL-restore the B-299 note called for: on the A100
# self-host, `docker inspect vwa-shopping --format '{{json .Mounts}}'` is `[]` —
# no volume, no bind mount, so Magento's MySQL data lives entirely in the
# container's writable layer. `docker rm -f` destroys it with the container and
# the rebuild comes up on the image's seed state. That makes shopping structurally
# identical to reddit's postmill (both self-seeding images), so it takes the same
# route. Enumerating dirty tables to truncate would be strictly more fragile here:
# it has to stay exhaustive as tasks touch new tables, whereas "throw the container
# away" cannot miss one. (SQL-restore is what a volume-backed deployment needs.)
#
# shopping_admin must be rebuilt in the same call: both sites share ONE container
# (docker inspect confirms 7770 AND 7780 both bind to vwa-shopping), so passing
# only `--sites shopping` would drop the 7780 mapping and leave admin unreachable.
#
# start_vwa_docker.sh already owns everything the rebuild needs afterwards —
# Magento base_url patch (config:set + the authoritative SQL UPDATE + a DB-side
# verify), cache:flush, and indexer:reindex polled to all-Ready (B-311; 5-10 min
# on the 68GB image). Not reimplemented here.
#
# FAIL-CLOSED, and this is the part that matters: start_vwa_docker.sh's
# find_running_container REUSES an existing container. So if `docker rm` fails,
# start would quietly attach to the dirty container and report success — exactly
# the silent-pass class B-299 was created to stop. The container ID is therefore
# the sentinel: unchanged ID means the rebuild never happened, and that is a hard
# failure rather than a warning.
_reset_vwa_local_shopping() {
    local label="$1"
    local _repo_root
    _repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

    local old_id
    old_id=$(docker inspect -f '{{.Id}}' vwa-shopping 2>/dev/null || echo "none")

    docker rm -f vwa-shopping >/dev/null 2>&1 || true

    # Rebuild + base_url patch + cache flush + indexer poll (all inside start_vwa_docker.sh).
    if ! bash "${_repo_root}/scripts/vwa/start_vwa_docker.sh" \
            --sites shopping,shopping_admin \
            --hostname "${VWA_SHOPPING_HOSTNAME:-localhost}"; then
        echo "[${label}][reset_vwa][local] shopping rebuild FAILED (start_vwa_docker.sh non-zero)" >&2
        return 1
    fi

    local new_id
    new_id=$(docker inspect -f '{{.Id}}' vwa-shopping 2>/dev/null || echo "none")
    if [[ "${new_id}" == "none" ]]; then
        echo "[${label}][reset_vwa][local] shopping container absent after rebuild" >&2
        return 1
    fi
    if [[ "${new_id}" == "${old_id}" ]]; then
        echo "[${label}][reset_vwa][local] shopping REBUILD DID NOT HAPPEN — container id unchanged (${old_id:0:12}). docker rm failed and start reused the dirty container; prior-condition cart/session state is still live." >&2
        return 1
    fi

    # HTTP warm-up. The indexer poll inside start_vwa_docker.sh usually outlasts
    # Magento's boot, but it polls the CLI rather than the storefront — a 500 from
    # the web tier would go unnoticed there.
    local i code
    for i in $(seq 1 60); do
        code=$(curl -sS -o /dev/null --max-time 5 -w "%{http_code}" \
            "http://localhost:7770/" 2>/dev/null || echo "000")
        [[ "${code}" == "200" ]] && break
        sleep 3
    done
    if [[ "${code}" != "200" ]]; then
        echo "[${label}][reset_vwa][local] shopping storefront warm-up TIMEOUT after 180s (last http=${code})" >&2
        return 1
    fi

    # Informational only — the rebuild already guarantees seed state, so these are
    # a baseline record for future comparison, NOT a gate. A failed query prints
    # `?` rather than 0: an unreadable count must not read as "measured zero".
    # MYSQL_PWD env injection per B-747 (keeps the password out of `ps auxe`).
    local quotes orders customers searches
    quotes=$(docker exec -e MYSQL_PWD=MyPassword vwa-shopping mysql -u magentouser magentodb -sN \
        -e "SELECT COUNT(*) FROM quote;" 2>/dev/null || echo "?")
    orders=$(docker exec -e MYSQL_PWD=MyPassword vwa-shopping mysql -u magentouser magentodb -sN \
        -e "SELECT COUNT(*) FROM sales_order;" 2>/dev/null || echo "?")
    customers=$(docker exec -e MYSQL_PWD=MyPassword vwa-shopping mysql -u magentouser magentodb -sN \
        -e "SELECT COUNT(*) FROM customer_entity;" 2>/dev/null || echo "?")
    searches=$(docker exec -e MYSQL_PWD=MyPassword vwa-shopping mysql -u magentouser magentodb -sN \
        -e "SELECT COUNT(*) FROM search_query;" 2>/dev/null || echo "?")

    echo "[${label}][reset_vwa][local] shopping OK (http=200, container ${old_id:0:12}→${new_id:0:12}; seed baseline: quote=${quotes}, sales_order=${orders}, customer_entity=${customers}, search_query=${searches}; shopping_admin 7780 shares this container)"
    return 0
}

# reset_vwa_sites <site> [label]
#   site:  all | classifieds | reddit | shopping | shopping_admin
#   label: 日志前缀（可选，默认 reset）
reset_vwa_sites() {
    local site="${1:-all}"
    local label="${2:-reset}"

    if [[ "${VWA_RESET_ENABLE}" != "1" ]]; then
        # P0-7-A (/stress Phase 0 unified bug list 2026-05-19, Claude unique):
        # paper-grade fire requires site reset (CLAUDE.md hard rule #2). Pre-fix
        # operator residual `VWA_RESET_ENABLE=0` from dev session + paper-grade
        # launch (RESET_BEFORE=1) → silent reset SKIP → runner starts on
        # contaminated cart/listings substrate. Now: under P79_PAPER_GRADE=1,
        # VWA_RESET_ENABLE != 1 is fail-closed; dev mode unaffected.
        if [[ "${P79_PAPER_GRADE:-0}" == "1" ]]; then
            echo "[${label}][reset_vwa][FATAL] VWA_RESET_ENABLE=${VWA_RESET_ENABLE} under P79_PAPER_GRADE=1 — paper-grade requires reset enabled (CLAUDE.md hard rule #2 跑实验必须 reset 站点)" >&2
            echo "[${label}][reset_vwa][FATAL] options: (a) export VWA_RESET_ENABLE=1; (b) unset P79_PAPER_GRADE for dev mode; (c) clear residual env from prior dev session" >&2
            return 1
        fi
        echo "[${label}][reset_vwa] VWA_RESET_ENABLE=0，跳过 reset（dev mode, site=${site}）"
        return 0
    fi

    # Auto-detect mode (B-300 A1.17 2026-05-16 cross-AI A+B P0 OOB):
    # pre-fix used only `[[ -f ${SSH_KEY} ]]` as proxy for "remote path available";
    # broken when A100 VM has legacy SSH key from dotfiles/rsync + paper-grade target
    # is local docker. Now hostname-first: A100 indicators force local, regardless
    # of SSH key presence. Remote only when explicitly DGX-shaped session.
    local mode="${VWA_RESET_MODE}"
    if [[ "${mode}" == "auto" ]]; then
        if [[ "$(hostname)" == *a100* ]] \
           || [[ "$(hostname)" == *condense* ]] \
           || [[ "${P79_PAPER_GRADE_HOST:-0}" == "1" ]] \
           || [[ -d /home/ubuntu/workspace/p79 ]]; then
            mode="local"
            echo "[${label}][reset_vwa] auto-detect: A100 indicator matched → mode=local (ignoring SSH key presence)" >&2
        elif [[ -f "${VWA_RESET_SSH_KEY}" ]]; then
            mode="remote"
        else
            mode="local"
        fi
    fi
    echo "[${label}][reset_vwa] resolved mode=${mode} for site=${site}" >&2

    # Local mode: dispatch per site via _reset_vwa_local_* helpers
    if [[ "${mode}" == "local" ]]; then
        echo "[${label}][reset_vwa] mode=local，开始 reset site=${site}..."
        local rc=0
        case "${site}" in
            all)
                # B-1839 P2-3: short-circuit on first failure (was: continued mutating
                # reddit/shopping after a cls failure + overwrote rc → masked root cause +
                # mutated unrelated sites post-hard-fail). Pass-1 uses specific sites, not
                # `all`; this hardens the documented `reset_vwa_sites all` path.
                _reset_vwa_local_classifieds "${label}" || { rc=$?; echo "[${label}][reset_vwa] all: classifieds failed (rc=$rc), aborting remaining sites" >&2; return $rc; }
                _reset_vwa_local_reddit "${label}" || { rc=$?; echo "[${label}][reset_vwa] all: reddit failed (rc=$rc), aborting remaining sites" >&2; return $rc; }
                _reset_vwa_local_shopping "${label}" || rc=$?
                ;;
            classifieds)         _reset_vwa_local_classifieds "${label}" || rc=$? ;;
            reddit)              _reset_vwa_local_reddit "${label}" || rc=$? ;;
            shopping|shopping_admin)
                                 _reset_vwa_local_shopping "${label}" || rc=$? ;;
            *)
                echo "[${label}][reset_vwa] unknown site=${site}" >&2
                return 2
                ;;
        esac
        return ${rc}
    fi

    # Remote mode: DGX→quark→Windows PowerShell (original path)
    if [[ ! -f "${VWA_RESET_SSH_KEY}" ]]; then
        echo "[${label}][reset_vwa] WARNING: SSH 私钥不存在: ${VWA_RESET_SSH_KEY}，跳过 reset"
        return 1
    fi
    # B-744 (cont): remote-mode-specific SSH host validation (moved from source-
    # time `:?` per A100 local-mode unblock).
    if [[ -z "${VWA_RESET_SSH_HOST:-}" ]]; then
        echo "[${label}][reset_vwa] FATAL: remote mode requires VWA_RESET_SSH_HOST (e.g., quark@YOUR_HOST_IP); see scripts/vwa_env_remote.sh" >&2
        return 1
    fi

    echo "[${label}][reset_vwa] mode=remote，开始 reset site=${site}..."
    ssh -i "${VWA_RESET_SSH_KEY}" \
        -o StrictHostKeyChecking=no \
        -o ConnectTimeout=30 \
        -o ServerAliveInterval=60 \
        "${VWA_RESET_SSH_HOST}" \
        "powershell -File ${VWA_RESET_SCRIPT} -Site ${site}" 2>&1
    local rc=$?
    if [[ $rc -eq 0 ]]; then
        echo "[${label}][reset_vwa] reset 完成"
    else
        echo "[${label}][reset_vwa] WARNING: reset 失败 rc=${rc}（继续执行）"
    fi

    # DGX-side defensive check: Magento base_url 复发是历史 bug, 持久化在 quark
    # side 已做三层 (commit on quark: magento_baseurl_fix.sh + start_vwa_docker.sh
    # + reset_shopping.sh), 这里加个 reachability+redirect 验证作为 belt-and-
    # suspenders. 仅 warn, 不 fail (避免 reset rc 干扰上游 chain).
    if [[ "${site}" == "shopping" || "${site}" == "shopping_admin" || "${site}" == "all" ]]; then
        for shop_site in shopping shopping_admin; do
            [[ "${site}" != "all" && "${site}" != "${shop_site}" ]] && continue
            local port="7770"
            [[ "${shop_site}" == "shopping_admin" ]] && port="7780"
            local url="${VWA_HOST_URL:-http://localhost}:${port}/"
            local redirect
            redirect=$(curl -sS -o /dev/null --max-time 10 -w "%{redirect_url}" -I "${url}" 2>/dev/null || echo "")
            if [[ "${redirect}" == *metis* ]]; then
                echo "[${label}][reset_vwa] ⚠️ ${shop_site} 又 redirect 到 metis (${redirect}). 需 quark side magento_baseurl_fix.sh 重跑."
            fi
        done
    fi

    return $rc
}
