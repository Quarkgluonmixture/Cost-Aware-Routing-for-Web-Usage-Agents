#!/usr/bin/env bash
# _lib_paper_grade_gates.sh — Shared helper for queue_*.sh paper-grade launch gates.
#
# A1.13 audit fix (2026-05-16) — addresses sibling-propagation defects:
#   - P0-1 (3-AI overlap): `auth_required_gate` hard-fail (B-224) was only in
#     queue_baseline.sh; phantom siblings still used soft `refresh_site_auth` warn.
#     Now centralized as `reset_and_auth_gate`.
#   - P0-2 (3-AI overlap): `BUG-2 preflight: assert all site URLs are local on
#     A100` block was only in queue_baseline.sh; phantom siblings missing it.
#     Now centralized as `assert_a100_url_locality`.
#   - P1-2 (codex+gemini): RUN_ID seconds-precision collision risk on FORCE_NEW
#     same-second double-fire. `mint_run_id` adds PID + $RANDOM suffix.
#
# Why a shared lib instead of inline copies:
#   Bug 2 + B-224 are both sibling-drift defects (codex stress v6 C2 / B-224 fix)
#   that landed in queue_baseline.sh and weren't manually propagated. Q4 A
#   (A1.13 user decision 2026-05-16): single source of truth prevents future
#   drift when next paper-grade gate is added.
#
# Source from queue_*.sh top:
#   source "${SCRIPT_DIR}/_lib_paper_grade_gates.sh"
# Then call functions explicitly. Each function is idempotent.

# ---------- 1. Core env exports + remote site sourcing ----------
# init_paper_grade_env <repo_dir>
#   Loads CUDA workaround, vwa_env_remote.sh (if present), WIKIPEDIA_ZIM version.
#   Always safe to call; environment-level idempotent.
init_paper_grade_env() {
  local repo_dir="$1"
  # DGX Spark CUDA workaround (also harmless on A100 host)
  export PYTORCH_NVML_BASED_CUDA_CHECK=1
  export CUDA_MPS_PIPE_DIRECTORY=""
  export CUDA_MPS_LOG_DIRECTORY=""
  # VWA endpoint env (file is per-host: A100 self-hosted = localhost vars;
  # DGX dev session = quark Tailscale IPs)
  if [[ -f "${repo_dir}/scripts/vwa_env_remote.sh" ]]; then
    # shellcheck disable=SC1091
    source "${repo_dir}/scripts/vwa_env_remote.sh"
  fi
  export WIKIPEDIA_ZIM_VERSION="${WIKIPEDIA_ZIM_VERSION:-wikipedia_en_all_maxi_2025-08}"
  # B-753 (/stress A1.17 cold-start P1-10 cont (init env P79_VWA_TZ default; reset_vwa_sites.sh:163 same fix) C* OOB, 2026-05-17): unified TZ env.
  # Pre-fix `start_vwa_docker.sh:247` used `${QUARK_TZ:-Europe/London}` while
  # `reset_vwa_sites.sh:110` used `${VWA_REDDIT_TZ:-${QUARK_TZ:-...}}` — different
  # first-layer env name → reddit container TZ could drift across reset (e.g.
  # operator sets VWA_REDDIT_TZ=UTC mid-session but not QUARK_TZ). Now both
  # consult P79_VWA_TZ first via single canonical default (Europe/London matches
  # historical Phase 1 paper-grade fires).
  export P79_VWA_TZ="${P79_VWA_TZ:-Europe/London}"
  # B-548 (/stress A1.5 P0-1-AB* codex+Claude OOB, 2026-05-17): paper_grade env
  # propagation to leaf queue scripts. Pre-fix only `queue_phase1_paper_grade.sh`
  # + `queue_phase1_router_paper_grade.sh` master orchestrators exported
  # `P79_PAPER_GRADE=1`; leaf queues (`queue_baseline.sh`, `queue_phantom_*.sh`)
  # only called this helper which did NOT set the env. Manual single-cell
  # reruns + watchdog re-spawn paths therefore ran in `paper_grade=False`
  # fail-open mode → B-544 evaluator hard-raise + B-486 paper_grade
  # diagnostic_controls hard-block + B-340 GLM hard-block all dormant.
  # Default-on with `${VAR:-1}` allows explicit `P79_PAPER_GRADE=0 bash
  # scripts/queues/queue_baseline.sh ...` dev opt-out for iteration speed.
  export P79_PAPER_GRADE="${P79_PAPER_GRADE:-1}"
}

# ---------- 1b. Per-(site, benchmark) flock — leaf script protection (B-704) ----------
# acquire_site_lock <site> <benchmark> [<label>]
# release_site_lock
#
# B-704 (/stress A1.14 Chunk d P1-4 codex F5 unique OOB B, 2026-05-17):
# pre-fix flock lived ONLY in queue_chain.sh (B-646 A1.13 P1-7) which is
# bypassed when users invoke leaf queue scripts directly (per CLAUDE.md hard
# rule: `RESET_BEFORE=1 bash scripts/queues/queue_baseline.sh B0 dom shopping`
# is documented and allowed). During an active chain, a manual rescue leaf
# invocation on the same site → race → RESET wipes the other's session.
#
# Parent-held detection via `P79_CHAIN_LOCK_HELD` env:
#   - queue_chain.sh exports `P79_CHAIN_LOCK_HELD="${site}:${benchmark}"`
#     after acquiring its own FD-9 lock.
#   - Leaf scripts check this var BEFORE attempting their own acquire — if
#     it matches the leaf's (site, benchmark) → skip (chain holds the lock).
#   - Manual leaf invocation outside any chain → env var absent → leaf
#     acquires its own per-process lock.
#
# FD 7 used (queue_chain uses FD 9) to avoid collision when a leaf script
# is hypothetically run inside a chain that DOESN'T export P79_CHAIN_LOCK_HELD
# (defense-in-depth; current chain always exports).
acquire_site_lock() {
  local site="${1:-}" benchmark="${2:-vwa}" label="${3:-leaf}"
  if [[ -z "${site}" ]]; then
    echo "[${label}][error] acquire_site_lock: site required" >&2
    return 1
  fi
  # B-905 (/stress A2.2 P0-4-A* OOB, 2026-05-17): env-bypass shortcut hardened
  # with fd-level verification. Pre-fix `P79_CHAIN_LOCK_HELD` env string match
  # alone allowed stale env leak (debug session / wrapper / cron leftover →
  # `export P79_CHAIN_LOCK_HELD=cls:vwa` 不删) → leaf bypass flock without
  # actually-held chain lock → two manual leaf invocations sharing the stale
  # env both skip → silent race (CLAUDE.md hard rule §106 violation silent).
  # Defense: env match + verify chain PID alive (kill -0) + verify chain's
  # fd 9 actually points to the expected per-site lock file (/proc readlink).
  # Stale env (chain dead OR chain fd 9 not pointing here) → FATAL with audit
  # surface, lock leak shows up as visible problem instead of silent race.
  #
  # Why this isn't pure-kernel-only (Option C as originally framed): Linux flock
  # treats two open file descriptions on the same file as INDEPENDENT lock holders;
  # same-process LOCK_EX|LOCK_NB on a second fd fails just like cross-process
  # contention. Pure deletion of the shortcut would break chain→leaf delegation
  # (subshell inherits fd 9 but opens its own fd 7 → second EX conflicts with
  # parent chain's first EX). Therefore we keep the shortcut but harden with
  # fs-level verification — the "stale env" attack is closed, re-entrance preserved.
  if [[ "${P79_CHAIN_LOCK_HELD:-}" == "${site}:${benchmark}" ]]; then
    local _chain_pid="${P79_CHAIN_PID:-}"
    if [[ -z "${_chain_pid}" ]]; then
      echo "[${label}][FATAL] P79_CHAIN_LOCK_HELD set but P79_CHAIN_PID missing — env-bypass refused (B-905)" >&2
      echo "[${label}][FATAL]   This indicates stale env leak. Unset P79_CHAIN_LOCK_HELD or re-launch via queue_chain.sh." >&2
      return 1
    fi
    if ! kill -0 "${_chain_pid}" 2>/dev/null; then
      echo "[${label}][FATAL] P79_CHAIN_LOCK_HELD set but chain PID=${_chain_pid} not alive — env-bypass refused (B-905 stale-env leak)" >&2
      echo "[${label}][FATAL]   Unset P79_CHAIN_LOCK_HELD + P79_CHAIN_PID and re-launch fresh." >&2
      return 1
    fi
    # Verify chain still holds fd 9 pointing at expected lock file
    local _expected_lock="${REPO_DIR:-$(pwd)}/.locks/p79_${site}_${benchmark}.lock"
    local _chain_fd9_target=""
    if [[ -L "/proc/${_chain_pid}/fd/9" ]]; then
      _chain_fd9_target="$(readlink "/proc/${_chain_pid}/fd/9" 2>/dev/null || true)"
    fi
    if [[ "${_chain_fd9_target}" != "${_expected_lock}" ]]; then
      echo "[${label}][FATAL] P79_CHAIN_LOCK_HELD/${_chain_pid} but /proc/${_chain_pid}/fd/9 = ${_chain_fd9_target:-<unset>}, expected ${_expected_lock} (B-905 stale-env leak)" >&2
      echo "[${label}][FATAL]   Chain process alive but no longer holds expected lock fd. Unset env vars + re-launch." >&2
      return 1
    fi
    echo "[${label}][lock] parent queue_chain pid=${_chain_pid} verified holds ${site}:${benchmark} (skip leaf acquire, B-905)" >&2
    SITE_LOCK_FD=""
    return 0
  fi
  local repo_dir="${REPO_DIR:-$(pwd)}"
  local lock_dir="${repo_dir}/.locks"
  mkdir -p "${lock_dir}" 2>/dev/null || true
  local lock_file="${lock_dir}/p79_${site}_${benchmark}.lock"
  exec 7>"${lock_file}"
  if ! flock -n 7; then
    echo "[${label}][FATAL] another paper-grade process holds lock for site=${site} benchmark=${benchmark}" >&2
    echo "[${label}][FATAL] lock file: ${lock_file}" >&2
    echo "[${label}][FATAL] if stale (prior process crashed), 'rm ${lock_file}' to force-release" >&2
    exec 7>&-
    return 78  # rc=78 = lock contention (matches reset_wa_sites convention)
  fi
  SITE_LOCK_FD=7
  SITE_LOCK_FILE="${lock_file}"
  echo "[${label}][lock] acquired ${lock_file} (pid $$)" >&2
  return 0
}

release_site_lock() {
  if [[ "${SITE_LOCK_FD:-}" == "7" ]]; then
    exec 7>&-
    unset SITE_LOCK_FD SITE_LOCK_FILE
  fi
}

# ---------- 1b. A100 hostname gate (B-1406 sibling-propagation consolidation) ----------
# require_paper_grade_host
#   Single canonical paper-grade hostname gate. Pre-fix this function was
#   inlined in 3 separate queue scripts (queue_phase1_paper_grade.sh:113,
#   queue_phase1_router_paper_grade.sh:92, queue_router_learned.sh:97-106
#   with a different older predicate) — Mode A F1 caught the regex permissive
#   match `(condense|a100|ubuntu)` substring; Mode B F5 caught the sibling
#   propagation gap. Consolidating to lib eliminates both attack vectors.
#
#   B-1406 (/stress A2.7 P1-4-AB* 2-AI overlap, Claude Mode A F1 + codex Mode B F5,
#   2026-05-18): canonical hostname allowlist with anchored regex. Pre-fix
#   `(condense|a100|ubuntu)` substring matched generic `ubuntu-server`,
#   `lubuntu-dev`, `a100something`, etc. Post-fix anchored regex requires
#   exact match against:
#     - `condense-a100.*` (canonical Condenser A100 VM family)
#     - `a100-jiaming.*` (current paper-grade target VM `a100-jiaming-test`)
#     - `ubuntu` (generic but EXACT match only — common cloud VM default)
#   Override via `P79_PAPER_GRADE_HOST=1` for explicit opt-in on CI / future
#   approved hostnames not matching the allowlist.
#
#   Function MUST be called inside a queue script that defines `log` and `fail`
#   shell functions (orchestrator + router_paper_grade + router_learned all do).
require_paper_grade_host() {
  local hn
  hn="$(hostname 2>/dev/null || true)"
  if [[ "${P79_PAPER_GRADE_HOST:-0}" == "1" ]]; then
    log "  paper-grade host: ${hn} (P79_PAPER_GRADE_HOST=1 override)"
  elif [[ "${hn}" =~ ^(condense-a100.*|a100-jiaming.*|ubuntu)$ ]]; then
    log "  paper-grade host: ${hn} (allowlist match: condense-a100*/a100-jiaming*/ubuntu)"
  else
    fail "Refusing paper-grade launch on non-A100 host '${hn}'.
       Phase 1a paper-grade target = Condenser A100 (memory project_paper_grade_target_host).
       Re-run on a100-jiaming-test (ssh condense-a100), OR set P79_PAPER_GRADE_HOST=1 explicitly.
       Allowlist: condense-a100*, a100-jiaming*, ubuntu (exact). Substring-permissive
       pre-A2.7 regex (condense|a100|ubuntu) retired (Mode A F1 + Mode B F5 OOB)."
  fi
  # URL locality — paper-grade target = A100 self-hosted docker, all VWA URLs
  # must point to localhost (no DGX→quark Tailscale substrate substitution).
  # Reuses lib helper which fail-loud on any non-local URL.
  assert_a100_url_locality
}

# ---------- 2. A100 URL-locality preflight (Bug 2 fix; B-298 A1.17 P0-1 hardening) ----------
# assert_a100_url_locality
#   On A100 self-hosted docker hosts, refuse launch if any site URL is non-local.
#   Paper-grade target = A100 self-hosted; non-local URL = silent prod substitution.
#
#   B-298 (A1.17 2026-05-16 cross-AI 2-AI overlap A+B P0 OOB): previous predicate
#   `hostname == *condense* OR -d /home/ubuntu/workspace/p79` failed on canonical
#   target VM `a100-jiaming-test` (hostname has no "condense" substring); only
#   directory fallback held, and only when user=ubuntu + canonical repo path.
#   New predicate broadens to (a) `*a100*` hostname (canonical VM name match),
#   (b) `P79_PAPER_GRADE_HOST=1` explicit env override (CI / future hostnames),
#   (c) cwd contains `workspace/p79` (user-agnostic), (d) legacy `*condense*` +
#   ubuntu path retained for back-compat.
#   On DGX dev sessions ($(hostname) = spark-9ea3, cwd = /home/jiaming/...)
#   none of (a)-(d) match → check harmlessly skips, no behavior change.
assert_a100_url_locality() {
  # B-755 (/stress A1.17 cold-start P1-12 AC* OOB, 2026-05-17): predicate logic
  # inverted via P79_PAPER_GRADE coupling. Pre-fix OR-chain whitelist was brittle:
  # CWD `*workspace/p79*` does NOT match the actual repo dir
  # `Cost-Aware-Routing-for-Web-Usage-Agents/`; hostname VM rename → gate silently
  # skipped on all DGX dev sessions AND on any A100 VM with non-canonical name.
  # New logic: under `P79_PAPER_GRADE=1` (default-on via init_paper_grade_env),
  # locality is ALWAYS enforced. DGX dev with Tailscale URLs must explicitly opt
  # out: `P79_PAPER_GRADE=0 bash scripts/queues/queue_baseline.sh ...`. Q14=A
  # (default invert) matches user decision /stress A1.17 2026-05-17.
  if [[ "${P79_PAPER_GRADE:-0}" != "1" ]]; then
    # Dev mode (P79_PAPER_GRADE=0) — operator explicitly opted out; no gate.
    return 0
  fi
  if true; then
    echo "[preflight] A100 URL-locality gate ACTIVE (P79_PAPER_GRADE=1) host=$(hostname), cwd=$(pwd)" >&2
    local _v
    for _v in CLASSIFIEDS REDDIT SHOPPING WIKIPEDIA HOMEPAGE; do
      # B-643 (A1.13 P1-8 Claude OOB, 2026-05-17): empty URL no longer silent
      # passes. Pre-fix `case "${!_v:-}" in *localhost*|*127.0.0.1*|"") ;; *) FATAL` —
      # empty string in the OK set meant `vwa_env_remote.sh` source failure (file
      # missing / syntax error) silently passed gate; runner then attempted
      # default fallback URLs (possibly prod via Tailscale). Post-fix: empty
      # URL = explicit FATAL with diagnostic hint at this env-loading layer.
      case "${!_v:-}" in
        *localhost*|*127.0.0.1*) ;;
        "")
          echo "✗ FATAL preflight: \$${_v} is EMPTY on A100 host; vwa_env_remote.sh may have failed to source. Check ${repo_dir:-scripts}/scripts/vwa_env_remote.sh" >&2
          exit 2
          ;;
        *)
          echo "✗ FATAL preflight: \$${_v}=${!_v} not local on A100 host; refusing launch" >&2
          exit 2
          ;;
      esac
    done
    unset _v
  fi
}

# ---------- 3. B0 PROXY API key loading ----------
# load_proxy_api_key <repo_dir> <log_prefix>
#   Only runs on B0 (caller checks BASELINE=B0 before invocation).
#   Loads PROXY_API_KEY from .auth/qwen_api (line prefixed `rp_`) and exports
#   PROXY_API_KEY / QWEN_API_KEY / DASHSCOPE_API_KEY for downstream consumers.
load_proxy_api_key() {
  local repo_dir="$1"
  local log_prefix="$2"
  if [[ -n "${PROXY_API_KEY:-}" ]]; then return 0; fi
  local auth_file="${repo_dir}/.auth/qwen_api"
  if [[ ! -f "${auth_file}" ]]; then
    echo "[${log_prefix}][error] ${auth_file} 不存在，且 PROXY_API_KEY 未设置" >&2
    exit 1
  fi
  local raw_key
  raw_key="$(grep -m1 '^rp_' "${auth_file}" | tr -d '[:space:]')"
  if [[ -z "${raw_key}" ]]; then
    echo "[${log_prefix}][error] ${auth_file} 存在但无 rp_ key" >&2
    exit 1
  fi
  export PROXY_API_KEY="${raw_key}"
  export QWEN_API_KEY="${raw_key}"
  export DASHSCOPE_API_KEY="${raw_key}"
  echo "[${log_prefix}] Loaded PROXY_API_KEY from ${auth_file}"
}

# ---------- 4. RUN_ID minting (P1-2 fix: nano + PID + RANDOM suffix) ----------
# mint_run_id <cfg_name> <phase_dir>
#   Echoes the chosen RUN_ID + sets RUN_ID global. FORCE_NEW=1 mints fresh
#   timestamp suffix (seconds + nanos + PID + $RANDOM) → no same-second collision
#   risk under any forseeable manual-retry pattern.
#   FORCE_NEW=0 (default) glob-resumes any existing dir matching ${CFG_NAME}_<digits>.
mint_run_id() {
  local cfg_name="$1"
  local phase_dir="$2"
  local log_prefix="${3:-queue}"
  local ts_date ts_full collision_token
  ts_date="$(date +%Y%m%d)"
  # B-640 (A1.13 P1-10 Claude, 2026-05-17): now actually using %N nanoseconds
  # alongside PID + $RANDOM. Pre-fix comment claimed "%N nanoseconds + $$ pid +
  # $RANDOM defeats same-second collision" but `date +%Y%m%d_%H%M%S` had no
  # `%N` token → only PID+RANDOM defended (Claude OOB comment-vs-code drift).
  # Format: YYYYMMDD_HHMMSS_NNNNNNNNN_PIDxxxx_Rxxxxx (~10⁻¹² collision).
  local nanos
  nanos="$(date +%N)"
  collision_token="${nanos}_$$_R${RANDOM}"
  ts_full="$(date +%Y%m%d_%H%M%S)_${collision_token}"
  if [[ "${FORCE_NEW:-0}" == "1" ]]; then
    RUN_ID="${cfg_name}_${ts_full}"
    echo "[${log_prefix}] FORCE_NEW=1 → fresh timestamped run_id=${RUN_ID} (resume-glob skipped)"
  else
    local existing
    existing="$(ls -dt "${phase_dir}/${cfg_name}_"[0-9]* 2>/dev/null | head -1 || true)"
    if [[ -n "${existing}" ]]; then
      # B-641 (A1.13 P1-9 Claude OOB, 2026-05-17): stale-resume fingerprint
      # check — schema_version v2 invariant.
      # B-836 (A1.16 cold-start P1-10-B*, 2026-05-17): expanded to also check
      # CONTENT invariants (not just format). Pre-fix: schema_version v2 was a
      # format-only check; 6-month replay on new git checkout could resume a
      # run_id whose env_snapshot.json points to stale git commit / stale VWA
      # submodule SHA — half episodes from old code/env, half from new. Now:
      # verify env_snapshot.json git commit + VWA submodule_sha match current
      # launch state. Mismatch → fresh timestamp.
      # B-1593 (/stress A1.24 post-fire P1-13-A*, 2026-05-18): HF model
      # loaded_revision check claim retracted from this docstring — HF
      # revisions are pinned at config-level (`exp_v2_base.yaml:103+138`
      # Qwen3-VL `ebb281e...` + Gemma3-VL `093f9f3...`); stale resume cannot
      # drift HF model. The pre-fix "(if present) HF model loaded_revision
      # match" claim was 2-week comment-code drift vapor (no implementing
      # branch existed at L335+). Claude Mode A solo OOB catch.
      local stale=0
      local stale_reason=""
      local meta
      for meta in "${existing}"/*/condition_meta.json; do
        if [[ -f "${meta}" ]]; then
          if ! grep -q '"schema_version"[[:space:]]*:[[:space:]]*"v2' "${meta}" 2>/dev/null; then
            stale=1
            stale_reason="schema_version != v2"
            break
          fi
        fi
      done

      # B-836 P1-10-B*: additional CONTENT checks against env_snapshot.json
      if [[ "${stale}" == "0" ]]; then
        local snap
        # B-1579 (/stress A1.24 hot follow-up, 2026-05-18): `|| true` guard against
        # `set -euo pipefail` exit when no env_snapshot.json exists at the
        # `${existing}/*/env_snapshot.json` path. Trigger condition: previous run
        # crashed pre-condition-init (env_snapshot was written at run_dir root, not
        # in a subdir) OR was cleaned up. Pre-fix bash -x trace: `+ snap=` →
        # immediate trap fire → silent exit 2 → smoke false-FAIL with no error
        # message. Now empty `snap` falls through to "no content check" branch
        # (gracefully skips stale-by-snap check; schema_version check above still
        # runs if any condition_meta.json exists).
        snap="$(ls "${existing}"/*/env_snapshot.json 2>/dev/null | head -1 || true)"
        if [[ -n "${snap}" && -f "${snap}" ]]; then
          # Current git HEAD
          local current_git_sha
          current_git_sha="$(git -C "${REPO_ROOT:-$PWD}" rev-parse HEAD 2>/dev/null || echo "")"
          # Snapshot-recorded git commit
          local snap_git_sha
          snap_git_sha="$(.venv/bin/python3 -c "
import json,sys
try:
    d=json.load(open('${snap}'))
    print(d.get('git',{}).get('commit',''))
except Exception:
    sys.exit(1)
" 2>/dev/null || echo "")"
          if [[ -n "${current_git_sha}" && -n "${snap_git_sha}" && "${current_git_sha}" != "${snap_git_sha}" ]]; then
            stale=1
            stale_reason="git_commit mismatch (current=${current_git_sha:0:8}, snap=${snap_git_sha:0:8})"
          fi

          # VWA submodule SHA (paper-grade SBOM consistency)
          if [[ "${stale}" == "0" ]]; then
            local current_vwa_sha
            current_vwa_sha="$(git -C "${REPO_ROOT:-$PWD}/external/visualwebarena" rev-parse HEAD 2>/dev/null || echo "")"
            local snap_vwa_sha
            snap_vwa_sha="$(.venv/bin/python3 -c "
import json,sys
try:
    d=json.load(open('${snap}'))
    print(d.get('vwa_sbom',{}).get('head_sha','') or d.get('vwa_source',{}).get('submodule_sha',''))
except Exception:
    sys.exit(1)
" 2>/dev/null || echo "")"
            if [[ -n "${current_vwa_sha}" && -n "${snap_vwa_sha}" && "${current_vwa_sha}" != "${snap_vwa_sha}" ]]; then
              stale=1
              stale_reason="vwa_submodule mismatch (current=${current_vwa_sha:0:8}, snap=${snap_vwa_sha:0:8})"
            fi
          fi
        fi
      fi

      if [[ "${stale}" == "1" ]]; then
        echo "[${log_prefix}] skipping stale resume candidate $(basename "${existing}") (${stale_reason}); minting fresh run_id"
        RUN_ID="${cfg_name}_${ts_full}"
      else
        RUN_ID="$(basename "${existing}")"
        echo "[${log_prefix}] resuming existing run_id=${RUN_ID}"
      fi
    else
      RUN_ID="${cfg_name}_${ts_date}"
      echo "[${log_prefix}] new run_id=${RUN_ID}"
    fi
  fi
  # B-634 (A1.13 P0-5 gemini G4 OOB, 2026-05-17): export RUN_TS_FULL so callers
  # don't independently recompute `date +%Y%m%d_%H%M%S` (different second from
  # mint = log filename collision when master orchestrator fires 2 chains in
  # same second). Callers should now use ${RUN_ID} directly for RUNNER_LOG
  # naming (RUN_ID has PID+RANDOM+nanos → 0-collision); RUN_TS_FULL kept for
  # callers wanting just the date portion of mint.
  export RUN_ID
  export RUN_TS_FULL="${ts_full}"
}

# ---------- 5. Reset + auth gate (B-224 hard-fail) ----------
# reset_and_auth_gate <site> <repo_dir> <python_bin> <log_prefix> <reset_label>
#   Source reset_vwa_sites.sh, call reset_vwa_sites, sleep 15s, run
#   auth_required_gate. Hard-fail on auth gate failure unless AUTH_GATE_BYPASS=1.
#   Caller responsible for guarding with RESET_BEFORE=1 + BENCHMARK!=wa.
reset_and_auth_gate() {
  # B-645 (A1.13 P1-12 gemini G9, 2026-05-17): named args. Pre-fix 5 positional
  # args (site, repo, python, log_prefix, reset_label); caller swap-order bugs
  # silently propagated wrong reset_label → trajectory event tags
  # cross-baseline-confused. Post-fix: both forms accepted (named when $1 starts
  # with `--`; legacy positional otherwise for back-compat) — new code should
  # use named form. Form 4 callers in queue_baseline / queue_phantom_{som,text,
  # prompt} migrated to named in this commit.
  local site="" repo_dir="" python_bin="" log_prefix="" reset_label=""
  if [[ "${1:-}" == --* ]]; then
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --site)         site="$2"; shift 2 ;;
        --repo|--repo-dir) repo_dir="$2"; shift 2 ;;
        --python|--python-bin) python_bin="$2"; shift 2 ;;
        --log-prefix)   log_prefix="$2"; shift 2 ;;
        --reset-label)  reset_label="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "[reset_and_auth_gate][error] unknown arg: $1" >&2; return 2 ;;
      esac
    done
  else
    # Legacy positional form (deprecated; use --site/--repo/--python/--log-prefix/--reset-label).
    site="$1"
    repo_dir="$2"
    python_bin="$3"
    log_prefix="$4"
    reset_label="$5"
  fi

  if [[ ! -f "${repo_dir}/scripts/maintenance/reset_vwa_sites.sh" ]]; then
    echo "[${log_prefix}][error] reset_vwa_sites.sh not found but RESET_BEFORE=1; aborting." >&2
    echo "[${log_prefix}][error] To bypass reset (paper-grade dirty), explicitly set RESET_BEFORE=0." >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "${repo_dir}/scripts/maintenance/reset_vwa_sites.sh"
  echo "[${log_prefix}] RESET_BEFORE=1 → resetting site=${site}..."
  # B-642 (A1.13 P1-5 codex F8 OOB, 2026-05-17): wrap reset_vwa_sites with
  # timeout to defend against hangs (reset script SSH stall / Docker compose
  # hang / Tailscale stall). Without timeout, hang at this stage blocks chain
  # with no watchdog yet attached + no runner pid + no ntfy → chain wedged
  # invisibly. 120s is generous (typical reset 5-15s); on timeout: ntfy +
  # abort gate with explicit FATAL surface so operator knows where it stuck.
  # reset_vwa_sites invoked in sub-bash since `timeout` operates on processes;
  # the sourced function is re-sourced in the sub-bash for isolation.
  # B-745 (/stress A1.17 cold-start P0-2 B* OOB, 2026-05-17): site-aware timeout.
  # Pre-fix hard 120s killed valid reddit reset whose own warm-up contract is up
  # to 180s (`_reset_vwa_local_reddit` polls 60 iters × 3s = 180s for HTTP 200
  # after postmill cold-start). Outer 120s < callee 180s = false-abort on healthy
  # cold container. Now: reddit gets 240s, classifieds 60s (curl <1s + ~12s sentinel
  # SQL + ~1s cleanup), shopping/all keeps 120s default. Empirically reddit needs
  # ~60-120s warm, occasional 130-160s; 240s is generous + matches callee 180s + 60s
  # buffer.
  local _reset_timeout
  case "${site}" in
    reddit) _reset_timeout=240 ;;
    classifieds) _reset_timeout=120 ;;
    *) _reset_timeout=120 ;;
  esac
  local _reset_rc
  # B-864 (/stress A1.23 P1-7 AB, 2026-05-17): process-group kill + SIGTERM trap.
  # Pre-fix `timeout ${N}s bash -c "..."` killed only the sub-bash on timeout;
  # docker compose / curl-to-reset-endpoint may have daemonized children that
  # continue asynchronously. Retry semantics then overlap: chain retry → new
  # reset_and_auth_gate → second `docker compose restart` overlaps the first
  # still in flight → container undefined intermediate state → site internal
  # cart / posted-listing silent corrupt → auth_gate still passes → looks
  # clean but data drift.
  # Fix: (a) `setsid` puts the bash in its own session/PGID so the kill
  # propagates to all children; (b) `--kill-after=10s` escalates to SIGKILL
  # if TERM ignored; (c) inner `trap SIGTERM` best-effort `docker stop` on
  # the canonical container names to preempt daemonized restart. Residual
  # gap: docker daemon's own async restart cannot be canceled by `docker stop`
  # mid-flight — disclosed in paper-2 forward stub.
  # B-1583 (/stress A1.24 post-fire P0-3-AC*, 2026-05-18): container names
  # corrected to match actual `reset_vwa_sites.sh` creates (`vwa-reddit` /
  # `classifieds` / `classifieds_db` / `vwa-shopping` / `vwa-wikipedia` per
  # `docker ps` live verify), NOT pre-fix `reddit-box/classifieds_box/
  # shopping_box` which were silent no-ops (`2>/dev/null || true` masked
  # the no-match). Pre-fix B-864 SIGTERM defense was cosmetic. 2-AI overlap
  # AC (Claude Mode A F1 + gemini Mode C F1).
  timeout --kill-after=10s --signal=TERM "${_reset_timeout}s" setsid bash -c "
    trap 'echo \"[reset_and_auth_gate] SIGTERM during reset; attempting docker stop fallback\" >&2; \
          docker stop vwa-reddit classifieds classifieds_db vwa-shopping vwa-wikipedia 2>/dev/null || true; exit 1' SIGTERM
    source '${repo_dir}/scripts/maintenance/reset_vwa_sites.sh'
    reset_vwa_sites '${site}' '${reset_label}'
  "
  _reset_rc=$?
  if [[ "${_reset_rc}" -eq 124 ]]; then
    echo "[${log_prefix}][FATAL] reset_vwa_sites timed out after ${_reset_timeout}s (B-745 site-aware)" >&2
    if command -v curl > /dev/null; then
      curl -d "queue ABORT: ${site} reset_vwa_sites timeout 120s; investigate SSH/Docker/Tailscale" \
        "ntfy.sh/${NTFY_TOPIC:-p79-exp-dgx-spark}" 2>/dev/null || true
    fi
    exit 1
  fi
  if [[ "${_reset_rc}" -eq 0 ]]; then
    echo "[${log_prefix}] reset OK; sleeping 15s for site to settle..."
    sleep 15

    # B-314 (A1.17 Option K Trajectory Event Log hook for reset events,
    # 2026-05-16): emit "reset_post_interrupt" event to a STAGING file that
    # the runner picks up on startup + merges into condition_dir/trajectory_events.jsonl.
    # Reason for staging: condition_dir doesn't exist at gate time (runner creates it).
    # Path: ${repo_dir}/logs/trajectory_events_staging/RUN_${RUN_ID:-unknown}.jsonl
    # Runner-side pickup is documented in [[phase1_plan]] §A1 follow-up.
    # Best-effort — failure does NOT block reset gate (event log is paper-§4
    # enrichment, not paper-grade integrity gate).
    if [[ -n "${RUN_ID:-}" ]]; then
      local _staging_dir="${repo_dir}/logs/trajectory_events_staging"
      mkdir -p "${_staging_dir}" 2>/dev/null || true
      local _staging_file="${_staging_dir}/RUN_${RUN_ID}.jsonl"
      local _ts
      _ts="$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
      # Simple JSON line — no jq dependency. Manual escaping of double-quotes
      # in site / reset_label assumed safe (controlled inputs from queue scripts).
      echo "{\"event_type\":\"reset_post_interrupt\",\"task_index\":null,\"wallclock_ts\":\"${_ts}\",\"metadata\":{\"site\":\"${site}\",\"reset_label\":\"${reset_label}\",\"source\":\"_lib_paper_grade_gates.reset_and_auth_gate\"}}" \
        >> "${_staging_file}" 2>/dev/null || true
      echo "[${log_prefix}] trajectory-event[reset]: staged at ${_staging_file}" >&2
    fi
    # B-224 hard-fail (2026-05-16 A1.13 P0-1, applied to ALL queue scripts):
    # post-reset auth refresh failure now aborts launch. Pre-fix soft-warn
    # let phantom paths start with NOT-LOGGED-IN tasks → step_record contamination.
    echo "[${log_prefix}] gating .auth/${site}_state.json post-reset via auth_required_gate..."
    # B-642 (A1.13 P1-5, 2026-05-17): wrap auth_required_gate with timeout to
    # defend against Playwright hangs (browser stall / network stall). 60s is
    # generous (typical auth refresh ~5-15s); timeout 124 → treat as auth FAIL
    # (handled by the outer else branch). Re-uses existing FATAL surface so
    # operator sees same error class.
    if timeout 60s "${python_bin}" -c "
import sys
sys.path.insert(0, '${repo_dir}')
from pathlib import Path
from p79.utils.auth_refresh import auth_required_gate, AuthRefreshFailure, AuthRefreshConfigError
try:
    auth_required_gate('${site}', Path('${repo_dir}/.auth'))
    print('[${log_prefix}][gate] auth_required_gate PASS')
    sys.exit(0)
except (AuthRefreshFailure, AuthRefreshConfigError) as exc:
    print(f'[${log_prefix}][gate][FATAL] {exc}', file=sys.stderr)
    sys.exit(1)
" 2>&1; then
      echo "[${log_prefix}] auth gate PASS — runner task=0 will be LOGGED IN"
    else
      echo "[${log_prefix}][error] post-reset auth gate FAILED — aborting launch to prevent paper-grade contamination." >&2
      echo "[${log_prefix}][error] Fix: (a) VWA_REMOTE_HOST env, (b) .auth/ dir writable, (c) site reachable, (d) VWA_${site^^}_USER/PASS env vars set." >&2
      echo "[${log_prefix}][error] To bypass (paper-grade dirty, watchdog reactive only), set AUTH_GATE_BYPASS=1." >&2
      # B-639 (A1.13 P1-3 codex F6 + gemini G2 2-AI, 2026-05-17): paper-grade
      # mode hard-blocks AUTH_GATE_BYPASS. Pre-fix a stray `AUTH_GATE_BYPASS=1`
      # in operator's .bashrc / dev session env could silently dissolve every
      # paper-grade gate with no log surface marking the cell dirty. Two-layer
      # defense: (1) under `P79_PAPER_GRADE=1` env (set by orchestrator), the
      # bypass is REFUSED — runs must abort + operator must investigate auth.
      # (2) when bypass IS legitimately set (dev session), emit a paper-grade
      # bypass audit log entry + ntfy alert so the bypass leaves a permanent
      # trail (reviewer / advisor / OSF audit-trail can grep for these).
      if [[ "${P79_PAPER_GRADE:-0}" == "1" && "${AUTH_GATE_BYPASS:-0}" == "1" ]]; then
        echo "[${log_prefix}][FATAL] AUTH_GATE_BYPASS=1 is FORBIDDEN under P79_PAPER_GRADE=1 — paper-grade requires hard-fail auth gate." >&2
        echo "[${log_prefix}][FATAL] Unset AUTH_GATE_BYPASS or unset P79_PAPER_GRADE (dirty/dev mode)." >&2
        exit 1
      fi
      if [[ "${AUTH_GATE_BYPASS:-0}" != "1" ]]; then
        exit 1
      fi
      # B-639: AUTH_GATE_BYPASS audit log + ntfy alert.
      local _bypass_log="${repo_dir}/logs/paper_grade_bypass_audit.log"
      local _bypass_ts
      _bypass_ts="$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
      mkdir -p "${repo_dir}/logs" 2>/dev/null || true
      echo "${_bypass_ts} AUTH_GATE_BYPASS=1 site=${site} reset_label=${reset_label} run_id=${RUN_ID:-unknown} log_prefix=${log_prefix} hostname=$(hostname) user=$(whoami)" \
        >> "${_bypass_log}" 2>/dev/null || true
      if command -v curl > /dev/null; then
        curl -d "AUTH_GATE_BYPASS active: site=${site} run=${RUN_ID:-unknown} (paper-grade dirty trail at ${_bypass_log})" \
          "ntfy.sh/${NTFY_TOPIC:-p79-exp-dgx-spark}" 2>/dev/null || true
      fi
      echo "[${log_prefix}][warn] AUTH_GATE_BYPASS=1 set — proceeding without auth gate; first 1-3 tasks at risk." >&2
      echo "[${log_prefix}][warn] Audit trail appended to ${_bypass_log} (B-639)." >&2
    fi
  else
    # B-642 (2026-05-17): _reset_rc captured from timeout-wrapped sub-bash above
    # (was `local rc=$?` pre-B-642 when reset_vwa_sites ran in current shell).
    local rc="${_reset_rc}"
    # B-299 (A1.17 P0-3): rc=78 is the "not implemented" sentinel from
    # _reset_vwa_local_shopping stub. Surface specific reason rather than generic
    # "reset failed" so Phase 1b launch operator knows what to implement.
    if [[ "${rc}" == "78" ]]; then
      echo "[${log_prefix}][error] reset NOT IMPLEMENTED for site=${site} (rc=78 sentinel)." >&2
      echo "[${log_prefix}][error] Implement reset_vwa_local_${site} body before paper-grade Phase 1b launch." >&2
    else
      echo "[${log_prefix}][error] reset failed (rc=${rc}); aborting to preserve paper-grade integrity." >&2
    fi
    echo "[${log_prefix}][error] To bypass reset (paper-grade dirty), explicitly set RESET_BEFORE=0." >&2
    exit 1
  fi
}

# ---------- 6. Cross-mode collision check (B-858 /stress A1.23 P0-1 ABC* OOB) ----------
# assert_no_cross_mode_collision <baseline> <site> <benchmark> <run_id> <log_prefix>
#
# B-858 (/stress A1.23 P0-1 ABC* OOB, 2026-05-17): defense against standalone-
# leaf cross-mode race window. Pre-fix leaf script `pgrep -f
# "run_experiment.py.*${RUN_ID}"` (queue_baseline.sh:139 + 3 phantom siblings)
# matches by FULL RUN_ID — second manual leaf invocation with DIFFERENT mode
# (e.g. `bash queue_baseline.sh B0 dom reddit` running + new
# `bash queue_baseline.sh B0 som reddit`) has different RUN_ID → pgrep doesn't
# match → reset_and_auth_gate proceeds → wipes site state under existing
# detached runner. queue_chain.sh:248 _collision_match enforces this at chain
# layer; B-858 propagates the check to standalone leaf entry point so the
# CLAUDE.md-documented `RESET_BEFORE=1 bash scripts/queues/queue_baseline.sh
# B0 dom shopping` pattern is paper-grade safe in isolation.
#
# Pattern: pgrep site+baseline+date prefix; exclude current RUN_ID exact match
# (avoid self-match); for VWA also exclude `_wa_` substring overlap. Non-empty
# → FATAL with diagnostic + ntfy alert.
assert_no_cross_mode_collision() {
  local baseline="${1:?baseline required}"
  local site="${2:?site required}"
  local benchmark="${3:?benchmark required}"
  local run_id="${4:?run_id required}"
  local log_prefix="${5:-leaf}"

  local site_pattern
  if [[ "${benchmark}" == "wa" ]]; then
    site_pattern="_wa_${site}_[0-9]{8}_"
  else
    # VWA: anchor `_<site>_<date>_` but filter WA siblings post-match.
    site_pattern="_${site}_[0-9]{8}_"
  fi

  # R2-P0-2-B* (/stress Phase 0 post-fix Mode B codex F2 OOB, 2026-05-19):
  # pre-fix `_${baseline}_` filter only matches same-baseline runners. Cross-
  # baseline collision (e.g., B0 cls running + manual leaf B1 cls) bypassed
  # this check → reaches reset_and_auth_gate → wipes session under B0's
  # live runner. Post-fix: TWO-PASS check. Pass 1 (same-baseline cross-mode,
  # existing B-858 semantic, FATAL). Pass 2 (cross-baseline same-site,
  # NEW R2-P0-2-B*, FATAL).
  local site_collisions
  site_collisions="$(pgrep -af "run_experiment.*${site_pattern}" 2>/dev/null \
                     | grep -v -F "${run_id}" || true)"
  if [[ "${benchmark}" == "vwa" && -n "${site_collisions}" ]]; then
    site_collisions="$(echo "${site_collisions}" | grep -v "_wa_" || true)"
  fi
  if [[ -n "${site_collisions}" ]]; then
    # Discriminate same-baseline vs cross-baseline within site_collisions.
    local same_baseline_collisions
    same_baseline_collisions="$(echo "${site_collisions}" | grep -E "_${baseline}_" || true)"
    local cross_baseline_collisions
    cross_baseline_collisions="$(echo "${site_collisions}" | grep -v -E "_${baseline}_" || true)"

    if [[ -n "${same_baseline_collisions}" ]]; then
      echo "[${log_prefix}][FATAL] same baseline+site different-mode runner already active:" >&2
      echo "${same_baseline_collisions}" | sed 's/^/  /' >&2
      echo "[${log_prefix}][FATAL] paper-grade hard rule: 同 site 单 baseline (cross-mode also forbidden)" >&2
      echo "[${log_prefix}][FATAL] B-858 (/stress A1.23 P0-1 ABC*): standalone-leaf cross-mode race vector." >&2
      echo "[${log_prefix}][FATAL] options: (a) 'pkill -f \"run_experiment.*_${baseline}_${site}_\"' kill existing; (b) wait for existing run; (c) use queue_chain.sh orchestration." >&2
      if command -v curl > /dev/null; then
        curl -L -d "leaf FATAL: ${baseline}/${site} cross-mode collision (B-858 A1.23 P0-1)" \
          "ntfy.sh/${NTFY_TOPIC:-p79-exp-dgx-spark}" 2>/dev/null || true
      fi
      exit 1
    fi
    if [[ -n "${cross_baseline_collisions}" ]]; then
      echo "[${log_prefix}][FATAL] cross-baseline same-site runner already active (R2-P0-2-B* /stress Phase 0 post-fix 2026-05-19):" >&2
      echo "${cross_baseline_collisions}" | sed 's/^/  /' >&2
      echo "[${log_prefix}][FATAL] paper-grade hard rule: 同 site 同时只能跑一个 baseline (B0 XOR B1 XOR B2)" >&2
      echo "[${log_prefix}][FATAL] CLAUDE.md hard rule #1: shared docker container + same user account → cross-pollination" >&2
      echo "[${log_prefix}][FATAL] options: (a) wait for existing run to complete; (b) 'pkill -f run_experiment.*_${site}_' (DESTRUCTIVE); (c) queue_chain.sh orchestration with cls/red/shop sequencing" >&2
      if command -v curl > /dev/null; then
        curl -L -d "leaf FATAL: ${baseline}/${site} cross-baseline collision (R2-P0-2-B*)" \
          "ntfy.sh/${NTFY_TOPIC:-p79-exp-dgx-spark}" 2>/dev/null || true
      fi
      exit 1
    fi
  fi
}

# ---------- 6b. Host-chain single-site refusal (R2-P0-1-B* /stress Phase 0 post-fix Mode B codex F1 OOB) ----------
# assert_no_other_site_chain_running <self_site> <log_prefix>
#
# R2-P0-1-B* (/stress Phase 0 post-fix Mode B codex F1 OOB, 2026-05-19): pre-fix
# `queue_phase1_paper_grade.sh launch all` correctly sequenced cls→red via
# sentinel-based wait (P0-2-B*), BUT single-site filters `launch cls` / `launch
# red` returned immediately after spawning detached chains. Operator could then
# run `launch red` while `launch cls` was active → recreates cross-site
# contention class (Fire-3 cls+red parallel = same failure mode pre-fix).
# Post-fix: refuse single-site launch if another site chain is alive,
# unless explicit `PHASE1A_PARALLEL=1` dev opt-in.
#
# Detection: pgrep -f "queue_phase1_(cls|red|shop)_<TS>_<PID>.pid"
# OR active runner with different `_<other-site>_<date>_` pattern.
assert_no_other_site_chain_running() {
  local self_site="${1:?self_site required}"  # cls | red | shop
  local log_prefix="${2:-leaf}"

  # PHASE1A_PARALLEL=1 dev opt-in skips check
  if [[ "${PHASE1A_PARALLEL:-0}" == "1" ]]; then
    echo "[${log_prefix}][lock] PHASE1A_PARALLEL=1 — host-chain check SKIPPED (dev mode)" >&2
    return 0
  fi

  local other_chains=""
  for site in cls red shop; do
    [[ "${site}" == "${self_site}" ]] && continue
    # Check active runner for that site
    local runner_match
    # For cls site filter look for run_experiment.*_classifieds_ pattern;
    # for red look for run_experiment.*_reddit_; for shop look for run_experiment.*_shopping_.
    local site_run_pattern=""
    case "${site}" in
      cls) site_run_pattern="run_experiment.*_classifieds_[0-9]{8}_" ;;
      red) site_run_pattern="run_experiment.*_reddit_[0-9]{8}_" ;;
      shop) site_run_pattern="run_experiment.*_shopping_[0-9]{8}_" ;;
    esac
    if [[ -n "${site_run_pattern}" ]]; then
      runner_match="$(pgrep -af "${site_run_pattern}" 2>/dev/null | grep -v "_wa_" || true)"
      if [[ -n "${runner_match}" ]]; then
        other_chains="${other_chains}\n[${site}]\n${runner_match}"
      fi
    fi
  done

  if [[ -n "${other_chains}" ]]; then
    echo "[${log_prefix}][FATAL] another site chain active — single-site launch refused (R2-P0-1-B*):" >&2
    printf "%b\n" "${other_chains}" | sed 's/^/  /' >&2
    echo "[${log_prefix}][FATAL] paper-grade hard rule #3: 同物理 host 同时只能跑一条 site chain (cls XOR red XOR shop)" >&2
    echo "[${log_prefix}][FATAL] options: (a) wait for existing chain to complete; (b) PHASE1A_PARALLEL=1 for dev opt-in (NOT paper-grade); (c) use 'launch all' for sequential cls→red orchestration" >&2
    if command -v curl > /dev/null; then
      curl -L -d "leaf FATAL: single-site ${self_site} launch refused (R2-P0-1-B* host-chain in progress)" \
        "ntfy.sh/${NTFY_TOPIC:-p79-exp-dgx-spark}" 2>/dev/null || true
    fi
    exit 1
  fi
}

# ---------- 7. Per-RUN_ID watchdog flock (B-907 /stress A2.2 P0-5-B* OOB) ----------
# acquire_watchdog_lock <run_id> [<label>]
# release_watchdog_lock
#
# B-907 (/stress A2.2 P0-5-B* codex F1 OOB, 2026-05-17): two queue leaves passing
# the pgrep TOCTOU window simultaneously spawn 2 watchdogs for SAME RUN_ID. Both
# share the same WD_STATE file + `.tmp` path → mutual overwrite of
# seen_keys / session_contaminated / error_retry_counts / seen_completions;
# may double-fire cleanup / post-analysis / gallery / figures. flock acquired on
# fd 8 (chain fd 9 + leaf-site fd 7 already used per acquire_site_lock).
#
# Lock file: `${REPO_DIR}/.locks/watchdog_${run_id}.lock`. Held until script
# exit (trap release in caller). Auto-released on subshell exit / `exec 8>&-`.
#
# Caller pattern:
#   acquire_watchdog_lock "${RUN_ID}" "queue_baseline" || exit $?
#   trap "release_watchdog_lock; release_site_lock" EXIT INT TERM
acquire_watchdog_lock() {
  local run_id="${1:-}" label="${2:-leaf}"
  if [[ -z "${run_id}" ]]; then
    echo "[${label}][error] acquire_watchdog_lock: run_id required" >&2
    return 1
  fi
  local repo_dir="${REPO_DIR:-$(pwd)}"
  local lock_dir="${repo_dir}/.locks"
  mkdir -p "${lock_dir}" 2>/dev/null || true
  local lock_file="${lock_dir}/watchdog_${run_id}.lock"
  exec 8>"${lock_file}"
  if ! flock -n 8; then
    echo "[${label}][FATAL] another watchdog holds lock for run_id=${run_id} (B-907)" >&2
    echo "[${label}][FATAL] lock file: ${lock_file}" >&2
    echo "[${label}][FATAL] if stale (prior watchdog crashed), 'rm ${lock_file}' to force-release" >&2
    exec 8>&-
    return 78  # rc=78 = lock contention (matches acquire_site_lock convention)
  fi
  WATCHDOG_LOCK_FD=8
  WATCHDOG_LOCK_FILE="${lock_file}"
  echo "[${label}][lock] acquired ${lock_file} (pid $$, B-907)" >&2
  return 0
}

release_watchdog_lock() {
  if [[ "${WATCHDOG_LOCK_FD:-}" == "8" ]]; then
    exec 8>&-
    unset WATCHDOG_LOCK_FD WATCHDOG_LOCK_FILE
  fi
}

# ============================== Gate G8 ==============================
# Fire-4 RCA Wave 2 M6 (/stress 3-AI 2026-05-19): cross-fire quarantine
# registry investigation gate. Pre-fix Fire-3 task 75 quarantined +
# Fire-4 task 75 re-quarantined (same URL id=84148), no infrastructure
# to halt Fire-5 from blindly rediscovering. Post-fix: registry at
# `docs/checkpoints/quarantine_registry.jsonl` tracks per-task
# quarantine + classification events; if any task has >= threshold
# unclassified quarantine events, Gate G8 halts the fire with
# "investigation required, NOT auto-skip" per user decision 2026-05-19.
#
# Usage: assert_quarantine_gate <site> <tasks_spec> [halt_threshold]
#   <site>: classifieds | reddit | shopping
#   <tasks_spec>: '0-233' or '1,3,5' or '75' (passed to registry CLI)
#   [halt_threshold]: default 1; env QUARANTINE_HALT_THRESHOLD overrides
#
# Exit: 0 ok, 1 halt (registry CLI's own exit code propagated).
assert_quarantine_gate() {
  local site="${1:?site required}"
  local tasks_spec="${2:?tasks_spec required (e.g., '0-233')}"
  local halt_threshold="${3:-1}"

  # Locate repo root via this file's path (one level up from scripts/queues/).
  local _lib_path="${BASH_SOURCE[0]}"
  local repo_root
  repo_root="$(cd "$(dirname "${_lib_path}")/../.." && pwd)"
  local registry_cli="${repo_root}/scripts/maintenance/quarantine_registry.py"

  if [[ ! -f "${registry_cli}" ]]; then
    echo "[gate_G8] WARN quarantine_registry.py not found at ${registry_cli} — skipping G8 (registry not deployed)" >&2
    return 0
  fi

  # Prefer .venv python3 if available, else system python3.
  local py_bin="python3"
  if [[ -x "${repo_root}/.venv/bin/python3" ]]; then
    py_bin="${repo_root}/.venv/bin/python3"
  fi

  echo "[gate_G8] preflight quarantine registry check: site=${site} tasks=${tasks_spec} threshold=${halt_threshold}" >&2
  if "${py_bin}" "${registry_cli}" preflight --site "${site}" --tasks "${tasks_spec}" --halt-threshold "${halt_threshold}"; then
    echo "[gate_G8] OK" >&2
    return 0
  else
    local _rc=$?
    echo "[gate_G8] FATAL: investigation required for site=${site}. Operator must:" >&2
    echo "[gate_G8]   1. Reproduce flagged task(s) via Wave 4 M7 (Playwright/manual)." >&2
    echo "[gate_G8]   2. Classify via: ${py_bin} ${registry_cli} classify --site=${site} --task-id=<N> --as=<class> --rationale=<...>" >&2
    echo "[gate_G8]   3. Then re-run preflight to confirm gate clears." >&2
    if command -v curl > /dev/null; then
      curl -L -d "Gate G8 HALT: site=${site} has unclassified quarantine events; Fire-5 BLOCKED until investigation (Wave 4 M7) completes." \
        "ntfy.sh/${NTFY_TOPIC:-p79-exp-dgx-spark}" 2>/dev/null || true
    fi
    return "${_rc}"
  fi
}
