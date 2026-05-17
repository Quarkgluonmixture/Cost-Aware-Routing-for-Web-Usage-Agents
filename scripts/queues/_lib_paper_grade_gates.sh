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
  if [[ "$(hostname)" == *a100* ]] \
     || [[ "$(hostname)" == *condense* ]] \
     || [[ "${P79_PAPER_GRADE_HOST:-0}" == "1" ]] \
     || [[ "$(pwd)" == *workspace/p79* ]] \
     || [[ -d /home/ubuntu/workspace/p79 ]]; then
    echo "[preflight] A100 URL-locality gate ACTIVE on host=$(hostname), cwd=$(pwd)" >&2
    local _v
    for _v in CLASSIFIEDS REDDIT SHOPPING WIKIPEDIA HOMEPAGE; do
      case "${!_v:-}" in
        *localhost*|*127.0.0.1*|"") ;;
        *) echo "✗ FATAL preflight: \$${_v}=${!_v} not local on A100 host; refusing launch" >&2; exit 2 ;;
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
  # B-587 (A1.13 P1-10 Claude, 2026-05-17): now actually using %N nanoseconds
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
      # B-588 (A1.13 P1-9 Claude OOB, 2026-05-17): stale-resume fingerprint
      # check. Pre-fix blindly resumed mtime-newest match even when its
      # condition_meta.json showed pre-fix schema_version. Now verify v2 schema
      # before resume; mismatch → fresh timestamp, log "skipping stale".
      local stale=0
      local meta
      for meta in "${existing}"/*/condition_meta.json; do
        if [[ -f "${meta}" ]]; then
          if ! grep -q '"schema_version"[[:space:]]*:[[:space:]]*"v2' "${meta}" 2>/dev/null; then
            stale=1
            break
          fi
        fi
      done
      if [[ "${stale}" == "1" ]]; then
        echo "[${log_prefix}] skipping stale resume candidate $(basename "${existing}") (schema_version != v2); minting fresh run_id"
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
  # B-581 (A1.13 P0-5 gemini G4 OOB, 2026-05-17): export RUN_TS_FULL so callers
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
  local site="$1"
  local repo_dir="$2"
  local python_bin="$3"
  local log_prefix="$4"
  local reset_label="$5"

  if [[ ! -f "${repo_dir}/scripts/maintenance/reset_vwa_sites.sh" ]]; then
    echo "[${log_prefix}][error] reset_vwa_sites.sh not found but RESET_BEFORE=1; aborting." >&2
    echo "[${log_prefix}][error] To bypass reset (paper-grade dirty), explicitly set RESET_BEFORE=0." >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "${repo_dir}/scripts/maintenance/reset_vwa_sites.sh"
  echo "[${log_prefix}] RESET_BEFORE=1 → resetting site=${site}..."
  if reset_vwa_sites "${site}" "${reset_label}"; then
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
    if "${python_bin}" -c "
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
      if [[ "${AUTH_GATE_BYPASS:-0}" != "1" ]]; then
        exit 1
      fi
      echo "[${log_prefix}][warn] AUTH_GATE_BYPASS=1 set — proceeding without auth gate; first 1-3 tasks at risk." >&2
    fi
  else
    local rc=$?
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
