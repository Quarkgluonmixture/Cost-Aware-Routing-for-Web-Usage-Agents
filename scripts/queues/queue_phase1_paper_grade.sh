#!/usr/bin/env bash
# queue_phase1_paper_grade.sh — Master orchestrator for Phase 1 paper-grade rerun.
# (Renamed 2026-05-13 from queue_16cell_paper_grade.sh; old name reflected prior
# 16-cell phantom-only scope that codex stress audit identified as incomplete.)
#
# Scope (revised 2026-05-14 — Gemma3-VL added as 3rd baseline, advisor 笔记 §138):
#   Phase 1a (THIS SCRIPT default): 36 operational conditions = 2 sites (cls, red)
#     × 3 models (B0, B1, B2) × 6 modes (DOM, SoM, Vision, P-text, P-prompt, P-SoM).
#     Statistical analysis: 6 (site, model) cells, pooled meta + TOST.
#     Target: workshop submission. (Was 24 conditions / 4 cells pre-2026-05-14;
#     B2 = Gemma3-VL google/gemma-3-4b-it, cross-family control.)
#   Phase 1b (deferred, requires explicit 'launch phase1b shop'): 18 additional
#     conditions = shop × 3 models × 6 modes. Feeds main paper R3→R1 framing
#     decision post-workshop submission.
#
# **Hard rule: Same site, one baseline only (B0 / B1 / B2)**. queue_chain wraps
# reset+watchdog+idempotent and enforces the 3-way same-site collision check.
# Splits into 2 parallel chains (cls / red) for Phase 1a, each chain internally sequential.
#
# Pre-launch gates (must all pass):
#   - Advisor email reply received → preregistration.md status `draft` → `locked`
#   - A100 SSH connectivity verified ('ssh condense-a100 nvidia-smi' returns OK)
#   - VWA stack running on chosen host (DGX→quark Tailscale OR A100 self-host)
#   - env_snapshot baseline committed (results/provenance/env_<host>_baseline.json)
#   - VWA snapshot baseline committed (results/provenance/vwa_<host>_baseline.json)
#
# Usage:
#   bash scripts/queues/queue_phase1_paper_grade.sh dry-run            # preview, no launch
#   bash scripts/queues/queue_phase1_paper_grade.sh launch             # Phase 1a (cls+red, 36 conditions)
#   bash scripts/queues/queue_phase1_paper_grade.sh launch cls         # only classifieds Phase 1a chain (18 conditions)
#   bash scripts/queues/queue_phase1_paper_grade.sh launch red         # only reddit Phase 1a chain (18 conditions)
#   bash scripts/queues/queue_phase1_paper_grade.sh launch phase1b     # Phase 1b VWA shop chain (18 conditions, deferred to post-workshop)
#   bash scripts/queues/queue_phase1_paper_grade.sh launch wa_shop        # WA shopping (18 conditions, 173 scored tasks each)
#   bash scripts/queues/queue_phase1_paper_grade.sh launch wa_shop_admin  # WA shopping_admin (18 conditions, 176 scored tasks each)
#
# B-1935 (2026-08-03): the three shop chains above all mutate the SAME Magento
# container (`vwa-shopping`, 7770 storefront + 7780 admin). WA is not a second
# stack — it is the same container reached through WA's task files. They share
# one container lock (B-1934) and must be run one after another; a second launch
# while one is alive aborts, which is the gate doing its job.
#
# Phase 1a conditions (36 total):
#   cls chain (18 conditions, B0 → B1 → B2 sequential):
#     - B0 cls × {DOM, SoM, Vision, P-text, P-SoM, P-prompt}  (6)
#     - B1 cls × {DOM, SoM, Vision, P-text, P-SoM, P-prompt}  (6)
#     - B2 cls × {DOM, SoM, Vision, P-text, P-SoM, P-prompt}  (6)
#   red chain (18 conditions, B0 → B1 → B2 sequential):
#     - B0 red × {DOM, SoM, Vision, P-text, P-SoM, P-prompt}  (6)
#     - B1 red × {DOM, SoM, Vision, P-text, P-SoM, P-prompt}  (6)
#     - B2 red × {DOM, SoM, Vision, P-text, P-SoM, P-prompt}  (6)
#
# Phase 1b conditions (18 total, deferred main-paper expansion):
#     - B0 shop × {DOM, SoM, Vision, P-text, P-SoM, P-prompt}  (6)
#     - B1 shop × {DOM, SoM, Vision, P-text, P-SoM, P-prompt}  (6)
#     - B2 shop × {DOM, SoM, Vision, P-text, P-SoM, P-prompt}  (6)
#
# Chain dependency:
#   cls and red can run in parallel (different sites = no resource contention beyond A100 GPU).
#   Within each chain B0 → B1 → B2 sequential (same-site baselines share the site
#   docker container + user account login).
#   Phase 1b shop launched separately after workshop submission to avoid Magento FPC bug
#   surface co-occurring with Phase 1a critical path.
#
# ETA estimates (A100 40GB; B2 ≈ B1 throughput, both ~4B local):
#   cls chain (18 conditions): B0 (~24h) → B1 (~48h) → B2 (~48h) = 120h ≈ 5 days
#   red chain (18 conditions): B0 (~20h) → B1 (~40h) → B2 (~40h) = 100h ≈ 4 days
#   Total Phase 1a wallclock with 2 parallel chains = max(120, 100) ≈ 5 days
#   Phase 1b shop chain (18 conditions): B0 (~32h) → B1 (~64h) → B2 (~64h) = 160h ≈ 6.5 days (deferred)
#
# Sentinel files (used by chain to detect completion):
#   results/visualwebarena/phase1/<run_id>/<condition_id>/condition_summary_v2.json

# B-677 (/stress A1.14 Chunk b P1-1 Claude+gemini 2-AI AC, 2026-05-17):
# `set -euo pipefail` (was `set -uo pipefail`) for fail-fast on uncaught errors.
# Pre-fix: `mkdir -p logs` / `echo "$pid" > pidfile` / unexpected nohup spawn
# failures silently swallowed; script could print "OK" while child failed.
# Sibling drift: 5/7 leaf queue scripts already use `set -euo pipefail`; only
# orchestrator + queue_chain were `-uo`. Defensive `|| true` / `|| rc=$?`
# wrappers added per check_gates command that legitimately tolerates non-zero
# exit (preflight rc capture / Gate 6 pgrep no-match / Gate 5 CUDA probe).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

# B-395 (/stress A1.1 v8 3-AI overlap P0-1, 2026-05-16): export P79_PAPER_GRADE
# so `p79/experiment/config.py:normalize_config` sets top-level
# `cfg["paper_grade"] = True` → runner propagates to backend cfg → B0
# `ApiProxyBackend` forwards to agent → B-340 GLM hard-block at
# `proxy_api_agent.py:179-186` raises if any yaml still has
# `use_glm_fallback: true`. Defense-in-depth: (a) all B0 paper-1 yamls
# explicit `use_glm_fallback: false` (B-396), (b) this env wire makes
# B-340 hard-block reachable, (c) B-340 RuntimeError fail-fast at init.
export P79_PAPER_GRADE=1

# B-1839 (Gate 3, 2026-05-23): per-condition docker restart for fresh substrate.
# canary R11315 ran on a 6-7 day-old classifieds container (substrate decay behind
# Fire-5/6 eval-timeout windows + cross-condition latency confound). When set,
# reset_vwa_sites.sh:_reset_vwa_local_classifieds restarts cls app+db per condition
# + waits db ping/http 200 (reddit already does docker rm+run = fresh). Exported
# here so it propagates queue_chain → queue_baseline → reset_and_auth_gate's
# `setsid bash -c` → reset_vwa_sites. Default 1 for paper-grade fire; opt-out with
# a `VWA_RESTART_DOCKER=0` prefix (e.g. fast dev rerun on an already-fresh stack).
export VWA_RESTART_DOCKER="${VWA_RESTART_DOCKER:-1}"

# P0-1-AB (/stress GRL audit 2026-05-20, user Q1=A): paper_grade XOR
# diagnostic_replay — queue-layer mirror of the runner hard block. A leaked
# P79_DIAGNOSTIC_REPLAY / QUARANTINE_DIAGNOSTIC_REPLAY env (e.g. left exported
# from a queue_diagnostic_replay.sh session) would route this canonical fire to
# results/diagnostic_replay/ + sr_excluded=True + suppress the M1 abort —
# silent zero-canonical-data waste. Fail BEFORE launch. Only
# queue_diagnostic_replay.sh (no P79_PAPER_GRADE) may set these.
if [ -n "${P79_DIAGNOSTIC_REPLAY:-}" ] || [ -n "${QUARANTINE_DIAGNOSTIC_REPLAY:-}" ]; then
  echo "[queue_phase1_paper_grade] FATAL: P79_PAPER_GRADE=1 is incompatible with" >&2
  echo "  P79_DIAGNOSTIC_REPLAY=${P79_DIAGNOSTIC_REPLAY:-<unset>} /" >&2
  echo "  QUARANTINE_DIAGNOSTIC_REPLAY=${QUARANTINE_DIAGNOSTIC_REPLAY:-<unset>}." >&2
  echo "  Diagnostic replay is non-canonical (sr_excluded + M1 abort suppressed)." >&2
  echo "  Unset both for a canonical fire, or use queue_diagnostic_replay.sh." >&2
  exit 2
fi

# B-673 (/stress A1.14 Chunk a P0-2, Claude+codex 2-AI OOB AB, 2026-05-17):
# orchestrator must enforce paper-grade target host (A100 self-hosted) BEFORE
# running gates. Header line 20-26 advertises "A100 SSH connectivity verified"
# + "VWA stack on chosen host" as required, but `check_gates` pre-fix had ZERO
# implementation of either — DGX dev session + Tailscale endpoints could pass
# every gate and produce hero numbers from the wrong substrate.
#
# Side-effect closure of P1-8 (Claude unique OOB): sourcing _lib_paper_grade_gates
# pulls in init_paper_grade_env which loads vwa_env_remote.sh — preflight Gate 4
# now has DATASET/SHOPPING/REDDIT env populated even from fresh shell.
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_lib_paper_grade_gates.sh"
init_paper_grade_env "${REPO_DIR}"

MODE="${1:-dry-run}"
SITE_FILTER="${2:-all}"

# B-1830 followup (2026-05-22, user directive): manifest auto-bind FAIL-CLOSED halt
# gate. The watchdog's in-pipeline _auto_bind_manifest drops this marker on
# ghost / ambiguous / write-error. A corrupt or ambiguous manifest means the
# aggregator cannot bind authoritative runs (the paper evidence chain) → refuse to
# relaunch (a fire would pile MORE conditions on top of an unresolvable manifest).
# dry-run (plan-only) is allowed so the operator can still inspect; launch is blocked.
_manifest_halt="${REPO_DIR}/.locks/manifest_bind_halt.marker"
if [ "${MODE}" = "launch" ] && [ -f "${_manifest_halt}" ]; then
  echo "[queue_phase1_paper_grade] FATAL: manifest auto-bind halt marker present:" >&2
  echo "  ${_manifest_halt}" >&2
  echo "  The watchdog's in-pipeline manifest bind hit ghost / ambiguous / write-error." >&2
  echo "  Resolve: 'python3 scripts/analysis/validate_fire_manifest.py' to inspect," >&2
  echo "  bind the authoritative run manually (or clear the ghost), then" >&2
  echo "  'rm ${_manifest_halt}' to re-enable relaunch." >&2
  exit 1
fi

log() { echo "[phase1 $(date '+%H:%M:%S')] $*"; }
fail() { log "FAIL: $*"; exit 1; }

# B-1604 (/stress 深入审 Mode A P1-7-A, 2026-05-18): cold-start warmup
# preflight. Empirical 2026-05-18 A100 probe: VWA classifieds cold-curl
# `index.php?page=login` = 9.96s (PHP-FPM worker spin-up + Postgres pool
# init + Magento FPC TTL); warm-curl after 3 sequential = 0.085-0.113s
# (~117× speedup). VWA shopping cold = 14.68s (Magento heavier), warm
# = 0.103-0.130s. If docker stack restarted mid-Pass-1 (or fresh
# `vwa_reset.sh` fires between conditions), head-of-chain task playwright
# `wait_until=load` 30s budget could trip on next cold path → silent
# 1-2 task crashes attributed to "agent / proxy failure" instead of
# infra fragility. Warm 3 sites × 3 curls/site = 9 cheap requests before
# any condition launches; total wallclock < 30s.
#
# WARMUP_SKIP=1 env opt-out: dev/debug runs where VWA stack just
# launched OR operator wants to test the cold-path failure mode itself.
warmup_vwa_sites() {
  if [ "${WARMUP_SKIP:-0}" = "1" ]; then
    log "  WARMUP_SKIP=1 — skipping VWA cold-start warmup"
    return 0
  fi
  log "=== Warmup: VWA cold-start defuse (cls / red / shop × 3 curls each) ==="
  local CLS_URL="${CLASSIFIEDS:-http://localhost:9980}/index.php?page=login"
  local RED_URL="${REDDIT:-http://localhost:9999}/"
  local SHOP_URL="${SHOPPING:-http://localhost:7770}/"
  local site url i code time_total
  for site_url_pair in "cls:${CLS_URL}" "red:${RED_URL}" "shop:${SHOP_URL}"; do
    site="${site_url_pair%%:*}"
    url="${site_url_pair#*:}"
    for i in 1 2 3; do
      # `--max-time 30` mirrors playwright wait_until=load budget; emit
      # latency for cold-vs-warm contrast in launch log.
      code_time=$(curl -sS -o /dev/null --max-time 30 -w "%{http_code} %{time_total}" "$url" 2>&1 || echo "FAIL FAIL")
      log "  warmup ${site} curl${i}: HTTP ${code_time%% *} time=${code_time##* }s"
    done
  done
  log "  warmup done"
}

# B-673: require paper-grade host (A100) before gates run.
# B-1406 (/stress A2.7 P1-4-AB* 2026-05-18): local definition retired,
# canonical `require_paper_grade_host` lives in `_lib_paper_grade_gates.sh`
# (already sourced above at L100). Mode A F1 + Mode B F5 caught the sibling-
# propagation regex permissive match + duplicate-def attack vectors.
# Override knob: `P79_PAPER_GRADE_HOST=1` for CI / future approved hostnames.

# ---------------------------------------------------------------------------
# Pre-launch gates
# ---------------------------------------------------------------------------

check_gates() {
  local errors=0

  log "=== Gate 1: preregistration.md threshold lock ==="
  # B-708 (/stress A1.14 Chunk d P2-5 Claude unique, 2026-05-17): broaden TBD
  # detection. Pre-fix narrow grep `K_h1.*TBD|K_h3.*TBD|TOST.*TBD` only matched
  # 3 specific phrase patterns; preregistration body had 2+ other TBDs (e.g.,
  # line 245 "implementing TBD on land", line 432 "Reproducible split via
  # scripts/analysis/router_split.py (TBD)") that pre-fix gate ignored. Paper-grade
  # principle: ANY TBD in a locked prereg is a draft-state marker. Allowlist via
  # `<!-- TBD-ALLOW: <reason> -->` HTML comment if intentional.
  if grep -nE "^[^<].*TBD" docs/checkpoints/pre_run/preregistration.md 2>/dev/null | grep -v 'TBD-ALLOW' | grep -q TBD; then
    log "  FAIL: preregistration.md still has un-allowlisted TBD markers."
    log "        Either resolve the TBD or add '<!-- TBD-ALLOW: <reason> -->' on the line for intentional placeholders."
    log "        Run: grep -n TBD docs/checkpoints/pre_run/preregistration.md    # locate"
    errors=$((errors+1))
  elif ! grep -q "^status: locked" docs/checkpoints/pre_run/preregistration.md 2>/dev/null; then
    # Gate 1b added 2026-05-13 (codex audit HIGH-1): launch_checklist.md
    # requires prereg `status: locked` before paper-grade rerun. Previously
    # only TBD threshold was checked; status=draft could pass.
    log "  FAIL: preregistration.md status is not 'locked' (still draft / pending advisor)."
    log "        Once advisor signs, flip 'status: draft' → 'status: locked' in"
    log "        docs/checkpoints/pre_run/preregistration.md before paper-grade launch."
    errors=$((errors+1))
  else
    log "  OK"
  fi

  # B-681 (/stress A1.14 Chunk c P1-6 Claude+codex 2-AI AB, 2026-05-17):
  # "committed" means git-tracked AND clean, not just file-exists. Pre-fix
  # Gate 2+Gate 3 only `ls -d` the path → untracked / dirty / wrong-host
  # provenance all passed → OSF reviewer running paper-grade re-fire would
  # produce different fingerprint than the committed baseline. 3-layer check:
  #   (a) `ls` file exists
  #   (b) `git ls-files --error-unmatch` file is tracked (catches new untracked)
  #   (c) `git diff --quiet HEAD --` file matches committed content (catches dirty)
  #   (d) JSON schema: `captured_at` + `host` fields present + `errors` array empty
  _check_provenance_baseline() {
    local label="$1" pattern="$2" snapshot_cmd="$3"
    log "=== ${label} ==="
    local files=()
    while IFS= read -r f; do [[ -n "$f" ]] && files+=("$f"); done < <(ls $pattern 2>/dev/null || true)
    if [[ ${#files[@]} -eq 0 ]]; then
      log "  FAIL: No matching file found ($pattern)"
      log "        Run: ${snapshot_cmd}"
      errors=$((errors+1))
      return
    fi
    local fail_this=0
    for f in "${files[@]}"; do
      # (b) git tracked
      if ! git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
        log "  FAIL: $f exists on disk but NOT git-tracked (untracked provenance ≠ reproducibility evidence)"
        log "        Run: git add $f && git commit -m 'provenance: capture baseline for paper-grade Phase 1a'"
        fail_this=$((fail_this+1)); continue
      fi
      # (c) clean (no uncommitted diff vs HEAD)
      if ! git diff --quiet HEAD -- "$f" 2>/dev/null; then
        log "  FAIL: $f is git-tracked but has uncommitted diff vs HEAD (dirty content invalidates fingerprint)"
        log "        Run: git diff -- $f    # inspect changes"
        log "        Then: git add $f && git commit -m 'provenance: refresh baseline'"
        fail_this=$((fail_this+1)); continue
      fi
      # (d) JSON schema sanity — captured_at + host required; errors should be empty array.
      local schema_check
      schema_check=$(.venv/bin/python3 -c "
import json, sys
try:
    d = json.load(open('$f'))
except Exception as e:
    print(f'json_parse:{e}'); sys.exit(1)
missing = [k for k in ('captured_at', 'host') if k not in d]
if missing:
    print(f'missing_fields:{missing}'); sys.exit(1)
errs = d.get('errors', [])
if isinstance(errs, list) and errs:
    # truncate to 2 errors for log brevity
    print(f'has_errors:{errs[:2]}'); sys.exit(1)
print(f'OK:host={d[\"host\"]},at={d[\"captured_at\"]}')
" 2>/dev/null || echo "schema_check_failed")
      if [[ "${schema_check}" != OK:* ]]; then
        log "  FAIL: $f schema check: ${schema_check}"
        log "        Re-run snapshot to refresh: ${snapshot_cmd}"
        fail_this=$((fail_this+1)); continue
      fi
      log "  ✓ $f  (${schema_check#OK:})"
    done
    if [[ $fail_this -gt 0 ]]; then
      errors=$((errors+1))
    fi
  }

  _check_provenance_baseline \
    "Gate 2: env_snapshot baseline committed" \
    "results/provenance/env_*_baseline.json" \
    "python3 scripts/provenance/snapshot_env.py results/provenance/env_<host>_baseline.json"

  # B-676 (A1.14 Chunk a P1-5) Gate 3 WARN→FAIL still in effect; helper now
  # adds git-tracked + clean + schema validation per B-681.
  _check_provenance_baseline \
    "Gate 3: VWA snapshot baseline committed" \
    "results/provenance/vwa_*.json" \
    "bash scripts/provenance/snapshot_vwa.sh    # captures docker fingerprint + endpoint state"

  # B-1604 (/stress 深入审 Mode A P1-7-A, 2026-05-18): warm VWA sites BEFORE
  # Gate 4 preflight so playwright endpoint probes don't trip on cold PHP-FPM
  # worker spin-up (empirical: cls cold 9.96s vs warm 0.085s; shop cold
  # 14.68s vs warm 0.13s). Without this, Gate 4 preflight (10s playwright
  # probe) could itself become the cold-start victim → false-positive FAIL
  # at gate level + 35-condition wallclock loss to mis-attributed retry.
  # WARMUP_SKIP=1 env opt-out for dev/debug.
  warmup_vwa_sites

  # Gate 4 — BLOCKING (codex stress v6 C2): preflight exit code now captured.
  # Strict ports for actual paper-grade fire (--no-strict-ports dropped).
  # B-680 (A1.14 Chunk b P1-11 gemini F4 unique OOB C, 2026-05-17): explicit
  # `STRICT_PORTS=1` export + `--strict-ports` flag — no longer relies on
  # preflight_v2.sh's internal default. Defense-in-depth against future
  # dev-convenience default flip.
  log "=== Gate 4: VWA reachability ==="
  if [ -f scripts/preflight_v2.sh ]; then
    # B-677 (P1-1): set -e safe rc capture via `|| preflight_rc=$?`. Pre-fix
    # `preflight_rc=$?` after `preflight_out=$(...)` was unreachable under
    # set -e since the failing $(...) substitution would exit the script
    # before line `preflight_rc=$?`.
    preflight_rc=0
    # B-793 (/stress A1.9 cold-start P1-9 root-cause fix, 2026-05-17):
    # `--paper-grade` instructs preflight to actually instantiate
    # `VwaEvaluator(paper_grade=True)`. Pre-fix B-544 init-time fail-loud
    # only surfaced AT batch launch (condition #1 crash → 35 conditions of
    # wallclock lost). Now: surface init failures here (10s probe) rather
    # than mid-fire.
    preflight_out=$(STRICT_PORTS=1 bash scripts/preflight_v2.sh --strict-ports --paper-grade 2>&1) || preflight_rc=$?
    echo "$preflight_out" | tail -8 | sed 's/^/    /'
    if [ "$preflight_rc" -ne 0 ]; then
      log "  FAIL: preflight_v2.sh exited rc=$preflight_rc — paper-grade fire requires all preflight checks pass"
      errors=$((errors+1))
    else
      log "  OK"
    fi
  else
    log "  FAIL: scripts/preflight_v2.sh not found"
    errors=$((errors+1))
  fi

  # Gate 5 — BLOCKING (codex stress v6 C2 sibling): CUDA availability now gates launch.
  # B-678 (A1.14 Chunk b P1-2 codex F3 unique OOB B, 2026-05-17): real model
  # load smoke replaces title-only "GPU + model load smoke" theater. Pre-fix
  # only `torch.cuda.is_available()` ran — Qwen/Gemma HF cache miss /
  # transformers parse failure / revision-pin drift surfaced only at first
  # cell launch ~24-48h into Phase 1a wallclock burn. `AutoConfig.from_pretrained
  # local_files_only=True` validates cache presence + revision parsability
  # WITHOUT allocating VRAM (no full weight load); revisions pinned per
  # `configs/exp_v2_base.yaml:103+138` (Qwen=ebb281e... / Gemma=093f9f3...).
  log "=== Gate 5: GPU + model load smoke ==="
  if command -v .venv/bin/python3 &>/dev/null; then
    cuda_ok=$(.venv/bin/python3 -c "import torch; print('YES' if torch.cuda.is_available() else 'NO')" 2>/dev/null || echo "ERROR")
    if [ "$cuda_ok" = "YES" ]; then
      gpu_name=$(.venv/bin/python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "unknown")
      log "  CUDA OK (${gpu_name})"
      # B-678: actual model config load (no VRAM, validates HF cache + parse).
      model_smoke=$(.venv/bin/python3 -c "
import warnings; warnings.filterwarnings('ignore')
try:
    from transformers import AutoConfig
    AutoConfig.from_pretrained('Qwen/Qwen3-VL-4B-Instruct', revision='ebb281ec70b05090aa6165b016eac8ec08e71b17', local_files_only=True)
    AutoConfig.from_pretrained('google/gemma-3-4b-it', revision='093f9f388b31de276ce2de164bdc2081324b9767', local_files_only=True)
    print('YES')
except Exception as e:
    msg = str(e).replace('\n', ' ')[:200]
    print(f'NO:{msg}')
" 2>/dev/null || echo "NO:python-failed")
      if [ "$model_smoke" = "YES" ]; then
        log "  Model load OK (Qwen3-VL-4B@ebb281e + Gemma3-VL@093f9f3 configs in HF cache, revisions pinned)"
      else
        log "  FAIL: model load smoke failed: $model_smoke"
        log "        Cause: HF cache miss / revision drift / transformers parse error"
        log "        Run: HF_HUB_OFFLINE=0 .venv/bin/python3 -c \"from transformers import AutoConfig; AutoConfig.from_pretrained('Qwen/Qwen3-VL-4B-Instruct', revision='ebb281ec70b05090aa6165b016eac8ec08e71b17')\""
        errors=$((errors+1))
      fi
    else
      log "  FAIL: CUDA not available — paper-grade B1/B2 local inference requires GPU"
      errors=$((errors+1))
    fi
  else
    log "  FAIL: .venv/bin/python3 not found"
    errors=$((errors+1))
  fi

  # Gate 6 — BLOCKING (codex stress v6 C6): active-run detection now fatal.
  # RESET_BEFORE=1 in launch_chain would reset site state under any active
  # same-site runner. Set ALLOW_ACTIVE_RUNS=1 only after verifying no collision.
  log "=== Gate 6: No conflicting active runs ==="
  # B-677 (P1-1) set -e safe: pgrep no-match returns 1 + pipefail → pipe rc 1
  # → set -e exit. `|| true` masks empty-match so active=0 path works.
  # B-709 (A1.14 Chunk d P2-6 gemini F5, 2026-05-17): regex-anchored pgrep
  # pattern `python[a-zA-Z0-9_.]*.*run_experiment\.py.*--config` matches only
  # actual python process invocations of run_experiment.py — not vim/less/grep
  # viewers happening to have "run_experiment --config" string in their argv.
  local pgrep_pattern='(python|\.venv/bin/python3?)[a-zA-Z0-9_./-]* .*run_experiment\.py.*--config'
  active=$(pgrep -fa "${pgrep_pattern}" 2>/dev/null | wc -l || true)
  log "  Active run_experiment processes: $active"
  if [ "$active" -gt 0 ]; then
    pgrep -fa "${pgrep_pattern}" 2>/dev/null | sed 's/^/    /' || true
    if [ "${ALLOW_ACTIVE_RUNS:-0}" == "1" ]; then
      log "  WARN: $active existing run(s) detected but ALLOW_ACTIVE_RUNS=1 — proceeding."
      log "        You are responsible for verifying no same-site B0+B1 collision."
    else
      log "  FAIL: $active existing run_experiment process(es) detected."
      log "        Phase 1a fire with RESET_BEFORE=1 would reset site state under any active same-site runner."
      log "        Stop conflicting runs, OR set ALLOW_ACTIVE_RUNS=1 if you verified no site collision."
      errors=$((errors+1))
    fi
  else
    log "  OK (no active runs)"
  fi

  # Gate 7 — BLOCKING (codex stress v6 C9): every chain config must exist on disk.
  # Prevents advertising a chain that fails mid-run on a missing config.
  log "=== Gate 7: All chain configs exist ==="
  local missing_cfg=0
  local builders_to_check
  case "${SITE_FILTER}" in
    all|cls|red)     builders_to_check="build_cls_chain build_red_chain" ;;
    phase1b)         builders_to_check="build_shop_chain" ;;
    # B-1960 (2026-08-05): shop_b0 / wa_shop_b0 were never registered here, so
    # they fell through to the `*)` fallback and this gate verified the cls+red
    # configs instead — "all chain configs exist" was answered about a chain that
    # was not being launched. The 2026-08-04 shop_b0 fire passed Gate 7 on cls/red
    # configs. Note Gate 8 handles the same situation the opposite way (it refuses
    # rather than inspect something irrelevant); this gate now names every label.
    shop_b0)         builders_to_check="build_shop_b0_chain" ;;
    shop_b0_tail)    builders_to_check="build_shop_b0_tail_chain" ;;    # B-1957
    wa_shop)         builders_to_check="build_wa_shop_chain" ;;         # B-1935
    wa_shop_b0)      builders_to_check="build_wa_shop_b0_chain" ;;
    wa_shop_admin)   builders_to_check="build_wa_shop_admin_chain" ;;   # B-1935
    *)               builders_to_check="build_cls_chain build_red_chain" ;;
  esac
  for builder in ${builders_to_check}; do
    while IFS= read -r cmd; do
      [ -z "$cmd" ] && continue
      cfg_path="$(config_for_cmd "$cmd")"
      # B-672 (A1.14): config_for_cmd's default branch now emits UNKNOWN_SCRIPT:<name>
      # to fail-loud instead of silent empty echo (pre-fix could silently bypass
      # this gate for any chain command not in the case statement).
      if [[ "$cfg_path" == UNKNOWN_SCRIPT:* ]]; then
        log "  FAIL: $cfg_path — chain command uses unrecognized leaf script (update config_for_cmd in queue_phase1_paper_grade.sh)"
        log "        command: $cmd"
        missing_cfg=$((missing_cfg+1))
      elif [ -n "$cfg_path" ] && [ ! -f "$cfg_path" ]; then
        log "  FAIL: missing config $cfg_path  (for: $cmd)"
        missing_cfg=$((missing_cfg+1))
      fi
    done < <($builder)
  done
  if [ "$missing_cfg" -gt 0 ]; then
    log "  $missing_cfg chain config(s) missing — cannot launch"
    errors=$((errors+1))
  else
    log "  OK (all chain configs present)"
  fi

  # Fire-4 RCA Wave 2 M6 (/stress 3-AI 2026-05-19): Gate 8 cross-fire
  # quarantine registry investigation gate. Halts if any task has >=1
  # unclassified quarantine event in docs/checkpoints/quarantine_registry.jsonl.
  # User decision 2026-05-19: investigation gate, NOT auto-skip — operator
  # must Wave 4 M7 reproduce + classify before the next fire can proceed.
  # cls 234 tasks + red 210 tasks (Phase 1a scope); shop deferred to Phase 1b.
  log "=== Gate 8: cross-fire quarantine registry investigation gate (Wave 2 M6) ==="
  if [ -f scripts/maintenance/quarantine_registry.py ]; then
    # Source the helper lib if not already loaded. Gates lib defines
    # assert_quarantine_gate which calls the registry CLI.
    if ! declare -f assert_quarantine_gate > /dev/null 2>&1; then
      # shellcheck disable=SC1091
      source scripts/queues/_lib_paper_grade_gates.sh
    fi
    # B-1939 (codex Mode B F5, 2026-08-03): the gate must examine the sites this
    # launch will actually touch. Pre-fix it hardcoded cls 0-233 + red 0-209
    # regardless of SITE_FILTER, so `launch wa_shop` passed a quarantine check on
    # two unrelated sites and printed "All gates passed" while any unresolved WA
    # shopping/admin quarantine event stayed invisible. A gate that always
    # inspects the same thing is not a gate for anything else.
    declare -a _g8_sites=()
    case "${SITE_FILTER}" in
      all|"")        _g8_sites=("classifieds:0-233" "reddit:0-209") ;;
      cls)           _g8_sites=("classifieds:0-233") ;;
      red)           _g8_sites=("reddit:0-209") ;;
      phase1b|shop_b0|shop_b0_tail)  _g8_sites=("shopping:0-465") ;;
      wa_shop|wa_shop_b0) _g8_sites=("wa_shopping:0-191") ;;
      wa_shop_admin) _g8_sites=("wa_shopping_admin:0-181") ;;
      *)
        log "  FAIL: Gate 8 has no quarantine policy for SITE_FILTER=${SITE_FILTER} — refusing to pass a gate that inspects nothing relevant"
        errors=$((errors+1)) ;;
    esac
    for _g8 in "${_g8_sites[@]}"; do
      _g8_site="${_g8%%:*}"; _g8_range="${_g8##*:}"
      g8_rc=0
      assert_quarantine_gate "${_g8_site}" "${_g8_range}" 1 || g8_rc=$?
      if [ "$g8_rc" -ne 0 ]; then
        log "  FAIL: ${_g8_site} quarantine registry gate HALT (rc=$g8_rc)"
        errors=$((errors+1))
      fi
    done
    if [ "$errors" -eq 0 ]; then
      log "  OK (no unclassified quarantine events for: ${_g8_sites[*]})"
    fi
  else
    # /stress 2026-05-20 Track A F3 P0-3-A: paper-grade hard rule "always clean"
    # contradicts "missing script = silent SKIP". Mirror Gate 4 (preflight_v2.sh
    # missing) fail-closed pattern. Operator workflow: `git pull` or restore
    # from backup before paper-grade fire.
    log "  FAIL: scripts/maintenance/quarantine_registry.py REQUIRED for Gate 8 — paper-grade fire must investigate cross-fire quarantine events; run 'git pull' or restore from backup before re-attempting"
    errors=$((errors+1))
  fi

  if [ "$errors" -gt 0 ]; then
    fail "$errors gate(s) failed; abort. Fix and re-run."
  fi
  log "All gates passed (or warnings only)."
}

# ---------------------------------------------------------------------------
# Chain definitions
# ---------------------------------------------------------------------------

# Map a chain command to its config file path. Mirrors the CFG_NAME convention
# in queue_baseline.sh / queue_phantom_*.sh (codex stress v6 C9).
#
# B-672 (/stress A1.14 Chunk a P0-1, codex Mode B F2 unique + Claude+gemini 2-AI
# brittle parse, 2026-05-17): full audit of this function:
#  1. Phantom-SoM config name typo — pre-fix built `exp_v2_<bl>_phantom_<site>.yaml`
#     (missing `_som_` infix); actual file is `..._phantom_som_<site>.yaml` per
#     `queue_phantom_som.sh:61-63` CFG_NAME. Gate 7 then FAILed all 6 P-SoM cells
#     (B0/B1/B2 × cls/red), launch-blocking Phase 1a (codex Mode B F2 OOB).
#  2. Unquoted `parts=( $cmd )` word-splitting — relied on chain command tokens
#     never containing whitespace/glob; future router-mode commands with paths
#     would silently break. Replaced with `read -r -a parts <<< "$cmd"`
#     (Claude F8 + Gemini F6 2-AI catch).
#  3. queue_phantom_dom.sh back-compat — symlink to queue_phantom_text.sh;
#     explicit case prevents silent fall-through to default empty echo if a
#     future chain definition mistakenly uses legacy mode value.
#  4. Default branch: fail loud (return non-empty error marker) so Gate 7
#     callers can detect "unknown script" vs "missing config" instead of the
#     pre-fix silent empty skip (which let any future leaf script bypass the
#     config-existence check).
config_for_cmd() {
  local cmd="$1"
  local -a parts
  # Safe word splitting (handles future quoted/whitespace arg values).
  read -r -a parts <<< "$cmd"
  local script="${parts[0]}"
  local baseline="${parts[1]}"
  # B-1935 (2026-08-03): benchmark-aware site token. The leaf scripts take
  # benchmark as a trailing optional arg (`... shopping wa`), and the config
  # naming convention puts it in FRONT of the site (`exp_v2_B0_dom_wa_shopping`),
  # so it cannot be appended positionally — pre-fix any WA chain step resolved to
  # the VWA config path and Gate 7 would have reported it missing.
  local site bench
  case "$script" in
    queue_baseline.sh)  site="${parts[3]}"; bench="${parts[4]:-vwa}" ;;
    *)                  site="${parts[2]}"; bench="${parts[3]:-vwa}" ;;
  esac
  [[ "${bench}" == "wa" ]] && site="wa_${site}"
  case "$script" in
    queue_baseline.sh)        # <baseline> <mode> <site> [benchmark]
      echo "configs/exp_v2_${baseline}_${parts[2]}_${site}.yaml" ;;
    queue_phantom_som.sh)     # <baseline> <site> [benchmark]
      echo "configs/exp_v2_${baseline}_phantom_som_${site}.yaml" ;;
    queue_phantom_text.sh|queue_phantom_dom.sh)  # <baseline> <site> [benchmark] (dom = back-compat symlink)
      echo "configs/exp_v2_${baseline}_phantom_text_${site}.yaml" ;;
    queue_phantom_prompt.sh)  # <baseline> <site> [benchmark]
      echo "configs/exp_v2_${baseline}_phantom_prompt_${site}.yaml" ;;
    *) echo "UNKNOWN_SCRIPT:${script}" ;;
  esac
}

# B-1825 (Fire-6 /stress P0-3-AC*): RESUME_MISSING done-detection. A condition is
# "complete" iff it has a MANIFEST-BOUND authoritative run
# (docs/checkpoints/pre_run/fire_manifest.json) whose condition_summary_v2.json
# has episodes >= the expected scored count. Manifest-bound (NOT glob-latest) so a
# re-fire's ghost run can never be mistaken for the authoritative one — this closes
# the `ls -dt` latest-run ambiguity in phase1a_status.sh:94. Not-in-manifest /
# summary-missing / under-count → return 1 → run fresh.
_condition_complete() {
  local cmd="$1"
  local -a p; read -r -a p <<< "$cmd"
  local script="${p[0]}" bl="${p[1]}" mode site bench
  case "$script" in
    queue_baseline.sh)                            mode="${p[2]}";        site="${p[3]}"; bench="${p[4]:-vwa}" ;;
    queue_phantom_som.sh)                         mode="phantom_som";    site="${p[2]}"; bench="${p[3]:-vwa}" ;;
    queue_phantom_text.sh|queue_phantom_dom.sh)   mode="phantom_text";   site="${p[2]}"; bench="${p[3]:-vwa}" ;;
    queue_phantom_prompt.sh)                      mode="phantom_prompt"; site="${p[2]}"; bench="${p[3]:-vwa}" ;;
    *) return 1 ;;
  esac
  # B-1935: qualify the manifest key by benchmark. Pre-fix a WA step looked up
  # the bare site, so `queue_baseline.sh B0 dom shopping wa` queried the VWA key
  # `shopping|B0|dom` — and under RESUME_MISSING=1 a COMPLETED VWA shopping
  # condition would mark the WA step done and silently drop it from the chain.
  # WA conditions are absent from the Phase 1a manifest by design, so the lookup
  # correctly misses and the step runs fresh.
  [[ "${bench}" == "wa" ]] && site="wa_${site}"
  REPO_DIR="${REPO_DIR}" python3 - "$site" "$bl" "$mode" "$bench" <<'PY'
import json, os, sys
site, bl, mode, bench = sys.argv[1:5]
repo = os.environ.get("REPO_DIR", ".")
try:
    d = json.load(open(os.path.join(repo, "docs/checkpoints/pre_run/fire_manifest.json")))
except Exception:
    sys.exit(1)
cond = d.get("conditions", {}).get(f"{site}|{bl}|{mode}")
if not cond:
    sys.exit(1)
scored = int(d.get("scored_task_count", {}).get(site, 10**9))
cond_id = cond.get("condition_id", f"phase1_{mode}_router_0")
# B-1938 (codex Mode B F3, 2026-08-03): results root must follow the benchmark.
# Pre-fix this was hardcoded to `results/visualwebarena/phase1`, but WA runs land
# in `results/webarena/phase1` (both dirs exist on disk). B-1935 exposed
# `_resume_filter_done` on the two new WA builders, so RESUME_MISSING advertised
# a resume that could never find WA data — interrupt at condition 17/18 and all
# 16 finished conditions would be re-run, each re-resetting the container.
# NOTE this alone does not make WA resume work: WA conditions are absent from the
# Phase 1a fire manifest by design, so the `conditions` lookup above still misses
# and every WA step runs fresh. That is correct-but-inert. Making WA resume
# actually functional needs its own manifest namespace (codex F3 defuse, 3-6h) —
# until then WA resume is a no-op, not a wrong answer.
_root = "results/webarena/phase1" if bench == "wa" else "results/visualwebarena/phase1"
summ = os.path.join(repo, _root, cond["run_id"], cond_id, "condition_summary_v2.json")
try:
    eps = int(json.load(open(summ)).get("episodes", 0))
except Exception:
    sys.exit(1)
sys.exit(0 if eps == scored else 1)  # B-1834: EXACT (>= would skip an over-complete/contaminated bound run forever)
PY
}

# B-1825: filter a build_*_chain heredoc — when RESUME_MISSING=1, drop conditions
# already complete (manifest-bound); passthrough otherwise. Logs each skip.
_resume_filter_done() {
  if [[ "${RESUME_MISSING:-0}" != "1" ]]; then cat; return 0; fi
  local cmd
  while IFS= read -r cmd; do
    [[ -z "${cmd// }" ]] && continue
    if _condition_complete "${cmd}"; then
      # B-1826: skip log MUST go to stderr — build_*_chain stdout is the chain-command
      # data channel that launch_chain collects; a log line on stdout becomes a bogus
      # chain command (Gate 7 UNKNOWN_SCRIPT abort, 2026-05-21 first relaunch).
      log "  [resume] SKIP done (manifest-bound): ${cmd}" >&2
    else
      echo "${cmd}"
    fi
  done
}

build_cls_chain() {
  # Phase 1a classifieds: 6 modes per model, B0 → B1 → B2 sequential = 18 conditions
  _resume_filter_done <<EOF  # B-1825: RESUME_MISSING=1 drops manifest-complete conditions
queue_baseline.sh B0 dom classifieds
queue_baseline.sh B0 som classifieds
queue_baseline.sh B0 vision classifieds
queue_phantom_text.sh B0 classifieds
queue_phantom_som.sh B0 classifieds
queue_phantom_prompt.sh B0 classifieds
queue_baseline.sh B1 dom classifieds
queue_baseline.sh B1 som classifieds
queue_baseline.sh B1 vision classifieds
queue_phantom_text.sh B1 classifieds
queue_phantom_som.sh B1 classifieds
queue_phantom_prompt.sh B1 classifieds
queue_baseline.sh B2 dom classifieds
queue_baseline.sh B2 som classifieds
queue_baseline.sh B2 vision classifieds
queue_phantom_text.sh B2 classifieds
queue_phantom_som.sh B2 classifieds
queue_phantom_prompt.sh B2 classifieds
EOF
}

build_red_chain() {
  # Phase 1a reddit: 6 modes per model, B0 → B1 → B2 sequential = 18 conditions
  _resume_filter_done <<EOF  # B-1825: RESUME_MISSING=1 drops manifest-complete conditions
queue_baseline.sh B0 dom reddit
queue_baseline.sh B0 som reddit
queue_baseline.sh B0 vision reddit
queue_phantom_text.sh B0 reddit
queue_phantom_som.sh B0 reddit
queue_phantom_prompt.sh B0 reddit
queue_baseline.sh B1 dom reddit
queue_baseline.sh B1 som reddit
queue_baseline.sh B1 vision reddit
queue_phantom_text.sh B1 reddit
queue_phantom_som.sh B1 reddit
queue_phantom_prompt.sh B1 reddit
queue_baseline.sh B2 dom reddit
queue_baseline.sh B2 som reddit
queue_baseline.sh B2 vision reddit
queue_phantom_text.sh B2 reddit
queue_phantom_som.sh B2 reddit
queue_phantom_prompt.sh B2 reddit
EOF
}

build_shop_chain() {
  # Phase 1b deferred: shop × 6 modes per model, B0 → B1 → B2 sequential = 18 conditions
  # B2 (Gemma3-VL) included for baseline consistency — a baseline applies to all
  # sites; Phase 1b's deferral is about launch TIMING, not model scope.
  # NOT launched as part of default `launch` (which is Phase 1a cls + red).
  # Launch via explicit `launch phase1b` after workshop submission.
  _resume_filter_done <<EOF  # B-1825: RESUME_MISSING=1 drops manifest-complete conditions
queue_baseline.sh B0 dom shopping
queue_baseline.sh B0 som shopping
queue_baseline.sh B0 vision shopping
queue_phantom_text.sh B0 shopping
queue_phantom_som.sh B0 shopping
queue_phantom_prompt.sh B0 shopping
queue_baseline.sh B1 dom shopping
queue_baseline.sh B1 som shopping
queue_baseline.sh B1 vision shopping
queue_phantom_text.sh B1 shopping
queue_phantom_som.sh B1 shopping
queue_phantom_prompt.sh B1 shopping
queue_baseline.sh B2 dom shopping
queue_baseline.sh B2 som shopping
queue_baseline.sh B2 vision shopping
queue_phantom_text.sh B2 shopping
queue_phantom_som.sh B2 shopping
queue_phantom_prompt.sh B2 shopping
EOF
}

build_shop_b0_chain() {
  # B0-only VWA shopping: 6 modes + 1 replicate arm = 7 conditions.
  #
  # WHY B0-ONLY (user decision 2026-08-03, option 1 of 3):
  # The frame needs the SITE axis extended (cls + red = 2 sites is too thin to
  # separate "site-specific" from noise). It does not need a third backbone on a
  # third site: the project's own wording discipline says three backbones sharing
  # one task set establish model robustness of the site interaction, NOT three
  # independent observations. Applying that rule consistently, the site axis +1
  # costs ONE backbone, not three.
  #
  # Measured (38 landed conditions on A100, 2026-08-03) — this is why:
  #   3 baselines × 2 shop sites = 36 cond = 46.8 GB / 54.3 days  ← 41 GB free
  #   B0 only    × 2 shop sites = 12 cond = 15.6 GB / 13.6 days   ← fits
  # Per-episode disk is 4.38 MB measured ON A100 (som 7.46). Do NOT re-measure
  # this on DGX: DGX lacks the synced artifacts and reports 132 KB, a 27× low
  # reading that is how the stale "12 cond ≈ 18.8G" estimate happened.
  #
  # REPLICATE ARM (last line, deliberate duplicate): re-runs B0 dom to give this
  # site a stochastic noise floor. Without it every shopping effect size has no
  # comparable noise band — and two ledger entries hang on exactly that:
  # §242 (drop-one oracle 1.7-3.3pp must be shown to clear the stochastic floor,
  # "重跑尚未做") and §293 (if H1 strict clears by only 1-2pp and the replicate
  # floor is also 1-2pp, the hero wording must be downgraded). Adding the floor
  # after the fact costs a whole second campaign; adding it here costs 1 condition.
  # `FORCE_NEW=1` (exported by launch_chain) mints a distinct run_id, so the
  # duplicate becomes its own run rather than resuming the first.
  _resume_filter_done <<EOF  # B-1825: RESUME_MISSING=1 drops manifest-complete conditions
queue_baseline.sh B0 dom shopping
queue_baseline.sh B0 som shopping
queue_baseline.sh B0 vision shopping
queue_phantom_text.sh B0 shopping
queue_phantom_som.sh B0 shopping
queue_phantom_prompt.sh B0 shopping
queue_baseline.sh B0 dom shopping
EOF
}

build_shop_b0_tail_chain() {
  # B-1957 follow-on (2026-08-05): cells 2-7 of build_shop_b0_chain — every mode
  # EXCEPT the dom main arm, which is resumed by hand under RESET_BEFORE=0 to
  # preserve B-304 trajectory continuity across the R3561 interrupt. These six
  # are all fresh cells, so launch_chain's hardcoded FORCE_NEW=1 RESET_BEFORE=1
  # is exactly right for them (and the replicate arm NEEDS FORCE_NEW=1 to get a
  # run_id distinct from the main arm's).
  #
  # Derived from the full builder rather than copied: one list, one place to
  # edit. A second literal copy is how a premise ends up with two versions that
  # drift (§428.8).
  #
  # RESUME_MISSING is forced off for the derivation so `tail -n +2` always drops
  # the dom main arm and not whatever line the filter happened to leave first.
  # Losing resume here costs nothing today: `fire_manifest.json` carries zero
  # shopping conditions, so RESUME_MISSING is inert for this site anyway.
  ( RESUME_MISSING=0; build_shop_b0_chain ) | tail -n +2
}

build_wa_shop_b0_chain() {
  # B0-only WA shopping: 6 modes + 1 replicate arm = 7 conditions, 173 scored
  # tasks each. Same rationale as build_shop_b0_chain; this is the second
  # independent evidence line the site-axis argument asks for.
  #
  # ⚠️ SAME Magento container as the VWA shop chain (7770/7780 both bind
  # vwa-shopping), so these two chains are SEQUENTIAL, never parallel — the
  # container lock (B-1934) enforces it and CLAUDE.md hard rule #3 requires it.
  # Budget the wallclock serially: ~11.3 days (VWA) + ~4.5 days (WA).
  _resume_filter_done <<EOF  # B-1825: RESUME_MISSING=1 drops manifest-complete conditions
queue_baseline.sh B0 dom shopping wa
queue_baseline.sh B0 som shopping wa
queue_baseline.sh B0 vision shopping wa
queue_phantom_text.sh B0 shopping wa
queue_phantom_som.sh B0 shopping wa
queue_phantom_prompt.sh B0 shopping wa
queue_baseline.sh B0 dom shopping wa
EOF
}

build_wa_shop_chain() {
  # WA shopping: 6 modes per model, B0 → B1 → B2 sequential = 18 conditions,
  # 173 scored tasks each (post-N/A-exclusion, B-1894).
  #
  # B-1935 (2026-08-03): launches under the SAME container key as the VWA shop
  # chain ("magento", see lib site_lock_key) — WA shopping is the `vwa-shopping`
  # container reached through WA's task file, not a second stack. So this chain
  # and build_shop_chain can never run concurrently; the container lock refuses
  # the second one. Run them one after the other, not in parallel, and expect
  # `launch wa_shop` to abort while a VWA shop chain is alive. That is the gate
  # working, not a misfire.
  _resume_filter_done <<EOF  # B-1825: RESUME_MISSING=1 drops manifest-complete conditions
queue_baseline.sh B0 dom shopping wa
queue_baseline.sh B0 som shopping wa
queue_baseline.sh B0 vision shopping wa
queue_phantom_text.sh B0 shopping wa
queue_phantom_som.sh B0 shopping wa
queue_phantom_prompt.sh B0 shopping wa
queue_baseline.sh B1 dom shopping wa
queue_baseline.sh B1 som shopping wa
queue_baseline.sh B1 vision shopping wa
queue_phantom_text.sh B1 shopping wa
queue_phantom_som.sh B1 shopping wa
queue_phantom_prompt.sh B1 shopping wa
queue_baseline.sh B2 dom shopping wa
queue_baseline.sh B2 som shopping wa
queue_baseline.sh B2 vision shopping wa
queue_phantom_text.sh B2 shopping wa
queue_phantom_som.sh B2 shopping wa
queue_phantom_prompt.sh B2 shopping wa
EOF
}

build_wa_shop_admin_chain() {
  # WA shopping_admin: 6 modes per model = 18 conditions, 176 scored tasks each
  # (B-1894). Same container as both shop chains above (7780 and 7770 are one
  # container), hence the same lock and the same one-at-a-time constraint.
  _resume_filter_done <<EOF  # B-1825: RESUME_MISSING=1 drops manifest-complete conditions
queue_baseline.sh B0 dom shopping_admin wa
queue_baseline.sh B0 som shopping_admin wa
queue_baseline.sh B0 vision shopping_admin wa
queue_phantom_text.sh B0 shopping_admin wa
queue_phantom_som.sh B0 shopping_admin wa
queue_phantom_prompt.sh B0 shopping_admin wa
queue_baseline.sh B1 dom shopping_admin wa
queue_baseline.sh B1 som shopping_admin wa
queue_baseline.sh B1 vision shopping_admin wa
queue_phantom_text.sh B1 shopping_admin wa
queue_phantom_som.sh B1 shopping_admin wa
queue_phantom_prompt.sh B1 shopping_admin wa
queue_baseline.sh B2 dom shopping_admin wa
queue_baseline.sh B2 som shopping_admin wa
queue_baseline.sh B2 vision shopping_admin wa
queue_phantom_text.sh B2 shopping_admin wa
queue_phantom_som.sh B2 shopping_admin wa
queue_phantom_prompt.sh B2 shopping_admin wa
EOF
}

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

dry_run() {
  log "DRY RUN — no launches will occur."
  log ""
  log "=== Phase 1a (default, workshop-target) ==="
  log ""
  log "Cls chain (18 conditions, 6 modes × B0+B1+B2):"
  build_cls_chain | sed 's/^/  /'
  log ""
  log "Red chain (18 conditions, 6 modes × B0+B1+B2):"
  build_red_chain | sed 's/^/  /'
  log ""
  log "Phase 1a total: 36 operational conditions across 6 statistical cells (= (site, model) tuples)."
  log ""
  log "=== Phase 1b (deferred, main paper expansion) ==="
  log ""
  log "Shop chain (18 conditions, 6 modes × B0+B1+B2):"
  build_shop_chain | sed 's/^/  /'
  log ""
  log "Phase 1b total: 18 conditions (launch separately via 'launch phase1b shop' post-workshop)."
  log ""
  log "=== ⭐ B0-only shopping (B-1950, user 决定 2026-08-03 — 这是当前要跑的) ==="
  log ""
  log "VWA shop B0-only (7 conditions = 6 modes + 1 replicate, 435 scored tasks each):"
  build_shop_b0_chain | sed 's/^/  /'
  log ""
  log "WA shop B0-only (7 conditions = 6 modes + 1 replicate, 173 scored tasks each):"
  build_wa_shop_b0_chain | sed 's/^/  /'
  log ""
  log "  实测成本 (38 个已落地 condition, A100 量): 4.38 MB/ep · B0 322 s/task"
  log "    VWA shop 7 cond ≈ 13.1 GB / 11.3 天    WA shop 7 cond ≈ 5.2 GB / 4.5 天"
  log "    合计 ≈ 18.3 GB / 15.8 天 (串行, 同一 Magento 容器)  vs A100 可用 41 GB"
  log "  对照 — 全 3-baseline 版本 (build_shop_chain + build_wa_shop_chain):"
  log "    36 cond ≈ 46.8 GB / 54.3 天 ⇒ 装不下 41 GB, 故未选"
  log ""
  log "=== WA shopping 全 3-baseline (B-1935; 未选, 保留) ==="
  log ""
  log "WA shop chain (18 conditions, 6 modes × B0+B1+B2, 173 scored tasks each):"
  build_wa_shop_chain | sed 's/^/  /'
  log ""
  log "WA shop_admin chain (18 conditions, 6 modes × B0+B1+B2, 176 scored tasks each):"
  build_wa_shop_admin_chain | sed 's/^/  /'
  log ""
  log "⚠️  All three shop chains (phase1b / wa_shop / wa_shop_admin) mutate ONE"
  log "    Magento container (vwa-shopping, ports 7770+7780). They hold the same"
  log "    container lock and must run sequentially; launching a second one while"
  log "    another is alive aborts by design (B-1934)."
  log ""
  log "Run with 'launch' for Phase 1a default, 'launch phase1b' for VWA shop,"
  log "or 'launch wa_shop' / 'launch wa_shop_admin' for the WA expansions."
}

launch_chain() {
  local label=$1
  local builder=$2
  # B-705 (/stress A1.14 Chunk d P2-2 Claude+codex 2-AI AB, 2026-05-17): log
  # + pid file names include timestamp + PID for collision-free re-fire. Pre-fix
  # `logs/queue_phase1_${label}.log` static name → re-fire overwrote previous
  # chain's transcript (forensic loss for multi-day paper-grade runs). New form
  # writes timestamped log + `latest` symlink for `tail -f` ergonomics.
  local ts="$(date +%Y%m%d_%H%M%S)"
  local logfile="logs/queue_phase1_${label}_${ts}_$$.log"
  local pidfile="logs/queue_phase1_${label}_${ts}_$$.pid"
  # P0-2-B* (/stress Phase 0 unified bug list 2026-05-19, codex unique OOB
  # SMOKING GUN): done sentinel writes `rc=N ts=...` on inner queue_chain.sh
  # exit. Master orchestrator reads sentinel rc after PID dies to decide
  # cls→red cascade safety. Pre-fix `kill -0` poll detected "alive vs dead"
  # but not exit code → cls chain B-486 quarantine exit 1 was silently
  # interpreted as "done" → red auto-started (Fire-3 2026-05-19 00:53→00:54
  # empirical signature).
  local donefile="logs/queue_phase1_${label}_${ts}_$$.done"
  local logsym="logs/queue_phase1_${label}.latest.log"
  local pidsym="logs/queue_phase1_${label}.latest.pid"
  local donesym="logs/queue_phase1_${label}.latest.done"
  mkdir -p logs

  # Convert chain commands to space-quoted args
  local args=()
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    args+=("$line")
  done < <($builder)

  log "Launching $label chain (${#args[@]} cells) → $logfile"
  # FORCE_NEW=1: paper-grade fresh rerun — each cell gets a timestamped run_id,
  # never resumes a pre-fix archived dir (codex stress v6 C1, 2026-05-14).
  # RESET_BEFORE=1: each condition resets site state for fair ablation.
  # P0-2-B* sentinel wrapper: subshell captures inner rc → writes donefile
  # before exit. `--` separates bash -c invocation flags from positional args.
  FORCE_NEW=1 RESET_BEFORE=1 nohup bash -c "
    bash scripts/queues/queue_chain.sh \"\$@\"
    _rc=\$?
    printf 'rc=%d ts=%s label=%s pid=%d\n' \"\$_rc\" \"\$(date -u +%FT%TZ)\" '$label' \"\$\$\" > '$donefile'
    exit \$_rc
  " _ "${args[@]}" > "$logfile" 2>&1 {ORCH_FD}>&- &  # B-1824 (Fire-6 /stress P1-2): close orchestrator lock fd → chain subtree (chain/leaf/daemons) never inherits it
  local pid=$!
  log "  PID $pid, log $logfile"
  echo "$pid" > "$pidfile"
  # Refresh `.latest.*` symlinks (force, atomic). Donesym target may not exist
  # yet (subshell still running); symlink resolves once sentinel lands on exit.
  ln -sfn "$(basename "$logfile")" "$logsym" 2>/dev/null || true
  ln -sfn "$(basename "$pidfile")" "$pidsym" 2>/dev/null || true
  ln -sfn "$(basename "$donefile")" "$donesym" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# B-1683 (/stress A2.11 P1-3-B* user Q4=C+B 2026-05-18): Phase 1a 42/42
# manifest gate. Pass-1 (baseline) = 36; Pass-2 (learned router) = 6;
# Phase 1a closed = 42/42. Pre-fix this orchestrator's `launch` fired 36
# only + log msg "Phase 1a rerun launched" → operator easily assumes
# Phase 1a complete at 36. Now: post-launch manifest check + new `status`
# mode for at-a-glance count + `launch-pass2` stub pointing to separate
# router orchestrator (LR training pipeline gates Pass-2; defer 1-script
# command split until LR pipeline lands).
check_phase1a_manifest_42() {
  local manifest="${REPO_DIR}/results/phantom_paper/run_manifest.yaml"
  if [[ ! -f "${manifest}" ]]; then
    log "  [warn] manifest not found: ${manifest}"
    return
  fi
  local counts pass1 pass2
  counts=$("${REPO_DIR}/.venv/bin/python3" - <<PY 2>/dev/null
import sys, yaml
try:
    with open("${manifest}") as f:
        m = yaml.safe_load(f) or {}
except Exception:
    print("0 0"); sys.exit(0)
cells = m.get("cells") or []
modes_pass1 = {"DOM","SoM","Vision","P-text","P-prompt","P-SoM"}
pass1 = sum(1 for c in cells if c.get("grade") == "paper-grade" and c.get("mode") in modes_pass1)
pass2 = sum(1 for c in cells if c.get("grade") == "paper-grade" and c.get("mode") == "learned")
print(f"{pass1} {pass2}")
PY
)
  pass1="${counts%% *}"
  pass2="${counts##* }"
  pass1="${pass1:-0}"
  pass2="${pass2:-0}"
  local total=$((pass1 + pass2))
  log "==================================================================="
  log "  Phase 1a Manifest Gate (B-1683 user Q4=C 2026-05-18):"
  log "  Pass-1 (baseline modes): ${pass1}/36 paper-grade cells"
  log "  Pass-2 (learned router): ${pass2}/6 paper-grade cells"
  log "  TOTAL:                   ${total}/42 (Phase 1a complete = 42/42)"
  if (( total < 42 )); then
    log "  STATUS: INCOMPLETE — Phase 1a NOT closed"
    if (( pass1 >= 36 && pass2 < 6 )); then
      log "  → Pass-1 done but Pass-2 NOT FIRED. Next:"
      log "    bash scripts/queues/queue_phase1_router_paper_grade.sh"
      log "    (DO NOT declare Phase 1a complete until 42/42)"
    elif (( pass1 < 36 )); then
      log "  → Pass-1 ${pass1}/36 — operator must complete Pass-1 conditions"
      log "    (run: bash scripts/queues/queue_phase1_paper_grade.sh launch)"
    fi
  else
    log "  STATUS: COMPLETE — Phase 1a 42/42 cells closed ✓"
  fi
  log "==================================================================="
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

case "$MODE" in
  dry-run)
    dry_run
    ;;
  status)
    # B-1683 Q4-C: 42/42 manifest gate diagnostic. Use this to verify
    # Phase 1a completeness state before declaring done.
    check_phase1a_manifest_42
    ;;
  launch-pass2)
    # B-1683 Q4-B 2026-05-18: Pass-2 router fire stub. Full impl in
    # separate orchestrator `queue_phase1_router_paper_grade.sh` (depends
    # on LR training pipeline running post-Pass-1; see paper §6 H10
    # operational-gate framework + docs/checkpoints/phase1_plan.md §C).
    log "Pass-2 (learned router) fire — delegated to separate orchestrator:"
    log "  bash ${SCRIPT_DIR}/queue_phase1_router_paper_grade.sh"
    log ""
    log "Reason: Pass-2 requires LR training pipeline (post-Pass-1 outcomes →"
    log "  oracle label matrix → entropy defer gate → fold assignments →"
    log "  per-cell LR heads + artifact smoke test). Bundling Pass-1+Pass-2"
    log "  in one orchestrator would auto-handoff before LR training completes."
    if [[ ! -f "${SCRIPT_DIR}/queue_phase1_router_paper_grade.sh" ]]; then
      log ""
      log "  [warn] queue_phase1_router_paper_grade.sh not present yet —"
      log "  Pass-2 orchestrator pending LR training pipeline land."
    fi
    fail "Pass-2 fire delegated; this orchestrator is Pass-1 only."
    ;;
  launch|launch-pass1)
    # B-674 (/stress A1.14 Chunk a P0-3, gemini Mode C F1 unique OOB, 2026-05-17):
    # TOCTOU defense around gate-check + launch — pre-fix two concurrent
    # orchestrator invocations could both see Gate 6 active=0 and both fire
    # cls+red chains, bypassing queue_chain.sh per-(site,benchmark) flock
    # which only fires AFTER chain start. flock at orchestrator level closes
    # the race window between Gate 6 read and chain spawn.
    ORCH_LOCK_DIR="${REPO_DIR}/.locks"
    mkdir -p "${ORCH_LOCK_DIR}"
    ORCH_LOCK="${ORCH_LOCK_DIR}/phase1_orchestrator.lock"
    exec {ORCH_FD}>"${ORCH_LOCK}"
    if ! flock -n -x "${ORCH_FD}"; then
      stale_pid="$(cat "${ORCH_LOCK}" 2>/dev/null || echo unknown)"
      fail "Another paper-grade orchestrator instance holds ${ORCH_LOCK} (pid ${stale_pid}). Wait for it to complete or kill stale lock holder."
    fi
    echo "$$" > "${ORCH_LOCK}"
    # Lock auto-releases on shell exit (FD closes); trap rm for clean cleanup.
    trap "rm -f '${ORCH_LOCK}' 2>/dev/null; exec {ORCH_FD}>&-; exit" EXIT INT TERM
    log "orchestrator lock acquired: ${ORCH_LOCK} (pid $$)"

    # B-673 (A1.14 P0-2): host + URL locality enforcement BEFORE gates run.
    # Pre-fix: DGX dev session could pass all gates; paper hero numbers risked
    # producing from wrong substrate.
    require_paper_grade_host
    check_gates
    case "$SITE_FILTER" in
      all)
        # Default = Phase 1a (cls + red only). Phase 1b shop requires explicit launch.
        # B-1663 (/stress A2.11 P0-5-A*C 2026-05-18, user Q3=A): paper-grade fire
        # SEQUENTIAL by default — cls + red 同时 fire 共享 A100 docker bridge +
        # Postgres/Redis underlay + B0 AWS proxy quota. Empirical 2026-05-18
        # 13:28:06 fire saw red 99s busy-wait + B-1581 asyncio race (cross-site
        # contention suspected root cause). Sequential ~2× wallclock but cross-cell
        # latency canonical clean. PHASE1A_PARALLEL=1 opt-in dev mode (NOT paper-
        # grade per CLAUDE.md hard rule #3).
        # B-1825 (Fire-6 /stress P0-3-AC*): resume is sequential-only — parallel
        # fresh/resume chains violate the single-site hard rule (the very defect
        # that made phase1a_relaunch_missing.sh unsafe).
        if [[ "${RESUME_MISSING:-0}" == "1" && "${PHASE1A_PARALLEL:-0}" == "1" ]]; then
          fail "RESUME_MISSING=1 + PHASE1A_PARALLEL=1 incompatible (B-1825): resume is sequential-only — no parallel chains. Unset PHASE1A_PARALLEL."
        fi
        if [[ "${PHASE1A_PARALLEL:-0}" != "1" ]]; then
          if [[ "${RESUME_MISSING:-0}" == "1" ]]; then
            log "RESUME_MISSING=1 (B-1825): only conditions WITHOUT a valid manifest-bound"
            log "  run (docs/checkpoints/pre_run/fire_manifest.json) will fire — same preflight/"
            log "  Gate8/quarantine gates, sequential cls→red. Completed conditions (e.g. R9755"
            log "  B0 dom cls) are SKIPPED, not re-run."
          fi
          log "Sequential paper-grade fire (B-1663): cls → red after cls completion."
          launch_chain "cls" build_cls_chain
          _cls_pid_file="logs/queue_phase1_cls.latest.pid"
          if [[ ! -f "$_cls_pid_file" ]]; then
            fail "cls pid file missing: $_cls_pid_file — launch_chain failed?"
          fi
          _cls_pid=$(cat "$_cls_pid_file")
          # P0-2-B* (/stress Phase 0 unified bug list 2026-05-19, codex unique
          # OOB SMOKING GUN): pre-fix `kill -0` poll detected "process alive vs
          # dead" but NOT exit code. cls chain exit 1 (B-486 quarantine /
          # B-1665 wallclock / fatal preflight) was silently interpreted as
          # "done" → red auto-started. Fire-3 2026-05-19 00:53→00:54 cascade
          # was this exact pattern. Post-fix: poll-loop preserved as runaway
          # watchdog (24h max), then read done sentinel for actual rc; non-zero
          # → fail (NOT launch red).
          # 2026-06-03 (user decision; same rationale as queue_chain.sh B1/B2 wallclock):
          # B1/B2 cls chain = 12 conditions × ~10-20h each = 120-240h, far beyond the
          # old 24h runaway-watchdog cap (P0-4 sized it for a single ~20h chain). With
          # MAX_CLS_WAIT_HOURS=0 (default) the orchestrator waits until the cls chain
          # exits on its own — real chain hangs are caught by the condition-level
          # watchdog idle-alert + liveness check inside queue_chain.sh, NOT by this
          # poll cap. Set MAX_CLS_WAIT_HOURS=N>0 to restore a hard cap.
          _max_cls_wait_secs=$(( ${MAX_CLS_WAIT_HOURS:-0} * 3600 ))
          log "Waiting for cls chain pid=${_cls_pid} to complete (max ${MAX_CLS_WAIT_HOURS:-0}h; 0=unlimited)..."
          _wait_elapsed=0
          while kill -0 "$_cls_pid" 2>/dev/null; do
            if (( _max_cls_wait_secs > 0 && _wait_elapsed >= _max_cls_wait_secs )); then
              fail "cls chain pid=${_cls_pid} alive after ${MAX_CLS_WAIT_HOURS}h max-wait — investigate manually"
            fi
            sleep 60
            _wait_elapsed=$((_wait_elapsed + 60))
            if (( _wait_elapsed % 1800 == 0 )); then
              log "  cls chain pid=${_cls_pid} still running (${_wait_elapsed}s elapsed)"
            fi
          done
          # P0-2-B* sentinel read: launch_chain wraps queue_chain.sh in a
          # subshell that writes `rc=N ts=...` to `.latest.done` BEFORE exit.
          # Brief sleep tolerates subshell-exit/sentinel-write race window.
          sleep 2
          _cls_done_sentinel="logs/queue_phase1_cls.latest.done"
          _cls_rc=1
          if [[ -f "$_cls_done_sentinel" ]]; then
            _cls_rc=$(grep -oE 'rc=-?[0-9]+' "$_cls_done_sentinel" | head -1 | cut -d= -f2)
            _cls_rc="${_cls_rc:-1}"
            log "  cls chain done sentinel: $(cat "$_cls_done_sentinel" 2>/dev/null | head -1)"
          else
            log "  cls chain done sentinel ABSENT at $_cls_done_sentinel — treating as rc=1 (chain crashed/killed without exit-handler running)"
          fi
          if (( _cls_rc != 0 )); then
            fail "cls chain pid=${_cls_pid} exited rc=${_cls_rc} — paper-grade cascade halt (P0-2-B*): NOT launching red. Investigate cls failure; manually re-fire after fix. Fire-3 2026-05-19 00:53/00:54 cascade was this exact pattern."
          fi
          log "cls chain pid=${_cls_pid} done rc=${_cls_rc}; launching red chain"
          launch_chain "red" build_red_chain
        else
          log "PHASE1A_PARALLEL=1 set — DEV MODE parallel fire (NOT paper-grade per CLAUDE.md hard rule #3)"
          launch_chain "cls" build_cls_chain
          launch_chain "red" build_red_chain
        fi
        ;;
      cls)
        # R2-P0-1-B* (/stress Phase 0 post-fix Mode B codex F1 OOB, 2026-05-19):
        # single-site launch refuses if another site chain is alive — closes
        # the cross-site contention class that the "launch all" sentinel-wait
        # (P0-2-B*) closed for sequential default. PHASE1A_PARALLEL=1 dev
        # opt-in preserved.
        assert_no_other_site_chain_running "cls" "queue_phase1"
        launch_chain "cls" build_cls_chain
        ;;
      red)
        assert_no_other_site_chain_running "red" "queue_phase1"
        launch_chain "red" build_red_chain
        ;;
      shop)
        log "WARN: 'launch shop' requested directly. shop is Phase 1b (main-paper expansion)."
        log "      Default Phase 1a does NOT include shop. Proceeding only if you confirm."
        log "      Use 'launch phase1b' to launch shop explicitly as Phase 1b."
        fail "Use 'launch phase1b' for shop chain (Phase 1b main-paper expansion)."
        ;;
      phase1b)
        log "=== Phase 1b launch (main-paper shop expansion) ==="
        assert_no_other_site_chain_running "shop" "queue_phase1"
        launch_chain "shop" build_shop_chain
        ;;
      shop_b0)
        # B-1950 (user decision 2026-08-03): B0-only VWA shopping, 6 modes + 1
        # replicate = 7 conditions. Declares self_site "shop" — same Magento
        # container as every other shop chain, so the host-chain check and the
        # container lock both treat them as one site.
        log "=== VWA shopping B0-only (7 conditions: 6 modes + 1 replicate; ~11.3 天, ~13.1 GB) ==="
        assert_no_other_site_chain_running "shop" "queue_phase1"
        launch_chain "shop" build_shop_b0_chain
        ;;
      shop_b0_tail)
        # B-1957 follow-on: the dom main arm is resumed separately (RESET_BEFORE=0,
        # B-304). This launches only what remains, so the six fresh cells still get
        # their per-condition reset while the interrupted arm keeps its trajectory.
        log "=== VWA shopping B0 tail (6 conditions: 5 modes + 1 replicate; dom main arm resumed separately) ==="
        assert_no_other_site_chain_running "shop" "queue_phase1"
        launch_chain "shop" build_shop_b0_tail_chain
        ;;
      wa_shop_b0)
        log "=== WA shopping B0-only (7 conditions: 6 modes + 1 replicate; ~4.5 天, ~5.2 GB) ==="
        assert_no_other_site_chain_running "shop" "queue_phase1"
        launch_chain "shop" build_wa_shop_b0_chain
        ;;
      wa_shop)
        # B-1935: WA shopping rides the same Magento container as VWA shopping,
        # so it declares self_site "shop" — the host-chain check must treat the
        # two as one site, and the container lock will refuse a second chain on
        # it regardless.
        log "=== WA shopping launch (18 conditions, 173 scored tasks/condition) ==="
        assert_no_other_site_chain_running "shop" "queue_phase1"
        launch_chain "shop" build_wa_shop_chain
        ;;
      wa_shop_admin)
        log "=== WA shopping_admin launch (18 conditions, 176 scored tasks/condition) ==="
        assert_no_other_site_chain_running "shop" "queue_phase1"
        launch_chain "shop" build_wa_shop_admin_chain
        ;;
      *) fail "Unknown site filter: $SITE_FILTER (expected: all|cls|red|phase1b|shop_b0|shop_b0_tail|wa_shop|wa_shop_b0|wa_shop_admin)" ;;
    esac
    # B-705 (A1.14 Chunk d P2-2): summary message now reflects actual SITE_FILTER
    # mode. Pre-fix line always said "Phase 1a rerun (36 conditions...)" even when
    # SITE_FILTER=phase1b (shop chain only) was active — misleading audit trail.
    log ""
    # B-1683 user Q4=C 2026-05-18: launch messages now name Pass-1 explicitly +
    # remind operator Phase 1a needs Pass-2 router fire too (42 total cells).
    case "$SITE_FILTER" in
      all)     log "Phase 1a Pass-1 launched (36/42 conditions, cls + red × B0+B1+B2 × 6 modes). Pass-2 router (6 cells) requires SEPARATE fire via queue_phase1_router_paper_grade.sh after Pass-1 done. Monitor:" ;;
      cls)     log "Phase 1a Pass-1 cls-only chain launched (18 conditions, B0+B1+B2 × 6 modes). Monitor:" ;;
      red)     log "Phase 1a Pass-1 red-only chain launched (18 conditions, B0+B1+B2 × 6 modes). Monitor:" ;;
      phase1b) log "Phase 1b shop chain launched (18 conditions, B0+B1+B2 × 6 modes; main-paper expansion). Monitor:" ;;
      shop_b0) log "VWA shopping B0-only launched (7 conditions = 6 modes + 1 replicate arm, 435 scored tasks each). Same Magento container as every shop chain — run wa_shop_b0 AFTER this finishes, not alongside. Monitor:" ;;
      shop_b0_tail) log "VWA shopping B0 tail launched (6 conditions = 5 modes + 1 replicate arm; the dom main arm was resumed separately under RESET_BEFORE=0). Monitor:" ;;
      wa_shop_b0) log "WA shopping B0-only launched (7 conditions = 6 modes + 1 replicate arm, 173 scored tasks each). Monitor:" ;;
      wa_shop) log "WA shopping chain launched (18 conditions, B0+B1+B2 × 6 modes, 173 scored tasks each). Shares the Magento container with VWA shop — run those two sequentially. Monitor:" ;;
      wa_shop_admin) log "WA shopping_admin chain launched (18 conditions, B0+B1+B2 × 6 modes, 176 scored tasks each). Same Magento container as both shop chains. Monitor:" ;;
      *)       log "Launch completed for SITE_FILTER=${SITE_FILTER}. Monitor:" ;;
    esac
    log ""
    # B-1683 Q4-C: post-launch manifest gate check — surfaces Phase 1a status
    # so operator sees "this fire is Pass-1 only; Pass-2 needs separate launch"
    # before terminal scroll. Re-run via `bash queue_phase1_paper_grade.sh status`
    # at any time to recheck (cells flip paper-grade as runs complete + manifest
    # updates land).
    check_phase1a_manifest_42
    log "  - PIDs: cat logs/queue_phase1_*.latest.pid"
    log "  - Logs: tail -f logs/queue_phase1_*.latest.log"
    log "  - Cells: open Obsidian Bases view 'cells.base' (cron 10min refresh)"
    log "  - Active: make active"
    log ""
    log "Post-completion analysis:"
    log "  make analysis              # full pipeline"
    log "  # Step 1: produce per_task_sr.csv (B-122 producer, 2026-05-15):"
    log "  python3 scripts/analysis/generate_per_task_sr.py \\"
    log "      --run-manifest results/phantom_paper/run_manifest.yaml \\"
    log "      --out results/phantom_paper/per_task_sr.csv"
    log "  # B-1052 (/stress A2.3c Mode B P0-2-B*, 2026-05-18): post-completion"
    log "  # handoff updated. Pre-fix invoked preregistration_decision_test.py"
    log "  # (synthetic test fixture, B-518) with retired --TOST-delta-pp +"
    log "  # --transparency-K_h1=4/K_h3=4 K-of-N gate flags. Canonical paper-grade"
    log "  # producer is aggregate_phase1_full_prereg_decision.py (A1.21 B-515)"
    log "  # invoked via 'make phase1-full-prereg-decision'. The retired flags"
    log "  # are deprecated post-B-957 TOST retire + Decision 3A K-of-N"
    log "  # transparency-only reclassification."
    log "  # Step 2: canonical paper-grade analysis pipeline (B-1052):"
    log "  make analysis"
    log "  # Step 3: canonical paper-grade gate artifact (B-515 A1.21 canonical):"
    log "  make phase1-full-prereg-decision"
    ;;
  *)
    fail "Unknown mode: $MODE (expected: dry-run | launch)"
    ;;
esac
