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
#   bash scripts/queues/queue_phase1_paper_grade.sh launch phase1b     # Phase 1b shop chain (18 conditions, deferred to post-workshop)
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

log() { echo "[phase1 $(date '+%H:%M:%S')] $*"; }
fail() { log "FAIL: $*"; exit 1; }

# B-673: require paper-grade host (A100) before gates run.
# Override knob: P79_PAPER_GRADE_HOST=1 for CI / future approved hostnames.
# Locality check uses _lib_paper_grade_gates.sh:assert_a100_url_locality
# already-loaded above.
require_paper_grade_host() {
  local hn
  hn="$(hostname 2>/dev/null || true)"
  if [[ "${P79_PAPER_GRADE_HOST:-0}" == "1" ]]; then
    log "  paper-grade host: ${hn} (P79_PAPER_GRADE_HOST=1 override)"
  elif [[ "${hn}" =~ (condense|a100|ubuntu) ]]; then
    log "  paper-grade host: ${hn} (matched condense|a100|ubuntu)"
  else
    fail "Refusing paper-grade launch on non-A100 host '${hn}'.
       Phase 1a paper-grade target = Condenser A100 (memory project_paper_grade_target_host).
       Re-run on a100-jiaming-test (ssh condense-a100), OR set P79_PAPER_GRADE_HOST=1 explicitly."
  fi
  # URL locality — paper-grade target = A100 self-hosted docker, all VWA URLs
  # must point to localhost (no DGX→quark Tailscale substrate substitution).
  # Reuses lib helper which fail-loud on any non-local URL.
  assert_a100_url_locality
}

# ---------------------------------------------------------------------------
# Pre-launch gates
# ---------------------------------------------------------------------------

check_gates() {
  local errors=0

  log "=== Gate 1: preregistration.md threshold lock ==="
  if grep -q "K_h1.*TBD\|K_h3.*TBD\|TOST.*TBD" docs/checkpoints/pre_run/preregistration.md 2>/dev/null; then
    log "  FAIL: preregistration.md still has TBD threshold values."
    log "        Wait for advisor email reply, then update preregistration.md."
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
    preflight_out=$(STRICT_PORTS=1 bash scripts/preflight_v2.sh --strict-ports 2>&1) || preflight_rc=$?
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
  active=$(pgrep -f "run_experiment.*--config" 2>/dev/null | wc -l || true)
  log "  Active run_experiment processes: $active"
  if [ "$active" -gt 0 ]; then
    pgrep -af "run_experiment.*--config" 2>/dev/null | sed 's/^/    /' || true
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
    all|cls|red) builders_to_check="build_cls_chain build_red_chain" ;;
    phase1b)     builders_to_check="build_shop_chain" ;;
    *)           builders_to_check="build_cls_chain build_red_chain" ;;
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
  case "$script" in
    queue_baseline.sh)        # <baseline> <mode> <site>
      echo "configs/exp_v2_${baseline}_${parts[2]}_${parts[3]}.yaml" ;;
    queue_phantom_som.sh)     # <baseline> <site>
      echo "configs/exp_v2_${baseline}_phantom_som_${parts[2]}.yaml" ;;
    queue_phantom_text.sh|queue_phantom_dom.sh)  # <baseline> <site> (dom = back-compat symlink)
      echo "configs/exp_v2_${baseline}_phantom_text_${parts[2]}.yaml" ;;
    queue_phantom_prompt.sh)  # <baseline> <site>
      echo "configs/exp_v2_${baseline}_phantom_prompt_${parts[2]}.yaml" ;;
    *) echo "UNKNOWN_SCRIPT:${script}" ;;
  esac
}

build_cls_chain() {
  # Phase 1a classifieds: 6 modes per model, B0 → B1 → B2 sequential = 18 conditions
  cat <<EOF
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
  cat <<EOF
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
  cat <<EOF
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
  log "Run with 'launch' for Phase 1a default, or 'launch phase1b shop' for shop expansion."
}

launch_chain() {
  local label=$1
  local builder=$2
  local logfile="logs/queue_phase1_${label}.log"
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
  FORCE_NEW=1 RESET_BEFORE=1 nohup bash scripts/queues/queue_chain.sh "${args[@]}" \
    > "$logfile" 2>&1 &
  local pid=$!
  log "  PID $pid, log $logfile"
  echo "$pid" > "logs/queue_phase1_${label}.pid"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

case "$MODE" in
  dry-run)
    dry_run
    ;;
  launch)
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
        launch_chain "cls" build_cls_chain
        launch_chain "red" build_red_chain
        ;;
      cls)  launch_chain "cls" build_cls_chain ;;
      red)  launch_chain "red" build_red_chain ;;
      shop)
        log "WARN: 'launch shop' requested directly. shop is Phase 1b (main-paper expansion)."
        log "      Default Phase 1a does NOT include shop. Proceeding only if you confirm."
        log "      Use 'launch phase1b' to launch shop explicitly as Phase 1b."
        fail "Use 'launch phase1b' for shop chain (Phase 1b main-paper expansion)."
        ;;
      phase1b)
        log "=== Phase 1b launch (main-paper shop expansion) ==="
        launch_chain "shop" build_shop_chain
        ;;
      *) fail "Unknown site filter: $SITE_FILTER (expected: all|cls|red|phase1b)" ;;
    esac
    log ""
    log "Phase 1a rerun launched (36 conditions, cls + red × B0+B1+B2 × 6 modes). Monitor:"
    log "  - PIDs: cat logs/queue_phase1_*.pid"
    log "  - Logs: tail -f logs/queue_phase1_*.log"
    log "  - Cells: open Obsidian Bases view 'cells.base' (cron 10min refresh)"
    log "  - Active: make active"
    log ""
    log "Post-completion analysis:"
    log "  make analysis              # full pipeline"
    log "  # Step 1: produce per_task_sr.csv (B-122 producer, 2026-05-15):"
    log "  python3 scripts/analysis/generate_per_task_sr.py \\"
    log "      --run-manifest results/phantom_paper/run_manifest.yaml \\"
    log "      --out results/phantom_paper/per_task_sr.csv"
    log "  # Step 2: decision test (B-120 6-cell K=6, 2026-05-15):"
    log "  python3 scripts/analysis/preregistration_decision_test.py \\"
    log "      --per-task-csv results/phantom_paper/per_task_sr.csv \\"
    log "      --primary-gate drop_one_pooled_meta_superiority \\"
    log "      --TOST-delta-pp 1.0 --H1-magnitude-pp 1.0 \\"
    log "      --transparency-K_h1 4 --transparency-K_h3 4 \\"
    log "      --out results/phantom_paper/preregistration_test_results.json"
    ;;
  *)
    fail "Unknown mode: $MODE (expected: dry-run | launch)"
    ;;
esac
