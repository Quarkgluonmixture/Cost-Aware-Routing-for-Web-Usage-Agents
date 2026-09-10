#!/usr/bin/env bash
# One live lane of the showcase demo: the ordinary p79 runner on a generated
# one-task config. Called by server.py, one process per lane.
#
#   run_lane.sh <config> <run_id> <max_steps>    run one lane
#   run_lane.sh login <auth_dir>                 log in to the demo site, write
#                                                <auth_dir>/classifieds_state.json
#
# This is NOT a paper-grade launch and must never look like one:
#   * output_root is set in the generated config to demo/live/runs/, never results/
#   * P79_PAPER_GRADE=0, so no fire gate, witness or manifest is consulted
#   * the site is quark's docker (the venue laptop), not the A100 — the A100 keeps
#     its one-site-chain rule untouched
#   * login state goes to demo/live/runs/auth/, not the repo's .auth/ — the runner
#     reuses whatever state file the task names on a lane's first episode, and the
#     repo's copy belongs to other work
# The queue scripts exist to make paper-grade launches safe (reset, watchdog,
# idempotent skip); none of that applies to an unscored demo run, which is why this
# wrapper exists instead of calling queue_baseline.sh.
#
# Env is set up the way the queue library does it (scripts/queues/_lib_paper_grade_gates.sh):
# the per-host VWA endpoint file (site credentials included), then the B0 key from
# .auth/qwen_api. Both are read by this shell only; nothing here prints them.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO"

export VWA_REMOTE_HOST="${VWA_REMOTE_HOST:-100.95.81.103}"
if [[ -f scripts/vwa_env_remote.sh ]]; then
  # shellcheck disable=SC1091
  source scripts/vwa_env_remote.sh
fi
# The demo's site is its own stack on this machine (live/site-compose.yml, 127.0.0.1:9981),
# whatever the endpoint file says. Backup: LIVE_CLASSIFIEDS=http://100.95.81.103:9980
# points the lanes at quark's docker instead.
export CLASSIFIEDS="${LIVE_CLASSIFIEDS:-http://localhost:9981}"
export P79_PAPER_GRADE=0
export PYTORCH_NVML_BASED_CUDA_CHECK=1 CUDA_MPS_PIPE_DIRECTORY="" CUDA_MPS_LOG_DIRECTORY=""

# Site account. DGX's endpoint file carries no site credentials (the A100's copy does).
# The demo's site is a local copy of the benchmark's own database, so the account is the
# benchmark's public test account, defined in the VWA submodule. It is read with `ast`
# (the module asserts on env at import, so it is not imported) and never printed.
# p79 itself keeps B-211's fail-loud rule; this fallback exists only in the demo wrapper.
if [[ -z "${VWA_CLASSIFIEDS_USER:-}" || -z "${VWA_CLASSIFIEDS_PASS:-}" ]]; then
  eval "$(.venv/bin/python3 - <<'PY'
import ast, shlex
src = open("external/visualwebarena/browser_env/env_config.py").read()
acc = next(ast.literal_eval(n.value) for n in ast.walk(ast.parse(src))
           if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "ACCOUNTS" for t in n.targets))
a = acc["classifieds"]
user = a.get("username") or a.get("email")
print(f"export VWA_CLASSIFIEDS_USER={shlex.quote(user)} VWA_CLASSIFIEDS_PASS={shlex.quote(a['password'])}")
PY
)"
fi

if [[ "${1:-}" == "login" ]]; then
  AUTH_DIR="$2"
  exec .venv/bin/python3 -c '
import sys
from pathlib import Path
from p79.utils.auth_refresh import refresh_site_auth
ok = refresh_site_auth("classifieds", Path(sys.argv[1]), benchmark="visualwebarena")
print("login", "ok" if ok else "FAILED")
sys.exit(0 if ok else 3)
' "$AUTH_DIR"
fi

CONFIG="$1"; RUN_ID="$2"; MAX_STEPS="$3"
if [[ -z "${PROXY_API_KEY:-}" ]]; then
  key="$(grep -m1 '^rp_' .auth/qwen_api | tr -d '[:space:]')"
  [[ -n "$key" ]] || { echo "[live] .auth/qwen_api has no rp_ key" >&2; exit 2; }
  export PROXY_API_KEY="$key" QWEN_API_KEY="$key" DASHSCOPE_API_KEY="$key"
fi

exec .venv/bin/python3 scripts/run_experiment.py \
  --config "$CONFIG" --run_id "$RUN_ID" --max_steps "$MAX_STEPS"
