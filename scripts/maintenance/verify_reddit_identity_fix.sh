#!/usr/bin/env bash
# B-1884 / Fix 4 — A100 one-shot live verification (PROTOCOL_NOTE_04 §4).
# Brings up vwa-reddit, simulates the task-138 rename, runs the DEPLOYED
# restore_reddit_identity() code, then proves a fresh login as the seed user
# succeeds — the decisive check on whether a username-only restore suffices
# (i.e. postmill login does not key off a separate canonical/normalized column).
set -uo pipefail
REPO=/home/ubuntu/workspace/p79
PY=/home/ubuntu/venvs/p79/bin/python
cd "$REPO"

pg() { docker exec vwa-reddit su - postgres -c "psql -d postmill -tAc \"$1\""; }

echo "### 1. ensure vwa-reddit is up"
if ! docker ps --format '{{.Names}}' | grep -qx vwa-reddit; then
  docker run --name vwa-reddit -p 9999:80 -d postmill-populated-exposed-withimg:latest >/dev/null
  echo "  (started fresh container)"
fi
for i in $(seq 1 40); do pg "select 1" >/dev/null 2>&1 && { echo "  postgres ready"; break; }; sleep 2; done

echo "### 2. users columns — does a canonical/normalized username column exist?"
pg "select column_name from information_schema.columns where table_name='users' order by ordinal_position" | paste -sd' '

echo "### 3. baseline @ id 13915 (id|username|normalized_username)"
pg "select id||'|'||username||'|'||normalized_username from users where id=13915"

echo "### 4. simulate the REAL task-138 rename -> Patrick (BOTH columns, as postmill does)"
pg "update users set username='Patrick', normalized_username='patrick' where id=13915" >/dev/null
pg "select id||'|'||username||'|'||normalized_username from users where id=13915"

echo "### 5. run THE DEPLOYED Fix 4 code: restore_reddit_identity({})"
$PY -c "import logging; logging.basicConfig(level=logging.INFO, format='  %(message)s'); from p79.utils.reddit_identity import restore_reddit_identity; print('  returned:', restore_reddit_identity({}))"

echo "### 6. confirm BOTH columns restored"
pg "select id||'|'||username||'|'||normalized_username from users where id=13915"

echo "### 7. LOGIN TEST as the seed user (decisive canonical-column behavioral check)"
[ -f scripts/vwa_env_remote.sh ] && source scripts/vwa_env_remote.sh 2>/dev/null || true
export REDDIT="http://localhost:9999"
export VWA_REDDIT_USER="${VWA_REDDIT_USER:-MarvelsGrantMan136}"
export VWA_REDDIT_PASS="${VWA_REDDIT_PASS:-test1234}"
export VWA_REMOTE_HOST="127.0.0.1"
echo "  (login as user=${VWA_REDDIT_USER})"
$PY -c "
from pathlib import Path
from p79.utils.auth_refresh import refresh_site_auth
ok = refresh_site_auth('reddit', Path('/tmp/p79_verify_auth'), base_urls={'reddit':'http://localhost:9999'})
print('  RESULT:', 'LOGIN_OK' if ok else 'LOGIN_FAIL')
"
echo "### done (account left at seed state)"
