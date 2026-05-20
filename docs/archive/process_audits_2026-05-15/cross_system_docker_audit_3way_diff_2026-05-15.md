# Cross-System Docker Audit — 3-AI Stress Diff (FINAL)

**Date**: 2026-05-15 → 2026-05-16
**Scope**: pre-fire (Phase 1a cls + reddit clean-run readiness on A100 self-host vs quark production archive)

## Verification status

- **Mode A** (Claude self-stress): **PASS** — 7 NEW findings + 5 meta-criticisms; output at `cross_system_docker_audit_claude_self_stress_2026-05-15.md`
- **Mode B** (codex `--sandbox danger-full-access`): **PASS** — 7 attacks, 4+ OOB, 1.3MB output with deep shell verification (ssh A100 + `docker exec` queries); output at `docs/checkpoints/codex_outputs/cross_system_docker_audit_codex_003729.md`
- **Mode C** (gemini `--yolo` retry 4): **PASS** — 3 OOB attacks + 5 Q-defuses with code citations + 1 meta-critique; output at `docs/checkpoints/codex_outputs/cross_system_docker_audit_gemini_005638.md`. Retries 1-3 failed for permission/gitignore/ripgrep issues, all root-caused + fixed → see `gemini_stress_skill_replica.md` v8.0.

## 3-AI agreement zones

### Where all 3 AIs agree (highest-confidence weakest, top priority)

**TZ drift** (Claude C4 meta + Gemini NEW-OOB-1 + implicit in codex docker-run audit): A100 UTC vs quark Windows TZ. Postmill timestamps render in server TZ. Reddit task evaluators have `must_include` with hardcoded dates (e.g. `"08-11-2023"`) baked against quark TZ. Midnight-boundary tasks display +/-1 day on A100. **1-2pp drift on reddit date-sensitive subset**.

### Where Codex + Gemini agree (Claude self-stress missed)

**Classifieds reset wrapper broken**: codex Attack 6 verified via `docker exec` → `oc_t_item_comment` count 2 stale post-"reset"; gemini independently found via `start_vwa_docker.sh:25` + OSClass source code (no `reset_database` controller, has `reset` with `$_POST['token']`). The wrapper does GET to wrong endpoint name; OSClass `index.php` swallows invalid `page=` returning homepage 200; bash sees 200 and reports success. **0 SQL executed across entire Phase 1a run**. Magnitude estimate diverges (3-5pp gemini vs 0.2-0.8pp codex) — Gemini is more aggressive because it assumes every-episode reset failure; codex bounds to `require_reset=true` task subset only.

### Where Codex alone caught (Claude + Gemini both missed)

- **Phase 1a operationally requires shop + wiki** (refutes my Claude N4 defuse) — 21 `__SHOPPING__` refs in cls + 33 `__WIKIPEDIA__` refs in reddit = 54/444 cross-link blast pool
- **`VWA_REMOTE_HOST="${VAR:-localhost}"` shell-inherit override** — single inherited env flips ALL site URLs to quark silently
- **`mysql:8.1` mutable tag** — DB engine version drift cls collation/sql_mode → search-order sensitive tasks
- **Playwright/Pillow/BS4 version drift** A100 vs DGX — SoM/vision bbox encoding diffs
- **Magento `|| true` swallowing patch failures** in setup script
- **444 vs 429 N/A exclusion denominator footgun** (`p79/experiment/tasks.py:104`)

### Where Gemini alone caught (Claude + Codex both missed)

- **🎯 `--host-resolver-rules` `localhost` literal invalid** — `auth_refresh.py:84` `_resolver_ip = ... "VWA_REMOTE_HOST", "..."`. When env is `localhost`, Chromium MAP rule mapping `metis.lti.cs.cmu.edu` → literal string `localhost` is invalid (MAP needs IP, not hostname). Chromium DNS resolver silently hangs 30s, runner episodes time out as fail. Phase 1a 0pp, **Phase 1b shop 100pp** auth-refresh dead. Subtle Chromium-DNS-corner-case impossible for non-deployed-AI to surface.
- **🎯 Evaluator-Environment State Divergence (meta)** — `external/visualwebarena/evaluation_harness/evaluators.py` `page.evaluate` runs against agent's FINAL navigation state. Agent posts comment ✅ → STOP from homepage → evaluator JS `document.querySelector("#main")` queries homepage → false negative. Entirely new attack vector category, NOT about deployment, about evaluator harness static-DOM-assumption brittleness. Affects every static-DOM-eval task across all baselines/modes.

### Where Claude alone caught (Codex + Gemini missed)

- **Postmill seeded post body absolute URLs** (NEW3, inference, not verified) — needs DB query to confirm
- **`auto_login.py` orphan cookie path** (NEW5) — operational, low impact
- **kiwix RW mount race with wget** (NEW7) — operational, fixable by sequencing
- **Eval-side LLM judge stochasticity** (C2 meta) — VWA `string_match` LLM judge calls = stochastic, A100 vs quark may hit different API state
- **Production-validated ≠ production-correct** (C3 meta) — "A100 = quark numerically" reproduces quark bugs; the bar is unclear

### Defuses Gemini contributed (Claude Q1-Q12 cleared)

- **Q1 OSClass webSiteUrl**: `scripts/vwa/start_vwa_docker.sh:153` sed-patches `CLASSIFIEDS` env in docker-compose.yml; OSClass `config.php` reads via `getenv("CLASSIFIEDS")` — fully populated ✅
- **Q6 reset container-ID drift**: Playwright always targets `localhost:9999`; container ID never leaks to agent or evaluator ✅
- **Q7 upload volume**: Postmill `public/media/` in writable container layer (NOT volume); `docker rm -f` obliterates → 100% reset to seeded state ✅
- **Q8 MySQL init race**: `reset_vwa_sites.sh:45-55` 60-attempt HTTP polling loop prevents race ✅
- **Q10 kiwix ZIM URL pattern**: `tasks.py:52` substring rewrite confirmed ✅

## Consolidated Bug List (16 total, prioritized)

### 🔴 P0 — Phase 1a launch blockers (must fix before fire)

#### BUG-1 — Phase 1a cls+reddit tasks reference shop+wiki URLs (codex CodexOnly-1)

**Evidence**: `external/visualwebarena/config_files/vwa/test_classifieds.raw.json:8072` (cls task 224 → `__SHOPPING__/`), `test_reddit.raw.json:1584` (reddit task 45 → `__WIKIPEDIA__/`). Counts: cls has 21 `__SHOPPING__` refs, reddit has 33 `__WIKIPEDIA__` refs = 54/444 affected tasks.

**Blast**: A100 currently has shop:7770 = `000` (no container) and wiki:8888 = `000` (downloading). Every cross-link from cls/reddit to those services 404s or hangs. Affected pool = **12.2pp max SR drift on Phase 1a**. Silent contamination because runner counts these as agent-failure not infra-failure.

**Fix**:
```bash
# (a) Phase 1a launch gate must verify every endpoint referenced in raw.json is reachable
python3 -c "
import json, os, re
sites = ['classifieds','reddit']
needed = set()
for s in sites:
    for t in json.load(open(f'external/visualwebarena/config_files/vwa/test_{s}.raw.json')):
        for v in str(t).split():
            m = re.search(r'__([A-Z_]+)__', v)
            if m: needed.add(m.group(1))
print('Endpoints needed:', needed)
"
# (b) Bring up shop + wiki BEFORE Phase 1a fire (waiting on wave-2 wiki download anyway)
# (c) Add to start_vwa_docker.sh: refuse start without all referenced containers up
```

#### BUG-2 — `VWA_REMOTE_HOST` shell-inherit override (codex CodexOnly-2)

**Evidence**: `scripts/vwa_env_remote.sh:6` on A100: `VWA_REMOTE_HOST="${VWA_REMOTE_HOST:-localhost}"`. Default-expansion form means an inherited shell env wins over the file's intent.

**Blast**: Worst case = any session with `export VWA_REMOTE_HOST=100.95.81.103` inherited (e.g. from `.bashrc` quark testing) silently flips ALL site URLs to quark. Up to **100% of 7992 Phase 1a episodes silently labeled A100 but hitting quark**. Mid-case = auth fails (different cookies expected), some tasks pass against quark while labeled A100 → corruption-in-data-attribution.

**Fix**:
```bash
# scripts/vwa_env_remote.sh on A100 — replace line 6
VWA_REMOTE_HOST="localhost"  # literal, not default-expansion
# In queue_baseline.sh add preflight assert AFTER source vwa_env_remote.sh:
for var in CLASSIFIEDS REDDIT SHOPPING WIKIPEDIA; do
  case "${!var}" in
    *localhost*|*127.0.0.1*) ;;
    *) echo "✗ FATAL: $var=${!var} contains non-localhost; refusing launch on A100"; exit 2 ;;
  esac
done
```

#### BUG-3 — Classifieds reset wrapper uses wrong endpoint + wrong HTTP method (codex Attack 6 + gemini NEW-OOB-2, 2-AI agree)

**Evidence**: `scripts/maintenance/reset_vwa_sites.sh:25` builds `GET http://localhost:9980/index.php?page=reset_database&token=...`. OSClass source at `/usr/src/myapp/oc-includes/osclass/controller/reset.php:15-18` expects `REQUEST_METHOD === 'POST'` + `$_POST['token']` + `page=reset` (NOT `reset_database`). Codex verified via `docker exec`: post-"reset" `oc_t_item_comment` still has 2 stale records.

**Blast**: Wrapper returns 200 (OSClass swallows invalid `page=`, serves homepage) → script reports success → **0 SQL executed across entire Phase 1a run**. Stale state from previous episode contaminates next: agent finds ads it shouldn't see, can't find ads it should create. Gemini estimate **3-5pp drift** on cls; codex bounds to require_reset tasks **0.2-0.8pp** worst-case. Use gemini estimate as upper bound.

**Fix**:
```bash
# scripts/maintenance/reset_vwa_sites.sh — rewrite cls reset:
_reset_vwa_local_classifieds() {
    local token="${CLASSIFIEDS_RESET_TOKEN:?missing token}"
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" \
           -X POST -d "token=${token}" \
           "http://localhost:9980/index.php?page=reset")
    [ "$code" = "200" ] || { echo "cls reset HTTP $code"; return 1; }
    # Mutation sentinel: verify reset actually happened
    local count
    count=$(docker exec classifieds_db mysql -uroot -ppassword osclass -sN -e \
            "SELECT COUNT(*) FROM oc_t_item_comment WHERE b_active=1;")
    [ "$count" = "0" ] || { echo "cls reset SQL not executed (still $count comments)"; return 1; }
}
```

#### BUG-4 — Playwright `--host-resolver-rules` `localhost` literal silently hangs (gemini NEW-OOB-3, gemini-unique)

**Evidence**: `p79/utils/auth_refresh.py:84` `_resolver_ip = os.environ.get("VWA_REMOTE_HOST", "100.95.81.103")` + line 92 `args=['--host-resolver-rules=MAP metis.lti.cs.cmu.edu {_resolver_ip}']`. On A100, `VWA_REMOTE_HOST=localhost`, so MAP target = literal string `localhost` not an IP.

**Blast**: Chromium `--host-resolver-rules` MAP syntax requires IP (or hostname that the chromium-native resolver can resolve). `localhost` works in normal lookups via /etc/hosts BUT in the resolver-rules pipeline it silently fails resolution → 30s DNS timeout per metis-redirect → Playwright `page.goto` timeouts. Phase 1a 0pp (no metis 302 surface for cls/reddit). **Phase 1b shop 100pp** — Magento dropdown JS issues `XHR /customer/section-data` which 302 → metis → resolver hangs → Playwright fails entire login flow.

**Fix**:
```python
# p79/utils/auth_refresh.py:84 — replace
_resolver_ip = os.environ.get("VWA_REMOTE_HOST", "100.95.81.103")
if _resolver_ip == "localhost":
    _resolver_ip = "127.0.0.1"
# Also patch external/visualwebarena/browser_env/envs.py:145 (same flag, hardcoded 100.95.81.103)
```

#### BUG-5 — Magento setup `|| true` swallows patch failures (codex Attack 2)

**Evidence**: `scripts/setup/a100_self_host_vwa.sh` Magento patch step (the `docker exec magento config:set` block) uses `|| true` to suppress errors. If container not ready or patch syntax wrong, error is swallowed and image-baked metis URL stays active → Magento 302→metis → silently activates BUG-4 hang path.

**Blast**: Phase 1a 0pp (no shop). Phase 1b: causes BUG-4 to manifest at 100pp shop SR drop even after operator "ran the fix". Compounding silent-failure surface.

**Fix**:
```bash
# scripts/setup/a100_self_host_vwa.sh — remove ALL `|| true` from Magento patch block
# After patches, assert:
docker exec vwa-shopping bash -c '
  base=$(php -r "include /var/www/magento2/app/etc/env.php; echo \$config[\"db\"][\"connection\"][\"default\"][\"host\"];")
  url=$(php /var/www/magento2/bin/magento config:show web/unsecure/base_url)
  [ "$url" = "http://localhost:7770/" ] || { echo "✗ Magento base_url=$url not patched"; exit 2; }
'
docker exec vwa-shopping bash -c 'php /var/www/magento2/bin/magento cache:flush'
```

#### BUG-6 — Timezone A100 UTC vs quark Windows TZ (3-AI agree: Claude C4 + gemini NEW-OOB-1)

**Evidence**: `scripts/vwa/start_vwa_docker.sh:119` `docker run --name vwa-reddit ...` without `-e TZ=`. A100 host = UTC. Quark = Windows local TZ. Postmill Symfony uses container TZ for rendering `created_at` in submission/comment headers.

**Blast**: Reddit task evaluators with `must_include: ["08-11-2023"]` baked against quark TZ. Crossing midnight UTC vs Quark-TZ → date renders +/-1 day → eval string-match fails → false negative. **~1-2pp drift** on date-sensitive reddit subset (gemini estimate, Claude C4 agrees order-of-magnitude).

**Fix**:
```bash
# scripts/vwa/start_vwa_docker.sh — add -e TZ=<quark-tz> to ALL docker run commands
# First: determine quark TZ
ssh quark "powershell -Command 'Get-TimeZone | Select Id'" || echo "Default to America/New_York"
# Then: in start_vwa_docker.sh, add to vwa-reddit/classifieds/vwa-shopping/vwa-wikipedia
docker run -d --name vwa-reddit -e TZ=America/New_York ...  # or whatever quark reports
```

#### BUG-7 — A100 disk 91% during wiki download (Claude NEW6)

**Evidence**: `df -h /home` shows 37GB free / 485GB total = 91% used at 2026-05-16 00:50 UTC. Wiki ZIM target ~95GB, currently 36GB downloaded.

**Blast**: If disk fills before wget completes → wget fail with "no space left" → wiki container never starts → 33 reddit wiki-cross-link tasks return 404 → **~28pp drift on those tasks = ~4.4pp on reddit total = ~2.2pp Phase 1a**.

**Fix**:
```bash
# (a) Monitor disk during wget; if <5GB free, kill wget + manual intervention
ssh condense-a100 'while pgrep -f "wget.*2025-08" > /dev/null; do
    FREE=$(df --output=avail /home | tail -1)
    [ "$FREE" -lt 5000000 ] && { echo "ALERT: <5GB free"; break; }
    sleep 300
  done'
# (b) Pre-emptively archive: A100 `results/_archive_smoke_20260515/` can be tar-balled to /tmp + deleted; DGX archive already separate
# (c) Confirm wave-2 ETA — at current speed (~50MB/s) ~20 more min needed
```

### 🟡 P1 — Paper-grade reviewer defense (fix before submission)

#### BUG-8 — `mysql:8.1` mutable tag (codex CodexOnly-3)

**Evidence**: `external/visualwebarena/environment_docker/classifieds_docker_compose/docker-compose.yml:15` uses `image: mysql:8.1`. Current A100 digest = `sha256:f61944ff3f296136...`. Quark digest unknown (likely older, given quark deployed earlier in 2026).

**Blast**: cls has 125 `url_match` + 72 `string_match` tasks; many depend on search-ordering, category-filtering, date-collation. MySQL minor-version refresh can change `collation_*` defaults or `sql_mode`. Estimate **0.9-1.8pp aggregate Phase 1a SR drift** if quark and A100 MySQL differ.

**Fix**:
```bash
# Pin to digest
sed -i 's|image: mysql:8.1$|image: mysql:8.1@sha256:f61944ff3f296136...|' \
  external/visualwebarena/environment_docker/classifieds_docker_compose/docker-compose.yml
# Record + diff against quark
docker exec classifieds_db mysql -uroot -ppassword -e \
  "SHOW VARIABLES LIKE 'collation_%'; SHOW VARIABLES LIKE 'sql_mode'; SELECT VERSION();" \
  > docs/reference/mysql_a100_baseline.txt
# Compare with quark equivalent before launch
```

#### BUG-9 — Playwright/Pillow/BS4 version drift A100 vs DGX (codex CodexOnly-5)

**Evidence**: A100 `playwright 1.59.0` / `Pillow 10.0.1` / `beautifulsoup4 4.12.2`. DGX `playwright 1.58.0` / `Pillow 12.1.1` / `beautifulsoup4 4.14.3`. Used in `external/visualwebarena/browser_env/envs.py:186` (observation generation) + `p79/experiment/runner/main.py:757` (reference image loading).

**Blast**: SoM/vision/phantom_som modes generate observations via Playwright + Pillow. Bbox encoding, screenshot DPR, AXTree serialization can shift between versions. Estimate **0.5-1.5pp aggregate SR drift**, concentrated in SoM/vision modes.

**Fix**:
```bash
# Pin to single version across A100/DGX (DGX 1.58 likely correct since archive data was generated on it)
# On A100:
.venv/bin/pip install 'playwright==1.58.0' 'Pillow==12.1.1' 'beautifulsoup4==4.14.3'
.venv/bin/playwright install chromium
# Record Chromium revision (Playwright bundles its own)
.venv/bin/playwright --version > docs/reference/playwright_pin_baseline.txt
```

#### BUG-10 — Paper denominator 444 vs runner scored 429 (codex meta)

**Evidence**: `p79/experiment/tasks.py:104` `exclude_na_tasks = bool(task_cfg.get("exclude_na_tasks", True))`. Default True. Codex grep counted 15 N/A tasks across cls+reddit raw.json.

**Blast**: Paper §1/§4 currently says "Phase 1a tested 444 tasks"; actual runner scored set is 429. Reviewer footgun ("you say 444 but show me where 444 numbers are computed → they're 429 → why?"). Not a Docker bug, but paper-grade defensibility issue.

**Fix**: Paper §3 (deployment scope) must explicitly state: "Of 444 nominal tasks in cls+reddit, 15 are N/A (unanswerable, see §139.8) and excluded from the scored set per pre-registered §139.8 decision → 429 scored episodes per condition." Cite preregistration.md.

#### BUG-11 — Evaluator-Environment State Divergence (gemini meta, gemini-unique) → **WONTFIX (user decision 2026-05-16)**

**Evidence**: `external/visualwebarena/evaluation_harness/evaluators.py` runs `page.evaluate` against final navigation state agent left.

**Blast**: Agent posts comment successfully ✅ → STOP from homepage → evaluator's `document.querySelector("#main")` returns wrong (homepage) DOM → false negative. Magnitude unknown but plausibly **1-3pp** across all baselines/modes.

**Status — WONTFIX (decision 2026-05-16)**: This is VWA upstream design, not P79 deployment bug. Reasoning to defer:
1. **Quark archive also affected** — same VWA eval code generated archive data; fix on A100 only would create cross-platform divergence WORSE than the original issue
2. **Baseline-invariant** — agent stop-state is LLM-random, not biased toward B0/B1/B2 → does not contaminate paper-1 phantom-routing claim
3. **Paper-1 scope discipline** — eval harness reform is paper-3 / VWA upstream contribution scope
4. **Mode-asymmetry hypothetical** — gemini's "vision-mode stays on task page more, DOM-mode wanders home more" is plausibility-argument, not evidence
5. **Cross-deploy consistency > absolute correctness** — A100 archive numbers must match quark archive numbers to validate paper-1 cross-platform claim; fixing VWA eval breaks that contract

**Disclosure (paper §Limitations)**: "VWA evaluator runs `page.evaluate` against agent's final navigation state. Mode-asymmetric agent stop behavior may introduce 1-3pp noise, but this noise is invariant across our quark archive baseline and A100 self-host deployment, preserving cross-platform comparability."

**If mode-asymmetry empirically demonstrates** in Phase 1a data (e.g., DOM-mode SR << vision-mode SR by an unexpected margin) → revisit as paper-2 / paper-3 investigation, not Phase 1a blocker.

### 🟢 P2 — Latent / Phase 1b blockers

#### BUG-12 — `envs.py:145` hardcodes 100.95.81.103 in runner Playwright (Claude NEW1 + codex Attack 1)

**Evidence**: `external/visualwebarena/browser_env/envs.py:145` `args=["--host-resolver-rules=MAP metis.lti.cs.cmu.edu 100.95.81.103"]`. Asymmetric with `auth_refresh.py:84` which respects env var; runner does not.

**Blast**: Phase 1a 0pp (cls/reddit don't 302→metis). Phase 1b: combined with BUG-5 (Magento patch silently failing) → Magento 302→metis → runner's chromium maps to 100.95.81.103 → A100 has no Tailscale → ECONNREFUSED. Or if A100 ever gets Tailscale (e.g. future eng debug), runner silently reads quark prod data ("ghost-prod-read").

**Fix**: Same as BUG-4 — env-var-ify and 127.0.0.1 fallback:
```python
# external/visualwebarena/browser_env/envs.py:145 (P79 fork patch)
import os
_resolver = os.environ.get("VWA_REMOTE_HOST", "127.0.0.1")
if _resolver == "localhost": _resolver = "127.0.0.1"
args=[f"--host-resolver-rules=MAP metis.lti.cs.cmu.edu {_resolver}"]
```

#### BUG-13 — Magento ES indexes baked with metis URLs (Claude NEW4)

**Evidence**: shopping_final_0712.tar single-container snapshot includes Magento + MySQL + Elasticsearch. ES indexes built at image-creation time → contain metis URLs in product detail fields.

**Blast**: Phase 1b ~2pp shop search-autocomplete tasks; eval may compare URL strings.

**Fix**: After BUG-5 (Magento base_url config:set):
```bash
docker exec vwa-shopping bash -c 'php /var/www/magento2/bin/magento indexer:reindex'
docker exec vwa-shopping bash -c 'php /var/www/magento2/bin/magento cache:flush'
```

### ⚪ P3 — Operational hygiene

#### BUG-14 — `_DEFAULT_BASE_URLS` 100.95.81.103 fallback (Claude NEW2)

**Evidence**: `p79/utils/auth_refresh.py:23-28`. Fallback path only when env unset.

**Blast**: 0pp if queue scripts source env (CLAUDE.md hard rule #3). Risk if anyone bypasses queue.

**Fix**: Change defaults to `localhost`:
```python
_DEFAULT_BASE_URLS = {
    "classifieds":    "http://localhost:9980",
    "reddit":         "http://localhost:9999",
    "shopping":       "http://localhost:7770",
    "shopping_admin": "http://localhost:7780",
}
```

#### BUG-15 — `auto_login.py` orphan cookie path (Claude NEW5)

**Evidence**: `external/visualwebarena/browser_env/auto_login.py:62,82` write cookies to CWD-relative `.auth/`.

**Blast**: 0pp on hot path (P79 uses `p79/utils/auth_refresh.py` which writes abs path).

**Fix**: Document in `docs/reference/master_bug_catalog.md`. Maybe patch VWA submodule to use abs path.

#### BUG-16 — kiwix container RW mount race with wget (Claude NEW7)

**Evidence**: `scripts/vwa/start_vwa_docker.sh:219` `docker run ... --volume="${ENV_DIR}/data/:/data" ...`.

**Blast**: 0pp if human waits wget exit before starting kiwix container.

**Fix**: Add wget-alive check:
```bash
# scripts/vwa/start_vwa_docker.sh — before start_wikipedia()
if pgrep -af "wget.*wikipedia.*\.zim" > /dev/null; then
    echo "✗ refusing to start kiwix-serve while wget still downloading ZIM"
    return 2
fi
```

## Synthesized launch sequence (before Phase 1a fire)

```bash
# Wait wave-2 wiki complete + verify file integrity
ssh condense-a100 'until ! pgrep -f "wget.*2025-08"; do sleep 60; done; ls -lh /home/ubuntu/workspace/p79/external/visualwebarena/environment_docker/data/wikipedia_en_all_maxi_2025-08.zim'

# Apply P0 fixes (in order):
# 1. BUG-2: literal VWA_REMOTE_HOST + preflight assert
# 2. BUG-3: cls reset wrapper rewrite (POST + mutation sentinel)
# 3. BUG-4: auth_refresh.py localhost → 127.0.0.1
# 4. BUG-6: docker run -e TZ=<quark> patch
# 5. BUG-7: archive smokes if disk pressure
# 6. BUG-1: bring up shop + wiki containers
# 7. BUG-5: Magento base_url config:set + verify

# Smoke test: 3 tasks per site to verify each P0 fix works
make launch BASELINE=B2 SITE=classifieds MODE=dom MAX_TASKS=3 RESET_BEFORE=1

# If smoke passes → Phase 1a paper-grade fire
```

## Versioning

- v1 (2026-05-15 23:50 UTC) — Claude self-stress only (Mode A)
- v2 (2026-05-16 00:43 UTC) — Mode B codex done, Mode C gemini failed ×3 (permission + gitignore + ripgrep)
- **v3 (2026-05-16 01:18 UTC) FINAL** — Mode C gemini retry 4 PASS after ripgrep install + flag fix; 16-bug consolidated list with blast + fix per bug; gemini contributed 2 unique findings (BUG-4 + BUG-11 meta) + 5 Q-defuses + cross-AI agreement on BUG-3 (with codex) and BUG-6 (with Claude)
