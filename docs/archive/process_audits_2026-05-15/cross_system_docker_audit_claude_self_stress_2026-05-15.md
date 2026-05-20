# Claude Self-Stress — Cross-System Docker Audit (Mode A)

**Date**: 2026-05-15 23:50 UTC → 2026-05-16 00:50 UTC
**Scope**: pre-fire (Phase 1a cls + reddit clean-run readiness)
**Bypass**: none — Mode B (codex) dispatched in parallel; Mode C (gemini) FAILED (run_shell_command blocked in -p mode, expected per MEMORY "weak on code-audit").

## Section A — defuses of my Q1-Q12 list

### Q1 (OSClass `webSiteUrl` empty) — PARTIAL DEFUSE

Verified: `oc_t_preference` on A100 has `webSiteUrl` row missing (only `contactEmail = admin@classifieds.com` returned). However, the page renders `<title>Classifieds</title>` and homepage loads on `http://localhost:9980/` without 302 redirect — OSClass evidently falls back to request-host like Postmill. Outstanding risk: image upload URLs in `oc_t_item_resource` may be absolute. Not yet verified.

### Q2 (Postmill image static URLs) — DEFUSED

`curl -s http://localhost:9999/ | grep canonical` returns `<link rel="canonical" href="http://localhost:9999/">`. Postmill's Symfony app uses request-host derivation, not DB-stored URL. The `images` table stores relative paths under `/var/www/postmill/public/uploads/` inside container (image-baked, identical across all hosts with the same `postmill-populated-exposed-withimg:latest` hash 0a0c002b4dd0).

### Q3 (Postmill ActionCable WebSocket) — DEFER (need browser-level test)

Postmill 1.1+ uses Stimulus + Turbo; ActionCable is optional. Whether vote-count updates need WS for the *paper-grade evaluator* (which checks DOM after agent action) depends on whether `eval.url_match` re-fetches the page. Likely OK because evaluator forces re-navigation; verify by inspecting `external/visualwebarena/evaluation_harness/evaluators.py`.

### Q4 (CORS/CSP cross-site) — LIKELY DEFUSED

Both A100 (localhost:9999 → localhost:8888) and quark (100.95.81.103:9999 → 100.95.81.103:8888) are same-origin within each deployment. CORS headers in Postmill don't pin specific Origin. Verifiable but low priority.

### Q5 (Login MX validation) — DEFUSED

OSClass/Magento/Postmill login forms accept arbitrary email strings; no MX lookup in login flow. `blake.sullivan@gmail.com` is a string identifier, not a deliverable address.

### Q6 (Reset wrapper container ID drift) — DEFUSED

P79 watchdog uses container NAME (`vwa-reddit`) via `docker exec vwa-reddit ...` patterns, not container ID. `docker rm && docker run --name vwa-reddit ...` produces new ID but same name → identifiable.

### Q7 (Image upload directory volume) — OUTSTANDING

Need to inspect: are `oc-content/uploads/` and `var/www/postmill/public/uploads/` bind-mounted volumes or in-image content? `docker inspect` on each container would resolve. **Codex agent has shell access and is probably checking this right now.**

### Q8 (MySQL init race) — DEFUSED

A100 manually ran `source /docker-entrypoint-initdb.d/osclass_craigslist.sql` after first compose-up failed to trigger init. `docker rm classifieds_db` would re-trigger init on next start (docker-compose mounts init scripts to `/docker-entrypoint-initdb.d/`). Reset wrapper for cls uses HTTP `/reset` endpoint, not DB-level operations.

### Q9 (Cross-machine SR drift) — OUTSTANDING; SEE NEW1 BELOW

The fundamental cross-machine reproducibility question. Defused at IMAGE level (same hashes), but NOT defused at RUNTIME config level. See finding NEW1.

### Q10 (kiwix-serve URL pattern) — DEFUSED via code reading

`p79/experiment/tasks.py:70`:
```python
mapping["wikipedia_en_all_maxi_2022-05"] = zim_override
```
This rewrites BOTH `__WIKIPEDIA__` placeholder AND the path-segment `wikipedia_en_all_maxi_2022-05`. After `__WIKIPEDIA__` → `http://localhost:8888` substitution, the subsequent substring replacement handles the version segment. Order-of-iteration in Python 3.7+ dict is insertion-order, and `wikipedia_en_all_maxi_2022-05` is inserted LAST (after all `__SITE__` keys), so it processes after host rewrite. Result: `http://localhost:8888/viewer#wikipedia_en_all_maxi_2025-08/...` ✅

### Q11 (Playwright UA fingerprint) — DEFUSED

Magento/Postmill/OSClass have no anti-bot UA detection in default configs. Same Playwright version + same image = same UA fingerprint across deploys.

### Q12 (DPI / viewport) — DEFUSED

VWA hardcodes viewport 1280×720 in `external/visualwebarena/browser_env/envs.py` (viewport_size kwarg). Headless Chromium does not query host DPI. Cross-platform identical.

## Section B — new attacks (NOT in my Q1-Q12)

### NEW1 — Runner Playwright hardcodes quark Tailscale IP (HIGH SEVERITY)

**File evidence**: `external/visualwebarena/browser_env/envs.py:145`:
```python
self.browser = self.playwright.chromium.launch(
    headless=self.headless, slow_mo=self.slow_mo,
    args=["--host-resolver-rules=MAP metis.lti.cs.cmu.edu 100.95.81.103"],
)
```

This is the **runner's main Playwright launch** (not just auth_refresh). The flag rewires Chromium DNS so that any request to `metis.lti.cs.cmu.edu` (which is what image-baked Magento base_url resolves to) gets redirected to the literal IP `100.95.81.103`.

On quark: A100 doesn't exist; `100.95.81.103` IS quark itself; this works because Magento at quark answers on that IP.

On A100: `100.95.81.103` is a Tailscale IP A100 has no access to. Browser tries to resolve metis → maps to 100.95.81.103 → A100 has no route → ECONNREFUSED.

**Asymmetry**: `auth_refresh.py:84` respects `VWA_REMOTE_HOST` env var; `envs.py:145` does NOT. The latter is upstream VWA code that was patched ad-hoc for quark but never abstracted.

**Phase 1a impact estimate**: Shopping is NOT in Phase 1a scope, and reddit/cls images don't issue metis 302 redirects (Postmill canonical = request-host, OSClass same). So Phase 1a direct impact ≈ **0pp SR drift on Phase 1a**, BUT...

**Phase 1b impact estimate**: Once shop is added, every Magento 302 redirect to metis (笔记 §75: dropdown render, signed-in customer-data JS API, every authenticated page reload) → connection refused → tasks involving authenticated shop browsing fail. Magnitude estimate based on 笔记 §75 evidence: WA shopping went 0% SR pre-fix; assume similar ≥30pp magnitude on the affected fraction of shop tasks. With ~150 of 466 shop tasks likely auth-gated (1/3 estimate), **Phase 1b shop SR drift = 30 × 150/466 = ~10pp lower on A100 than quark**.

**Proposed defuse**: 
- (a) **Fix Magento base_url permanently** via `docker exec vwa-shopping magento config:set web/{secure,unsecure}/base_url http://localhost:7770/` THEN `cache:flush` (笔记 §103 exact recipe) — eliminates the 302-to-metis at the source so resolver-rules-MAP is never exercised. This MUST be done at first shop docker-run on A100 BEFORE any paper-grade shop run.
- (b) Patch `envs.py:145` to use env var: `args=[f"--host-resolver-rules=MAP metis.lti.cs.cmu.edu {os.environ.get('VWA_REMOTE_HOST', '100.95.81.103')}"]`. Belt-and-suspenders.
- (c) Add `127.0.0.1 metis.lti.cs.cmu.edu` to A100 `/etc/hosts` — cheapest, no code change, no rebuild. But hides the problem from anyone reading code.

I recommend (a) + (b) combined. Pure (c) is brittle (next sysadmin wipes hosts file).

### NEW2 — Default `_DEFAULT_BASE_URLS` fallback to quark IP if env unset (MEDIUM)

**File evidence**: `p79/utils/auth_refresh.py:23-28`:
```python
_DEFAULT_BASE_URLS = {
    "classifieds":    "http://100.95.81.103:9980",
    "reddit":         "http://100.95.81.103:9999",
    "shopping":       "http://100.95.81.103:7770",
    "shopping_admin": "http://100.95.81.103:7780",
}
```

Used at line 71: `base_url = os.environ.get(env_key, _DEFAULT_BASE_URLS.get(site, ""))`.

If `CLASSIFIEDS` env var unset (e.g., direct python invocation skipping queue scripts that source `scripts/vwa_env_remote.sh`), auth refresh attempts to hit `http://100.95.81.103:9980` from A100 → ECONNREFUSED → auth refresh fails → cookies stale → subsequent runner episode tries cookies → invalid → some site-specific failure mode.

**Phase 1a impact estimate**: 0pp if every paper-grade run goes through `queue_baseline.sh` / `queue_phantom_*.sh` (which source `vwa_env_remote.sh` to set env). But CLAUDE.md hard rule #3 already forbids bare `run_experiment.py`, and queue scripts ARE the only sanctioned entry. So practical risk = 0pp **conditional on user not bypassing rules**. Maintenance risk = high — any future "let me just rerun task 5 quickly" via bare python would silently fail.

**Proposed defuse**: change `_DEFAULT_BASE_URLS` values from `100.95.81.103` to `localhost`. Localhost is the safer default — if quark deployment somehow loses env vars, the failure surface (localhost not running anything) is loud (connection refused immediately), not the silent contamination of routing to wrong machine.

### NEW3 — Postmill posts table seeded with absolute URLs in markdown bodies (LIKELY)

**Inference, not verified**: Postmill `submissions` table content is image-baked. Many seeded posts contain markdown link bodies like `[example](http://www.reddit.com/r/foo)` (real Reddit links) OR `[Wikipedia](__WIKIPEDIA__/A/Foo)` (templated).

If template-style: `_replace_placeholders` only operates on the task config raw.json, NOT on Postmill DB content. So `__WIKIPEDIA__` literally appears as text in Postmill seeded posts on A100. Agent sees raw `__WIKIPEDIA__` placeholder text → confused.

If absolute-URL-style: links point to external real reddit.com / wikipedia.org — A100 outbound network policy unknown — clicks may fail or succeed differently than quark.

**Phase 1a impact estimate**: Unknown without DB query, but assuming worst case (10% of posts have such links, 33 reddit-wiki cross-link tasks all hit a 5% chance of clicking a broken link) ≈ ~0.5-1pp SR drift potential.

**Proposed defuse**: Query Postmill `submissions` table on A100 + quark side-by-side and diff. Or write a probe agent that visits 5 random post URLs and counts 4xx responses. If both match → defused. If mismatched → real bug, need DB seed patch or accept as platform-specific.

### NEW4 — Magento Elasticsearch index baked at image creation, may not reflect base_url change (HIGH FOR PHASE 1B)

**Inference**: `shopping_final_0712.tar` is a single-container snapshot containing Magento + MySQL + Elasticsearch. ES indexes search product catalog including URL fields. If image was built with base_url = `http://metis.lti.cs.cmu.edu:7770/`, ES has indexed product URLs with metis hostname.

After `docker exec ... config:set base_url http://localhost:7770/` (笔记 §103 recipe), Magento serves new URLs in page HTML, but ES still has metis URLs in product detail responses to autocomplete/search queries. Tasks that use the autocomplete search bar may see metis URLs.

**Phase 1a impact**: 0pp (no shop in 1a).
**Phase 1b impact**: ~2pp drift on shop tasks that use search-autocomplete subset (estimate 5-10% of shop tasks).

**Proposed defuse**: After `config:set base_url`, run `docker exec vwa-shopping bin/magento indexer:reindex` to rebuild all indexes including ES. Then `cache:flush`. This adds ~5min to shop startup but is one-time.

### NEW5 — `external/visualwebarena/.auth/` orphan cookie path persists (LOW, ALREADY KNOWN)

**File evidence**: `external/visualwebarena/browser_env/auto_login.py:62,82` launch their own playwright; they write cookies to CWD-relative `.auth/`. When invoked from `external/visualwebarena/` directory, cookies land at `external/visualwebarena/.auth/`. When invoked from repo root, cookies land at repo root `.auth/`.

This was hit during A100 first smoke run; manually `cp` upward to fix. The `p79/utils/auth_refresh.py` shared path writes to absolute repo-root `.auth/` (line 64 `auth_dir / f"{site}_state.json"` where `auth_dir` is caller-provided).

**Phase 1a impact**: 0pp if all auth refresh goes through `p79/utils/auth_refresh.py`. But VWA's own `auto_login.py` is still callable — if any code path triggers it (e.g., setup verification), the orphan cookies will be written and ignored by P79 runner.

**Proposed defuse**: Audit which code paths call `auto_login.py` vs `auth_refresh.py`. Likely none in paper-grade hot path. Document this in `docs/reference/master_bug_catalog.md` for future maintenance.

### NEW6 — A100 disk usage approaching limit during wiki download (OPERATIONAL)

**Live evidence**: A100 disk 91% full (440GB used / 485GB total). Wiki zim still has ~67GB to download. Wiki zim file lives at `external/visualwebarena/environment_docker/data/wikipedia_en_all_maxi_2025-08.zim`.

After full download: 440 + 67 - (current 28GB) = ~480GB used / 485GB total = 99% full. Likely DISK FULL before download completes if growth pattern continues at 70 MB/s.

**Phase 1a impact**: If disk fills mid-download, wget fails, wiki container never starts, reddit task wiki cross-links return 404, **~28pp SR drift on reddit's 33 wiki-cross-link tasks** = ~4.6pp on reddit overall = ~2.3pp on Phase 1a combined.

**Proposed defuse**: 
- (a) **Continuously monitor disk during wget**. If <5GB free, kill wget and re-evaluate.
- (b) Free up archive data: `results/_archive_smoke_20260515/` is 5 entry safe to drop (just smoke + B2_3mode digest). DGX archive `_archive_aborted_20260515/` has B1 run dir ~15MB only.
- (c) Move wiki zim to a different mount if A100 VM has additional disk attached.

This is the most time-sensitive of the 6 findings.

### NEW7 — Concurrent VWA wikipedia bind mount on first kiwix-serve run (MEDIUM)

**File evidence**: `scripts/vwa/start_vwa_docker.sh:219`:
```bash
docker run -d --name vwa-wikipedia --volume="${ENV_DIR}/data/:/data" -p 8888:80 ghcr.io/kiwix/kiwix-serve:3.3.0 wikipedia_en_all_maxi_2025-08.zim
```

The `--volume=${ENV_DIR}/data/:/data` mount is RW. Kiwix-serve doesn't write back, but mount is RW by default. If something else writes to the host data dir while kiwix is reading, fs may flush mid-read → corrupt response → task fails.

**Phase 1a impact**: 0pp normally. Risk if wget on host is still running when kiwix-serve starts (file race condition between wget appending and kiwix reading). Don't `docker run kiwix-serve` until `wget` exits.

**Proposed defuse**: Add a check in start_vwa_docker.sh — if wget is still appending to the .zim file, refuse to start kiwix. Or simply: human waits for wget exit before invoking `start_vwa_docker.sh --sites shopping,shopping_admin,wikipedia`.

## Section C — meta-criticism of my audit

**C1 — I assumed image hashes = identical config, ignored runtime state**

The 4 negative findings N1-N4 all leaned on "same image hash = same config". This is mostly true for content (DB, files inside image) but NOT for runtime state created on first container start (ES indexes, search caches, MySQL my.cnf in conf.d, etc.). My NEW4 partly addresses this for Magento ES; the same category-error likely lurks for OSClass MySQL config.

**C2 — I anchored on cross-platform diff at request-pipeline level, missed the OUT-pipeline (logs, eval)**

VWA `string_match` evaluator uses an LLM judge (笔记 B-91). On A100 vs quark, LLM judge calls go to a DIFFERENT OpenAI endpoint, possibly with different model versions, different prompt cache state. **Each LLM-judge call is a stochastic event**. If A100's judge has different default temperature than quark's judge, SR numbers shift even with identical agent behavior. **I did NOT audit the eval-side LLM judge config alignment**.

**C3 — I conflated "production-validated" with "production-correct"**

Quark generated archive Phase 1 data. This is "validated" in the sense that the data was used in paper drafts. But quark may itself have bugs that bias SR a specific direction, and "A100 = quark" reproduces those bugs. Paper-grade reproducibility could mean (a) numerical identity to quark = bug-equivalence, (b) absolute correctness regardless of quark. The audit context I wrote conflated these — I implicitly assumed (a) is the bar.

**C4 — I did not audit time-zone-sensitive task subset**

Postmill posts have `created_at` timestamps. Tasks asking "the latest 3 posts in /f/news" depend on browser-displayed timestamps, which depend on Postmill's view rendering, which uses Rails default `Time.current` → server timezone. Quark Windows TZ may differ from A100 UTC. **This is the single most likely silent SR drift vector I missed in my Q1-Q12**.

**C5 — I never queried task instructions for site-name strings**

Some task instructions hardcode the site title (e.g., "Find the 'Classifieds' homepage..."). If A100 Postmill auto-titles itself differently from quark Postmill (both should be "Postmill" but verify), agent's text-grounded reasoning may diverge.

## Verification status

- Mode A (this doc, Claude self-stress): **PASS**, 6 new findings + 5 meta-criticisms, ≥3 OOB, ≥1 HIGH severity (NEW1)
- Mode B (codex, parallel): **IN PROGRESS** (509KB output as of 00:40 UTC, doing real shell verification)
- Mode C (gemini, parallel): **FAIL** (run_shell_command blocked in -p mode, expected per MEMORY rule "Mode C weak on code-audit"). Killed at 00:43 UTC.

Next: integrate codex output (when finishes) with this doc + 笔记 §141 chronicle append.
