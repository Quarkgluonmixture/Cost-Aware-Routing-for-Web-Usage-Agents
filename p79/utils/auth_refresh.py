"""Shared auth refresh logic for Magento (shopping/shopping_admin) sites.

Extracted from scripts/maintenance/experiment_watchdog.py so that both the watchdog
and the experiment runner can reuse the same login routine.

A1.5 refactor (2026-05-16, B-211 + B-212 + B-220/221/224):
- Credentials moved to env vars (B-211 double-leak cleanup); no plaintext in tracked code.
  Required env: VWA_<SITE>_USER + VWA_<SITE>_PASS for each site. See scripts/vwa_env_remote.sh
  template (gitignored by design per CLAUDE.md hard rule).
- LOGIN_FAILED detection replaced from URL substring to urlparse path-equal + login_qs subset
  match (B-212); pre-fix substring match systematically false-positived on shopping_admin
  (every post-login URL contains "/admin/*" substring) and was fragile on reddit ("/login").
- New `auth_required_gate(site, auth_dir, benchmark)` raises on failure (B-220/221/224 fail-cascade
  closure); use at queue post-reset / runner pre-episode / watchdog launch where stale session
  is paper-grade unacceptable. Retains existing `refresh_site_auth(...) -> bool` for soft-fail
  consumers that already handle the False return.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


class AuthRefreshConfigError(Exception):
    """Raised when required env vars (VWA_<SITE>_USER/PASS) are missing."""


class AuthRefreshFailure(Exception):
    """Raised by auth_required_gate when refresh fails after retries.

    Callers (queue / runner / watchdog) should let this propagate to abort
    the current launch path — paper-grade contamination prevention.
    """


# B-211 (2026-05-16): credentials loaded from env vars, NOT hardcoded.
# Layout matches the canonical VWA reference test accounts; the env file
# scripts/vwa_env_remote.sh (gitignored) is the runtime source.
_ACCOUNT_ENV_KEYS = {
    "classifieds":    ("VWA_CLASSIFIEDS_USER",    "VWA_CLASSIFIEDS_PASS"),
    "reddit":         ("VWA_REDDIT_USER",         "VWA_REDDIT_PASS"),
    "shopping":       ("VWA_SHOPPING_USER",       "VWA_SHOPPING_PASS"),
    "shopping_admin": ("VWA_SHOPPING_ADMIN_USER", "VWA_SHOPPING_ADMIN_PASS"),
}

_DEFAULT_BASE_URLS = {
    # BUG-14 fix (2026-05-16, Claude NEW2): defaults changed from quark Tailscale
    # IP (private Tailscale IP) to localhost. CLAUDE.md hard rule #3 already mandates
    # going through queue scripts which source vwa_env_remote.sh setting env vars
    # — so this fallback is 0pp on hot path. But if anyone bypasses queue and
    # invokes bare python, loud-fail (localhost not running anything on DGX) is
    # safer than silent-route-to-quark-prod (could pollute prod state).
    "classifieds":    "http://localhost:9980",
    "reddit":         "http://localhost:9999",
    "shopping":       "http://localhost:7770",
    "shopping_admin": "http://localhost:7780",
}

_LOGIN_PATHS = {
    "classifieds":    "/index.php?page=login",
    "reddit":         "/login",
    "shopping":       "/customer/account/login/",
    "shopping_admin": "/admin",
}

_ENV_KEYS = {
    "classifieds":    "CLASSIFIEDS",
    "reddit":         "REDDIT",
    "shopping":       "SHOPPING",
    "shopping_admin": "SHOPPING_ADMIN",
}


def _load_account(site: str) -> Tuple[str, str]:
    """Load (username, password) from env. Raise AuthRefreshConfigError on missing.

    B-211 (2026-05-16): replaces the hardcoded _ACCOUNTS table. Set env via
    `scripts/vwa_env_remote.sh` (gitignored). Missing creds → fail loud rather
    than silent fallback to canonical demo accounts.
    """
    if site not in _ACCOUNT_ENV_KEYS:
        raise AuthRefreshConfigError(f"unknown site {site!r}")
    user_var, pass_var = _ACCOUNT_ENV_KEYS[site]
    user = os.environ.get(user_var)
    pwd = os.environ.get(pass_var)
    if not user or not pwd:
        raise AuthRefreshConfigError(
            f"VWA credentials for {site!r} missing — set both {user_var} and {pass_var} "
            f"(template in scripts/vwa_env_remote.sh; gitignored by design)"
        )
    return user, pwd


def refresh_site_auth(
    site: str,
    auth_dir: Path,
    *,
    base_urls: dict | None = None,
    benchmark: str = "",
) -> bool:
    """Re-login to *site* and overwrite ``auth_dir/{site}_state.json``.

    Uses a Playwright subprocess (same approach as the watchdog) so that
    the runner's own Playwright instance is not affected.

    Returns True on success, False on any failure (logged as warning).
    Callers needing hard-fail semantics should use ``auth_required_gate()``.
    """
    if site not in _ACCOUNT_ENV_KEYS:
        logger.warning("auth_refresh: unknown site %r", site)
        return False

    auth_dir = Path(auth_dir)
    auth_file = auth_dir / f"{site}_state.json"

    try:
        username, password = _load_account(site)
    except AuthRefreshConfigError as exc:
        logger.error("auth_refresh: %s", exc)
        return False

    if base_urls and site in base_urls:
        base_url = base_urls[site]
    else:
        env_key = _ENV_KEYS.get(site, site.upper())
        base_url = os.environ.get(env_key, _DEFAULT_BASE_URLS.get(site, ""))

    if not base_url:
        logger.warning("auth_refresh: no base_url for site %r", site)
        return False

    login_path = _LOGIN_PATHS[site]

    # Resolve repo root for VWA sys.path injection
    repo_dir = Path(__file__).resolve().parent.parent.parent

    # Resolve host IP for --host-resolver-rules from env. /stress A1.18 P0-2
    # (2026-05-16): default to 127.0.0.1 so committed code doesn't leak any
    # private Tailscale IP; reproducers set VWA_REMOTE_HOST to their host.
    _resolver_ip = os.environ.get("VWA_REMOTE_HOST", "127.0.0.1")
    # BUG-4 fix (2026-05-16, gemini NEW-OOB-3): Chromium --host-resolver-rules
    # MAP syntax requires IP, not hostname. Literal "localhost" causes silent
    # 30s DNS hang in resolver-rules pipeline → Playwright page.goto timeout.
    # Phase 1a 0pp impact, Phase 1b shop 100pp auth-refresh dead.
    if _resolver_ip == "localhost":
        _resolver_ip = "127.0.0.1"

    # B-212 fix (2026-05-16): LOGIN_FAILED detection — replace URL substring match
    # with urlparse-based path-equal + login_qs subset. Pre-fix substring match
    # systematically false-positived on shopping_admin (every "/admin/*" post-login
    # URL contained "/admin" marker substring). For classifieds with query-bearing
    # login (e.g. /index.php?page=login), require post-login query to also retain
    # the login key for "still on login" to be true. See A1.5 §5 explanation.
    script = f"""
import sys, time
from urllib.parse import urlparse, parse_qs
sys.path.insert(0, {str(repo_dir / 'external' / 'visualwebarena')!r})
from playwright.sync_api import sync_playwright
cm = sync_playwright()
pw = cm.__enter__()
browser = pw.chromium.launch(headless=True, args=['--host-resolver-rules=MAP metis.lti.cs.cmu.edu {_resolver_ip}'])
ctx = browser.new_context()
page = ctx.new_page()
page.goto({(base_url + login_path)!r})
site = {site!r}
if site == 'classifieds':
    page.locator('#email').fill({username!r})
    page.locator('#password').fill({password!r})
    page.get_by_role('button', name='Log in').click()
elif site == 'reddit':
    page.get_by_label('Username').fill({username!r})
    page.get_by_label('Password').fill({password!r})
    page.get_by_role('button', name='Log in').click()
elif site == 'shopping':
    page.get_by_label('Email', exact=True).fill({username!r})
    page.get_by_label('Password', exact=True).fill({password!r})
    page.get_by_role('button', name='Sign In').click()
elif site == 'shopping_admin':
    page.locator('#username').fill({username!r})
    page.locator('#login').fill({password!r})
    page.get_by_role('button', name='Sign in').click()
time.sleep(2)
# B-212 (2026-05-16): structured login-success check via urlparse.
# Path-equal + login-query subset = "still on login" gate.
# Examples (verified A1.5 §5):
#   classifieds login=/index.php?page=login; user-page=/index.php?page=user → path same,
#     query 'page' differs → NOT still on login → success
#   shopping_admin login=/admin; dashboard=/admin/dashboard/index → path differs (after rstrip)
#     → NOT still on login → success (pre-fix substring approach false-positived 100% of time)
#   reddit login=/login; redirected post-login=/f/foo → path differs → success
final_url = page.url
_parsed_final = urlparse(final_url)
_parsed_login = urlparse({(base_url + login_path)!r})
_final_qs = parse_qs(_parsed_final.query)
_login_qs = parse_qs(_parsed_login.query)
_final_path = _parsed_final.path.rstrip('/').lower()
_login_path_norm = _parsed_login.path.rstrip('/').lower()
_still_on_login = (
    _final_path == _login_path_norm
    and all(_final_qs.get(_k) == _v for _k, _v in _login_qs.items())
)
if _still_on_login:
    cm.__exit__(None, None, None)
    print('LOGIN_FAILED ->', final_url)
    sys.exit(2)  # distinct exit code so caller knows it's a login failure
ctx.storage_state(path={str(auth_file)!r})
cm.__exit__(None, None, None)
print('ok ->', final_url)
"""

    # Infer dataset for DATASET env var
    if benchmark == "webarena" or site == "shopping_admin":
        dataset = "webarena"
    else:
        dataset = "visualwebarena"

    # B-214 (2026-05-16, deferred): minimal env propagation would drop LLM API
    # keys + GITHUB_TOKEN from subprocess. Currently kept on for backwards-
    # compatibility with Playwright env (HOME, PATH, DISPLAY, etc.). Future
    # tightening: filter to {PATH, DATASET, VWA_REMOTE_HOST, HOME, USER,
    # DISPLAY, PLAYWRIGHT_*}.
    env = {**os.environ, "DATASET": dataset}

    # B-217 (2026-05-16, deferred): timeout override via AUTH_REFRESH_TIMEOUT.
    _timeout = int(os.environ.get("AUTH_REFRESH_TIMEOUT", "30"))

    try:
        r = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=_timeout, env=env,
        )
        if r.returncode == 0 and auth_file.exists():
            logger.info("auth_refresh: %s refreshed: %s", site, r.stdout.strip())
            return True
        # rc=2 → script detected login failure (still on login page, no
        # storage_state written). Distinguish for clearer logging.
        if r.returncode == 2:
            logger.warning(
                "auth_refresh: %s LOGIN VERIFICATION FAILED (still on login page) — "
                "credentials wrong, site down, or page structure changed: %s",
                site, r.stdout.strip(),
            )
            return False
        logger.warning(
            "auth_refresh: %s failed rc=%d: %s",
            site, r.returncode, r.stderr[-300:],
        )
        return False
    except Exception as exc:
        logger.warning("auth_refresh: %s error: %s", site, exc)
        return False


def auth_required_gate(
    site: str,
    auth_dir: Path,
    *,
    base_urls: dict | None = None,
    benchmark: str = "",
    retry_count: int = 1,
    retry_sleep_s: float = 2.0,
) -> None:
    """Hard-fail variant of refresh_site_auth: raise AuthRefreshFailure on persistent failure.

    Use at paper-grade gate points where stale session contamination is unacceptable:
    - queue post-reset launch (queue_baseline.sh) — first task starting clean
    - runner pre-episode (runner/main.py) — periodic mid-condition refresh
    - watchdog reactive launch (experiment_watchdog.py) — after session-loss-streak

    B-220 / B-221 / B-224 (2026-05-16): closes the three-layer fail-cascade where
    auth refresh failure was silently logged-warning + continue, leading to
    NOT-LOGGED-IN episodes contaminating condition_summary_v2.json.

    Args:
      site, auth_dir, base_urls, benchmark — same as refresh_site_auth
      retry_count — additional retries after first attempt (default 1 → 2 total attempts)
      retry_sleep_s — sleep between attempts (default 2.0s)

    Raises:
      AuthRefreshFailure — wraps the last error after exhausting retries
      AuthRefreshConfigError — propagated immediately (no retry on config error)
    """
    if site not in _ACCOUNT_ENV_KEYS:
        raise AuthRefreshConfigError(f"unknown site {site!r}")

    # Validate creds early — config errors don't benefit from retry
    _load_account(site)  # raises AuthRefreshConfigError on missing env

    attempts = retry_count + 1
    last_error: str | None = None
    for i in range(attempts):
        try:
            ok = refresh_site_auth(
                site, auth_dir, base_urls=base_urls, benchmark=benchmark
            )
            if ok:
                if i > 0:
                    logger.info(
                        "auth_required_gate(%s) succeeded on attempt %d/%d after retry",
                        site, i + 1, attempts,
                    )
                return
            last_error = f"refresh_site_auth returned False (attempt {i + 1}/{attempts})"
        except Exception as exc:
            last_error = f"{exc} (attempt {i + 1}/{attempts})"
        if i < attempts - 1:
            time.sleep(retry_sleep_s)

    raise AuthRefreshFailure(
        f"auth_required_gate({site!r}) FAILED after {attempts} attempts: {last_error}. "
        f"NOT proceeding — paper-grade contamination prevented. "
        f"Check VWA_REMOTE_HOST env, .auth/ writable, site reachability, "
        f"and VWA_{site.upper()}_USER / VWA_{site.upper()}_PASS env vars."
    )


def should_refresh(
    site: str,
    episodes_since_refresh: int,
    cfg: dict,
    *,
    seconds_since_refresh: float | None = None,
) -> bool:
    """Return True if auth should be refreshed for *site* based on config.

    B-35 fix (笔记 §116.9, Tier 7 audit 2026-04-30):
    Original logic was episode-count only. PHP session.gc_maxlifetime=1440s on
    cls/shopping (§39 / B-49b) means long episodes (max_step=30 × 60s/step) can
    expire session mid-episode before episode count threshold is crossed.

    Now: refresh fires if EITHER condition met:
      (a) episodes_since_refresh >= interval (existing)
      (b) seconds_since_refresh >= time_interval (new — default 1200s, below 1440s)

    `seconds_since_refresh` is optional; callers that don't pass it get original
    episode-count-only behavior (backward compat for tests / older callsites).
    """
    auth_cfg = cfg.get("auth_refresh", {})
    if not auth_cfg.get("enabled", False):
        return False
    allowed_sites = auth_cfg.get("sites", [])
    if site not in allowed_sites:
        return False
    interval = int(auth_cfg.get("interval", 5))
    if episodes_since_refresh >= interval:
        return True
    # Time-based check (B-35 fix)
    time_interval = float(auth_cfg.get("time_interval_seconds", 1200))
    if seconds_since_refresh is not None and seconds_since_refresh >= time_interval:
        return True
    return False
