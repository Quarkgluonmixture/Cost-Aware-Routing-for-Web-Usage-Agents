"""Shared auth refresh logic for Magento (shopping/shopping_admin) sites.

Extracted from scripts/maintenance/experiment_watchdog.py so that both the watchdog
and the experiment runner can reuse the same login routine.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_ACCOUNTS = {
    "classifieds":    ("blake.sullivan@gmail.com", "Password.123"),
    "reddit":         ("MarvelsGrantMan136",       "test1234"),
    "shopping":       ("emma.lopez@gmail.com",     "Password.123"),
    "shopping_admin": ("admin",                    "admin1234"),
}

_DEFAULT_BASE_URLS = {
    "classifieds":    "http://100.95.81.103:9980",
    "reddit":         "http://100.95.81.103:9999",
    "shopping":       "http://100.95.81.103:7770",
    "shopping_admin": "http://100.95.81.103:7780",
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
    """
    if site not in _ACCOUNTS:
        logger.warning("auth_refresh: unknown site %r", site)
        return False

    auth_dir = Path(auth_dir)
    auth_file = auth_dir / f"{site}_state.json"
    username, password = _ACCOUNTS[site]

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

    # Resolve host IP for --host-resolver-rules from env (default = legacy
    # value). Lets users with different Tailscale IPs override without code edit.
    _resolver_ip = os.environ.get("VWA_REMOTE_HOST", "100.95.81.103")
    # BUG-4 fix (2026-05-16, gemini NEW-OOB-3): Chromium --host-resolver-rules
    # MAP syntax requires IP, not hostname. Literal "localhost" causes silent
    # 30s DNS hang in resolver-rules pipeline → Playwright page.goto timeout.
    # Phase 1a 0pp impact, Phase 1b shop 100pp auth-refresh dead.
    if _resolver_ip == "localhost":
        _resolver_ip = "127.0.0.1"

    script = f"""
import sys, time
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
# Verify login actually succeeded BEFORE writing storage_state.
# Was: storage_state written unconditionally → empty/stale cookies on failed
# login → caller (watchdog) believed auth was refreshed but next episode
# still NOT-LOGGED-IN. Heuristic: post-login URL no longer on login page.
final_url = page.url
# Bug fix (2026-04-26): previous code did `.split('?')[0]` which collapsed
# `/index.php?page=login` to `/index.php` for OSClass classifieds — this matches
# ALL OSClass pages (including the post-login dashboard at
# /index.php?page=user&action=items), causing every successful login to be
# misclassified as LOGIN_FAILED. Fix: keep the full login path (with query)
# when checking for "still on login page".
login_marker = {login_path!r}.lower().rstrip('/')
still_on_login = bool(login_marker) and login_marker in final_url.lower()
if still_on_login:
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
    env = {**os.environ, "DATASET": dataset}

    try:
        r = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=30, env=env,
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
