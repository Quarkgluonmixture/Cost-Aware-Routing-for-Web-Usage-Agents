#!/usr/bin/env python3
"""Phase 1a launch gate — parse cls+reddit raw.json, derive endpoints, verify all reachable.

BUG-1 fix (2026-05-16, codex CodexOnly-1): Phase 1a cls+red task configs
embed cross-site URLs (cls task 224 → __SHOPPING__, reddit task 45 →
__WIKIPEDIA__). 21 cls + 33 red = 54/444 cross-link tasks. If shop or
wiki endpoint unreachable on A100, these tasks silently 404 → counted
as agent-fail by P79 runner (no infra-error tagging) → 12.2pp SR drift
vs quark archive.

This gate parses task configs, collects all `__SITE__` placeholders after
P79 env substitution, curls each, refuses launch if any returns non-2xx/3xx.
Mirrors P79 tasks.py:_placeholder_mapping() logic.

Usage:
    python3 scripts/maintenance/phase1a_launch_gate.py
    # exit 0 = all endpoints reachable, safe to fire Phase 1a
    # exit 2 = some endpoint missing, refuse fire with itemized report

Run automatically by queue_phase1_paper_grade.sh before launch.
"""
from __future__ import annotations
import json, os, re, subprocess, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_CFG = {
    "classifieds": REPO_ROOT / "external/visualwebarena/config_files/vwa/test_classifieds.raw.json",
    "reddit":      REPO_ROOT / "external/visualwebarena/config_files/vwa/test_reddit.raw.json",
}
# Mirror p79/experiment/tasks.py:PLACEHOLDER_DEFAULTS — env override is the source of truth
DEFAULTS = {
    "__CLASSIFIEDS__":    "http://localhost:9980",
    "__REDDIT__":         "http://localhost:9999",
    "__SHOPPING__":       "http://localhost:7770",
    "__SHOPPING_ADMIN__": "http://localhost:7780/admin",
    "__WIKIPEDIA__":      "http://localhost:8888",
    "__HOMEPAGE__":       "http://localhost:4399",
    "__GITLAB__":         "http://localhost:8023",
    "__MAP__":            "http://localhost:3000",
}
ENV_KEYS = {
    "__CLASSIFIEDS__":    "CLASSIFIEDS",
    "__REDDIT__":         "REDDIT",
    "__SHOPPING__":       "SHOPPING",
    "__SHOPPING_ADMIN__": "SHOPPING_ADMIN",
    "__WIKIPEDIA__":      "WIKIPEDIA",
    "__HOMEPAGE__":       "HOMEPAGE",
    "__GITLAB__":         "GITLAB",
    "__MAP__":            "MAP",
}

def resolve_url(placeholder: str) -> str:
    env_key = ENV_KEYS.get(placeholder)
    if env_key and env_key in os.environ:
        return os.environ[env_key]
    return DEFAULTS[placeholder]

def collect_placeholders(raw_path: Path) -> set[str]:
    """Find all __XXX__ tokens referenced in task config."""
    content = raw_path.read_text(encoding="utf-8")
    return set(re.findall(r"__[A-Z_]+__", content))

def check_endpoint(url: str, timeout: int = 5) -> tuple[bool, str]:
    """Returns (ok, status_string). 2xx/3xx = ok, else fail."""
    # Strip trailing path for root probe (e.g. __SHOPPING_ADMIN__/admin → check :7780/)
    base = url.split("/admin", 1)[0]
    base = base.rstrip("/")
    try:
        result = subprocess.run(
            ["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}",
             "--max-time", str(timeout), "-I", base],
            capture_output=True, text=True, timeout=timeout + 2
        )
        code = result.stdout.strip()
        ok = code.startswith("2") or code.startswith("3")
        return ok, f"HTTP {code}"
    except (subprocess.TimeoutExpired, Exception) as e:
        return False, f"TIMEOUT/{type(e).__name__}"

def main() -> int:
    print("=== Phase 1a Launch Gate ===")
    print("Phase 1a scope: cls + reddit. But raw.json cross-link task subsets")
    print("(cls→shop, reddit→wiki) require shop + wiki endpoints reachable too.")
    print()

    # Collect referenced placeholders from cls + reddit task configs
    all_refs: dict[str, list[str]] = {}
    for site, raw_path in RAW_CFG.items():
        if not raw_path.exists():
            print(f"✗ FATAL: task config missing: {raw_path}", file=sys.stderr)
            return 2
        placeholders = collect_placeholders(raw_path)
        # Filter to site-level placeholders (skip __WIKIPEDIA_ZIM_VERSION__ etc not in DEFAULTS)
        placeholders = {p for p in placeholders if p in DEFAULTS}
        all_refs[site] = sorted(placeholders)
        print(f"  {site}: {len(placeholders)} placeholder(s) → {sorted(placeholders)}")

    # Union of referenced endpoints
    needed = set()
    for site, refs in all_refs.items():
        needed.update(refs)

    print()
    print("=== Endpoint reachability check ===")
    failures = []
    for placeholder in sorted(needed):
        url = resolve_url(placeholder)
        ok, status = check_endpoint(url)
        marker = "✓" if ok else "✗"
        print(f"  {marker} {placeholder} = {url}  ({status})")
        if not ok:
            failures.append((placeholder, url, status))

    print()
    if failures:
        print(f"✗ FATAL: {len(failures)} endpoint(s) unreachable; refusing Phase 1a launch", file=sys.stderr)
        for ph, url, status in failures:
            print(f"   - {ph}: {url} ({status})", file=sys.stderr)
        print()
        print("Affected task subset (codex CodexOnly-1):", file=sys.stderr)
        print("  cls task 224 + 20 more reference __SHOPPING__", file=sys.stderr)
        print("  reddit task 45 + 32 more reference __WIKIPEDIA__", file=sys.stderr)
        print("Up to 12.2pp silent SR drift vs quark archive if launched.", file=sys.stderr)
        return 2

    print(f"✓ All {len(needed)} required endpoints reachable. Phase 1a fire SAFE.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
