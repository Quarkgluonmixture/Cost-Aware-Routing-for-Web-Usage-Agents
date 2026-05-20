#!/usr/bin/env python3
"""P2-1 (/stress GRL audit 2026-05-20): one-time verification of the C1 eval-context
classification (`VwaEvaluator._classify_eval_context`).

The Fire-6 C1 eval isolation (B-1768) routes program_html-with-explicit-URL tasks
to a fresh isolated `page.context.new_page()` for evaluation instead of the agent's
stateful runner page. The original claim "232/234 cls + ~136 red program_html tasks
qualify" was **inspected, not empirically enumerated** (P2-1) — and this script's
first run (2026-05-20, B-1783) proved it an ~8× / ~2× OVERCOUNT: cls has only 31
program_html tasks (29 isolate), the bulk being url_match (131) / string_match (78);
red isolates 71. The docstring + this baseline are now corrected. The script
classifies every per-task config on disk and verifies the safety invariant.

Static (no browser / no agent trajectory needed) — run before a paper-grade fire:

    python scripts/maintenance/verify_eval_context_classification.py --site classifieds
    python scripts/maintenance/verify_eval_context_classification.py --site reddit

Checks per task:
  1. classify → one of {isolated_program_html_context, agent_page, no_browser_required}.
  2. SAFETY INVARIANT for isolated_program_html_context: EVERY program_html target
     must be an explicit URL (no `last` / `__last_url__` / `func`-url) AND there is
     no url_match in eval_types — i.e. the isolated fresh-page eval is semantically
     identical to the agent-page eval (agent DOM is discarded by the goto). A task
     classified isolated that violates this would silently produce a WRONG eval
     result → wrong SR. (The classifier itself enforces this; re-deriving here is
     the independent cross-check.)

Exit 0 = all tasks classify cleanly + every isolated task satisfies the invariant.
Exit 1 = any classify error OR any isolated task violates the safety invariant.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# EMPIRICALLY VERIFIED baseline (B-1783, this script's 2026-05-20 first run) —
# the script reports observed-vs-verified so future drift (e.g. a task-pool
# regen) is visible. NOTE: this REPLACES the original B-1768 docstring figure
# ("232/234 cls + ~136 red") which this very script proved was an ~8× / ~2×
# overcount (there are only 31 cls program_html tasks total; the bulk are
# url_match / string_match). See environment.py:_classify_eval_context.
_CLAIMED = {
    "classifieds": {"isolated_program_html_context": 29, "total": 234},
    "reddit": {"isolated_program_html_context": 71, "total": 210},
}


def _default_config_dir(site: str) -> Path:
    return REPO_ROOT / "external" / "visualwebarena" / "config_files" / "vwa" / f"test_{site}"


def _independent_safety_check(config_file: Path) -> Tuple[bool, str]:
    """Re-derive the isolation safety invariant from the raw config, INDEPENDENT of
    `_classify_eval_context`, so a classifier bug cannot hide behind itself.

    Returns (is_isolation_safe, reason). is_isolation_safe=True means: program_html
    present, every target an explicit URL (no last/__last_url__/func), no url_match.
    """
    try:
        cfg = json.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"config unreadable: {exc}"
    eval_block = cfg.get("eval", {}) or {}
    eval_types = eval_block.get("eval_types", []) or []
    if "program_html" not in eval_types:
        return False, "no program_html"
    if "url_match" in eval_types:
        return False, "url_match alongside program_html (needs agent page.url)"
    targets = eval_block.get("program_html", []) or []
    for t in targets:
        url = str(t.get("url", "") or "")
        if url == "last" or "__last_url__" in url or url.startswith("func"):
            return False, f"agent-page-dependent target url={url!r}"
    return True, "ok (all explicit-URL targets, no url_match)"


def verify_site(site: str, config_dir: Optional[Path] = None) -> int:
    from p79.experiment.environment import VwaEvaluator

    config_dir = config_dir or _default_config_dir(site)
    if not config_dir.is_dir():
        print(f"[verify] FATAL config dir not found: {config_dir}", file=sys.stderr)
        return 1

    task_files = sorted(
        config_dir.glob("*.json"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else 1_000_000,
    )
    # Skip non-task aux files (e.g. *.raw.json) — only pure-int stems are tasks.
    task_files = [p for p in task_files if p.stem.isdigit()]
    if not task_files:
        print(f"[verify] FATAL no per-task <id>.json configs in {config_dir}", file=sys.stderr)
        return 1

    mode_counts: Counter = Counter()
    violations: List[str] = []
    classify_errors: List[str] = []

    for tf in task_files:
        try:
            mode, _target = VwaEvaluator._classify_eval_context(str(tf))  # noqa: SLF001
        except Exception as exc:  # pragma: no cover - defensive
            classify_errors.append(f"{tf.name}: classify raised {type(exc).__name__}: {exc}")
            continue
        mode_counts[mode] += 1
        # Cross-check: a task classified isolated MUST pass the independent safety
        # invariant; otherwise the fresh-page eval could differ from agent-page eval.
        if mode == "isolated_program_html_context":
            safe, reason = _independent_safety_check(tf)
            if not safe:
                violations.append(f"{tf.name}: classified isolated but UNSAFE — {reason}")

    total = sum(mode_counts.values())
    print(f"[verify] site={site} config_dir={config_dir}")
    print(f"[verify] {total} tasks classified:")
    for mode in ("isolated_program_html_context", "agent_page", "no_browser_required"):
        print(f"    {mode}: {mode_counts.get(mode, 0)}")

    claimed = _CLAIMED.get(site)
    if claimed:
        obs_iso = mode_counts.get("isolated_program_html_context", 0)
        clm_iso = claimed.get("isolated_program_html_context")
        tag = "≈" if claimed.get("approx") else "="
        match = "OK" if (claimed.get("approx") or obs_iso == clm_iso) else "DRIFT"
        print(f"[verify] isolated observed={obs_iso} {tag} claimed={clm_iso} [{match}]")
        if claimed.get("total") and total != claimed["total"]:
            print(f"[verify] NOTE total observed={total} != claimed {claimed['total']}")

    ok = not violations and not classify_errors
    if classify_errors:
        print(f"[verify] {len(classify_errors)} CLASSIFY ERROR(S):", file=sys.stderr)
        for e in classify_errors:
            print(f"    ✗ {e}", file=sys.stderr)
    if violations:
        print(f"[verify] {len(violations)} ISOLATION SAFETY VIOLATION(S) "
              f"(isolated task whose fresh-page eval may differ from agent-page):", file=sys.stderr)
        for v in violations:
            print(f"    ✗ {v}", file=sys.stderr)
    print(f"[verify] {'PASS' if ok else 'FAIL'} — "
          f"{len(violations)} safety violation(s), {len(classify_errors)} classify error(s).")
    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="P2-1 verify C1 eval-context classification (static)")
    p.add_argument("--site", required=True, choices=["classifieds", "reddit", "shopping"])
    p.add_argument("--config-dir", default=None,
                   help="override per-task config dir (default: "
                        "external/visualwebarena/config_files/vwa/test_<site>/)")
    args = p.parse_args(argv)
    return verify_site(args.site, Path(args.config_dir) if args.config_dir else None)


if __name__ == "__main__":
    raise SystemExit(main())
