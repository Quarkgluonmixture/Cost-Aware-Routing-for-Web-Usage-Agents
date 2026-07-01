#!/usr/bin/env python3
"""Bound-but-unpromoted cell watch (队列⑧, 2026-07-02).

GAP (§360.3 root cause): A100 fire-side `validate_fire_manifest --apply` binds
conditions into fire_manifest.json, but aggregators read run_manifest.yaml
`cells:` — and promotion is a MANUAL step. The episode-data sync cron never
looked at binding state, so B1/B2 cls sat bound-but-unaggregated for 3 weeks.

This checker diffs a staged A100 fire_manifest copy against the registry's
paper-grade cells and ntfy-alerts when the un-promoted set CHANGES (state-file
dedup — no 15-min spam). It never edits either manifest: promotion stays a
deliberate operator step (NUMBERS_TODO §0 配方).

Cron usage (wired into sync_a100_results.sh):
  check_manifest_promotion_gap.py --fire logs/cron/fire_manifest_a100_latest.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "results/phantom_paper/run_manifest.yaml"
STATE = ROOT / "logs/cron/promotion_gap_state.json"
NTFY_TOPIC = "p79-claude"

# fire_manifest lowercase op-mode → registry display mode
MODE_MAP = {
    "dom": "DOM",
    "som": "SoM",
    "vision": "Vision",
    "phantom_text": "P-text",
    "phantom_som": "P-SoM",
    "phantom_prompt": "P-prompt",
}


def bound_cells(fire_path: Path) -> set[tuple[str, str, str]]:
    data = json.loads(fire_path.read_text())
    out = set()
    for key in data.get("conditions", {}):
        site, baseline, mode = key.split("|", 2)
        out.add((baseline, site, MODE_MAP.get(mode, mode)))
    return out


def promoted_cells(registry_path: Path) -> set[tuple[str, str, str]]:
    data = yaml.safe_load(registry_path.read_text()) or {}
    out = set()
    for cell in data.get("cells", []) or []:
        if cell.get("grade") == "paper-grade":
            out.add((cell.get("baseline"), cell.get("site"), cell.get("mode")))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fire", type=Path, required=True, help="staged A100 fire_manifest.json copy")
    ap.add_argument("--registry", type=Path, default=REGISTRY)
    ap.add_argument("--state", type=Path, default=STATE)
    ap.add_argument("--no-ntfy", action="store_true")
    args = ap.parse_args()

    if not args.fire.exists():
        print(f"[promotion-gap] staged fire manifest missing: {args.fire} (skip)")
        return 0

    gap = sorted(bound_cells(args.fire) - promoted_cells(args.registry))
    gap_strs = [f"{b}·{s}·{m}" for b, s, m in gap]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    print(f"[promotion-gap] {now} bound-not-promoted: {len(gap_strs)}"
          + (f" → {', '.join(gap_strs)}" if gap_strs else ""))

    prev: list[str] = []
    if args.state.exists():
        try:
            prev = json.loads(args.state.read_text()).get("gap", [])
        except Exception:
            prev = []
    args.state.parent.mkdir(parents=True, exist_ok=True)
    args.state.write_text(json.dumps({"ts": now, "gap": gap_strs}, indent=2))

    if gap_strs and args.no_ntfy:
        print("[promotion-gap] --no-ntfy: alert suppressed"
              + (" (gap set CHANGED)" if gap_strs != prev else " (gap unchanged)"))
    elif gap_strs and gap_strs != prev:
        msg = (f"[{now}] {len(gap_strs)} cell(s) bound on A100 but NOT promoted into "
               f"run_manifest.yaml → 不进聚合. 走 NUMBERS_TODO §0 sync 配方: "
               + ", ".join(gap_strs))
        subprocess.run(
            ["curl", "-s", "--max-time", "3",
             "-H", "Title: P79 registry promotion gap",
             "-H", "Priority: high",
             "-d", msg,
             f"https://ntfy.sh/{NTFY_TOPIC}"],
            check=False, capture_output=True,
        )
        print("[promotion-gap] ntfy sent (gap set changed)")
    elif gap_strs:
        print("[promotion-gap] gap unchanged since last tick — no ntfy (dedup)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
