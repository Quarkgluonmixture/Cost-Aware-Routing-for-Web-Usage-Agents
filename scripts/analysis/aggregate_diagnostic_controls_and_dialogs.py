#!/usr/bin/env python3
"""Per (site, model, mode) aggregator for diagnostic-control fires +
dialog-meta misclick blast-radius — paper §3 + §3.5.1 evidence layer.

B-555 (/stress A1.5 P1-5-ABC* 3-AI overlap consumer-wiring closure,
2026-05-17): A1.5b Phase 2 B-546 + B-547 wrote `control_intervention`
+ `dialog_meta_primary` + `runtime_sleep_primary` fields into step JSONL
but NO downstream aggregator consumed them.

- B-546 write site: `runner/main.py:2602` writes
  `step_record["control_intervention"]` when diagnostic_controls fire.
  Phase 1 B-497 declared the schema; Phase 2 wired the writer; this
  script wires the reader.
- B-547 write site: `runner/main.py:2579-2581` writes
  `dialog_meta_primary` (primary-action dialogs only) + `dialog_meta_retry`
  (retry-action dialogs only). Paper §3.5.1 cross-baseline misclick
  blast-radius rate needs primary-only view to avoid B0-vs-B1/B2 retry
  differential bias.

Outputs per (site, model, mode):

Diagnostic controls (paper §3 — auditable JSONL-only):
  - n_steps_total
  - n_steps_with_control_fire    (any of 3 controls fired)
  - control_fire_rate_pct
  - fires_by_type {"query_sanitization", "anti_repeat", "no_early_finish"}
  - original_action_distribution (action_type histogram pre-control)

Dialog blast-radius (paper §3.5.1 cross-baseline parity):
  - n_steps_total
  - n_steps_with_dialog_primary  (primary-only, B-547 cross-baseline fair)
  - dialog_rate_pct_primary
  - dialog_type_distribution     (confirm/alert/prompt types)

Usage:
    python3 scripts/analysis/aggregate_diagnostic_controls_and_dialogs.py \\
        --run-dir results/visualwebarena/phase1/<RUN_ID> \\
        --out-md docs/analysis/<run>/diagnostic_controls_and_dialogs.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def _safe_load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield parsed JSON lines, skipping corrupt/empty lines."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue


def _aggregate_cell(condition_dir: Path) -> Dict[str, Any]:
    """Walk every episode step JSONL in this condition and tally fires."""
    n_steps_total = 0
    n_with_control = 0
    fires_by_type: Counter = Counter()
    original_action_dist: Counter = Counter()
    n_with_dialog_primary = 0
    dialog_type_dist: Counter = Counter()

    episodes_dir = condition_dir / "episodes"
    if not episodes_dir.is_dir():
        return {"available": False, "reason": f"no episodes dir under {condition_dir.name}"}

    for steps_path in sorted(episodes_dir.glob("*_steps_v2.jsonl")):
        for step in _safe_load_jsonl(steps_path):
            n_steps_total += 1
            ci = step.get("control_intervention")
            if isinstance(ci, dict):
                n_with_control += 1
                fires = ci.get("fires") or []
                for f in fires:
                    ft = (f or {}).get("type")
                    if ft:
                        fires_by_type[ft] += 1
                orig = ci.get("original_action") or {}
                at = orig.get("action_type")
                if at:
                    original_action_dist[at] += 1
            # B-547 primary-only dialog (paper §3.5.1 cross-baseline parity)
            dialog_primary = step.get("dialog_meta_primary")
            if dialog_primary:
                n_with_dialog_primary += 1
                # dialog_meta_primary is a list of dialog events
                if isinstance(dialog_primary, list):
                    for evt in dialog_primary:
                        if isinstance(evt, dict):
                            dt = evt.get("type") or evt.get("dialog_type")
                            if dt:
                                dialog_type_dist[dt] += 1

    return {
        "available": True,
        "n_steps_total": n_steps_total,
        "n_steps_with_control_fire": n_with_control,
        "control_fire_rate_pct": round(100.0 * n_with_control / n_steps_total, 4) if n_steps_total else 0.0,
        "fires_by_type": dict(fires_by_type),
        "original_action_distribution": dict(original_action_dist),
        "n_steps_with_dialog_primary": n_with_dialog_primary,
        "dialog_rate_pct_primary": round(100.0 * n_with_dialog_primary / n_steps_total, 4) if n_steps_total else 0.0,
        "dialog_type_distribution": dict(dialog_type_dist),
    }


def _walk_run_dir(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Walk run_dir/<condition>/episodes/*.jsonl and aggregate per condition."""
    out: Dict[str, Dict[str, Any]] = {}
    for condition_dir in sorted(run_dir.iterdir()):
        if not condition_dir.is_dir():
            continue
        if not (condition_dir / "episodes").is_dir():
            continue
        out[condition_dir.name] = _aggregate_cell(condition_dir)
    return out


def _render_markdown(stats: Dict[str, Dict[str, Any]]) -> str:
    """Two tables: diagnostic-control fires + dialog-primary blast radius."""
    lines = ["# Diagnostic Controls + Dialog Blast Radius (B-555)\n"]
    lines.append(
        "Paper §3 disclosure (control_intervention auditable from JSONL) + "
        "§3.5.1 cross-baseline parity (dialog_meta_primary, B-547 retry-overwrite split).\n"
    )

    lines.append("## Diagnostic-Control Fires\n")
    lines.append("| Condition | Steps | Fires | Rate% | query_sanitization | anti_repeat | no_early_finish |")
    lines.append("|---|---|---|---|---|---|---|")
    for cond, s in stats.items():
        if not s.get("available"):
            lines.append(f"| {cond} | – | – | – | – | – | – |")
            continue
        fb = s["fires_by_type"]
        lines.append(
            f"| {cond} | {s['n_steps_total']} | {s['n_steps_with_control_fire']} | "
            f"{s['control_fire_rate_pct']} | {fb.get('query_sanitization', 0)} | "
            f"{fb.get('anti_repeat', 0)} | {fb.get('no_early_finish', 0)} |"
        )

    lines.append("\n## Dialog-Primary Blast Radius (B-547 cross-baseline fair)\n")
    lines.append("| Condition | Steps | DialogPrimary | Rate% | Top dialog types |")
    lines.append("|---|---|---|---|---|")
    for cond, s in stats.items():
        if not s.get("available"):
            lines.append(f"| {cond} | – | – | – | – |")
            continue
        dt = s["dialog_type_distribution"]
        top = ", ".join(f"{k}({v})" for k, v in sorted(dt.items(), key=lambda kv: -kv[1])[:3])
        lines.append(
            f"| {cond} | {s['n_steps_total']} | {s['n_steps_with_dialog_primary']} | "
            f"{s['dialog_rate_pct_primary']} | {top or '–'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, type=Path, help="results/.../<RUN_ID> directory")
    p.add_argument("--out-md", type=Path, default=None, help="optional markdown output path")
    p.add_argument("--out-json", type=Path, default=None, help="optional JSON output path")
    args = p.parse_args()

    if not args.run_dir.is_dir():
        print(f"✗ run-dir not a directory: {args.run_dir}", file=sys.stderr)
        return 2

    stats = _walk_run_dir(args.run_dir)
    md = _render_markdown(stats)

    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md, encoding="utf-8")
        print(f"✓ wrote {args.out_md}")
    else:
        print(md)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"✓ wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
