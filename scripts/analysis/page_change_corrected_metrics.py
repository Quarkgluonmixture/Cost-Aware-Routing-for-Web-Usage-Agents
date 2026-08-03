#!/usr/bin/env python3
"""Recompute page_changed-derived metrics under a corrected definition — analysis layer only.

WHY THIS EXISTS (B-1926, 2026-08-03 数据质量审计)
-------------------------------------------------
`state_digest`-based `page_changed` fires on a `content_changed` reason whenever the
visible-text SequenceMatcher ratio drops below `similarity_threshold = 0.95`. Pages
carrying several *volatile* fragments (relative timestamps such as "3 minutes ago")
therefore report `page_changed = True` on every render even when nothing the agent did
had any effect — WA task 651 抖 4.7% 的 text_length 却把 similarity 压到 0.42.

Two consequences, both measured:
  * `content_changed` ∈ `AGENT_VISIBLE_REASONS`, so `agent_visible_changed` is顶起 too →
    the existing `visibility_gap_rate` metric CANNOT see this class of false positive.
  * `router.py` 的 `unchanged_streak` 由 `prev_page_changed` 驱动 → the escalation
    trigger never accumulates on those episodes (`trigger_distribution` 全空).

The fix at source would live in `p79/experiment/state_change.py`, which is INSIDE the
fire import path and feeds an estimand-adjacent field. Per
`feedback_analysis_layer_fire_immutability_and_witness` that needs a witness, and it
would split Pass-1 / Pass-2 into two measurement conventions. User decision 2026-08-03:
**correct in the analysis layer instead** — already-collected data stays byte-stable and
reproducible, while the corrected numbers are derived on read.

CORRECTED DEFINITION
--------------------
A step counts as `page_changed` only if it ALSO shows a substantive change:

    page_changed AND NOT (url unchanged AND scroll_y unchanged AND text_similarity > 0.95)

i.e. a step whose URL and scroll offset are identical and whose visible text is >95%
similar is treated as *unchanged* regardless of which cosmetic reason fired.

Usage:
    .venv/bin/python3 scripts/analysis/page_change_corrected_metrics.py \
        --out docs/analysis/cross_sites/page_change_corrected.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCAN_DIRS = (("v11_vwa", "visualwebarena"), ("v11_wa", "webarena"))
SIM_FLOOR = 0.95


def _is_cosmetic(rec: dict) -> bool:
    """True when page_changed=True is not backed by url / scroll / text movement."""
    if rec.get("page_changed") is not True:
        return False
    sd = rec.get("state_digest") or {}
    if sd.get("url_before") != sd.get("url_after"):
        return False
    if sd.get("scroll_y_before") != sd.get("scroll_y_after"):
        return False
    ts = rec.get("text_similarity")
    return isinstance(ts, (int, float)) and ts > SIM_FLOOR


def _conditions():
    for scan_dir, bench_root in SCAN_DIRS:
        for sf in sorted(glob.glob(str(REPO / "results" / "diag_scans" / scan_dir / "*.json"))):
            tag = os.path.basename(sf)[:-5]
            run_id = json.load(open(sf)).get("run_id")
            runs = glob.glob(str(REPO / "results" / bench_root / "phase1" / run_id))
            if not runs:
                continue
            conds = [
                p for p in glob.glob(runs[0] + "/*")
                if os.path.isdir(p) and os.path.basename(p).startswith("phase1")
            ]
            if conds:
                yield tag, conds[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    per_cond = {}
    for tag, cond_dir in _conditions():
        n_steps = n_changed = n_cosmetic = 0
        n_nochange_obs = n_nochange_corr = 0
        n_inert_obs = n_inert_corr = 0
        ep_streak_obs = ep_streak_corr = n_ep = 0

        for f in sorted(glob.glob(cond_dir + "/episodes/*_steps_v2.jsonl")):
            rows = []
            for line in open(f, errors="ignore"):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
            if not rows:
                continue
            n_ep += 1
            streak_obs = streak_corr = mx_obs = mx_corr = 0
            for r in rows:
                n_steps += 1
                pc = r.get("page_changed")
                cosmetic = _is_cosmetic(r)
                pc_corr = False if cosmetic else pc
                if pc is True:
                    n_changed += 1
                if cosmetic:
                    n_cosmetic += 1
                if pc is False:
                    n_nochange_obs += 1
                if pc_corr is False:
                    n_nochange_corr += 1
                if pc is False and r.get("action_success") is True:
                    n_inert_obs += 1
                if pc_corr is False and r.get("action_success") is True:
                    n_inert_corr += 1
                # router unchanged_streak (router.py:80-83)
                streak_obs = streak_obs + 1 if pc is False else (0 if pc is True else streak_obs)
                streak_corr = streak_corr + 1 if pc_corr is False else (0 if pc_corr is True else streak_corr)
                mx_obs = max(mx_obs, streak_obs)
                mx_corr = max(mx_corr, streak_corr)
            ep_streak_obs += mx_obs >= 2
            ep_streak_corr += mx_corr >= 2

        per_cond[tag] = {
            "steps": n_steps,
            "episodes": n_ep,
            "page_changed_steps": n_changed,
            "cosmetic_false_positive_steps": n_cosmetic,
            "cosmetic_rate_of_changed": round(n_cosmetic / n_changed, 6) if n_changed else None,
            "no_change_rate_observed": round(n_nochange_obs / n_steps, 6) if n_steps else None,
            "no_change_rate_corrected": round(n_nochange_corr / n_steps, 6) if n_steps else None,
            "noop_inert_rate_observed": round(n_inert_obs / n_steps, 6) if n_steps else None,
            "noop_inert_rate_corrected": round(n_inert_corr / n_steps, 6) if n_steps else None,
            "router_streak2_episodes_observed": ep_streak_obs,
            "router_streak2_episodes_corrected": ep_streak_corr,
        }

    # per-mode rollup + rank stability of the headline Micro claim
    by_mode = defaultdict(lambda: [0, 0, 0, 0])  # mode -> [nochange_obs, nochange_corr, steps, cosmetic]
    for tag, v in per_cond.items():
        mode = "_".join(tag.split("_")[1:-1]) if not tag.endswith("wa_reddit") \
            else "_".join(tag.split("_")[1:-2])
        acc = by_mode[mode]
        acc[0] += (v["no_change_rate_observed"] or 0) * v["steps"]
        acc[1] += (v["no_change_rate_corrected"] or 0) * v["steps"]
        acc[2] += v["steps"]
        acc[3] += v["cosmetic_false_positive_steps"]

    modes = {m: {
        "no_change_rate_observed": round(a[0] / a[2], 6) if a[2] else None,
        "no_change_rate_corrected": round(a[1] / a[2], 6) if a[2] else None,
        "cosmetic_fp_steps": a[3],
        "steps": a[2],
    } for m, a in by_mode.items()}

    tot_obs = sum(v["router_streak2_episodes_observed"] for v in per_cond.values())
    tot_corr = sum(v["router_streak2_episodes_corrected"] for v in per_cond.values())

    payload = {
        "schema_version": "1.0",
        "definition": (
            "page_changed AND NOT (url unchanged AND scroll_y unchanged AND "
            f"text_similarity > {SIM_FLOOR})"
        ),
        "source_note": "analysis-layer correction for B-1926; runner untouched (fire-path immutability)",
        "conditions": per_cond,
        "per_mode": modes,
        "router_streak2_total_observed": tot_obs,
        "router_streak2_total_corrected": tot_corr,
    }

    print(f"{'mode':<16} {'no_change 实测':>14} {'修正':>10} {'cosmetic FP step':>18}")
    for m, v in sorted(modes.items(), key=lambda kv: -(kv[1]["no_change_rate_corrected"] or 0)):
        print(f"{m:<16} {100*(v['no_change_rate_observed'] or 0):>13.2f}% "
              f"{100*(v['no_change_rate_corrected'] or 0):>9.2f}% {v['cosmetic_fp_steps']:>18}")
    print(f"\nrouter streak>=2 episode: 实测 {tot_obs} → 修正 {tot_corr} "
          f"(+{tot_corr - tot_obs}, {100*(tot_corr-tot_obs)/tot_obs:.1f}%)")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n✓ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
