#!/usr/bin/env python3
"""Rescan every canonical VWA condition at the current RULESET_VERSION.

The /diag skill's discover-then-freeze discipline requires that all digests carry
ONE ruleset_version before any cross-mode or cross-model aggregation. Landing a
rule batch therefore obliges a full rescan, not just a rescan of the site that
motivated the batch — a rule can change behaviour on the other site too (the
2026-07-27 batch's P33 path extension added a genuine cls hit, and the B-1890
P35/P39 fix removed stale cls hits).

Writes one JSON per condition into --out-dir and prints a version-consistency
check plus, when --baseline-dir is given, a per-rule diff against a previous scan.

Usage:
  python3 scripts/analysis/diag_rescan_all.py --out-dir /tmp/diag_v8
  python3 scripts/analysis/diag_rescan_all.py --out-dir /tmp/diag_v8 --sites reddit
  python3 scripts/analysis/diag_rescan_all.py --out-dir /tmp/diag_v8 --baseline-dir /tmp/diag_v7
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PHASE1 = REPO / "results" / "visualwebarena" / "phase1"
SCANNER = REPO / "scripts" / "analysis" / "diag_pattern_match.py"
PY = str(REPO / ".venv" / "bin" / "python3")

# Canonical run per (model, mode, site). Keys are the digest basenames, so the
# output filenames line up 1:1 with docs/analysis/vwa_<site>/<key>_diag_digest.md.
CANONICAL = {
    # ---- reddit (6 modes x 3 models) ----
    "B0_dom_reddit": "B0_dom_reddit_20260625_154833_928747130_2827521_R11344",
    "B0_som_reddit": "B0_som_reddit_20260627_035453_162107997_3024022_R20936",
    "B0_vision_reddit": "B0_vision_reddit_20260628_094255_184327569_3222015_R17559",
    "B0_phantom_text_reddit": "B0_phantom_text_reddit_20260629_140253_060787566_3384189_R32139",
    "B0_phantom_som_reddit": "B0_phantom_som_reddit_20260701_223127_661875492_3649813_R28173",
    "B0_phantom_prompt_reddit": "B0_phantom_prompt_reddit_20260709",
    "B1_dom_reddit": "B1_dom_reddit_20260703",
    "B1_som_reddit": "B1_som_reddit_20260706",
    "B1_vision_reddit": "B1_vision_reddit_20260708_002122_732634080_205180_R16847",
    "B1_phantom_text_reddit": "B1_phantom_text_reddit_20260710",
    "B1_phantom_som_reddit": "B1_phantom_som_reddit_20260711",
    "B1_phantom_prompt_reddit": "B1_phantom_prompt_reddit_20260713",
    "B2_dom_reddit": "B2_dom_reddit_20260715",
    "B2_som_reddit": "B2_som_reddit_20260717",
    "B2_vision_reddit": "B2_vision_reddit_20260719",
    "B2_phantom_text_reddit": "B2_phantom_text_reddit_20260720",
    "B2_phantom_som_reddit": "B2_phantom_som_reddit_20260722",
    "B2_phantom_prompt_reddit": "B2_phantom_prompt_reddit_20260723",
}

MODE_OF = {
    "dom": "dom", "som": "som", "vision": "vision",
    "phantom_text": "phantom_text", "phantom_som": "phantom_som",
    "phantom_prompt": "phantom_prompt",
}


def _discover_cls(baseline_dir=None) -> dict:
    """Resolve cls conditions by globbing — their run dirs carry random suffixes.

    Digest basenames without an explicit R-suffix (e.g. `B0_som_classifieds`) refer
    to whichever run the digest header names; we pick the newest matching dir and
    print it so the mapping is auditable rather than silent.

    ⚠️ B-1927 (2026-08-03): "newest by mtime" is NOT the same as "canonical". A
    replicate run lands in the SAME `phase1/` directory under the SAME
    `experiment.name` with the SAME seed — nothing in `run_meta.json` marks it as a
    replicate, so only the run_id timestamp distinguishes it. On 2026-08-03 a
    51-episode `B0_som_classifieds` replicate hijacked the rescan away from the
    224-episode canonical run, and the resulting per-rule diff showed ~20 rules
    "changing" that the rule batch had never touched.

    Fix: when a baseline scan dir is supplied, PIN each condition to the run_id the
    baseline used. A diff against a baseline is only meaningful if both scans cover
    the same runs; pinning makes that structural rather than a thing to remember.
    """
    pinned = {}
    if baseline_dir is not None:
        for f in Path(baseline_dir).glob("*_classifieds.json"):
            try:
                rid = json.loads(f.read_text(encoding="utf-8")).get("run_id")
            except Exception:
                continue
            if rid:
                pinned[f.stem] = rid

    out = {}
    for model in ("B0", "B1", "B2"):
        for mode in MODE_OF:
            key = f"{model}_{mode}_classifieds"
            if key in pinned and (PHASE1 / pinned[key]).is_dir():
                out[key] = pinned[key]
                continue
            pat = f"{model}_{mode}_classifieds*"
            cands = sorted(PHASE1.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
            # exclude longer-mode false prefixes (som must not match phantom_som)
            cands = [c for c in cands if re.match(rf"^{model}_{mode}_classifieds(_|$)", c.name)]
            if cands:
                out[key] = cands[0].name
                if key in pinned:
                    print(f"  ⚠ {key}: baseline run {pinned[key]} 不存在, 回退到最新 {cands[0].name}")
    return out


def scan(key: str, run: str, out_dir: Path) -> dict | None:
    mode = None
    for m in sorted(MODE_OF, key=len, reverse=True):
        if f"_{m}_" in f"_{key}_".replace(key.split("_")[0] + "_", "_", 1):
            mode = m
            break
    if mode is None:                                   # fall back to name parse
        parts = key.split("_")
        mode = "_".join(parts[1:-1])
    run_dir = PHASE1 / run
    if not run_dir.is_dir():
        print(f"  ✗ {key}: run dir missing ({run})")
        return None
    out = out_dir / f"{key}.json"
    r = subprocess.run(
        [PY, str(SCANNER), "--run-dir", str(run_dir), "--output", str(out)],
        capture_output=True, text=True, cwd=str(REPO),
    )
    if r.returncode != 0 or not out.exists():
        print(f"  ✗ {key}: scanner rc={r.returncode} {r.stderr.strip()[:120]}")
        return None
    return json.load(open(out))


def per_rule(d: dict) -> collections.Counter:
    c = collections.Counter()
    for e in d["results"]:
        for h in e["hits"]:
            c[h["rule_id"] if isinstance(h, dict) else str(h)] += 1
    return c


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--sites", nargs="+", default=["reddit", "classifieds"])
    ap.add_argument("--baseline-dir", type=Path, help="prior scan dir for a per-rule diff")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    targets = dict(CANONICAL)
    if "classifieds" in args.sites:
        cls = _discover_cls(args.baseline_dir)
        print(f"cls 目录解析 ({len(cls)}):")
        for k, v in sorted(cls.items()):
            print(f"    {k:28s} → {v}")
        targets.update(cls)
    targets = {k: v for k, v in targets.items() if any(s in k for s in args.sites)}

    print(f"\n扫描 {len(targets)} 个 condition …")
    versions = collections.Counter()
    scanned = {}
    for key in sorted(targets):
        d = scan(key, targets[key], args.out_dir)
        if d is None:
            continue
        versions[d.get("ruleset_version", "?")] += 1
        scanned[key] = d
        n = len(d["results"]); s = sum(1 for e in d["results"] if e["success"])
        print(f"  ✓ {key:30s} n={n:3d} SR={100*s/n if n else 0:5.2f}%")

    print(f"\nruleset_version 一致性: {dict(versions)}")
    if len(versions) != 1:
        print("  ⚠️ 版本不一致 — cross-mode 聚合被禁止 (discover-then-freeze 硬纪律)")
        return 1
    print("  ✅ 全部同版本 → cross-mode 聚合解锁")

    if args.baseline_dir:
        print("\n与 baseline 的 per-rule 差异 (仅列变化):")
        for key in sorted(scanned):
            b = args.baseline_dir / f"{key}.json"
            if not b.exists():
                print(f"  {key:30s} (baseline 缺失, 新增 condition)")
                continue
            cb = per_rule(json.load(open(b))); cn = per_rule(scanned[key])
            diff = {k: (cb.get(k, 0), cn.get(k, 0)) for k in set(cb) | set(cn)
                    if cb.get(k, 0) != cn.get(k, 0)}
            if diff:
                print(f"  {key:30s} {diff}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
