#!/usr/bin/env python3
"""Inventory every run/condition across all copies of the P79 results trees.

A *condition* is any directory holding `condition_meta.json` (its parent is the run
dir, which normally holds `run_meta.json`). For each one we record where it lives,
what it is (model / mode / site) and an episode fingerprint, so copies of the same
condition on different hosts can be compared without reading artifacts.

Read-only. Does not descend into `artifacts/` or `episodes/` beyond one listdir.

Usage:
    python scripts/maintenance/inventory_all_runs.py --out docs/analysis/run_inventory
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

DEFAULT_SOURCES = {
    "a100_scratch": "E:/a100-condenser-backup/mnt-scratch/p79_results_active_visualwebarena",
    "a100_archives": "E:/a100-condenser-backup/mnt-scratch/p79_archives",
    "a100_ws": "E:/a100-condenser-backup/home-ubuntu/workspace/p79/results",
    "dgx": "E:/dgx-jiaming-backup/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/results",
    "local": "C:/Workspace/Cost-Aware-Routing-for-Web-Usage-Agents/results",
}

SKIP_DESCEND = {"artifacts", "episodes", "task_configs", "node_modules", ".git", "__pycache__"}


def _load_json(p: Path):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        return {"_error": f"{type(e).__name__}: {e}"}


def _episodes(cond: Path):
    ep = cond / "episodes"
    rows = []
    if ep.is_dir():
        with os.scandir(ep) as it:
            for e in it:
                if e.is_file(follow_symlinks=False):
                    st = e.stat(follow_symlinks=False)
                    rows.append((e.name, st.st_size, st.st_mtime))
    return rows


def _count_dirs(p: Path) -> int:
    if not p.is_dir():
        return 0
    with os.scandir(p) as it:
        return sum(1 for e in it if e.is_dir(follow_symlinks=False))


def scan_condition(source: str, root: Path, cond: Path) -> dict:
    meta = _load_json(cond / "condition_meta.json")
    run_dir = cond.parent
    run_meta = _load_json(run_dir / "run_meta.json") if (run_dir / "run_meta.json").exists() else {}
    eps = _episodes(cond)
    summaries = [n for n, _, _ in eps if n.endswith("_summary_v2.json")]
    steps = [n for n, _, _ in eps if n.endswith("_steps_v2.jsonl")]
    fp = hashlib.sha1(
        "\n".join(f"{n}\t{s}" for n, s, _ in sorted(eps)).encode("utf-8")
    ).hexdigest()
    cfg = run_meta.get("config", {}) if isinstance(run_meta, dict) else {}
    exp = cfg.get("experiment", {}) if isinstance(cfg, dict) else {}
    md = meta.get("metadata", {}) if isinstance(meta, dict) else {}
    return {
        "source": source,
        "relpath": cond.relative_to(root).as_posix(),
        "run_id": run_dir.name,
        "condition_id": meta.get("condition_id"),
        "observation_mode": meta.get("observation_mode"),
        "backend_id": meta.get("backend_id"),
        "model_name": md.get("model_name"),
        "router_on": meta.get("router_on"),
        "experiment_name": exp.get("name"),
        "benchmark": exp.get("benchmark"),
        "run_timestamp": run_meta.get("timestamp") if isinstance(run_meta, dict) else None,
        "n_summary": len(summaries),
        "n_steps": len(steps),
        "n_episode_files": len(eps),
        "episode_bytes": sum(s for _, s, _ in eps),
        "episode_max_mtime": max((m for _, _, m in eps), default=None),
        "n_artifact_dirs": _count_dirs(cond / "artifacts"),
        "episode_fingerprint": fp,
        "meta_error": meta.get("_error") if isinstance(meta, dict) else None,
    }


def lp(p) -> str:
    """Extended-length path (some DGX mirror paths exceed MAX_PATH)."""
    s = os.path.abspath(str(p))
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        s = "\\\\?\\" + s
    return s


def walk(source: str, root: Path):
    root = Path(lp(root))
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False, onerror=lambda e: print(f"WARN {e}", file=sys.stderr)):
        d = Path(dirpath)
        if "condition_meta.json" in filenames:
            yield scan_condition(source, root, d)
            dirnames[:] = []
            continue
        dirnames[:] = [x for x in dirnames if x not in SKIP_DESCEND and not (d / x).is_symlink()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", action="append", help="name=path (repeatable); default = the five known roots")
    args = ap.parse_args()
    sources = dict(s.split("=", 1) for s in args.source) if args.source else DEFAULT_SOURCES
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, path in sources.items():
        root = Path(path)
        if not root.is_dir():
            print(f"[{name}] MISSING root {root}", file=sys.stderr)
            continue
        n0 = len(rows)
        rows.extend(walk(name, root))
        print(f"[{name}] {len(rows) - n0} conditions", file=sys.stderr, flush=True)
    with open(out / "conditions.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {out / 'conditions.jsonl'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
