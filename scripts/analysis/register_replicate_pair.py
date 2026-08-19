#!/usr/bin/env python3
"""Register a same-condition replicate pair into CLEAN_PAIRS — safely (笔记 §471.8).

WHY THIS CANNOT BE A DATA FILE
------------------------------
The obvious design is an external registry (JSONL) that the inventory appends to. It
does not work, and the reason is worth stating because it will be proposed again:
``validate_fire_manifest.registered_replicate_run_ids()`` reads CLEAN_PAIRS with
``ast.literal_eval`` on the *source*, deliberately, so the check stays side-effect-free.
``literal_eval`` accepts literals only. The moment CLEAN_PAIRS becomes
``[...] + load_registry()`` it raises ValueError, the validator returns an empty
frozenset, and **every registered replicate becomes a ghost at once** — the exact
failure §469.5 recorded, where one syntax problem in this file emptied the registry
and no report mentioned the offending line.

So registration edits the literal, and this script exists to make that edit checkable
rather than manual.

WHAT IT VERIFIES AFTER WRITING (all of it, or it rolls back)
  1. the file still parses
  2. CLEAN_PAIRS still ``literal_eval``s — the property the validator depends on
  3. the new pair is present in what the validator itself returns
  4. both run directories exist and carry a complete condition summary
  5. the pair count went up by exactly one

Usage:
  register_replicate_pair.py --label B5.cls.dom \\
      --canonical results/visualwebarena/phase1/<runA>/phase1_dom_router_0 \\
      --replicate results/visualwebarena/phase1/<runB>/phase1_dom_router_0 \\
      --expected-n 224 --note "reframe chain A1/A2, intent 20260819"
"""
from __future__ import annotations

import argparse
import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
INVENTORY = REPO / "scripts" / "analysis" / "aggregate_noise_floor_inventory.py"


def _pairs_from_source(src: str):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "CLEAN_PAIRS" for t in node.targets
        ):
            return ast.literal_eval(node.value), node
    raise SystemExit("CLEAN_PAIRS assignment not found")


def _condition_ok(rel: str, expected_n: int) -> tuple[bool, str]:
    d = REPO / rel
    if not d.is_dir():
        return False, f"not a directory: {rel}"
    f = d / "condition_summary_v2.json"
    if not f.is_file():
        return False, f"no condition_summary_v2.json under {rel}"
    try:
        s = json.loads(f.read_text())
    except Exception as e:  # noqa: BLE001
        return False, f"summary unparseable: {e}"
    ep = s.get("episodes", s.get("total_tasks", s.get("num_tasks", s.get("scored_task_count", 0))))
    if ep != expected_n:
        return False, f"episodes={ep} != expected {expected_n}"
    return True, f"ok ({ep} episodes)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="e.g. B5.cls.dom")
    ap.add_argument("--canonical", required=True, help="repo-relative condition dir (arm A)")
    ap.add_argument("--replicate", required=True, help="repo-relative condition dir (arm B)")
    ap.add_argument("--expected-n", type=int, required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = INVENTORY.read_text(encoding="utf-8")
    pairs, node = _pairs_from_source(src)
    before = len(pairs)

    if any(p[0] == args.label for p in pairs):
        print(f"already registered: {args.label} — nothing to do")
        return 0

    # both arms must be real and complete BEFORE the file is touched
    for name, rel in (("canonical", args.canonical), ("replicate", args.replicate)):
        ok, why = _condition_ok(rel, args.expected_n)
        print(f"  {name:10} {rel}\n             {why}")
        if not ok:
            print(f"REFUSING to register: {name} arm is not complete", file=sys.stderr)
            return 2

    entry = (
        f'    # Registered {args.note or "by register_replicate_pair.py"}.\n'
        f'    ("{args.label}",\n'
        f'     "{args.canonical}",\n'
        f'     "{args.replicate}"),\n'
    )
    # insert before the closing bracket of the CLEAN_PAIRS literal
    lines = src.splitlines(keepends=True)
    end_line = node.value.end_lineno  # 1-indexed, the line carrying `]`
    new_src = "".join(lines[: end_line - 1]) + entry + "".join(lines[end_line - 1:])

    if args.dry_run:
        print("--- would insert ---"); print(entry, end="")
        return 0

    backup = INVENTORY.with_suffix(".py.bak")
    shutil.copy2(INVENTORY, backup)
    INVENTORY.write_text(new_src, encoding="utf-8")

    def rollback(why: str) -> int:
        shutil.copy2(backup, INVENTORY)
        backup.unlink(missing_ok=True)
        print(f"ROLLED BACK: {why}", file=sys.stderr)
        return 3

    # 1+2. parses, and still literal_eval-able (what the validator depends on)
    try:
        after_pairs, _ = _pairs_from_source(INVENTORY.read_text(encoding="utf-8"))
    except SyntaxError as e:
        return rollback(f"file no longer parses: {e}")
    except ValueError as e:
        return rollback(f"CLEAN_PAIRS no longer literal_eval-able: {e} "
                        "(this is the §469.5 failure — it would ghost EVERY replicate)")

    # 5. exactly one more
    if len(after_pairs) != before + 1:
        return rollback(f"pair count {before} -> {len(after_pairs)}, expected +1")

    # 3. the validator itself must see it
    sys.path.insert(0, str(REPO / "scripts" / "analysis"))
    try:
        import importlib
        import validate_fire_manifest as vfm  # noqa: E402
        importlib.reload(vfm)
        seen = vfm.registered_replicate_run_ids()
    except Exception as e:  # noqa: BLE001
        return rollback(f"validate_fire_manifest could not read the registry: {e}")
    rep_run = args.replicate.split("/")[3] if args.replicate.startswith("results/") else ""
    if rep_run and rep_run not in seen:
        return rollback(f"validator does not see {rep_run} after registration "
                        f"(sees {len(seen)} run ids) — the entry is present but not counted")

    backup.unlink(missing_ok=True)
    print(f"registered {args.label}: CLEAN_PAIRS {before} -> {len(after_pairs)} pairs")
    print(f"validator now sees {len(seen)} replicate run ids incl. {rep_run}")

    # 4-bis. the inventory must still run end to end
    r = subprocess.run([sys.executable, str(INVENTORY)], capture_output=True, text=True, cwd=REPO)
    if r.returncode != 0:
        print("WARNING: aggregate_noise_floor_inventory.py exited nonzero after registration:",
              file=sys.stderr)
        print(r.stderr[-800:], file=sys.stderr)
        return 4
    tail = [l for l in r.stderr.splitlines() if "clean pair" in l or "wrote" in l]
    print("\n".join("  " + l.split("INFO ")[-1] for l in tail[-10:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
