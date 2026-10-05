#!/usr/bin/env python3
"""Merge every copy of the P79 results trees into one store, without touching the copies.

Sources (priority order, highest first — on a path conflict the higher one wins):
  local         C:/Workspace/Cost-Aware-Routing-for-Web-Usage-Agents/results   (episode text synced from A100 up to 2026-09-24)
  a100_ws       E:/a100-condenser-backup/home-ubuntu/workspace/p79/results
  a100_scratch  E:/a100-condenser-backup/mnt-scratch/p79_results_active_visualwebarena -> visualwebarena/
  a100_archives E:/a100-condenser-backup/mnt-scratch/p79_archives                    -> _a100_archives/
  dgx           E:/dgx-jiaming-backup/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/results

Why this order: local is a later A100 sync than the 09-22 pull (B1_som_shopping_20260923: 93 vs 4
episodes); the DGX mirror holds stale steps files that A100 later replaced (R28173 tasks 87/149,
R3561 task 346 — the §400.1 summary/steps identity case).

Same-volume sources are hard-linked (zero extra space; the originals stay intact), other volumes
are copied. A file that exists in several sources is "the same" when size and mtime match;
size-equal/mtime-different pairs are hashed. A losing variant that really differs is linked under
`_conflicts/<source>/<relpath>` and listed in the manifest — nothing is dropped.

Symlinks are skipped (A100's results/visualwebarena is a symlink to the scratch tree, which is
merged directly). Re-running is idempotent: files already present with the same size are skipped.

Usage:
  python scripts/maintenance/merge_run_stores.py --dest E:/p79-runs --dry-run
  python scripts/maintenance/merge_run_stores.py --dest E:/p79-runs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

SOURCES = [
    ("local", "C:/Workspace/Cost-Aware-Routing-for-Web-Usage-Agents/results", ""),
    ("a100_ws", "E:/a100-condenser-backup/home-ubuntu/workspace/p79/results", ""),
    ("a100_scratch", "E:/a100-condenser-backup/mnt-scratch/p79_results_active_visualwebarena", "visualwebarena/"),
    ("a100_archives", "E:/a100-condenser-backup/mnt-scratch/p79_archives", "_a100_archives/"),
    ("dgx", "E:/dgx-jiaming-backup/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/results", ""),
]
SKIP_NAMES = {".rsync-partial", "__pycache__"}


def lp(p) -> str:
    """Extended-length path: the DGX mechanistic `_obs_mirror` tree has paths > 260 chars,
    which plain Win32 calls cannot list (17,722 dirs were silently unlistable before this)."""
    s = os.path.abspath(str(p))
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        s = "\\\\?\\" + s
    return s


unlistable: list[str] = []
reparse_skipped: list[str] = []


def iter_files(root: Path):
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            it = os.scandir(lp(d))
        except OSError as e:
            print(f"WARN cannot list {d}: {e}", file=sys.stderr)
            unlistable.append(str(d))
            continue
        with it:
            for e in it:
                if e.name in SKIP_NAMES or e.is_symlink():
                    continue
                # WSL symlinks rsync'd onto NTFS are LX reparse points: is_symlink() is False and
                # they look like files, but they point into the A100 filesystem (e.g. the
                # phase1_paper_grade/_vwa -> external/visualwebarena link). Skip every reparse point.
                try:
                    if getattr(e.stat(follow_symlinks=False), "st_file_attributes", 0) & 0x400:
                        reparse_skipped.append(str(Path(d) / e.name))
                        continue
                except OSError:
                    reparse_skipped.append(str(Path(d) / e.name))
                    continue
                if e.is_dir(follow_symlinks=False):
                    stack.append(Path(d) / e.name)
                elif e.is_file(follow_symlinks=False):
                    st = e.stat(follow_symlinks=False)
                    yield Path(d) / e.name, st.st_size, st.st_mtime


def sha1(p: str) -> str:
    h = hashlib.sha1()
    with open(lp(p), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def same_volume(a: str, b: str) -> bool:
    return os.path.splitdrive(os.path.abspath(a))[0].lower() == os.path.splitdrive(os.path.abspath(b))[0].lower()


def place(src: str, dst: str, link: bool):
    os.makedirs(lp(os.path.dirname(dst)), exist_ok=True)
    if link:
        os.link(lp(src), lp(dst))
    else:
        shutil.copy2(lp(src), lp(dst))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", help="dry-run: write conflicts + summary here")
    args = ap.parse_args()
    dest = Path(args.dest)

    # pass 1: index every file of every source (path -> winner + variants)
    winners: dict[str, tuple[str, str, int, float]] = {}
    # an identical copy on dest's volume, used instead of copying a cross-volume winner
    alt_link: dict[str, str] = {}
    conflicts: list[dict] = []
    stats = {s: {"files": 0, "bytes": 0} for s, _, _ in SOURCES}
    t0 = time.time()
    for name, root, prefix in SOURCES:
        rootp = Path(root)
        if not rootp.is_dir():
            print(f"[{name}] MISSING {root}", file=sys.stderr)
            continue
        n = 0
        for p, size, mtime in iter_files(rootp):
            rel = prefix + p.relative_to(rootp).as_posix()
            stats[name]["files"] += 1
            stats[name]["bytes"] += size
            n += 1
            w = winners.get(rel)
            if w is None:
                winners[rel] = (name, str(p), size, mtime)
                continue
            wname, wpath, wsize, wmtime = w
            identical = (size == wsize and int(mtime) == int(wmtime)) or (
                size == wsize and sha1(str(p)) == sha1(wpath))
            if identical:
                if (rel not in alt_link and not same_volume(wpath, str(dest))
                        and same_volume(str(p), str(dest))):
                    alt_link[rel] = str(p)
                continue
            conflicts.append({"relpath": rel, "winner": wname, "winner_size": wsize,
                              "loser": name, "loser_size": size, "loser_path": str(p)})
        print(f"[{name}] indexed {n} files  ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)

    by_src = {}
    for rel, (name, path, size, _) in winners.items():
        d = by_src.setdefault(name, {"files": 0, "bytes": 0, "copy_bytes": 0})
        d["files"] += 1
        d["bytes"] += size
        if not same_volume(path, str(dest)) and rel not in alt_link:
            d["copy_bytes"] += size
    summary = {
        "dest": str(dest),
        "sources_scanned": stats,
        "merged_files_by_winning_source": by_src,
        "merged_files": len(winners),
        "merged_bytes": sum(w[2] for w in winners.values()),
        "conflicts": len(conflicts),
        "unlistable_dirs": len(unlistable),
        "reparse_points_skipped": len(reparse_skipped),
        "cross_volume_winners_linked_from_identical_copy": len(alt_link),
        "copy_bytes_total": sum(d["copy_bytes"] for d in by_src.values()),
    }
    print(json.dumps(summary, indent=2), file=sys.stderr)
    if args.dry_run:
        if args.report:
            Path(args.report).mkdir(parents=True, exist_ok=True)
            with open(Path(args.report) / "dry_conflicts.jsonl", "w", encoding="utf-8") as f:
                for c in conflicts:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
            with open(Path(args.report) / "dry_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        return 0
    if unlistable:
        print(f"REFUSING: {len(unlistable)} unlistable dirs — the merge would silently miss files", file=sys.stderr)
        return 2

    # pass 2: materialise
    meta = dest / "_merge"
    meta.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    for rel, (name, path, size, _) in winners.items():
        dst = dest / rel
        if os.path.exists(lp(dst)) and os.stat(lp(dst)).st_size == size:
            skipped += 1
            continue
        src = alt_link.get(rel, path)
        place(src, str(dst), same_volume(src, str(dest)))
        done += 1
        if done % 50000 == 0:
            print(f"placed {done} (skipped {skipped})", file=sys.stderr, flush=True)
    for c in conflicts:
        dst = dest / "_conflicts" / c["loser"] / c["relpath"]
        if not os.path.exists(lp(dst)):
            place(c["loser_path"], str(dst), same_volume(c["loser_path"], str(dest)))
    with open(meta / "conflicts.jsonl", "w", encoding="utf-8") as f:
        for c in conflicts:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(meta / "provenance.jsonl", "w", encoding="utf-8") as f:
        for rel, (name, path, size, _) in sorted(winners.items()):
            f.write(json.dumps({"relpath": rel, "source": name, "size": size}, ensure_ascii=False) + "\n")
    summary.update({"placed": done, "already_present": skipped, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    with open(meta / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"DONE placed={done} skipped={skipped} conflicts={len(conflicts)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
