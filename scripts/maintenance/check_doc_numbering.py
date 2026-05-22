#!/usr/bin/env python3
"""P79 doc-numbering guard.

Detects cross-session sequence-number collisions in the two monolithic,
sequentially-numbered chronicle docs BEFORE they reach the shared remote:

  - docs/checkpoints/实验笔记.md          top-level sections  `## N. ...`
  - docs/reference/master_bug_catalog.md  bug ids `B-NNNN`

Why this is needed: two independent Claude Code sessions (DGX / A100 / quark /
Myriad) that each compute "max+1" allocate the SAME next number. git then
merges the two additions in *different file regions* WITHOUT a conflict,
silently creating a duplicate (§268 twice / two B-1835 entries). git's own
conflict detection never sees it. This guard catches it by diffing the
working tree against the merge-base and the remote tip.

Core idea (noise-proof):
  added_local  = ids(working tree) - ids(merge-base)
  added_remote = ids(origin/master) - ids(merge-base)
  collision    = added_local & added_remote     # both sessions created same id
Pre-existing duplicates (e.g. the historical §132/§133/§173/§175 pairs) and
the bug catalog's normal "B-1830 followup" multi-mentions live in the base on
both sides, so they are never flagged.

Used by .githooks/pre-push. Also runnable standalone:
    python3 scripts/maintenance/check_doc_numbering.py --remote-ref origin/master
    python3 scripts/maintenance/check_doc_numbering.py --local-only
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter

# Top-level chronicle headers only: exactly `## N.` (## + whitespace + digits + dot).
# `###`-level sub-headers (e.g. §127.1) do NOT match because the char after `##`
# is `#`, not whitespace.
SECTION_RE = re.compile(r"^##[ \t]+(\d+)\.", re.M)
BUG_RE = re.compile(r"\bB-(\d+)\b")

# (path, kind, check_selfdup)
# bug catalog: check_selfdup=False — same B-number legitimately appears many
# times (original entry + followup entries + triage table + cross-refs).
DOCS = [
    ("docs/checkpoints/实验笔记.md", "section", True),
    ("docs/reference/master_bug_catalog.md", "bug", False),
]


def extract(text: str, kind: str) -> list[int]:
    rx = SECTION_RE if kind == "section" else BUG_RE
    return [int(x) for x in rx.findall(text)]


def analyze(local_text: str, base_text: str, remote_text: str,
            kind: str, check_selfdup: bool) -> dict:
    """Pure logic (no git) — unit-testable. See module docstring for the model."""
    local = Counter(extract(local_text, kind))
    base = Counter(extract(base_text, kind))
    remote = Counter(extract(remote_text, kind))

    added_local = set(local) - set(base)
    added_remote = set(remote) - set(base)
    cross = sorted(added_local & added_remote)

    selfdup: list[int] = []
    if check_selfdup:
        # a number duplicated in YOUR working tree that was not already
        # duplicated in the base (so the historical dups never trip it).
        selfdup = sorted(n for n, c in local.items() if c >= 2 and base.get(n, 0) < 2)

    all_known = set(local) | set(remote)
    next_free = (max(all_known) if all_known else 0) + 1
    return {"cross": cross, "selfdup": selfdup, "next_free": next_free}


# ---- git plumbing (thin wrappers; everything fail-open to "") -------------

def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True,
                              text=True, check=True).stdout
    except subprocess.CalledProcessError:
        return ""


def git_show(ref: str, path: str) -> str:
    return _git("show", f"{ref}:{path}")


def merge_base(a: str, b: str) -> str:
    return _git("merge-base", a, b).strip()


def read_local(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def label(kind: str, n: int) -> str:
    return f"§{n}" if kind == "section" else f"B-{n}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--remote-ref", default="origin/master",
                    help="remote tip to compare for cross-session collisions")
    ap.add_argument("--local-only", action="store_true",
                    help="skip remote comparison (only working-tree self-dups vs HEAD)")
    args = ap.parse_args()

    hard = False
    for path, kind, check_selfdup in DOCS:
        local_text = read_local(path)
        if not local_text:
            continue

        if args.local_only:
            base_text = git_show("HEAD", path)
            remote_text = ""
        else:
            base = merge_base("HEAD", args.remote_ref)
            base_text = git_show(base, path) if base else git_show("HEAD", path)
            remote_text = git_show(args.remote_ref, path)

        res = analyze(local_text, base_text, remote_text, kind, check_selfdup)

        if res["cross"]:
            hard = True
            ids = ", ".join(label(kind, n) for n in res["cross"])
            print(f"❌ [doc-guard] {path}")
            print(f"   跨 session 撞号: {ids} —— 另一个 session 已在 {args.remote_ref} 用了同样的号。")
            print(f"   修法: git pull --rebase, 把你的条目改成下一个空号 "
                  f"(≥ {label(kind, res['next_free'])}), 再 push。")
        if res["selfdup"]:
            hard = True
            ids = ", ".join(label(kind, n) for n in res["selfdup"])
            print(f"❌ [doc-guard] {path}")
            print(f"   本地新增重复号: {ids} —— 同一个号在你的改动里出现 ≥2 次, 改掉再 push。")

    if hard:
        print("\n[doc-guard] push BLOCKED. (紧急绕过: git push --no-verify —— 不建议)")
        return 1

    print("[doc-guard] ✓ 无跨 session 撞号 / 新增重复号。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
