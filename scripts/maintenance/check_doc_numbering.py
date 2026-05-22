#!/usr/bin/env python3
"""P79 doc-numbering guard (v2).

Blocks cross-session sequence-number collisions before they reach the shared
remote, for two append-only docs edited by multiple Claude Code sessions
(DGX / A100 / quark / Myriad, one GitHub remote):

  - docs/checkpoints/实验笔记.md          sections `## N.`   — ENFORCING (clean format)
  - docs/reference/master_bug_catalog.md  bug ids `B-NNNN`   — ADVISORY  (see #4 note)

Detection model (noise-proof via merge-base delta):
  added_local  = ids(pushed commit) - ids(merge-base)
  added_remote = ids(origin/master) - ids(merge-base)
  cross        = added_local & added_remote      # both branches created the same id
  self_dup     = id duplicated WITHIN the push beyond what the base already had

v2 changes — codex race/correctness review 2026-05-22 (docs/checkpoints/codex_outputs/
hook_race_review_2026-05-22.md):
  #1  Reads the COMMIT being pushed via `git show <local_sha>:path` (was the working
      tree), fed from pre-push stdin. Closes the "committed dup + clean working tree"
      and the non-current-ref push holes (codex P0).
  #2  git helpers report ok/fail; a degraded check (missing merge-base / git object,
      e.g. a shallow clone) NO LONGER prints a false "✓ clean" — it warns instead
      (codex P1 fail-open hole).
  #5  Section self-dup rule fixed to `count > base_count and count > 1`, so a *third*
      occurrence of an already-historically-duplicated number is still caught (codex P2).
  #4  The bug catalog is ADVISORY-ONLY. Its format is heterogeneous (`### B-NN.`,
      `## B-NNNN —`, `**B-NNNN**`) AND uses a followup convention that legitimately
      repeats a number across allocation lines, so count-based self-dup is impossible
      by regex (empirically B-1..B-9 each appear 50-128x in heading form). We keep a
      best-effort CROSS-SESSION warning (robust because the set-diff cancels count
      noise) but NEVER BLOCK on bugs, to avoid false-positives on every followup.
      ⚠ Known residual: two sessions on a SHARED local .git (neither pushed yet) and the
      post-pull/post-rebase case are NOT caught for bugs by this regex approach. Reliable
      bug dedup needs a reservation counter (.coord/) or a per-entry restructure, or a
      server-side CI check on the merge result — see hook_race_review.md #3.

Standalone:
    python3 scripts/maintenance/check_doc_numbering.py --local-sha HEAD --remote-ref origin/master
    python3 scripts/maintenance/check_doc_numbering.py --local-only
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter

SECTION_RE = re.compile(r"^##[ \t]+(\d+)\.", re.M)   # `## N.` top-level only (### excluded)
BUG_RE = re.compile(r"\bB-(\d+)\b")                   # all mentions; set-diff cancels noise

# (path, kind, enforce) — enforce=True blocks the push; False is advisory (warn only).
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

    # #5: count > base_count (not "base < 2") so a 3rd occurrence is still caught.
    selfdup = (sorted(n for n, c in local.items() if c > base.get(n, 0) and c > 1)
               if check_selfdup else [])

    all_known = set(local) | set(remote)
    next_free = (max(all_known) if all_known else 0) + 1
    return {"cross": cross, "selfdup": selfdup, "next_free": next_free}


# ---- git plumbing — every helper reports (ok, text); callers track `degraded` ----

def _git(*args: str) -> tuple[bool, str]:
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    return r.returncode == 0, r.stdout


def ref_exists(ref: str) -> bool:
    ok, _ = _git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    return ok


def git_show(ref: str, path: str) -> tuple[bool, str]:
    return _git("show", f"{ref}:{path}")


def merge_base(a: str, b: str) -> str:
    ok, out = _git("merge-base", a, b)
    return out.strip() if ok else ""


def read_local(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def label(kind: str, n: int) -> str:
    return f"§{n}" if kind == "section" else f"B-{n}"


def main() -> int:
    ap = argparse.ArgumentParser(description="P79 doc-numbering guard")
    ap.add_argument("--local-sha", default=None,
                    help="commit being pushed (from pre-push stdin); default = working tree")
    ap.add_argument("--remote-ref", default="origin/master",
                    help="remote tip to compare for cross-session collisions")
    ap.add_argument("--local-only", action="store_true",
                    help="skip remote comparison (working-tree self-dups only)")
    args = ap.parse_args()

    local_rev = args.local_sha or "HEAD"
    remote_present = (not args.local_only) and ref_exists(args.remote_ref)

    hard = False
    degraded: list[str] = []
    advisory: list[str] = []

    for path, kind, enforce in DOCS:
        # --- local doc content (the COMMIT being pushed, or the working tree) ---
        if args.local_sha:
            ok, local_text = git_show(args.local_sha, path)
            if not ok:
                if enforce:
                    degraded.append(f"{path}: 无法读取 push commit ({args.local_sha[:8]}) 内容")
                continue
        else:
            local_text = read_local(path)
            if not local_text:
                continue

        # --- base + remote content ---
        base_text = remote_text = ""
        if remote_present:
            base_rev = merge_base(local_rev, args.remote_ref)
            if not base_rev:
                if enforce:
                    degraded.append(f"{path}: 无法定位 merge-base (shallow clone?) — 跨-session 校验降级")
            else:
                ok_b, base_text = git_show(base_rev, path)
                ok_r, remote_text = git_show(args.remote_ref, path)
                if not (ok_b and ok_r) and enforce:
                    degraded.append(f"{path}: base/remote 对象缺失 — 跨-session 校验降级")
                base_text = base_text if ok_b else ""
                remote_text = remote_text if ok_r else ""

        res = analyze(local_text, base_text, remote_text, kind, check_selfdup=enforce)
        prefix = "❌" if enforce else "⚠ (advisory)"
        for ftype in ("cross", "selfdup"):
            if not res[ftype]:
                continue
            ids = ", ".join(label(kind, n) for n in res[ftype])
            if ftype == "cross":
                line = (f"{prefix} [doc-guard] {path}\n"
                        f"   跨 session 撞号: {ids} —— 另一 session 已在 {args.remote_ref} 用了同号。"
                        f" git pull --rebase 后改用 ≥ {label(kind, res['next_free'])} 再 push。")
            else:
                line = (f"{prefix} [doc-guard] {path}\n"
                        f"   本地新增重复号: {ids} —— 改掉再 push。")
            if enforce:
                hard = True
                print(line)
            else:
                advisory.append(line)

    for m in advisory:
        print(m)
    for m in degraded:
        print(f"⚠ [doc-guard] {m}")

    if hard:
        print("\n[doc-guard] push BLOCKED (enforcing doc 撞号). 紧急绕过: git push --no-verify —— 不建议")
        return 1
    if degraded:
        print("[doc-guard] ⚠ 部分校验降级未完成 — 未确认安全 (见上); 放行 push。")
        return 0
    tail = " (bug 为 advisory)" if advisory else ""
    print(f"[doc-guard] ✓ 无 enforcing-doc 撞号 / 新增重复号{tail}。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
