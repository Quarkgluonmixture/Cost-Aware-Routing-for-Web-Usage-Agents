#!/usr/bin/env python3
"""Dead link check — scan all .md files in docs/ for broken references.

Catches:
- Path-based: `docs/checkpoints/foo.md`, `docs/reference/X.md`
- Wikilinks: `[[note]]`, `[[note#heading]]`, `[[note|alias]]`
- Embeds: `![[X]]`, `![[X.base#view]]`

Pure Python, no GLM. Run weekly via cron or `make check-links`.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DOCS = REPO_ROOT / "docs"

# Path-based reference (relative to repo root): `docs/...md` or `docs/X.base`
PATH_RE = re.compile(r"`?(docs/[\w./_\-]+\.(?:md|canvas|base|py|sh))`?")
# Wikilink: [[note]] or [[note#heading]] or [[note|alias]] or ![[note]]
WIKI_RE = re.compile(r"!?\[\[([^\]|#]+?)(?:#([^\]|]+))?(?:\|[^\]]+)?\]\]")


def find_all_md(root: Path):
    return [p for p in root.rglob("*.md") if ".obsidian" not in p.parts and "_archive" not in p.parts]


def find_vault_files(vault: Path) -> dict:
    """Build map: basename (no ext) → list of relative paths under vault."""
    m: dict = {}
    for p in vault.rglob("*"):
        if p.is_file() and ".obsidian" not in p.parts:
            stem = p.stem
            m.setdefault(stem, []).append(p)
    return m


def check_file(md: Path, vault_files: dict, repo_root: Path) -> list:
    """Return list of broken-reference reports for this md."""
    broken = []
    try:
        text = md.read_text(encoding="utf-8")
    except Exception as e:
        return [(md, "read-error", str(e))]

    for m in PATH_RE.finditer(text):
        rel = m.group(1)
        target = repo_root / rel
        if not target.exists():
            broken.append((md, "path", rel))

    for m in WIKI_RE.finditer(text):
        target = m.group(1).strip()
        if target in vault_files:
            continue
        # Try with .md / .canvas / .base extensions
        if any((Path(target).stem) in vault_files or (Path(target).with_suffix(ext).stem in vault_files) for ext in (".md", ".canvas", ".base")):
            continue
        broken.append((md, "wikilink", f"[[{target}{('#'+m.group(2)) if m.group(2) else ''}]]"))

    return broken


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--vault", type=Path, default=DOCS)
    parser.add_argument("--quiet", action="store_true", help="suppress per-file 'OK'")
    args = parser.parse_args()

    vault_files = find_vault_files(args.vault)
    md_files = find_all_md(args.root)
    print(f"📋 Scanning {len(md_files)} markdown files (vault has {sum(len(v) for v in vault_files.values())} files)")

    total_broken = 0
    for md in sorted(md_files):
        broken = check_file(md, vault_files, args.root)
        if broken:
            rel = md.relative_to(args.root)
            print(f"\n❌ {rel}")
            for _, kind, ref in broken:
                print(f"    [{kind}] {ref}")
            total_broken += len(broken)
        elif not args.quiet:
            pass  # 默认不打 OK

    print(f"\n{'='*60}")
    if total_broken == 0:
        print(f"✅ No broken references found across {len(md_files)} files.")
        return 0
    else:
        print(f"⚠️  Found {total_broken} broken references.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
