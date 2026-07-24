#!/usr/bin/env python3
"""Grammar-only check against a LOCAL LanguageTool server.

Privacy: this script refuses to talk to the public languagetool.org API;
unpublished manuscripts should never leave the machine. Start a local
server first, e.g.:

    docker run --rm -d -p 8010:8010 erikvl87/languagetool

or download the standalone server from https://languagetool.org/download/
and run: java -cp languagetool-server.jar org.languagetool.server.HTTPServer --port 8010

Only mechanical categories are enabled (grammar, typos, punctuation,
casing, confused words). Style is the job of the skill + Vale layers;
letting LanguageTool restyle prose would fight them.

Usage: grammar_check.py FILE [FILE...] [--url http://localhost:8010] [--lang en-US]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tex_to_text import convert  # noqa: E402

CATEGORIES = "GRAMMAR,TYPOS,PUNCTUATION,CASING,CONFUSED_WORDS"
MAX_CHARS = 20000  # keep requests well under typical server limits


def check_chunk(url: str, lang: str, text: str) -> list[dict]:
    data = urllib.parse.urlencode({
        "text": text,
        "language": lang,
        "enabledCategories": CATEGORIES,
        "enabledOnly": "true",
    }).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v2/check", data=data)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp).get("matches", [])


def chunks(text: str):
    """Split on paragraph boundaries into <= MAX_CHARS pieces."""
    buf: list[str] = []
    size = 0
    for para in text.split("\n\n"):
        if size + len(para) > MAX_CHARS and buf:
            yield "\n\n".join(buf)
            buf, size = [], 0
        buf.append(para)
        size += len(para) + 2
    if buf:
        yield "\n\n".join(buf)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--url", default="http://localhost:8010")
    ap.add_argument("--lang", default="en-US")
    ap.add_argument("--allow-remote", action="store_true",
                    help="permit a non-localhost server you trust "
                         "(e.g. LanguageTool on your own LAN box); the "
                         "public languagetool.org API stays refused")
    args = ap.parse_args()

    host = urlparse(args.url).hostname or ""
    if host.endswith("languagetool.org"):
        sys.exit("refusing to send manuscript text to the public API "
                 "(even with --allow-remote); run a local server (see --help)")
    if host not in {"localhost", "127.0.0.1", "::1"} and not args.allow_remote:
        sys.exit(f"refusing non-local server {host!r}: manuscript text would "
                 "leave this machine. Use a localhost server, or pass "
                 "--allow-remote for a private server you trust.")

    total = 0
    for name in args.files:
        path = Path(name)
        raw = path.read_text(errors="replace")
        text = convert(raw) if path.suffix == ".tex" else raw
        for chunk in chunks(text):
            try:
                matches = check_chunk(args.url, args.lang, chunk)
            except (urllib.error.URLError, OSError):
                sys.exit(f"error: no LanguageTool server at {args.url}\n"
                         "start one locally, e.g.:\n"
                         "  docker run --rm -d -p 8010:8010 erikvl87/languagetool")
            for m in matches:
                ctx = m["context"]["text"]
                off = m["context"]["offset"]
                length = max(m["context"]["length"], 1)
                print(f"{name}: {m['message']}")
                print(f"    {ctx}")
                print(f"    {' ' * off}{'^' * length}")
                reps = [r["value"] for r in m.get("replacements", [])[:3]]
                if reps:
                    print(f"    suggest: {', '.join(reps)}")
            total += len(matches)

    print(f"\n{total} grammar issue(s) found.")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
