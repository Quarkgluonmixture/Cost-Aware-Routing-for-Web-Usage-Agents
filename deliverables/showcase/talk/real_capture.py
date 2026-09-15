#!/usr/bin/env python3
"""Real captures for the talk's `agents` slide — no mock-up.

capture  drives the real @playwright/mcp server (the server Claude Code uses) over stdio on one public page and
         saves its `browser_snapshot` reply and its viewport screenshot under fig/. Needs the network; the page
         drifts, so the saved files — not a re-run — are what the slide shows.
render   draws an excerpt of the saved snapshot as fig/real_wiki_snapshot_excerpt.png. Lines are verbatim; each
         elided run is replaced by a count of the lines it held, never by a paraphrase.

Usage::

    .venv/bin/python3 deliverables/showcase/talk/real_capture.py capture
    .venv/bin/python3 deliverables/showcase/talk/real_capture.py render
"""
from __future__ import annotations

import html
import json
import os
import re
import select
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG = HERE / "fig"
URL = "https://en.wikipedia.org/wiki/Special:CreateAccount"
SNAPSHOT = FIG / "real_wiki_createaccount_snapshot.md"
SCREENSHOT = FIG / "real_wiki_createaccount_screenshot.png"
EXCERPT = FIG / "real_wiki_snapshot_excerpt.png"
# this machine's copies; override when running elsewhere
MCP = os.environ.get("PLAYWRIGHT_MCP", str(Path.home() / ".npm/_npx/9833c18b2d85bc59/node_modules/.bin/playwright-mcp"))
CHROME = os.environ.get("PLAYWRIGHT_CHROME", str(Path.home() / ".cache/ms-playwright/chromium-1223/chrome-linux/chrome"))


def capture() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        # --no-sandbox: this host's AppArmor blocks Chromium's user-namespace sandbox (MCP launch fails without it)
        cmd = [MCP, "--headless", "--isolated", "--no-sandbox", "--viewport-size", "1280x800",
               "--output-dir", tmp, "--executable-path", CHROME]
        proc = subprocess.Popen(cmd, cwd=tmp, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True, bufsize=1)

        def send(msg: dict) -> None:
            proc.stdin.write(json.dumps(msg) + "\n")
            proc.stdin.flush()

        def reply(want: int, timeout: float = 120) -> dict:
            end = time.time() + timeout
            while time.time() < end:
                if not select.select([proc.stdout], [], [], 1)[0]:
                    continue
                line = proc.stdout.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if msg.get("id") == want:
                    return msg
            raise TimeoutError(f"no MCP reply for id {want}")

        def call(i: int, name: str, args: dict) -> str:
            send({"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": name, "arguments": args}})
            msg = reply(i)
            if "error" in msg:
                raise RuntimeError(msg["error"])
            return "\n".join(c.get("text", "") for c in msg["result"].get("content", []) if c.get("type") == "text")

        send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                         "clientInfo": {"name": "talk-real-capture", "version": "1.0"}}})
        server = reply(1)["result"].get("serverInfo")
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        call(2, "browser_navigate", {"url": URL})
        snapshot = call(3, "browser_snapshot", {})
        call(4, "browser_take_screenshot", {"filename": "page.png", "scale": "css", "type": "png"})
        call(5, "browser_close", {})
        proc.terminate()
        SNAPSHOT.write_text(f"<!-- captured {time.strftime('%Y-%m-%d %H:%M %Z')} · {server} · {URL} -->\n{snapshot}",
                            encoding="utf-8")
        (Path(tmp) / "page.png").replace(SCREENSHOT)
    print(f"server {server}\n  {SNAPSHOT.name}\n  {SCREENSHOT.name}")


def yaml_lines() -> list[str]:
    text = SNAPSHOT.read_text(encoding="utf-8")
    block = re.search(r"```yaml\n(.*?)```", text, re.S)
    assert block, "no yaml snapshot in the saved reply"
    return block.group(1).rstrip("\n").split("\n")


def render() -> None:
    from playwright.sync_api import sync_playwright

    lines = yaml_lines()
    find = lambda pat: next(i for i, l in enumerate(lines) if re.search(pat, l))
    heading = find(r'- heading "Create account"')
    form = find(r'- generic "Create account"')
    button = find(r'- button "Create your account"')
    rows: list[tuple[str, str]] = [("line", lines[0])]
    for lo, hi in ((1, heading), (heading + 1, form)):
        rows.append(("gap", f"⋮  {hi - lo} lines"))
        if hi == heading:
            rows.append(("line", lines[heading]))
    rows += [("line", l) for l in lines[form:button + 1]]
    rows.append(("gap", f"⋮  {len(lines) - button - 1} more lines"))
    body = []
    for kind, text in rows:
        cls = "gap" if kind == "gap" else ("hit" if re.search(r"- (textbox|button) ", text) else "")
        body.append(f'<div class="{cls}">{html.escape(text)}</div>')
    page = f"""<!doctype html><meta charset="utf-8"><style>
      body{{margin:0;background:#1b1c26;font:17px/1.42 Consolas,'DejaVu Sans Mono',monospace;color:#cfd3e4;width:1280px}}
      .bar{{display:flex;justify-content:space-between;padding:12px 22px;background:#12131b;color:#a3a9c6;font-size:16px}}
      .bar b{{color:#e8845c;font-weight:700}}
      pre{{margin:0;padding:14px 22px 18px;white-space:pre}} .hit{{color:#ffd580}} .gap{{color:#6f7695;font-style:italic}}
    </style><div class="bar"><span><b>playwright</b> · browser_snapshot — real reply, excerpt</span>
    <span>{len(lines)} lines for this page</span></div><pre>{''.join(body)}</pre>"""
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        tab = browser.new_page(viewport={"width": 1280, "height": 400}, device_scale_factor=2)
        tab.set_content(page)
        tab.screenshot(path=str(EXCERPT), full_page=True)
        browser.close()
    print(f"  {EXCERPT.name}: {len(rows)} rows shown of {len(lines)} snapshot lines")


if __name__ == "__main__":
    {"capture": capture, "render": render}[sys.argv[1] if len(sys.argv) > 1 else "render"]()
