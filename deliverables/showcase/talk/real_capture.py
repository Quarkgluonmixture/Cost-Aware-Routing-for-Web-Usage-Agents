#!/usr/bin/env python3
"""Real captures for the talk's `agents` slide — no mock-up.

capture  drives the real @playwright/mcp server (the server Claude Code uses) over stdio on one public page and
         saves, from ONE session: its `browser_snapshot` reply, its viewport screenshot, and the on-screen boxes of
         the page's interactive elements (`browser_evaluate` + getBoundingClientRect). Needs the network; the page
         drifts, so the saved files — not a re-run — are what the slide shows.
marks    draws those boxes on the saved screenshot the way this project's BOTH view draws them (same outline colour,
         width, numbered pill labels — `p79.experiment.som._draw_label`), giving the slide's third panel:
         fig/real_wiki_createaccount_som.png. The numbering is 1..K in page order, like the project's [SOM_MARKS].
render   draws an excerpt of the saved snapshot as fig/real_wiki_snapshot_excerpt.png (wide, for reference) and
         fig/real_wiki_snapshot_excerpt_compact.png (narrow, for the three-column slide). Lines are verbatim; each
         elided run is replaced by a count of the lines it held, never by a paraphrase.

Usage::

    .venv/bin/python3 deliverables/showcase/talk/real_capture.py capture
    .venv/bin/python3 deliverables/showcase/talk/real_capture.py marks
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
REPO = HERE.parents[2]
URL = "https://en.wikipedia.org/wiki/Special:CreateAccount"
SNAPSHOT = FIG / "real_wiki_createaccount_snapshot.md"
SCREENSHOT = FIG / "real_wiki_createaccount_screenshot.png"
BOXES = FIG / "real_wiki_createaccount_boxes.json"
MARKED = FIG / "real_wiki_createaccount_som.png"
EXCERPT = FIG / "real_wiki_snapshot_excerpt.png"
EXCERPT_COMPACT = FIG / "real_wiki_snapshot_excerpt_compact.png"
# this machine's copies; override when running elsewhere
MCP = os.environ.get("PLAYWRIGHT_MCP", str(Path.home() / ".npm/_npx/9833c18b2d85bc59/node_modules/.bin/playwright-mcp"))
CHROME = os.environ.get("PLAYWRIGHT_CHROME", str(Path.home() / ".cache/ms-playwright/chromium-1223/chrome-linux/chrome"))

# The elements a Set-of-Marks view boxes: what the page lets you act on, as the browser exposes it. Nested hits
# (a link wrapping an image) keep the outer one; off-screen and invisible ones are skipped.
BOXES_JS = r"""() => {
  const sel = 'a[href], button, input, select, textarea, [role="button"], [role="link"], [role="textbox"], ' +
              '[role="radio"], [role="checkbox"], [role="menuitem"], [role="tab"], [role="combobox"]';
  const seen = [], out = [];
  for (const el of document.querySelectorAll(sel)) {
    if (seen.some(p => p.contains(el))) continue;
    const r = el.getBoundingClientRect();
    if (r.width < 4 || r.height < 4) continue;
    if (r.bottom <= 0 || r.right <= 0 || r.top >= innerHeight || r.left >= innerWidth) continue;
    const st = getComputedStyle(el);
    if (st.visibility === 'hidden' || st.display === 'none' || +st.opacity === 0) continue;
    seen.push(el);
    const name = (el.getAttribute('aria-label') || el.getAttribute('placeholder') || el.textContent || '').trim();
    out.push({tag: el.tagName.toLowerCase(), role: el.getAttribute('role') || '', name: name.slice(0, 60),
              x: r.left, y: r.top, w: r.width, h: r.height});
  }
  return out;
}"""


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
                         "clientInfo": {"name": "talk-real-capture", "version": "1.1"}}})
        server = reply(1)["result"].get("serverInfo")
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        call(2, "browser_navigate", {"url": URL})
        snapshot = call(3, "browser_snapshot", {})
        call(4, "browser_take_screenshot", {"filename": "page.png", "scale": "css", "type": "png"})
        boxes_reply = call(5, "browser_evaluate", {"function": BOXES_JS})
        call(6, "browser_close", {})
        proc.terminate()
        stamp = time.strftime("%Y-%m-%d %H:%M %Z")
        SNAPSHOT.write_text(f"<!-- captured {stamp} · {server} · {URL} -->\n{snapshot}", encoding="utf-8")
        (Path(tmp) / "page.png").replace(SCREENSHOT)
        # the reply is "### Result\n<json array>" followed by other sections; decode the array that starts there
        head = boxes_reply.find("### Result")
        start = boxes_reply.find("[", head)
        assert head >= 0 and start > head, f"no json result in the evaluate reply:\n{boxes_reply[:400]}"
        boxes, _ = json.JSONDecoder().raw_decode(boxes_reply[start:])
        BOXES.write_text(json.dumps({"captured": stamp, "server": server, "url": URL, "viewport": [1280, 800],
                                     "selector_note": "interactive elements in the viewport, page order, outer of nested",
                                     "boxes": boxes}, indent=1), encoding="utf-8")
    print(f"server {server}\n  {SNAPSHOT.name}: {len(yaml_lines())} lines\n  {SCREENSHOT.name}\n  {BOXES.name}: {len(boxes)} boxes")


def marks() -> None:
    """The BOTH panel: the saved screenshot with the saved boxes drawn as this project's SoM view draws them."""
    from PIL import Image, ImageDraw

    sys.path.insert(0, str(REPO))
    from p79.experiment.som import _draw_label, _get_font   # the production drawing helpers, unchanged

    boxes = json.loads(BOXES.read_text(encoding="utf-8"))["boxes"]
    img = Image.open(SCREENSHOT).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _get_font(size=14)
    for seq, b in enumerate(boxes, start=1):
        x1, y1, x2, y2 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
        draw.rectangle([x1, y1, x2, y2], outline="#00BCD4", width=2)      # p79/experiment/som.py::_build_som_result
        _draw_label(draw, x1, y1, str(seq), font)
    img.save(MARKED)
    print(f"  {MARKED.name}: {len(boxes)} marks drawn")


def yaml_lines() -> list[str]:
    text = SNAPSHOT.read_text(encoding="utf-8")
    block = re.search(r"```yaml\n(.*?)```", text, re.S)
    assert block, "no yaml snapshot in the saved reply"
    return block.group(1).rstrip("\n").split("\n")


def _rows(lines: list[str], compact: bool) -> list[tuple[str, str]]:
    find = lambda pat: next(i for i, l in enumerate(lines) if re.search(pat, l))
    heading = find(r'- heading "Create account"')
    form = find(r'- generic "Create account"')
    button = find(r'- button "Create your account"')
    rows: list[tuple[str, str]] = [("line", lines[0])]
    for lo, hi in ((1, heading), (heading + 1, form)):
        rows.append(("gap", f"⋮  {hi - lo} lines"))
        if hi == heading:
            rows.append(("line", lines[heading]))
    if compact:
        # the form's textboxes and its button only, each elided run counted — still every line verbatim
        keep = [i for i in range(form, button + 1)
                if re.search(r'- (textbox|button) |- generic "Create account"|/placeholder', lines[i])]
        prev = form - 1
        for i in keep:
            if i - prev > 1:
                rows.append(("gap", f"⋮  {i - prev - 1} lines"))
            rows.append(("line", lines[i]))
            prev = i
    else:
        rows += [("line", l) for l in lines[form:button + 1]]
    rows.append(("gap", f"⋮  {len(lines) - button - 1} more lines"))
    return rows


def render() -> None:
    from playwright.sync_api import sync_playwright

    lines = yaml_lines()
    jobs = ((EXCERPT, False, 1280, 17), (EXCERPT_COMPACT, True, 760, 20))
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        for out, compact, width, px in jobs:
            rows = _rows(lines, compact)
            body = []
            for kind, text in rows:
                cls = "gap" if kind == "gap" else ("hit" if re.search(r"- (textbox|button) ", text) else "")
                shown = re.sub(r"^ {6}", "", text) if compact else text     # compact: drop six of the indent columns
                body.append(f'<div class="{cls}">{html.escape(shown)}</div>')
            left = "browser_snapshot — real reply" if compact else "browser_snapshot — real reply, excerpt"
            right = f"{len(lines)} lines" if compact else f"{len(lines)} lines for this page"
            page = f"""<!doctype html><meta charset="utf-8"><style>
              body{{margin:0;background:#1b1c26;font:{px}px/1.42 Consolas,'DejaVu Sans Mono',monospace;color:#cfd3e4;width:{width}px}}
              .bar{{display:flex;justify-content:space-between;gap:16px;padding:12px 22px;background:#12131b;color:#a3a9c6;font-size:{px - 1}px;white-space:nowrap}}
              .bar b{{color:#e8845c;font-weight:700}}
              pre{{margin:0;padding:14px 22px 18px;white-space:pre;overflow:hidden}} .hit{{color:#ffd580}} .gap{{color:#6f7695;font-style:italic}}
            </style><div class="bar"><span><b>playwright</b> · {left}</span>
            <span>{right}</span></div><pre>{''.join(body)}</pre>"""
            tab = browser.new_page(viewport={"width": width, "height": 400}, device_scale_factor=2)
            tab.set_content(page)
            tab.screenshot(path=str(out), full_page=True)
            tab.close()
            print(f"  {out.name}: {len(rows)} rows shown of {len(lines)} snapshot lines")
        browser.close()


if __name__ == "__main__":
    {"capture": capture, "marks": marks, "render": render}[sys.argv[1] if len(sys.argv) > 1 else "render"]()
