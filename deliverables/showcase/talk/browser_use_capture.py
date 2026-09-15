#!/usr/bin/env python3
"""BOTH panel of the talk's `agents` slide: browser-use's OWN highlighted screenshot of the same public page.

browser-use (0.13.10 when this was written) boxes and numbers the page's interactive elements — mechanically the
Set-of-Marks family, so it is the tool the room knows for the BOTH view. It has two renderings: the screenshot
overlay (`highlight_elements`, thin dashed boxes, indices only on short-text elements) and the DOM overlay
(`dom_highlight_elements`, its `add_highlights` JS: a dashed box and a numbered badge on every interactive
element — the look of its early demos). The slide uses the DOM overlay: this script drives its BrowserSession
headless on the same URL as real_capture.py, asks it for the browser state, calls its own `add_highlights`, and
screenshots the page. Nothing is drawn by us.

Setup (a scratch venv is enough; browser-use is not a project dependency)::

    python3 -m venv /tmp/bu && /tmp/bu/bin/pip install browser-use
    /tmp/bu/bin/python deliverables/showcase/talk/browser_use_capture.py            # writes into talk/fig/

Outputs: browser_use_wiki_highlighted.png (its DOM-overlay boxes), browser_use_wiki_text.txt (its text side,
`[index]<tag …>` lines). The slide shows a crop of the first
(real_capture-style crop box 30,55–540,600 — see the PIL crop in 笔记 §527.5). chromium_sandbox=False
because this host's AppArmor blocks Chromium's user-namespace sandbox.
"""
import asyncio, base64, sys
from pathlib import Path
from browser_use import BrowserSession, BrowserProfile
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "fig"; URL = 'https://en.wikipedia.org/wiki/Special:CreateAccount'
async def main():
    profile = BrowserProfile(headless=True, chromium_sandbox=False, highlight_elements=True, dom_highlight_elements=True,
        window_size={'width': 1280, 'height': 800}, viewport={'width': 1280, 'height': 800},
        executable_path='/home/jiaming/.cache/ms-playwright/chromium-1223/chrome-linux/chrome')
    s = BrowserSession(browser_profile=profile)
    await s.start(); await s.navigate_to(URL); await asyncio.sleep(4)
    st = await s.get_browser_state_summary(include_screenshot=True)
    sm = st.dom_state.selector_map
    await s.add_highlights(sm)            # browser-use's own DOM overlay: a dashed box + numbered badge per element
    await asyncio.sleep(1)
    png = await s.take_screenshot()
    (OUT / 'browser_use_wiki_highlighted.png').write_bytes(png)
    (OUT / 'browser_use_wiki_text.txt').write_text(st.dom_state.llm_representation(), encoding='utf-8')
    print('elements', len(sm), '| highlighted bytes', len(png))
    await s.kill()
asyncio.run(main())
