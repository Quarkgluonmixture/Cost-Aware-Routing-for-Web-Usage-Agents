#!/usr/bin/env python3
"""BOTH panel of the talk's `agents` slide: browser-use's OWN highlighted screenshot of the same public page.

browser-use (0.13.10 when this was written) boxes and numbers the page's interactive elements by default
(`highlight_elements=True`) — mechanically the Set-of-Marks family, so it is the tool the room knows for the
BOTH view. This script drives its BrowserSession headless on the same URL as real_capture.py, asks it for the
browser state, and lets its own `create_highlighted_screenshot_async` draw the boxes; nothing is drawn by us.

Setup (a scratch venv is enough; browser-use is not a project dependency)::

    python3 -m venv /tmp/bu && /tmp/bu/bin/pip install browser-use
    /tmp/bu/bin/python deliverables/showcase/talk/browser_use_capture.py            # writes into talk/fig/

Outputs: browser_use_wiki_highlighted.png (its screenshot with its boxes), bu_wiki_plain.png (without),
browser_use_wiki_text.txt (its text side, `[index]<tag …>` lines). The slide shows a crop of the first
(real_capture-style crop box 30,55–540,600 — see the PIL crop in 笔记 §527.5). chromium_sandbox=False
because this host's AppArmor blocks Chromium's user-namespace sandbox.
"""
import asyncio, base64, sys
from pathlib import Path
from browser_use import BrowserSession, BrowserProfile
from browser_use.browser.python_highlights import create_highlighted_screenshot_async
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "fig"; URL = 'https://en.wikipedia.org/wiki/Special:CreateAccount'
async def main():
    profile = BrowserProfile(headless=True, chromium_sandbox=False, highlight_elements=True,
        window_size={'width': 1280, 'height': 800}, viewport={'width': 1280, 'height': 800},
        executable_path='/home/jiaming/.cache/ms-playwright/chromium-1223/chrome-linux/chrome')
    s = BrowserSession(browser_profile=profile)
    await s.start(); await s.navigate_to(URL); await asyncio.sleep(4)
    st = await s.get_browser_state_summary(include_screenshot=True)
    sm = st.dom_state.selector_map
    cdp = await s.get_or_create_cdp_session()
    hl = await create_highlighted_screenshot_async(st.screenshot, sm, cdp_session=cdp, filter_highlight_ids=True)
    (OUT / 'browser_use_wiki_highlighted.png').write_bytes(base64.b64decode(hl))
    (OUT / 'bu_wiki_plain.png').write_bytes(base64.b64decode(st.screenshot))
    (OUT / 'browser_use_wiki_text.txt').write_text(st.dom_state.llm_representation(), encoding='utf-8')
    print('elements', len(sm), '| highlighted bytes', len(base64.b64decode(hl)))
    await s.kill()
asyncio.run(main())
