"""Export the talk after print-only background images have decoded.

Usage: .venv/bin/python3 deliverables/showcase/talk/export_pdf.py
"""
from pathlib import Path

from playwright.sync_api import sync_playwright


HERE = Path(__file__).resolve().parent


def main() -> None:
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        # Print exposes all slides, including dark slides hidden on screen.
        page.emulate_media(media="print")
        page.goto((HERE / "index.html").as_uri(), wait_until="load")
        assets = page.evaluate(r"""async () => {
            await document.fonts.ready;
            const urls = new Set();
            for (const el of document.querySelectorAll('*')) {
                const bg = getComputedStyle(el).backgroundImage;
                for (const match of bg.matchAll(/url\(["']?([^"')]+)["']?\)/g)) {
                    urls.add(match[1]);
                }
            }
            const backgrounds = [...urls].map(src => {
                const img = new Image();
                img.src = src;
                return img;
            });
            await Promise.all([...document.images, ...backgrounds].map(img => img.decode()));
            await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
            return [...urls];
        }""")
        assert any(url.endswith("tpl_bg.jpg") for url in assets), assets
        for name in ("tpl_logo_hai.png", "tpl_logo_ucl_cdi.png"):
            assert any(url.endswith(name) for url in assets), name
        page.pdf(path=str(HERE / "talk.pdf"), width="13.333333in", height="7.5in",
                 print_background=True)
        browser.close()
    print(f"Exported {HERE / 'talk.pdf'}; decoded {len(assets)} background assets")


if __name__ == "__main__":
    main()
