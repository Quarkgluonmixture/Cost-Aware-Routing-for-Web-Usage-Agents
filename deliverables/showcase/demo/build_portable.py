"""Build a single-file, fully portable copy of the demo.

WHY
---
The board copy has to survive being carried on a USB stick to a venue with no network
and no guarantee about what is installed. `index.html` + `data.js` + 82 files under
`frames/` is three things that can go missing independently; one file cannot.

QUALITY IS NOT TRADED AWAY
--------------------------
Frames are re-encoded to **lossless WebP**, not to a lossy format and not resized. The
build asserts pixel-for-pixel equality against the source PNG for every frame and
fails if any pixel differs, so "smaller" here never means "worse". Measured on this
set: ~63% of the PNG bytes for identical pixels, which is what keeps the inlined file
near 12 MB instead of 20 MB.

WHY THE DATA URIS LIVE IN A JS OBJECT
-------------------------------------
Base64 dropped straight into 82 `<img src="data:...">` attributes makes the HTML
parser walk ~12 MB of attribute text before the page exists. Held in one JS object
instead, the parser sees a single string literal and the decode happens per image when
`src` is assigned. Same bytes, markedly faster to first paint.

Usage:  .venv/bin/python3 deliverables/showcase/demo/build_portable.py
Output: deliverables/showcase/demo_portable.html   (open by double-click)
"""
from __future__ import annotations

import base64
import io as _io
import json
import re
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "demo_portable.html"


class BuildError(RuntimeError):
    """Fail loud: a silently degraded portable copy is worse than none."""


def encode(png: Path) -> tuple[str, int, int]:
    """Lossless WebP data URI + (png_bytes, webp_bytes), verified pixel-identical."""
    src = Image.open(png).convert("RGB")
    buf = _io.BytesIO()
    src.save(buf, "WEBP", lossless=True, quality=100, method=6)
    raw = buf.getvalue()
    if Image.open(_io.BytesIO(raw)).convert("RGB").tobytes() != src.tobytes():
        raise BuildError(f"{png}: WebP round-trip changed pixels — refusing to ship")
    return ("data:image/webp;base64," + base64.b64encode(raw).decode(),
            png.stat().st_size, len(raw))


def main() -> int:
    html = (HERE / "index.html").read_text(encoding="utf-8")
    data_js = (HERE / "data.js").read_text(encoding="utf-8")
    if "data.js" not in html:
        raise BuildError("index.html no longer references data.js")

    frames = sorted((HERE / "frames").rglob("*.png"))
    if not frames:
        raise BuildError("no frames found — run build_demo_data.py first")

    table, tp, tw = {}, 0, 0
    for i, f in enumerate(frames, 1):
        uri, p, w = encode(f)
        table[f.relative_to(HERE).as_posix()] = uri
        tp += p; tw += w
        print(f"\r  encoding {i}/{len(frames)}  {100*tw/tp:.1f}% of PNG bytes", end="")
    print()

    # every frame the data references must exist in the table, or the portable copy
    # would show a broken image exactly where the served copy works
    demo = json.loads(re.search(r"window\.DEMO = (\{.*\});", data_js, re.S).group(1))
    missing = {fr["img"] for t in demo.values() for ln in t["lanes"].values()
               for fr in ln["frames"]} - set(table)
    if missing:
        raise BuildError(f"{len(missing)} referenced frames not encoded: {sorted(missing)[:3]}")

    inline = ("<script>\n" + data_js + "\nwindow.FRAMES = "
              + json.dumps(table, separators=(",", ":")) + ";\n</script>")
    out = html.replace('<script src="data.js"></script>', inline)
    if out == html:
        raise BuildError("data.js script tag not replaced")
    OUT.write_text(out, encoding="utf-8")

    print(f"\n✓ {OUT.relative_to(HERE.parents[1])}")
    print(f"  frames {len(frames)}   PNG {tp/1e6:.1f} MB -> WebP {tw/1e6:.1f} MB "
          f"({100*tw/tp:.1f}%, pixel-identical)")
    print(f"  single file: {OUT.stat().st_size/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
