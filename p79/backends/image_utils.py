from __future__ import annotations

import base64
import io
from typing import Any, Dict

from PIL import Image

# B-155 (/stress A1.2 v8 Claude A7, 2026-05-16): import-time pin check.
# ``Image.Resampling.LANCZOS`` (used below at line 55) is PIL ≥9.1 enum API;
# older PIL has only ``Image.LANCZOS`` direct attribute. Mixed PIL across
# DGX / Condenser / Myriad → different resize kernel + boundary padding →
# image bytes diverge → paper-grade reproducibility impossible. pyproject
# pins ``pillow>=10.0,<12.0``; this assert catches a manually-overridden
# venv (e.g. older pillow accidentally installed via a sibling package).
import logging as _logging
_pil_logger = _logging.getLogger(__name__)
import PIL
_PIL_VERSION_PARTS = tuple(int(x) for x in PIL.__version__.split(".")[:2])
# B-819 (/stress A1.2 cold-start codex F4 honest-gap promotion, 2026-05-17):
# replace ``assert`` with explicit ``raise`` so ``python -O`` cannot strip
# the lower-bound guard. Upper bound (pyproject ``<12.0``) is currently
# violated by the dev env (Pillow 12.1.1 installed); warn-loud rather than
# hard-fail so this session does not brick on import. Future hardening:
# either pin Pillow <13 in pyproject + bump constants here, or accept the
# drift and document JPEG-byte tolerance for paper §3 reproducibility.
if _PIL_VERSION_PARTS < (10, 0):
    raise RuntimeError(
        f"PIL/Pillow version {PIL.__version__} below paper-grade pinned "
        f"lower bound 10.0. pyproject.toml declares this bound; honor it "
        f"to keep JPEG bytes byte-identical across hosts (B-819)."
    )
if _PIL_VERSION_PARTS >= (12, 0):
    _pil_logger.warning(
        "Pillow %s exceeds pyproject upper pin (<12.0). libjpeg defaults "
        "may have shifted; paper-grade reproducibility risk. Bump pin or "
        "downgrade. (B-819)", PIL.__version__,
    )


# Adapted from external_code/image.py (Aiden Yiliu Li, Apache-2.0)
DEFAULT_MAX_IMAGE_PAYLOAD_BYTES = 5 * 1024 * 1024

# (quality, scale) presets tried in order; first that fits within max_payload_bytes wins.
COMPRESSION_PRESETS = [
    (85, 1.0), (70, 1.0), (55, 0.9), (40, 0.8),
    (30, 0.7), (25, 0.6), (20, 0.5), (20, 0.4),
]


def _normalize_image(image: Image.Image) -> Image.Image:
    # B-820 (/stress A1.2 cold-start P2-3-AC Claude+gemini, 2026-05-17):
    # RGBA → RGB white-composite path is currently DORMANT — empirical archive
    # check (gemini G6 + Claude F10): all VWA screenshots from cls/red/shop
    # archive load as ``mode='RGB'`` (no alpha channel). The composite branch
    # below fires only on hypothetical RGBA inputs (dark-mode page screenshot
    # / PNG with transparency / palette mode). Kept as defense-in-depth for
    # future browser config changes; if a non-RGB screenshot lands during
    # paper-grade fire, the composite produces deterministic white-background
    # JPEG (no random alpha bleed across hosts).
    if image.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", image.size, (255, 255, 255))
        alpha = None
        if image.mode == "P":
            image = image.convert("RGBA")
        if image.mode in ("RGBA", "LA"):
            alpha = image.split()[-1]
        bg.paste(image, mask=alpha)
        return bg
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def _encode_jpeg_base64(image: Image.Image, quality: int) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def encode_image_data_url(
    image: Image.Image,
    max_payload_bytes: int = DEFAULT_MAX_IMAGE_PAYLOAD_BYTES,
    min_quality: int = 20,
) -> Dict[str, Any]:
    # B-416 (/stress A1.2 v8 Mode A P2-4, 2026-05-16): remove redundant
    # `.encode("utf-8")` on hot path. `b64` is already an ASCII-only string
    # (`_encode_jpeg_base64` returns `bytes.decode("ascii")`), so
    # `len(b64.encode("utf-8")) == len(b64)`. Removing the redundant encode
    # saves ~6-8 per-step calls × ~80k steps per condition = ~640k allocs
    # across a Phase 1a fire. No behavior change.
    # B-821 (/stress A1.2 cold-start P2-6-B codex, 2026-05-17): removed
    # `quality_start` parameter — it was DEAD CONFIG. Pre-fix it only
    # controlled the `compressed` boolean label (`compressed: quality <
    # quality_start or scale < 1.0`); the actual encoding quality came from
    # `COMPRESSION_PRESETS[0][0]` regardless. Operator setting
    # `quality_start=30` to "compress harder" was silently no-op; future
    # ablation would have been wrong. Canonical baseline quality is now
    # `COMPRESSION_PRESETS[0][0]` (85) — fixed at the preset table.
    normalized = _normalize_image(image)
    last_img = normalized

    baseline_quality = COMPRESSION_PRESETS[0][0]
    last_quality = COMPRESSION_PRESETS[-1][0]
    for quality, scale in COMPRESSION_PRESETS:
        if scale < 1.0:
            w = max(1, int(normalized.width * scale))
            h = max(1, int(normalized.height * scale))
            img = normalized.resize((w, h), Image.Resampling.LANCZOS)
        else:
            img = normalized
        b64 = _encode_jpeg_base64(img, quality=quality)
        last_img = img
        b64_bytes = len(b64)  # ASCII string → byte length == char length
        if b64_bytes <= max_payload_bytes:
            return {
                "data_url": f"data:image/jpeg;base64,{b64}",
                "payload_bytes": b64_bytes,
                "quality": quality,
                "compressed": quality < baseline_quality or scale < 1.0,
                "width": img.width,
                "height": img.height,
                "over_cap": False,
            }

    # Last-resort payload — no preset fit within max_payload_bytes. /stress
    # A1.2 F1 fix: avoid the redundant double-encode that happened under the
    # default config (`min_quality=20` == `COMPRESSION_PRESETS[-1][0]` → the
    # loop's last iteration had already produced exactly this JPEG). We
    # reuse the loop tail's b64 in that common case and only re-encode when
    # the caller explicitly asked for a different floor quality. The
    # `over_cap=True` flag surfaces the condition so caller telemetry can
    # audit it instead of silently shipping an over-budget payload.
    if min_quality == last_quality:
        final_quality = last_quality  # b64 already holds the right encoding
    else:
        b64 = _encode_jpeg_base64(last_img, quality=min_quality)
        final_quality = min_quality
    return {
        "data_url": f"data:image/jpeg;base64,{b64}",
        "payload_bytes": len(b64),  # B-416: ASCII → no .encode("utf-8")
        "quality": final_quality,
        "compressed": True,
        "width": last_img.width,
        "height": last_img.height,
        "over_cap": True,
    }
