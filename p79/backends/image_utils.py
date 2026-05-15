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
import PIL
_PIL_VERSION_PARTS = tuple(int(x) for x in PIL.__version__.split(".")[:2])
assert _PIL_VERSION_PARTS >= (10, 0), (
    f"PIL/Pillow version too old for paper-grade reproducibility: "
    f"got {PIL.__version__}, need ≥10.0 (see pyproject.toml + B-155)"
)


# Adapted from external_code/image.py (Aiden Yiliu Li, Apache-2.0)
DEFAULT_MAX_IMAGE_PAYLOAD_BYTES = 5 * 1024 * 1024

# (quality, scale) presets tried in order; first that fits within max_payload_bytes wins.
COMPRESSION_PRESETS = [
    (85, 1.0), (70, 1.0), (55, 0.9), (40, 0.8),
    (30, 0.7), (25, 0.6), (20, 0.5), (20, 0.4),
]


def _normalize_image(image: Image.Image) -> Image.Image:
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
    quality_start: int = 85,
    min_quality: int = 20,
) -> Dict[str, Any]:
    normalized = _normalize_image(image)
    last_img = normalized

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
        if len(b64.encode("utf-8")) <= max_payload_bytes:
            return {
                "data_url": f"data:image/jpeg;base64,{b64}",
                "payload_bytes": len(b64.encode("utf-8")),
                "quality": quality,
                "compressed": quality < quality_start or scale < 1.0,
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
        "payload_bytes": len(b64.encode("utf-8")),
        "quality": final_quality,
        "compressed": True,
        "width": last_img.width,
        "height": last_img.height,
        "over_cap": True,
    }
