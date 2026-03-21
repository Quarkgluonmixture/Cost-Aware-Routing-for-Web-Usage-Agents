from __future__ import annotations

import base64
import io
from typing import Any, Dict

from PIL import Image


# Adapted from external_code/image.py (Aiden Yiliu Li, Apache-2.0)
DEFAULT_MAX_IMAGE_PAYLOAD_BYTES = 5 * 1024 * 1024


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

    for quality in range(quality_start, min_quality - 1, -10):
        b64 = _encode_jpeg_base64(normalized, quality=quality)
        if len(b64.encode("utf-8")) <= max_payload_bytes:
            return {
                "data_url": f"data:image/jpeg;base64,{b64}",
                "payload_bytes": len(b64.encode("utf-8")),
                "quality": quality,
                "compressed": quality < quality_start,
                "width": normalized.width,
                "height": normalized.height,
            }

    current = normalized
    scale = 0.9
    while scale >= 0.3:
        target_size = (max(1, int(normalized.width * scale)), max(1, int(normalized.height * scale)))
        current = normalized.resize(target_size, Image.Resampling.LANCZOS)
        b64 = _encode_jpeg_base64(current, quality=70)
        if len(b64.encode("utf-8")) <= max_payload_bytes:
            return {
                "data_url": f"data:image/jpeg;base64,{b64}",
                "payload_bytes": len(b64.encode("utf-8")),
                "quality": 70,
                "compressed": True,
                "width": current.width,
                "height": current.height,
            }
        scale -= 0.1

    # Last-resort payload, even if above target limit.
    b64 = _encode_jpeg_base64(current, quality=min_quality)
    return {
        "data_url": f"data:image/jpeg;base64,{b64}",
        "payload_bytes": len(b64.encode("utf-8")),
        "quality": min_quality,
        "compressed": True,
        "width": current.width,
        "height": current.height,
    }
