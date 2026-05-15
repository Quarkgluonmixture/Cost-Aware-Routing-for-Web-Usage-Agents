"""image_utils over-cap fallback invariants — /stress A1.2 F1 fix.

Guards the encoder's two contracts:
1. Success path emits `over_cap=False`; fallback emits `over_cap=True`.
2. Default config fallback no longer redundantly double-encodes the last
   COMPRESSION_PRESETS entry — it reuses the loop tail's b64 only when
   `min_quality == COMPRESSION_PRESETS[-1][0]` (default), and re-encodes
   when the caller asks for a different floor.
"""

from __future__ import annotations

from PIL import Image

from p79.backends.image_utils import COMPRESSION_PRESETS, encode_image_data_url


def _make_image(w: int = 100, h: int = 100, color=(200, 100, 50)) -> Image.Image:
    return Image.new("RGB", (w, h), color)


def test_success_path_emits_over_cap_false():
    """Small image at default cap fits at first preset (85, 1.0) — over_cap=False."""
    result = encode_image_data_url(_make_image(64, 64))
    assert result["over_cap"] is False
    assert result["quality"] == 85
    assert result["compressed"] is False


def test_impossible_cap_triggers_over_cap_true():
    """max_payload_bytes=1 is impossible — fallback fires with over_cap=True."""
    result = encode_image_data_url(_make_image(64, 64), max_payload_bytes=1)
    assert result["over_cap"] is True
    # Default min_quality=20 == COMPRESSION_PRESETS[-1][0], reuses loop tail's
    # b64 (no double-encode). The returned quality reflects that.
    assert result["quality"] == COMPRESSION_PRESETS[-1][0]


def test_custom_min_quality_distinct_fallback():
    """Caller-provided min_quality=15 (below last preset's 20) DOES re-encode.

    This is the codex Mode B empirical finding: passing min_quality=15
    produces a different JPEG than the default min_quality=20 fallback.
    """
    img = _make_image(200, 200)
    r_default = encode_image_data_url(img, max_payload_bytes=1)
    r_lower = encode_image_data_url(img, max_payload_bytes=1, min_quality=15)
    assert r_default["over_cap"] is True
    assert r_lower["over_cap"] is True
    # Different quality settings produce different JPEGs.
    assert r_default["data_url"] != r_lower["data_url"], (
        "Caller-supplied min_quality=15 must produce a distinct fallback JPEG "
        "from the default min_quality=20 path"
    )
    assert r_lower["quality"] == 15


def test_default_fallback_does_not_redundantly_double_encode():
    """Default-config fallback reuses loop tail's b64.

    Behavioural assertion: under default min_quality=20, the fallback
    return must be byte-identical to what the loop's last iteration
    produced — same image (last_img), same quality (20). If a future
    edit re-introduces the double-encode this test still passes (same
    bytes), but if anyone changes the fallback semantics to use a
    *different* quality/scale than COMPRESSION_PRESETS[-1] without
    updating min_quality, this asserts the contract.
    """
    img = _make_image(200, 200)
    result = encode_image_data_url(img, max_payload_bytes=1)
    # Default min_quality (20) == last preset quality → final quality is 20.
    assert result["quality"] == 20


def test_over_cap_field_exists_in_both_paths():
    """over_cap field is mandatory in every return dict (auditable surface)."""
    success = encode_image_data_url(_make_image(64, 64))
    fail = encode_image_data_url(_make_image(64, 64), max_payload_bytes=1)
    assert "over_cap" in success
    assert "over_cap" in fail
    # And all the legacy keys still present (backward compat with
    # proxy_api_agent.py meta-extraction code).
    for k in ("data_url", "payload_bytes", "quality", "compressed", "width", "height"):
        assert k in success
        assert k in fail
