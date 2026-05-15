from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import ImageDraw, ImageFont

logger = logging.getLogger(__name__)


@dataclass
class SomResult:
    som_text: str
    marked_image_path: Optional[str]
    marked_image: Optional[Any]  # PIL Image with bounding boxes drawn, None if unavailable
    degraded_som: bool
    mark_count: int


# /stress A1.4 F2 fix: explicit known-modes set so unknown / typo modes raise
# instead of silently falling through to the DOM-like branch. Previously a
# config typo like 'phantum_som' would silently run the DOM observation path
# while the runner's metadata recorded the typo'd condition_id, producing
# paper-grade data that mixed phantom-mode rows with secretly-DOM rows.
KNOWN_OBSERVATION_MODES = frozenset({
    "dom",
    "som",
    "vision",
    "phantom_som",
    "phantom_dom",     # legacy alias for phantom_text
    "phantom_text",
    "phantom_prompt",
})


def _extract_text_marks(obs_text: str, max_marks: Optional[int] = None) -> List[Dict[str, Any]]:
    # B-84: max_marks defaults to None (no cap). The former 200 cap fired on
    # ~0.03% of steps but only on marks modes — an axis-1 asymmetry vs the
    # uncapped AXTree modes. The viewport filter bounds element count in practice.
    marks: List[Dict[str, Any]] = []
    for line in (obs_text or "").splitlines():
        m = re.search(r"\[(\d+)\]", line)
        if not m:
            continue
        eid = int(m.group(1))
        label = re.sub(r"\[(\d+)\]", "", line).strip()
        marks.append({"id": eid, "label": label})
        if max_marks is not None and len(marks) >= max_marks:
            break
    return marks


def build_som_text_from_obs_text(obs_text: str, max_marks: Optional[int] = None) -> str:
    """Canonical [SOM_MARKS] text builder from an AXTree obs_text string.

    SINGLE SOURCE OF TRUTH for SoM text construction. `_build_som_result`
    (production agent path) and every mechanistic extractor MUST call this so
    the hidden-state NPZ / patching SoM text is byte-identical to the SoM text
    the deployed agent saw.

    Includes the `_options_map` recovery pass: `_inject_select_options` /
    `_inject_css_dropdown_options` write `[OPTIONS]` / `[DROPDOWN OPTIONS]`
    lines into obs_text, but `_extract_text_marks` strips them (they carry no
    `[N]` element id). This re-scans obs_text and re-attaches them under their
    owning mark.

    History: phantom-mode mechanistic extractors (run_stage4_*) re-implemented
    a marks-only builder that DROPPED this options recovery — master bug
    catalog B-82. The dropdown fix (B-06 / §51) predates phantom modes, so the
    omission was a phantom-design oversight, not a B-06 regression.
    """
    text_marks = _extract_text_marks(obs_text, max_marks=max_marks)
    if len(text_marks) == 0:
        return "[SOM_MARKS]\n[/SOM_MARKS]"
    _options_map: Dict[int, str] = {}
    _obs_lines = (obs_text or "").splitlines()
    for _i, _line in enumerate(_obs_lines):
        _m = re.search(r"\[(\d+)\]", _line)
        if not _m:
            continue
        _eid = int(_m.group(1))
        for _j in range(_i + 1, min(_i + 3, len(_obs_lines))):
            _stripped = _obs_lines[_j].strip()
            if _stripped.startswith("[OPTIONS") or _stripped.startswith("[DROPDOWN OPTIONS"):
                _options_map[_eid] = _stripped
                break
            if re.search(r"\[(\d+)\]", _obs_lines[_j]):
                break  # next element reached
    mark_lines = []
    for _mark in text_marks:
        _entry = f"[id={_mark['id']}] {_mark['label']}"
        if _mark["id"] in _options_map:
            _entry += f"\n    {_options_map[_mark['id']]}"
        mark_lines.append(_entry)
    return "\n".join(["[SOM_MARKS]"] + mark_lines + ["[/SOM_MARKS]"])


def _collect_bbox_map(raw: Any, bbox_map: Dict[int, List[float]]) -> None:
    if isinstance(raw, dict):
        maybe_id = None
        for id_key in ("id", "node_id", "nodeId", "element_id"):
            if id_key in raw:
                try:
                    maybe_id = int(raw[id_key])
                    break
                except Exception:
                    logger.debug("Failed to parse element ID from key %s=%r", id_key, raw[id_key])
                    maybe_id = None

        bbox = None
        for bbox_key in ("bbox", "bounding_box", "bounds", "rect"):
            if bbox_key in raw and isinstance(raw[bbox_key], (list, tuple)) and len(raw[bbox_key]) == 4:
                bbox = [float(x) for x in raw[bbox_key]]
                break

        if maybe_id is not None and bbox is not None:
            bbox_map[maybe_id] = bbox

        for v in raw.values():
            _collect_bbox_map(v, bbox_map)
    elif isinstance(raw, list):
        for v in raw:
            _collect_bbox_map(v, bbox_map)


_FONT_CACHE: Dict[int, Any] = {}

_CANDIDATE_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Windows/Fonts/arialbd.ttf",
]


def _get_font(size: int = 14) -> Any:
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    font = None
    for path in _CANDIDATE_FONTS:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                break
            except Exception:
                continue
    if font is None:
        try:
            font = ImageFont.load_default(size=size)  # Pillow >= 9.2.0
        except TypeError:
            font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def _draw_label(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    text: str,
    font: Any,
    bg_color: str = "#00BCD4",
    fg_color: str = "white",
    pad: int = 2,
) -> None:
    """Draw a filled pill-shaped label with white text at (x, y)."""
    try:
        bb = font.getbbox(text)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
    except AttributeError:
        tw, th = 8 * len(text), 12
    label_h = th + 2 * pad
    rx0 = int(x)
    # Place label above the bbox; fall back to inside if near top edge
    ry0 = int(y) - label_h - 1 if int(y) - label_h - 1 >= 0 else int(y)
    rx1, ry1 = rx0 + tw + 2 * pad, ry0 + label_h
    draw.rectangle([rx0, ry0, rx1, ry1], fill=bg_color)
    draw.text((rx0 + pad, ry0 + pad), text, fill=fg_color, font=font)


def _normalize_bbox(bbox: List[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = bbox
    # If normalized values in [0, 1], scale to pixels.
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    return [x1, y1, x2, y2]


def prepare_observation_for_mode(
    obs: Any,
    mode: str,
    artifact_dir: Path,
    step_idx: int,
) -> SomResult:
    """Prepare observation representation for the given observation mode.

    mode == "dom":            Full AXTree text, no image (consistent: prompt expects AXTree, gets it).
    mode == "som":            SOM_MARKS compressed index + marked image (consistent SoM).
    mode == "phantom_som":    SOM_MARKS index, NO image (P-SoM: image-mismatched — prompt
                              promises screenshot but agent gets none).
    mode == "phantom_dom"
       or "phantom_text":     SOM_MARKS index, NO image, but DOM-prompt (P-text: text-mismatched —
                              prompt expects AXTree, agent receives [SOM_MARKS]).
                              phantom_dom is the legacy mode value (paper-grade run dirs use it);
                              phantom_text is the current name. Both dispatch identically.
    mode == "phantom_prompt": Full AXTree text, NO image, with SoM-prompt (P-prompt: text+image
                              mismatched — symmetric counterpart of phantom_text across the
                              prompt × text axes; isolates prompt swap effect on AXTree text).
    mode == "vision":         Empty text, raw screenshot as image.
    """
    # /stress A1.4 F2: strict mode validation. Previously the `if mode != "som"`
    # fallback at the bottom silently accepted any string and behaved like DOM,
    # producing paper-grade data with secretly-DOM rows under a typo'd
    # condition_id label. The agent layer also has `.get(mode, dom_default)`
    # secondary silent-fallback — this raise closes both layers at the entry.
    if mode not in KNOWN_OBSERVATION_MODES:
        raise ValueError(
            f"Unknown observation_mode {mode!r}; expected one of "
            f"{sorted(KNOWN_OBSERVATION_MODES)}"
        )

    obs_text = getattr(obs, "text", "") or ""

    if mode == "vision":
        return SomResult(
            som_text="",
            marked_image_path=None,
            marked_image=getattr(obs, "image", None),
            degraded_som=False,
            mark_count=0,
        )

    if mode in ("phantom_som", "phantom_dom", "phantom_text"):
        # phantom_som: SoM prompt + [SOM_MARKS] text + no image (image-mismatched)
        # phantom_dom / phantom_text: DOM prompt + [SOM_MARKS] text + no image (text-mismatched)
        # Obs construction identical — only system prompt differs (handled in agent).
        result = _build_som_result(obs, obs_text, artifact_dir, step_idx)
        return SomResult(
            som_text=result.som_text,
            marked_image_path=result.marked_image_path,  # keep artifact for inspection
            marked_image=None,                           # model receives no image
            degraded_som=result.degraded_som,
            mark_count=result.mark_count,
        )

    if mode == "phantom_prompt":
        # P-prompt: AXTree text (same as DOM mode) + no image, but SoM prompt (set in agent).
        # Symmetric ablation of phantom_text: only the prompt axis is swapped from DOM.
        return SomResult(som_text=obs_text, marked_image_path=None, marked_image=None,
                         degraded_som=False, mark_count=0)

    if mode != "som":
        # "dom" mode or any unknown mode — full AXTree, no image
        return SomResult(som_text=obs_text, marked_image_path=None, marked_image=None, degraded_som=False, mark_count=0)

    # --- "som" mode: SOM_MARKS compressed index + marked image ---
    return _build_som_result(obs, obs_text, artifact_dir, step_idx)


def _build_som_result(
    obs: Any,
    obs_text: str,
    artifact_dir: Path,
    step_idx: int,
    include_full_axtree: bool = False,
) -> SomResult:
    """Core SOM logic: build SOM_MARKS index + marked image.

    Args:
        include_full_axtree: If True, appends the full AXTree after SOM_MARKS
            (legacy behavior). If False (new SOM mode), only SOM_MARKS is returned.
    """
    text_marks = _extract_text_marks(obs_text)
    # Only zero marks is a hard SOM fallback.
    # A single mark can still be a valid interactive page and should not be
    # forced into text-only degradation.
    if len(text_marks) == 0:
        # Do NOT leak the full AXTree into SOM mode — the model uses a SOM-specific
        # system prompt and expects [SOM_MARKS] format. An empty block signals
        # "no interactive elements detected" while keeping prompt/input consistent.
        # Fall back to the raw (unmarked) screenshot so the model can still use vision.
        return SomResult(
            som_text="[SOM_MARKS]\n[/SOM_MARKS]",
            marked_image_path=None,
            marked_image=getattr(obs, "image", None),
            degraded_som=True,
            mark_count=0,
        )

    # SoM text construction delegated to the canonical single-source builder
    # (master bug B-82 fix, 2026-05-14). build_som_text_from_obs_text does the
    # _extract_text_marks + _options_map recovery + [SOM_MARKS] wrapper; every
    # mechanistic extractor shares it so NPZ / patching SoM text is byte-
    # identical to this production path. text_marks is still computed above for
    # the empty-marks degradation check + bbox drawing + mark_count below.
    som_header = build_som_text_from_obs_text(obs_text)
    som_text = f"{som_header}\n\n{obs_text}" if include_full_axtree else som_header

    bbox_map: Dict[int, List[float]] = {}

    # Prefer VWA's obs_nodes_info (populated from CDP via observation_metadata).
    # Each entry: str(element_id) -> {"union_bound": [x, y, width, height], ...}
    # Convert [x, y, w, h] → [x1, y1, x2, y2] for _normalize_bbox.
    obs_nodes_info = getattr(obs, "obs_nodes_info", None)
    if obs_nodes_info:
        for node_id_str, node_info in obs_nodes_info.items():
            try:
                eid = int(node_id_str)
                ub = node_info.get("union_bound") if isinstance(node_info, dict) else None
                if ub and len(ub) == 4 and all(v is not None for v in ub):
                    x, y, w, h = (float(v) for v in ub)
                    if w > 0 and h > 0:
                        bbox_map[eid] = [x, y, x + w, y + h]
            except (ValueError, TypeError):
                continue
    else:
        # Fallback: attempt to collect bboxes from raw obs dict (legacy path).
        raw = getattr(obs, "raw", None)
        _collect_bbox_map(raw, bbox_map)

    image = getattr(obs, "image", None)
    marked_image_path: Optional[str] = None
    marked_image: Optional[Any] = None

    if image is not None and bbox_map:
        try:
            drawn = image.copy()
            draw = ImageDraw.Draw(drawn)
            width, height = drawn.size
            font = _get_font(size=14)
            for mark in text_marks:
                bbox = bbox_map.get(mark["id"])
                if not bbox:
                    continue
                x1, y1, x2, y2 = _normalize_bbox(bbox, width, height)
                draw.rectangle([x1, y1, x2, y2], outline="#00BCD4", width=2)
                _draw_label(draw, x1, y1, str(mark["id"]), font)

            marked_image = drawn  # PIL Image passed to the model

            som_dir = artifact_dir / "som"
            som_dir.mkdir(parents=True, exist_ok=True)
            marked_image_path = str(som_dir / f"step_{step_idx:03d}_som.png")
            drawn.save(marked_image_path)
        except Exception:
            logger.warning("SOM image rendering failed; degrading to unmarked screenshot", exc_info=True)
            marked_image_path = None
            marked_image = None

    # degraded_som=True here means marks were extracted but image rendering failed.
    # Zero-mark complete degradation is handled in the early return above.
    degraded = marked_image is None
    return SomResult(
        som_text=som_text,
        marked_image_path=marked_image_path,
        marked_image=marked_image,
        degraded_som=degraded,
        mark_count=len(text_marks),
    )


def apply_som(
    obs: Any,
    som_on: bool,
    artifact_dir: Path,
    step_idx: int,
) -> SomResult:
    """Backward-compatible function. Prefer prepare_observation_for_mode for new code."""
    obs_text = getattr(obs, "text", "") or ""
    if not som_on:
        return SomResult(som_text=obs_text, marked_image_path=None, marked_image=None, degraded_som=False, mark_count=0)
    return _build_som_result(obs, obs_text, artifact_dir, step_idx, include_full_axtree=True)
