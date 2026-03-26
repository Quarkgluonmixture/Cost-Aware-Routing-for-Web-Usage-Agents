from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import ImageDraw


@dataclass
class SomResult:
    som_text: str
    marked_image_path: Optional[str]
    degraded_som: bool
    mark_count: int


def _extract_text_marks(obs_text: str, max_marks: int = 80) -> List[Dict[str, Any]]:
    marks: List[Dict[str, Any]] = []
    for line in (obs_text or "").splitlines():
        m = re.search(r"\[(\d+)\]", line)
        if not m:
            continue
        eid = int(m.group(1))
        label = re.sub(r"\[(\d+)\]", "", line).strip()
        marks.append({"id": eid, "label": label})
        if len(marks) >= max_marks:
            break
    return marks


def _collect_bbox_map(raw: Any, bbox_map: Dict[int, List[float]]) -> None:
    if isinstance(raw, dict):
        maybe_id = None
        for id_key in ("id", "node_id", "nodeId", "element_id"):
            if id_key in raw:
                try:
                    maybe_id = int(raw[id_key])
                    break
                except Exception:
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


def _normalize_bbox(bbox: List[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = bbox
    # If normalized values in [0, 1], scale to pixels.
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    return [x1, y1, x2, y2]


def apply_som(
    obs: Any,
    som_on: bool,
    artifact_dir: Path,
    step_idx: int,
) -> SomResult:
    obs_text = getattr(obs, "text", "") or ""
    if not som_on:
        return SomResult(som_text=obs_text, marked_image_path=None, degraded_som=False, mark_count=0)

    text_marks = _extract_text_marks(obs_text)
    if not text_marks:
        return SomResult(
            som_text=obs_text,
            marked_image_path=None,
            degraded_som=True,
            mark_count=0,
        )

    mark_lines = [f"[id={m['id']}] {m['label']}" for m in text_marks]
    som_header = "\n".join(["[SOM_MARKS]"] + mark_lines + ["[/SOM_MARKS]"])
    som_text = f"{som_header}\n\n{obs_text}"

    raw = getattr(obs, "raw", None)
    bbox_map: Dict[int, List[float]] = {}
    _collect_bbox_map(raw, bbox_map)

    image = getattr(obs, "image", None)
    marked_image_path: Optional[str] = None

    if image is not None and bbox_map:
        try:
            image = image.copy()
            draw = ImageDraw.Draw(image)
            width, height = image.size
            for mark in text_marks:
                bbox = bbox_map.get(mark["id"])
                if not bbox:
                    continue
                x1, y1, x2, y2 = _normalize_bbox(bbox, width, height)
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                draw.text((x1 + 2, y1 + 2), str(mark["id"]), fill="red")

            som_dir = artifact_dir / "som"
            som_dir.mkdir(parents=True, exist_ok=True)
            marked_image_path = str(som_dir / f"step_{step_idx:03d}_som.png")
            image.save(marked_image_path)
        except Exception:
            marked_image_path = None

    degraded = marked_image_path is None
    return SomResult(
        som_text=som_text,
        marked_image_path=marked_image_path,
        degraded_som=degraded,
        mark_count=len(text_marks),
    )
