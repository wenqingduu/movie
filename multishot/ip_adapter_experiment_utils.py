"""Shared utilities for IP-Adapter trajectory-residual experiments."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = (
    "cinematic medium close-up portrait of a man standing beneath warm neon lights "
    "on a rainy night street, three-quarter view, natural skin texture, shallow depth "
    "of field, photorealistic, subtle rim light"
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _face_app():
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="antelopev2",
        root=str(PROJECT_ROOT / "third_party" / "PuLID"),
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def _largest_face(app, image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    faces = app.get(image)
    if not faces:
        return None
    return max(
        faces,
        key=lambda face: float(
            (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
        ),
    )


def _face_record(face, width: int, height: int) -> dict:
    if face is None:
        return {
            "face_id": "face_0",
            "face_bbox": [
                int(width * 0.33),
                int(height * 0.16),
                int(width * 0.67),
                int(height * 0.68),
            ],
            "pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
            "detection_source": "fallback_center_bbox",
        }
    pose = getattr(face, "pose", [0.0, 0.0, 0.0])
    return {
        "face_id": "face_0",
        "face_bbox": [round(float(value), 2) for value in face.bbox],
        "pose": {
            "pitch": round(float(pose[0]), 4),
            "yaw": round(float(pose[1]), 4),
            "roll": round(float(pose[2]), 4),
        },
        "detection_source": "antelopev2",
    }


def _embedding(face):
    if face is None:
        return None
    return np.asarray(face.normed_embedding, dtype=np.float32)


def _cosine(left, right):
    if left is None or right is None:
        return None
    return round(float(np.dot(left, right)), 6)


def _pixel_delta(baseline_path: Path, candidate_path: Path, bbox: list[float]) -> dict:
    baseline = np.asarray(Image.open(baseline_path).convert("RGB"), dtype=np.float32)
    candidate = np.asarray(Image.open(candidate_path).convert("RGB"), dtype=np.float32)
    delta = np.abs(candidate - baseline)
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(delta.shape[1], x2), min(delta.shape[0], y2)
    face_delta = delta[y1:y2, x1:x2]
    return {
        "whole_image_rgb_mae_vs_baseline": round(float(delta.mean()), 6),
        "target_face_bbox_rgb_mae_vs_baseline": (
            round(float(face_delta.mean()), 6) if face_delta.size else None
        ),
    }


def _make_contact_sheet(items: list[tuple[str, Path]], output_path: Path) -> None:
    thumb_size = (420, 420)
    label_height = 42
    columns = 3
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * thumb_size[0], rows * (thumb_size[1] + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for index, (label, path) in enumerate(items):
        image = Image.open(path).convert("RGB")
        image.thumbnail(thumb_size, Image.Resampling.LANCZOS)
        x = (index % columns) * thumb_size[0] + (thumb_size[0] - image.width) // 2
        y = (index // columns) * (thumb_size[1] + label_height)
        sheet.paste(image, (x, y))
        draw.text(
            ((index % columns) * thumb_size[0] + 8, y + thumb_size[1] + 10),
            label,
            fill="black",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _compact_state(state: dict) -> dict:
    return {key: value for key, value in state.items() if not key.startswith("_")}
