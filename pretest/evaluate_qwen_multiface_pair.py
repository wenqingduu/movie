#!/usr/bin/env python3
"""Evaluate per-character InsightFace identity for a paired multi-face image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from multishot.ip_adapter_experiment_utils import _face_app
from multishot.multi_face_reference import detected_faces


def _embedding(face) -> np.ndarray:
    value = np.asarray(face.embedding, dtype=np.float32)
    return value / (np.linalg.norm(value) + 1e-8)


def _bbox(face) -> list[float]:
    return [float(value) for value in face.bbox]


def _iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _match_by_bbox(targets: list[dict], faces: list) -> list:
    if len(faces) < len(targets):
        raise RuntimeError(f"detected {len(faces)} faces, expected at least {len(targets)}")
    remaining = set(range(len(faces)))
    matched = []
    for target in targets:
        target_bbox = [float(value) for value in target["bbox"]]
        index = max(remaining, key=lambda item: _iou(target_bbox, _bbox(faces[item])))
        overlap = _iou(target_bbox, _bbox(faces[index]))
        if overlap < 0.25:
            raise RuntimeError(
                f"{target['name']}: best detected-face IoU {overlap:.3f} is unreliable"
            )
        matched.append((faces[index], overlap))
        remaining.remove(index)
    return matched


def run(args: argparse.Namespace, *, app=None) -> dict:
    prepared = json.loads(args.prepared.read_text(encoding="utf-8"))
    targets = prepared["targets"]
    app = app or _face_app()
    images = {
        "control": Image.open(args.control).convert("RGB"),
        "treatment": Image.open(args.treatment).convert("RGB"),
    }
    matches = {
        condition: _match_by_bbox(
            targets, detected_faces(app, image, args.min_face_confidence)
        )
        for condition, image in images.items()
    }

    characters = []
    for index, target in enumerate(targets):
        reference = Image.open(target["reference_image"]).convert("RGB")
        reference_faces = detected_faces(app, reference, args.min_face_confidence)
        if not reference_faces:
            raise RuntimeError(f"{target['name']}: no face in reference image")
        reference_face = max(
            reference_faces,
            key=lambda face: float(
                (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
            ),
        )
        reference_embedding = _embedding(reference_face)
        scores = {}
        detections = {}
        for condition in ("control", "treatment"):
            face, overlap = matches[condition][index]
            scores[condition] = float(np.dot(reference_embedding, _embedding(face)))
            detections[condition] = {
                "bbox": _bbox(face),
                "target_bbox_iou": overlap,
                "det_score": float(face.det_score),
            }
        characters.append(
            {
                "name": target["name"],
                "reference_image": target["reference_image"],
                "control_cosine": scores["control"],
                "treatment_cosine": scores["treatment"],
                "delta": scores["treatment"] - scores["control"],
                "detections": detections,
            }
        )

    result = {
        "kind": "qwen_image21_multiface_pair_insightface_evaluation",
        "prepared": str(args.prepared.resolve()),
        "control": str(args.control.resolve()),
        "treatment": str(args.treatment.resolve()),
        "characters": characters,
        "macro": {
            "control_mean": float(np.mean([item["control_cosine"] for item in characters])),
            "treatment_mean": float(
                np.mean([item["treatment_cosine"] for item in characters])
            ),
            "mean_paired_delta": float(np.mean([item["delta"] for item in characters])),
            "positive_character_count": sum(item["delta"] > 0 for item in characters),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-face-confidence", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
