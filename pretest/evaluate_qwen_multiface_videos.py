#!/usr/bin/env python3
"""Evaluate per-character InsightFace identity in paired one-to-many videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from multishot.ip_adapter_experiment_utils import _face_app
from multishot.multi_face_reference import detected_faces


def _embedding(face) -> np.ndarray:
    value = np.asarray(face.embedding, dtype=np.float32)
    return value / (np.linalg.norm(value) + 1e-8)


def _largest_reference_face(app, path: Path, confidence: float):
    faces = detected_faces(app, Image.open(path).convert("RGB"), confidence)
    if not faces:
        raise RuntimeError(f"no reference face: {path}")
    return max(
        faces,
        key=lambda face: float(
            (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
        ),
    )


def _summarize(values: list[float | None]) -> dict:
    detected = np.asarray([value for value in values if value is not None], dtype=np.float32)
    if not len(detected):
        return {
            "decoded_frame_count": len(values),
            "detected_frame_count": 0,
            "detection_coverage": 0.0,
        }
    indices = np.asarray(
        [index for index, value in enumerate(values) if value is not None], dtype=np.float32
    )
    slope = float(np.polyfit(indices, detected, 1)[0]) if len(detected) > 1 else 0.0
    return {
        "decoded_frame_count": len(values),
        "detected_frame_count": int(len(detected)),
        "detection_coverage": float(len(detected) / len(values)),
        "first": float(detected[0]),
        "mean": float(np.mean(detected)),
        "median": float(np.median(detected)),
        "p10": float(np.percentile(detected, 10)),
        "minimum": float(np.min(detected)),
        "last_detected": float(detected[-1]),
        "regression_slope_per_frame": slope,
        "regression_change_over_full_video": slope * max(0, len(values) - 1),
    }


def _evaluate_video(
    app,
    video_path: Path,
    ordered_targets: list[dict],
    reference_embeddings: dict[str, np.ndarray],
    confidence: float,
    initial_canvas_width: float,
) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    values = {target["name"]: [] for target in ordered_targets}
    initial_centers = {
        target["name"]: (float(target["bbox"][0]) + float(target["bbox"][2])) / 2.0
        for target in ordered_targets
    }
    decoded = 0
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        decoded += 1
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        faces = detected_faces(app, Image.fromarray(rgb), confidence)
        if len(faces) > len(ordered_targets):
            faces = sorted(
                faces,
                key=lambda face: float(
                    (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
                ),
                reverse=True,
            )[: len(ordered_targets)]
        faces = sorted(faces, key=lambda face: float((face.bbox[0] + face.bbox[2]) / 2.0))
        assigned: dict[str, object] = {}
        if len(faces) == len(ordered_targets):
            assigned = {
                target["name"]: face for target, face in zip(ordered_targets, faces)
            }
        elif faces:
            # Preserve role identity by position rather than using identity
            # embeddings to choose the face being evaluated.  When only a
            # subset is detected, solve the small spatial assignment greedily.
            remaining = list(ordered_targets)
            for face in faces:
                normalized_center = (
                    float(face.bbox[0] + face.bbox[2]) / 2.0 / bgr.shape[1]
                )
                target = min(
                    remaining,
                    key=lambda item: abs(
                        initial_centers[item["name"]] / initial_canvas_width
                        - normalized_center
                    ),
                )
                assigned[target["name"]] = face
                remaining.remove(target)
        for target in ordered_targets:
            face = assigned.get(target["name"])
            values[target["name"]].append(
                None
                if face is None
                else float(np.dot(reference_embeddings[target["name"]], _embedding(face)))
            )
    capture.release()
    if decoded == 0:
        raise RuntimeError(f"decoded 0 frames: {video_path}")
    return {
        "video": str(video_path.resolve()),
        "characters": {name: _summarize(scores) for name, scores in values.items()},
        "frame_values": values,
    }


def run(args: argparse.Namespace, *, app=None) -> dict:
    prepared = json.loads(args.prepared.read_text(encoding="utf-8"))
    targets = sorted(prepared["targets"], key=lambda item: float(item["bbox"][0]))
    if not targets:
        raise ValueError("prepared reference contains no targets")
    app = app or _face_app()
    reference_embeddings = {
        target["name"]: _embedding(
            _largest_reference_face(
                app, Path(target["reference_image"]), args.min_face_confidence
            )
        )
        for target in targets
    }
    initial_canvas_width = float(prepared["control_size"][0])
    conditions = {
        "control": _evaluate_video(
            app,
            args.control,
            targets,
            reference_embeddings,
            args.min_face_confidence,
            initial_canvas_width,
        ),
        "treatment": _evaluate_video(
            app,
            args.treatment,
            targets,
            reference_embeddings,
            args.min_face_confidence,
            initial_canvas_width,
        ),
    }
    deltas = {}
    for target in targets:
        name = target["name"]
        control = conditions["control"]["characters"][name]
        treatment = conditions["treatment"]["characters"][name]
        deltas[name] = {
            metric: treatment.get(metric) - control.get(metric)
            for metric in (
                "first",
                "mean",
                "median",
                "p10",
                "minimum",
                "last_detected",
                "detection_coverage",
                "regression_change_over_full_video",
            )
            if control.get(metric) is not None and treatment.get(metric) is not None
        }
    mean_deltas = [value["mean"] for value in deltas.values() if "mean" in value]
    result = {
        "kind": "qwen_image21_multiface_paired_video_insightface_evaluation",
        "assignment_policy": (
            "spatial left-to-right when all target faces are detected; "
            "otherwise greedy nearest normalized initial x-position"
        ),
        "prepared": str(args.prepared.resolve()),
        "conditions": conditions,
        "treatment_minus_control": deltas,
        "macro": {
            "mean_paired_delta": float(np.mean(mean_deltas)) if mean_deltas else None,
            "positive_character_count": sum(value > 0 for value in mean_deltas),
            "character_count": len(targets),
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
