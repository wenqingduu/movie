#!/usr/bin/env python3
"""Evaluate reference-anchored DINOv2 similarity for the Qwen full episodes.

Face selection follows the paired InsightFace evaluator: first-frame boxes are
reused from ``identity_metrics.json`` and video roles are assigned spatially,
without using identity similarity to choose the evaluated person.  DINOv2 only
embeds the resulting expanded face crops.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multishot.ip_adapter_experiment_utils import _face_app  # noqa: E402
from multishot.multi_face_reference import detected_faces  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _bbox(face) -> list[float]:
    return [float(value) for value in face.bbox]


def _crop(image: Image.Image, bbox: list[float], expansion: float) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = [float(value) for value in bbox]
    face_width = max(1.0, x2 - x1)
    face_height = max(1.0, y2 - y1)
    x1 = max(0, int(np.floor(x1 - face_width * expansion)))
    y1 = max(0, int(np.floor(y1 - face_height * expansion)))
    x2 = min(width, int(np.ceil(x2 + face_width * expansion)))
    y2 = min(height, int(np.ceil(y2 + face_height * expansion)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid crop bbox: {(x1, y1, x2, y2)}")
    return image.crop((x1, y1, x2, y2))


class DinoEncoder:
    def __init__(self, model: Path, device: str, batch_size: int):
        self.processor = AutoImageProcessor.from_pretrained(str(model), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(model), local_files_only=True)
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.batch_size = batch_size

    @torch.inference_mode()
    def encode(self, images: list[Image.Image]) -> np.ndarray:
        chunks = []
        for start in range(0, len(images), self.batch_size):
            inputs = self.processor(
                images=images[start : start + self.batch_size], return_tensors="pt"
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            output = self.model(**inputs)
            features = getattr(output, "pooler_output", None)
            if features is None:
                features = output.last_hidden_state[:, 0]
            chunks.append(F.normalize(features.float(), dim=-1).cpu().numpy())
        return np.concatenate(chunks, axis=0)


def _reference_embedding(
    app,
    encoder: DinoEncoder,
    path: Path,
    confidence: float,
    expansion: float,
    cache: dict[str, np.ndarray],
) -> np.ndarray:
    key = str(path.resolve())
    if key in cache:
        return cache[key]
    image = Image.open(path).convert("RGB")
    faces = detected_faces(app, image, confidence)
    if not faces:
        raise RuntimeError(f"no reference face: {path}")
    face = max(
        faces,
        key=lambda item: float(
            (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])
        ),
    )
    cache[key] = encoder.encode([_crop(image, _bbox(face), expansion)])[0]
    return cache[key]


def _video_crops(
    app,
    video_path: Path,
    ordered_targets: list[dict[str, Any]],
    confidence: float,
    expansion: float,
    initial_width: float,
) -> tuple[dict[str, list[Image.Image | None]], dict[str, list[list[float] | None]]]:
    crops = {target["name"]: [] for target in ordered_targets}
    boxes = {target["name"]: [] for target in ordered_targets}
    initial_centers = {
        target["name"]: (float(target["bbox"][0]) + float(target["bbox"][2])) / 2.0
        for target in ordered_targets
    }
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        faces = detected_faces(app, image, confidence)
        if len(faces) > len(ordered_targets):
            faces = sorted(
                faces,
                key=lambda face: float(
                    (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
                ),
                reverse=True,
            )[: len(ordered_targets)]
        faces = sorted(faces, key=lambda face: float(face.bbox[0] + face.bbox[2]))
        assigned: dict[str, Any] = {}
        if len(faces) == len(ordered_targets):
            assigned = {
                target["name"]: face for target, face in zip(ordered_targets, faces)
            }
        elif faces:
            remaining = list(ordered_targets)
            for face in faces:
                normalized_center = float(face.bbox[0] + face.bbox[2]) / 2.0 / image.width
                target = min(
                    remaining,
                    key=lambda item: abs(
                        initial_centers[item["name"]] / initial_width - normalized_center
                    ),
                )
                assigned[target["name"]] = face
                remaining.remove(target)
        for target in ordered_targets:
            name = target["name"]
            face = assigned.get(name)
            if face is None:
                crops[name].append(None)
                boxes[name].append(None)
            else:
                bbox = _bbox(face)
                crops[name].append(_crop(image, bbox, expansion))
                boxes[name].append(bbox)
    capture.release()
    if not next(iter(crops.values()), []):
        raise RuntimeError(f"decoded 0 frames: {video_path}")
    return crops, boxes


def _encode_optional_crops(
    encoder: DinoEncoder,
    reference: np.ndarray,
    crops: list[Image.Image | None],
) -> list[float | None]:
    indexes = [index for index, crop in enumerate(crops) if crop is not None]
    embeddings = encoder.encode([crops[index] for index in indexes]) if indexes else []
    scores: list[float | None] = [None] * len(crops)
    for index, embedding in zip(indexes, embeddings):
        scores[index] = float(np.dot(reference, embedding))
    return scores


def _video_summary(values: list[float | None]) -> dict[str, Any]:
    valid = [float(value) for value in values if value is not None]
    total = len(values)
    return {
        "decoded_frame_count": total,
        "scored_frame_count": len(valid),
        "detection_coverage": len(valid) / total if total else 0.0,
        "video_first": values[0] if values else None,
        "video_frame_mean": float(np.mean(valid)) if valid else None,
        "video_shot_median": float(np.median(valid)) if valid else None,
        "video_p10": float(np.percentile(valid, 10)) if valid else None,
        "video_minimum": float(np.min(valid)) if valid else None,
        "video_last": values[-1] if values else None,
    }


def _delta(control: dict[str, Any], treatment: dict[str, Any]) -> dict[str, float]:
    result = {}
    for key in ("first_frame", "video_first", "video_frame_mean", "video_shot_median", "video_p10", "video_minimum", "video_last", "detection_coverage"):
        if control.get(key) is not None and treatment.get(key) is not None:
            result[key] = float(treatment[key] - control[key])
    return result


def _evaluate_shot(
    shot_dir: Path,
    app,
    encoder: DinoEncoder,
    confidence: float,
    expansion: float,
    reference_cache: dict[str, np.ndarray],
) -> dict[str, Any]:
    prepared = _load(shot_dir / "prepared/prepared.json")
    targets = sorted(prepared["targets"], key=lambda item: float(item["bbox"][0]))
    insight = _load(shot_dir / "identity_metrics.json")
    image_detections = {item["name"]: item["detections"] for item in insight["characters"]}
    references = {
        target["name"]: _reference_embedding(
            app,
            encoder,
            Path(target["reference_image"]),
            confidence,
            expansion,
            reference_cache,
        )
        for target in targets
    }
    conditions: dict[str, Any] = {}
    for condition in ("control", "treatment"):
        first_image = Image.open(shot_dir / f"{condition}.png").convert("RGB")
        first_crops = {
            target["name"]: _crop(
                first_image,
                image_detections[target["name"]][condition]["bbox"],
                expansion,
            )
            for target in targets
        }
        first_embeddings = encoder.encode([first_crops[target["name"]] for target in targets])
        first_scores = {
            target["name"]: float(np.dot(references[target["name"]], embedding))
            for target, embedding in zip(targets, first_embeddings)
        }
        video_crops, video_boxes = _video_crops(
            app,
            shot_dir / f"videos/{condition}.mp4",
            targets,
            confidence,
            expansion,
            float(prepared["control_size"][0]),
        )
        characters = {}
        for target in targets:
            name = target["name"]
            values = _encode_optional_crops(
                encoder, references[name], video_crops[name]
            )
            characters[name] = {
                "first_frame": first_scores[name],
                **_video_summary(values),
                "frame_values": values,
                "frame_bboxes": video_boxes[name],
            }
        conditions[condition] = {
            "image": str((shot_dir / f"{condition}.png").resolve()),
            "video": str((shot_dir / f"videos/{condition}.mp4").resolve()),
            "characters": characters,
        }
    deltas = {
        target["name"]: _delta(
            conditions["control"]["characters"][target["name"]],
            conditions["treatment"]["characters"][target["name"]],
        )
        for target in targets
    }
    return {
        "kind": "qwen_image21_reference_anchored_dinov2_evaluation",
        "methodology": {
            "crop": f"InsightFace bbox expanded by {expansion:.2f} face widths/heights",
            "first_frame_assignment": "same bbox selected by paired InsightFace evaluator",
            "video_assignment": "spatial role tracking; identity is not used for selection",
            "embedding": "DINOv2 normalized CLS cosine to original identity portrait crop",
        },
        "prepared": str((shot_dir / "prepared/prepared.json").resolve()),
        "conditions": conditions,
        "treatment_minus_control": deltas,
        "macro": {
            "first_frame_mean_paired_delta": statistics.mean(
                value["first_frame"] for value in deltas.values()
            ),
            "video_frame_mean_paired_delta": statistics.mean(
                value["video_frame_mean"] for value in deltas.values()
            ),
            "character_count": len(targets),
        },
    }


def _aggregate(shots: list[dict[str, Any]]) -> dict[str, Any]:
    first_pairs = [
        (
            shot["dino_metrics"]["conditions"]["control"]["characters"][name]["first_frame"],
            shot["dino_metrics"]["conditions"]["treatment"]["characters"][name]["first_frame"],
        )
        for shot in shots
        for name in shot["dino_metrics"]["treatment_minus_control"]
    ]
    video_pairs = [
        (
            shot["dino_metrics"]["conditions"]["control"]["characters"][name]["video_frame_mean"],
            shot["dino_metrics"]["conditions"]["treatment"]["characters"][name]["video_frame_mean"],
        )
        for shot in shots
        for name in shot["dino_metrics"]["treatment_minus_control"]
    ]
    first = [treatment - control for control, treatment in first_pairs]
    video = [treatment - control for control, treatment in video_pairs]
    return {
        "evaluated_shot_count": len(shots),
        "evaluated_character_count": len(first),
        "first_frame_control_mean": statistics.mean(control for control, _ in first_pairs)
        if first_pairs
        else None,
        "first_frame_treatment_mean": statistics.mean(treatment for _, treatment in first_pairs)
        if first_pairs
        else None,
        "first_frame_mean_paired_delta": statistics.mean(first) if first else None,
        "first_frame_positive_character_count": sum(value > 0 for value in first),
        "video_frame_control_mean": statistics.mean(control for control, _ in video_pairs)
        if video_pairs
        else None,
        "video_frame_treatment_mean": statistics.mean(treatment for _, treatment in video_pairs)
        if video_pairs
        else None,
        "video_frame_mean_paired_delta": statistics.mean(video) if video else None,
        "video_positive_character_count": sum(value > 0 for value in video),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/entitybench_qwen_image21_v7_full3",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/root/autodl-tmp/models/dinov2-base"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--min-face-confidence", type=float, default=0.5)
    parser.add_argument("--face-expansion", type=float, default=0.35)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = _load(args.output_root / "manifest.json")
    pending = [
        record
        for record in manifest["shots"]
        if (Path(record["shot_dir"]) / "prepared/prepared.json").is_file()
        and (
            args.overwrite
            or not (Path(record["shot_dir"]) / "dino_identity_metrics.json").is_file()
        )
    ]
    app = _face_app() if pending else None
    encoder = DinoEncoder(args.model, args.device, args.batch_size) if pending else None
    reference_cache: dict[str, np.ndarray] = {}
    rows = []
    failures = []
    for index, record in enumerate(manifest["shots"], 1):
        shot_dir = Path(record["shot_dir"])
        output = shot_dir / "dino_identity_metrics.json"
        if not (shot_dir / "prepared/prepared.json").is_file():
            print(f"[{index}/{len(manifest['shots'])}] skip unprepared {record['run']} {record['shot_key']}")
            continue
        try:
            if output.is_file() and not args.overwrite:
                metrics = _load(output)
            else:
                assert app is not None and encoder is not None
                metrics = _evaluate_shot(
                    shot_dir,
                    app,
                    encoder,
                    args.min_face_confidence,
                    args.face_expansion,
                    reference_cache,
                )
                _write(output, metrics)
        except Exception as exc:
            failure = {
                "run": record["run"],
                "shot_key": record["shot_key"],
                "error": repr(exc),
            }
            failures.append(failure)
            print(
                f"[{index}/{len(manifest['shots'])}] failed "
                f"{record['run']} {record['shot_key']}: {exc!r}",
                flush=True,
            )
            continue
        rows.append({
            "run": record["run"],
            "shot_key": record["shot_key"],
            "injection_applied": _load(shot_dir / "result.json").get("injection_applied", False),
            "dino_metrics": metrics,
        })
        print(f"[{index}/{len(manifest['shots'])}] evaluated {record['run']} {record['shot_key']}", flush=True)

    episodes = {
        run_name: _aggregate([row for row in rows if row["run"] == run_name])
        for run_name in sorted({row["run"] for row in rows})
    }
    report = {
        "kind": "entitybench_qwen_image21_v7_full3_reference_dinov2_report",
        "methodology": {
            "aggregation": "micro mean over evaluated scheduled character-shot pairs",
            "fallback_policy": "unprepared Control-reuse shots are excluded because role mapping is unavailable",
        },
        "episodes": episodes,
        "overall": _aggregate(rows),
        "prepared_shot_count": sum(
            (Path(record["shot_dir"]) / "prepared/prepared.json").is_file()
            for record in manifest["shots"]
        ),
        "failed_shot_count": len(failures),
        "failures": failures,
        "shots": rows,
    }
    _write(args.output_root / "dino_evaluation_report.json", report)
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
