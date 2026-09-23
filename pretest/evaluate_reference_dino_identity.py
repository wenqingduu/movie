#!/usr/bin/env python3
"""Evaluate reference-anchored DINOv2 similarity for first frames and videos.

This is a project metric, not an official EntityBench metric.  Target faces are
selected with the existing InsightFace identity backend; DINOv2 is used only to
embed the resulting face crops.  Video face boxes are reused from
``evaluate_video_identity.py`` so both identity encoders score the same target.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from multishot.face_analysis_backend import get_face_backend  # noqa: E402


METRICS = (
    "first_frame",
    "video_first",
    "video_frame_mean",
    "video_shot_median",
    "video_p10",
    "video_minimum",
    "video_last",
    "detection_coverage",
)


def _resolve(value: str, relative_to: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (relative_to.parent / path).resolve()


def _cosine(left, right) -> float:
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def _largest_face(faces: list[dict]) -> dict | None:
    if not faces:
        return None
    return max(
        faces,
        key=lambda face: (face["face_bbox"][2] - face["face_bbox"][0])
        * (face["face_bbox"][3] - face["face_bbox"][1]),
    )


def _crop_face(image: Image.Image, bbox: list[float], expansion: float) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = (float(value) for value in bbox)
    face_width = max(1.0, x2 - x1)
    face_height = max(1.0, y2 - y1)
    x1 = max(0, int(np.floor(x1 - face_width * expansion)))
    y1 = max(0, int(np.floor(y1 - face_height * expansion)))
    x2 = min(width, int(np.ceil(x2 + face_width * expansion)))
    y2 = min(height, int(np.ceil(y2 + face_height * expansion)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid expanded face crop: {(x1, y1, x2, y2)}")
    return image.crop((x1, y1, x2, y2))


class DinoEncoder:
    def __init__(self, model_path: Path, device: str, batch_size: int):
        self.processor = AutoImageProcessor.from_pretrained(
            str(model_path), local_files_only=True
        )
        self.model = AutoModel.from_pretrained(str(model_path), local_files_only=True)
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.batch_size = batch_size

    @torch.inference_mode()
    def encode(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, 0), dtype=np.float32)
        chunks = []
        for start in range(0, len(images), self.batch_size):
            batch = images[start : start + self.batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            output = self.model(**inputs)
            features = getattr(output, "pooler_output", None)
            if features is None:
                features = output.last_hidden_state[:, 0]
            chunks.append(F.normalize(features.float(), dim=-1).cpu().numpy())
        return np.concatenate(chunks, axis=0)


def _select_input_crop(
    image_path: Path,
    reference_identity_embedding,
    expansion: float,
) -> tuple[Image.Image | None, dict[str, Any]]:
    backend = get_face_backend()
    faces = backend.analyze(str(image_path))
    candidates = [
        (_cosine(reference_identity_embedding, face["face_embedding"]), face)
        for face in faces
    ]
    if not candidates:
        return None, {"face_detected": False, "face_count": 0}
    identity_cosine, face = max(candidates, key=lambda item: item[0])
    image = Image.open(image_path).convert("RGB")
    crop = _crop_face(image, face["face_bbox"], expansion)
    return crop, {
        "face_detected": True,
        "face_count": len(faces),
        "selected_face_bbox": face["face_bbox"],
        "selector_identity_cosine": identity_cosine,
    }


def _decode_video_crops(
    video_path: Path,
    frame_records: list[dict],
    expansion: float,
) -> tuple[list[Image.Image], list[int], int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    crops: list[Image.Image] = []
    indexes: list[int] = []
    decoded = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if decoded >= len(frame_records):
            capture.release()
            raise ValueError(f"Identity report has fewer frames than video: {video_path}")
        bbox = frame_records[decoded].get("selected_face_bbox")
        if bbox is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            crops.append(_crop_face(Image.fromarray(rgb), bbox, expansion))
            indexes.append(decoded)
        decoded += 1
    capture.release()
    if decoded != len(frame_records):
        raise ValueError(
            f"Frame-count mismatch for {video_path}: decoded={decoded}, "
            f"report={len(frame_records)}"
        )
    return crops, indexes, decoded


def _video_summary(similarities: list[float | None]) -> dict[str, Any]:
    valid = [float(value) for value in similarities if value is not None]
    total = len(similarities)
    return {
        "decoded_frame_count": total,
        "scored_frame_count": len(valid),
        "detection_coverage": round(len(valid) / total, 6) if total else 0.0,
        "video_first": round(similarities[0], 6) if total and similarities[0] is not None else None,
        "video_frame_mean": round(float(np.mean(valid)), 6) if valid else None,
        "video_shot_median": round(float(np.median(valid)), 6) if valid else None,
        "video_p10": round(float(np.percentile(valid, 10)), 6) if valid else None,
        "video_minimum": round(float(np.min(valid)), 6) if valid else None,
        "video_last": round(similarities[-1], 6)
        if total and similarities[-1] is not None
        else None,
    }


def _aggregate_pairs(pairs: list[dict], applied_only: bool) -> dict[str, Any]:
    selected = [pair for pair in pairs if pair["plugin_applied"] or not applied_only]
    metrics: dict[str, Any] = {}
    for metric in METRICS:
        values = [
            (pair["control"][metric], pair["treatment"][metric])
            for pair in selected
            if pair["control"].get(metric) is not None
            and pair["treatment"].get(metric) is not None
        ]
        metrics[metric] = {
            "valid_pair_count": len(values),
            "control_mean": round(float(np.mean([value[0] for value in values])), 6)
            if values
            else None,
            "treatment_mean": round(float(np.mean([value[1] for value in values])), 6)
            if values
            else None,
            "mean_paired_delta": round(
                float(np.mean([value[1] - value[0] for value in values])), 6
            )
            if values
            else None,
            "positive_delta_count": sum(value[1] > value[0] for value in values),
        }
    first_delta = metrics["first_frame"]["mean_paired_delta"]
    video_delta = metrics["video_shot_median"]["mean_paired_delta"]
    retention = (
        video_delta / first_delta
        if first_delta is not None and video_delta is not None and abs(first_delta) >= 1e-6
        else None
    )
    return {
        "applied_only": applied_only,
        "pair_count": len(selected),
        "metrics": metrics,
        "ref_dino_gain_retention_ratio": round(retention, 6)
        if retention is not None
        else None,
    }


def _evaluate_case(
    route: str,
    manifest_path: Path,
    identity_report_path: Path,
    encoder: DinoEncoder,
    expansion: float,
    reference_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity_report = json.loads(identity_report_path.read_text(encoding="utf-8"))
    identity_jobs = {job["job_id"]: job for job in identity_report["jobs"]}
    references = manifest["identity_references"]
    backend = get_face_backend()

    for character, value in references.items():
        path = _resolve(value, manifest_path)
        cache_key = str(path.resolve())
        if cache_key in reference_cache:
            continue
        faces = backend.analyze(str(path))
        face = _largest_face(faces)
        if face is None:
            raise RuntimeError(f"No face found in reference: {path}")
        image = Image.open(path).convert("RGB")
        crop = _crop_face(image, face["face_bbox"], expansion)
        reference_cache[cache_key] = {
            "path": str(path),
            "face_bbox": face["face_bbox"],
            "identity_embedding": face["face_embedding"],
            "dino_embedding": encoder.encode([crop])[0],
        }

    jobs = []
    by_job_id: dict[str, dict[str, Any]] = {}
    for manifest_job in manifest["jobs"]:
        job_id = manifest_job["job_id"]
        character = manifest_job["target_character"]
        reference_path = _resolve(references[character], manifest_path)
        reference = reference_cache[str(reference_path.resolve())]
        input_path = _resolve(manifest_job["input_image"], manifest_path)
        first_crop, first_detection = _select_input_crop(
            input_path, reference["identity_embedding"], expansion
        )
        first_similarity = None
        if first_crop is not None:
            first_embedding = encoder.encode([first_crop])[0]
            first_similarity = _cosine(reference["dino_embedding"], first_embedding)

        identity_job = identity_jobs[job_id]
        reuse_source = manifest_job.get("reuse_video_from")
        if reuse_source and reuse_source in by_job_id:
            video = copy.deepcopy(by_job_id[reuse_source]["video"])
            video["reused_from"] = reuse_source
        else:
            video_path = _resolve(manifest_job["output_video"], manifest_path)
            frame_records = identity_job["video"]["frames"]
            crops, indexes, decoded = _decode_video_crops(
                video_path, frame_records, expansion
            )
            similarities: list[float | None] = [None] * decoded
            if crops:
                embeddings = encoder.encode(crops)
                for index, embedding in zip(indexes, embeddings):
                    similarities[index] = _cosine(reference["dino_embedding"], embedding)
            video = {
                "path": str(video_path),
                **_video_summary(similarities),
                "frame_similarities": [
                    round(value, 6) if value is not None else None
                    for value in similarities
                ],
            }

        result = {
            "job_id": job_id,
            "episode_id": manifest_job.get("episode_id"),
            "shot_key": manifest_job["shot_key"],
            "condition": manifest_job["condition"],
            "target_character": character,
            "reuse_video_from": reuse_source,
            "reference_image": reference["path"],
            "input_image": str(input_path),
            "first_frame": round(first_similarity, 6)
            if first_similarity is not None
            else None,
            "first_frame_detection": first_detection,
            "video": video,
        }
        jobs.append(result)
        by_job_id[job_id] = result

    grouped: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for job in jobs:
        grouped[(job["shot_key"], job["target_character"])][job["condition"]] = job
    pairs = []
    for (shot_key, character), conditions in grouped.items():
        if "control" not in conditions or "treatment" not in conditions:
            continue
        control_job = conditions["control"]
        treatment_job = conditions["treatment"]
        control = {"first_frame": control_job["first_frame"]}
        treatment = {"first_frame": treatment_job["first_frame"]}
        for metric in METRICS[1:]:
            control[metric] = control_job["video"].get(metric)
            treatment[metric] = treatment_job["video"].get(metric)
        deltas = {
            metric: round(treatment[metric] - control[metric], 6)
            if control.get(metric) is not None and treatment.get(metric) is not None
            else None
            for metric in METRICS
        }
        first_delta = deltas["first_frame"]
        video_delta = deltas["video_shot_median"]
        pairs.append(
            {
                "episode_id": control_job.get("episode_id"),
                "shot_key": shot_key,
                "target_character": character,
                "plugin_applied": not bool(treatment_job.get("reuse_video_from")),
                "control": control,
                "treatment": treatment,
                "treatment_minus_control": deltas,
                "gain_retention_ratio": round(video_delta / first_delta, 6)
                if first_delta is not None
                and video_delta is not None
                and abs(first_delta) >= 1e-6
                else None,
            }
        )

    episode_id = manifest.get("episode_id") or jobs[0].get("episode_id")
    return {
        "route": route,
        "episode_id": episode_id,
        "manifest": str(manifest_path),
        "identity_report": str(identity_report_path),
        "jobs": jobs,
        "pairs": pairs,
        "aggregate_all_single_character_pairs": _aggregate_pairs(pairs, False),
        "aggregate_applied_pairs": _aggregate_pairs(pairs, True),
    }


def _aggregate_route(cases: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [pair for case in cases for pair in case["pairs"]]
    return {
        "episode_ids": [case["episode_id"] for case in cases],
        "aggregate_all_single_character_pairs": _aggregate_pairs(pairs, False),
        "aggregate_applied_pairs": _aggregate_pairs(pairs, True),
    }


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Reference-Anchored DINOv2 Evaluation",
        "",
        "> 项目扩展指标，不是 EntityBench 官方指标。目标脸由 InsightFace 选择，DINOv2 只对统一脸部裁剪编码。",
        "",
    ]
    for route, route_report in report["routes"].items():
        lines += [f"## {route}", ""]
        for scope_key, scope_name in (
            ("aggregate_applied_pairs", "实际注入镜头"),
            ("aggregate_all_single_character_pairs", "全部单角色镜头"),
        ):
            aggregate = route_report[scope_key]
            lines += [f"### {scope_name}（{aggregate['pair_count']} 对）", ""]
            lines += [
                "| 指标 | Control | v7 | Δ | 正增益数 |",
                "|---|---:|---:|---:|---:|",
            ]
            for metric in ("first_frame", "video_shot_median", "video_p10", "detection_coverage"):
                item = aggregate["metrics"][metric]
                lines.append(
                    f"| `{metric}` | {_fmt(item['control_mean'])} | "
                    f"{_fmt(item['treatment_mean'])} | "
                    f"{_fmt(item['mean_paired_delta'])} | "
                    f"{item['positive_delta_count']}/{item['valid_pair_count']} |"
                )
            lines += [
                "",
                f"聚合增益保留率：`{_fmt(aggregate['ref_dino_gain_retention_ratio'])}`",
                "",
            ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        nargs=3,
        metavar=("ROUTE", "MANIFEST", "VIDEO_IDENTITY_REPORT"),
        required=True,
    )
    parser.add_argument(
        "--model-path", type=Path, default=Path("/root/autodl-tmp/models/dinov2-base")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--face-crop-expansion", type=float, default=0.15)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    encoder = DinoEncoder(args.model_path.resolve(), args.device, args.batch_size)
    reference_cache: dict[str, dict[str, Any]] = {}
    cases = []
    for route, manifest, identity_report in args.case:
        cases.append(
            _evaluate_case(
                route,
                Path(manifest).resolve(),
                Path(identity_report).resolve(),
                encoder,
                args.face_crop_expansion,
                reference_cache,
            )
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[case["route"]].append(case)
    report = {
        "kind": "reference_anchored_dinov2_face_similarity",
        "metric_scope": "project_extension_not_official_entitybench",
        "model_path": str(args.model_path.resolve()),
        "face_crop_expansion": args.face_crop_expansion,
        "aggregation": "frame similarities -> shot median -> equal-weight shot macro mean",
        "cases": cases,
        "routes": {route: _aggregate_route(items) for route, items in grouped.items()},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(_render_markdown(report), encoding="utf-8")
    print(args.output_json)
    print(args.output_md)


if __name__ == "__main__":
    main()
