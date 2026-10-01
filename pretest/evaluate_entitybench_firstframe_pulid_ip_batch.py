#!/usr/bin/env python3
"""Evaluate the frozen 20-episode PuLID/IP first-frame batch.

InsightFace values and final face boxes are read from the generation-time
paired reports. DINOv2 embeds the same final face crops (15% expansion) against
the episode-local identity portrait. Headline means include only shots where
the v7 plugin was actually applied and both conditions were measurable.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from multishot.face_analysis_backend import get_face_backend  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _crop(image: Image.Image, bbox: list[float], expansion: float) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = (float(value) for value in bbox)
    fw, fh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    box = (
        max(0, int(np.floor(x1 - fw * expansion))),
        max(0, int(np.floor(y1 - fh * expansion))),
        min(width, int(np.ceil(x2 + fw * expansion))),
        min(height, int(np.ceil(y2 + fh * expansion))),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"invalid crop for {bbox}: {box}")
    return image.crop(box)


class DinoEncoder:
    def __init__(self, model_path: Path, device: str, batch_size: int):
        self.processor = AutoImageProcessor.from_pretrained(str(model_path), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(model_path), local_files_only=True)
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.batch_size = batch_size

    @torch.inference_mode()
    def encode(self, images: list[Image.Image]) -> np.ndarray:
        chunks = []
        for start in range(0, len(images), self.batch_size):
            inputs = self.processor(images=images[start:start + self.batch_size], return_tensors="pt")
            output = self.model(**{key: value.to(self.device) for key, value in inputs.items()})
            features = getattr(output, "pooler_output", None)
            if features is None:
                features = output.last_hidden_state[:, 0]
            chunks.append(F.normalize(features.float(), dim=-1).cpu().numpy())
        return np.concatenate(chunks, axis=0)


def _largest_reference_bbox(path: Path) -> list[float]:
    faces = get_face_backend().analyze(str(path))
    if not faces:
        raise RuntimeError(f"no face detected in identity reference: {path}")
    face = max(
        faces,
        key=lambda item: (item["face_bbox"][2] - item["face_bbox"][0])
        * (item["face_bbox"][3] - item["face_bbox"][1]),
    )
    return [float(value) for value in face["face_bbox"]]


def _shot_values(route: str, shot: dict[str, Any]) -> dict[str, Any]:
    if route == "PuLID-FLUX":
        root = Path(shot["output_dir"])
        result_path = root / "metrics.json"
        result = _load(result_path)
        control = result.get("reference_control_insightface_cosine")
        treatment = result.get("reference_treatment_insightface_cosine")
        pose = result.get("pose") or {}
        control_bbox = (pose.get("control_final") or {}).get("bbox")
        treatment_bbox = (pose.get("treatment_final") or {}).get("bbox")
        control_path = root / "control/final.png"
        treatment_path = root / "treatment/final.png"
        comparison = root / "comparison.jpg"
    else:
        result_path = Path(shot["result_json"])
        result = _load(result_path)
        metrics = result.get("metrics", {})
        control_result = metrics.get("ip_adapter_baseline", {})
        treatment_result = metrics.get("ip_adapter_plus_pulid_style_residual", {})
        control = control_result.get("reference_portrait_cosine")
        treatment = treatment_result.get("reference_portrait_cosine")
        control_bbox = control_result.get("final_face", {}).get("face_bbox") if control_result.get("face_detected") else None
        treatment_bbox = treatment_result.get("final_face", {}).get("face_bbox") if treatment_result.get("face_detected") else None
        control_path = Path(shot["control_image"])
        treatment_path = Path(shot["treatment_image"])
        comparison = Path(result.get("paths", {}).get("comparison", result_path.parent / "comparison.jpg"))
    return {
        "result_path": str(result_path),
        "insightface_control": control,
        "insightface_treatment": treatment,
        "control_bbox": control_bbox,
        "treatment_bbox": treatment_bbox,
        "control_image": str(control_path),
        "treatment_image": str(treatment_path),
        "comparison": str(comparison),
    }


def _mean_pairs(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    pairs = [
        (float(row[f"{prefix}_control"]), float(row[f"{prefix}_treatment"]))
        for row in rows
        if row.get("plugin_applied")
        and row.get(f"{prefix}_control") is not None
        and row.get(f"{prefix}_treatment") is not None
    ]
    if not pairs:
        return {
            "valid_pair_count": 0,
            "control_mean": None,
            "treatment_mean": None,
            "mean_paired_delta": None,
            "positive_delta_count": 0,
            "negative_delta_count": 0,
        }
    deltas = [treatment - control for control, treatment in pairs]
    return {
        "valid_pair_count": len(pairs),
        "control_mean": round(statistics.mean(control for control, _ in pairs), 6),
        "treatment_mean": round(statistics.mean(treatment for _, treatment in pairs), 6),
        "mean_paired_delta": round(statistics.mean(deltas), 6),
        "positive_delta_count": sum(value > 0 for value in deltas),
        "negative_delta_count": sum(value < 0 for value in deltas),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "single_character_shot_count": len(rows),
        "injected_shot_count": sum(row["plugin_applied"] for row in rows),
        "control_reuse_shot_count": sum(not row["plugin_applied"] for row in rows),
        "insightface": _mean_pairs(rows, "insightface"),
        "reference_dinov2": _mean_pairs(rows, "dinov2"),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    manifest = _load(output_root / "manifest.json")
    rows: list[dict[str, Any]] = []
    crop_jobs: list[tuple[int, str, Image.Image]] = []
    reference_job: dict[str, int] = {}
    images: list[Image.Image] = []

    for route, output_key, report_name in (
        ("PuLID-FLUX", "pulid_output", "pulid_first_frame_report.json"),
        ("IP-Adapter", "ip_output", "ip_adapter_first_frame_report.json"),
    ):
        for episode in manifest["episodes"]:
            report = _load(Path(episode[output_key]) / report_name)
            for shot in report.get("shots", []):
                if len(shot.get("characters", [])) != 1:
                    continue
                values = _shot_values(route, shot)
                row = {
                    "route": route,
                    "episode_id": episode["episode_id"],
                    "run": episode["run"],
                    "difficulty": episode["difficulty"],
                    "source": episode["source"],
                    "shot_key": shot["shot_key"],
                    "character": shot.get("target_character") or shot["characters"][0],
                    "plugin_applied": not bool(shot.get("plugin_skipped", False)),
                    "skip_reason": shot.get("skip_reason"),
                    "reference_image": shot["reference_image"],
                    **values,
                }
                row["insightface_delta"] = (
                    float(values["insightface_treatment"] - values["insightface_control"])
                    if values["insightface_control"] is not None and values["insightface_treatment"] is not None
                    else None
                )
                rows.append(row)
                row_index = len(rows) - 1
                reference = str(Path(shot["reference_image"]).resolve())
                if reference not in reference_job:
                    image = Image.open(reference).convert("RGB")
                    reference_job[reference] = len(images)
                    images.append(_crop(image, _largest_reference_bbox(Path(reference)), args.crop_expansion))
                for condition in ("control", "treatment"):
                    bbox = values[f"{condition}_bbox"]
                    path = Path(values[f"{condition}_image"])
                    if bbox is None or not path.is_file():
                        continue
                    crop_jobs.append((row_index, condition, _crop(Image.open(path).convert("RGB"), bbox, args.crop_expansion)))

    crop_offset = len(images)
    images.extend(job[2] for job in crop_jobs)
    encoder = DinoEncoder(args.dino_model.resolve(), args.device, args.batch_size)
    embeddings = encoder.encode(images)
    crop_embeddings: dict[tuple[int, str], np.ndarray] = {}
    for index, (row_index, condition, _) in enumerate(crop_jobs):
        crop_embeddings[(row_index, condition)] = embeddings[crop_offset + index]
    for row_index, row in enumerate(rows):
        reference = embeddings[reference_job[str(Path(row["reference_image"]).resolve())]]
        for condition in ("control", "treatment"):
            value = crop_embeddings.get((row_index, condition))
            row[f"dinov2_{condition}"] = round(float(np.dot(reference, value)), 6) if value is not None else None
        row["dinov2_delta"] = (
            round(row["dinov2_treatment"] - row["dinov2_control"], 6)
            if row["dinov2_control"] is not None and row["dinov2_treatment"] is not None
            else None
        )

    by_route = {route: _aggregate([row for row in rows if row["route"] == route]) for route in ("PuLID-FLUX", "IP-Adapter")}
    by_episode = defaultdict(dict)
    for episode in manifest["episodes"]:
        for route in by_route:
            selected = [row for row in rows if row["episode_id"] == episode["episode_id"] and row["route"] == route]
            by_episode[episode["episode_id"]][route] = _aggregate(selected)
    by_difficulty = defaultdict(dict)
    for difficulty in ("easy", "medium", "hard"):
        for route in by_route:
            selected = [row for row in rows if row["difficulty"] == difficulty and row["route"] == route]
            by_difficulty[difficulty][route] = _aggregate(selected)
    result = {
        "kind": "entitybench_firstframe_pulid_ip_20ep_evaluation",
        "scope": "single-character first frames; actual-injection means exclude safety-gated Control reuse",
        "manifest": str((output_root / "manifest.json").resolve()),
        "dino_model": str(args.dino_model.resolve()),
        "crop_expansion": args.crop_expansion,
        "routes": by_route,
        "difficulty": dict(by_difficulty),
        "episodes": dict(by_episode),
        "shots": rows,
    }
    report_path = output_root / "evaluation/firstframe_evaluation.json"
    _write(report_path, result)
    lines = [
        "# EntityBench 20-episode first-frame evaluation",
        "",
        "Headline means use only shots where v7 injection was actually applied and both paired values exist.",
        "",
        "| Route | Single shots | Injected | Reused Control | InsightFace Control | Treatment | Delta | DINOv2 Control | Treatment | Delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for route, value in by_route.items():
        identity, dino = value["insightface"], value["reference_dinov2"]
        lines.append(
            f"| {route} | {value['single_character_shot_count']} | {value['injected_shot_count']} | "
            f"{value['control_reuse_shot_count']} | {identity['control_mean']:.4f} | "
            f"{identity['treatment_mean']:.4f} | {identity['mean_paired_delta']:+.4f} | "
            f"{dino['control_mean']:.4f} | {dino['treatment_mean']:.4f} | {dino['mean_paired_delta']:+.4f} |"
        )
    (output_root / "evaluation/firstframe_evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dino-model", type=Path, default=Path("/root/autodl-tmp/models/dinov2-base"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--crop-expansion", type=float, default=0.15)
    return parser.parse_args()


if __name__ == "__main__":
    value = evaluate(parse_args())
    print(json.dumps(value["routes"], ensure_ascii=False, indent=2))
