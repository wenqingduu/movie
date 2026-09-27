#!/usr/bin/env python3
"""Evaluate all completed Qwen full-episode image/video pairs and summarize."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def run(args) -> dict:
    from multishot.ip_adapter_experiment_utils import _face_app
    from pretest.evaluate_qwen_multiface_pair import run as evaluate_images
    from pretest.evaluate_qwen_multiface_videos import run as evaluate_videos

    root = args.output_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    app = _face_app()
    shots = []
    for index, record in enumerate(manifest["shots"], 1):
        shot_dir = Path(record["shot_dir"])
        result = json.loads((shot_dir / "result.json").read_text(encoding="utf-8"))
        item = {
            "run": record["run"], "shot_key": record["shot_key"],
            "character_count": record["character_count"],
            "characters": [value["name"] for value in record["characters"]],
            "injection_applied": bool(result.get("injection_applied")),
            "fallback_reason": result.get("reason"),
        }
        prepared = shot_dir / "prepared/prepared.json"
        if prepared.is_file():
            try:
                image_metrics = evaluate_images(SimpleNamespace(
                    prepared=prepared, control=shot_dir / "control.png",
                    treatment=shot_dir / "treatment.png",
                    output=shot_dir / "identity_metrics.json",
                    min_face_confidence=args.min_face_confidence,
                ), app=app)
                item["image_metrics"] = image_metrics
            except Exception as exc:
                item["image_evaluation_error"] = repr(exc)
            control_video = shot_dir / "videos/control.mp4"
            treatment_video = shot_dir / "videos/treatment.mp4"
            if control_video.is_file() and treatment_video.is_file():
                try:
                    video_metrics = evaluate_videos(SimpleNamespace(
                        prepared=prepared, control=control_video,
                        treatment=treatment_video,
                        output=shot_dir / "video_identity_metrics.json",
                        min_face_confidence=args.min_face_confidence,
                    ), app=app)
                    item["video_metrics"] = video_metrics
                except Exception as exc:
                    item["video_evaluation_error"] = repr(exc)
        shots.append(item)
        print(f"[{index}/{len(manifest['shots'])}] evaluated {record['run']} {record['shot_key']}", flush=True)

    episodes = {}
    for run_name in sorted({item["run"] for item in shots}):
        selected = [item for item in shots if item["run"] == run_name]
        image_deltas = [
            char["delta"] for item in selected
            for char in item.get("image_metrics", {}).get("characters", [])
        ]
        video_deltas = [
            value.get("mean") for item in selected
            for value in item.get("video_metrics", {}).get("treatment_minus_control", {}).values()
            if value.get("mean") is not None
        ]
        episodes[run_name] = {
            "shot_count": len(selected),
            "single_shot_count": sum(item["character_count"] == 1 for item in selected),
            "multi_shot_count": sum(item["character_count"] > 1 for item in selected),
            "injected_shot_count": sum(item["injection_applied"] for item in selected),
            "fallback_shot_count": sum(not item["injection_applied"] for item in selected),
            "image_evaluated_shot_count": sum("image_metrics" in item for item in selected),
            "image_evaluated_character_count": len(image_deltas),
            "image_identity_mean_paired_delta": _mean(image_deltas),
            "image_positive_character_count": sum(value > 0 for value in image_deltas),
            "video_evaluated_shot_count": sum("video_metrics" in item for item in selected),
            "video_evaluated_character_count": len(video_deltas),
            "video_identity_mean_paired_delta": _mean(video_deltas),
            "video_positive_character_count": sum(value > 0 for value in video_deltas),
        }
    all_image = [
        char["delta"] for item in shots
        for char in item.get("image_metrics", {}).get("characters", [])
    ]
    all_video = [
        value.get("mean") for item in shots
        for value in item.get("video_metrics", {}).get("treatment_minus_control", {}).values()
        if value.get("mean") is not None
    ]
    report = {
        "kind": "entitybench_qwen_image21_v7_full3_paired_identity_report",
        "methodology": {
            "identity_encoder": "InsightFace normalized embedding cosine",
            "image_pairing": "same target bbox by IoU",
            "video_pairing": "spatial role tracking; no identity embedding used for assignment",
            "aggregation": "micro mean over evaluated scheduled character-shot pairs",
            "fallback_policy": "Treatment exactly reuses Control; excluded where role mapping is unavailable",
        },
        "episodes": episodes,
        "overall": {
            "shot_count": len(shots),
            "injected_shot_count": sum(item["injection_applied"] for item in shots),
            "fallback_shot_count": sum(not item["injection_applied"] for item in shots),
            "image_evaluated_character_count": len(all_image),
            "image_identity_mean_paired_delta": _mean(all_image),
            "image_positive_character_count": sum(value > 0 for value in all_image),
            "video_evaluated_character_count": len(all_video),
            "video_identity_mean_paired_delta": _mean(all_video),
            "video_positive_character_count": sum(value > 0 for value in all_video),
        },
        "shots": shots,
    }
    _write(root / "evaluation_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/entitybench_qwen_image21_v7_full3")
    parser.add_argument("--min-face-confidence", type=float, default=0.5)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
