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


def _paired_summary(values: list[tuple[float, float]]) -> dict:
    return {
        "valid_pair_count": len(values),
        "control_mean": _mean([control for control, _ in values]),
        "treatment_mean": _mean([treatment for _, treatment in values]),
        "mean_paired_delta": _mean(
            [treatment - control for control, treatment in values]
        ),
        "positive_delta_count": sum(treatment > control for control, treatment in values),
    }


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
        scheduled_characters = [value["name"] for value in record["characters"]]
        applied_characters = result.get("applied_characters")
        if applied_characters is None:
            applied_characters = [
                value.get("name")
                for value in result.get("targets", [])
                if value.get("name")
            ]
        skipped_characters = [
            name for name in scheduled_characters if name not in set(applied_characters)
        ]
        item = {
            "run": record["run"], "shot_key": record["shot_key"],
            "character_count": record["character_count"],
            "characters": scheduled_characters,
            "injection_applied": bool(result.get("injection_applied")),
            "partial_injection": bool(applied_characters and skipped_characters),
            "applied_characters": applied_characters,
            "skipped_characters": skipped_characters,
            "target_failures": result.get("target_failures", []),
            "fallback_reason": result.get("reason"),
        }
        prepared = shot_dir / "prepared/prepared.json"
        if prepared.is_file():
            try:
                image_metrics_path = shot_dir / "identity_metrics.json"
                if image_metrics_path.is_file():
                    image_metrics = json.loads(image_metrics_path.read_text(encoding="utf-8"))
                else:
                    image_metrics = evaluate_images(SimpleNamespace(
                        prepared=prepared, control=shot_dir / "control.png",
                        treatment=shot_dir / "treatment.png",
                        output=image_metrics_path,
                        min_face_confidence=args.min_face_confidence,
                    ), app=app)
                item["image_metrics"] = image_metrics
            except Exception as exc:
                item["image_evaluation_error"] = repr(exc)
            control_video = shot_dir / "videos/control.mp4"
            treatment_video = shot_dir / "videos/treatment.mp4"
            if control_video.is_file() and treatment_video.is_file():
                try:
                    video_metrics_path = shot_dir / "video_identity_metrics.json"
                    if video_metrics_path.is_file():
                        video_metrics = json.loads(video_metrics_path.read_text(encoding="utf-8"))
                    else:
                        video_metrics = evaluate_videos(SimpleNamespace(
                            prepared=prepared, control=control_video,
                            treatment=treatment_video,
                            output=video_metrics_path,
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
        image_pairs = [
            (char["control_cosine"], char["treatment_cosine"])
            for item in selected
            for char in item.get("image_metrics", {}).get("characters", [])
        ]
        video_pairs = [
            (
                item["video_metrics"]["conditions"]["control"]["characters"][name]["mean"],
                item["video_metrics"]["conditions"]["treatment"]["characters"][name]["mean"],
            )
            for item in selected
            for name in item.get("video_metrics", {}).get("treatment_minus_control", {})
            if item["video_metrics"]["conditions"]["control"]["characters"][name].get("mean") is not None
            and item["video_metrics"]["conditions"]["treatment"]["characters"][name].get("mean") is not None
        ]
        image_summary = _paired_summary(image_pairs)
        video_summary = _paired_summary(video_pairs)
        episodes[run_name] = {
            "shot_count": len(selected),
            "single_shot_count": sum(item["character_count"] == 1 for item in selected),
            "multi_shot_count": sum(item["character_count"] > 1 for item in selected),
            "injected_shot_count": sum(item["injection_applied"] for item in selected),
            "partial_injection_shot_count": sum(item["partial_injection"] for item in selected),
            "fallback_shot_count": sum(not item["injection_applied"] for item in selected),
            "scheduled_character_count": sum(item["character_count"] for item in selected),
            "injected_character_count": sum(len(item["applied_characters"]) for item in selected),
            "skipped_character_count": sum(len(item["skipped_characters"]) for item in selected),
            "image_evaluated_shot_count": sum("image_metrics" in item for item in selected),
            "image_evaluated_character_count": len(image_pairs),
            "image_identity": image_summary,
            "image_identity_control_mean": image_summary["control_mean"],
            "image_identity_treatment_mean": image_summary["treatment_mean"],
            "image_identity_mean_paired_delta": image_summary["mean_paired_delta"],
            "image_positive_character_count": image_summary["positive_delta_count"],
            "video_evaluated_shot_count": sum("video_metrics" in item for item in selected),
            "video_evaluated_character_count": len(video_pairs),
            "video_identity": video_summary,
            "video_identity_control_mean": video_summary["control_mean"],
            "video_identity_treatment_mean": video_summary["treatment_mean"],
            "video_identity_mean_paired_delta": video_summary["mean_paired_delta"],
            "video_positive_character_count": video_summary["positive_delta_count"],
        }
    all_image = [
        (char["control_cosine"], char["treatment_cosine"])
        for item in shots
        for char in item.get("image_metrics", {}).get("characters", [])
    ]
    all_video = [
        (
            item["video_metrics"]["conditions"]["control"]["characters"][name]["mean"],
            item["video_metrics"]["conditions"]["treatment"]["characters"][name]["mean"],
        )
        for item in shots
        for name in item.get("video_metrics", {}).get("treatment_minus_control", {})
        if item["video_metrics"]["conditions"]["control"]["characters"][name].get("mean") is not None
        and item["video_metrics"]["conditions"]["treatment"]["characters"][name].get("mean") is not None
    ]
    all_image_summary = _paired_summary(all_image)
    all_video_summary = _paired_summary(all_video)
    report = {
        "kind": "entitybench_qwen_image21_v7_full3_paired_identity_report",
        "methodology": {
            "identity_encoder": "InsightFace normalized embedding cosine",
            "image_pairing": "same target bbox by IoU",
            "video_pairing": "spatial role tracking; no identity embedding used for assignment",
            "aggregation": "micro mean over evaluated scheduled character-shot pairs",
            "fallback_policy": (
                "per-role tolerance: mapped/prepared roles are injected independently; "
                "unmapped or failed roles are reported as skipped; when no role is usable, "
                "Treatment exactly reuses Control"
            ),
        },
        "episodes": episodes,
        "overall": {
            "shot_count": len(shots),
            "injected_shot_count": sum(item["injection_applied"] for item in shots),
            "partial_injection_shot_count": sum(item["partial_injection"] for item in shots),
            "fallback_shot_count": sum(not item["injection_applied"] for item in shots),
            "scheduled_character_count": sum(item["character_count"] for item in shots),
            "injected_character_count": sum(len(item["applied_characters"]) for item in shots),
            "skipped_character_count": sum(len(item["skipped_characters"]) for item in shots),
            "image_evaluation_error_count": sum(
                "image_evaluation_error" in item for item in shots
            ),
            "video_evaluation_error_count": sum(
                "video_evaluation_error" in item for item in shots
            ),
            "image_evaluated_character_count": len(all_image),
            "image_identity": all_image_summary,
            "image_identity_control_mean": all_image_summary["control_mean"],
            "image_identity_treatment_mean": all_image_summary["treatment_mean"],
            "image_identity_mean_paired_delta": all_image_summary["mean_paired_delta"],
            "image_positive_character_count": all_image_summary["positive_delta_count"],
            "video_evaluated_character_count": len(all_video),
            "video_identity": all_video_summary,
            "video_identity_control_mean": all_video_summary["control_mean"],
            "video_identity_treatment_mean": all_video_summary["treatment_mean"],
            "video_identity_mean_paired_delta": all_video_summary["mean_paired_delta"],
            "video_positive_character_count": all_video_summary["positive_delta_count"],
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
