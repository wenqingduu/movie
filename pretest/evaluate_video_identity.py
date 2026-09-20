"""Evaluate reference-anchored face identity consistency for paired videos.

The input is the same manifest consumed by ``run_wan22_i2v_manifest.py``.
Every decoded frame is retained in the report: frames where no face is found
have a null similarity and contribute to the detection coverage denominator.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from multishot.face_analysis_backend import get_face_backend  # noqa: E402


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def _cosine(left, right) -> float:
    left_array = np.asarray(left, dtype=np.float32)
    right_array = np.asarray(right, dtype=np.float32)
    denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left_array, right_array) / denominator)


def _largest_face(faces: list[dict]) -> dict | None:
    if not faces:
        return None
    return max(
        faces,
        key=lambda face: (face["face_bbox"][2] - face["face_bbox"][0])
        * (face["face_bbox"][3] - face["face_bbox"][1]),
    )


def _summary(frame_records: list[dict]) -> dict:
    scored = [item for item in frame_records if item["identity_cosine"] is not None]
    scores = np.asarray([item["identity_cosine"] for item in scored], dtype=np.float64)
    total = len(frame_records)
    first = frame_records[0]["identity_cosine"] if total else None
    last = frame_records[-1]["identity_cosine"] if total else None
    last_detected = scored[-1]["identity_cosine"] if scored else None

    if len(scored) >= 2:
        x = np.asarray([item["frame_index"] for item in scored], dtype=np.float64)
        slope_per_frame = float(np.polyfit(x, scores, 1)[0])
        span = max(1, total - 1)
        slope_full_video = slope_per_frame * span
    else:
        slope_per_frame = None
        slope_full_video = None

    return {
        "decoded_frame_count": total,
        "detected_frame_count": len(scored),
        "detection_coverage": round(len(scored) / total, 6) if total else 0.0,
        "first": round(first, 6) if first is not None else None,
        "mean": round(float(scores.mean()), 6) if scores.size else None,
        "median": round(float(np.median(scores)), 6) if scores.size else None,
        "p10": round(float(np.percentile(scores, 10)), 6) if scores.size else None,
        "minimum": round(float(scores.min()), 6) if scores.size else None,
        "last": round(last, 6) if last is not None else None,
        "last_detected": round(last_detected, 6) if last_detected is not None else None,
        "last_minus_first": (
            round(last - first, 6) if first is not None and last is not None else None
        ),
        "first_minus_last": (
            round(first - last, 6) if first is not None and last is not None else None
        ),
        "regression_slope_per_frame": (
            round(slope_per_frame, 8) if slope_per_frame is not None else None
        ),
        "regression_change_over_full_video": (
            round(slope_full_video, 6) if slope_full_video is not None else None
        ),
    }


def _annotate(frame, record: dict, title: str):
    canvas = frame.copy()
    bbox = record.get("selected_face_bbox")
    if bbox:
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (65, 220, 65), 3)
    score = record.get("identity_cosine")
    score_text = "no face" if score is None else f"cos={score:.4f}"
    text = f"{title}  frame={record['frame_index']}  {score_text}"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 48), (0, 0, 0), -1)
    cv2.putText(canvas, text, (14, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas


def _make_contact_sheet(
    selected_frames: list[tuple[np.ndarray, dict, str]], output_path: Path
) -> None:
    if not selected_frames:
        return
    target_width = 560
    panels = []
    for frame, record, title in selected_frames:
        scale = target_width / frame.shape[1]
        resized = cv2.resize(
            frame,
            (target_width, max(1, int(round(frame.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        panels.append(_annotate(resized, {
            **record,
            "selected_face_bbox": (
                [value * scale for value in record["selected_face_bbox"]]
                if record.get("selected_face_bbox") else None
            ),
        }, title))
    height = max(panel.shape[0] for panel in panels)
    padded = []
    for panel in panels:
        if panel.shape[0] < height:
            panel = cv2.copyMakeBorder(
                panel, 0, height - panel.shape[0], 0, 0, cv2.BORDER_CONSTANT,
                value=(20, 20, 20),
            )
        padded.append(panel)
    sheet = np.concatenate(padded, axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)


def _evaluate_video(video_path: Path, reference_embedding, visual_path: Path | None) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    backend = get_face_backend()
    records = []
    frames = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        faces = backend.analyze_bgr(frame)
        candidates = []
        for face in faces:
            candidates.append((_cosine(reference_embedding, face["face_embedding"]), face))
        selected = max(candidates, key=lambda item: item[0]) if candidates else None
        record = {
            "frame_index": frame_index,
            "timestamp_seconds": round(frame_index / fps, 6) if fps > 0 else None,
            "detected_face_count": len(faces),
            "identity_cosine": round(selected[0], 6) if selected else None,
            "selected_face_bbox": selected[1]["face_bbox"] if selected else None,
            "selected_face_confidence": selected[1]["face_confidence"] if selected else None,
            "selected_face_pose": selected[1]["pose"] if selected else None,
            "detection_source": selected[1]["detection_source"] if selected else None,
        }
        records.append(record)
        frames.append(frame)
        frame_index += 1
    capture.release()
    if not records:
        raise RuntimeError(f"No decodable frames in video: {video_path}")

    summary = _summary(records)
    if visual_path is not None:
        scored_indexes = [i for i, item in enumerate(records) if item["identity_cosine"] is not None]
        worst_index = min(scored_indexes, key=lambda i: records[i]["identity_cosine"]) if scored_indexes else 0
        picks = [
            (0, "first"),
            (len(records) // 2, "middle"),
            (worst_index, "minimum"),
            (len(records) - 1, "last"),
        ]
        # Preserve semantic order while avoiding duplicate panels for very short videos.
        seen = set()
        selected_frames = []
        for index, title in picks:
            if index not in seen:
                selected_frames.append((frames[index], records[index], title))
                seen.add(index)
        _make_contact_sheet(selected_frames, visual_path)

    return {
        "video_path": str(video_path),
        "fps": round(fps, 6),
        "duration_seconds": round(len(records) / fps, 6) if fps > 0 else None,
        "summary": summary,
        "frames": records,
        "contact_sheet": str(visual_path) if visual_path else None,
    }


def _paired_deltas(results: list[dict]) -> list[dict]:
    groups = defaultdict(dict)
    for item in results:
        key = (item.get("episode_id"), item.get("shot_key"), item.get("target_character"))
        groups[key][item.get("condition")] = item
    metric_names = (
        "first", "mean", "median", "p10", "minimum", "last", "last_detected",
        "last_minus_first", "regression_change_over_full_video", "detection_coverage",
    )
    pairs = []
    for (episode_id, shot_key, target_character), conditions in groups.items():
        control = conditions.get("control")
        treatment = conditions.get("treatment")
        if control is None or treatment is None:
            continue
        control_summary = control["video"]["summary"]
        treatment_summary = treatment["video"]["summary"]
        deltas = {}
        for name in metric_names:
            left = control_summary.get(name)
            right = treatment_summary.get(name)
            deltas[name] = round(right - left, 6) if left is not None and right is not None else None
        pairs.append({
            "episode_id": episode_id,
            "shot_key": shot_key,
            "target_character": target_character,
            "control_job_id": control["job_id"],
            "treatment_job_id": treatment["job_id"],
            "plugin_applied": not bool(treatment.get("reuse_video_from")),
            "treatment_minus_control": deltas,
        })
    return pairs


def _aggregate_pairs(pairs: list[dict], results: list[dict], applied_only: bool) -> dict:
    result_by_id = {item["job_id"]: item for item in results}
    selected = [item for item in pairs if item["plugin_applied"] or not applied_only]
    metric_names = (
        "first", "mean", "median", "p10", "minimum", "last", "last_detected",
        "last_minus_first", "regression_change_over_full_video", "detection_coverage",
    )
    metrics = {}
    for name in metric_names:
        values = []
        for pair in selected:
            control = result_by_id[pair["control_job_id"]]["video"]["summary"].get(name)
            treatment = result_by_id[pair["treatment_job_id"]]["video"]["summary"].get(name)
            if control is not None and treatment is not None:
                values.append((control, treatment))
        metrics[name] = {
            "valid_pair_count": len(values),
            "control_mean": round(float(np.mean([item[0] for item in values])), 6)
            if values else None,
            "treatment_mean": round(float(np.mean([item[1] for item in values])), 6)
            if values else None,
            "mean_paired_delta": round(float(np.mean([item[1] - item[0] for item in values])), 6)
            if values else None,
            "positive_delta_count": sum(item[1] > item[0] for item in values),
        }
    return {
        "applied_only": applied_only,
        "pair_count": len(selected),
        "metrics": metrics,
    }


def run(args) -> dict:
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference_paths = manifest.get("identity_references", {})
    if not reference_paths:
        raise ValueError("Manifest has no identity_references mapping")
    jobs = manifest.get("jobs", [])
    if not jobs:
        raise ValueError("Manifest has no jobs")

    backend = get_face_backend()
    references = {}
    reference_report = {}
    for character, value in reference_paths.items():
        path = _resolve(value, manifest_path)
        faces = backend.analyze(str(path))
        face = _largest_face(faces)
        if face is None:
            raise RuntimeError(f"No face detected in identity reference for {character}: {path}")
        references[character] = face["face_embedding"]
        reference_report[character] = {
            "path": str(path),
            "face_bbox": face["face_bbox"],
            "face_confidence": face["face_confidence"],
            "pose": face["pose"],
        }

    visual_dir = Path(args.visual_dir).resolve() if args.visual_dir else None
    results = []
    for job in jobs:
        character = job.get("target_character")
        if character not in references:
            raise KeyError(f"{job.get('job_id')}: no identity reference for {character!r}")
        video_path = _resolve(job["output_video"], manifest_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"{job.get('job_id')}: video is missing: {video_path}")
        visual_path = visual_dir / f"{job['job_id']}_contact.jpg" if visual_dir else None
        video_result = _evaluate_video(video_path, references[character], visual_path)
        results.append({
            "job_id": job["job_id"],
            "episode_id": job.get("episode_id"),
            "shot_key": job.get("shot_key"),
            "condition": job.get("condition"),
            "target_character": character,
            "reuse_video_from": job.get("reuse_video_from"),
            "video": video_result,
        })
        if args.output_json:
            partial_pairs = _paired_deltas(results)
            partial = {
                "kind": "reference_anchored_video_identity_evaluation",
                "manifest": str(manifest_path),
                "identity_references": reference_report,
                "jobs": results,
                "paired_comparisons": partial_pairs,
                "aggregate_all_pairs": _aggregate_pairs(partial_pairs, results, applied_only=False),
                "aggregate_applied_pairs": _aggregate_pairs(partial_pairs, results, applied_only=True),
            }
            _write_json(Path(args.output_json).resolve(), partial)

    pairs = _paired_deltas(results)
    report = {
        "kind": "reference_anchored_video_identity_evaluation",
        "manifest": str(manifest_path),
        "identity_references": reference_report,
        "jobs": results,
        "paired_comparisons": pairs,
        "aggregate_all_pairs": _aggregate_pairs(pairs, results, applied_only=False),
        "aggregate_applied_pairs": _aggregate_pairs(pairs, results, applied_only=True),
    }
    if args.output_json:
        _write_json(Path(args.output_json).resolve(), report)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--visual-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    result = run(cli_args)
    compact = {
        "output": str(Path(cli_args.output_json).resolve()),
        "jobs": [
            {"job_id": item["job_id"], **item["video"]["summary"]}
            for item in result["jobs"]
        ],
        "paired_comparisons": result["paired_comparisons"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
