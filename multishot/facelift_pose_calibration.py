"""Calibrate FaceLift Gaussian camera poses against InsightFace pose estimates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from multishot.face_analysis_backend import get_face_backend
from multishot.mcp_asset_server import _pose_to_facelift_camera


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AXES = ("pitch", "yaw", "roll")


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _largest_face_record(image_path: Path) -> dict | None:
    faces = get_face_backend().analyze(str(image_path))
    if not faces:
        return None
    return max(
        faces,
        key=lambda face: (
            (face["face_bbox"][2] - face["face_bbox"][0])
            * (face["face_bbox"][3] - face["face_bbox"][1])
        ),
    )


def _render(pc, render_opencv_cam, pose: dict, output_path: Path, image_size: int) -> None:
    c2w_np, fxfycxcy_np, camera_meta = _pose_to_facelift_camera(pose, image_size)
    device = pc.get_xyz.device
    c2w = torch.from_numpy(c2w_np).float().to(device)
    fxfycxcy = torch.from_numpy(fxfycxcy_np).float().to(device)
    with torch.no_grad():
        rendered = render_opencv_cam(pc, image_size, image_size, c2w, fxfycxcy)["render"]
    image = rendered.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((image * 255.0).round().astype(np.uint8)).save(output_path)
    _write_json(output_path.with_suffix(".camera.json"), camera_meta)


def _pose_vector(pose: dict) -> np.ndarray:
    return np.asarray([float(pose[axis]) for axis in AXES], dtype=np.float64)


def _fit_profile(name: str, samples: list[dict], target_bounds: dict) -> dict:
    camera = np.asarray([_pose_vector(sample["camera_pose"]) for sample in samples])
    detected = np.asarray([_pose_vector(sample["detected_pose"]) for sample in samples])
    camera_augmented = np.concatenate([camera, np.ones((camera.shape[0], 1))], axis=1)
    detected_augmented = np.concatenate([detected, np.ones((detected.shape[0], 1))], axis=1)

    camera_to_detected, _, _, _ = np.linalg.lstsq(camera_augmented, detected, rcond=None)
    target_to_camera, _, _, _ = np.linalg.lstsq(detected_augmented, camera, rcond=None)
    detected_fit = camera_augmented @ camera_to_detected
    camera_fit = detected_augmented @ target_to_camera
    detected_rmse = np.sqrt(np.mean((detected_fit - detected) ** 2, axis=0))
    camera_rmse = np.sqrt(np.mean((camera_fit - camera) ** 2, axis=0))

    margin = {"pitch": 5.0, "yaw": 7.5, "roll": 5.0}
    camera_bounds = {
        axis: [
            float(camera[:, index].min() - margin[axis]),
            float(camera[:, index].max() + margin[axis]),
        ]
        for index, axis in enumerate(AXES)
    }
    return {
        "name": name,
        "target_bounds": target_bounds,
        "camera_bounds": camera_bounds,
        "target_to_camera_affine": target_to_camera.T.tolist(),
        "camera_to_detected_affine": camera_to_detected.T.tolist(),
        "fit_rmse_degrees": {
            axis: float(detected_rmse[index]) for index, axis in enumerate(AXES)
        },
        "inverse_camera_rmse_degrees": {
            axis: float(camera_rmse[index]) for index, axis in enumerate(AXES)
        },
        "sample_count": len(samples),
        "detected_pose_range": {
            axis: [float(detected[:, index].min()), float(detected[:, index].max())]
            for index, axis in enumerate(AXES)
        },
    }


def _profile_for_target(profiles: list[dict], target: dict) -> dict:
    for profile in profiles:
        if all(
            float(profile["target_bounds"][axis][0])
            <= float(target[axis])
            <= float(profile["target_bounds"][axis][1])
            for axis in AXES
        ):
            return profile
    raise ValueError(f"No calibration profile covers validation target: {target}")


def _correct_target(profile: dict, target: dict) -> dict:
    vector = np.asarray(
        [float(target["pitch"]), float(target["yaw"]), float(target["roll"]), 1.0],
        dtype=np.float64,
    )
    corrected = np.asarray(profile["target_to_camera_affine"], dtype=np.float64) @ vector
    result = {axis: float(corrected[index]) for index, axis in enumerate(AXES)}
    for axis in AXES:
        lower, upper = profile["camera_bounds"][axis]
        result[axis] = max(float(lower), min(float(upper), result[axis]))
    return result


def _parse_pose(value: str) -> dict:
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("pose must be pitch,yaw,roll")
    return dict(zip(AXES, parts))


def run(args) -> Path:
    model_path = args.model.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    os.environ["MULTISHOT_INSIGHTFACE_MODEL_NAME"] = "antelopev2"
    os.environ["MULTISHOT_INSIGHTFACE_ROOT"] = str(PROJECT_ROOT / "third_party" / "PuLID")
    os.environ["MULTISHOT_FACELIFT_RENDER_SIZE"] = str(args.image_size)

    facelift_root = PROJECT_ROOT / "third_party" / "FaceLift"
    if str(facelift_root) not in sys.path:
        sys.path.insert(0, str(facelift_root))
    from gslrm.model.gaussians_renderer import GaussianModel, render_opencv_cam

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pc = GaussianModel(sh_degree=3)
    pc.load_ply(str(model_path))
    pc = pc.to(device)

    samples_path = output / "samples.json"
    if args.reuse_samples:
        if not samples_path.exists():
            raise FileNotFoundError(f"Cannot reuse missing samples file: {samples_path}")
        samples = json.loads(samples_path.read_text(encoding="utf-8"))
    else:
        samples = []
        for yaw in args.yaw_values:
            for pitch in args.pitch_values:
                for roll in args.roll_values:
                    camera_pose = {"pitch": pitch, "yaw": yaw, "roll": roll}
                    tag = f"p{pitch:+05.1f}_y{yaw:+05.1f}_r{roll:+05.1f}".replace("+", "p").replace("-", "m").replace(".", "p")
                    image_path = output / "samples" / f"{tag}.png"
                    _render(pc, render_opencv_cam, camera_pose, image_path, args.image_size)
                    face = _largest_face_record(image_path)
                    sample = {
                        "camera_pose": camera_pose,
                        "image": str(image_path),
                        "face_detected": face is not None,
                        "detected_pose": (
                            {axis: float(face["pose"][axis]) for axis in AXES}
                            if face
                            else None
                        ),
                        "detection_confidence": face.get("face_confidence") if face else None,
                    }
                    samples.append(sample)
                    print(json.dumps(sample, ensure_ascii=False))

    valid = [sample for sample in samples if sample["face_detected"]]
    positive = [sample for sample in valid if sample["camera_pose"]["yaw"] > 0]
    negative = [sample for sample in valid if sample["camera_pose"]["yaw"] < 0]
    if len(positive) < 8 or len(negative) < 8:
        raise RuntimeError(
            f"Insufficient detected calibration samples: positive={len(positive)}, negative={len(negative)}"
        )

    profiles = [
        _fit_profile(
            "positive_high_yaw_v1",
            positive,
            {"pitch": [-20.0, 30.0], "yaw": [35.0, 65.0], "roll": [-30.0, 35.0]},
        ),
        _fit_profile(
            "negative_high_yaw_v1",
            negative,
            {"pitch": [-20.0, 30.0], "yaw": [-65.0, -35.0], "roll": [-30.0, 35.0]},
        ),
    ]

    validation = []
    for index, target in enumerate(args.validation_pose):
        profile = _profile_for_target(profiles, target)
        camera_pose = _correct_target(profile, target)
        image_path = output / "validation" / f"validation_{index}_{profile['name']}.png"
        _render(pc, render_opencv_cam, camera_pose, image_path, args.image_size)
        face = _largest_face_record(image_path)
        detected_pose = (
            {axis: float(face["pose"][axis]) for axis in AXES}
            if face
            else None
        )
        validation.append({
            "profile": profile["name"],
            "target_pose": target,
            "corrected_camera_pose": camera_pose,
            "detected_pose": detected_pose,
            "absolute_error_degrees": (
                {
                    axis: abs(float(detected_pose[axis]) - float(target[axis]))
                    for axis in AXES
                }
                if detected_pose
                else None
            ),
            "image": str(image_path),
        })

    calibration = {
        "version": 1,
        "method": "local_affine_inverse_of_insightface_measured_facelift_camera_grid",
        "gaussian_model": str(model_path),
        "gaussian_model_sha256": _sha256(model_path),
        "image_size": args.image_size,
        "axes": list(AXES),
        "camera_grid": {
            "pitch": args.pitch_values,
            "yaw": args.yaw_values,
            "roll": args.roll_values,
        },
        "detected_samples": len(valid),
        "total_samples": len(samples),
        "profiles": profiles,
        "validation": validation,
    }
    calibration_path = args.calibration_output.resolve()
    _write_json(calibration_path, calibration)
    _write_json(samples_path, samples)
    _write_json(output / "calibration_result.json", calibration)
    print(json.dumps(calibration, ensure_ascii=False, indent=2))
    return calibration_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiment_output"
            / "pulid_flux_conservative_mask_04"
            / "input"
            / "facelift"
            / "facelift_raw"
            / "input"
            / "gaussians.ply"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "experiment_output" / "facelift_pose_calibration",
    )
    parser.add_argument("--calibration-output", type=Path)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--reuse-samples", action="store_true")
    parser.add_argument("--pitch-values", type=float, nargs="+", default=[-15.0, 0.0, 15.0])
    parser.add_argument(
        "--yaw-values",
        type=float,
        nargs="+",
        default=[-55.0, -47.5, -40.0, 40.0, 47.5, 55.0],
    )
    parser.add_argument("--roll-values", type=float, nargs="+", default=[-20.0, 0.0, 20.0])
    parser.add_argument(
        "--validation-pose",
        type=_parse_pose,
        action="append",
        default=[
            {"pitch": 13.4097, "yaw": 54.6798, "roll": 10.6905},
            {"pitch": 0.2023, "yaw": -43.7855, "roll": -6.7964},
        ],
    )
    args = parser.parse_args()
    if args.calibration_output is None:
        args.calibration_output = args.model.with_suffix(".pose_calibration.json")
    return args


if __name__ == "__main__":
    run(parse_args())
