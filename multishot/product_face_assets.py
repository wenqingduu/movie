"""Prepare project-local FaceLift pose calibration for product face assets."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def calibrate_product_face(face_3d: dict) -> dict:
    result = dict(face_3d)
    model = Path(result.get("model_path", "")).resolve()
    if result.get("facelift_status") == "failed" or model.suffix != ".ply" or not model.is_file():
        raise RuntimeError(f"FaceLift did not produce a usable Gaussian model: {model}")
    calibration = model.with_suffix(".pose_calibration.json")
    if not calibration.is_file():
        root = Path(__file__).resolve().parents[1]
        output = Path(result["path"]) / "pose_calibration"
        output.mkdir(parents=True, exist_ok=True)
        with (output / "calibration.log").open("w") as log:
            subprocess.run(
                [
                    os.getenv("MULTISHOT_FACE_CALIBRATION_PYTHON", sys.executable),
                    "-m", "multishot.facelift_pose_calibration",
                    "--model", str(model), "--output", str(output),
                    "--calibration-output", str(calibration),
                    "--image-size", os.getenv("MULTISHOT_FACE_CALIBRATION_SIZE", "512"),
                ],
                cwd=root, check=True, stdout=log, stderr=subprocess.STDOUT,
                timeout=int(os.getenv("MULTISHOT_FACE_CALIBRATION_TIMEOUT", "1800")),
            )
    if not calibration.is_file():
        raise FileNotFoundError(f"Missing role-specific pose calibration: {calibration}")
    result["pose_calibration_path"] = str(calibration)
    return result
