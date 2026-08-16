"""Compare the project's three SDXL face-conditioning paths on one 3D face.

The three branches share the same initial noise and the same state at step 30:

1. ``ip_adapter_baseline``: the original portrait conditions IP-Adapter only.
2. ``dynamic_ip_adapter_only``: the 3D render is blended into IP-Adapter embeds.
3. ``ip_adapter_plus_self_attention``: dynamic IP-Adapter plus the project's
   masked mutual self-attention injection of the scale-matched 3D face.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from multishot.diffusion_backend import OpenSourceDiffusionBackend
from multishot.mcp_asset_server import (
    _prepare_reference_face_crop,
    _render_3d_face_reference,
    _write_face_mask,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = (
    "cinematic medium close-up portrait of a man standing beneath warm neon lights "
    "on a rainy night street, three-quarter view, natural skin texture, shallow depth "
    "of field, photorealistic, subtle rim light"
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _face_app():
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="antelopev2",
        root=str(PROJECT_ROOT / "third_party" / "PuLID"),
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def _largest_face(app, image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    faces = app.get(image)
    if not faces:
        return None
    return max(
        faces,
        key=lambda face: float(
            (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
        ),
    )


def _face_record(face, width: int, height: int) -> dict:
    if face is None:
        return {
            "face_id": "face_0",
            "face_bbox": [int(width * 0.33), int(height * 0.16), int(width * 0.67), int(height * 0.68)],
            "pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
            "detection_source": "fallback_center_bbox",
        }
    pose = getattr(face, "pose", [0.0, 0.0, 0.0])
    return {
        "face_id": "face_0",
        "face_bbox": [round(float(value), 2) for value in face.bbox],
        "pose": {
            "pitch": round(float(pose[0]), 4),
            "yaw": round(float(pose[1]), 4),
            "roll": round(float(pose[2]), 4),
        },
        "detection_source": "antelopev2",
    }


def _embedding(face):
    if face is None:
        return None
    return np.asarray(face.normed_embedding, dtype=np.float32)


def _cosine(left, right):
    if left is None or right is None:
        return None
    return round(float(np.dot(left, right)), 6)


def _pixel_delta(baseline_path: Path, candidate_path: Path, bbox: list[float]) -> dict:
    baseline = np.asarray(Image.open(baseline_path).convert("RGB"), dtype=np.float32)
    candidate = np.asarray(Image.open(candidate_path).convert("RGB"), dtype=np.float32)
    delta = np.abs(candidate - baseline)
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(delta.shape[1], x2), min(delta.shape[0], y2)
    face_delta = delta[y1:y2, x1:x2]
    return {
        "whole_image_rgb_mae_vs_baseline": round(float(delta.mean()), 6),
        "target_face_bbox_rgb_mae_vs_baseline": round(float(face_delta.mean()), 6) if face_delta.size else None,
    }


def _make_contact_sheet(items: list[tuple[str, Path]], output_path: Path) -> None:
    thumb_size = (420, 420)
    label_height = 42
    columns = 3
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_size[0], rows * (thumb_size[1] + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, path) in enumerate(items):
        image = Image.open(path).convert("RGB")
        image.thumbnail(thumb_size, Image.Resampling.LANCZOS)
        x = (index % columns) * thumb_size[0] + (thumb_size[0] - image.width) // 2
        y = (index // columns) * (thumb_size[1] + label_height)
        sheet.paste(image, (x, y))
        draw.text(((index % columns) * thumb_size[0] + 8, y + thumb_size[1] + 10), label, fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _compact_state(state: dict) -> dict:
    return {
        key: value
        for key, value in state.items()
        if not key.startswith("_")
    }


def run(args) -> dict:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference_path = args.reference.resolve()
    facelift_result_path = args.facelift_result.resolve()
    facelift_result = (
        json.loads(facelift_result_path.read_text(encoding="utf-8"))
        if facelift_result_path.exists()
        else {}
    )
    model_path_value = facelift_result.get("model_path")
    model_path = Path(model_path_value) if model_path_value else None
    input_dir = output / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    copied_render = input_dir / "rendered_3d_face.png"
    if (model_path is None or not model_path.exists()) and not copied_render.exists():
        raise FileNotFoundError(
            f"Neither the FaceLift Gaussian model nor a cached continuous render exists: {model_path}"
        )

    # Reuse the already downloaded AntelopeV2 package rather than triggering a
    # second InsightFace model download through the main wrapper.
    os.environ["MULTISHOT_INSIGHTFACE_MODEL_NAME"] = "antelopev2"
    os.environ["MULTISHOT_INSIGHTFACE_ROOT"] = str(PROJECT_ROOT / "third_party" / "PuLID")
    os.environ["MULTISHOT_DIFFUSION_SEED"] = str(args.seed)
    os.environ["MULTISHOT_DIFFUSION_STEPS"] = str(args.steps)
    os.environ["MULTISHOT_FINAL_STEP"] = str(args.steps)
    os.environ["MULTISHOT_IP_ADAPTER_IMAGE"] = str(reference_path)
    os.environ["MULTISHOT_REFERENCE_LAYOUT_MODE"] = "match_target_scale"
    os.environ["MULTISHOT_REFERENCE_FACE_SCALE_RATIO"] = str(args.reference_scale)
    os.environ["MULTISHOT_REFERENCE_CROP_SIZE"] = "1024"
    os.environ["MULTISHOT_DYNAMIC_IP_ADAPTER_SCALE"] = "1.0"
    os.environ["MULTISHOT_ATTENTION_INJECTION_SCALE"] = str(args.attention_scale)

    backend = OpenSourceDiffusionBackend("sdxl-base-1.0-ip-adapter")
    started = time.time()
    runtime = backend.prepare_generation("ip_adapter_face_comparison", args.prompt, args.steps)
    conditioning = {"prompt": args.prompt, "reference_portrait": str(reference_path)}

    # The first 30 steps are shared exactly by all branches.
    os.environ["MULTISHOT_INJECTION_MODE"] = "attention"
    os.environ["MULTISHOT_DYNAMIC_IP_ADAPTER_REFERENCE"] = "0"
    shared_state = backend.denoise_window(
        runtime,
        0,
        args.fork_step,
        previous_denoise_state=None,
        injection_plan={"lambda": 0.0, "targets": []},
        conditioning=conditioning,
    )
    step30_path = output / "shared" / f"step_{args.fork_step:02d}_x0.png"
    backend.estimate_x0_preview(shared_state, str(step30_path))

    app = _face_app()
    target_image = Image.open(step30_path)
    target_face = _face_record(_largest_face(app, step30_path), *target_image.size)
    target_mask_path = Path(_write_face_mask(str(step30_path), target_face))

    if not copied_render.exists():
        render_path = _render_3d_face_reference(
            {
                "model_path": str(model_path),
                "path": facelift_result.get("face_asset_dir") or str(model_path.parent),
            },
            target_face["pose"],
            target_face["face_bbox"],
        )
        if not render_path:
            raise RuntimeError("Continuous FaceLift Gaussian render failed; no discrete-view fallback is used")
        shutil.copy2(render_path, copied_render)
    copied_reference = input_dir / "original_reference.jpg"
    shutil.copy2(reference_path, copied_reference)

    reference_layout = _prepare_reference_face_crop(
        str(copied_render),
        target_face["face_bbox"],
    )
    target = {
        "face_id": "face_0",
        "mask_path": str(target_mask_path),
        "face_bbox": target_face["face_bbox"],
        "matched_character_id": "experiment_subject",
        **reference_layout,
    }
    injection_plan = {
        "lambda": args.injection_lambda,
        "candidate_id": "fixed_lambda",
        "targets": [target],
    }

    branch_specs = [
        ("ip_adapter_baseline", "off", "0", {"lambda": 0.0, "targets": []}),
        ("dynamic_ip_adapter_only", "off", "1", injection_plan),
        ("ip_adapter_plus_self_attention", "attention", "1", injection_plan),
    ]
    branch_outputs = {}
    branch_logs = {}
    for branch_name, injection_mode, dynamic_ip, branch_plan in branch_specs:
        os.environ["MULTISHOT_INJECTION_MODE"] = injection_mode
        os.environ["MULTISHOT_DYNAMIC_IP_ADAPTER_REFERENCE"] = dynamic_ip
        branch_state = backend.denoise_window(
            runtime,
            args.fork_step,
            args.steps,
            previous_denoise_state=shared_state,
            injection_plan=branch_plan,
            conditioning=conditioning,
        )
        branch_path = output / "branches" / f"{branch_name}.png"
        backend.decode_final_image(branch_state, str(branch_path))
        branch_outputs[branch_name] = branch_path
        branch_logs[branch_name] = _compact_state(branch_state)
        _write_json(output / "logs" / f"{branch_name}.json", branch_logs[branch_name])

    reference_face = _largest_face(app, copied_reference)
    render_face = _largest_face(app, copied_render)
    reference_embedding = _embedding(reference_face)
    render_embedding = _embedding(render_face)
    metrics = {}
    baseline_path = branch_outputs["ip_adapter_baseline"]
    for branch_name, branch_path in branch_outputs.items():
        output_face = _largest_face(app, branch_path)
        output_embedding = _embedding(output_face)
        metrics[branch_name] = {
            "reference_portrait_cosine": _cosine(reference_embedding, output_embedding),
            "continuous_3d_render_cosine": _cosine(render_embedding, output_embedding),
            "face_detected": output_face is not None,
            **_pixel_delta(baseline_path, branch_path, target_face["face_bbox"]),
        }

    contact_sheet = output / "comparison.jpg"
    _make_contact_sheet(
        [
            ("original portrait / IP-Adapter", copied_reference),
            ("continuous FaceLift 3D render", copied_render),
            (f"shared x0 at step {args.fork_step}", step30_path),
            ("IP-Adapter baseline", branch_outputs["ip_adapter_baseline"]),
            ("dynamic IP-Adapter only", branch_outputs["dynamic_ip_adapter_only"]),
            ("IP-Adapter + masked self-attention", branch_outputs["ip_adapter_plus_self_attention"]),
        ],
        contact_sheet,
    )

    result = {
        "status": "completed",
        "model": "sdxl-base-1.0-ip-adapter",
        "prompt": args.prompt,
        "seed": args.seed,
        "total_steps": args.steps,
        "fork_step": args.fork_step,
        "injection_lambda": args.injection_lambda,
        "attention_scale": args.attention_scale,
        "effective_self_attention_strength": round(args.injection_lambda * args.attention_scale, 4),
        "reference_face_scale_ratio": args.reference_scale,
        "target_face": target_face,
        "reference_layout": reference_layout,
        "paths": {
            "reference_portrait": str(copied_reference),
            "continuous_3d_render": str(copied_render),
            "scale_matched_3d_layout": reference_layout["reference_image"],
            "target_face_mask": str(target_mask_path),
            "shared_step_x0": str(step30_path),
            "comparison": str(contact_sheet),
            "branches": {name: str(path) for name, path in branch_outputs.items()},
        },
        "branch_definition": {
            "ip_adapter_baseline": "original portrait through IP-Adapter; no 3D injection",
            "dynamic_ip_adapter_only": "blend continuous 3D render into IP-Adapter image embeds; no self-attention injection",
            "ip_adapter_plus_self_attention": "same dynamic IP-Adapter blend plus masked mutual self-attention from scale-matched 3D face",
        },
        "metrics": metrics,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    _write_json(output / "result.json", result)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=Path,
        default=PROJECT_ROOT / "experiment_assets" / "pulid_reference.jpg",
    )
    parser.add_argument(
        "--facelift-result",
        type=Path,
        default=PROJECT_ROOT / "experiment_output" / "pulid_flux_smoke" / "input" / "facelift_result.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "experiment_output" / "ip_adapter_self_attention_comparison",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--fork-step", type=int, default=30)
    parser.add_argument("--injection-lambda", type=float, default=0.6)
    parser.add_argument("--attention-scale", type=float, default=0.85)
    parser.add_argument("--reference-scale", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
