"""Compare IP-Adapter self-attention with PuLID-style trajectory residual injection."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

from multishot.diffusion_backend import OpenSourceDiffusionBackend
from multishot.ip_adapter_self_attention_experiment import (
    DEFAULT_PROMPT,
    PROJECT_ROOT,
    _compact_state,
    _cosine,
    _embedding,
    _face_app,
    _face_record,
    _largest_face,
    _make_contact_sheet,
    _pixel_delta,
    _write_json,
)
from multishot.mcp_asset_server import _prepare_reference_face_crop, _render_3d_face_reference


def _write_conservative_face_mask(
    image_path: Path,
    face_bbox: list[float],
    output_path: Path,
) -> Path:
    """Use the same geometric facial-core policy as the PuLID conservative mask.

    The SDXL comparison does not load PuLID's semantic face parser, so this is
    the geometric core component only; that deviation is recorded in result.json.
    """

    width, height = Image.open(image_path).size
    x1, y1, x2, y2 = [float(value) for value in face_bbox]
    face_width = max(1.0, x2 - x1)
    face_height = max(1.0, y2 - y1)
    center_x = (x1 + x2) / 2.0
    center_y = y1 + 0.55 * face_height
    radius_x = 0.43 * face_width
    radius_y = 0.43 * face_height

    ellipse = [
        int(round(center_x - radius_x)),
        int(round(center_y - radius_y)),
        int(round(center_x + radius_x)),
        int(round(center_y + radius_y)),
    ]
    vertical = [
        0,
        int(round(y1 + 0.16 * face_height)),
        width,
        int(round(y1 + 0.90 * face_height)),
    ]
    ellipse_mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(ellipse_mask).ellipse(ellipse, fill=255)
    vertical_mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(vertical_mask).rectangle(vertical, fill=255)
    mask = ImageChops.multiply(ellipse_mask, vertical_mask)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=2.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(output_path)
    return output_path


def run(args) -> dict:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference_path = args.reference.resolve()
    continuous_render = args.continuous_render.resolve()
    gaussian_model = args.gaussian_model.resolve() if args.gaussian_model else None
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference portrait not found: {reference_path}")
    if gaussian_model and not gaussian_model.exists():
        raise FileNotFoundError(f"FaceLift Gaussian model not found: {gaussian_model}")
    if not gaussian_model and not continuous_render.exists():
        raise FileNotFoundError(f"Cached continuous FaceLift render not found: {continuous_render}")

    os.environ["MULTISHOT_INSIGHTFACE_MODEL_NAME"] = "antelopev2"
    os.environ["MULTISHOT_INSIGHTFACE_ROOT"] = str(PROJECT_ROOT / "third_party" / "PuLID")
    os.environ["MULTISHOT_DIFFUSION_SEED"] = str(args.seed)
    os.environ["MULTISHOT_DIFFUSION_STEPS"] = str(args.steps)
    os.environ["MULTISHOT_FINAL_STEP"] = str(args.steps)
    os.environ["MULTISHOT_IP_ADAPTER_IMAGE"] = str(reference_path)
    os.environ["MULTISHOT_IP_ADAPTER_SCALE"] = str(args.ip_adapter_scale)
    os.environ["MULTISHOT_REFERENCE_LAYOUT_MODE"] = "match_target_scale"
    os.environ["MULTISHOT_REFERENCE_FACE_SCALE_RATIO"] = str(args.reference_scale)
    os.environ["MULTISHOT_REFERENCE_CROP_SIZE"] = "1024"
    os.environ["MULTISHOT_ATTENTION_INJECTION_SCALE"] = str(args.attention_scale)
    os.environ["MULTISHOT_TRAJECTORY_INJECTION_SCALE"] = "1.0"

    input_dir = output / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    copied_reference = input_dir / "original_reference.jpg"
    copied_render = input_dir / "rendered_3d_face.png"
    shutil.copy2(reference_path, copied_reference)
    if not gaussian_model:
        shutil.copy2(continuous_render, copied_render)

    backend = OpenSourceDiffusionBackend("sdxl-base-1.0-ip-adapter")
    started = time.time()
    runtime = backend.prepare_generation("ip_adapter_pulid_style_comparison", args.prompt, args.steps)
    conditioning = {"prompt": args.prompt, "reference_portrait": str(reference_path)}

    os.environ["MULTISHOT_INJECTION_MODE"] = "off"
    os.environ["MULTISHOT_DYNAMIC_IP_ADAPTER_REFERENCE"] = "0"
    shared_state = backend.denoise_window(
        runtime,
        0,
        args.fork_step,
        previous_denoise_state=None,
        injection_plan={"lambda": 0.0, "targets": []},
        conditioning=conditioning,
    )
    shared_path = output / "shared" / f"step_{args.fork_step:02d}_x0.png"
    backend.estimate_x0_preview(shared_state, str(shared_path))

    app = _face_app()
    target_image = Image.open(shared_path)
    target_face = _face_record(_largest_face(app, shared_path), *target_image.size)
    target_yaw = float(target_face["pose"]["yaw"])
    yaw_gate = {
        "minimum_absolute_yaw": float(args.min_abs_yaw),
        "maximum_absolute_yaw": float(args.max_abs_yaw),
        "detected_yaw": target_yaw,
        "accepted": args.min_abs_yaw <= abs(target_yaw) <= args.max_abs_yaw,
    }
    _write_json(output / "shared" / f"step_{args.fork_step:02d}_yaw_gate.json", yaw_gate)
    if not yaw_gate["accepted"]:
        raise RuntimeError(
            f"Step-{args.fork_step} absolute yaw {abs(target_yaw):.4f} is outside "
            f"the requested [{args.min_abs_yaw:.4f}, {args.max_abs_yaw:.4f}] range"
        )
    if gaussian_model:
        rendered_path = _render_3d_face_reference(
            {
                "model_path": str(gaussian_model),
                "path": str(input_dir / "continuous_gaussian"),
            },
            target_face["pose"],
            target_face["face_bbox"],
        )
        if not rendered_path:
            raise RuntimeError("FaceLift continuous Gaussian pose render failed")
        shutil.copy2(rendered_path, copied_render)
    target_mask = _write_conservative_face_mask(
        shared_path,
        target_face["face_bbox"],
        output / "shared" / f"step_{args.fork_step:02d}_conservative_face_mask.png",
    )
    reference_layout = _prepare_reference_face_crop(str(copied_render), target_face["face_bbox"])
    target = {
        "face_id": "face_0",
        "mask_path": str(target_mask),
        "face_bbox": target_face["face_bbox"],
        "matched_character_id": "experiment_subject",
        **reference_layout,
    }
    injection_plan = {
        "lambda": args.injection_lambda,
        "candidate_id": "fixed_lambda",
        "targets": [target],
    }

    # Dynamic 3D IP-Adapter is disabled in every branch. The original portrait
    # remains the same global IP-Adapter condition; only the local 3D operator changes.
    branch_specs = [
        ("ip_adapter_baseline", "off", {"lambda": 0.0, "targets": []}),
        ("ip_adapter_plus_self_attention", "attention", injection_plan),
        ("ip_adapter_plus_pulid_style_residual", "trajectory_residual", injection_plan),
    ]
    branch_outputs: dict[str, Path] = {}
    for branch_name, injection_mode, branch_plan in branch_specs:
        os.environ["MULTISHOT_INJECTION_MODE"] = injection_mode
        os.environ["MULTISHOT_DYNAMIC_IP_ADAPTER_REFERENCE"] = "0"
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
        _write_json(output / "logs" / f"{branch_name}.json", _compact_state(branch_state))

    reference_embedding = _embedding(_largest_face(app, copied_reference))
    render_face = _largest_face(app, copied_render)
    render_embedding = _embedding(render_face)
    render_image = Image.open(copied_render)
    render_face_record = _face_record(render_face, *render_image.size)
    baseline_path = branch_outputs["ip_adapter_baseline"]
    metrics = {}
    for branch_name, branch_path in branch_outputs.items():
        output_face = _largest_face(app, branch_path)
        output_embedding = _embedding(output_face)
        output_image = Image.open(branch_path)
        metrics[branch_name] = {
            "reference_portrait_cosine": _cosine(reference_embedding, output_embedding),
            "continuous_3d_render_cosine": _cosine(render_embedding, output_embedding),
            "face_detected": output_face is not None,
            "final_face": _face_record(output_face, *output_image.size),
            **_pixel_delta(baseline_path, branch_path, target_face["face_bbox"]),
        }

    contact_sheet = output / "comparison.jpg"
    _make_contact_sheet(
        [
            ("original portrait / IP-Adapter", copied_reference),
            ("continuous FaceLift 3D render", copied_render),
            (f"shared x0 at step {args.fork_step}", shared_path),
            ("IP-Adapter baseline", branch_outputs["ip_adapter_baseline"]),
            ("IP-Adapter + masked self-attention", branch_outputs["ip_adapter_plus_self_attention"]),
            ("IP-Adapter + PuLID-style residual", branch_outputs["ip_adapter_plus_pulid_style_residual"]),
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
        "ip_adapter_scale": args.ip_adapter_scale,
        "injection_lambda": args.injection_lambda,
        "attention_scale": args.attention_scale,
        "effective_self_attention_strength": round(args.injection_lambda * args.attention_scale, 4),
        "effective_trajectory_residual_strength": args.injection_lambda,
        "dynamic_3d_ip_adapter_enabled": False,
        "target_face": target_face,
        "yaw_gate": yaw_gate,
        "continuous_3d_render_face": render_face_record,
        "continuous_3d_reference": {
            "mode": "target_pose_gaussian_render" if gaussian_model else "pre_rendered_image",
            "gaussian_model": str(gaussian_model) if gaussian_model else None,
            "provided_render": str(continuous_render) if not gaussian_model else None,
        },
        "reference_layout": reference_layout,
        "mask_policy": {
            "name": "conservative_geometric_face_core",
            "matches_pulid_geometric_core": True,
            "semantic_face_parser_intersection": False,
            "documented_deviation": "SDXL comparison uses the PuLID conservative geometric core without BiSeNet semantic intersection",
        },
        "trajectory_policy": {
            "formula": "target_next += strength * mask * (reference_next - target_next)",
            "reference_state": "VAE x0 noised with fixed reference noise at the target next scheduler timestep",
            "final_state": "clean reference VAE x0",
        },
        "paths": {
            "comparison": str(contact_sheet),
            "reference_portrait": str(copied_reference),
            "continuous_3d_render": str(copied_render),
            "scale_matched_3d_layout": reference_layout["reference_image"],
            "target_face_mask": str(target_mask),
            "shared_step_x0": str(shared_path),
            "branches": {name: str(path) for name, path in branch_outputs.items()},
        },
        "branch_definition": {
            "ip_adapter_baseline": "original portrait IP-Adapter only",
            "ip_adapter_plus_self_attention": "original portrait IP-Adapter plus local 3D masked mutual self-attention",
            "ip_adapter_plus_pulid_style_residual": "original portrait IP-Adapter plus local same-timestep 3D trajectory residual",
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
        "--continuous-render",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiment_output"
            / "ip_adapter_self_attention_comparison"
            / "input"
            / "rendered_3d_face.png"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "experiment_output" / "ip_adapter_pulid_style_comparison",
    )
    parser.add_argument("--gaussian-model", type=Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--fork-step", type=int, default=30)
    parser.add_argument("--ip-adapter-scale", type=float, default=0.6)
    parser.add_argument("--injection-lambda", type=float, default=0.4)
    parser.add_argument("--attention-scale", type=float, default=0.85)
    parser.add_argument("--reference-scale", type=float, default=1.0)
    parser.add_argument("--min-abs-yaw", type=float, default=0.0)
    parser.add_argument("--max-abs-yaw", type=float, default=90.0)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
