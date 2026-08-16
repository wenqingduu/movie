"""Compare IP-Adapter self-attention with PuLID-style trajectory residual injection."""

from __future__ import annotations

import argparse
import gc
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
    """Write the geometric facial-core fallback and pre-harmonization diagnostic.

    Harmonized runs replace this with the target/reference BiSeNet semantic
    intersection produced by ``_harmonize_3d_reference_layout``.
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


def _harmonize_3d_reference_layout(
    target_preview_path: Path,
    reference_layout_path: Path,
    target_bbox: list[float],
    app,
    output_dir: Path,
) -> tuple[Path, Path, dict]:
    """Apply the PuLID pure-3D harmonization policy to an aligned SDXL reference."""
    import numpy as np
    import torch
    from facexlib.parsing import init_parsing_model

    from multishot.pulid_flux_inner_face_experiment import (
        HARMONIZATION_POLICY,
        INNER_FACE_LABELS,
        _conservative_face_core_mask,
        _harmonize_reference,
        _semantic_inner_face_mask,
    )

    target_preview = Image.open(target_preview_path).convert("RGB")
    aligned_reference = Image.open(reference_layout_path).convert("RGB")
    target_bbox_int = [int(round(value)) for value in target_bbox]
    aligned_face = _largest_face(app, reference_layout_path)
    if aligned_face is None:
        raise RuntimeError("No face detected in the aligned 3D reference for harmonization")
    aligned_bbox = [int(round(float(value))) for value in aligned_face.bbox]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parsing_model = init_parsing_model(model_name="bisenet", device=device)
    target_semantic = _semantic_inner_face_mask(
        target_preview,
        target_bbox_int,
        parsing_model,
        device,
        included_labels=INNER_FACE_LABELS,
    )
    reference_semantic = _semantic_inner_face_mask(
        aligned_reference,
        aligned_bbox,
        parsing_model,
        device,
        included_labels=INNER_FACE_LABELS,
    )
    target_skin = _semantic_inner_face_mask(
        target_preview,
        target_bbox_int,
        parsing_model,
        device,
        included_labels=(HARMONIZATION_POLICY["skin_label"],),
    )
    reference_skin = _semantic_inner_face_mask(
        aligned_reference,
        aligned_bbox,
        parsing_model,
        device,
        included_labels=(HARMONIZATION_POLICY["skin_label"],),
    )
    parsing_model.cpu()
    del parsing_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    skin_intersection = Image.fromarray(
        np.minimum(np.asarray(target_skin), np.asarray(reference_skin)).astype(np.uint8),
        mode="L",
    )
    color_application = Image.fromarray(
        np.minimum(np.asarray(target_semantic), np.asarray(reference_semantic)).astype(np.uint8),
        mode="L",
    )
    target_core = _conservative_face_core_mask(target_semantic, target_bbox_int)
    reference_core = _conservative_face_core_mask(reference_semantic, aligned_bbox)
    injection_intersection = np.minimum(
        np.asarray(target_core), np.asarray(reference_core)
    ).astype(np.uint8)
    final_injection_mask = Image.fromarray(injection_intersection, mode="L").filter(
        ImageFilter.GaussianBlur(radius=2.0)
    )
    _, harmonization_images, metadata = _harmonize_reference(
        aligned_reference,
        target_preview,
        skin_intersection,
        color_application,
        final_injection_mask,
        target_bbox_int,
    )
    metadata["reference_mode"] = "pure_3d"
    metadata["estimation_mask"] = "target/reference skin-label intersection"
    metadata["application_mask"] = "target/reference complete inner-face intersection"
    metadata["target_bbox"] = target_bbox_int
    metadata["reference_bbox"] = aligned_bbox

    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = {
        "target_semantic_inner_face_mask.png": target_semantic,
        "reference_semantic_inner_face_mask.png": reference_semantic,
        "target_skin_mask.png": target_skin,
        "reference_skin_mask.png": reference_skin,
        "skin_intersection_mask.png": skin_intersection,
        "target_conservative_inner_face_mask.png": target_core,
        "reference_conservative_inner_face_mask.png": reference_core,
        "final_injection_mask.png": final_injection_mask,
        **harmonization_images,
    }
    for filename, image in diagnostics.items():
        if filename in {"harmonization_blend_mask.png", "harmonized_reference.png"}:
            continue
        image.save(output_dir / filename)
    _write_json(output_dir / "harmonization.json", metadata)
    return (
        output_dir / "harmonized_3d_face.png",
        output_dir / "final_injection_mask.png",
        metadata,
    )


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
    unharmonized_reference_image = reference_layout["reference_image"]
    harmonization_metadata = None
    if args.harmonize_reference:
        harmonized_reference, target_mask, harmonization_metadata = _harmonize_3d_reference_layout(
            shared_path,
            Path(unharmonized_reference_image),
            target_face["face_bbox"],
            app,
            input_dir / "harmonization",
        )
        reference_layout["unharmonized_reference_image"] = unharmonized_reference_image
        reference_layout["reference_image"] = str(harmonized_reference)
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
            (
                "harmonized aligned 3D reference"
                if args.harmonize_reference
                else "aligned 3D reference",
                Path(reference_layout["reference_image"]),
            ),
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
        "harmonize_reference": args.harmonize_reference,
        "harmonization_policy": harmonization_metadata,
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
            "name": (
                "conservative_semantic_inner_face_intersection"
                if args.harmonize_reference
                else "conservative_geometric_face_core"
            ),
            "matches_pulid_geometric_core": True,
            "semantic_face_parser_intersection": args.harmonize_reference,
            "documented_deviation": (
                None
                if args.harmonize_reference
                else "SDXL comparison uses the PuLID conservative geometric core without BiSeNet semantic intersection"
            ),
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
            "unharmonized_scale_matched_3d_layout": unharmonized_reference_image,
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
    parser.add_argument(
        "--harmonize-reference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use target low-frequency illumination on a pure 3D inner-face reference",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
