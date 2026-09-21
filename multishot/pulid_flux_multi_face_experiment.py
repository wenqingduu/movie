"""Simultaneous multi-face PuLID-FLUX trajectory-residual experiment.

Unlike the single-face runner, the target denoising branch does not receive a
global PuLID identity embedding.  Each character owns an independent 3D render,
pose, harmonization mask and reference trajectory.  All local residuals are
combined simultaneously, so results do not depend on character iteration order.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from itertools import permutations
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

from multishot import pulid_flux_inner_face_experiment as single


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_character_spec(value: str) -> dict:
    """Parse NAME=REFERENCE_IMAGE=FACELIFT_RESULT."""
    parts = value.split("=", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError(
            "--character must be NAME=REFERENCE_IMAGE=FACELIFT_RESULT"
        )
    name, reference, facelift = parts
    return {
        "name": name,
        "reference_image": str(Path(reference).resolve()),
        "facelift_result": str(Path(facelift).resolve()),
    }


def _detected_faces(app, image: Image.Image, minimum_confidence: float) -> list:
    rgb = np.asarray(image.convert("RGB"))
    faces = app.get(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)) or []
    faces = [face for face in faces if float(face.det_score) >= minimum_confidence]
    return sorted(faces, key=lambda face: float((face.bbox[0] + face.bbox[2]) / 2.0))


def _select_faces(faces: list, expected_count: int) -> list:
    """Keep the largest expected faces, then restore deterministic left-to-right order."""
    if len(faces) < expected_count:
        raise RuntimeError(f"detected {len(faces)} reliable faces; expected {expected_count}")
    if len(faces) > expected_count:
        faces = sorted(
            faces,
            key=lambda face: float(
                (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
            ),
            reverse=True,
        )[:expected_count]
    return sorted(faces, key=lambda face: float((face.bbox[0] + face.bbox[2]) / 2.0))


def _face_embedding(face) -> np.ndarray:
    value = np.asarray(face.embedding, dtype=np.float32)
    return value / (np.linalg.norm(value) + 1e-8)


def _assign_characters_to_faces(app, characters: list[dict], faces: list) -> tuple[list[dict], dict]:
    """Assign each detected face to one character with a global one-to-one optimum.

    Prompt mention order is not a spatial contract.  The coarse step-30 faces
    usually retain enough attributes (hair, age, face shape) for InsightFace to
    prevent obvious identity swaps.  We record the full matrix because low
    assignment confidence should remain auditable by a future gate.
    """
    references = [
        _face_embedding(single._largest_face(app, item["reference"]))
        for item in characters
    ]
    candidates = [_face_embedding(face) for face in faces]
    matrix = np.asarray(
        [[float(np.dot(reference, candidate)) for candidate in candidates] for reference in references],
        dtype=np.float32,
    )
    best_score = -float("inf")
    best_columns = None
    for columns in permutations(range(len(faces)), len(characters)):
        score = float(sum(matrix[row, column] for row, column in enumerate(columns)))
        if score > best_score:
            best_score = score
            best_columns = columns
    character_for_face = [None] * len(faces)
    pairs = []
    for character_index, face_index in enumerate(best_columns):
        character_for_face[face_index] = characters[character_index]
        pairs.append({
            "character": characters[character_index]["name"],
            "face_index_left_to_right": int(face_index),
            "cosine": float(matrix[character_index, face_index]),
        })
    return character_for_face, {
        "method": "insightface_global_one_to_one_maximum_cosine",
        "total_score": best_score,
        "cosine_matrix_character_by_face": matrix.tolist(),
        "input_character_order": [item["name"] for item in characters],
        "assigned_characters_left_to_right": [item["name"] for item in character_for_face],
        "pairs": pairs,
    }


def _identity_metrics(app, image: Image.Image, characters: list[dict]) -> dict:
    faces = _detected_faces(app, image, 0.0)
    references = [_face_embedding(single._largest_face(app, item["reference"])) for item in characters]
    detected = [_face_embedding(face) for face in faces]
    matrix = np.asarray(
        [[float(np.dot(reference, candidate)) for candidate in detected] for reference in references],
        dtype=np.float32,
    ) if detected else np.empty((len(references), 0), dtype=np.float32)

    fixed = {}
    for index, item in enumerate(characters):
        fixed[item["name"]] = float(matrix[index, index]) if index < matrix.shape[1] else None

    best = None
    if len(detected) >= len(characters):
        best_score = -float("inf")
        best_assignment = None
        for columns in permutations(range(len(detected)), len(characters)):
            score = float(sum(matrix[row, column] for row, column in enumerate(columns)))
            if score > best_score:
                best_score = score
                best_assignment = columns
        best = {
            item["name"]: {
                "detected_face_index": int(best_assignment[index]),
                "cosine": float(matrix[index, best_assignment[index]]),
            }
            for index, item in enumerate(characters)
        }
    return {
        "detected_face_count": len(faces),
        "fixed_left_to_right": fixed,
        "best_one_to_one": best,
        "cosine_matrix_reference_by_detected_face": matrix.tolist(),
        "detected_bboxes_left_to_right": [
            [float(value) for value in face.bbox] for face in faces
        ],
    }


def _mask_preview(packed_mask, height: int, width: int) -> Image.Image:
    values = (
        packed_mask.float().mean(dim=-1)
        .reshape(math.ceil(height / 16), math.ceil(width / 16))
        .mul(255.0).round().clamp(0, 255).byte().cpu().numpy()
    )
    return Image.fromarray(values, mode="L")


def _build_character_reference(
    *,
    item: dict,
    face,
    preview: Image.Image,
    pulid,
    device: torch.device,
    output_dir: Path,
    strategy: str,
) -> dict:
    target_bbox = single._bbox(face, preview.width, preview.height)
    pose = [float(value) for value in getattr(face, "pose", [0.0, 0.0, 0.0])]
    facelift_asset = single._load_facelift_asset_record(Path(item["facelift_result"]))

    from multishot.mcp_asset_server import _render_3d_face_reference

    gaussian_model_path = Path(facelift_asset["model_path"])
    gaussian_asset_dir = Path(
        facelift_asset.get("facelift_output_dir") or gaussian_model_path.parent
    )
    rendered_path = _render_3d_face_reference(
        {"model_path": str(gaussian_model_path), "path": str(gaussian_asset_dir)},
        {"pitch": pose[0], "yaw": pose[1], "roll": pose[2]},
        target_bbox,
    )
    if not rendered_path:
        raise RuntimeError(f"3D render failed for {item['name']}")
    rendered = Image.open(rendered_path).convert("RGB")
    source_face = single._largest_face(pulid.app, rendered)
    source_bbox = single._bbox(source_face, rendered.width, rendered.height)
    aligned, alignment = single._align_reference(
        rendered,
        source_bbox,
        target_bbox,
        (preview.width, preview.height),
        "facelift_continuous_gaussian_pose_render",
        None,
    )
    aligned_face = single._largest_face(pulid.app, aligned)
    aligned_bbox = single._bbox(aligned_face, aligned.width, aligned.height)

    parser = pulid.face_helper.face_parse
    target_inner = single._semantic_inner_face_mask(preview, target_bbox, parser, device)
    target_features = single._semantic_inner_face_mask(
        preview, target_bbox, parser, device, included_labels=single.IDENTITY_FEATURE_LABELS
    )
    reference_inner = single._semantic_inner_face_mask(aligned, aligned_bbox, parser, device)
    reference_features = single._semantic_inner_face_mask(
        aligned, aligned_bbox, parser, device, included_labels=single.IDENTITY_FEATURE_LABELS
    )
    target_context = single._semantic_probability_mask(
        preview, target_bbox, parser, device, included_labels=single.COLOR_CONTEXT_LABELS
    )
    reference_context = single._semantic_probability_mask(
        aligned, aligned_bbox, parser, device, included_labels=single.COLOR_CONTEXT_LABELS
    )
    target_skin = single._semantic_inner_face_mask(
        preview,
        target_bbox,
        parser,
        device,
        included_labels=(single.HARMONIZATION_POLICY["skin_label"],),
    )
    reference_skin = single._semantic_inner_face_mask(
        aligned,
        aligned_bbox,
        parser,
        device,
        included_labels=(single.HARMONIZATION_POLICY["skin_label"],),
    )

    target_core = single._connected_identity_feature_core_mask(
        target_inner, target_features, target_bbox
    )
    reference_core = single._connected_identity_feature_core_mask(
        reference_inner, reference_features, aligned_bbox
    )
    core_intersection = Image.fromarray(
        np.minimum(np.asarray(target_core), np.asarray(reference_core)).astype(np.uint8),
        mode="L",
    )
    skin_intersection = Image.fromarray(
        np.minimum(np.asarray(target_skin), np.asarray(reference_skin)).astype(np.uint8),
        mode="L",
    )
    color_application, color_metadata = single._color_context_application_mask(
        core_intersection, target_context, reference_context, target_bbox
    )
    if strategy == "legacy_core":
        hard_injection = core_intersection
        safety_metadata = {"source": "identity_core_without_color_safety_margin"}
    elif strategy == "color_safe_core":
        hard_injection, safety_metadata = single._safe_injection_mask_from_color_application(
            core_intersection, color_application, target_bbox
        )
    else:
        raise ValueError(strategy)
    final_mask = hard_injection.filter(
        ImageFilter.GaussianBlur(radius=single.CONSERVATIVE_MASK_POLICY["feather_radius_px"])
    )
    harmonized, harmonization_images, harmonization_metadata = single._harmonize_reference(
        aligned,
        preview,
        skin_intersection,
        color_application,
        final_mask,
        target_bbox,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    rendered.save(output_dir / "rendered_3d_face.png")
    aligned.save(output_dir / "aligned_3d_face.png")
    target_core.save(output_dir / "target_identity_core.png")
    reference_core.save(output_dir / "reference_identity_core.png")
    core_intersection.save(output_dir / "identity_core_intersection.png")
    color_application.save(output_dir / "harmonization_application_mask.png")
    final_mask.save(output_dir / "final_injection_mask.png")
    for filename, image in harmonization_images.items():
        image.save(output_dir / filename)
    _write_json(
        output_dir / "reference_metadata.json",
        {
            "character": item["name"],
            "strategy": strategy,
            "target_bbox": target_bbox,
            "pitch_yaw_roll": pose,
            "det_score": float(face.det_score),
            "alignment": alignment,
            "color_context": color_metadata,
            "injection_safety": safety_metadata,
            "harmonization": harmonization_metadata,
        },
    )
    return {
        **item,
        "bbox": target_bbox,
        "pose": pose,
        "trajectory_reference": harmonized,
        "mask": final_mask,
    }


def _simultaneous_residual(base, references: list[torch.Tensor], alphas: list[torch.Tensor]):
    if not references:
        return base, torch.zeros_like(base), 0.0
    alpha_stack = torch.stack(alphas, dim=0)
    alpha_sum = alpha_stack.sum(dim=0)
    scale = torch.where(alpha_sum > 1.0, alpha_sum.reciprocal(), torch.ones_like(alpha_sum))
    normalized = alpha_stack * scale.unsqueeze(0)
    effective_sum = normalized.sum(dim=0)
    result = base * (1.0 - effective_sum)
    for alpha, reference in zip(normalized, references):
        result = result + alpha * reference
    return result, result - base, float(alpha_sum.max().item())


def run(args) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if len(args.character) < 2:
        raise ValueError("At least two --character specifications are required")
    names = [item["name"] for item in args.character]
    if len(set(names)) != len(names):
        raise ValueError("Character names must be unique")
    for item in args.character:
        for key in ("reference_image", "facelift_result"):
            if not Path(item[key]).is_file():
                raise FileNotFoundError(item[key])
    single._preflight_model_files(args)

    output = Path(args.output_dir).resolve()
    for path in (output / "input", output / "step_30", output / "control"):
        path.mkdir(parents=True, exist_ok=True)
    for strategy in args.strategy:
        (output / strategy / "treatment").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    old_cwd = Path.cwd()
    os.chdir(single.PULID_ROOT)
    try:
        model, ae, t5, clip, pulid = single._load_models(args, device)
        for item in args.character:
            item["reference"] = Image.open(item["reference_image"]).convert("RGB")
            item["reference"].save(output / "input" / f"{item['name']}_reference.png")
        (output / "input" / "prompt.txt").write_text(args.prompt, encoding="utf-8")

        noise = single.get_noise(
            1, args.height, args.width, device=device, dtype=torch.bfloat16, seed=args.seed
        )
        timesteps = single.get_schedule(
            args.steps, noise.shape[-1] * noise.shape[-2] // 4, shift=True
        )
        t5.to(device)
        clip.to(device)
        target_cond = single.prepare(t5=t5, clip=clip, img=noise, prompt=args.prompt)
        reference_cond = target_cond
        t5.cpu(); clip.cpu(); del t5, clip
        gc.collect(); torch.cuda.empty_cache()

        pulid.components_to_device(device)
        pulid.device = device
        pulid.face_helper.device = device
        pulid.face_helper.face_det.device = device
        pulid.face_helper.face_det.mean_tensor = pulid.face_helper.face_det.mean_tensor.to(device)
        for item in args.character:
            resized = single.resize_numpy_image_long(np.asarray(item["reference"]), 1024)
            item["id_embedding"], _ = pulid.get_id_embedding(resized, cal_uncond=False)
        cpu = torch.device("cpu")
        pulid.components_to_device(cpu)
        pulid.device = cpu
        pulid.face_helper.device = cpu
        pulid.face_helper.face_det.device = cpu
        pulid.face_helper.face_det.mean_tensor = pulid.face_helper.face_det.mean_tensor.to(cpu)
        model.to(device); ae.to(device); torch.cuda.empty_cache()

        neutral_id = args.character[0]["id_embedding"]
        current = target_cond["img"]
        logs = []
        with torch.inference_mode():
            for step in range(args.inject_start):
                t, t_next = timesteps[step], timesteps[step + 1]
                pred = single._velocity(
                    model, current, target_cond, t, args.guidance, neutral_id, 0.0
                )
                current = current + (t_next - t) * pred
                logs.append({"step": step, "injected": False})

        actual_start = args.inject_start
        failures = []
        while True:
            detect_t = timesteps[actual_start]
            with torch.inference_mode():
                detect_pred = single._velocity(
                    model, current, target_cond, detect_t, args.guidance, neutral_id, 0.0
                )
                pred_x0 = current - detect_t * detect_pred
                preview = single._decode(ae, pred_x0, args.height, args.width, device)
            preview.save(output / "step_30" / f"pred_x0_step_{actual_start}.png")
            try:
                faces = _select_faces(
                    _detected_faces(pulid.app, preview, args.min_face_confidence),
                    len(args.character),
                )
                break
            except RuntimeError as exc:
                failures.append({"step": actual_start, "reason": str(exc)})
                if len(failures) >= args.max_face_detection_retries or actual_start >= args.steps - 1:
                    raise RuntimeError(f"multi-face detection failed: {failures}") from exc
                t_next = timesteps[actual_start + 1]
                with torch.inference_mode():
                    current = current + (t_next - detect_t) * detect_pred
                actual_start += 1
        args.character, assignment = _assign_characters_to_faces(
            pulid.app, args.character, faces
        )
        names = [item["name"] for item in args.character]
        preview.save(output / "step_30" / "pred_x0.png")
        labeled_preview = preview.copy()
        ImageDraw.Draw(labeled_preview).text(
            (8, 8), "left-to-right: " + ", ".join(names), fill="white"
        )
        labeled_preview.save(output / "step_30" / "pred_x0_labeled.png")

        pulid.face_helper.face_parse.to(device)
        strategy_targets = {}
        for strategy in args.strategy:
            targets = []
            for item, face in zip(args.character, faces):
                character_dir = output / strategy / "references" / item["name"]
                target = _build_character_reference(
                    item=item,
                    face=face,
                    preview=preview,
                    pulid=pulid,
                    device=device,
                    output_dir=character_dir,
                    strategy=strategy,
                )
                packed_mask, latent_mask = single._token_mask(
                    target["mask"], args.height, args.width, current.shape[-1], device, current.dtype
                )
                target["packed_mask"] = packed_mask
                target["latent_mask"] = latent_mask
                _mask_preview(packed_mask, args.height, args.width).save(
                    character_dir / "packed_token_mask.png"
                )
                target["adaptation"] = single._adaptive_face_injection_policy(
                    target["bbox"],
                    args.injection_strength,
                    actual_start,
                    args.steps,
                    enabled=args.adaptive_small_face,
                    max_active_steps=args.max_active_injection_steps,
                )
                target["reference_x0"] = single._encode(
                    ae, target["trajectory_reference"], args.height, args.width, device
                ).to(current.dtype)
                targets.append(target)
            strategy_targets[strategy] = targets
        pulid.face_helper.face_parse.cpu()

        for strategy, targets in strategy_targets.items():
            for target in targets:
                started = time.perf_counter()
                target["trajectory"] = single._reference_trajectory(
                    model,
                    target["reference_x0"],
                    reference_cond,
                    timesteps,
                    args.guidance,
                    target["id_embedding"],
                    args.pulid_id_weight,
                    target["packed_mask"],
                )
                target["trajectory_seconds"] = time.perf_counter() - started

        control = current.detach().clone()
        treatments = {name: current.detach().clone() for name in args.strategy}
        with torch.inference_mode():
            for step in range(actual_start, args.steps):
                t, t_next = timesteps[step], timesteps[step + 1]
                control_pred = detect_pred if step == actual_start else single._velocity(
                    model, control, target_cond, t, args.guidance, neutral_id, 0.0
                )
                control = control + (t_next - t) * control_pred
                step_record = {"step": step, "strategies": {}}
                for strategy, targets in strategy_targets.items():
                    state = treatments[strategy]
                    pred = detect_pred if step == actual_start else single._velocity(
                        model, state, target_cond, t, args.guidance, neutral_id, 0.0
                    )
                    base = state + (t_next - t) * pred
                    references = []
                    alphas = []
                    active_names = []
                    for target in targets:
                        if step >= target["adaptation"]["end_step_exclusive"]:
                            continue
                        references.append(target["trajectory"][step + 1].to(device=device, dtype=base.dtype))
                        alphas.append(
                            target["packed_mask"] * target["adaptation"]["effective_strength"]
                        )
                        active_names.append(target["name"])
                    next_state, residual, raw_alpha_max = _simultaneous_residual(
                        base, references, alphas
                    )
                    if not bool(torch.isfinite(next_state).all().item()):
                        raise FloatingPointError(f"NaN/Inf in {strategy} at step {step}")
                    treatments[strategy] = next_state
                    step_record["strategies"][strategy] = {
                        "active_characters": active_names,
                        "raw_combined_alpha_max": raw_alpha_max,
                        "residual_norm": float(residual.float().norm().item()),
                    }
                logs.append(step_record)

        control_image = single._decode(ae, control, args.height, args.width, device)
        control_image.save(output / "control" / "final.png")
        metrics = {"control": _identity_metrics(pulid.app, control_image, args.character)}
        comparison = [("Control", output / "control" / "final.png")]
        for strategy, state in treatments.items():
            image = single._decode(ae, state, args.height, args.width, device)
            result_path = output / strategy / "treatment" / "final.png"
            image.save(result_path)
            metrics[strategy] = _identity_metrics(pulid.app, image, args.character)
            comparison.append((strategy, result_path))
        single._make_comparison_sheet(comparison, output / "comparison.jpg")

        for strategy in args.strategy:
            for name, value in metrics[strategy]["fixed_left_to_right"].items():
                baseline = metrics["control"]["fixed_left_to_right"].get(name)
                metrics[strategy].setdefault("fixed_delta_vs_control", {})[name] = (
                    value - baseline if value is not None and baseline is not None else None
                )
        _write_json(output / "metrics.json", metrics)
        _write_json(output / "step_log.json", logs)
        _write_json(
            output / "config.json",
            {
                "prompt": args.prompt,
                "seed": args.seed,
                "width": args.width,
                "height": args.height,
                "steps": args.steps,
                "requested_injection_start": args.inject_start,
                "actual_injection_start": actual_start,
                "target_global_pulid_id_weight": 0.0,
                "reference_trajectory_pulid_id_weight": args.pulid_id_weight,
                "injection_strength": args.injection_strength,
                "max_active_injection_steps": args.max_active_injection_steps,
                "character_assignment": assignment,
                "characters_left_to_right": [
                    {
                        "name": item["name"],
                        "reference_image": item["reference_image"],
                        "facelift_result": item["facelift_result"],
                        "bbox": single._bbox(face, preview.width, preview.height),
                        "pitch_yaw_roll": [float(x) for x in getattr(face, "pose", [0, 0, 0])],
                    }
                    for item, face in zip(args.character, faces)
                ],
                "strategies": {
                    "legacy_core": "identity core chosen independently of color-mask boundary",
                    "color_safe_core": "identity core derived from eroded harmonization application mask",
                },
                "detection_failures": failures,
            },
        )
        return output
    finally:
        os.chdir(old_cwd)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--character",
        action="append",
        type=_parse_character_spec,
        required=True,
        help="Repeat once per identity: NAME=REFERENCE_IMAGE=FACELIFT_RESULT",
    )
    parser.add_argument("--strategy", action="append", choices=("legacy_core", "color_safe_core"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--inject-start", type=int, default=30)
    parser.add_argument("--injection-strength", type=float, default=0.4)
    parser.add_argument("--max-active-injection-steps", type=int, default=12)
    parser.add_argument("--adaptive-small-face", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--pulid-id-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=719005)
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--onnx-provider", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--min-face-confidence", type=float, default=0.5)
    parser.add_argument("--max-face-detection-retries", type=int, default=5)
    parser.add_argument("--fp8", action=argparse.BooleanOptionalAction, default=True)
    parser.set_defaults(build_facelift=False, facelift_result=True)
    args = parser.parse_args()
    if not args.strategy:
        args.strategy = ["legacy_core", "color_safe_core"]
    return args


if __name__ == "__main__":
    print(run(parse_args()))
