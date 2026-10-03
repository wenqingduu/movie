"""Product Qwen-Image-2.1 first-frame generation with v7 3D face injection.

One denoising trajectory consumes the scene and character references. At step
12 its predicted clean image supplies face mapping and pose for per-character
color-safe 3D residuals. Only the final first frame is generated.
"""

from __future__ import annotations

import hashlib
import gc
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image

from .prompt_injection_safety import prompt_requests_closed_eyes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PIPELINE = None
_PIPELINE_KEY: tuple[str, str] | None = None
_FACE_APP = None
_FACE_PARSER = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _asset_path(asset: dict[str, Any], label: str) -> Path:
    value = asset.get("path")
    if not value:
        raise FileNotFoundError(f"Missing {label} path")
    path = Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def _build_reference_inputs(
    scene_asset: dict[str, Any],
    character_assets: dict[str, Any],
    character_ids: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ordered Qwen inputs and character-only injection records."""

    references = [
        {
            "kind": "scene",
            "id": scene_asset.get("asset_id", "scene_background"),
            "path": _asset_path(scene_asset, "scene reference"),
        }
    ]
    characters = []
    for character_id in character_ids:
        asset = character_assets.get(character_id)
        if not asset:
            raise FileNotFoundError(f"Missing character asset for {character_id}")
        reference_path = _asset_path(asset, f"character reference for {character_id}")
        display_name = asset.get("character_name") or character_id
        references.append(
            {
                "kind": "character",
                "id": character_id,
                "name": display_name,
                "path": reference_path,
            }
        )
        characters.append(
            {
                "name": character_id,
                "display_name": display_name,
                "reference_image": str(reference_path),
                "face_3d": asset.get("face_3d"),
            }
        )
    return references, characters


def _build_qwen_prompt(prompt: str, references: list[dict[str, Any]]) -> str:
    scene = references[0]
    characters = [item for item in references if item["kind"] == "character"]
    clauses = [
        f"Image 1 is the exact scene and background reference ({scene['id']}).",
        "Preserve its location, architecture, lighting, weather, and visual continuity.",
    ]
    for index, item in enumerate(characters, 2):
        clauses.append(
            f"Image {index} is the exact identity reference for {item['name']} "
            f"(role id {item['id']})."
        )
    if characters:
        roster = ", ".join(item["name"] for item in characters)
        clauses.extend(
            [
                f"Generate exactly {len(characters)} named "
                f"{'person' if len(characters) == 1 else 'people'}: {roster}.",
                "Preserve each referenced person's identity, age, facial structure, "
                "hairstyle, clothing cues, and assigned role.",
                "Do not merge, swap, duplicate, or add named identities.",
            ]
        )
    else:
        clauses.append("Generate the scene without any named foreground character.")
    clauses.append(prompt.strip())
    return " ".join(clause for clause in clauses if clause)


def _load_pipeline(model: Path, offload_mode: str):
    global _PIPELINE, _PIPELINE_KEY

    import torch
    from diffusers import QwenImage21Pipeline

    key = (str(model), offload_mode)
    if _PIPELINE is not None and _PIPELINE_KEY == key:
        return _PIPELINE
    pipe = QwenImage21Pipeline.from_pretrained(
        str(model), dtype=torch.bfloat16, local_files_only=True
    )
    if offload_mode == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif offload_mode == "model":
        pipe.enable_model_cpu_offload()
    elif offload_mode == "cuda":
        pipe.to("cuda")
    else:
        raise ValueError(f"Unsupported QWEN_IMAGE21_OFFLOAD_MODE: {offload_mode}")
    _PIPELINE = pipe
    _PIPELINE_KEY = key
    return pipe


def _load_face_runtime():
    global _FACE_APP, _FACE_PARSER

    import torch
    from facexlib.parsing import init_parsing_model

    from .ip_adapter_experiment_utils import _face_app

    if _FACE_APP is None:
        _FACE_APP = _face_app()
    if _FACE_PARSER is None:
        _FACE_PARSER = init_parsing_model(
            model_name="bisenet", device=torch.device("cuda")
        )
    return _FACE_APP, _FACE_PARSER


def _decode(pipe, packed, width: int, height: int) -> Image.Image:
    import torch

    latents = pipe._unpack_latents(packed, height, width, pipe.vae_scale_factor)
    latents = latents.to(pipe.vae.dtype)
    mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    std = (
        torch.tensor(pipe.vae.config.latents_std)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    decoded = pipe.vae.decode(latents * std + mean, return_dict=False)[0][:, :, 0]
    return pipe.image_processor.postprocess(decoded, output_type="pil")[0]




def _prepare_conditioning(
    pipe,
    *,
    prompt: str,
    images: list[Image.Image],
    width: int,
    height: int,
    output_resolution: int,
    steps: int,
    seed: int,
    use_kv_cache: bool,
) -> dict[str, Any]:
    import torch
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import (
        calculate_dimensions,
        calculate_shift,
        retrieve_timesteps,
    )

    device = pipe._execution_device
    condition_images = [image.convert("RGBA") for image in images]
    input_sizes, input_images, vae_images = [], [], []
    for image in condition_images:
        input_width, input_height, _ = calculate_dimensions(
            output_resolution * output_resolution, image.width / image.height
        )
        input_sizes.append((input_width, input_height))
        input_images.append(
            pipe.image_processor.resize(image, width=input_width, height=input_height)
        )
        vae_images.append(
            pipe.image_processor.preprocess(
                image, width=input_width, height=input_height
            ).unsqueeze(2)
        )

    prompt_embeds, prompt_mask, image_pad_mask = pipe.encode_prompt(
        image=input_images,
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
    )
    generator = torch.Generator(device="cuda").manual_seed(seed)
    latent_channels = pipe.transformer.config.in_channels
    initial, condition_latents = pipe.prepare_latents(
        vae_images,
        1,
        latent_channels,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        None,
    )
    latent_height = 2 * (height // (pipe.vae_scale_factor * 2))
    latent_width = 2 * (width // (pipe.vae_scale_factor * 2))
    img_shapes = [[
        *[
            (1, image_height // pipe.vae_scale_factor, image_width // pipe.vae_scale_factor)
            for image_width, image_height in input_sizes
        ],
        (1, height // pipe.vae_scale_factor, width // pipe.vae_scale_factor),
    ]]
    image_pad_mask = torch.cat(
        [
            image_pad_mask,
            image_pad_mask.new_ones(image_pad_mask.shape[0], initial.shape[1] // 4),
        ],
        dim=1,
    )
    sigmas_input = np.linspace(1.0, 1 / steps, steps)
    mu = calculate_shift(
        initial.shape[1],
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, _ = retrieve_timesteps(
        pipe.scheduler, steps, device, sigmas=sigmas_input, mu=mu
    )
    cache = (
        QwenImage21KVCache(len(pipe.transformer.transformer_blocks))
        if use_kv_cache and pipe.transformer.config.causal_condition
        else None
    )
    return {
        "device": device,
        "prompt_embeds": prompt_embeds,
        "prompt_mask": prompt_mask,
        "image_pad_mask": image_pad_mask,
        "condition_latents": condition_latents,
        "initial": initial,
        "latent_height": latent_height,
        "latent_width": latent_width,
        "img_shapes": img_shapes,
        "timesteps": timesteps,
        "sigmas": pipe.scheduler.sigmas,
        "cache": cache,
    }


def _predict(pipe, state: dict[str, Any], latents, step_index: int):
    import torch

    timestep = state["timesteps"][step_index]
    model_input = torch.cat([state["condition_latents"], latents], dim=1)
    expanded_timestep = timestep.expand(latents.shape[0]).to(latents.dtype)
    cache = state["cache"]
    kv_mode = (
        "extract"
        if cache is not None and step_index == 0
        else ("cached" if cache is not None else None)
    )
    with pipe.transformer.cache_context("cond"):
        prediction = pipe.transformer(
            hidden_states=model_input,
            timestep=expanded_timestep / 1000,
            encoder_hidden_states=state["prompt_embeds"],
            encoder_hidden_states_mask=state["prompt_mask"],
            img_shapes=state["img_shapes"],
            img_mask=state["image_pad_mask"],
            attention_kwargs={},
            kv_cache=cache,
            kv_cache_mode=kv_mode,
            return_dict=False,
        )[0]
    return prediction[:, -latents.size(1) :]


def _euler_step(sample, model_output, sigmas, step_index: int):
    dt = sigmas[step_index + 1] - sigmas[step_index]
    return (sample.float() + dt * model_output.float()).to(model_output.dtype)


def _encode_reference(pipe, image: Image.Image, width: int, height: int, generator):
    value = pipe.image_processor.preprocess(
        image.convert("RGBA"), width=width, height=height
    )
    value = value.unsqueeze(2).to(device=pipe._execution_device, dtype=pipe.vae.dtype)
    encoded = pipe._encode_vae_image(value, generator)
    return pipe._pack_latents(
        encoded,
        encoded.shape[0],
        encoded.shape[1],
        encoded.shape[3],
        encoded.shape[4],
    )


def _slug(value: str) -> str:
    return "_".join(value.lower().replace("-", " ").split())


def _prepare_color_safe_references(
    *,
    preview: Image.Image,
    characters: list[dict[str, Any]],
    output_dir: Path,
    minimum_confidence: float,
    minimum_face_height: float,
    app,
    parser,
) -> dict[str, Any]:
    import torch

    from .multi_face_reference import (
        assign_characters_to_faces,
        build_color_safe_reference,
        detected_faces,
        select_faces,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    preview = preview.convert("RGB")
    preview.save(output_dir / "pred_x0.png")
    detected = detected_faces(app, preview, minimum_confidence)
    faces = select_faces(detected, len(characters))

    runtime_characters = []
    for item in characters:
        copied = dict(item)
        copied["reference"] = Image.open(item["reference_image"]).convert("RGB")
        runtime_characters.append(copied)
    ordered, assignment = assign_characters_to_faces(app, runtime_characters, faces)

    pulid_like = SimpleNamespace(
        app=app,
        face_helper=SimpleNamespace(face_parse=parser),
    )
    targets = []
    target_failures = []
    for item, face in zip(ordered, faces):
        bbox = [float(value) for value in face.bbox]
        face_height = bbox[3] - bbox[1]
        if face_height < minimum_face_height:
            target_failures.append(
                {
                    "name": item["name"],
                    "bbox": bbox,
                    "det_score": float(face.det_score),
                    "reason": "face_below_minimum_height",
                    "face_height": face_height,
                    "minimum_face_height": minimum_face_height,
                }
            )
            continue
        character_dir = output_dir / "characters" / _slug(item["name"])
        try:
            face_3d = item.get("face_3d") or {}
            model_path = Path(face_3d.get("model_path", ""))
            if face_3d.get("facelift_status") == "failed" or not model_path.is_file():
                raise FileNotFoundError(f"Missing usable 3D face for {item['name']}")
            if not model_path.with_suffix(".pose_calibration.json").is_file():
                raise FileNotFoundError(f"Missing role-specific pose calibration for {item['name']}")
            record_path = character_dir / "facelift_result.json"
            _write_json(record_path, face_3d)
            target = build_color_safe_reference(
                item={**item, "facelift_result": str(record_path)},
                face=face,
                preview=preview,
                pulid=pulid_like,
                device=torch.device("cuda"),
                output_dir=character_dir,
            )
        except torch.OutOfMemoryError:
            raise
        except Exception as exc:
            target_failures.append(
                {
                    "name": item["name"],
                    "bbox": bbox,
                    "det_score": float(face.det_score),
                    "reason": "reference_preparation_failed",
                    "error": repr(exc),
                }
            )
            continue
        targets.append(
            {
                "name": item["name"],
                "display_name": item["display_name"],
                "reference_image": item["reference_image"],
                "bbox": target["bbox"],
                "pose": target["pose"],
                "harmonized_reference": str(
                    (character_dir / "harmonized_3d_face.png").resolve()
                ),
                "mask": str((character_dir / "final_injection_mask.png").resolve()),
            }
        )

    result = {
        "kind": "qwen_image21_product_v7_reference_preparation",
        "preview": str((output_dir / "pred_x0.png").resolve()),
        "preview_size": list(preview.size),
        "reliable_face_count": len(detected),
        "selected_face_count": len(faces),
        "extra_reliable_face_count": max(0, len(detected) - len(faces)),
        "assignment": assignment,
        "requested_characters": [item["name"] for item in characters],
        "unassigned_scheduled_characters": assignment[
            "unassigned_scheduled_characters"
        ],
        "target_failures": target_failures,
        "successful_target_count": len(targets),
        "targets": targets,
    }
    _write_json(output_dir / "preparation_attempt.json", result)
    if targets:
        _write_json(output_dir / "prepared.json", result)
    return result


def _prepare_target_records(
    pipe,
    prepared: dict[str, Any],
    state: dict[str, Any],
    *,
    width: int,
    height: int,
    seed: int,
    injection_strength: float,
    identity_step_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import torch

    assignment_cosines = {
        item["character"]: float(item["cosine"])
        for item in prepared["assignment"]["pairs"]
    }
    generator = torch.Generator(device="cuda").manual_seed(seed)
    records, failures = [], []
    for target in prepared["targets"]:
        try:
            reference_x0 = _encode_reference(
                pipe,
                Image.open(target["harmonized_reference"]),
                width,
                height,
                generator,
            ).to(device=state["device"], dtype=torch.bfloat16)
            mask = Image.open(target["mask"]).convert("L").resize(
                (state["latent_width"], state["latent_height"]),
                Image.Resampling.BILINEAR,
            )
            alpha = torch.from_numpy(np.asarray(mask, dtype=np.float32) / 255.0)
            alpha = alpha.reshape(
                1, state["latent_height"] * state["latent_width"], 1
            ).to(device=state["device"], dtype=torch.bfloat16)
            face_height = max(1.0, float(target["bbox"][3] - target["bbox"][1]))
            size_scale = min(1.0, max(0.3, face_height / 128.0))
            identity_cosine = assignment_cosines[target["name"]]
            active_steps = 6 if identity_cosine < identity_step_threshold else 1
            records.append(
                {
                    **target,
                    "reference_x0": reference_x0,
                    "alpha": alpha,
                    "size_scale": size_scale,
                    "effective_strength": injection_strength * size_scale,
                    "step12_identity_cosine": identity_cosine,
                    "identity_step_threshold": identity_step_threshold,
                    "active_injection_steps": active_steps,
                    "schedule_class": (
                        "low_identity_six_steps"
                        if active_steps == 6
                        else "high_identity_one_step"
                    ),
                }
            )
        except torch.OutOfMemoryError:
            raise
        except Exception as exc:
            failures.append(
                {
                    "name": target["name"],
                    "reason": "latent_reference_preparation_failed",
                    "error": repr(exc),
                }
            )
    return records, failures


def _fallback_reason(exc: Exception) -> str:
    message = str(exc).lower()
    if "detected 0 reliable faces" in message:
        return "no_reliable_face_detected"
    if "reliable faces" in message or ("expected" in message and "faces" in message):
        return "reliable_face_detection_failed"
    if "color-safe identity injection core is empty" in message:
        return "empty_v7_color_safe_core"
    if "render" in message or "pose" in message:
        return "gaussian_render_or_pose_check_failed"
    return "reference_preparation_failed"


def _execution_mode(offload_mode: str) -> str:
    return {
        "sequential": "sequential_cpu_offload",
        "model": "model_cpu_offload",
        "cuda": "full_cuda",
    }[offload_mode]


@torch.no_grad()
def generate_qwen_image21_first_frame(
    *,
    shot_id: str,
    prompt: str,
    scene_asset: dict[str, Any],
    character_assets: dict[str, Any],
    character_ids: list[str],
    project_dir: Path,
) -> dict[str, Any]:
    """Generate a single trajectory, preparing and injecting 3D faces mid-way."""

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen-Image-2.1 product generation requires CUDA")
    model = Path(os.getenv(
        "QWEN_IMAGE21_MODEL",
        str(PROJECT_ROOT / "models/diffusion/qwen-image-2.1"),
    )).resolve()
    if not model.is_dir():
        raise FileNotFoundError(f"Qwen-Image-2.1 model is missing: {model}")

    width = _env_int("QWEN_IMAGE21_WIDTH", 1024)
    height = _env_int("QWEN_IMAGE21_HEIGHT", 576)
    steps = _env_int("QWEN_IMAGE21_STEPS", 20)
    inject_start = _env_int("QWEN_IMAGE21_INJECT_START", 12)
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError("Qwen first-frame dimensions must be positive multiples of 32")
    if not 0 < inject_start < steps:
        raise ValueError("QWEN_IMAGE21_INJECT_START must be between 1 and steps - 1")
    seed = _env_int("QWEN_IMAGE21_SEED", 719005)
    strength = _env_float("QWEN_IMAGE21_INJECTION_STRENGTH", 0.4)
    threshold = _env_float("QWEN_IMAGE21_IDENTITY_STEP_THRESHOLD", 0.4)
    if not 0 <= strength <= 1 or not 0 <= threshold <= 1:
        raise ValueError("Qwen injection strength and identity threshold must be in [0, 1]")
    offload_mode = os.getenv("QWEN_IMAGE21_OFFLOAD_MODE", "sequential").strip().lower()
    references, characters = _build_reference_inputs(
        scene_asset, character_assets, character_ids
    )
    qwen_prompt = _build_qwen_prompt(prompt, references)
    images = [Image.open(item["path"]).convert("RGB") for item in references]
    project_dir = project_dir.resolve()
    frame_path = project_dir / "frames" / f"{shot_id}.png"
    artifact_dir = project_dir / "frames" / f"{shot_id}_qwen"
    prepared_dir = artifact_dir / "prepared"
    result_path = artifact_dir / "result.json"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    pipe = _load_pipeline(model, offload_mode)
    pipe._attention_kwargs = {}
    torch.cuda.reset_peak_memory_stats()
    targets, failures, step_log = [], [], []
    prepared = None
    reason = (
        "no_characters_requested" if not characters else
        "prompt_eye_closure_conflict" if prompt_requests_closed_eyes(prompt) else None
    )
    state = None
    try:
        state = _prepare_conditioning(
            pipe, prompt=qwen_prompt, images=images, width=width, height=height,
            output_resolution=_env_int("QWEN_IMAGE21_OUTPUT_RESOLUTION", 1024),
            steps=steps, seed=seed,
            use_kv_cache=_env_bool("QWEN_IMAGE21_USE_KV_CACHE", True),
        )
        current = state["initial"].clone()
        for index in range(steps):
            prediction = _predict(pipe, state, current, index)
            if index == inject_start and reason is None:
                sigma = state["sigmas"][index].to(current.device, current.dtype)
                preview = _decode(pipe, current - sigma * prediction, width, height)
                try:
                    app, parser = _load_face_runtime()
                    prepared = _prepare_color_safe_references(
                        preview=preview, characters=characters, output_dir=prepared_dir,
                        minimum_confidence=_env_float("QWEN_IMAGE21_MIN_FACE_CONFIDENCE", 0.5),
                        minimum_face_height=_env_float("QWEN_IMAGE21_MIN_FACE_HEIGHT", 24.0),
                        app=app, parser=parser,
                    )
                    failures.extend(prepared["target_failures"])
                    targets, latent_failures = _prepare_target_records(
                        pipe, prepared, state, width=width, height=height, seed=seed,
                        injection_strength=strength, identity_step_threshold=threshold,
                    )
                    failures.extend(latent_failures)
                    if not targets:
                        reason = "no_usable_3d_targets"
                except torch.OutOfMemoryError:
                    raise
                except Exception as exc:
                    reason = _fallback_reason(exc)
                    failures.append({"reason": reason, "error": repr(exc)})
                del preview
                gc.collect()
                torch.cuda.empty_cache()

            next_latent = _euler_step(current, prediction, state["sigmas"], index)
            active_targets = [
                target for target in targets
                if inject_start <= index < inject_start + target["active_injection_steps"]
            ]
            residual = torch.zeros_like(next_latent)
            if active_targets:
                sigma_next = state["sigmas"][index + 1].to(current.device, current.dtype)
                alpha_stack = torch.stack([
                    target["alpha"] * target["effective_strength"]
                    for target in active_targets
                ])
                alpha_sum = alpha_stack.sum(dim=0)
                normalization = torch.where(
                    alpha_sum > 1.0, alpha_sum.reciprocal(), torch.ones_like(alpha_sum)
                )
                for alpha, target in zip(alpha_stack * normalization.unsqueeze(0), active_targets):
                    reference = (
                        sigma_next * state["initial"]
                        + (1.0 - sigma_next) * target["reference_x0"]
                    )
                    residual += alpha * (reference - next_latent)
            current = next_latent + residual
            step_log.append({
                "step": index,
                "injected": bool(active_targets),
                "active_characters": [target["name"] for target in active_targets],
                "residual_norm": float(residual.float().norm().cpu()),
            })
        _decode(pipe, current, width, height).save(frame_path)
    finally:
        # Persistent MCP sessions reuse model weights but never a shot's tensors.
        state = None
        for target in targets:
            target.pop("alpha", None)
            target.pop("reference_x0", None)
        pipe.maybe_free_model_hooks()
        gc.collect()
        torch.cuda.empty_cache()

    applied = [target["name"] for target in targets]
    result = {
        "kind": "qwen_image21_product_single_trajectory_v7",
        "backend": "qwen_image21", "model": str(model), "shot_id": shot_id,
        "prompt": prompt, "qwen_prompt": qwen_prompt,
        "references": [
            {**{k: v for k, v in item.items() if k != "path"},
             "path": str(item["path"]), "sha256": _sha256(item["path"])}
            for item in references
        ],
        "seed": seed, "width": width, "height": height, "steps": steps,
        "inject_start": inject_start, "identity_step_threshold": threshold,
        "injection_strength": strength, "execution_mode": _execution_mode(offload_mode),
        "protocol": "single trajectory; mid-denoise pred_x0; per-role v7 residual",
        "status": "completed", "injection_applied": bool(targets), "skip_reason": reason,
        "frame_path": str(frame_path), "frame_sha256": _sha256(frame_path),
        "prepared": str(prepared_dir / "preparation_attempt.json") if prepared else None,
        "targets": targets, "requested_characters": character_ids,
        "applied_characters": applied,
        "skipped_characters": [name for name in character_ids if name not in applied],
        "unassigned_scheduled_characters": (
            prepared["unassigned_scheduled_characters"] if prepared else []
        ),
        "target_failures": failures, "step_log": step_log,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "peak_cuda_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
    }
    _write_json(result_path, result)
    frame_path.with_suffix(".prompt.txt").write_text(prompt, encoding="utf-8")
    return {"frame_path": str(frame_path), "denoise_log_path": str(result_path), "metadata": result}
