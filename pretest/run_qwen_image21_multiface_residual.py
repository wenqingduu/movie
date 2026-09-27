#!/usr/bin/env python3
"""Run paired Qwen-Image-2.1 Control and simultaneous multi-face latent residual."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import shutil
from pathlib import Path

import numpy as np
import torch
from diffusers import QwenImage21Pipeline
from PIL import Image, ImageDraw


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_comparison(control: Path, treatment: Path, output: Path) -> None:
    images = [Image.open(control).convert("RGB"), Image.open(treatment).convert("RGB")]
    labels = ["Qwen-Image-2.1 multi-reference Control", "Control + simultaneous v7 3D residual"]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    label_height = 36
    sheet = Image.new("RGB", (width * 2, height + label_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (image, label) in enumerate(zip(images, labels)):
        sheet.paste(image, (index * width, 0))
        draw.text((index * width + 8, height + 8), label, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def _encode_reference(pipe, image: Image.Image, width: int, height: int, generator) -> torch.Tensor:
    # Qwen-Image-2.1's VAE is four-channel. Match the pipeline's own
    # condition-image path, which converts PIL inputs to RGBA before
    # preprocessing and VAE encoding.
    value = pipe.image_processor.preprocess(image.convert("RGBA"), width=width, height=height)
    value = value.unsqueeze(2).to(device=pipe._execution_device, dtype=pipe.vae.dtype)
    encoded = pipe._encode_vae_image(value, generator)
    return pipe._pack_latents(
        encoded,
        encoded.shape[0],
        encoded.shape[1],
        encoded.shape[3],
        encoded.shape[4],
    )


def run(args: argparse.Namespace, *, pipe=None) -> dict:
    model = args.model.resolve()
    prepared_path = args.prepared.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    references = [path.resolve() for path in args.reference]

    started = time.perf_counter()
    if pipe is None:
        pipe = QwenImage21Pipeline.from_pretrained(
            str(model), dtype=torch.bfloat16, local_files_only=True
        )
        if args.offload_mode == "sequential":
            pipe.enable_sequential_cpu_offload()
            execution_mode = "sequential_cpu_offload"
        elif args.offload_mode == "model":
            pipe.enable_model_cpu_offload()
            execution_mode = "model_cpu_offload"
        else:
            pipe.to("cuda")
            execution_mode = "full_cuda"
    else:
        execution_mode = f"shared_pipeline_{args.offload_mode}"

    device = pipe._execution_device
    condition_images = [Image.open(path).convert("RGB") for path in references]
    latent_height = 2 * (args.height // (pipe.vae_scale_factor * 2))
    latent_width = 2 * (args.width // (pipe.vae_scale_factor * 2))
    latent_channels = pipe.transformer.config.in_channels
    noise_generator = torch.Generator(device="cuda").manual_seed(args.seed)
    initial = torch.randn(
        (1, 1, latent_channels, latent_height, latent_width),
        generator=noise_generator,
        device=device,
        dtype=torch.bfloat16,
    )
    initial = pipe._pack_latents(
        initial, 1, latent_channels, latent_height, latent_width
    )

    target_records = []
    encode_generator = torch.Generator(device="cuda").manual_seed(args.seed)
    for target in prepared["targets"]:
        reference_path = Path(target["harmonized_reference"])
        mask_path = Path(target["mask"])
        reference_x0 = _encode_reference(
            pipe,
            Image.open(reference_path),
            args.width,
            args.height,
            encode_generator,
        ).to(device=device, dtype=torch.bfloat16)
        mask = Image.open(mask_path).convert("L").resize(
            (latent_width, latent_height), Image.Resampling.BILINEAR
        )
        alpha = torch.from_numpy(np.asarray(mask, dtype=np.float32) / 255.0)
        alpha = alpha.reshape(1, latent_height * latent_width, 1).to(
            device=device, dtype=torch.bfloat16
        )
        bbox = target["bbox"]
        face_height = max(1.0, float(bbox[3] - bbox[1]))
        size_scale = min(1.0, max(0.3, face_height / 128.0))
        target_records.append({
            **target,
            "reference_x0": reference_x0,
            "alpha": alpha,
            "effective_strength": args.injection_strength * size_scale,
            "size_scale": size_scale,
        })

    common = {
        "prompt": args.prompt,
        "image": condition_images,
        "width": args.width,
        "height": args.height,
        "num_inference_steps": args.steps,
        "true_cfg_scale": 1.0,
        "use_kv_cache": True,
    }
    torch.cuda.reset_peak_memory_stats()
    control_path = output / "control.png"
    if getattr(args, "existing_control", None):
        source_control = Path(args.existing_control).resolve()
        if not source_control.is_file():
            raise FileNotFoundError(source_control)
        shutil.copy2(source_control, control_path)
    else:
        control_image = pipe(
            **common,
            latents=initial.clone(),
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        ).images[0]
        control_image.save(control_path)

    step_log = []
    end_step = min(args.steps, args.inject_start + args.max_active_injection_steps)

    def inject_callback(pipeline, step_index, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        active = args.inject_start <= step_index < end_step
        record = {
            "step": int(step_index),
            "timestep": float(timestep.detach().float().cpu()),
            "injected": active,
        }
        if active:
            sigma = pipeline.scheduler.sigmas[step_index + 1].to(
                device=latents.device, dtype=latents.dtype
            )
            alphas = []
            reference_states = []
            for target in target_records:
                alphas.append(target["alpha"] * target["effective_strength"])
                reference_states.append(
                    sigma * initial + (1.0 - sigma) * target["reference_x0"]
                )
            alpha_stack = torch.stack(alphas, dim=0)
            alpha_sum = alpha_stack.sum(dim=0)
            normalization = torch.where(
                alpha_sum > 1.0, alpha_sum.reciprocal(), torch.ones_like(alpha_sum)
            )
            normalized = alpha_stack * normalization.unsqueeze(0)
            combined = normalized.sum(dim=0)
            updated = latents * (1.0 - combined)
            for alpha, reference_state in zip(normalized, reference_states):
                updated = updated + alpha * reference_state
            record.update({
                "sigma_next": float(sigma.detach().float().cpu()),
                "raw_combined_alpha_max": float(alpha_sum.max().detach().float().cpu()),
                "residual_norm": float((updated - latents).float().norm().detach().cpu()),
                "active_characters": [target["name"] for target in target_records],
            })
            latents = updated
        step_log.append(record)
        return {"latents": latents}

    treatment_image = pipe(
        **common,
        latents=initial.clone(),
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
        callback_on_step_end=inject_callback,
        callback_on_step_end_tensor_inputs=["latents"],
    ).images[0]
    treatment_path = output / "treatment.png"
    treatment_image.save(treatment_path)
    comparison_path = output / "comparison.jpg"
    _make_comparison(control_path, treatment_path, comparison_path)

    serializable_targets = [
        {key: value for key, value in target.items() if key not in {"reference_x0", "alpha"}}
        for target in target_records
    ]
    metadata = {
        "kind": "qwen_image21_multi_reference_control_vs_simultaneous_v7_3d_residual",
        "model": str(model),
        "prepared": str(prepared_path),
        "prompt": args.prompt,
        "references": [str(path) for path in references],
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "seed": args.seed,
        "injection_start": args.inject_start,
        "end_step_exclusive": end_step,
        "injection_strength": args.injection_strength,
        "targets": serializable_targets,
        "execution_mode": execution_mode,
        "control": str(control_path),
        "treatment": str(treatment_path),
        "comparison": str(comparison_path),
        "control_sha256": _sha256(control_path),
        "treatment_sha256": _sha256(treatment_path),
        "peak_cuda_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "trajectory_policy": (
            "after scheduler step: z <- z + normalized_alpha * "
            "((sigma_next * fixed_noise + (1-sigma_next) * reference_x0) - z)"
        ),
        "step_log": step_log,
    }
    (output / "result.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--existing-control", type=Path)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=719005)
    parser.add_argument("--inject-start", type=int, default=24)
    parser.add_argument("--injection-strength", type=float, default=0.4)
    parser.add_argument("--max-active-injection-steps", type=int, default=10)
    parser.add_argument(
        "--offload-mode",
        choices=("sequential", "model", "none"),
        default="sequential",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
