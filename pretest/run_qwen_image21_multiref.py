#!/usr/bin/env python3
"""Generate one deterministic Qwen-Image-2.1 multi-reference Control frame."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from diffusers import QwenImage21Pipeline
from PIL import Image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace, *, pipe=None) -> dict:
    model = args.model.resolve()
    references = [path.resolve() for path in args.reference]
    output = args.output.resolve()
    if not model.is_dir():
        raise FileNotFoundError(model)
    for path in references:
        if not path.is_file():
            raise FileNotFoundError(path)

    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if pipe is None:
        pipe = QwenImage21Pipeline.from_pretrained(
            str(model),
            dtype=torch.bfloat16,
            local_files_only=True,
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

    images = [Image.open(path).convert("RGB") for path in references]
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    result = pipe(
        prompt=args.prompt,
        image=images,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        generator=generator,
        true_cfg_scale=1.0,
        use_kv_cache=True,
    ).images[0]
    result.save(output)

    metadata = {
        "kind": "qwen_image21_multi_reference_control",
        "model": str(model),
        "prompt": args.prompt,
        "references": [
            {"path": str(path), "sha256": _sha256(path)} for path in references
        ],
        "output": str(output),
        "output_sha256": _sha256(output),
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "seed": args.seed,
        "execution_mode": execution_mode,
        "peak_cuda_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    metadata_path = output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=719005)
    parser.add_argument(
        "--offload-mode",
        choices=("sequential", "model", "none"),
        default="sequential",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
