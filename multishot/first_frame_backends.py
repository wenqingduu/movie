from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def generate_first_frame(
    *,
    shot_id: str,
    first_frame_prompt: str,
    backend: str,
    scene_asset: dict[str, Any],
    character_assets: dict[str, Any],
    character_ids: list[str],
    project_dir: str | Path,
    legacy_generator: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Generate one product first frame through the selected backend."""

    normalized = backend.strip().lower()
    if normalized in {"legacy", "juggernaut-xl-v9", "sdxl-base-1.0", "pseudo_diffusion_v1"}:
        generation_model = (
            os.getenv("MULTISHOT_LEGACY_FIRST_FRAME_MODEL", "juggernaut-xl-v9")
            if normalized == "legacy"
            else backend
        )
        return legacy_generator(
            shot_id=shot_id,
            first_frame_prompt=first_frame_prompt,
            generation_model=generation_model,
            scene_asset=scene_asset,
            character_assets=character_assets,
            character_ids=character_ids,
        )
    if normalized == "qwen_image21":
        return _generate_qwen_image21_first_frame(
            shot_id=shot_id,
            prompt=first_frame_prompt,
            character_assets=character_assets,
            character_ids=character_ids,
            project_dir=Path(project_dir),
        )
    if normalized in {"pulid_flux", "ip_adapter"}:
        raise NotImplementedError(
            f"{backend} first-frame backend is not wired into the product MCP yet"
        )
    raise ValueError(f"Unsupported first-frame backend: {backend}")


def _generate_qwen_image21_first_frame(
    *,
    shot_id: str,
    prompt: str,
    character_assets: dict[str, Any],
    character_ids: list[str],
    project_dir: Path,
) -> dict[str, Any]:
    """Qwen-Image-2.1 multi-reference Control path extracted from evaluation code."""

    import torch
    from diffusers import QwenImage21Pipeline

    model = Path(
        os.getenv(
            "QWEN_IMAGE21_MODEL",
            str(PROJECT_ROOT / "models/diffusion/qwen-image-2.1"),
        )
    ).resolve()
    if not model.is_dir():
        raise FileNotFoundError(f"Qwen-Image-2.1 model is missing: {model}")

    references = []
    for character_id in character_ids:
        asset = character_assets.get(character_id)
        if not asset or not asset.get("path"):
            raise FileNotFoundError(f"Missing character reference for {character_id}")
        path = Path(asset["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        references.append(path)
    if not references:
        raise ValueError("Qwen-Image-2.1 first-frame generation needs at least one reference image")

    output = project_dir / "frames" / f"{shot_id}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    pipe = QwenImage21Pipeline.from_pretrained(
        str(model),
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    offload_mode = os.getenv("QWEN_IMAGE21_OFFLOAD_MODE", "sequential")
    if offload_mode == "sequential":
        pipe.enable_sequential_cpu_offload()
        execution_mode = "sequential_cpu_offload"
    elif offload_mode == "model":
        pipe.enable_model_cpu_offload()
        execution_mode = "model_cpu_offload"
    else:
        pipe.to("cuda")
        execution_mode = "full_cuda"

    images = [Image.open(path).convert("RGB") for path in references]
    seed = _env_int("QWEN_IMAGE21_SEED", 719005)
    generator_device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    result = pipe(
        prompt=prompt,
        image=images,
        width=_env_int("QWEN_IMAGE21_WIDTH", 1024),
        height=_env_int("QWEN_IMAGE21_HEIGHT", 576),
        num_inference_steps=_env_int("QWEN_IMAGE21_STEPS", 40),
        generator=generator,
        true_cfg_scale=1.0,
        use_kv_cache=True,
    ).images[0]
    result.save(output)

    metadata = {
        "kind": "qwen_image21_product_first_frame",
        "backend": "qwen_image21",
        "model": str(model),
        "prompt": prompt,
        "references": [
            {"path": str(path), "sha256": _sha256(path)} for path in references
        ],
        "output": str(output),
        "output_sha256": _sha256(output),
        "seed": seed,
        "execution_mode": execution_mode,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    if torch.cuda.is_available():
        metadata["peak_cuda_memory_mib"] = round(torch.cuda.max_memory_allocated() / 1024**2, 2)

    metadata_path = output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_suffix(".prompt.txt").write_text(prompt, encoding="utf-8")
    return {
        "frame_path": str(output),
        "denoise_log_path": str(metadata_path),
        "metadata": metadata,
    }
