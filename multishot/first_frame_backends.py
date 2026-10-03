from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable


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
        from .qwen_image21_first_frame import generate_qwen_image21_first_frame

        return generate_qwen_image21_first_frame(
            shot_id=shot_id,
            prompt=first_frame_prompt,
            scene_asset=scene_asset,
            character_assets=character_assets,
            character_ids=character_ids,
            project_dir=Path(project_dir),
        )
    if normalized in {"pulid_flux", "ip_adapter"}:
        raise NotImplementedError(
            f"{backend} first-frame backend is not wired into the product MCP yet"
        )
    raise ValueError(f"Unsupported first-frame backend: {backend}")
