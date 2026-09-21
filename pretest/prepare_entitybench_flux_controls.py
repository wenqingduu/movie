"""Generate text-only FLUX Control frames for selected multi-character fallbacks.

These are shots for which the current evaluation does not have complete
per-character identity/Gaussian assets or a reliable assignment.  Each Control
follows the same PuLID-FLUX execution path as the multi-face prototype,
including a valid identity embedding, but fixes its global identity weight to
zero and applies no 3D residual.  Treatment reuses Control so a complete
ordered episode does not pretend that the plugin was applied.  This fallback
does not imply that the multi-face injection algorithm itself is unsupported;
shot 4:2 is evaluated through ``pulid_flux_multi_face_experiment.py``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from PIL import Image
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from multishot import pulid_flux_inner_face_experiment as single  # noqa: E402


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_shots(episode: dict) -> list[dict]:
    shots = []
    global_index = 0
    for scene in episode.get("scenes", []):
        prompts = scene.get("video_prompts", [])
        for local_index, prompt in enumerate(prompts, start=1):
            shot_key = f"{scene['scene_num']}:{local_index}"
            shots.append({
                "shot_key": shot_key,
                "shot_index": global_index,
                "prompt": prompt,
                "characters": episode.get("entity_schedule", {}).get(
                    shot_key, {}
                ).get("characters", []) or [],
            })
            global_index += 1
    return shots


def run(args) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    episode_path = Path(args.episode).resolve()
    output_dir = Path(args.output_dir).resolve()
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    selected = set(args.shot) if args.shot else None
    shots = [
        shot for shot in _ordered_shots(episode)
        if len(shot["characters"]) != 1
        and (selected is None or shot["shot_key"] in selected)
    ]
    if not shots:
        raise ValueError("No non-single-character shots selected")
    neutral_reference_path = Path(args.neutral_reference).resolve()
    if not neutral_reference_path.is_file():
        raise FileNotFoundError(neutral_reference_path)
    args.build_facelift = False
    args.facelift_result = None
    single._preflight_model_files(args)  # noqa: SLF001

    report_path = output_dir / "flux_control_first_frame_report.json"
    manifest_path = output_dir / "wan_manifest_multi_character_control_reuse.json"
    report = {
        "kind": "entitybench_text_only_flux_control_frames",
        "episode_id": episode_path.stem,
        "episode_path": str(episode_path),
        "settings": {
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "guidance": args.guidance,
            "base_seed": args.base_seed,
            "pulid_id_weight": 0.0,
            "neutral_reference": str(neutral_reference_path),
            "neutral_reference_sha256": _sha256(neutral_reference_path),
            "neutral_reference_effective_weight": 0.0,
            "plugin_applied": False,
            "treatment_policy": "reuse_control_for_selected_multi_character_fallback",
        },
        "shots": [],
    }
    jobs = []
    device = torch.device("cuda")
    old_cwd = Path.cwd()
    os.chdir(single.PULID_ROOT)
    try:
        model, ae, t5, clip, pulid = single._load_models(args, device)  # noqa: SLF001
        prepared = []
        for shot in shots:
            seed = args.base_seed + shot["shot_index"]
            noise = single.get_noise(
                1, args.height, args.width, device=device, dtype=torch.bfloat16, seed=seed
            )
            conditioning = single.prepare(
                t5=t5, clip=clip, img=noise, prompt=shot["prompt"]
            )
            prepared.append((shot, seed, conditioning))
        t5.cpu()
        clip.cpu()
        del t5, clip
        gc.collect()
        torch.cuda.empty_cache()

        # Match the multi-face experiment's initialization order exactly:
        # prompt conditioning first, then the zero-weight identity carrier.
        pulid.components_to_device(device)
        pulid.device = device
        pulid.face_helper.device = device
        pulid.face_helper.face_det.device = device
        pulid.face_helper.face_det.mean_tensor = (
            pulid.face_helper.face_det.mean_tensor.to(device)
        )
        neutral_reference = Image.open(neutral_reference_path).convert("RGB")
        resized_reference = single.resize_numpy_image_long(
            np.asarray(neutral_reference), 1024
        )
        neutral_id, _ = pulid.get_id_embedding(
            resized_reference, cal_uncond=False
        )
        cpu = torch.device("cpu")
        pulid.components_to_device(cpu)
        pulid.device = cpu
        pulid.face_helper.device = cpu
        pulid.face_helper.face_det.device = cpu
        pulid.face_helper.face_det.mean_tensor = (
            pulid.face_helper.face_det.mean_tensor.to(cpu)
        )
        model.to(device)
        ae.to(device)

        for shot, seed, conditioning in prepared:
            shot_name = f"shot_{shot['shot_key'].replace(':', '_')}"
            frame_root = output_dir / "first_frames" / shot_name / "flux_control"
            control_path = frame_root / "control/final.png"
            treatment_path = frame_root / "treatment/final.png"
            prompt_path = frame_root / "input/prompt.txt"
            config_path = frame_root / "config.json"
            started = time.perf_counter()
            if control_path.is_file() and config_path.is_file() and not args.overwrite:
                status = "skipped_existing"
            else:
                state = conditioning["img"]
                timesteps = single.get_schedule(
                    args.steps, state.shape[-1] * state.shape[-2] // 4, shift=True
                )
                with torch.inference_mode():
                    for step in range(args.steps):
                        t, t_next = timesteps[step], timesteps[step + 1]
                        velocity = single._velocity(  # noqa: SLF001
                            model,
                            state,
                            conditioning,
                            t,
                            args.guidance,
                            neutral_id,
                            0.0,
                        )
                        state = state + (t_next - t) * velocity
                image = single._decode(  # noqa: SLF001
                    ae, state, args.height, args.width, device
                )
                control_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(control_path)
                status = "generated"
            treatment_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(control_path, treatment_path)
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(shot["prompt"], encoding="utf-8")
            config = {
                "shot_key": shot["shot_key"],
                "characters": shot["characters"],
                "prompt": shot["prompt"],
                "seed": seed,
                "width": args.width,
                "height": args.height,
                "steps": args.steps,
                "guidance": args.guidance,
                "generation_model": "FLUX.1-dev local FP8",
                "pulid_id_weight": 0.0,
                "neutral_reference": str(neutral_reference_path),
                "neutral_reference_effective_weight": 0.0,
                "plugin_applied": False,
                "treatment_reuses_control": True,
                "reason": (
                    "selected fallback: complete per-character assets or reliable "
                    "role assignment unavailable in this evaluation run"
                ),
            }
            _write_json(config_path, config)
            record = {
                **config,
                "status": status,
                "control_image": str(control_path.resolve()),
                "treatment_image": str(treatment_path.resolve()),
                "control_sha256": _sha256(control_path),
                "treatment_sha256": _sha256(treatment_path),
                "elapsed_seconds": time.perf_counter() - started,
            }
            report["shots"].append(record)

            common = {
                "episode_id": episode_path.stem,
                "shot_key": shot["shot_key"],
                "shot_index": shot["shot_index"],
                "characters": shot["characters"],
                "prompt": shot["prompt"],
                "seed": seed,
            }
            control_job_id = f"{shot_name}_control"
            video_root = output_dir / "videos" / shot_name
            jobs.extend([
                {
                    "job_id": control_job_id,
                    **common,
                    "condition": "control",
                    "plugin_applied": False,
                    "input_image": str(control_path.resolve()),
                    "output_video": str((video_root / "control.mp4").resolve()),
                },
                {
                    "job_id": f"{shot_name}_treatment",
                    **common,
                    "condition": "treatment",
                    "plugin_applied": False,
                    "input_image": str(treatment_path.resolve()),
                    "output_video": str((video_root / "treatment.mp4").resolve()),
                    "reuse_video_from": control_job_id,
                },
            ])
            _write_json(report_path, report)
            _write_json(manifest_path, {"jobs": jobs})
    finally:
        os.chdir(old_cwd)

    report["completed_shots"] = len(report["shots"])
    report["wan_job_count"] = len(jobs)
    _write_json(report_path, report)
    _write_json(manifest_path, {"jobs": jobs})
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shot", action="append", default=[])
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--base-seed", type=int, default=719000)
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--onnx-provider", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--neutral-reference", required=True)
    parser.add_argument("--fp8", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
