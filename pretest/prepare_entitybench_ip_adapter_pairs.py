"""Prepare a complete EntityBench episode with SDXL/IP-Adapter first frames.

Single-character shots use the original identity portrait as the global
IP-Adapter condition.  Treatment adds the v7 harmonized, color-safe local 3D
trajectory residual.  Reliability-gate failures and multi-character shots
reuse Control; the latter use text-only SDXL because one global IP-Adapter
portrait cannot represent multiple identities fairly.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(name: str) -> str:
    return "_".join(name.lower().replace("-", " ").split())


def _ordered_shots(episode: dict) -> list[dict]:
    shots = []
    shot_index = 0
    for scene in episode.get("scenes", []):
        prompts = scene.get("video_prompts", [])
        actions = scene.get("action_descriptions", [])
        cuts = scene.get("cut", [])
        for local_index, prompt in enumerate(prompts, start=1):
            shot_key = f"{scene['scene_num']}:{local_index}"
            shots.append({
                "shot_key": shot_key,
                "shot_index": shot_index,
                "prompt": prompt,
                "action_description": (
                    actions[local_index - 1] if local_index <= len(actions) else prompt
                ),
                "cut": cuts[local_index - 1] if local_index <= len(cuts) else None,
                "characters": episode.get("entity_schedule", {}).get(
                    shot_key, {}
                ).get("characters", []) or [],
            })
            shot_index += 1
    return shots


def _asset_paths(asset_root: Path, character: str) -> tuple[Path, Path]:
    slug = _slug(character)
    return (
        asset_root / "characters" / f"{slug}_reference.png",
        asset_root / "faces_3d" / slug / "facelift_result.json",
    )


def _sdxl_prompt(shot: dict) -> str:
    """Move the official action sentence ahead of entity definitions.

    SDXL's CLIP text encoder keeps only 77 tokens.  EntityBench stores entity
    definitions before the actual shot direction, which otherwise truncates
    the action entirely.  This preserves every original text span exactly once
    and changes only their order.
    """

    prompt = shot["prompt"].strip()
    action = shot["action_description"].strip()
    if not action or action not in prompt:
        return prompt
    prefix, suffix = prompt.rsplit(action, 1)
    remainder = "\n\n".join(item.strip() for item in (prefix, suffix) if item.strip())
    return f"{action}\n\n{remainder}" if remainder else action


def _job_pair(
    *,
    episode_id: str,
    shot: dict,
    control_image: Path,
    treatment_image: Path,
    output_dir: Path,
    seed: int,
    plugin_applied: bool,
    target_character: str | None,
) -> list[dict]:
    shot_name = f"shot_{shot['shot_key'].replace(':', '_')}"
    common = {
        "episode_id": episode_id,
        "shot_key": shot["shot_key"],
        "shot_index": shot["shot_index"],
        "characters": shot["characters"],
        "prompt": shot["prompt"],
        "seed": seed,
        "plugin_applied": plugin_applied,
    }
    if target_character is not None:
        common["target_character"] = target_character
    control_id = f"{shot_name}_control"
    video_dir = output_dir / "videos" / shot_name
    jobs = [
        {
            "job_id": control_id,
            **common,
            "condition": "control",
            "input_image": str(control_image.resolve()),
            "output_video": str((video_dir / "control.mp4").resolve()),
        },
        {
            "job_id": f"{shot_name}_treatment",
            **common,
            "condition": "treatment",
            "input_image": str(treatment_image.resolve()),
            "output_video": str((video_dir / "treatment.mp4").resolve()),
        },
    ]
    if not plugin_applied:
        jobs[1]["reuse_video_from"] = control_id
    return jobs


def _generate_text_only_controls(
    shots: list[dict], output_dir: Path, args
) -> dict[str, dict]:
    """Generate all multi-character Control frames with one SDXL model load."""

    if not shots:
        return {}
    import torch

    from multishot.diffusion_backend import OpenSourceDiffusionBackend

    os.environ["MULTISHOT_IMAGE_WIDTH"] = str(args.width)
    os.environ["MULTISHOT_IMAGE_HEIGHT"] = str(args.height)
    os.environ["MULTISHOT_DIFFUSION_STEPS"] = str(args.steps)
    os.environ["MULTISHOT_FINAL_STEP"] = str(args.steps)
    backend = OpenSourceDiffusionBackend("sdxl-base-1.0")
    records = {}
    try:
        for shot in shots:
            shot_name = f"shot_{shot['shot_key'].replace(':', '_')}"
            root = output_dir / "first_frames" / shot_name / "sdxl_control"
            control = root / "control" / "final.png"
            treatment = root / "treatment" / "final.png"
            config_path = root / "config.json"
            seed = args.base_seed + shot["shot_index"]
            started = time.perf_counter()
            if control.is_file() and config_path.is_file() and not args.overwrite:
                status = "skipped_existing"
            else:
                os.environ["MULTISHOT_DIFFUSION_SEED"] = str(seed)
                model_prompt = _sdxl_prompt(shot)
                runtime = backend.prepare_generation(
                    f"entitybench_{shot_name}_control", model_prompt, args.steps
                )
                os.environ["MULTISHOT_INJECTION_MODE"] = "off"
                state = backend.denoise_window(
                    runtime,
                    0,
                    args.steps,
                    previous_denoise_state=None,
                    injection_plan={"lambda": 0.0, "targets": []},
                    conditioning={"prompt": model_prompt},
                )
                backend.decode_final_image(state, str(control))
                del state, runtime
                gc.collect()
                torch.cuda.empty_cache()
                status = "generated"
            treatment.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(control, treatment)
            config = {
                "shot_key": shot["shot_key"],
                "characters": shot["characters"],
                "prompt": shot["prompt"],
                "model_prompt": _sdxl_prompt(shot),
                "prompt_adaptation": "official action_description moved before entity definitions for CLIP-77",
                "seed": seed,
                "width": args.width,
                "height": args.height,
                "steps": args.steps,
                "generation_model": "sdxl-base-1.0 text-only",
                "plugin_applied": False,
                "treatment_reuses_control": True,
                "reason": (
                    "multi-character shot: a single global IP-Adapter portrait cannot "
                    "represent all identities; no validated multi-target IP-Adapter path"
                ),
            }
            _write_json(config_path, config)
            records[shot["shot_key"]] = {
                **shot,
                **config,
                "status": status,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "control_image": str(control.resolve()),
                "treatment_image": str(treatment.resolve()),
                "control_sha256": _sha256(control),
                "treatment_sha256": _sha256(treatment),
            }
    finally:
        if getattr(backend, "_pipe", None) is not None:
            del backend._pipe
        del backend
        gc.collect()
        torch.cuda.empty_cache()
    return records


def run(args) -> dict:
    episode_path = Path(args.episode).resolve()
    output_dir = Path(args.output_dir).resolve()
    asset_root = Path(args.asset_root).resolve()
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    episode_id = episode_path.stem
    selected = set(args.shot) if args.shot else None
    shots = [
        item for item in _ordered_shots(episode)
        if selected is None or item["shot_key"] in selected
    ]
    single_shots = [item for item in shots if len(item["characters"]) == 1]
    fallback_shots = [item for item in shots if len(item["characters"]) != 1]
    report_path = output_dir / "ip_adapter_first_frame_report.json"
    full_manifest_path = output_dir / "wan_manifest_full_episode.json"
    single_manifest_path = output_dir / "wan_manifest_single_character.json"
    records: dict[str, dict] = {}
    identity_references = {}

    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    shared_backend = None
    shared_face_app = None
    for shot in single_shots:
        character = shot["characters"][0]
        reference, facelift_result = _asset_paths(asset_root, character)
        if not reference.is_file() or not facelift_result.is_file():
            raise FileNotFoundError(
                f"Missing assets for {character}: {reference}, {facelift_result}"
            )
        facelift = json.loads(facelift_result.read_text(encoding="utf-8"))
        gaussian_model = Path(facelift["model_path"]).resolve()
        if not gaussian_model.is_file():
            raise FileNotFoundError(gaussian_model)
        identity_references[character] = str(reference.resolve())
        shot_name = f"shot_{shot['shot_key'].replace(':', '_')}"
        pair_dir = output_dir / "first_frames" / shot_name / "ip_adapter"
        result_path = pair_dir / "result.json"
        control = pair_dir / "branches" / "ip_adapter_baseline.png"
        treatment = pair_dir / "branches" / "ip_adapter_plus_pulid_style_residual.png"
        seed = args.base_seed + shot["shot_index"]
        started = time.perf_counter()
        if (
            result_path.is_file() and control.is_file() and treatment.is_file()
            and not args.overwrite
        ):
            status = "skipped_existing"
        else:
            from argparse import Namespace
            from multishot.diffusion_backend import OpenSourceDiffusionBackend
            from multishot.ip_adapter_experiment_utils import _face_app
            from multishot.ip_adapter_pulid_style_injection_experiment import run as run_pair

            if shared_backend is None:
                os.environ.update({
                    "HF_HUB_OFFLINE": env["HF_HUB_OFFLINE"],
                    "TRANSFORMERS_OFFLINE": env["TRANSFORMERS_OFFLINE"],
                    "MULTISHOT_IMAGE_WIDTH": str(args.width),
                    "MULTISHOT_IMAGE_HEIGHT": str(args.height),
                    "MULTISHOT_IP_ADAPTER_SCALE": str(args.ip_adapter_scale),
                })
                shared_backend = OpenSourceDiffusionBackend(
                    "sdxl-base-1.0-ip-adapter"
                )
                shared_face_app = _face_app()
            pair_args = Namespace(
                reference=reference,
                continuous_render=(
                    PROJECT_ROOT / "experiment_output"
                    / "ip_adapter_small_yaw_harmonized_soft_context_v4_04"
                    / "input" / "rendered_3d_face.png"
                ),
                output=pair_dir,
                gaussian_model=gaussian_model,
                prompt=_sdxl_prompt(shot),
                seed=seed,
                width=args.width,
                height=args.height,
                steps=args.steps,
                fork_step=args.fork_step,
                ip_adapter_scale=args.ip_adapter_scale,
                injection_lambda=args.injection_strength,
                max_active_injection_steps=args.max_active_injection_steps,
                min_injection_face_height=args.min_injection_face_height,
                max_face_detection_retries=args.max_face_detection_retries,
                skip_unreliable_face=True,
                adaptive_small_face=True,
                reference_scale=1.0,
                min_abs_yaw=0.0,
                max_abs_yaw=90.0,
                harmonize_reference=True,
            )
            try:
                run_pair(pair_args, backend=shared_backend, app=shared_face_app)
            except Exception:
                status = "failed"
                if not args.continue_on_error:
                    raise
            else:
                status = "generated"
        if not result_path.is_file() or not control.is_file() or not treatment.is_file():
            status = "failed_missing_output"
            result = {}
        else:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        records[shot["shot_key"]] = {
            **shot,
            "target_character": character,
            "reference_image": str(reference.resolve()),
            "reference_sha256": _sha256(reference),
            "facelift_result": str(facelift_result.resolve()),
            "gaussian_model": str(gaussian_model),
            "seed": seed,
            "model_prompt": _sdxl_prompt(shot),
            "prompt_adaptation": "official action_description moved before entity definitions for CLIP-77",
            "status": status,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "plugin_skipped": bool(result.get("plugin_skipped", False)),
            "skip_reason": result.get("skip_reason"),
            "control_image": str(control.resolve()),
            "treatment_image": str(treatment.resolve()),
            "control_sha256": _sha256(control) if control.is_file() else None,
            "treatment_sha256": _sha256(treatment) if treatment.is_file() else None,
            "result_json": str(result_path.resolve()),
        }
        _write_json(report_path, {"shots": [records[key] for key in records]})
        if status.startswith("failed") and not args.continue_on_error:
            raise RuntimeError(f"{shot['shot_key']} failed; see {report_path}")

    if shared_backend is not None:
        import torch

        if getattr(shared_backend, "_pipe", None) is not None:
            del shared_backend._pipe
        del shared_backend, shared_face_app
        gc.collect()
        torch.cuda.empty_cache()
    records.update(_generate_text_only_controls(fallback_shots, output_dir, args))

    jobs = []
    single_jobs = []
    for shot in shots:
        record = records[shot["shot_key"]]
        is_single = len(shot["characters"]) == 1
        plugin_applied = is_single and not record.get("plugin_skipped", False)
        pair = _job_pair(
            episode_id=episode_id,
            shot=shot,
            control_image=Path(record["control_image"]),
            treatment_image=Path(record["treatment_image"]),
            output_dir=output_dir,
            seed=args.base_seed + shot["shot_index"],
            plugin_applied=plugin_applied,
            target_character=record.get("target_character"),
        )
        jobs.extend(pair)
        if is_single:
            single_jobs.extend(pair)

    report = {
        "kind": "entitybench_sdxl_ip_adapter_v7_first_frame_pairs",
        "episode_id": episode_id,
        "episode_path": str(episode_path),
        "asset_root": str(asset_root),
        "settings": {
            "model": "sdxl-base-1.0-ip-adapter",
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "fork_step": args.fork_step,
            "base_seed": args.base_seed,
            "ip_adapter_scale": args.ip_adapter_scale,
            "injection_strength": args.injection_strength,
            "max_active_injection_steps": args.max_active_injection_steps,
            "minimum_injection_face_height_px": args.min_injection_face_height,
            "maximum_face_detection_retries": args.max_face_detection_retries,
            "mask_policy": "v7 connected identity feature core intersected with eroded color application",
            "multi_character_policy": "text-only SDXL Control reused",
            "prompt_adaptation": "official action_description moved before entity definitions for CLIP-77",
        },
        "shots": [records[item["shot_key"]] for item in shots],
        "shot_count": len(shots),
        "single_character_shot_count": len(single_shots),
        "fallback_shot_count": len(fallback_shots),
        "injected_shots": sum(
            len(item["characters"]) == 1
            and not records[item["shot_key"]].get("plugin_skipped", False)
            for item in shots
        ),
        "control_reuse_shots": sum(
            len(item["characters"]) != 1
            or records[item["shot_key"]].get("plugin_skipped", False)
            for item in shots
        ),
        "wan_job_count": len(jobs),
    }
    _write_json(report_path, report)
    _write_json(full_manifest_path, {
        "kind": "entitybench_complete_ordered_episode_ip_adapter_v7_wan22_manifest",
        "episode_id": episode_id,
        "jobs": jobs,
    })
    _write_json(single_manifest_path, {
        "kind": "entitybench_single_character_ip_adapter_v7_wan22_manifest",
        "episode_id": episode_id,
        "identity_references": identity_references,
        "jobs": single_jobs,
    })
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--shot", action="append", default=[])
    parser.add_argument("--python", default=str(PROJECT_ROOT / ".venv/bin/python"))
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--fork-step", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=719000)
    parser.add_argument("--ip-adapter-scale", type=float, default=0.2)
    parser.add_argument("--injection-strength", type=float, default=0.4)
    parser.add_argument("--max-active-injection-steps", type=int, default=12)
    parser.add_argument("--min-injection-face-height", type=int, default=24)
    parser.add_argument("--max-face-detection-retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
