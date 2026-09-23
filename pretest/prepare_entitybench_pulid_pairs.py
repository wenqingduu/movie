"""Prepare resumable PuLID-FLUX Control/Treatment first-frame pairs.

This pilot helper reads one EntityBench episode, keeps official shot order, and
runs only shots whose entity schedule contains exactly one character. Character
reference images and FaceLift results follow the episode asset layout documented
in LLM_HANDOFF.md. Failed/no-face shots are recorded and do not enter the Wan
manifest as if injection had succeeded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    global_index = 0
    for scene in episode.get("scenes", []):
        prompts = scene.get("video_prompts", [])
        actions = scene.get("action_descriptions", [])
        cuts = scene.get("cut", [])
        for local_index, prompt in enumerate(prompts, start=1):
            shot_key = f"{scene['scene_num']}:{local_index}"
            shots.append({
                "shot_key": shot_key,
                "shot_index": global_index,
                "prompt": prompt,
                "action_description": (
                    actions[local_index - 1] if local_index <= len(actions) else prompt
                ),
                "cut": cuts[local_index - 1] if local_index <= len(cuts) else None,
                "characters": episode.get("entity_schedule", {}).get(shot_key, {}).get(
                    "characters", []
                ) or [],
            })
            global_index += 1
    return shots


def _asset_paths(asset_root: Path, character: str) -> tuple[Path, Path]:
    slug = _slug(character)
    return (
        asset_root / "characters" / f"{slug}_reference.png",
        asset_root / "faces_3d" / slug / "facelift_result.json",
    )


def _wan_jobs(
    episode_id: str,
    shot: dict,
    character: str,
    output_dir: Path,
    seed: int,
    reuse_treatment: bool,
):
    shot_name = f"shot_{shot['shot_key'].replace(':', '_')}"
    pair_dir = output_dir / "first_frames" / shot_name / "pulid_flux"
    video_dir = output_dir / "videos" / shot_name
    common = {
        "episode_id": episode_id,
        "shot_key": shot["shot_key"],
        "shot_index": shot["shot_index"],
        "target_character": character,
        "prompt": shot["prompt"],
        "seed": seed,
    }
    control_job_id = f"{shot_name}_control"
    jobs = [
        {
            "job_id": control_job_id,
            **common,
            "condition": "control",
            "input_image": str((pair_dir / "control" / "final.png").resolve()),
            "output_video": str((video_dir / "control.mp4").resolve()),
        },
        {
            "job_id": f"{shot_name}_treatment",
            **common,
            "condition": "treatment",
            "input_image": str((pair_dir / "treatment" / "final.png").resolve()),
            "output_video": str((video_dir / "treatment.mp4").resolve()),
        },
    ]
    if reuse_treatment:
        jobs[1]["reuse_video_from"] = control_job_id
    return jobs


def run(args) -> dict:
    episode_path = Path(args.episode).resolve()
    output_dir = Path(args.output_dir).resolve()
    asset_root = Path(args.asset_root).resolve()
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    episode_id = episode_path.stem
    selected_keys = set(args.shot) if args.shot else None
    report_path = output_dir / "pulid_first_frame_report.json"
    wan_manifest_path = output_dir / "wan_manifest_single_character.json"
    report = {
        "kind": "entitybench_pulid_flux_first_frame_pairs",
        "episode_id": episode_id,
        "episode_path": str(episode_path),
        "asset_root": str(asset_root),
        "settings": {
            "width": args.width,
            "height": args.height,
            "base_seed": args.base_seed,
            "guidance": args.guidance,
            "pulid_id_weight": args.pulid_id_weight,
            "injection_strength": args.injection_strength,
            "max_active_injection_steps": args.max_active_injection_steps,
            "minimum_injection_face_height_px": args.min_injection_face_height,
            "maximum_face_detection_retries": args.max_face_detection_retries,
            "harmonize_reference": True,
            "reference_conditioning": "target",
        },
        "shots": [],
    }
    identity_references = {}
    wan_jobs = []
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    # Loading FLUX, the text encoders and PuLID once per shot dominates runtime.
    # In batch mode, keep the model objects alive and let the experiment move
    # individual components between CPU/GPU exactly as it already does. The
    # subprocess path remains available as a conservative fallback.
    shared_single = None
    original_loader = None
    shared_models = None
    if args.shared_model_process:
        from multishot import pulid_flux_inner_face_experiment as shared_single

        original_loader = shared_single._load_models

        def _cached_loader(pair_args, device):
            nonlocal shared_models
            if shared_models is None:
                shared_models = original_loader(pair_args, device)
            return shared_models

        shared_single._load_models = _cached_loader

    for shot in _ordered_shots(episode):
        if selected_keys is not None and shot["shot_key"] not in selected_keys:
            continue
        if len(shot["characters"]) != 1:
            continue
        character = shot["characters"][0]
        reference, facelift_result = _asset_paths(asset_root, character)
        if not reference.is_file() or not facelift_result.is_file():
            raise FileNotFoundError(
                f"Missing assets for {character}: {reference}, {facelift_result}"
            )
        identity_references[character] = str(reference)
        shot_name = f"shot_{shot['shot_key'].replace(':', '_')}"
        pair_dir = output_dir / "first_frames" / shot_name / "pulid_flux"
        control = pair_dir / "control" / "final.png"
        treatment = pair_dir / "treatment" / "final.png"
        config_path = pair_dir / "config.json"
        metrics_path = pair_dir / "metrics.json"
        seed = args.base_seed + shot["shot_index"]
        record = {
            **shot,
            "target_character": character,
            "reference_image": str(reference),
            "reference_sha256": _sha256(reference),
            "facelift_result": str(facelift_result),
            "seed": seed,
            "output_dir": str(pair_dir),
        }
        started = time.perf_counter()
        if (
            control.is_file()
            and treatment.is_file()
            and config_path.is_file()
            and metrics_path.is_file()
            and not args.overwrite
        ):
            record["status"] = "skipped_existing"
        else:
            command = [
                args.python,
                "-m", "multishot.pulid_flux_inner_face_experiment",
                "--reference-image", str(reference),
                "--reference-origin",
                "EntityBench character description; locally generated SDXL identity asset",
                "--reference-generated",
                "--output-dir", str(pair_dir),
                "--prompt", shot["prompt"],
                "--width", str(args.width),
                "--height", str(args.height),
                "--seed", str(seed),
                "--guidance", str(args.guidance),
                "--pulid-id-weight", str(args.pulid_id_weight),
                "--injection-strength", str(args.injection_strength),
                "--facelift-result", str(facelift_result),
                "--harmonize-reference",
                "--reference-conditioning", "target",
                "--onnx-provider", args.onnx_provider,
                "--min-injection-face-height", str(args.min_injection_face_height),
                "--skip-unreliable-face",
                "--max-face-detection-retries", str(args.max_face_detection_retries),
            ]
            if args.max_active_injection_steps is not None:
                command.extend(
                    ["--max-active-injection-steps", str(args.max_active_injection_steps)]
                )
            if args.shared_model_process:
                old_argv = sys.argv
                try:
                    sys.argv = ["pulid_flux_inner_face_experiment", *command[3:]]
                    pair_args = shared_single.parse_args()
                    shared_single.run(pair_args)
                except Exception as exc:
                    record["return_code"] = 1
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    record["status"] = "failed"
                    if not args.continue_on_error:
                        raise
                else:
                    record["return_code"] = 0
                    record["status"] = "generated"
                finally:
                    sys.argv = old_argv
                    import gc
                    import torch

                    gc.collect()
                    torch.cuda.empty_cache()
            else:
                completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
                record["return_code"] = completed.returncode
                record["status"] = "generated" if completed.returncode == 0 else "failed"
        record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        if record["status"] in {"generated", "skipped_existing"}:
            if not control.is_file() or not treatment.is_file():
                record["status"] = "failed_missing_output"
            else:
                record["control_sha256"] = _sha256(control)
                record["treatment_sha256"] = _sha256(treatment)
                pair_config = (
                    json.loads(config_path.read_text(encoding="utf-8"))
                    if config_path.is_file() else {}
                )
                record["plugin_skipped"] = bool(pair_config.get("plugin_skipped", False))
                record["skip_reason"] = pair_config.get("skip_reason")
                wan_jobs.extend(
                    _wan_jobs(
                        episode_id,
                        shot,
                        character,
                        output_dir,
                        seed,
                        reuse_treatment=record["plugin_skipped"],
                    )
                )
        report["shots"].append(record)
        _write_json(report_path, report)
        _write_json(wan_manifest_path, {
            "identity_references": identity_references,
            "jobs": wan_jobs,
        })
        if record["status"].startswith("failed") and not args.continue_on_error:
            raise RuntimeError(f"{shot['shot_key']} failed; see {report_path}")

    report["successful_shots"] = sum(
        item["status"] in {"generated", "skipped_existing"} for item in report["shots"]
    )
    report["failed_shots"] = sum(item["status"].startswith("failed") for item in report["shots"])
    report["injected_shots"] = sum(
        item["status"] in {"generated", "skipped_existing"}
        and not item.get("plugin_skipped", False)
        for item in report["shots"]
    )
    report["control_reuse_shots"] = sum(
        item["status"] in {"generated", "skipped_existing"}
        and item.get("plugin_skipped", False)
        for item in report["shots"]
    )
    report["wan_job_count"] = len(wan_jobs)
    _write_json(report_path, report)
    _write_json(wan_manifest_path, {
        "identity_references": identity_references,
        "jobs": wan_jobs,
    })
    if shared_single is not None and original_loader is not None:
        shared_single._load_models = original_loader
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--shot", action="append", default=[])
    parser.add_argument("--python", default=str(PROJECT_ROOT / ".venv/bin/python"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--base-seed", type=int, default=719000)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--pulid-id-weight", type=float, default=1.0)
    parser.add_argument("--injection-strength", type=float, default=0.4)
    parser.add_argument("--max-active-injection-steps", type=int, default=12)
    parser.add_argument("--onnx-provider", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--min-injection-face-height", type=int, default=24)
    parser.add_argument("--max-face-detection-retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--shared-model-process",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse one FLUX/PuLID model load across all selected shots.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
