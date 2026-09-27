#!/usr/bin/env python3
"""Run resumable Qwen-Image-2.1 Control/Treatment stages for three episodes.

The stages are deliberately separate because Qwen and the FaceLift preparation
stack live in different virtual environments.  ``control`` and ``treatment``
reuse one Qwen pipeline per process; ``prepare`` reuses one InsightFace and one
BiSeNet instance for every shot.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/entitybench_qwen_image21_v7_full3"
CASES = (
    {
        "run": "run719",
        "episode": PROJECT_ROOT / "benchmarks/entitybench/data/scripts/00053051-5f7e-314f-85e0-517ec18f3b08__run719__i35_j44__T120.json",
        "assets": PROJECT_ROOT / "outputs/entitybench_wan22_smoke/episode_00053051/assets",
        "seed": 719000,
    },
    {
        "run": "run893",
        "episode": PROJECT_ROOT / "benchmarks/entitybench/data/scripts/000785f7-48d3-3a68-8ba7-82566cd6d77c__run893__i95_j118__T300.json",
        "assets": PROJECT_ROOT / "outputs/entitybench_pilot3/episode_run893/assets",
        "seed": 893000,
    },
    {
        "run": "run1517",
        "episode": PROJECT_ROOT / "benchmarks/entitybench/data/scripts/000cf326-8770-3c4e-a43a-6f9e3d57a3e0__run1517__i15_j40__T300.json",
        "assets": PROJECT_ROOT / "outputs/entitybench_pilot3/episode_run1517/assets",
        "seed": 1517000,
    },
)


def _slug(value: str) -> str:
    return "_".join(value.lower().replace("-", " ").split())


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _shot_prompts(episode: dict) -> dict[str, str]:
    result = {}
    for scene in episode["scenes"]:
        for index, prompt in enumerate(scene["video_prompts"], 1):
            result[f"{scene['scene_num']}:{index}"] = prompt
    return result


def _qwen_prompt(prompt: str, names: list[str]) -> str:
    identities = " ".join(
        f"Image {index} is the exact identity reference for {name}."
        for index, name in enumerate(names, 1)
    )
    count = len(names)
    roster = ", ".join(names)
    return (
        f"{identities} Generate a cinematic frame containing exactly {count} named "
        f"{'person' if count == 1 else 'people'}: {roster}. Preserve each referenced "
        "person's exact identity, age, facial structure, hairstyle and assigned role. "
        "Do not merge, swap, duplicate, or add identities. " + prompt
    )


def build_manifest(output_root: Path) -> dict:
    records = []
    for case in CASES:
        episode = json.loads(case["episode"].read_text(encoding="utf-8"))
        prompts = _shot_prompts(episode)
        for shot_index, shot_key in enumerate(episode["entity_schedule"], 1):
            names = episode["entity_schedule"][shot_key]["characters"]
            if not names:
                continue
            shot_dir = output_root / case["run"] / "shots" / f"shot_{shot_key.replace(':', '_')}"
            characters = []
            for name in names:
                slug = _slug(name)
                reference = case["assets"] / "characters" / f"{slug}_reference.png"
                facelift = case["assets"] / "faces_3d" / slug / "facelift_result.json"
                if not reference.is_file() or not facelift.is_file():
                    raise FileNotFoundError(f"missing assets for {case['run']} {shot_key} {name}")
                characters.append({
                    "name": name,
                    "reference_image": str(reference.resolve()),
                    "facelift_result": str(facelift.resolve()),
                })
            records.append({
                "run": case["run"],
                "episode": str(case["episode"].resolve()),
                "story_name": episode["story_name"],
                "shot_key": shot_key,
                "shot_index": shot_index,
                "character_count": len(names),
                "characters": characters,
                "official_prompt": prompts[shot_key],
                "qwen_prompt": _qwen_prompt(prompts[shot_key], names),
                "seed": case["seed"] + shot_index,
                "shot_dir": str(shot_dir.resolve()),
            })
    manifest = {
        "kind": "entitybench_qwen_image21_v7_full_episode_manifest",
        "shot_count": len(records),
        "single_character_shots": sum(r["character_count"] == 1 for r in records),
        "multi_character_shots": sum(r["character_count"] > 1 for r in records),
        "shots": records,
    }
    _write(output_root / "manifest.json", manifest)
    return manifest


def _copy_fallback(record: dict, reason: str) -> dict:
    shot_dir = Path(record["shot_dir"])
    control = shot_dir / "control.png"
    treatment = shot_dir / "treatment.png"
    shutil.copy2(control, treatment)
    status = {
        "status": "treatment_reuses_control",
        "injection_applied": False,
        "reason": reason,
        "control": str(control),
        "treatment": str(treatment),
    }
    _write(shot_dir / "result.json", status)
    return status


def run_control(args, manifest: dict) -> None:
    import torch
    from diffusers import QwenImage21Pipeline
    from pretest.run_qwen_image21_multiref import run

    pipe = QwenImage21Pipeline.from_pretrained(
        str(args.model), dtype=torch.bfloat16, local_files_only=True
    )
    pipe.enable_sequential_cpu_offload()
    for index, record in enumerate(manifest["shots"], 1):
        shot_dir = Path(record["shot_dir"])
        output = shot_dir / "control.png"
        if output.is_file() and not args.overwrite:
            print(f"[{index}/{len(manifest['shots'])}] control exists {record['run']} {record['shot_key']}", flush=True)
            continue
        ns = SimpleNamespace(
            model=args.model,
            reference=[Path(item["reference_image"]) for item in record["characters"]],
            prompt=record["qwen_prompt"], output=output, width=args.width,
            height=args.height, steps=args.steps, seed=record["seed"],
            offload_mode="sequential",
        )
        metadata = run(ns, pipe=pipe)
        print(f"[{index}/{len(manifest['shots'])}] control {record['run']} {record['shot_key']} {metadata['elapsed_seconds']}s", flush=True)


def run_prepare(args, manifest: dict) -> None:
    import torch
    from facexlib.parsing import init_parsing_model
    from multishot.ip_adapter_experiment_utils import _face_app
    from multishot.prompt_injection_safety import prompt_requests_closed_eyes
    from pretest.prepare_qwen_multiface_residual import run

    app = _face_app()
    parser = init_parsing_model(model_name="bisenet", device=torch.device("cuda"))
    for index, record in enumerate(manifest["shots"], 1):
        shot_dir = Path(record["shot_dir"])
        prepared_path = shot_dir / "prepared" / "prepared.json"
        status_path = shot_dir / "prepare_status.json"
        if prepared_path.is_file() and not args.overwrite:
            print(f"[{index}/{len(manifest['shots'])}] prepared exists {record['run']} {record['shot_key']}", flush=True)
            continue
        if prompt_requests_closed_eyes(record["official_prompt"]):
            status = {"status": "skipped", "reason": "prompt_eye_closure_conflict"}
            _write(status_path, status)
            print(f"[{index}/{len(manifest['shots'])}] prepare skipped eye gate {record['run']} {record['shot_key']}", flush=True)
            continue
        ns = SimpleNamespace(
            control=shot_dir / "control.png", output_dir=shot_dir / "prepared",
            character=record["characters"], min_face_confidence=args.min_face_confidence,
        )
        try:
            result = run(ns, app=app, parser=parser)
            status = {
                "status": "prepared", "target_count": len(result["targets"]),
                "reliable_face_count": result["reliable_face_count"],
                "extra_reliable_face_count": result["extra_reliable_face_count"],
                "assignment": result["assignment"],
            }
        except Exception as exc:
            status = {"status": "skipped", "reason": "reference_preparation_failed", "error": repr(exc)}
        _write(status_path, status)
        print(f"[{index}/{len(manifest['shots'])}] prepare {record['run']} {record['shot_key']} {status['status']} {status.get('reason','')}", flush=True)


def run_treatment(args, manifest: dict) -> None:
    import torch
    from diffusers import QwenImage21Pipeline
    from pretest.run_qwen_image21_multiface_residual import run

    eligible = [r for r in manifest["shots"] if (Path(r["shot_dir"]) / "prepared/prepared.json").is_file()]
    pipe = None
    if eligible:
        pipe = QwenImage21Pipeline.from_pretrained(
            str(args.model), dtype=torch.bfloat16, local_files_only=True
        )
        pipe.enable_sequential_cpu_offload()
    for index, record in enumerate(manifest["shots"], 1):
        shot_dir = Path(record["shot_dir"])
        treatment = shot_dir / "treatment.png"
        if treatment.is_file() and not args.overwrite:
            print(f"[{index}/{len(manifest['shots'])}] treatment exists {record['run']} {record['shot_key']}", flush=True)
            continue
        prepared = shot_dir / "prepared/prepared.json"
        if not prepared.is_file():
            status = json.loads((shot_dir / "prepare_status.json").read_text(encoding="utf-8"))
            _copy_fallback(record, status.get("reason", "not_prepared"))
            print(f"[{index}/{len(manifest['shots'])}] fallback {record['run']} {record['shot_key']}", flush=True)
            continue
        ns = SimpleNamespace(
            model=args.model, prepared=prepared,
            reference=[Path(item["reference_image"]) for item in record["characters"]],
            prompt=record["qwen_prompt"], output_dir=shot_dir / "pair",
            existing_control=shot_dir / "control.png", width=args.width,
            height=args.height, steps=args.steps, seed=record["seed"],
            inject_start=args.inject_start, injection_strength=args.injection_strength,
            max_active_injection_steps=args.injection_steps, offload_mode="sequential",
        )
        try:
            result = run(ns, pipe=pipe)
            shutil.copy2(result["treatment"], treatment)
            shutil.copy2(result["comparison"], shot_dir / "comparison.jpg")
            status = {
                "status": "generated", "injection_applied": True,
                "control": str((shot_dir / "control.png").resolve()),
                "treatment": str(treatment.resolve()), "details": result,
            }
            _write(shot_dir / "result.json", status)
        except Exception as exc:
            status = _copy_fallback(record, "treatment_generation_failed")
            status["error"] = repr(exc)
            _write(shot_dir / "result.json", status)
        print(f"[{index}/{len(manifest['shots'])}] treatment {record['run']} {record['shot_key']} {status['status']}", flush=True)


def build_wan_manifest(output_root: Path, manifest: dict) -> dict:
    jobs = []
    for record in manifest["shots"]:
        shot_dir = Path(record["shot_dir"])
        result = json.loads((shot_dir / "result.json").read_text(encoding="utf-8"))
        prefix = f"{record['run']}_{record['shot_key'].replace(':', '_')}"
        control_id = prefix + "_control"
        common = {
            "episode_id": record["run"], "shot_key": record["shot_key"],
            "shot_index": record["shot_index"], "prompt": record["official_prompt"],
            "seed": record["seed"],
        }
        jobs.append({**common, "job_id": control_id, "condition": "control",
                     "input_image": str((shot_dir / "control.png").resolve()),
                     "output_video": str((shot_dir / "videos/control.mp4").resolve())})
        treatment_job = {**common, "job_id": prefix + "_treatment", "condition": "treatment",
                         "input_image": str((shot_dir / "treatment.png").resolve()),
                         "output_video": str((shot_dir / "videos/treatment.mp4").resolve())}
        if not result.get("injection_applied"):
            treatment_job["reuse_video_from"] = control_id
        jobs.append(treatment_job)
    value = {"kind": "entitybench_qwen_image21_v7_full3_wan_manifest", "jobs": jobs}
    _write(output_root / "wan_manifest.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("manifest", "control", "prepare", "treatment", "wan-manifest"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models/diffusion/qwen-image-2.1")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--inject-start", type=int, default=12)
    parser.add_argument("--injection-steps", type=int, default=6)
    parser.add_argument("--injection-strength", type=float, default=0.4)
    parser.add_argument("--min-face-confidence", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    args.model = args.model.resolve()
    manifest = build_manifest(args.output_root)
    if args.stage == "control":
        run_control(args, manifest)
    elif args.stage == "prepare":
        run_prepare(args, manifest)
    elif args.stage == "treatment":
        run_treatment(args, manifest)
    elif args.stage == "wan-manifest":
        build_wan_manifest(args.output_root, manifest)
    print(json.dumps({"stage": args.stage, "shot_count": manifest["shot_count"], "output_root": str(args.output_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
