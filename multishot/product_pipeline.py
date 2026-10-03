"""Product-oriented entrypoint for story-to-video projects.

This module is intentionally separate from ``pretest``.  The pretest package is
for benchmark/evaluation workflows; this file provides a stable surface that a
CLI, FastAPI service, or worker can call for user-facing projects.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_ROOT = Path("outputs/projects")


def _read_story(args: argparse.Namespace) -> str:
    if args.story_file:
        return Path(args.story_file).read_text(encoding="utf-8").strip()
    if args.story:
        return args.story.strip()
    raise ValueError("Provide --story or --story-file.")


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _shot_prompt(shot: dict[str, Any]) -> str:
    parts = [
        shot.get("shot_content", ""),
        shot.get("action_prompt", ""),
        shot.get("camera_prompt", ""),
        shot.get("first_frame_prompt", ""),
    ]
    return ". ".join(part.strip().rstrip(".") for part in parts if part and part.strip())


def build_wan_manifest(
    project_dir: str | Path,
    *,
    base_seed: int = 1000,
    output_name: str = "wan_manifest_product.json",
) -> Path:
    """Build a Wan-compatible manifest from a generated project plan.

    The manifest follows the same simple job structure as the current Wan
    runner, but this function does not depend on ``pretest``.
    """

    project_path = Path(project_dir)
    plan_path = project_path / "project_plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"Missing project plan: {plan_path}")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    jobs = []
    for index, shot in enumerate(plan.get("shots", [])):
        shot_id = shot["shot_id"]
        first_frame = shot.get("first_frame_path")
        if not first_frame:
            raise ValueError(f"{shot_id} is missing first_frame_path")

        output_video = project_path / "shot_videos" / f"{shot_id}.mp4"
        jobs.append(
            {
                "job_id": f"{shot_id}_video",
                "shot_id": shot_id,
                "subscript_id": shot.get("subscript_id"),
                "character_ids": shot.get("character_ids", []),
                "input_image": str(Path(first_frame).resolve()),
                "output_video": str(output_video.resolve()),
                "prompt": _shot_prompt(shot),
                "seed": base_seed + index,
            }
        )

    manifest = {
        "kind": "product_story_to_video_wan_manifest",
        "project_dir": str(project_path.resolve()),
        "project_title": plan.get("project_title"),
        "jobs": jobs,
    }
    return _write_json(project_path / output_name, manifest)


def assemble_videos(
    project_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
    output_path: str | Path | None = None,
    reencode: bool = False,
) -> Path:
    """Assemble shot videos into one final mp4 with ffmpeg."""

    project_path = Path(project_dir)
    manifest_file = Path(manifest_path) if manifest_path else project_path / "wan_manifest_product.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    final_path = Path(output_path) if output_path else project_path / "final" / "final_video.mp4"
    final_path.parent.mkdir(parents=True, exist_ok=True)

    concat_path = project_path / "final" / "concat_list.txt"
    lines = []
    for job in manifest.get("jobs", []):
        video_path = Path(job["output_video"])
        if not video_path.is_file():
            raise FileNotFoundError(f"Missing shot video for assembly: {video_path}")
        escaped = str(video_path.resolve()).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if reencode:
        command = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "24",
            "-c:a",
            "aac",
            str(final_path),
        ]
    else:
        command = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-c",
            "copy",
            str(final_path),
        ]
    subprocess.run(command, check=True)
    return final_path


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def generate_shot_videos(
    project_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
    report_path: str | Path | None = None,
    wan_root: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    size: str | None = None,
    frame_num: int | None = None,
    sample_steps: int | None = None,
    sample_shift: float | None = None,
    guide_scale: float | None = None,
    base_seed: int | None = None,
    limit_jobs: int | None = None,
    overwrite: bool | None = None,
    continue_on_error: bool | None = None,
) -> dict[str, Any]:
    """Generate shot videos for a project using the existing Wan manifest runner."""

    project_path = Path(project_dir).resolve()
    manifest_file = Path(manifest_path).resolve() if manifest_path else project_path / "wan_manifest_product.json"
    report_file = Path(report_path).resolve() if report_path else project_path / "wan_video_report.json"

    if not manifest_file.is_file():
        raise FileNotFoundError(f"Missing Wan manifest: {manifest_file}")

    args = argparse.Namespace(
        manifest=str(manifest_file),
        report=str(report_file),
        wan_root=str(Path(wan_root or os.getenv("WAN_ROOT", "/root/autodl-tmp/Wan2.2")).resolve()),
        wan_commit=os.getenv("WAN_COMMIT", "42bf4cfaa384bc21833865abc2f9e6c0e67233dc"),
        checkpoint_dir=str(Path(
            checkpoint_dir
            or os.getenv(
                "WAN_CHECKPOINT_DIR",
                str(Path(__file__).resolve().parents[1] / "models/video/Wan2.2-TI2V-5B"),
            )
        ).resolve()),
        size=size or os.getenv("WAN_SIZE", "1280*704"),
        frame_num=int(frame_num if frame_num is not None else os.getenv("WAN_FRAME_NUM", "49")),
        sample_steps=int(
            sample_steps if sample_steps is not None else os.getenv("WAN_SAMPLE_STEPS", "50")
        ),
        sample_shift=float(
            sample_shift if sample_shift is not None else os.getenv("WAN_SAMPLE_SHIFT", "5.0")
        ),
        guide_scale=float(
            guide_scale if guide_scale is not None else os.getenv("WAN_GUIDE_SCALE", "5.0")
        ),
        base_seed=int(base_seed if base_seed is not None else os.getenv("WAN_BASE_SEED", "719000")),
        limit_jobs=limit_jobs
        if limit_jobs is not None
        else (int(os.getenv("WAN_LIMIT_JOBS")) if os.getenv("WAN_LIMIT_JOBS") else None),
        overwrite=_env_bool("WAN_OVERWRITE") if overwrite is None else overwrite,
        continue_on_error=_env_bool("WAN_CONTINUE_ON_ERROR") if continue_on_error is None else continue_on_error,
    )
    # Wan has its own CUDA and model dependencies. Run the existing CLI in its
    # environment and leave the Celery process free of video model weights.
    python = Path(os.getenv("WAN_PYTHON", "/root/autodl-tmp/wan22-venv/bin/python"))
    if not python.is_file():
        raise FileNotFoundError(f"Wan Python environment is missing: {python}")
    command = [str(python), "-m", "pretest.run_wan22_i2v_manifest"]
    for name, value in vars(args).items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
        elif value is not None:
            command.extend([flag, str(value)])
    subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)
    if not report_file.is_file():
        raise FileNotFoundError(f"Wan did not write its report: {report_file}")
    return json.loads(report_file.read_text(encoding="utf-8"))


def run_story_project(
    story: str,
    project_dir: str | Path,
    *,
    backend: str = "qwen_image21",
    base_seed: int = 1000,
    write_manifest: bool = True,
) -> dict[str, Any]:
    """Run the product pipeline up to first frames and optional Wan manifest."""

    project_path = Path(project_dir).resolve()
    project_path.mkdir(parents=True, exist_ok=True)

    from .graph import build_multishot_graph

    graph = build_multishot_graph()
    state = graph.invoke(
        {
            "story": story,
            "project_dir": str(project_path),
            "backend": backend,
        }
    )

    result = {
        "project_dir": str(project_path.resolve()),
        "project_plan_path": state.get("project_plan_path"),
        "asset_index_path": state.get("asset_index_path"),
    }
    if write_manifest:
        manifest_path = build_wan_manifest(project_path, base_seed=base_seed)
        result["wan_manifest_path"] = str(manifest_path)
    return result


def run_full_project(
    story: str,
    project_dir: str | Path,
    *,
    backend: str = "qwen_image21",
    base_seed: int = 1000,
    reencode: bool = True,
) -> dict[str, Any]:
    """Run the full product chain from story to final assembled video."""

    result = run_story_project(
        story,
        project_dir,
        backend=backend,
        base_seed=base_seed,
        write_manifest=True,
    )
    video_report = generate_shot_videos(project_dir)
    final_video = assemble_videos(project_dir, reencode=reencode)
    result["wan_video_report_path"] = str(Path(project_dir) / "wan_video_report.json")
    result["completed_video_jobs"] = video_report.get("completed_jobs")
    result["failed_video_jobs"] = video_report.get("failed_jobs")
    result["final_video"] = str(final_video)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the product story-to-video pipeline without using benchmark code."
    )
    parser.add_argument("--story", help="Raw story text.")
    parser.add_argument("--story-file", help="Path to a text file containing the story.")
    parser.add_argument(
        "--project-dir",
        default=str(DEFAULT_OUTPUT_ROOT / "demo_project"),
        help="Output project directory.",
    )
    parser.add_argument(
        "--backend",
        default="qwen_image21",
        help="First-frame backend: qwen_image21, or an existing legacy diffusion model.",
    )
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only build wan_manifest_product.json from an existing project_plan.json.",
    )
    parser.add_argument(
        "--assemble-only",
        action="store_true",
        help="Only assemble existing shot videos listed in wan_manifest_product.json.",
    )
    parser.add_argument("--manifest-path", help="Manifest path for --assemble-only.")
    parser.add_argument("--final-video", help="Final assembled mp4 path.")
    parser.add_argument("--reencode", action="store_true", help="Reencode while assembling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir)

    if args.assemble_only:
        final_path = assemble_videos(
            project_dir,
            manifest_path=args.manifest_path,
            output_path=args.final_video,
            reencode=args.reencode,
        )
        print(json.dumps({"final_video": str(final_path)}, ensure_ascii=False, indent=2))
        return

    if args.manifest_only:
        manifest_path = build_wan_manifest(project_dir, base_seed=args.base_seed)
        print(json.dumps({"wan_manifest_path": str(manifest_path)}, ensure_ascii=False, indent=2))
        return

    story = _read_story(args)
    result = run_story_project(
        story,
        project_dir,
        backend=args.backend,
        base_seed=args.base_seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
