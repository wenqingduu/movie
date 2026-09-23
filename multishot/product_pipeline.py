"""Product-oriented entrypoint for story-to-video projects.

This module is intentionally separate from ``pretest``.  The pretest package is
for benchmark/evaluation workflows; this file provides a stable surface that a
CLI, FastAPI service, or worker can call for user-facing projects.
"""

from __future__ import annotations

import argparse
import json
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
                "shot_index": index,
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


def run_story_project(
    story: str,
    project_dir: str | Path,
    *,
    generation_model: str = "juggernaut-xl-v9",
    base_seed: int = 1000,
    write_manifest: bool = True,
) -> dict[str, Any]:
    """Run the product pipeline up to first frames and optional Wan manifest."""

    project_path = Path(project_dir)
    project_path.mkdir(parents=True, exist_ok=True)

    from .graph import build_multishot_graph

    graph = build_multishot_graph()
    state = graph.invoke(
        {
            "story": story,
            "project_dir": str(project_path),
            "generation_model": generation_model,
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
        "--generation-model",
        default="juggernaut-xl-v9",
        help="First-frame image generation model configured in diffusion_backend.py.",
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
        generation_model=args.generation_model,
        base_seed=args.base_seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
