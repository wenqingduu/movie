#!/usr/bin/env python3
"""Stage existing paired videos for the official EntityBench evaluator.

The evaluator expects one directory per method with files laid out as::

    <results_dir>/<episode_id>/<scene>_<shot>.mp4

Our generation manifests instead keep Control and Treatment videos together.
This utility creates symlink-only staging trees, so no videos are copied or
modified.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        action="append",
        help="Wan manifest to stage. Repeat to build a multi-episode split.",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--treatment-conditions",
        default="treatment,color_safe_core",
        help="Comma-separated manifest conditions accepted as v7 Treatment.",
    )
    parser.add_argument(
        "--split-name",
        default="split.json",
        help="Filename written below output-root for the combined episode split.",
    )
    return parser.parse_args()


def _replace_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"Refusing to replace non-symlink: {link}")
    os.symlink(target.resolve(), link)


def main() -> None:
    args = _parse_args()
    accepted_treatments = {
        item.strip() for item in args.treatment_conditions.split(",") if item.strip()
    }
    episode_ids: list[str] = []
    staged_shots = 0
    for raw_manifest_path in args.manifest:
        manifest_path = raw_manifest_path.resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        episode_id = manifest["episode_id"]
        if episode_id in episode_ids:
            raise ValueError(f"Duplicate episode manifest: {episode_id}")
        episode_ids.append(episode_id)

        selected: dict[tuple[str, str], Path] = {}
        for job in manifest.get("jobs", []):
            condition = job.get("condition")
            if condition == "control":
                staged_condition = "control"
            elif condition in accepted_treatments:
                staged_condition = "v7"
            else:
                continue
            shot_key = job["shot_key"]
            key = (staged_condition, shot_key)
            if key in selected:
                raise ValueError(
                    f"{episode_id}: duplicate {staged_condition} video for shot {shot_key}"
                )
            source = Path(job["output_video"])
            if not source.is_file():
                raise FileNotFoundError(source)
            selected[key] = source

        shot_keys = sorted(
            {shot_key for condition, shot_key in selected if condition == "control"},
            key=lambda value: tuple(int(part) for part in value.split(":")),
        )
        missing = [
            f"{condition}:{shot_key}"
            for condition in ("control", "v7")
            for shot_key in shot_keys
            if (condition, shot_key) not in selected
        ]
        if missing:
            raise ValueError(f"{episode_id}: missing paired videos: {', '.join(missing)}")

        for condition in ("control", "v7"):
            episode_dir = args.output_root / condition / episode_id
            for shot_key in shot_keys:
                scene, shot = (int(part) for part in shot_key.split(":"))
                destination = episode_dir / f"{scene:03d}_{shot:03d}.mp4"
                _replace_symlink(destination, selected[(condition, shot_key)])
        staged_shots += len(shot_keys)
        print(f"Staged {len(shot_keys)} paired shots for {episode_id}")

    split_path = args.output_root / args.split_name
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        json.dumps({"easy": episode_ids, "medium": [], "hard": []}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"Staged {staged_shots} paired shots across {len(episode_ids)} episodes")
    print(f"Control: {args.output_root / 'control'}")
    print(f"v7:      {args.output_root / 'v7'}")
    print(f"Split:   {split_path}")


if __name__ == "__main__":
    main()
