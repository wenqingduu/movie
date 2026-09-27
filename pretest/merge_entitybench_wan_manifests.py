#!/usr/bin/env python3
"""Merge single-character and fallback jobs into one ordered Wan manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--single-manifest", type=Path, required=True)
    parser.add_argument("--fallback-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    episode = json.loads(args.episode.read_text(encoding="utf-8"))
    single = json.loads(args.single_manifest.read_text(encoding="utf-8"))
    fallback = json.loads(args.fallback_manifest.read_text(encoding="utf-8"))
    expected = []
    for scene in episode.get("scenes", []):
        expected.extend(
            f"{scene['scene_num']}:{index}"
            for index in range(1, len(scene.get("video_prompts", [])) + 1)
        )

    jobs = single.get("jobs", []) + fallback.get("jobs", [])
    by_key: dict[tuple[str, str], dict] = {}
    for job in jobs:
        key = (job["shot_key"], job["condition"])
        if key in by_key:
            raise ValueError(f"Duplicate job: {key}")
        by_key[key] = job
    missing = [
        (shot_key, condition)
        for shot_key in expected
        for condition in ("control", "treatment")
        if (shot_key, condition) not in by_key
    ]
    extras = sorted(set(by_key) - {
        (shot_key, condition)
        for shot_key in expected
        for condition in ("control", "treatment")
    })
    if missing or extras:
        raise ValueError(f"Manifest coverage mismatch; missing={missing}, extras={extras}")

    ordered = [
        by_key[(shot_key, condition)]
        for shot_key in expected
        for condition in ("control", "treatment")
    ]
    payload = {
        "kind": "entitybench_complete_ordered_episode_pulid_v7_wan22_manifest",
        "episode_id": args.episode.stem,
        "identity_references": single.get("identity_references", {}),
        "jobs": ordered,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Merged {len(expected)} shots / {len(ordered)} jobs -> {args.output}")


if __name__ == "__main__":
    main()
