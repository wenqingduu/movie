import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from multishot.diffusion_backend import get_diffusion_backend  # noqa: E402


def _shot_sort_key(key: str):
    scene, shot = key.split(":")
    return int(scene), int(shot)


def _iter_shots(episode: dict):
    for scene in episode.get("scenes", []):
        scene_num = scene["scene_num"]
        prompts = scene.get("video_prompts", [])
        actions = scene.get("action_descriptions", [])
        cuts = scene.get("cut", [])
        for index, prompt in enumerate(prompts, start=1):
            shot_key = f"{scene_num}:{index}"
            yield {
                "shot_key": shot_key,
                "scene_num": scene_num,
                "shot_num": index,
                "prompt": prompt,
                "action_description": actions[index - 1] if index - 1 < len(actions) else prompt,
                "cut": cuts[index - 1] if index - 1 < len(cuts) else None,
            }


def _load_split_ids(split_json: Path | None, tier: str | None):
    if split_json is None:
        return None
    data = json.loads(split_json.read_text(encoding="utf-8"))
    if tier:
        return set(data.get(tier, []))
    ids = set()
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                ids.update(value)
    elif isinstance(data, list):
        ids.update(data)
    return ids


def _select_episode_paths(scripts_dir: Path, episode_ids: list[str], split_ids: set[str] | None, limit_episodes: int | None):
    if episode_ids:
        paths = [scripts_dir / f"{episode_id}.json" for episode_id in episode_ids]
    else:
        paths = sorted(scripts_dir.glob("*.json"))
        if split_ids is not None:
            paths = [path for path in paths if path.stem in split_ids]
    if limit_episodes is not None:
        paths = paths[:limit_episodes]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing episode scripts:\n" + "\n".join(missing))
    return paths


def _scheduled_characters(episode: dict, shot_key: str):
    return episode.get("entity_schedule", {}).get(shot_key, {}).get("characters", []) or []


def _entity_descriptions(episode: dict):
    descriptions = {}
    for label, info in episode.get("entity_descriptions", {}).items():
        if not label.startswith("Character") or not isinstance(info, dict):
            continue
        name = info.get("name")
        if name:
            descriptions[name] = info.get("description", "")
    return descriptions


def generate_first_frames(
    scripts_dir: Path,
    output_dir: Path,
    episode_ids: list[str],
    split_json: Path | None,
    tier: str | None,
    limit_episodes: int | None,
    limit_shots: int | None,
    generation_model: str,
    overwrite: bool,
    dry_run: bool,
):
    split_ids = _load_split_ids(split_json, tier)
    episode_paths = _select_episode_paths(scripts_dir, episode_ids, split_ids, limit_episodes)
    backend = None if dry_run else get_diffusion_backend(generation_model)

    manifest = {
        "kind": "entitybench_first_frame_baseline",
        "generation_model": generation_model,
        "scripts_dir": str(scripts_dir),
        "output_dir": str(output_dir),
        "dry_run": dry_run,
        "episodes": [],
    }

    for episode_path in episode_paths:
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        episode_id = episode_path.stem
        episode_dir = output_dir / episode_id
        frame_dir = episode_dir / "frames"
        prompt_dir = episode_dir / "prompts"
        frame_dir.mkdir(parents=True, exist_ok=True)
        prompt_dir.mkdir(parents=True, exist_ok=True)

        shots = list(_iter_shots(episode))
        if limit_shots is not None:
            shots = shots[:limit_shots]

        episode_record = {
            "episode_id": episode_id,
            "script_path": str(episode_path),
            "story_overview": episode.get("story_overview", ""),
            "character_descriptions": _entity_descriptions(episode),
            "shots": [],
        }
        manifest["episodes"].append(episode_record)

        print(f"episode {episode_id}: {len(shots)} shots")
        for shot in shots:
            shot_key_for_file = shot["shot_key"].replace(":", "_")
            frame_path = frame_dir / f"shot_{shot_key_for_file}.png"
            prompt_path = prompt_dir / f"shot_{shot_key_for_file}.txt"
            prompt_path.write_text(shot["prompt"], encoding="utf-8")

            if dry_run:
                status = "dry_run"
            elif frame_path.exists() and not overwrite:
                status = "skipped_existing"
            else:
                backend.generate_image(shot["prompt"], str(frame_path))
                status = "generated"

            scheduled_characters = _scheduled_characters(episode, shot["shot_key"])
            episode_record["shots"].append({
                **shot,
                "scheduled_characters": scheduled_characters,
                "frame_path": str(frame_path),
                "prompt_path": str(prompt_path),
                "status": status,
            })
            print(f"  {shot['shot_key']} {status}: {frame_path}")

        episode_manifest_path = episode_dir / "manifest.json"
        episode_manifest_path.write_text(
            json.dumps(episode_record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest -> {manifest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Generate first-frame baseline images from EntityBench shot prompts without LLM/API planning."
    )
    parser.add_argument("--scripts-dir", default="benchmarks/entitybench/data/scripts")
    parser.add_argument("--output-dir", default="pretest/first_frame_outputs/entitybench_baseline")
    parser.add_argument("--episode-id", action="append", default=[], help="Episode id without .json; repeatable.")
    parser.add_argument("--split-json", default=None, help="Optional EntityBench split json.")
    parser.add_argument("--tier", default=None, help="Optional split tier, e.g. easy/medium/hard.")
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--limit-shots", type=int, default=None)
    parser.add_argument("--generation-model", default="segmind/tiny-sd")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    generate_first_frames(
        scripts_dir=Path(args.scripts_dir),
        output_dir=Path(args.output_dir),
        episode_ids=args.episode_id,
        split_json=Path(args.split_json) if args.split_json else None,
        tier=args.tier,
        limit_episodes=args.limit_episodes,
        limit_shots=args.limit_shots,
        generation_model=args.generation_model,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
