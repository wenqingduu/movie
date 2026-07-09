import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def analyze_episode(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    schedule = data.get("entity_schedule", {})
    shot_order = sorted(schedule.keys(), key=_shot_sort_key)
    shot_to_index = {shot_key: index for index, shot_key in enumerate(shot_order)}
    character_to_shots = defaultdict(list)

    for shot_key in shot_order:
        for character in schedule[shot_key].get("characters", []) or []:
            character_to_shots[character].append(shot_key)

    recurrent = {
        name: shots
        for name, shots in character_to_shots.items()
        if len(shots) >= 2
    }
    max_gap = 0
    for shots in recurrent.values():
        positions = [shot_to_index[key] for key in shots]
        if len(positions) >= 2:
            max_gap = max(max_gap, max(positions) - min(positions))

    return {
        "episode_id": path.stem,
        "path": str(path),
        "story_overview": data.get("story_overview", ""),
        "num_shots": len(schedule),
        "num_characters": len(character_to_shots),
        "num_recurrent_characters": len(recurrent),
        "max_recurrence_gap": max_gap,
        "recurrent_characters": recurrent,
    }


def _shot_sort_key(key: str):
    scene, shot = key.split(":")
    return int(scene), int(shot)



def main():
    parser = argparse.ArgumentParser(
        description="Analyze EntityBench episodes for cross-shot character recurrence."
    )
    parser.add_argument(
        "--scripts-dir",
        default="benchmarks/entitybench/data/scripts",
        help="Directory containing EntityBench episode JSON files.",
    )
    parser.add_argument(
        "--output-json",
        default="pretest/entitybench_recurrence_report.json",
        help="Where to save the recurrence report.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of strongest candidate episodes to print.",
    )
    args = parser.parse_args()

    scripts = sorted(Path(args.scripts_dir).glob("*.json"))
    episodes = [analyze_episode(path) for path in scripts]
    candidates = [item for item in episodes if item["num_recurrent_characters"] > 0]
    candidates.sort(
        key=lambda item: (
            item["num_recurrent_characters"],
            item["max_recurrence_gap"],
            item["num_shots"],
        ),
        reverse=True,
    )

    summary = {
        "scripts_dir": args.scripts_dir,
        "num_episodes": len(episodes),
        "num_candidate_episodes": len(candidates),
        "shot_count_histogram": Counter(item["num_shots"] for item in episodes),
        "top_candidates": candidates[: args.top_k],
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"episodes: {len(episodes)}")
    print(f"episodes with recurrent characters: {len(candidates)}")
    print(f"report: {output_path}")
    for item in candidates[: args.top_k]:
        chars = ", ".join(
            f"{name}({len(shots)})"
            for name, shots in item["recurrent_characters"].items()
        )
        print(
            f"{item['episode_id']} | shots={item['num_shots']} | "
            f"recurrent={item['num_recurrent_characters']} | "
            f"max_gap={item['max_recurrence_gap']} | {chars}"
        )


if __name__ == "__main__":
    main()
