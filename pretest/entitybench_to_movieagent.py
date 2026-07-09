import argparse
import json
from pathlib import Path


def convert_entitybench_episode(input_path: Path, output_path: Path):
    data = json.loads(input_path.read_text(encoding="utf-8"))

    entity_descriptions = data.get("entity_descriptions", {})
    characters = []
    for label, info in entity_descriptions.items():
        if label.startswith("Character") and isinstance(info, dict):
            name = info.get("name")
            if name and name not in characters:
                characters.append(name)

    movieagent_data = {
        "MovieScript": data.get("story_overview", ""),
        "Character": characters,
        "Source": {
            "benchmark": "EntityBench",
            "story_name": data.get("story_name"),
            "source_path": str(input_path),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(movieagent_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return movieagent_data


def main():
    parser = argparse.ArgumentParser(
        description="Convert one EntityBench episode JSON into MovieAgent script_synopsis format."
    )
    parser.add_argument("input_json", help="Path to EntityBench data/scripts/*.json")
    parser.add_argument("output_json", help="Output MovieAgent-style script_synopsis.json")
    args = parser.parse_args()

    result = convert_entitybench_episode(Path(args.input_json), Path(args.output_json))
    print(f"wrote {args.output_json}")
    print(f"characters: {', '.join(result['Character'])}")


if __name__ == "__main__":
    main()
