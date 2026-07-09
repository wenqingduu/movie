import argparse
import json
import os
import sys
from pathlib import Path


MOVIEAGENT_ROOT = Path(__file__).resolve().parent / "MovieAgent"
MOVIEAGENT_CODE = MOVIEAGENT_ROOT / "movie_agent"
sys.path.insert(0, str(MOVIEAGENT_CODE))

from base_agent import BaseAgent  # noqa: E402
from system_prompts import sys_prompts  # noqa: E402


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_script(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["MovieScript"], data["Character"]


def ensure_openai_key_for_dashscope():
    if not os.getenv("OPENAI_API_KEY") and os.getenv("DASHSCOPE_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["DASHSCOPE_API_KEY"]


def run_planning(script_path: Path, output_dir: Path, llm: str):
    ensure_openai_key_for_dashscope()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY or DASHSCOPE_API_KEY before running planning.")

    movie_script, characters = load_script(script_path)
    characters_text = json.dumps(characters, ensure_ascii=False)

    screenwriter = BaseAgent(
        llm,
        system_prompt=sys_prompts["screenwriterCoT-sys"],
        use_history=False,
        temp=0.7,
    )
    sceneplanner = BaseAgent(
        llm,
        system_prompt=sys_prompts["ScenePlanningCoT-sys"],
        use_history=False,
        temp=0.7,
    )
    shotplanner = BaseAgent(
        llm,
        system_prompt=sys_prompts["ShotPlotCreateCoT-sys"],
        use_history=False,
        temp=0.7,
    )

    script_query = f"""
Script Synopsis: {movie_script}
Character: {characters_text}
"""
    step1 = screenwriter(script_query, parse=True)
    step1_path = output_dir / "Step_1_script_results.json"
    save_json(step1, step1_path)

    data_scene = step1
    relationships = step1.get("Relationships", {})
    sub_scripts = step1.get("Sub-Script", {})
    for sub_script_name, sub_script_info in sub_scripts.items():
        sub_script = sub_script_info.get("Plot", "")
        scene_query = f"""
Given the following inputs:
- Script Synopsis: "{sub_script}"
- Character Relationships: {json.dumps(relationships, ensure_ascii=False)}
"""
        scene_result = sceneplanner(scene_query, parse=True)
        data_scene["Sub-Script"][sub_script_name]["Scene Annotation"] = scene_result
        save_json(data_scene, output_dir / "Step_2_scene_results.json")

    data_shot = data_scene
    for sub_script_name, sub_script_info in data_shot.get("Sub-Script", {}).items():
        scene_annotation = sub_script_info.get("Scene Annotation", {})
        scenes = scene_annotation.get("Scene", {})
        for scene_name, scene_details in scenes.items():
            shot_query = f"""
Given the following Scene Details:
- Involving Characters: "{scene_details.get('Involving Characters', '')}"
- Plot: "{scene_details.get('Plot', '')}"
- Scene Description: "{scene_details.get('Scene Description', '')}"
- Emotional Tone: "{scene_details.get('Emotional Tone', '')}"
- Key Props: {scene_details.get('Key Props', [])}
- Cinematography Notes: "{scene_details.get('Cinematography Notes', '')}"
"""
            shot_result = shotplanner(shot_query, parse=True)
            data_shot["Sub-Script"][sub_script_name]["Scene Annotation"]["Scene"][scene_name]["Shot Annotation"] = shot_result
            save_json(data_shot, output_dir / "Step_3_shot_results.json")

    return {
        "step1": str(step1_path),
        "step2": str(output_dir / "Step_2_scene_results.json"),
        "step3": str(output_dir / "Step_3_shot_results.json"),
    }


def main():
    parser = argparse.ArgumentParser(description="Run MovieAgent CoT planning without image/video/audio generation.")
    parser.add_argument("--script-path", required=True, help="MovieAgent-style script_synopsis.json")
    parser.add_argument("--output-dir", required=True, help="Directory for Step_*.json outputs")
    parser.add_argument("--llm", default="qwen-plus", help="OpenAI-compatible model name, default qwen-plus")
    args = parser.parse_args()

    outputs = run_planning(Path(args.script_path), Path(args.output_dir), args.llm)
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
