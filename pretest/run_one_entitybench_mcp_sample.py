import asyncio
import json
import os
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from multishot.agents import _tool_result  # noqa: E402
from multishot.face_analysis_backend import cosine_similarity, get_face_backend  # noqa: E402
from multishot.mcp_asset_server import PROJECT_DIR_ENV  # noqa: E402


EPISODE_PATH = Path("benchmarks/entitybench/data/scripts/000020ca-f66b-3542-8ad1-5c3172105f7b__run15__i0_j19__T240.json")
PROJECT_DIR = Path("pretest/one_sample_outputs/entitybench_000020ca_shot_1_1")
GENERATION_MODEL = "juggernaut-xl-v9"


def mcp_env(project_dir: Path):
    env = os.environ.copy()
    env[PROJECT_DIR_ENV] = str(project_dir)
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    # Smoke-test settings: fast enough to inspect while still using the real code path.
    env.setdefault("MULTISHOT_IMAGE_HEIGHT", "512")
    env.setdefault("MULTISHOT_IMAGE_WIDTH", "512")
    env.setdefault("MULTISHOT_DIFFUSION_STEPS", "24")
    env.setdefault("MULTISHOT_FINAL_STEP", "30")
    env.setdefault("MULTISHOT_DENOISE_WINDOW", "5")
    env.setdefault("MULTISHOT_ROLLOUT_CANDIDATES", "2")
    env.setdefault("MULTISHOT_ROLLOUT_PARALLELISM", "1")
    env.setdefault("MULTISHOT_ROLLOUT_TOPK", "1")
    env.setdefault("MULTISHOT_DIFFUSION_SEED", "123")
    return env


def load_sample():
    episode = json.loads(EPISODE_PATH.read_text(encoding="utf-8"))
    shot_key = "1:1"
    shot_prompt = episode["scenes"][0]["video_prompts"][0]
    action_description = episode["scenes"][0]["action_descriptions"][0]
    schedule = episode["entity_schedule"][shot_key]

    character_name = "Victor"
    character_description = None
    for label, info in episode["entity_descriptions"].items():
        if label.startswith("Character") and info.get("name") == character_name:
            character_description = info["description"]
            break
    location_name = schedule["places"][0]
    location_description = None
    for label, info in episode["entity_descriptions"].items():
        if label.startswith("Location") and info.get("name") == location_name:
            location_description = info["description"]
            break

    return {
        "episode_id": EPISODE_PATH.stem,
        "shot_key": shot_key,
        "shot_id": "shot_001",
        "subscript_id": "scene_001",
        "scene_name": "the_scholars_study",
        "character_id": "char_001",
        "character_name": "victor",
        "display_character_name": character_name,
        "character_description": character_description,
        "location_name": location_name,
        "location_description": location_description,
        "shot_prompt": shot_prompt,
        "action_description": action_description,
    }


def external_drift(reference_path: str, frame_path: str):
    backend = get_face_backend()
    reference_embedding = backend.first_face_embedding(reference_path)
    frame_faces = backend.analyze(frame_path)
    if reference_embedding is None:
        return {"error": "no face detected in reference image"}
    if not frame_faces:
        return {"error": "no face detected in generated first frame"}
    scored_faces = []
    for face in frame_faces:
        similarity = cosine_similarity(face["face_embedding"], reference_embedding)
        scored_faces.append({
            "face_id": face["face_id"],
            "face_bbox": face["face_bbox"],
            "face_confidence": face["face_confidence"],
            "similarity_to_reference": similarity,
            "drift_score": round(max(0.0, 1.0 - similarity), 4),
        })
    best = max(scored_faces, key=lambda item: item["similarity_to_reference"])
    return {"best_match": best, "all_faces": scored_faces}


def extract_log_drift(log_path: str):
    log = json.loads(Path(log_path).read_text(encoding="utf-8"))
    records = []
    for item in log:
        if item.get("action") == "check_face_clarity" and "face_state" in item:
            records.append({"step": item.get("step"), "source": "initial", "face_state": item["face_state"]})
        if item.get("action") == "rollout_select_best_denoise_state" and "best_face_state" in item:
            records.append({"step": item.get("step"), "source": "rollout", "face_state": item["best_face_state"]})
    final = records[-1] if records else None
    if not final:
        return {"error": "no face_state found in denoise log", "num_log_records": len(log)}
    faces = []
    for face in final["face_state"].get("faces", []):
        faces.append({
            "face_id": face.get("face_id"),
            "matched_character_id": face.get("matched_character_id"),
            "similarity": face.get("similarity"),
            "drift_score": face.get("drift_score"),
            "face_bbox": face.get("face_bbox"),
            "face_confidence": face.get("face_confidence"),
            "feature_source": face.get("feature_source"),
            "assignment_similarity": face.get("assignment_similarity"),
        })
    return {"step": final["step"], "source": final["source"], "faces": faces}


async def main():
    sample = load_sample()
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_DIR / "sample_input.json").write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")

    client = MultiServerMCPClient({
        "multishot_assets": {
            "command": sys.executable,
            "args": ["-m", "multishot.mcp_asset_server"],
            "transport": "stdio",
            "env": mcp_env(PROJECT_DIR),
        }
    })
    tools = {tool.name: tool for tool in await client.get_tools()}

    character_prompt = (
        f"{sample['display_character_name']}: {sample['character_description']}. "
        "single person, frontal face, looking at camera, head-and-shoulders portrait, "
        "sharp facial features, clear eyes, face occupies large area, simple clean background, "
        "photorealistic cinematic character reference, no sunglasses, no mask"
    )
    scene_prompt = (
        f"{sample['location_name']}: {sample['location_description']}. "
        "cinematic realistic background plate, dimly lit study interior, wooden desk, no people visible"
    )

    scene_asset = _tool_result(await tools["generate_scene_asset"].ainvoke({
        "subscript_id": sample["subscript_id"],
        "scene_name": sample["scene_name"],
        "prompt": scene_prompt,
    }))
    char_asset = _tool_result(await tools["generate_character_asset"].ainvoke({
        "character_id": sample["character_id"],
        "character_name": sample["character_name"],
        "prompt": character_prompt,
    }))
    face_asset = _tool_result(await tools["build_3d_face_asset"].ainvoke({
        "character_id": sample["character_id"],
        "reference_image_path": char_asset["path"],
    }))
    first_frame = _tool_result(await tools["generate_shot_first_frame"].ainvoke({
        "shot_id": sample["shot_id"],
        "subscript_id": sample["subscript_id"],
        "character_ids": [sample["character_id"]],
        "first_frame_prompt": sample["shot_prompt"],
        "generation_model": GENERATION_MODEL,
    }))

    drift = external_drift(char_asset["path"], first_frame["frame_path"])
    log_drift = extract_log_drift(first_frame["denoise_log_path"])

    summary = {
        "project_dir": str(PROJECT_DIR),
        "generation_model": GENERATION_MODEL,
        "sample": sample,
        "prompts": {
            "character_reference_prompt": character_prompt,
            "scene_asset_prompt": scene_prompt,
            "shot_first_frame_prompt": sample["shot_prompt"],
        },
        "assets": {
            "scene_asset": scene_asset,
            "character_asset": char_asset,
            "face_3d_asset": face_asset,
            "first_frame": first_frame,
        },
        "drift_external_final_frame_vs_reference": drift,
        "drift_from_denoise_log": log_drift,
    }
    summary_path = PROJECT_DIR / "sample_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
