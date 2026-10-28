"""Dispatch single- or multi-character frames to the existing v7 implementation."""
from pathlib import Path
from types import SimpleNamespace
import json
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def generate(task: dict, output: Path):
    import torch
    from facexlib.parsing import init_parsing_model
    from multishot.ip_adapter_experiment_utils import _face_app
    from qwen140_pilot_qwen import make_pipe
    from qwen140_integrated_episodes import sha, digest, write_once
    from run_qwen_image21_shared_prefix_adaptive import run_shot

    spec = task["face_injection"]
    for item in spec["frozen_files"]:
        assert sha(Path(item["path"])) == item["sha256"], item["path"]
    characters = spec["characters"] if "characters" in spec else [spec["character"]]
    separate = spec.get("separate_conditioning_images", False)
    for image in task["images"]:
        assert image["sha256"] == sha(Path(image["path"]))
    if not separate:
        assert len(task["images"]) == len(characters)
        for image, character in zip(task["images"], characters):
            assert image["sha256"] == sha(Path(character["reference_image"]))
    for character in characters:
        result3d = json.loads(Path(character["facelift_result"]).read_text())
        assert result3d["source_reference_sha256"] == sha(Path(character["reference_image"]))
    if separate:
        # Keep the original v7 algorithm intact. Its two conditioning builders
        # receive exactly the Agent-selected image sequence, including the tail;
        # face assignment/preparation still receive all visible role identities.
        import run_qwen_image21_shared_prefix_adaptive as v7
        original_control = v7._official_control_and_fork_latent
        original_conditioning = v7._prepare_conditioning
        def conditioning_record(record):
            return {**record, "characters": [{"reference_image": image["path"]} for image in task["images"]]}
        def control_with_images(pipe, record, args):
            return original_control(pipe, conditioning_record(record), args)
        def state_with_images(pipe, record, args):
            return original_conditioning(pipe, conditioning_record(record), args)
        v7._official_control_and_fork_latent = control_with_images
        v7._prepare_conditioning = state_with_images
    backend = spec.get("preparation_backend", "original_area_selection")
    if backend == "all_detected_faces":
        import run_qwen_image21_shared_prefix_adaptive as v7
        from pretest.prepare_qwen_all_detected_faces import prepare_from_image
        v7.prepare_from_image = prepare_from_image
    elif backend != "original_area_selection":
        raise ValueError(f"Unknown preparation backend: {backend}")
    pipe = make_pipe("frames")
    app = _face_app()
    parser = init_parsing_model(model_name="bisenet", device=torch.device("cuda"))
    record = {"run": spec["run"], "episode_id": task["episode_id"], "shot_key": task["shot_key"],
              "seed": task["seed"], "official_prompt": spec["official_prompt"],
              "qwen_prompt": task["prompt"], "characters": characters}
    args = SimpleNamespace(**spec["settings"], overwrite=False)
    directory = output.parent / "injection"
    result = run_shot(pipe, app, parser, record, directory, args)
    if not result["injection_applied"] and not spec.get("allow_control_reuse", False):
        raise RuntimeError("Required v7 injection did not execute; stopped without using Control: "
                           + json.dumps({k: result.get(k) for k in ["reason", "error", "target_failures"]}, ensure_ascii=False))
    assert set(result["applied_characters"]) <= {c["name"] for c in characters}
    if result["injection_applied"]:
        assert any(s["injected"] and s.get("residual_norm", 0) > 0 for s in result["step_log"])
    else:
        assert sha(Path(result["control"])) == sha(Path(result["treatment"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    assert not output.exists()
    shutil.copyfile(result["treatment"], output)
    write_once(output.with_suffix(".json"), {
        "condition_sha256": digest(task), "inputs": task, "sha256": sha(output),
        "injection_applied": result["injection_applied"], "injection_result": str(directory / "result.json"),
        "skip_reason": None if result["injection_applied"] else result.get("reason"),
        "applied_characters": result["applied_characters"],
        "elapsed_seconds": result["elapsed_seconds"]})
    print("v7 keyframe", task["shot_key"], "injection_applied", result["injection_applied"], output, flush=True)
