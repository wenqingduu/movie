"""Durable 4-step rollout and official non-VLM evaluation, with exclusive GPU use.

The integration preflight uses two existing pilot videos in an isolated folder.
Formal generation starts from scratch; no pilot or old 40-step video is reused.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from prepare_entitybench_a14b_4step import OUTPUT, PROJECT, prepare, sha256, write_json
from run_lightx2v_a14b_4step_continuation import fingerprint

EVAL_PYTHON = "/root/autodl-tmp/envs/entitybench-eval/bin/python"
GEN_PYTHON = "/root/autodl-tmp/envs/lightx2v-trial/bin/python"
METHOD = "pulid_a14b_4step_continuation_20ep_no_vlm"
DIMENSIONS = ["subject_consistency", "temporal_flickering", "motion_smoothness",
              "dynamic_degree", "aesthetic_quality", "imaging_quality"]


def state(root, phase, **extra):
    write_json(root / "status.json", dict(phase=phase, updated_at=datetime.now(timezone.utc).isoformat(),
        vlm_enabled=False, evaluation_scope="partial_without_vlm", project_identity_enabled=False, **extra))


def environment():
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        PYTHONPATH=str(PROJECT) + os.pathsep + env.get("PYTHONPATH", ""),
        DINO_MODEL_PATH="/root/autodl-tmp/models/dinov2-base",
        CLIP_MODEL_PATH="/root/autodl-tmp/models/clip-vit-base-patch32",
        GDINO_CONFIG="/root/autodl-tmp/models/entitybench_groundingdino/GroundingDINO_SwinT_OGC.py",
        GDINO_WEIGHTS="/root/autodl-tmp/models/entitybench_groundingdino/groundingdino_swint_ogc.pth",
        VBENCH_CACHE_DIR="/root/autodl-tmp/models/entitybench_vbench",
        TORCH_HOME="/root/autodl-tmp/models/entitybench_torch_cache",
        HF_HOME="/root/autodl-tmp/models/entitybench_hf_cache")
    for key in list(env):
        if any(prefix in key.upper() for prefix in ("OPENAI", "AZURE", "GEMINI", "GOOGLE_API_KEY")):
            env.pop(key)
    return env


def child(root, phase, command, env):
    state(root, phase, command=command)
    log = root / "logs" / f"{phase}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"Starting {phase}: {' '.join(command)}", flush=True)
    with log.open("a") as stream:
        subprocess.run(command, cwd=PROJECT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def evaluation_command(root, results, scripts, split, out, method):
    return [EVAL_PYTHON, "-u", str(root / "runtime/evaluate_benchmark_partial.py"),
        "--results_dir", str(results), "--scripts_dir", str(scripts), "--split_json", str(split),
        "--out_dir", str(out), "--method_name", method, "--pillars", "1,2,3", "--skip_vlm",
        "--resume", "--device", "cuda", "--vbench_cache_dir", "/root/autodl-tmp/models/entitybench_vbench"]


def protect_evaluation_resume(out, inputs):
    """Official episode caches do not hash videos: refuse reuse when inputs change."""
    expected = {"fingerprint": fingerprint(inputs), "inputs": inputs}
    path = out / "input_fingerprint.json"
    if path.exists():
        assert json.loads(path.read_text()) == expected, "Evaluation inputs changed; use a fresh score directory"
    else:
        assert not (out / "run_manifest.json").exists(), "Existing scores have no input provenance"
        write_json(path, expected)


def verify_evaluation(out, episode_ids, method):
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["skip_vlm"] is True and manifest["evaluation_scope"] == "partial_without_vlm"
    assert manifest["strict"] is True and not manifest["allow_config_override"] and not manifest["allow_partial"]
    assert manifest["vlm_call_stats"]["total_calls"] == 0
    for eid in episode_ids:
        report = json.loads((out / f"{eid}.json").read_text())
        assert report["episode_id"] == eid
        for dimension in DIMENSIONS:
            metric = report["vbench_means"][dimension]
            assert metric["value"] is not None and metric["n_evaluated"] > 0, (eid, dimension)
            assert metric["n_failed"] == 0, (eid, dimension, metric)
    assert (out / f"report_{method}.json").is_file()
    assert (out / f"report_{method}.md").is_file()


def integration_preflight(root, manifest, env):
    """Test actual official metrics, never select candidates by their scores."""
    preflight = root / "preflight"
    eid = manifest["episodes"][0]["episode_id"]
    original = Path(manifest["episodes"][0]["episode_path"])
    script = copy.deepcopy(json.loads(original.read_text()))
    scene = next(s for s in script["scenes"] if s["scene_num"] == 4)
    for key in ("video_prompts", "cut", "action_descriptions"):
        if key in scene:
            scene[key] = scene[key][:2]
    script["scenes"] = [scene]
    script["entity_schedule"] = {k: v for k, v in script["entity_schedule"].items() if k in {"4:1", "4:2"}}
    script["_window_shot_ids"] = ["4:1", "4:2"]
    script["_num_scenes"], script["_num_shots"] = 1, 2
    script_path = preflight / "scripts" / f"{eid}.json"
    write_json(script_path, script)
    write_json(preflight / "split.json", {"easy": [eid], "medium": [], "hard": []})
    inputs = {"scope": "integration_only_not_formal_scores", "original_script_sha256": sha256(original),
        "derived_script_sha256": sha256(script_path), "evaluator_sha256": sha256(root / "runtime/evaluate_benchmark_partial.py"),
        "videos": {}}
    for number in (1, 2):
        video = PROJECT / f"outputs/entitybench_a14b_acceleration_trial/lightx2v_distill_bf16_4step/shot_4_{number}/video.mp4"
        assert video.is_file()
        inputs["videos"][str(number)] = sha256(video)
        link = preflight / "results" / eid / f"004_{number:03d}.mp4"
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            link.unlink()
        assert not link.exists()
        link.symlink_to(video)
    out = preflight / "official"
    protect_evaluation_resume(out, inputs)
    method = "integration_only_4step_no_vlm"
    child(root, "official_preflight", evaluation_command(root, preflight / "results", preflight / "scripts",
        preflight / "split.json", out, method), env)
    verify_evaluation(out, [eid], method)
    write_json(preflight / "validation.json", dict(status="passed", official_six_vbench_dimensions=DIMENSIONS,
        scope="integration_only; excluded from formal results; no quality selection", zero_vlm_calls=True))


def episode_video_path(root, episode):
    # A run may appear in multiple windows of the frozen cohort.
    return root / "episodes" / (episode["episode_id"] + ".mp4")


def audit_and_concat(root):
    from PIL import Image, ImageDraw
    manifest = json.loads((root / "manifest.json").read_text())
    report = json.loads((root / "generation_report.json").read_text())
    assert report["status"] == "complete" and len(report["jobs"]) == 363
    jobs = {(r["episode_id"], r["shot_key"]): r for r in report["jobs"]}
    audit, episode_rows = [], []
    for episode in manifest["episodes"]:
        previous, videos = None, []
        for shot in episode["shots"]:
            row = jobs[(episode["episode_id"], shot["shot_key"])]
            video = Path(row["output_video"])
            assert row["status"] in {"generated", "resumed_verified"}
            assert sha256(video) == row["video_sha256"]
            assert sha256(row["last_frame"]) == row["last_frame_sha256"]
            assert sha256(video.parent / "first.png") == row["first_frame_sha256"]
            assert sha256(row["input_image"]) == row["input_sha256"]
            assert row["decoded_frames"] == 81 and row["decoded_fps"] == 16
            if shot["cut"]:
                assert row["input_sha256"] == shot["firstframe_source"]["input_sha256"]
            else:
                assert previous is not None and row["previous_shot_key"] == previous["shot_key"]
                assert row["parent_video_sha256"] == previous["video_sha256"]
                assert row["input_sha256"] == previous["last_frame_sha256"]
            scene, local = map(int, shot["shot_key"].split(":"))
            link = root / "official_results" / episode["episode_id"] / f"{scene:03d}_{local:03d}.mp4"
            assert link.is_symlink() and link.resolve() == video.resolve()
            panels = [("hard-cut source" if shot["cut"] else "own previous tail", Path(row["source_input"])),
                      ("actual I2V input", Path(row["input_image"])),
                      ("actual video first", video.parent / "first.png"),
                      ("actual video last", video.parent / "last.png")]
            canvas = Image.new("RGB", (1600, 265), "#151515")
            draw = ImageDraw.Draw(canvas)
            for index, (label, path) in enumerate(panels):
                image = Image.open(path).convert("RGB")
                image.thumbnail((400, 220))
                canvas.paste(image, (index * 400, 35))
                draw.text((index * 400 + 8, 8), label, fill="white")
            contact = root / "audit" / episode["episode_id"] / ("shot_" + shot["shot_key"].replace(":", "_") + ".jpg")
            contact.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(contact, quality=90)
            audit.append({"episode_id": episode["episode_id"], "shot_key": shot["shot_key"],
                          "cut": shot["cut"], "source_type": row["input_source_type"],
                          "video_sha256": row["video_sha256"], "contact_sheet": str(contact), "verified": True})
            videos.append(video)
            previous = row
        destination = episode_video_path(root, episode)
        destination.parent.mkdir(parents=True, exist_ok=True)
        concat_file = destination.with_suffix(".concat.txt")
        concat_file.write_text("".join(f"file '{video}'\n" for video in videos))
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i",
                        str(concat_file), "-c", "copy", str(destination)], check=True)
        probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                   "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "json", str(destination)], text=True))
        assert int(probe["streams"][0]["nb_read_frames"]) == len(videos) * 81
        episode_rows.append({"episode_id": episode["episode_id"], "video": str(destination),
                             "sha256": sha256(destination), "shot_count": len(videos), "frame_count": len(videos) * 81,
                             "concat_policy": "retain every canonical frame; no trimming; official evaluation uses original shots"})
    write_json(root / "audit/validation.json", {"verified_shots": len(audit), "verified_continuations": sum(not r["cut"] for r in audit),
               "shots": audit, "episodes": episode_rows})


def main(root):
    root = Path(root).resolve()
    env = environment()
    try:
        state(root, "prepare")
        manifest = prepare(root)
        integration_preflight(root, manifest, env)
        child(root, "generate_formal", [GEN_PYTHON, "-u", "pretest/run_lightx2v_a14b_4step_continuation.py",
            "--output-root", str(root)], env)
        state(root, "audit_and_concat")
        audit_and_concat(root)
        generation = json.loads((root / "generation_report.json").read_text())
        inputs = dict(manifest_sha256=sha256(root / "manifest.json"),
            evaluator_sha256=sha256(root / "runtime/evaluate_benchmark_partial.py"),
            split_sha256=sha256(root / "split_20ep.json"),
            scripts={e["episode_id"]: sha256(e["episode_path"]) for e in manifest["episodes"]},
            videos={f'{r["episode_id"]}/{r["shot_key"]}': r["video_sha256"] for r in generation["jobs"]})
        out = root / "evaluation/official"
        protect_evaluation_resume(out, inputs)
        child(root, "evaluate_official_non_vlm", evaluation_command(root, root / "official_results",
            PROJECT / "benchmarks/entitybench/data/scripts", root / "split_20ep.json", out, METHOD), env)
        verify_evaluation(out, [e["episode_id"] for e in manifest["episodes"]], METHOD)
        state(root, "non_vlm_complete_pending_visual_review", generated_shots=363, complete_episodes=20)
        (root / "report.md").write_text("# 20 episode 4步A14B接续路线\n\n"
            "363镜头、20 episode已生成并通过来源/尾帧/视频完整性审计。单组系统评测，无C/T增量。\n\n"
            f"[官方非VLM报告](evaluation/official/report_{METHOD}.md)；"
            "[全部镜头审计](audit/validation.json)；完整视频见episodes/。\n\n"
            "只运行官方6项VBench、跨镜头DINO、接续边界与实体出现率，不运行项目身份辅助指标。\n"
            "VLM未运行，保真度gate未应用，cs_face/cs_object为暂定值，不能直接对齐完整论文表格。\n"
            "失败、缺失和多人镜头保持原样；代表性质量/漂移仍待人工检查。\n")
    except Exception as error:
        current = json.loads((root / "status.json").read_text()) if (root / "status.json").exists() else {}
        state(root, "failed", failed_phase=current.get("phase"), error=repr(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    main(parser.parse_args().output_root)
