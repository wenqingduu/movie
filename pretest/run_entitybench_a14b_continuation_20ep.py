"""Durable smoke -> missing hard cuts -> full rollout -> non-VLM evaluation.

Run in tmux. Each subprocess owns the GPU sequentially. No VLM credentials
are passed to evaluators, and no generation/metrics in old runs are rewritten.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from prepare_entitybench_a14b_continuation import OUTPUT, PROJECT, prepare, sha256, write_json


def state(root, phase, **extra):
    write_json(root / "status.json", {"phase": phase, "updated_at": datetime.now(timezone.utc).isoformat(),
               "vlm_enabled": False, "evaluation_scope": "partial_without_vlm", **extra})


def environment():
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               PYTHONPATH=str(PROJECT) + os.pathsep + env.get("PYTHONPATH", ""))
    for key in ["GEMINI_API_KEYS", "AZURE_OPENAI_API_KEY", "OPENAI_API_KEY"]:
        env.pop(key, None)
    return env


def child(root, phase, command, env):
    state(root, phase, command=command)
    log = root / "logs" / f"{phase}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"Starting {phase}: {' '.join(command)}", flush=True)
    with log.open("a") as stream:
        subprocess.run(command, cwd=PROJECT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def verify_smoke(root):
    report = json.loads((root / "smoke/generation_report.json").read_text())
    assert report["status"] == "complete" and len(report["jobs"]) == 2
    first, second = report["jobs"]
    assert first["cut"] and not second["cut"]
    assert second["previous_shot_key"] == first["shot_key"]
    assert second["parent_video_sha256"] == first["video_sha256"]
    assert second["input_sha256"] == first["last_frame_sha256"]
    assert sha256(second["input_image"]) == sha256(first["last_frame"])
    assert all(row["decoded_frames"] == 81 and row["decoded_fps"] == 16 for row in report["jobs"])
    audit = {"status": "passed", "scope": "integration_and_memory_only; no quality selection",
             "same_route_parent_and_exact_tail_input": True, "decoded_frames_per_video": 81,
             "max_cuda_peak_allocated_bytes": max(r["cuda_peak_allocated_bytes"] for r in report["jobs"]),
             "config": json.loads((root / "smoke/generation_manifest.json").read_text())["config"]}
    write_json(root / "smoke/validation.json", audit)


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
        destination = root / "episodes" / (episode["run"] + ".mp4")
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
        prepare(root)
        if subprocess.run(["tmux", "has-session", "-t", "entitybench_a14b20_smoke"], capture_output=True).returncode == 0:
            state(root, "smoke_running", note="Waiting for the already-running integration test; no concurrent GPU load")
            deadline = time.monotonic() + 7200
            while subprocess.run(["tmux", "has-session", "-t", "entitybench_a14b20_smoke"], capture_output=True).returncode == 0:
                if time.monotonic() > deadline:
                    raise TimeoutError("Smoke test still running after two hours")
                time.sleep(10)
        elif not (root / "smoke/generation_report.json").is_file():
            child(root, "smoke", ["/root/autodl-tmp/wan22-venv/bin/python", "-u",
                  "pretest/run_wan22_i2v_a14b_continuation.py", "--output-root", str(root), "--smoke"], env)
        verify_smoke(root)
        child(root, "prepare_missing_firstframes", [str(PROJECT / ".venv/bin/python"), "-u",
              "pretest/prepare_entitybench_a14b_continuation.py", "--output-root", str(root), "--generate-missing"], env)
        child(root, "generate_formal", ["/root/autodl-tmp/wan22-venv/bin/python", "-u",
              "pretest/run_wan22_i2v_a14b_continuation.py", "--output-root", str(root)], env)
        state(root, "audit_and_concat")
        audit_and_concat(root)
        aux_env = dict(env)
        libs = ["/root/autodl-tmp/onnx_cuda12_libs/nvidia/cublas/lib", "/root/autodl-tmp/onnx_cuda12_libs/nvidia/cuda_runtime/lib",
                "/root/autodl-tmp/onnx_cuda12_libs/nvidia/cufft/lib", str(PROJECT / ".venv/lib/python3.10/site-packages/nvidia/cudnn/lib")]
        aux_env["LD_LIBRARY_PATH"] = ":".join(libs) + ":" + env.get("LD_LIBRARY_PATH", "")
        aux_env["MULTISHOT_INSIGHTFACE_PROVIDERS"] = "CUDAExecutionProvider,CPUExecutionProvider"
        child(root, "evaluate_project_identity", [str(PROJECT / ".venv/bin/python"), "-u",
              "pretest/evaluate_a14b_project_identity.py", "--output-root", str(root)], aux_env)
        eval_env = dict(env)
        eval_env.update(DINO_MODEL_PATH="/root/autodl-tmp/models/dinov2-base",
                        CLIP_MODEL_PATH="/root/autodl-tmp/models/clip-vit-base-patch32",
                        GDINO_CONFIG="/root/autodl-tmp/models/entitybench_groundingdino/GroundingDINO_SwinT_OGC.py",
                        GDINO_WEIGHTS="/root/autodl-tmp/models/entitybench_groundingdino/groundingdino_swint_ogc.pth",
                        VBENCH_CACHE_DIR="/root/autodl-tmp/models/entitybench_vbench",
                        TORCH_HOME="/root/autodl-tmp/models/entitybench_torch_cache",
                        HF_HOME="/root/autodl-tmp/models/entitybench_hf_cache")
        child(root, "evaluate_official_partial", ["/root/autodl-tmp/envs/entitybench-eval/bin/python", "-u",
              str(root / "runtime/evaluate_benchmark_partial.py"), "--results_dir", str(root / "official_results"),
              "--scripts_dir", str(PROJECT / "benchmarks/entitybench/data/scripts"), "--split_json", str(root / "split_20ep.json"),
              "--out_dir", str(root / "evaluation/official"), "--method_name", "pulid_a14b_continuation_20ep_no_vlm",
              "--pillars", "1,2,3", "--skip_vlm", "--resume", "--device", "cuda",
              "--vbench_cache_dir", eval_env["VBENCH_CACHE_DIR"]], eval_env)
        evaluation_manifest = json.loads((root / "evaluation/official/run_manifest.json").read_text())
        assert evaluation_manifest["skip_vlm"] is True
        assert evaluation_manifest["vlm_call_stats"]["total_calls"] == 0
        state(root, "non_vlm_complete_pending_visual_review", generated_shots=363, complete_episodes=20)
        (root / "report.md").write_text("# 20 episode A14B 接续路线\n\n"
            "363 shot、20 episode已生成并通过来源/尾帧/视频完整性审计。单组，不计算C/T增量。\n\n"
            "[官方非VLM部分报告](evaluation/official/report_pulid_a14b_continuation_20ep_no_vlm.md)；"
            "[项目单人身份与质心报告](evaluation/project/identity_report.json)；"
            "[全部镜头审计](audit/validation.json)。\n\n"
            "VLM未运行，保真度gate未应用；跨镜头分数为暂定值，不能直接与完整官方结果比较。"
            "多人/无人镜头保留在官方评测；项目身份辅助评分限132个单人shot。"
            "API缺失项不填零。代表性质量/漂移例仍待人工查看，不把流水线退出当作视觉结论。\n")
    except Exception as error:
        current = json.loads((root / "status.json").read_text()) if (root / "status.json").exists() else {}
        state(root, "failed", failed_phase=current.get("phase"), error=repr(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    main(parser.parse_args().output_root)
