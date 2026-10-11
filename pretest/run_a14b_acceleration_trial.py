"""Isolated LightX2V feasibility trial; never writes to formal generation outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "outputs/entitybench_a14b_acceleration_trial"
LIGHT = Path("/root/autodl-tmp/LightX2V")
sys.path.insert(0, str(PROJECT))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def build(route, preflight):
    import torch
    # The CUDA pinned-host allocator retained ~81 GiB during the first attempt,
    # exceeding this container's 96 GiB budget when UMT5 was loaded. Restrict this
    # process to pageable CPU storage; values and BF16 precision are unchanged.
    if not preflight:
        original_empty = torch.empty
        def pageable_empty(*args, **kwargs):
            if kwargs.get("pin_memory"):
                kwargs["pin_memory"] = False
            return original_empty(*args, **kwargs)
        torch.empty = pageable_empty
    sys.path.insert(0, str(LIGHT))
    from lightx2v import LightX2VPipeline
    from lightx2v.utils.set_config import build_startup_config
    plan = json.loads((ROOT / "plan.json").read_text())
    assert subprocess.check_output(["git", "-C", str(LIGHT), "rev-parse", "HEAD"], text=True).strip() == plan["lightx2v_revision"]
    settings = plan["routes"][route]
    pipe = LightX2VPipeline(model_path=plan["base_model"], model_cls="wan2.2_moe", task="i2v",
                          distill_method="dmd2" if settings["distillation"] else None)
    pipe.enable_offload(cpu_offload=True, offload_granularity="model", text_encoder_offload=True,
                        image_encoder_offload=False, vae_offload=False)
    pipe.startup_config.update(rms_norm_type="torch", t5_rms_norm_type="torch", layer_norm_type="torch",
                               clean_cuda_cache=False, use_image_encoder=False, fps=16)
    pipe.startup_config["trial_cpu_pinned_memory"] = False
    if settings["distillation"]:
        download = json.loads(Path(plan["lora_download_manifest"]).read_text())
        loras = [{"name": "high_noise_model" if "high_noise" in f["path"] else "low_noise_model",
                  "path": str(Path(download["destination"]) / f["path"]), "strength": 1.0} for f in download["files"]]
        if not preflight:
            assert json.loads((Path(plan["lora_download_manifest"]).parent / "status.json").read_text())["state"] == "complete"
            for f in download["files"]:
                assert sha(Path(download["destination"]) / f["path"]) == f["sha256"]
        pipe.enable_lora(loras, lora_dynamic_apply=False)
    if preflight:
        pipe.set_infer_config("torch_sdpa", "torch_complex_rope", settings["steps"], 81, (480, 832),
                              settings["guidance"], settings["shift"], "16:9", .9, 2,
                              (1000, 750, 500, 250), True, "torch", None)
        config = build_startup_config(pipe.startup_config)
        from lightx2v.models.runners.wan.wan_runner import Wan22MoeRunner
        runner = Wan22MoeRunner(config)  # Config/scheduler only; no weight allocation.
        assert runner.config["infer_steps"] == settings["steps"]
        write(ROOT / route / "preflight.json", {"status": "passed", "torch": torch.__version__, "config": dict(config)})
        return None, None
    begin = time.perf_counter()
    pipe.create_generator(attn_mode="torch_sdpa", infer_steps=settings["steps"], num_frames=81,
                           size=(480, 832), guidance_scale=settings["guidance"], sample_shift=settings["shift"],
                           fps=16, modulate_type="torch")
    return pipe, time.perf_counter() - begin


def run(route, preflight=False):
    import torch
    from PIL import Image
    import imageio.v2 as imageio
    torch.set_num_threads(8)
    plan = json.loads((ROOT / "plan.json").read_text())
    pipe, loading = build(route, preflight)
    if preflight:
        return
    dest = ROOT / route
    metadata = {"route": route, "torch": torch.__version__, "cuda": torch.version.cuda,
                "implementation_sha256": sha(__file__), "lightx2v_revision": plan["lightx2v_revision"],
                "config": dict(pipe.runner.config), "model_load_seconds": loading, "jobs": [],
                "benchmark_scope": "pilot only; no official metrics", "status": "running"}
    write(dest / "report.json", metadata)
    wanted = plan["routes"][route]["shots"]
    parent = None
    sys.path.insert(0, "/root/autodl-tmp/Wan2.2")
    from wan.configs.shared_config import wan_shared_cfg
    for shot in plan["shots"]:
        if shot["shot_key"] not in wanted:
            continue
        name = "shot_" + shot["shot_key"].replace(":", "_")
        folder = dest / name
        folder.mkdir(parents=True, exist_ok=True)
        if shot["cut"]:
            source = Path(shot["firstframe_source"]["input_image"])
            assert sha(source) == shot["firstframe_source"]["input_sha256"]
        else:
            assert parent and parent["shot_key"] == shot["previous_shot_key"]
            source = Path(parent["last_frame"])
            assert sha(source) == parent["last_frame_sha256"]
        shutil.copy2(source, folder / "input.png")
        record = {"shot_key": shot["shot_key"], "cut": shot["cut"], "prompt": shot["prompt"],
                  "seed": shot["seed"], "source_image": str(source), "input_sha256": sha(source),
                  "parent_video_sha256": None if shot["cut"] else parent["video_sha256"],
                  "negative_prompt": wan_shared_cfg.sample_neg_prompt}
        record["negative_prompt_active"] = bool(pipe.runner.config.get("enable_cfg", False))
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        begin = time.perf_counter()
        temporary = folder / "video.tmp.mp4"
        print(f"TRIAL GENERATE {route} {shot['shot_key']}", flush=True)
        try:
            request = dict(seed=shot["seed"], prompt=shot["prompt"],
                           image_path=str(folder / "input.png"), save_result_path=str(temporary))
            if record["negative_prompt_active"]:
                request["negative_prompt"] = record["negative_prompt"]
            pipe.generate(**request)
            torch.cuda.synchronize()
            record["generate_seconds"] = time.perf_counter() - begin
            record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            record["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            reader = imageio.get_reader(str(temporary), "ffmpeg")
            frames = [frame for frame in reader]
            fps = reader.get_meta_data()["fps"]
            reader.close()
            assert len(frames) == 81 and abs(fps - 16) < .01, (len(frames), fps)
            Image.fromarray(frames[0]).save(folder / "first.png")
            Image.fromarray(frames[-1]).save(folder / "last.png")
            for index in [0, 20, 40, 60, 80]:
                Image.fromarray(frames[index]).save(folder / f"frame_{index:03d}.jpg")
            record.update(frames=81, fps=fps, width=frames[0].shape[1], height=frames[0].shape[0])
            baseline = plan["baseline_records"][shot["shot_key"]]
            assert [record["width"], record["height"]] == [baseline["width"], baseline["height"]]
            temporary.replace(folder / "video.mp4")
            record.update(status="generated", video=str(folder / "video.mp4"), video_sha256=sha(folder / "video.mp4"),
                          last_frame=str(folder / "last.png"), last_frame_sha256=sha(folder / "last.png"))
            print(f"TRIAL COMPLETE {route} {shot['shot_key']} {record['generate_seconds']:.2f}s", flush=True)
            parent = record
        except BaseException as error:
            record.update(status="failed", error=repr(error))
            metadata["jobs"].append(record)
            metadata["status"] = "failed"
            write(folder / "generation.json", record)
            write(dest / "report.json", metadata)
            raise
        write(folder / "generation.json", record)
        metadata["jobs"].append(record)
        write(dest / "report.json", metadata)
    metadata["status"] = "complete"
    write(dest / "report.json", metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", required=True, choices=["lightx2v_bf16_40step", "lightx2v_distill_bf16_4step"])
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    run(args.route, args.preflight)
