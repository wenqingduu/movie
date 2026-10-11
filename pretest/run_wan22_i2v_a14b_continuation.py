"""Generate ordinary Wan2.2-I2V-A14B with audited, own-tail dependencies.

The smoke output is separate from the formal rollout. Resume requires matching
input, parent video, implementation, model and configuration fingerprints.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

from prepare_entitybench_a14b_continuation import OUTPUT, PROJECT, sha256, write_json

WAN = Path("/root/autodl-tmp/Wan2.2")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def reusable(record, expected, directory):
    if record.get("status") not in {"generated", "resumed_verified"}:
        return False
    if record.get("dependency_fingerprint") != fingerprint(expected):
        return False
    for field, filename in [("video_sha256", "video.mp4"), ("last_frame_sha256", "last.png"),
                            ("first_frame_sha256", "first.png")]:
        path = directory / filename
        if not path.is_file() or sha256(path) != record.get(field):
            return False
    return True


def decode_video(video_path, first_path, last_path, expected_frames, expected_fps):
    import imageio.v2 as imageio
    from PIL import Image
    reader = imageio.get_reader(str(video_path), "ffmpeg")
    count, first, last = 0, None, None
    try:
        metadata = reader.get_meta_data()
        for frame in reader:
            if first is None:
                first = frame
            last = frame
            count += 1
    finally:
        reader.close()
    assert count == expected_frames, f"Expected {expected_frames} frames, decoded {count}"
    assert abs(metadata["fps"] - expected_fps) < 0.01, metadata
    Image.fromarray(first).save(first_path)
    Image.fromarray(last).save(last_path)
    return {"decoded_frames": count, "decoded_fps": metadata["fps"],
            "width": int(first.shape[1]), "height": int(first.shape[0]),
            "last_frame_index": count - 1}


def load_pipeline(manifest):
    import torch
    sys.path.insert(0, str(WAN))
    import wan
    from wan.configs import WAN_CONFIGS
    torch.set_num_threads(8)
    config = WAN_CONFIGS["i2v-A14B"]
    assert config.sample_fps == manifest["video_config"]["fps"]
    logging.info("Loading ordinary I2V-A14B; CPU T5 and expert offload; no LoRA")
    return wan.WanI2V(config=config, checkpoint_dir=manifest["model_path"],
                      device_id=0, rank=0, t5_cpu=True, init_on_cpu=True,
                      convert_model_dtype=True)


def run(args):
    import torch
    from PIL import Image
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", stream=sys.stdout)
    manifest = json.loads((args.output_root / "manifest.json").read_text())
    config = dict(manifest["video_config"])
    if args.smoke:
        config["sampling_steps"] = 6
        episodes = [manifest["episodes"][0]]
        selected = {"4:1", "4:2"}
        root = args.output_root / "smoke"
    else:
        episodes, selected, root = manifest["episodes"], None, args.output_root
        assert not manifest["totals"]["firstframe_source_counts"].get("missing_multi_hardcut"), "Prepare all hard cuts first"
    implementation = {str(path): sha256(path) for path in [Path(__file__).resolve(),
        WAN / "wan/image2video.py", WAN / "wan/modules/model.py", WAN / "wan/modules/attention.py",
        WAN / "wan/__init__.py", WAN / "wan/configs/wan_i2v_A14B.py", WAN / "wan/configs/shared_config.py"]}
    model_status = json.loads((PROJECT / "outputs/model_downloads/wan22_i2v_a14b/status.json").read_text())
    assert model_status["state"] == "complete"
    assert model_status["revision"] == manifest["model_revision"]
    model_inventory = json.loads((PROJECT / "outputs/model_downloads/wan22_i2v_a14b/manifest.json").read_text())
    for row in model_inventory["files"]:
        path = Path(manifest["model_path"]) / row["path"]
        assert path.is_file() and path.stat().st_size == row["size"], path
    runtime = {"model_revision": manifest["model_revision"], "model_inventory_hash": fingerprint(model_inventory),
               "wan_commit": subprocess.check_output(["git", "-C", str(WAN), "rev-parse", "HEAD"], text=True).strip(),
               "implementation_sha256": implementation, "config": config, "torch": torch.__version__,
               "cuda": torch.version.cuda, "attention": "existing PyTorch SDPA adapter",
               "mode": "smoke_not_for_scoring" if args.smoke else "formal",
               "prompt_policy": "original_script_prompt_verbatim", "vlm_enabled": False}
    write_json(root / "generation_manifest.json", runtime)
    report = {"kind": "a14b_own_tail_rollout", "mode": runtime["mode"], "jobs": [], "status": "loading_model"}
    write_json(root / "generation_report.json", report)
    pipeline = load_pipeline(manifest)
    sys.path.insert(0, str(WAN))
    from wan.utils.utils import save_video
    started = time.perf_counter()
    for episode in episodes:
        assert sha256(episode["episode_path"]) == episode["episode_sha256"]
        parent = None
        for shot in episode["shots"]:
            if selected and shot["shot_key"] not in selected:
                continue
            name = "shot_" + shot["shot_key"].replace(":", "_")
            directory = root / "videos" / episode["episode_id"] / name
            directory.mkdir(parents=True, exist_ok=True)
            input_path = root / "inputs" / episode["episode_id"] / name / "input.png"
            input_path.parent.mkdir(parents=True, exist_ok=True)
            record = {"episode_id": episode["episode_id"], "shot_key": shot["shot_key"],
                      "shot_index": shot["shot_index"], "seed": shot["seed"], "cut": shot["cut"],
                      "entity_schedule": shot["entity_schedule"], "prompt": shot["prompt"],
                      "input_image": str(input_path), "output_video": str(directory / "video.mp4")}
            if not shot["cut"] and (parent is None or parent["shot_key"] != shot["previous_shot_key"]):
                record.update(status="dependency_blocked", reason="Previous shot unavailable in this route")
                write_json(directory / "generation.json", record)
                report["jobs"].append(record)
                write_json(root / "generation_report.json", report)
                parent = None
                continue
            try:
                if shot["cut"]:
                    source = shot["firstframe_source"]
                    source_path = Path(source["input_image"])
                    assert sha256(source_path) == source["input_sha256"], "Changed frozen hard-cut input"
                    assert sha256(source["source_config"]) == source["source_config_sha256"]
                    record.update(input_source_type=source["source_kind"], source_input=str(source_path),
                                  parent_video_sha256=None, previous_shot_key=None)
                else:
                    source_path = Path(parent["last_frame"])
                    assert sha256(source_path) == parent["last_frame_sha256"]
                    record.update(input_source_type="own_previous_video_last_frame", source_input=str(source_path),
                                  parent_video_sha256=parent["video_sha256"],
                                  previous_shot_key=parent["shot_key"],
                                  parent_last_frame_index=parent["last_frame_index"])
                shutil.copy2(source_path, input_path)
                record["input_sha256"] = sha256(input_path)
                expected = {"runtime": runtime, "script_sha256": episode["episode_sha256"],
                            "shot_key": shot["shot_key"], "seed": shot["seed"], "prompt": shot["prompt"],
                            "input_sha256": record["input_sha256"],
                            "parent_video_sha256": record["parent_video_sha256"]}
                record["dependency_fingerprint"] = fingerprint(expected)
                previous_record = directory / "generation.json"
                existing = json.loads(previous_record.read_text()) if previous_record.is_file() else {}
                if reusable(existing, expected, directory):
                    record = dict(existing, status="resumed_verified")
                    logging.info("Verified existing %s %s", episode["run"], shot["shot_key"])
                else:
                    if shutil.disk_usage(root).free < 2 * 1024**3:
                        raise RuntimeError("Less than 2 GiB free; cannot continue safely")
                    begin = time.perf_counter()
                    torch.cuda.reset_peak_memory_stats()
                    logging.info("Generating %s %s (%s)", episode["run"], shot["shot_key"], record["input_source_type"])
                    image = Image.open(input_path).convert("RGB")
                    video = pipeline.generate(shot["prompt"], img=image, max_area=config["max_area"],
                        frame_num=config["frame_num"], shift=config["shift"], sample_solver=config["solver"],
                        sampling_steps=config["sampling_steps"], guide_scale=tuple(config["guide_scale"]),
                        seed=shot["seed"], offload_model=True)
                    temporary = directory / "video.tmp.mp4"
                    save_video(tensor=video[None], save_file=str(temporary), fps=config["fps"], nrow=1,
                               normalize=True, value_range=(-1, 1))
                    del video
                    gc.collect()
                    torch.cuda.empty_cache()
                    assert temporary.is_file() and temporary.stat().st_size
                    details = decode_video(temporary, directory / "first.png", directory / "last.png",
                                           config["frame_num"], config["fps"])
                    temporary.replace(directory / "video.mp4")
                    record.update(details, status="generated", elapsed_seconds=time.perf_counter() - begin,
                                  cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                  cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
                    record.update(video_sha256=sha256(directory / "video.mp4"),
                                  first_frame_sha256=sha256(directory / "first.png"),
                                  last_frame_sha256=sha256(directory / "last.png"),
                                  last_frame=str(directory / "last.png"))
                    write_json(directory / "generation.json", record)
                    logging.info("Completed %s %s in %.1fs", episode["run"], shot["shot_key"], record["elapsed_seconds"])
                parent = record
                if not args.smoke:
                    scene, local = map(int, shot["shot_key"].split(":"))
                    link = root / "official_results" / episode["episode_id"] / f"{scene:03d}_{local:03d}.mp4"
                    link.parent.mkdir(parents=True, exist_ok=True)
                    if link.is_symlink():
                        link.unlink()
                    assert not link.exists(), f"Refusing to replace ordinary file: {link}"
                    link.symlink_to(directory / "video.mp4")
            except Exception as error:
                logging.exception("Shot failed: %s %s", episode["run"], shot["shot_key"])
                record.update(status="failed", error=repr(error))
                write_json(directory / "generation.json", record)
                parent = None
                report["jobs"].append(record)
                report.update(status="failed", elapsed_seconds=time.perf_counter() - started)
                write_json(root / "generation_report.json", report)
                # Resource/runtime failures need diagnosis; avoid hundreds of repeated failures.
                raise
            report["jobs"].append(record)
            report.update(status="running", elapsed_seconds=time.perf_counter() - started)
            write_json(root / "generation_report.json", report)
    target = 2 if args.smoke else 363
    assert len(report["jobs"]) == target
    assert all(row["status"] in {"generated", "resumed_verified"} for row in report["jobs"])
    report.update(status="complete", elapsed_seconds=time.perf_counter() - started)
    write_json(root / "generation_report.json", report)
    logging.info("Finished %s: %d verified videos", runtime["mode"], target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
