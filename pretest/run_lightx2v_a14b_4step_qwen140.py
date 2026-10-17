"""Generate full-cohort LightX2V 4-step A14B with audited own-tail dependencies.

The smoke output is separate from the formal rollout. Resume requires matching
input, parent video, implementation, model and configuration fingerprints.
"""

from __future__ import annotations

import argparse
import gc
import os
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

from prepare_entitybench_a14b_4step import PROJECT, LIGHT, LIGHT_REVISION, sha256, write_json
from prepare_entitybench_qwen21_full140 import OUTPUT, SOURCE



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
    sys.path.insert(0, str(LIGHT))
    from lightx2v import LightX2VPipeline
    torch.set_num_threads(8)
    original_empty = torch.empty
    def pageable_empty(*args, **kwargs):
        if kwargs.get("pin_memory"):
            kwargs["pin_memory"] = False
        return original_empty(*args, **kwargs)
    torch.empty = pageable_empty
    config = manifest["video_config"]
    assert config["sampling_steps"] == 4 and config["negative_prompt_active"] is False
    loras = manifest["generation_backend"]["lora_inventory"]
    for row in loras["files"]:
        assert sha256(Path(loras["destination"]) / row["path"]) == row["sha256"]
    pipe = LightX2VPipeline(model_path=manifest["model_path"], model_cls="wan2.2_moe", task="i2v", distill_method="dmd2")
    pipe.enable_offload(cpu_offload=True, offload_granularity="model", text_encoder_offload=True,
                        image_encoder_offload=False, vae_offload=False)
    pipe.startup_config.update(rms_norm_type="torch", t5_rms_norm_type="torch", layer_norm_type="torch",
                              clean_cuda_cache=False, use_image_encoder=False, fps=16)
    pipe.startup_config["trial_cpu_pinned_memory"] = False
    pipe.enable_lora([dict(name="high_noise_model" if "high_noise" in row["path"] else "low_noise_model",
                          path=str(Path(loras["destination"]) / row["path"]), strength=1.0) for row in loras["files"]],
                     lora_dynamic_apply=False)
    pipe.create_generator(attn_mode="torch_sdpa", infer_steps=4, num_frames=81, size=(480, 832),
                          guidance_scale=1.0, sample_shift=5.0, fps=16, modulate_type="torch")
    assert pipe.runner.config["enable_cfg"] is False
    assert list(pipe.runner.config["denoising_step_list"]) == [1000, 750, 500, 250]
    assert pipe.runner.config["boundary_step_index"] == 2
    return pipe


def import_episode(root, episode, runtime):
    """Import a whole completed episode without changing its original provenance."""
    source = Path(episode['reuse_generation_root'])
    original_runtime = json.loads((source / 'generation_manifest.json').read_text())
    for key in ['model_revision', 'model_inventory_hash', 'lightx2v_commit', 'lightx2v_source_sha256',
                'lora_inventory', 'config', 'torch', 'cuda', 'attention', 'cpu_pinned_memory', 'resolved_lightx2v_config']:
        assert fingerprint(original_runtime[key]) == fingerprint(runtime[key]), f'Incompatible reused backend: {key}'
    for path, expected in original_runtime['implementation_sha256'].items():
        assert sha256(path) == expected, f'Changed original implementation: {path}'
    report = json.loads((source / 'generation_report.json').read_text())
    assert report['status'] == 'complete'
    original_episodes = {e['episode_id']: e for e in json.loads((source / 'manifest.json').read_text())['episodes']}
    original_ep = original_episodes[episode['episode_id']]
    assert episode['episode_sha256'] == original_ep['episode_sha256']
    assert [(s['shot_key'], s['seed'], s['prompt'], s['cut']) for s in episode['shots']] == [
        (s['shot_key'], s['seed'], s['prompt'], s['cut']) for s in original_ep['shots']]
    rows = {r['shot_key']: r for r in report['jobs'] if r['episode_id'] == episode['episode_id']}
    imported, previous = [], None
    for shot in episode['shots']:
        row = dict(rows[shot['shot_key']])
        assert row['status'] in {'generated', 'resumed_verified'}
        original_directory = Path(row['output_video']).parent
        assert row['decoded_frames'] == 81 and row['decoded_fps'] == 16
        for field, name in [('video_sha256', 'video.mp4'), ('first_frame_sha256', 'first.png'), ('last_frame_sha256', 'last.png')]:
            assert sha256(original_directory / name) == row[field]
        assert sha256(row['input_image']) == row['input_sha256']
        if shot['cut']:
            assert row['input_sha256'] == shot['firstframe_source']['input_sha256']
        else:
            assert previous and row['previous_shot_key'] == previous['shot_key']
            assert row['input_sha256'] == previous['last_frame_sha256']
            assert row['parent_video_sha256'] == previous['video_sha256']
        directory = root / 'videos' / episode['episode_id'] / ('shot_' + shot['shot_key'].replace(':', '_'))
        directory.mkdir(parents=True, exist_ok=True)
        input_path = root / 'inputs' / episode['episode_id'] / directory.name / 'input.png'
        input_path.parent.mkdir(parents=True, exist_ok=True)
        links = [(directory / name, original_directory / name) for name in ['video.mp4', 'first.png', 'last.png']]
        links.append((input_path, Path(row['input_image'])))
        scene, local = map(int, shot['shot_key'].split(':'))
        canonical = root / 'official_results' / episode['episode_id'] / f'{scene:03d}_{local:03d}.mp4'
        canonical.parent.mkdir(parents=True, exist_ok=True)
        links.append((canonical, directory / 'video.mp4'))
        for link, target in links:
            if link.is_symlink():
                assert link.resolve() == target.resolve()
            else:
                assert not link.exists(), f'Refusing to overwrite reused media: {link}'
                link.symlink_to(target)
        row.update(status='resumed_verified', output_video=str(directory / 'video.mp4'),
            last_frame=str(directory / 'last.png'), input_image=str(input_path),
            reuse_generation_origin=str(source), reuse_generation_record=str(original_directory / 'generation.json'),
            reuse_record_sha256=sha256(original_directory / 'generation.json'),
            provenance_note='Original dependency_fingerprint retained; whole-episode backend/input/media compatibility audited')
        write_json(directory / 'generation.json', row)
        imported.append(row)
        previous = row
    return imported


def run(args):
    from audit_entitybench_qwen21_full140 import verify_adapters
    verify_adapters(args.output_root)
    os.environ.update(DTYPE="BF16", SENSITIVE_LAYER_DTYPE="None")
    import torch
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", stream=sys.stdout)
    manifest = json.loads((args.output_root / "manifest.json").read_text())
    config = dict(manifest["video_config"])
    assert not args.smoke, "Full-cohort runner has no separate smoke mode; backend was already validated"
    if args.smoke:
        # Same actual 4-step backend; isolated smoke is never scored formally.
        episodes = [manifest["episodes"][0]]
        selected = {"4:1", "4:2"}
        root = args.output_root / "smoke"
    else:
        episodes, selected, root = manifest["episodes"], None, args.output_root
        assert not manifest["totals"]["firstframe_source_counts"].get("missing_multi_hardcut"), "Prepare all hard cuts first"
    assert subprocess.check_output(["git", "-C", str(LIGHT), "rev-parse", "HEAD"], text=True).strip() == LIGHT_REVISION
    implementation = {str(path): sha256(path) for path in [Path(__file__).resolve(),
        PROJECT / "pretest/prepare_entitybench_a14b_4step.py",
        PROJECT / "pretest/prepare_entitybench_qwen21_full140.py",
        PROJECT / "pretest/run_lightx2v_a14b_4step_continuation.py",
        PROJECT / "pretest/prepare_entitybench_a14b_continuation.py"]}
    light_sources = {str(path.relative_to(LIGHT)): sha256(path)
                     for folder in ["lightx2v", "lightx2v_platform"]
                     for path in (LIGHT / folder).rglob("*.py")}
    assert light_sources, "LightX2V source inventory is empty"
    model_status = json.loads((PROJECT / "outputs/model_downloads/wan22_i2v_a14b/status.json").read_text())
    assert model_status["state"] == "complete"
    assert model_status["revision"] == manifest["model_revision"]
    model_inventory = json.loads((PROJECT / "outputs/model_downloads/wan22_i2v_a14b/manifest.json").read_text())
    for row in model_inventory["files"]:
        path = Path(manifest["model_path"]) / row["path"]
        assert path.is_file() and path.stat().st_size == row["size"], path
    runtime = {"model_revision": manifest["model_revision"], "model_inventory_hash": fingerprint(model_inventory),
               "lightx2v_commit": LIGHT_REVISION, "lightx2v_source_sha256": light_sources,
               "lora_inventory": manifest["generation_backend"]["lora_inventory"],
               "implementation_sha256": implementation, "config": config, "torch": torch.__version__,
               "cuda": torch.version.cuda, "attention": "torch_sdpa", "cpu_pinned_memory": False,
               "mode": "smoke_not_for_scoring" if args.smoke else "formal",
               "prompt_policy": "original_script_prompt_verbatim", "vlm_enabled": False}
    report = {"kind": "lightx2v_4step_a14b_own_tail_rollout", "mode": runtime["mode"], "jobs": [], "status": "loading_model"}
    write_json(root / "generation_report.json", report)
    load_begin = time.perf_counter()
    pipeline = load_pipeline(manifest)
    report["model_load_seconds"] = time.perf_counter() - load_begin
    runtime["resolved_lightx2v_config"] = dict(pipeline.runner.config)
    from prepare_entitybench_qwen21_20ep import immutable
    immutable(root / "generation_manifest.json", runtime)
    started = time.perf_counter()
    for episode in episodes:
        assert sha256(episode["episode_path"]) == episode["episode_sha256"]
        if episode.get("reuse_generation_root"):
            imported = import_episode(root, episode, runtime)
            report["jobs"].extend(imported)
            report.update(status="running", elapsed_seconds=time.perf_counter() - started)
            write_json(root / "generation_report.json", report)
            logging.info("Imported verified completed episode %s (%d videos)", episode["episode_id"], len(imported))
            continue
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
                    if shutil.disk_usage(root).free < 10 * 1024**3:
                        raise RuntimeError("Less than 10 GiB free; cannot continue safely")
                    begin = time.perf_counter()
                    torch.cuda.reset_peak_memory_stats()
                    logging.info("Generating %s %s (%s)", episode["run"], shot["shot_key"], record["input_source_type"])
                    temporary = directory / "video.tmp.mp4"
                    torch.cuda.synchronize()
                    pipeline.generate(seed=shot["seed"], prompt=shot["prompt"], image_path=str(input_path),
                                      save_result_path=str(temporary))
                    torch.cuda.synchronize()
                    # CFG is disabled in this distilled route; no negative_prompt field is sent.
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
    target = sum(len(e["shots"]) for e in episodes)
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
