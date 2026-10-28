#!/usr/bin/env python3
"""Generate isolated Qwen140 pilot clips from audited keyframes; never touch the baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from qwen140_pilot_qwen import BASE, OUT, data, sha

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pretest"))
from run_lightx2v_a14b_4step_qwen140 import decode_video

MANIFEST = BASE / "manifest.json"

PROMPTS = {
    "run714": (
        "Photorealistic live-action, clean white photography studio. Leo is the adult blond man in a navy shirt "
        "holding a DSLR camera with both hands. He calmly raises the camera from chest level to his eye, "
        "aims it forward and takes a photograph. One man only, natural detailed movement, steady medium close-up, "
        "realistic hands, consistent face and clothing."
    ),
    "run714_identity_anchor_v2": (
        "Photorealistic live-action in the plain white studio. Leo is the SAME blond man from the episode's "
        "earlier outdoor shots: side-parted blond hair, consistent face shape, and the same black jacket with "
        "white sleeve stripes and light gray hood over a dark top. He holds "
        "a DSLR with both hands. In a medium close-up, he calmly raises the camera from chest level to his eye "
        "and looks through the viewfinder. One man only, no Clara, natural detailed movement, steady lighting, "
        "realistic hands and face, no identity change."
    ),
    "run714_identity_anchor_slow_v3": (
        "Photorealistic live-action in the plain white studio. Leo is the SAME blond man from the episode's "
        "earlier outdoor shots: side-parted blond hair, consistent face shape, and the same black jacket with "
        "white sleeve stripes and light gray hood over a dark top. He holds a DSLR at chest height with both hands. "
        "For the first two seconds his face stays clearly visible while he looks down at the camera and adjusts "
        "his grip. Then he slowly raises the DSLR to his eye and looks through the viewfinder. One continuous "
        "shot, one man only, no Clara, natural deliberate hand and arm movement, consistent face and clothing."
    ),
    "run298": (
        "Photorealistic live-action inside The Gilded Stage auditorium. Julian is the dark-haired man at left in "
        "a white suit; Simon is the blond man at right in a white button-down shirt. The two men stand behind "
        "the sound mixing desk and converse. Julian turns his head toward Simon as he speaks; Simon nods, "
        "then both glance toward the empty dark red auditorium chairs. Exactly two men, no singer, no new person. "
        "Subtle natural gestures, realistic faces and hands, steady chest-up framing."
    ),
    "run298_stage_lit_v2": (
        "Photorealistic live-action inside The Gilded Stage auditorium, with vivid blue and red stage rim lighting "
        "and a soft warm front fill that keeps faces clear. Julian is the dark-haired man at left in a white suit; "
        "Simon is the blond man at right in a white button-down shirt. The two men stand behind the sound mixing "
        "desk and converse. Julian turns toward Simon as he speaks; Simon nods and gestures once toward the "
        "empty dark-red auditorium chairs, then both look out over the chairs. Exactly two men, no singer. "
        "Natural expressive gestures, visible colorful light, steady chest-up framing, realistic faces and hands."
    ),
    "run755": (
        "Photorealistic live-action inside the rear seat of a black London taxi at night. Dante is the bald man "
        "with a goatee on the LEFT, wearing a hooded white forensic garment. Chloe is the long curly-haired woman "
        "with a blonde extension on the RIGHT, wearing a patterned robe. They sit together and watch the small "
        "CAB NEWS screen located outside this camera framing. Screen light flickers across their faces. Dante "
        "looks shocked and gives a slight head shake; Chloe looks concerned and turns her eyes toward Dante. "
        "Exactly two rear-seat passengers, both visible throughout, no broadcast screen filling the frame, "
        "no extra people, steady medium two-shot, natural skin and cinematic low-light detail."
    ),
    "run1404": (
        "Photorealistic live-action medium-wide shot inside the Red Lotus commercial restaurant kitchen. "
        "Mei, the woman in a pink apron, cooks at the flaming wok on the LEFT; a background chef in white "
        "works at a prep station. Kenji, the short black-haired man in a dark charcoal suit, and Ryan, the blond "
        "man in a shiny green bomber jacket with red and white letters, rush in together from the RIGHT kitchen "
        "doorway and look around urgently. The camera tracks their entrance while keeping Mei and the wok "
        "visible. Every person, including the background chef, remains a photorealistic human; no anime, "
        "no illustrated face, no cel shading. Consistent lighting, natural movement and realistic hands."
    ),
    "run484_transition": (
        "Photorealistic live-action inside the same townhouse living room. Nora, the brunette woman in a green "
        "blouse, stays seated on the patterned sofa. The camera pans toward the nearby doorway. Chloe, an adult "
        "black-haired woman in a white blazer, and Leo, a young boy in a yellow jacket over a red polo, enter "
        "through the doorway together and STOP just inside the entrance, still away from the sofa. Exactly three "
        "people: Nora, Chloe and the boy Leo. No extra woman, no duplicates. Smooth continuous entrance only."
    ),
    "run484_main": (
        "Photorealistic live-action in the townhouse living room. Chloe is the adult black-haired woman in the "
        "white blazer, Leo is the young boy in a yellow jacket over a red polo, and Nora is the brunette woman "
        "in green seated on the patterned sofa. Starting near the doorway, Chloe and Leo walk together toward "
        "Nora and stop close to the sofa. Nora watches them. Exactly three people, no duplicates, natural walking "
        "motion, consistent faces and clothing, steady camera."
    ),
}


def fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def load_pipeline(task: str, frames: int):
    import torch

    sys.path.insert(0, "/root/autodl-tmp/LightX2V")
    from lightx2v import LightX2VPipeline

    manifest = json.loads(MANIFEST.read_text())
    torch.set_num_threads(8)
    original_empty = torch.empty

    def pageable_empty(*args, **kwargs):
        if kwargs.get("pin_memory"):
            kwargs["pin_memory"] = False
        return original_empty(*args, **kwargs)

    torch.empty = pageable_empty
    pipe = LightX2VPipeline(model_path=manifest["model_path"], model_cls="wan2.2_moe", task=task, distill_method="dmd2")
    pipe.enable_offload(cpu_offload=True, offload_granularity="model", text_encoder_offload=True,
                        image_encoder_offload=False, vae_offload=False)
    pipe.startup_config.update(rms_norm_type="torch", t5_rms_norm_type="torch", layer_norm_type="torch",
                               clean_cuda_cache=False, use_image_encoder=False, fps=16)
    pipe.startup_config["trial_cpu_pinned_memory"] = False
    inventory = manifest["generation_backend"]["lora_inventory"]
    loras = []
    for row in inventory["files"]:
        path = Path(inventory["destination"]) / row["path"]
        assert sha(path) == row["sha256"]
        loras.append(dict(name="high_noise_model" if "high_noise" in row["path"] else "low_noise_model",
                          path=str(path), strength=1.0))
    pipe.enable_lora(loras, lora_dynamic_apply=False)
    pipe.create_generator(attn_mode="torch_sdpa", infer_steps=4, num_frames=frames, size=(480, 832),
                          guidance_scale=1.0, sample_shift=5.0, fps=16, modulate_type="torch")
    assert not pipe.runner.config["enable_cfg"]
    assert list(pipe.runner.config["denoising_step_list"]) == [1000, 750, 500, 250]
    return pipe, manifest


def generate(pipe, run: str, segment: str, input_path: Path, frames: int, last_frame_path: Path | None = None) -> Path:
    import torch

    ep, shot, _, parent = data(run)
    dest = OUT / "videos" / run / segment
    dest.mkdir(parents=True, exist_ok=True)
    video = dest / "video.mp4"
    record = dest / "generation.json"
    seed = int(shot["seed"]) + (2000000 if segment == "transition" else 3000000)
    prompt_key = ("run714_identity_anchor_slow_v3" if run == "run714" and segment == "main_identity_anchor_slow_v3"
                  else "run714_identity_anchor_v2" if run == "run714" and segment == "main_identity_anchor_v2"
                  else "run298_stage_lit_v2" if run == "run298" and segment == "main_stage_lit_v2"
                  else run if run != "run484" else f"run484_{segment}")
    condition = {"episode_id": ep["episode_id"], "shot_key": shot["shot_key"],
                 "parent_video_sha256": parent["video_sha256"], "parent_tail_sha256": parent["last_frame_sha256"],
                 "input_path": str(input_path.resolve()), "input_sha256": sha(input_path),
                 "last_frame_path": str(last_frame_path.resolve()) if last_frame_path else None,
                 "last_frame_sha256": sha(last_frame_path) if last_frame_path else None,
                 "prompt": PROMPTS[prompt_key], "seed": seed, "task": "flf2v" if last_frame_path else "i2v",
                 "frames": frames, "fps": 16, "size": [480, 832], "steps": 4,
                 "baseline_model_revision": json.loads(MANIFEST.read_text())["model_revision"]}
    if record.exists():
        old = json.loads(record.read_text())
        assert old["fingerprint"] == fingerprint(condition)
        if old["status"] == "generated" and video.is_file() and sha(video) == old["video_sha256"]:
            print("verified", run, segment, video, flush=True)
            return video
        raise RuntimeError(f"Existing incomplete or modified candidate at {dest}")
    if shutil.disk_usage(OUT).free < 10 * 1024**3:
        raise RuntimeError("Less than 10 GiB free")
    temp = dest / "video.tmp.mp4"
    assert not temp.exists() and not video.exists()
    t0 = time.monotonic()
    kwargs = {"seed": seed, "prompt": PROMPTS[prompt_key], "image_path": str(input_path),
              "save_result_path": str(temp)}
    if last_frame_path:
        kwargs["last_frame_path"] = str(last_frame_path)
    print("generating", run, segment, condition["task"], flush=True)
    torch.cuda.reset_peak_memory_stats()
    pipe.generate(**kwargs)
    torch.cuda.synchronize()
    assert temp.is_file() and temp.stat().st_size
    decoded = decode_video(temp, dest / "first.png", dest / "last.png", frames, 16)
    temp.replace(video)
    row = {"status": "generated", "fingerprint": fingerprint(condition), "inputs": condition,
           "decoded": decoded, "video_sha256": sha(video), "first_sha256": sha(dest / "first.png"),
           "last_sha256": sha(dest / "last.png"), "elapsed_seconds": time.monotonic() - t0,
           "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    record.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
    print("generated", run, segment, video, "seconds", round(row["elapsed_seconds"], 1), flush=True)
    return video


def concat_run484() -> Path:
    first = OUT / "videos/run484/transition/video.mp4"
    second = OUT / "videos/run484/main/video.mp4"
    assert first.is_file() and second.is_file()
    dest = OUT / "videos/run484/combined"
    dest.mkdir(parents=True, exist_ok=True)
    video = dest / "video.mp4"
    record = dest / "generation.json"
    inputs = {"first_video": str(first), "first_sha256": sha(first),
              "second_video": str(second), "second_sha256": sha(second),
              "join": "41 transition frames + main frames 1..40", "fps": 16}
    if record.exists():
        old = json.loads(record.read_text())
        assert old["fingerprint"] == fingerprint(inputs) and sha(video) == old["video_sha256"]
        return video
    assert not video.exists()
    temp = dest / "video.tmp.mp4"
    if not temp.exists():
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(first), "-i", str(second),
                        "-filter_complex", "[0:v]setpts=PTS-STARTPTS[v0];[1:v]trim=start_frame=1,setpts=PTS-STARTPTS[v1];[v0][v1]concat=n=2:v=1:a=0[out]",
                        "-map", "[out]", "-an", "-r", "16", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p",
                        str(temp)], check=True)
    decoded = decode_video(temp, dest / "first.png", dest / "last.png", 81, 16)
    temp.replace(video)
    row = {"status": "generated", "fingerprint": fingerprint(inputs), "inputs": inputs,
           "decoded": decoded, "video_sha256": sha(video),
           "first_sha256": sha(dest / "first.png"), "last_sha256": sha(dest / "last.png")}
    record.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
    print("generated run484 combined", video, flush=True)
    return video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", choices=["run714", "run298", "run484", "run755", "run1404"], required=True)
    parser.add_argument("--segment", choices=["main", "main_stage_lit_v2", "main_identity_anchor_v2", "main_identity_anchor_slow_v3", "transition", "combine"], default="main")
    parser.add_argument("--target-variant", default="doorway_v2")
    args = parser.parse_args()
    if args.segment == "combine":
        assert args.runs == ["run484"]
        concat_run484()
        return
    assert args.segment == "main" or (args.segment == "main_stage_lit_v2" and args.runs == ["run298"]) or (args.segment in {"main_identity_anchor_v2", "main_identity_anchor_slow_v3"} and args.runs == ["run714"]) or (args.segment == "transition" and args.runs == ["run484"])
    frames = 41 if "run484" in args.runs else 81
    pipe, _ = load_pipeline("flf2v" if args.segment == "transition" else "i2v", frames)
    for run in args.runs:
        if args.segment == "transition":
            _, _, _, parent = data(run)
            start, end = Path(parent["last_frame"]), OUT / "keyframes/run484" / f"frame_{args.target_variant}.png"
        else:
            start = (OUT / "videos/run484/transition/last.png" if run == "run484"
                     else OUT / "keyframes" / run / ("frame_stage_lit_v2.png" if args.segment == "main_stage_lit_v2"
                                                     else "frame_identity_anchor_v2.png" if args.segment in {"main_identity_anchor_v2", "main_identity_anchor_slow_v3"}
                                                      else "frame_style_clean_v2.png" if run == "run1404" else "frame.png"))
            end = None
        assert start.is_file() and (end is None or end.is_file())
        generate(pipe, run, args.segment, start, frames, end)


if __name__ == "__main__":
    main()
