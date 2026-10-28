#!/usr/bin/env python3
"""One planned B-route sample: lookup -> target -> Wan3 transition -> Wan action.

Each generation runs once. There is no visual acceptance gate, repair, or fallback.
Separate stages let the operator stop and report a problem without regenerating.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("QWEN140_INTEGRATED_OUTPUT_ROOT", str(
    ROOT / "outputs/entitybench_qwen140_integrated_episodes_v2_20261026"))

import qwen140_integrated_episodes as legacy
from qwen140_shot_agent import compile_prompts, plan_shot, validate_plan
from qwen140_wan_api import generate_video
from run_lightx2v_a14b_4step_qwen140 import decode_video


def inputs(run: str, shot_key: str) -> tuple:
    manifest, prep, old, _ = legacy.source()
    ep = legacy.episode_for(run, manifest)
    i = next(i for i, s in enumerate(ep["shots"]) if s["shot_key"] == shot_key)
    previous, shot = ep["shots"][i - 1], ep["shots"][i]
    parent_path = legacy.OUT / "episodes" / ep["episode_id"] / f"shot_{previous['shot_key'].replace(':', '_')}" / "result.json"
    parent = json.loads(parent_path.read_text())
    tail = Path(parent["last_frame"])
    if legacy.sha(tail) != parent["last_sha256"]:
        raise ValueError("Actual parent tail changed")
    descriptions = legacy.descriptions(ep)
    catalog = {}
    for name in shot["entity_schedule"].get("characters", []):
        ref = legacy.role_ref(run, ep, name, prep, old)
        origin = "episode_character_asset"
        if (run, name) == ("run700", "Leo"):
            ref = legacy.OUT / "episodes" / ep["episode_id"] / "shot_1_2/start_attempt0.png"
            origin = "existing_solo_Leo_frame; original reference is a multi-person montage"
        catalog[name] = {"description": descriptions[name], "reference": str(ref) if ref else "",
                         "reference_source": origin}
    return ep, previous, shot, tail, parent, catalog


def decode(path: Path, folder: Path) -> dict:
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=nb_frames,r_frame_rate", "-of", "json", str(path)],
                           capture_output=True, text=True, check=True)
    stream = json.loads(probe.stdout)["streams"][0]
    a, b = stream["r_frame_rate"].split("/")
    info = decode_video(path, folder / "first.png", folder / "last.png", int(stream["nb_frames"]), float(a) / float(b))
    legacy.write_once(folder / "decoded.json", {**info, "video_sha256": legacy.sha(path),
                                                "last_sha256": legacy.sha(folder / "last.png")})
    return info


def execute(action: str, run: str, shot_key: str, folder: Path) -> None:
    ep, previous, shot, tail, parent, catalog = inputs(run, shot_key)
    folder.mkdir(parents=True, exist_ok=True)
    style = legacy.style_for(run)["medium"]
    if action == "plan":
        context = {"previous_shot": previous["action_description"], "current_shot": shot["action_description"],
                   "scheduled_characters": list(catalog), "episode_style": style,
                   "trial_requirement": "本例路线B；根据上一尾帧的可见角色决定需要哪些新增人物素材。过渡中Leo从右侧进入，Stefan提起藤篮，镜头拉宽露出关闭的柜子；第二段再移动到柜前、开抽屉等。"}
        plan = plan_shot(context, tail, catalog, folder / "agent")
        if plan["decision"]["route"] != "B":
            raise ValueError("Plan differs from the requested B sample; stop for user decision")
        legacy.write_once(folder / "compiled_prompts.json", compile_prompts(plan, style))
        legacy.write_once(folder / "input_manifest.json", {"episode_id": ep["episode_id"], "shot_key": shot_key,
                          "previous_tail": str(tail), "parent_video_sha256": parent["video_sha256"],
                          "tail_sha256": legacy.sha(tail), "transition_seconds": 3, "main_frames": 81,
                          "duration_scope": "exploratory extended-duration sample, not a 5s-budget comparison"})
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        return
    plan = json.loads((folder / "agent/plan.json").read_text())
    validate_plan(plan["decision"])
    prompts = json.loads((folder / "compiled_prompts.json").read_text())
    if action == "frame":
        dest = folder / "frame/target.png"
        images = [{"label": "图1：上一实际尾帧", "path": str(tail), "sha256": legacy.sha(tail)}]
        images += [{"label": f"图{a['image_index']}：{a['name']}", "path": a["reference"], "sha256": a["sha256"]}
                   for a in plan["assets"]]
        task = {"episode_id": ep["episode_id"], "shot_key": shot_key, "stage": "single_pass_target",
                "prompt": prompts["frame"], "images": images, "seed": int(shot["seed"]) + 14_000_000,
                "steps": 20, "width": 1024, "height": 576,
                "model_revision": "790c92633540aa0cb11d9abf19eb46d861714758"}
        legacy.write_once(dest.with_suffix(".task.json"), task)
        if not dest.exists():
            legacy.run_worker(legacy.QWEN_PYTHON, "worker-frame", "--task", str(dest.with_suffix(".task.json")), "--output", str(dest))
        print(dest, flush=True)
    elif action == "transition":
        video = generate_video(tail, folder / "frame/target.png", prompts["transition"], 3, folder / "transition")
        decode(video, folder / "transition")
        print(video, flush=True)
    elif action == "main":
        actual_tail = folder / "transition/last.png"
        media = legacy.video_spec(ep, shot, "main", actual_tail, prompts["shot"], 81, folder / "main")
        print(media["video"], flush=True)
    elif action == "assemble":
        out = folder / "combined.mp4"
        if not out.exists():
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                            "-i", str(folder / "transition/video.mp4"), "-i", str(folder / "main/video.mp4"),
                            "-filter_complex", "[0:v]fps=16,scale=832:464,setsar=1,setpts=PTS-STARTPTS[a];"
                            "[1:v]trim=start_frame=1,scale=832:464,setsar=1,setpts=PTS-STARTPTS[b];"
                            "[a][b]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-an", "-c:v", "libx264",
                            "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], check=True)
        info = decode(out, folder)
        print(out, info, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["plan", "frame", "transition", "main", "assemble"])
    parser.add_argument("--run", default="run700", choices=list(legacy.TAGS))
    parser.add_argument("--shot", default="3:3")
    parser.add_argument("--output", type=Path, default=legacy.OUT / "experiments/run700_3_3_single_pass_agent_20261028")
    args = parser.parse_args()
    try:
        execute(args.action, args.run, args.shot, args.output.resolve())
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "stopped.json").write_text(json.dumps({"stage": args.action, "error": str(error),
                                                           "decision": "stopped_for_user; no retry or fallback"}, ensure_ascii=False, indent=2))
        raise
