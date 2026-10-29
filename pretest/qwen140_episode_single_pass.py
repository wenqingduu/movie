#!/usr/bin/env python3
"""Whole-episode single-pass rerun. Stages are explicit so problems stop for the user."""
from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("QWEN140_INTEGRATED_OUTPUT_ROOT", str(
    ROOT / "outputs/entitybench_qwen140_run700_v7_restored_20261028"))

import qwen140_integrated_episodes as core
from qwen140_shot_agent import compile_prompts, frame_inputs, plan_shot
from qwen140_single_pass_shot import decode
from qwen140_wan_api import generate_video


def load(run: str, key: str | None):
    manifest, _, _, jobs = core.source()
    ep = core.episode_for(run, manifest)
    directory = core.OUT / "episodes" / ep["episode_id"]
    if key is None:
        return ep, directory, jobs
    index = next(i for i, shot in enumerate(ep["shots"]) if shot["shot_key"] == key)
    shot = ep["shots"][index]
    folder = directory / f"shot_{key.replace(':', '_')}"
    parent = None
    if index:
        parent = json.loads((directory / f"shot_{ep['shots'][index-1]['shot_key'].replace(':', '_')}"
                             / "result.json").read_text())
        assert core.sha(Path(parent["video"])) == parent["video_sha256"]
        assert core.sha(Path(parent["last_frame"])) == parent["last_sha256"]
    return ep, shot, folder, parent, jobs, index


def prepare(run: str):
    if not (core.OUT / "face_injection_policy.json").exists():
        raise RuntimeError("Explicit face_injection_policy.json required; native-only historical trial is retired")
    ep, directory, _ = load(run, None)
    style = core.style_for(run)
    desc = core.descriptions(ep)
    names = sorted({n for s in ep["shots"] for n in s["entity_schedule"].get("characters", [])})
    policy = json.loads((core.OUT / "face_injection_policy.json").read_text())
    assert policy["mode"] in {"original_full140_single_person_hardcut_v7", "all_generated_keyframes_v7"}
    items = []
    for name in names:
        source = policy["references"][name]
        original = Path(source["path"])
        assert core.sha(original) == source["sha256"]
        dest = core.OUT / "references" / run / f"{name.lower()}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.symlink_to(original.resolve())
        assert core.sha(dest) == source["sha256"]
        items.append({"name": name, "description": desc[name], "reference": str(dest), **source})
    core.write_once(directory / "reference_manifest.json", {"episode_id": ep["episode_id"], "items": items})
    core.write_once(core.OUT / "rerun_manifest.json", {
        "episode_id": ep["episode_id"], "run": run, "shots": ep["shots"],
        "baseline_manifest_sha256": core.sha(core.BASE / "manifest.json"),
        "character_reference_policy": policy.get("character_reference_policy", "frozen clean references; original Stefan 2D/3D identity pair restored; no random regeneration"),
        "identity_scope": policy.get("identity_scope", "Stefan original full140 anchor; other five roles retain the frozen clean-reference trial anchors"),
        "generation_policy": "all hardcuts new; continuations A/B/C from own actual parent tail; single pass; no fallback",
        "face_injection": policy["mode"],
        "face_injection_scope": policy.get("scope", "original single-character hardcuts only"),
        "face_injection_policy_sha256": core.sha(core.OUT / "face_injection_policy.json"),
        "transition_model": "wan3.0-video API; first+last frame; 3 seconds; same task polled up to 30 minutes",
        "main_model": "Wan2.2 I2V A14B, LightX2V 4-step, 81 frames at 16fps",
        "duration_scope": "B is 3s transition plus 5.0625s action minus one duplicate frame; other routes 5.0625s",
        "style_sha256": core.sha(core.STYLE_FILE)})
    print(directory, flush=True)


def stage(action: str, run: str, key: str | None):
    if action == "prepare":
        return prepare(run)
    if action == "refs":
        ep, directory, _ = load(run, None)
        for item in json.loads((directory / "reference_manifest.json").read_text())["items"]:
            assert core.sha(Path(item["reference"])) == item["sha256"]
        print("Frozen references verified; no identity regeneration", flush=True)
        return
    if action == "assemble":
        return assemble(run)
    ep, shot, folder, parent, jobs, index = load(run, key)
    folder.mkdir(parents=True, exist_ok=True)
    tail = Path(parent["last_frame"]) if parent and not shot["cut"] else None
    style = core.style_for(run)["medium"]
    if action == "plan":
        desc = core.descriptions(ep)
        catalog = {n: {"description": desc[n], "reference": str(core.OUT / "references" / run / f"{n.lower()}.png")}
                   for n in shot["entity_schedule"].get("characters", [])}
        context = {"episode_id": ep["episode_id"], "shot_key": key, "original_cut": shot["cut"],
                   "previous_shot": ep["shots"][index-1]["action_description"] if index else "",
                   "current_shot": shot["action_description"], "entity_schedule": shot["entity_schedule"],
                   "setting_and_object_descriptions": {n: desc.get(n, n) for group in ("places", "objects")
                                                        for n in shot["entity_schedule"].get(group, [])},
                   "episode_style": style,
                   "instruction": "按原文安排动作与道具，不增加其他人物行动。关键帧只画此刻可见角色；frame_characters必须返回。所有图像都是单幅连续电影画面。"}
        plan = plan_shot(context, tail, catalog, folder / "agent")
        decision = plan["decision"]
        assert not shot["cut"] or decision["route"] == "C", "Hardcut must establish a new frame"
        frame_names = decision["frame_characters"]
        assert isinstance(frame_names, list) and set(frame_names) <= set(catalog)
        assert len(frame_names) == len(set(frame_names))
        if decision["route"] == "C" and tail is None:
            assert set(frame_names) <= {a["name"] for a in plan["assets"]}, "New hardcut lacks a visible role reference"
        missing = [n for group in ("places", "objects") for n in shot["entity_schedule"].get(group, []) if n not in desc]
        if missing:
            core.write_once(folder / "missing_setting_descriptions.json", {"names": missing, "source": "Exact scheduled label used when dataset provides no description; no invented description"})
        selected, _ = frame_inputs(plan)
        core.write_once(folder / "frame_asset_projection.json", {"original_assets": plan["assets"], "visible_assets": selected, "frame_characters": frame_names})
        core.write_once(folder / "compiled_prompts.json", compile_prompts(plan, style))
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        return
    plan = json.loads((folder / "agent/plan.json").read_text())
    decision = plan["decision"]
    route = decision["route"]
    prompts = json.loads((folder / "compiled_prompts.json").read_text())
    if action == "frame":
        assert route in {"B", "C"}, "Route A uses actual parent tail without frame generation"
        images = [{"label": "图1：上一实际尾帧", "path": str(tail), "sha256": core.sha(tail)}] if tail else []
        selected, _ = frame_inputs(plan)
        images += [{"label": f"图{a['image_index']}：{a['name']}", "path": a["reference"], "sha256": a["sha256"]}
                   for a in selected]
        dest = folder / "frame/target.png"
        task = {"episode_id": ep["episode_id"], "shot_key": key,
                "stage": "transition_target" if route == "B" else "new_shot_start",
                "prompt": prompts["frame"], "images": images,
                "seed": int(shot["seed"]) + (0 if shot["cut"] else 17_000_000),
                "steps": 20, "width": 1024, "height": 576,
                "model_revision": "790c92633540aa0cb11d9abf19eb46d861714758"}
        policy_path = core.OUT / "face_injection_policy.json"
        active = core.OUT / "active_face_injection_policy.json"
        if active.exists():
            pointer = json.loads(active.read_text())
            policy_path = Path(pointer["path"])
            assert core.sha(policy_path) == pointer["sha256"]
        policy = json.loads(policy_path.read_text())
        assert policy["mode"] in {"original_full140_single_person_hardcut_v7", "all_generated_keyframes_v7"}
        task["face_injection_policy_sha256"] = core.sha(policy_path)
        names = shot["entity_schedule"].get("characters", [])
        if policy["mode"] == "all_generated_keyframes_v7" and decision["frame_characters"]:
            frame_roles = decision["frame_characters"]
            available = [n for n in frame_roles if n in policy["roles"]]
            if not policy.get("allow_control_reuse", False):
                assert available == frame_roles, "Required 3D asset missing"
            task["unavailable_3d_characters"] = [n for n in frame_roles if n not in available]
            task["face_injection"] = {"run": run,
                "characters": [policy["roles"][n] for n in available],
                "separate_conditioning_images": True,
                "preparation_backend": policy.get("preparation_backend", "original_area_selection"),
                "allow_control_reuse": policy.get("allow_control_reuse", False),
                "settings": policy["settings"], "frozen_files": policy["frozen_files"],
                "official_prompt": shot["prompt"]}
            if not available:
                task.pop("face_injection")
                task["injection_skip_reason"] = "no_usable_3d_asset"
        elif shot["cut"] and len(names) == 1:
            character = policy["roles"][names[0]]
            task["face_injection"] = {"run": run, "character": character,
                "settings": policy["settings"], "frozen_files": policy["frozen_files"],
                "official_prompt": shot["prompt"]}
        else:
            task["injection_skip_reason"] = ("original_policy_multi_character_no_multi_face_injection"
                if len(names) != 1 else "original_policy_continuation_not_new_injection")
        core.write_once(dest.with_suffix(".task.json"), task)
        core.run_worker(core.QWEN_PYTHON, "worker-frame", "--task", str(dest.with_suffix(".task.json")), "--output", str(dest))
    elif action == "transition":
        assert route == "B"
        video = generate_video(tail, folder / "frame/target.png", prompts["transition"], 3, folder / "transition")
        decode(video, folder / "transition")
    elif action == "main":
        start = tail if route == "A" else folder / ("transition/last.png" if route == "B" else "frame/target.png")
        segment = "direct" if route == "A" else "main" if route == "B" else "hardcut" if shot["cut"] else "new"
        media = core.video_spec(ep, shot, segment, start, prompts["shot"], 81, folder / "main")
        core.write_once(folder / "main_media.json", {k: str(v) if isinstance(v, Path) else v for k, v in media.items()})
    elif action == "finish":
        media = json.loads((folder / "main_media.json").read_text())
        if route == "B":
            combined = folder / "combined"
            combined.mkdir(parents=True, exist_ok=True)
            video = combined / "video.mp4"
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-i", str(folder / "transition/video.mp4"),
                            "-i", media["video"], "-filter_complex",
                            "[0:v]fps=16,scale=832:464,setsar=1,setpts=PTS-STARTPTS[a];"
                            "[1:v]trim=start_frame=1,scale=832:464,setsar=1,setpts=PTS-STARTPTS[b];"
                            "[a][b]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-an", "-c:v", "libx264", "-crf", "17",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)], check=True)
            decode(video, combined)
            media = {"video": str(video), "last": str(combined / "last.png"), "sha256": core.sha(video)}
        link = core.link_official(ep, shot, {"video": Path(media["video"])})
        record = {"episode_id": ep["episode_id"], "shot_key": key, "cut": shot["cut"],
                  "source_kind": "single_pass_fresh_hardcut" if shot["cut"] else f"single_pass_route_{route}",
                  "route": decision, "style_sha256": core.sha(core.STYLE_FILE),
                  "video": media["video"], "video_sha256": media["sha256"],
                  "last_frame": media["last"], "last_sha256": core.sha(Path(media["last"])),
                  "parent_video_sha256": parent["video_sha256"] if parent else None,
                  "parent_tail_sha256": parent["last_sha256"] if parent else None,
                  "baseline_video_sha256": jobs[ep["episode_id"], key]["video_sha256"], "official_link": str(link.resolve())}
        frame_record = json.loads((folder / "frame/target.json").read_text()) if route != "A" else {}
        record["injection_applied"] = frame_record.get("injection_applied", False)
        record["injection_result"] = frame_record.get("injection_result")
        record["injection_skip_reason"] = (None if record["injection_applied"] else
            frame_record.get("skip_reason", "direct_continuation_inherits_parent"))
        core.write_once(folder / "result.json", record)
        print("completed", key, route, media["video"], flush=True)


def assemble(run: str):
    ep, directory, jobs = load(run, None)
    records = [json.loads((directory / f"shot_{s['shot_key'].replace(':', '_')}" / "result.json").read_text()) for s in ep["shots"]]
    core.write_once(directory / "episode_result.json", {"episode_id": ep["episode_id"], "run": run,
                    "style_sha256": core.sha(core.STYLE_FILE), "shots": records})
    review = core.OUT / "review"
    review.mkdir(exist_ok=True)
    for label, paths in (("before", [jobs[ep["episode_id"], s["shot_key"]]["output_video"] for s in ep["shots"]]),
                         ("after", [r["video"] for r in records])):
        listing = review / f"{label}.ffconcat"
        listing.write_text("ffconcat version 1.0\n" + "".join(f"file '{p}'\n" for p in paths))
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-an", "-c", "copy", "-movflags", "+faststart", str(review / f"{label}.mp4")], check=True)
    details = ""
    for row in records:
        original = jobs[ep["episode_id"], row["shot_key"]]["output_video"]
        old_url = html.escape(os.path.relpath(original, review), quote=True)
        new_url = html.escape(os.path.relpath(row["video"], review), quote=True)
        details += (f'<tr><td>{row["shot_key"]}</td><td>{row["route"]["route"]}</td>'
                    f'<td><a href="{old_url}">原shot</a></td><td><a href="{new_url}">新shot</a></td></tr>')
    (review / "index.html").write_text(f'<meta charset="utf-8"><title>{run} 全集对比</title>'
        f'<h1>{run}：原结果 / 当前集成版</h1><p>B路线增加过渡时长，非等时长对照。</p>'
        '<p>人物参考按本轮manifest冻结，未在逐shot阶段重生人物身份。</p>'
        '<h2>原结果</h2><video controls width="832" src="before.mp4"></video>'
        '<h2>新结果</h2><video controls width="832" src="after.mp4"></video>'
        '<table><tr><th>shot</th><th>路由</th><th>之前</th><th>之后</th></tr>' + details + '</table>')
    print(review / "index.html", flush=True)


def run_remaining(run: str):
    """Continue the accepted plan once; keep semantic deviations for review."""
    ep, directory, _ = load(run, None)
    reference_count = len(json.loads((directory / "reference_manifest.json").read_text())["items"])
    policy = json.loads((core.OUT / "face_injection_policy.json").read_text())

    def progress(key, action):
        completed = [json.loads(p.read_text()) for p in directory.glob("shot_*/result.json")]
        (core.OUT / "progress.json").write_text(json.dumps({
            "status": "running", "active_shot": key, "active_stage": action,
            "completed_shots": len(completed), "total_shots": len(ep["shots"]),
            "references_frozen": reference_count,
            "references_generated_this_run": policy.get("reference_generation_count", 0),
            "routes_completed": {r["shot_key"]: r["route"]["route"] for r in completed},
            "policy": "single pass; semantic deviations retained following user instruction; technical errors stop",
        }, ensure_ascii=False, indent=2))
        print("STAGE", key, action, flush=True)

    for shot in ep["shots"]:
        key = shot["shot_key"]
        folder = directory / f"shot_{key.replace(':', '_')}"
        result = folder / "result.json"
        if result.exists():
            row = json.loads(result.read_text())
            assert core.sha(Path(row["video"])) == row["video_sha256"]
            assert core.sha(Path(row["last_frame"])) == row["last_sha256"]
            continue
        if not (folder / "agent/plan.json").exists() or not (folder / "compiled_prompts.json").exists():
            progress(key, "plan")
            stage("plan", run, key)
        route = json.loads((folder / "agent/plan.json").read_text())["decision"]["route"]
        if route in {"B", "C"}:
            frame = folder / "frame/target.png"
            if frame.exists():
                record = json.loads(frame.with_suffix(".json").read_text())
                task = json.loads(frame.with_suffix(".task.json").read_text())
                assert core.sha(frame) == record["sha256"]
                assert core.digest(task) == record["condition_sha256"]
            else:
                progress(key, "frame")
                stage("frame", run, key)
        if route == "B":
            progress(key, "transition")
            stage("transition", run, key)
        progress(key, "main")
        stage("main", run, key)
        progress(key, "finish")
        stage("finish", run, key)
    progress(None, "assemble")
    stage("assemble", run, None)
    progress(None, "complete")
    p = core.OUT / "progress.json"
    row = json.loads(p.read_text())
    row.update(status="generation_complete_pending_visual_review_and_evaluation")
    p.write_text(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "refs", "plan", "frame", "transition", "main", "finish", "assemble", "run-remaining"])
    parser.add_argument("--run", default="run700")
    parser.add_argument("--shot")
    args = parser.parse_args()
    try:
        if args.action == "run-remaining":
            run_remaining(args.run)
        else:
            stage(args.action, args.run, args.shot)
    except Exception as error:
        dest = core.OUT / "stopped.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({"stage": args.action, "shot": args.shot, "error": str(error),
                                   "policy": "stopped for user; no automatic retry or fallback"}, ensure_ascii=False, indent=2))
        raise
