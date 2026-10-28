#!/usr/bin/env python3
"""One paired run700 4:1 trial using the existing multi-role v7 path."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess

import qwen140_integrated_episodes as core

ROOT = core.ROOT
SOURCE = ROOT / "outputs/entitybench_qwen140_run700_v7_restored_20261028"
OUT = ROOT / "outputs/entitybench_qwen140_run700_multiface_v7_20261028"
EID = "00050e2d-6c9a-38e6-b702-87f8df189087__run700__i25_j50__T300"
SHOT = OUT / "shot_4_1"
ORIGINAL = SOURCE / "episodes" / EID / "shot_4_1"
ASSETS = OUT / "assets" / EID
ASSET_PYTHON = str(ROOT / ".venv/bin/python")


def load(path):
    return json.loads(Path(path).read_text())


def env():
    result = dict(os.environ)
    result.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
                  PYTHONPATH=str(ROOT), MULTISHOT_ALLOW_FACELIFT_FALLBACK="0",
                  MULTISHOT_DISABLE_FACELIFT="0", MULTISHOT_FACELIFT_STEP_2D="50",
                  HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if (OUT / "runtime/sitecustomize.py").exists():
        result.update(PYTHONPATH=str(OUT / "runtime") + os.pathsep + str(ROOT),
                      ENTITYBENCH_ORT_THREADS="2")
    return result


def command(label, args, environment=None):
    path = OUT / "logs" / f"{label}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    print(label, flush=True)
    with path.open("a") as stream:
        subprocess.run(args, cwd=ROOT, env=environment or env(),
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def assets():
    from run_entitybench_a14b_full140 import asset_environment

    policy = load(SOURCE / "face_injection_policy.json")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "runtime").mkdir(exist_ok=True)
    asset_env = asset_environment(OUT, env())
    for name in ("Elias", "Walter"):
        reference = policy["references"][name]
        assert core.sha(Path(reference["path"])) == reference["sha256"]
        link = ASSETS / "characters" / f"{name.lower()}_reference.png"
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(reference["path"])
        assert core.sha(link) == reference["sha256"]
        # Frozen existing portraits only; never run the reference-generation stage.
        for stage in ("facelift", "calibrate"):
            command(f"{name.lower()}_{stage}", [ASSET_PYTHON, "-u",
                    "pretest/prepare_entitybench_character_assets.py", "--episode",
                    str(ROOT / "benchmarks/entitybench/data/scripts" / f"{EID}.json"),
                    "--asset-root", str(ASSETS), "--stage", stage,
                    "--character-scope", "scheduled", "--only-character", name,
                    "--base-seed", "30018000", "--calibration-size", "512",
                    "--python", ASSET_PYTHON], asset_env)
    roles = [policy["roles"]["Stefan"]]
    roles += [{"name": name, "reference_image": policy["references"][name]["path"],
               "facelift_result": str(ASSETS / "faces_3d" / name.lower() / "facelift_result.json")}
              for name in ("Elias", "Walter")]
    files = list(policy["frozen_files"])
    for role in roles:
        result = load(role["facelift_result"])
        assert result["facelift_status"] == "success"
        assert result["source_reference_sha256"] == core.sha(Path(role["reference_image"]))
        model = Path(result["model_path"])
        calibration = model.with_suffix(".pose_calibration.json")
        assert load(calibration)["gaussian_model_sha256"] == core.sha(model)
        for path in (Path(role["reference_image"]), Path(role["facelift_result"]), model, calibration):
            files.append({"path": str(path), "sha256": core.sha(path)})
    settings = dict(policy["settings"], multi_face_injection=True)
    core.write_once(OUT / "manifest.json", {
        "episode_id": EID, "shot_key": "4:1", "scope": "one paired multi-role first-frame and video trial",
        "characters": roles, "settings": settings, "frozen_files": files,
        "source_frame_task_sha256": core.sha(ORIGINAL / "frame/target.task.json"),
        "source_video_task_sha256": core.sha(ORIGINAL / "main/task.json"),
        "policy": "per-role existing v7 gates; no new portraits, semantic retries or whole-shot fallback"})


def frame():
    from PIL import Image
    manifest = load(OUT / "manifest.json")
    assert manifest["source_frame_task_sha256"] == core.sha(ORIGINAL / "frame/target.task.json")
    task = load(ORIGINAL / "frame/target.task.json")
    task.pop("injection_skip_reason")
    task.pop("face_injection_policy_sha256")
    task["face_injection"] = {"run": "run700", "characters": manifest["characters"],
                              "settings": manifest["settings"], "frozen_files": manifest["frozen_files"],
                              "official_prompt": load(ORIGINAL / "agent/planning_inputs.json")["context"]["current_shot"]}
    task_path = SHOT / "frame/target.task.json"
    core.write_once(task_path, task)
    target = SHOT / "frame/target.png"
    if not target.exists():
        command("frame", [core.QWEN_PYTHON, "-u", str(ROOT / "pretest/qwen140_integrated_episodes.py"),
                          "worker-frame", "--task", str(task_path), "--output", str(target)])
    result = load(SHOT / "frame/injection/result.json")
    assert result["injection_applied"] and len(result["applied_characters"]) >= 2, result
    assert core.sha(target) == core.sha(Path(result["treatment"]))
    # The original native worker writes RGB; run_shot preserves Qwen's RGBA.
    # Compare decoded pixels and a canonical RGB PNG, retaining the raw original.
    canonical = SHOT / "frame/injection/control_rgb.png"
    Image.open(result["control"]).convert("RGB").save(canonical)
    assert core.sha(canonical) == core.sha(ORIGINAL / "frame/target.png")
    core.write_once(OUT / "control_validation.json", {
        "native_png_sha256": core.sha(ORIGINAL / "frame/target.png"),
        "raw_control_rgba_png_sha256": core.sha(Path(result["control"])),
        "canonical_control_rgb_png_sha256": core.sha(canonical),
        "canonical_png_and_decoded_rgb_equal": True})
    print("actually injected", result["applied_characters"], "skipped", result["skipped_characters"], flush=True)


def videos():
    result = load(SHOT / "frame/injection/result.json")
    assert len(result["applied_characters"]) >= 2
    manifest = load(OUT / "manifest.json")
    assert manifest["source_video_task_sha256"] == core.sha(ORIGINAL / "main/task.json")
    for condition in ("control", "treatment"):
        task = copy.deepcopy(load(ORIGINAL / "main/task.json"))
        image = Path(result[condition])
        task.update(input_path=str(image), input_sha256=core.sha(image))
        directory = SHOT / "videos" / condition
        task_path = directory / "task.json"
        core.write_once(task_path, task)
        if not (directory / "video.mp4").exists():
            command(f"video_{condition}", [core.WAN_PYTHON, "-u", str(ROOT / "pretest/qwen140_integrated_episodes.py"),
                    "worker-video", "--task", str(task_path), "--output", str(directory)],
                    dict(env(), DTYPE="BF16", SENSITIVE_LAYER_DTYPE="None"))
        generated = load(directory / "generation.json")
        assert generated["inputs"] == task
        assert generated["video_sha256"] == core.sha(directory / "video.mp4")
        link = SHOT / "frame/injection/videos" / f"{condition}.mp4"
        link.parent.mkdir(exist_ok=True)
        if not link.exists():
            link.symlink_to(directory / "video.mp4")


def evaluate():
    from types import SimpleNamespace
    from multishot.ip_adapter_experiment_utils import _face_app
    from evaluate_qwen_multiface_pair import run as image_evaluation
    from evaluate_qwen_multiface_videos import run as video_evaluation
    from evaluate_qwen_reference_dino import DinoEncoder, _evaluate_shot
    from render_qwen_evaluation_comparisons import _render_shot

    directory = SHOT / "frame/injection"
    app = _face_app()
    common = {"prepared": directory / "prepared/prepared.json", "min_face_confidence": 0.5}
    injection = load(directory / "result.json")
    for condition in ("control", "treatment"):
        assert core.sha(directory / f"{condition}.png") == injection[f"{condition}_sha256"]
    if (directory / "identity_metrics.json").exists():
        image_scores = load(directory / "identity_metrics.json")
        for condition in ("control", "treatment"):
            assert Path(image_scores[condition]).resolve() == (directory / f"{condition}.png").resolve()
    else:
        image_scores = image_evaluation(SimpleNamespace(**common, control=directory / "control.png",
            treatment=directory / "treatment.png", output=directory / "identity_metrics.json"), app=app)
    for condition in ("control", "treatment"):
        media_dir = SHOT / "videos" / condition
        assert core.sha(media_dir / "video.mp4") == load(media_dir / "generation.json")["video_sha256"]
    if (directory / "video_identity_metrics.json").exists():
        video_scores = load(directory / "video_identity_metrics.json")
        for condition in ("control", "treatment"):
            assert Path(video_scores["conditions"][condition]["video"]).resolve() == (directory / f"videos/{condition}.mp4").resolve()
    else:
        video_scores = video_evaluation(SimpleNamespace(**common, control=directory / "videos/control.mp4",
            treatment=directory / "videos/treatment.mp4", output=directory / "video_identity_metrics.json"), app=app)
    # DINO uses boxes, not InsightFace embeddings/age/3D landmarks. Keep the
    # exact same SCRFD detector and confidence while omitting those unused heads.
    from insightface.app import FaceAnalysis
    detector = FaceAnalysis(name="antelopev2", root=str(ROOT / "third_party/PuLID"),
                           providers=["CPUExecutionProvider"], allowed_modules=["detection"])
    detector.prepare(ctx_id=-1, det_size=(640, 640))
    encoder = DinoEncoder(Path("/root/autodl-tmp/models/dinov2-base"), "cpu", 4)
    dino_scores = _evaluate_shot(directory, detector, encoder, 0.5, 0.15, {})
    core.write_once(directory / "dino_identity_metrics.json", dino_scores)
    manifest = load(OUT / "manifest.json")
    _render_shot({"shot_dir": str(directory), "characters": manifest["characters"]})
    result = load(directory / "result.json")
    comparison = {
        "scope": "one shot, character-shot means; reference identity diagnostics, not official cs_face",
        "scheduled_role_count": len(manifest["characters"]),
        "injected_role_count": len(result["applied_characters"]),
        "skipped_characters": result["skipped_characters"],
        "original_native_frame_sha256": core.sha(ORIGINAL / "frame/target.png"),
        "paired_control_sha256": core.sha(directory / "control.png"),
        "control_validation": load(OUT / "control_validation.json"),
        "first_frame_insightface": image_scores,
        "video_insightface": video_scores,
        "reference_dino": dino_scores}
    core.write_once(OUT / "evaluation_report.json", comparison)
    core.write_once(OUT / "dino_evaluation_report.json", dino_scores)
    import html
    import statistics
    rows = []
    report_rows = []
    for item in image_scores["characters"]:
        name = item["name"]
        dc = dino_scores["conditions"]["control"]["characters"][name]
        dt = dino_scores["conditions"]["treatment"]["characters"][name]
        vc = video_scores["conditions"]["control"]["characters"][name]
        vt = video_scores["conditions"]["treatment"]["characters"][name]
        pairs = [(item["control_cosine"], item["treatment_cosine"]),
                 (vc.get("mean"), vt.get("mean")),
                 (dc["first_frame"], dt["first_frame"]),
                 (dc["video_frame_mean"], dt["video_frame_mean"])]
        cells = [f"{a:.4f} → {b:.4f} ({b-a:+.4f})" if a is not None and b is not None else "未评分" for a, b in pairs]
        rows.append("<tr><td>" + html.escape(name) + "</td><td>" + "</td><td>".join(cells) + "</td></tr>")
        report_rows.append("| " + name + " | " + " | ".join(cells) + " |")
    summary = "<table border='1'><tr><th>角色</th><th>首帧 IF</th><th>视频 IF</th><th>首帧参考 DINO</th><th>视频参考 DINO</th></tr>" + "".join(rows) + "</table>"
    (OUT / "index.html").write_text("<meta charset='utf-8'><title>run700 多人3D注入</title><h1>4:1 多人 v7 对照</h1>"
        "<p>同提示词、同人物参考、同随机种子；仅增加局部3D注入。视频81帧、16fps。</p>"
        "<p>这里的参考身份相似度不等于官方 cs_face；未修改整集结果。</p>" + summary +
        "<h2>Control</h2><video controls width='832' src='shot_4_1/videos/control/video.mp4'></video>"
        "<h2>多人v7</h2><video controls width='832' src='shot_4_1/videos/treatment/video.mp4'></video>"
        "<p><a href='shot_4_1/frame/injection/comparison.jpg'>参考／3D／mask／首帧完整对照</a></p>"
        "<img width='100%' src='shot_4_1/frame/injection/comparison.jpg'>", encoding="utf-8")
    (OUT / "report.md").write_text("# run700 4:1 多人首帧注入试验\n\n"
        f"计划3角色，实际注入{len(result['applied_characters'])}；跳过：{result['skipped_characters']}。\n\n"
        "| 角色 | 首帧IF | 视频IF | 首帧参考DINO | 视频参考DINO |\n|---|---|---|---|---|\n" +
        "\n".join(report_rows) + "\n\n仅此镜头的配对试验，不替代整集官方指标。视频按空间位置跟踪角色，未用身份最高分选脸。"
        "缺脸按检测覆盖率保留；负增益也保留。\n", encoding="utf-8")
    def mean_pair(metric, kind):
        if kind == "if":
            if metric == "image":
                return tuple(statistics.mean(item[key] for item in image_scores["characters"])
                             for key in ("control_cosine", "treatment_cosine"))
            return tuple(statistics.mean(v["mean"] for v in video_scores["conditions"][condition]["characters"].values()
                                         if v.get("mean") is not None) for condition in ("control", "treatment"))
        key = "first_frame" if metric == "image" else "video_frame_mean"
        return tuple(statistics.mean(v[key] for v in dino_scores["conditions"][condition]["characters"].values()
                                     if v.get(key) is not None) for condition in ("control", "treatment"))
    with (OUT / "report.md").open("a") as stream:
        for kind, title in (("if", "InsightFace"), ("dino", "参考DINO")):
            a, b = mean_pair("image", kind), mean_pair("video", kind)
            stream.write(f"\n| 路线 | 实际注入镜头数 | {title} 首帧 | {title} 视频均值 |\n"
                         "|---|---:|---|---|\n"
                         f"| Qwen多人v7 | 1 | {a[0]:.4f}→{a[1]:.4f} ({a[1]-a[0]:+.4f}) | "
                         f"{b[0]:.4f}→{b[1]:.4f} ({b[1]-b[0]:+.4f}) |\n")
        stream.write("\n一个episode的一个镜头；表内按成功映射/注入的角色—镜头对等权平均。\n")
    print("review", OUT / "index.html", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["assets", "frame", "videos", "evaluate"])
    args = parser.parse_args()
    globals()[args.action]()
