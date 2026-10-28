#!/usr/bin/env python3
"""Run a complete episode with v7 on every generated B/C keyframe; stop on errors."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/entitybench_qwen140_run700_multiface_episode_20261028"
os.environ["QWEN140_INTEGRATED_OUTPUT_ROOT"] = str(OUT)
import qwen140_integrated_episodes as core

SINGLE = ROOT / "outputs/entitybench_qwen140_run700_v7_restored_20261028"
TRIAL = ROOT / "outputs/entitybench_qwen140_run700_multiface_v7_20261028"
EID = "00050e2d-6c9a-38e6-b702-87f8df189087__run700__i25_j50__T300"
ASSETS = OUT / "assets" / EID
ASSET_PYTHON = str(ROOT / ".venv/bin/python")
EVAL_PYTHON = "/root/autodl-tmp/envs/entitybench-eval/bin/python"
EVAL_NAME = "run700_multiface_episode_v1"


def load(path):
    return json.loads(Path(path).read_text())


def environment():
    result = dict(os.environ)
    result.update(QWEN140_INTEGRATED_OUTPUT_ROOT=str(OUT), OMP_NUM_THREADS="4",
                  OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
                  HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  PYTHONPATH=str(OUT / "runtime") + os.pathsep + str(ROOT),
                  ENTITYBENCH_ORT_THREADS="2", MULTISHOT_ALLOW_FACELIFT_FALLBACK="0",
                  MULTISHOT_DISABLE_FACELIFT="0", MULTISHOT_FACELIFT_STEP_2D="50")
    return result


def command(label, arguments, env=None):
    path = OUT / "logs" / f"{label}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    print("STAGE", label, flush=True)
    with path.open("a") as stream:
        subprocess.run(arguments, cwd=ROOT, env=env or environment(),
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def prepare_assets():
    from run_entitybench_a14b_full140 import asset_environment
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "runtime").mkdir(exist_ok=True)
    env = asset_environment(OUT, environment())
    source = load(SINGLE / "face_injection_policy.json")
    for name in ("Arthur", "Leo", "Nora"):
        reference = source["references"][name]
        assert core.sha(Path(reference["path"])) == reference["sha256"]
        link = ASSETS / "characters" / f"{name.lower()}_reference.png"
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(reference["path"])
        assert core.sha(link) == reference["sha256"]
        for stage in ("facelift", "calibrate"):
            command(f"{name.lower()}_{stage}", [ASSET_PYTHON, "-u",
                "pretest/prepare_entitybench_character_assets.py", "--episode",
                str(ROOT / "benchmarks/entitybench/data/scripts" / f"{EID}.json"),
                "--asset-root", str(ASSETS), "--stage", stage,
                "--character-scope", "scheduled", "--only-character", name,
                "--base-seed", "30018000", "--calibration-size", "512", "--python", ASSET_PYTHON], env)
    roles = {item["name"]: item for item in load(TRIAL / "manifest.json")["characters"]}
    for name in ("Arthur", "Leo", "Nora"):
        roles[name] = {"name": name, "reference_image": source["references"][name]["path"],
                       "facelift_result": str(ASSETS / "faces_3d" / name.lower() / "facelift_result.json")}
    frozen = {}
    def freeze(path):
        path = Path(path)
        frozen[str(path)] = {"path": str(path), "sha256": core.sha(path)}
    for role in roles.values():
        asset = load(role["facelift_result"])
        assert asset["facelift_status"] == "success"
        assert asset["source_reference_sha256"] == core.sha(Path(role["reference_image"]))
        model = Path(asset["model_path"])
        calibration = model.with_suffix(".pose_calibration.json")
        assert load(calibration)["gaussian_model_sha256"] == core.sha(model)
        for path in (role["reference_image"], role["facelift_result"], model, calibration):
            freeze(path)
    for path in ["pretest/run_qwen_image21_shared_prefix_adaptive.py",
                 "pretest/prepare_qwen_multiface_residual.py", "pretest/qwen140_v7_frame.py"]:
        freeze(ROOT / path)
    policy = {**source, "mode": "all_generated_keyframes_v7", "roles": roles,
              "settings": {**source["settings"], "multi_face_injection": True},
              "frozen_files": list(frozen.values()), "required_shots": [],
              "scope": "all new B transition targets and C action starts; A inherits actual tail; per-role existing v7 gates",
              "conditioning": "Agent-selected previous tail/new-role references only; all frame_characters have identity assets for 3D mapping, without adding redundant conditioning portraits"}
    core.write_once(OUT / "face_injection_policy.json", policy)
    core.write_once(OUT / "experiment_contract.json", {
        "episode_id": EID, "scope": "full 10-shot episode, all B/C keyframes support single/multiple visible roles",
        "references": source["references"], "reference_regeneration": False,
        "comparison": "original full140 run700 and previous single-hardcut-only integrated run700; no isolated causal gain claim",
        "new_scope_vs_previous_single": "multi-person hardcuts and newly generated continuation B/C keyframes now receive v7; previous single version injected only 2:1/3:1",
        "failure_policy": "stop; no automatic semantic repair, rerouting, regeneration, model substitution or whole-shot Control fallback"})


def report():
    comparison = load(OUT / "evaluation" / EVAL_NAME / "comparison.json")["rows"][0]["metrics"]
    previous = load(SINGLE / "evaluation/run700_v7_restored_v1/comparison.json")["rows"][0]["metrics"]
    episode = load(OUT / "episodes" / EID / "episode_result.json")
    role_rows = []
    for shot in episode["shots"]:
        if shot.get("injection_result"):
            injection = load(shot["injection_result"])
            if shot["injection_applied"]:
                assert injection["injection_applied"]
                assert any(s.get("injected") and s.get("residual_norm", 0) > 0 for s in injection["step_log"])
            else:
                assert shot["shot_key"] == "1:2" and (OUT / "user_decisions/shot_1_2_skip_injection.json").is_file()
            role_rows.append({"shot": shot["shot_key"], "route": shot["route"]["route"],
                              "injected": injection["applied_characters"], "skipped": injection["skipped_characters"],
                              "targets": injection.get("targets", []), "failures": injection["target_failures"],
                              "reason": injection.get("error") or injection.get("reason")})
    core.write_once(OUT / "injection_audit.json", {"shots": role_rows})
    labels = {"subject_consistency": "主体一致性", "temporal_flickering": "时间稳定性",
              "motion_smoothness": "运动平滑度", "dynamic_degree": "动态程度",
              "aesthetic_quality": "美学质量", "imaging_quality": "成像质量（0–100）",
              "intra_character_presence": "角色出现率", "intra_object_presence": "物体出现率",
              "intra_location_presence": "场景出现率", "cs_face": "人物区域cs_face",
              "cs_object": "物体区域cs_object", "cs_transition_boundary": "接续边界一致性"}
    lines = ["# run700：多人v7接入后的完整episode", "", "完整10镜头；原始结果为full140中的同一run700，不是全140集均值。", "",
             "所有新生成的B目标尾帧/C动作首帧支持逐角色v7；A直接继承上一实际尾帧。本轮同时扩大到接续新关键帧，不是只改变硬切多人开关的单因素实验。", "",
             "| 指标 | 原始run700 | 上轮单人集成版 | 本轮多人集成版 | 相对上轮 |", "|---|---:|---:|---:|---:|"]
    for group in ("vbench_means", "metrics"):
        for key, item in comparison[group].items():
            if key not in labels: continue
            a,b,c=item["baseline"],previous[group][key]["candidate"],item["candidate"]
            lines.append(f"| {labels[key]} | {a:.4f} | {b:.4f} | {c:.4f} | {c-b:+.4f} |")
    lines += ["", "官方strict/skip_vlm，调用0；cs_face/cs_object为缺少保真度门控的暂定人物/物体区域指标。没有调低阈值或筛掉负样本。", "",
              "| shot | 路由 | 实际注入角色 | 跳过角色 |", "|---|---|---|---|"]
    for row in role_rows:
        lines.append(f"| {row['shot']} | {row['route']} | {', '.join(row['injected'])} | {', '.join(row['skipped']) or '无'} |")
    lines += ["", "未另生成全套Control视频，不把整集差异解释为3D插件独立因果增益。", "", "[视频](review/index.html) · [逐角色注入记录](injection_audit.json)"]
    (OUT / "report.md").write_text("\n".join(lines)+"\n")
    p = OUT / "review/index.html"
    text = p.read_text()
    url = os.path.relpath(SINGLE / "review/after.mp4", p.parent)
    text = text.replace("<h2>新结果</h2>", f'<h2>上轮：仅单人硬切注入</h2><video controls width="832" src="{url}"></video><h2>本轮：所有新关键帧支持多人注入</h2>')
    text += '<p><a href="../report.md">完整指标与注入范围</a></p>'
    p.write_text(text)
    status = load(OUT / "progress.json")
    status.update(status="generation_and_official_non_vlm_complete_pending_visual_review",
                  evaluation=f"evaluation/{EVAL_NAME}")
    (OUT / "progress.json").write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n")


def main():
    actions = ("run-remaining",)
    if not (OUT / "rerun_manifest.json").exists():
        prepare_assets()
        actions = ("prepare", "refs", "run-remaining")
    for action in actions:
        command(action, [core.WAN_PYTHON, "-u", "pretest/qwen140_episode_single_pass.py", action, "--run", "run700"])
    for action in ("prepare", "run", "baseline-aggregate", "compare"):
        command("evaluation_"+action, [EVAL_PYTHON, "-u", "pretest/qwen140_integrated_episode_eval.py",
                action, "--name", EVAL_NAME, "--runs", "run700"])
    report()
    print("COMPLETE", OUT / "review/index.html", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        OUT.mkdir(parents=True,exist_ok=True)
        p = OUT / "progress.json"
        (OUT / "stopped.json").write_text(json.dumps({"error":str(error),
            "progress":load(p) if p.exists() else None,
            "policy":"stopped for user decision; no retry or fallback"},ensure_ascii=False,indent=2)+"\n")
        raise
