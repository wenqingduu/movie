#!/usr/bin/env python3
"""Wait for weak4, then aggregate matched complete episodes with the official code."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/entitybench_qwen140_multiface_weak4_20261028"
BASE = ROOT / "outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140"


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def aggregate():
    plan = read(OUT / "cohort.json")
    selected = [item for item in plan["episodes"] if (Path(item["output_root"]) / "complete.json").exists()]
    missing = [i["episode_id"] for i in plan["episodes"] if i not in selected]
    summary = {"expected_episodes": plan["num_episodes"], "paired_completed_episodes": len(selected), "missing_episodes": missing, "scope": "Same complete episode subset in both columns; strict official non-VLM; missing episodes not scored or filled with zero"}
    if not selected:
        (OUT / "cohort_comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    evaluator = BASE / "runtime/evaluate_benchmark_partial.py"
    spec = importlib.util.spec_from_file_location("weak4_frozen_official", evaluator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    split = {"easy": [], "medium": [], "hard": []}
    old, new, fingerprints = [], [], []
    for item in selected:
        ep, d = item["episode"], Path(item["output_root"])
        inputs = read(d / "evaluation/weak4_v1/input_manifest.json")
        assert inputs["evaluator_sha256"] == sha(evaluator)
        for info in inputs["videos"].values():
            assert sha(info["path"]) == info["sha256"]
        assert sha(item["baseline_result"]) == item["baseline_result_sha256"]
        result_path = d / "evaluation/weak4_v1/results" / f"{ep['episode_id']}.json"
        raw = read(result_path)
        manifest = read(result_path.parent / "run_manifest.json")
        assert manifest["strict"] and manifest["skip_vlm"] and manifest["vlm_call_stats"]["total_calls"] == 0
        assert raw["num_shots"] == len(ep["shots"])
        split[ep["difficulty"]].append(ep["episode_id"])
        old.append(module._episode_result_from_json(ep["episode_id"], ep["difficulty"], read(item["baseline_result"])))
        new.append(module._episode_result_from_json(ep["episode_id"], ep["difficulty"], raw))
        fingerprints.append({"episode_id": ep["episode_id"], "baseline_sha256": sha(item["baseline_result"]), "candidate_sha256": sha(result_path), "input_manifest_sha256": sha(d / "evaluation/weak4_v1/input_manifest.json")})
    a, b = module.aggregate_results(old, split), module.aggregate_results(new, split)
    summary.update(evaluator_sha256=sha(evaluator), input_fingerprints=fingerprints, split=split, baseline=a, candidate=b)
    (OUT / "cohort_comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    labels = {"subject_consistency": "主体一致性", "temporal_flickering": "时间稳定性", "motion_smoothness": "运动平滑度", "dynamic_degree": "动态程度", "aesthetic_quality": "美学质量", "imaging_quality": "成像质量（0–100）", "intra_character_presence": "角色出现率", "intra_object_presence": "物体出现率", "intra_location_presence": "场景出现率", "cs_face": "人物区域cs_face", "cs_object": "物体区域cs_object", "cs_transition_boundary": "接续边界一致性"}
    lines = ["# 弱指标episode：原始与多人3D集成", "", f"计划{len(plan['episodes'])}集，实际相同配对完成集合{len(selected)}集；缺失{len(missing)}集。官方集等权聚合；VLM调用0；不代表全140集。", "", "| 指标↑ | 原始同一子集 | 多人3D集成 | 差值 |", "|---|---:|---:|---:|"]
    for key, label in labels.items():
        av = a["all"]["vbench"][key]["mean"] if key in a["all"]["vbench"] else a["all"][key].get("mean")
        bv = b["all"]["vbench"][key]["mean"] if key in b["all"]["vbench"] else b["all"][key].get("mean")
        if av is None or bv is None:
            lines.append(f"| {label} | {av} | {bv} | 未评分 |")
        else:
            lines.append(f"| {label} | {av:.4f} | {bv:.4f} | {bv-av:+.4f} |")
    lines += ["", "cs_face/cs_object缺VLM保真度门控；素材身份、路由、提示词及注入范围同时变化，不作插件独立因果结论。门控未通过的Control沿用镜头仍保留在整集评分中。完整人工视频审阅待完成。", "", "缺失episode：" + (", ".join(missing) or "无"), "", "[各集视频和指标入口](index.html)"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print("Official paired cohort aggregation complete", len(selected), "episodes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    if args.wait:
        terminal = {"complete_non_vlm_pending_video_visual_review", "batch_finished_with_recorded_exceptions", "batch_setup_exception_recorded"}
        while read(OUT / "status.json")["status"] not in terminal:
            time.sleep(20)
    try:
        aggregate()
    except Exception as exc:
        (OUT / "aggregation_exception.json").write_text(json.dumps({"error": repr(exc), "action": "recorded; existing per-episode results preserved"}, ensure_ascii=False, indent=2))
        raise
