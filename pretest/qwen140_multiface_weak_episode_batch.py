#!/usr/bin/env python3
"""Frozen four-episode weak-case rerun using the approved all-face v7 pipeline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140"
OUT = ROOT / "outputs/entitybench_qwen140_multiface_weak4_20261028"
TEMPLATE = ROOT / "outputs/entitybench_qwen140_run700_multiface_episode_20261028/face_injection_policy_v2.json"
WAN = "/root/autodl-tmp/envs/lightx2v-trial/bin/python"
QWEN = "/root/autodl-tmp/qwen-image21-venv/bin/python"
ASSET = str(ROOT / ".venv/bin/python")
EVAL = "/root/autodl-tmp/envs/entitybench-eval/bin/python"
SELECTION = {"run771": "i15_j24", "run455": "i0_j24", "run484": "i15_j47", "run1404": "i170_j220"}
METRICS = ["dynamic_degree", "aesthetic_quality", "imaging_quality", "intra_character_presence", "intra_location_presence", "cs_face"]
LABELS = {"subject_consistency": "主体一致性", "temporal_flickering": "时间稳定性", "motion_smoothness": "运动平滑度", "dynamic_degree": "动态程度", "aesthetic_quality": "美学质量", "imaging_quality": "成像质量（0–100）", "intra_character_presence": "角色出现率", "intra_object_presence": "物体出现率", "intra_location_presence": "场景出现率", "cs_face": "跨镜头人物区域DINO cs_face", "cs_object": "cs_object", "cs_transition_boundary": "接续边界一致性"}
CODE = ["pretest/qwen140_multiface_weak_episode_batch.py", "pretest/qwen140_integrated_episodes.py", "pretest/qwen140_episode_single_pass.py", "pretest/qwen140_shot_agent.py", "pretest/qwen140_v7_frame.py", "pretest/prepare_qwen_all_detected_faces.py", "pretest/prepare_qwen_multiface_residual.py", "pretest/run_qwen_image21_shared_prefix_adaptive.py", "pretest/qwen140_integrated_episode_eval.py", "pretest/qwen140_wan_api.py", "multishot/multi_face_reference.py", "docs/entitybench_qwen140_route_agent_prompt_v1_20261026.md"]


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(p, obj, *, frozen=False):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    if frozen and p.exists():
        assert p.read_text() == blob, f"Immutable input changed: {p}"
    else:
        p.write_text(blob)


def now():
    return datetime.now(timezone.utc).isoformat()


def status(state, episode=None, **extra):
    write(OUT / "status.json", {"updated_utc": now(), "status": state, "active_episode": episode, **extra})


def prepare():
    if (OUT / "cohort.json").exists():
        return read(OUT / "cohort.json")
    manifest = read(BASE / "manifest.json")
    episodes = []
    rows = []
    for ep in manifest["episodes"]:
        matches = list((BASE / "evaluation/official_parallel_v2/.shards").glob(f"worker_*/results/{ep['episode_id']}.json"))
        assert len(matches) == 1
        result = read(matches[0])
        scores = {k: (result["vbench_means"] if k in result["vbench_means"] else result["metrics"])[k]["value"] for k in METRICS}
        rows.append({"episode_id": ep["episode_id"], "scores": scores, "baseline_result": str(matches[0]), "baseline_result_sha256": sha(matches[0])})
    thresholds = {k: sum(r["scores"][k] for r in rows if r["scores"][k] is not None) / sum(r["scores"][k] is not None for r in rows) for k in METRICS}
    for run, tag in SELECTION.items():
        hits = [e for e in manifest["episodes"] if e["run"] == run and tag in e["episode_id"]]
        assert len(hits) == 1
        ep = hits[0]
        assert sha(ep["episode_path"]) == ep["episode_sha256"]
        score = next(r for r in rows if r["episode_id"] == ep["episode_id"])
        low = [k for k in METRICS if score["scores"][k] is not None and score["scores"][k] < thresholds[k]]
        assert low
        episodes.append({"episode": ep, **score, "below_benchmark": low, "output_root": str(OUT / "experiments" / f"{run}__{tag}")})
    frozen = [{"path": str(ROOT / p), "sha256": sha(ROOT / p)} for p in CODE]
    for entry in frozen:
        dest = OUT / "runtime/frozen_source" / Path(entry["path"]).relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(entry["path"]).read_bytes())
    plan = {"created_utc": now(), "run_tags": SELECTION, "benchmark_episode_count": len(rows), "weak_any_count": sum(any(r["scores"][k] is not None and r["scores"][k] < thresholds[k] for k in METRICS) for r in rows), "selection_policy": "First bounded representative batch: two all-six-low episodes plus previously examined entry/style cases; whole episodes, not all qualifying 138 episodes", "thresholds_exact_episode_means": thresholds, "num_episodes": len(episodes), "num_shots": sum(len(e["episode"]["shots"]) for e in episodes), "episodes": episodes, "frozen_code": frozen, "baseline_manifest_sha256": sha(BASE / "manifest.json"), "v7_template_sha256": sha(TEMPLATE), "failure_policy": "Gate skips including all-role skips reuse same Control; log technical errors and continue other episodes without confirmation, threshold relaxation, rerouting or model substitution", "reference_policy": "Fresh single-portrait role-description-only references, one per scheduled role; same reference binds new episode-local Gaussian; identities differ from original baseline, not a plugin-only causal comparison", "style": "photorealistic live-action film", "evaluation_scope": "12 official non-VLM metrics, strict skip_vlm, zero VLM calls"}
    write(OUT / "cohort.json", plan, frozen=True)
    write(OUT / "cohort_config.json", {"run_tags": SELECTION}, frozen=True)
    styles = {"scope": "four complete weak-case episodes", "episodes": {f"{r}__{t}": {"medium": plan["style"], "evidence": "Baseline first-frame visual inspection; one consistent live-action medium across references, keyframes and video"} for r, t in SELECTION.items()}}
    for item in episodes:
        write(Path(item["output_root"]) / "style_specs.json", styles, frozen=True)
    write(OUT / "eta.json", {"estimated_start_utc": plan["created_utc"], "estimated_finish_utc": (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat(), "range_hours": [5, 7], "basis": "run700 measured main sampling ~80s, keyframe ~80-100s plus per-shot model loading and Agent calls; 48 shots plus 22 fresh references/Gaussians/calibrations and official evaluation", "condition": "Uninterrupted GPU/API execution; technical/pose-all-role failure stops queue for user"}, frozen=True)
    return plan


def environment(directory):
    env = dict(os.environ)
    env.update(QWEN140_INTEGRATED_OUTPUT_ROOT=str(directory), QWEN140_COHORT_CONFIG=str(OUT / "cohort_config.json"), OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", ENTITYBENCH_ORT_THREADS="2", MULTISHOT_ALLOW_FACELIFT_FALLBACK="0", MULTISHOT_DISABLE_FACELIFT="0", MULTISHOT_FACELIFT_STEP_2D="50", PYTHONPATH=str(directory / "runtime") + os.pathsep + str(ROOT))
    return env


def command(item, label, args, env):
    directory = Path(item["output_root"])
    log = directory / "logs" / f"{label}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    status("running", item["episode_id"], stage=label, output_root=str(directory))
    print(now(), item["episode"]["run"], label, flush=True)
    start = time.monotonic()
    with log.open("a") as stream:
        subprocess.run(args, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    with (OUT / "stage_timings.jsonl").open("a") as stream:
        stream.write(json.dumps({"episode_id": item["episode_id"], "stage": label, "seconds": time.monotonic() - start, "finished_utc": now()}) + "\n")


def assets(item, plan):
    from run_entitybench_a14b_full140 import asset_environment
    ep = item["episode"]
    directory = Path(item["output_root"])
    (directory / "runtime").mkdir(parents=True, exist_ok=True)
    env = asset_environment(directory, environment(directory))
    policy_path = directory / "face_injection_policy.json"
    active_policy = directory / "active_face_injection_policy.json"
    if active_policy.exists():
        pointer = read(active_policy)
        policy_path = Path(pointer["path"])
        assert sha(policy_path) == pointer["sha256"]
    if policy_path.exists():
        for f in read(policy_path)["frozen_files"]:
            assert sha(f["path"]) == f["sha256"], f["path"]
        return env
    names = sorted({n for s in ep["shots"] for n in s["entity_schedule"].get("characters", [])})
    descriptions = {v["name"]: v["description"] for v in ep["entity_descriptions"].values()}
    reference_seed = ep["base_seed"] if "base_seed" in ep else ep["shots"][0]["seed"]
    ref_items = [{"name": n, "description": descriptions[n], "path": str(directory / "references" / ep["run"] / f"{n.lower()}.png"), "seed": reference_seed + 22_000_000 + i, "style": {"medium": plan["style"]}} for i, n in enumerate(names)]
    task = directory / "references.task.json"
    write(task, {"episode_id": ep["episode_id"], "items": ref_items, "style_sha256": sha(directory / "style_specs.json")}, frozen=True)
    command(item, "references", [QWEN, "-u", "pretest/qwen140_integrated_episodes.py", "worker-refs", "--task", str(task)], env)
    roles = {}
    asset_failures = []
    for ref in ref_items:
        name = ref["name"]
        slug = "_".join(name.lower().replace("-", " ").split())
        asset_root = directory / "assets" / ep["episode_id"]
        link = asset_root / "characters" / f"{slug}_reference.png"
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(ref["path"])
        assert sha(link) == sha(ref["path"])
        try:
            for stage in ("facelift", "calibrate"):
                command(item, f"{slug}_{stage}", [ASSET, "-u", "pretest/prepare_entitybench_character_assets.py", "--episode", ep["episode_path"], "--asset-root", str(asset_root), "--stage", stage, "--character-scope", "scheduled", "--only-character", name, "--base-seed", "30018000", "--calibration-size", "512", "--python", ASSET], env)
            result_path = asset_root / "faces_3d" / slug / "facelift_result.json"
            result = read(result_path)
            assert result["facelift_status"] == "success"
            assert result["source_reference_sha256"] == sha(ref["path"])
            model = Path(result["model_path"])
            assert read(model.with_suffix(".pose_calibration.json"))["gaussian_model_sha256"] == sha(model)
            roles[name] = {"name": name, "reference_image": ref["path"], "facelift_result": str(result_path)}
        except Exception as exc:
            asset_failures.append({"name": name, "error": repr(exc), "stage": stage, "action": "Retain 2D reference, omit unavailable 3D for this role, continue other roles"})
            write(directory / "asset_failures.json", asset_failures)
    files = {f["path"]: f for f in plan["frozen_code"]}
    refs = {r["name"]: {"path": r["path"], "sha256": sha(r["path"]), "origin": "fresh_clean_single_portrait_for_this_episode"} for r in ref_items}
    for r in refs.values():
        files[r["path"]] = {"path": r["path"], "sha256": r["sha256"]}
    for name, role in roles.items():
        result = read(role["facelift_result"])
        assert result["facelift_status"] == "success"
        assert result["source_reference_sha256"] == sha(role["reference_image"])
        model = Path(result["model_path"])
        calibration = model.with_suffix(".pose_calibration.json")
        assert read(calibration)["gaussian_model_sha256"] == sha(model)
        for p in [role["reference_image"], role["facelift_result"], str(model), str(calibration)]:
            files[p] = {"path": p, "sha256": sha(p)}
    template = read(TEMPLATE)
    policy = {**template, "references": refs, "roles": roles, "frozen_files": list(files.values()), "required_shots": [], "reference_generation_count": len(refs), "character_reference_policy": "Frozen clean single portraits from role sentence and episode style only; corresponding new Gaussian; no per-shot identity regeneration", "identity_scope": "Every identity anchor and Gaussian is episode-local and newly paired in this batch; original benchmark identity anchors are not held fixed", "revision": "weak4_all_detected_faces_v2", "scope": "All generated B/C keyframes; A inherits actual own tail; original per-role v7 gates, no new area threshold"}
    policy.pop("previous_policy_sha256", None)
    policy.update(allow_control_reuse=True, asset_failures=asset_failures,
                  failure_policy="Gate skips, including all-role skips, reuse the same Control; log technical exceptions and continue other episodes; no threshold relaxation or model substitution")
    write(policy_path, policy, frozen=True)
    return env


def summarize(item):
    directory = Path(item["output_root"])
    comparison = read(directory / "evaluation/weak4_v1/comparison.json")["rows"][0]["metrics"]
    result_root = directory / "evaluation/weak4_v1/results"
    result = read(result_root / f"{item['episode_id']}.json")
    manifest = read(result_root / "run_manifest.json")
    assert manifest["strict"] and manifest["skip_vlm"] and manifest["vlm_call_stats"]["total_calls"] == 0
    coverage = {}
    lines = [f"# {item['episode']['run']}：原始benchmark与多人3D集成", "", "完整episode；同一官方非VLM口径，VLM调用0。cs_face/cs_object缺保真度门控，属于partial_without_vlm。", "", "| 指标↑ | 原始episode | 多人3D集成 | 差值 |", "|---|---:|---:|---:|"]
    for group in ("vbench_means", "metrics"):
        for k, v in comparison[group].items():
            if k not in LABELS:
                continue
            stat = result[group][k]
            assert stat["value"] is not None and stat["n_failed"] == 0, (k, stat)
            coverage[k] = stat
            a, b = v["baseline"], v["candidate"]
            if a is not None and b is not None:
                lines.append(f"| {LABELS[k]} | {a:.4f} | {b:.4f} | {b-a:+.4f} |")
    assert len(coverage) == 12
    episode = read(directory / "episodes" / item["episode_id"] / "episode_result.json")
    injections = []
    for i, shot in enumerate(episode["shots"]):
        assert sha(shot["video"]) == shot["video_sha256"]
        assert sha(shot["last_frame"]) == shot["last_sha256"]
        if i and not shot["cut"]:
            assert shot["parent_video_sha256"] == episode["shots"][i-1]["video_sha256"]
            assert shot["parent_tail_sha256"] == episode["shots"][i-1]["last_sha256"]
        if shot.get("injection_result"):
            inj = read(shot["injection_result"])
            if inj["injection_applied"]:
                assert any(s.get("injected") and s.get("residual_norm", 0) > 0 for s in inj["step_log"])
            else:
                assert sha(inj["control"]) == sha(inj["treatment"])
            injections.append({"shot": shot["shot_key"], "injected": inj["applied_characters"], "skipped": inj["skipped_characters"], "target_failures": inj["target_failures"], "reason": inj.get("reason"), "error": inj.get("error")})
    write(directory / "evaluation_validation.json", {"coverage": coverage, "vlm_calls": 0, "actual_injections": injections, "visual_review": "pending user/assistant video review; do not equate process completion with visual success"}, frozen=True)
    lines += ["", "新人物锚点、路由、关键帧与注入范围同时变化，不是插件独立因果消融。完整视频人工审阅待完成。", "", "[原始/新整集及逐shot视频](review/index.html)"]
    (directory / "report.md").write_text("\n".join(lines) + "\n")
    write(directory / "complete.json", {"episode_id": item["episode_id"], "completed_utc": now(), "generation_and_official_non_vlm": "complete", "video_visual_review": "pending"}, frozen=True)


def index(plan):
    rows = []
    for item in plan["episodes"]:
        d = Path(item["output_root"])
        complete = (d / "complete.json").exists()
        progress = read(d / "progress.json") if (d / "progress.json").exists() else {}
        url = os.path.relpath(d / "review/index.html", OUT)
        report = os.path.relpath(d / "report.md", OUT)
        rows.append(f'<tr><td>{html.escape(item["episode_id"])}</td><td>{len(item["episode"]["shots"])}</td><td>{"生成/评测完成，视频待审" if complete else str(progress.get("completed_shots",0))+"镜头完成"}</td><td><a href="{url}">前后视频</a> · <a href="{report}">指标</a></td></tr>')
    (OUT / "index.html").write_text('<meta charset="utf-8"><h1>四个低指标episode：多人3D完整重跑</h1><p>48shot；结果按完整episode统计，VLM调用0；未完成链接会在对应阶段生成。</p><table><tr><th>episode</th><th>shot数</th><th>状态</th><th>结果</th></tr>'+"".join(rows)+'</table>')


def run(plan):
    for f in plan["frozen_code"]:
        assert sha(f["path"]) == f["sha256"], f["path"]
    assert sha(BASE / "manifest.json") == plan["baseline_manifest_sha256"]
    assert sha(TEMPLATE) == plan["v7_template_sha256"]
    index(plan)
    while True:
        probe = subprocess.run([QWEN, "-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"],
                               env={**os.environ, "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"},
                               capture_output=True, text=True, timeout=60)
        if probe.returncode == 0:
            break
        completed_prefix = sum(len(list((Path(i["output_root"]) / "episodes").glob("*/shot_*/result.json"))) for i in plan["episodes"])
        status("waiting_for_gpu", completed_prefix_shots=completed_prefix, expected_shots=plan["num_shots"],
               reason="CUDA unavailable; preserve valid work and resume automatically when GPU returns")
        time.sleep(30)
    errors = []
    for item in plan["episodes"]:
        d = Path(item["output_root"])
        if (d / "complete.json").exists():
            continue
        try:
            env = assets(item, plan)
            if not (d / "rerun_manifest.json").exists():
                for stage in ("prepare", "refs"):
                    command(item, stage, [WAN, "-u", "pretest/qwen140_episode_single_pass.py", stage, "--run", item["episode"]["run"]], env)
            if not (d / "episodes" / item["episode_id"] / "episode_result.json").exists():
                command(item, "generation", [WAN, "-u", "pretest/qwen140_episode_single_pass.py", "run-remaining", "--run", item["episode"]["run"]], env)
            for stage in ("prepare", "run", "baseline-aggregate", "compare"):
                command(item, "evaluation_"+stage, [EVAL, "-u", "pretest/qwen140_integrated_episode_eval.py", stage, "--name", "weak4_v1", "--runs", item["episode"]["run"]], env)
            summarize(item)
        except Exception as exc:
            error = {"episode_id": item["episode_id"], "recorded_utc": now(), "error": repr(exc), "stage_status": read(OUT / "status.json"), "action": "Preserve valid prefix and exception; continue next complete episode; no silent model or route change"}
            errors.append(error)
            write(d / "technical_exception.json", error)
            write(OUT / "technical_exceptions.json", errors)
            print("RECORDED EPISODE EXCEPTION; CONTINUING", item["episode_id"], repr(exc), flush=True)
        index(plan)
    completed = [i for i in plan["episodes"] if (Path(i["output_root"]) / "complete.json").exists()]
    status("complete_non_vlm_pending_video_visual_review" if len(completed) == len(plan["episodes"]) else "batch_finished_with_recorded_exceptions", completed_episodes=len(completed), completed_shots=sum(len(i["episode"]["shots"]) for i in completed), expected_episodes=len(plan["episodes"]), technical_exceptions=len(errors))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run"])
    args = parser.parse_args()
    try:
        plan = prepare()
        if args.action == "prepare":
            index(plan)
            status("prepared", num_episodes=plan["num_episodes"], num_shots=plan["num_shots"])
            print(OUT / "cohort.json")
        else:
            run(plan)
    except Exception as exc:
        previous = read(OUT / "status.json") if (OUT / "status.json").exists() else None
        status("batch_setup_exception_recorded", previous=previous, error=repr(exc))
        write(OUT / "stopped.json", {"error": repr(exc), "previous": previous, "policy": "Setup/integrity exception recorded; no pending user confirmation and no silent input/model replacement."})
        raise
