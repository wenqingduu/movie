#!/usr/bin/env python3
"""Prepare and compare complete-episode official non-VLM evaluations for pilot replacements."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from qwen140_pilot_qwen import BASE, OUT, TARGETS, sha

ROOT = OUT / "official_subset_v1"
REPLACEMENTS = {
    "run714": "main",
    "run484": "combined",
    "run755": "main",
    "run1404": "main",
}
METRICS = ("intra_character_presence", "cs_face", "intra_object_presence", "cs_object",
           "intra_location_presence", "sc_background", "ec_background")
VBENCH = ("subject_consistency", "temporal_flickering", "motion_smoothness",
          "dynamic_degree", "aesthetic_quality", "imaging_quality")


def source_data():
    manifest = json.loads((BASE / "manifest.json").read_text())
    report = json.loads((BASE / "generation_report.json").read_text())
    jobs = {(r["episode_id"], r["shot_key"]): r for r in report["jobs"]}
    return manifest, jobs


def prepare(runs: list[str]):
    manifest, jobs = source_data()
    split = {"easy": [], "medium": [], "hard": []}
    selected = []
    videos = {}
    for run in runs:
        assert run in REPLACEMENTS
        tag, shot_key, _ = TARGETS[run]
        ep = next(e for e in manifest["episodes"] if e["run"] == run and tag in e["episode_id"])
        eid = ep["episode_id"]
        split[ep["difficulty"]].append(eid)
        index = next(i for i, s in enumerate(ep["shots"]) if s["shot_key"] == shot_key)
        assert index + 1 == len(ep["shots"]) or ep["shots"][index + 1]["cut"], \
            f"Replacement {run} requires downstream rerun before complete-episode evaluation"
        candidate = OUT / "videos" / run / REPLACEMENTS[run] / "video.mp4"
        rec = json.loads((candidate.parent / "generation.json").read_text())
        assert candidate.is_file() and sha(candidate) == rec["video_sha256"]
        replacements = {shot_key: candidate}
        for shot in ep["shots"]:
            sk = shot["shot_key"]
            original = jobs[eid, sk]
            old = Path(original["output_video"])
            assert sha(old) == original["video_sha256"]
            source = replacements.get(sk, old)
            scene, local = map(int, sk.split(":"))
            link = ROOT / "official_results" / eid / f"{scene:03d}_{local:03d}.mp4"
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink():
                assert link.resolve() == source.resolve()
            else:
                assert not link.exists(), f"Refusing to overwrite {link}"
                link.symlink_to(source.resolve())
            videos[f"{eid}/{sk}"] = {"path": str(source.resolve()), "sha256": sha(source),
                                      "original_sha256": original["video_sha256"], "replaced": sk == shot_key}
        selected.append({"episode_id": eid, "run": run, "difficulty": ep["difficulty"],
                         "shot_key": shot_key, "n_shots": len(ep["shots"]), "candidate": str(candidate),
                         "candidate_sha256": sha(candidate), "next_is_cut_or_end": True})
    ROOT.mkdir(parents=True, exist_ok=True)
    plan = {"kind": "complete_episode_pilot_subset", "source_manifest_sha256": sha(BASE / "manifest.json"),
            "source_generation_report_sha256": sha(BASE / "generation_report.json"),
            "evaluator_sha256": sha(BASE / "runtime/evaluate_benchmark_partial.py"),
            "scope": "strict official non-VLM; selected complete episodes, no full140 aggregate",
            "episodes": selected, "videos": videos}
    path = ROOT / "input_manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == plan, "Evaluation input manifest changed; use new root"
    else:
        path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    split_path = ROOT / "split.json"
    if split_path.exists():
        assert json.loads(split_path.read_text()) == split
    else:
        split_path.write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"root": str(ROOT), "episodes": selected, "n_videos": len(videos)}, ensure_ascii=False, indent=2))


def compare():
    plan = json.loads((ROOT / "input_manifest.json").read_text())
    assert sha(BASE / "runtime/evaluate_benchmark_partial.py") == plan["evaluator_sha256"]
    for key, info in plan["videos"].items():
        assert sha(Path(info["path"])) == info["sha256"], key
    rows = []
    for ep in plan["episodes"]:
        eid = ep["episode_id"]
        original = list((BASE / "evaluation/official_parallel_v2/.shards").glob(f"worker_*/results/{eid}.json"))
        assert len(original) == 1, (eid, original)
        baseline = json.loads(original[0].read_text())
        candidate_path = ROOT / "results" / f"{eid}.json"
        candidate = json.loads(candidate_path.read_text())
        assert baseline["episode_id"] == candidate["episode_id"] == eid
        assert baseline["num_shots"] == candidate["num_shots"] == ep["n_shots"]
        detail = {}
        for category, names in (("metrics", METRICS), ("vbench_means", VBENCH)):
            detail[category] = {}
            for name in names:
                old, new = baseline[category].get(name, {}), candidate[category].get(name, {})
                ov, nv = old.get("value"), new.get("value")
                detail[category][name] = {"baseline": ov, "candidate": nv,
                                           "delta": nv - ov if ov is not None and nv is not None else None,
                                           "baseline_coverage": {k: old.get(k) for k in ("n_evaluated", "n_failed", "n_skipped")},
                                           "candidate_coverage": {k: new.get(k) for k in ("n_evaluated", "n_failed", "n_skipped")}}
        shot_part = ep["shot_key"].replace(":", "_")
        original_shot_files = list((BASE / "evaluation/official_parallel_v2/.shards").glob(
            f"worker_*/results/.work/eval_artifacts/{eid}/intra_shot/{shot_part}/_eval.json"))
        assert len(original_shot_files) == 1, (eid, original_shot_files)
        candidate_shot_file = ROOT / "results/.work/eval_artifacts" / eid / "intra_shot" / shot_part / "_eval.json"
        old_shot = json.loads(original_shot_files[0].read_text())
        new_shot = json.loads(candidate_shot_file.read_text())
        old_roles = {x["name"]: x for x in old_shot["presence"]["character"]["items"]}
        new_roles = {x["name"]: x for x in new_shot["presence"]["character"]["items"]}
        assert set(old_roles) == set(new_roles)
        role_details = {}
        for name in sorted(old_roles):
            old_face = baseline["per_entity_scores"].get(f"face_{name}", {})
            new_face = candidate["per_entity_scores"].get(f"face_{name}", {})
            role_details[name] = {
                "presence_baseline": {k: old_roles[name].get(k) for k in ("status", "clip_sim", "frame_idx")},
                "presence_candidate": {k: new_roles[name].get(k) for k in ("status", "clip_sim", "frame_idx")},
                "cs_face_role_mean_baseline": old_face.get("mean_sim_to_centroid"),
                "cs_face_role_mean_candidate": new_face.get("mean_sim_to_centroid"),
                "cs_face_role_n_baseline": old_face.get("n_appearances"),
                "cs_face_role_n_candidate": new_face.get("n_appearances"),
                "cs_face_target_shot_baseline": next((x["sim_to_centroid"] for x in old_face.get("per_shot_sim_to_centroid", [])
                                                       if x["shot"] == ep["shot_key"]), None),
                "cs_face_target_shot_candidate": next((x["sim_to_centroid"] for x in new_face.get("per_shot_sim_to_centroid", [])
                                                        if x["shot"] == ep["shot_key"]), None),
            }
        rows.append({"episode_id": eid, "run": ep["run"], "shot_key": ep["shot_key"],
                     "n_shots": ep["n_shots"], "candidate_sha256": ep["candidate_sha256"],
                     "scores": detail, "target_roles": role_details})
    dest = ROOT / "comparison.json"
    dest.write_text(json.dumps({"scope": plan["scope"], "rows": rows}, ensure_ascii=False, indent=2) + "\n")
    for row in rows:
        print(row["run"], row["shot_key"], "presence", row["scores"]["metrics"]["intra_character_presence"],
              "cs_face", row["scores"]["metrics"]["cs_face"])


def run_official():
    assert (ROOT / "input_manifest.json").exists() and (ROOT / "split.json").exists()
    plan = json.loads((ROOT / "input_manifest.json").read_text())
    assert sha(BASE / "runtime/evaluate_benchmark_partial.py") == plan["evaluator_sha256"]
    for key, info in plan["videos"].items():
        assert sha(Path(info["path"])) == info["sha256"], key
    env = {k: v for k, v in os.environ.items() if not any(t in k.upper() for t in
           ("API_KEY", "SECRET", "CREDENTIAL", "ACCESS_TOKEN", "DASHSCOPE", "OPENAI", "AZURE", "GEMINI"))}
    env.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", NUMEXPR_NUM_THREADS="4",
               CUDA_VISIBLE_DEVICES="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", PYTHONPATH=str(BASE.parents[1]),
               DINO_MODEL_PATH="/root/autodl-tmp/models/dinov2-base",
               CLIP_MODEL_PATH="/root/autodl-tmp/models/clip-vit-base-patch32",
               GDINO_CONFIG="/root/autodl-tmp/models/entitybench_groundingdino/GroundingDINO_SwinT_OGC.py",
               GDINO_WEIGHTS="/root/autodl-tmp/models/entitybench_groundingdino/groundingdino_swint_ogc.pth",
               VBENCH_CACHE_DIR="/root/autodl-tmp/models/entitybench_vbench",
               TORCH_HOME="/root/autodl-tmp/models/entitybench_torch_cache",
               HF_HOME="/root/autodl-tmp/models/entitybench_hf_cache")
    cmd = ["/root/autodl-tmp/envs/entitybench-eval/bin/python", "-u",
           str(BASE / "runtime/evaluate_benchmark_partial.py"),
           "--results_dir", str(ROOT / "official_results"),
           "--scripts_dir", str(BASE.parents[1] / "benchmarks/entitybench/data/scripts"),
           "--split_json", str(ROOT / "split.json"), "--out_dir", str(ROOT / "results"),
           "--method_name", "qwen140_pilot_routed_subset", "--pillars", "1,2,3",
           "--skip_vlm", "--resume", "--device", "cuda:0",
           "--vbench_cache_dir", "/root/autodl-tmp/models/entitybench_vbench"]
    with (ROOT / "evaluation.log").open("a") as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print("Official strict non-VLM subset evaluation completed", ROOT / "results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run", "compare"])
    parser.add_argument("--runs", nargs="+", choices=list(REPLACEMENTS), default=list(REPLACEMENTS))
    parser.add_argument("--root-name", default="official_subset_v1")
    parser.add_argument("--run714-segment", choices=["main", "main_identity_anchor_v2", "main_identity_anchor_slow_v3"], default="main")
    args = parser.parse_args()
    assert "/" not in args.root_name and args.root_name not in {"", ".", ".."}
    ROOT = OUT / args.root_name
    REPLACEMENTS["run714"] = args.run714_segment
    if args.action == "prepare":
        prepare(args.runs)
    elif args.action == "run":
        run_official()
    else:
        compare()
