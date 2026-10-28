#!/usr/bin/env python3
"""Frozen official non-VLM evaluation of complete integrated pilot episodes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
from pathlib import Path

from qwen140_integrated_episodes import BASE, OUT, TAGS, episode_for, sha, source, write_once

EVAL_PYTHON = "/root/autodl-tmp/envs/entitybench-eval/bin/python"
EVALUATOR = BASE / "runtime/evaluate_benchmark_partial.py"
VBENCH = ("subject_consistency", "temporal_flickering", "motion_smoothness",
          "dynamic_degree", "aesthetic_quality", "imaging_quality")


def root(name: str) -> Path:
    assert name and name not in {".", ".."} and "/" not in name
    return OUT / "evaluation" / name


def prepare(name: str, runs: list[str]) -> None:
    manifest, _, _, jobs = source()
    dst = root(name)
    selected = [episode_for(run, manifest) for run in runs]
    split = {"easy": [], "medium": [], "hard": []}
    inputs = {}
    for ep in selected:
        eid = ep["episode_id"]
        result_path = OUT / "episodes" / eid / "episode_result.json"
        row = json.loads(result_path.read_text())
        assert row["episode_id"] == eid and len(row["shots"]) == len(ep["shots"])
        assert row["style_sha256"] == sha(OUT / "style_specs.json")
        split[ep["difficulty"]].append(eid)
        for i, (shot, record) in enumerate(zip(ep["shots"], row["shots"])):
            sk = shot["shot_key"]
            assert sk == record["shot_key"] and shot["cut"] == record["cut"]
            video, tail = Path(record["video"]), Path(record["last_frame"])
            assert sha(video) == record["video_sha256"] and sha(tail) == record["last_sha256"]
            original = jobs[eid, sk]
            assert original["video_sha256"] == record["baseline_video_sha256"]
            if not shot["cut"]:
                assert i > 0
                prev = row["shots"][i - 1]
                assert record["parent_video_sha256"] == prev["video_sha256"]
                assert record["parent_tail_sha256"] == prev["last_sha256"]
            scene, local = map(int, sk.split(":"))
            link = OUT / "official_results" / eid / f"{scene:03d}_{local:03d}.mp4"
            assert link.is_symlink() and link.resolve() == video
            inputs[f"{eid}/{sk}"] = {"path": str(video), "sha256": record["video_sha256"],
                                      "baseline_sha256": original["video_sha256"],
                                      "source_kind": record["source_kind"], "route": record["route"]["route"] if record["route"] else None}
    plan = {"scope": "complete selected episodes; frozen official partial_without_vlm; no full140 claim",
            "runs": runs, "episodes": [{"episode_id": e["episode_id"], "run": e["run"],
                                   "tier": e["difficulty"], "num_shots": len(e["shots"])} for e in selected],
            "source_manifest_sha256": sha(BASE / "manifest.json"),
            "source_generation_report_sha256": sha(BASE / "generation_report.json"),
            "evaluator_sha256": sha(EVALUATOR), "style_sha256": sha(OUT / "style_specs.json"),
            "videos": inputs}
    write_once(dst / "input_manifest.json", plan)
    write_once(dst / "split.json", split)
    print("prepared", dst, len(selected), "episodes", len(inputs), "videos")


def run_official(name: str) -> None:
    dst = root(name)
    plan = json.loads((dst / "input_manifest.json").read_text())
    assert sha(EVALUATOR) == plan["evaluator_sha256"]
    assert sha(OUT / "style_specs.json") == plan["style_sha256"]
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
    cmd = [EVAL_PYTHON, "-u", str(EVALUATOR), "--results_dir", str(OUT / "official_results"),
           "--scripts_dir", str(BASE.parents[1] / "benchmarks/entitybench/data/scripts"),
           "--split_json", str(dst / "split.json"), "--out_dir", str(dst / "results"),
           "--method_name", f"qwen140_integrated_{name}", "--pillars", "1,2,3",
           "--skip_vlm", "--resume", "--device", "cuda:0",
           "--vbench_cache_dir", "/root/autodl-tmp/models/entitybench_vbench"]
    with (dst / "evaluation.log").open("a") as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print("evaluated", dst)


def original_result(eid: str) -> dict:
    paths = list((BASE / "evaluation/official_parallel_v2/.shards").glob(f"worker_*/results/{eid}.json"))
    assert len(paths) == 1, (eid, paths)
    return json.loads(paths[0].read_text())


def compare(name: str) -> None:
    dst = root(name)
    plan = json.loads((dst / "input_manifest.json").read_text())
    assert sha(EVALUATOR) == plan["evaluator_sha256"]
    for key, info in plan["videos"].items():
        assert sha(Path(info["path"])) == info["sha256"], key
    rows = []
    for ep in plan["episodes"]:
        eid = ep["episode_id"]
        old = original_result(eid)
        new = json.loads((dst / "results" / f"{eid}.json").read_text())
        assert old["episode_id"] == new["episode_id"] == eid
        assert old["num_shots"] == new["num_shots"] == ep["num_shots"]
        values = {}
        for group in ("vbench_means", "metrics"):
            names = sorted(set(old[group]) | set(new[group]))
            values[group] = {}
            for metric in names:
                a, b = old[group].get(metric, {}), new[group].get(metric, {})
                av, bv = a.get("value"), b.get("value")
                values[group][metric] = {"baseline": av, "candidate": bv,
                                          "delta": bv - av if av is not None and bv is not None else None,
                                          "baseline_coverage": {k: a.get(k) for k in ("n_evaluated", "n_failed", "n_skipped")},
                                          "candidate_coverage": {k: b.get(k) for k in ("n_evaluated", "n_failed", "n_skipped")}}
        rows.append({"episode_id": eid, "run": ep["run"], "tier": ep["tier"],
                     "num_shots": ep["num_shots"], "metrics": values})
    write_once(dst / "comparison.json", {"scope": plan["scope"], "rows": rows})
    for row in rows:
        print(row["run"], row["num_shots"],
              "presence", row["metrics"]["metrics"]["intra_character_presence"],
              "cs_face", row["metrics"]["metrics"]["cs_face"])


def aggregate_baseline(name: str) -> None:
    # Run this action with EVAL_PYTHON. It imports the frozen official implementation,
    # deserializes baseline episode results, then uses its own canonical aggregation.
    dst = root(name)
    plan = json.loads((dst / "input_manifest.json").read_text())
    assert sha(EVALUATOR) == plan["evaluator_sha256"]
    spec = importlib.util.spec_from_file_location("frozen_entitybench", EVALUATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    split = json.loads((dst / "split.json").read_text())
    rows = [module._episode_result_from_json(e["episode_id"], e["tier"], original_result(e["episode_id"]))
            for e in plan["episodes"]]
    report = module.aggregate_results(rows, split)
    write_once(dst / "baseline_official_aggregate.json", report)
    print("aggregated baseline", dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run", "compare", "baseline-aggregate"])
    parser.add_argument("--name", required=True)
    parser.add_argument("--runs", nargs="+", choices=list(TAGS), default=list(TAGS))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.name, args.runs)
    elif args.action == "run":
        run_official(args.name)
    elif args.action == "compare":
        compare(args.name)
    else:
        aggregate_baseline(args.name)
