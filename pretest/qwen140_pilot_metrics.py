#!/usr/bin/env python3
"""Run the frozen VBench implementation on isolated pilot videos only."""

import argparse
import hashlib
import json
from pathlib import Path

from qwen140_pilot_qwen import BASE, OUT, TARGETS, sha

DIMS = ["dynamic_degree", "aesthetic_quality", "imaging_quality"]
CACHE = "/root/autodl-tmp/models/entitybench_vbench"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", choices=list(TARGETS), required=True)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--run298-variant", choices=["v1", "stage_lit_v2"], default="v1")
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    from vbench import VBench

    assert args.run298_variant == "v1" or (args.runs == ["run298"] and not args.baseline_only)
    suffix = "_baseline_check" if args.baseline_only else "" if args.run298_variant == "v1" else "_stage_lit_v2"
    videos = OUT / "evaluation" / ("inputs" + suffix)
    result_dir = OUT / "evaluation" / "results"
    videos.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    selected = {}
    report = json.loads((BASE / "generation_report.json").read_text()) if args.baseline_only else None
    for run in args.runs:
        if args.baseline_only:
            path = Path(next(row["output_video"] for row in report["jobs"] if f"__{run}__" in row["episode_id"] and row["shot_key"] == TARGETS[run][1]))
        else:
            segment = "combined" if run == "run484" else "main_stage_lit_v2" if suffix and run == "run298" else "main"
            path = OUT / "videos" / run / segment / "video.mp4"
        assert path.is_file()
        if not args.baseline_only:
            record = json.loads((path.parent / "generation.json").read_text())
            assert sha(path) == record["video_sha256"]
        link = videos / (run + ".mp4")
        if link.is_symlink():
            assert link.resolve() == path.resolve()
        else:
            assert not link.exists()
            link.symlink_to(path)
        selected[run] = {"video": str(path), "sha256": sha(path)}
    if not args.aggregate_only:
        evaluator = VBench("cuda:0", CACHE, str(result_dir))
        for dim in DIMS:
            evaluator.evaluate(videos_path=str(videos), name=f"qwen140_pilot{suffix}_{dim}",
                               dimension_list=[dim], mode="custom_input", output_path=str(result_dir))
    baseline = json.loads((BASE / "audit/quality_diagnosis_20261026/shot_metrics.json").read_text())
    rows = []
    for run, paths in selected.items():
        shot_key = TARGETS[run][1]
        base_row = next(r for r in baseline if f"__{run}__" in r["episode_id"] and r["shot_key"] == shot_key)
        current = {}
        for dim in DIMS:
            result = json.loads((result_dir / f"qwen140_pilot{suffix}_{dim}_eval_results.json").read_text())
            match = next(r for r in result[dim][1] if Path(r["video_path"]).stem == run)
            current[dim] = float(match["video_results"])
        rows.append({"run": run, "shot_key": shot_key, **paths,
                     "baseline": {dim: base_row[dim] for dim in DIMS}, "candidate": current,
                     "delta": {dim: current[dim] - base_row[dim] for dim in DIMS}})
    if args.baseline_only:
        assert all(abs(value) < 1e-5 for row in rows for value in row["delta"].values()), "Local VBench differs from frozen baseline"
    payload = {"scope": "isolated shot VBench comparison, not episode EntityBench aggregate",
               "vbench_source_sha256": {name: hashlib.sha256((Path(__import__('vbench').__file__).parent / name).read_bytes()).hexdigest()
                                         for name in [f"{d}.py" for d in DIMS]}, "rows": rows}
    dest = OUT / "evaluation" / ("comparison" + suffix + ".json")
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
