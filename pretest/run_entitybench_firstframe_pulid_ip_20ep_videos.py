#!/usr/bin/env python3
"""Resume paired Wan videos and evaluate the frozen 20-episode single-face cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/entitybench_firstframe_pulid_ip_20ep"
OUTPUT = SOURCE / "video_evaluation"
WAN_PYTHON = Path("/root/autodl-tmp/wan22-venv/bin/python")
FACE_PYTHON = ROOT / ".venv/bin/python"
ROUTES = (("PuLID-FLUX", "pulid_output", "pulid_flux"),
          ("IP-Adapter", "ip_output", "ip_adapter"))


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def old_verified_job(source_root: Path, job: dict, input_hash: str) -> dict | None:
    video = Path(job["output_video"])
    if not video.is_file():
        return None
    for report_path in sorted(source_root.glob("wan_report*.json")):
        report = read(report_path)
        settings = report.get("settings", {})
        if any(settings.get(key) != value for key, value in (
            ("size", "1280*704"), ("frame_num", 49), ("sample_steps", 50),
            ("sample_shift", 5.0), ("guide_scale", 5.0),
        )):
            continue
        for prior in report.get("jobs", []):
            if prior.get("job_id") != job["job_id"]:
                continue
            if prior.get("status") not in {
                "generated", "reused_control", "reused_external_verified", "skipped_existing"
            }:
                continue
            if prior.get("input_sha256") != input_hash:
                continue
            if prior.get("seed") != job.get("seed"):
                continue
            video_hash = prior.get("output_sha256")
            if video_hash and sha256(video) == video_hash:
                return {"reuse_video_path": str(video.resolve()),
                        "reuse_verified_input_sha256": input_hash,
                        "reuse_verified_output_sha256": video_hash}
    return None


def prepare() -> dict:
    cohort = read(SOURCE / "manifest.json")
    first = read(SOURCE / "evaluation/firstframe_evaluation.json")
    first_rows = {(row["episode_id"], row["route"], row["shot_key"]): row
                  for row in first["shots"]}
    previous = read(OUTPUT / "wan_report.json") if (OUTPUT / "wan_report.json").is_file() else {}
    previous_jobs = {job["job_id"]: job for job in previous.get("jobs", [])}
    lock_path = OUTPUT / "first_frame_hashes.json"
    locked = read(lock_path) if lock_path.is_file() else {}
    new_lock = {}
    all_jobs = []
    cases = []
    counts = {"verified_external_reuse": 0, "verified_resume": 0, "to_generate": 0,
              "reuse_control": 0}
    for episode in cohort["episodes"]:
        episode_id = episode["episode_id"]
        for route, key, route_slug in ROUTES:
            source_root = Path(episode[key])
            original = read(source_root / "wan_manifest_single_character.json")
            expected = episode["single_character_shot_count"] * 2
            if len(original["jobs"]) != expected:
                raise ValueError(f"{episode_id} {route}: expected {expected} jobs")
            case_jobs = []
            for item in original["jobs"]:
                item = dict(item)
                source_id = item["job_id"]
                unified_id = f"{episode_id}__{route_slug}__{source_id}"
                image = Path(item["input_image"])
                if not image.is_file():
                    raise FileNotFoundError(image)
                input_hash = sha256(image)
                if unified_id in locked and locked[unified_id] != input_hash:
                    raise ValueError(f"{unified_id}: frozen first frame changed")
                new_lock[unified_id] = input_hash
                item["expected_input_sha256"] = input_hash
                item["output_video"] = str(
                    OUTPUT / "videos" / episode_id / route_slug
                    / f"shot_{item['shot_key'].replace(':', '_')}" / f"{item['condition']}.mp4"
                )
                first_row = first_rows[(episode_id, route, item["shot_key"])]
                bbox = first_row.get(f"{item['condition']}_bbox")
                if bbox:
                    item["target_initial_bbox"] = bbox
                    item["target_initial_image_size"] = [640, 352]
                if item["condition"] == "treatment" and item.get("reuse_video_from"):
                    if input_hash != new_lock.get(f"{episode_id}__{route_slug}__{item['reuse_video_from']}"):
                        raise ValueError(f"{unified_id}: Control-reuse frame differs")
                    counts["reuse_control"] += 1
                output = Path(item["output_video"])
                prior = previous_jobs.get(unified_id)
                if output.is_file():
                    if not prior or prior.get("input_sha256") != input_hash \
                            or prior.get("output_sha256") != sha256(output):
                        raise ValueError(f"{unified_id}: existing video lacks matching provenance")
                    item["expected_output_sha256"] = prior["output_sha256"]
                    counts["verified_resume"] += 1
                elif not item.get("reuse_video_from"):
                    verified = old_verified_job(source_root, original_job_by_id(original, source_id), input_hash)
                    if verified:
                        item.update(verified)
                        counts["verified_external_reuse"] += 1
                    else:
                        counts["to_generate"] += 1
                case_jobs.append(item)
                combined = dict(item)
                combined["job_id"] = unified_id
                if combined.get("reuse_video_from"):
                    combined["reuse_video_from"] = (
                        f"{episode_id}__{route_slug}__{combined['reuse_video_from']}"
                    )
                all_jobs.append(combined)
            case_path = OUTPUT / "cases" / episode_id / route_slug / "wan_manifest.json"
            write(case_path, {"kind": "entitybench_20ep_single_character_wan_case",
                              "episode_id": episode_id,
                              "identity_references": original["identity_references"],
                              "jobs": case_jobs})
            cases.append({"episode_id": episode_id, "run": episode["run"],
                          "difficulty": episode["difficulty"], "route": route,
                          "route_slug": route_slug, "manifest": str(case_path),
                          "identity_report": str(case_path.parent / "video_identity.json")})
    if len(all_jobs) != 528:
        raise ValueError(f"Expected 528 jobs, got {len(all_jobs)}")
    write(OUTPUT / "wan_manifest.json",
          {"kind": "entitybench_20ep_single_character_paired_wan",
           "jobs": all_jobs})
    write(OUTPUT / "cases.json", cases)
    write(lock_path, new_lock)
    write(OUTPUT / "progress.json", {"phase": "prepared", "jobs": len(all_jobs), **counts})
    print(json.dumps({"jobs": len(all_jobs), **counts}, ensure_ascii=False), flush=True)
    return counts


def original_job_by_id(manifest: dict, job_id: str) -> dict:
    return next(job for job in manifest["jobs"] if job["job_id"] == job_id)


def run_command(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def run_wan() -> None:
    run_command([str(WAN_PYTHON), "pretest/run_wan22_i2v_manifest.py",
                 "--manifest", str(OUTPUT / "wan_manifest.json"),
                 "--report", str(OUTPUT / "wan_report.json"),
                 "--continue-on-error"])
    report = read(OUTPUT / "wan_report.json")
    if report.get("failed_jobs") or report.get("completed_jobs") != 528:
        raise RuntimeError(f"Wan incomplete: {report.get('completed_jobs')}/528, "
                           f"failed={report.get('failed_jobs')}")
    outputs = {job["job_id"]: job for job in report["jobs"]}
    for job in read(OUTPUT / "wan_manifest.json")["jobs"]:
        source = job.get("reuse_video_from")
        if source and outputs[job["job_id"]]["output_sha256"] != outputs[source]["output_sha256"]:
            raise ValueError(f"{job['job_id']}: Control-reuse video hash differs")
    write(OUTPUT / "progress.json", {"phase": "wan_complete", "jobs": 528})


def evaluate() -> None:
    cases = read(OUTPUT / "cases.json")
    if len(cases) != 40:
        raise ValueError(f"Expected 40 cases, got {len(cases)}")
    for index, case in enumerate(cases, 1):
        report_path = Path(case["identity_report"])
        manifest = read(Path(case["manifest"]))
        if report_path.is_file():
            prior = read(report_path)
            if prior.get("tracking_policy") == "spatial" and len(prior.get("jobs", [])) == len(manifest["jobs"]):
                if any(job["video"]["summary"]["decoded_frame_count"] != 49 for job in prior["jobs"]):
                    raise ValueError(f"{report_path}: expected 49 frames in every video")
                print(f"[insightface] {index}/40 reuse {case['run']} {case['route']}", flush=True)
                continue
        run_command([str(FACE_PYTHON), "pretest/evaluate_video_identity.py",
                     "--manifest", case["manifest"], "--output-json", str(report_path),
                     "--tracking-policy", "spatial"])
        completed = read(report_path)
        if len(completed["jobs"]) != len(manifest["jobs"]) or any(
            job["video"]["summary"]["decoded_frame_count"] != 49
            for job in completed["jobs"]
        ):
            raise ValueError(f"{report_path}: incomplete or wrong-length video evaluation")
        print(f"[insightface] {index}/40 {case['run']} {case['route']}", flush=True)
        write(OUTPUT / "progress.json", {"phase": "insightface", "completed_cases": index,
                                         "total_cases": len(cases)})
    dino_command = [str(FACE_PYTHON), "pretest/evaluate_reference_dino_identity.py"]
    for case in cases:
        dino_command += ["--case", case["route"], case["manifest"], case["identity_report"]]
    dino_command += ["--output-json", str(OUTPUT / "dinov2_video.json"),
                     "--output-md", str(OUTPUT / "dinov2_video.md")]
    run_command(dino_command)
    summarize()


def aggregate(rows: list[tuple[float, float]]) -> dict:
    if not rows:
        return {"valid_pair_count": 0, "control_mean": None,
                "treatment_mean": None, "mean_paired_delta": None}
    count = len(rows)
    return {"valid_pair_count": count,
            "control_mean": round(sum(a for a, _ in rows) / count, 6),
            "treatment_mean": round(sum(b for _, b in rows) / count, 6),
            "mean_paired_delta": round(sum(b - a for a, b in rows) / count, 6),
            "positive_delta_count": sum(b > a for a, b in rows),
            "negative_delta_count": sum(b < a for a, b in rows)}


def summarize() -> None:
    cases = read(OUTPUT / "cases.json")
    first = read(SOURCE / "evaluation/firstframe_evaluation.json")
    dino = read(OUTPUT / "dinov2_video.json")
    dino_cases = {(case["episode_id"], case["route"]): case for case in dino["cases"]}
    details = []
    for case in cases:
        report = read(Path(case["identity_report"]))
        manifest = read(Path(case["manifest"]))
        lookup = {job["job_id"]: job for job in report["jobs"]}
        dino_lookup = {(pair["shot_key"], pair["target_character"]): pair
                       for pair in dino_cases[(case["episode_id"], case["route"])]["pairs"]}
        for idx in range(0, len(manifest["jobs"]), 2):
            left, right = manifest["jobs"][idx:idx + 2]
            if left["condition"] != "control" or right["condition"] != "treatment":
                raise ValueError("Manifest pairing order changed")
            ic = lookup[left["job_id"]]["video"]["summary"]["mean"]
            it = lookup[right["job_id"]]["video"]["summary"]["mean"]
            dp = dino_lookup[(left["shot_key"], left["target_character"])]
            dc = dp["control"].get("video_frame_mean")
            dt = dp["treatment"].get("video_frame_mean")
            details.append({"episode_id": case["episode_id"], "run": case["run"],
                            "difficulty": case["difficulty"], "route": case["route"],
                            "shot_key": left["shot_key"], "character": left["target_character"],
                            "plugin_applied": not bool(right.get("reuse_video_from")),
                            "control_video": left["output_video"],
                            "treatment_video": right["output_video"],
                            "insightface_control": ic, "insightface_treatment": it,
                            "dinov2_control": dc, "dinov2_treatment": dt})
    if len(details) != 264:
        raise ValueError(f"Expected 264 paired route-shot rows, got {len(details)}")
    routes = {}
    for route, _, _ in ROUTES:
        selected = [row for row in details if row["route"] == route and row["plugin_applied"]]
        if_pairs = [(row["insightface_control"], row["insightface_treatment"])
                    for row in selected if row["insightface_control"] is not None
                    and row["insightface_treatment"] is not None]
        dino_pairs = [(row["dinov2_control"], row["dinov2_treatment"])
                      for row in selected if row["dinov2_control"] is not None
                      and row["dinov2_treatment"] is not None]
        routes[route] = {"single_shots": 132, "injected": len(selected),
                         "reused_control": 132 - len(selected),
                         "firstframe_insightface": first["routes"][route]["insightface"],
                         "firstframe_dinov2": first["routes"][route]["reference_dinov2"],
                         "video_insightface": aggregate(if_pairs),
                         "video_dinov2": aggregate(dino_pairs)}
    episodes = {}
    for case in cases:
        episode_id = case["episode_id"]
        if episode_id in episodes:
            continue
        episode_rows = [row for row in details if row["episode_id"] == episode_id]
        episode_summary = {"run": case["run"], "difficulty": case["difficulty"], "routes": {}}
        for route, _, _ in ROUTES:
            selected = [row for row in episode_rows if row["route"] == route and row["plugin_applied"]]
            episode_summary["routes"][route] = {
                "injected": len(selected),
                "reused_control": sum(row["route"] == route and not row["plugin_applied"]
                                      for row in episode_rows),
                "video_insightface": aggregate([
                    (row["insightface_control"], row["insightface_treatment"])
                    for row in selected if row["insightface_control"] is not None
                    and row["insightface_treatment"] is not None]),
                "video_dinov2": aggregate([
                    (row["dinov2_control"], row["dinov2_treatment"])
                    for row in selected if row["dinov2_control"] is not None
                    and row["dinov2_treatment"] is not None]),
            }
        episodes[episode_id] = episode_summary
    result = {"kind": "entitybench_20ep_single_character_paired_video_evaluation",
              "scope": "20 episodes; 132 single-character shots per route; 49-frame Wan clips",
              "tracking_policy": "initial face bbox then spatial nearest-neighbor",
              "routes": routes, "episodes": episodes, "shots": details}
    write(OUTPUT / "video_evaluation.json", result)
    lines = ["# EntityBench 20-episode paired video evaluation", "",
             "49-frame Wan2.2-TI2V-5B videos, 24 fps. Means cover injected shots with both paired values.", "",
             "| Route | Single shots | Injected | Reused Control | First-frame IF C→T (Δ) | First-frame DINOv2 C→T (Δ) | Video IF mean C→T (Δ) | Video DINOv2 mean C→T (Δ) |",
             "|---|---:|---:|---:|---|---|---|---|"]
    def fmt(value: dict) -> str:
        if value["valid_pair_count"] == 0:
            return "—"
        return f"{value['control_mean']:.4f}→{value['treatment_mean']:.4f} ({value['mean_paired_delta']:+.4f}; n={value['valid_pair_count']})"
    for route in routes:
        row = routes[route]
        lines.append(f"| {route} | 132 | {row['injected']} | {row['reused_control']} | "
                     f"{fmt(row['firstframe_insightface'])} | {fmt(row['firstframe_dinov2'])} | "
                     f"{fmt(row['video_insightface'])} | "
                     f"{fmt(row['video_dinov2'])} |")
    lines += ["", "## Per-episode video results", "",
              "| Episode | Difficulty | Route | Injected | Reused Control | Video IF mean C→T (Δ) | Video DINOv2 mean C→T (Δ) |",
              "|---|---|---|---:|---:|---|---|"]
    for episode in episodes.values():
        for route, row in episode["routes"].items():
            lines.append(f"| {episode['run']} | {episode['difficulty']} | {route} | "
                         f"{row['injected']} | {row['reused_control']} | "
                         f"{fmt(row['video_insightface'])} | {fmt(row['video_dinov2'])} |")
    lines += ["", "Video identity uses spatial tracking from the frozen first-frame face box.",
              "Missing face scores are excluded from similarity means and retained in per-frame detection coverage.", ""]
    (OUTPUT / "video_evaluation.md").write_text("\n".join(lines), encoding="utf-8")
    write(OUTPUT / "progress.json", {"phase": "complete", "episodes": 20,
                                     "paired_route_shots": 264, "jobs": 528})
    print("\n".join(lines), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "wan", "evaluate", "summarize", "all"))
    args = parser.parse_args()
    if args.stage in ("prepare", "all"):
        prepare()
    if args.stage in ("wan", "all"):
        run_wan()
    if args.stage in ("evaluate", "all"):
        evaluate()
    if args.stage == "summarize":
        summarize()


if __name__ == "__main__":
    main()
