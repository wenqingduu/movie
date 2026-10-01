#!/usr/bin/env python3
"""Run a resumable 20-episode PuLID-FLUX/IP-Adapter first-frame batch.

The cohort is frozen before generation: four existing episodes plus a
deterministic stratified sample of seven Easy, six Medium, and three Hard
episodes.  Only single-character shots are generated for the two routes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/entitybench_firstframe_pulid_ip_20ep"
SCRIPTS_ROOT = PROJECT_ROOT / "benchmarks/entitybench/data/scripts"
SPLIT_PATH = PROJECT_ROOT / "benchmarks/entitybench/data/splits/final_split_validated_ids.json"
SELECTION_SEED = 20260929
ADDED_QUOTAS = {"easy": 7, "medium": 6, "hard": 3}

EXISTING = {
    "00053051-5f7e-314f-85e0-517ec18f3b08__run719__i35_j44__T120": {
        "base_seed": 719000,
        "asset_root": PROJECT_ROOT / "outputs/entitybench_wan22_smoke/episode_00053051/assets",
        "pulid_output": PROJECT_ROOT / "outputs/entitybench_wan22_colornested_v7_s04_12/episode_00053051",
        "ip_output": PROJECT_ROOT / "outputs/entitybench_wan22_ip_adapter_v7_s04_12/episode_00053051",
    },
    "000785f7-48d3-3a68-8ba7-82566cd6d77c__run893__i95_j118__T300": {
        "base_seed": 893000,
        "asset_root": PROJECT_ROOT / "outputs/entitybench_pilot3/episode_run893/assets",
        "pulid_output": PROJECT_ROOT / "outputs/entitybench_wan22_pulid_v7_pilot3/episode_run893",
        "ip_output": PROJECT_ROOT / "outputs/entitybench_wan22_ip_adapter_v7_pilot3/episode_run893",
    },
    "000cf326-8770-3c4e-a43a-6f9e3d57a3e0__run1517__i15_j40__T300": {
        "base_seed": 1517000,
        "asset_root": PROJECT_ROOT / "outputs/entitybench_pilot3/episode_run1517/assets",
        "pulid_output": PROJECT_ROOT / "outputs/entitybench_wan22_pulid_v7_pilot3/episode_run1517",
        "ip_output": PROJECT_ROOT / "outputs/entitybench_wan22_ip_adapter_v7_pilot3/episode_run1517",
    },
    "000ec577-e87f-320f-a6c1-43adf38af0ce__run1775__i70_j85__T180": {
        "base_seed": 1775000,
        "asset_root": PROJECT_ROOT / "outputs/entitybench_large_face_prompt_pilot/run1775/assets",
        "pulid_output": PROJECT_ROOT / "outputs/entitybench_large_face_prompt_pilot/run1775",
        "ip_output": PROJECT_ROOT / "outputs/entitybench_large_face_prompt_pilot_ip_adapter/run1775",
    },
}


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _slug(value: str) -> str:
    return "_".join(value.lower().replace("-", " ").split())


def _run_name(episode_id: str) -> str:
    match = re.search(r"__run(\d+)__", episode_id)
    if not match:
        raise ValueError(f"Cannot parse run from {episode_id}")
    return f"run{match.group(1)}"


def _ordered_shots(episode: dict[str, Any]) -> list[dict[str, Any]]:
    shots = []
    for scene in episode.get("scenes", []):
        prompts = scene.get("video_prompts", [])
        for local_index, prompt in enumerate(prompts, 1):
            key = f"{scene['scene_num']}:{local_index}"
            shots.append({
                "shot_key": key,
                "prompt": prompt,
                "characters": episode.get("entity_schedule", {}).get(key, {}).get("characters", []) or [],
            })
    return shots


def _episode_stats(path: Path) -> dict[str, Any]:
    episode = _load(path)
    shots = _ordered_shots(episode)
    singles = [item for item in shots if len(item["characters"]) == 1]
    return {
        "shot_count": len(shots),
        "single_character_shot_count": len(singles),
        "multi_character_shot_count": sum(len(item["characters"]) > 1 for item in shots),
        "zero_character_shot_count": sum(not item["characters"] for item in shots),
        "single_character_asset_count": len({item["characters"][0] for item in singles}),
        "single_character_names": sorted({item["characters"][0] for item in singles}),
    }


def build_manifest(output_root: Path, rebuild: bool = False) -> dict[str, Any]:
    path = output_root / "manifest.json"
    if path.is_file() and not rebuild:
        return _load(path)
    split = _load(SPLIT_PATH)
    lookup = {item.stem: item for item in SCRIPTS_ROOT.glob("*.json")}
    difficulty = {
        episode_id: level
        for level in ("easy", "medium", "hard")
        for episode_id in split[level]
    }
    rng = random.Random(SELECTION_SEED)
    selected_new: list[tuple[str, str]] = []
    for level, count in ADDED_QUOTAS.items():
        candidates = []
        for episode_id in split[level]:
            if episode_id in EXISTING:
                continue
            stats = _episode_stats(lookup[episode_id])
            if stats["single_character_shot_count"]:
                candidates.append(episode_id)
        rng.shuffle(candidates)
        selected_new.extend((level, episode_id) for episode_id in candidates[:count])

    records = []
    for episode_id, existing in EXISTING.items():
        episode_path = lookup[episode_id].resolve()
        records.append({
            "episode_id": episode_id,
            "run": _run_name(episode_id),
            "difficulty": difficulty[episode_id],
            "source": "existing",
            "episode_path": str(episode_path),
            "base_seed": existing["base_seed"],
            "asset_root": str(existing["asset_root"].resolve()),
            "pulid_output": str(existing["pulid_output"].resolve()),
            "ip_output": str(existing["ip_output"].resolve()),
            **_episode_stats(episode_path),
        })
    for index, (level, episode_id) in enumerate(selected_new, 1):
        episode_path = lookup[episode_id].resolve()
        episode_root = output_root / "episodes" / episode_id
        records.append({
            "episode_id": episode_id,
            "run": _run_name(episode_id),
            "difficulty": level,
            "source": "new",
            "episode_path": str(episode_path),
            "base_seed": 20_000_000 + index * 1_000,
            "asset_root": str((episode_root / "assets").resolve()),
            "pulid_output": str((episode_root / "pulid_flux").resolve()),
            "ip_output": str((episode_root / "ip_adapter").resolve()),
            **_episode_stats(episode_path),
        })
    manifest = {
        "kind": "entitybench_firstframe_pulid_ip_stratified_20_episode_manifest",
        "created_at_unix": time.time(),
        "selection_seed": SELECTION_SEED,
        "selection_policy": (
            "four fixed existing Easy episodes plus deterministic random selection "
            "of 7 Easy, 6 Medium, and 3 Hard episodes with at least one single-character shot"
        ),
        "scope": "single-character first frames only; PuLID-FLUX and IP-Adapter; no Wan video",
        "settings": {
            "width": 640,
            "height": 352,
            "steps": 50,
            "fork_step": 30,
            "injection_strength": 0.4,
            "max_active_injection_steps": 12,
            "minimum_injection_face_height_px": 24,
            "maximum_face_detection_retries": 4,
            "ip_adapter_scale": 0.2,
            "pulid_id_weight": 1.0,
            "mask_policy": "v7 color-safe",
        },
        "episodes": records,
        "totals": {
            "episode_count": len(records),
            "existing_episode_count": sum(item["source"] == "existing" for item in records),
            "new_episode_count": sum(item["source"] == "new" for item in records),
            "shot_count": sum(item["shot_count"] for item in records),
            "single_character_shot_count": sum(item["single_character_shot_count"] for item in records),
            "new_single_character_shot_count": sum(
                item["single_character_shot_count"] for item in records if item["source"] == "new"
            ),
            "new_single_character_asset_count": sum(
                item["single_character_asset_count"] for item in records if item["source"] == "new"
            ),
            "difficulty_episode_counts": {
                level: sum(item["difficulty"] == level for item in records)
                for level in ("easy", "medium", "hard")
            },
        },
    }
    _write(path, manifest)
    return manifest


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", ""),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "1",
    })
    return env


def _run_logged(command: list[str], log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] $ {' '.join(command)}\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=_environment(),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "command": command,
        "return_code": completed.returncode,
        "elapsed_seconds": round(time.time() - started, 3),
        "log": str(log_path.resolve()),
    }


def _asset_ready(record: dict[str, Any]) -> bool:
    root = Path(record["asset_root"])
    for name in record["single_character_names"]:
        slug = _slug(name)
        reference = root / "characters" / f"{slug}_reference.png"
        result_path = root / "faces_3d" / slug / "facelift_result.json"
        if not reference.is_file() or not result_path.is_file():
            return False
        result = _load(result_path)
        model = Path(result["model_path"])
        if not model.is_file() or not model.with_suffix(".pose_calibration.json").is_file():
            return False
    return True


def _reported_single_shots(report: dict[str, Any]) -> int:
    """Count only single-character records in legacy full-episode reports."""

    return sum(
        len(item.get("characters", [])) == 1
        for item in report.get("shots", [])
    )


def run_assets(manifest: dict[str, Any], output_root: Path, workers: int) -> list[dict[str, Any]]:
    records = [item for item in manifest["episodes"] if item["source"] == "new"]

    def task(record: dict[str, Any]) -> dict[str, Any]:
        status_path = output_root / "status/assets" / f"{record['episode_id']}.json"
        if _asset_ready(record):
            result = {"episode_id": record["episode_id"], "status": "ready_existing"}
            _write(status_path, result)
            return result
        command = [
            str(PROJECT_ROOT / ".venv/bin/python"),
            "pretest/prepare_entitybench_character_assets.py",
            "--episode", record["episode_path"],
            "--asset-root", record["asset_root"],
            "--stage", "all",
            "--base-seed", str(record["base_seed"]),
            "--character-scope", "single-shot",
            "--candidates", "3",
            "--size", "1024",
            "--steps", "30",
            "--guidance", "5.0",
            "--calibration-size", "512",
            "--python", str(PROJECT_ROOT / ".venv/bin/python"),
        ]
        run = _run_logged(command, output_root / "logs/assets" / f"{record['episode_id']}.log")
        result = {
            "episode_id": record["episode_id"],
            "status": "ready" if run["return_code"] == 0 and _asset_ready(record) else "failed",
            **run,
        }
        _write(status_path, result)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(task, record) for record in records]
        results = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"[assets] {result['episode_id']} {result['status']}", flush=True)
    return results


def _pair_command(route: str, record: dict[str, Any]) -> list[str]:
    common = [
        "--episode", record["episode_path"],
        "--asset-root", record["asset_root"],
        "--width", "640",
        "--height", "352",
        "--base-seed", str(record["base_seed"]),
        "--injection-strength", "0.4",
        "--max-active-injection-steps", "12",
        "--min-injection-face-height", "24",
        "--max-face-detection-retries", "4",
        "--continue-on-error",
    ]
    if route == "pulid_flux":
        return [
            str(PROJECT_ROOT / ".venv/bin/python"),
            "pretest/prepare_entitybench_pulid_pairs.py",
            "--output-dir", record["pulid_output"],
            *common,
            "--guidance", "4.0",
            "--pulid-id-weight", "1.0",
            "--onnx-provider", "cpu",
            "--shared-model-process",
        ]
    return [
        str(PROJECT_ROOT / ".venv/bin/python"),
        "pretest/prepare_entitybench_ip_adapter_pairs.py",
        "--output-dir", record["ip_output"],
        *common,
        "--steps", "50",
        "--fork-step", "30",
        "--ip-adapter-scale", "0.2",
        "--single-character-only",
    ]


def run_pairs(manifest: dict[str, Any], output_root: Path) -> list[dict[str, Any]]:
    records = [item for item in manifest["episodes"] if item["source"] == "new"]

    def route_worker(route: str) -> list[dict[str, Any]]:
        results = []
        for record in records:
            status_path = output_root / "status/pairs" / route / f"{record['episode_id']}.json"
            if not _asset_ready(record):
                result = {"episode_id": record["episode_id"], "route": route, "status": "skipped_assets_not_ready"}
                _write(status_path, result)
                results.append(result)
                continue
            report_name = "pulid_first_frame_report.json" if route == "pulid_flux" else "ip_adapter_first_frame_report.json"
            report = Path(record["pulid_output"] if route == "pulid_flux" else record["ip_output"]) / report_name
            expected = record["single_character_shot_count"]
            if report.is_file() and _reported_single_shots(_load(report)) == expected:
                result = {"episode_id": record["episode_id"], "route": route, "status": "complete_existing", "report": str(report)}
                _write(status_path, result)
                results.append(result)
                print(f"[{route}] {record['episode_id']} complete_existing", flush=True)
                continue
            run = _run_logged(
                _pair_command(route, record),
                output_root / "logs/pairs" / route / f"{record['episode_id']}.log",
            )
            complete = report.is_file() and _reported_single_shots(_load(report)) == expected
            result = {
                "episode_id": record["episode_id"],
                "route": route,
                "status": "complete" if run["return_code"] == 0 and complete else "failed",
                "report": str(report.resolve()),
                **run,
            }
            _write(status_path, result)
            results.append(result)
            print(f"[{route}] {record['episode_id']} {result['status']}", flush=True)
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        pulid = pool.submit(route_worker, "pulid_flux")
        ip = pool.submit(route_worker, "ip_adapter")
        return pulid.result() + ip.result()


def status(manifest: dict[str, Any], output_root: Path) -> dict[str, Any]:
    rows = []
    for record in manifest["episodes"]:
        row = {
            "episode_id": record["episode_id"],
            "difficulty": record["difficulty"],
            "source": record["source"],
            "single_character_shot_count": record["single_character_shot_count"],
            "assets_ready": _asset_ready(record),
        }
        for route, report_name, key in (
            ("pulid_flux", "pulid_first_frame_report.json", "pulid_output"),
            ("ip_adapter", "ip_adapter_first_frame_report.json", "ip_output"),
        ):
            report_path = Path(record[key]) / report_name
            route_report = _load(report_path) if report_path.is_file() else {}
            reported_single_shots = _reported_single_shots(route_report)
            row[route] = {
                "report": str(report_path),
                "reported_shots": len(route_report.get("shots", [])),
                "reported_single_shots": reported_single_shots,
                "complete_for_single_scope": reported_single_shots == record["single_character_shot_count"],
                "injected_shots": route_report.get("injected_shots"),
                "control_reuse_shots": route_report.get("control_reuse_shots"),
                "failed_shots": route_report.get("failed_shots"),
            }
        rows.append(row)
    value = {
        "kind": "entitybench_firstframe_pulid_ip_batch_status",
        "updated_at_unix": time.time(),
        "manifest": str((output_root / "manifest.json").resolve()),
        "episodes": rows,
        "summary": {
            "episode_count": len(rows),
            "assets_ready": sum(row["assets_ready"] for row in rows),
            "pulid_complete": sum(row["pulid_flux"]["complete_for_single_scope"] for row in rows),
            "ip_complete": sum(row["ip_adapter"]["complete_for_single_scope"] for row in rows),
        },
    }
    _write(output_root / "status.json", value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("manifest", "assets", "pairs", "all", "status"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--asset-workers", type=int, default=2)
    parser.add_argument("--rebuild-manifest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    manifest = build_manifest(args.output_root, args.rebuild_manifest)
    if args.stage in {"assets", "all"}:
        run_assets(manifest, args.output_root, args.asset_workers)
        status(manifest, args.output_root)
    if args.stage in {"pairs", "all"}:
        run_pairs(manifest, args.output_root)
        status(manifest, args.output_root)
    value = status(manifest, args.output_root)
    print(json.dumps(value["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
