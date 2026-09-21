"""Run a resumable Wan2.2-TI2V-5B I2V manifest with one model load.

This runner is intentionally separate from the first-frame generator.  Each
manifest row names an already frozen Control or Treatment first frame, which
keeps the video backbone identical across the paired comparison.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _path(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def _validate_manifest(data: dict, manifest_path: Path) -> list[dict]:
    jobs = data.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Manifest must contain a non-empty jobs list")
    ids = set()
    for index, job in enumerate(jobs):
        job_id = job.get("job_id")
        if not job_id or job_id in ids:
            raise ValueError(f"Invalid or duplicate job_id at row {index}: {job_id!r}")
        ids.add(job_id)
        for key in ("input_image", "output_video"):
            if not job.get(key):
                raise ValueError(f"{job_id}: missing {key}")
        input_path = _path(job["input_image"], manifest_path)
        if not input_path.is_file():
            raise FileNotFoundError(f"{job_id}: input image is missing: {input_path}")
        if not job.get("reuse_video_from") and not job.get("prompt"):
            raise ValueError(f"{job_id}: prompt is required for generated videos")
    return jobs


def run(args) -> dict:
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = _validate_manifest(manifest, manifest_path)
    if args.limit_jobs is not None:
        jobs = jobs[: args.limit_jobs]
    if args.frame_num < 1 or (args.frame_num - 1) % 4:
        raise ValueError("Wan frame_num must have the form 4n+1")

    wan_root = Path(args.wan_root).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    if not (wan_root / "wan" / "textimage2video.py").is_file():
        raise FileNotFoundError(f"Wan source tree is missing: {wan_root}")
    if not (checkpoint_dir / "config.json").is_file():
        raise FileNotFoundError(f"Wan checkpoint is missing: {checkpoint_dir}")
    sys.path.insert(0, str(wan_root))

    import wan  # noqa: E402
    from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS  # noqa: E402
    from wan.utils.utils import save_video  # noqa: E402

    report_path = Path(args.report).resolve()
    report = {
        "kind": "wan22_ti2v_5b_manifest_run",
        "manifest": str(manifest_path),
        "wan_root": str(wan_root),
        "wan_commit": args.wan_commit,
        "checkpoint_dir": str(checkpoint_dir),
        "settings": {
            "size": args.size,
            "frame_num": args.frame_num,
            "fps": 24,
            "sample_steps": args.sample_steps,
            "sample_shift": args.sample_shift,
            "guide_scale": args.guide_scale,
            "base_seed": args.base_seed,
            "offload_model": True,
            "convert_model_dtype": True,
            "t5_cpu": True,
        },
        "jobs": [],
    }

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )
    cfg = WAN_CONFIGS["ti2v-5B"]
    logging.info("Loading Wan2.2-TI2V-5B once for %d manifest jobs", len(jobs))
    pipeline = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=str(checkpoint_dir),
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=True,
        convert_model_dtype=True,
    )
    if torch.cuda.is_available():
        report["cuda_memory_after_model_load_bytes"] = {
            "allocated": int(torch.cuda.memory_allocated()),
            "reserved": int(torch.cuda.memory_reserved()),
        }

    completed_outputs: dict[str, Path] = {}
    started = time.perf_counter()
    for index, job in enumerate(jobs):
        job_started = time.perf_counter()
        job_id = job["job_id"]
        input_path = _path(job["input_image"], manifest_path)
        output_path = _path(job["output_video"], manifest_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        seed = int(job.get("seed", args.base_seed + int(job.get("shot_index", index))))
        record = {
            "job_id": job_id,
            "episode_id": job.get("episode_id"),
            "shot_key": job.get("shot_key"),
            "condition": job.get("condition"),
            "input_image": str(input_path),
            "input_sha256": _sha256(input_path),
            "output_video": str(output_path),
            "seed": seed,
        }
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            if output_path.is_file() and output_path.stat().st_size > 0 and not args.overwrite:
                record["status"] = "skipped_existing"
            elif job.get("reuse_video_from"):
                source_job_id = job["reuse_video_from"]
                source_path = completed_outputs.get(source_job_id)
                if source_path is None:
                    source_job = next((item for item in jobs if item["job_id"] == source_job_id), None)
                    if source_job is None:
                        raise KeyError(f"reuse source job not found: {source_job_id}")
                    source_path = _path(source_job["output_video"], manifest_path)
                if not source_path.is_file():
                    raise FileNotFoundError(f"reuse source video is missing: {source_path}")
                shutil.copy2(source_path, output_path)
                record["status"] = "reused_control"
                record["reuse_video_from"] = source_job_id
            else:
                image = Image.open(input_path).convert("RGB")
                video = pipeline.generate(
                    job["prompt"],
                    img=image,
                    size=SIZE_CONFIGS[args.size],
                    max_area=MAX_AREA_CONFIGS[args.size],
                    frame_num=args.frame_num,
                    shift=args.sample_shift,
                    sample_solver="unipc",
                    sampling_steps=args.sample_steps,
                    guide_scale=args.guide_scale,
                    seed=seed,
                    offload_model=True,
                )
                save_video(
                    tensor=video[None],
                    save_file=str(output_path),
                    fps=cfg.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )
                del video
                gc.collect()
                torch.cuda.empty_cache()
                if not output_path.is_file() or output_path.stat().st_size == 0:
                    raise RuntimeError(f"Wan save_video did not produce {output_path}")
                record["status"] = "generated"
                if torch.cuda.is_available():
                    record["cuda_peak_allocated_bytes"] = int(
                        torch.cuda.max_memory_allocated()
                    )
                    record["cuda_peak_reserved_bytes"] = int(
                        torch.cuda.max_memory_reserved()
                    )
            completed_outputs[job_id] = output_path
            record["output_bytes"] = output_path.stat().st_size
            record["output_sha256"] = _sha256(output_path)
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = repr(exc)
            report["jobs"].append(record)
            report["elapsed_seconds"] = time.perf_counter() - started
            _write_json(report_path, report)
            if not args.continue_on_error:
                raise
        else:
            record["elapsed_seconds"] = time.perf_counter() - job_started
            report["jobs"].append(record)
            report["elapsed_seconds"] = time.perf_counter() - started
            _write_json(report_path, report)
            logging.info(
                "%s %s in %.1fs -> %s",
                job_id,
                record["status"],
                record["elapsed_seconds"],
                output_path,
            )

    report["elapsed_seconds"] = time.perf_counter() - started
    report["completed_jobs"] = sum(
        item["status"] in {"generated", "reused_control", "skipped_existing"}
        for item in report["jobs"]
    )
    report["failed_jobs"] = sum(item["status"] == "failed" for item in report["jobs"])
    _write_json(report_path, report)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--wan-root", default="/root/autodl-tmp/Wan2.2")
    parser.add_argument(
        "--wan-commit", default="42bf4cfaa384bc21833865abc2f9e6c0e67233dc"
    )
    parser.add_argument(
        "--checkpoint-dir", default=str(PROJECT_ROOT / "models/video/Wan2.2-TI2V-5B")
    )
    parser.add_argument("--size", default="1280*704", choices=("1280*704", "704*1280"))
    parser.add_argument("--frame-num", type=int, default=49)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--sample-shift", type=float, default=5.0)
    parser.add_argument("--guide-scale", type=float, default=5.0)
    parser.add_argument("--base-seed", type=int, default=719000)
    parser.add_argument("--limit-jobs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
