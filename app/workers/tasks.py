from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

from app.services import store
from multishot.product_gpu import product_gpu_lock

from .celery_app import celery_app


@celery_app.task(name="movie.generate_project", queue="gpu")
def generate_project(project_id: str) -> dict[str, Any]:
    with product_gpu_lock():
        return _generate_project(project_id)


def _generate_project(project_id: str) -> dict[str, Any]:
    project = store.get_project(project_id)
    if project is None:
        return {"ok": False, "error": f"Unknown project: {project_id}"}

    project_dir = Path(project["project_dir"])
    try:
        from multishot.product_pipeline import (
            assemble_videos,
            generate_shot_videos,
            run_story_project,
        )

        store.update_project(
            project_id,
            status="running",
            message="Generating script, assets, 3D faces, and first frames",
            error="",
        )
        result = run_story_project(
            project["story"],
            project_dir,
            backend=project.get("backend") or "qwen_image21",
            write_manifest=True,
        )

        plan_path = Path(result["project_plan_path"])
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        shot_ids = [shot["shot_id"] for shot in plan.get("shots", [])]
        store.upsert_project_shots(project_id, shot_ids, status="succeeded")
        for shot_id in shot_ids:
            store.update_shot(project_id, shot_id, status="succeeded", message="First frame ready")

        store.update_project(project_id, message="Generating shot videos")
        for shot_id in shot_ids:
            store.update_shot(project_id, shot_id, status="running", message="Generating video")

        video_report = generate_shot_videos(project_dir)
        completed = set()
        failed = set()
        for item in video_report.get("jobs", []):
            shot_id = item.get("shot_id") or _shot_id_from_wan_job_id(item.get("job_id"))
            if not shot_id:
                continue
            if item.get("status") in {
                "generated",
                "reused_control",
                "reused_external_verified",
                "skipped_existing",
            }:
                completed.add(shot_id)
            elif item.get("status") == "failed":
                failed.add(shot_id)
                store.update_shot(
                    project_id,
                    shot_id,
                    status="failed",
                    message="Video generation failed",
                    error=item.get("error") or "",
                )
        for shot_id in completed:
            store.update_shot(project_id, shot_id, status="succeeded", message="Video ready")
        for shot_id in set(shot_ids) - completed - failed:
            store.update_shot(project_id, shot_id, status="succeeded", message="Video ready")
        if failed:
            raise RuntimeError(f"{len(failed)} shot video generation job(s) failed")

        store.update_project(project_id, message="Assembling final video")
        final_video = assemble_videos(project_dir, reencode=True)
        store.update_project(project_id, status="succeeded", message="Completed", error="")
        return {
            "project_dir": str(project_dir),
            "project_plan_path": str(plan_path),
            "wan_manifest_path": str(project_dir / "wan_manifest_product.json"),
            "wan_video_report_path": str(project_dir / "wan_video_report.json"),
            "final_video": str(final_video),
        }
    except Exception as exc:
        error = f"{exc}\n\n{traceback.format_exc()}"
        store.update_project(project_id, status="failed", message="Generation failed", error=error)
        raise


def _shot_id_from_wan_job_id(job_id: str | None) -> str | None:
    if not job_id:
        return None
    if job_id.endswith("_video"):
        return job_id[:-len("_video")]
    return job_id
