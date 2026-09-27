from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Callable

from app.services import store

from .celery_app import celery_app


def _missing_project(job_id: str, project_id: str) -> None:
    store.update_job(
        job_id,
        status="failed",
        progress=100,
        message="Project is missing",
        error=f"Unknown project: {project_id}",
        finished=True,
    )


def _run_project_task(
    *,
    job_id: str,
    project_id: str,
    running_status: str,
    running_message: str,
    success_status: str,
    success_message: str,
    handler: Callable[[dict[str, Any]], dict[str, Any]],
    start_progress: int = 5,
) -> dict[str, Any]:
    project = store.get_project(project_id)
    if project is None:
        _missing_project(job_id, project_id)
        return {"ok": False, "error": f"Unknown project: {project_id}"}

    try:
        store.update_project_status(project_id, running_status)
        store.update_job(
            job_id,
            status="running",
            progress=start_progress,
            message=running_message,
            started=True,
        )
        result = handler(project)
        store.update_project_status(project_id, success_status)
        store.update_job(
            job_id,
            status="succeeded",
            progress=100,
            message=success_message,
            result=result,
            finished=True,
        )
        return result
    except Exception as exc:
        error = f"{exc}\n\n{traceback.format_exc()}"
        store.update_project_status(project_id, "failed")
        store.update_job(
            job_id,
            status="failed",
            progress=100,
            message=f"{running_message} failed",
            error=error,
            finished=True,
        )
        raise


@celery_app.task(name="movie.pipeline", queue="gpu")
def run_pipeline_job(job_id: str, project_id: str) -> dict[str, Any]:
    def handler(project: dict[str, Any]) -> dict[str, Any]:
        from multishot.product_pipeline import run_story_project

        return run_story_project(
            project["story"],
            project["project_dir"],
            generation_model=project["generation_model"],
            base_seed=int(project["base_seed"]),
        )

    return _run_project_task(
        job_id=job_id,
        project_id=project_id,
        running_status="running",
        running_message="Starting story pipeline",
        success_status="first_frames_ready",
        success_message="Project pipeline completed",
        handler=handler,
        start_progress=5,
    )


@celery_app.task(name="movie.wan_video", queue="gpu")
def generate_videos_job(job_id: str, project_id: str) -> dict[str, Any]:
    def handler(project: dict[str, Any]) -> dict[str, Any]:
        from multishot.product_pipeline import generate_shot_videos

        report = generate_shot_videos(Path(project["project_dir"]))
        return {
            "wan_video_report_path": str(Path(project["project_dir"]) / "wan_video_report.json"),
            "completed_jobs": report.get("completed_jobs"),
            "failed_jobs": report.get("failed_jobs"),
        }

    return _run_project_task(
        job_id=job_id,
        project_id=project_id,
        running_status="video_generating",
        running_message="Generating shot videos",
        success_status="videos_ready",
        success_message="Shot videos generated",
        handler=handler,
        start_progress=10,
    )


@celery_app.task(name="movie.assemble", queue="gpu")
def assemble_video_job(job_id: str, project_id: str, reencode: bool = True) -> dict[str, Any]:
    def handler(project: dict[str, Any]) -> dict[str, Any]:
        from multishot.product_pipeline import assemble_videos

        final_video = assemble_videos(
            project["project_dir"],
            reencode=reencode,
        )
        return {"final_video": str(final_video)}

    return _run_project_task(
        job_id=job_id,
        project_id=project_id,
        running_status="assembling",
        running_message="Assembling shot videos",
        success_status="completed",
        success_message="Final video assembled",
        handler=handler,
        start_progress=20,
    )
