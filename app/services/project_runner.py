from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from multishot.product_pipeline import assemble_videos, run_story_project

from . import store


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "projects"

# Local MVP: one GPU-heavy task at a time.
EXECUTOR = ThreadPoolExecutor(max_workers=1)


def project_dir(project_id: str) -> Path:
    return OUTPUT_ROOT / project_id


def submit_pipeline_job(job_id: str, project_id: str) -> None:
    EXECUTOR.submit(_run_pipeline_job, job_id, project_id)


def submit_assemble_job(job_id: str, project_id: str, reencode: bool = True) -> None:
    EXECUTOR.submit(_run_assemble_job, job_id, project_id, reencode)


def _run_pipeline_job(job_id: str, project_id: str) -> None:
    project = store.get_project(project_id)
    if project is None:
        store.update_job(
            job_id,
            status="failed",
            progress=100,
            message="Project is missing",
            error=f"Unknown project: {project_id}",
            finished=True,
        )
        return

    try:
        store.update_project_status(project_id, "running")
        store.update_job(
            job_id,
            status="running",
            progress=5,
            message="Starting story pipeline",
            started=True,
        )
        result = run_story_project(
            project["story"],
            project["project_dir"],
            generation_model=project["generation_model"],
            base_seed=int(project["base_seed"]),
        )
        store.update_project_status(project_id, "first_frames_ready")
        store.update_job(
            job_id,
            status="succeeded",
            progress=100,
            message="Project pipeline completed",
            result=result,
            finished=True,
        )
    except Exception as exc:
        store.update_project_status(project_id, "failed")
        store.update_job(
            job_id,
            status="failed",
            progress=100,
            message="Project pipeline failed",
            error=f"{exc}\n\n{traceback.format_exc()}",
            finished=True,
        )


def _run_assemble_job(job_id: str, project_id: str, reencode: bool) -> None:
    project = store.get_project(project_id)
    if project is None:
        store.update_job(
            job_id,
            status="failed",
            progress=100,
            message="Project is missing",
            error=f"Unknown project: {project_id}",
            finished=True,
        )
        return

    try:
        store.update_project_status(project_id, "assembling")
        store.update_job(
            job_id,
            status="running",
            progress=20,
            message="Assembling shot videos",
            started=True,
        )
        final_video = assemble_videos(
            project["project_dir"],
            reencode=reencode,
        )
        result = {"final_video": str(final_video)}
        store.update_project_status(project_id, "completed")
        store.update_job(
            job_id,
            status="succeeded",
            progress=100,
            message="Final video assembled",
            result=result,
            finished=True,
        )
    except Exception as exc:
        store.update_project_status(project_id, "failed")
        store.update_job(
            job_id,
            status="failed",
            progress=100,
            message="Assembly failed",
            error=f"{exc}\n\n{traceback.format_exc()}",
            finished=True,
        )
