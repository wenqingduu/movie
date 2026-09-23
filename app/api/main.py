from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.services import store
from app.services.project_runner import OUTPUT_ROOT, project_dir, submit_assemble_job, submit_pipeline_job


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_ROOT = PROJECT_ROOT / "frontend"
FRONTEND_DIST = FRONTEND_ROOT / "dist"


class CreateProjectRequest(BaseModel):
    story: str = Field(min_length=1)
    title: str | None = None
    generation_model: str = "juggernaut-xl-v9"
    base_seed: int = 1000
    autorun: bool = False


class RunProjectRequest(BaseModel):
    job_type: str = "pipeline"


class AssembleRequest(BaseModel):
    reencode: bool = True


def make_app() -> FastAPI:
    store.init_db()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="Movie Local Studio", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health():
        return {
            "ok": True,
            "project_root": str(PROJECT_ROOT),
            "output_root": str(OUTPUT_ROOT),
        }

    @app.post("/api/projects")
    def create_project(payload: CreateProjectRequest):
        story = payload.story.strip()
        if not story:
            raise HTTPException(status_code=400, detail="story is required")

        project_id = _new_project_id(payload.title or story)
        directory = project_dir(project_id).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "input_story.txt").write_text(story, encoding="utf-8")

        project = store.insert_project(
            project_id=project_id,
            title=(payload.title or _title_from_story(story)),
            story=story,
            project_dir=directory,
            generation_model=payload.generation_model,
            base_seed=payload.base_seed,
        )
        response = {"project": project, "job": None}
        if payload.autorun:
            response["job"] = _start_job(project_id, "pipeline")
        return response

    @app.get("/api/projects")
    def list_projects():
        return {"projects": [_project_with_latest_job(item) for item in store.list_projects()]}

    @app.get("/api/projects/{project_id}")
    def get_project(project_id: str):
        project = store.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        return {
            "project": _project_with_latest_job(project),
            "artifacts": _list_artifacts(Path(project["project_dir"])),
        }

    @app.post("/api/projects/{project_id}/run")
    def run_project(project_id: str, payload: RunProjectRequest):
        if payload.job_type != "pipeline":
            raise HTTPException(status_code=400, detail="unsupported job_type")
        _require_project(project_id)
        return {"job": _start_job(project_id, "pipeline")}

    @app.post("/api/projects/{project_id}/assemble")
    def assemble_project(project_id: str, payload: AssembleRequest):
        _require_project(project_id)
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job = store.insert_job(job_id=job_id, project_id=project_id, job_type="assemble")
        submit_assemble_job(job_id, project_id, reencode=payload.reencode)
        return {"job": job}

    @app.get("/api/jobs")
    def list_jobs(project_id: str | None = None):
        return {"jobs": store.list_jobs(project_id)}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return {"job": job}

    @app.get("/api/projects/{project_id}/project-plan")
    def get_project_plan(project_id: str):
        return _read_project_json(project_id, "project_plan.json")

    @app.get("/api/projects/{project_id}/asset-index")
    def get_asset_index(project_id: str):
        return _read_project_json(project_id, "asset_index.json")

    @app.get("/api/projects/{project_id}/wan-manifest")
    def get_wan_manifest(project_id: str):
        return _read_project_json(project_id, "wan_manifest_product.json")

    @app.get("/api/projects/{project_id}/artifacts")
    def get_artifacts(project_id: str):
        project = _require_project(project_id)
        return {"artifacts": _list_artifacts(Path(project["project_dir"]))}

    @app.get("/api/projects/{project_id}/final-video")
    def get_final_video(project_id: str):
        project = _require_project(project_id)
        final_video = Path(project["project_dir"]) / "final" / "final_video.mp4"
        if not final_video.is_file():
            raise HTTPException(status_code=404, detail="final video not found")
        return FileResponse(final_video, media_type="video/mp4")

    if OUTPUT_ROOT.exists():
        app.mount("/media", StaticFiles(directory=str(OUTPUT_ROOT)), name="media")
    if FRONTEND_DIST.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")

    return app


def _start_job(project_id: str, job_type: str) -> dict:
    latest = store.latest_job_for_project(project_id)
    if latest and latest["status"] in {"queued", "running"}:
        raise HTTPException(
            status_code=409,
            detail=f"project already has an active job: {latest['id']}",
        )
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    job = store.insert_job(job_id=job_id, project_id=project_id, job_type=job_type)
    submit_pipeline_job(job_id, project_id)
    return job


def _require_project(project_id: str) -> dict:
    project = store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    return project


def _read_project_json(project_id: str, filename: str) -> dict:
    project = _require_project(project_id)
    path = Path(project["project_dir"]) / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"{filename} not found")
    return json.loads(path.read_text(encoding="utf-8"))


def _project_with_latest_job(project: dict) -> dict:
    enriched = dict(project)
    enriched["latest_job"] = store.latest_job_for_project(project["id"])
    return enriched


def _new_project_id(seed: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", seed.strip().lower())[:28].strip("_")
    if not slug:
        slug = "project"
    return f"proj_{slug}_{uuid.uuid4().hex[:8]}"


def _title_from_story(story: str) -> str:
    compact = " ".join(story.split())
    return compact[:24] or "Untitled Project"


def _list_artifacts(project_path: Path) -> list[dict]:
    if not project_path.exists():
        return []
    artifacts = []
    for path in sorted(project_path.rglob("*")):
        if not path.is_file():
            continue
        rel_to_project = path.relative_to(project_path)
        rel_to_output = path.relative_to(OUTPUT_ROOT)
        artifacts.append(
            {
                "name": str(rel_to_project),
                "path": str(path),
                "url": "/media/" + quote(str(rel_to_output).replace("\\", "/")),
                "bytes": path.stat().st_size,
                "kind": _artifact_kind(path),
            }
        )
    return artifacts


def _artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".mp4", ".mov", ".webm"}:
        return "video"
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return "image"
    if suffix == ".json":
        return "json"
    if suffix in {".txt", ".md", ".log"}:
        return "text"
    return "file"


app = make_app()
