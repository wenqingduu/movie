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
from app.workers.tasks import generate_project


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "projects"
FRONTEND_ROOT = PROJECT_ROOT / "frontend"
FRONTEND_DIST = FRONTEND_ROOT / "dist"


class CreateProjectRequest(BaseModel):
    story: str = Field(min_length=1)
    backend: str = "qwen_image21"


def project_dir(project_id: str) -> Path:
    return OUTPUT_ROOT / project_id


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
            "queue": "celery",
        }

    @app.post("/api/projects")
    def create_project(payload: CreateProjectRequest):
        story = payload.story.strip()
        if not story:
            raise HTTPException(status_code=400, detail="story is required")

        project_id = _new_project_id(story)
        directory = project_dir(project_id).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "input_story.txt").write_text(story, encoding="utf-8")

        project = store.insert_project(
            project_id=project_id,
            story=story,
            project_dir=directory,
            backend=payload.backend,
        )
        return {"project": project}

    @app.get("/api/projects")
    def list_projects():
        return {"projects": store.list_projects()}

    @app.get("/api/projects/{project_id}")
    def get_project(project_id: str):
        project = _require_project(project_id)
        return {
            "project": project,
            "shots": store.list_shots(project_id),
            "artifacts": _list_artifacts(Path(project["project_dir"])),
        }

    @app.get("/api/projects/{project_id}/shots")
    def get_project_shots(project_id: str):
        _require_project(project_id)
        return {"shots": store.list_shots(project_id)}

    @app.post("/api/projects/{project_id}/run")
    def run_project(project_id: str):
        project = _require_project(project_id)
        if store.active_project(project_id):
            raise HTTPException(status_code=409, detail="project is already running")
        try:
            store.update_project(
                project_id,
                status="running",
                message="Queued",
                error="",
            )
            result = generate_project.apply_async(args=(project_id,), queue="gpu")
            store.update_project(project_id, celery_task_id=result.id)
        except Exception as exc:
            store.update_project(
                project_id,
                status="failed",
                message="Failed to enqueue Celery task",
                error=str(exc),
            )
            raise HTTPException(status_code=503, detail=f"failed to enqueue task: {exc}") from exc
        return {"project": store.get_project(project_id) or project}

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


def _new_project_id(seed: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", seed.strip().lower())[:28].strip("_")
    if not slug:
        slug = "project"
    return f"proj_{slug}_{uuid.uuid4().hex[:8]}"


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
