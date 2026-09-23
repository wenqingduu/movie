from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "app.db"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def db_path() -> Path:
    return Path(os.getenv("MOVIE_APP_DB", str(DEFAULT_DB_PATH)))


def connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.executescript(
            """
            create table if not exists projects (
                id text primary key,
                title text not null,
                story text not null,
                status text not null,
                project_dir text not null,
                generation_model text not null,
                base_seed integer not null,
                created_at text not null,
                updated_at text not null
            );

            create table if not exists jobs (
                id text primary key,
                project_id text not null,
                type text not null,
                status text not null,
                progress integer not null default 0,
                message text not null default '',
                error text,
                result_json text,
                created_at text not null,
                started_at text,
                finished_at text,
                foreign key(project_id) references projects(id)
            );
            """
        )


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    data = dict(row)
    if data.get("result_json"):
        try:
            data["result"] = json.loads(data["result_json"])
        except json.JSONDecodeError:
            data["result"] = data["result_json"]
    data.pop("result_json", None)
    return data


def insert_project(
    *,
    project_id: str,
    title: str,
    story: str,
    project_dir: Path,
    generation_model: str,
    base_seed: int,
) -> dict[str, Any]:
    now = utc_now()
    with connect() as conn:
        conn.execute(
            """
            insert into projects (
                id, title, story, status, project_dir, generation_model,
                base_seed, created_at, updated_at
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                title,
                story,
                "created",
                str(project_dir),
                generation_model,
                base_seed,
                now,
                now,
            ),
        )
    project = get_project(project_id)
    if project is None:
        raise RuntimeError(f"Failed to create project {project_id}")
    return project


def list_projects() -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "select * from projects order by created_at desc"
        ).fetchall()
    return [row_to_dict(row) for row in rows if row is not None]


def get_project(project_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute(
            "select * from projects where id = ?",
            (project_id,),
        ).fetchone()
    return row_to_dict(row)


def update_project_status(project_id: str, status: str) -> None:
    with connect() as conn:
        conn.execute(
            "update projects set status = ?, updated_at = ? where id = ?",
            (status, utc_now(), project_id),
        )


def insert_job(*, job_id: str, project_id: str, job_type: str) -> dict[str, Any]:
    now = utc_now()
    with connect() as conn:
        conn.execute(
            """
            insert into jobs (
                id, project_id, type, status, progress, message,
                created_at
            )
            values (?, ?, ?, ?, ?, ?, ?)
            """,
            (job_id, project_id, job_type, "queued", 0, "Queued", now),
        )
    job = get_job(job_id)
    if job is None:
        raise RuntimeError(f"Failed to create job {job_id}")
    return job


def get_job(job_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute(
            "select * from jobs where id = ?",
            (job_id,),
        ).fetchone()
    return row_to_dict(row)


def latest_job_for_project(project_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute(
            """
            select * from jobs
            where project_id = ?
            order by created_at desc
            limit 1
            """,
            (project_id,),
        ).fetchone()
    return row_to_dict(row)


def list_jobs(project_id: str | None = None) -> list[dict[str, Any]]:
    query = "select * from jobs"
    params: tuple[Any, ...] = ()
    if project_id:
        query += " where project_id = ?"
        params = (project_id,)
    query += " order by created_at desc"
    with connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [row_to_dict(row) for row in rows if row is not None]


def update_job(
    job_id: str,
    *,
    status: str | None = None,
    progress: int | None = None,
    message: str | None = None,
    error: str | None = None,
    result: dict[str, Any] | None = None,
    started: bool = False,
    finished: bool = False,
) -> None:
    fields = []
    values: list[Any] = []
    if status is not None:
        fields.append("status = ?")
        values.append(status)
    if progress is not None:
        fields.append("progress = ?")
        values.append(max(0, min(100, int(progress))))
    if message is not None:
        fields.append("message = ?")
        values.append(message)
    if error is not None:
        fields.append("error = ?")
        values.append(error)
    if result is not None:
        fields.append("result_json = ?")
        values.append(json.dumps(result, ensure_ascii=False))
    if started:
        fields.append("started_at = ?")
        values.append(utc_now())
    if finished:
        fields.append("finished_at = ?")
        values.append(utc_now())
    if not fields:
        return
    values.append(job_id)
    with connect() as conn:
        conn.execute(
            f"update jobs set {', '.join(fields)} where id = ?",
            values,
        )
