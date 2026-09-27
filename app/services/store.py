from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, unquote, urlparse

import pymysql
from pymysql.cursors import DictCursor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_URL = "mysql+pymysql://movie_user:movie_local_pass@127.0.0.1:3306/movie_app?charset=utf8mb4"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def database_url() -> str:
    return os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL).strip()


def connect() -> pymysql.Connection:
    parsed = urlparse(database_url())
    if parsed.scheme not in {"mysql", "mysql+pymysql"}:
        raise ValueError("DATABASE_URL must use mysql+pymysql://")
    query = dict(parse_qsl(parsed.query))
    return pymysql.connect(
        host=parsed.hostname or "127.0.0.1",
        port=parsed.port or 3306,
        user=unquote(parsed.username or ""),
        password=unquote(parsed.password or ""),
        database=parsed.path.lstrip("/"),
        charset=query.get("charset", "utf8mb4"),
        autocommit=True,
        cursorclass=DictCursor,
    )


def init_db() -> None:
    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
            create table if not exists projects (
                id varchar(96) primary key,
                title varchar(255) not null,
                story longtext not null,
                status varchar(32) not null,
                project_dir varchar(1024) not null,
                generation_model varchar(128) not null,
                base_seed int not null,
                created_at varchar(64) not null,
                updated_at varchar(64) not null,
                index idx_projects_created_at (created_at)
            ) engine=InnoDB default charset=utf8mb4 collate=utf8mb4_unicode_ci
                """
            )
            cursor.execute(
                """
            create table if not exists jobs (
                id varchar(96) primary key,
                project_id varchar(96) not null,
                `type` varchar(32) not null,
                status varchar(32) not null,
                progress int not null default 0,
                message text not null,
                error longtext null,
                result_json longtext null,
                celery_task_id varchar(255) null,
                created_at varchar(64) not null,
                started_at varchar(64) null,
                finished_at varchar(64) null,
                index idx_jobs_project_created_at (project_id, created_at),
                constraint fk_jobs_project
                    foreign key(project_id) references projects(id)
                    on delete cascade
            ) engine=InnoDB default charset=utf8mb4 collate=utf8mb4_unicode_ci
                """
            )
            cursor.execute("show columns from jobs like 'celery_task_id'")
            if cursor.fetchone() is None:
                cursor.execute("alter table jobs add column celery_task_id varchar(255) null")


def row_to_dict(row: dict[str, Any] | None) -> dict[str, Any] | None:
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
        with conn.cursor() as cursor:
            cursor.execute(
                """
            insert into projects (
                id, title, story, status, project_dir, generation_model,
                base_seed, created_at, updated_at
            )
            values (%s, %s, %s, %s, %s, %s, %s, %s, %s)
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
        with conn.cursor() as cursor:
            cursor.execute("select * from projects order by created_at desc")
            rows = cursor.fetchall()
    return [row_to_dict(row) for row in rows if row is not None]


def get_project(project_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "select * from projects where id = %s",
                (project_id,),
            )
            row = cursor.fetchone()
    return row_to_dict(row)


def update_project_status(project_id: str, status: str) -> None:
    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "update projects set status = %s, updated_at = %s where id = %s",
                (status, utc_now(), project_id),
            )


def insert_job(*, job_id: str, project_id: str, job_type: str) -> dict[str, Any]:
    now = utc_now()
    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
            insert into jobs (
                id, project_id, `type`, status, progress, message,
                created_at
            )
            values (%s, %s, %s, %s, %s, %s, %s)
                """,
                (job_id, project_id, job_type, "queued", 0, "Queued", now),
            )
    job = get_job(job_id)
    if job is None:
        raise RuntimeError(f"Failed to create job {job_id}")
    return job


def get_job(job_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "select * from jobs where id = %s",
                (job_id,),
            )
            row = cursor.fetchone()
    return row_to_dict(row)


def latest_job_for_project(project_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
            select * from jobs
            where project_id = %s
            order by created_at desc
            limit 1
                """,
                (project_id,),
            )
            row = cursor.fetchone()
    return row_to_dict(row)


def active_job_for_project(project_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
            select * from jobs
            where project_id = %s and status in ('queued', 'running')
            order by created_at desc
            limit 1
                """,
                (project_id,),
            )
            row = cursor.fetchone()
    return row_to_dict(row)


def list_jobs(project_id: str | None = None) -> list[dict[str, Any]]:
    query = "select * from jobs"
    params: tuple[Any, ...] = ()
    if project_id:
        query += " where project_id = %s"
        params = (project_id,)
    query += " order by created_at desc"
    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
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
        fields.append("status = %s")
        values.append(status)
    if progress is not None:
        fields.append("progress = %s")
        values.append(max(0, min(100, int(progress))))
    if message is not None:
        fields.append("message = %s")
        values.append(message)
    if error is not None:
        fields.append("error = %s")
        values.append(error)
    if result is not None:
        fields.append("result_json = %s")
        values.append(json.dumps(result, ensure_ascii=False))
    if started:
        fields.append("started_at = %s")
        values.append(utc_now())
    if finished:
        fields.append("finished_at = %s")
        values.append(utc_now())
    if not fields:
        return
    values.append(job_id)
    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f"update jobs set {', '.join(fields)} where id = %s",
                values,
            )


def set_job_celery_task_id(job_id: str, celery_task_id: str) -> None:
    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "update jobs set celery_task_id = %s where id = %s",
                (celery_task_id, job_id),
            )
