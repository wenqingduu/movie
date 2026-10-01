from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, unquote, urlparse

import pymysql
from pymysql.cursors import DictCursor


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
                story longtext not null,
                status varchar(32) not null,
                message varchar(255) not null,
                error longtext null,
                project_dir varchar(1024) not null,
                backend varchar(64) not null,
                celery_task_id varchar(255) null,
                created_at varchar(64) not null,
                updated_at varchar(64) not null,
                index idx_projects_created_at (created_at)
            ) engine=InnoDB default charset=utf8mb4 collate=utf8mb4_unicode_ci
                """
            )
            _ensure_project_columns(cursor)
            cursor.execute(
                """
            create table if not exists shots (
                id varchar(160) primary key,
                project_id varchar(96) not null,
                shot_id varchar(96) not null,
                status varchar(32) not null,
                message varchar(255) not null,
                error longtext null,
                created_at varchar(64) not null,
                updated_at varchar(64) not null,
                unique key uq_shots_project_shot (project_id, shot_id),
                index idx_shots_project (project_id),
                constraint fk_shots_project
                    foreign key(project_id) references projects(id)
                    on delete cascade
            ) engine=InnoDB default charset=utf8mb4 collate=utf8mb4_unicode_ci
                """
            )


def _columns(cursor, table: str) -> set[str]:
    cursor.execute(f"show columns from {table}")
    return {row["Field"] for row in cursor.fetchall()}


def _ensure_project_columns(cursor) -> None:
    columns = _columns(cursor, "projects")
    additions = {
        "message": "alter table projects add column message varchar(255) not null default ''",
        "error": "alter table projects add column error longtext null",
        "backend": "alter table projects add column backend varchar(64) not null default 'qwen_image21'",
        "celery_task_id": "alter table projects add column celery_task_id varchar(255) null",
    }
    for name, statement in additions.items():
        if name not in columns:
            cursor.execute(statement)


def row_to_dict(row: dict[str, Any] | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def insert_project(
    *,
    project_id: str,
    story: str,
    project_dir: Path,
    backend: str,
) -> dict[str, Any]:
    now = utc_now()
    with connect() as conn:
        with conn.cursor() as cursor:
            columns = _columns(cursor, "projects")
            fields = [
                "id",
                "story",
                "status",
                "message",
                "project_dir",
                "backend",
                "created_at",
                "updated_at",
            ]
            values: list[Any] = [
                project_id,
                story,
                "created",
                "Created",
                str(project_dir),
                backend,
                now,
                now,
            ]
            # Existing local DBs may still have old NOT NULL columns from the
            # earlier MVP. Fill them for compatibility without using them.
            legacy_defaults = {
                "title": project_id,
                "generation_model": backend,
                "base_seed": 1000,
            }
            for field, value in legacy_defaults.items():
                if field in columns:
                    fields.append(field)
                    values.append(value)
            placeholders = ", ".join(["%s"] * len(fields))
            cursor.execute(
                f"""
            insert into projects ({', '.join(fields)})
            values ({placeholders})
                """,
                values,
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
            cursor.execute("select * from projects where id = %s", (project_id,))
            row = cursor.fetchone()
    return row_to_dict(row)


def update_project(
    project_id: str,
    *,
    status: str | None = None,
    message: str | None = None,
    error: str | None = None,
    celery_task_id: str | None = None,
) -> None:
    fields = ["updated_at = %s"]
    values: list[Any] = [utc_now()]
    if status is not None:
        fields.append("status = %s")
        values.append(status)
    if message is not None:
        fields.append("message = %s")
        values.append(message)
    if error is not None:
        fields.append("error = %s")
        values.append(error)
    if celery_task_id is not None:
        fields.append("celery_task_id = %s")
        values.append(celery_task_id)
    values.append(project_id)
    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f"update projects set {', '.join(fields)} where id = %s",
                values,
            )


def active_project(project_id: str) -> dict[str, Any] | None:
    project = get_project(project_id)
    if project and project.get("status") == "running":
        return project
    return None


def upsert_project_shots(project_id: str, shot_ids: list[str], *, status: str = "pending") -> None:
    now = utc_now()
    with connect() as conn:
        with conn.cursor() as cursor:
            for shot_id in shot_ids:
                cursor.execute(
                    """
                insert into shots (
                    id, project_id, shot_id, status, message,
                    created_at, updated_at
                )
                values (%s, %s, %s, %s, %s, %s, %s)
                on duplicate key update
                    status = values(status),
                    message = values(message),
                    error = null,
                    updated_at = values(updated_at)
                    """,
                    (
                        f"{project_id}:{shot_id}",
                        project_id,
                        shot_id,
                        status,
                        "Pending",
                        now,
                        now,
                    ),
                )


def list_shots(project_id: str) -> list[dict[str, Any]]:
    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
            select * from shots
            where project_id = %s
            order by shot_id asc
                """,
                (project_id,),
            )
            rows = cursor.fetchall()
    return [row_to_dict(row) for row in rows if row is not None]


def update_shot(
    project_id: str,
    shot_id: str,
    *,
    status: str | None = None,
    message: str | None = None,
    error: str | None = None,
) -> None:
    fields = ["updated_at = %s"]
    values: list[Any] = [utc_now()]
    if status is not None:
        fields.append("status = %s")
        values.append(status)
    if message is not None:
        fields.append("message = %s")
        values.append(message)
    if error is not None:
        fields.append("error = %s")
        values.append(error)
    values.extend([project_id, shot_id])
    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f"""
            update shots
            set {', '.join(fields)}
            where project_id = %s and shot_id = %s
                """,
                values,
            )
