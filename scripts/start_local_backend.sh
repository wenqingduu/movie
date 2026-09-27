#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/movie}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

cd "$PROJECT_ROOT"
source .venv/bin/activate

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export DATABASE_URL="${DATABASE_URL:-mysql+pymysql://movie_user:movie_local_pass@127.0.0.1:3306/movie_app?charset=utf8mb4}"
export CELERY_BROKER_URL="${CELERY_BROKER_URL:-redis://127.0.0.1:6379/0}"
export CELERY_RESULT_BACKEND="${CELERY_RESULT_BACKEND:-redis://127.0.0.1:6379/1}"
export DASHSCOPE_BASE_URL="${DASHSCOPE_BASE_URL:-http://127.0.0.1:8001/v1}"
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-local}"
export MULTISHOT_QWEN_MODEL="${MULTISHOT_QWEN_MODEL:-qwen-local}"

exec uvicorn app.api.main:app --host "$HOST" --port "$PORT"
