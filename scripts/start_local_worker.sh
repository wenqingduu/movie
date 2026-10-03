#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/movie}"
QUEUE="${QUEUE:-gpu}"
CONCURRENCY="${CONCURRENCY:-1}"

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
export MULTISHOT_VLLM_SLEEP_ENABLED="${MULTISHOT_VLLM_SLEEP_ENABLED:-1}"
ONNX_CUDA_LIBS="${ONNX_CUDA_LIBS:-/root/autodl-tmp/onnx_cuda12_libs}"
if [ -d "$ONNX_CUDA_LIBS" ]; then
  export LD_LIBRARY_PATH="$ONNX_CUDA_LIBS/nvidia/cublas/lib:$ONNX_CUDA_LIBS/nvidia/cuda_runtime/lib:$ONNX_CUDA_LIBS/nvidia/cufft/lib:$PROJECT_ROOT/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export MULTISHOT_INSIGHTFACE_PROVIDERS="${MULTISHOT_INSIGHTFACE_PROVIDERS:-CUDAExecutionProvider,CPUExecutionProvider}"
fi

exec celery -A app.workers.celery_app worker \
  --loglevel=info \
  --concurrency="$CONCURRENCY" \
  -Q "$QUEUE"
