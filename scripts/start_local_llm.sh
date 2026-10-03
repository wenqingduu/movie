#!/usr/bin/env bash
set -euo pipefail

VLLM_VENV="${VLLM_VENV:-/root/autodl-tmp/llm-vllm-venv}"
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/movie/models/llm/Qwen2.5-14B-Instruct-AWQ}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen-local}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"

if [ ! -d "$VLLM_VENV" ]; then
  echo "Missing vLLM venv: $VLLM_VENV" >&2
  echo "Create it first: python3 -m venv $VLLM_VENV && source $VLLM_VENV/bin/activate && pip install vllm" >&2
  exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
  echo "Missing model directory: $MODEL_DIR" >&2
  exit 1
fi

source "$VLLM_VENV/bin/activate"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_SERVER_DEV_MODE=1

python - <<'PY'
from pathlib import Path
import site
import re

paths = site.getsitepackages()
target = None
for base in paths:
    candidate = Path(base) / "vllm/model_executor/warmup/kernel_warmup.py"
    if candidate.exists():
        target = candidate
        break

if target is None:
    raise SystemExit("Cannot find vllm kernel_warmup.py")

text = target.read_text()
old_import = '''def kernel_warmup(worker: "Worker", *, process_local_only: bool = False):
    from vllm.model_executor.warmup.minimax_m3_msa_warmup import (
        minimax_m3_msa_warmup,
    )
'''
new_import = '''def kernel_warmup(worker: "Worker", *, process_local_only: bool = False):
    minimax_m3_msa_warmup = None
    try:
        from vllm.model_executor.warmup.minimax_m3_msa_warmup import (
            minimax_m3_msa_warmup,
        )
    except Exception:
        logger.exception("Skipping MiniMax M3 MSA warmup import.")
'''
new_call = "    if minimax_m3_msa_warmup is not None:\n        minimax_m3_msa_warmup(worker)\n"

changed = False
if old_import in text:
    text = text.replace(old_import, new_import)
    changed = True
patched = re.sub(
    r"(?m)^(?:[ \t]+if minimax_m3_msa_warmup is not None:\n)*"
    r"[ \t]+minimax_m3_msa_warmup\(worker\)\n",
    lambda match: new_call,
    text,
)
if patched != text:
    text = patched
    changed = True
if changed:
    compile(text, str(target), "exec")
    target.write_text(text)
    print(f"Patched {target}")
PY

exec vllm serve "$MODEL_DIR" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --quantization awq \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enable-sleep-mode \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --enforce-eager
