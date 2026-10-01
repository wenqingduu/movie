#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/movie"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/entitybench_firstframe_pulid_ip_20ep"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_ROOT}/logs"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
export MULTISHOT_INSIGHTFACE_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider
ONNX_CUDA_LIBS="/root/autodl-tmp/onnx_cuda12_libs"
export LD_LIBRARY_PATH="${ONNX_CUDA_LIBS}/nvidia/cublas/lib:${ONNX_CUDA_LIBS}/nvidia/cuda_runtime/lib:${ONNX_CUDA_LIBS}/nvidia/cufft/lib:${PROJECT_ROOT}/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

"${PYTHON}" pretest/run_entitybench_firstframe_pulid_ip_batch.py manifest \
  --output-root "${OUTPUT_ROOT}"

# Asset creation is resumable. Two workers fit the 48 GB GPU and overlap the
# independent reference/3D preparation jobs without duplicating generation
# outputs.
"${PYTHON}" pretest/run_entitybench_firstframe_pulid_ip_batch.py assets \
  --output-root "${OUTPUT_ROOT}" \
  --asset-workers 2

# Use one persistent worker per image route. This keeps model duplication
# bounded while allowing PuLID-FLUX and IP-Adapter to advance concurrently.
"${PYTHON}" pretest/run_entitybench_firstframe_pulid_ip_batch.py pairs \
  --output-root "${OUTPUT_ROOT}"

"${PYTHON}" pretest/run_entitybench_firstframe_pulid_ip_batch.py status \
  --output-root "${OUTPUT_ROOT}"

"${PYTHON}" pretest/evaluate_entitybench_firstframe_pulid_ip_batch.py \
  --output-root "${OUTPUT_ROOT}"
