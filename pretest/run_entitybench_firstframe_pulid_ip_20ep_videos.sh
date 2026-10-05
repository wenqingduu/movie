#!/usr/bin/env bash
set -euo pipefail

project_root=/root/autodl-tmp/movie
output_root="$project_root/outputs/entitybench_firstframe_pulid_ip_20ep/video_evaluation"
onnx_cuda_libs=/root/autodl-tmp/onnx_cuda12_libs

cd "$project_root"
mkdir -p "$output_root/logs"
export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
export MULTISHOT_INSIGHTFACE_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider
export LD_LIBRARY_PATH="$onnx_cuda_libs/nvidia/cublas/lib:$onnx_cuda_libs/nvidia/cuda_runtime/lib:$onnx_cuda_libs/nvidia/cufft/lib:$project_root/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$project_root/.venv/bin/python" -u \
  pretest/run_entitybench_firstframe_pulid_ip_20ep_videos.py all \
  2>&1 | tee -a "$output_root/logs/tmux.log"
