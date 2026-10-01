#!/usr/bin/env bash
set -euo pipefail

project_root=/root/autodl-tmp/movie
output_root="$project_root/outputs/entitybench_qwen_image21_v7_full3"
qwen_python=/root/autodl-tmp/qwen-image21-venv/bin/python
face_python="$project_root/.venv/bin/python"
wan_python=/root/autodl-tmp/wan22-venv/bin/python
export MULTISHOT_INSIGHTFACE_PROVIDERS=CPUExecutionProvider
onnx_cuda_libs=/root/autodl-tmp/onnx_cuda12_libs

cd "$project_root"
mkdir -p "$output_root/logs"

"$qwen_python" pretest/run_qwen_image21_entitybench_batch.py control \
  --output-root "$output_root" 2>&1 | tee "$output_root/logs/01_control.log"

"$face_python" pretest/run_qwen_image21_entitybench_batch.py prepare \
  --output-root "$output_root" 2>&1 | tee "$output_root/logs/02_prepare.log"

"$qwen_python" pretest/run_qwen_image21_entitybench_batch.py treatment \
  --output-root "$output_root" 2>&1 | tee "$output_root/logs/03_treatment.log"

"$qwen_python" pretest/run_qwen_image21_entitybench_batch.py wan-manifest \
  --output-root "$output_root" 2>&1 | tee "$output_root/logs/04_wan_manifest.log"

"$wan_python" pretest/run_wan22_i2v_manifest.py \
  --manifest "$output_root/wan_manifest.json" \
  --report "$output_root/wan_report.json" \
  --continue-on-error 2>&1 | tee "$output_root/logs/05_wan.log"

if [[ -d "$onnx_cuda_libs" ]]; then
  export LD_LIBRARY_PATH="$onnx_cuda_libs/nvidia/cublas/lib:$onnx_cuda_libs/nvidia/cuda_runtime/lib:$onnx_cuda_libs/nvidia/cufft/lib:$project_root/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export MULTISHOT_INSIGHTFACE_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider
fi

"$face_python" pretest/evaluate_qwen_image21_entitybench_batch.py \
  --output-root "$output_root" 2>&1 | tee "$output_root/logs/06_evaluation.log"

"$face_python" pretest/evaluate_qwen_reference_dino.py \
  --output-root "$output_root" 2>&1 | tee "$output_root/logs/07_dinov2.log"

"$face_python" pretest/render_qwen_evaluation_comparisons.py \
  --output-root "$output_root" 2>&1 | tee "$output_root/logs/08_comparisons.log"

"$face_python" pretest/summarize_current_episode_results.py \
  2>&1 | tee "$output_root/logs/09_summary.log"
