#!/usr/bin/env bash
set -euo pipefail

project_root=/root/autodl-tmp/movie
output_root="$project_root/outputs/entitybench_qwen_image21_official_control_partial_adaptive_040_full3"
qwen_python=/root/autodl-tmp/qwen-image21-venv/bin/python
face_python="$project_root/.venv/bin/python"
wan_python=/root/autodl-tmp/wan22-venv/bin/python
onnx_cuda_libs=/root/autodl-tmp/onnx_cuda12_libs

cd "$project_root"
mkdir -p "$output_root/logs"

if [[ -d "$onnx_cuda_libs" ]]; then
  export LD_LIBRARY_PATH="$onnx_cuda_libs/nvidia/cublas/lib:$onnx_cuda_libs/nvidia/cuda_runtime/lib:$onnx_cuda_libs/nvidia/cufft/lib:$project_root/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export MULTISHOT_INSIGHTFACE_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider
else
  export MULTISHOT_INSIGHTFACE_PROVIDERS=CPUExecutionProvider
fi
"$qwen_python" pretest/run_qwen_image21_shared_prefix_adaptive.py \
  --output-root "$output_root" \
  --identity-step-threshold 0.40 \
  --low-identity-active-steps 6 \
  --high-identity-active-steps 1 \
  2>&1 | tee -a "$output_root/logs/01_first_frames.log"

"$qwen_python" pretest/run_qwen_image21_entitybench_batch.py wan-manifest \
  --output-root "$output_root" \
  2>&1 | tee "$output_root/logs/02_wan_manifest.log"

"$wan_python" pretest/run_wan22_i2v_manifest.py \
  --manifest "$output_root/wan_manifest.json" \
  --report "$output_root/wan_report.json" \
  --continue-on-error \
  2>&1 | tee -a "$output_root/logs/03_wan.log"

"$face_python" pretest/evaluate_qwen_image21_entitybench_batch.py \
  --output-root "$output_root" \
  2>&1 | tee "$output_root/logs/04_insightface.log"

"$face_python" pretest/evaluate_qwen_reference_dino.py \
  --output-root "$output_root" \
  2>&1 | tee "$output_root/logs/05_dinov2.log"

"$face_python" pretest/render_qwen_evaluation_comparisons.py \
  --output-root "$output_root" \
  2>&1 | tee "$output_root/logs/06_comparisons.log"

"$face_python" pretest/compare_qwen_historical_adaptive.py \
  --new-root "$output_root" \
  2>&1 | tee "$output_root/logs/07_historical_comparison.log"

"$face_python" pretest/summarize_current_episode_results.py \
  2>&1 | tee "$output_root/logs/08_project_summary.log"
