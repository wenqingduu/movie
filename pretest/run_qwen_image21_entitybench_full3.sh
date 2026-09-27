#!/usr/bin/env bash
set -euo pipefail

project_root=/root/autodl-tmp/movie
output_root="$project_root/outputs/entitybench_qwen_image21_v7_full3"
qwen_python=/root/autodl-tmp/qwen-image21-venv/bin/python
face_python="$project_root/.venv/bin/python"
wan_python=/root/autodl-tmp/wan22-venv/bin/python
export MULTISHOT_INSIGHTFACE_PROVIDERS=CPUExecutionProvider

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

"$face_python" pretest/evaluate_qwen_image21_entitybench_batch.py \
  --output-root "$output_root" 2>&1 | tee "$output_root/logs/06_evaluation.log"
