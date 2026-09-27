#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${PROJECT_ROOT}/models/diffusion/qwen-image-2.1"
REVISION="790c92633540aa0cb11d9abf19eb46d861714758"
BASE_URL="https://huggingface.co/Qwen/Qwen-Image-2.1/resolve/${REVISION}"

# aria2 reads the lower-case proxy variables on this host.
export http_proxy="${HTTP_PROXY:-${http_proxy:-}}"
export https_proxy="${HTTPS_PROXY:-${https_proxy:-}}"

weights=(
  "transformer/diffusion_pytorch_model-00001-of-00002.safetensors"
  "transformer/diffusion_pytorch_model-00002-of-00002.safetensors"
  "text_encoder/model-00001-of-00004.safetensors"
  "text_encoder/model-00002-of-00004.safetensors"
  "text_encoder/model-00003-of-00004.safetensors"
  "text_encoder/model-00004-of-00004.safetensors"
  "vae/diffusion_pytorch_model.safetensors"
)

for relative_path in "${weights[@]}"; do
  destination="${MODEL_ROOT}/$(dirname "${relative_path}")"
  mkdir -p "${destination}"
  aria2c \
    -c -x 16 -s 16 -k 4M \
    --file-allocation=none \
    --summary-interval=10 \
    --console-log-level=notice \
    -d "${destination}" \
    -o "$(basename "${relative_path}")" \
    "${BASE_URL}/${relative_path}"
done

# Fetch/verify the small configuration and tokenizer files after all large
# shards are present.  The single worker avoids resolver-rate-limit bursts.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=30
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen-Image-2.1', revision='${REVISION}', local_dir='${MODEL_ROOT}', local_dir_use_symlinks=False, max_workers=1, ignore_patterns=['*.safetensors'])"
