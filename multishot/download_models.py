from huggingface_hub import snapshot_download


DIFFUSION_MODELS = {
    "segmind/tiny-sd": "models/diffusion/segmind-tiny-sd",
}


def download_diffusion_models():
    """下载当前项目默认使用的开源 diffusion 模型。"""

    for repo_id, local_dir in DIFFUSION_MODELS.items():
        snapshot_download(repo_id=repo_id, local_dir=local_dir)
        print(f"downloaded {repo_id} -> {local_dir}")


if __name__ == "__main__":
    download_diffusion_models()
