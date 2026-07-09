from huggingface_hub import snapshot_download


DIFFUSION_MODELS = {
    "RunDiffusion/Juggernaut-XL-v9": {
        "local_dir": "models/diffusion/juggernaut-xl-v9",
        "allow_patterns": [
            "model_index.json",
            "README.md",
            "scheduler/*",
            "tokenizer/*",
            "tokenizer_2/*",
            "text_encoder/config.json",
            "text_encoder/model.fp16.safetensors",
            "text_encoder_2/config.json",
            "text_encoder_2/model.fp16.safetensors",
            "unet/config.json",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.fp16.safetensors",
        ],
    },
    "Lykon/dreamshaper-8": {
        "local_dir": "models/diffusion/dreamshaper-8",
        "allow_patterns": [
            "model_index.json",
            "README.md",
            "scheduler/*",
            "tokenizer/*",
            "feature_extractor/*",
            "text_encoder/config.json",
            "text_encoder/model.fp16.safetensors",
            "unet/config.json",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.fp16.safetensors",
        ],
    },
    "segmind/tiny-sd": {
        "local_dir": "models/diffusion/segmind-tiny-sd",
        "allow_patterns": None,
    },
}


def download_diffusion_models():
    """下载当前项目使用的开源 diffusion 模型。"""

    for repo_id, config in DIFFUSION_MODELS.items():
        snapshot_download(
            repo_id=repo_id,
            local_dir=config["local_dir"],
            allow_patterns=config["allow_patterns"],
        )
        print(f"downloaded {repo_id} -> {config['local_dir']}")


if __name__ == "__main__":
    download_diffusion_models()
