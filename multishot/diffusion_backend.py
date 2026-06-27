import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME = "segmind/tiny-sd"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "diffusion" / "segmind-tiny-sd"


class OpenSourceDiffusionBackend:
    """基于 diffusers 的开源文生图后端。

    这个类只负责“图像模型本身”的事情：
    - 加载本地开源 diffusion 模型。
    - 普通文生图，用于场景/人物资产生成。
    - 分窗口推进去噪，用于 shot 首帧实验。
    - 从当前 latent 反解 x0 preview，供 VLM/InsightFace 判断。
    - 从最终 latent decode 出最终首帧。

    人脸漂移检测、3D 人脸检索、rollout 选择这些仍然放在 MCP 工具流程里。
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.model_path = Path(os.getenv("MULTISHOT_DIFFUSION_MODEL_PATH", DEFAULT_MODEL_PATH))
        self.height = int(os.getenv("MULTISHOT_IMAGE_HEIGHT", "256"))
        self.width = int(os.getenv("MULTISHOT_IMAGE_WIDTH", "256"))
        self.default_steps = int(os.getenv("MULTISHOT_DIFFUSION_STEPS", "50"))
        self.guidance_scale = float(os.getenv("MULTISHOT_GUIDANCE_SCALE", "7.5"))
        self.negative_prompt = os.getenv(
            "MULTISHOT_NEGATIVE_PROMPT",
            "blurry, low quality, distorted face, extra fingers, bad anatomy",
        )

        self._pipe = None
        self._torch = None
        self.device = None
        self.dtype = None

    def _load(self):
        """懒加载 diffusers pipeline。

        MCP server 启动后可能会调用多个工具。懒加载可以避免刚启动 server
        就占 GPU 显存；真正第一次生成图像时才加载模型。
        """

        if self._pipe is not None:
            return self._pipe

        import torch
        from diffusers import DDIMScheduler, StableDiffusionPipeline

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True,
        )

        # DDIM 的 x0 反解公式更直接，适合做中间预览和实验日志。
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)

        self._pipe = pipe
        return pipe

    def generate_image(self, prompt: str, output_path: str, steps: int = 30):
        """直接用开源 diffusion pipeline 生成一张图片。"""

        pipe = self._load()
        image_path = Path(output_path)
        image_path.parent.mkdir(parents=True, exist_ok=True)

        image = pipe(
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            height=self.height,
            width=self.width,
            num_inference_steps=steps,
            guidance_scale=self.guidance_scale,
        ).images[0]
        image.save(image_path)

        image_path.with_suffix(".prompt.txt").write_text(prompt, encoding="utf-8")
        return str(image_path)

    def prepare_generation(self, shot_id: str, prompt: str, total_steps: int | None = None):
        """准备一次可分段推进的 diffusion 生成上下文。"""

        pipe = self._load()
        torch = self._torch
        total_steps = total_steps or self.default_steps
        do_cfg = self.guidance_scale > 1.0

        pipe.scheduler.set_timesteps(total_steps, device=self.device)
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=self.negative_prompt,
        )
        if do_cfg:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        generator_device = self.device if self.device == "cuda" else "cpu"
        seed = int(os.getenv("MULTISHOT_DIFFUSION_SEED", "42"))
        generator = torch.Generator(device=generator_device).manual_seed(seed)

        latent_channels = pipe.unet.config.in_channels
        vae_scale_factor = getattr(pipe, "vae_scale_factor", 8)
        latents = torch.randn(
            (1, latent_channels, self.height // vae_scale_factor, self.width // vae_scale_factor),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        latents = latents * pipe.scheduler.init_noise_sigma

        return {
            "shot_id": shot_id,
            "prompt": prompt,
            "total_steps": total_steps,
            "do_classifier_free_guidance": do_cfg,
            "prompt_embeds": prompt_embeds,
            "initial_latents": latents,
            "timesteps": pipe.scheduler.timesteps,
        }

    def denoise_window(
        self,
        runtime: dict,
        from_step: int,
        to_step: int,
        previous_denoise_state: dict | None,
        injection_plan: dict | None,
        conditioning: dict,
    ):
        """从 from_step 推进到 to_step。

        目前这里已经是真实 diffusion 去噪；reference face 注入先作为 hook
        保留在 injection_plan 里。后续要改 attention/latent 注入，只需要在
        每一步 scheduler.step 前后插入对应逻辑。
        """

        pipe = self._load()
        torch = self._torch
        injection_plan = injection_plan or {"lambda": 0.0, "targets": []}
        latents = (
            previous_denoise_state["_latents"].detach().clone()
            if previous_denoise_state is not None and "_latents" in previous_denoise_state
            else runtime["initial_latents"].detach().clone()
        )

        last_noise_pred = None
        last_timestep = None
        for step_index in range(from_step, to_step):
            timestep = runtime["timesteps"][step_index]
            latent_model_input = (
                torch.cat([latents] * 2)
                if runtime["do_classifier_free_guidance"]
                else latents
            )
            latent_model_input = pipe.scheduler.scale_model_input(
                latent_model_input,
                timestep,
            )

            with torch.no_grad():
                noise_pred = pipe.unet(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=runtime["prompt_embeds"],
                    return_dict=False,
                )[0]

            if runtime["do_classifier_free_guidance"]:
                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_pred = noise_uncond + self.guidance_scale * (noise_text - noise_uncond)

            # 真实 reference 注入入口：
            # injection_plan["lambda"] > 0 时，后续可在这里做 cross-attention /
            # latent region / IP-Adapter 形式的参考脸注入。现在不伪造注入效果，
            # 只把计划写进日志，让主流程先跑通。
            latents = pipe.scheduler.step(noise_pred, timestep, latents).prev_sample
            last_noise_pred = noise_pred.detach()
            last_timestep = timestep

        return {
            "backend": "diffusers",
            "model_name": self.model_name,
            "from_step": from_step,
            "step": to_step,
            "window_size": to_step - from_step,
            "latent_shape": list(latents.shape),
            "timestep": int(last_timestep.item()) if hasattr(last_timestep, "item") else int(last_timestep),
            "prediction_type": pipe.scheduler.config.get("prediction_type", "epsilon"),
            "injection_plan": injection_plan,
            "conditioning": conditioning,
            "_latents": latents,
            "_noise_pred": last_noise_pred,
            "_timestep": last_timestep,
        }

    def estimate_x0_preview(self, denoise_state: dict, output_path: str):
        """把当前 x_t 反解为 x0 preview 并保存成自然图像。"""

        pipe = self._load()
        torch = self._torch

        latents = denoise_state["_latents"]
        noise_pred = denoise_state["_noise_pred"]
        timestep = denoise_state["_timestep"]

        alpha_prod_t = pipe.scheduler.alphas_cumprod[timestep].to(latents.device, latents.dtype)
        beta_prod_t = 1 - alpha_prod_t
        prediction_type = pipe.scheduler.config.get("prediction_type", "epsilon")

        if prediction_type == "v_prediction":
            pred_x0 = alpha_prod_t.sqrt() * latents - beta_prod_t.sqrt() * noise_pred
        else:
            pred_x0 = (latents - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt()

        preview_path = Path(output_path)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        self._decode_latents_to_image(pred_x0, preview_path)

        return {
            "step": denoise_state["step"],
            "pred_x0_latent": f"diffusers_x0_latent_step_{denoise_state['step']}",
            "x0_preview_path": str(preview_path),
            "x0_image_features": {
                "backend": "diffusers",
                "model_name": self.model_name,
                "prediction_type": prediction_type,
                "latent_shape": list(pred_x0.shape),
            },
        }

    def decode_final_image(self, denoise_state: dict, output_path: str):
        """从最终 latent decode 得到最终首帧。"""

        image_path = Path(output_path)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        self._decode_latents_to_image(denoise_state["_latents"], image_path)
        return str(image_path)

    def _decode_latents_to_image(self, latents, image_path: Path):
        pipe = self._load()
        torch = self._torch

        scaling_factor = getattr(pipe.vae.config, "scaling_factor", 0.18215)
        with torch.no_grad():
            image = pipe.vae.decode(latents / scaling_factor, return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type="pil")[0]
        image.save(image_path)


_BACKENDS: dict[str, OpenSourceDiffusionBackend] = {}


def get_diffusion_backend(model_name: str | None = None):
    """按模型名获取后端实例。

    目前先把 segmind/tiny-sd 作为默认开源模型。后续要接 SD1.5/SDXL，
    可以通过 MULTISHOT_DIFFUSION_MODEL_PATH 指向新的 diffusers 模型目录。
    """

    model_name = model_name or DEFAULT_MODEL_NAME
    if model_name not in _BACKENDS:
        _BACKENDS[model_name] = OpenSourceDiffusionBackend(model_name)
    return _BACKENDS[model_name]
