import os
import threading
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME = "juggernaut-xl-v9"

MODEL_CONFIGS = {
    "juggernaut-xl-v9": {
        "repo_id": "RunDiffusion/Juggernaut-XL-v9",
        "path": PROJECT_ROOT / "models" / "diffusion" / "juggernaut-xl-v9",
        "variant": "fp16",
        "pipeline": "sdxl",
        "height": 1024,
        "width": 1024,
        "steps": 50,
        "guidance_scale": 5.0,
        "negative_prompt": "",
    },
    "RunDiffusion/Juggernaut-XL-v9": {
        "repo_id": "RunDiffusion/Juggernaut-XL-v9",
        "path": PROJECT_ROOT / "models" / "diffusion" / "juggernaut-xl-v9",
        "variant": "fp16",
        "pipeline": "sdxl",
        "height": 1024,
        "width": 1024,
        "steps": 50,
        "guidance_scale": 5.0,
        "negative_prompt": "",
    },
    "sdxl-base-1.0-ip-adapter": {
        "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "path": PROJECT_ROOT / "models" / "diffusion" / "sdxl-base-1.0",
        "variant": "fp16",
        "pipeline": "sdxl",
        "height": 1024,
        "width": 1024,
        "steps": 50,
        "guidance_scale": 5.0,
        "negative_prompt": "",
        "ip_adapter_path": PROJECT_ROOT / "models" / "ip_adapter" / "h94-IP-Adapter",
        "ip_adapter_subfolder": "sdxl_models",
        "ip_adapter_weight_name": "ip-adapter_sdxl.safetensors",
        "ip_adapter_scale": 0.6,
    },
    "stabilityai/stable-diffusion-xl-base-1.0-ip-adapter": {
        "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "path": PROJECT_ROOT / "models" / "diffusion" / "sdxl-base-1.0",
        "variant": "fp16",
        "pipeline": "sdxl",
        "height": 1024,
        "width": 1024,
        "steps": 50,
        "guidance_scale": 5.0,
        "negative_prompt": "",
        "ip_adapter_path": PROJECT_ROOT / "models" / "ip_adapter" / "h94-IP-Adapter",
        "ip_adapter_subfolder": "sdxl_models",
        "ip_adapter_weight_name": "ip-adapter_sdxl.safetensors",
        "ip_adapter_scale": 0.6,
    },
    "dreamshaper-8": {
        "repo_id": "Lykon/dreamshaper-8",
        "path": PROJECT_ROOT / "models" / "diffusion" / "dreamshaper-8",
        "variant": "fp16",
    },
    "Lykon/dreamshaper-8": {
        "repo_id": "Lykon/dreamshaper-8",
        "path": PROJECT_ROOT / "models" / "diffusion" / "dreamshaper-8",
        "variant": "fp16",
    },
    "segmind/tiny-sd": {
        "repo_id": "segmind/tiny-sd",
        "path": PROJECT_ROOT / "models" / "diffusion" / "segmind-tiny-sd",
        "variant": None,
    },
}


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
        model_config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS[DEFAULT_MODEL_NAME])
        self.model_path = Path(os.getenv("MULTISHOT_DIFFUSION_MODEL_PATH", model_config["path"]))
        self.variant = os.getenv("MULTISHOT_DIFFUSION_VARIANT", model_config.get("variant") or "") or None
        self.pipeline_type = model_config.get("pipeline", "sd15")
        self.height = int(os.getenv("MULTISHOT_IMAGE_HEIGHT", str(model_config.get("height", 512))))
        self.width = int(os.getenv("MULTISHOT_IMAGE_WIDTH", str(model_config.get("width", 512))))
        self.default_steps = int(os.getenv("MULTISHOT_DIFFUSION_STEPS", str(model_config.get("steps", 50))))
        self.guidance_scale = float(os.getenv("MULTISHOT_GUIDANCE_SCALE", str(model_config.get("guidance_scale", 7.5))))
        self.negative_prompt = os.getenv(
            "MULTISHOT_NEGATIVE_PROMPT",
            model_config.get("negative_prompt", "blurry, low quality, distorted face, extra fingers, bad anatomy"),
        )
        self.ip_adapter_path = Path(os.getenv("MULTISHOT_IP_ADAPTER_PATH", str(model_config.get("ip_adapter_path", ""))))
        self.ip_adapter_subfolder = os.getenv("MULTISHOT_IP_ADAPTER_SUBFOLDER", model_config.get("ip_adapter_subfolder", ""))
        self.ip_adapter_weight_name = os.getenv("MULTISHOT_IP_ADAPTER_WEIGHT", model_config.get("ip_adapter_weight_name", ""))
        self.ip_adapter_scale = float(os.getenv("MULTISHOT_IP_ADAPTER_SCALE", str(model_config.get("ip_adapter_scale", 0.6))))
        self.ip_adapter_enabled = bool(self.ip_adapter_weight_name and str(self.ip_adapter_path))

        self._pipe = None
        self._torch = None
        self._reference_latent_cache = {}
        self._reference_noise_cache = {}
        self._ip_adapter_reference_embed_cache = {}
        self._denoise_lock = threading.Lock()
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
        from diffusers import DDIMScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        load_kwargs = {
            "torch_dtype": self.dtype,
            "local_files_only": True,
        }
        if self.pipeline_type != "sdxl":
            load_kwargs["safety_checker"] = None
            load_kwargs["requires_safety_checker"] = False
        if self.variant:
            load_kwargs["variant"] = self.variant
            load_kwargs["use_safetensors"] = True

        pipeline_cls = StableDiffusionXLPipeline if self.pipeline_type == "sdxl" else StableDiffusionPipeline
        pipe = pipeline_cls.from_pretrained(
            self.model_path,
            **load_kwargs,
        )

        # DDIM 的 x0 反解公式更直接，适合做中间预览和实验日志。
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        if self.ip_adapter_enabled:
            pipe.load_ip_adapter(
                str(self.ip_adapter_path),
                subfolder=self.ip_adapter_subfolder or None,
                weight_name=self.ip_adapter_weight_name,
            )
            pipe.set_ip_adapter_scale(self.ip_adapter_scale)
        self._pipe = pipe
        return pipe

    def _reference_noise(self, reference_image: str, reference_latents):
        """为参考图生成稳定噪声，保证不同 lambda rollout 的对比只差注入强度。"""

        import hashlib

        torch = self._torch
        seed_base = int(os.getenv("MULTISHOT_DIFFUSION_SEED", "42"))
        digest = hashlib.sha256(f"{reference_image}:{seed_base}".encode("utf-8")).hexdigest()
        seed = (int(digest[:8], 16) + seed_base) % (2**31 - 1)
        cache_key = (reference_image, tuple(reference_latents.shape), str(reference_latents.dtype), str(reference_latents.device), seed)
        if cache_key not in self._reference_noise_cache:
            generator_device = self.device if self.device == "cuda" else "cpu"
            generator = torch.Generator(device=generator_device).manual_seed(seed)
            self._reference_noise_cache[cache_key] = torch.randn(
                reference_latents.shape,
                generator=generator,
                device=reference_latents.device,
                dtype=reference_latents.dtype,
            )
        return self._reference_noise_cache[cache_key]

    def _clone_added_cond_kwargs(self, added_cond_kwargs):
        torch = self._torch
        if added_cond_kwargs is None:
            return None
        cloned = {}
        for key, value in added_cond_kwargs.items():
            if isinstance(value, list):
                cloned[key] = [item.detach().clone() if torch.is_tensor(item) else item for item in value]
            elif isinstance(value, tuple):
                cloned[key] = tuple(item.detach().clone() if torch.is_tensor(item) else item for item in value)
            elif torch.is_tensor(value):
                cloned[key] = value.detach().clone()
            else:
                cloned[key] = value
        return cloned

    def _ip_adapter_reference_target(self, injection_plan: dict):
        """取动态 IP-Adapter 要用的参考图。

        默认用 reference_source_image，也就是干净 3D render。
        """

        source_key = os.getenv("MULTISHOT_DYNAMIC_IP_ADAPTER_SOURCE", "reference_source_image")
        for target in injection_plan.get("targets", []):
            reference_image = target.get(source_key) or target.get("reference_view_image") or target.get("reference_image")
            if not reference_image:
                continue
            reference_path = Path(reference_image)
            if not reference_path.is_absolute():
                reference_path = PROJECT_ROOT / reference_path
            if reference_path.exists():
                return str(reference_path), source_key
        return None, source_key

    def _ip_adapter_reference_embeds(self, reference_image: str, do_cfg: bool):
        pipe = self._load()
        cache_key = (reference_image, do_cfg, self.device, str(self.dtype))
        if cache_key in self._ip_adapter_reference_embed_cache:
            return self._ip_adapter_reference_embed_cache[cache_key]

        from PIL import Image

        image = Image.open(reference_image).convert("RGB")
        embeds = pipe.prepare_ip_adapter_image_embeds(
            ip_adapter_image=[image],
            ip_adapter_image_embeds=None,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
        )
        self._ip_adapter_reference_embed_cache[cache_key] = embeds
        return embeds

    def _blend_ip_adapter_embeds(self, original, reference, strength: float):
        torch = self._torch
        if isinstance(original, list) and isinstance(reference, list):
            return [
                self._blend_ip_adapter_embeds(orig_item, ref_item, strength)
                for orig_item, ref_item in zip(original, reference)
            ]
        if isinstance(original, tuple) and isinstance(reference, tuple):
            return tuple(
                self._blend_ip_adapter_embeds(orig_item, ref_item, strength)
                for orig_item, ref_item in zip(original, reference)
            )
        if torch.is_tensor(original) and torch.is_tensor(reference):
            reference = reference.to(device=original.device, dtype=original.dtype)
            return original * (1.0 - strength) + reference * strength
        return reference

    def _apply_dynamic_ip_adapter_reference(self, added_cond_kwargs, runtime: dict, injection_plan: dict, step_index: int):
        """在当前窗口把 IP-Adapter 图像条件换/混成检索视角参考脸。"""

        if os.getenv("MULTISHOT_DYNAMIC_IP_ADAPTER_REFERENCE", "0") != "1":
            return added_cond_kwargs, None
        if not self.ip_adapter_enabled or not added_cond_kwargs or "image_embeds" not in added_cond_kwargs:
            return added_cond_kwargs, None
        injection_lambda = float(injection_plan.get("lambda", 0.0) or 0.0)
        targets = injection_plan.get("targets", [])
        if injection_lambda <= 0 or not targets:
            return added_cond_kwargs, None

        reference_image, source_key = self._ip_adapter_reference_target(injection_plan)
        if not reference_image:
            return added_cond_kwargs, {
                "step_index": step_index,
                "mode": "dynamic_ip_adapter_reference",
                "status": "skipped",
                "reason": "reference image not found",
            }

        scale = float(os.getenv("MULTISHOT_DYNAMIC_IP_ADAPTER_SCALE", "1.0"))
        strength = max(0.0, min(1.0, injection_lambda * scale))
        if strength <= 0:
            return added_cond_kwargs, None

        updated = self._clone_added_cond_kwargs(added_cond_kwargs)
        reference_embeds = self._ip_adapter_reference_embeds(reference_image, runtime["do_classifier_free_guidance"])
        updated["image_embeds"] = self._blend_ip_adapter_embeds(
            updated["image_embeds"],
            reference_embeds,
            strength,
        )
        return updated, {
            "step_index": step_index,
            "mode": "dynamic_ip_adapter_reference",
            "reference_image": reference_image,
            "reference_source_key": source_key,
            "lambda": injection_lambda,
            "effective_strength": round(strength, 4),
            "target_count": len(targets),
            "status": "applied",
        }

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
        requested_final_step = int(os.getenv("MULTISHOT_FINAL_STEP", str(self.default_steps)))
        total_steps = total_steps or max(self.default_steps, requested_final_step)
        do_cfg = self.guidance_scale > 1.0

        pipe.scheduler.set_timesteps(total_steps, device=self.device)
        added_cond_kwargs = None
        if self.pipeline_type == "sdxl":
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(
                prompt=prompt,
                prompt_2=prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=do_cfg,
                negative_prompt=self.negative_prompt or None,
                negative_prompt_2=self.negative_prompt or None,
            )
            add_time_ids = pipe._get_add_time_ids(
                (self.height, self.width),
                (0, 0),
                (self.height, self.width),
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim,
            ).to(self.device)
            if do_cfg:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
                pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])
                add_time_ids = torch.cat([add_time_ids, add_time_ids])
            added_cond_kwargs = {
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids,
            }
            ip_adapter_image = os.getenv("MULTISHOT_IP_ADAPTER_IMAGE", "").strip()
            if self.ip_adapter_enabled and ip_adapter_image:
                from PIL import Image

                image = Image.open(ip_adapter_image).convert("RGB")
                added_cond_kwargs["image_embeds"] = pipe.prepare_ip_adapter_image_embeds(
                    ip_adapter_image=[image],
                    ip_adapter_image_embeds=None,
                    device=self.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=do_cfg,
                )
        else:
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
            "added_cond_kwargs": added_cond_kwargs,
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

        默认使用 trajectory_residual：scheduler 每步更新 target 后，把同一
        next timestep 的 3D reference latent residual 按脸区 mask 混入。
        设置 MULTISHOT_INJECTION_MODE=latent_blend 时可回退到静态 VAE latent blending；
        设置为 off 时完全关闭局部参考注入。
        """

        pipe = self._load()
        torch = self._torch
        injection_plan = injection_plan or {"lambda": 0.0, "targets": []}
        latents = (
            previous_denoise_state["_latents"].detach().clone()
            if previous_denoise_state is not None and "_latents" in previous_denoise_state
            else runtime["initial_latents"].detach().clone()
        )

        applied_injections = []
        last_noise_pred = None
        last_timestep = None
        injection_mode = os.getenv("MULTISHOT_INJECTION_MODE", "trajectory_residual")

        # 同一 backend 的多个 rollout 共用 pipeline 与缓存，因此串行进入 UNet。
        with self._denoise_lock:
            for step_index in range(from_step, to_step):
                timestep = runtime["timesteps"][step_index]
                target_model_input = (
                    torch.cat([latents] * 2)
                    if runtime["do_classifier_free_guidance"]
                    else latents
                )
                target_model_input = pipe.scheduler.scale_model_input(
                    target_model_input,
                    timestep,
                )

                step_injections = []
                model_input = target_model_input
                prompt_embeds = runtime["prompt_embeds"]
                added_cond_kwargs = self._clone_added_cond_kwargs(
                    runtime.get("added_cond_kwargs")
                )
                added_cond_kwargs, dynamic_ip_log = self._apply_dynamic_ip_adapter_reference(
                    added_cond_kwargs,
                    runtime,
                    injection_plan,
                    step_index,
                )
                if dynamic_ip_log:
                    step_injections.append(dynamic_ip_log)

                with torch.no_grad():
                    noise_pred = pipe.unet(
                        model_input,
                        timestep,
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]
                if runtime["do_classifier_free_guidance"]:
                    noise_uncond, noise_text = noise_pred.chunk(2)
                    noise_pred = noise_uncond + self.guidance_scale * (noise_text - noise_uncond)

                latents = pipe.scheduler.step(noise_pred, timestep, latents).prev_sample

                if injection_mode == "latent_blend":
                    latents, blend_logs = self._apply_reference_injection(
                        latents,
                        injection_plan,
                        step_index,
                    )
                    step_injections.extend(blend_logs)
                elif injection_mode == "trajectory_residual":
                    latents, trajectory_logs = self._apply_reference_trajectory_injection(
                        latents,
                        runtime,
                        injection_plan,
                        step_index,
                    )
                    step_injections.extend(trajectory_logs)

                applied_injections.extend(step_injections)
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
            "injection_mode": injection_mode,
            "injection_plan": injection_plan,
            "applied_injections": applied_injections,
            "conditioning": conditioning,
            "_latents": latents,
            "_noise_pred": last_noise_pred,
            "_timestep": last_timestep,
        }

    def _apply_reference_injection(self, latents, injection_plan: dict, step_index: int):
        """把参考脸 latent 按 mask 混入当前 latent。

        输入的 injection_plan 来自 MCP rollout：
        {
          "lambda": 0.5,
          "targets": [
            {"reference_image": ".../front.png", "face_bbox": [x1, y1, x2, y2], ...}
          ]
        }

        返回新的 latents 和可写日志的 applied_injections。
        """

        injection_lambda = float(injection_plan.get("lambda", 0.0) or 0.0)
        targets = injection_plan.get("targets", [])
        if injection_lambda <= 0 or not targets:
            return latents, []

        # 这里给 lambda 再乘一个全局缩放，避免 VAE latent 直接硬替换导致脸区过亮/破碎。
        # 想观察更强效果时可以调环境变量 MULTISHOT_INJECTION_SCALE。
        scale = float(os.getenv("MULTISHOT_INJECTION_SCALE", "0.85"))
        strength = max(0.0, min(1.0, injection_lambda * scale))
        if strength <= 0:
            return latents, []

        mixed_latents = latents
        applied = []
        for target in targets:
            reference_image = target.get("reference_image")
            if not reference_image or not Path(reference_image).exists():
                continue

            try:
                reference_latents = self._encode_reference_image(reference_image)
            except Exception as exc:
                applied.append({
                    "step_index": step_index,
                    "face_id": target.get("face_id"),
                    "reference_image": reference_image,
                    "status": "skipped",
                    "reason": f"failed to encode reference image: {exc}",
                })
                continue

            mask = self._build_latent_mask(target, mixed_latents)
            alpha = (mask * strength).to(device=mixed_latents.device, dtype=mixed_latents.dtype)
            mixed_latents = mixed_latents * (1 - alpha) + reference_latents * alpha
            applied.append({
                "step_index": step_index,
                "face_id": target.get("face_id"),
                "matched_character_id": target.get("matched_character_id"),
                "reference_image": reference_image,
                "lambda": injection_lambda,
                "effective_strength": round(strength, 4),
                "mask_source": "face_mask_path" if target.get("mask_path") else "face_bbox",
                "status": "applied",
            })

        return mixed_latents, applied

    def _apply_reference_trajectory_injection(
        self,
        latents,
        runtime: dict,
        injection_plan: dict,
        step_index: int,
    ):
        """按脸区混合同一 scheduler 时刻的 3D reference residual。

        这对应 PuLID-FLUX 实验中的核心更新：
        ``target_next += strength * mask * (reference_next - target_next)``。
        区别只在 reference trajectory 的构造：SDXL 这里用固定 reference noise
        和 scheduler.add_noise 得到与 target next timestep 对齐的 reference state。
        """

        pipe = self._load()
        torch = self._torch
        injection_lambda = float(injection_plan.get("lambda", 0.0) or 0.0)
        targets = injection_plan.get("targets", [])
        if injection_lambda <= 0 or not targets:
            return latents, []

        scale = float(os.getenv("MULTISHOT_TRAJECTORY_INJECTION_SCALE", "1.0"))
        strength = max(0.0, min(1.0, injection_lambda * scale))
        if strength <= 0:
            return latents, []

        next_step_index = step_index + 1
        has_next_timestep = next_step_index < len(runtime["timesteps"])
        mixed_latents = latents
        applied = []
        for target in targets:
            reference_image = target.get("reference_image")
            if not reference_image or not Path(reference_image).exists():
                applied.append({
                    "step_index": step_index,
                    "mode": "masked_reference_trajectory_residual",
                    "status": "skipped",
                    "reason": "reference image not found",
                })
                continue

            try:
                reference_x0 = self._encode_reference_image(reference_image)
                reference_noise = self._reference_noise(reference_image, reference_x0)
                if has_next_timestep:
                    next_timestep = runtime["timesteps"][next_step_index]
                    reference_next = pipe.scheduler.add_noise(
                        reference_x0,
                        reference_noise,
                        next_timestep,
                    )
                    trajectory_source = "scheduler_add_noise"
                    next_timestep_value = (
                        int(next_timestep.item())
                        if hasattr(next_timestep, "item")
                        else int(next_timestep)
                    )
                else:
                    reference_next = reference_x0
                    trajectory_source = "clean_reference_x0"
                    next_timestep_value = None
            except Exception as exc:
                applied.append({
                    "step_index": step_index,
                    "mode": "masked_reference_trajectory_residual",
                    "reference_image": reference_image,
                    "status": "skipped",
                    "reason": f"failed to build reference trajectory state: {exc}",
                })
                continue

            mask = self._build_latent_mask(target, mixed_latents)
            alpha = (mask * strength).to(device=mixed_latents.device, dtype=mixed_latents.dtype)
            residual = alpha * (reference_next - mixed_latents)
            mixed_latents = mixed_latents + residual
            applied.append({
                "step_index": step_index,
                "mode": "masked_reference_trajectory_residual",
                "face_id": target.get("face_id"),
                "matched_character_id": target.get("matched_character_id"),
                "reference_image": reference_image,
                "reference_trajectory_source": trajectory_source,
                "reference_next_timestep": next_timestep_value,
                "lambda": injection_lambda,
                "effective_strength": round(strength, 4),
                "mask_source": "face_mask_path" if target.get("mask_path") else "face_bbox",
                "mask_latent_token_count": int((mask > 0.01).sum().item()),
                "residual_norm": round(float(residual.float().norm().item()), 6),
                "reference_state_norm": round(float(reference_next.float().norm().item()), 6),
                "target_state_norm_before": round(float((mixed_latents - residual).float().norm().item()), 6),
                "finite": bool(torch.isfinite(mixed_latents).all().item()),
                "status": "applied",
            })

        return mixed_latents, applied

    def _encode_reference_image(self, reference_image_path: str):
        """把参考图编码成和当前生成图同尺寸的 VAE latent。"""

        pipe = self._load()
        torch = self._torch
        cache_key = (reference_image_path, self.width, self.height, str(self.dtype), self.device)
        if cache_key in self._reference_latent_cache:
            return self._reference_latent_cache[cache_key]

        from PIL import Image

        image = Image.open(reference_image_path).convert("RGB")
        image = image.resize((self.width, self.height))
        vae_dtype = torch.float32 if getattr(pipe.vae.config, "force_upcast", False) else self.dtype
        pipe.vae.to(dtype=vae_dtype)
        image_tensor = pipe.image_processor.preprocess(image).to(
            device=self.device,
            dtype=vae_dtype,
        )
        scaling_factor = getattr(pipe.vae.config, "scaling_factor", 0.18215)
        with torch.no_grad():
            latents = pipe.vae.encode(image_tensor).latent_dist.mean * scaling_factor
        latents = latents.to(device=self.device, dtype=self.dtype)

        self._reference_latent_cache[cache_key] = latents.detach()
        return self._reference_latent_cache[cache_key]

    def _build_latent_mask(self, target: dict, latents):
        """根据 face mask 或 bbox 生成 latent 尺度的 soft mask。"""

        torch = self._torch
        from PIL import Image, ImageDraw, ImageFilter

        mask_path = target.get("mask_path")
        if mask_path and Path(mask_path).exists() and Path(mask_path).stat().st_size > 0:
            mask_image = Image.open(mask_path).convert("L").resize((self.width, self.height))
        else:
            mask_image = Image.new("L", (self.width, self.height), 0)
            draw = ImageDraw.Draw(mask_image)
            bbox = target.get("face_bbox") or [
                int(self.width * 0.32),
                int(self.height * 0.20),
                int(self.width * 0.68),
                int(self.height * 0.62),
            ]
            x1, y1, x2, y2 = bbox
            pad_x = int((x2 - x1) * 0.25)
            pad_y = int((y2 - y1) * 0.35)
            expanded = [
                max(0, x1 - pad_x),
                max(0, y1 - pad_y),
                min(self.width, x2 + pad_x),
                min(self.height, y2 + pad_y),
            ]
            draw.ellipse(expanded, fill=255)

        mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=max(4, self.width // 80)))
        latent_h, latent_w = latents.shape[-2:]
        mask_image = mask_image.resize((latent_w, latent_h))
        values = torch.tensor(
            list(mask_image.getdata()),
            device=latents.device,
            dtype=latents.dtype,
        ).view(1, 1, latent_h, latent_w) / 255.0
        return values.clamp(0, 1)

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
        vae_dtype = torch.float32 if getattr(pipe.vae.config, "force_upcast", False) else self.dtype
        pipe.vae.to(dtype=vae_dtype)
        decode_latents = (latents / scaling_factor).to(device=self.device, dtype=vae_dtype)
        with torch.no_grad():
            image = pipe.vae.decode(decode_latents, return_dict=False)[0]
        image = image.float().clamp(-1, 1)
        image = pipe.image_processor.postprocess(image, output_type="pil")[0]
        image.save(image_path)


_BACKENDS: dict[str, OpenSourceDiffusionBackend] = {}


def get_diffusion_backend(model_name: str | None = None):
    """按模型名获取后端实例。

    默认使用 Juggernaut XL v9。DreamShaper 8 / tiny-sd 仍可作为对照或快速 smoke test。
    如果要接其他 SD1.5/SDXL 模型，可以通过 MULTISHOT_DIFFUSION_MODEL_PATH
    指向新的 diffusers 模型目录。
    """

    model_name = model_name or DEFAULT_MODEL_NAME
    if model_name not in _BACKENDS:
        _BACKENDS[model_name] = OpenSourceDiffusionBackend(model_name)
    return _BACKENDS[model_name]
