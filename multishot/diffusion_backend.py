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


class ReferenceSelfAttentionController:
    """管理 masked mutual self-attention 的运行状态。

    真实注入时，reference latent 和 target latent 会拼在同一个 batch 里跑 UNet。
    processor 在每个 self-attention 层里手动拆出 reference / target 的 Q/K/V，
    然后让 target face query 去关注 reference face key/value。
    """

    def __init__(self, image_width: int, image_height: int):
        self.image_width = image_width
        self.image_height = image_height
        self.mode = "off"
        self.strength = 0.0
        self.targets = []
        self.source_batch_size = 0
        self.target_batch_size = 0
        self.start_layer = 0
        self.end_layer = 10**9
        self._mask_cache = {}
        self.layer_stats = []
        self.debug_prefix = None
        self.debug_reference_image = None
        self.debug_written = 0

    def start_mutual(
        self,
        strength: float,
        targets: list[dict],
        source_batch_size: int,
        target_batch_size: int,
        debug_prefix: str | None = None,
        debug_reference_image: str | None = None,
    ):
        self.mode = "mutual"
        self.strength = max(0.0, min(1.0, float(strength)))
        self.targets = targets
        self.source_batch_size = source_batch_size
        self.target_batch_size = target_batch_size
        self._mask_cache = {}
        self.layer_stats = []
        self.debug_prefix = debug_prefix
        self.debug_reference_image = debug_reference_image
        self.debug_written = 0

    def stop(self):
        self.mode = "off"
        self.strength = 0.0
        self.targets = []
        self.source_batch_size = 0
        self.target_batch_size = 0
        self._mask_cache = {}
        self.layer_stats = []
        self.debug_prefix = None
        self.debug_reference_image = None
        self.debug_written = 0

    def configure_layers(self, total_layers: int, pipeline_type: str):
        """配置哪些 self-attention 层参与注入。

        MasaCtrl 的 SDXL demo 重点使用中后层，例如 44/54/64。
        我们默认从 54 层开始；如果模型层数不足，则退化为后 25% 的层。
        """

        if pipeline_type == "sdxl":
            default_start = 54 if total_layers > 54 else int(total_layers * 0.75)
        else:
            default_start = 10 if total_layers > 10 else int(total_layers * 0.65)
        self.start_layer = int(os.getenv("MULTISHOT_MUTUAL_ATTENTION_START_LAYER", str(default_start)))
        self.end_layer = int(os.getenv("MULTISHOT_MUTUAL_ATTENTION_END_LAYER", str(total_layers - 1)))
        self.start_layer = max(0, min(self.start_layer, total_layers - 1))
        self.end_layer = max(self.start_layer, min(self.end_layer, total_layers - 1))

    def layer_enabled(self, layer_index: int):
        return self.start_layer <= layer_index <= self.end_layer

    @staticmethod
    def layer_region(layer_name: str):
        if layer_name.startswith("down_blocks"):
            return "down"
        if layer_name.startswith("mid_block"):
            return "mid"
        if layer_name.startswith("up_blocks"):
            return "up"
        return "unknown"

    def record_layer_stat(
        self,
        layer_index: int,
        layer_name: str,
        height: int | None,
        width: int | None,
        sequence_length: int,
        target_token_count: int,
        reference_token_count: int,
    ):
        self.layer_stats.append({
            "layer_index": layer_index,
            "layer_name": layer_name,
            "region": self.layer_region(layer_name),
            "height": height,
            "width": width,
            "sequence_length": sequence_length,
            "target_face_token_count": target_token_count,
            "reference_face_token_count": reference_token_count,
            "reference_to_target_token_ratio": round(reference_token_count / max(1, target_token_count), 4),
        })


    def should_write_debug_map(self, layer_index: int, height: int | None, width: int | None, sequence_length: int):
        if os.getenv("MULTISHOT_ATTENTION_DEBUG", "0") != "1":
            return False
        if not self.debug_prefix:
            return False
        if height is None or width is None:
            side = int(sequence_length ** 0.5)
            if side * side != sequence_length:
                return False
        max_maps = int(os.getenv("MULTISHOT_ATTENTION_DEBUG_MAX_MAPS", "4"))
        if self.debug_written >= max_maps:
            return False
        max_tokens = int(os.getenv("MULTISHOT_ATTENTION_DEBUG_MAX_TOKENS", "1024"))
        if sequence_length > max_tokens:
            return False
        layer_filter = os.getenv("MULTISHOT_ATTENTION_DEBUG_LAYER", "").strip()
        if layer_filter and str(layer_index) != layer_filter:
            return False
        return True

    def write_debug_attention_map(
        self,
        layer_index: int,
        layer_name: str,
        height: int,
        width: int,
        attention_heat,
        ref_mask,
    ):
        from PIL import Image, ImageOps
        import numpy as np
        import torch

        if attention_heat is None:
            return None
        heat = attention_heat.detach().float().cpu()
        if heat.numel() != height * width:
            return None
        heat = heat.view(height, width)
        heat = heat - heat.min()
        heat = heat / (heat.max() + 1e-8)
        heat_np = (heat.numpy() * 255.0).round().astype(np.uint8)

        out_dir = Path(os.getenv("MULTISHOT_ATTENTION_DEBUG_DIR", PROJECT_ROOT / "outputs" / "attention_debug"))
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_layer = layer_name.replace(".", "_").replace("/", "_")
        base = f"{self.debug_prefix}.layer_{layer_index:03d}.{safe_layer}.{width}x{height}"
        heat_path = out_dir / f"{base}.heat.png"
        overlay_path = out_dir / f"{base}.overlay.png"
        mask_path = out_dir / f"{base}.ref_mask.png"

        heat_img = Image.fromarray(heat_np, mode="L").resize((self.image_width, self.image_height), Image.Resampling.BICUBIC)
        heat_img.save(heat_path)

        ref_mask_img = ref_mask.detach().float().cpu().view(height, width)
        ref_mask_np = (ref_mask_img.clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
        Image.fromarray(ref_mask_np, mode="L").resize((self.image_width, self.image_height), Image.Resampling.NEAREST).save(mask_path)

        if self.debug_reference_image and Path(self.debug_reference_image).exists():
            base_img = Image.open(self.debug_reference_image).convert("RGB").resize((self.image_width, self.image_height))
        else:
            base_img = Image.new("RGB", (self.image_width, self.image_height), "black")
        color = ImageOps.colorize(heat_img, black="black", white="red")
        overlay = Image.blend(base_img, color, 0.45)
        overlay.save(overlay_path)

        self.debug_written += 1
        return {
            "layer_index": layer_index,
            "layer_name": layer_name,
            "height": height,
            "width": width,
            "heatmap_path": str(heat_path),
            "overlay_path": str(overlay_path),
            "reference_mask_path": str(mask_path),
        }

    def target_query_mask(self, sequence_length: int, batch_size: int, dtype, device, height: int | None, width: int | None):
        """生成 [target_batch, sequence, 1]，控制当前图哪些 query token 被注入。"""

        return self._token_mask(
            mask_field="mask_path",
            bbox_field="face_bbox",
            sequence_length=sequence_length,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            height=height,
            width=width,
            cache_prefix="target",
        )

    def reference_key_mask(self, sequence_length: int, batch_size: int, dtype, device, height: int | None, width: int | None):
        """生成 [source_batch, sequence]，控制参考图哪些 key/value token 可以被关注。"""

        mask = self._token_mask(
            mask_field="reference_mask_path",
            bbox_field="reference_face_bbox",
            sequence_length=sequence_length,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            height=height,
            width=width,
            cache_prefix="reference",
        )
        return mask.squeeze(-1)

    def part_guided_enabled(self):
        return os.getenv("MULTISHOT_ATTENTION_MASK_MODE", "face").strip().lower() in {
            "part",
            "parts",
            "part_guided",
        }

    def part_names(self):
        value = os.getenv("MULTISHOT_ATTENTION_PARTS", "eyes,nose,mouth")
        parts = [item.strip().lower() for item in value.split(",") if item.strip()]
        allowed = {"eyes", "nose", "mouth", "face"}
        return [part for part in parts if part in allowed] or ["eyes", "nose", "mouth"]

    @staticmethod
    def _part_bbox_from_face_bbox(bbox, part: str, image_width: int, image_height: int):
        x1, y1, x2, y2 = [float(v) for v in bbox]
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(image_width), x2), min(float(image_height), y2)
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        if part == "eyes":
            return [
                x1 + 0.08 * w,
                y1 + 0.16 * h,
                x2 - 0.08 * w,
                y1 + 0.44 * h,
            ]
        if part == "nose":
            return [
                x1 + 0.25 * w,
                y1 + 0.30 * h,
                x2 - 0.25 * w,
                y1 + 0.68 * h,
            ]
        if part == "mouth":
            return [
                x1 + 0.18 * w,
                y1 + 0.58 * h,
                x2 - 0.18 * w,
                y1 + 0.86 * h,
            ]
        return [x1, y1, x2, y2]

    def target_part_query_mask(self, part: str, sequence_length: int, batch_size: int, dtype, device, height: int | None, width: int | None):
        return self._token_part_mask(
            part=part,
            bbox_field="face_bbox",
            sequence_length=sequence_length,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            height=height,
            width=width,
            cache_prefix="target_part",
        )

    def reference_part_key_mask(self, part: str, sequence_length: int, batch_size: int, dtype, device, height: int | None, width: int | None):
        mask = self._token_part_mask(
            part=part,
            bbox_field="reference_face_bbox",
            sequence_length=sequence_length,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            height=height,
            width=width,
            cache_prefix="reference_part",
        )
        return mask.squeeze(-1)

    def _token_part_mask(
        self,
        part: str,
        bbox_field: str,
        sequence_length: int,
        batch_size: int,
        dtype,
        device,
        height: int | None,
        width: int | None,
        cache_prefix: str,
    ):
        cache_key = (cache_prefix, part, bbox_field, sequence_length, batch_size, str(dtype), str(device), height, width)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        from PIL import Image, ImageDraw, ImageFilter
        import torch

        if height is None or width is None:
            side = int(sequence_length ** 0.5)
            if side * side != sequence_length:
                mask = torch.ones((batch_size, sequence_length, 1), device=device, dtype=dtype)
                self._mask_cache[cache_key] = mask
                return mask
            height = side
            width = side

        full_mask = Image.new("L", (self.image_width, self.image_height), 0)
        for target in self.targets:
            bbox = target.get(bbox_field)
            if not bbox:
                bbox = [
                    int(self.image_width * 0.35),
                    int(self.image_height * 0.18),
                    int(self.image_width * 0.65),
                    int(self.image_height * 0.58),
                ]
            x1, y1, x2, y2 = self._part_bbox_from_face_bbox(
                bbox,
                part,
                self.image_width,
                self.image_height,
            )
            mask_image = Image.new("L", (self.image_width, self.image_height), 0)
            draw = ImageDraw.Draw(mask_image)
            shape = [int(round(v)) for v in [x1, y1, x2, y2]]
            if part == "face":
                draw.ellipse(shape, fill=255)
            else:
                draw.rounded_rectangle(shape, radius=max(2, int((shape[2] - shape[0]) * 0.18)), fill=255)
            full_mask = Image.composite(Image.new("L", full_mask.size, 255), full_mask, mask_image)

        blur = max(1, self.image_width // 260)
        full_mask = full_mask.filter(ImageFilter.GaussianBlur(radius=blur))
        token_mask = full_mask.resize((width, height))
        values = torch.tensor(list(token_mask.getdata()), device=device, dtype=dtype).view(1, height * width, 1) / 255.0
        values = values.clamp(0, 1)
        if height * width != sequence_length:
            values = torch.ones((1, sequence_length, 1), device=device, dtype=dtype)
        mask = values.repeat(batch_size, 1, 1)
        self._mask_cache[cache_key] = mask
        return mask

    def _token_mask(
        self,
        mask_field: str,
        bbox_field: str,
        sequence_length: int,
        batch_size: int,
        dtype,
        device,
        height: int | None,
        width: int | None,
        cache_prefix: str,
    ):
        cache_key = (cache_prefix, mask_field, bbox_field, sequence_length, batch_size, str(dtype), str(device), height, width)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        from PIL import Image, ImageDraw, ImageFilter
        import torch

        if height is None or width is None:
            side = int(sequence_length ** 0.5)
            if side * side != sequence_length:
                mask = torch.ones((batch_size, sequence_length, 1), device=device, dtype=dtype)
                self._mask_cache[cache_key] = mask
                return mask
            height = side
            width = side

        full_mask = Image.new("L", (self.image_width, self.image_height), 0)
        for target in self.targets:
            mask_path = target.get(mask_field)
            if mask_path and Path(mask_path).exists() and Path(mask_path).stat().st_size > 0:
                mask_image = Image.open(mask_path).convert("L").resize((self.image_width, self.image_height))
            else:
                mask_image = Image.new("L", (self.image_width, self.image_height), 0)
                draw = ImageDraw.Draw(mask_image)
                bbox = target.get(bbox_field) or [
                    int(self.image_width * 0.35),
                    int(self.image_height * 0.18),
                    int(self.image_width * 0.65),
                    int(self.image_height * 0.58),
                ]
                x1, y1, x2, y2 = [int(v) for v in bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(self.image_width, x2), min(self.image_height, y2)
                draw.ellipse([x1, y1, x2, y2], fill=255)
            full_mask = Image.composite(Image.new("L", full_mask.size, 255), full_mask, mask_image)

        full_mask = full_mask.filter(ImageFilter.GaussianBlur(radius=max(2, self.image_width // 180)))
        token_mask = full_mask.resize((width, height))
        values = torch.tensor(list(token_mask.getdata()), device=device, dtype=dtype).view(1, height * width, 1) / 255.0
        values = values.clamp(0, 1)
        if height * width != sequence_length:
            values = torch.ones((1, sequence_length, 1), device=device, dtype=dtype)
        mask = values.repeat(batch_size, 1, 1)
        self._mask_cache[cache_key] = mask
        return mask


class ReferenceSelfAttnProcessor:
    """Diffusers attention processor：masked mutual self-attention。

    输入 batch 布局：
    [reference_uncond, reference_cond, target_uncond, target_cond]  # CFG 时
    [reference, target]                                            # 非 CFG 时

    默认 attention 仍然按 batch 内各自 self-attend。这里额外让 target 的 query
    去 reference face K/V 上做一次 attention，再用当前脸 mask 和 lambda 融合。
    """

    def __init__(self, layer_name: str, layer_index: int, controller: ReferenceSelfAttentionController):
        self.layer_name = layer_name
        self.layer_index = layer_index
        self.controller = controller

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        import torch
        import torch.nn.functional as F

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        height = width = None
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        is_self_attention = encoder_hidden_states is None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        self_output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        output = self_output

        should_inject = (
            is_self_attention
            and self.controller.mode == "mutual"
            and self.controller.strength > 0
            and self.controller.layer_enabled(self.layer_index)
        )
        source_batch = self.controller.source_batch_size
        target_batch = self.controller.target_batch_size
        if should_inject and batch_size == source_batch + target_batch and source_batch == target_batch:
            ref_slice = slice(0, source_batch)
            target_slice = slice(source_batch, source_batch + target_batch)

            q_target = query[target_slice]
            k_ref = key[ref_slice]
            v_ref = value[ref_slice]

            output = output.clone()
            part_stats = []
            if self.controller.part_guided_enabled():
                target_combined = None
                reference_combined = None
                target_current = output[target_slice]
                for part in self.controller.part_names():
                    part_ref_mask = self.controller.reference_part_key_mask(
                        part=part,
                        sequence_length=sequence_length,
                        batch_size=source_batch,
                        dtype=query.dtype,
                        device=query.device,
                        height=height,
                        width=width,
                    )
                    part_target_mask_raw = self.controller.target_part_query_mask(
                        part=part,
                        sequence_length=sequence_length,
                        batch_size=target_batch,
                        dtype=query.dtype,
                        device=query.device,
                        height=height,
                        width=width,
                    )
                    if int((part_ref_mask[0, :] > 0.05).sum().item()) <= 0:
                        continue
                    if int((part_target_mask_raw[0, :, 0] > 0.05).sum().item()) <= 0:
                        continue
                    part_ref_attn_mask = (1.0 - part_ref_mask).view(source_batch, 1, 1, sequence_length) * -10000.0
                    part_ref_output = F.scaled_dot_product_attention(
                        q_target,
                        k_ref,
                        v_ref,
                        attn_mask=part_ref_attn_mask,
                        dropout_p=0.0,
                        is_causal=False,
                    )
                    part_target_mask = part_target_mask_raw.view(target_batch, 1, sequence_length, 1)
                    alpha = part_target_mask * self.controller.strength
                    target_current = target_current * (1 - alpha) + part_ref_output * alpha
                    target_combined = part_target_mask_raw if target_combined is None else torch.maximum(target_combined, part_target_mask_raw)
                    reference_combined = part_ref_mask if reference_combined is None else torch.maximum(reference_combined, part_ref_mask)
                    part_stats.append({
                        "part": part,
                        "target_tokens": int((part_target_mask_raw[0, :, 0] > 0.05).sum().item()),
                        "reference_tokens": int((part_ref_mask[0, :] > 0.05).sum().item()),
                    })
                output[target_slice] = target_current
                if target_combined is None:
                    target_mask_raw = self.controller.target_query_mask(
                        sequence_length=sequence_length,
                        batch_size=target_batch,
                        dtype=query.dtype,
                        device=query.device,
                        height=height,
                        width=width,
                    )
                    ref_mask = self.controller.reference_key_mask(
                        sequence_length=sequence_length,
                        batch_size=source_batch,
                        dtype=query.dtype,
                        device=query.device,
                        height=height,
                        width=width,
                    )
                else:
                    target_mask_raw = target_combined
                    ref_mask = reference_combined
            else:
                ref_mask = self.controller.reference_key_mask(
                    sequence_length=sequence_length,
                    batch_size=source_batch,
                    dtype=query.dtype,
                    device=query.device,
                    height=height,
                    width=width,
                )
                # PyTorch SDPA 的 float mask 是加到 attention logits 上的；非脸 token 给大负数。
                ref_attn_mask = (1.0 - ref_mask).view(source_batch, 1, 1, sequence_length) * -10000.0
                ref_output = F.scaled_dot_product_attention(
                    q_target,
                    k_ref,
                    v_ref,
                    attn_mask=ref_attn_mask,
                    dropout_p=0.0,
                    is_causal=False,
                )

                target_mask_raw = self.controller.target_query_mask(
                    sequence_length=sequence_length,
                    batch_size=target_batch,
                    dtype=query.dtype,
                    device=query.device,
                    height=height,
                    width=width,
                )
                target_mask = target_mask_raw.view(target_batch, 1, sequence_length, 1)
                alpha = target_mask * self.controller.strength
                output[target_slice] = self_output[target_slice] * (1 - alpha) + ref_output * alpha

            target_tokens = int((target_mask_raw[0, :, 0] > 0.05).sum().item())
            reference_tokens = int((ref_mask[0, :] > 0.05).sum().item())
            debug_map = None
            if self.controller.should_write_debug_map(self.layer_index, height, width, sequence_length):
                debug_batch = -1
                q_dbg = q_target[debug_batch].float()
                k_dbg = k_ref[debug_batch].float()
                logits = torch.matmul(q_dbg, k_dbg.transpose(-1, -2)) * (head_dim ** -0.5)
                ref_attn_mask = (1.0 - ref_mask).view(source_batch, 1, 1, sequence_length) * -10000.0
                logits = logits + ref_attn_mask[debug_batch].float()
                probs = logits.softmax(dim=-1)
                query_weights = target_mask_raw[debug_batch, :, 0].float()
                query_weights = query_weights / (query_weights.sum() + 1e-8)
                attention_heat = (probs.mean(dim=0) * query_weights[:, None]).sum(dim=0)
                debug_height, debug_width = height, width
                if debug_height is None or debug_width is None:
                    debug_side = int(sequence_length ** 0.5)
                    debug_height = debug_width = debug_side
                debug_map = self.controller.write_debug_attention_map(
                    layer_index=self.layer_index,
                    layer_name=self.layer_name,
                    height=debug_height,
                    width=debug_width,
                    attention_heat=attention_heat,
                    ref_mask=ref_mask[debug_batch],
                )

            self.controller.record_layer_stat(
                layer_index=self.layer_index,
                layer_name=self.layer_name,
                height=height,
                width=width,
                sequence_length=sequence_length,
                target_token_count=target_tokens,
                reference_token_count=reference_tokens,
            )
            if self.controller.layer_stats:
                self.controller.layer_stats[-1]["mask_mode"] = (
                    "part_guided" if self.controller.part_guided_enabled() else "face"
                )
                if part_stats:
                    self.controller.layer_stats[-1]["part_stats"] = part_stats
            if debug_map and self.controller.layer_stats:
                self.controller.layer_stats[-1]["attention_debug"] = debug_map

        hidden_states = output.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


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
        self._reference_renoise_cache = {}
        self._ip_adapter_reference_embed_cache = {}
        self._attention_controller = None
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
        self._install_reference_attention_processors(pipe)

        self._pipe = pipe
        return pipe

    def _install_reference_attention_processors(self, pipe):
        """把 reference self-attention processor 挂到 UNet 的 attention 层上。

        diffusers 的 UNet 里有很多 attention processor：
        - self-attention：当前图像 token 彼此关注，名字里通常是 attn1。
        - cross-attention：图像 token 关注文本 token，名字里通常是 attn2。

        我们只替换 self-attention。这样注入发生在 UNet 多个尺度的自注意力块里，
        不改文本条件的 cross-attention，避免把“角色参考脸”误当作文本语义去混。
        """

        self._attention_controller = ReferenceSelfAttentionController(
            image_width=self.width,
            image_height=self.height,
        )

        self_attn_names = [name for name in pipe.unet.attn_processors if name.endswith("attn1.processor")]
        self._attention_controller.configure_layers(
            total_layers=len(self_attn_names),
            pipeline_type=self.pipeline_type,
        )

        processors = {}
        self_attn_index = 0
        for name, processor in pipe.unet.attn_processors.items():
            if name.endswith("attn1.processor"):
                processors[name] = ReferenceSelfAttnProcessor(
                    name,
                    self_attn_index,
                    self._attention_controller,
                )
                self_attn_index += 1
            else:
                processors[name] = processor
        pipe.unet.set_attn_processor(processors)

    def _reference_attention_enabled(self, injection_plan: dict):
        """判断当前窗口是否需要启用 reference self-attention 注入。"""

        if os.getenv("MULTISHOT_INJECTION_MODE", "attention") != "attention":
            return False
        injection_lambda = float(injection_plan.get("lambda", 0.0) or 0.0)
        targets = injection_plan.get("targets", [])
        return injection_lambda > 0 and bool(targets)

    def _reference_attention_strength(self, injection_plan: dict):
        """把策略层的 lambda 映射成 attention 融合强度。"""

        injection_lambda = float(injection_plan.get("lambda", 0.0) or 0.0)
        scale = float(os.getenv("MULTISHOT_ATTENTION_INJECTION_SCALE", "0.85"))
        return max(0.0, min(1.0, injection_lambda * scale))

    def _reference_latents_for_attention(self, injection_plan: dict):
        """读取当前窗口要用的参考脸 x0 latent。

        当前先取第一个目标人脸作为 reference attention 的 source。
        多人脸更完整的版本应该为每张脸构造独立 reference 分支。
        """

        for target in injection_plan.get("targets", []):
            reference_image = target.get("reference_image")
            if reference_image and Path(reference_image).exists():
                reference_latents = self._encode_reference_image(reference_image)
                return reference_latents.to(device=self.device, dtype=self.dtype), reference_image
        return None, None

    def _reference_noise_for_attention(self, reference_image: str, reference_latents):
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

    def _reference_renoise_prompt(self):
        return os.getenv(
            "MULTISHOT_RENOISE_REFERENCE_PROMPT",
            (
                "side profile portrait of a middle-aged Asian man with metal-framed glasses, "
                "short black hair, clean white background, studio portrait lighting, "
                "realistic face, sharp facial features"
            ),
        )

    def _reference_renoise_trajectory(self, reference_image: str, runtime: dict):
        """用 ReNoise inversion 为参考脸预计算同一 scheduler 上的 x_t 轨迹。"""

        if self.pipeline_type != "sdxl":
            raise RuntimeError("ReNoise reference trajectory is only wired for SDXL pipelines")

        renoise_root = Path(os.getenv("MULTISHOT_RENOISE_ROOT", "/tmp/ReNoise-Inversion"))
        if not renoise_root.exists():
            raise FileNotFoundError(f"ReNoise repo not found: {renoise_root}")

        prompt = self._reference_renoise_prompt()
        steps = int(runtime.get("total_steps") or len(runtime.get("timesteps", [])) or self.default_steps)
        renoise_steps = int(os.getenv("MULTISHOT_RENOISE_STEPS", "1"))
        guidance_scale = float(os.getenv("MULTISHOT_RENOISE_GUIDANCE_SCALE", "0.0"))
        seed = int(os.getenv("MULTISHOT_RENOISE_SEED", os.getenv("MULTISHOT_DIFFUSION_SEED", "42")))
        cache_key = (
            reference_image,
            prompt,
            steps,
            renoise_steps,
            guidance_scale,
            seed,
            self.width,
            self.height,
            str(self.dtype),
            self.device,
        )
        if cache_key in self._reference_renoise_cache:
            return self._reference_renoise_cache[cache_key]

        import sys
        from PIL import Image

        if str(renoise_root) not in sys.path:
            sys.path.insert(0, str(renoise_root))
        from src.config import RunConfig
        from src.eunms import Model_Type, Scheduler_Type
        from src.pipes.sdxl_inversion_pipeline import SDXLDDIMPipeline
        from src.schedulers.ddim_scheduler import MyDDIMScheduler

        pipe = self._load()
        torch = self._torch
        reference = Image.open(reference_image).convert("RGB").resize((self.width, self.height))
        inversion_pipe = SDXLDDIMPipeline(**pipe.components)
        inversion_pipe.scheduler = MyDDIMScheduler.from_config(pipe.scheduler.config)
        inversion_pipe.cfg = RunConfig(
            model_type=Model_Type.SDXL,
            scheduler_type=Scheduler_Type.DDIM,
            seed=seed,
            num_inference_steps=steps,
            num_inversion_steps=steps,
            guidance_scale=guidance_scale,
            num_renoise_steps=renoise_steps,
            perform_noise_correction=False,
            noise_regularization_num_reg_steps=0,
        )
        inversion_pipe.set_progress_bar_config(disable=True)

        generator_device = self.device if self.device == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        with torch.no_grad():
            _, all_latents = inversion_pipe(
                prompt=prompt,
                image=reference,
                ip_adapter_image=reference if self.ip_adapter_enabled else None,
                num_inversion_steps=steps,
                num_inference_steps=steps,
                generator=generator,
                guidance_scale=guidance_scale,
                strength=1.0,
                denoising_start=0.0,
                num_renoise_steps=renoise_steps,
            )
        trajectory = [latent.detach().to(device=self.device, dtype=self.dtype) for latent in all_latents]
        self._reference_renoise_cache[cache_key] = trajectory
        return trajectory

    def _reference_xt_for_attention(self, reference_image: str, reference_latents, runtime: dict, step_index: int, timestep):
        """返回当前 denoise step 对应的参考脸 x_t。"""

        source = os.getenv("MULTISHOT_REFERENCE_TRAJECTORY", "add_noise").strip().lower()
        if source == "renoise":
            trajectory = self._reference_renoise_trajectory(reference_image, runtime)
            trajectory_index = max(0, min(len(trajectory) - 1, len(trajectory) - 1 - step_index))
            return trajectory[trajectory_index], f"renoise:{trajectory_index}"

        pipe = self._load()
        reference_noise = self._reference_noise_for_attention(reference_image, reference_latents)
        return pipe.scheduler.add_noise(reference_latents, reference_noise, timestep), "add_noise"

    def _duplicate_conditioning_for_reference(self, runtime: dict):
        """把 target 的文本条件复制一份给 reference batch。"""

        torch = self._torch
        prompt_embeds = torch.cat([runtime["prompt_embeds"], runtime["prompt_embeds"]], dim=0)
        added_cond_kwargs = runtime.get("added_cond_kwargs")
        if added_cond_kwargs is None:
            return prompt_embeds, None
        duplicated = {}
        for key, value in added_cond_kwargs.items():
            if isinstance(value, (list, tuple)):
                duplicated[key] = [torch.cat([item, item], dim=0) for item in value]
            else:
                duplicated[key] = torch.cat([value, value], dim=0)
        return prompt_embeds, duplicated

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

        默认用 reference_source_image，也就是干净 3D render；self-attention 仍使用
        reference_image 这个 match_target_scale 版本。
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

        默认注入模式是 masked mutual self-attention：
        1. 把 3D 参考脸图编码成 reference x0 latent。
        2. 用固定 reference noise 把 reference x0 加噪到当前 timestep，得到 reference x_t。
        3. 把 [reference x_t, target x_t] 拼到同一个 UNet batch。
        4. self-attention processor 手动拆出 reference / target 的 Q/K/V。
        5. target face query 关注 reference face K/V，再用 M_target 和 lambda 融合。

        如果设置 MULTISHOT_INJECTION_MODE=latent_blend，则回退到旧的 VAE latent blending。
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
        injection_mode = os.getenv("MULTISHOT_INJECTION_MODE", "attention")

        # controller 状态仍然挂在同一个 UNet processor 上；同进程多个 rollout 需要串行进入 UNet。
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
                use_mutual_attention = self._reference_attention_enabled(injection_plan)
                if use_mutual_attention:
                    reference_latents, reference_image = self._reference_latents_for_attention(injection_plan)
                    if reference_latents is not None:
                        target_added_cond_kwargs = self._clone_added_cond_kwargs(runtime.get("added_cond_kwargs"))
                        target_added_cond_kwargs, dynamic_ip_log = self._apply_dynamic_ip_adapter_reference(
                            target_added_cond_kwargs,
                            runtime,
                            injection_plan,
                            step_index,
                        )
                        if dynamic_ip_log:
                            step_injections.append(dynamic_ip_log)
                        reference_xt, reference_xt_source = self._reference_xt_for_attention(
                            reference_image,
                            reference_latents,
                            runtime,
                            step_index,
                            timestep,
                        )
                        source_batch_size = target_model_input.shape[0]
                        if reference_xt.shape[0] != source_batch_size:
                            reference_xt = reference_xt.repeat(source_batch_size, 1, 1, 1)
                        reference_model_input = pipe.scheduler.scale_model_input(reference_xt, timestep)
                        model_input = torch.cat([reference_model_input, target_model_input], dim=0)
                        target_runtime = dict(runtime)
                        target_runtime["added_cond_kwargs"] = target_added_cond_kwargs
                        prompt_embeds, added_cond_kwargs = self._duplicate_conditioning_for_reference(target_runtime)
                        strength = self._reference_attention_strength(injection_plan)
                        debug_prefix = None
                        if os.getenv("MULTISHOT_ATTENTION_DEBUG", "0") == "1":
                            shot_id = runtime.get("shot_id", "shot")
                            candidate_id = injection_plan.get("candidate_id", "candidate")
                            debug_prefix = f"{shot_id}.step_{step_index + 1:03d}.{candidate_id}"
                        self._attention_controller.start_mutual(
                            strength=strength,
                            targets=injection_plan.get("targets", []),
                            source_batch_size=source_batch_size,
                            target_batch_size=target_model_input.shape[0],
                            debug_prefix=debug_prefix,
                            debug_reference_image=reference_image,
                        )
                        step_injections.append({
                            "step_index": step_index,
                            "mode": "masked_mutual_self_attention",
                            "reference_image": reference_image,
                            "reference_xt_source": reference_xt_source,
                            "lambda": float(injection_plan.get("lambda", 0.0) or 0.0),
                            "effective_strength": round(strength, 4),
                            "target_count": len(injection_plan.get("targets", [])),
                            "layer_range": [
                                self._attention_controller.start_layer,
                                self._attention_controller.end_layer,
                            ],
                            "attention_mask_mode": os.getenv("MULTISHOT_ATTENTION_MASK_MODE", "face"),
                            "attention_parts": os.getenv("MULTISHOT_ATTENTION_PARTS", "eyes,nose,mouth"),
                            "status": "applied",
                        })
                    else:
                        model_input = target_model_input
                        prompt_embeds = runtime["prompt_embeds"]
                        added_cond_kwargs = runtime.get("added_cond_kwargs")
                        self._attention_controller.stop()
                        step_injections.append({
                            "step_index": step_index,
                            "mode": "masked_mutual_self_attention",
                            "status": "skipped",
                            "reason": "reference image not found",
                        })
                else:
                    model_input = target_model_input
                    prompt_embeds = runtime["prompt_embeds"]
                    added_cond_kwargs = self._clone_added_cond_kwargs(runtime.get("added_cond_kwargs"))
                    added_cond_kwargs, dynamic_ip_log = self._apply_dynamic_ip_adapter_reference(
                        added_cond_kwargs,
                        runtime,
                        injection_plan,
                        step_index,
                    )
                    if dynamic_ip_log:
                        step_injections.append(dynamic_ip_log)
                    self._attention_controller.stop()

                with torch.no_grad():
                    noise_pred = pipe.unet(
                        model_input,
                        timestep,
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]
                if use_mutual_attention and step_injections and step_injections[0].get("status") == "applied":
                    step_injections[0]["attention_layer_stats"] = list(self._attention_controller.layer_stats)
                self._attention_controller.stop()

                if use_mutual_attention and noise_pred.shape[0] == target_model_input.shape[0] * 2:
                    noise_pred = noise_pred[target_model_input.shape[0]:]

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
