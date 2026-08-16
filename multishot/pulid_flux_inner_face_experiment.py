"""Single-shot PuLID-FLUX inner-face injection smoke experiment.

This runner deliberately lives outside the vendored PuLID snapshot.  It keeps
official PuLID cross attention enabled for both branches, forks one denoising
state at step 30, and only adds masked latent blending to the treatment branch.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image, ImageFilter
from safetensors.torch import load_file as load_sft
from torchvision.transforms.functional import normalize
from transformers import CLIPTextModel, CLIPTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PULID_ROOT = PROJECT_ROOT / "third_party" / "PuLID"
if str(PULID_ROOT) not in sys.path:
    sys.path.insert(0, str(PULID_ROOT))

from flux.sampling import get_noise, get_schedule, prepare, unpack  # noqa: E402
from flux.modules.conditioner import HFEmbedder  # noqa: E402
from flux.model import Flux  # noqa: E402
from flux.util import (  # noqa: E402
    configs,
    load_ae,
    load_flow_model,
)
from pulid.pipeline_flux import PuLIDPipeline  # noqa: E402
from pulid.utils import resize_numpy_image_long  # noqa: E402


DEFAULT_PROMPT = (
    "cinematic medium close-up portrait of a man standing beneath warm neon lights "
    "on a rainy night street, three-quarter view, natural skin texture, shallow depth "
    "of field, photorealistic, subtle rim light"
)
REFERENCE_PROMPT = (
    "a neutral studio portrait of the same person, centered face, natural skin texture, "
    "soft even light, plain gray background, photorealistic"
)
INNER_FACE_LABELS = (1, 2, 3, 4, 5, 10, 11, 12, 13)
CONSERVATIVE_MASK_POLICY = {
    "name": "conservative_inner_face_v2",
    "center_y_fraction": 0.55,
    "radius_x_fraction": 0.43,
    "radius_y_fraction": 0.43,
    "top_fraction": 0.16,
    "bottom_fraction": 0.90,
    "erosion_fraction": 0.02,
    "feather_radius_px": 2.0,
    "excluded_regions": [
        "hairline_and_upper_forehead",
        "temples",
        "outer_cheeks",
        "chin_edge",
        "ears_and_earrings",
    ],
}


class _LocalCLIPEmbedder(HFEmbedder):
    """HFEmbedder variant whose CLIP type does not depend on a repo-id prefix."""

    def __init__(self, path: str, max_length: int, **hf_kwargs):
        torch.nn.Module.__init__(self)
        self.is_clip = True
        self.max_length = max_length
        self.output_key = "pooler_output"
        self.tokenizer = CLIPTokenizer.from_pretrained(path, max_length=max_length)
        self.hf_module = CLIPTextModel.from_pretrained(path, **hf_kwargs)
        self.hf_module = self.hf_module.eval().requires_grad_(False)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _model_record(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _largest_face(app, image: Image.Image):
    rgb = np.asarray(image.convert("RGB"))
    faces = app.get(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not faces:
        raise RuntimeError("InsightFace did not detect a face")
    return max(faces, key=lambda face: float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])))


def _bbox(face, width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [float(value) for value in face.bbox]
    return [
        max(0, min(width - 1, int(round(x1)))),
        max(0, min(height - 1, int(round(y1)))),
        max(1, min(width, int(round(x2)))),
        max(1, min(height, int(round(y2)))),
    ]


def _expanded_square(bbox: list[int], width: int, height: int, margin: float = 0.18) -> list[int]:
    x1, y1, x2, y2 = bbox
    side = max(x2 - x1, y2 - y1) * (1.0 + margin * 2.0)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    left = max(0, int(round(cx - side / 2)))
    top = max(0, int(round(cy - side / 2)))
    right = min(width, int(round(cx + side / 2)))
    bottom = min(height, int(round(cy + side / 2)))
    return [left, top, right, bottom]


def _align_reference(
    reference: Image.Image,
    source_bbox: list[int],
    target_bbox: list[int],
    canvas_size: tuple[int, int],
    reference_mode: str,
    document_deviation: str | None,
) -> tuple[Image.Image, dict]:
    """Match the rendered face scale and center to the target face bbox.

    This is the rectangular-canvas equivalent of the main experiment's
    ``match_target_scale`` layout: preserve the source aspect ratio, make the
    face no larger than the target bbox times the configured ratio, and align
    the two face centers.  The surrounding patch is retained for parsing and
    feathering, while the rest of the reference canvas stays neutral gray.
    """

    source_x1, source_y1, source_x2, source_y2 = [float(value) for value in source_bbox]
    target_x1, target_y1, target_x2, target_y2 = [float(value) for value in target_bbox]
    source_face_width = max(1.0, source_x2 - source_x1)
    source_face_height = max(1.0, source_y2 - source_y1)
    target_face_width = max(1.0, target_x2 - target_x1)
    target_face_height = max(1.0, target_y2 - target_y1)
    source_center_x = (source_x1 + source_x2) / 2.0
    source_center_y = (source_y1 + source_y2) / 2.0
    target_center_x = (target_x1 + target_x2) / 2.0
    target_center_y = (target_y1 + target_y2) / 2.0

    crop_ratio = 1.45
    source_crop_bbox = [
        max(0, int(round(source_center_x - source_face_width * crop_ratio / 2.0))),
        max(0, int(round(source_center_y - source_face_height * crop_ratio / 2.0))),
        min(reference.width, int(round(source_center_x + source_face_width * crop_ratio / 2.0))),
        min(reference.height, int(round(source_center_y + source_face_height * crop_ratio / 2.0))),
    ]
    source_crop = reference.crop(tuple(source_crop_bbox)).convert("RGB")

    face_scale_ratio = float(os.getenv("MULTISHOT_REFERENCE_FACE_SCALE_RATIO", "1.0"))
    desired_face_width = min(canvas_size[0] * 0.85, target_face_width * face_scale_ratio)
    desired_face_height = min(canvas_size[1] * 0.85, target_face_height * face_scale_ratio)
    scale = min(
        desired_face_width / source_face_width,
        desired_face_height / source_face_height,
    )
    resized_width = max(1, int(round(source_crop.width * scale)))
    resized_height = max(1, int(round(source_crop.height * scale)))
    source_resized = source_crop.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

    face_bbox_in_crop = [
        (source_x1 - source_crop_bbox[0]) * scale,
        (source_y1 - source_crop_bbox[1]) * scale,
        (source_x2 - source_crop_bbox[0]) * scale,
        (source_y2 - source_crop_bbox[1]) * scale,
    ]
    face_center_x = (face_bbox_in_crop[0] + face_bbox_in_crop[2]) / 2.0
    face_center_y = (face_bbox_in_crop[1] + face_bbox_in_crop[3]) / 2.0
    paste_x = int(round(target_center_x - face_center_x))
    paste_y = int(round(target_center_y - face_center_y))

    canvas = Image.new("RGB", canvas_size, (127, 127, 127))
    canvas.paste(source_resized, (paste_x, paste_y))
    face_bbox_on_layout = [
        int(round(face_bbox_in_crop[0] + paste_x)),
        int(round(face_bbox_in_crop[1] + paste_y)),
        int(round(face_bbox_in_crop[2] + paste_x)),
        int(round(face_bbox_in_crop[3] + paste_y)),
    ]
    target_paste_bbox = [
        max(0, paste_x),
        max(0, paste_y),
        min(canvas_size[0], paste_x + resized_width),
        min(canvas_size[1], paste_y + resized_height),
    ]
    return canvas, {
        "source_face_bbox": source_bbox,
        "source_crop_bbox": source_crop_bbox,
        "target_face_bbox": target_bbox,
        "target_paste_bbox": target_paste_bbox,
        "face_bbox_on_reference_layout": face_bbox_on_layout,
        "scale_x": scale,
        "scale_y": scale,
        "reference_face_scale_ratio": face_scale_ratio,
        "reference_to_target_face_scale": [
            round((face_bbox_on_layout[2] - face_bbox_on_layout[0]) / target_face_width, 4),
            round((face_bbox_on_layout[3] - face_bbox_on_layout[1]) / target_face_height, 4),
        ],
        "target_face_size": [round(target_face_width, 2), round(target_face_height, 2)],
        "reference_face_size_on_layout": [
            face_bbox_on_layout[2] - face_bbox_on_layout[0],
            face_bbox_on_layout[3] - face_bbox_on_layout[1],
        ],
        "layout_mode": "match_target_scale",
        "reference_mode": reference_mode,
        "document_deviation": document_deviation,
    }


def _build_facelift_asset_record(reference_path: Path, output: Path) -> dict:
    """Build FaceLift before FLUX is loaded so the two large models stay serial."""
    from multishot.facelift_backend import build_facelift_asset

    face_dir = output / "input" / "facelift"
    cached_result = output / "input" / "facelift_result.json"
    if cached_result.exists():
        result = json.loads(cached_result.read_text(encoding="utf-8"))
        model_path = Path(result.get("model_path") or "")
        if result.get("facelift_status") == "success" and model_path.exists():
            return result
    result = build_facelift_asset(str(reference_path), str(face_dir))
    if result.get("facelift_status") != "success":
        raise RuntimeError(f"FaceLift did not complete successfully: {result}")
    model_path = Path(result.get("model_path") or "")
    if not model_path.exists():
        raise RuntimeError(f"FaceLift Gaussian model is missing: {model_path}")
    _write_json(output / "input" / "facelift_result.json", result)
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _semantic_inner_face_mask(
    image: Image.Image,
    face_bbox: list[int],
    parsing_model,
    device: torch.device,
) -> Image.Image:
    crop_bbox = _expanded_square(face_bbox, image.width, image.height)
    crop = image.crop(tuple(crop_bbox)).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    tensor = torch.from_numpy(np.asarray(crop).copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    tensor = normalize(tensor.to(device), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    with torch.inference_mode():
        logits = parsing_model(tensor)[0]
        labels = logits.argmax(dim=1, keepdim=True)
        keep = torch.zeros_like(labels, dtype=torch.bool)
        for label in INNER_FACE_LABELS:
            keep |= labels == label
        mask = keep.float()
        mask = F.interpolate(
            mask,
            size=(crop_bbox[3] - crop_bbox[1], crop_bbox[2] - crop_bbox[0]),
            mode="nearest",
        )[0, 0]
    canvas = torch.zeros((image.height, image.width), dtype=torch.float32)
    canvas[crop_bbox[1] : crop_bbox[3], crop_bbox[0] : crop_bbox[2]] = mask.cpu()
    return Image.fromarray((canvas.numpy() * 255).astype(np.uint8), mode="L")


def _conservative_face_core_mask(mask: Image.Image, face_bbox: list[int]) -> Image.Image:
    """Restrict a semantic face mask to a conservative, scale-aware facial core.

    Face parsing already excludes ears, earrings, hair, neck, and clothing. The
    additional oval support and small erosion keep the skin class away from the
    hairline, temples, outer cheeks, and chin boundary where a rendered-domain
    mismatch is most visible.
    """
    values = np.asarray(mask.convert("L"), dtype=np.uint8)
    x1, y1, x2, y2 = [float(value) for value in face_bbox]
    face_width = max(1.0, x2 - x1)
    face_height = max(1.0, y2 - y1)
    center_x = (x1 + x2) / 2.0
    center_y = y1 + CONSERVATIVE_MASK_POLICY["center_y_fraction"] * face_height
    radius_x = CONSERVATIVE_MASK_POLICY["radius_x_fraction"] * face_width
    radius_y = CONSERVATIVE_MASK_POLICY["radius_y_fraction"] * face_height

    yy, xx = np.ogrid[: values.shape[0], : values.shape[1]]
    oval = ((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2 <= 1.0
    vertical_window = (
        (yy >= y1 + CONSERVATIVE_MASK_POLICY["top_fraction"] * face_height)
        & (yy <= y1 + CONSERVATIVE_MASK_POLICY["bottom_fraction"] * face_height)
    )
    restricted = np.where(oval & vertical_window, values, 0).astype(np.uint8)

    erosion_radius = max(
        1,
        int(round(min(face_width, face_height) * CONSERVATIVE_MASK_POLICY["erosion_fraction"])),
    )
    kernel_size = erosion_radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    restricted = cv2.erode(restricted, kernel, iterations=1)
    return Image.fromarray(restricted, mode="L")


def _token_mask(mask: Image.Image, height: int, width: int, device, dtype):
    token_h = math.ceil(height / 16)
    token_w = math.ceil(width / 16)
    resized = mask.resize((token_w, token_h), Image.Resampling.BILINEAR)
    values = torch.from_numpy(np.asarray(resized).copy()).float().div_(255.0)
    values = values.clamp(0, 1).reshape(1, token_h * token_w, 1)
    return values.to(device=device, dtype=dtype)


def _decode(ae, packed, height: int, width: int, device: torch.device) -> Image.Image:
    latent = unpack(packed.float(), height, width)
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        decoded = ae.decode(latent)
    decoded = decoded.clamp(-1, 1)[0]
    array = rearrange(decoded, "c h w -> h w c")
    return Image.fromarray((127.5 * (array + 1.0)).byte().cpu().numpy())


def _encode(ae, image: Image.Image, height: int, width: int, device: torch.device):
    resized = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    value = torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1).float().unsqueeze(0)
    value = value.to(device=device).div_(127.5).sub_(1.0)
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        latent = ae.encode(value)
    return rearrange(latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)


def _velocity(
    model,
    state,
    conditioning,
    timestep: float,
    guidance: float,
    id_embedding,
    id_weight: float,
):
    t_vec = torch.full((state.shape[0],), timestep, dtype=state.dtype, device=state.device)
    guidance_vec = torch.full((state.shape[0],), guidance, dtype=state.dtype, device=state.device)
    return model(
        img=state,
        img_ids=conditioning["img_ids"],
        txt=conditioning["txt"],
        txt_ids=conditioning["txt_ids"],
        y=conditioning["vec"],
        timesteps=t_vec,
        guidance=guidance_vec,
        id=id_embedding,
        id_weight=id_weight,
        aggressive_offload=False,
    )


def _reference_trajectory(
    model,
    reference_x0,
    conditioning,
    timesteps: list[float],
    guidance: float,
    id_embedding,
    id_weight: float,
    cache_mask,
):
    """Integrate the FLUX vector field from t=0 to t=1."""
    # Keep no reference content outside the injection support in the cache.
    # A binary support mask is used here so the feather is applied exactly once
    # later by the actual residual blend.
    support = (cache_mask > 0).to(dtype=reference_x0.dtype)
    trajectory = {len(timesteps) - 1: reference_x0.detach().clone() * support}
    current = reference_x0
    with torch.inference_mode():
        for target_index in range(len(timesteps) - 2, -1, -1):
            t_current = timesteps[target_index + 1]
            t_next = timesteps[target_index]
            pred = _velocity(
                model, current, conditioning, t_current, guidance, id_embedding, id_weight
            )
            current = current + (t_next - t_current) * pred
            trajectory[target_index] = current.detach().clone() * support
    return trajectory


def _face_similarity(app, first: Image.Image, second: Image.Image) -> float:
    a = np.asarray(_largest_face(app, first).embedding, dtype=np.float32)
    b = np.asarray(_largest_face(app, second).embedding, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _load_models(args, device: torch.device):
    text_root = PULID_ROOT / "models" / "xflux_text_encoders"
    clip_root = PULID_ROOT / "models" / "clip-vit-large-patch14"
    if not text_root.exists() or not clip_root.exists():
        raise FileNotFoundError(
            "Local FLUX text encoders are missing; run the experiment model downloader first"
        )
    t5 = HFEmbedder(
        str(text_root), max_length=args.max_sequence_length, torch_dtype=torch.bfloat16
    ).to(device)
    clip = _LocalCLIPEmbedder(str(clip_root), max_length=77, torch_dtype=torch.bfloat16).to(device)
    if args.fp8:
        from optimum.quanto import requantize

        checkpoint = PULID_ROOT / "models" / "flux-dev-fp8.safetensors"
        quant_map = PULID_ROOT / "models" / "flux_dev_quantization_map.json"
        if not checkpoint.exists() or not quant_map.exists():
            raise FileNotFoundError("Local FP8 FLUX checkpoint or quantization map is missing")
        model = Flux(configs["flux-dev"].params).to(torch.bfloat16)
        state_dict = load_sft(str(checkpoint), device="cpu")
        requantize(model, state_dict, json.loads(quant_map.read_text(encoding="utf-8")), device="cpu")
        del state_dict
    else:
        model = load_flow_model("flux-dev", device="cpu")
    model.eval()
    ae = load_ae("flux-dev", device="cpu")

    eva_path = PULID_ROOT / "models" / "eva_clip" / "EVA02_CLIP_L_336_psz14_s6B.pt"
    antelope_path = PULID_ROOT / "models" / "antelopev2"
    if not eva_path.exists() or not antelope_path.exists():
        raise FileNotFoundError("Local EVA-CLIP or AntelopeV2 weights are missing")
    import eva_clip.pretrained as eva_pretrained
    import pulid.pipeline_flux as pipeline_flux_module

    original_eva_download = eva_pretrained.download_pretrained_from_hf

    def local_eva_download(model_id, filename="open_clip_pytorch_model.bin", revision=None, cache_dir=None):
        if model_id == "QuanSun/EVA-CLIP" and filename == eva_path.name:
            return str(eva_path)
        return original_eva_download(model_id, filename, revision=revision, cache_dir=cache_dir)

    eva_pretrained.download_pretrained_from_hf = local_eva_download
    pipeline_flux_module.snapshot_download = lambda *unused_args, **unused_kwargs: str(antelope_path)
    pulid = PuLIDPipeline(model, device="cpu", weight_dtype=torch.bfloat16, onnx_provider=args.onnx_provider)
    pulid_weight = PULID_ROOT / "models" / "pulid_flux_v0.9.1.safetensors"
    state_dict = load_sft(str(pulid_weight), device="cpu")
    grouped = {}
    for key, value in state_dict.items():
        module, child_key = key.split(".", 1)
        grouped.setdefault(module, {})[child_key] = value
    for module, weights in grouped.items():
        getattr(pulid, module).load_state_dict(weights, strict=True)
    del state_dict, grouped
    return model, ae, t5, clip, pulid


def _preflight_model_files(args) -> None:
    required = {
        PULID_ROOT / "models" / "xflux_text_encoders" / "model-00001-of-00002.safetensors": 4_900_000_000,
        PULID_ROOT / "models" / "xflux_text_encoders" / "model-00002-of-00002.safetensors": 4_400_000_000,
        PULID_ROOT / "models" / "clip-vit-large-patch14" / "model.safetensors": 1_700_000_000,
        PULID_ROOT / "models" / "eva_clip" / "EVA02_CLIP_L_336_psz14_s6B.pt": 850_000_000,
        PULID_ROOT / "models" / "pulid_flux_v0.9.1.safetensors": 1_000_000_000,
        PULID_ROOT / "models" / "antelopev2" / "glintr100.onnx": 250_000_000,
        PULID_ROOT / "models" / "ae.safetensors": 150_000_000,
    }
    if args.fp8:
        required[PULID_ROOT / "models" / "flux-dev-fp8.safetensors"] = 11_000_000_000
    if args.build_facelift:
        facelift_root = PROJECT_ROOT / "third_party" / "FaceLift"
        required.update(
            {
                facelift_root / "checkpoints" / "gslrm" / "ckpt_0000000000021125.pt": 3_700_000_000,
                facelift_root
                / "checkpoints/mvdiffusion/pipeckpts/unet/diffusion_pytorch_model.safetensors": 3_400_000_000,
                facelift_root
                / "checkpoints/mvdiffusion/pipeckpts/image_encoder/model.safetensors": 1_200_000_000,
                facelift_root
                / "checkpoints/mvdiffusion/pipeckpts/text_encoder/model.safetensors": 650_000_000,
                facelift_root
                / "checkpoints/mvdiffusion/pipeckpts/vae/diffusion_pytorch_model.safetensors": 160_000_000,
                facelift_root
                / "mvdiffusion/data/fixed_prompt_embeds_6view/clr_embeds.pt": 500_000,
            }
        )
    bad = [
        f"{path} (missing or smaller than {minimum} bytes)"
        for path, minimum in required.items()
        if not path.exists() or path.stat().st_size < minimum
    ]
    partials = list((PULID_ROOT / "models").rglob("*.aria2"))
    if args.build_facelift:
        partials.extend((PROJECT_ROOT / "third_party" / "FaceLift" / "checkpoints").rglob("*.aria2"))
    if bad or partials:
        details = bad + [f"unfinished aria2 download: {path}" for path in partials]
        raise FileNotFoundError("Model preflight failed:\n- " + "\n- ".join(details))


def run(args) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    reference_path = Path(args.reference_image).resolve()
    if not reference_path.exists():
        raise FileNotFoundError(reference_path)
    _preflight_model_files(args)

    output = Path(args.output_dir).resolve()
    step_dir = output / "step_30"
    control_dir = output / "control"
    treatment_dir = output / "treatment"
    for path in (step_dir, control_dir, treatment_dir, output / "input", output / "trajectory"):
        path.mkdir(parents=True, exist_ok=True)

    facelift_asset = _build_facelift_asset_record(reference_path, output) if args.build_facelift else None

    device = torch.device("cuda")
    old_cwd = Path.cwd()
    os.chdir(PULID_ROOT)
    try:
        model, ae, t5, clip, pulid = _load_models(args, device)
        reference = Image.open(reference_path).convert("RGB")
        reference.save(output / "input" / "identity_reference.png")
        (output / "input" / "prompt.txt").write_text(args.prompt, encoding="utf-8")

        noise = get_noise(1, args.height, args.width, device=device, dtype=torch.bfloat16, seed=args.seed)
        timesteps = get_schedule(args.steps, noise.shape[-1] * noise.shape[-2] // 4, shift=True)

        t5.to(device)
        clip.to(device)
        target_cond = prepare(t5=t5, clip=clip, img=noise, prompt=args.prompt)
        reference_cond = prepare(t5=t5, clip=clip, img=noise, prompt=args.reference_prompt)
        t5.cpu()
        clip.cpu()
        del t5, clip
        gc.collect()
        torch.cuda.empty_cache()

        pulid.components_to_device(device)
        # PuLID's helper moves the RetinaFace module but leaves its explicit
        # device attributes unchanged. Keep those fields synchronized so the
        # detector constructs inputs on the same device as its weights.
        pulid.device = device
        pulid.face_helper.device = device
        pulid.face_helper.face_det.device = device
        pulid.face_helper.face_det.mean_tensor = pulid.face_helper.face_det.mean_tensor.to(device)
        resized_reference = resize_numpy_image_long(np.asarray(reference), 1024)
        id_embedding, _ = pulid.get_id_embedding(resized_reference, cal_uncond=False)
        cpu_device = torch.device("cpu")
        pulid.components_to_device(cpu_device)
        pulid.device = cpu_device
        pulid.face_helper.device = cpu_device
        pulid.face_helper.face_det.device = cpu_device
        pulid.face_helper.face_det.mean_tensor = pulid.face_helper.face_det.mean_tensor.to(cpu_device)
        model.to(device)
        ae.to(device)
        torch.cuda.empty_cache()

        current = target_cond["img"]
        step_logs = []
        with torch.inference_mode():
            for step in range(args.inject_start):
                t, t_next = timesteps[step], timesteps[step + 1]
                pred = _velocity(
                    model,
                    current,
                    target_cond,
                    t,
                    args.guidance,
                    id_embedding,
                    args.pulid_id_weight,
                )
                current = current + (t_next - t) * pred
                step_logs.append({"step": step, "timestep": t, "injected": False})

            detect_t = timesteps[args.inject_start]
            detect_pred = _velocity(
                model,
                current,
                target_cond,
                detect_t,
                args.guidance,
                id_embedding,
                args.pulid_id_weight,
            )
            pred_x0 = current - detect_t * detect_pred
            preview = _decode(ae, pred_x0, args.height, args.width, device)
            preview.save(step_dir / "pred_x0.png")

        actual_inject_start = args.inject_start
        detection_failures = []
        while True:
            try:
                target_face = _largest_face(pulid.app, preview)
                if float(target_face.det_score) < args.min_face_confidence:
                    raise RuntimeError(
                        f"face confidence {float(target_face.det_score):.4f} is below "
                        f"{args.min_face_confidence:.4f}"
                    )
                break
            except RuntimeError as exc:
                detection_failures.append({"step": actual_inject_start, "reason": str(exc)})
                if actual_inject_start >= args.steps - 1:
                    raise RuntimeError(
                        "No reliable face was detected before the final denoising step: "
                        f"{detection_failures}"
                    ) from exc
                t, t_next = timesteps[actual_inject_start], timesteps[actual_inject_start + 1]
                current = current + (t_next - t) * detect_pred
                step_logs.append(
                    {
                        "step": actual_inject_start,
                        "timestep": t,
                        "next_timestep": t_next,
                        "injected": False,
                        "reason": "face_detection_delayed",
                        "detection_error": str(exc),
                    }
                )
                actual_inject_start += 1
                detect_t = timesteps[actual_inject_start]
                detect_pred = _velocity(
                    model,
                    current,
                    target_cond,
                    detect_t,
                    args.guidance,
                    id_embedding,
                    args.pulid_id_weight,
                )
                pred_x0 = current - detect_t * detect_pred
                preview = _decode(ae, pred_x0, args.height, args.width, device)
                preview.save(step_dir / f"pred_x0_retry_step_{actual_inject_start}.png")
        preview.save(step_dir / "pred_x0.png")
        target_bbox = _bbox(target_face, preview.width, preview.height)
        pose = [float(value) for value in getattr(target_face, "pose", [0.0, 0.0, 0.0])]
        _write_json(
            step_dir / "face_detection.json",
            {
                "bbox": target_bbox,
                "pitch_yaw_roll": pose,
                "det_score": float(target_face.det_score),
                "requested_start_step": args.inject_start,
                "actual_start_step": actual_inject_start,
                "detection_failures": detection_failures,
            },
        )
        absolute_yaw = abs(pose[1])
        if not args.min_abs_yaw <= absolute_yaw <= args.max_abs_yaw:
            raise RuntimeError(
                f"Detected absolute yaw {absolute_yaw:.4f} degrees is outside the required "
                f"range [{args.min_abs_yaw:.4f}, {args.max_abs_yaw:.4f}]"
            )

        if facelift_asset:
            # Reuse the main experiment's continuous Gaussian renderer.  The
            # FaceLift reconstruction model has already been released; this
            # path only reloads the filtered PLY and invokes the lightweight
            # Gaussian rasterizer for the detected target pose.
            from multishot.mcp_asset_server import _render_3d_face_reference

            pitch, yaw, roll = (pose + [0.0, 0.0, 0.0])[:3]
            face_pose = {"pitch": pitch, "yaw": yaw, "roll": roll}
            gaussian_model_path = Path(facelift_asset["model_path"])
            gaussian_asset_dir = Path(
                facelift_asset.get("facelift_output_dir") or gaussian_model_path.parent
            )
            rendered_path = _render_3d_face_reference(
                {
                    "model_path": str(gaussian_model_path),
                    "path": str(gaussian_asset_dir),
                },
                face_pose,
                target_bbox,
            )
            if not rendered_path:
                raise RuntimeError(
                    "Continuous FaceLift Gaussian rendering failed; discrete-view fallback is disabled"
                )
            rendered_reference = Image.open(rendered_path).convert("RGB")
            render_meta_path = Path(rendered_path).with_suffix(".meta.json")
            render_meta = (
                json.loads(render_meta_path.read_text(encoding="utf-8"))
                if render_meta_path.exists()
                else {}
            )
            view_meta = {
                "target_pitch_yaw_roll": [pitch, yaw, roll],
                "gaussian_model_path": str(gaussian_model_path),
                "gaussian_render_path": str(rendered_path),
                "gaussian_render_meta_path": str(render_meta_path),
                "gaussian_camera": render_meta.get("camera"),
            }
            reference_mode = "facelift_continuous_gaussian_pose_render"
            document_deviation = None
            gc.collect()
            torch.cuda.empty_cache()
        else:
            rendered_reference = reference
            view_meta = {"target_pitch_yaw_roll": pose}
            reference_mode = "aligned_2d_identity_reference_smoke"
            document_deviation = "FaceLift was explicitly disabled for this smoke run"
        source_face = _largest_face(pulid.app, rendered_reference)
        source_bbox = _bbox(source_face, rendered_reference.width, rendered_reference.height)
        aligned_reference, align_meta = _align_reference(
            rendered_reference,
            source_bbox,
            target_bbox,
            (preview.width, preview.height),
            reference_mode,
            document_deviation,
        )
        align_meta.update(view_meta)
        aligned_reference.save(step_dir / "aligned_3d_face.png")
        rendered_reference.save(step_dir / "rendered_3d_face.png")
        _write_json(step_dir / "alignment.json", align_meta)

        pulid.face_helper.face_parse.to(device)
        target_semantic_mask = _semantic_inner_face_mask(
            preview, target_bbox, pulid.face_helper.face_parse, device
        )
        aligned_face = _largest_face(pulid.app, aligned_reference)
        aligned_bbox = _bbox(aligned_face, aligned_reference.width, aligned_reference.height)
        reference_semantic_mask = _semantic_inner_face_mask(
            aligned_reference, aligned_bbox, pulid.face_helper.face_parse, device
        )
        pulid.face_helper.face_parse.cpu()
        target_semantic_mask.save(step_dir / "target_semantic_inner_face_mask.png")
        reference_semantic_mask.save(step_dir / "reference_semantic_inner_face_mask.png")
        target_mask = _conservative_face_core_mask(target_semantic_mask, target_bbox)
        reference_mask = _conservative_face_core_mask(reference_semantic_mask, aligned_bbox)
        target_mask.save(step_dir / "target_inner_face_mask.png")
        reference_mask.save(step_dir / "reference_inner_face_mask.png")
        intersection = np.minimum(np.asarray(target_mask), np.asarray(reference_mask)).astype(np.uint8)
        final_mask = Image.fromarray(intersection, mode="L").filter(
            ImageFilter.GaussianBlur(radius=CONSERVATIVE_MASK_POLICY["feather_radius_px"])
        )
        final_mask.save(step_dir / "final_inner_face_mask.png")
        packed_mask = _token_mask(final_mask, args.height, args.width, device, current.dtype)

        reference_x0 = _encode(ae, aligned_reference, args.height, args.width, device).to(current.dtype)
        trajectory_started = time.perf_counter()
        trajectory = _reference_trajectory(
            model,
            reference_x0,
            reference_cond,
            timesteps,
            args.guidance,
            id_embedding,
            args.pulid_id_weight,
            packed_mask,
        )
        _write_json(
            output / "trajectory" / "metadata.json",
            {
                "states": len(trajectory),
                "dtype": str(reference_x0.dtype),
                "shape": list(reference_x0.shape),
                "seconds": time.perf_counter() - trajectory_started,
                "reference_mode": align_meta["reference_mode"],
            },
        )

        control = current.detach().clone()
        treatment = current.detach().clone()
        with torch.inference_mode():
            for step in range(actual_inject_start, args.steps):
                t, t_next = timesteps[step], timesteps[step + 1]
                control_pred = detect_pred if step == actual_inject_start else _velocity(
                    model,
                    control,
                    target_cond,
                    t,
                    args.guidance,
                    id_embedding,
                    args.pulid_id_weight,
                )
                control_next = control + (t_next - t) * control_pred

                treatment_pred = detect_pred if step == actual_inject_start else _velocity(
                    model,
                    treatment,
                    target_cond,
                    t,
                    args.guidance,
                    id_embedding,
                    args.pulid_id_weight,
                )
                treatment_base = treatment + (t_next - t) * treatment_pred
                reference_next = trajectory[step + 1].to(device=device, dtype=treatment_base.dtype)
                residual = args.injection_strength * packed_mask * (reference_next - treatment_base)
                treatment_next = treatment_base + residual
                finite = bool(torch.isfinite(treatment_next).all().item())
                step_logs.append(
                    {
                        "step": step,
                        "timestep": t,
                        "next_timestep": t_next,
                        "injected": True,
                        "actual_plugin_start_step": actual_inject_start,
                        "render_pitch_yaw_roll": pose,
                        "reference_trajectory_index": step + 1,
                        "mask_token_count": int((packed_mask > 0.01).sum().item()),
                        "mask_fraction": float((packed_mask > 0.01).float().mean().item()),
                        "target_next_base_norm": float(treatment_base.float().norm().item()),
                        "reference_next_norm": float(reference_next.float().norm().item()),
                        "injection_residual_norm": float(residual.float().norm().item()),
                        "target_next_norm": float(treatment_next.float().norm().item()),
                        "finite": finite,
                    }
                )
                if not finite:
                    raise FloatingPointError(f"NaN/Inf after injection at step {step}")
                control, treatment = control_next, treatment_next

        control_image = _decode(ae, control, args.height, args.width, device)
        treatment_image = _decode(ae, treatment, args.height, args.width, device)
        control_image.save(control_dir / "final.png")
        treatment_image.save(treatment_dir / "final.png")
        with (output / "step_log.jsonl").open("w", encoding="utf-8") as handle:
            for item in step_logs:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

        metrics = {
            "reference_control_insightface_cosine": _face_similarity(pulid.app, reference, control_image),
            "reference_treatment_insightface_cosine": _face_similarity(pulid.app, reference, treatment_image),
            "rendered_3d_control_insightface_cosine": _face_similarity(
                pulid.app, rendered_reference, control_image
            ),
            "rendered_3d_treatment_insightface_cosine": _face_similarity(
                pulid.app, rendered_reference, treatment_image
            ),
        }
        metrics["treatment_minus_control"] = (
            metrics["reference_treatment_insightface_cosine"]
            - metrics["reference_control_insightface_cosine"]
        )
        metrics["rendered_3d_treatment_minus_control"] = (
            metrics["rendered_3d_treatment_insightface_cosine"]
            - metrics["rendered_3d_control_insightface_cosine"]
        )
        _write_json(output / "metrics.json", metrics)

        model_path = PULID_ROOT / "models" / (
            "flux-dev-fp8.safetensors" if args.fp8 else "flux1-dev.safetensors"
        )
        pulid_path = PULID_ROOT / "models" / "pulid_flux_v0.9.1.safetensors"
        _write_json(
            output / "config.json",
            {
                "prompt": args.prompt,
                "reference_prompt": args.reference_prompt,
                "seed": args.seed,
                "width": args.width,
                "height": args.height,
                "steps": args.steps,
                "guidance": args.guidance,
                "true_cfg": 1.0,
                "pulid_id_weight": args.pulid_id_weight,
                "pulid_start_step": 0,
                "plugin_start_step": args.inject_start,
                "plugin_actual_start_step": actual_inject_start,
                "plugin_strength": args.injection_strength,
                "required_absolute_yaw_range": [args.min_abs_yaw, args.max_abs_yaw],
                "mask_type": CONSERVATIVE_MASK_POLICY["name"],
                "mask_policy": CONSERVATIVE_MASK_POLICY,
                "fp8": args.fp8,
                "reference_image": str(reference_path),
                "reference_image_sha256": _sha256(reference_path),
                "reference_image_generated": args.reference_generated,
                "reference_image_origin": args.reference_origin,
                "reference_mode": align_meta["reference_mode"],
                "facelift_step_2d": int(os.getenv("MULTISHOT_FACELIFT_STEP_2D", "50")),
                "document_deviations": [align_meta["document_deviation"]]
                if align_meta["document_deviation"]
                else [],
                "models": {
                    "flux": _model_record(model_path),
                    "pulid": _model_record(pulid_path),
                    "ae": _model_record(PULID_ROOT / "models" / "ae.safetensors"),
                    "t5_shard_1": _model_record(
                        PULID_ROOT
                        / "models/xflux_text_encoders/model-00001-of-00002.safetensors"
                    ),
                    "t5_shard_2": _model_record(
                        PULID_ROOT
                        / "models/xflux_text_encoders/model-00002-of-00002.safetensors"
                    ),
                    "clip": _model_record(
                        PULID_ROOT / "models/clip-vit-large-patch14/model.safetensors"
                    ),
                    "eva_clip": _model_record(
                        PULID_ROOT
                        / "models/eva_clip/EVA02_CLIP_L_336_psz14_s6B.pt"
                    ),
                    **(
                        {
                            "facelift_gslrm": _model_record(
                                PROJECT_ROOT
                                / "third_party/FaceLift/checkpoints/gslrm/ckpt_0000000000021125.pt"
                            ),
                            "facelift_unet": _model_record(
                                PROJECT_ROOT
                                / "third_party/FaceLift/checkpoints/mvdiffusion/pipeckpts/unet/diffusion_pytorch_model.safetensors"
                            ),
                        }
                        if args.build_facelift
                        else {}
                    ),
                },
            },
        )
        return output
    finally:
        os.chdir(old_cwd)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-image", required=True)
    parser.add_argument(
        "--reference-origin",
        default="external_input_source_url_not_recorded",
        help="Human-readable provenance recorded in config.json",
    )
    parser.add_argument(
        "--reference-generated",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record whether the reference image was generated by this project",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "experiment_output" / "pulid_flux_conservative_mask_04"),
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--reference-prompt", default=REFERENCE_PROMPT)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--inject-start", type=int, default=30)
    parser.add_argument(
        "--injection-strength",
        type=float,
        default=0.4,
        help="Fixed at 0.4 for the conservative-mask experiment",
    )
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--pulid-id-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--onnx-provider", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--min-face-confidence", type=float, default=0.5)
    parser.add_argument(
        "--min-abs-yaw",
        type=float,
        default=0.0,
        help="Reject the run at detection time when absolute yaw is below this value",
    )
    parser.add_argument(
        "--max-abs-yaw",
        type=float,
        default=180.0,
        help="Reject the run at detection time when absolute yaw is above this value",
    )
    parser.add_argument("--fp8", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--build-facelift", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.steps != 50 or args.inject_start != 30:
        raise ValueError("This experiment is fixed to 50 steps with injection starting at step 30")
    if abs(args.injection_strength - 0.4) > 1e-8:
        raise ValueError("Injection strength is fixed to 0.4 for the conservative-mask experiment")
    if args.pulid_id_weight < 0:
        raise ValueError("PuLID id weight must be non-negative")
    if args.min_abs_yaw < 0 or args.max_abs_yaw < args.min_abs_yaw:
        raise ValueError("Yaw bounds must satisfy 0 <= min_abs_yaw <= max_abs_yaw")
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(f"experiment complete: {result}")
