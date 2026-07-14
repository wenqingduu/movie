import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import os
import random
from pathlib import Path
from mcp.server.fastmcp import FastMCP

from .diffusion_backend import get_diffusion_backend
from .face_analysis_backend import cosine_similarity, get_face_backend
from .facelift_backend import build_facelift_asset


# MCP server 通过环境变量拿当前项目目录。
# 这样工具 schema 里不会出现 project_dir，模型也不能决定文件写到哪里。
PROJECT_DIR_ENV = "MULTISHOT_PROJECT_DIR"


def _project_dir():
    return Path(os.environ[PROJECT_DIR_ENV])


def _load_index():
    """读取项目级资产索引。

    asset_index.json 是当前阶段的“资产记忆”：
    后续生成 shot 首帧时，会根据 subscript_id / character_id 从这里查路径。
    """

    index_path = _project_dir() / "asset_index.json"
    if index_path.exists():
        return json.loads(index_path.read_text(encoding="utf-8"))
    return {"scene_assets": {}, "character_assets": {}}


def _save_index(data: dict):
    """保存项目级资产索引。"""

    index_path = _project_dir() / "asset_index.json"
    index_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(index_path)


def _json_safe(data):
    """把运行时对象转成可以写入 JSON 的结构。

    diffusion 真实运行时会把 torch tensor 放在 _latents / _noise_pred
    这类私有字段里，供后续窗口继续去噪。日志只保存可读元数据，
    所有下划线开头的字段都过滤掉。
    """

    if isinstance(data, dict):
        return {
            key: _json_safe(value)
            for key, value in data.items()
            if not key.startswith("_")
        }
    if isinstance(data, list):
        return [_json_safe(item) for item in data]
    return data


def _generate_image(prompt: str, output_path: str, generation_model: str | None = None):
    """调用开源 diffusion 模型生成图片。

    当前默认模型是 Juggernaut XL v9。
    这不是自写模型，只是用 diffusers 加载本地开源模型。
    """

    backend = get_diffusion_backend(generation_model)
    return backend.generate_image(prompt, output_path)


mcp = FastMCP("multishot-assets")


@mcp.tool()
def generate_scene_asset(subscript_id: str, scene_name: str, prompt: str):
    """生成并保存一个场景背景资产。

    subscript_id: 场景子剧本 id，例如 scene_001。
    scene_name: 稳定英文场景名，例如 rainy_alley。
    prompt: 用于图像生成模型的背景图提示词。
    """

    image_path = _project_dir() / "assets" / "scenes" / f"{subscript_id}.png"
    saved_path = _generate_image(prompt, str(image_path))

    asset = {
        "asset_id": f"{subscript_id}_background",
        "scene_name": scene_name,
        "prompt": prompt,
        "path": saved_path,
    }

    # 在工具侧更新资产记忆，而不是让 Agent 自己维护路径细节。
    index = _load_index()
    index["scene_assets"][subscript_id] = asset
    index_path = _save_index(index)

    return {"subscript_id": subscript_id, "index_path": index_path, **asset}


@mcp.tool()
def generate_character_asset(character_id: str, character_name: str, prompt: str):
    """生成并保存一个人物参考图资产。

    character_id: 人物 id，例如 char_001。
    character_name: 稳定英文人物名，例如 young_girl。
    prompt: 用于图像生成模型的人物参考图提示词。
    """

    image_path = _project_dir() / "assets" / "characters" / f"{character_id}.png"
    saved_path = _generate_image(prompt, str(image_path))

    asset = {
        "asset_id": f"{character_id}_reference",
        "character_name": character_name,
        "prompt": prompt,
        "path": saved_path,
    }

    index = _load_index()
    index["character_assets"][character_id] = asset
    index_path = _save_index(index)

    return {"character_id": character_id, "index_path": index_path, **asset}


def _build_3d_face(reference_image_path: str, output_dir: str):
    """调用 FaceLift 构建 3D 人脸资产。"""

    face_dir = Path(output_dir)
    face_dir.mkdir(parents=True, exist_ok=True)
    return build_facelift_asset(reference_image_path, str(face_dir))


@mcp.tool()
def build_3d_face_asset(character_id: str, reference_image_path: str):
    """根据人物正脸参考图生成 3D 人脸资产。

    character_id: 人物 id，例如 char_001。
    reference_image_path: 人物参考图路径，通常来自 asset_index.character_assets[character_id].path。
    """

    face_dir = _project_dir() / "assets" / "faces_3d" / character_id
    face_result = _build_3d_face(reference_image_path, str(face_dir))

    face_3d_asset = {
        "asset_id": f"{character_id}_face_3d",
        "source_image_path": reference_image_path,
        "path": str(face_dir),
        **face_result,
    }

    index = _load_index()
    index["character_assets"][character_id]["face_3d"] = face_3d_asset
    index_path = _save_index(index)

    return {"character_id": character_id, "index_path": index_path, **face_3d_asset}


def _injection_memory_path():
    """返回跨项目共享的注入强度记忆库路径。

    这份记忆不依赖单个 shot，也不放在某个项目目录里；它按模型、去噪阶段、
    漂移档位累计经验，供后续所有 shot 复用。
    """

    default_path = Path(__file__).resolve().parents[1] / "outputs" / "global_injection_memory.json"
    return Path(os.getenv("MULTISHOT_INJECTION_MEMORY_PATH", str(default_path)))


def _load_injection_memory():
    """读取跨 shot / 跨项目的注入强度记忆库。"""

    memory_path = _injection_memory_path()
    if memory_path.exists():
        return json.loads(memory_path.read_text(encoding="utf-8"))
    return {}


def _save_injection_memory(memory: dict):
    memory_path = _injection_memory_path()
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(memory_path)


def _drift_bucket(drift_score: float):
    """把漂移程度分成三档，避免早期样本过稀。"""

    if drift_score < 0.2:
        return "low"
    if drift_score < 0.5:
        return "mid"
    return "high"


def _initial_memory_record():
    """初始化 TN(0.5, 0.2^2) 对应的可更新分布记录。"""

    return {
        "mu": 0.5,
        "sigma": 0.2,
        "samples": 0,
        "history": [],
        "selected_lambda_counts": {},
    }


def _sample_truncated_normal(mu: float, sigma: float, count: int, low: float = 0.0, high: float = 1.0):
    """从截断正态分布采样注入强度 lambda。"""

    values = []
    max_attempts = count * 50
    attempts = 0
    while len(values) < count and attempts < max_attempts:
        attempts += 1
        value = random.gauss(mu, sigma)
        if low <= value <= high:
            values.append(round(value, 4))
    while len(values) < count:
        values.append(round(min(high, max(low, random.gauss(mu, sigma))), 4))
    return values


def _parse_lambda_list(value: str):
    lambdas = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        lambdas.append(round(max(0.0, min(1.0, float(item))), 4))
    return lambdas


def _fixed_exploration_lambdas(candidate_count: int):
    """测试阶段使用固定探索分布，并强制包含高强度候选。"""

    mu = float(os.getenv("MULTISHOT_FIXED_EXPLORATION_MU", "0.5"))
    sigma = float(os.getenv("MULTISHOT_FIXED_EXPLORATION_SIGMA", "0.2"))
    high_points = _parse_lambda_list(os.getenv("MULTISHOT_FIXED_HIGH_LAMBDAS", "0.95,1.0"))
    high_points = high_points[:candidate_count]
    sampled_count = max(0, candidate_count - len(high_points))
    return _sample_truncated_normal(mu, sigma, sampled_count) + high_points


def _weighted_mean_std(samples: list[dict], old_mu: float, old_sigma: float):
    """按漂移改善作为权重，估计本轮候选强度的均值和标准差。"""

    weights = [max(0.0, item["improvement"]) for item in samples]
    weight_sum = sum(weights)
    if weight_sum <= 0:
        return old_mu, old_sigma

    weighted_mean = sum(item["lambda"] * weight for item, weight in zip(samples, weights)) / weight_sum
    weighted_var = sum(
        weight * ((item["lambda"] - weighted_mean) ** 2)
        for item, weight in zip(samples, weights)
    ) / weight_sum
    return weighted_mean, math.sqrt(max(0.0, weighted_var))


def _update_distribution(memory_record: dict, accepted_samples: list[dict], selected_lambda: float):
    """用本轮有改善的样本更新注入强度分布。"""

    alpha = float(os.getenv("MULTISHOT_MEMORY_UPDATE_ALPHA", "0.3"))
    sigma_min = float(os.getenv("MULTISHOT_MEMORY_SIGMA_MIN", "0.05"))
    sigma_max = float(os.getenv("MULTISHOT_MEMORY_SIGMA_MAX", "0.25"))
    old_mu = float(memory_record.get("mu", 0.5))
    old_sigma = float(memory_record.get("sigma", 0.2))

    if accepted_samples:
        batch_mu, batch_sigma = _weighted_mean_std(accepted_samples, old_mu, old_sigma)
        memory_record["mu"] = round((1 - alpha) * old_mu + alpha * batch_mu, 4)
        memory_record["sigma"] = round(min(sigma_max, max(sigma_min, (1 - alpha) * old_sigma + alpha * batch_sigma)), 4)
    else:
        memory_record["mu"] = round(old_mu, 4)
        memory_record["sigma"] = round(old_sigma, 4)

    memory_record["samples"] = int(memory_record.get("samples", 0)) + len(accepted_samples)
    selected_key = f"{selected_lambda:.4f}"
    selected_counts = memory_record.setdefault("selected_lambda_counts", {})
    selected_counts[selected_key] = selected_counts.get(selected_key, 0) + 1
    memory_record.setdefault("history", []).append({
        "selected_lambda": round(selected_lambda, 4),
        "accepted_samples": [
            {
                "lambda": round(item["lambda"], 4),
                "improvement": round(item["improvement"], 4),
                "avg_drift_score": round(item["avg_drift_score"], 4),
            }
            for item in accepted_samples
        ],
    })
    memory_record["history"] = memory_record["history"][-50:]
    return memory_record


def _select_natural_candidate_from_topk(topk_candidates: list[dict]):
    """主 agent / VLM 自然性选择接口。

    当前先选择漂移改善最大的候选；后续可以在这里接 Qwen-VL，让它从 topk 图里
    按自然程度从高到低挑选。
    """

    if not topk_candidates:
        return None
    return topk_candidates[0]


def _fake_predict_denoise_step(generation_state: dict, step: int):
    """伪造单步去噪模型输出。

    真实 diffusion 接入时，这里对应一次 UNet/DiT forward：
    输入当前 x_t latent、timestep、prompt/condition，输出 noise_pred 或 velocity。
    当前只记录结构化占位信息，方便后续替换成真实 tensor 和特征。
    """

    return {
        "step": step,
        "timestep": f"t_{step}",
        "prediction_type": "epsilon",
        "current_latent": f"x_t_placeholder_for_{generation_state['shot_id']}_step_{step}",
        "noise_prediction": f"epsilon_pred_placeholder_for_{generation_state['shot_id']}_step_{step}",
        "conditioning": {
            "prompt": generation_state["prompt"],
            "scene_asset_path": generation_state["scene_asset_path"],
            "character_ids": generation_state["character_ids"],
        },
    }


def _fake_estimate_x0_preview(denoise_state: dict, output_path: str):
    """根据当前 x_t 和模型噪声预测估计 x0，并解码为可评估图像。

    真实 diffusion 接入时，这里应该和 scheduler / prediction_type 强绑定：
    1. 用 x_t、noise_pred 或 velocity、timestep 和 scheduler 参数估计 pred_x0_latent。
    2. 用 VAE 把 pred_x0_latent decode 成 RGB 图像。
    3. 保存 x0 preview，后续交给 VLM/InsightFace 做人脸清晰度、角度和身份判断。
    """

    preview_path = Path(output_path)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.touch()

    x0_result = {
        "step": denoise_state["step"],
        "pred_x0_latent": f"x0_latent_estimated_from_{denoise_state['current_latent']}",
        "x0_preview_path": str(preview_path),
        "x0_image_features": {
            "image_embedding": f"pseudo_image_embedding_step_{denoise_state['step']}",
            "face_pose_summary": "unknown_until_insightface_runs",
            "face_quality": "unknown_until_vlm_runs",
        },
    }

    preview_path.with_suffix(".meta.json").write_text(
        json.dumps({
            "kind": "x0_preview",
            "denoise_state": denoise_state,
            **x0_result,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return x0_result


def _fake_decode_final_image(denoise_state: dict, output_path: str):
    """从最终去噪 latent 解码得到最终首帧图像。

    中间过程用 _fake_estimate_x0_preview 生成观察图；最终输出应该从最终
    denoise_state 通过 VAE decode 得到。当前是伪实现，只创建图片占位和
    final_meta 文件。
    """

    image_path = Path(output_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.touch()
    image_path.with_suffix(".final_meta.json").write_text(
        json.dumps({
            "kind": "final_decoded_image",
            "denoise_state": denoise_state,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(image_path)


def _use_real_diffusion(generation_model: str):
    """判断当前 shot 是否使用真实开源 diffusion 后端。"""

    return generation_model != "pseudo_diffusion_v1"


def _prepare_diffusion_runtime(generation_state: dict):
    """为当前 shot 准备真实 diffusion 运行时。

    runtime 和 backend 都是 Python 对象，不会写进日志；日志只保留公开字段。
    """

    if not _use_real_diffusion(generation_state["generation_model"]):
        return

    backend = get_diffusion_backend(generation_state["generation_model"])
    generation_state["_diffusion_backend"] = backend
    generation_state["_diffusion_runtime"] = backend.prepare_generation(
        shot_id=generation_state["shot_id"],
        prompt=generation_state["prompt"],
    )


def _denoise_window(
    generation_state: dict,
    from_step: int,
    to_step: int,
    previous_denoise_state: dict | None = None,
    injection_plan: dict | None = None,
):
    """统一去噪窗口接口：真实 diffusion 优先，pseudo 作为 fallback。"""

    backend = generation_state.get("_diffusion_backend")
    if backend is not None:
        return backend.denoise_window(
            runtime=generation_state["_diffusion_runtime"],
            from_step=from_step,
            to_step=to_step,
            previous_denoise_state=previous_denoise_state,
            injection_plan=injection_plan,
            conditioning={
                "prompt": generation_state["prompt"],
                "scene_asset_path": generation_state["scene_asset_path"],
                "character_ids": generation_state["character_ids"],
            },
        )

    return _fake_denoise_window(
        generation_state,
        from_step,
        to_step,
        previous_denoise_state=previous_denoise_state,
        injection_plan=injection_plan,
    )


def _estimate_x0_preview(generation_state: dict, denoise_state: dict, output_path: str):
    """统一 x0 预览接口。"""

    backend = generation_state.get("_diffusion_backend")
    if backend is not None:
        return backend.estimate_x0_preview(denoise_state, output_path)
    return _fake_estimate_x0_preview(denoise_state, output_path)


def _decode_final_image(generation_state: dict, denoise_state: dict, output_path: str):
    """统一最终图 decode 接口。"""

    backend = generation_state.get("_diffusion_backend")
    if backend is not None:
        return backend.decode_final_image(denoise_state, output_path)
    return _fake_decode_final_image(denoise_state, output_path)


def _assess_face_clarity(x0_result: dict):
    """用 InsightFace 判断当前 x0 预览里的人脸是否已经可检测。"""

    step = x0_result["step"]
    min_step = int(os.getenv("MULTISHOT_FACE_CLEAR_MIN_STEP", "30"))
    if step < min_step:
        return {
            "is_clear": False,
            "reason": f"step {step} is earlier than MULTISHOT_FACE_CLEAR_MIN_STEP={min_step}",
            "x0_preview_path": x0_result["x0_preview_path"],
            "x0_image_features": x0_result["x0_image_features"],
            "face_count": 0,
        }

    try:
        faces = get_face_backend().analyze(x0_result["x0_preview_path"])
    except Exception as exc:
        return {
            "is_clear": False,
            "reason": f"InsightFace clarity check failed: {exc}",
            "x0_preview_path": x0_result["x0_preview_path"],
            "x0_image_features": x0_result["x0_image_features"],
            "face_count": 0,
        }

    min_confidence = float(os.getenv("MULTISHOT_FACE_CLEAR_MIN_CONFIDENCE", "0.45"))
    clear_faces = [face for face in faces if face.get("face_confidence", 0.0) >= min_confidence]
    return {
        "is_clear": bool(clear_faces),
        "reason": "InsightFace detected a clear face" if clear_faces else "InsightFace did not detect a reliable face yet",
        "x0_preview_path": x0_result["x0_preview_path"],
        "x0_image_features": x0_result["x0_image_features"],
        "face_count": len(faces),
        "clear_face_count": len(clear_faces),
        "faces": [
            {
                "face_id": face["face_id"],
                "face_bbox": face["face_bbox"],
                "face_confidence": face["face_confidence"],
                "pose": face["pose"],
            }
            for face in faces
        ],
    }


def _fake_embedding_from_key(key: str, dims: int = 8):
    """检测失败时生成稳定 fallback embedding，保证流程不中断。"""

    values = []
    salt_index = 0
    while len(values) < dims:
        digest = hashlib.sha256(f"{key}:{salt_index}".encode("utf-8")).digest()
        values.extend((byte / 127.5) - 1.0 for byte in digest)
        salt_index += 1
    values = values[:dims]
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [round(value / norm, 6) for value in values]


def _face_pose_value(face_pose: dict, key: str, default: float = 0.0):
    try:
        return float((face_pose or {}).get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _pose_to_facelift_camera(face_pose: dict, image_size: int):
    """把 InsightFace yaw/pitch/roll 转成 FaceLift Gaussian renderer 的 OpenCV camera。"""

    import numpy as np

    yaw = max(-55.0, min(55.0, _face_pose_value(face_pose, "yaw")))
    pitch = max(-35.0, min(35.0, _face_pose_value(face_pose, "pitch")))
    roll = max(-45.0, min(45.0, _face_pose_value(face_pose, "roll")))
    radius = float(os.getenv("MULTISHOT_FACELIFT_RENDER_RADIUS", "2.7"))
    hfov = float(os.getenv("MULTISHOT_FACELIFT_RENDER_HFOV", "50"))
    yaw_sign = float(os.getenv("MULTISHOT_FACELIFT_YAW_SIGN", "-1"))
    roll_sign = float(os.getenv("MULTISHOT_FACELIFT_ROLL_SIGN", "1"))

    # FaceLift turntable 的 frontal camera 对应 azimuth=270, elevation=0。
    # FaceLift renderer 的水平相机方向与 InsightFace yaw 符号相反；默认用 -yaw 对齐画面左右朝向。
    azim = np.deg2rad(270.0 + yaw_sign * yaw)
    elev = np.deg2rad(-pitch)
    up_vector = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    z = radius * np.sin(elev)
    base = radius * np.cos(elev)
    x = base * np.cos(azim)
    y = base * np.sin(azim)
    cam_pos = np.array([x, y, z], dtype=np.float32)
    forward = -cam_pos / np.linalg.norm(cam_pos)
    right = np.cross(forward, up_vector)
    right = right / (np.linalg.norm(right) or 1.0)
    up = np.cross(right, forward)

    roll_rad = np.deg2rad(roll_sign * roll)
    cos_r = np.cos(roll_rad)
    sin_r = np.sin(roll_rad)
    rolled_right = right * cos_r + up * sin_r
    rolled_up = -right * sin_r + up * cos_r
    rotation = np.stack((rolled_right, -rolled_up, forward), axis=1)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :4] = np.concatenate((rotation, cam_pos[:, None]), axis=1)

    focal = image_size / (2 * np.tan(np.deg2rad(hfov) / 2.0))
    fxfycxcy = np.array([focal, focal, image_size / 2.0, image_size / 2.0], dtype=np.float32)
    return c2w, fxfycxcy, {
        "yaw": round(yaw, 4),
        "pitch": round(pitch, 4),
        "roll": round(roll, 4),
        "yaw_sign": yaw_sign,
        "roll_sign": roll_sign,
        "azimuth": round(float(np.rad2deg(azim)), 4),
        "elevation": round(float(np.rad2deg(elev)), 4),
    }


def _render_3d_face_reference(face_3d: dict, face_pose: dict, target_face_bbox: list[float] | None = None):
    """从 FaceLift gaussians.ply 按当前帧 pose 连续渲染参考脸。"""

    model_path = face_3d.get("model_path")
    asset_dir = face_3d.get("path")
    if not model_path or not Path(model_path).exists() or not asset_dir:
        return None

    try:
        import sys
        import numpy as np
        import torch
        from PIL import Image
    except Exception:
        return None

    facelift_root = Path(os.getenv("MULTISHOT_FACELIFT_ROOT", Path(__file__).resolve().parents[1] / "third_party" / "FaceLift"))
    if str(facelift_root) not in sys.path:
        sys.path.insert(0, str(facelift_root))

    try:
        from gslrm.model.gaussians_renderer import GaussianModel, render_opencv_cam
    except Exception as exc:
        print(f"FaceLift renderer import failed; falling back to multiview image: {exc}")
        return None

    image_size = int(os.getenv("MULTISHOT_FACELIFT_RENDER_SIZE", "1024"))
    yaw = _face_pose_value(face_pose, "yaw")
    pitch = _face_pose_value(face_pose, "pitch")
    roll = _face_pose_value(face_pose, "roll")
    target_tag = "none"
    if target_face_bbox:
        target = [float(v) for v in target_face_bbox]
        target_w = max(1.0, target[2] - target[0])
        target_h = max(1.0, target[3] - target[1])
        target_cx = (target[0] + target[2]) / 2.0
        target_cy = (target[1] + target[3]) / 2.0
        target_tag = f"{int(round(target_w))}x{int(round(target_h))}_{int(round(target_cx))}_{int(round(target_cy))}"
    yaw_sign_tag = str(os.getenv("MULTISHOT_FACELIFT_YAW_SIGN", "-1"))
    roll_sign_tag = str(os.getenv("MULTISHOT_FACELIFT_ROLL_SIGN", "1"))
    tag = f"pose_ys_{yaw_sign_tag}_rs_{roll_sign_tag}_yaw_{yaw:+.1f}_pitch_{pitch:+.1f}_roll_{roll:+.1f}_{target_tag}".replace("+", "p").replace("-", "m").replace(".", "p")
    render_dir = Path(asset_dir) / "pose_renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    render_path = render_dir / f"{tag}.png"
    meta_path = render_dir / f"{tag}.meta.json"
    if render_path.exists() and meta_path.exists():
        return str(render_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        pc = GaussianModel(sh_degree=3)
        pc.load_ply(model_path)
        pc = pc.to(device)
        c2w_np, fxfycxcy_np, camera_meta = _pose_to_facelift_camera(face_pose, image_size)
        c2w = torch.from_numpy(c2w_np).float().to(device)
        fxfycxcy = torch.from_numpy(fxfycxcy_np).float().to(device)
        with torch.no_grad():
            rendered = render_opencv_cam(pc, image_size, image_size, c2w, fxfycxcy)["render"]
        image = rendered.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
        image = (image * 255.0).round().astype(np.uint8)
        Image.fromarray(image).save(render_path)
        meta = {
            "source_model_path": str(model_path),
            "render_image": str(render_path),
            "input_face_pose": face_pose or {},
            "camera": camera_meta,
            "target_face_bbox": [round(float(v), 2) for v in target_face_bbox] if target_face_bbox else None,
            "render_size": image_size,
            "renderer": "FaceLift GaussianModel/render_opencv_cam",
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(render_path)
    except Exception as exc:
        print(f"FaceLift pose render failed; falling back to multiview image: {exc}")
        return None
    finally:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _select_3d_face_view(face_3d: dict, face_pose: dict, target_face_bbox: list[float] | None = None):
    """优先按当前帧 pose 从 FaceLift 3D 高斯脸渲染参考图，失败时回退到离散多视角。"""

    if os.getenv("MULTISHOT_FACELIFT_POSE_RENDER", "1") != "0":
        rendered = _render_3d_face_reference(face_3d, face_pose, target_face_bbox)
        if rendered:
            return rendered

    multi_view_images = face_3d.get("multi_view_images", [])
    if not multi_view_images:
        return None

    yaw = _face_pose_value(face_pose, "yaw")
    threshold = float(os.getenv("MULTISHOT_FACE_VIEW_YAW_THRESHOLD", "20"))
    # FaceLift 当前导出的顺序是 [front, right, left]。
    if yaw < -threshold and len(multi_view_images) > 2:
        return multi_view_images[2]
    if yaw > threshold and len(multi_view_images) > 1:
        return multi_view_images[1]
    return multi_view_images[0]


def _fake_denoise_window(
    generation_state: dict,
    from_step: int,
    to_step: int,
    previous_denoise_state: dict | None = None,
    injection_plan: dict | None = None,
):
    """统一的伪去噪窗口函数。

    所有去噪都走这个函数：
    - 不注入时 injection_plan 的 lambda 为 0。
    - rollout 候选时传入不同 lambda 和参考脸。
    - 当前先固定每次推进 5 步，真实模型接入时这里会包 diffusion scheduler loop。

    参数：
    generation_state: shot 级生成上下文，比如 prompt、scene asset、character ids。
    from_step / to_step: 本次去噪窗口的起止步数。
    previous_denoise_state: 上一轮保留下来的最优噪声/latent 状态。
    injection_plan: 参考脸注入计划，包含每张脸的 mask、reference、lambda。

    返回：
    denoise_state: 推进到 to_step 后的噪声状态，后续可以继续从这里去噪。
    """

    injection_plan = injection_plan or {"lambda": 0.0, "targets": []}
    source_latent = (
        previous_denoise_state.get("current_latent")
        if previous_denoise_state
        else f"initial_latent_for_{generation_state['shot_id']}"
    )

    return {
        "shot_id": generation_state["shot_id"],
        "from_step": from_step,
        "step": to_step,
        "window_size": to_step - from_step,
        "current_latent": f"latent_{generation_state['shot_id']}_{from_step}_to_{to_step}_lambda_{injection_plan.get('lambda', 0.0)}",
        "source_latent": source_latent,
        "noise_prediction": f"noise_pred_{generation_state['shot_id']}_{from_step}_to_{to_step}",
        "injection_plan": injection_plan,
        "conditioning": {
            "prompt": generation_state["prompt"],
            "scene_asset_path": generation_state["scene_asset_path"],
            "character_ids": generation_state["character_ids"],
        },
    }


def _write_face_mask(x0_preview_path: str, face: dict):
    """根据 InsightFace bbox 写一个可用于 latent injection 的 soft mask。"""

    from PIL import Image, ImageDraw, ImageFilter

    image = Image.open(x0_preview_path)
    width, height = image.size
    x1, y1, x2, y2 = [int(value) for value in face["face_bbox"]]
    pad_x = int((x2 - x1) * float(os.getenv("MULTISHOT_FACE_MASK_PAD_X", "0.20")))
    pad_y = int((y2 - y1) * float(os.getenv("MULTISHOT_FACE_MASK_PAD_Y", "0.30")))
    bbox = [
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    ]
    mask_path = str(Path(x0_preview_path).with_suffix(f".{face['face_id']}.mask.png"))
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse(bbox, fill=255)
    mask.filter(ImageFilter.GaussianBlur(radius=max(3, width // 120))).save(mask_path)
    return mask_path


def _write_reference_face_mask(reference_image_path: str):
    """为 3D 参考脸采样图写 reference mask，限制 reference K/V 只来自脸部区域。"""

    from PIL import Image, ImageDraw, ImageFilter

    image = Image.open(reference_image_path)
    width, height = image.size
    try:
        faces = get_face_backend().analyze(reference_image_path)
    except Exception:
        faces = []

    if faces:
        bbox = [int(value) for value in faces[0]["face_bbox"]]
        source = "insightface"
    else:
        bbox = [
            int(width * 0.30),
            int(height * 0.15),
            int(width * 0.70),
            int(height * 0.62),
        ]
        source = "fallback_center_bbox"

    x1, y1, x2, y2 = bbox
    pad_x = int((x2 - x1) * 0.18)
    pad_y = int((y2 - y1) * 0.25)
    expanded = [
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    ]
    mask_path = str(Path(reference_image_path).with_suffix(".reference_face.mask.png"))
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse(expanded, fill=255)
    mask.filter(ImageFilter.GaussianBlur(radius=max(3, width // 120))).save(mask_path)
    return {
        "reference_mask_path": mask_path,
        "reference_face_bbox": expanded,
        "reference_mask_source": source,
    }


def _detect_reference_face_bbox(source_path: Path):
    """检测参考图中的人脸 bbox；失败时返回中心 fallback。"""

    from PIL import Image

    image = Image.open(source_path).convert("RGB")
    width, height = image.size
    try:
        faces = get_face_backend().analyze(str(source_path))
    except Exception:
        faces = []

    if faces:
        face_bbox = [float(value) for value in faces[0]["face_bbox"]]
        detection_source = "insightface"
    else:
        face_bbox = [
            width * 0.30,
            height * 0.14,
            width * 0.70,
            height * 0.64,
        ]
        detection_source = "fallback_center_bbox"
    return image, face_bbox, detection_source


def _save_reference_layout_mask(mask_path: Path, output_size: int, face_bbox: list[int]):
    """保存 reference layout 上的人脸 mask。"""

    from PIL import Image, ImageDraw, ImageFilter

    fx1, fy1, fx2, fy2 = face_bbox
    pad_face_x = int((fx2 - fx1) * float(os.getenv("MULTISHOT_FACE_MASK_PAD_X", "0.20")))
    pad_face_y = int((fy2 - fy1) * float(os.getenv("MULTISHOT_FACE_MASK_PAD_Y", "0.30")))
    mask_bbox = [
        max(0, fx1 - pad_face_x),
        max(0, fy1 - pad_face_y),
        min(output_size, fx2 + pad_face_x),
        min(output_size, fy2 + pad_face_y),
    ]
    mask = Image.new("L", (output_size, output_size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse(mask_bbox, fill=255)
    mask.filter(ImageFilter.GaussianBlur(radius=max(3, output_size // 120))).save(mask_path)
    return mask_bbox


def _prepare_reference_face_crop(reference_image_path: str, target_face_bbox: list[float] | None = None):
    """准备 reference attention 使用的人脸参考图。

    MULTISHOT_REFERENCE_LAYOUT_MODE=crop：
      裁出参考图中人脸并放大到 1024，reference token 更干净。
      现在默认不使用这个模式，因为它会让 reference face token 尺度明显大于小脸 target。

    MULTISHOT_REFERENCE_LAYOUT_MODE=match_target_scale：
      根据当前帧 target face bbox，把参考脸缩放到 target_face_bbox * ratio 的大小；默认 ratio=1.0，
      并放到与当前脸相同的中心位置，方便 M_ref 与 M_cur 有接近的 token 尺度。
    """

    from PIL import Image, ImageOps

    source_path = Path(reference_image_path)
    mode = os.getenv("MULTISHOT_REFERENCE_LAYOUT_MODE", "match_target_scale")
    ratio = float(os.getenv("MULTISHOT_REFERENCE_FACE_SCALE_RATIO", "1.0"))
    output_size = int(os.getenv("MULTISHOT_REFERENCE_CROP_SIZE", "1024"))

    if mode == "match_target_scale" and target_face_bbox:
        target = [float(v) for v in target_face_bbox]
        target_w = max(1.0, target[2] - target[0])
        target_h = max(1.0, target[3] - target[1])
        target_cx = (target[0] + target[2]) / 2.0
        target_cy = (target[1] + target[3]) / 2.0
        tag = f"reference_match_{ratio:.2f}_{int(round(target_w))}x{int(round(target_h))}_{int(round(target_cx))}_{int(round(target_cy))}".replace(".", "p")
        layout_path = source_path.with_suffix(f".{tag}.png")
        mask_path = source_path.with_suffix(f".{tag}.mask.png")
        meta_path = source_path.with_suffix(f".{tag}.meta.json")

        if layout_path.exists() and mask_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return {
                "reference_source_image": str(source_path),
                "reference_image": str(layout_path),
                "reference_mask_path": str(mask_path),
                "reference_face_bbox": meta["face_bbox_on_reference_layout"],
                "reference_crop_bbox_on_source": meta["source_face_crop_bbox"],
                "reference_crop_meta_path": str(meta_path),
                "reference_mask_source": meta["face_detection_source"],
                "reference_layout_mode": mode,
                "reference_face_scale_ratio": ratio,
                "reference_to_target_face_scale": meta.get("reference_to_target_face_scale"),
                "target_face_size": meta.get("target_face_size"),
                "reference_face_size_on_layout": meta.get("reference_face_size_on_layout"),
            }

        image, face_bbox, detection_source = _detect_reference_face_bbox(source_path)
        width, height = image.size
        sx1, sy1, sx2, sy2 = face_bbox
        source_face_w = max(1.0, sx2 - sx1)
        source_face_h = max(1.0, sy2 - sy1)

        # 先裁出包含完整脸的 source patch，再缩放到目标人脸尺度附近。
        source_crop_w = source_face_w * 1.45
        source_crop_h = source_face_h * 1.45
        scx = (sx1 + sx2) / 2.0
        scy = (sy1 + sy2) / 2.0
        crop_bbox = [
            max(0, int(round(scx - source_crop_w / 2.0))),
            max(0, int(round(scy - source_crop_h / 2.0))),
            min(width, int(round(scx + source_crop_w / 2.0))),
            min(height, int(round(scy + source_crop_h / 2.0))),
        ]
        source_crop = image.crop(crop_bbox)

        desired_face_w = max(4.0, min(output_size * 0.85, target_w * ratio))
        desired_face_h = max(4.0, min(output_size * 0.85, target_h * ratio))
        scale = min(desired_face_w / source_face_w, desired_face_h / source_face_h)
        resized_w = max(1, int(round(source_crop.width * scale)))
        resized_h = max(1, int(round(source_crop.height * scale)))
        source_resized = source_crop.resize((resized_w, resized_h), Image.Resampling.LANCZOS)

        face_bbox_in_crop = [
            (sx1 - crop_bbox[0]) * scale,
            (sy1 - crop_bbox[1]) * scale,
            (sx2 - crop_bbox[0]) * scale,
            (sy2 - crop_bbox[1]) * scale,
        ]
        face_cx_in_resized = (face_bbox_in_crop[0] + face_bbox_in_crop[2]) / 2.0
        face_cy_in_resized = (face_bbox_in_crop[1] + face_bbox_in_crop[3]) / 2.0

        # 放在 target face 的中心位置，让 reference mask token 数与 target mask 接近。
        paste_x = int(round(target_cx - face_cx_in_resized))
        paste_y = int(round(target_cy - face_cy_in_resized))
        paste_x = max(-resized_w + 1, min(output_size - 1, paste_x))
        paste_y = max(-resized_h + 1, min(output_size - 1, paste_y))

        canvas = Image.new("RGB", (output_size, output_size), (127, 127, 127))
        canvas.paste(source_resized, (paste_x, paste_y))
        canvas.save(layout_path)

        face_bbox_on_layout = [
            int(round(face_bbox_in_crop[0] + paste_x)),
            int(round(face_bbox_in_crop[1] + paste_y)),
            int(round(face_bbox_in_crop[2] + paste_x)),
            int(round(face_bbox_in_crop[3] + paste_y)),
        ]
        face_bbox_on_layout = [
            max(0, min(output_size, face_bbox_on_layout[0])),
            max(0, min(output_size, face_bbox_on_layout[1])),
            max(0, min(output_size, face_bbox_on_layout[2])),
            max(0, min(output_size, face_bbox_on_layout[3])),
        ]
        mask_bbox = _save_reference_layout_mask(mask_path, output_size, face_bbox_on_layout)

        meta = {
            "source_image": str(source_path),
            "reference_layout_image": str(layout_path),
            "reference_layout_mask": str(mask_path),
            "layout_mode": mode,
            "reference_face_scale_ratio": ratio,
            "source_size": [width, height],
            "output_size": [output_size, output_size],
            "target_face_bbox": [round(v, 2) for v in target],
            "target_face_size": [round(target_w, 2), round(target_h, 2)],
            "detected_face_bbox_on_source": [round(v, 2) for v in face_bbox],
            "source_face_crop_bbox": crop_bbox,
            "resized_source_crop_size": [resized_w, resized_h],
            "paste_xy": [paste_x, paste_y],
            "face_bbox_on_reference_layout": face_bbox_on_layout,
            "mask_bbox_on_reference_layout": mask_bbox,
            "reference_face_size_on_layout": [
                round(face_bbox_on_layout[2] - face_bbox_on_layout[0], 2),
                round(face_bbox_on_layout[3] - face_bbox_on_layout[1], 2),
            ],
            "reference_to_target_face_scale": [
                round((face_bbox_on_layout[2] - face_bbox_on_layout[0]) / target_w, 4),
                round((face_bbox_on_layout[3] - face_bbox_on_layout[1]) / target_h, 4),
            ],
            "face_detection_source": detection_source,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "reference_source_image": str(source_path),
            "reference_image": str(layout_path),
            "reference_mask_path": str(mask_path),
            "reference_face_bbox": face_bbox_on_layout,
            "reference_crop_bbox_on_source": crop_bbox,
            "reference_crop_meta_path": str(meta_path),
            "reference_mask_source": detection_source,
            "reference_layout_mode": mode,
            "reference_face_scale_ratio": ratio,
            "reference_to_target_face_scale": [
                round((face_bbox_on_layout[2] - face_bbox_on_layout[0]) / target_w, 4),
                round((face_bbox_on_layout[3] - face_bbox_on_layout[1]) / target_h, 4),
            ],
            "target_face_size": [round(target_w, 2), round(target_h, 2)],
            "reference_face_size_on_layout": [
                round(face_bbox_on_layout[2] - face_bbox_on_layout[0], 2),
                round(face_bbox_on_layout[3] - face_bbox_on_layout[1], 2),
            ],
        }

    # 默认 crop 模式：裁出参考脸并放大，让 reference latent 主要包含脸部信息。
    crop_path = source_path.with_suffix(".reference_crop.png")
    mask_path = source_path.with_suffix(".reference_crop.mask.png")
    meta_path = source_path.with_suffix(".reference_crop.meta.json")

    if crop_path.exists() and mask_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return {
            "reference_source_image": str(source_path),
            "reference_image": str(crop_path),
            "reference_mask_path": str(mask_path),
            "reference_face_bbox": meta["face_bbox_on_crop"],
            "reference_crop_bbox_on_source": meta["crop_bbox_on_source"],
            "reference_crop_meta_path": str(meta_path),
            "reference_mask_source": meta["face_detection_source"],
            "reference_layout_mode": "crop",
            "reference_face_scale_ratio": None,
        }

    image, face_bbox, detection_source = _detect_reference_face_bbox(source_path)
    width, height = image.size
    x1, y1, x2, y2 = face_bbox
    face_w = max(1.0, x2 - x1)
    face_h = max(1.0, y2 - y1)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    crop_side = max(face_w * 1.85, face_h * 1.65)
    crop_side = min(crop_side, float(max(width, height)))
    crop_x1 = center_x - crop_side / 2.0
    crop_y1 = center_y - crop_side * 0.46
    crop_x2 = crop_x1 + crop_side
    crop_y2 = crop_y1 + crop_side

    if crop_x1 < 0:
        crop_x2 -= crop_x1
        crop_x1 = 0.0
    if crop_y1 < 0:
        crop_y2 -= crop_y1
        crop_y1 = 0.0
    if crop_x2 > width:
        crop_x1 -= crop_x2 - width
        crop_x2 = float(width)
    if crop_y2 > height:
        crop_y1 -= crop_y2 - height
        crop_y2 = float(height)
    crop_x1 = max(0.0, crop_x1)
    crop_y1 = max(0.0, crop_y1)
    crop_x2 = min(float(width), crop_x2)
    crop_y2 = min(float(height), crop_y2)

    crop_bbox = [int(round(v)) for v in [crop_x1, crop_y1, crop_x2, crop_y2]]
    crop = image.crop(crop_bbox)
    crop_resized = ImageOps.pad(crop, (output_size, output_size), method=Image.Resampling.LANCZOS, color=(127, 127, 127), centering=(0.5, 0.5))
    crop_resized.save(crop_path)

    crop_w = max(1, crop_bbox[2] - crop_bbox[0])
    crop_h = max(1, crop_bbox[3] - crop_bbox[1])
    scale = min(output_size / crop_w, output_size / crop_h)
    resized_w = int(round(crop_w * scale))
    resized_h = int(round(crop_h * scale))
    pad_x = (output_size - resized_w) / 2.0
    pad_y = (output_size - resized_h) / 2.0

    face_bbox_on_crop = [
        int(round((x1 - crop_bbox[0]) * scale + pad_x)),
        int(round((y1 - crop_bbox[1]) * scale + pad_y)),
        int(round((x2 - crop_bbox[0]) * scale + pad_x)),
        int(round((y2 - crop_bbox[1]) * scale + pad_y)),
    ]
    face_bbox_on_crop = [
        max(0, min(output_size, face_bbox_on_crop[0])),
        max(0, min(output_size, face_bbox_on_crop[1])),
        max(0, min(output_size, face_bbox_on_crop[2])),
        max(0, min(output_size, face_bbox_on_crop[3])),
    ]
    mask_bbox = _save_reference_layout_mask(mask_path, output_size, face_bbox_on_crop)

    meta = {
        "source_image": str(source_path),
        "reference_crop_image": str(crop_path),
        "reference_crop_mask": str(mask_path),
        "layout_mode": "crop",
        "source_size": [width, height],
        "output_size": [output_size, output_size],
        "detected_face_bbox_on_source": [round(v, 2) for v in face_bbox],
        "crop_bbox_on_source": crop_bbox,
        "face_bbox_on_crop": face_bbox_on_crop,
        "mask_bbox_on_crop": mask_bbox,
        "face_detection_source": detection_source,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "reference_source_image": str(source_path),
        "reference_image": str(crop_path),
        "reference_mask_path": str(mask_path),
        "reference_face_bbox": face_bbox_on_crop,
        "reference_crop_bbox_on_source": crop_bbox,
        "reference_crop_meta_path": str(meta_path),
        "reference_mask_source": detection_source,
        "reference_layout_mode": "crop",
        "reference_face_scale_ratio": None,
    }


def _extract_face_observations(x0_preview_path: str, expected_face_count: int):
    """用 InsightFace 提取 x0 预览图中的人脸 bbox / pose / embedding。"""

    try:
        faces = get_face_backend().analyze(x0_preview_path)
    except Exception as exc:
        faces = []
        insightface_error = str(exc)
    else:
        insightface_error = None

    if not faces:
        # fallback 只在检测失败时保留流程不中断，同时日志会标出来源。
        from PIL import Image
        try:
            width, height = Image.open(x0_preview_path).size
        except Exception:
            width, height = 512, 512
        face_count = max(1, expected_face_count)
        faces = []
        for index in range(face_count):
            left = int(width * 0.28) + index * int(width * 0.18)
            faces.append({
                "face_id": f"face_{index}",
                "face_index": index,
                "face_embedding": _fake_embedding_from_key(f"fallback-face:{x0_preview_path}:{index}", dims=512),
                "face_bbox": [left, int(height * 0.18), min(width, left + width // 4), min(height, int(height * 0.18) + height // 3)],
                "face_confidence": 0.0,
                "pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "view": "unknown"},
                "feature_source": "fallback_hash_embedding",
                "insightface_error": insightface_error,
            })
    else:
        for face in faces:
            face["feature_source"] = "insightface"

    observations = []
    for face in faces:
        face = dict(face)
        face["face_mask_path"] = _write_face_mask(x0_preview_path, face)
        observations.append(face)
    return observations


def _character_reference_features(character_assets: dict):
    """为每个角色参考图提取 InsightFace 身份 embedding。"""

    features = {}
    for character_id, character_asset in character_assets.items():
        reference_path = character_asset.get("path", character_id)
        try:
            embedding = get_face_backend().first_face_embedding(reference_path)
        except Exception as exc:
            embedding = None
            error = str(exc)
        else:
            error = None
        if embedding is None:
            embedding = _fake_embedding_from_key(f"fallback-character:{reference_path}", dims=512)
            source = "fallback_hash_embedding"
        else:
            source = "insightface"
        features[character_id] = {
            "reference_path": reference_path,
            "character_embedding": embedding,
            "feature_source": source,
            "insightface_error": error,
        }
    return features


def _assign_faces_to_characters_by_similarity(
    face_observations: list[dict],
    character_reference_features: dict,
    character_ids: list[str],
):
    """用 InsightFace cosine similarity 进行 face -> character 匹配。"""

    assignments = []
    candidate_ids = [cid for cid in character_ids if cid in character_reference_features]
    for face in face_observations:
        best_character_id = candidate_ids[0] if candidate_ids else "unknown"
        best_similarity = -1.0
        for character_id in candidate_ids:
            similarity = cosine_similarity(
                face["face_embedding"],
                character_reference_features[character_id]["character_embedding"],
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_character_id = character_id
        assignments.append({
            "face_id": face["face_id"],
            "character_id": best_character_id,
            "similarity": round(best_similarity, 4),
            "reason": "matched by InsightFace cosine similarity",
        })
    return assignments


def _compute_face_drift(face_embedding: list[float], reference_embedding: list[float]):
    similarity = cosine_similarity(face_embedding, reference_embedding)
    return {
        "similarity": similarity,
        "drift_score": round(max(0.0, 1.0 - similarity), 4),
    }


def _initialize_face_state(
    x0_preview_path: str,
    character_assets: dict,
    character_ids: list[str],
    first_frame_prompt: str,
):
    """人脸状态初始化：InsightFace 检测、角色匹配、3D 视角检索和漂移计算。"""

    face_observations = _extract_face_observations(
        x0_preview_path,
        len(character_ids),
    )
    character_reference_features = _character_reference_features(character_assets)
    assignments = _assign_faces_to_characters_by_similarity(
        face_observations,
        character_reference_features,
        character_ids,
    )
    assignment_by_face = {item["face_id"]: item for item in assignments}

    faces = []
    for face in face_observations:
        assignment = assignment_by_face[face["face_id"]]
        character_id = assignment["character_id"]
        character_asset = character_assets[character_id]
        reference_features = character_reference_features[character_id]
        reference_view_image = _select_3d_face_view(character_asset.get("face_3d", {}), face["pose"], face.get("face_bbox")) or character_asset.get("path")
        reference_crop = _prepare_reference_face_crop(reference_view_image, face["face_bbox"]) if reference_view_image else {}
        retrieved_face = {
            "face_3d_asset": character_asset.get("face_3d", {}),
            "reference_view_image": reference_view_image,
            "reference_embedding": reference_features["character_embedding"],
            "reference_feature_source": reference_features["feature_source"],
            **reference_crop,
        }
        drift = _compute_face_drift(face["face_embedding"], retrieved_face["reference_embedding"])
        faces.append({
            **face,
            "matched_character_id": character_id,
            "assignment_reason": assignment["reason"],
            "assignment_similarity": assignment["similarity"],
            "retrieved_3d_face": retrieved_face,
            **drift,
        })

    return {
        "source_frame_path": x0_preview_path,
        "character_reference_features": character_reference_features,
        "first_frame_prompt": first_frame_prompt,
        "faces": faces,
    }


def _evaluate_rollout_candidate(
    generation_state: dict,
    current_step: int,
    next_step: int,
    denoise_state: dict,
    face_state: dict,
    injection_lambda: float,
    candidate_label: str,
):
    """执行一个短程 rollout 候选，并计算平均漂移。"""

    injection_plan = {
        "lambda": injection_lambda,
        "candidate_id": candidate_label,
        "targets": [
            {
                "face_id": face["face_id"],
                "mask_path": face["face_mask_path"],
                "face_bbox": face["face_bbox"],
                "matched_character_id": face["matched_character_id"],
                "reference_image": face["retrieved_3d_face"].get("reference_image"),
                "reference_source_image": face["retrieved_3d_face"].get("reference_source_image"),
                "reference_view_image": face["retrieved_3d_face"].get("reference_view_image"),
                "reference_mask_path": face["retrieved_3d_face"].get("reference_mask_path"),
                "reference_face_bbox": face["retrieved_3d_face"].get("reference_face_bbox"),
                "reference_crop_meta_path": face["retrieved_3d_face"].get("reference_crop_meta_path"),
                "reference_layout_mode": face["retrieved_3d_face"].get("reference_layout_mode"),
                "reference_face_scale_ratio": face["retrieved_3d_face"].get("reference_face_scale_ratio"),
            }
            for face in face_state["faces"]
        ],
    }
    candidate_state = _denoise_window(
        generation_state,
        current_step,
        next_step,
        previous_denoise_state=denoise_state,
        injection_plan=injection_plan,
    )
    candidate_x0 = _estimate_x0_preview(
        generation_state,
        candidate_state,
        str(_project_dir() / "frames" / f"{generation_state['shot_id']}.step_{next_step}.{candidate_label}.x0.png"),
    )

    updated_faces = []
    total_drift = 0.0
    candidate_faces = _extract_face_observations(
        candidate_x0["x0_preview_path"],
        len(face_state["faces"]),
    )
    for face in face_state["faces"]:
        matching_faces = [item for item in candidate_faces if item["face_index"] == face["face_index"]]
        observed_face = matching_faces[0] if matching_faces else face
        rollout_embedding = observed_face["face_embedding"]
        drift = _compute_face_drift(
            rollout_embedding,
            face["retrieved_3d_face"]["reference_embedding"],
        )
        updated_face = {
            **face,
            "face_embedding": rollout_embedding,
            "face_bbox": observed_face.get("face_bbox", face["face_bbox"]),
            "face_confidence": observed_face.get("face_confidence", face.get("face_confidence", 0.0)),
            "feature_source": observed_face.get("feature_source", face.get("feature_source")),
            **drift,
        }
        updated_faces.append(updated_face)
        total_drift += drift["drift_score"]

    avg_drift = round(total_drift / max(1, len(updated_faces)), 4)
    candidate_face_state = {
        **face_state,
        "source_frame_path": candidate_x0["x0_preview_path"],
        "faces": updated_faces,
    }
    return {
        "candidate_id": candidate_label,
        "lambda": round(float(injection_lambda), 4),
        "denoise_state": candidate_state,
        "x0_result": candidate_x0,
        "face_state": candidate_face_state,
        "avg_drift_score": avg_drift,
    }


def _rollout_injection_window(
    generation_state: dict,
    current_step: int,
    next_step: int,
    denoise_state: dict,
    face_state: dict,
    memory: dict,
):
    """短程 rollout 自适应选择注入强度，并更新跨 shot 记忆库。

    逻辑：
    1. 用 [模型名, 去噪阶段, 当前漂移档位] 检索全局记忆。
    2. 未命中时初始化 TN(mu=0.5, sigma=0.2)。
    3. 先跑 lambda=0 的不注入 baseline。
    4. 未收敛时从 TN 采样 10 个候选；收敛时直接使用 lambda=mu。
       测试模式 MULTISHOT_ROLLOUT_SAMPLING_STRATEGY=fixed_exploration 时，
       每轮从固定 TN(0.5, 0.2^2) 探索，并强制加入高强度点。
    5. 只保留相对 baseline 有漂移改善的候选，按改善从高到低取 topk。
    6. 当前没有 VLM 自然性判断时，从 topk 里选择改善最大的候选。
    7. 用有改善样本更新该 [阶段, 漂移量] 下的 lambda 分布。
    """

    current_drift = round(
        sum(face.get("drift_score", 0.0) for face in face_state["faces"]) / max(1, len(face_state["faces"])),
        4,
    )
    drift_bucket = _drift_bucket(current_drift)
    stage = f"step_{current_step}_to_{next_step}"
    memory_key = f"{generation_state['generation_model']}:{stage}:drift_{drift_bucket}"
    memory_record = memory.get(memory_key, _initial_memory_record())

    candidate_count = int(os.getenv("MULTISHOT_ROLLOUT_CANDIDATES", "2"))
    topk = int(os.getenv("MULTISHOT_ROLLOUT_TOPK", "3"))
    min_improvement = float(os.getenv("MULTISHOT_MIN_DRIFT_IMPROVEMENT", "0.01"))
    sampling_strategy = os.getenv("MULTISHOT_ROLLOUT_SAMPLING_STRATEGY", "memory").strip()
    memory_disabled = os.getenv("MULTISHOT_DISABLE_INJECTION_MEMORY", "0") == "1"
    converge_n = int(os.getenv("MULTISHOT_MEMORY_CONVERGE_N", "12"))
    converge_sigma = float(os.getenv("MULTISHOT_MEMORY_CONVERGE_SIGMA", "0.08"))
    mu = float(memory_record.get("mu", 0.5))
    sigma = float(memory_record.get("sigma", 0.2))
    use_fixed_exploration = sampling_strategy == "fixed_exploration"
    converged = (
        not use_fixed_exploration
        and not memory_disabled
        and int(memory_record.get("samples", 0)) >= converge_n
        and sigma < converge_sigma
    )

    baseline = _evaluate_rollout_candidate(
        generation_state,
        current_step,
        next_step,
        denoise_state,
        face_state,
        0.0,
        "baseline_no_injection",
    )
    baseline_drift = baseline["avg_drift_score"]

    if use_fixed_exploration:
        sampled_lambdas = _fixed_exploration_lambdas(candidate_count)
        sampling_mode = "fixed_exploration_with_high_points"
    elif converged:
        sampled_lambdas = [round(max(0.0, min(1.0, mu)), 4)]
        sampling_mode = "converged_use_mu"
    else:
        sampled_lambdas = _sample_truncated_normal(mu, sigma, candidate_count)
        sampling_mode = "truncated_normal_sample"

    candidates = [baseline]
    parallelism = max(1, int(os.getenv("MULTISHOT_ROLLOUT_PARALLELISM", "2")))

    def run_sampled_candidate(item):
        candidate_index, injection_lambda = item
        candidate = _evaluate_rollout_candidate(
            generation_state,
            current_step,
            next_step,
            denoise_state,
            face_state,
            injection_lambda,
            f"rollout_{candidate_index}",
        )
        candidate["candidate_index"] = candidate_index
        candidate["baseline_avg_drift_score"] = baseline_drift
        candidate["improvement"] = round(baseline_drift - candidate["avg_drift_score"], 4)
        return candidate

    sampled_items = list(enumerate(sampled_lambdas))
    sampled_candidates = []
    if parallelism == 1 or len(sampled_items) <= 1:
        sampled_candidates = [run_sampled_candidate(item) for item in sampled_items]
    else:
        with ThreadPoolExecutor(max_workers=min(parallelism, len(sampled_items))) as executor:
            futures = [executor.submit(run_sampled_candidate, item) for item in sampled_items]
            for future in as_completed(futures):
                sampled_candidates.append(future.result())
        sampled_candidates.sort(key=lambda item: item["candidate_index"])

    candidates.extend(sampled_candidates)

    improved_candidates = [
        candidate for candidate in candidates[1:]
        if candidate.get("improvement", 0.0) >= min_improvement
    ]
    improved_candidates.sort(key=lambda item: item["improvement"], reverse=True)
    topk_candidates = improved_candidates[:topk]

    if converged:
        selected = candidates[1]
        accepted_samples = improved_candidates
    else:
        selected = _select_natural_candidate_from_topk(topk_candidates)
        if selected is None:
            selected = baseline
            accepted_samples = []
        else:
            accepted_samples = improved_candidates

    if use_fixed_exploration or memory_disabled:
        updated_memory_record = memory_record
    else:
        updated_memory_record = _update_distribution(
            memory_record,
            accepted_samples,
            selected["lambda"],
        )
        memory[memory_key] = updated_memory_record

    return {
        "memory_key": memory_key,
        "memory_path": str(_injection_memory_path()),
        "stage": stage,
        "drift_bucket": drift_bucket,
        "current_avg_drift_score": current_drift,
        "from_step": current_step,
        "to_step": next_step,
        "sampling_mode": sampling_mode,
        "sampling_strategy": sampling_strategy,
        "rollout_parallelism": parallelism,
        "distribution_before": {
            "mu": round(mu, 4),
            "sigma": round(sigma, 4),
            "samples": int(memory_record.get("samples", 0)),
            "converged": converged,
            "memory_disabled": memory_disabled,
        },
        "baseline_candidate": baseline,
        "rollout_candidates": candidates,
        "improved_candidates": improved_candidates,
        "topk_candidates_for_naturalness": topk_candidates,
        "selected_candidate": selected,
        "selected_denoise_state": selected["denoise_state"],
        "selected_face_state": selected["face_state"],
        "selected_x0_result": selected["x0_result"],
        "updated_distribution": updated_memory_record,
    }


def _diffusion_first_frame(
    shot_id: str,
    first_frame_prompt: str,
    generation_model: str,
    scene_asset: dict,
    character_assets: dict,
    character_ids: list[str],
):
    """真实 diffusion 首帧生成流程。

    设计逻辑：
    1. 所有去噪都走统一 _denoise_window；无注入时 lambda=0。
    2. 从 30 步开始把当前 latent 反解成 x0 预览图。
    3. InsightFace 判断人脸是否可检测，并初始化 face_state。
    4. face_state 包含 bbox、mask、pose、角色匹配、FaceLift 参考视角和漂移值。
    5. 后续每轮 rollout 都推进 5 步，选漂移最低的候选作为当前最优噪声状态。
    6. 最终从当前最优 latent decode 成 shot 首帧。
    """

    frame_path = _project_dir() / "frames" / f"{shot_id}.png"
    frame_path.parent.mkdir(parents=True, exist_ok=True)

    generation_state = {
        "shot_id": shot_id,
        "prompt": first_frame_prompt,
        "generation_model": generation_model,
        "scene_asset_path": scene_asset.get("path"),
        "character_ids": character_ids,
    }
    _prepare_diffusion_runtime(generation_state)

    denoise_log = []
    injection_memory = _load_injection_memory()
    runtime = generation_state.get("_diffusion_runtime") or {}
    runtime_steps = len(runtime.get("timesteps", []))
    requested_final_step = os.getenv("MULTISHOT_FINAL_STEP")
    final_step = int(requested_final_step) if requested_final_step else (runtime_steps or 50)
    if runtime_steps:
        final_step = min(final_step, runtime_steps)
    window_size = int(os.getenv("MULTISHOT_DENOISE_WINDOW", "5"))

    current_step = 0
    denoise_state = None
    face_state = None
    x0_result = None

    while current_step < final_step:
        next_step = min(current_step + window_size, final_step)

        if face_state is None:#尚未清晰时
            # 人脸状态尚未初始化时，所有去噪都等价于不注入：lambda=0。
            denoise_state = _denoise_window(
                generation_state,
                current_step,
                next_step,
                previous_denoise_state=denoise_state,
                injection_plan={"lambda": 0.0, "targets": []},
            )
            current_step = next_step

            record = {
                "step": current_step,
                "action": "denoise_without_injection",
                "denoise_state": denoise_state,
            }

            # 早期阶段先不判断人脸清晰度，让模型自然形成主体结构。
            face_clear_min_step = int(os.getenv("MULTISHOT_FACE_CLEAR_MIN_STEP", "30"))
            if current_step < face_clear_min_step:
                record["next_action"] = "continue_denoising_without_face_check"
                record["reason"] = f"step {current_step} is earlier than MULTISHOT_FACE_CLEAR_MIN_STEP={face_clear_min_step}"
                denoise_log.append(record)
                continue

            x0_result = _estimate_x0_preview(
                generation_state,
                denoise_state,
                str(frame_path.with_suffix(f".step_{current_step}.x0.png")),
            )
            clarity = _assess_face_clarity(x0_result)
            record.update({
                "action": "check_face_clarity",
                "x0_preview_path": x0_result["x0_preview_path"],
                "clarity": clarity,
            })

            if not clarity["is_clear"]:
                record["next_action"] = "continue_denoising_and_check_later"
                denoise_log.append(record)
                continue
            #清晰之后
            face_state = _initialize_face_state(
                x0_result["x0_preview_path"],
                character_assets,
                character_ids,
                first_frame_prompt,
            )
            record.update({
                "next_action": "initialize_face_state_and_start_rollout",
                "face_state": face_state,
            })
            denoise_log.append(record)
            continue

        # face_state 初始化之后，每一轮都用 rollout 选择最优注入强度和最优噪声状态。
        rollout = _rollout_injection_window(
            generation_state,
            current_step,
            next_step,
            denoise_state,
            face_state,
            injection_memory,
        )
        denoise_state = rollout["selected_denoise_state"]
        face_state = rollout["selected_face_state"]
        x0_result = rollout["selected_x0_result"]
        current_step = next_step
        denoise_log.append({
            "step": current_step,
            "action": "rollout_select_best_denoise_state",
            "rollout": rollout,
            "best_denoise_state": denoise_state,
            "best_face_state": face_state,
            "x0_preview_path": x0_result["x0_preview_path"],
        })

    _save_injection_memory(injection_memory)

    final_frame_path = _decode_final_image(generation_state, denoise_state, str(frame_path))
    frame_path.with_suffix(".prompt.txt").write_text(first_frame_prompt, encoding="utf-8")
    log_path = frame_path.with_suffix(".denoise_log.json")
    log_path.write_text(json.dumps(_json_safe(denoise_log), ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "frame_path": final_frame_path,
        "denoise_log_path": str(log_path),
        "denoise_log": _json_safe(denoise_log),
    }


@mcp.tool()
def generate_shot_first_frame(
    shot_id: str,
    subscript_id: str,
    character_ids: list[str],
    first_frame_prompt: str,
    generation_model: str = "juggernaut-xl-v9",
):
    """生成单个 shot 的首帧。

    shot_id: 镜头 id，例如 shot_001。
    subscript_id: 该 shot 所属场景子剧本 id，用来检索场景背景资产。
    character_ids: 该 shot 中出现的人物 id，用来检索人物参考图和 3D 人脸资产。
    first_frame_prompt: 首帧生成提示词。
    generation_model: 图像生成模型名称，便于后续比较不同开源模型。
    """

    index = _load_index()
    scene_asset = index["scene_assets"][subscript_id]
    character_assets = {
        character_id: index["character_assets"][character_id]
        for character_id in character_ids
    }

    result = _diffusion_first_frame(
        shot_id=shot_id,
        first_frame_prompt=first_frame_prompt,
        generation_model=generation_model,
        scene_asset=scene_asset,
        character_assets=character_assets,
        character_ids=character_ids,
    )

    return {
        "shot_id": shot_id,
        "frame_path": result["frame_path"],
        "denoise_log_path": result["denoise_log_path"],
    }


if __name__ == "__main__":
    mcp.run()
