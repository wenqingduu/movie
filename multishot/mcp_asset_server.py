import hashlib
import json
import math
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .diffusion_backend import get_diffusion_backend


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

    当前默认模型是 models/diffusion/segmind-tiny-sd。
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


def _fake_build_3d_face(reference_image_path: str, output_dir: str):
    """临时 3D 人脸建模占位函数。

    真实版本后面可以替换成 FaceLift 或其他 3D head reconstruction 工具。
    当前先创建一个目录和若干占位文件，让资产链路跑通。
    """

    face_dir = Path(output_dir)
    face_dir.mkdir(parents=True, exist_ok=True)

    model_path = face_dir / "head_3d.placeholder.txt"
    model_path.write_text(
        "TODO: build 3D face model from reference image:\n" + reference_image_path,
        encoding="utf-8",
    )

    multi_view_images = []
    for view_name in ["front", "left", "right"]:
        view_path = face_dir / f"{view_name}.png"
        view_path.touch()
        view_path.with_suffix(".prompt.txt").write_text(
            f"TODO: render {view_name} face view from 3D model built from {reference_image_path}",
            encoding="utf-8",
        )
        multi_view_images.append(str(view_path))

    return {
        "model_path": str(model_path),
        "multi_view_images": multi_view_images,
    }


@mcp.tool()
def build_3d_face_asset(character_id: str, reference_image_path: str):
    """根据人物正脸参考图生成 3D 人脸资产。

    character_id: 人物 id，例如 char_001。
    reference_image_path: 人物参考图路径，通常来自 asset_index.character_assets[character_id].path。
    """

    face_dir = _project_dir() / "assets" / "faces_3d" / character_id
    face_result = _fake_build_3d_face(reference_image_path, str(face_dir))

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


def _load_injection_memory():
    """读取注入强度记忆库。

    key 先用 denoise_step + drift_bucket 表示。
    真实实验里可以扩展成：模型名、镜头类型、人脸角度、漂移类型等维度。
    """

    memory_path = _project_dir() / "injection_memory.json"
    if memory_path.exists():
        return json.loads(memory_path.read_text(encoding="utf-8"))
    return {}


def _save_injection_memory(memory: dict):
    memory_path = _project_dir() / "injection_memory.json"
    memory_path.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(memory_path)


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


def _fake_llm_assess_face_clarity(x0_result: dict):
    """伪 LLM/VLM 评估：只根据 x0 预览图判断人脸是否清晰。

    真实版本应该只把 x0_preview_path 对应的自然图像交给 VLM/LLM，
    让模型根据图像内容返回：{"is_clear": bool, "reason": str}。
    denoise_state 只用于外层日志和后续注入，不应该作为清晰度判断输入。
    """

    step = x0_result["step"]
    mock_results = {
        30: {"is_clear": False, "reason": "pseudo VLM: face is still blurry"},
        35: {"is_clear": False, "reason": "pseudo VLM: facial structure is visible but not reliable"},
        40: {"is_clear": True, "reason": "pseudo VLM: face is clear enough for identity check"},
    }
    mock_result = mock_results.get(step, {
        "is_clear": False,
        "reason": f"pseudo VLM check at step {step}: face is not clear yet",
    })
    return {
        "x0_preview_path": x0_result["x0_preview_path"],
        "x0_image_features": x0_result["x0_image_features"],
        **mock_result,
    }


def _fake_embedding_from_key(key: str, dims: int = 8):
    """从稳定字符串生成伪 embedding，模拟 InsightFace embedding。"""

    digest = hashlib.sha256(key.encode("utf-8")).digest()
    values = [(digest[index] / 127.5) - 1.0 for index in range(dims)]
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [round(value / norm, 6) for value in values]


def _cosine_similarity(left: list[float], right: list[float]):
    return round(sum(a * b for a, b in zip(left, right)), 4)


def _fake_insightface_extract_and_match(
    frame_path: str,
    character_assets: dict,
    expected_face_count: int,
):
    """伪 InsightFace 多脸检测、角色匹配和 3D 视角检索。

    这是首次发现 x0 人脸清晰时触发的初始化步骤：
    1. 检测当前 x0 里的每张脸并提取 embedding / bbox / pose。
    2. 用检测脸 embedding 和角色库参考脸 embedding 做相似度匹配，得到 character_id。
    3. 根据 character_id + pose 从该角色 3D 人脸资产里检索指定视角参考图。

    返回值可以直接写进当前 step record；后续 rollout 只更新 faces 的一致性结果，
    不再重复做 3D 检索。
    """

    character_reference_features = {}
    for character_id, character_asset in character_assets.items():
        reference_path = character_asset.get("path", character_id)
        character_reference_features[character_id] = {
            "reference_path": reference_path,
            "character_embedding": _fake_embedding_from_key(f"character:{reference_path}"),
        }

    face_count = max(1, expected_face_count)
    faces = []
    for index in range(face_count):
        left = 100 + index * 90
        face_embedding = _fake_embedding_from_key(f"face:{frame_path}:{index}")
        pose = {
            "yaw": 5.0 + index * 3.0,
            "pitch": -2.0,
            "roll": 1.0,
            "view": "near_frontal",
        }

        best_character_id = "unknown"
        best_embedding = []
        best_similarity = 0.0
        for character_id, reference_features in character_reference_features.items():
            similarity = _cosine_similarity(
                face_embedding,
                reference_features["character_embedding"],
            )
            if best_character_id == "unknown" or similarity > best_similarity:
                best_character_id = character_id
                best_embedding = reference_features["character_embedding"]
                best_similarity = similarity

        matched_character = character_assets.get(best_character_id, {})
        face_3d = matched_character.get("face_3d", {})
        reference_image = _fake_select_3d_face_view(face_3d, pose)
        if reference_image is None:
            reference_image = matched_character.get("path")

        faces.append({
            "face_id": f"face_{index}",
            "face_index": index,
            "face_embedding": face_embedding,
            "face_bbox": [left, 80, left + 160, 260],
            "face_confidence": 0.98,
            "pose": pose,
            "matched_character_id": best_character_id,
            "character_embedding": best_embedding,
            "similarity": best_similarity,
            "drift_score": round(max(0.0, 1.0 - best_similarity), 4),
            "retrieved_3d_face": {
                "face_3d_asset": face_3d,
                "reference_image": reference_image,
                "reference_embedding": _fake_embedding_from_key(f"3d_reference:{reference_image}"),
            },
        })

    face_identity_features = {
        "source_frame_path": frame_path,
        "character_reference_features": character_reference_features,
        "faces": faces,
    }
    return {
        "face_identity_features": face_identity_features,
        "faces": faces,
    }


def _fake_select_3d_face_view(face_3d: dict, face_pose: dict):
    """根据当前检测脸角度，从 3D 人脸资产中选择最接近的参考视角。"""

    multi_view_images = face_3d.get("multi_view_images", [])
    if not multi_view_images:
        return None

    yaw = face_pose.get("yaw", 0.0)
    if yaw < -20 and len(multi_view_images) > 1:
        return multi_view_images[1]
    if yaw > 20 and len(multi_view_images) > 2:
        return multi_view_images[2]
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


def _fake_extract_face_observations(x0_preview_path: str, expected_face_count: int):
    """伪人脸检测/分割/特征提取。

    真实版本会调用：
    - 人脸检测/landmark/pose 模型
    - InsightFace embedding
    - 人脸分割模型得到每张脸的 mask

    这里只返回结构化占位结果，供 LLM 做角色匹配。
    """

    face_count = max(1, expected_face_count)
    observations = []
    for index in range(face_count):
        left = 100 + index * 90
        observations.append({
            "face_id": f"face_{index}",
            "face_index": index,
            "face_embedding": _fake_embedding_from_key(f"face:{x0_preview_path}:{index}"),
            "face_bbox": [left, 80, left + 160, 260],
            "face_mask_path": x0_preview_path.replace(".x0.png", f".face_{index}.mask.png"),
            "pose": {
                "yaw": 5.0 + index * 3.0,
                "pitch": -2.0,
                "roll": 1.0,
                "view": "near_frontal",
            },
            "face_confidence": 0.98,
        })
        Path(observations[-1]["face_mask_path"]).touch()
    return observations


def _fake_llm_assign_faces_to_characters(
    face_observations: list[dict],
    character_assets: dict,
    character_ids: list[str],
    first_frame_prompt: str,
):
    """伪 LLM 角色匹配。

    真实版本里，LLM/VLM 可以看到：
    - 当前帧提示词
    - 每张脸的 bbox/mask/局部裁剪
    - 当前 shot 的 character_ids 和人物描述/资产

    然后输出 face_id -> character_id 的对应关系。
    这里用顺序匹配做占位。
    """

    assignments = []
    for index, face in enumerate(face_observations):
        character_id = character_ids[min(index, len(character_ids) - 1)] if character_ids else "unknown"
        assignments.append({
            "face_id": face["face_id"],
            "character_id": character_id,
            "reason": "pseudo LLM assignment by face order and shot character list",
        })
    return assignments


def _fake_compute_face_drift(face_embedding: list[float], reference_embedding: list[float]):
    similarity = _cosine_similarity(face_embedding, reference_embedding)
    return {
        "similarity": similarity,
        "drift_score": round(max(0.0, 1.0 - similarity), 4),
    }


def _fake_initialize_face_state(
    x0_preview_path: str,
    character_assets: dict,
    character_ids: list[str],
    first_frame_prompt: str,
):
    """人脸状态初始化。

    只在人脸首次被 VLM/LLM 判断清晰时执行一次：
    1. 提取当前 x0 图的人脸特征、bbox、mask、pose。
    2. 让 LLM/VLM 根据当前帧提示和检测结果确定每张脸对应哪个 character_id。
    3. 根据 character_id + pose 检索该角色的 3D 人脸参考图。
    4. 计算初始漂移值。

    后续 rollout 默认人脸 id、角度、mask 不再变化，只更新 embedding/drift。
    """

    face_observations = _fake_extract_face_observations(
        x0_preview_path,
        len(character_ids),
    )
    assignments = _fake_llm_assign_faces_to_characters(
        face_observations,
        character_assets,
        character_ids,
        first_frame_prompt,
    )
    assignment_by_face = {item["face_id"]: item for item in assignments}

    faces = []
    for face in face_observations:
        assignment = assignment_by_face[face["face_id"]]
        character_id = assignment["character_id"]
        character_asset = character_assets[character_id]
        reference_embedding = _fake_embedding_from_key(f"character:{character_asset.get('path', character_id)}")
        retrieved_face = {
            "face_3d_asset": character_asset.get("face_3d", {}),
            "reference_image": _fake_select_3d_face_view(character_asset.get("face_3d", {}), face["pose"])
            or character_asset.get("path"),
            "reference_embedding": reference_embedding,
        }
        drift = _fake_compute_face_drift(face["face_embedding"], reference_embedding)
        faces.append({
            **face,
            "matched_character_id": character_id,
            "assignment_reason": assignment["reason"],
            "retrieved_3d_face": retrieved_face,
            **drift,
        })

    return {
        "source_frame_path": x0_preview_path,
        "faces": faces,
    }


def _fake_rollout_injection_window(
    generation_state: dict,
    current_step: int,
    next_step: int,
    denoise_state: dict,
    face_state: dict,
    memory: dict,
):
    """从当前 step 开始做一轮 5 步 rollout，选择最优噪声状态。

    输入：
    - denoise_state: 当前最优噪声状态。
    - face_state: 初始化后固定的人脸状态，包含 face_id、mask、pose、角色 id、3D 参考脸。
    - memory: 注入强度经验库。

    返回：
    - selected_denoise_state: 被选中的下一步噪声状态。
    - selected_face_state: 用 rollout 结果更新 drift 后的人脸状态。
    - selected_x0_result: 选中候选对应的 x0 预览。
    - rollout_candidates: 所有候选的日志。
    """

    memory_key = f"step_{current_step}_to_{next_step}:joint_faces"
    memory_record = memory.get(memory_key, {
        "samples": 0,
        "lambda_values": [0.0, 0.25, 0.5, 0.75],
        "selected_lambda_counts": {},
    })

    candidates = []
    for candidate_index, injection_lambda in enumerate(memory_record["lambda_values"]):
        injection_plan = {
            "lambda": injection_lambda,
            "targets": [
                {
                    "face_id": face["face_id"],
                    "mask_path": face["face_mask_path"],
                    "matched_character_id": face["matched_character_id"],
                    "reference_image": face["retrieved_3d_face"]["reference_image"],
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
            str(_project_dir() / "frames" / f"{generation_state['shot_id']}.step_{next_step}.rollout_{candidate_index}.x0.png"),
        )

        updated_faces = []
        total_drift = 0.0
        for face in face_state["faces"]:
            rollout_embedding = _fake_embedding_from_key(
                f"rollout:{candidate_x0['x0_preview_path']}:{face['face_id']}:{injection_lambda}"
            )
            drift = _fake_compute_face_drift(
                rollout_embedding,
                face["retrieved_3d_face"]["reference_embedding"],
            )
            updated_face = {
                **face,
                "face_embedding": rollout_embedding,
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
        candidates.append({
            "candidate_id": f"lambda_{injection_lambda}",
            "lambda": injection_lambda,
            "denoise_state": candidate_state,
            "x0_result": candidate_x0,
            "face_state": candidate_face_state,
            "avg_drift_score": avg_drift,
        })

    selected = min(candidates, key=lambda item: item["avg_drift_score"])
    selected_lambda = str(selected["lambda"])
    memory_record["samples"] += 1
    memory_record["selected_lambda_counts"][selected_lambda] = (
        memory_record["selected_lambda_counts"].get(selected_lambda, 0) + 1
    )
    memory[memory_key] = memory_record

    return {
        "memory_key": memory_key,
        "from_step": current_step,
        "to_step": next_step,
        "rollout_candidates": candidates,
        "selected_candidate": selected,
        "selected_denoise_state": selected["denoise_state"],
        "selected_face_state": selected["face_state"],
        "selected_x0_result": selected["x0_result"],
        "updated_distribution": memory_record,
    }


def _fake_diffusion_first_frame(
    shot_id: str,
    first_frame_prompt: str,
    generation_model: str,
    scene_asset: dict,
    character_assets: dict,
    character_ids: list[str],
):
    """伪 diffusion 首帧生成流程。

    设计逻辑：
    1. 所有去噪都走 _fake_denoise_window；无注入时 lambda=0。
    2. 30/35/40/45 这些 step 只负责触发 x0 解码和清晰度判断。
    3. 一旦 VLM/LLM 判断人脸清晰，就初始化 face_state。
    4. face_state 固定 face id / mask / pose / 角色匹配 / 3D 参考脸。
    5. 后续每轮 rollout 都推进 5 步，选漂移最低的候选作为当前最优噪声状态。
    6. 最终把当前最优 x0 写成 shot 首帧。
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
    final_step = int(os.getenv("MULTISHOT_FINAL_STEP", "50"))
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
            if current_step < 30:
                record["next_action"] = "continue_denoising_without_face_check"
                record["reason"] = "early denoising stage, face may be unstable"
                denoise_log.append(record)
                continue

            x0_result = _estimate_x0_preview(
                generation_state,
                denoise_state,
                str(frame_path.with_suffix(f".step_{current_step}.x0.png")),
            )
            clarity = _fake_llm_assess_face_clarity(x0_result)
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
            face_state = _fake_initialize_face_state(
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
        rollout = _fake_rollout_injection_window(
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
    generation_model: str = "dreamshaper-8",
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

    result = _fake_diffusion_first_frame(
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
