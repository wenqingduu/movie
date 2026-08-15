# PuLID-FLUX 3D Inner-Face 实验状态

> 给后续接手本仓库的大模型：本文是当前 PuLID-FLUX 实验的交接入口。不要重新下载模型或从头实现；主链路已经跑通，当前问题是注入结果的视觉融合质量，而不是环境或推理可用性。

## 当前结论

截至 2026-08-16，已在单张 NVIDIA GeForce RTX 4090（48 GB）上完成以下验证：

1. FLUX.1-dev FP8、PuLID-FLUX v0.9.1、本地 T5/CLIP、EVA-CLIP、InsightFace AntelopeV2 可以共同加载并完成 50 步生成。
2. FaceLift 可以从同一人物参考图生成 3D Gaussian 人脸资产，并从 `gaussians.ply` 按目标 yaw/pitch/roll 连续渲染。
3. step 30 的 `pred_x0` 可以检测到可靠人脸并估计姿态。
4. FaceLift 参考脸可以完成空间对齐、语义 inner-face mask 构建和 FLUX packed-token mask 映射。
5. step 30～49 的 masked residual injection 已按实验方案运行；Control、0.6 Treatment 和 0.4 Diagnostic 均成功结束，没有 NaN/Inf。
6. 注入在数值上明显生效，但当前视觉结果不合格：Treatment 的内脸区域过度接近 FaceLift 渲染域，出现肤色、材质和边缘不连续，呈现明显“贴脸感”。

因此，当前阶段不是“PuLID-FLUX 能否运行”，而是“如何让 3D inner-face 参考在 latent/token 域中自然融合”。

## 实验设置

- Prompt：`cinematic medium close-up portrait of a man standing beneath warm neon lights on a rainy night street, three-quarter view, natural skin texture, shallow depth of field, photorealistic, subtle rim light`
- Seed：`20260815`
- 分辨率：`512×640`
- FLUX steps：`50`
- PuLID `id_weight`：`1.0`
- Guidance：`4.0`
- 插件注入区间：step `30～49`
- 主实验注入强度：`0.6`
- 诊断组注入强度：`0.4`
- Mask：目标语义 inner-face mask 与对齐后 3D 参考脸语义 mask 的交集，羽化后映射到 packed tokens
- FaceLift 多视图扩散 smoke 设置：`10` steps
- 3D 参考模式：`facelift_continuous_gaussian_pose_render`，没有离散视图 fallback
- 参考布局：`match_target_scale`，比例 `1.0`，保持长宽比并对齐目标脸中心

本轮输入 `experiment_assets/pulid_reference.jpg` 是下载的外部真实照片，不是本项目生成的人物资产；原始 URL 未记录。文件 SHA256 为：

`1d163eb4cc3244e063895263490ee5abc199fe915e6dae9aadbdfb435523644c`

完整参数与模型文件 SHA256 记录在：

- `experiment_output/pulid_flux_smoke/config.json`
- `experiment_output/pulid_flux_diagnostic_04/config.json`

## 定量结果

| 组别 | 对原始身份参考图的 InsightFace cosine | 对匹配角度 3D 渲染脸的 cosine |
|---|---:|---:|
| Control（仅官方 PuLID） | 0.829754 | 0.685264 |
| Treatment（3D 注入 0.6） | 0.769998 | 0.973062 |
| Diagnostic（3D 注入 0.4） | 0.777916 | 0.973876 |

Control 在两次独立运行中的 PNG SHA256 都是：

`91d0903cae4a1cd7820e7eaa6268995f3671ca487e9703687ef4319bacf4d2c0`

这证明 0.6 和 0.4 对比使用了相同 seed、prompt 和基础生成轨迹。结果表明插件成功把生成脸拉向 3D 参考，但同时降低了对原始人物照片的身份相似度。

## 连续渲染与尺度验证

独立实验已复用项目主实验 `mcp_asset_server.py` 中的连续 Gaussian 渲染函数。step-30 检测姿态与对应相机为：

- InsightFace pose：pitch `-4.9918°`、yaw `1.3447°`、roll `-2.3509°`
- FaceLift camera：azimuth `268.6553°`、elevation `4.9918°`
- 目标脸尺寸：`192×270`
- 缩放后参考脸尺寸：`190×270`
- reference/target 比例：`[0.9896, 1.0000]`

`step_30/alignment.json` 记录了 PLY 路径、渲染图路径、相机参数、源/目标 bbox、缩放比例与粘贴位置。本轮 `document_deviations` 为空。

## 代码与结果入口

- 实验设计：`pulid_flux_inner_face_injection_experiment.md`
- 可执行实现：`multishot/pulid_flux_inner_face_experiment.py`
- 0.6 主实验：`experiment_output/pulid_flux_smoke/`
- 0.4 诊断组：`experiment_output/pulid_flux_diagnostic_04/`
- Control：`experiment_output/pulid_flux_smoke/control/final.png`
- 0.6 Treatment：`experiment_output/pulid_flux_smoke/treatment/final.png`
- 0.4 Diagnostic：`experiment_output/pulid_flux_diagnostic_04/treatment/final.png`
- step-30 预测、对齐图和 mask：`experiment_output/pulid_flux_smoke/step_30/`
- 指标：两组目录下的 `metrics.json`
- 每步数值日志：两组目录下的 `step_log.jsonl`

为避免仓库膨胀，本次不提交本地模型权重、虚拟环境、下载缓存、重复的 Gaussian PLY 和 turntable 视频。提交的图片、JSON 和日志足以核验本轮结论；完整模型仍保存在执行服务器上。

## 本地复现命令

模型已下载到 `third_party/PuLID/models/` 和 `third_party/FaceLift/checkpoints/`。在原执行服务器上可离线复现：

```bash
cd /root/autodl-tmp/movie
export MULTISHOT_FACELIFT_STEP_2D=10
export MULTISHOT_FACELIFT_TIMEOUT=1800
export MULTISHOT_ALLOW_FACELIFT_FALLBACK=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD"

.venv/bin/python -m multishot.pulid_flux_inner_face_experiment \
  --reference-image experiment_assets/pulid_reference.jpg \
  --reference-origin 'downloaded_external_real_photo; original_url_not_recorded' \
  --no-reference-generated \
  --output-dir experiment_output/pulid_flux_smoke \
  --injection-strength 0.6
```

诊断组只需更换输出目录并将 `--injection-strength` 改为 `0.4`。

## 建议的下一步

1. 不要继续简单降低全程固定注入强度；0.4 与 0.6 都出现同类伪影。
2. 对 FaceLift 渲染图先做颜色、曝光和低频统计匹配，再编码参考轨迹。
3. 缩小 mask 或排除高风险边缘区域，并尝试随 timestep 衰减的注入权重。
4. 评估只注入中间若干步，而不是从 step 30 一直持续到 step 49。
5. 用项目自身生成的角色参考图另做一组实验；不要与本轮外部真实照片结果混为同一输入条件。
6. 每次修改必须保留相同 Control，并同时报告原始身份相似度、3D 相似度和人工自然度检查。
