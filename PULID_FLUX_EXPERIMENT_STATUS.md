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
7. 已完成固定强度 0.4 的保守 mask 复验：排除发际线/上额头、太阳穴、脸颊外轮廓、下巴边缘、耳朵和耳饰后，mask token 覆盖率从 14.77% 降至 9.77%。外轮廓融合明显改善，但中心五官区仍有偏白、偏蜡的 FaceLift 域差，尚未达到合格视觉质量。
8. 已完成高 yaw 复验：step-30 检测 yaw `-43.7855°`，连续 Gaussian 按该姿态成功渲染并注入。原始身份 cosine 从 `0.535710` 提升到 `0.658383`，3D render cosine 从 `0.451441` 提升到 `0.855195`；身份注入在侧脸仍有效，但偏白、平光和贴脸域差比近正脸更明显。

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
- 当前保守 mask 实验注入强度：固定为 `0.4`
- 高 yaw 组：seed `20260818`、PuLID `id_weight=0.5`、要求绝对 yaw `25°～45°`
- Mask：目标语义 inner-face mask 与对齐后 3D 参考脸语义 mask 的交集，羽化后映射到 packed tokens
- 保守 Mask：在上述语义交集上进一步施加随人脸 bbox 缩放的椭圆安全区和轻微腐蚀，排除高风险外沿区域
- FaceLift 多视图扩散 smoke 设置：`10` steps
- 3D 参考模式：`facelift_continuous_gaussian_pose_render`，没有离散视图 fallback
- 参考布局：`match_target_scale`，比例 `1.0`，保持长宽比并对齐目标脸中心

本轮输入 `experiment_assets/pulid_reference.jpg` 是下载的外部真实照片，不是本项目生成的人物资产；原始 URL 未记录。文件 SHA256 为：

`1d163eb4cc3244e063895263490ee5abc199fe915e6dae9aadbdfb435523644c`

完整参数与模型文件 SHA256 记录在：

- `experiment_output/pulid_flux_conservative_mask_04/config.json`
- `experiment_output/pulid_flux_high_yaw_44_mask_04/config.json`

## 定量结果

| 组别 | 对原始身份参考图的 InsightFace cosine | 对匹配角度 3D 渲染脸的 cosine |
|---|---:|---:|
| Control（仅官方 PuLID） | 0.829754 | 0.685264 |
| Treatment（3D 注入 0.6） | 0.769998 | 0.973062 |
| Diagnostic（3D 注入 0.4） | 0.777916 | 0.973876 |
| Conservative mask（3D 注入 0.4） | 0.776806 | 0.913053 |
| High-yaw Control（yaw -43.7855°） | 0.535710 | 0.451441 |
| High-yaw Treatment（3D 注入 0.4） | 0.658383 | 0.855195 |

历史各次独立运行与当前保守 mask 组的 Control PNG SHA256 都是：

`91d0903cae4a1cd7820e7eaa6268995f3671ca487e9703687ef4319bacf4d2c0`

这证明 0.6 和 0.4 对比使用了相同 seed、prompt 和基础生成轨迹。结果表明插件成功把生成脸拉向 3D 参考，但同时降低了对原始人物照片的身份相似度。

保守 mask 复验继续得到相同的 Control SHA256。相对旧 0.4 组：

- packed mask 从 `189/1280` tokens（`14.7656%`）降至 `125/1280`（`9.7656%`）。
- step 30 首次注入 residual norm 从 `34.2796` 降至 `28.0097`。
- Treatment 相对 Control 的全图 RGB MAE 从 `9.0925` 降至 `6.3850`。
- 对 3D 渲染脸的 cosine 从 `0.973876` 降至 `0.913053`，说明 3D 参考复制强度下降；对原始照片的 cosine 基本持平。

人工检查显示耳侧、太阳穴、脸颊外轮廓和下巴的接缝明显改善，但中心额头、眼鼻和嘴部仍保留 FaceLift 的偏白、平光和蜡感。mask 收缩有效，但不能单独解决参考渲染域不一致。

高 yaw 组使用更严格的侧脸 prompt、seed `20260818` 和 PuLID `id_weight=0.5`。20/20 个注入步骤均 finite，packed mask 为 `149/1280` tokens；residual norm 从 step 30 的 `30.6833` 衰减到 step 49 的 `2.5429`。人工检查确认五官和脸型明显向 3D reference 移动，但目标脸中心的亮度、肤色和渲染材质与霓虹场景不一致，因此高 cosine 不能单独视为视觉合格。

## 连续渲染与尺度验证

独立实验已复用项目主实验 `mcp_asset_server.py` 中的连续 Gaussian 渲染函数。step-30 检测姿态与对应相机为：

- InsightFace pose：pitch `-4.9918°`、yaw `1.3447°`、roll `-2.3509°`
- FaceLift camera：azimuth `268.6553°`、elevation `4.9918°`
- 目标脸尺寸：`192×270`
- 缩放后参考脸尺寸：`190×270`
- reference/target 比例：`[0.9896, 1.0000]`

高 yaw 组的连续渲染与对齐记录为：

- InsightFace pose：pitch `0.2023°`、yaw `-43.7855°`、roll `-6.7964°`
- FaceLift camera：azimuth `313.7855°`、elevation `-0.2023°`
- 目标脸尺寸：`199×290`
- 缩放后参考脸尺寸：`197×290`
- reference/target 比例：`[0.9899, 1.0000]`

`step_30/alignment.json` 记录了 PLY 路径、渲染图路径、相机参数、源/目标 bbox、缩放比例与粘贴位置。本轮 `document_deviations` 为空。

## 代码与结果入口

- 实验设计：`pulid_flux_inner_face_injection_experiment.md`
- 可执行实现：`multishot/pulid_flux_inner_face_experiment.py`
- 0.4 保守 mask 组：`experiment_output/pulid_flux_conservative_mask_04/`
- 当前 Control：`experiment_output/pulid_flux_conservative_mask_04/control/final.png`
- 0.4 Conservative mask：`experiment_output/pulid_flux_conservative_mask_04/treatment/final.png`
- 高 yaw 组：`experiment_output/pulid_flux_high_yaw_44_mask_04/`
- 高 yaw Control/Treatment：`experiment_output/pulid_flux_high_yaw_44_mask_04/control/final.png`、`experiment_output/pulid_flux_high_yaw_44_mask_04/treatment/final.png`
- 最新 step-30 预测、对齐图和 mask：`experiment_output/pulid_flux_conservative_mask_04/step_30/`
- 指标：各实验目录下的 `metrics.json`
- 每步数值日志：各实验目录下的 `step_log.jsonl`

旧的 `facelift_smoke`、`pulid_flux_smoke` 和 `pulid_flux_diagnostic_04` 结果目录已按要求移出项目，仅保留上表中的历史指标。当前保守 mask 目录已包含独立的 Gaussian PLY、FaceLift 渲染和状态文件，不再依赖旧实验目录。为避免仓库膨胀，本地模型权重、虚拟环境和下载缓存仍不提交。

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
  --output-dir experiment_output/pulid_flux_conservative_mask_04 \
  --injection-strength 0.4
```

当前代码将 `--injection-strength` 固定为 `0.4`；传入其他值会直接报错。独立输出目录可以复用已生成的 FaceLift Gaussian，避免重复构建 3D 资产。

高 yaw 组复现命令：

```bash
.venv/bin/python -m multishot.pulid_flux_inner_face_experiment \
  --reference-image experiment_assets/pulid_reference.jpg \
  --reference-origin 'downloaded_external_real_photo; original_url_not_recorded' \
  --no-reference-generated \
  --output-dir experiment_output/pulid_flux_high_yaw_44_mask_04 \
  --seed 20260818 \
  --guidance 4.0 \
  --pulid-id-weight 0.5 \
  --min-abs-yaw 25 \
  --max-abs-yaw 45 \
  --prompt 'strict right-facing side profile portrait of the same man, face looking to frame right, only one eye visible, one ear visible, clear nose silhouette, far half of face hidden, no frontal face, cinematic warm neon rainy night street, photorealistic natural skin, medium close-up'
```

## 建议的下一步

1. 不要继续简单降低全程固定注入强度；0.4 与 0.6 都出现同类伪影。
2. 优先对 FaceLift 渲染图做颜色、曝光和低频统计匹配，再编码参考轨迹；保守 mask 和高 yaw 组都证明只能改善外沿，无法修复中心区域域差。
3. 保留当前保守 mask 作为新基线，并尝试随 timestep 衰减的注入权重。
4. 评估只注入中间若干步，而不是从 step 30 一直持续到 step 49。
5. 用项目自身生成的角色参考图另做一组实验；不要与本轮外部真实照片结果混为同一输入条件。
6. 每次修改必须保留相同 Control，并同时报告原始身份相似度、3D 相似度和人工自然度检查。
