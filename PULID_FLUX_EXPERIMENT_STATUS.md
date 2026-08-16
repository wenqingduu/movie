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
9. 已完成 3D 参考光照迁移实验：skin-only log-RGB 低频增益可以基本消除白色面具并恢复红蓝霓虹光，但 16px 像素合成羽化会混合目标与 3D 两套五官，使身份 cosine 降到 `0.490374`；缩至 4px 后恢复到 `0.530356`，但出现局部硬边，仍未超过 Control。因此调色实现保留为可选实验，暂不替换默认保守 mask 基线。
10. 已完成“调色后的纯 3D 脸直接送入 AE”对照：`pred_x0` 只用于在重叠纯皮肤区估计低频 log-RGB 光影，增益应用到完整 inner-face（包含眉眼、鼻、嘴和唇），不再有任何 `pred_x0` 像素进入参考图。收缩 mask、step 30～49 和权重 `0.4` 均不变。原始身份 cosine 为 `0.688840`，3D cosine 为 `0.858792`，最终 yaw 为 `-44.8903°`；鼻子和五官边缘的未调色白边已经消除。
11. 当前同策略小角度复验也已完成：step-30 yaw `+1.3447°`，原始身份 cosine `0.796340`，3D cosine `0.889137`，最终 yaw `+2.9400°`。高 yaw 标定 profile 未错误应用到该小角度姿态。PuLID 与 IP-Adapter 的最终四组对比见 `HARMONIZED_3D_INJECTION_COMPARISON_STATUS.md`。

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
- 大 yaw 组：seed `20260818`、guidance `4.0`、PuLID `id_weight=0.5`、实测绝对 yaw 验收范围 `25°～45°`

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

## 大 yaw 验证

为避免原始正脸参考在 `id_weight=1.0` 下压制姿态，先用 yaw gate 检查多个候选。强 prompt 和 guidance `8.0` 在 `id_weight=1.0` 下仍只得到约 `1°～10°` yaw；将 PuLID `id_weight` 降至 `0.5` 后，step 30 得到：

- pitch `0.2023°`、yaw `-43.7855°`、roll `-6.7964°`
- 检测置信度 `0.8129`
- 连续 Gaussian 相机 azimuth `313.7855°`、elevation `-0.2023°`
- 目标脸 `199×290`，对齐参考脸 `197×290`

| 大 yaw 组别 | 对原始照片 cosine | 对匹配角度 3D 脸 cosine | 最终 yaw |
|---|---:|---:|---:|
| Control（PuLID 0.5） | 0.535710 | 0.451441 | -42.5001° |
| Treatment（PuLID 0.5 + 3D 注入 0.4） | 0.658383 | 0.855195 | -51.7436° |

大 yaw Treatment 相对 Control 的原始身份 cosine 提升 `0.122673`，3D cosine 提升 `0.403755`。20 个注入步骤全部有限，mask 覆盖 `149/1280` tokens（`11.6406%`），首次 residual norm 为 `30.6833`。

人工检查结论是：大 yaw 下 3D 注入明显恢复了参考人物的眼、鼻、嘴和脸型特征，但 FaceLift 的偏白平光材质覆盖了目标的红蓝霓虹光照，中心脸部出现明显大块贴片；同时最终 yaw 估计比 step 30 多侧转约 `8°`。因此这组证明了 3D 约束在大姿态下的价值，也更清楚地暴露了下一阶段必须解决的颜色域匹配与姿态感知 mask 问题。

### 光照迁移消融

新增可选 `--harmonize-reference` 路径：仅用双方 skin label 交集估计 log-linear RGB 低频光照，将目标 `pred_x0` 的红蓝光照施加到对齐后 3D 脸。早期消融以目标图作为 AE 编码上下文；最新 `--harmonization-reference-mode pure_3d` 则直接编码调色后的纯 3D 图，不混入目标像素。各组使用相同 Control、yaw、收缩 mask 和注入强度 0.4。

| 大 yaw Treatment | 原始照片 cosine | 3D cosine | 最终 yaw | 人工结果 |
|---|---:|---:|---:|---|
| 原始 3D 白脸参考 | 0.658383 | 0.855195 | -51.7436° | 身份最强，白色面具明显 |
| 调色 + 16px 向内羽化 | 0.490374 | 0.594839 | -50.5332° | 光影最自然，身份明显回退 |
| 调色 + 4px 向内羽化 | 0.530356 | 0.643797 | -52.1169° | 身份部分恢复，但鼻侧有硬边 |
| 纯 3D 完整 inner-face 调色后直接 AE | 0.688840 | 0.858792 | -44.8903° | 不混入目标像素，鼻子与五官边缘连续，身份约束强 |

neutral reference prompt 与 target prompt 的调色结果几乎相同（原始身份 cosine 分别为 `0.491652` 与 `0.490374`），说明身份损失主要来自像素层合成，而不是参考轨迹文本条件。额外检查显示：原始 3D 脸对原图 cosine 为 `0.698078`，单纯调色后仍有 `0.613386`，但与 `pred_x0` 做 16px 合成后降至 `0.485823`。因此后续不应继续扩大像素羽化，而应尝试在 latent 中保留目标低频统计、只注入 3D 高频结构。

纯 3D 对照进一步验证了这一判断。初版若把光照增益硬限制在 skin label 或注入核心区，会导致鼻子、眉眼、嘴唇周围未调色，或产生明显多边形色块。最终实现拆成三个 mask：纯皮肤交集只负责估计光照，完整 inner-face 负责应用光照，保守核心 mask 只负责 latent 注入。调色只在完整内脸外边界向内平滑衰减，鼻子和五官不再形成内部空洞。最终 Control PNG SHA256 仍为 `d31555651470215b50b6d9b2698688b9a0aca1356e5977eaf8f1143b6a4c5e64`，50 步日志无 NaN/Inf。

注意：最新纯 3D 调色组自动应用了 `negative_high_yaw_v1` 姿态标定，输入 yaw `-43.7855°` 被修正为 renderer yaw `-36.1814°`；旧的原始 3D 白脸组未使用该标定。因此两者的数值不能视为严格单变量调色消融，下一轮应以当前标定再跑一次未调色 3D 基线。

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
- 高 yaw 纯 3D 调色组：`experiment_output/pulid_flux_high_yaw_44_harmonized_pure_3d_04/`
- 小角度纯 3D 调色组：`experiment_output/pulid_flux_small_yaw_harmonized_pure_3d_04/`
- 最新 step-30 预测、对齐图和 mask：`experiment_output/pulid_flux_conservative_mask_04/step_30/`
- 大 yaw step-30 预测、对齐图和 mask：`experiment_output/pulid_flux_high_yaw_44_harmonized_pure_3d_04/step_30/`
- 指标：各实验目录下的 `metrics.json`
- 每步数值日志：各实验目录下的 `step_log.jsonl`

旧的 `facelift_smoke`、`pulid_flux_smoke` 和 `pulid_flux_diagnostic_04` 结果目录已按要求移出项目，仅保留上表中的历史指标。当前保守 mask 目录已包含独立的 Gaussian PLY、FaceLift 渲染和状态文件；大 yaw 组复用该资产。为避免仓库膨胀，本地模型权重、虚拟环境和下载缓存仍不提交。

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
  --output-dir experiment_output/pulid_flux_high_yaw_44_harmonized_pure_3d_04 \
  --seed 20260818 \
  --guidance 4.0 \
  --pulid-id-weight 0.5 \
  --min-abs-yaw 25 \
  --max-abs-yaw 45 \
  --harmonize-reference \
  --harmonization-reference-mode pure_3d \
  --reference-conditioning target \
  --prompt 'strict right-facing side profile portrait of the same man, face looking to frame right, only one eye visible, one ear visible, clear nose silhouette, far half of face hidden, no frontal face, cinematic warm neon rainy night street, photorealistic natural skin, medium close-up'
```

## 建议的下一步

1. 不要继续简单降低全程固定注入强度；0.4 与 0.6 都出现同类伪影。
2. 不再使用像素层 `pred_x0` 合成；保留“纯皮肤估计、完整 inner-face 应用、保守核心注入”的纯 3D 调色方案。
3. 用当前姿态标定补跑未调色 3D 基线，再严格量化纯 3D 调色本身的收益。
4. 保留当前保守 mask 作为新基线，并尝试随 timestep 衰减的注入权重。
5. 评估只注入中间若干步，而不是从 step 30 一直持续到 step 49。
6. 大 yaw 下使用姿态感知的非对称 mask，进一步排除远侧脸轮廓和被遮挡区域；当前 bbox 椭圆 mask 在侧脸上仍偏宽。
7. 用项目自身生成的角色参考图另做一组实验；不要与本轮外部真实照片结果混为同一输入条件。
8. 每次修改必须保留相同 Control，并同时报告原始身份相似度、3D 相似度、最终姿态和人工自然度检查。
