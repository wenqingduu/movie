# 3D 人脸注入实验大模型交接文档

> 最后更新：2026-08-28。仓库根目录为 `/root/autodl-tmp/movie`。后续模型应先读本文，再读 `SMALL_FACE_ADAPTIVE_INJECTION_STATUS.md` 和 `HARMONIZED_3D_INJECTION_COMPARISON_STATUS.md`；不要重新下载模型。reference self-attention 策略已从代码和最终结果中删除，不要恢复。

## 1. 当前任务状态

项目已经跑通两条 3D 人脸局部注入链路：

- FLUX.1-dev + PuLID-FLUX：在 step 30 分叉，构建 3D reference FLUX 轨迹，并在 step 30～49 做 masked residual injection。
- SDXL + IP-Adapter：原始照片始终作为全局 IP-Adapter 条件，3D 脸只作为 step 30～49 的局部 same-timestep trajectory residual。

当前最终实现同时具备：

1. FaceLift Gaussian 连续姿态渲染。
2. 当前人物 Gaussian 专属的正/负高 yaw camera 标定。
3. 纯 3D 光照与色调匹配，不把 `pred_x0` 像素合成到参考图。
4. target/reference BiSeNet 语义 inner-face 交集。
5. 排除发际线、太阳穴、外脸颊、下巴边缘和耳朵的保守核心 mask。
6. 正常脸以 `0.4`、step 30～49 为基准；小脸按 step-30 绝对像素高度自动降低 strength 并提前停止注入。
7. 两条链路都使用 v5 soft BiSeNet 调色上下文环和小脸稳健调色 fallback；PuLID-FLUX 额外使用 VAE 子位置级 packed mask。硬注入核心没有扩大。
8. BiSeNet 在极小脸上失效时，使用继续排除发际线、耳朵、外脸颊和下巴边缘的保守几何 core fallback。

当前六组正面/小角度全身小脸实验及两组新增侧脸小脸实验按尺寸自动执行 10～15 个注入步骤，实际步数均与策略一致，全部 finite，无 NaN/Inf；两组近景回归仍执行 20/20 步。

## 2. 权威代码入口

- PuLID-FLUX 主实验：`multishot/pulid_flux_inner_face_experiment.py`
- IP-Adapter trajectory residual 实验：`multishot/ip_adapter_pulid_style_injection_experiment.py`
- IP-Adapter 实验公共评估工具：`multishot/ip_adapter_experiment_utils.py`
- SDXL/IP-Adapter 后端：`multishot/diffusion_backend.py`
- FaceLift 连续渲染与姿态标定接入：`multishot/mcp_asset_server.py`
- 姿态标定生成器：`multishot/facelift_pose_calibration.py`
- 最终结果摘要：`HARMONIZED_3D_INJECTION_COMPARISON_STATUS.md`
- 小脸 v5 策略与六组验证：`SMALL_FACE_ADAPTIVE_INJECTION_STATUS.md`
- PuLID 历史与详细消融：`PULID_FLUX_EXPERIMENT_STATUS.md`
- 姿态标定详情：`FACELIFT_POSE_CALIBRATION_STATUS.md`

其他 IP-Adapter 状态文档只保留历史记录；其中旧输出路径可能已被清理，不应作为当前入口。

## 3. 当前算法

### 3.1 姿态与 3D 渲染

```text
step-30 latent
→ 估计 pred_x0
→ InsightFace 检测 pitch/yaw/roll
→ 查找 gaussians.pose_calibration.json
→ 若 pose 落在已标定 profile，则修正 renderer camera pose
→ FaceLift gaussians.ply 连续渲染
→ 将 3D 脸按目标 bbox 等比例对齐
```

标定文件：

`experiment_output/pulid_flux_conservative_mask_04/input/facelift/facelift_raw/input/gaussians.pose_calibration.json`

当前 profile 只覆盖绝对 yaw `35°～65°`：

- 正大 yaw：`positive_high_yaw_v1`
- 负大 yaw：`negative_high_yaw_v1`
- 小角度不应用补偿，继续使用原始 camera mapping。

该标定是当前人物 Gaussian 的模型级标定，不能未经验证直接用于其他人物的 `gaussians.ply`。

### 3.2 纯 3D 调色

PuLID-FLUX 最新调色策略名：`target_low_frequency_log_rgb_v5_small_face_robust`。

```text
pred_x0 RGB ─┐
             ├→ 仅在双方纯皮肤交集估计低频 log-linear RGB gain
3D reference ┘

gain → 应用到完整 inner-face（含眉眼、鼻、嘴、唇）
     → soft face parsing 构建约 19～20px 的调色上下文环
     → 强制实际注入区域的调色覆盖不低于注入 alpha
     → 只在上下文环外缘衰减
     → 得到 harmonized_3d_face.png
     → 直接送入 AE/VAE
```

关键原则：

- `pred_x0` 只提供低频光照参数。
- 不把任何 `pred_x0` 像素混入 3D 参考图。
- 纯皮肤 mask 只负责估光，不能同时作为调色应用 mask；否则鼻子和五官边缘会漏色。
- 当前参数：光照强度 `0.8`、低频 sigma 为脸宽 `0.09`、gain `[0.1, 1.7]`、向内羽化 `16 px`。
- 调色上下文可包含临近头发和耳部，只用于改善 AE 边界上下文；实际 latent 注入仍由保守核心限制。

### 3.3 注入 mask

最终注入 mask 与调色应用 mask 是两套职责不同的 mask：

```text
target semantic inner-face → 保守椭圆核心 + 腐蚀 ─┐
reference semantic inner-face → 保守椭圆核心 + 腐蚀 ─┴→ 交集 → 2px 羽化
→ BOX 面积采样到 32×40 packed tokens → 0.5 token Gaussian 羽化
```

排除区域：

- 发际线与上额头
- 太阳穴
- 外脸颊轮廓
- 下巴边缘
- 耳朵和耳饰

IP-Adapter 旧版只有几何椭圆，会在小角度额头注入 3D 头发。当前 IP-Adapter 已加载同一 BiSeNet parser，并使用 target/reference 保守语义交集；不要退回 geometric-only mask。

## 4. 最终结果

| 模型与角度 | Baseline 原图 cosine | 3D 注入原图 cosine | Baseline 3D cosine | 3D 注入 3D cosine | step-30 yaw | 最终 yaw |
|---|---:|---:|---:|---:|---:|---:|
| PuLID-FLUX v4 小角度 | 0.829754 | 0.812630 | 0.685264 | 0.910659 | +1.3447° | +2.9851° |
| PuLID-FLUX v4 大角度 | 0.535710 | 0.695429 | 0.462680 | 0.865799 | -43.7855° | -45.1850° |
| IP-Adapter v4 小角度 | 0.316020 | 0.689039 | 0.314810 | 0.809916 | -2.1247° | -2.7698° |
| IP-Adapter v4 大角度 | 0.059616 | 0.595695 | 0.052479 | 0.698383 | +54.6798° | +53.0692° |

注意：PuLID 与 IP-Adapter 使用不同模型、分辨率、prompt 条件和大 yaw 方向，不能将表中差值视为严格架构排名。

### v5 全身小脸

| 模型与场景 | 脸尺寸 | 实际 strength / 步数 | Control | Treatment | 变化 |
|---|---:|---:|---:|---:|---:|
| PuLID 霓虹 | 36×49 | 0.1531 / 10 | 0.5191 | 0.5470 | +0.0278 |
| PuLID 日光 | 39×55 | 0.1719 / 11 | 0.4723 | 0.5254 | +0.0531 |
| PuLID 室内 | 34×44 | 0.1375 / 10 | 0.3348 | 0.4395 | +0.1047 |
| IP 霓虹 | 69×90 | 0.2797 / 15 | 0.1463 | 0.3256 | +0.1793 |
| IP 日光 | 51×68 | 0.2114 / 12 | -0.0668 | 0.0748 | +0.1416 |
| IP 室内 | 52×66 | 0.2058 / 12 | -0.0347 | 0.1208 | +0.1555 |
| PuLID 侧脸（step-30 yaw +74.02°） | 44×71 | 0.2219 / 13 | 0.1337 | 0.1694 | +0.0357 |
| IP 侧脸（step-30 yaw -30.20°） | 70×80 | 0.2485 / 14 | 0.0099 | 0.2612 | +0.2513 |

近景回归：PuLID v5 `0.868990`，IP v5 `0.764934`，均高于对应 v4。详见 `SMALL_FACE_ADAPTIVE_INJECTION_STATUS.md`。

## 5. 当前保留的输出

`experiment_output/` 只保留以下目录：

- `facelift_pose_calibration/`：54 个姿态样本、拟合与留出验证。
- `pulid_flux_conservative_mask_04/`：Gaussian、标定 JSON 和保守 mask 基线。
- `pulid_flux_small_yaw_harmonized_soft_context_v4_04/`：PuLID 小角度最新 v4 组。
- `pulid_flux_high_yaw_44_harmonized_soft_context_v4_04/`：PuLID 大角度最新 v4 组。
- `ip_adapter_small_yaw_harmonized_soft_context_v4_04/`：IP-Adapter 小角度 v4 组。
- `ip_adapter_high_yaw_harmonized_soft_context_v4_04/`：IP-Adapter 大角度 v4 组。
- `pulid_flux_full_body_{neon,daylight,interior}_small_face_adaptive_v5/`：PuLID 三组全身小脸。
- `ip_adapter_full_body_{neon,daylight,interior}_small_face_adaptive_v5_scale02/`：IP 三组全身小脸。
- `pulid_flux_full_body_side_face_small_face_adaptive_v5/`、`ip_adapter_full_body_side_face_small_face_adaptive_v5_scale02/`：两组全身小脸侧脸。
- `ip_adapter_side_face_medium_face_regression_adaptive_v5_scale02/`：一次未保持极小脸但有效验证 `0.4 / 20步` 大脸回退的侧脸回归。
- `pulid_flux_closeup_regression_adaptive_v5/`、`ip_adapter_closeup_regression_adaptive_v5/`：近景回归。
- `small_face_adaptive_v5_validation_contact_sheet.jpg`：六组局部放大总览。

每个最终实验目录包含：输入/纯 3D 调色诊断图、mask、Control/Baseline、Treatment/residual、结构化指标和逐步数值日志。IP-Adapter 目录只保留 baseline 与 trajectory residual，不再包含 self-attention 分支。pure-3D 模式不保存旧的 `pred_x0` 像素合成图，避免将诊断分支误认为 AE/VAE 输入。

## 6. 已清理的结果

10 个被当前方案取代的目录已移出项目，包括：

- 旧的像素合成调色与不同羽化宽度消融
- 未标定 PuLID 大 yaw
- IP-Adapter scale 失败 probe
- 大 yaw 方向错误诊断
- self-attention 旧实验
- 未调色 IP-Adapter 小角度与大角度

可恢复位置：

`/root/.local/share/Trash/files/movie_obsolete_experiments_20260816_2/`

随后从两组最终 IP-Adapter 输出中移除的 4 个 self-attention 分支文件可恢复于：

`/root/.local/share/Trash/files/movie_retired_self_attention_20260816/`

2026-08-25 被 v4 替代的四个 PuLID/IP-Adapter v3 目录可恢复于：

`/root/.local/share/Trash/files/movie_superseded_harmonization_v3_20260825/`

Git 中相应旧结果会表现为删除，这是预期状态。

## 7. 运行环境

- GPU：NVIDIA RTX 4090 48 GB
- Python 环境：`.venv/`
- PuLID/FLUX 本地模型：`third_party/PuLID/models/`
- FaceLift checkpoints：`third_party/FaceLift/checkpoints/`
- 参考照片：`experiment_assets/pulid_reference.jpg`
- 参考照片 SHA256：`1d163eb4cc3244e063895263490ee5abc199fe915e6dae9aadbdfb435523644c`
- 参考照片为外部下载的真实照片，不是本项目生成；原始 URL 未记录。

模型权重、虚拟环境和第三方下载缓存受 `.gitignore` 管理，不应提交到 GitHub。

建议环境变量：

```bash
cd /root/autodl-tmp/movie
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD"
```

## 8. 精确复现命令

### PuLID-FLUX 小角度

```bash
.venv/bin/python -m multishot.pulid_flux_inner_face_experiment \
  --reference-image experiment_assets/pulid_reference.jpg \
  --reference-origin 'downloaded_external_real_photo; original_url_not_recorded' \
  --no-reference-generated \
  --output-dir experiment_output/pulid_flux_small_yaw_harmonized_soft_context_v4_04 \
  --seed 20260815 \
  --guidance 4.0 \
  --pulid-id-weight 1.0 \
  --min-abs-yaw 0 \
  --max-abs-yaw 15 \
  --harmonize-reference \
  --harmonization-reference-mode pure_3d \
  --reference-conditioning target
```

### PuLID-FLUX 大角度

```bash
.venv/bin/python -m multishot.pulid_flux_inner_face_experiment \
  --reference-image experiment_assets/pulid_reference.jpg \
  --reference-origin 'downloaded_external_real_photo; original_url_not_recorded' \
  --no-reference-generated \
  --output-dir experiment_output/pulid_flux_high_yaw_44_harmonized_soft_context_v4_04 \
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

### IP-Adapter 小角度

```bash
.venv/bin/python -m multishot.ip_adapter_pulid_style_injection_experiment \
  --reference experiment_assets/pulid_reference.jpg \
  --gaussian-model experiment_output/pulid_flux_conservative_mask_04/input/facelift/facelift_raw/input/gaussians.ply \
  --output experiment_output/ip_adapter_small_yaw_harmonized_soft_context_v4_04 \
  --seed 42 \
  --ip-adapter-scale 0.6 \
  --injection-lambda 0.4 \
  --min-abs-yaw 0 \
  --max-abs-yaw 15 \
  --harmonize-reference
```

### IP-Adapter 大角度

```bash
.venv/bin/python -m multishot.ip_adapter_pulid_style_injection_experiment \
  --reference experiment_assets/pulid_reference.jpg \
  --gaussian-model experiment_output/pulid_flux_conservative_mask_04/input/facelift/facelift_raw/input/gaussians.ply \
  --output experiment_output/ip_adapter_high_yaw_harmonized_soft_context_v4_04 \
  --seed 20260818 \
  --ip-adapter-scale 0.3 \
  --injection-lambda 0.4 \
  --min-abs-yaw 25 \
  --max-abs-yaw 55 \
  --harmonize-reference \
  --prompt 'strict right-facing side profile portrait of the same man, face looking to frame right, only one eye visible, one ear visible, clear nose silhouette, far half of face hidden, no frontal face, cinematic warm neon rainy night street, photorealistic natural skin, medium close-up'
```

已有输出目录内包含复用 Gaussian 的 `facelift_result.json`。若改用全新 PuLID 输出目录，需要先复制该记录或允许脚本重新构建 FaceLift 资产。

## 9. 已知限制

1. 正常脸仍以 `0.4` 为基准；小脸必须保留默认尺寸自适应，否则会带入 FaceLift 平滑材质并破坏五官。
2. PuLID 小角度 Control 的原始身份 cosine 已很高；v5 子位置 mask 回归为 `0.868990`，但仍需跨人物验证。
3. IP-Adapter 小角度虽然身份显著提升，但仍能看出 3D 五官与材质域。
4. 当前光照迁移只有低频 RGB gain，不处理镜面高光、阴影几何和材质分解。
5. 高 yaw 标定只针对当前人物 Gaussian，跨人物泛化尚未验证。
6. 大 yaw mask 仍是 bbox 驱动的保守核心，可继续尝试随 yaw 非对称收缩远侧脸。

## 10. 推荐下一步

1. 将 v5 尺寸自适应作为新基线；后续 timestep 衰减必须与其做相同 Control 对照。
2. 对人脸低于约 `40 px` 的场景评估跳过局部注入、高分辨率生成或 ROI 二次精修。
3. 尝试在 latent/token 域分离低频外观与高频几何，减少直接复制 FaceLift 材质。
4. 对新人物 Gaussian 运行轻量姿态标定，不要复用当前人物标定 JSON。
5. 每次新实验都必须保留 Control 哈希和逐步 finite 日志。
6. IP-Adapter 小脸侧脸的局部 Laplacian 方差由 Control `821.18` 降到 Treatment `623.41`，但整图几乎不变；后续若优化清晰度，应针对 mask 内 reference 高频，而不是全图锐化。

## 11. 交接检查清单

- 不要恢复旧像素合成路径作为默认方案。
- 不要恢复 reference self-attention processor、实验分支或参数；IP-Adapter 局部注入只保留 trajectory residual。
- 不要用 skin label 作为最终调色应用 mask；它会漏掉鼻子和五官边缘。
- 不要让 IP-Adapter 回退到 geometric-only 注入 mask。
- 不要按绝对 yaw 选择缓存侧脸；必须根据带符号 pose 连续渲染。
- 不要提交 `.venv`、模型权重、checkpoint 或第三方缓存。
- 先确认 GPU 空闲和磁盘空间，再运行新实验。
