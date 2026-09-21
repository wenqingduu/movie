# 3D 人脸注入实验大模型交接文档

> 最后更新：2026-09-22。仓库根目录为 `/root/autodl-tmp/movie`。后续模型应先读本文，再读 `SMALL_FACE_ADAPTIVE_INJECTION_STATUS.md` 和 `EVALUATION_PLAN.md`。当前代码只有 pure-3D 调色与 trajectory residual 注入路径。EntityBench + Wan2.2 单 episode 的 12 个有序镜头视频已全部生成；N 人同时注入代码、`4:2` 自动角色匹配首帧及 Control/legacy/color-safe 三条件视频已跑通，但匹配置信度 gate 与多人逐角色视频指标尚未完成。首轮错配输出不能作为算法结果。

## 1. 当前任务状态

项目已经跑通两条 3D 人脸局部注入链路：

- FLUX.1-dev + PuLID-FLUX：在 step 30 分叉，构建 3D reference FLUX 轨迹，并在 step 30～49 做 masked residual injection。
- SDXL + IP-Adapter：原始照片始终作为全局 IP-Adapter 条件，3D 脸只作为 step 30～49 的局部 same-timestep trajectory residual。

当前最终实现同时具备：

1. FaceLift Gaussian 连续姿态渲染。
2. 每个人物 Gaussian 独立执行 near-front、正高 yaw、负高 yaw camera 标定。
3. 纯 3D 光照与色调匹配，不把 `pred_x0` 像素合成到参考图。
4. target/reference BiSeNet 语义 inner-face 交集。
5. 排除发际线、太阳穴、外脸颊、下巴边缘和耳朵的保守核心 mask。
6. 单步权重以 `0.4` 为基准；PuLID-FLUX v7 从 step 30 开始最多注入 12 步，小脸再按 step-30 绝对像素高度自动降低 strength 并提前停止。
7. PuLID-FLUX 当前使用 v7 的“中央五官连通 core + 更大调色区 + 向内安全边距”；IP-Adapter 代码已同步中央五官/调色 mask 构建，但尚未用 v7 批量复跑。PuLID-FLUX 额外使用 VAE 子位置级 packed mask。
8. BiSeNet 在极小脸上失效时，使用继续排除发际线、耳朵、外脸颊和下巴边缘的保守几何 core fallback。

当前六组正面/小角度全身小脸实验及两组新增侧脸小脸实验按尺寸自动执行 10～15 个注入步骤，实际步数均与策略一致，全部 finite，无 NaN/Inf；两组近景回归仍执行 20/20 步。

## 2. 权威代码入口

- PuLID-FLUX 主实验：`multishot/pulid_flux_inner_face_experiment.py`
- PuLID-FLUX N 人同时注入实验：`multishot/pulid_flux_multi_face_experiment.py`
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

旧实验人物的历史标定文件只覆盖绝对 yaw `35°～65°`：

- 正大 yaw：`positive_high_yaw_v1`
- 负大 yaw：`negative_high_yaw_v1`
- 小角度不应用补偿，继续使用原始 camera mapping。

该标定是当前人物 Gaussian 的模型级标定，不能未经验证直接用于其他人物的 `gaussians.ply`。

EntityBench 新身份已改为对每个 Gaussian 单独生成新版标定，默认相机网格为 pitch `[-25,-10,5,20]`、yaw `[-55,-47.5,-40,-20,0,20,40,47.5,55]`、roll `[-20,0,20]`，共 108 个样本，并拟合：

- `near_frontal_v2`：覆盖 yaw `±35°`、pitch `±35°`、roll `±45°`。
- `positive_high_yaw_v1`：正大 yaw。
- `negative_high_yaw_v1`：负大 yaw。

因此新人物即使是小 yaw，也必须先读取自己的 `<gaussians.ply>.pose_calibration.json`；不要复用历史人物的标定参数。

### 3.2 纯 3D 调色

PuLID-FLUX 最新调色策略名：`target_low_frequency_log_rgb_v7_nested_color_safe_core`。

```text
pred_x0 RGB ─┐
             ├→ 仅在双方纯皮肤交集估计低频 log-linear RGB gain
3D reference ┘

gain → 应用到比身份核心更大的语义脸部调色区
     → 从调色区向内留出约脸宽 9% 的 AE/latent 安全边距
     → 安全区与中央五官核心相交后才得到注入 mask
     → 强制实际注入区域的调色覆盖不低于注入 alpha
     → 只在上下文环外缘衰减
     → 得到 harmonized_3d_face.png
     → 直接送入 AE/VAE
```

关键原则：

- `pred_x0` 只提供低频光照参数。
- 不把任何 `pred_x0` 像素混入 3D 参考图。
- 纯皮肤 mask 只负责估光，不能同时作为调色应用 mask；否则鼻子和五官边缘会漏色。
- 当前参数：光照强度 `0.8`、低频 sigma 为脸宽 `0.09`、gain `[0.1, 1.7]`、调色区向内羽化 `12 px`。
- v7 调色上下文只使用 inner-face 语义，不再包含头发、耳朵或耳饰；实际 latent 注入位于调色区内部。

### 3.3 注入 mask

最终注入 mask 与调色应用 mask 是两套职责不同的 mask：

```text
target/reference 眉眼鼻嘴语义 → 连通凸包 + 小幅扩张
→ 与双方保守 inner-face 核心相交
→ 与“向内腐蚀后的调色安全区”相交 → 2px 羽化
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

每个最终实验目录包含：输入/纯 3D 调色诊断图、mask、Control/Baseline、Treatment/residual、结构化指标和逐步数值日志。IP-Adapter 目录只包含 baseline 与 trajectory residual；pure-3D 调色不会保存或编码 `pred_x0` 像素合成图。

## 6. 已清理的结果

10 个被当前方案取代的目录已移出项目，包括：

- 旧的像素合成调色与不同羽化宽度消融
- 未标定 PuLID 大 yaw
- IP-Adapter scale 失败 probe
- 大 yaw 方向错误诊断
- 未调色 IP-Adapter 小角度与大角度

可恢复位置：

`/root/.local/share/Trash/files/movie_obsolete_experiments_20260816_2/`

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
5. 姿态标定参数不能跨人物 Gaussian 直接复用；当前做法是每个新身份独立采样和拟合。
6. 大 yaw mask 仍是 bbox 驱动的保守核心，可继续尝试随 yaw 非对称收缩远侧脸。

## 10. 推荐下一步

1. 将 v5 尺寸自适应作为新基线；后续 timestep 衰减必须与其做相同 Control 对照。
2. 对人脸低于约 `40 px` 的场景评估跳过局部注入、高分辨率生成或 ROI 二次精修。
3. 尝试在 latent/token 域分离低频外观与高频几何，减少直接复制 FaceLift 材质。
4. 对新人物 Gaussian 运行轻量姿态标定，不要复用当前人物标定 JSON。
5. 每次新实验都必须保留 Control 哈希和逐步 finite 日志。
6. IP-Adapter 小脸侧脸的局部 Laplacian 方差由 Control `821.18` 降到 Treatment `623.41`，但整图几乎不变；后续若优化清晰度，应针对 mask 内 reference 高频，而不是全图锐化。

## 11. 交接检查清单

- 调色只有 pure-3D 路径；IP-Adapter 局部注入只有 trajectory residual 路径。
- 不要用 skin label 作为最终调色应用 mask；它会漏掉鼻子和五官边缘。
- 不要让 IP-Adapter 回退到 geometric-only 注入 mask。
- 不要按绝对 yaw 选择缓存侧脸；必须根据带符号 pose 连续渲染。
- 不要提交 `.venv`、模型权重、checkpoint 或第三方缓存。
- 先确认 GPU 空闲和磁盘空间，再运行新实验。

## 12. 2026-09-20 EntityBench + Wan2.2 冒烟评测进展

### 12.1 当前评测范围

当前先验证“插件式单角色首帧注入”，不评测多角色注入，也暂不引入跨镜头历史记忆：

```text
完整有序 EntityBench episode
├── Control：首帧不做 3D trajectory residual 注入
└── Treatment：单角色镜头启用当前插件
    └── 两者使用相同提示词、seed、Wan 参数，只改变首帧是否注入
```

首轮 episode：

`00053051-5f7e-314f-85e0-517ec18f3b08__run719__i35_j44__T120`

- 共 12 个有序镜头，其中 8 个单角色镜头、4 个多角色镜头。
- 单角色镜头包含 Viktor 7 个、Roman 1 个。
- 当前只完成首个单角色镜头 `2:1` 的 PuLID-FLUX 首帧和 Wan2.2 Control/Treatment 视频。
- 原计划的首轮上限仍为 20 个视频；必须先完成本镜头视频级指标和人工检查，再决定是否扩展剩余镜头。
- 多角色镜头当前不启用插件；若纳入 episode 级完整性统计，应直接复用同一个 Control 视频，不能把它解释为“技术路线不支持多角色”。多角色代码与评测另行实现。

### 12.2 模型、源码与运行环境

- Wan checkpoint：`models/video/Wan2.2-TI2V-5B/`
- 固定模型 revision：`921dbaf3f1674a56f47e83fb80a34bac8a8f203e`
- 已核验 23/23 文件，目录约 32 GiB，五个核心文件 SHA256 与上游一致。
- 官方 Wan 源码：`/root/autodl-tmp/Wan2.2`
- 固定 Wan commit：`42bf4cfaa384bc21833865abc2f9e6c0e67233dc`
- Wan 独立环境：`/root/autodl-tmp/wan22-venv`
- CUDA 11 兼容 ONNX Runtime 层：`/root/autodl-tmp/ort-cuda11`，版本 `onnxruntime-gpu==1.17.1`。

姿态标定和视频帧身份评估需显式使用 CUDA 11 兼容层：

```bash
export PYTHONPATH=/root/autodl-tmp/ort-cuda11:$PWD
export MULTISHOT_INSIGHTFACE_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider
```

项目现有 `models/insightface/models/buffalo_l.zip` 不完整，失败解压还留下了空的 `buffalo_l/`，不要使用。代码已改为检测实际 `.onnx` 文件，并在 buffalo_l 不可用时回退到完整的 `third_party/PuLID/models/antelopev2/`。

官方 Wan 源码目录不属于 movie Git 仓库，当前有两个本地兼容修改，迁移时必须保留或重新应用：

1. `wan/__init__.py`：将与 TI2V 无关的 WanS2V/WanAnimate 改为可选导入，避免额外音频/SAM 依赖阻断 TI2V。
2. `wan/modules/model.py`：使用项目已有 attention 包装器，使其走 PyTorch SDPA fallback；没有安装 `flash-attn`。

SDPA 会显示 padding-mask 警告；当前 batch size 为 1 且本次输入无 padding，冒烟运行正常。

### 12.3 本轮代码改动（已提交并推送）

本轮代码和文档已提交并推送到 `origin/main`；交接时以远端 `main` 最新 HEAD 为当前基线。改动包括：

- `pretest/run_wan22_i2v_manifest.py`：新增 manifest 驱动的 Wan2.2 I2V 批处理入口。模型只加载一次，支持断点续跑、`reuse_video_from`，记录输入/输出 SHA256、seed 和耗时；正式默认 49 帧、50 steps、24 fps。
- `multishot/pulid_flux_inner_face_experiment.py`：新增 `--facelift-result`，允许一个角色跨镜头复用同一份 FaceLift Gaussian，不必每个镜头重建 3D 资产。
- `multishot/diffusion_backend.py`：修复旧 diffusers 对 `Path` 参数的兼容问题，并加入纯本地 SDXL Base 配置用于身份资产生成。
- `multishot/face_analysis_backend.py`：增加 AntelopeV2 fallback 与内存 BGR 帧分析入口。
- `multishot/facelift_pose_calibration.py`：新增 `near_frontal_v2` 标定，并扩展为每身份 108 个 camera 样本。
- `pretest/prepare_entitybench_pulid_pairs.py`：新增有序 episode 的 PuLID-FLUX 首帧对批处理；按角色复用身份图和 Gaussian，支持断点续跑，并为不适用镜头写入 Control 复用任务。
- `pretest/evaluate_video_identity.py`：新增逐帧原始身份锚定评估；输出 first/mean/median/P10/min/last/drift/回归斜率/检测覆盖率、成对差值、仅实际注入镜头聚合及抽帧图。
- `multishot/pulid_flux_inner_face_experiment.py`：另增加批量评测 face gate。检测脸高小于 24 px 或从 step 30 起连续 3 次无可靠脸时，不做 3D 注入，完成 Control 并让 Treatment 明确复用；同时修复延迟检测循环未使用 inference mode 导致无脸镜头计算图累积和 OOM 的问题。默认参数仍保持旧行为，pilot runner 显式启用 gate。

正常交接时 `git status --short` 应为空。`outputs/`、模型、独立环境及外部 Wan 源码均被忽略，不会随 movie 仓库提交。

### 12.4 角色资产和姿态标定

本轮资产根目录：

`outputs/entitybench_wan22_smoke/episode_00053051/assets/`

已生成并通过 InsightFace 检测的 SDXL 身份参考：

| 角色 | 身份参考 | SHA256 | 检测置信度 | yaw |
|---|---|---|---:|---:|
| Viktor | `characters/viktor_reference.png` | `65d63d708810ed005875e42e2a20bb458d5a58a955ba539d86b62ac37e678bd5` | 0.8221 | +1.1261° |
| Roman | `characters/roman_reference.png` | `cd5b0ea0f30dccf8dfd2122f620322698e1fbc5fe47482b2a6d7decc6207232c` | 0.7777 | +6.9414° |

两名角色均已成功构建 FaceLift 资产并分别完成 108 样本标定：

- Viktor：`assets/faces_3d/viktor/facelift_result.json`
- Roman：`assets/faces_3d/roman/facelift_result.json`
- 标定文件位于各自 Gaussian 旁边：`facelift_raw/input/gaussians.pose_calibration.json`

留出验证误差（pitch/yaw/roll，单位为度）：

| 角色 | near-front | 正大 yaw | 负大 yaw |
|---|---|---|---|
| Viktor | 0.6565 / 0.6956 / 2.0023 | 3.3162 / 2.0568 / 5.0931 | 3.0171 / 4.1205 / 1.9944 |
| Roman | 1.2850 / 1.4996 / 0.8031 | 0.8563 / 0.9625 / 0.1494 | 1.9169 / 3.2425 / 3.3183 |

### 12.5 首个镜头的关键结果

镜头 `2:1` 是 Viktor 的 Quarry Tunnels 近景。第一次用未覆盖 near-front 的旧标定方式运行时：

- Control 原始身份 cosine：`0.634860`
- Treatment 原始身份 cosine：`0.310273`
- 变化：`-0.324586`
- 3D Control / Treatment cosine：`0.433145 / 0.576538`
- Treatment pitch 被错误推到 `-37.52°`，而目标约为 `-21.50°`。

这不是注入强度或调色问题，而是新人物 Gaussian 的 canonical camera 映射不同。加入该人物自己的 `near_frontal_v2` 标定后，以完全相同 prompt、seed 和 Control 重跑：

| 指标 | Control | Treatment | 变化 |
|---|---:|---:|---:|
| 原始身份 cosine | 0.634860 | 0.692452 | +0.057593 |
| 匹配角度 3D cosine | 0.548087 | 0.770440 | +0.222353 |

- step-30 目标姿态：pitch/yaw/roll `[-21.498, -1.553, +5.250]°`
- 最终 Control：`[-22.498, -2.227, +5.089]°`
- 最终 Treatment：`[-21.684, -0.639, +3.968]°`
- 对比图：`outputs/entitybench_wan22_smoke/episode_00053051/first_frames/shot_2_1/pulid_flux/comparison.jpg`
- 结构化结果：同目录的 `metrics.json`、`config.json` 和 `step_log.jsonl`

结论仅限该镜头：每个新 Gaussian 必须做自己的 near-front 标定，否则即使检测 yaw 很小，渲染 pitch 仍可能严重偏离并破坏身份。

### 12.6 已完成的 Wan2.2 Control/Treatment 视频

运行清单与报告：

- `outputs/entitybench_wan22_smoke/episode_00053051/wan_manifest_shot_2_1.json`
- `outputs/entitybench_wan22_smoke/episode_00053051/wan_report_shot_2_1.json`

固定参数：Wan2.2-TI2V-5B，`1280×704`，49 帧，24 fps，50 steps，shift 5.0，guidance 5.0，seed `719001`。两组除首帧外使用相同条件。

| 条件 | 视频 | SHA256 | 耗时 |
|---|---|---|---:|
| Control | `videos/shot_2_1/control.mp4` | `1f0399d6b9af6e57d821d5d64c2ab40590978f0beec52f2441e9aed436b1ea39` | 180.97 s |
| Treatment | `videos/shot_2_1/treatment.mp4` | `7f5dd6f72c2b049eee1294069ff3cb33bba780c1231ea7cc75f5e230b615bde6` | 184.71 s |

最小 Wan 预检视频另存于：

`outputs/entitybench_wan22_smoke/preflight/wan_preflight.mp4`

### 12.7 单角色镜头批量结果

本 episode 的 8 个单角色镜头已全部完成 PuLID-FLUX 首帧处理：

| 镜头 | 首帧处理 | 原因或原始身份 cosine 变化 |
|---|---|---|
| `2:1` | 实际注入 | `0.634860 → 0.692452`，`+0.057593` |
| `4:1` | 复用 Control | step-30 脸框约 `10×13 px`，低于 24 px gate |
| `4:4` | 实际注入 | `0.584101 → 0.740027`，`+0.155926` |
| `4:5` | 实际注入 | `0.461436 → 0.590090`，`+0.128654` |
| `4:6` | 复用 Control | 后视镜头，连续 3 次无可靠脸 |
| `5:1` | 复用 Control | 坠落远景，连续 3 次无可靠脸 |
| `5:2` | 复用 Control | 亮窗剪影，连续 3 次无可靠脸 |
| `6:1` | 实际注入 | `0.654499 → 0.624917`，`-0.029583`；必须保留的负例 |

首帧批处理记录和完整 Wan manifest：

- `outputs/entitybench_wan22_smoke/episode_00053051/pulid_first_frame_report.json`
- `outputs/entitybench_wan22_smoke/episode_00053051/wan_manifest_single_character.json`

Wan2.2 已完成 manifest 中全部 16 个 job，0 失败。其中 12 条是实际生成的视频：首个 `2:1` 两条既有视频、新生成 10 条；4 个被 gate 排除的 Treatment 直接复制对应 Control。其余参数仍固定为 `1280×704`、49 帧、24 fps、50 steps、shift 5.0、guidance 5.0；完整运行耗时约 1818.9 秒。

- Wan 报告：`outputs/entitybench_wan22_smoke/episode_00053051/wan_report_single_character.json`
- 视频目录：`outputs/entitybench_wan22_smoke/episode_00053051/videos/`

### 12.8 视频逐帧身份结果

评估始终以原始角色身份图为锚点，使用同一 AntelopeV2/InsightFace 后端逐帧检测，并且不丢弃检测失败帧。完整结构化报告与抽帧：

- `outputs/entitybench_wan22_smoke/episode_00053051/video_identity_report_single_character.json`
- `outputs/entitybench_wan22_smoke/episode_00053051/video_identity_visuals_all/`

仅统计 4 个实际注入镜头的宏平均：

| 视频指标 | Control | Treatment | 平均成对变化 | 正增益镜头 |
|---|---:|---:|---:|---:|
| first | 0.580580 | 0.659802 | +0.079222 | 3/4 |
| mean | 0.471482 | 0.524104 | +0.052622 | 4/4 |
| median | 0.483601 | 0.542074 | +0.058473 | 4/4 |
| P10 | 0.357968 | 0.390752 | +0.032784 | 3/4 |
| minimum | 0.313891 | 0.317463 | +0.003572 | 2/4 |
| detection coverage | 0.816326 | 0.811224 | -0.005102 | 1/4 |

逐镜头 mean/P10 变化：

| 镜头 | mean Δ | P10 Δ | 解释 |
|---|---:|---:|---|
| `2:1` | +0.054721 | +0.050342 | 近静态镜头，身份增益全程稳定 |
| `4:4` | +0.095317 | +0.083142 | 全程正增益，但从首帧 +0.162829 衰减到末帧 +0.070241 |
| `4:5` | +0.049390 | +0.008334 | 动态走近后身份明显漂移；最低帧差值 -0.075667 |
| `6:1` | +0.011115 | -0.010488 | 首帧 -0.024962、覆盖率 -0.040817；低照消失过程不构成明确成功 |

结论只能写成：在这个单 episode pilot 的 4 个可注入镜头中，平均身份分数提高，但最差帧、时序衰减和检测覆盖没有稳定改善；不能宣称插件已经普遍解决视频身份一致性。`4:5` 和 `6:1` 是后续定位时序传播失败的重要负例，不能删除或只展示较好镜头。

### 12.9 下一步

1. 先确认工作树干净、模型路径及当前输出都存在；输出、视频和模型不进 Git。
2. 为本 episode 的 4 个多角色镜头生成 Control，并在 Treatment 侧复用，才能形成完整 12 镜头有序 episode；当前只完成 8 个单角色镜头。
3. 在同一个 episode 上跑 IP-Adapter 首帧对照；保持 Wan、seed、提示词和视频评估后端不变。
4. 将首帧/视频结果整理成统一 episode 汇总，明确区分 `plugin_applied`、`control_reused` 和检测失败。
5. 再决定是否扩展到另外两个 pilot episode；当前样本量仍不足以做普遍结论。

不要恢复 self-attention、旧像素合成调色或 geometric-only mask，也不要因为 `6:1` 是负例而从评测中删除。

## 13. 2026-09-21 v7 调色安全核心复跑

针对旧版 `6:1` 中五官过深、周围黄绿色割裂的问题，当前 PuLID-FLUX 默认策略改为：

1. 用眉毛、眼睛、鼻子、嘴和嘴唇语义构建一个连续中央身份核心，并继续与 target/reference 的保守 inner-face 相交。
2. 先构建更大的纯 3D 调色应用区，再从该区域向内留出约脸宽 `9%` 的安全边距；注入 mask 必须位于这个安全区内。
3. 单步权重继续固定为 `0.4`，但 v7 默认最多注入 12 步，避免最后几步把 3D 阴影和色调锁死。小脸自适应仍可进一步减少步数。

新输出根目录：

`outputs/entitybench_wan22_colornested_v7_s04_12/episode_00053051/`

8 个单角色镜头全部成功：4 个实际注入、4 个 Control 复用。4 个注入镜头首帧均为正增益：

| 镜头 | Control | Treatment | 变化 |
|---|---:|---:|---:|
| `2:1` | 0.634599 | 0.689214 | +0.054616 |
| `4:4` | 0.581630 | 0.660613 | +0.078983 |
| `4:5` | 0.446398 | 0.529125 | +0.082727 |
| `6:1` | 0.657453 | 0.695309 | +0.037856 |

Wan2.2 的 16 个 job 全部完成，12 条实际生成、4 条复用、0 失败。仅统计4个实际注入镜头的视频成对宏平均：

| 指标 | 旧版平均变化 | v7 平均变化 |
|---|---:|---:|
| first | +0.079222 | +0.065982 |
| mean | +0.052622 | +0.056064 |
| median | +0.058473 | +0.046387 |
| P10 | +0.032784 | +0.052329 |
| minimum | +0.003572 | +0.089353 |
| detection coverage | -0.005102 | +0.005102 |

v7 的平均首帧增益更保守，但 mean、P10、最低帧和检测覆盖更稳。`6:1` 的视频变化由旧版 first/P10/minimum 均为负，修正为：

- first `+0.037491`
- mean `+0.034569`
- P10 `+0.040788`
- minimum `+0.128121`
- detection coverage `0.0`（两组均为 27/49）

完整报告：

- `outputs/entitybench_wan22_colornested_v7_s04_12/episode_00053051/pulid_first_frame_report.json`
- `outputs/entitybench_wan22_colornested_v7_s04_12/episode_00053051/wan_report_single_character.json`
- `outputs/entitybench_wan22_colornested_v7_s04_12/episode_00053051/video_identity_report_single_character.json`
- `outputs/entitybench_wan22_colornested_v7_s04_12/episode_00053051/video_identity_visuals_all/`

仍需注意：`4:5` 的最后可检测帧差值为负，说明快速走近/出画时的传播衰减尚未解决；v7 不能据此宣称已经普遍解决时序身份一致性。

## 14. 2026-09-21 多人同时注入原型

### 14.1 Git 基线与运行状态

v7 调色安全核心、多人同时注入原型、EntityBench Control 准备和 Wan 显存记录代码已提交为 `bc18c45`（`Add color-safe multi-face injection evaluation`）。交接时以包含该提交的最新 `main` 为基线，并运行：

```bash
cd /root/autodl-tmp/movie
git status --short
git diff --check
.venv/bin/python -m py_compile \
  multishot/pulid_flux_inner_face_experiment.py \
  multishot/ip_adapter_pulid_style_injection_experiment.py \
  multishot/pulid_flux_multi_face_experiment.py \
  pretest/prepare_entitybench_pulid_pairs.py \
  pretest/prepare_entitybench_flux_controls.py \
  pretest/run_wan22_i2v_manifest.py
```

当前没有残留的 PuLID/FLUX 实验进程。自动匹配重跑目录
`outputs/entitybench_wan22_multiface_v7/episode_00053051/official_shot_4_2_autoassign/`
已经生成完整有效首帧结果，详见 14.7。除非需要验证复现性，否则不应把它误判为中断目录再次覆盖。

### 14.2 新多人实现

新入口：

`multishot/pulid_flux_multi_face_experiment.py`

已实现的逻辑：

1. 一个 prompt/seed 只生成一次共同的 step-30 target state 和 Control。
2. target 去噪不使用任何单一人物的全局 PuLID 身份条件，`target_global_pulid_id_weight=0.0`，避免把一个身份污染到所有人脸。
3. 对 step-30 图检测 N 张可靠脸；若多于 N 张，保留面积最大的 N 张。
4. 每个角色分别加载身份参考、FaceLift Gaussian 和 PuLID embedding；分别使用该目标脸的 pitch/yaw/roll 连续渲染 3D 脸。
5. 每张脸分别执行语义分割、纯 3D 调色、mask 构建、小脸 strength/结束步自适应和 same-timestep reference trajectory。
6. 每个去噪步骤把所有角色的局部 residual **同时**合成；不采用“先注入角色 A、再覆盖角色 B”的顺序式更新。
7. 若多个 mask 意外重叠，会把总 alpha 归一化到不超过 1；因此结果不依赖角色遍历顺序。
8. 同时输出两个处理分支，共享同一个 Control、seed、检测框与姿态：
   - `legacy_core`：中央身份 core 独立于调色边界选取；
   - `color_safe_core`：中央身份 core 必须落入向内腐蚀后的调色应用区，即“基于调色 mask 得到注入 mask”的 v7 方式。
9. 已用小 tensor 测试确认同时融合对角色遍历顺序不敏感，且重叠 alpha 能正确归一化；模块 `py_compile` 通过。

当前角色分配不再依赖 prompt 中的名字顺序。代码会计算：

```text
所有角色参考 InsightFace embedding
×
step-30 所有检测脸 embedding
→ cosine 矩阵
→ 穷举一对一最大总 cosine（当前 N 很小）
→ 角色 ↔ 检测脸映射
```

完整矩阵、总分和映射都会写入 `config.json`。该匹配仍需要加入置信度/间隔 gate；当 step-30 脸太小、模糊或多个角色外观接近时，不能无条件相信自动分配。

### 14.3 可用多人镜头与资产限制

run719 episode 有 4 个多人镜头：

| 镜头 | 角色 | 当前资产状态 |
|---|---|---|
| `1:1` | Viktor、Julian | 缺 Julian 身份图和 Gaussian |
| `3:1` | Silas、Viktor | 缺 Silas 身份图和 Gaussian |
| `4:2` | Viktor、Roman | 两人资产均已存在，可直接测试 |
| `4:3` | Silas、Leo | 缺 Silas、Leo 身份图和 Gaussian |

现有资产根目录：

`outputs/entitybench_wan22_smoke/episode_00053051/assets/`

因此当前唯一可做严格真实双身份注入的官方镜头是 `4:2`。不要用 Viktor/Roman 资产冒充 Julian、Silas 或 Leo。若扩到另外三个镜头，应先按各自 `entity_descriptions` 固定生成身份参考，再分别构建和标定 Gaussian；资产生成仍不是 Control/Treatment 的实验变量。

### 14.4 首轮 `4:2` 结果为什么无效

首轮完整输出：

`outputs/entitybench_wan22_multiface_v7/episode_00053051/official_shot_4_2/`

输出包含共同 Control、`legacy_core`、`color_safe_core`、两角色的渲染/调色/mask 诊断图、`metrics.json` 和 `comparison.jpg`。原始 benchmark prompt 虽写“两人 sprint away from camera”，该 seed 实际生成了两张正面可见脸，step-30 检测框为：

- 左脸约 `38×57 px`，有头发，对应 Viktor；
- 右脸约 `52×66 px`，光头，对应 Roman，yaw 约 `-29.25°`。

但首轮命令把角色按 `Roman, Viktor` 的输入顺序硬绑定到左、右脸，实际空间身份正好相反，造成交叉注入：Roman 3D 注入 Viktor 脸，Viktor 3D 注入 Roman 脸。该输出只用于证明“按 prompt/参数顺序绑定角色”不可行，所有 identity 增益均为无效数据，不能用于比较 `legacy_core` 与 `color_safe_core`。

正因为这个失败，代码随后加入了 14.2 所述的一对一自动匹配。有效自动匹配重跑现已完成，结果见 14.7。

### 14.5 自动匹配精确复现命令（已完成）

先确保 GPU 空闲，然后在 repo 根目录运行：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD"

PROMPT="$(.venv/bin/python - <<'PY'
import json
p = 'benchmarks/entitybench/data/scripts/00053051-5f7e-314f-85e0-517ec18f3b08__run719__i35_j44__T120.json'
d = json.load(open(p, encoding='utf-8'))
print(d['scenes'][3]['video_prompts'][1])
PY
)"

.venv/bin/python -m multishot.pulid_flux_multi_face_experiment \
  --character "Roman=outputs/entitybench_wan22_smoke/episode_00053051/assets/characters/roman_reference.png=outputs/entitybench_wan22_smoke/episode_00053051/assets/faces_3d/roman/facelift_result.json" \
  --character "Viktor=outputs/entitybench_wan22_smoke/episode_00053051/assets/characters/viktor_reference.png=outputs/entitybench_wan22_smoke/episode_00053051/assets/faces_3d/viktor/facelift_result.json" \
  --output-dir outputs/entitybench_wan22_multiface_v7/episode_00053051/official_shot_4_2_autoassign \
  --prompt "$PROMPT" \
  --seed 719004 \
  --max-face-detection-retries 3
```

一次运行会生成：

```text
control/final.png
legacy_core/treatment/final.png
color_safe_core/treatment/final.png
comparison.jpg
metrics.json
config.json
step_log.json
legacy_core/references/<角色>/...
color_safe_core/references/<角色>/...
```

预期耗时约 8 分钟（RTX 4090 48GB）。完成后必须先核对：

1. `config.json.character_assignment.assigned_characters_left_to_right` 应为 `Viktor, Roman`；
2. 每个角色的 bbox、姿态、Gaussian 与参考图路径正确；
3. 两个处理分支相对于 Control 的逐角色 cosine，而不是只看多人平均；
4. 局部放大检查五官、肤色、mask 边界和角色是否互换；
5. `step_log.json` 中每步 finite，组合 alpha 未产生异常重叠。

该自动匹配首帧实验已有效完成。若继续运行 Wan2.2，建议 manifest 包含 `control`、`legacy_core`、`color_safe_core` 三个条件，并按每个角色分别逐帧匹配和计分。现有 `pretest/evaluate_video_identity.py` 是单目标评估器，不能直接把最大脸当作两个人共同的身份分数；多人评估应做每帧检测脸与两张身份参考的一对一匹配，并报告每个角色的 first/mean/P10/min/coverage 及宏平均。

### 14.6 仍需实现或验证

- 自动角色匹配的置信度 gate 和歧义处理；不能在低分/小间隔时静默交换身份。
- 多人首帧自动批处理 runner、Wan 三条件 manifest 和逐角色视频评估器。
- 同一镜头多个 mask 接近或遮挡时的空间冲突测试。
- 至少增加 2～3 个多人镜头/seed 后再判断 v7 是否优于 legacy；单个 `4:2` 不能形成普遍结论。
- 当前只实现 PuLID-FLUX 多人路径；IP-Adapter 多目标路径虽在通用后端保留了 `targets[]` 接口，但尚未按同一条件做有效多人验证，不能宣称两模型均支持等价的多人实验。
- 完成验证后再更新 `EVALUATION_PLAN.md` 的多人范围，并提交/推送代码；提交前不要删除首轮错配输出，它是角色绑定失败的诊断证据。

### 14.7 `4:2` 自动角色匹配有效重跑结果

已按 14.5 的命令完整重跑，进程退出码为 0。有效输出目录为：

`outputs/entitybench_wan22_multiface_v7/episode_00053051/official_shot_4_2_autoassign/`

首轮 `official_shot_4_2/` 仍只作为“按输入顺序绑定会交叉注入”的诊断证据，下面所有数值均来自 `official_shot_4_2_autoassign/`，不得混用首轮错配指标。

自动匹配结果正确恢复了视觉空间身份：

- 左脸：Viktor，bbox `[194,98,232,155]`，约 `38×57 px`，pitch/yaw/roll `[-12.782,-11.159,-4.487]°`。
- 右脸：Roman，bbox `[340,59,392,125]`，约 `52×66 px`，pitch/yaw/roll `[-15.687,-29.249,-1.783]°`。
- `config.json.character_assignment.assigned_characters_left_to_right` 为 `["Viktor","Roman"]`。
- 角色参考图与 FaceLift 记录均指向各自资产，没有交叉使用。

step-30 自动匹配 cosine 矩阵的行顺序为 Roman、Viktor，列顺序为左脸、右脸：

```text
[
  [-0.031663,  0.105518],  # Roman
  [-0.005628, -0.112998],  # Viktor
]
```

选中映射总分为 `0.099891`，交换映射总分为 `-0.144661`，全局间隔为 `0.244552`；Roman 和 Viktor 的逐角色候选间隔分别为 `0.137181`、`0.107370`。映射方向与发型/光头外观一致，但绝对分数很低，尤其 Viktor 的选中 cosine 仍为负数，因此仍必须实现绝对置信度与歧义 gate，不能把本次成功方向外推为自动匹配已经稳健。

逐角色首帧身份结果：

| 角色 | Control | legacy_core | legacy Δ | color_safe_core | color-safe Δ |
|---|---:|---:|---:|---:|---:|
| Viktor（左） | 0.018376 | 0.024857 | +0.006481 | 0.030058 | +0.011681 |
| Roman（右） | 0.093390 | 0.149169 | +0.055779 | 0.156139 | +0.062749 |
| 两角色宏平均 | 0.055883 | 0.087013 | +0.031130 | 0.093098 | +0.037215 |

`color_safe_core` 在两名角色上都比 Control 和 `legacy_core` 略高，但只有一个镜头，且 Viktor 绝对分数很低，不能据此宣称 v7 普遍优于 legacy。视觉放大检查未见角色交换、明显发际线/外轮廓泄漏或硬接缝；Roman 的眼神和中央五官变化较明显，Viktor 因脸更小、packed mask 很弱而变化较轻。

数值日志检查：

- `config.json`、`metrics.json`、`step_log.json` 中所有浮点值 finite。
- step 30～40 两名角色同时注入，step 41 仅 Roman 继续，step 42～49 均停止，符合小脸自适应与 12 步上限。
- 两个分支每步 `raw_combined_alpha_max` 最大均为 `0.16015625`，未发生异常 mask 重叠或 alpha 超限。
- `color_safe_core` 的 Viktor/Roman 安全核心保留率分别为 `1.0`、`0.993758`。

关键结构化文件：

- `official_shot_4_2_autoassign/config.json`
- `official_shot_4_2_autoassign/metrics.json`
- `official_shot_4_2_autoassign/step_log.json`
- `official_shot_4_2_autoassign/comparison.jpg`

Control、legacy、color-safe 三个首帧的 Wan2.2 视频现已生成，详见第 15 节。下一步仍是补角色匹配 gate，并按每帧两张参考图与检测脸做一对一逐角色视频评估；不能复用单目标“最大脸”评估逻辑。

## 15. 2026-09-22 完整有序 episode 视频与 Wan 显存实测

### 15.1 完整 episode 状态

权威输出根目录：

`outputs/entitybench_wan22_colornested_v7_s04_12/episode_00053051/`

12 个有序 shot 的主 Control/Treatment 已全部存在，共 24 条主视频；`4:2` 另保留 1 条 `legacy_core` 消融视频。所有单 shot 视频均为 `1280×704`、49 帧、24 fps、50 steps，已用 ffprobe 逐条读取验证。

主 Treatment 定义：

- 实际应用插件：`2:1`、`4:2`、`4:4`、`4:5`、`6:1`。
- Treatment 复用 Control：`1:1`、`3:1`、`4:1`、`4:3`、`4:6`、`5:1`、`5:2`。
- `4:2` 主 Treatment 使用 `color_safe_core`；`legacy_core` 只作为额外消融。
- `1:1`、`3:1`、`4:3` 缺少完整多角色身份资产，因此只生成零全局身份权重的多人 Control 并复用，不冒充插件成功。
- `4:2` 使用 14.7 的有效自动角色匹配首帧，Control 保持 `target_global_pulid_id_weight=0.0`，legacy/color-safe 是两个真实多人局部注入条件。

7 个 Control 复用镜头不是随机关闭插件：

- `4:1`：step-30 检测脸高仅 `13 px`，低于当前 `24 px` 可靠注入阈值。
- `4:6`：后脑/背影视角；连续尝试到去噪 step 32 仍没有可靠人脸。
- `5:1`：高机位快速坠落的全身人物；没有可靠人脸。
- `5:2`：窗内远距离剪影；没有可靠人脸。
- `1:1`：缺 Julian 身份参考和 Gaussian。
- `3:1`：缺 Silas 身份参考和 Gaussian。
- `4:3`：缺 Silas、Leo 身份参考和 Gaussian。

前三类检测失败保护用于避免把错误身份 residual 注入背景或错误部位；多人资产缺失则用于避免拿 Viktor/Roman 冒充其他角色。它们不表示算法原则上不能处理这些镜头，补齐资产并通过检测/角色映射 gate 后可以再启用。

完整结构化入口：

- `wan_manifest_full_episode.json`：12-shot 主对照加 `4:2 legacy`，共 25 个 job。
- `wan_report_full_episode_summary.json`：逐文件 SHA256、ffprobe、复用关系、插件应用状态和显存观察。
- `wan_report_remaining_worker_a_retry.json`：`1:1`、`4:2 Control/legacy` 的最终权威报告。
- `wan_report_remaining_worker_b.json`：`3:1`、`4:3`、`4:2 color-safe` 的权威报告。
- `wan_report_remaining_worker_a.json`：双进程切换时 OOM 的诊断报告，不是最终完成状态。

按官方顺序无重编码拼接出的完整视频：

- `episode_sequences/episode_control.mp4`
- `episode_sequences/episode_treatment_color_safe.mp4`
- `episode_sequences/episode_treatment_4_2_legacy.mp4`

三条拼接序列均为 588 帧、约 24.50 秒、1280×704。concat 后容器的平均帧率表示为近似 24 fps 的有理数；各原始 shot 文件仍是严格 `24/1`。

### 15.2 `4:2` 三条件 Wan 视频

三条视频使用相同官方 prompt、seed `719004` 和 Wan 参数，只改变首帧：

- Control：`videos/shot_4_2/control.mp4`
- legacy：`videos/shot_4_2/legacy_core.mp4`
- color-safe：`videos/shot_4_2/treatment.mp4`

首帧 SHA256 分别对应 14.7 已验证的：

- Control：`9eba30a4c9255b7cab2177ee1eab8c71e4437b07c51f316d40fe707ed602accb`
- legacy：`1b7c345763d0eae1a8c058e2cd4d794d090e88d518bf60bd27aa9ec0e3b909e4`
- color-safe：`f1eb0940520f6392189fd61be8b688abf2a9f3bc1b933a18e7c53e37b236bcb7`

尚未计算多人视频逐角色身份指标，不能用单人评估器把最大脸当作两个人的共同分数。

### 15.3 Wan2.2-5B 显存和并发结论

`pretest/run_wan22_i2v_manifest.py` 已增加每个实际生成 job 的 CUDA peak allocated/reserved 记录。本轮配置仍为 `offload_model=True`、`convert_model_dtype=True`、`t5_cpu=True`。

实测：

- 模型刚加载后约 allocated `2.82 GB`、reserved `2.87 GB`。
- 单任务最大 allocated `23,832,273,920 bytes`，约 `22.19 GiB`。
- 连续任务最大 reserved `34,345,058,304 bytes`，约 `31.99 GiB`。
- 两进程第一轮稳定采样时，nvidia-smi 合计约 `33,486 MiB`，GPU 100%。
- 两进程都进入后续任务时出现阶段性峰值：一个进程约占 `32.42 GiB`，另一个约 `14.85 GiB`，最终因只剩约 86 MiB 而 OOM。
- OOM 只影响尚未开始的 `4:2 legacy_core`；已有视频保留，随后单进程断点续跑成功。

结论：Wan2.2-5B 在 offload 下“模型加载常驻显存低”，但完整生成峰值并不低。单张 48GB GPU 上两个进程可以短暂并行，却不适合可靠的连续批处理；默认应保持一个 Wan 进程、一次加载模型、manifest 内串行生成。真正并行应使用多 GPU，或把每个进程限制为单个 job 并严密监控峰值。

### 15.4 本轮新增代码与未完成项

- 新增 `pretest/prepare_entitybench_flux_controls.py`：为本轮缺少完整身份资产或可靠角色映射的指定多人回退镜头，生成与多人原型一致的零全局 PuLID 权重 Control，并生成 Control 复用 manifest。该回退不代表多人注入算法本身不受支持。
- 更新 `pretest/run_wan22_i2v_manifest.py`：记录模型加载后显存和每个生成 job 的 CUDA 峰值。
- 上述代码已提交为 `bc18c45`；不要恢复旧 self-attention、旧像素合成调色或 geometric-only mask。

后续优先级：

1. 实现多人视频逐帧“一对一身份匹配”评估器，评估 `4:2` Control/legacy/color-safe 的 Viktor 与 Roman。
2. 人工检查三条完整 episode 和 `4:2` 三条件的运动、闪烁、身份交换及首帧伪影传播。
3. 实现自动角色匹配的绝对分数、候选间隔和全局间隔 gate。
4. 再决定是否补齐 Julian、Silas、Leo 资产并扩展其他多人镜头；不要因为本轮 Control 复用而宣称这些镜头已经验证多人插件。
