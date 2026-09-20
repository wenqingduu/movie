# 3D 人脸注入实验大模型交接文档

> 最后更新：2026-09-20。仓库根目录为 `/root/autodl-tmp/movie`。后续模型应先读本文，再读 `SMALL_FACE_ADAPTIVE_INJECTION_STATUS.md` 和 `EVALUATION_PLAN.md`。当前代码只有 pure-3D 调色与 trajectory residual 注入路径。EntityBench + Wan2.2 的首个单镜头 Control/Treatment 端到端冒烟实验已经跑通，但视频级指标尚未计算，不能据此下最终结论。

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
