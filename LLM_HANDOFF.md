# 3D 人脸注入实验大模型交接文档

> 最后更新：2026-08-16。仓库根目录为 `/root/autodl-tmp/movie`。后续模型应先读本文，再读 `HARMONIZED_3D_INJECTION_COMPARISON_STATUS.md`；不要重新下载模型。reference self-attention 策略已从代码和最终结果中删除，不要恢复。

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
6. 固定 `0.4` 的 step 30～49 局部注入。

所有最终四组实验均完成 20/20 个注入步骤，全部 finite，无 NaN/Inf。

## 2. 权威代码入口

- PuLID-FLUX 主实验：`multishot/pulid_flux_inner_face_experiment.py`
- IP-Adapter trajectory residual 实验：`multishot/ip_adapter_pulid_style_injection_experiment.py`
- IP-Adapter 实验公共评估工具：`multishot/ip_adapter_experiment_utils.py`
- SDXL/IP-Adapter 后端：`multishot/diffusion_backend.py`
- FaceLift 连续渲染与姿态标定接入：`multishot/mcp_asset_server.py`
- 姿态标定生成器：`multishot/facelift_pose_calibration.py`
- 最终四组结果摘要：`HARMONIZED_3D_INJECTION_COMPARISON_STATUS.md`
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

调色策略名：`target_low_frequency_log_rgb_v3_inner_face_apply`。

```text
pred_x0 RGB ─┐
             ├→ 仅在双方纯皮肤交集估计低频 log-linear RGB gain
3D reference ┘

gain → 应用到完整 inner-face（含眉眼、鼻、嘴、唇）
     → 在完整内脸外边界向内羽化
     → 得到 harmonized_3d_face.png
     → 直接送入 AE/VAE
```

关键原则：

- `pred_x0` 只提供低频光照参数。
- 不把任何 `pred_x0` 像素混入 3D 参考图。
- 纯皮肤 mask 只负责估光，不能同时作为调色应用 mask；否则鼻子和五官边缘会漏色。
- 当前参数：光照强度 `0.8`、低频 sigma 为脸宽 `0.09`、gain `[0.1, 1.7]`、向内羽化 `16 px`。

### 3.3 注入 mask

最终注入 mask 与调色应用 mask 是两套职责不同的 mask：

```text
target semantic inner-face → 保守椭圆核心 + 腐蚀 ─┐
reference semantic inner-face → 保守椭圆核心 + 腐蚀 ─┴→ 交集 → 2px 羽化
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
| PuLID-FLUX 小角度 | 0.829754 | 0.796340 | 0.685264 | 0.889137 | +1.3447° | +2.9400° |
| PuLID-FLUX 大角度 | 0.535710 | 0.688840 | 0.462680 | 0.858792 | -43.7855° | -44.8903° |
| IP-Adapter 小角度 | 0.316020 | 0.713831 | 0.314810 | 0.820282 | -2.1247° | -2.6143° |
| IP-Adapter 大角度 | 0.059616 | 0.605702 | 0.052479 | 0.689535 | +54.6798° | +53.3364° |

注意：PuLID 与 IP-Adapter 使用不同模型、分辨率、prompt 条件和大 yaw 方向，不能将表中差值视为严格架构排名。

## 5. 当前保留的输出

`experiment_output/` 只保留以下目录：

- `facelift_pose_calibration/`：54 个姿态样本、拟合与留出验证。
- `pulid_flux_conservative_mask_04/`：Gaussian、标定 JSON 和保守 mask 基线。
- `pulid_flux_small_yaw_harmonized_pure_3d_04/`：PuLID 小角度最终组。
- `pulid_flux_high_yaw_44_harmonized_pure_3d_04/`：PuLID 大角度最终组。
- `ip_adapter_small_yaw_harmonized_calibrated_04/`：IP-Adapter 小角度最终组。
- `ip_adapter_high_yaw_harmonized_calibrated_04/`：IP-Adapter 大角度最终组。

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
  --output-dir experiment_output/pulid_flux_small_yaw_harmonized_pure_3d_04 \
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

### IP-Adapter 小角度

```bash
.venv/bin/python -m multishot.ip_adapter_pulid_style_injection_experiment \
  --reference experiment_assets/pulid_reference.jpg \
  --gaussian-model experiment_output/pulid_flux_conservative_mask_04/input/facelift/facelift_raw/input/gaussians.ply \
  --output experiment_output/ip_adapter_small_yaw_harmonized_calibrated_04 \
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
  --output experiment_output/ip_adapter_high_yaw_harmonized_calibrated_04 \
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

1. 固定 `0.4` 注入仍会带入 FaceLift 的平滑材质，不是生产级最终参数。
2. PuLID 小角度 Control 的原始身份 cosine 已很高，加入 3D 后从 `0.829754` 降至 `0.796340`；小角度未必需要全程强注入。
3. IP-Adapter 小角度虽然身份显著提升，但仍能看出 3D 五官与材质域。
4. 当前光照迁移只有低频 RGB gain，不处理镜面高光、阴影几何和材质分解。
5. 高 yaw 标定只针对当前人物 Gaussian，跨人物泛化尚未验证。
6. 大 yaw mask 仍是 bbox 驱动的保守核心，可继续尝试随 yaw 非对称收缩远侧脸。

## 10. 推荐下一步

1. 在相同 Control 下测试 timestep 衰减权重或只注入中间窗口，优先改善小角度自然度。
2. 将 `0.4` 与 `0.2/0.3` 做严格同 Control 对照，同时报告原图身份、3D 身份、姿态和人工自然度。
3. 尝试在 latent/token 域分离低频外观与高频几何，减少直接复制 FaceLift 材质。
4. 对新人物 Gaussian 运行轻量姿态标定，不要复用当前人物标定 JSON。
5. 每次新实验都必须保留 Control 哈希和逐步 finite 日志。

## 11. 交接检查清单

- 不要恢复旧像素合成路径作为默认方案。
- 不要恢复 reference self-attention processor、实验分支或参数；IP-Adapter 局部注入只保留 trajectory residual。
- 不要用 skin label 作为最终调色应用 mask；它会漏掉鼻子和五官边缘。
- 不要让 IP-Adapter 回退到 geometric-only 注入 mask。
- 不要按绝对 yaw 选择缓存侧脸；必须根据带符号 pose 连续渲染。
- 不要提交 `.venv`、模型权重、checkpoint 或第三方缓存。
- 先确认 GPU 空闲和磁盘空间，再运行新实验。
