# 小脸自适应 3D 一致性注入状态

> 2026-08-28：PuLID-FLUX 与 IP-Adapter/SDXL 已完成单人全身、小脸首帧及小脸侧脸验证。多人物逻辑未启用。

## 结论

固定 `0.4`、step 30～49 全程注入不适合小脸。首个 `36×49 px` 全身测试中，PuLID-FLUX 身份 cosine 从 Control `0.519150` 降到 `0.347885`，并出现红光硬贴和五官破碎。

当前 v5 改为按检测到的人脸绝对像素尺寸自动调整局部注入。相同提示词、相同 seed、相同 Control 下，该失败样例的新结果为 `0.546983`，相对 Control 提升 `+0.027833`，相对旧 v4 Treatment 提升 `+0.199098`。

三种全身场景在两个模型上均获得正向身份变化；两张近景回归也均优于旧 v4。

新增侧脸测试同样为正向：PuLID-FLUX 在约 `74°`、`44×71 px` 的极端小侧脸上，原图身份 cosine 相对 Control 提升 `+0.035690`；IP-Adapter 在约 `30°`、`70×80 px` 的小侧脸上提升 `+0.251340`。

## 当前算法

### 尺度自适应 strength

正常尺寸的基准强度仍固定为 `0.4`。设 step-30 检测脸高为 `h`：

```text
size_scale = clamp(h / 128, 0.30, 1.00)
effective_strength = 0.4 * size_scale
```

当 `h >= 128 px` 时仍使用完整 `0.4`；小脸最低使用 `0.12`。

### 尺度自适应注入窗口

基准窗口为 step 30～49，共 20 步。小脸提前停止，让剩余步骤由生成模型融合：

```text
progress = clamp((h - 48) / (128 - 48), 0, 1)
active_fraction = 0.5 + 0.5 * progress
active_steps = round(20 * active_fraction)
```

约 `48 px` 高的人脸注入 10 步；`128 px` 及以上仍注入 20 步。

### FLUX 子 token mask

旧实现先把像素 mask 压成 `32×40`，每个 packed token 只有一个 alpha，约对应 `16×16` 输出像素。

v5 先在 `64×80` VAE latent 网格生成 soft mask，再按 FLUX 的 `2×2` 打包顺序扩展到 64 个 packed channel，使四个 latent 子位置分别拥有 alpha。等效 mask 粒度由约 `16 px` 提升到约 `8 px`，且参考轨迹缓存和实际 residual 使用同一通道级 support。

### 小脸稳健调色

正常脸继续使用 v4 的空间低频 log-RGB gain。当皮肤交集少于 `1500` 像素时切换为：

- 全脸稳健中位数 log-RGB offset；
- 调色强度 `0.4`；
- gain 限制为 `[0.7, 1.3]`；
- 调色应用区仍是 soft semantic context，注入区仍是收缩 inner-face core。

当 BiSeNet 在极小脸上输出的语义区域过少时，使用同一个保守几何椭圆作为 core fallback。该区域继续排除发际线、上额头、耳朵、外脸颊和下巴边缘，不使用完整 bbox，也不扩大到背景。

## 全身小脸验证

### PuLID-FLUX，512×640

| 场景 | step-30 脸尺寸 | 有效 strength | 注入步数 | Control | Treatment | 变化 |
|---|---:|---:|---:|---:|---:|---:|
| 霓虹雨夜 | 36×49 | 0.1531 | 10 | 0.5191 | 0.5470 | +0.0278 |
| 日光站台 | 39×55 | 0.1719 | 11 | 0.4723 | 0.5254 | +0.0531 |
| 室内大厅 | 34×44 | 0.1375 | 10 | 0.3348 | 0.4395 | +0.1047 |

三个结果均完成 50 步采样，实际注入步数与自动策略一致，全部 latent finite。

### IP-Adapter/SDXL，1024×1024

全身首帧需将全局 `ip_adapter_scale` 设为 `0.2`。`0.6` 会让参考肖像压过文本构图，把 full-body prompt 拉回大头照；该构图失败样例保留在 `ip_adapter_full_body_neon_scale06_composition_failure_v5/`，不计入小脸结果。

| 场景 | step-30 脸尺寸 | 有效 strength | 注入步数 | Control | Treatment | 变化 |
|---|---:|---:|---:|---:|---:|---:|
| 霓虹雨夜 | 69×90 | 0.2797 | 15 | 0.1463 | 0.3256 | +0.1793 |
| 日光站台 | 51×68 | 0.2114 | 12 | -0.0668 | 0.0748 | +0.1416 |
| 室内大厅 | 52×66 | 0.2058 | 12 | -0.0347 | 0.1208 | +0.1555 |

IP-Adapter 三张图的注入方向均为正，但极小脸的绝对 cosine 明显低于 PuLID-FLUX。InsightFace 在几十像素人脸上的绝对数值也更不稳定，因此必须同时查看局部放大图。

## 近景回归

| 模型 | Control | 旧 v4 Treatment | v5 Treatment | v5 相对 Control |
|---|---:|---:|---:|---:|
| PuLID-FLUX | 0.8298 | 0.8126 | 0.8690 | +0.0392 |
| IP-Adapter/SDXL | 0.3160 | 0.6890 | 0.7649 | +0.4489 |

两张近景均自动保持 `0.4` 和 20 个注入步骤。PuLID Control 与旧 v4 SHA-256 完全一致。

## 小脸侧脸验证

| 模型 | step-30 yaw / 脸尺寸 | 有效 strength / 步数 | Control | Treatment | 变化 | Control 3D cosine | Treatment 3D cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| PuLID-FLUX | `+74.02°` / `44×71` | `0.2219` / 13 | 0.1337 | 0.1694 | +0.0357 | 0.1593 | 0.2649 |
| IP-Adapter | `-30.20°` / `70×80` | `0.2485` / 14 | 0.0099 | 0.2612 | +0.2513 | 0.0078 | 0.3225 |

两组均保持完整单人全身构图并通过真实 step-30 yaw 检测。IP-Adapter 的极小侧脸 Control embedding 接近失效，绝对起点不稳定，因此必须同时查看局部图；Treatment 的方向和 3D 匹配均明显为正。

PuLID 的 `74°` 样例验证了连续 Gaussian 渲染与注入链路可以运行，但超出了当前人物显式姿态标定的 `35°～65°` 覆盖范围，不能表述为“74° 已标定”。IP 的 `30°` 样例同样低于该标定 profile 的启用下限，走连续原始 camera mapping。

### 清晰度诊断

对最终检测框外扩 35% 的原生分辨率 crop 计算 Laplacian 方差：

| 模型 | Control 面部 | Treatment 面部 | Control 整图 | Treatment 整图 | 观察 |
|---|---:|---:|---:|---:|---|
| PuLID-FLUX | 339.99 | 461.64 | 198.60 | 202.98 | 注入后局部和整图均未变糊 |
| IP-Adapter | 821.18 | 623.41 | 343.59 | 341.49 | 整图不变，平滑化集中在注入脸内 |

因此当前“图像都比较模糊”的主要视觉来源有两种：几十像素小脸在诊断图里被放大后的天然像素不足，以及 IP trajectory residual 带入 FaceLift reference 的平滑低频材质。小脸自适应本身不是全局模糊来源；正常大脸仍自动回退到原来的 `0.4 / 20步`。

## 结果入口

- 六组小脸局部总览：`experiment_output/small_face_adaptive_v5_validation_contact_sheet.jpg`
- PuLID 霓虹：`experiment_output/pulid_flux_full_body_neon_small_face_adaptive_v5/`
- PuLID 日光：`experiment_output/pulid_flux_full_body_daylight_small_face_adaptive_v5/`
- PuLID 室内：`experiment_output/pulid_flux_full_body_interior_small_face_adaptive_v5/`
- IP 霓虹：`experiment_output/ip_adapter_full_body_neon_small_face_adaptive_v5_scale02/`
- IP 日光：`experiment_output/ip_adapter_full_body_daylight_small_face_adaptive_v5_scale02/`
- IP 室内：`experiment_output/ip_adapter_full_body_interior_small_face_adaptive_v5_scale02/`
- PuLID 小脸侧脸：`experiment_output/pulid_flux_full_body_side_face_small_face_adaptive_v5/`
- IP 小脸侧脸：`experiment_output/ip_adapter_full_body_side_face_small_face_adaptive_v5_scale02/`
- IP 中等脸侧脸回归：`experiment_output/ip_adapter_side_face_medium_face_regression_adaptive_v5_scale02/`
- PuLID 近景回归：`experiment_output/pulid_flux_closeup_regression_adaptive_v5/`
- IP 近景回归：`experiment_output/ip_adapter_closeup_regression_adaptive_v5/`

每个小脸目录包含完整 `comparison.jpg` 和 `face_crop_comparison.jpg`。

## 代码入口

- PuLID-FLUX：`multishot/pulid_flux_inner_face_experiment.py`
- IP-Adapter 实验：`multishot/ip_adapter_pulid_style_injection_experiment.py`
- SDXL trajectory residual：`multishot/diffusion_backend.py`

`--adaptive-small-face` 默认开启；`--no-adaptive-small-face` 仅用于严格消融。基准 `--injection-strength` / `--injection-lambda` 仍为 `0.4`。

## 当前限制

1. 本轮只验证单人，不代表多人物注入链路已完成。
2. 小于约 `40 px` 的脸接近当前 512 分辨率的实用下限，建议跳过局部轨迹注入或改用高分辨率/ROI 二次精修。
3. IP-Adapter 的全局参考权重会直接影响首帧景别；full-body 场景建议 `0.2`，近景仍可使用 `0.6`。
4. cosine 在极小脸上噪声较大，应同时报告检测置信度、局部图和人工自然度。
5. 当前尺寸策略按单张 step-30 最大脸计算；多人物版本必须对每张脸独立计算 strength、结束步和 mask。
