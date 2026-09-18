# IP-Adapter + 3D Trajectory Residual 当前状态

> 2026-08-28 更新：reference self-attention 策略及其实验入口已经删除。当前 v5 只保留原始照片 IP-Adapter baseline 与局部 3D trajectory residual，并加入按脸高自适应 strength/结束步、小脸稳健调色和极小脸几何 core fallback。完整六组验证见 `SMALL_FACE_ADAPTIVE_INJECTION_STATUS.md`。

## 当前策略

两条分支共享相同的 SDXL 初始噪声、prompt、seed、step-30 latent，以及原始照片全局 IP-Adapter 条件：

| 分支 | 原始照片 IP-Adapter | 3D trajectory residual |
|---|---|---|
| `ip_adapter_baseline` | 是 | 否 |
| `ip_adapter_plus_pulid_style_residual` | 是 | 是 |

正常尺寸脸的局部更新仍为：

```text
target_next += 0.4 * face_mask * (reference_next - target_next)
```

`reference_next` 来自调色后的纯 3D reference VAE `x0`，使用固定 reference noise 加噪到相同 DDIM next timestep；最后一步使用干净 reference `x0`。正常脸使用 step 30～49；小脸根据 step-30 脸高自动降低 strength 并提前停止。

调色优先使用 target/reference 纯皮肤交集估计低频 RGB gain，并用 BiSeNet soft 概率构建调色上下文环。小脸支持不足时改用稳健中位数 gain；BiSeNet 极小脸失败时使用同样排除发际线、上额头、太阳穴、外脸颊、下巴边缘和耳朵的几何 core fallback。

## 小角度最终结果

输出目录：`experiment_output/ip_adapter_small_yaw_harmonized_soft_context_v4_04/`

| 分支 | 原始照片 cosine | 连续 3D cosine | 最终 yaw |
|---|---:|---:|---:|
| IP-Adapter baseline | 0.316020 | 0.314810 | -2.7397° |
| + trajectory residual | 0.689039 | 0.809916 | -2.7698° |

- step-30 yaw：`-2.1247°`
- 注入：20/20 步成功且全部 finite
- 姿态标定：未应用，原因是 `pose_outside_calibrated_profiles`
- 权重：固定 `0.4`

## 代码与结果入口

- 实验：`multishot/ip_adapter_pulid_style_injection_experiment.py`
- 公共评估工具：`multishot/ip_adapter_experiment_utils.py`
- 后端：`multishot/diffusion_backend.py` 的 `trajectory_residual` 模式
- 对照图：`experiment_output/ip_adapter_small_yaw_harmonized_soft_context_v4_04/comparison.jpg`
- 结构化指标：`experiment_output/ip_adapter_small_yaw_harmonized_soft_context_v4_04/result.json`
- 最终图：`experiment_output/ip_adapter_small_yaw_harmonized_soft_context_v4_04/branches/`
- 每步日志：`experiment_output/ip_adapter_small_yaw_harmonized_soft_context_v4_04/logs/`

完整复现命令与四组结果见 `LLM_HANDOFF.md`。

## v5 全身小脸结果

全身首帧使用 `ip_adapter_scale=0.2`；`0.6` 会把 full-body 文本构图拉回近景肖像。

| 场景 | 脸尺寸 | 实际 strength / 步数 | Control | Treatment | 变化 |
|---|---:|---:|---:|---:|---:|
| 霓虹雨夜 | 69×90 | 0.2797 / 15 | 0.1463 | 0.3256 | +0.1793 |
| 日光站台 | 51×68 | 0.2114 / 12 | -0.0668 | 0.0748 | +0.1416 |
| 室内大厅 | 52×66 | 0.2058 / 12 | -0.0347 | 0.1208 | +0.1555 |

近景回归保持 `0.4 / 20`，Treatment 从 v4 的 `0.689039` 提升到 `0.764934`。
