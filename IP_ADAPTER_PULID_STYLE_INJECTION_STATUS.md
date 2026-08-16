# IP-Adapter + 3D Trajectory Residual 当前状态

> 2026-08-16 更新：reference self-attention 策略及其实验入口已经删除。当前只保留原始照片 IP-Adapter baseline 与调色后的局部 3D trajectory residual。

## 当前策略

两条分支共享相同的 SDXL 初始噪声、prompt、seed、step-30 latent，以及原始照片全局 IP-Adapter 条件：

| 分支 | 原始照片 IP-Adapter | 3D trajectory residual |
|---|---|---|
| `ip_adapter_baseline` | 是 | 否 |
| `ip_adapter_plus_pulid_style_residual` | 是 | 是 |

局部更新为：

```text
target_next += 0.4 * face_mask * (reference_next - target_next)
```

`reference_next` 来自调色后的纯 3D reference VAE `x0`，使用固定 reference noise 加噪到相同 DDIM next timestep；最后一步使用干净 reference `x0`。注入区间固定为 step 30～49。

调色使用 target/reference 纯皮肤交集估计低频 RGB gain，但将 gain 应用到完整 inner-face。最终注入 mask 是双方 BiSeNet inner-face 保守核心的交集，排除发际线、上额头、太阳穴、外脸颊、下巴边缘和耳朵。

## 小角度最终结果

输出目录：`experiment_output/ip_adapter_small_yaw_harmonized_calibrated_04/`

| 分支 | 原始照片 cosine | 连续 3D cosine | 最终 yaw |
|---|---:|---:|---:|
| IP-Adapter baseline | 0.316020 | 0.314810 | -2.7397° |
| + trajectory residual | 0.713831 | 0.820282 | -2.6143° |

- step-30 yaw：`-2.1247°`
- 注入：20/20 步成功且全部 finite
- 姿态标定：未应用，原因是 `pose_outside_calibrated_profiles`
- 权重：固定 `0.4`

## 代码与结果入口

- 实验：`multishot/ip_adapter_pulid_style_injection_experiment.py`
- 公共评估工具：`multishot/ip_adapter_experiment_utils.py`
- 后端：`multishot/diffusion_backend.py` 的 `trajectory_residual` 模式
- 对照图：`experiment_output/ip_adapter_small_yaw_harmonized_calibrated_04/comparison.jpg`
- 结构化指标：`experiment_output/ip_adapter_small_yaw_harmonized_calibrated_04/result.json`
- 最终图：`experiment_output/ip_adapter_small_yaw_harmonized_calibrated_04/branches/`
- 每步日志：`experiment_output/ip_adapter_small_yaw_harmonized_calibrated_04/logs/`

完整复现命令与四组结果见 `LLM_HANDOFF.md`。
