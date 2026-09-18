# IP-Adapter + 3D Trajectory Residual 大角度状态

> 2026-08-25 更新：大角度 v4 路径只包含 baseline 与 soft-context 调色、标定后的 trajectory residual。

## 当前策略

- 模型：SDXL Base 1.0 + `ip-adapter_sdxl.safetensors`
- 原始照片 IP-Adapter scale：`0.3`
- Seed：`20260818`
- 分叉/注入区间：step `30～49`
- trajectory residual 权重：固定 `0.4`
- yaw gate：绝对 yaw `25°～55°`
- 3D 参考：按 step-30 带符号 pitch/yaw/roll 从 FaceLift Gaussian 连续渲染
- 姿态标定：自动应用 `positive_high_yaw_v1`
- 调色：纯 3D 低频 RGB 光照匹配 + BiSeNet soft 调色上下文环
- mask：target/reference BiSeNet 保守 inner-face 交集

## 最终结果

输出目录：`experiment_output/ip_adapter_high_yaw_harmonized_soft_context_v4_04/`

| 分支 | 原始照片 cosine | 动态 3D cosine | 最终 pitch / yaw / roll |
|---|---:|---:|---|
| IP-Adapter baseline | 0.059616 | 0.052479 | `14.1808 / 54.5110 / 11.0857°` |
| + trajectory residual | 0.595695 | 0.698383 | `11.7775 / 53.0692 / 9.2449°` |

- step-30 pose：`13.4097 / 54.6798 / 10.6905°`
- 标定后的实际 3D render：`13.1607 / 55.8899 / 10.6976°`
- 注入：20/20 步成功且全部 finite
- 最终 yaw 与 step-30 yaw 同号，差 `1.6106°`

## 代码与结果入口

- 实验：`multishot/ip_adapter_pulid_style_injection_experiment.py`
- 后端：`multishot/diffusion_backend.py` 的 `trajectory_residual` 模式
- 姿态标定接入：`multishot/mcp_asset_server.py`
- 对照图：`experiment_output/ip_adapter_high_yaw_harmonized_soft_context_v4_04/comparison.jpg`
- 结构化指标：`experiment_output/ip_adapter_high_yaw_harmonized_soft_context_v4_04/result.json`
- 最终图：`experiment_output/ip_adapter_high_yaw_harmonized_soft_context_v4_04/branches/`
- 每步日志：`experiment_output/ip_adapter_high_yaw_harmonized_soft_context_v4_04/logs/`

完整复现命令、调色和标定说明见 `LLM_HANDOFF.md` 与 `FACELIFT_POSE_CALIBRATION_STATUS.md`。
