# FaceLift Gaussian 极端姿态标定状态

> 2026-08-16 最终更新：当前带调色与 BiSeNet 保守语义 mask 的 IP-Adapter 大角度结果位于 `experiment_output/ip_adapter_high_yaw_harmonized_calibrated_04/`；旧的仅标定、未调色输出已移到回收站。四组最终对比见 `HARMONIZED_3D_INJECTION_COMPARISON_STATUS.md`。

> 给后续接手本仓库的大模型：本标定已真实完成并接入项目连续 Gaussian 渲染路径。补偿以具体 `gaussians.ply` 为单位，只在正、负高 yaw profile 的有效范围内自动启用；正脸和范围外姿态仍使用原始映射。

## 为什么需要标定

项目先用 InsightFace 从 step-30 `pred_x0` 检测 pitch/yaw/roll，再将这三个角度转换为 FaceLift camera。原始映射在大 yaw 下保持了左右方向，但存在明显轴耦合：

- IP-Adapter step-30 目标：`13.4097 / 54.6798 / 10.6905°`
- 未补偿 3D render 检测：`24.8189 / 59.7947 / 32.7208°`

因此不能只给 pitch 或 roll 减一个固定偏移，需要同时建模三个轴之间的交叉影响。

## 标定方法

- Gaussian：当前参考人物的 `gaussians.ply`
- 回测器：与实验指标一致的 InsightFace AntelopeV2
- 渲染尺寸：`1024×1024`
- camera pitch 网格：`[-15, 0, 15]°`
- camera yaw 网格：`[-55, -47.5, -40, 40, 47.5, 55]°`
- camera roll 网格：`[-20, 0, 20]°`
- 总样本：`54`
- 成功检测：`54/54`
- 正、负高 yaw 分别拟合局部仿射逆映射：

```text
corrected_camera_pose = A × [desired_insightface_pitch, yaw, roll, 1]
```

拟合包含 pitch/yaw/roll 交叉项，不假设左右侧完全对称。

## 留出姿态验证

### IP-Adapter 正侧大 yaw

目标：`13.4097 / 54.6798 / 10.6905°`

| 版本 | render 检测 pitch / yaw / roll | 绝对误差 pitch / yaw / roll |
|---|---|---|
| 未补偿 | `24.8189 / 59.7947 / 32.7208` | `11.4092 / 5.1149 / 22.0303` |
| 标定补偿 | `13.1607 / 55.8899 / 10.6976` | `0.2490 / 1.2101 / 0.0071` |

补偿后的实际 camera pose 为：

`9.5873 / 48.9789 / -0.8292°`

### PuLID 负侧大 yaw

目标：`0.2023 / -43.7855 / -6.7964°`

补偿后的 render 检测为：

`-1.7400 / -45.3300 / -5.0237°`

绝对误差为：

`1.9423 / 1.5445 / 1.7727°`

## IP-Adapter 端到端复验

相同 seed、prompt、step-30 latent、baseline、IP-Adapter scale `0.3` 和 residual λ=`0.4` 下：

| 项目 | 未补偿 | 标定补偿 |
|---|---:|---:|
| 3D render pitch | 24.8189 | 13.1607 |
| 3D render yaw | 59.7947 | 55.8899 |
| 3D render roll | 32.7208 | 10.6976 |
| residual 最终 pitch | 24.5662 | 14.1149 |
| residual 最终 yaw | 57.5780 | 53.9328 |
| residual 最终 roll | 29.0267 | 10.3548 |
| 对原始照片 cosine | 0.593941 | 0.616340 |
| 对各自 3D render cosine | 0.900819 | 0.817205 |
| 目标脸 bbox RGB MAE vs baseline | 27.157984 | 24.560553 |

标定后的 residual 最终姿态相对 step-30 目标仅差：

`pitch 0.7052° / yaw 0.7470° / roll 0.3357°`

Baseline PNG SHA256 在两次实验中完全相同：

`1adf2f3176453131154537c047c71b8b7faef53dacebb717b66a9d0c4a440ea7`

这证明姿态改善来自 3D reference camera 补偿，而不是基础生成轨迹变化。20/20 个 residual 注入步骤均成功且 finite。

## 实现和自动启用规则

- 标定脚本：`multishot/facelift_pose_calibration.py`
- 渲染接入：`multishot/mcp_asset_server.py`
- 模型标定文件：`experiment_output/pulid_flux_conservative_mask_04/input/facelift/facelift_raw/input/gaussians.pose_calibration.json`
- 完整样本和拟合结果：`experiment_output/facelift_pose_calibration/`
- 标定后 IP-Adapter 结果：`experiment_output/ip_adapter_pulid_style_high_yaw_44_calibrated_comparison/`
- 标定前后拼图：`experiment_output/ip_adapter_pulid_style_high_yaw_44_calibrated_comparison/comparison_uncalibrated_vs_calibrated.jpg`

渲染器默认查找与模型同名的 `gaussians.pose_calibration.json`。找到覆盖目标 pose 的 profile 时自动应用，并在 render meta 中记录：

- 原始期望 pose
- 补偿后的 camera pose
- 使用的 profile
- 标定文件路径
- 拟合 RMSE

设置以下环境变量可显式关闭自动补偿：

```bash
export MULTISHOT_FACELIFT_POSE_CALIBRATION=0
```

也可以将该变量指向其他 calibration JSON。

## 当前限制

1. 这是当前人物 Gaussian 的模型级标定，不能未经验证直接复制给其他人物的 Gaussian。
2. 当前 profile 只覆盖绝对 yaw `35°～65°`；正脸和中等 yaw 不应用补偿。
3. 正侧训练网格的 pose forward-fit RMSE 为 `3.56/1.19/4.20°`，负侧为 `3.29/1.77/4.08°`；留出目标明显更准，但不能把单次验证误差当作整个区间的上界。
4. 补偿解决的是视角姿态，不解决 FaceLift 渲染的肤色、光照、材质和贴脸域差。
5. 后续应对新的角色 Gaussian 自动运行轻量标定，或建立跨人物共享标定后再验证泛化性。
