# 调色 + 姿态标定 3D 人脸注入最终对比

> 后续大模型的完整操作入口为 `LLM_HANDOFF.md`；本文聚焦最终结果和清理状态。

截至 2026-08-16，PuLID-FLUX 与 SDXL IP-Adapter 的小角度、大角度最终实验均已完成。当前统一策略为：

1. step 30 检测目标 pitch/yaw/roll，并从同一 FaceLift Gaussian 连续渲染。
2. 高 yaw 自动使用与 `gaussians.ply` 同名的姿态标定；小 yaw 因不在 `35°～65°` profile 内而保持原始映射。
3. `pred_x0` 与 3D 参考的纯皮肤交集只用于估计低频 log-linear RGB 光照。
4. 光照增益应用到完整 inner-face，包含眉眼、鼻、嘴和唇；不混入任何 `pred_x0` 像素。
5. 最终注入 mask 使用 target/reference BiSeNet inner-face 的保守核心交集，排除发际线、太阳穴、外脸颊、下巴边缘和耳朵。
6. step 30～49 固定注入权重/残差系数 `0.4`。

## 最终结果

| 模型与角度 | Baseline 原图 cosine | 3D 注入原图 cosine | Baseline 3D cosine | 3D 注入 3D cosine | step-30 yaw | 最终 yaw |
|---|---:|---:|---:|---:|---:|---:|
| PuLID-FLUX 小角度 | 0.829754 | 0.796340 | 0.685264 | 0.889137 | +1.3447° | +2.9400° |
| PuLID-FLUX 大角度 | 0.535710 | 0.688840 | 0.462680 | 0.858792 | -43.7855° | -44.8903° |
| IP-Adapter 小角度 | 0.316020 | 0.713831 | 0.314810 | 0.820282 | -2.1247° | -2.6143° |
| IP-Adapter 大角度 | 0.059616 | 0.605702 | 0.052479 | 0.689535 | +54.6798° | +53.3364° |

四组插件分支均完成 20/20 个注入步骤，全部 finite，无 NaN/Inf。

2026-08-16 后续收缩：IP-Adapter reference self-attention 策略已从后端、实验入口和最终结果中删除。两组 IP-Adapter 对照图现在均为六格，只包含输入、调色参考、共享 step-30、baseline 和 trajectory residual。

## 姿态标定验收

- PuLID 小角度 `+1.3447°` 在 profile 范围外，使用原始 camera mapping；该缓存 render 的 meta 生成于标定字段接入前，因此没有 `pose_calibration` 字段。IP-Adapter 小角度 `-2.1247°` 明确记录为 `pose_outside_calibrated_profiles`。两者都没有错误套用高 yaw 补偿。
- PuLID 大角度应用 `negative_high_yaw_v1`：目标 yaw `-43.7855°`，renderer camera yaw 修正为 `-36.1814°`。
- IP-Adapter 大角度应用 `positive_high_yaw_v1`：目标 pose `13.4097/54.6798/10.6905°`，renderer camera pose 修正为 `9.5873/48.9789/-0.8292°`；实际 3D render 检测为 `13.1607/55.8899/10.6976°`。

## 人工检查

- PuLID 小角度：肤色已经匹配暖色街景，发际线和外轮廓没有注入；3D 材质与五官形状仍较明显，原始照片身份 cosine 比 Control 低 `0.033414`。
- PuLID 大角度：红蓝光照连续覆盖鼻子和五官边缘，身份与姿态均明显优于 Control；中心脸仍略平滑，但不再是白色面具。
- IP-Adapter 小角度：语义交集移除了几何-only mask 产生的额头发片，身份显著增强；近正脸仍能看到 FaceLift 的平滑材质和五官域差。
- IP-Adapter 大角度：标定后姿态稳定，调色与语义 mask 使边界最自然；身份增益明确，但对原始照片 cosine 低于 PuLID 大角度。由于两者模型、分辨率、yaw 符号和角度不完全一致，不能把该差值视为严格架构排名。

## 当前保留结果

- FaceLift 标定：`experiment_output/facelift_pose_calibration/`
- Gaussian 与保守基线：`experiment_output/pulid_flux_conservative_mask_04/`
- PuLID 小角度：`experiment_output/pulid_flux_small_yaw_harmonized_pure_3d_04/`
- PuLID 大角度：`experiment_output/pulid_flux_high_yaw_44_harmonized_pure_3d_04/`
- IP-Adapter 小角度：`experiment_output/ip_adapter_small_yaw_harmonized_calibrated_04/`
- IP-Adapter 大角度：`experiment_output/ip_adapter_high_yaw_harmonized_calibrated_04/`

## 清理记录

10 个已被最终方案取代的目录已移到：

`/root/.local/share/Trash/files/movie_obsolete_experiments_20260816_2/`

其中包括旧像素合成调色、未标定高 yaw、失败 yaw probe、方向错误诊断、self-attention 旧实验，以及未调色 IP-Adapter 小/大角度结果。该操作可恢复。

两组最终 IP-Adapter 目录中原有的 self-attention PNG 与逐步日志另外移到：

`/root/.local/share/Trash/files/movie_retired_self_attention_20260816/`

## 代码入口

- PuLID-FLUX：`multishot/pulid_flux_inner_face_experiment.py`
- IP-Adapter：`multishot/ip_adapter_pulid_style_injection_experiment.py`
- 姿态标定接入：`multishot/mcp_asset_server.py`
- 标定生成器：`multishot/facelift_pose_calibration.py`
