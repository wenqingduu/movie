# IP-Adapter 移植 PuLID 式 3D Trajectory Residual 实验

> 2026-08-16 最终更新：当前小/大角度结果已经加入纯 3D 光照匹配、FaceLift 姿态标定自动接入和 PuLID 同款 BiSeNet 保守语义交集 mask。本文以下内容保留为历史记录；最新权威结果与入口请以 `HARMONIZED_3D_INJECTION_COMPARISON_STATUS.md` 为准。

> 给后续接手本仓库的大模型：本实验已经真实完成。目标不是把 IP-Adapter 改造成 PuLID 原生 IDFormer，而是把此前 PuLID-FLUX 3D 实验中直观有效的“同 timestep reference trajectory residual + 目标脸 mask”策略移植到 SDXL + IP-Adapter。

大角度复验也已完成；结论、姿态诊断和结果入口见 `IP_ADAPTER_HIGH_YAW_PULID_STYLE_EXPERIMENT_STATUS.md`。

## 结论

这个移植方向有效，而且明显强于当前 masked mutual self-attention：

- IP-Adapter baseline 对原始照片/连续 3D render 的 InsightFace cosine 为 `0.316020/0.314810`。
- masked self-attention 为 `0.309894/0.317285`，身份基本没有变化。
- PuLID 式 trajectory residual 达到 `0.739416/0.893245`，五官身份明显向原始人物和 3D render 靠近。

人工检查同样确认 residual 分支不只是改变皮肤纹理，眉眼、鼻、嘴和脸型都发生了明确身份迁移。但 λ=`0.4` 下仍有额头/发际线、肤色、光照和材质域不连续，属于“身份注入有效，视觉融合仍未完成”，不能作为最终生产参数。

## 公平对照

三组共享完全相同的 SDXL 初始噪声、prompt、seed 和 step-30 latent。所有分支始终用原始照片作为全局 IP-Adapter 条件，并关闭 dynamic 3D IP-Adapter，唯一变量是局部 3D 注入算子：

| 分支 | 原始照片 IP-Adapter | 3D self-attention | 3D trajectory residual |
|---|---|---|---|
| `ip_adapter_baseline` | 是 | 否 | 否 |
| `ip_adapter_plus_self_attention` | 是 | 是 | 否 |
| `ip_adapter_plus_pulid_style_residual` | 是 | 否 | 是 |

- 模型：SDXL Base 1.0 + `ip-adapter_sdxl.safetensors`
- 分辨率：`1024×1024`
- Seed：`42`
- 总步数：`50`
- 分叉/注入区间：step `30～49`
- 注入 lambda：`0.4`
- self-attention 有效强度：`0.4 × 0.85 = 0.34`
- trajectory residual 有效强度：`0.4`
- 参考布局：`match_target_scale`，参考脸/目标脸大小比例约 `[0.9571, 1.0008]`
- 目标 mask：复用当前 PuLID 保守 mask 的几何 facial-core 参数；未加载 BiSeNet 语义交集，这是本轮明确记录的实现差异。

## 移植公式

每个 scheduler step 先正常得到 `target_next`，再执行：

```text
target_next += strength * face_mask * (reference_next - target_next)
```

其中 `reference_next` 是尺度和位置已经匹配目标脸的 3D reference VAE `x0`，使用固定 reference noise 加噪到同一个 DDIM next timestep；最后一步使用干净 reference `x0`。这与 PuLID-FLUX 实验的更新形式一致，但 reference trajectory 的构造适配成了 SDXL scheduler，而不是 FLUX ODE 轨迹。

## 定量结果

| 分支 | 原始照片 cosine | 连续 3D render cosine | 全图 RGB MAE vs baseline | 目标脸 bbox RGB MAE vs baseline |
|---|---:|---:|---:|---:|
| IP-Adapter baseline | 0.316020 | 0.314810 | 0.000000 | 0.000000 |
| IP-Adapter + self-attention | 0.309894 | 0.317285 | 1.455034 | 2.963926 |
| IP-Adapter + PuLID-style residual | 0.739416 | 0.893245 | 9.465489 | 24.080063 |

residual 分支相对 baseline：

- 原始照片身份 cosine 增加 `0.423396`。
- 连续 3D render cosine 增加 `0.578435`。
- 目标脸 bbox 的变化显著大于全图，说明 mask 起到了空间约束作用。
- 20/20 个注入步骤成功，20/20 均为 finite。
- latent mask 覆盖 `3295/16384` 个位置，约 `20.11%`。
- residual norm 从 step 30 的 `44.659584` 衰减到 step 49 的 `4.056949`。

最终 PNG SHA256：

- baseline：`d2de3985100c7b25e57f4088609b61dcd40414787301af4b408b2e92c96db720`
- self-attention：`1d595a552b3c157db055505ba3cf7b4157163de8e1d4a4c70fd9df3a1b2ec108`
- PuLID-style residual：`bec39d5b458656174209d4284a1d31bfcb613abc34fcb44ff94806c99c7d5d2c`

## 入口

- 可执行脚本：`multishot/ip_adapter_pulid_style_injection_experiment.py`
- 后端实现：`multishot/diffusion_backend.py` 中的 `trajectory_residual` 模式
- 对照拼图：`experiment_output/ip_adapter_pulid_style_comparison/comparison.jpg`
- 结构化结果：`experiment_output/ip_adapter_pulid_style_comparison/result.json`
- 最终图：`experiment_output/ip_adapter_pulid_style_comparison/branches/`
- 每步日志：`experiment_output/ip_adapter_pulid_style_comparison/logs/`
- 共享 step-30 x0 与保守 mask：`experiment_output/ip_adapter_pulid_style_comparison/shared/`

## 本地复现

```bash
cd /root/autodl-tmp/movie
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD"

.venv/bin/python -m multishot.ip_adapter_pulid_style_injection_experiment
```

## 下一步

1. 优先测试 λ=`0.20/0.30`，并保持相同 baseline；目标是在保留身份增益的同时减轻贴脸和域差。
2. 加入与 PuLID 当前版本完全一致的 BiSeNet semantic mask intersection，进一步排除发际线和高风险边缘。
3. 做 reference 的颜色、曝光和低频统计匹配，再编码 VAE x0。
4. 测试只在中间窗口注入或随 timestep 衰减，而不是固定 λ 持续到最后一步。
