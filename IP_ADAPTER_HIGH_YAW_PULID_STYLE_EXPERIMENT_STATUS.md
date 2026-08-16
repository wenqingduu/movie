# IP-Adapter + PuLID 式 3D Residual 大角度实验

> 给后续接手本仓库的大模型：本实验已真实完成。原始人物照片始终作为全局 IP-Adapter 条件，3D 人脸只作为 step 30～49 的局部 trajectory residual 插件；不要把结果误解为“仅使用 3D 图生成”。

## 结论

大角度下，PuLID 式 3D trajectory residual 在 IP-Adapter 基线上仍能显著注入身份，并保持目标脸朝向：

- step-30 目标 yaw：`+54.6798°`
- 按该目标 pose 动态生成的 FaceLift Gaussian render yaw：`+59.7947°`
- IP-Adapter baseline 最终 yaw：`+54.5110°`
- residual 分支最终 yaw：`+57.5780°`
- 对原始照片的 InsightFace cosine：`0.059616 → 0.593941`
- 对动态 3D render 的 cosine：`0.069173 → 0.900819`

人工检查确认 residual 分支的眉眼、鼻、嘴和脸型都明显向参考人物移动，不只是皮肤纹理变化；并且没有像错误方向参考组那样把侧脸翻转。但是 λ=`0.4` 仍把 FaceLift 的偏亮、偏粉材质带入目标脸，pitch/roll 也被参考 render 额外拉动，因此结论仍是“身份插件有效，视觉融合与极端姿态标定尚未完成”。

## 公平对照

三个最终分支共享相同的 SDXL 初始噪声、prompt、seed、step-30 latent 和原始照片 IP-Adapter 条件。唯一变量是 step 30～49 的局部 3D 算子：

| 分支 | 原始照片全局 IP-Adapter | 3D self-attention | 3D trajectory residual |
|---|---|---|---|
| `ip_adapter_baseline` | 是 | 否 | 否 |
| `ip_adapter_plus_self_attention` | 是 | 是 | 否 |
| `ip_adapter_plus_pulid_style_residual` | 是 | 否 | 是 |

- 模型：SDXL Base 1.0 + `ip-adapter_sdxl.safetensors`
- 分辨率：`1024×1024`
- Seed：`20260818`
- 总步数：`50`
- 分叉/注入区间：step `30～49`
- 原始照片 IP-Adapter scale：`0.3`
- trajectory residual λ：`0.4`
- self-attention 有效强度：`0.4 × 0.85 = 0.34`
- yaw gate：绝对 yaw `25°～55°`
- 3D 参考：从现有 FaceLift `gaussians.ply` 按 SDXL step-30 的实际 pitch/yaw/roll 动态连续渲染
- 参考布局：`match_target_scale`，参考脸/目标脸比例约 `[0.9514, 0.9992]`
- Mask：PuLID 保守几何 facial-core；本轮仍未加入 BiSeNet semantic intersection

## 定量结果

| 分支 | 原始照片 cosine | 动态 3D render cosine | 最终 pitch / yaw / roll | 全图 RGB MAE vs baseline | 目标脸 bbox RGB MAE vs baseline |
|---|---:|---:|---|---:|---:|
| IP-Adapter baseline | 0.059616 | 0.069173 | `14.1808 / 54.5110 / 11.0857` | 0.000000 | 0.000000 |
| + masked self-attention | 0.094987 | 0.094468 | `14.1638 / 54.4053 / 11.0109` | 0.562638 | 2.937699 |
| + PuLID-style residual | 0.593941 | 0.900819 | `24.5662 / 57.5780 / 29.0267` | 3.526088 | 27.157984 |

Residual 相对 baseline：

- 原始照片身份 cosine 增加 `0.534325`。
- 3D render cosine 增加 `0.831646`。
- 20/20 个注入步骤成功且全部 finite。
- latent mask 覆盖 `1094/16384` 个位置，约 `6.68%`。
- residual norm 从 step 30 的 `23.301918` 衰减到 step 49 的 `2.237650`。
- 最终 yaw 与 target 同号，差约 `2.90°`；但 pitch 和 roll 明显向 3D render 偏移。

## 两个必要的诊断组

### IP-Adapter scale 0.6 姿态失败

保持普通实验的 IP-Adapter scale `0.6` 时，即使使用严格侧脸 prompt，step-30 绝对 yaw 也只有 `6.4612°`，没有达到 `25°` 下限，因此 yaw gate 在分叉前中止。本轮把 scale 降到 `0.3` 后才得到 `54.6798°`。原始参考图条件没有关闭，只是降低了它对正脸姿态的压制。

失败候选保存在：

`experiment_output/ip_adapter_high_yaw_probe_scale_06_seed_20260818/`

### 直接复用 PuLID 缓存 PNG 的方向错误

第一次完整计算直接复用了 PuLID-FLUX 大 yaw 组的缓存侧脸 PNG。SDXL step-30 是 `+54.6798°`，该缓存 render 被 InsightFace 测为 `-54.7166°`，方向相反；residual 最终把脸翻到 `-52.3109°`。这证明 3D 插件不能只按“绝对大角度”选择缓存图，必须根据每次生成的带符号 pose 动态渲染 Gaussian。

错误方向诊断组保存在：

`experiment_output/ip_adapter_high_yaw_mismatched_reference_scale_03_seed_20260818/`

## 代码与结果入口

- 实现：`multishot/ip_adapter_pulid_style_injection_experiment.py`
- 正确姿态匹配组：`experiment_output/ip_adapter_pulid_style_high_yaw_44_comparison/`
- 对照拼图：`experiment_output/ip_adapter_pulid_style_high_yaw_44_comparison/comparison.jpg`
- 结构化结果：`experiment_output/ip_adapter_pulid_style_high_yaw_44_comparison/result.json`
- 三张最终图：`experiment_output/ip_adapter_pulid_style_high_yaw_44_comparison/branches/`
- 动态 Gaussian render 与尺度匹配布局：`experiment_output/ip_adapter_pulid_style_high_yaw_44_comparison/input/`
- step-30 x0、yaw gate 和目标 mask：`experiment_output/ip_adapter_pulid_style_high_yaw_44_comparison/shared/`
- 每步注入日志：`experiment_output/ip_adapter_pulid_style_high_yaw_44_comparison/logs/`

## 本地复现

```bash
cd /root/autodl-tmp/movie
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD"

.venv/bin/python -m multishot.ip_adapter_pulid_style_injection_experiment \
  --reference experiment_assets/pulid_reference.jpg \
  --gaussian-model experiment_output/pulid_flux_conservative_mask_04/input/facelift/facelift_raw/input/gaussians.ply \
  --output experiment_output/ip_adapter_pulid_style_high_yaw_44_comparison \
  --seed 20260818 \
  --ip-adapter-scale 0.3 \
  --injection-lambda 0.4 \
  --min-abs-yaw 25 \
  --max-abs-yaw 55 \
  --prompt 'strict right-facing side profile portrait of the same man, face looking to frame right, only one eye visible, one ear visible, clear nose silhouette, far half of face hidden, no frontal face, cinematic warm neon rainy night street, photorealistic natural skin, medium close-up'
```

## 下一步

1. 在相同 seed、prompt 和 baseline 下测试 λ=`0.2/0.3`，降低偏亮贴脸感。
2. 将 PuLID 当前的 BiSeNet semantic intersection 移植到 SDXL mask，排除更多发际线和轮廓风险区。
3. 校准极端 yaw 下 FaceLift camera pose 与渲染结果的 InsightFace pitch/roll 偏差；本轮请求 pose 为 `13.4097/54.6798/10.6905`，render 检测为 `24.8189/59.7947/32.7208`。
4. 对动态 3D render 做目标场景颜色、曝光和低频照明匹配后再编码 reference x0。
5. 保持原始照片 IP-Adapter 为基础条件；3D 路径继续作为可开关插件，不要替换基础身份条件。
