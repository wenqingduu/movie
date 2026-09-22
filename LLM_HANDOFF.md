# 3D 人脸一致性注入项目交接文档

> 最后更新：2026-09-22。仓库根目录：`/root/autodl-tmp/movie`。
>
> 当前只以 **v7 color-safe 注入策略**为有效实现。旧 v4/v5 调色实验已从项目目录移走；不要恢复 self-attention、旧像素合成调色、geometric-only mask 或旧实验参数。

## 1. 接手时先做什么

1. 保留当前未提交工作树，不要 reset、checkout 或覆盖。
2. 阅读本文和 `EVALUATION_PLAN.md`；其他旧状态文档只作历史背景，不是当前运行入口。
3. 核对 GPU、模型路径和 Python 环境。
4. 新实验必须继续使用 v7 color-safe core，并分别报告首帧、实际注入镜头和完整 episode 指标。

当前 Git 状态：

- `HEAD` / `origin/main`：`9f1474db34f6d6bf87104c03f403aabab16872f6`
- 工作树包含尚未提交的 IP-Adapter v7 批处理实现、矩形画布支持和本轮文档更新。
- 主要未提交文件：
  - `multishot/ip_adapter_experiment_utils.py`
  - `multishot/ip_adapter_pulid_style_injection_experiment.py`
  - `multishot/mcp_asset_server.py`
  - `pretest/prepare_entitybench_ip_adapter_pairs.py`
  - `EVALUATION_PLAN.md`
  - `LLM_HANDOFF.md`

## 2. 当前任务定位

当前研究对象是一个首帧/关键帧身份增强插件：

```text
文本与身份资产
→ 基础图像模型生成 Control 首帧
→ v7 局部 3D 身份注入生成 Treatment 首帧
→ 同一个 Wan2.2 模型生成成对视频
→ 以原始身份参考为锚点做逐帧评估
```

现阶段不是完整的一句话多镜头智能体，也没有接入跨镜头历史记忆。EntityBench 用于提供有序 episode、实体定义和逐镜头提示词；Wan2.2 对每个镜头独立首帧驱动。

已验证两条基础首帧路线：

- FLUX.1-dev + PuLID-FLUX；
- SDXL base 1.0 + IP-Adapter。

## 3. 当前唯一有效的 v7 算法

### 3.1 姿态与 3D 参考

```text
目标去噪轨迹的 pred_x0
→ InsightFace 检测目标脸和 pitch/yaw/roll
→ 读取该人物 Gaussian 自己的 pose calibration
→ FaceLift Gaussian 连续姿态渲染
→ 按目标 bbox 等比例放置纯 3D 脸
```

每个人物使用自己的：

- 原始身份参考图；
- `gaussians.ply`；
- `<gaussians.ply>.pose_calibration.json`。

不能复用其他人物的标定参数。

### 3.2 v7 纯 3D 调色

策略名：`target_low_frequency_log_rgb_v7_nested_color_safe_core`。

```text
pred_x0 目标脸 + 纯 3D reference
→ 在双方皮肤语义交集估计低频 log-RGB gain
→ 将 gain 应用到较大的 inner-face 调色区
→ 从调色区向内留安全边距
→ 安全区与中央五官身份核心相交
→ 得到最终注入 mask
```

必须保持：

- `pred_x0` 只用于估计光影和色调，不把目标像素混入 3D reference；
- 调色参考仍是纯 3D 脸；
- 调色区覆盖鼻子和五官边缘；
- 不把头发、耳朵、耳饰和外脸轮廓纳入身份核心；
- 实际注入区域必须完全落在可靠调色区内部。

### 3.3 v7 注入 mask

策略名：`connected_identity_feature_core_v7_color_safe`。

最终核心排除：

- 发际线和上额头；
- 太阳穴；
- 外脸颊；
- 下巴边缘；
- 耳朵和耳饰。

PuLID-FLUX 还会把像素 mask 变换到 packed latent/token 空间；IP-Adapter 使用同一语义核心，但走 SDXL VAE latent。

### 3.4 trajectory residual

Control 与 Treatment 共享 prompt、seed、基础身份条件和分叉前状态。Treatment 在每个允许步骤执行：

```text
target_next = scheduler_step(target_state)
reference_next = 同 timestep 的带噪 3D reference latent
residual = reference_next - target_next
target_next += mask * strength * residual
```

当前约束：

- 基准强度 `0.4`；
- 默认从 step 30 分叉；
- 最多连续注入 12 步；
- 小脸会降低 strength 并提前停止；
- 人脸检测不可靠或脸高低于实用阈值时，Treatment 直接复用 Control；
- 不使用 self-attention 特征替换。

### 3.5 IP-Adapter Control 的定义

单人物 Control 已经是条件生成：

```text
官方文本 prompt
+ 原始身份参考图（全局 IP-Adapter condition，当前 scale 0.2）
→ SDXL Control
```

Treatment 在完全相同的全局 IP 条件上增加 v7 局部 3D residual。因此 IP-Adapter 实验测的是插件的增量收益，不是“无身份条件 vs 有身份条件”。

当前没有验证 IP-Adapter 多身份分别绑定多张目标脸。多人 IP 镜头使用 text-only SDXL Control 并让 Treatment 复用；这不代表 PuLID-FLUX 多人局部注入路线不存在。

## 4. 权威代码入口

- PuLID-FLUX 单人 v7：`multishot/pulid_flux_inner_face_experiment.py`
- PuLID-FLUX 多人原型：`multishot/pulid_flux_multi_face_experiment.py`
- IP-Adapter 单人 v7：`multishot/ip_adapter_pulid_style_injection_experiment.py`
- IP-Adapter 公共工具：`multishot/ip_adapter_experiment_utils.py`
- SDXL/IP-Adapter 后端：`multishot/diffusion_backend.py`
- FaceLift 渲染与资产接入：`multishot/mcp_asset_server.py`
- Gaussian 姿态标定：`multishot/facelift_pose_calibration.py`
- EntityBench IP 成对首帧：`pretest/prepare_entitybench_ip_adapter_pairs.py`
- Wan manifest 批量运行：`pretest/run_wan22_i2v_manifest.py`
- 单人物视频逐帧身份评估：`pretest/evaluate_video_identity.py`

## 5. 环境与模型路径

主要环境：

- 图像、注入与评估：`/root/autodl-tmp/movie/.venv`
- Wan2.2：`/root/autodl-tmp/wan22-venv`
- GPU InsightFace 额外包：`/root/autodl-tmp/ort-cuda11`

模型：

- SDXL：`models/diffusion/sdxl-base-1.0`
- IP-Adapter：`models/ip_adapter/h94-IP-Adapter`
- Wan2.2-TI2V-5B：`models/video/Wan2.2-TI2V-5B`
- PuLID / InsightFace：`third_party/PuLID`
- Wan 源码：`/root/autodl-tmp/Wan2.2`
- 本轮 Wan 源码提交：`42bf4cfaa384bc21833865abc2f9e6c0e67233dc`

Wan2.2-5B 的 offload 常驻显存低，但单任务峰值约 allocated `22.2 GiB`、reserved `32.0 GiB`。48GB 单卡连续批处理默认只开一个 Wan 进程；双进程在任务切换时已经发生过 OOM。

## 6. EntityBench 数据与当前资产

Benchmark：

- 数据根：`benchmarks/entitybench/`
- 当前 episode：`benchmarks/entitybench/data/scripts/00053051-5f7e-314f-85e0-517ec18f3b08__run719__i35_j44__T120.json`
- 共 12 个官方有序镜头：8 个单人物、4 个多人物。

当前身份与 3D 资产：

`outputs/entitybench_wan22_smoke/episode_00053051/assets/`

已有完整资产：Viktor、Roman。Julian、Silas、Leo 尚无完整身份参考和 Gaussian，因此不能用 Viktor/Roman 冒充。

旧调色实验已从 `experiment_output` 移走；该目录现在只保留：

`experiment_output/facelift_pose_calibration/`

## 7. 当前有效结果

### 7.1 PuLID-FLUX v7 完整 episode

权威目录：

`outputs/entitybench_wan22_colornested_v7_s04_12/episode_00053051/`

主 Treatment 使用 v7 color-safe 策略。完整 Control/Treatment：

- `episode_sequences/episode_control.mp4`
- `episode_sequences/episode_treatment_color_safe.mp4`

单人物视频评估中，实际注入且可严格成对评估的 4 个镜头：

| 指标 | Control | Treatment | Δ |
|---|---:|---:|---:|
| 视频可检测帧 mean 宏平均 | 0.467014 | 0.523078 | +0.056064 |
| 视频首帧宏平均 | 0.578188 | 0.644170 | +0.065982 |

4/4 的视频 mean 为正增益。`4:2` 是双人物 v7 镜头，不包含在单人物评估器的宏平均中。

### 7.2 SDXL/IP-Adapter v7 完整 episode

权威目录：

`outputs/entitybench_wan22_ip_adapter_v7_s04_12/episode_00053051/`

完整视频：

- `episode_sequences/episode_control.mp4`
- `episode_sequences/episode_treatment.mp4`

12 个镜头中实际注入 `2:1`、`4:4`、`4:5`、`5:2`、`6:1`；其余 7 个安全跳过或多人复用。

| 范围 | Control | Treatment | Δ |
|---|---:|---:|---:|
| 5 个应用镜头首帧宏平均 | 0.118361 | 0.282797 | +0.164436 |
| 5 个应用镜头视频 mean 宏平均 | 0.104617 | 0.219083 | +0.114466 |
| 完整 12-shot 首帧归一化增益 | — | — | +0.068515 |
| 完整 12-shot 视频归一化增益 | — | — | +0.047694 |

结构化入口：

- `ip_adapter_first_frame_report.json`
- `wan_report_full_episode.json`
- `video_identity_report_single_character.json`
- `episode_identity_summary.json`
- `video_identity_visuals/`

重要质量结论：

- `2:1` 身份增益稳定传播；
- `4:5` 首帧提升明显，但视频 mean 仅 `+0.004987`；
- `5:2` 是低信号远景小脸；
- `6:1` cosine 提升大，但写实脸与插画场景冲突，属于视觉质量失败；
- 不能只按 cosine 宣称全部样本成功。

### 7.3 PuLID-FLUX 双人物 v7

唯一有效的自动角色映射输出：

`outputs/entitybench_wan22_multiface_v7/episode_00053051/official_shot_4_2_autoassign/`

正确映射：左脸 Viktor，右脸 Roman。v7 color-safe 首帧逐角色结果：

| 角色 | Control | v7 Treatment | Δ |
|---|---:|---:|---:|
| Viktor | 0.018376 | 0.030058 | +0.011681 |
| Roman | 0.093390 | 0.156139 | +0.062749 |
| 宏平均 | 0.055883 | 0.093098 | +0.037215 |

当前映射方向正确，但绝对匹配分数偏低。尚未实现可靠的绝对分数、候选间隔和歧义 gate，也尚未完成多人视频逐角色评估。

## 8. IP-Adapter v7 复现命令

先生成或续跑成对首帧：

```bash
cd /root/autodl-tmp/movie
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD"

.venv/bin/python pretest/prepare_entitybench_ip_adapter_pairs.py \
  --episode benchmarks/entitybench/data/scripts/00053051-5f7e-314f-85e0-517ec18f3b08__run719__i35_j44__T120.json \
  --asset-root outputs/entitybench_wan22_smoke/episode_00053051/assets \
  --output-dir outputs/entitybench_wan22_ip_adapter_v7_s04_12/episode_00053051 \
  --continue-on-error
```

再运行 Wan：

```bash
export PYTHONPATH="/root/autodl-tmp/wan22-venv/lib/python3.10/site-packages:$PWD"
export OMP_NUM_THREADS=8

/root/autodl-tmp/wan22-venv/bin/python pretest/run_wan22_i2v_manifest.py \
  --manifest outputs/entitybench_wan22_ip_adapter_v7_s04_12/episode_00053051/wan_manifest_full_episode.json \
  --report outputs/entitybench_wan22_ip_adapter_v7_s04_12/episode_00053051/wan_report_full_episode.json \
  --continue-on-error
```

评估单人物视频：

```bash
export PYTHONPATH="/root/autodl-tmp/ort-cuda11:$PWD"
export MULTISHOT_INSIGHTFACE_PROVIDERS="CUDAExecutionProvider,CPUExecutionProvider"
export OMP_NUM_THREADS=8

.venv/bin/python pretest/evaluate_video_identity.py \
  --manifest outputs/entitybench_wan22_ip_adapter_v7_s04_12/episode_00053051/wan_manifest_single_character.json \
  --output-json outputs/entitybench_wan22_ip_adapter_v7_s04_12/episode_00053051/video_identity_report_single_character.json \
  --visual-dir outputs/entitybench_wan22_ip_adapter_v7_s04_12/episode_00053051/video_identity_visuals
```

## 9. 评测口径

必须同时报告：

1. 首帧应用镜头宏平均；
2. 视频应用镜头宏平均；
3. 完整 episode 归一化成对增益；
4. 插件触发覆盖率与人脸检测覆盖率；
5. 明显风格、边缘、五官重绘和时序失败。

身份指标始终是：

```text
cosine(原始身份参考 embedding, 当前生成帧目标脸 embedding)
```

不能只计算生成帧之间的相似度，也不能把“稳定地生成成另一个人”算作成功。

完整 episode 归一化增益将安全跳过和严格复用镜头按成对差值 0 计入；它不能与旧单张精挑首帧的增益直接比较。

## 10. 当前未完成事项

优先级从高到低：

1. 对 IP-Adapter 当前镜头做 v7 `12 / 16 / 20` 注入步数消融，确认保守上限与身份增益/伪影的权衡。
2. 加入风格/OOD gate，避免 `6:1` 这类写实脸注入插画场景。
3. 分析 `4:5` 的 Wan 身份传播衰减；仅增强首帧未必足够，需要关键帧或视频级身份条件。
4. 实现多人自动映射置信度 gate。
5. 实现多人视频逐帧一对一角色匹配评估器。
6. 再扩展到更多完整 EntityBench episode 和随机种子。

## 11. 禁止回退的路径

- self-attention 特征注入；
- 将调色后的 3D 脸与 `pred_x0` 做像素合成再编码；
- 灰色画布旧参考；
- geometric-only 注入 mask；
- 按 prompt 顺序或输入参数顺序绑定多人角色；
- 用单人物评估器评价多人镜头；
- 用高 cosine 掩盖明显的风格、构图或五官质量失败。

## 12. 清理状态

旧 v4/v5 调色与回归结果、旧小脸总览以及已确认错配的首轮多人结果已移出项目。可恢复位置：

`/root/.local/share/Trash/files/movie_pre_v7_results_20260922/`

项目中当前应保留：

- `experiment_output/facelift_pose_calibration/`
- `outputs/entitybench_wan22_colornested_v7_s04_12/`
- `outputs/entitybench_wan22_ip_adapter_v7_s04_12/`
- `outputs/entitybench_wan22_multiface_v7/.../official_shot_4_2_autoassign/`
- v7 依赖的 `outputs/entitybench_wan22_smoke/episode_00053051/assets/`
