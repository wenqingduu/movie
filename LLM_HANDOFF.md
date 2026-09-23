# 3D 人脸一致性注入项目交接文档

> 最后更新：2026-09-23。仓库根目录：`/root/autodl-tmp/movie`。
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

当前没有验证 IP-Adapter 多身份分别绑定多张目标脸。PuLID-FLUX 曾试验多人局部注入，但自动角色映射不可靠；两条路线的多人镜头均已退出当前正式评测。

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

旧 pilot 仅有 Viktor、Roman 等主要角色资产。2026-09-22 已完成正式补建：run719
含 Viktor、Julian、Silas、Roman、Leo；run1517 含 Tae-hwan、Hyeon-sik、Jin-woo；
run893 含 Chloe、Julian、Leo、Eleanor。三个 episode 共 12/12 个角色均已核对固定参考图、
对应 FaceLift Gaussian 和独立姿态标定，FaceLift 输入哈希与参考图一致。资产构建必须扫描
完整 `entity_schedule`，不再只选择至少有一次单独出镜的角色，且绝不能跨 episode 按同名
人物复用资产。

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

### 7.3 PuLID-FLUX 双人物 v7（历史结果，不再纳入评测）

旧自动角色映射试验已随多人路线退出当前范围，并于 2026-09-23 移至：

`/root/.local/share/Trash/files/movie_outputs_obsolete_20260923/outputs/entitybench_wan22_multiface_v7/`

不要从该历史结果推导当前算法结论；PuLID-FLUX 与当前 IP-Adapter 只评单角色镜头。

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

1. 增加单人姿态可靠性 gate：保留正常侧脸，只拦截标定外推失败、渲染回检误差过大和退化 mask。
2. 对 IP-Adapter 当前镜头做 v7 `12 / 16 / 20` 注入步数消融，确认保守上限与身份增益/伪影的权衡。
3. 加入风格/OOD gate，避免 `6:1` 这类写实脸注入插画场景。
4. 分析 `4:5` 的 Wan 身份传播衰减；仅增强首帧未必足够，需要关键帧或视频级身份条件。
5. 再扩展到更多单角色 EntityBench 镜头和随机种子。
6. 多人能力改由原生多参考图像模型单独立项，不继续扩展当前 PuLID-FLUX/IP-Adapter 多人路径。

## 11. 禁止回退的路径

- self-attention 特征注入；
- 将调色后的 3D 脸与 `pred_x0` 做像素合成再编码；
- 灰色画布旧参考；
- geometric-only 注入 mask；
- 按 prompt 顺序或输入参数顺序绑定多人角色；
- 用单人物评估器评价多人镜头；
- 用高 cosine 掩盖明显的风格、构图或五官质量失败。

## 12. 清理状态

旧 v4/v5 调色与回归结果、旧小脸总览以及已确认错配的首轮多人结果已移出项目。2026-09-23 又清理了旧 v6 mask、旧 smoke 生成物、run719 重复评测、多人试验结果及可重建 evaluator `.work` 缓存。可恢复位置：

`/root/.local/share/Trash/files/movie_pre_v7_results_20260922/`

`/root/.local/share/Trash/files/movie_outputs_obsolete_20260923/`

项目中当前应保留：

- `experiment_output/facelift_pose_calibration/`
- `outputs/entitybench_wan22_colornested_v7_s04_12/`
- `outputs/entitybench_wan22_pulid_v7_pilot3/`
- `outputs/entitybench_wan22_ip_adapter_v7_s04_12/`
- `outputs/entitybench_wan22_ip_adapter_v7_pilot3/`
- `outputs/entitybench_reference_dino_pilot3/`
- `outputs/entitybench_official_eval_pilot3/` 中最终 JSON/Markdown 报告（`.work` 已清理）
- `outputs/entitybench_pilot3/*/assets/`
- v7 依赖的 `outputs/entitybench_wan22_smoke/episode_00053051/assets/`

## 13. EntityBench 非 API 重评状态（2026-09-22）

run719 的 Control vs v7 已完成部分官方重评，覆盖 PuLID-FLUX 与 IP-Adapter 两条首帧路线。为了不调用外部大模型 API，`benchmarks/entitybench/eval/evaluate_benchmark.py` 新增了 `--skip_vlm`：保留 VBench、GroundingDINO/CLIP presence、DINOv2 跨镜头一致性与镜头边界连续性，跳过 VLM 属性/动作评分和 LLM judge。

run719 单 episode 的旧重复评测目录已在 2026-09-23 清理；三 episode 历史汇总保留在：

- `outputs/entitybench_official_eval_pilot3/control_v7_summary.md`
- `outputs/entitybench_official_eval_pilot3/control_v7_summary.json`

该历史官方格式汇总包含完整 episode 中复用 Control 的非注入镜头。当前算法结论应优先使用
`outputs/entitybench_reference_dino_pilot3/` 和各路线的 `video_identity_report_single_character.json`，
只统计单角色且实际触发插件的成对样本。

| 路线 | `cs_face` Δ | 边界连续性 Δ | subject consistency Δ | aesthetic Δ |
|---|---:|---:|---:|---:|
| PuLID-FLUX | +0.0022 | +0.0099 | +0.0009 | +0.0022 |
| IP-Adapter | +0.0170 | -0.0042 | -0.0007 | -0.0014 |

这轮说明 v7 对 EntityBench 的跨镜头自一致性和基础视频质量总体近似中性；不能据一个 episode 判断显著改善或退化。`cs_face` 是同一角色不同生成镜头之间的 DINOv2 一致性，并不以原始身份图为锚点。它与第 7 节 InsightFace 原始身份 cosine 回答不同问题，所以必须并列报告，不能相互替代。

本轮环境与本地模型：

- 独立评测环境：`/root/autodl-tmp/envs/entitybench-eval`
- VBench：`/root/autodl-tmp/models/entitybench_vbench`
- GroundingDINO：`/root/autodl-tmp/models/entitybench_groundingdino`
- BERT：`/root/autodl-tmp/models/bert-base-uncased`
- CLIP：`/root/autodl-tmp/models/clip-vit-base-patch32`
- DINOv2：`/root/autodl-tmp/models/dinov2-base`

复现链路：

1. `pretest/prepare_entitybench_official_eval.py` 从现有 generation manifest 建立 Control/v7 官方目录软链接，不复制视频；
2. evaluator 分别执行 `--pillars 1` 和 `--pillars 2,3 --skip_vlm`；
3. `pretest/summarize_entitybench_control_v7.py` 合并四组报告并计算 v7-Control。

报告 manifest 必须保留 `evaluation_scope: partial_without_vlm`。VLM/LLM 字段为 `null` 表示未运行，绝不能解释为 0；当前结果也不是完整 EntityBench 排名。

## 14. 三 episode 当前正式范围：单角色插件评测（2026-09-23）

PuLID-FLUX 与当前 SDXL IP-Adapter 接法都没有可靠的多身份空间绑定，因此不再把多人局部注入
纳入这两条路线的正式结论。旧多人输出已移到回收站；多人相关代码暂时保留，不能把它产生的
历史结果混入当前单人汇总。

当前固定流程：

```text
读取完整 episode 及官方顺序
→ 仅选择 entity_schedule 明确为单角色的镜头
→ 固定角色参考图、FaceLift Gaussian 和姿态标定
→ 生成成对 Control / v7 首帧
→ 仅通过预先固定可靠性 gate 的 Treatment 送入 Wan
→ 计算参考身份锚定的首帧与视频逐帧指标
```

资产覆盖审计仍保留：

- run719：Viktor、Julian、Silas、Roman、Leo，共 5 人；
- run1517：Tae-hwan、Hyeon-sik、Jin-woo，共 3 人；
- run893：Chloe、Julian、Leo、Eleanor，共 4 人。

当前入口：

- `pretest/prepare_entitybench_character_assets.py`：默认
  `--character-scope scheduled`，构建 episode 全角色资产；旧
  `--character-scope single-shot` 只用于复现 pilot；
- `pretest/prepare_entitybench_pulid_pairs.py`：PuLID-FLUX 单角色成对首帧；
- `pretest/prepare_entitybench_ip_adapter_pairs.py`：IP-Adapter 单角色成对首帧；
- `pretest/evaluate_reference_dino_identity.py`：参考角色图锚定的单人 DINOv2 评估。

执行状态：

- 全角色资产已完成：12/12 角色完整；虽然当前只评单人，这些资产继续保留；
- 三 episode 的 PuLID-FLUX、IP-Adapter 单人首帧、Wan 视频、InsightFace 与 DINOv2 报告已完成；
- 下一轮必须补充姿态可靠性 gate：侧脸可以注入，但 `pose_calibration.applied=false`、渲染后
  姿态回检误差过大或有效 mask 退化为狭长碎片时应预先跳过并复用 Control；
- 多人镜头不进入当前两条路线的 Control/Treatment 聚合。后续如加入原生多参考模型，另建实验根目录。
