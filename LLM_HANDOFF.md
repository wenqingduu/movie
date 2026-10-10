# 3D 人脸一致性注入项目交接文档

最后更新：2026-10-10（北京时间）。仓库：`/root/autodl-tmp/movie`。

最新进展：PuLID-FLUX、SDXL/IP-Adapter 的 20 episode 单人首帧与视频评测已完成，
详见第 11、12 节。528/528 个视频任务完成，失败 0；2026-10-10 核对时 tmux 无运行会话。
本会话后续补齐了四项指标解释、动漫风格来源核查和 20 张首帧正例对比图（其中 14 张为
两批写实正例）。新窗口先读第 13 节和 [20 episode 评估说明](docs/entitybench_20ep_evaluation.md)。

本轮随后补齐了 20 episode 的 DINOv2 跨镜头人脸外观质心相似度；见第 14 节。
该指标采用论文质心公式，但复用现有脸部裁剪，不是官方 `cs_face`，不替代原有四项身份指标。

## 1. 接手顺序

1. 保留未提交工作树，不要 reset 或 checkout；前后端目录还有用户的并行改动。
2. 阅读本文和 [20 episode 评估说明](docs/entitybench_20ep_evaluation.md)；
   `EPISODE_TEST_RESULTS.md` 仍是三 episode/Qwen 结果，不能当作 20 episode 总表。执行或审计评测时使用
   `/root/.codex/skills/evaluate-face-injection/SKILL.md`（`$evaluate-face-injection`）。
3. 核对 GPU、模型路径和 Python 环境。
4. `c3c9fcccd558befb7302a046794e789c9f71f615` 是本轮 Qwen 全 episode 批处理前的基线；
   以仓库最新提交和本文为准，不要恢复已删除的旧调色、自注意力或旧多人路径。

## 2. 当前任务定义

本项目当前评测一个首帧/关键帧身份增强插件：

```text
文本 + 固定身份/3D 资产
→ 官方基础图像管线正常生成 Control，并截取进入注入点的 latent
→ 从该 latent 预测 pred_x0，完成人脸检测、角色映射、姿态标定和 v7 reference
→ Treatment 从截取的同一 latent 继续去噪
→ 仅 Treatment 接收逐角色 v7 局部 3D residual
→ 相同 Wan2.2 生成成对视频
→ 原始身份参考锚定的逐帧评测
```

EntityBench 提供完整有序 episode、实体调度和逐镜头提示词。Wan2.2 当前逐镜头独立运行，
没有跨镜头历史记忆。不能把本轮结果表述为完整的一句话多镜头智能体评测。

## 3. 当前唯一有效算法

只保留 v7 color-safe：

1. 从注入起点 `pred_x0` 检测目标人脸和 pitch/yaw/roll；
2. 读取该角色自己的 `gaussians.ply.pose_calibration.json`；
3. 从 FaceLift Gaussian 连续渲染目标姿态并做 InsightFace 姿态回检；
4. 在目标/3D 皮肤交集估计低频 log-RGB gain，并应用到纯 3D inner-face；
5. BiSeNet v7 mask 排除发际线、太阳穴、外脸颊、下巴边缘、耳朵和耳饰；
6. 将调色后的纯 3D reference 编码为 latent，在相同 timestep 做局部 trajectory residual；
7. 正常脸基准强度 `0.4`；小脸继续使用 v7 的自适应降强度和提前结束。
8. Qwen 路线在 step 12 用角色参考图与 `pred_x0` 目标脸的 InsightFace cosine
   选择窗口：`<0.40` 注入 6 步（step 12～17），`>=0.40` 只注入 step 12。

公式：

```text
target_next = scheduler_step(target_state)
reference_next = 同 timestep 的带噪 3D reference latent
target_next += mask * strength * (reference_next - target_next)
```

### 3.1 统一执行协议（不得自行改动）

PuLID-FLUX、SDXL/IP-Adapter 和 Qwen-Image-2.1 统一使用“中期 `pred_x0` +
同一中间 latent 分叉”。Qwen 的严格实现是：Control 必须完整走未修改的官方
`QwenImage21Pipeline.__call__`；callback 只读取并保存进入 step 12 的 latent，不能用手写
采样器代替 Control。Control 完成后 Treatment 才从该 latent 继续：

```text
官方 Control trajectory
  → callback 截取 shared latent
  → 注入点 pred_x0
  → 检测 / 角色映射 / pitch-yaw-roll / 3D 渲染 / 调色 / v7 mask
  → Treatment 在同 timestep 加局部 residual 后继续去噪
```

Qwen 20 步的 step 12 对应 PuLID/IP 50 步的 step 30，均为约 60% 去噪进度。
不得再用“完整生成 Control 后，依据最终 Control 准备 3D，然后从噪声重跑
Treatment”作为当前方案。该两遍式仅作为历史 pilot，不能与统一协议结果混合。

上述 `0.40` 阈值和 1/6 步二选一规则已经用户确认。除此之外仍只允许单变量消融；
不得自行加入新 mask、新权重规则、调色下限或其他 gate。

## 4. 安全回退

PuLID-FLUX/IP-Adapter 单人路线遇到以下情况时 Treatment 复用 Control。Qwen 多参考路线
改为逐角色容错：缺脸、映射失败、3D 渲染/调色/mask/latent reference 失败时只跳过该角色，
其余成功角色继续注入；只有没有任何可用角色或命中整镜头闭眼冲突 gate 时才整张复用 Control。

相关 gate 包括：

- 无可靠脸或脸高 `<24 px`；
- Gaussian 渲染失败或姿态回检不通过；
- v7 color-safe core 为空；
- 提示词明确要求闭眼，记录 `skip_reason=prompt_eye_closure_conflict`；
- 当前 PuLID-FLUX/IP-Adapter 路线的多人镜头。

“复用”意味着 Control 正常生成，Treatment 文件与 Control 完全相同；不是删除难例。

run1517 / 4:2 已验证闭眼 gate。IP-Adapter 当前结果状态为
`completed_control_reuse`，Control/Treatment SHA256 相同，身份 cosine 均为 `0.339437`；PuLID-FLUX
对应身份 cosine 均为 `0.822210`。两条路线的 Wan 视频也已重跑并验证 Control/Treatment 文件哈希
完全相同，视频身份均值差为 `0`。

## 5. 当前代码入口

- PuLID-FLUX 单人 v7：`multishot/pulid_flux_inner_face_experiment.py`
- IP-Adapter 单人 v7：`multishot/ip_adapter_pulid_style_injection_experiment.py`
- 闭眼提示词 gate：`multishot/prompt_injection_safety.py`
- FaceLift 渲染/姿态回检：`multishot/mcp_asset_server.py`
- Gaussian 标定：`multishot/facelift_pose_calibration.py`
- 当前多人 v7 reference 工具：`multishot/multi_face_reference.py`
- IP episode 首帧：`pretest/prepare_entitybench_ip_adapter_pairs.py`
- PuLID episode 首帧：`pretest/prepare_entitybench_pulid_pairs.py`
- Wan：`pretest/run_wan22_i2v_manifest.py`
- 视频 InsightFace：`pretest/evaluate_video_identity.py`
- 参考图 DINOv2：`pretest/evaluate_reference_dino_identity.py`
- 当前 episode 总结：`pretest/summarize_current_episode_results.py`
- Qwen 多参考 Control：`pretest/run_qwen_image21_multiref.py`
- Qwen 多脸 reference 准备：`pretest/prepare_qwen_multiface_residual.py`
- Qwen 同步多人 residual：`pretest/run_qwen_image21_multiface_residual.py`
- Qwen 多人首帧逐角色评测：`pretest/evaluate_qwen_multiface_pair.py`
- Qwen 官方 Control latent 分叉、逐角色容错、自适应 1/6 步入口：
  `pretest/run_qwen_image21_shared_prefix_adaptive.py`
- Qwen 自适应三 episode 全链路：
  `pretest/run_qwen_image21_shared_prefix_adaptive_full3.sh`
- Qwen 单/多人视频逐角色评测：`pretest/evaluate_qwen_multiface_videos.py`
- Qwen 三 episode 可恢复批处理：`pretest/run_qwen_image21_entitybench_batch.py`
- Qwen 三 episode InsightFace 汇总：`pretest/evaluate_qwen_image21_entitybench_batch.py`
- Qwen 三 episode DINOv2：`pretest/evaluate_qwen_reference_dino.py`
- Qwen 完整逐 shot 可视化：`pretest/render_qwen_evaluation_comparisons.py`
- Qwen 历史/当前汇总：`pretest/compare_qwen_historical_adaptive.py`

`--shot` 单镜头重跑已改成增量更新：只替换指定镜头，保留 episode 报告和 manifest 中其余镜头。

## 6. 环境和模型

环境：

- 主图像/注入/评测：`/root/autodl-tmp/movie/.venv`
- Wan2.2：`/root/autodl-tmp/wan22-venv`
- EntityBench evaluator：`/root/autodl-tmp/envs/entitybench-eval`
- Qwen-Image-2.1：`/root/autodl-tmp/qwen-image21-venv`

模型：

- FLUX/PuLID：`third_party/PuLID`
- SDXL：`models/diffusion/sdxl-base-1.0`
- IP-Adapter：`models/ip_adapter/h94-IP-Adapter`
- Wan2.2-TI2V-5B：`models/video/Wan2.2-TI2V-5B`（约 32 GB）
- Qwen-Image-2.1：`models/diffusion/qwen-image-2.1`（约 31 GB，全部权重哈希已校验）
- DINOv2：`/root/autodl-tmp/models/dinov2-base`

Qwen 固定 revision：`790c92633540aa0cb11d9abf19eb46d861714758`。48 GB 单卡使用
`sequential_cpu_offload`；不要在未知显存占用下直接 full-CUDA。

## 7. 数据、资产和正式范围

EntityBench：`benchmarks/entitybench/`。

原始三 episode 范围（20 episode 扩展另见第 11、12 节）：

- run719：12 镜头，8 个单角色；资产在
  `outputs/entitybench_wan22_smoke/episode_00053051/assets/`；
- run893：10 镜头，5 个单角色；
- run1517：12 镜头，10 个单角色；
- run893/run1517 资产在 `outputs/entitybench_pilot3/episode_*/assets/`。

三个 episode 共 12 个角色，均有固定参考图、FaceLift Gaussian 和独立姿态标定。资产必须
episode-local，不能跨 episode 按同名角色复用。

## 8. 当前权威结果

- PuLID run719：`outputs/entitybench_wan22_colornested_v7_s04_12/episode_00053051/`
- IP run719：`outputs/entitybench_wan22_ip_adapter_v7_s04_12/episode_00053051/`
- PuLID run893/run1517：`outputs/entitybench_wan22_pulid_v7_pilot3/`
- IP run893/run1517：`outputs/entitybench_wan22_ip_adapter_v7_pilot3/`
- 参考图 DINOv2：`outputs/entitybench_reference_dino_pilot3/reference_dino_summary.json`
- Qwen 历史两遍式三 episode：`outputs/entitybench_qwen_image21_v7_full3/`，其中
  `evaluation_report.json` 为 InsightFace，`dino_evaluation_report.json` 为参考图锚定 DINOv2；
- Qwen 当前官方 Control 分叉、逐角色容错三 episode：
  `outputs/entitybench_qwen_image21_official_control_partial_adaptive_040_full3/`；
  其中 `HISTORICAL_VS_ADAPTIVE.md` 是历史/当前完整路线对照；
- 当前机器可读总表：`outputs/entitybench_current_summary.json`
- 当前 Markdown 总表：`EPISODE_TEST_RESULTS.md`

以上项目根目录总表仍对应三 episode 范围。最新 PuLID-FLUX/IP-Adapter 的 20 episode
首帧、视频结果分别以第 11、12 节的独立报告为准，不能把三 episode 总表当作最新扩展汇总。

重新汇总：

```bash
cd /root/autodl-tmp/movie
.venv/bin/python pretest/summarize_current_episode_results.py
```

完整评测流程、可视化契约和验收规则已迁移到自动发现的 Codex skill：
`/root/.codex/skills/evaluate-face-injection/SKILL.md`。后续要求大模型使用
`$evaluate-face-injection` 执行或审计评测；不再维护独立的评测方案 Markdown。

该报告包含每个 episode 和每个镜头的触发状态、跳过原因、首帧 InsightFace、视频
InsightFace 与参考图 DINOv2。EntityBench 非 API 指标需要按当前闭眼 gate 重算；VLM/LLM API
指标未运行，缺失项不能当作 0。

闭眼冲突 gate 加入前曾报告过一版历史快照：PuLID-FLUX 17 个注入镜头，InsightFace
首帧/视频 Δ `+0.0331/+0.0344`，DINOv2 `+0.0071/+0.0072`；IP-Adapter 18 个注入镜头，
InsightFace `+0.1029/+0.0831`，DINOv2 `+0.0273/+0.0092`。这组数仅用于追溯，当前主表必须
以自动重算的 gate 后 16/17 镜头口径为准。

## 9. 当前多人边界

PuLID-FLUX 与当前 SDXL/IP-Adapter 不再执行多人注入，因为它们没有可靠的多参考身份绑定。
多人研究保留在 Qwen-Image-2.1 路线：

```text
多张角色参考图 + 文本
→ Qwen-Image-2.1 原生 multi-reference Control
→ 每张检测脸独立姿态渲染、调色和 v7 mask
→ 同一步同步叠加多脸 residual
```

通用多人能力没有取消，只是从不合适的基础模型迁移到原生多参考模型。

最初已完成 run719 / shot 1:1 的 Qwen 多人 pilot：

- 首帧 Julian：`0.674653 → 0.735510`，Δ `+0.060858`；
- 首帧 Viktor：`0.445645 → 0.454448`，Δ `+0.008803`；
- 视频 mean Julian：Δ `+0.074233`；
- 视频 mean Viktor：Δ `-0.014289`；
- 视频两角色宏平均：Δ `+0.029972`，两角色检测覆盖率均为 `1.0`。

结果目录：`outputs/qwen_image21_multiface_v7/run719_shot_1_1/`。该 pilot 证明同步多人链路可运行，
但存在角色间增益不均匀。

2026-09-27 已完成三个 episode 的 Qwen 两遍式历史路线：

- 范围：34 个 shot，23 个单人、11 个多人；
- 首帧：34/34 Control/Treatment 已完成；28 个实际注入、6 个安全复用 Control；
- 回退构成：3 个背脸/缺脸、1 个 Qwen 多人漏角色、1 个闭眼冲突、1 个 v7 core 为空；
- 所有 28 个可准备 shot 的 Treatment 均成功生成，没有运行期降级；
- Wan manifest：68 个 job，其中 62 个实际推理、6 个 Treatment 复用对应 Control；
- 运行目录：`outputs/entitybench_qwen_image21_v7_full3/`；
- Wan 视频已全部完成：68 个 job 中 62 个实际推理、6 个 Treatment 复用 Control，失败 0；
- 逐角色评测已完成：40 个角色-shot 对。InsightFace 首帧/视频 mean Δ 分别为
  `-0.0627/-0.0510`；DINOv2 首帧/视频 mean Δ 分别为 `+0.0043/+0.0024`；
- 34 个 shot 的 `comparison.jpg` 已统一为身份参考、连续 3D 渲染、调色 3D reference、
  最终 mask、Control、Treatment，并标注逐角色指标。

这批数值保留用于追溯，但其 Treatment 使用了最终 Control 准备姿态、mask
和调色 reference，不再作为统一插件协议的算法结论。

### 9.1 Qwen 官方 Control 分叉、逐角色容错、自适应 1/6 步

用户已确认只保留两种窗口：step 12 单步和历史 6 步。调度依据是 step 12 `pred_x0`
中角色参考图与已分配目标脸的 InsightFace cosine：低于 `0.40` 用 6 步，否则用 1 步。
多人镜头逐角色独立调度，同一步仍使用归一化后的多 mask residual 同步叠加。

阈值依据之一是 run719 / 4:1：step-12 cosine 为 `0.2656`；单步只得到
`0.3483→0.3527 (+0.0044)`，而 canonical shared-prefix 6 步为
`0.3483→0.4502 (+0.1018)`，且接近历史两遍式 6 步的 `+0.1009`。

2026-09-28 已完成修正后的 34/34 个首帧、68/68 个 Wan 视频及全部非 API 评测：

- Control 完整走官方 `QwenImage21Pipeline.__call__`，callback 只截取进入 step 12 的 latent；
  34/34 个 Control PNG 与历史官方 Control 逐字节一致；
- 30 个 shot 实际注入、4 个整镜头安全复用 Control，其中 3 个 shot 是部分角色注入；
- 48 个计划角色中 41 个注入、7 个跳过；28 个角色走 1 步、13 个走 6 步；
- 首帧 InsightFace 为 `0.5376→0.5264 (-0.0112)`，`18/41` 个角色为正；
- 视频 InsightFace mean 为 `0.4078→0.3942 (-0.0136)`，`17/41` 个角色为正；
- DINOv2 首帧为 `0.6093→0.6037 (-0.0056)`，视频 mean 为
  `0.5222→0.5238 (+0.0016)`；两者分别有 `21/41`、`19/41` 个角色为正；
- Wan 68 个任务包括 34 个哈希验证后的历史 Control 视频复用、30 个新 Treatment 推理、
  4 个整镜头 Treatment 复用对应 Control；失败 0；
- 历史两遍式 6 步为 IF 首帧/视频 `-0.0627/-0.0510`。修正版明显降低破坏，但 IF 均值仍为负，
  不能表述为稳定身份提升。

当前权威输出根目录：
`outputs/entitybench_qwen_image21_official_control_partial_adaptive_040_full3/`。最终报告为
`evaluation_report.json`、`dino_evaluation_report.json`，34 个 shot 均有 `comparison.jpg`；
历史/当前完整路线对照见 `HISTORICAL_VS_ADAPTIVE.md`。

相同的 34 个 Control 图像已经哈希冻结。历史/当前共有的 39 个可评角色对中，Control identity
cosine 最大绝对差为 `0.001216`；这来自两条路线使用不同注入点 bbox/角色映射元数据评测同一张
Control，而不是 Control 像素发生变化。两条路线的 Treatment 执行协议也不同，因此仍属于完整
路线对比，不是仅注入步数不同的严格消融。

肉眼检查单人正例、负例、部分角色注入和闭眼回退：角色映射与面板完整，未观察到明显 mask
越界或头发泄漏；高相似度正脸仍会出现轻微脸型/质感重绘，与 run1517 的 IF 负增益一致。

旧目录 `outputs/entitybench_qwen_image21_shared_prefix_adaptive_040_full3/` 使用手写 Control
采样路径，仅保留为排错记录，不得再作为当前算法结论或与官方 Control 结果混算。

此前 1/2/5 步和临时 `gain floor=0.55` 的实验目录已移到系统回收站；当前入口不再暴露
该临时调色参数。这里删除的是试验开关，不是 v7 调色：低频 log-RGB harmonization、姿态标定、
纯 3D reference 和 color-safe mask 均保持不变。

run893 / 6:1 的 Qwen Control 计划 4 人但生成了额外背景人物；当前只选择面积最大的 4 张主脸。
该镜头仍需在最终完整 comparison 中单独检查，不得只报告宏平均。

## 10. 下一步

1. 20 episode 单人首帧和视频扩展已完成；优先按脸部尺寸、难度和 episode 分析视频增益、
   负例与检测覆盖率，尤其审计 IP-Adapter 较低的绝对身份分数；
2. Qwen 34-shot 修正版全链路已完成，继续分析高身份单步组，尤其 run1517 的高置信正脸重绘；
3. 决定 `0.40` 阈值是否保留时，继续同时报告 IF、DINOv2、绝对值、差值和肉眼结果；
4. 不重新引入已删除的 2/5 步、临时调色下限、self-attention、旧调色路径或两遍式
   “先完整 Control 再从噪声重跑 Treatment”。

## 11. 2026-09-29 至 2026-10-01：20 episode 首帧扩展评测

用户要求先只扩展 PuLID-FLUX 和 SDXL/IP-Adapter 的首帧评测，不跑 Wan 视频。正式清单在：

`outputs/entitybench_firstframe_pulid_ip_20ep/manifest.json`

清单在看结果前冻结，包含已有 4 个 episode 和按难度确定性抽取的新增 16 个 episode：

- 20 个 episode：Easy 11、Medium 6、Hard 3；
- 363 个官方 shot，其中 132 个单人 shot 进入这两条插件路线；
- 新增部分为 105 个单人 shot、46 份 episode-local 角色资产；
- 多人和无人镜头不在本轮范围内；没有把它们伪装成负例或零分；
- 固定 v7 color-safe mask、注入强度 `0.4`、最多 12 个 active steps、最小脸高 24 px；
- Control/Treatment 保持同 prompt、seed/noise、资产、分辨率和模型，仅 v7 residual 不同；
- 明确闭眼、未检测到可靠脸、脸过小或 v7 core 为空时安全复用 Control。

可恢复运行入口：

- `pretest/run_entitybench_firstframe_pulid_ip_batch.py`
- `pretest/run_entitybench_firstframe_pulid_ip_20ep.sh`
- IP 单人范围通过 `pretest/prepare_entitybench_ip_adapter_pairs.py --single-character-only`
- 汇总器：`pretest/evaluate_entitybench_firstframe_pulid_ip_batch.py`

本轮 tmux `entitybench_ff20` 已正常结束；不存在仍在后台运行的任务。生成完整性：资产 20/20、
PuLID 报告 20/20、IP 报告 20/20。逐 shot 的 Control、Treatment、原参考、3D render、调色 reference
和 comparison 均已落盘。此阶段只生成首帧指标；后续第 12 节已补齐视频和逐帧评测。
VLM/LLM API 指标仍未运行。

角色原始正脸素材由 `pretest/prepare_entitybench_character_assets.py` 显式指定
SDXL Base 1.0（`stabilityai/stable-diffusion-xl-base-1.0`）生成，再用 FaceLift 构建 Gaussian。
通用 diffusion 后端的默认 `juggernaut-xl-v9` 不代表本轮评测素材使用的模型。

实际触发注入且 Control/Treatment 都可测量的 micro mean：

| 路线 | 单人 shot | 实际注入 | 安全复用 Control | InsightFace C→T (Δ) | DINOv2 C→T (Δ) |
|---|---:|---:|---:|---:|---:|
| PuLID-FLUX | 132 | 104 | 28 | `0.6825→0.7160 (+0.0335)` | `0.7513→0.7685 (+0.0172)` |
| IP-Adapter | 132 | 84 | 48 | `0.0831→0.1997 (+0.1166)` | `0.4432→0.4793 (+0.0360)` |

PuLID 的 104 个有效注入对中 86 正、18 负。IP 有 84 个实际注入 shot，但其中 3 个最终条件缺少
可靠 InsightFace/DINO face box，因此聚合使用 81 个有效对：69 正、12 负。IP 的绝对身份分数很低，
后续必须结合逐 shot 参考资产、远景/动画风格和检测质量审计，不能只用较大的正增益下结论。

权威输出：

- Markdown 总表：`outputs/entitybench_firstframe_pulid_ip_20ep/evaluation/firstframe_evaluation.md`
- 逐难度、逐 episode、逐 shot JSON：
  `outputs/entitybench_firstframe_pulid_ip_20ep/evaluation/firstframe_evaluation.json`
- 运行状态：`outputs/entitybench_firstframe_pulid_ip_20ep/status.json`
- 总日志：`outputs/entitybench_firstframe_pulid_ip_20ep/logs/tmux.log`

已肉眼抽查最大正负例。PuLID 最大正例为 run53/3:3，最大负例为 run323/2:3；IP 最大正例为
run146/28:1，最大负例为 run354/4:1。负例必须保留，不得依据分数事后剔除。

### 11.1 Git 状态与交接注意事项

2026-10-01 交接时首帧扩展代码尚未提交；后续已通过 `b27393f`
（`Add 20-episode first-frame evaluation pipeline`）提交。2026-10-04 核对时分支为 `main`，
`HEAD` 和本地 `origin/main` 均为 `b6c06e9`。本次视频协调器、视频评测/续跑修改和交接文档
当时仍在工作树中，尚未提交。后续已通过 `65a07c2`
（`Add 20-episode paired video evaluation and update handoff`）提交并推送到 GitHub `main`。
2026-10-10 核对时本地 `main` / `origin/main` 为 `a62e5b4`（后续产品文档/同源访问提交）。
`docs/frontend_agent_architecture.md` 有用户未提交改动，`experiment_output/` 为未跟踪输出，
均须保留；本次交接更新另产生评测文档改动。`outputs/` 下报告、视频与精选图为本机忽略文件，
没有随着 `65a07c2` 推送到 GitHub。不要 stage 实验输出。

后续提交评测代码时仍应保留用户工作树，检查是否有并行 app/frontend/product-pipeline 改动：

- 保留整个脏工作树，禁止 reset/checkout 覆盖用户改动；
- 只 stage 评测文档、评测/运行脚本、Qwen 当前评测链路、
  `multishot/face_analysis_backend.py` 和 `multishot/facelift_pose_calibration.py`；
- 不 stage `app/`、`frontend/`、产品 pipeline 改动、`experiment_output/` 或任何 `outputs/`；
- 提交前重新执行 Python compile、shell `bash -n`、`git diff --check`，检查 staged diff 后再 push。

## 12. 2026-10-04：20 episode 单人镜头视频扩展已完成

用户要求将第 11 节的 PuLID-FLUX 和 SDXL/IP-Adapter 首帧队列补齐 Wan 视频。
tmux `entitybench_video20` 已结束；`progress.json` 为 `complete`，当前没有后台评测任务。

- 执行入口：`pretest/run_entitybench_firstframe_pulid_ip_20ep_videos.sh`。
- 可恢复协调器：`pretest/run_entitybench_firstframe_pulid_ip_20ep_videos.py`。
- 输出：`outputs/entitybench_firstframe_pulid_ip_20ep/video_evaluation/`，其中 `wan_manifest.json`、`first_frame_hashes.json`、`wan_report.json`、`progress.json` 和 `logs/tmux.log` 可核对进度与复跑。
- 范围：20 episode × 两条路线，各 132 个单人 shot，共 264 组 Control/Treatment、528 个视频任务。多人与无人镜头不在这次扩展范围。
- 最终完整性：528/528 个视频任务完成，408 个新生成、44 个历史视频经过输入/输出 SHA256 及 Wan 参数验证后复用、76 个安全回退 Treatment 复用 Control；失败 0。
- 参数沿用 Wan2.2-TI2V-5B、1280×704、49 帧、24 fps、50 steps、shift 5.0、guidance 5.0、各镜头原 seed。Control/Treatment 仅首帧的 v7 注入不同。
- 20 episode × 两路线的 40 份视频 InsightFace 报告、40 个 DINOv2 case 和 264 组配对明细已完成。InsightFace 报告中全部 528 条视频均解码为 49 帧。
- 身份视频跟踪使用首帧 bbox 初始化的空间连续跟踪，缺少初始 bbox 时从面积最大的检测脸启动；不按最高身份相似度逐帧挑脸。DINOv2 视频复用相同的跟踪 bbox，face crop expansion 为 0.15。

实际注入且两侧可测量的配对均值如下。首帧与视频均报告 Control、Treatment 实际值及差值：

| 路线 | 单人 shot | 实际注入 / 安全复用 | 首帧 InsightFace C→T (Δ) | 视频 InsightFace mean C→T (Δ) | 首帧 DINOv2 C→T (Δ) | 视频 DINOv2 mean C→T (Δ) |
|---|---:|---:|---:|---:|---:|---:|
| PuLID-FLUX | 132 | 104 / 28 | `0.6825→0.7160 (+0.0335)` | `0.5697→0.5852 (+0.0155)` | `0.7513→0.7685 (+0.0172)` | `0.6522→0.6624 (+0.0102)` |
| IP-Adapter | 132 | 84 / 48 | `0.0831→0.1997 (+0.1166)` | `0.0782→0.1653 (+0.0870)` | `0.4432→0.4793 (+0.0360)` | `0.4165→0.4425 (+0.0261)` |

统计口径与限制：

- 视频先对每条视频可检测/可评分帧求均值，再对有效注入镜头等权求均值；上表视频 DINOv2 使用 `video_frame_mean`，不是 shot median。
- PuLID 首帧/视频有效对均为 104；视频 InsightFace 74 正、30 负，视频 DINOv2 63 正、41 负。
- IP 首帧有效对为 81，视频有效对为 84；视频 InsightFace 71 正、13 负，视频 DINOv2 58 正、26 负。
  首帧与视频样本集合不同，不能直接用两组整体增量相除宣称严格的增益保留率。
- 缺脸帧分数保留为 null，并计入检测覆盖率分母，不作为 0 分；安全复用镜头保留在完整清单，未计入实际注入均值。
- 两路线首帧、视频均值均为正增益，但存在负例；IP 视频 Treatment 身份分数仅 `0.1653`，不能仅凭较大增量宣称高质量身份保持。
- EntityBench validated 清单共 140 episode、2,491 shot、847 个单人 shot；本轮覆盖 20 episode 的 132 个单人 shot。
  这是 benchmark 子集评测，每镜头为独立的 49 帧短视频（约 2.04 秒），没有跨镜头记忆，多人和无人镜头未纳入本轮扩展。
- VLM/LLM API 指标未运行，缺失项不能当作 0。

权威输出（均在 `outputs/entitybench_firstframe_pulid_ip_20ep/video_evaluation/`）：

- `video_evaluation.md`：路线首帧/视频总表；当前落盘版本没有逐 episode Markdown 汇总。
- `video_evaluation.json`：路线指标和 264 组逐 shot、逐角色视频配对明细；当前落盘版本顶层为 `routes`、`shots`。
- `cases.json`：40 个 case 的 manifest 与 InsightFace 报告路径索引，**不是逐帧评分文件**。
- `cases/<episode_id>/<route>/video_identity.json`：每条视频的逐帧身份分数、跟踪 bbox、检测覆盖率、first/mean/p10/last 和配对差值；route 为 `pulid_flux` 或 `ip_adapter`。
- `dinov2_video.json`、`dinov2_video.md`：40 个 case 的参考图锚定 DINOv2 和逐 episode/路线聚合。
- `videos/<episode_id>/<route>/shot_<scene>_<shot>/control.mp4` 与 `treatment.mp4`：正式视频文件。
- `wan_report.json`、`wan_manifest.json`、`first_frame_hashes.json`、`progress.json`、`logs/tmux.log`：任务完整性、输入/输出哈希与运行记录。

用户要求实测双进程并行：隔离目录 `parallel_probe/` 的测试视频完成，耗时约 365 秒；
单进程近期每条约 184 秒，两进程同时去噪时每步约 6.7 秒（单进程约 3.1 秒），
合计显存一度约 43 GB。没有观察到明显吞吐提升，正式队列保持单进程。
`parallel_probe/` 不属于正式 528 条视频/264 组配对，不得混入评测。

后续只审计现有结果时读取上述报告，不要重新生成视频。若需要恢复中断任务，可重新运行同一入口，
协调器会校验已有输入图及视频哈希。当前代码的汇总器比后台启动时的版本多了字段和逐 episode 表，
再次执行 `summarize` 会扩展报告结构；不可假定这些新字段已存在于本次落盘报告。

## 13. 2026-10-10：新窗口接续、对比图与风格核查

本会话最新任务是展示有首帧增益的对比图，随后用户要求改找非动漫例子，并再次追加一批。
全部使用已有实验文件排版，没有重新生成首帧、视频，也没有修改正式配对分数、清单或 gate。
当前只是交接文档更新，未启动新实验。

### 13.1 首帧 DINOv2 已完成，旧视频表只是漏列

首帧 InsightFace、首帧 DINOv2、视频 InsightFace mean、视频 DINOv2 mean 四项都已完成。
旧 `video_evaluation/video_evaluation.md` 只有三项，旧 JSON 路线汇总也缺少
`firstframe_dinov2` 字段；首帧 DINOv2 应从 `evaluation/firstframe_evaluation.json` 的
`routes.<route>.reference_dinov2` 和 `shots` 读取，不能解释成“尚未评测”。四项总表见第 12 节。

### 13.2 对比图格式与本机索引

用户认可的顺序是 **2 行 × 3 列**：

| 左列 | 中列 | 右列 |
|---|---|---|
| 原始身份参考 | 连续 FaceLift 3D 渲染 | 调色并对齐后的纯 3D reference |
| 最终 v7 注入 mask | Control 最终首帧 | Treatment 最终首帧 |

标题写路线、run/episode、shot、角色；Control/Treatment 下分别写实际首帧 InsightFace 和
DINOv2，底部写两项 `Control→Treatment (Δ)`。mask 必须来自实际实验，不能用 `pred_x0`、
3D 参考 mask 或手绘区域代替。首帧展示图明确标注 `FIRST FRAME`；完整首帧/视频评估图再补
视频 InsightFace/DINOv2 mean，首帧分数不能冒充视频分数。详细文件映射、复现方式、多人和回退
格式见 [评估说明的对比图契约](docs/entitybench_20ep_evaluation.md#对比图格式)。

输出根目录：`outputs/entitybench_firstframe_pulid_ip_20ep/`。

| 子目录 | 已展示数量 | 内容 |
|---|---:|---|
| `selected_positive_firstframes/` | 6 | 首批大增益正例，混合写实/动画风格；4 张明显偏动画 |
| `selected_realistic_positive_firstframes/` | 6 | 第一批写实正例，PuLID/IP 各 3 张 |
| `selected_realistic_positive_firstframes_batch2/` | 8 | 第二批写实正例，PuLID/IP 各 4 张，与前两批不重复 |

每批均有 `README.md`、`selection.json`、`render_examples.py` 和完整六面板 PNG（1800×1080）。
`selection.json` 保留正式结果行、生成图路径、六个原始面板文件路径；继续找例子前读取三批索引，
按 `(route, run, shot_key)` 去重。同一 run/shot 的不同路线是不同配对，允许并列。
20 张的分数和可点击图索引已写入 [20 episode 评估说明](docs/entitybench_20ep_evaluation.md)。

注意原实验目录里的旧 `comparison.jpg`：本轮查看的 PuLID/IP 旧版第四格是中期 `pred_x0`，
不是最终 mask，且缺少指标文字。新精选 PNG 补齐了这些内容，旧图未覆盖。
不能因文件名叫 `comparison.jpg` 就声称满足六面板评估契约，也不能声称全部 264 组旧图已修好。

### 13.3 动漫风格来源与正例偏差

已经核对原始 EntityBench 镜头提示词及实际生成配置：

- PuLID run53/3:1 明确写 `In an animated medium shot ...`，官方要求动画风格；
- PuLID run53/3:3 只描述 Kaito 在广场张望、产生灵感，没有明确动画要求；
- IP run146/28:1（Nia 起床）和 run44/11:1（Daniel 在办公室走廊）都没有动画要求；
  这些图的风格来自基础模型输出，具体成因尚未做受控实验，不可声称由上一镜头“记忆”导致；
- PuLID 使用官方镜头提示词；IP 的 `_sdxl_prompt` 为 CLIP-77 调整动作/实体描述顺序，
  保留原文，没有统一加动漫词，也没有强制所有镜头写实；
- 身份参考与 3D 素材是写实资产。注入把动画脸拉向写实参考可能提高身份指标，同时破坏画风；
  写实正例也有皮肤平滑、年龄感、表情和肤色变化，不能把正增量直接等同于全面视觉改善；
- 首批按大增益选例，6 张中 4 张偏动画，不代表整个 132-shot 队列的风格比例。
  两批 14 张写实正例是肉眼筛选的展示样本，尚未做全量风格标注/写实子集总体均值。

第二批查看后因风格化而未纳入的 IP run1775/7:1、run444/4:1 图保留在
`selected_positive_firstframes/realistic_review/excluded/`，原因见其 `README.md`；
正式 benchmark 行仍保留，未据展示筛选删除或重算。

### 13.4 接下来如何接着做

1. 用户再要例子：从正式首帧 JSON 筛 `plugin_applied=true`，同时读两项 C/T/Δ，
   先查看实际 Control/Treatment 风格、检测与可见变化，再按上述格式排版；不重跑生成。
2. 需要写实子集结论时：先按可复核规则标注全部镜头风格，再报告人数、有效对、正负例、
   四项绝对值和差值；不能将这 14 张挑选正例的均值当作正式写实子集结果。
3. 用户若要求提交本次更新，使用 `$movie-github-ssh-push`；仅提交本次评测文档/相关说明，
   保留 `docs/frontend_agent_architecture.md` 等用户改动，不提交 `outputs/`。

## 14. 新增 20 episode 跨镜头人脸外观质心评测

用户要求补充“外观质心相似度”。已在现有 20 episode PuLID-FLUX / IP-Adapter 的
264 组首帧和视频配对上完成；没有重新生成图像/视频，没有改动原有评分、清单或 gate。
Qwen 三 episode 不在这次新增范围内。

- 入口：`pretest/evaluate_cross_shot_face_centroid.py`。
- 回归测试：`tests/test_cross_shot_face_centroid.py`。
- 报告根目录：`outputs/entitybench_firstframe_pulid_ip_20ep/cross_shot_evaluation/`。
- 主报告：`face_centroid_evaluation.json` / `.md`；源文件完整性与独立复算见 `validation.json`。
- `features/` 是源文件内容、模型与 bbox 参数共同确定的可恢复特征缓存；`crops/` 为实际评分图，
  `audit/` 有 88 张跨镜头 C/T 联系表。原有六面板 `comparison.jpg` 未改写，旧格式缺口仍存在。

计算口径：按路线、episode-local 角色分组，Control/Treatment 在相同的配对有效镜头集合上
各自计算归一化 DINOv2 CLS 向量均值质心；每个镜头与该侧质心求 cosine，至少两个不同镜头。
采用论文公式（质心包含该镜头自身），但沿用现有人脸 bbox、0.15 扩展裁剪和空间视频跟踪，
不包含官方 GroundingDINO 全角色定位、CLIP 筛选、VLM fidelity gate；必须标为项目指标。
视频从 5 个等距采样帧按清晰度×人脸面积选代表帧，不按身份 cosine 选帧，也不是逐帧均值。

| 口径 | 路线 | 首帧有效角色-shot / C→T (Δ) | 视频代表帧有效角色-shot / C→T (Δ) |
|---|---|---|---|
| 全部单人，含安全复用 | PuLID-FLUX | 85 / 0.8925→0.8958 (+0.0033) | 91 / 0.8286→0.8338 (+0.0052) |
| 全部单人，含安全复用 | IP-Adapter | 60 / 0.7783→0.8067 (+0.0284) | 78 / 0.7565→0.7647 (+0.0083) |
| 仅实际注入，重算质心 | PuLID-FLUX | 78 / 0.9120→0.9159 (+0.0039) | 78 / 0.8581→0.8636 (+0.0055) |
| 仅实际注入，重算质心 | IP-Adapter | 58 / 0.7836→0.8135 (+0.0299) | 61 / 0.7856→0.7946 (+0.0091) |

每路线 132 个 shot 包含 54 个 episode 内角色，其中 29 个只出现一次；剩余 25 个角色的
103 个 shot 是全量口径的潜在有效集合。缺失和单次出现有明确原因，不填零。
路线/episode 均按可评分角色-shot 等权；仅注入子集独立重算质心。
回退图像 C/T 相同，但因为其他镜头导致两侧质心不同，其质心分数 Δ 可以非零，不能强制置零。

独立复算了 151 个有效角色组（两种口径×两种模态）、核对了 1,122 个输入/模型/原报告 SHA256，
均通过；安全回退媒体哈希一致。已查看首帧/视频正负例及回退裁剪；正分数也可能伴随画风变化、
鼻部异常、皮肤平滑或年龄感改变，不得表述为统一视觉改善。

本轮新增质心评测使用 **CPU**：首帧/视频代表帧的 DINOv2 特征提取及质心汇总均在 CPU 上完成，
参数为 `--device cpu --cpu-threads 1 --batch-size 1`。本轮执行环境无可用 GPU，CPU 配额约
0.5 核、内存 2 GB；这不代表此前图像/视频生成或原四项评测也使用 CPU。
7 项回归测试通过；续跑完整复用 1,056 条 appearance 特征记录，分数与首次完成的报告一致。
汇总命中缓存时不加载模型；复现使用评估说明中的单线程命令。全部产物仍在忽略的 `outputs/` 下。
本轮 Git 提交范围为质心评测脚本、回归测试、本文和 20 episode 评估说明；
用户前端文档及其他已有未提交改动保留在本机，不纳入本轮提交。
