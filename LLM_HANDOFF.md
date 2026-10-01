# 3D 人脸一致性注入项目交接文档

最后更新：2026-09-28。仓库：`/root/autodl-tmp/movie`。

## 1. 接手顺序

1. 保留未提交工作树，不要 reset 或 checkout；前后端目录还有用户的并行改动。
2. 阅读本文和自动生成的 `EPISODE_TEST_RESULTS.md`；执行或审计评测时使用
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

当前三个 episode：

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

1. 当前 34-shot 修正版全链路已完成；后续优先分析高身份单步组，尤其 run1517 的高置信正脸重绘；
2. 决定 `0.40` 阈值是否保留时，继续同时报告 IF、DINOv2、绝对值、差值和肉眼结果；
3. 不重新引入已删除的 2/5 步、临时调色下限、self-attention、旧调色路径或两遍式
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
和 comparison 均已落盘。本轮只有首帧指标，没有视频指标，也没有 VLM/LLM API 指标。

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

截至 2026-10-01，当前分支为 `main`，`HEAD` 和 `origin/main` 都仍在 `0a1b44d`。用户随后要求
“把目前的评测代码推送一下”，但在 commit/push 前又要求切换新窗口，因此最新评测代码**尚未提交、
也尚未推送**。

工作树同时包含另一组 app/frontend/product-pipeline 改动，所有权不属于本次评测提交。后续提交时：

- 保留整个脏工作树，禁止 reset/checkout 覆盖用户改动；
- 只 stage 评测文档、评测/运行脚本、Qwen 当前评测链路、
  `multishot/face_analysis_backend.py` 和 `multishot/facelift_pose_calibration.py`；
- 不 stage `app/`、`frontend/`、产品 pipeline 改动、`experiment_output/` 或任何 `outputs/`；
- 提交前重新执行 Python compile、shell `bash -n`、`git diff --check`，检查 staged diff 后再 push。
