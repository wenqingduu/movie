# 3D 人脸一致性注入项目交接文档

最后更新：2026-09-27。仓库：`/root/autodl-tmp/movie`。

## 1. 接手顺序

1. 保留未提交工作树，不要 reset 或 checkout；前后端目录还有用户的并行改动。
2. 阅读本文、`EVALUATION_PLAN.md` 和自动生成的 `EPISODE_TEST_RESULTS.md`。
3. 核对 GPU、模型路径和 Python 环境。
4. `c3c9fcccd558befb7302a046794e789c9f71f615` 是本轮 Qwen 全 episode 批处理前的基线；
   以仓库最新提交和本文为准，不要恢复已删除的旧调色、自注意力或旧多人路径。

## 2. 当前任务定义

本项目当前评测一个首帧/关键帧身份增强插件：

```text
文本 + 固定身份/3D 资产
→ 基础图像模型生成 Control
→ v7 局部 3D residual 生成 Treatment
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
7. 正常脸基准强度 `0.4`、最多 12 个活跃注入步；小脸自动降强度和提前结束。

公式：

```text
target_next = scheduler_step(target_state)
reference_next = 同 timestep 的带噪 3D reference latent
target_next += mask * strength * (reference_next - target_next)
```

## 4. 安全回退

以下情况 Treatment 复用 Control，并保留在 episode 中按差值 0 统计：

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
- Qwen 单/多人视频逐角色评测：`pretest/evaluate_qwen_multiface_videos.py`
- Qwen 三 episode 可恢复批处理：`pretest/run_qwen_image21_entitybench_batch.py`
- Qwen → Wan → 评测全链路：`pretest/run_qwen_image21_entitybench_full3.sh`
- Qwen 三 episode 汇总：`pretest/evaluate_qwen_image21_entitybench_batch.py`

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
- 当前机器可读总表：`outputs/entitybench_current_summary.json`
- 当前 Markdown 总表：`EPISODE_TEST_RESULTS.md`

重新汇总：

```bash
cd /root/autodl-tmp/movie
.venv/bin/python pretest/summarize_current_episode_results.py
```

该报告包含每个 episode 和每个镜头的触发状态、跳过原因、首帧 InsightFace、视频
InsightFace 与参考图 DINOv2。EntityBench 非 API 指标需要按当前闭眼 gate 重算；VLM/LLM API
指标未运行，缺失项不能当作 0。

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

2026-09-27 已进一步冻结并启动三个完整 episode 的 Qwen 路线：

- 范围：34 个 shot，23 个单人、11 个多人；
- 首帧：34/34 Control/Treatment 已完成；28 个实际注入、6 个安全复用 Control；
- 回退构成：3 个背脸/缺脸、1 个 Qwen 多人漏角色、1 个闭眼冲突、1 个 v7 core 为空；
- 所有 28 个可准备 shot 的 Treatment 均成功生成，没有运行期降级；
- Wan manifest：68 个 job，其中 62 个实际推理、6 个 Treatment 复用对应 Control；
- 运行目录：`outputs/entitybench_qwen_image21_v7_full3/`；
- tmux：`qwen_full3`。视频生成和最终逐角色评测仍可能在后台运行，以
  `wan_report.json`、`evaluation_report.json` 是否完整为准。

run893 / 6:1 的 Qwen Control 计划 4 人但生成了额外背景人物；当前只选择面积最大的 4 张主脸，
并记录 `extra_reliable_face_count=1`。run893 / 6:1 中 Leo 的自动映射 margin 仅 `0.0678`，最终分析
必须单独检查，不得只报告宏平均。

## 10. 下一步

1. 等待 `qwen_full3` 完成并读取 `evaluation_report.json`，先报告 34-shot 全量结果；
2. 分开报告实际注入、完整 episode（回退 Δ=0）、单人和多人，不用宏平均掩盖负角色；
3. 根据完整结果再决定是否做逐角色自适应强度/停止步；
4. 增加风格/OOD gate 和极端侧脸局部关键点可靠性；
5. 扩大 episode/seed 并加入 bootstrap 置信区间；如需完整 EntityBench，再配置 VLM/LLM API。
