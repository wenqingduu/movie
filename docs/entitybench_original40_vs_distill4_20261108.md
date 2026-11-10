> **最新用户指令：大队列保持暂停，不再自动恢复。首次试跑因遗漏RoPE配置失败，当前重试见 `docs/entitybench_original40_torch_rope_20261108.md`；下文原自动恢复与旧ETA为历史记录。**

# 原版40步与4步蒸馏：固定输入的8条对照

准备时间：2026-11-08 20:19（北京时间）。用户明确要求试跑；使用已存在完整原版权重，不下载、不修改正式benchmark。

## 对照条件

Same exact first-frame PNG, positive prompt, seed, resolution,81frames/16fps, BF16 and torch_sdpa. Compare original40/CFG3.5/noLoRA/WanScheduler/boundary.9 against existing4/DMD2/LoRA1022/CFGoff; empty negative prompt for original CFG. No image/person/3D regeneration, route change, following-shot generation, paidAPI, VLM or benchmark replacement.

原版：同Wan2.2-I2V-A14B底座，不加载任何蒸馏LoRA，原WanScheduler多步求解40步，CFG3.5，shift5，boundary=.9按timestep切换high/low；不是延长4步DMD2到40步。负提示词为空，避免另加运动/画质指令。蒸馏参照：已有1022蒸馏LoRA、DMD2四步、CFGoff的实际视频，精确SHA绑定，不重新采样。

## 八条冻结名单

| 对照 | 来源 | 镜头 | 类型 | 4步动态 |
|---|---|---|---|---:|
| N01 | M0007 | run323/i60_j94/2:1 | camera | 0 |
| N02 | M0008 | run466/i15_j24/2:1 | camera | 0 |
| N03 | M0001 | run1/i50_j84/18:1 | pilot_walking | 0 |
| N04 | M0002 | run4/i80_j98/7:6 | pilot_opening_door | 0 |
| N05 | M0006 | run264/i50_j98/1:6 | pilot_strumming | 0 |
| N06 | M0003 | run103/i30_j54/24:2 | pilot_walking | 1 |
| N07 | M0004 | run146/i0_j46/7:1 | pilot_one_off_arm_raise | 1 |
| N08 | M0005 | run221/i0_j18/16:5 | pilot_continuous_prop_motion | 1 |

其中5条4步动态0、3条4步动态1参照；覆盖运镜、步行、开柜门、弹吉他、抬手与双节棍。预先固定名单，不因原版结果再换案例/改词/换种子。首条N01为明确右摇镜的run323/2:1，先确认81帧输出/原版配置正确，再继续其余7条。

## 状态与调度

- 新tmux：`qwen140_original40_test_20261108`。入口：`pretest/qwen140_original40_vs_distill4_20261108.py schedule`。
- 原后台父调度45449已STOP以阻止派下一批，当前GPU worker69726继续完成M0135–M0166，不取消当前shot；新测试等这32项结果全部保存、worker退出释放GPU/CPU模型内存后，才加载原版模型。不是原队列出错或卡住。
- 原版8条生成结束后释放模型，再算原6项VBench，旧4步分数仅按SHA验证后复用。所有8条及原动态0的5条子集分别汇总，不只展示正例。
- 成功/技术失败均在finally SIGCONT恢复原后台父调度，receipt写在`original_queue_resume.json`。不自动改采样参数、减步数、关CFG或重试来凑成功。
- 配置/输入CPU预检已通过：`preflight.json`。此预检没有加载模型权重或生成视频，不能写成已通过实际GPU试跑。
- 第一条实际生成验证在`first_case_validation.json`（完成后才出现），每条视频与生成记录在`cases/N##/`。状态`status.json`，原版生成日志`generate.log`，评分日志`evaluate.log`。

预计等待当前批次后，整个8条40步测试约2–4小时；先用第一条实际时间更新，不将理论20倍前向次数当精确时间。原1333队列会相应延后测试占用时间。

## 解释边界

Small purposive backend-recipe comparison, not an isolated step-count or distillation-weight ablation. Different CFG/scheduler are required original-vs-distilled recipes. Same seed does not itself prove identical latent trajectory. Report all8 and5 zero-baseline subset, not a full benchmark score.

这是“原版推理配方 vs 蒸馏推理配方”配对测试；CFG/求解器/权重同时对应各自适配配置，不是孤立步数消融。只有8条目的性样本，不推断全2491shot平均。6项VBench不会证明角色/身份/动作正确，视频需要复核；不在这里计算跨镜头DINO或做benchmark替换。
