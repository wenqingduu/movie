# A14B 加速小样本试验

**后续更新：用户已选定4步全量，旧40步队列停止并保留7段视频。**
本页保留小样本历史与实测；当前363shot正式任务见[4步执行说明](entitybench_a14b_4step_run.md)。
新正式任务只跑官方非VLM指标，下面IF/参考DINO表不会作为其官方结果。

2026-10-11，用户授权尝试开源加速。独立输出为
`outputs/entitybench_a14b_acceleration_trial/`，不替代正式20episode的40步BF16结果。
试验准备、权重下载期间正式队列继续运行；等当前镜头完整保存后，临时让出唯一GPU，
有限试验结束或失败后自动按原代码、配置与输入依赖续跑。

## 当前范围

单张48GB RTX4090；81帧、16fps、max_area=832×480，保持输入比例；原参考图及首帧不重新生成。
范围预先固定为run719的2:1、4:1、4:2：2:1为实际中间步注入首帧，4:1为安全回退首帧，
4:2为多人接续。本路线4:2只能使用本候选4:1视频的真实解码尾帧，不使用正式40步路线尾帧。

| 路线 | 镜头 | 设置 |
|---|---|---|
| 原官方实现基线 | 2:1、4:1、4:2 | 复用已有正式81帧/16fps、40步、shift3、CFG3.5 BF16视频及实测耗时 |
| LightX2V普通BF16 | 2:1 | 相同基础权重、40步、shift3、CFG3.5，无蒸馏、缓存或量化 |
| LightX2V蒸馏BF16 | 2:1、4:1、4:2 | 原基础权重加两份4step_1022蒸馏LoRA；官方DMD2时间步1000/750/500/250、专家切换index2、shift5、CFG关闭 |

本轮先使用`torch_sdpa`，以免把注意力量化和蒸馏同时混入首轮可行性结果。
模型级CPU offload，T5 offload，Torch RMSNorm/LayerNorm与modulation，原VAE，禁用轻量VAE。
LightX2V与原实现有版本、编码/采样/数值细节差异，同一seed不等于完全相同的latent。
4步版本同时改变LoRA、时间步、CFG及shift，是新的生成配置，不能只归因于减少步数。

## 版本、权重与环境

- LightX2V：`7bd01b7d5d04bf99cdd9ca9082f20c184c66d991`，目录`/root/autodl-tmp/LightX2V`。
- 基础模型沿用原官方A14B已下载版本，原文件不修改、不另存完整合并模型。
- `lightx2v/Wan2.2-Distill-Loras`：`570044187a5219776ef30a5c60c6f76428a3a10a`。
  仅下载high/low两个`wan2.2_i2v_A14b_*_noise_lora_rank64_lightx2v_4step_1022.safetensors`，
  共1,374,118,048字节，官方LFS SHA256均已通过。
- 环境`/root/autodl-tmp/envs/lightx2v-trial`：只在本环境安装额外小型依赖，通过pth只读复用
  已有Qwen环境的Torch2.5.1+cu118及项目依赖。原正式Wan环境及其Torch2.4.0不改变。
- 入口`pretest/run_a14b_acceleration_trial.py`；GPU调度、检查点快照与原队列自动恢复入口
  `pretest/supervise_a14b_acceleration_trial.py`，tmux `a14b_acceleration_trial`。

## 输出和判定

`plan.json`冻结镜头、原基线、配置及限制；每路`report.json`独立记录模型加载耗时、
每镜头生成耗时、显存峰值、实际分辨率/帧率、输入与视频/尾帧哈希。
`formal_snapshot.json`和`formal_preservation.json`审计原正式视频、尾帧及生成配置未变化。
正式实验恢复仍使用原来的依赖校验，不把候选视频写入正式目录。

`pretest/evaluate_a14b_acceleration_trial.py`在CPU复用现有InsightFace与参考DINO评测逻辑。
实际视频首帧可用固定身份参考映射，后续仅空间跟踪；保留逐帧缺脸和检测覆盖率。
2:1、4:1给出身份诊断；4:2多人映射未接入，明确缺失，不填零。
所有镜头出具输入及0/20/40/60/80帧联系表；不筛掉负例。
这些是小样本诊断，不是官方EntityBench分数、完整跨镜头质心评测或插件Control/Treatment消融。

本机测出速度与人脸/运动差异后再决定后续配置，不能直接套用8×H100或5090的宣传倍数。
FA3面向Hopper，NVFP4+稀疏50×结果面向Blackwell；缓存、量化也改变计算或精度，需独立验证。

## 加载阶段的适配记录

首次两路测试均在UMT5加载阶段收到SIGKILL（退出码-9），没有生成候选视频。
观察到进程约81GiB的共享/固定页内存，容器总内存96GiB；cgroup的oom计数为0，
因此记录为内存压力下被终止，不能声称已经确认是cgroup OOM killer。
原日志与退出码保留在`attempt1_pinned_host/`。
重试仅在候选进程拦截`torch.empty(pin_memory=True)`，使用普通CPU存储，
不改变权重值、精度、步数及正式Wan代码。这个适配可能影响CPU到GPU传输速度。

对全部safetensors头部逐键审计（`lora_header_audit.json`）：高噪声LoRA无未匹配键；
低噪声LoRA的290个未匹配张量键全部属于原A14B没有的图像编码器分支
（`img_emb`、`k_img`、`v_img`、`norm_k_img`），没有未识别命名。
实际加载日志高噪声应用1095/1095项调整、低噪声应用1054/1262项调整；
不能把低噪声描述为全键匹配。试验使用普通A14B的无CLIP图像编码器路径。

依据：[LightX2V普通I2V示例](https://github.com/ModelTC/LightX2V/blob/main/examples/wan/wan_i2v.py)、
[蒸馏LoRA示例](https://github.com/ModelTC/LightX2V/blob/main/examples/wan/wan_i2v_with_distill_loras.py)、
[LoRA发布页](https://huggingface.co/lightx2v/Wan2.2-Distill-Loras)、
[8×H100优化报告](https://github.com/morphicfilms/wan2.2_optimizations/blob/main/blog.md)。

## 2026-10-11 小样本完成后的结果

三段4步候选与一段普通40步候选均生成完成；每段81帧、16fps、848×464，输入和尾帧依赖已验证。
模型加载耗时单列：LightX2V普通40步83.6秒，4步蒸馏95.4秒；下表为加载后的单镜头生成时间。

| 镜头 | 原官方40步 / 秒 | LightX2V普通40步 / 秒 | LightX2V蒸馏4步 / 秒 | 4步相对原版速度 |
|---|---:|---:|---:|---:|
| 2:1 | 1238.2 | 993.1 | 79.9 | 15.49× |
| 4:1 | 1234.1 | 未测 | 93.6 | 13.19× |
| 4:2 | 1218.8 | 未测 | 93.5 | 13.03× |

普通40步仅测2:1，速度提升1.25倍。4步三镜头平均89.0秒；若363镜头均保持这个吞吐，
纯生成粗略外推约9小时，未包含评测/失败重试，也不是整批实测或质量承诺。

| 路线 | 单人镜头 | IF视频均值 | 参考DINO均值 | 人脸检测覆盖率 |
|---|---|---:|---:|---:|
| original_bf16_40step | 2:1 | 0.5817 | 0.5703 | 1.0000 |
| original_bf16_40step | 4:1 | 0.0642 | 0.2973 | 0.4444 |
| lightx2v_bf16_40step | 2:1 | 0.6445 | 0.5817 | 1.0000 |
| lightx2v_distill_bf16_4step | 2:1 | 0.6522 | 0.6637 | 1.0000 |
| lightx2v_distill_bf16_4step | 4:1 | 0.0575 | 0.3558 | 0.8395 |

IF与DINO均使用固定原身份参考，CPU完成；它们是项目身份诊断，不是EntityBench官方跨镜头分数。
4:1原版与候选的可检测帧范围不同，不能把这两个均值当作相同帧上的配对差值；
原首帧属于小脸安全回退，该负例保持在结果中。4:2多人身份辅助缺失，不能填零或宣称一致性达标。

已人工查看三张完整对照表：2:1的4步人物较少转头，身份分数较高不能替代动作保真；
4:1两路人物均最终离开主体画面，时间进程不同；4:2的4步抽样帧出现第三个人，原版抽样为两人。
因此结论为“4步加速可运行、显著节省时间”，不能认定质量等价或直接证明3D插件效果。

正式20episode原40步队列已恢复：6个已保存镜头视频/尾帧以及冻结manifest哈希均未改变，
生成器将6个镜头全部标记为`resumed_verified`，继续run719 4:4。候选没有写入正式视频目录；VLM继续不测。

完整表与图片见`outputs/entitybench_a14b_acceleration_trial/report.md`及`quality/report.json`；
失败尝试、配置适配和源码快照保留在`attempt1_pinned_host/`、`attempt2_pageable/`、`adaptations.json`。
普通40步运行的源码哈希对应`attempt2_pageable/runner_40step.py`；修正后的4步对应当前入口源码。
所有候选已生成并完成诊断；正式363shot仍在生成，尚不能报完整实验完成。
