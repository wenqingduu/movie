# 新会话接手提示词

复制下面内容到新会话。状态数字为静态交接，先读实际进程和实时文件。

```text
请接手 /root/autodl-tmp/movie，使用中文。先读LLM_HANDOFF.md（最新记录在前）、docs/entitybench_full140_completed41_snapshot_20261103.md，再查git status、tmux、进程、GPU、磁盘；保留所有无关改动，不重复启动GPU任务。

当前后台是tmux qwen140_episode_repairs；入口pretest/qwen140_repair_priority_queue_v2_20261103.py run，技术适配器pretest/qwen140_technical_resume_v2_20261103.py。状态、日志和不可变计划位于outputs/entitybench_episode_repairs_20261103/（status.json、progress.json、queue.log、queue_v2.json）。canonical pointer为outputs/entitybench_qwen140_multiface_remaining126_20261030/active_queue.json。旧remaining126原次序和旧qwen140_csface_priority队列已退出，不启动它们；守护修复Agent保持关闭。

该队列继续原55集1413shot低DINO cohort，先补原16个中断集，再接原55集未完成部分。运行中的19个冻结生成源码不可原地修改。仅T5 checkpoint torch.load mmap=True，242张量bit-exact且整集实跑验证；人物查询有效ID约束、匿名人物保留行动；只对原task_id的GET网络错误作有限重试，POST不重试。模型、BF16、LoRA、种子、v7身份/面积/姿态门控、画风策略不变。正常全图注入跳过无需确认；技术异常留档继续其它集，全局额度/认证或再现SIGKILL会停止留档。

11月4日17:29静态状态：本55队列19集完整、原16中断集修复10集；加此前30集累计49集完整生成并完成常规非VLM评测。当前run34/i0_j23，当时12/15shot、6:5首帧；数字以实时各集progress为准。run53/i65_j78在4:3过渡API返回IPInfringementSuspect，保留前8shot，未自动改提示词，整集未完成。后台修复后无新增T5加载SIGKILL。

用户只授权并已删除models/llm/Qwen2.5-14B-Instruct-AWQ（9.31GiB），SDXL和IP-Adapter仍保留；不要把被中断的先前三模型请求当继续删除授权。清理记录outputs/disk_model_cleanup_audit_20261103/。当前空闲约20.6GiB，队列保留8GiB，空间不足等待，不删历史结果/参考/Gaussian。

最新已发布全量指标为outputs/entitybench_full140_completed41_snapshot_20261103/report.md：按11月3日17:20冻结41集670shot重跑＋99集1821shot原版，完整140集2491shot；不是140集统一新版重跑。默认DINO0.7863→0.7920（131有效集、1380→1417次），raw cs_face0.7155→0.7214，角色出现0.9212→0.9437，成像59.6517→60.4016；动态/场景/边界下降。之后新完成集尚未纳入该冻结表。完整CSV、all/easy/medium/hard、输入SHA验证同目录。

评测按skills/evaluate-face-injection-v2/SKILL.md及references/visible_single_face_region_dino.md。默认人物一致性是可见单脸筛选＋确认定位错误剔除后的原人物区域DINO；保留官方raw cs_face。原裁剪三项排除只按精确SHA沿用，新图重新复查，排除后重算质心。不能把自动门控中间结果当最终复查结果，不删除真实身份/画风失败/重复人物或未决案例。官方strict/skip_vlm、VLM0，保真度门控缺失如实说明。

用户接受新版run594/i75_j98的3:4单Leo6步注入和3:5原目标图复用＋新过渡/主段。新版已纳入此前30集评测；绘图素材和noised references、实际去噪张量、Leo/Marcus四角渲染均已导出，入口见LLM_HANDOFF。不要重跑已接受媒体，不补Control视频，除非用户新要求。

授权范围：继续后台实验，正常门控跳过不用确认；异常记录并推进其它集。不自动变更模型、降低阈值、语义重试或复活守护Agent。用户请求状态则查实时证据；评测只纳入完整episode，冻结快照不随运行中队列滚动变动。Git提交按用户明确请求进行，outputs/模型/环境密钥不上传。
```
