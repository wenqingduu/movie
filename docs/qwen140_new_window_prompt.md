# 新会话接手：原始benchmark与当前多人3D集成

```text
请接手 /root/autodl-tmp/movie，使用中文。先读 LLM_HANDOFF.md，再检查 git status、tmux、实际进程和GPU/cgroup，不重复启动后台队列。

主交接只保留：
1. 原始完整140集/2491镜头benchmark设计和正式结果（只读保留）。
2. 当前多人3D集成：Agent A/B/C路由、角色描述提示词强化、统一episode画风、Qwen逐角色v7、Wan四步；run700已完整验证，首批低指标4集/48shot在后台执行。

运行目录 outputs/entitybench_qwen140_multiface_weak4_20261028/；tmux qwen140_multiface_weak4；入口 pretest/qwen140_multiface_weak_episode_batch.py。先读 cohort.json、status.json、各集progress.json/日志以及异常记录。视频和指标会进入根index.html与各集review/index.html、report.md。

用户最新授权：整张无角色通过3D门控属于正常跳过，沿用同次Control继续，不必确认；真正技术异常记录并继续其他episode，尽量把整批跑完。不得放宽阈值、改模型、隐藏未完成项或把未注入记为实际注入。新授权覆盖之前“遇事先停问用户”的旧要求。

当前人物匹配先在全部检测脸中做全局一对一分配，不先按面积取前N；身份cosine0.4只决定1步/6步，不是可靠性通过线。没有新增面积禁注入阈值。B固定3秒过渡＋81帧行动，行动用实际过渡尾帧，不使用计划目标图代替。

官方评分保持strict/skip_vlm、原采样与阈值，VLM调用0；cs_face是跨镜头人物区域DINO，非人脸身份准确率，缺保真度门控。逐episode完整评分和聚合，不把片段或run700分数冒充140集。保留所有负例、缺项和异常。

先确认实时进度；如队列已在运行就继续监测，勿重启。保留素材、视频、3D、原始报告、代码快照及无关未提交改动；不自行推送Git。
```
