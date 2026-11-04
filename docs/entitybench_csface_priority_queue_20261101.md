# 人物一致性低分优先队列（2026-11-01启动）

用户要求继续按当前多人3D集成重跑，优先原版cs人物一致性低于平均的episode。当前默认使用可见单脸＋定位纠错人物区域DINO；读取原full140已冻结逐集诊断，131有效集精确均值0.7862577181641647（显示0.7863）。排除已有30个完整重跑，选出尚未完成且低于均值的55集、1413shot，按原分数从低到高。只改调度次序，原19个冻结生成/评分源码及cohort未改，角色/3D资产、门控与模型策略均不变。

[队列及进度](../outputs/entitybench_csface_priority_20261101/README.md) · [实时状态](../outputs/entitybench_csface_priority_20261101/status.json) · [冻结清单](../outputs/entitybench_csface_priority_20261101/queue.json)

当前tmux为`qwen140_csface_priority`，入口`pretest/qwen140_csface_priority_queue.py run`。生成/评分输出继续放在`outputs/entitybench_qwen140_multiface_remaining126_20261030/experiments/<完整episode_id>/`，状态与调度日志改看新优先队列根。旧remaining126/status.json为历史状态，新入口也在其active_queue.json登记。共享原.queue.lock，防止两份调度同时写相同episode。

前五：run323/i0_j14（15shot，0.6277）、run146/i0_j46（50shot，0.6298）、run221/i0_j18（50shot，0.6545）、run406/i5_j41（16shot，0.6645）、run719/i35_j44（12shot，0.6661）。已完成30集以及run594刚修复的独立媒体不重跑或覆盖。原run155查询名称错误的未完成前缀不在此次低分集合，未修改。

资源已实查RTX4090 48GB，启动时约35.6GiB空闲磁盘，GPU起初空闲且无旧队列，19份生成源码SHA一致。qwen3.8-max接口单次24token连通预检通过，不属于VLM评分。首次观察run323进入references阶段，GPU约40GiB已有负载，不把prepared状态当作已运行。

按旧批次阶段统计估算55集连续运行约140小时／5.8天，并非完成承诺。415个episode-local角色、1413shot预计新增约51GiB；比当前空间多，且每集开始前另保留8GiB和25%单集估算余量。需要后续扩容；空间不足时等待，不自动删除历史结果。

逐集流程保持干净角色素材＋对应FaceLift Gaussian、Agent A/B/C、角色动作提示词、episode写实风格、单/多人v7门控、Wan本地4步主视频与B首尾帧API。正常门控跳过沿用同次Control，技术异常保留前缀后继续其他集；全局API认证/额度异常停止新任务。没有守护/修复Agent、自动语义重试或换模型。

每集完成后自动跑官方12项strict/skip_vlm评分；新可见单脸＋定位纠错DINO需另做新裁剪检测与定位复查，明确标pending，不能把raw cs_face标成新默认DINO。为避免旧混合快照发布问题，本队列不调用原每集full140滚动聚合，也不覆盖最新30集独立评估报告。完整数据和评分仍写到原合法episode目录。

下次先看实际tmux/进程、queue/status.json、queue.log、对应episode/logs。若仍运行不要重复启动；重启使用同一队列清单，SHA与已完成缓存必须相符。SSH断开不影响已启动tmux任务；关机/释放实例会终止进程，恢复后需要续跑。
