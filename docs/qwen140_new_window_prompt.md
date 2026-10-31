# 新会话接手提示词（2026-10-31 23:07北京时间）

复制以下内容给新会话：

```text
请接手 /root/autodl-tmp/movie，使用中文。先读 docs/entitybench_session_handoff_20261031.md、LLM_HANDOFF.md，再检查 git status、tmux、实际进程、GPU和磁盘；不要重复启动已有队列。

当前任务未完成：用户授权把剩余126集2316shot按当前多人3D集成重跑，保留此前14集175shot。北京时间2026-10-31 23:07快照：本批15集158shot完整生成并完成官方非VLM；共生成175shot（包括run155的10/11shot和正在跑run654/i20_j41的7/10shot）。合计29集333shot完成重跑并评分，不能把本批175shot都说成评分完成。后台仍在运行：tmux qwen140_remaining126，入口pretest/qwen140_multiface_remaining_episode_batch.py，结果根outputs/entitybench_qwen140_multiface_remaining126_20261030/（R）。先看R/status.json、queue.log、index.html及实际进程，数据会随运行变化。

10月31日约15:09因下一集空间不足等待，20:05扩容320→350GiB后恢复；旧进程退出原因未证实，别归因于SSH。现在约36.80GiB空余；每集准入加25%余量并保留8GiB，不足时等待，不删除历史。预计原单卡9–12天需要GPU/API/存储连续可用，已有暂停使日期顺延，现扩容不保证够跑完全队列。

两项未解决问题：
1. run155/i5_j39的8:1，Agent lookup请求the yellow electric pokémon不在人物catalog中而KeyError；前10shot保留，20:05恢复又从缓存命中同错后继续其他集。看该集logs/generation.log、shot_8_1/agent/planning_inputs.json和lookup_call.json，核实别名/类型再设计修复，不随便映射身份或删请求。
2. 全140滚动汇总helper在旧规范视频软链接指向原版、新完成集需改指重跑版时触发assert。R/full140_aggregation_exception.json及logs/full140_snapshot.log留档；R/full140_snapshot/可能部分写入，validation还停留14集替换时，不能当最新一致快照。本批15集逐集评分与R/report.md、cohort_comparison.json有效。后续用独立版本快照及明确发布入口修复，不能改分数或原媒体。
另外run617/i75_j127的Jordan FaceLift因4通道图输入、VAE期望3通道而失败；asset_failures.json记录保留2D、该角色无可用3D、其他角色继续；该集已评分。不能把资产技术错误叫正常姿态门控，也不能声称所有角色注入成功。

本批15集配对（非全140）：角色出现0.9649→0.9919、人物区域cs_face 0.7618→0.7807、成像65.5121→67.1443、美学0.5968→0.6036、场景0.5914→0.6135；动态0.5165→0.4852、物体区域cs_object 0.8606→0.8283、边界0.9868→0.7564下降。完整人工视频审阅待完成，不作插件独立因果结论。

当前方案保留：qwen3.8-max Agent A/B/C路由、角色原句和身份绑定提示词、episode统一画风、干净单人参考及对应FaceLift Gaussian、全部检测脸先全局一对一匹配、逐角色v7原门控。A自身实际尾帧；B进入过渡＋行动固定1＋1，主行动用真实过渡尾帧；C重建起始帧，硬切不传前尾帧。原单人/多人3D都属于当前方案，不能取消3D。cosine0.4决定1/6步，不是匹配可靠性门控；不新增面积禁注入线。

用户授权正常门控跳过（包括整张无角色通过）沿用同次Control，不需确认；技术异常记录后继续其他集，全局API额度/认证异常停止新任务；禁止放宽阈值、换模型、重生挑高分或隐藏缺项。守护修复Agent明确关闭，不重新启用；普通队列和镜头路由Agent继续。当前没有自动修复行为。

冻结约束：R/cohort.json及各集face_injection_policy.json中冻结了19个源码SHA，本次交接核验一致。不要原地改shot_agent、路由提示词、重聚合helper或driver后直接续跑，否则后续校验失败；如修复，必须保留已有前缀、输入指纹和旧版，明确新代码/资产版本及独立输出，不能只改SHA记录。不要为了修汇总打断正在生成的shot。

评测保持官方strict/skip_vlm、原采样/阈值、零VLM评分调用。cs_face是跨镜头人物区域DINO，不是人脸识别分数；缺VLM保真度门控，scope partial_without_vlm。合法null不填0。使用evaluate-face-injection-v2 skill，完整episode评分后官方集等权聚合。API配置只进程内读取.env.portrait，禁止输出密钥。

原始完整140集2491shot在outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140/只读保留。此前14集替换的有效历史全量快照在outputs/entitybench_qwen140_partial_rerun_full140_20261030/，不是新29集全量快照。之前14集包含已留档代码版本差异，最终不能声称全140都从首shot同一实现。主文档只保留原始benchmark和当前多人3D两条主线。

R/handoff/snapshot_20261031.json是本次静态完成集清单和冻结源码检查，report_20261031.md为当时报告。实时结果看R及每集complete.json、episode_result.json、evaluation_validation.json。review/index.html及before.mp4、after.mp4可看前后视频；完整episode_id用于区分同run不同窗口。仅在确实无旧队列/工作进程、CUDA及冻结校验通过时恢复，具体命令在本次交接文档；已有合法结果自动跳过。

大量未提交代码/文档是已有工作成果，不reset/clean或自行推送Git；保留无关docs/frontend_agent_architecture.md、experiment_output/及中文绘图文档。首帧绘图skill及Clara原配置重放已完成，见docs/entitybench_firstframe_figure_skill_selection_20261030.md，不重复生成，Clara4:1无明显最终IF提升。

请继续用户已授权的后台实验，先报告实际进度和异常，再处理需要的后续工作；不要只输出计划或反复要用户确认。
```
