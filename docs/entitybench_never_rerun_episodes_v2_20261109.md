## 最新恢复：2026-11-09 22:09（北京时间），扩容后继续78集4步整集队列

用户明确“继续吧，我刚刚扩容了一下”。实查数据盘450GiB、可用约113GiB，覆盖此前约60～65GiB的新增估计；不设reserve、不删除旧产物。旧整集tmux已退出，状态停留run714/i20_j34的1/10shot完成、2:1首帧第8/20步，日志无退出异常尾部，不能断言是磁盘不足。已完成的人物3D资产、1:1视频及2:1 Agent/提示词/帧task全部保留。

恢复tmux仍为`qwen140_never_rerun_v2_20261109`，**实际新入口为 `pretest/qwen140_never_rerun_resume_20261109.py run`**，日志及范围仍`outputs/entitybench_never_rerun_episodes_v2_20261109/`。原78集/1218shot冻结plan、原生成与评测源代码不改。新恢复入口只绕开旧v1迁移器对“无成品帧”的自动archive/replan，直接安装同一v2并继续`run_remaining`，从2:1相同输入重算未完成的采样，不新查角色/重写Agent计划。17项已存结果与计划SHA绑定在`resume_after_expansion_20261109.json`，编译与v2/run/tag探测通过，预检API0。正常后续新shot仍按原流程调Agent与B过渡API；主视频仍4步、原多人v7门控，不恢复40步动态队列。

恢复前另一已生成8条的完整上下文评测已释放GPU，状态为official_full140_complete_default_dino_review_pending（新增7、累计35替换），不覆盖其结果，不把默认DINO未复核当完成。该评测的“暂停所有生成”交接是此前状态，**本轮用户明确继续78集整集生成为当前授权**，无新增动态40步授权。第一集实际恢复阶段以新status/progress/log为准，完成进度不重复清零。预计仍约5～6天，首集完成后校准。无守护Agent、无自动改模型/提示词/阈值、无推送。

## 最新：2026-11-09 21:53 北京时间，暂停40步动态实验，先跑从未开始整集生成的78集

用户明确切换优先级，并在异步回答中确认“沿用原整集4步配置”。这覆盖下方19:42的1277条40步自动执行授权：**动态队列已停，不自动恢复；此前28条完整替换版及所有历史媒体保留。** 停止时第一批8/1277条已全部生成、0条完成新评分，保存于`outputs/entitybench_original40_priority_remaining_20261109/cases/`，未作为新增增长替换；status已标记用户暂停。

新普通后台tmux `qwen140_never_rerun_v2_20261109`，入口 `pretest/qwen140_never_rerun_episodes_20261109.py run`；调度输出 `outputs/entitybench_never_rerun_episodes_v2_20261109/`，`plan.json`为唯一新范围，`inventory_at_launch.json`保留启动前证据。78集／1218shot均在冻结140快照中为original，原整集目录无shot目录、无rerun_manifest、未complete。其间允许此前独立shot实验；“未整集重跑”不等于从未生成过任何独立候选。run714只存在人物素材构建前缀，复用精确素材，尚无整集shot。另4个整集部分完成（run155、run53、run44、run237）本次不纳入；run445已有complete但未纳入11-06冻结快照，亦不重复。首次预检验证所有原冻结代码SHA、78/1218范围、v2编译器与run/tag绑定通过，API0。

沿用已经验证的v2_20261104 Agent提示词与身份绑定（非本轮运动强化词）、episode统一画风、全部B/C关键帧多人v7 3D、A自身真实上一尾帧。主视频 **Wan2.2-I2V-A14B LightX2V 4步／81帧16fps**；B首尾帧过渡仍为原有wan3.0-video API，本次未换本地首尾帧模型。3D正常门控跳过可沿用Control并记录，不改阈值，不取消3D，不做语义检查重试／自动降配／修复Agent。不修改旧冻结生成代码。新wrapper只选范围、串行调用原素材/episode/eval入口，原媒体仍位于`outputs/entitybench_qwen140_multiface_remaining126_20261030/experiments/<完整episode_id>/`。

最新启动阶段：run714/i20_j34（10shot）references模型已加载，正在准备/核验素材；尚无新shot出片，不把预检当完成。每集生成完整序列后跑官方strict/skip_vlm12项，VLM0；默认可见单脸＋定位纠错DINO仍需单独定位复核，不用raw代替。主体一致性与人物出现可能受路由改善，但不预设必涨。日志为新OUT/status.json、queue.log、stage_timings.jsonl以及各episode/logs/never_*。单卡串行，SSH可断开，机器须保持运行。初估78集约5～6天，首集完成后校准；约13GiB空余，全队列预计新增约60～65GiB，需扩容才能全跑完。不设磁盘reserve/人为额度、不自动删除；真实磁盘/认证/OOM等系统异常停队列记录，独立技术异常留档后继续其他集。旧整集大队列、动态实验都不另行启动。未推送。

完整78集名单见 `outputs/entitybench_never_rerun_episodes_v2_20261109/README.md`。
