# 低指标四集：兼容修复与续跑（2026-10-29）

首轮队列于北京时间10月29日01:44左右结束，四集均记录技术异常，未完成整集评测；实际已生成22/48个shot，不能称为没有生成结果或已经跑完。按用户“不等确认、记录异常、尽量跑完”的授权修复后续跑。

| episode | 已完成shot | 首轮中断原因 | 修复 |
|---|---:|---|---|
| run771 / i15_j24 | 6/11 | 2:6调度含`the dark suit`，实体描述表没有该键 | 仅对缺失的场景/道具描述使用原调度名称，并记录缺失；不编造描述 |
| run455 / i0_j24 | 8/15 | 2:2调度含`SearchO`，实体描述表没有该键 | 同上 |
| run484 / i15_j47 | 0/12 | 旧窗口没有episode级`base_seed` | 用该窗口原首shot的seed作为参考素材种子基数；该集尚无已生成素材 |
| run1404 / i170_j220 | 8/10 | 5:1查询3人参考，但首帧可见名单只有Kenji、Ryan | 保留原Agent计划；按可见名单投影帧参考，重排图像编号及帧提示词引用，主视频仍保留Chloe描述 |

22个已完成shot的result/video/tail SHA均验证并保持不变；人物参考、Gaussian、注入参数、路由和已生成媒体不重做。修复只作用于后续未生成节点，5:1复用已缓存的Agent计划，不重新调用规划API。三项绑定回归测试和48个shot的描述/种子预检通过。

旧cohort、冻结源码、异常与聚合记录保存在：

`outputs/entitybench_qwen140_multiface_weak4_20261028/runtime/revisions/data_compat_20261029/`

已建立素材的三集保留原`face_injection_policy.json`，后续通过`active_face_injection_policy.json`指向版本化兼容策略；不修改原有task或结果记录。run484首次生成使用新版本策略。

恢复时环境无`/dev/nvidia*`，PyTorch返回CUDA不可用、device_count=0。tmux队列会等待CUDA恢复，再自动从有效前缀继续；不因缺GPU把所有episode再次标成失败。原结束时间估计作废，GPU恢复后预计尚需约3–5小时，以API和实际进度为准。

主状态、日志、视频与指标入口仍在`outputs/entitybench_qwen140_multiface_weak4_20261028/`。门控全跳过继续使用同次Control，技术异常记录并继续其他episode；缺失整集不会填0或被冒充完整配对结果。Git推送不含outputs里的本机媒体与运行记录。
