# 夜间实验实际结果：2026-10-30

> 14:08更新：用户充值并授权继续，已启动新监督队列；实时状态见[充值后续跑](entitybench_recharge_resume_20261030.md)。以下保留本轮续跑前的夜间失败记录。

## 10月30日最终核查（北京时间）

夜间没有全部跑完。API返回HTTP 429，错误类型/code为`insufficient_quota`，正文明确为月度token套餐额度耗尽，不是可通过短暂退避解决的普通限流。规划与修复Agent使用同一接口，修复调用也失败，未发生成功的自动修复。GPU未崩溃；监督器于03:30结束，目前没有实验进程或tmux，GPU空闲。中午截止已过，本次核查没有重启队列。

| 队列/episode | 完成shot | 完整官方评分 |
|---|---:|---|
| focus4：run317 / i25_j60 | 15/15 | 12项非VLM完成 |
| focus4：run15 / i0_j19 | 10/10 | 12项非VLM完成 |
| focus4：run354 / i90_j105 | 6/15 | 未完成，4:1素材查询因额度停止 |
| focus4：run348 / i30_j49 | 0/15 | 未完成，首次规划因额度停止 |
| extra5：新增5集 | 0/62 | 全部首次规划因额度停止；已生成素材保留 |
| Clara原配置绘图首帧 | 2/2首帧 | 两帧Control及Treatment均与原图逐像素一致 |

focus4实际已有31/55个shot，状态中的completed_shots=25只统计两集完整episode。配对聚合只包含run317和run15，不计run354有效前缀，不冒充四集或全140集结果。相同两集：人物出现率0.7278→1.0000，cs_face 0.6864→0.7140，美学0.5817→0.6014，成像60.6950→66.0211；动态0.2833不变，场景出现率0.4000→0.3333，边界0.9851→0.8438，仍非全面改善。VLM调用0，完整人工视频审阅待完成。

- [两集前后视频及完整指标](../outputs/entitybench_qwen140_multiface_focus4_20261029/index.html)
- [两集配对报告](../outputs/entitybench_qwen140_multiface_focus4_20261029/report.md)
- [Clara逐步去噪素材](../outputs/model_figure_clara_denoising_replay_20261029/index.html)：4:1和5:1各82张主轨迹图，另有注入前后解码及原始张量；02:23完成。
- 证据：focus4与extra5各shot下`agent/lookup_call.error.json`、中央`state.json`、Clara两份`complete.json`。有效视频、素材与3D均保留。

恢复执行需要先恢复配置API可用额度，再重新确定执行时段；原授权截止时间为2026-10-30 12:00北京时间。接口返回预计11月13日16:00 UTC重置（北京时间11月14日00:00）。不要无限重试、自动换模型或把未完成项计为零分。
