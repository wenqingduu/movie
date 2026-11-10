> 最新评估政策：用户要求本轮不设指标质量门控，全部生成结果（含指标下降）均纳入对照，不按此前阈值过滤。只保留技术完整性检查；不自动替换正式benchmark。详见 `unfiltered_evaluation_policy.json`。

> 22:31:56更新：旧队列已正式暂停，保留230生成/198初评/11初筛结果；旧GPU worker和父调度均退出。新40步试验已开始加载模型，结束不会恢复大队列。

# 原版40步：Torch RoPE修正后重试（大队列保持暂停）

用户已于2026-11-08明确要求：低收益的大规模4步候选队列先不跑，只继续非蒸馏思路。此指令覆盖此前“试验结束自动恢复大队列”的授权。当前不会自动恢复或启动任何旧整集/1333候选队列。

## 当前执行

- 大队列父调度45449已STOP；关闭尚未执行的context评分waiter48699。当前worker93303只完成在途M0199–M0230批次最后两条，保存后退出；新执行器核对该32结果齐全后结束空闲父进程，并把旧status明确标为paused_by_user。
- 新tmux：`qwen140_original40_torch_rope_20261108`。
- 入口：`pretest/qwen140_original40_torch_rope_20261108.py schedule`。
- 新输出：`outputs/entitybench_original40_vs_distill4_torch_rope_20261108/`。
- 8条名单不变、5个4步动态0＋3个动态1参照。先N01/run323/2:1，真实81帧通过后其余7条，结束自动评分原6项VBench。
- 保留全部已有视频/评分、第一次失败目录、计划与日志；不改正式benchmark、不重新下载模型、不新建首帧或人物资产、不调用付费API。

## 第一次失败与本次修正

第一次原版试跑在20:41加载模型后进入首条推理，但customJSON未指定rope_type，WanSelfAttention默认走flashinfer_rope，而环境无flashinfer，首帧尚未产生就退出。未发生OOM、权重缺失或证明原版不能生成；原队列按当时计划已恢复，此历史行为现在被用户暂停指令覆盖。

本次仅显式增加两个配置，与原4步API已有默认值相同：

```json
{"rope_type":"torch_complex_rope","double_precision_rope":true}
```

不加载蒸馏LoRA、不用DMD2，仍是WanScheduler40步、CFG3.5、shift5、boundary=.9按timestep切换高低噪声、空negative；BF16、SDPA和所有8条PNG/提示词/种子/81帧16fps/尺寸不变。不是降低步数、关闭CFG或换模型来绕过失败。

CPU预检已真实实例化WanSelfAttention的TorchComplexRope，执行q/k旋转且与直接复数旋转逐元素一致，8条输入与第一次计划相同；证明在preflight.json。此时还不能把预检称为首条视频生成成功，实际GPU状态以status.json和first_case_validation.json为准。

## 结果与时间

状态：`status.json`；推理日志：`generate.log`；评分日志：`evaluate.log`；视频：`cases/N##/video.mp4`；完成后对照表：`report.md`及`comparison.json`。原大队列暂停回执：`large_queue_paused.json`。本执行器没有SIGCONT恢复大队列逻辑，即使40步测试报错也保持暂停、保存错误。

预计8条约2–4小时，需首条真实耗时校准，之前20:17的排队时间估计已失效。比较是两套适配推理配方的目的性8例，不是纯步数/蒸馏独立消融或完整benchmark结果；不根据个别成功直接替换。
