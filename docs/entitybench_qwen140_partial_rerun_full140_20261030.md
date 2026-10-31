# 已完成重跑替换后的完整140集评估

用户要求按当前已重跑结果重新评估全部episode。已完成输入核验与官方全量重新聚合，输出独立目录，不覆盖原始benchmark。

- 范围：140集2491shot，原split保持easy/medium/hard=80/40/20。
- 替换：run700多人版10shot＋首批weak4的48shot＋focus4/extra5九集117shot，共14集175shot；其余126集2316shot沿用原版。
- 以完整episode_id替换整集，不按run简称合并窗口，不按得分选择新旧版本，下降结果也纳入。run700与weak4存在已记录的版本前缀，不能称为统一新版本全140实验。
- 逐集评分复用已验证同一代码/配置/模型/库的官方结果，本次无新神经推理或媒体生成。VLM调用0，保留partial_without_vlm与缺失值。
- 冻结官方解序列化器和聚合器重算完整split；原始all/easy/medium/hard报告逐项精确复现；2491个原始/所选视频、175个新尾帧及父链、规范路径核验通过。17份评分manifest一致，共2981个唯一媒体/脚本来源核验哈希。
- 140行逐集CSV已生成；126个未替换episode的所有有效指标差值均为0。边界指标仅137集有适用值，沿用官方分母。

[完整12项前后指标](../outputs/entitybench_qwen140_partial_rerun_full140_20261030/report.md) · [14集前后视频入口](../outputs/entitybench_qwen140_partial_rerun_full140_20261030/index.html) · [140集逐集CSV](../outputs/entitybench_qwen140_partial_rerun_full140_20261030/per_episode_comparison.csv) · [核验记录](../outputs/entitybench_qwen140_partial_rerun_full140_20261030/validation.json)

这是部分重跑后的当前混合全量快照。只替换10%的episode，局部改善对全量均值的影响有限；不能外推成全部episode统一重跑后的效果。完整人工视频审阅待完成。

复现入口：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/root/autodl-tmp/envs/entitybench-eval/bin/python \
  pretest/qwen140_reaggregate_partial_reruns.py
```

源数据、原始140集报告、14集各自评分以及此前九集联合报告均保留。未启用修复Agent或额外收费API。
