# 完整benchmark更新：替换41集快照

本次按请求时冻结的41个完整重跑episode／670shot替换原版，其余99集／1821shot不变；全量140集／2491shot。默认DINO0.7863→0.7920，角色出现率0.9212→0.9437，成像59.6517→60.4016；动态、场景和边界一致性下降。

完整13项表及全量逐集CSV：

- [报告](../outputs/entitybench_full140_completed41_snapshot_20261103/report.md)
- [140集CSV](../outputs/entitybench_full140_completed41_snapshot_20261103/per_episode_comparison.csv)
- [JSON与覆盖](../outputs/entitybench_full140_completed41_snapshot_20261103/report.json)
- [输入及官方验证](../outputs/entitybench_full140_completed41_snapshot_20261103/validation.json)
- [最终追加核验](../outputs/entitybench_full140_completed41_snapshot_20261103/post_aggregation_validation.json)

冻结时间2026-11-03 17:20、汇总完成约20:10（北京时间）。运行中的队列未修改；其后新增完成集留待下一快照。本次仅评估已有结果，不生成媒体、不调用付费API。官方12项复用同设置逐集缓存，原all/easy/medium/hard聚合精确复现；默认DINO补新6集150shot的CPU RetinaFace（177.8秒），80项隐藏分数定位审核，排除6项后重算质心，其余保留。此前35集的相同输入与复查结果复用。默认DINO131/140有效，计分出现1380→1417；是条件诊断。

入口：`pretest/evaluate_full140_completed_snapshot_20261103.py`（prepare/official/final）、`pretest/evaluate_full140_new6_dino_20261103.py`（extract/score/final）；冻结清单在报告目录cohort_snapshot.json。新6集裁剪/原_eval及全部670shot缓存的crop SHA、评分SHA、审核清单SHA绑定再次核验；99个未替换集所有13项差值均0。原版媒体、旧报告及冻结源码未修改。

采用[evaluate-face-injection-v2](/root/.codex/skills/evaluate-face-injection-v2/SKILL.md)。strict/skip_vlm，VLM0，raw cs_face/cs_object未加VLM保真度门控；不声称新版140集统一重跑或单独注入消融。
