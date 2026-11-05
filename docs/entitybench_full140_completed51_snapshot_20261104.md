# 完整benchmark更新：替换51集快照

冻结时间2026-11-04 20:54（北京时间）：51完整重跑episode／988shot＋89原版episode／1503shot，合计140集2491shot，包含下降案例，不纳入未完成前缀。

默认人物一致性DINO0.7863→0.7943（上次替换41集后0.7920），131集有效、计分出现1380→1469。raw cs_face0.7155→0.7230；角色出现0.9212→0.9458；成像59.6517→60.8480；美学0.5817→0.5854；动态0.4853→0.4634，场景0.6303→0.6246，边界0.9822→0.8679。

- [完整13项表](../outputs/entitybench_full140_completed51_snapshot_20261104/report.md)
- [140集逐集CSV](../outputs/entitybench_full140_completed51_snapshot_20261104/per_episode_comparison.csv)
- [JSON及DINO覆盖](../outputs/entitybench_full140_completed51_snapshot_20261104/report.json)
- [原版／41集替换／51集替换趋势](../outputs/entitybench_full140_completed51_snapshot_20261104/comparison_original_41_51.json)
- [全部输入与规范视频映射](../outputs/entitybench_full140_completed51_snapshot_20261104/official/selection_manifest.json)
- [最终追加核验](../outputs/entitybench_full140_completed51_snapshot_20261104/post_aggregation_validation.json)

官方12项复用同配置输入验证缓存，由冻结官方实现按episode等权重新汇总；原all/easy/medium/hard完全复现。4607份媒体来源哈希、54份评分manifest、全部988shot裁剪及_eval SHA通过。未替换89集的全部13项差值均0；全部13项独立均值计算吻合。新10集318shot补CPU RetinaFace（589次前向，约6分钟），80项分数隐藏定位复查确认6项错误（N013/N038/N051/N053/N060/N079），之后重新计算质心；此前41集240项复查保持原判定。N079为Silas背部区域内把后方Nadia的脸算作目标可见脸；重复人物、身份/画风/衣服漂移和未决N001/N004/N036/N043/N062/N069保留。复核表及证据同目录new10_visible_region/review/。

入口：`pretest/evaluate_full140_completed51_snapshot_20261104.py`（prepare/official/final）、`pretest/evaluate_full140_new10_dino_20261104.py`（extract/score/final）；代码和验证/复核写入脚本快照在输出runtime，source SHA见evaluation_source_fingerprints.json。旧41集脚本、冻结19生成源码、原媒体及旧报告未改动，GPU队列继续。无新视频、付费API、官方评分或DINO特征推理。

范围为strict/skip_vlm、partial_without_vlm，VLM调用0；raw cs_face/cs_object缺少VLM保真度门控。默认DINO是可见单脸条件诊断，不是参考人脸身份准确率。完整视频视觉审核仍待完成。原run53因API IPInfringementSuspect未完整，继续沿用整集原版参与全140，不纳入重跑完成集。
