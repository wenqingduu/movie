# 30集最新评估交接

用户确认范围：30个已完成重跑episode、343shot；run594/i75_j98使用刚确认的3:4单Leo与3:5过渡＋主视频。其余29集使用清单中原有完整重跑版本，未扩展到140集。

[完整13项表](../outputs/entitybench_rerun30_latest_evaluation_20261101/report.md) · [30集逐集CSV](../outputs/entitybench_rerun30_latest_evaluation_20261101/per_episode_comparison.csv) · [核验](../outputs/entitybench_rerun30_latest_evaluation_20261101/validation.json)

入口`pretest/evaluate_rerun30_latest.py`；先prepare、extract、score，保存80项定位决策后final。已完成，不要重复启动。默认人物区域DINO由保存Gram重建等价特征，原官方raw逐角色/逐集重现后再过滤重算质心；无新增DINO推理、无VLM或收费API。原29集官方评分经视频、脚本、配置、权重、库、SHA与接续链核验后复用，新run594完整10shot重新评估。未运行首帧InsightFace/参考脸DINO消融。

主指标原版同30集0.8022→当前0.7991，所有30集均有效，保留区域262→291，计分出现202→237；本表原版是同30集的原完整benchmark子集，不是全140原0.7863。80个隐藏逐项分数的定位上下文均已检查，5项确认为另一个人物且目标同帧可辨认，75项保留。N014、N016、N056关联存在歧义，保守保留；生成身份/衣服漂移、重复、叠画不按低分删除。决策绑定manifest和原裁剪SHA。此为抽样定位复查，非全部角色定位或全片人工审阅。

交集补充为21个有效集、77次可计分出现，0.8064→0.8168；只是同角色同shot条件样本的辅助诊断，不能替代30集主结果或视为3D插件因果消融。

run594的evaluation_media_overrides.json明确两个新版媒体及正确注入记录；合成episode_result中的原route/任务字段只保留祖先记录，不能声称本次重新调用Agent或继续使用旧双Leo图。原视频、原评分、独立去噪图片包保持原样。当前无模型worker，守护修复Agent仍关闭。
