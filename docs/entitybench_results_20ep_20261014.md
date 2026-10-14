# 当前20集三路线结果汇总（2026-10-14）

三路线各20个完整episode／363视频，生成失败0；旧SDXL→FLUX、Qwen、新摄影版FLUX已完成官方非VLM评分及独立核验。
FLUX全140仍paused_by_user；Qwen全140仅有[耗时估算](entitybench_qwen21_full140_estimate.md)，未启动。
以下数值从已完成原始报告及脸部版本report.json读取，不覆盖原报告。完整视频、权重、PNG、原始JSON与查看页只保留在本机outputs，不随Git推送。

## 原人物区域官方非VLM结果

固定作者revision `3776439e7037b2128ac3b5d2d1ce63e424cca166`、strict skip_vlm适配，按完整episode官方宏平均聚合。
Qwen／新FLUX使用两路评分；旧FLUX正式评分串行，v2重新聚合已核对一致。
六项VBench及presence／DINO／边界保持原公开代码口径；VLM调用0，39项缺失null，fidelity gate不可用。
论文140集／2,491镜头；项目20集／363镜头且素材、视频条件不同。原人物定位污染、边界gate与聚合差异见[指标口径审计](entitybench_metric_provenance_audit_20261014.md)。
**下表仅作初步观察，不能据此与论文直接排名或证明3D插件因果增益。**

| 指标 | EntityMem论文 | StoryMem论文 | HoloCine论文 | CineTrans论文 | SDXL→FLUX | Qwen | FLUX→FLUX |
|---|---:|---:|---:|---:|---:|---:|---:|
| subject_consistency | 0.8810 | 0.7590 | 0.8600 | 0.9680 | 0.9281 | 0.9388 | 0.9293 |
| temporal_flickering | 0.9760 | 0.8380 | 0.9570 | 0.9790 | 0.9810 | 0.9859 | 0.9803 |
| motion_smoothness | 0.9880 | 0.8490 | 0.9640 | 0.9900 | 0.9895 | 0.9919 | 0.9893 |
| dynamic_degree | 0.6570 | 0.5620 | 0.7210 | 0.6880 | 0.5930 | 0.4866 | 0.5999 |
| aesthetic_quality | 0.5930 | 0.4750 | 0.5180 | 0.5960 | 0.5772 | 0.5690 | 0.5814 |
| imaging_quality | 66.0000 | 56.4100 | 49.9700 | 68.5700 | 63.0349 | 60.2959 | 63.8872 |
| intra_character_presence | 0.9670 | 0.8490 | 0.8820 | 0.7960 | 0.8649 | 0.9335 | 0.8391 |
| intra_object_presence | 0.8880 | 0.8930 | 0.7230 | 0.7760 | 0.8685 | 0.8710 | 0.8742 |
| intra_location_presence | 0.6870 | 0.6810 | 0.6240 | 0.6510 | 0.7032 | 0.6279 | 0.7121 |
| cs_face | 0.7370 | 0.7920 | 0.7510 | 0.7720 | 0.6853 | 0.6958 | 0.6836 |
| cs_object | 0.7980 | 0.8390 | 0.8030 | 0.7940 | 0.7964 | 0.8186 | 0.8038 |
| cs_transition_boundary | 0.7380 | 0.6630 | 0.4980 | 0.5080 | 0.9884 | 0.9820 | 0.9882 |

论文数值来源：[arXiv v1 Table4及附录F.1](https://arxiv.org/html/2605.15199v1)。
原cs_face为人物区域DINO质心指标，仍为暂定值；cs_transition_boundary近1对应本路线末帧I2V保留，不能解释为整体故事一致性。

## 用户要求的只裁脸版本

本地RetinaFace只检测脸；脸框加10%边缘、原五采样点、不对齐；DINOv2单位CLS特征、角色质心公式和episode宏平均沿用。
角色仅作空间匹配，同脸轨迹被多角色竞争或归属不确定时记缺失。没有身份级角色真值，不把匹配覆盖率说成人脸检测召回率。
21页／1,002候选脸图全部目视核验，2张相机严重遮挡无可用脸部内容排除；保留模糊、侧脸、人物／画风变化。

| 路线 | 脸部版cs_face | 有效集数 | 可匹配角色-镜头／调度 | 跨镜头appearance | 共同样本cs_face |
|---|---:|---:|---:|---:|---:|
| SDXL→FLUX | 0.8072 | 20／20 | 285／726（39.26%） | 239 | 0.8196 |
| Qwen | 0.8225 | 20／20 | 396／726（54.55%） | 365 | 0.8323 |
| FLUX→FLUX | 0.7907 | 19／20 | 319／726（43.94%） | 283 | 0.8119 |

共同角色／镜头交集164项；至少两次appearance后，实际共同评分为16集／126项。
新FLUX的run542窗口i35_j49缺少同一角色至少两个有效脸部镜头，保留null，有效评分19集。
脸部版本改变输入与样本集合，单独保存；不能替代原人物区域cs_face或直接与论文表格横比。详见[执行与冻结协议](entitybench_face_only_dino_20ep_run.md)。

## 核验及来源

三路线全部1,089视频、2,178调度角色-镜头和1,002候选脸图来源核验；60episode脸部评分独立复算，官方反序列化／聚合复核一致。
原3份官方报告与归档SHA保持；全部516接续输入等于自身父视频实际解码末帧。全量人工审片未完成，抽样视觉检查另存。

| 原官方报告 | SHA256 |
|---|---|
| flux_sdxl20 | `35e2b1ce3a39f68be74f53b7c2526dbece6c54390c07d6063b16dc7f9606a8ba` |
| qwen20 | `6ed4d81d7aa575830fd101bd9701f621d4dd2bad77cf6a808ee803df001a4e86` |
| flux_native20 | `f9c6e85e604e71f64dd1a07abc7c46c928237cd3087b8923f9ed67a6cb90e5fa` |

脸部report.json SHA256：`42baf3742d5464e73447532e23ef139c1473c471efeca88460b9ca6b03ca6ad3`。

本机查看：

- [三路线／论文完整归档及视频查看页](../outputs/entitybench_paper_flux_qwen_portraitflux_photo20_comparison/README.md)。
- [脸部版本完整结果及裁剪查看页](../outputs/entitybench_face_only_dino_20ep_20261014/README.md)。
- 定位修正人物区域中间版本保留在`outputs/entitybench_cs_face_corrected_character_crops_20ep_20261014/`，与脸部版本分开。

GitHub仅有本汇总与复现代码；上述outputs链接用于原运行机器，远端没有媒体文件。
