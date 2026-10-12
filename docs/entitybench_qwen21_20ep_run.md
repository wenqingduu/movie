# Qwen：相同20episode的最终路线实验

2026-10-12北京时间21:03用户要求停止FLUX全140扩展，改跑已有官方评测的同一20episode。
新Qwen队列21:12启动。只生成最终一组视频，不追加无插件视频组，暂不扩展到140episode。

## 范围与基准

严格复用`outputs/entitybench_pulid_i2v_a14b_4step_continuation_20ep/split_20ep.json`
和manifest内20个完整episode_id。363shot=191硬切+172接续，原cut、顺序、视频prompt/seed保持。
同run不同窗口分开目录。FLUX20episode已有官方非VLM报告保持只读，无需重跑。
旧正式评分为串行，v2缓存重新聚合已核对一致，22视频的v2两路推理也已核验；新Qwen用v2并发。
评分算法和范围一致，不能把旧项目身份指标当作这组官方基准。

FLUX全140已按要求停止，状态`paused_by_user`，新增标定资产144/217完成，全部保留。
停止时两个未完成角色不算完成；旧20episode资产和363视频已完成并保留。
新Qwen人物素材改变，对应Gaussian/标定要重新准备，不能按名字复用旧资产。

## 固定路线

- 本机Qwen-Image-2.1，revision `790c92633540aa0cb11d9abf19eb46d861714758`，BF16；
  素材阶段CUDA驻留，首帧仍sequential CPU offload（验证细节见文末）。
  启动重新核验7份权重SHA，并冻结代码和本地Diffusers实现。
- 为硬切所需112个episode-local角色生成人物素材，另19个仅接续角色不额外生成参考。
  原131个角色及全部363镜头仍保留。按原描述和故事风格生成，不统一成年/写实/人形。
- 素材1024×1024、40步；40个单角色硬切角色各3候选，其余72角色各1候选，共192候选。
  固定seed，按唯一可靠脸、近正面姿态和面积选择，不能按正式评测分数挑身份。
  三候选均无可靠单脸时保留基础参考和资格缺失记录，不删镜头。
- 对符合资格的单角色硬切素材做FaceLift与273点/512姿态标定，两路有界CPU/ONNX线程。
  现有资产助手只调用`facelift`、`calibrate`，不会重新走SDXL参考阶段。
- Qwen硬切首帧1024×576、20步；单角色沿用官方Control共享latent分叉，step12 pred_x0
  准备3D/mask，v7强度0.4，cosine阈值0.40选择1/6步。内部Control是已验证实现所需轨迹，
  只把最终Treatment送入视频；不生成消融视频组，新入口关闭附带InsightFace配对评分。
- 多人硬切用原生多参考，不新增多人3D注入；子集最多7角色，低于官方10图上限。
  非人脸、缺脸、闭眼或空mask保留基础结果及原因，资源/文件故障须中断而非伪装成回退。
- 接续只用本路线前一段新视频真实解码尾帧，不编辑、不再注入、不借用FLUX尾帧。
- A14B与基准同配置：LightX2V high/low 4step_1022 LoRA、DMD2四步、shift5、CFG关、
  BF16/torch_sdpa、81帧/16fps/480p面积。视频prompt/seed与FLUX对应shot一致。

## 执行、进度与续跑

当前入口`pretest/run_entitybench_qwen21_20ep_accelerated.py`，tmux `entitybench_qwen21_a14b4step20`。
原入口和核心参考/注入实现保持冻结；新入口只重定向素材阶段的模型放置方式。
设置remain-on-exit，判断是否运行须检查pane_dead。
输出`outputs/entitybench_qwen21_i2v_a14b_4step_continuation_20ep/`。
启动日志`outputs/entitybench_qwen21_20ep_pipeline.log`，阶段日志在新输出`logs/`。

自动执行：固定子集/seed→权重核验→Qwen素材→新FaceLift/标定→硬切首帧→冻结manifest→
363段A14B→363来源/172接续审计→20成片→v2两路评分→与已有FLUX报告比较。
图像/FaceLift/视频分阶段加载，不与已停FLUX队列并跑。

整体阶段见`status.json`，素材见`references_status.json`，3D见`asset_status/`及
`asset_preparation_report.json`，硬切见`firstframes_status.json`，视频见`generation_report.json`，
评分见`evaluation/official_parallel_v2/parallel_status.json`。
缓存必须匹配输入、模型、代码、素材及Gaussian/标定SHA；Qwen与FLUX/旧三episode产物隔离。

启动前约18.5GiB空间，没有重复下载权重。生成阶段保留至少5GiB守护；
20episode产物也需观察实际增长，不保证空间必然足够，不自动删除历史实验。
首个已保存素材候选107.3秒；只代表素材阶段一个样本，不能用来估整条路线完工时间。

## 官方评价与最终比较

使用[evaluate-face-injection-v2](/root/.codex/skills/evaluate-face-injection-v2/SKILL.md)，
同一官方revision `3776439e7037b2128ac3b5d2d1ce63e424cca166`及skip_vlm适配。
完整episode按两进程、每路4线程分片评分，独立缓存，最后一次官方聚合。
主表六项VBench、角色/物体/场景presence、cs_face、cs_object、真实接续边界。
VLM仍跳过，gate不可用，跨shot分数暂定，scope=`partial_without_vlm`。
采用公开代码episode宏平均，旧InsightFace/参考DINO/脸框质心不填入官方表。

官方报告位于`evaluation/official_parallel_v2/report_qwen21_native_reference_a14b_4step_continuation_20ep_no_vlm.json/.md`。
`official_comparison.md`列all/easy/medium/hard的两路线均值、差值和有效episode数；
`official_comparison_sources.json`保留原始官方聚合报告及SHA。
素材与首帧系统同时改变，差值只解释为最终路线效果；消融沿用之前实验。
人工审查单人、多人、回退、接续及漂移，与指标完成分别记录，不预设新路线一定改善。

4项新增子集/冻结/种子/资格检查、25项A14B协议检查、9项既有Qwen产品检查通过；
新入口编译通过。Qwen既有插件诊断及负增益事实见[历史与计划](entitybench_qwen21_next_experiment_plan.md)。

## 2026-10-12 22:59：素材阶段采用GPU驻留

用户同意测试减少offload与两路并发。在独立
`outputs/entitybench_qwen21_reference_acceleration_probe/`复算已生成的Julian/Viktor两个固定候选；
测试期间暂时SIGSTOP素材进程，测试入口finally恢复，之后按验证结果重启正式队列。

| 模式 | 两张图推理总耗时 | 冷启动进程墙钟 | PyTorch显存峰值 | 输出核验 |
|---|---:|---:|---:|---|
| 原逐层sequential offload，既有候选 | 约211.9秒 | 不作冷启动比较 | 低显存逐层搬运 | 基准 |
| 单进程model offload | 105.57秒 | 107.01秒（不含Python导入） | 约16.5GiB allocated | 两图PNG SHA完全相同 |
| 单进程全部CUDA驻留 | 42.11秒 | 55.73秒（含权重搬到GPU，不含导入） | 约36.9GiB allocated、37.8GiB reserved | 两图PNG SHA完全相同 |
| 两进程model offload，各生成一图 | 每进程约81.6–81.8秒 | 两子进程整体89.41秒（含导入/初始化） | 各约16.5GiB allocated | 两图PNG SHA完全相同 |

两路model offload比其单进程稍有吞吐改善，但单进程CUDA驻留更快。
正式素材阶段采用单进程CUDA驻留：约21秒/张，相对原候选约5倍；是两样本推理比较，
不保证全量或整条视频路线获得5倍加速。40步、权重、prompt、seed、KV-cache均保持。
29份完整角色素材及已有候选先校验复用，未完成候选从同seed重跑，不覆盖有效既有结果。

新素材执行入口`pretest/run_qwen21_gpu_resident_references.py`直接复用原冻结
`references()`函数；原始core源文件/模型SHA及preparation_plan不变。
`runtime/reference_acceleration.json`记录新入口/worker SHA、原implementation SHA及性能报告SHA，
明确仅素材阶段改变设备放置。后续首帧仍用已验证的sequential offload，标定仍两路，
视频和官方评价不变；多参考首帧GPU驻留还未测，不直接套用该结果。
新增2项重定向/输出一致性准入检查通过；原队列已恢复到同名tmux。
