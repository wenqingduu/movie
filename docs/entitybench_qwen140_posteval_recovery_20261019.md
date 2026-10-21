# Qwen140：最终核验恢复与完成记录

2026-10-19北京时间21:57:23已完成官方非VLM、脸部版DINO及归档，原tmux正常退出0。
全量主状态为 `complete_non_vlm_pending_full_video_visual_review`。后文故障／接续步骤为保留的恢复记录，勿据此重复启动。
总报告为输出根目录 `report.md`，完整结果及范围见 `archive/completion_20261019.md`，视频入口 `archive/viewer.html`。

2026-10-19北京时间21:04:24，原协调器在官方评分后的最终核验退出1。
视频已全部完成：140集／2491镜头，其中2128新增、363核验复用，生成失败0。
媒体和1355项实际尾帧接续核验通过；两进程官方非VLM评分用时16485.84秒，140集结果均已保存。

## 原因和独立修复

固定官方聚合器对没有任何可评分episode的指标输出
`{"value": null, "n_episodes_evaluated": 0, "n_episodes_total": ...}`。
有有效结果时才输出 `mean/std/min/max`。原最终核验把39个未运行VLM指标按 `mean` 读取，因此触发 `KeyError: 'mean'`。
另外，原skip_vlm适配器允许逐集JSON直接省略未运行的VLM指标；官方反序列化和聚合本身已支持这种缺失。

新增独立核验器 `pretest/audit_entitybench_qwen140_final.py`：

- 严格核对39项aggregate空值及各难度的覆盖分母；拒绝缺字段、非null值和错误覆盖计数。
- 原helper重验完整split、脚本／canonical视频映射及SHA、输入fingerprint、两个worker清单与原报告。
- 逐集六项VBench完整覆盖、零失败，以及2491项原实体schedule核对。
- 用原官方反序列化器和聚合器重算140集，逐tier结果与原报告完全一致。
- 保留逐集实体／接续指标的失败／跳过／缺失计数；核对39项VLM未评分、无密钥及调用0。
- 复核原独立尾帧解码核验的输入SHA，并重新检查全部媒体及父尾帧依赖；不重复模型推理。

原冻结协调器／生成代码／评测器未修改，原评分JSON／Markdown和全部媒体不改。
4项新回归检查、3文件Python编译和shell语法检查通过。修复后的真实140集最终核验已通过。
正式证据：全量输出 `audit/final_official.json`；新日志 `logs/official_audit_schema_recovery.log`。

## 接续

独立入口：`pretest/resume_entitybench_qwen140_posteval.py`。
用户回复“继续”后，已在原tmux `entitybench_qwen21_full140` 接续，仅执行最终核验→脸部裁剪→真实审阅→脸部评分→归档。
入口仍持有全仓GPU锁，保留10GiB磁盘守护，离线运行且不传VLM凭据；不重复生成视频或官方评分。

本机启动脚本为全量输出 `runtime/resume_posteval_20261019.sh`，最新日志 `logs/posteval_resume_20261019_attempt2.log`，第一次启动日志 `logs/posteval_resume_20261019.log`保留。
恢复源码和已完成产物的SHA保存在 `runtime/posteval_recovery_20261019.json`；
原失败状态及日志保存在 `runtime/posteval_before_20261019/`，启动记录为 `runtime/posteval_resume_launch_record_20261019.json`。
后续查看 `logs/face_extract.log`、`evaluation/face_only_dino/status.json`、主status及实际进程，勿重启原全流水线入口。

第一次脸部提取在首个视频解码前发现ASSET环境缺少 `imageio_ffmpeg` 插件，未产生crop。
已从本机原评测环境复制现有 `imageio-ffmpeg 0.6.0` 和其FFmpeg 7.0.2静态二进制至ASSET环境，无下载。
两环境在一个硬切和一个接续视频上各解码81帧，原五采样点共10个像素hash完全一致，首尾帧像素与保存媒体完全一致。
14个依赖文件逐个核对SHA，加入恢复清单；原清单先归档为 `runtime/posteval_recovery_20261019.before_decoder.json`，既有受保护项保持。
依赖来源和验证见 `runtime/posteval_decoder_dependency_20261019.json`、`audit/decoder_validation_20261019.json`；第一次提取故障已归档，第二次启动记录为 `runtime/posteval_resume_launch_record_20261019_attempt2.json`。

脸部提取仍使用原冻结RetinaFace、五采样点、10% padding、空间匹配和DINO公式。
实际提取完成后，用view_image查看全部51页、2428个裁剪；排除2个被不透明相机遮挡而没有可用脸区的错误定位。
侧脸、遮挡、头盔、动画／非人类脸及身份／风格漂移按原规则保留，没有按分数过滤。
实际查看页与最终审阅页逐文件SHA完全一致，2428项crop SHA和ID集合全部绑定manifest；评分前冻结 `review/decisions.json`。
审阅证据为 `evaluation/face_only_dino/review_work/visual_checks.json`、`review/visual_review_validation.json` 和 `frozen_review.json`。

## 脸部评分核验与归档完成

原脸部提取耗时807.79秒；2426个有效脸部特征与140集分数均已生成。
原独立核验把官方四位小数均值与未舍入均值按1e-6比较而退出：原始集均值0.82370728686，官方发布0.8237；评分本身完整。
新增 `pretest/audit_entitybench_qwen140_face_final.py` 逐项校验单位向量、角色质心、140集原始值、覆盖／失败／跳过计数，使用原官方反序列化器和聚合器重算并精确比较全难度结果，按原四位小数规则检查发布均值。
新核验通过：有效2426／4991角色-镜头（48.6075%），2124个appearance进入至少双镜头角色的质心评分，有效140／140集，脸部DINO **0.8237**。
缺失原因保持：角色区域多脸歧义379、同脸角色冲突1692、无空间匹配336、无最小尺寸脸156、审阅排除2；不填0，不转作失败。
证据为 `evaluation/face_only_dino/independent_validation.json`；原脸部报告SHA `38a33147bf77bd3a85b6209b4a1b2cfcd4cfb1398c801781e70f4974e0767536` 未改变。
最终入口 `pretest/finalize_entitybench_qwen140.py` 先验证恢复契约、旧产物SHA及新源码，再核验并调用原归档函数。
最终日志 `logs/finalize_precision_20261019.log`；新增冻结契约 `runtime/face_audit_precision_recovery_20261019.json`，原失败现场 `runtime/face_audit_before_20261019/` 保留。
7项回归检查和新增源码编译通过；原冻结生成器／评测器、旧20集及完整官方结果均保持。

官方 `cs_face/cs_object` 仍为无VLM／无fidelity gate的暂定人物区域指标；脸部版单独保存。
只做单路线绝对表现，裁剪定位审阅及代表性联系表检查不等于逐片人工质量审查；全片审查仍待做。
恢复代码、相关测试和交接文档纳入本次提交范围，实际提交与推送状态以Git核对为准；原并行改动及所有历史产物保留。
