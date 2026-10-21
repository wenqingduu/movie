> 历史快照（2026-10-21归档），不是当前执行指令。原文件：`docs/LLM_HANDOFF_QWEN140_20261014.md`。
> 当前实验见[主交接](../../LLM_HANDOFF.md)；归档前原文SHA256：`51552099a314d5a574753db35ef1e0074c027d32e8b69e56923cebaf39c0db2b`。

# Qwen140执行交接

更新：2026-10-19（北京时间）。当前方向和固定参数以[主交接](../../LLM_HANDOFF.md)为准；本文只列执行与核验步骤。
历史模型组合／进度／清理前容量见[历史索引](README.md)，当前任务无需通读。

历史准备核验、准入及逐图裁剪决策要求见[准备检查](../entitybench_qwen21_full140_preparation_check.md)。
该文记录2026-10-14的未启动状态；当前运行阶段以本页最新快照、实时JSON及实际进程为准。

## 现状与资源

2026-10-19 21:57：140集／2491视频、官方非VLM、脸部评分及归档均完成；生成失败0，1355实际尾帧依赖核验通过。
主状态 `complete_non_vlm_pending_full_video_visual_review`，原tmux正常退出0，勿重启生成／评分。
官方人物区域cs_face为0.7155；独立脸部版为0.8237（140集等权），51页2428裁剪实际审阅后排除2个错误定位，有效2426／4991角色-镜头。
六项VBench／39项VLM缺失及零调用均核验通过；无fidelity gate，官方cs_face/cs_object仍暂定。全片人工质量审阅待做。
最后独立入口 `pretest/finalize_entitybench_qwen140.py`，日志 `logs/finalize_precision_20261019.log`。
原冻结代码、分数和媒体保持；空指标格式、FFmpeg本地依赖及四位小数核验修复详见[恢复与完成记录](../entitybench_qwen140_posteval_recovery_20261019.md)。

2026-10-17 11:05快照：711参考／1163候选、208份有效3D／标定及945新增硬切首帧全部完成。
原tmux在08:26进入视频生成，新增98／2128、失败0，旧20集363视频继续核验复用。
视频实时日志 `logs/generate_formal.log`；协调器日志 `logs/resume_routing_20261016.log`，不要因主日志暂不更新重复启动。
用户已要求本轮全量代码、恢复适配器、测试及交接提交推送；原核心与生成中任务保持，提交／远端以git检查为准。

2026-10-16再次接续：Blaze（斑点狗）在13:44:58因人脸标定样本不足致队列退出1。
已按既有9项非人脸角色规则纠正Qwen参考metadata资格，原参考／候选／选择fingerprint／冻结核心保持；所有镜头仍保留。
有效3D任务208个，接续前111个可复用、97个待补；原tmux已使用 `runtime/resume_routing_20261016.sh` 恢复。
优先检查 `logs/resume_routing_20261016.log`，入口 `pretest/run_entitybench_qwen21_full140_resume_routing.py`；后续不得重复启动旧RGBA入口。
续跑磁盘按原40GiB产物总预算扣除本轮已有13.656GiB再留10GiB核验，本次46GiB可用通过；详见[分流恢复](../entitybench_qwen21_full140_routing_recovery_20261016.md)。
14:59:33已确认两份新资产完成且Gaussian／标定SHA绑定通过，实时113／208、失败0，队列继续后续角色。

2026-10-16最新接续：Annelise的RGBA→VAE通道格式故障已独立兼容修复并实际通过3D构建；711参考／1163候选和原67资产保留。
原tmux `entitybench_qwen21_full140` 使用 `runtime/resume_rgba_20261016.sh` 补原规格标定并恢复完整流水线，入口 `pretest/run_entitybench_qwen21_full140_resume_rgba.py`。
看 `logs/resume_rgba_20261016.log` 和实际进程；原失败现场已归档，核心／参考／参数不变。详见[故障与恢复](../entitybench_qwen21_full140_rgba_recovery_20261016.md)。

2026-10-15更新：用户已授权开始，GPU／CPU／内存已恢复；tmux `entitybench_qwen140_preflight` 只读预检退出0。
三个真实Python环境CUDA检查、完整旧20集复用及全部模型SHA重算通过；7项适配器检查通过。
用户已对50GiB预算准入回复“开始”，完整队列已在tmux `entitybench_qwen21_full140` 启动；启动时59.746GiB可用，运行中仍保留10GiB。
启动脚本为全量输出 `runtime/launch_full140_20261015.sh`，日志 `logs/pipeline_20261015.log`；来源复核及已有视频并发验证后自动接续其余阶段，脸部真实审阅等待规则保留。
详见全量输出 `audit/prelaunch_20261015.json`。旧GPU不可见快照及下文未启动记录属于历史状态；其他模型／参数／保留边界继续适用。

全量入口已存在：`pretest/run_entitybench_qwen21_full140.py`。
输出：`outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140/`。
2026-10-14 22:06:27的`status.json`为`prepared_waiting_resources`；计划已准备，尚未开始全量生成。
该状态的admission来自清理前，记录GPU不可见、CPU配额0.5核、内存上限2GiB及空间不足；不是生成失败。
接手重新核对tmux／进程、GPU／CPU／内存、`admission.json`、`preparation_plan.json`和阶段日志；不重复启动。
用户确认GPU尚未准备好，当前先做准备与只读核验，资源未满足时不运行生成或GPU评分、不下载新模型。

清理后数据盘约59.75GiB可用、系统盘约9.30GiB可用；新增产物预算30–40GiB，至少保留10GiB守护。
当前协调器`resources()`仍检查启动空闲≥60GiB、GPU空闲显存及利用率、CPU／内存准入；不把旧admission里的27.9GiB当实时容量。
预算目标与实际准入代码的差异需核对；本次只整理文档，不修改冻结脚本、准入阈值或status。
资源满足并且适配器／复用检查通过后，按用户已确定方向在tmux后台接续，断开SSH仍运行。
先检查现有队列；重启须使用相同计划、源文件／模型／输入fingerprint和自己的尾帧依赖。

## 范围和复用

| 范围 | episode | 视频 | 硬切 | 接续 |
|---|---:|---:|---:|---:|
| 完整 | 140 | 2,491 | 1,136 | 1,355 |
| 已完成、核验后复用 | 20 | 363 | 191 | 172 |
| 本次新增 | 120 | 2,128 | 945 | 1,183 |

新增711个硬切角色参考、226个单角色3D候选键、约1,163参考候选图，以冻结计划重算核对。
来源为`outputs/entitybench_qwen21_i2v_a14b_4step_continuation_20ep/`；整个episode核验后只读链接复用，不覆盖或复制媒体。
旧20集保持seed、模型／LoRA、协议／实现SHA、素材／Gaussian／标定和视频SHA；新增episode seed按完整split冻结。
角色键episode-local，同run不同窗口用完整episode_id；角色素材改变时不能按名字借用其他参考的Gaussian。

## 入口及自动阶段

| 用途 | 文件 |
|---|---|
| 全量计划 | `pretest/prepare_entitybench_qwen21_full140.py` |
| 素材／首帧 | `pretest/run_qwen21_entitybench_full140_images.py` |
| 视频 | `pretest/run_lightx2v_a14b_4step_qwen140.py` |
| 复用／完成核验 | `pretest/audit_entitybench_qwen21_full140.py` |
| 脸部版 | `pretest/evaluate_entitybench_face_only_dino_qwen140.py` |
| 协议检查 | `tests/test_qwen21_full140.py` |

旧20集协调器写死20集及112／191／363，不直接用于扩量；新适配器独立冻结，核心算法保持。
必要的协议、依赖续跑、输入和聚合检查完成后，流水线接续：

1. 核验模型／环境、源20集和完整split，冻结计划／适配器及全部输入来源。
2. 新Qwen参考→符合资格角色的新FaceLift／273点512标定→945硬切首帧。
3. 新2,128段四步A14B视频；接续严格使用自己的实际解码尾帧，旧363段核验复用。
4. 2,491视频／首尾依赖核验，拼接140集成片，官方非VLM按完整episode两路评分及统一聚合。
5. 独立提取脸部crop，逐图检查并冻结决策，脸部DINO质心评分和最终归档。

当前入口在脸部提取后等待`evaluation/face_only_dino/review/decisions.json`，需完成目视检查才能继续，不能伪造自动验收。
不运行额外生成消融；并发验证只使用已有视频，GPU生成／评分分阶段，避免与其他重任务竞争。

## 评分与完成核验

按[evaluate-face-injection-v2](/root/.codex/skills/evaluate-face-injection-v2/SKILL.md)，固定作者revision／skip_vlm适配、strict、无config override。
API／VLM调用0、39缺失指标null、无fidelity gate；官方原人物区域指标和用户要求的脸部版分开保存。
默认两进程各4CPU线程，完整episode分片、独立输入绑定cache，在原140-ID split上官方反序列化／聚合，不拼接分片均值。
核验140集各一次、2,491镜头、1,355自身尾帧接续、逐shot schedule、六项VBench覆盖／失败计数及来源SHA。
脸部版保存检测＋角色空间匹配覆盖率、有效集数、appearance与缺失原因；同角色至少两镜头，不把缺失填0。
原报告与脸部版输入／样本范围不同，论文只作外部参考；绝对路线效果不证明插件因果增益。
成片／裁剪查看页、抽样视觉审查与评分完成分别记录；原完整报告、manifest、模型／实现SHA和分片证据保存。

## 保留和文档入口

人物素材与首帧均Qwen，视频仍A14B四步；全部本地，其他旧队列保持停止，不加API／VLM／消融。
已批准删除的只有旧Wan5B、pip缓存与11个pip解包残留；其他模型、素材、3D、消融、报告、环境与/tmp保留。
前端默认旧5B视频路线暂不可用；不要为本实验自动改前端配置。Antelope仍为素材／注入／标定依赖。
用户并行改动及另一个会话新增的全量代码保持，不reset／clean或全量stage。
本轮提交范围为Qwen全量代码、RGBA及非人脸分流恢复、测试和交接文档；媒体／模型／运行时输出不入Git，其他并行改动保留，提交／推送状态以git检查为准。

- 固定生成协议来源：[Qwen20运行记录](../entitybench_qwen21_20ep_run.md)。
- 脸部协议：[只裁脸评分说明](../entitybench_face_only_dino_20ep_run.md)。
- 实测时间／存储：[预算](../entitybench_qwen21_full140_estimate.md)。
- 清理审批与完成证明：[数据盘](../disk_cleanup_review_20261014.md)、[系统盘](../system_disk_usage_20261014.md)。
