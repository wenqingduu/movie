# Qwen140执行交接

更新：2026-10-14（北京时间）。当前方向和固定参数以[主交接](../LLM_HANDOFF.md)为准；本文只列执行与核验步骤。
历史模型组合／进度／清理前容量见[历史索引](history/README.md)，当前任务无需通读。

## 现状与资源

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
本次推送仅包含交接、历史和资源文档；新全量代码／测试仍在本机未提交，保留另一会话工作，提交／推送状态以git检查为准。

- 固定生成协议来源：[Qwen20运行记录](entitybench_qwen21_20ep_run.md)。
- 脸部协议：[只裁脸评分说明](entitybench_face_only_dino_20ep_run.md)。
- 实测时间／存储：[预算](entitybench_qwen21_full140_estimate.md)。
- 清理审批与完成证明：[数据盘](disk_cleanup_review_20261014.md)、[系统盘](system_disk_usage_20261014.md)。
