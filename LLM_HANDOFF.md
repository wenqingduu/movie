# Qwen人脸注入：当前工作交接

更新：2026-10-14（北京时间）。仓库：`/root/autodl-tmp/movie`。

## 当前目标与状态

后续只接续Qwen路线：Qwen人物素材→对应3D／标定→Qwen参考首帧＋单人3D注入→Wan A14B四步蒸馏视频。
目标完整EntityBench **140集／2,491镜头**，核验复用已完成20集／363镜头，只补120集／2,128镜头。
新增范围为945硬切／1,183接续、711角色参考、226个单角色3D候选键、约1,163参考候选图；正式数量以冻结计划为准。

现有全量入口和计划已在本机工作树。2026-10-14 22:06:27状态为`prepared_waiting_resources`，尚未开始全量生成；
没有活跃tmux生成队列。该状态的资源快照来自清理前，接手须重新检查实际进程、GPU、CPU／内存、磁盘及JSON。
用户已选择Qwen全量，资源与准入检查通过后在tmux后台接续；已有正常队列时只检查，不重复启动。
用户确认目前GPU尚未准备好；先整理／核验准备工作，不启动生成或GPU评分，不重复下载模型。

先读[Qwen140执行交接](docs/LLM_HANDOFF_QWEN140_20261014.md)，需要跨会话时使用[精简提示词](docs/qwen140_new_window_prompt.md)。
执行或审计官方评测使用[evaluate-face-injection-v2](/root/.codex/skills/evaluate-face-injection-v2/SKILL.md)。

## 固定生成设置

| 阶段 | 固定设置 |
|---|---|
| 图像模型 | Qwen-Image-2.1，revision `790c92633540aa0cb11d9abf19eb46d861714758`，原权重／环境 |
| 人物素材 | 1024×1024、40步、原prompt模板；单角色硬切所需参考3候选，其余1候选；素材CUDA驻留 |
| 3D | 参考对应的FaceLift Gaussian、273点／512标定；episode-local角色键，按参考／Gaussian SHA绑定 |
| 首帧 | 1024×576、20步、sequential CPU offload；官方Control共享latent分叉，step12 pred_x0准备3D／mask |
| 单人注入 | v7 color-safe，强度0.4；身份cosine阈值0.40，低于阈值6步，否则1步 |
| 多人硬切 | Qwen原生多参考，无多人3D注入 |
| 视频 | Wan2.2-I2V-A14B＋LightX2V high/low `4step_1022` LoRA，DMD2四步、shift5、CFG关、BF16／torch_sdpa |
| 视频尺寸 | 81帧／16fps／480p面积；接续只用本路线前一视频实际解码尾帧 |

保持benchmark prompt、cut、镜头次序及旧20集seed；内部Control不追加消融视频组。
缺脸／闭眼／空mask等按既有规则记录基础结果，保留难例；文件／资源故障中断，不伪装成安全回退。
不新增消融，不调用云端API，不测VLM；旧的其他路线保持停止。

## 入口与数据位置

- 全量协调器：`pretest/run_entitybench_qwen21_full140.py`。
- 全量准备／素材首帧：`pretest/prepare_entitybench_qwen21_full140.py`、`pretest/run_qwen21_entitybench_full140_images.py`。
- 全量视频／核验：`pretest/run_lightx2v_a14b_4step_qwen140.py`、`pretest/audit_entitybench_qwen21_full140.py`。
- 全量脸部版：`pretest/evaluate_entitybench_face_only_dino_qwen140.py`；协议检查：`tests/test_qwen21_full140.py`。
- 全量输出：`outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140/`。
- 已完成20集来源：`outputs/entitybench_qwen21_i2v_a14b_4step_continuation_20ep/`，保持只读。

先读全量`status.json`、`admission.json`、`preparation_plan.json`，再核对阶段日志与实际进程。
已有20集的入口写死20集，不用于扩量；新适配器独立冻结，不原地修改冻结核心代码。
复用旧20集前核验模型／配置／实现、script、参考／3D、首帧、视频／父尾帧SHA，记录完整episode来源；不复制旧媒体。

## 评测与完成条件

官方非VLM：固定作者revision `3776439e7037b2128ac3b5d2d1ce63e424cca166`及skip_vlm适配，strict、无config覆盖。
六项VBench、presence、人物／物体区域DINO和接续边界按原口径；API／VLM调用0，39项缺失保持null。
没有fidelity gate，跨shot分数仍为暂定；原人物区域`cs_face`与用户要求的脸部版分别保存。
默认两进程、每路4CPU线程，按完整episode分片、独立缓存，在完整140-ID split上一次官方反序列化／聚合。

脸部版：本地RetinaFace仅检测、原五采样点、脸框10%pad、不对齐、DINOv2单位CLS与角色质心公式；
空间匹配冲突／缺失如实记录，角色至少两个有效镜头。逐图检查并冻结决策后评分，不用身份特征评分或DINO数值挑脸。
当前全量入口在脸部提取后等待`evaluation/face_only_dino/review/decisions.json`；不得伪造验收以跳过检查。
完成核验140集各一次、2,491视频映射／SHA、1,355真实接续依赖、覆盖／失败／缺失计数、评分／归档和查看页。
论文只作外部参考；素材、生成和gate／聚合不同，不宣称严格横比或插件因果增益。

## 已完成Qwen基线

20集／363视频及官方非VLM评分已完成，视频失败0；原人物区域`cs_face=0.6958`。
另存的只裁脸DINO为`0.8225`，有效20集、396／726可匹配角色-镜头（54.55%）；原报告不覆盖。
脸部协议见[执行说明](docs/entitybench_face_only_dino_20ep_run.md)，冻结生成协议见[Qwen20记录](docs/entitybench_qwen21_20ep_run.md)。
本轮四步视频363段平均95.904秒，视频阶段9.672小时；剩余端到端预算约4–5天，见[时间／存储预算](docs/entitybench_qwen21_full140_estimate.md)。

## 资源与保留边界

最新清理后快照：数据盘320GiB、空闲约59.75GiB；系统盘30GiB、空闲约9.30GiB。新产物预算30–40GiB，逐任务至少保留10GiB。
当前全量协调器还有启动空闲≥60GiB的硬检查；与预算目标及实际资源状态需接手核对，本次整理文档不修改代码或准入状态。
只按用户批准删除了旧Wan5B权重、pip缓存及11个pip解包目录；其他模型、素材、3D、报告／消融、其他/tmp和安装环境保留。
旧Wan5B删除使前端默认视频路线暂不可用；当前A14B实验不受影响。清理详情分别见[数据盘](docs/disk_cleanup_review_20261014.md)和[系统盘](docs/system_disk_usage_20261014.md)。
保留Qwen／A14B／LoRA、FaceLift、Antelope、facexlib及官方评测依赖。未知资源环境不降低参数强行启动。
本次提交／推送范围为当前交接、历史归档及存储清理文档；全量适配器由另一会话编写，仍在本机未提交。提交／远端状态以git检查为准。
保留用户并行改动：`EPISODE_TEST_RESULTS.md`、`docs/frontend_agent_architecture.md`、`pretest/summarize_current_episode_results.py`、`experiment_output/`及新增全量代码。不得reset／clean或全量stage。

## 历史资料

早期首帧／视频评测、其他模型组合、阶段快照和既有消融已完整移到[历史交接档案](docs/history/LLM_HANDOFF_experiments_20261014.md)。
需要查旧实验或证据时按[历史索引](docs/history/README.md)进入；当前接手不要求阅读全部历史。
