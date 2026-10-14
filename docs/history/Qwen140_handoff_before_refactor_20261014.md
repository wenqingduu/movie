# Qwen140交接整理前记录（归档）

本文保留精简前交接内容，其中阶段描述和资源数字是历史快照。
当前执行以[主交接](../../LLM_HANDOFF.md)及[Qwen140执行交接](../LLM_HANDOFF_QWEN140_20261014.md)为准。

---

# Qwen完整140集：新会话交接（2026-10-14，北京时间）

用户最新方向：后续人物素材和首帧统一用本地Qwen，将现有设置扩到完整EntityBench；不补FLUX素材＋Qwen首帧的交叉组合。
本会话只核算存储、按明确批准清理文件并写交接，**没有启动全量生成、下载、API或评分**。
最初交接时全量入口未存在；后续盘点已看到工作树出现`pretest/run_entitybench_qwen21_full140.py`及配套准备／图像／视频／评分脚本和测试，可能由另一个会话继续实现。
本清理会话没有修改或验证这些新增代码的完成状态；接手先查实际文件、进程、status和测试记录，不重复创建入口或启动队列。
FLUX全140的paused_by_user继续有效，只禁止恢复FLUX队列，不禁止用户新决定的Qwen全量。
主交接／旧提示词中“不扩140”是上一阶段20集边界；本次Qwen方向优先。所有其他保存边界仍有效。

**清理完成：用户仅批准A1旧Wan5B，已删除 `/root/autodl-tmp/movie/models/video/Wan2.2-TI2V-5B`，53个文件，释放31.85GiB。**
数据盘空闲约59.76GiB；A2／A3／B1／B2／B3全部未批准、未删，见[精确清单及影响](../disk_cleanup_review_20261014.md)。
前端默认视频仍引用已删Wan5B，暂不可直接运行；不得自动改前端，当前Qwen benchmark用A14B，不受影响。
antelopev2仍必需，不能整删PuLID模型目录。官方报告、视频、参考、3D／标定、消融和其他改动保持。
系统盘另为30GiB overlay；用户后续批准pip缓存及11个pip-unpack残留，已释放7.11GiB，目前约9.30GiB可用，详见[系统盘核查](../system_disk_usage_20261014.md)。
已安装环境未搬迁／删除；其他/tmp没有清理，当前VS Code通信socket、代理锁及实验调试产物保留，不能整删/tmp。
容量以实时df为准；60GiB为预算目标，不是59.76与60之间的机械门槛，应核验新产物30–40GiB及至少10GiB运行余量，再决定是否还需要扩容。

## 先读与先查

完整阅读本文、[主交接](../../LLM_HANDOFF.md)、[Qwen20协议](../entitybench_qwen21_20ep_run.md)、
[20集结果汇总](../entitybench_results_20ep_20261014.md)、[耗时与存储预算](../entitybench_qwen21_full140_estimate.md)、
[脸部评分协议](../entitybench_face_only_dino_20ep_run.md)及 `/root/.codex/skills/evaluate-face-injection-v2/SKILL.md`。
先检查实际tmux pane_dead、Python进程、GPU／内存／磁盘、实时status与manifest，避免重复启动。
本次检查tmux没有server，未发现生成／评分进程；这只是交接快照，不替代新会话实时检查。
用户希望任务在tmux后台运行，断开SSH仍继续，完成后自动核验、评分和归档；不能只启动前半阶段后退出。
本会话不提前启动；新会话读取本提示词后，在存储和资源条件满足、全量入口检查通过时接续，无需再次确认已确定的模型路线。

## 存储预算（不含已有模型）

当前数据盘总320GiB，删除旧Wan5B后可用约59.76GiB；清理前27.94GiB是历史预算快照，开跑前重新df。
Qwen20集目录实占约5.37GiB，包含素材／3D／首帧／输入／363视频／20成片／评分缓存；不含权重。

| 类别 | 20集实测 | 剩120集外推与预算 |
|---|---:|---:|
| 参考图及候选 | 0.385GiB／192候选 | 约2–3GiB／预计1,163候选 |
| FaceLift／273点512标定 | 3.768GiB／39角色 | 约22–25GiB；最多226新单角色键，资格筛选后可少 |
| 硬切首帧及内部诊断 | 0.296GiB／191镜头 | 约1.5–3GiB／945硬切 |
| 视频＋输入＋episode成片 | 0.663GiB／363镜头 | 约4–6GiB／2,128新镜头 |
| 官方评分、脸部版裁剪、日志及核验 | 原官方评分0.217GiB；脸部版三路线约0.116GiB | 约2–4GiB |

按当前文件种类线性外推约30.8GiB，新增产物建议按**30–40GiB**预算；另留临时文件／I/O和运行余量。
**建议开跑前约60GiB可用**；现在清理后约59.76GiB，新产物预算30–40GiB、剩余约20–30GiB，不再固定要求扩40–60GiB。
先实现全量入口、冻结计划并核验实际存储预算和至少10GiB守护；只有预计空间不足时等待用户处理，不删除未批准项目。
不自动扩容，不删除任何旧产物、模型或用户改动；逐阶段及逐任务空间守护至少保留10GiB，停止要明确且可续跑。
已存在模型不重新下载，不复制一份全套权重，不为复用20集再复制媒体。
实测字节、来源SHA与规模计算见[存储证据JSON](../entitybench_qwen21_full140_storage_budget.json)。
原2026-10-12下一轮计划的100GiB建议是旧粗预算，本次已有实测，仍以实时增长和守护为准。

## 完整范围与复用

- 原validated split为140个完整episode／2,491shot，硬切1,136、接续1,355。
- 已完成Qwen20集／363shot：191硬切、172接续；核验兼容后复用完整episode，保留原seed和媒体／报告SHA。
- 新增120集／2,128shot：945硬切、1,183接续。按原benchmark entity schedule统计，硬切需711个新episode-local角色参考；226个单角色键各3候选，其余各1，共约1,163候选。
- 以上数量是预算检查值，正式计划应从原split／script自行重算并核对，不以文档或旧FLUX plan替代事实源。
- 新输出建议 `outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140/`；入口`pretest/run_entitybench_qwen21_full140.py`已在工作树出现，完成状态／输出路径需先核对，不重复创建或启动。
- 已完成20集来自 `outputs/entitybench_qwen21_i2v_a14b_4step_continuation_20ep/`；不覆盖原目录，不重写原正式结果。
- 旧FLUX全量计划可参考split与完整episode_id，但其SDXL／FLUX参考和144份暂停3D资产不能借给Qwen。
- 同名人物必须保持episode-local键，同run不同窗口使用完整episode_id，不能合并身份或覆盖目录。
- 新episode seed按完整split冻结，旧20保持原seed；prompt、cut、entity schedule与镜头次序保持benchmark。

## 固定模型与生成路线

人物素材和首帧均为本地Qwen-Image-2.1，revision `790c92633540aa0cb11d9abf19eb46d861714758`，沿用本地权重／环境。
素材1024×1024、40步、原prompt模板、CUDA驻留；单角色硬切所需参考沿用3候选资格筛选，其余1候选。
保留儿童、性别不明确、非人实体及明确动画风格，不统一成人或强制全部写实；不根据正式DINO分数选素材。
符合资格的新素材准备自己的FaceLift Gaussian和273点／512标定；不复用不同参考SHA的3D。
单人首帧1024×576、20步、原官方Control共享latent分叉，step12 pred_x0准备3D／mask，v7强度0.4，
cosine0.40阈值决定1／6步；内部Control只是实现轨迹，不追加无插件视频消融组。
多人硬切用Qwen原生多参考，无多人3D注入；按模型参考图上限检查，不能默默删除角色。
首帧保留原验证过的sequential CPU offload，不擅自把素材驻GPU加速套到多人首帧。
缺脸／闭眼／空mask／非人脸按原资格与安全回退记录，保留镜头；文件／资源错误中断，不伪装成回退。

视频仍为 **Wan2.2-I2V-A14B＋LightX2V high/low 4step_1022蒸馏LoRA**，不是Qwen视频模型。
DMD2四步、1000／750／500／250、shift5、CFG关闭、BF16／torch_sdpa、81帧／16fps／480p面积，
原LoRA／模型／运行环境SHA冻结。接续只用本路线前一视频实际解码尾帧，不编辑、不再注入、不借其他路线视频。
当前363段实测平均95.904秒，视频阶段9.672小时；剩2,128段约56.7小时，加全部准备／评分建议预留4–5天。
本路线不引入云端Portrait Agent、不需API key、不测VLM。不得把已有新FLUX云端prompt改编混入Qwen模板。

## 全量入口实施与自动完成

现有 `pretest/run_entitybench_qwen21_20ep_accelerated.py` 和相关准备／图像脚本写死20集及112／191／363等数量，
不能直接调用旧入口期待扩量，也不要原地修改已冻结源码。新入口／适配器独立写入和冻结，复用核心算法。
参考Qwen20已有资产、首帧、视频、输入依赖检查以及完整FLUX的whole-episode复用方式，但执行新Qwen路线。
先核验原20集模型、环境、参数、脚本、script、参考／Gaussian／标定、首帧、视频与父尾帧SHA；复用后记录来源。
路径链接复用须在源输入SHA验证后创建，保持旧目录只读；新全量评分目录和输入fingerprint独立。
按所改范围做必要协议／续跑／聚合检查，不启动新增生成消融；小样本只用于验证新入口与实际依赖。
有进程锁、按完整episode缓存、disk guard、清楚日志与status；同一GPU生成／评分分阶段，避免重复进程。
后台流水线完整串起：新素材→新3D／标定→硬切首帧→视频／自己的尾帧接续→来源核验／成片→非VLM评分→脸部版评分→最终归档。

## 评测口径与结果

官方非VLM按evaluate-face-injection-v2：固定作者revision `3776439e7037b2128ac3b5d2d1ce63e424cca166`，
原版／skip_vlm适配SHA、strict、无config override、VLM调用0、39缺失指标null；无fidelity gate，跨shot分数暂定。
先按两进程各4CPU线程、完整episode分片、独立cache，原完整140-ID split一次官方反序列化／聚合。
沿用已验证并发协议，不能把不同shot分片均值当角色质心，不能把旧20宏平均加权拼成全量平均。
完成核验140episode各一次、2,491视频映射／SHA、逐shot schedule、六项VBench覆盖／失败计数和1,355实际尾帧接续。

另外按用户最新要求保留**只裁脸DINOv2质心版本**：RetinaFace仅检测、原5采样点、脸框10%pad、不对齐，
角色空间匹配与冲突缺失规则、先查看／冻结裁剪决策再评分，至少两个有效镜头；不加载InsightFace身份特征。
现有脸部脚本写死三路线20集，完整Qwen需新适配入口／输出，不能覆盖已完成20集或填入原官方cs_face。
脸部检测及角色匹配覆盖率、missing、valid episode数、appearance数一起报告；同角色空间匹配不等于身份真值。
原人物区域官方报告与脸部版分开保存；论文gate／聚合／素材／视频不同，只作外部参考，不声称公平排名或插件因果增益。
最终保存全量Qwen与原20集／论文参考对照，保留原报告、输入来源与SHA；视频／裁剪查看页及抽样视觉审查独立于评分完成。

## Git与保留

GitHub main已验证提交 `6342dff7d0107cb8b9d801b426ad7b8cb4ab0e33`；那次提交34个本轮代码／文档，40项检查通过。
**本文、全140新窗口提示词和本次预算更新尚未提交／推送**；本机后续文档优先，不能假设GitHub已包含此次方向变化。
用户其他未提交改动保留：`EPISODE_TEST_RESULTS.md`、`docs/frontend_agent_architecture.md`、
`pretest/summarize_current_episode_results.py`及`experiment_output/`。不reset／clean，不全量stage，不上传密钥／权重／媒体。
本次交接未启动或恢复任何FLUX／Qwen生成，也未修改冻结代码或已有评分。
