# 2026-10-31会话交接：剩余126集重跑继续中

状态核对时间：北京时间2026-10-31 23:07。这是交接快照，接手后先看实际进程、日志和结果，后台进度会继续变化。本次只更新文档、读取已有报告及核验19个冻结源码SHA；没有重启任务、修改实验代码、重新推理评分或推送Git。

## 当前执行情况

**尚未跑完，后台正常运行，不要重复启动。**

| 范围 | 已完整生成并完成官方非VLM评测 |
|---|---:|
| 本批126集／2316shot | 15集／158shot |
| 此前保留的多人3D重跑 | 14集／175shot |
| 合计140集中的已重跑部分 | 29集／333shot |

本批已生成175shot，包含未完成集的17shot，不可把175当作本批已评测shot数。未完成集包括run155/i5_j39的10/11shot，以及正在执行的run654/i20_j41的7/10shot。最近完成的是run629/i115_j139和run629/i95_j115两个不同窗口。其余111集尚未完整完成，原始140集媒体和分数仍保留。

当前实际tmux：`qwen140_remaining126`；父队列PID1997、run654生成调度PID26585，PID只是本次快照，后续可能变化。GPU为RTX4090 48GB，实际有工作负载。数据盘350GiB、空余约36.80GiB；仍可能不足以跑完全队列，不能承诺目前扩容已足够。

10月30日22:59首次启动；10月31日约15:09完成run617/i75_j127评测后因下一集空间准入不足等待。20:05检查旧tmux和进程已不在，退出原因没有日志证明，不能归因于SSH。用户扩容320→350GiB后，20:05原配置恢复，13集结果原样保留，现增加到15集。恢复记录在根目录`resume_records/20261031T120529Z/`。

## 接手入口

所有相对路径均相对于`/root/autodl-tmp/movie`。

- 主队列：`pretest/qwen140_multiface_remaining_episode_batch.py`。
- 结果根：`outputs/entitybench_qwen140_multiface_remaining126_20261030/`（下称R）。
- R的`status.json`是当前阶段，`queue.log`是调度日志；`cohort.json`是冻结完整清单，`stage_timings.jsonl`是已成功阶段计时。
- R的`index.html`链接每集前后视频、指标及人物区域裁剪，`report.md`和`cohort_comparison.json`是本批已完成相同子集的官方聚合。
- 每集目录为`R/experiments/<完整episode_id>/`，不能只凭run号区分窗口。`review/index.html`、`review/before.mp4`、`review/after.mp4`供人工查看。
- 每集`complete.json`表示生成和非VLM评分完成；`evaluation/weak4_v1/`保留输入manifest、逐集结果、对比和评分manifest，`evaluation_validation.json`记录覆盖率与实际注入。
- 每集`episodes/<episode_id>/shot_<scene>_<shot>/result.json`为成功shot，`episode_result.json`为整集完成记录。异常见该集`technical_exception.json`及`logs/`。
- 交接时静态快照：`R/handoff/snapshot_20261031.json`，包含完成集明细、部分完成、异常指针、空余空间及19个源码SHA核验；`R/handoff/report_20261031.md`保留当时报告。它们不替代实时状态。

## 两个未解决问题及一个资产缺失

1. **run155/i5_j39的8:1未生成，前10shot保留。** 日志`experiments/0001a639-957f-3e52-b962-a5b69747bbc6__run155__i5_j39__T420/logs/generation.log`显示，Agent素材查询names含`the yellow electric pokémon`，`qwen140_shot_agent.lookup_characters`直接用名称索引catalog导致KeyError。这不是全局API故障或磁盘故障。20:05重启从缓存再次命中同一错误，随后继续run629；没有生成修复结果。后续要结合8:1的`agent/planning_inputs.json`和`lookup_call.json`核实它是别名、未建模角色还是不该查询的人物；不可随便映射到已有角色或偷偷删除请求。
2. **全140集滚动快照更新失败，本批逐集评分和15集配对汇总有效。** `pretest/qwen140_reaggregate_partial_reruns.py`约第195行要求现有规范视频软链接已经指向当前选中视频；初始化时该链接指向原版，后续新集完成后需要改选重跑版，从而触发assert。R的`full140_aggregation_exception.json`和`logs/full140_snapshot.log`留有错误，`full140_snapshot/validation.json`仍停留在10月30日22:59的14集替换状态。后续失败可能已部分写入其目录，**不能把该目录当作最新一致快照**。修复方向是验证输入后构建独立版本快照，再发布明确的完整快照入口；不能改分数或删原媒体来绕过断言。
3. **run617/i75_j127的Jordan没有可用3D。** 该集`logs/jordan_facelift.log`为FaceLift VAE收到4通道图、期望3通道的技术错误。`asset_failures.json`记录保留2D参考、该角色不注入3D，其余角色继续；整集已完成并进入评分。不能把它说成所有角色都成功注入，也不能把资产错误称作正常姿态门控。后续修复属于另一个有记录的资产版本，不覆盖已冻结参考和既有结果。

上述问题均未在本次交接中修改代码处理。继续批次生成与修复这些问题是不同工作；不要为修复汇总而打断正在生成的shot。

## 本批15集配对指标

来源：R的`report.md`，截至23:07读取的已完成15集158shot，官方集等权。不是全140集，也不是3D插件的单独消融。

| 指标↑ | 原始同一子集 | 当前多人3D集成 |
|---|---:|---:|
| 主体一致性 | 0.9442 | 0.9483 |
| 时间稳定性 | 0.9809 | 0.9828 |
| 运动平滑度 | 0.9890 | 0.9900 |
| 动态程度 | 0.5165 | 0.4852 |
| 美学质量 | 0.5968 | 0.6036 |
| 成像质量（0–100） | 65.5121 | 67.1443 |
| 角色出现率 | 0.9649 | 0.9919 |
| 物体出现率 | 0.9031 | 0.9210 |
| 场景出现率 | 0.5914 | 0.6135 |
| 人物区域cs_face | 0.7618 | 0.7807 |
| 物体区域cs_object | 0.8606 | 0.8283 |
| 接续边界一致性 | 0.9868 | 0.7564 |

角色出现、人物区域一致性、成像、美学等提高；动态、物体区域一致性、边界下降。完整视频人工审阅未完成，不声称所有效果达标。

## 实验和操作约束

- 只保留原始完整benchmark与当前多人3D集成两条主线。原始140集2491shot在`outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140/`；不能覆盖。历史14集版本细节已留档，最终也不能宣称全140集都从首shot使用完全相同代码。
- 当前方案：qwen3.8-max Agent路由、数据集人物一句话描述、episode统一画风；A复用自身实际尾帧，B进入过渡＋行动固定1＋1，C重建起始帧。干净单人参考和对应FaceLift Gaussian逐episode绑定；全部检测脸先做一对一身份匹配，逐角色v7原门控，不新增面积禁注入阈值。
- 整张或部分角色未通过3D门控时按已授权策略沿用同次Control并记录，不要求用户确认，不计作实际注入。技术异常记录后继续其他集；全局认证/额度问题停止新任务。不放宽门控、换模型或重生挑高分。
- **守护修复Agent保持关闭。** 普通队列和镜头路由Agent仍工作；不能因为用户换会话而启动修复Agent。
- 每集完整官方strict/skip_vlm，无VLM评分调用；cs_face是人物区域跨镜头DINO，非人脸识别分数，缺VLM保真度门控。合法不适用值保留null，不填0。评测接手使用`evaluate-face-injection-v2` skill。
- **冻结约束：19个源文件SHA进入cohort及各集face_injection_policy。** 不要原地改`qwen140_shot_agent.py`、路由提示词、重聚合helper或队列driver后直接续跑，否则后续校验会失败。正式修复要保留已有前缀/旧版代码与输入指纹，明确新版本、独立输出和兼容续跑范围；不能只改记录里的SHA假装是原实验。文档更新不涉及本次19个冻结源码。
- 每集前按资产及剩余shot估算磁盘，加25%余量并保留8GiB；不足时普通队列60秒检查空间，不自动删除历史。原9–12天预估基于持续GPU/API/存储，已有等待使完成日期顺延。
- API配置已授权使用`.env.portrait`，只在进程内读取；禁止把密钥写进文档、日志、命令行输出或聊天。
- 工作区有大量未提交改动和新脚本，不能reset/clean或自行推送。尤其保留无关`docs/frontend_agent_architecture.md`、`experiment_output/`及中文绘图说明。本次未提交Git。

## 下一会话步骤

1. 读本页及`LLM_HANDOFF.md`；检查tmux、实际进程、GPU、磁盘、R/status和最新日志，核对完成数。活动队列存在就继续，不再启动第二份。
2. 只在队列确实退出且冻结配置/素材及CUDA检查通过时，按原入口恢复；成功episode和合法shot缓存会跳过。run155已知未修复，会从缓存再次失败后继续，恢复不能宣称修好它。
3. 后续修复优先处理独立全140快照发布与run155查询名称问题，版本策略须保护当前运行及历史输出。Jordan资产缺失单独留档。
4. 继续逐集官方非VLM，完成后再汇总140集；完成数、评分覆盖率、失败/缺失、实际注入、人工审阅分别报告。保留下降指标。

仅当核实无原进程时使用的恢复命令：

```bash
tmux new-session -d -s qwen140_remaining126 -c /root/autodl-tmp/movie 'OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 ENTITYBENCH_ORT_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /root/autodl-tmp/envs/lightx2v-trial/bin/python -u pretest/qwen140_multiface_remaining_episode_batch.py run >> outputs/entitybench_qwen140_multiface_remaining126_20261030/queue.log 2>&1'
```

Clara绘图任务和skill也已完成，不需重复跑：原去噪重放`outputs/model_figure_clara_denoising_replay_20261029/`；183张图片目录`outputs/clara_figure_images_20261030/`；选例`outputs/firstframe_figure_reviewed_20261030/`；skill及限制见`docs/entitybench_firstframe_figure_skill_selection_20261030.md`。Clara4:1不具有明显最终IF增益，不能把流程图示说成身份显著提升证据。
