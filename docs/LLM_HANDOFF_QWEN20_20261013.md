# Qwen20episode实验：新窗口交接

**2026-10-14接手说明：本文保留Qwen20原运行快照；当前生成、非VLM评分、三路线归档和脸部版评分均已完成。**
最新入口为[主交接](../LLM_HANDOFF.md)、[三路线结果汇总](entitybench_results_20ep_20261014.md)及[Qwen全140预算](entitybench_qwen21_full140_estimate.md)。
下文“当前任务／正在运行／之后自动评测”属于2026-10-13历史，不据此重启完成队列。全140仍未启动Qwen，FLUX仍停止。

**最新完成状态（2026-10-13）：北京时间14:09已完成Qwen20/363官方非VLM评分并归档；16:48之后独立核验通过。下面11:30进度为历史快照。**
结果见[论文/FLUX/Qwen完整对比](../outputs/entitybench_paper_flux_qwen20_comparison/README.md)，[完成核验](../outputs/entitybench_paper_flux_qwen20_comparison/review/final_integrity_audit.md)。全量人工审片仍待完成，FLUX全140保持暂停。

快照时间：2026-10-13 11:30–11:32，北京时间。工作目录`/root/autodl-tmp/movie`。
这是本次新窗口的快速入口；先读本文件，再读`LLM_HANDOFF.md`及`docs/entitybench_qwen21_20ep_run.md`。
用户不希望重复确认，已有生成/官方非VLM评价/结果归档已授权；先核对活跃队列，不重复启动。

## 用户已经确定的范围

当前只跑与已有FLUX官方评价完全相同的20episode、363shot，不跑全140，不增加消融组。
新路线是Qwen生成人物素材→Qwen参考条件首帧＋单人3D插件→同一普通4步A14B→自己尾帧接续。
只看最终路线指标和视觉效果，消融沿用以前的实验。用户知道论文全量与本项目子集不同，
要求先保存论文、FLUX和本轮Qwen结果供初步观察，不要求严格论文排名。
VLM继续不测；不新增InsightFace/参考DINO/项目脸框质心正式评测。

## 快照进度与队列

| 阶段 | 快照状态 |
|---|---|
| Qwen素材 | 112/112完成，40单角色硬切角色各3候选，其余72各1候选 |
| 新3D/标定 | 39/39可用角色完成；Nora三候选无可靠单脸，保留基础参考并记录gate |
| 硬切首帧 | 191/191完成；66实际注入，125未注入 |
| 未注入构成 | 117多人、6目标缺可靠脸、1参考人脸gate、1闭眼gate |
| 新A14B视频 | 约292/363完成，生成失败0，仍在运行 |
| 官方Qwen评分 | 尚未启动，生成/审计完成后自动开始 |
| 初步结果存档 | 论文与FLUX已存；等待Qwen完整官方结果后自动补齐 |

这些是快照，实时视频完成数以`generation_report.json`为准。`status.json`只在切换阶段更新，
更新时间旧不等于队列卡住。当前视频均值约95秒，最近20段约98秒；余下约71段约2小时，
另加审计及约半小时到1小时非VLM评分，快照预计今天13:30–15:00左右完成，不能作为保证。

- 主tmux：`entitybench_qwen21_a14b4step20`，pane_dead=0，视频阶段GPU约36GiB/100%。
- 当前入口：`pretest/run_entitybench_qwen21_20ep_accelerated.py`，调用冻结的原始路线。
- 主输出：`outputs/entitybench_qwen21_i2v_a14b_4step_continuation_20ep/`。
- 总日志：`outputs/entitybench_qwen21_20ep_pipeline.log`；视频日志为主输出`logs/generate_formal.log`。
- 归档tmux：`entitybench_results_archive20`，CPU轻量后台每30秒看完整官方报告。
- 已停FLUX全140tmux：`entitybench_a14b4step_full140`，保留退出pane，不是活跃队列。
  全140状态`paused_by_user`，新增资产144/217保留。不要自动恢复；旧FLUX20episode已完成。
- 数据盘300GiB，目前约14GiB空闲，不能自动删除历史产物；各阶段有空间守护。

## 新窗口先检查

```bash
cd /root/autodl-tmp/movie
tmux list-sessions
tmux list-panes -t entitybench_qwen21_a14b4step20 -F '#{pane_dead} #{pane_current_command}'
cat outputs/entitybench_qwen21_i2v_a14b_4step_continuation_20ep/status.json
tail -n 15 outputs/entitybench_qwen21_i2v_a14b_4step_continuation_20ep/logs/generate_formal.log
```

查看`generation_report.json`的jobs长度、status及逐shot失败；确认实际Python进程。
tmux设置remain-on-exit，会话名字存在不代表程序活着。若队列正常，继续等待，不再开第二套。
若失败，读日志诊断、保留已完成产物，再用当前入口续跑；不要为了续跑改冻结模型/seed/原代码SHA。

## 首帧是什么、在哪里

主输出`firstframes/<完整episode_id>/shot_<scene>_<shot>/frame_record.json`：
`final_image`才是实际视频首帧，`plugin_applied`才是是否实际注入，不能仅凭文件名判断。

- `control.png`：固定人物参考条件生成、没有3D注入的内部基础图，注入镜头也会有它。
- `treatment.png`：共享中间latent继续去噪的最终输出；安全回退时也可能是Control原样副本。
- `final.png`：多人等基础路线的最终输出。
- 单人注入从官方Control callback保存的step12 latent准备pred_x0姿态/mask/3D，再继续Treatment。
  不根据最终Control图准备注入，不从噪声再生成Treatment；内部Control不生成消融视频。

例如run1517的路径：
`firstframes/000cf326-8770-3c4e-a43a-6f9e3d57a3e0__run1517__i15_j40__T300/shot_3_1/control.png`
确实是该镜头条件生图的未注入内部图；不能据它存在断言最终镜头没有注入。

## 固定生成协议与加速证据

- 图像模型已在`models/diffusion/qwen-image-2.1`，7份权重约30.84GiB；启动已重新核验SHA。
  revision `790c92633540aa0cb11d9abf19eb46d861714758`，不下载Turbo、不升级环境。
- 素材1024×1024、40步，112个硬切角色素材；19个仅接续角色不额外造参考，原131角色调度保留。
- 单人硬切需要新素材自己的FaceLift Gaussian与273点/512标定，不能借用旧SDXL身份资产。
- 首帧1024×576、20步，step12 v7强度0.4，参考cosine阈值0.40决定1/6步。
  多人只原生多参考、不新增多人注入。缺脸/闭眼/空mask等保留基础结果，不删负例。
- 接续直接用本路线前一视频的解码尾帧，无二次编辑/注入，无借用旧视频。
- 视频同FLUX：Wan2.2-I2V-A14B＋LightX2V high/low 4step_1022 LoRA，DMD2四步、shift5、
  CFG关、BF16/torch_sdpa、81帧/16fps/480p面积，原video prompt/seed/cut顺序保留。

素材已经加速：原sequential offload约105秒/图，CUDA驻留约21秒/图（约5倍，两个样本）。
两候选PNG SHA与原图完全一致；allocated约36.9GiB/reserved约37.8GiB。
两路model offload两图89秒，慢于单进程CUDA驻留两图42秒，素材不做双进程。
只改变素材模型放置：后续多参考首帧仍sequential offload，标定两路，视频/评价配置不变。
证据为`outputs/entitybench_qwen21_reference_acceleration_probe/report.json`及
主输出`runtime/reference_acceleration.json`。原计划/源码/模型SHA冻结，没有重做先前已完成素材。

## 官方评价：用skill v2

使用`/root/.codex/skills/evaluate-face-injection-v2/SKILL.md`，仓库副本在
`skills/evaluate-face-injection-v2/`。主流水线视频后自动审计363来源/172接续、拼接20episode，
再调用skill helper两进程、按完整episode分片、独立缓存、一次官方统一聚合。
不要把episode内部shot拆开算质心，也不要平均两个分片的总分。

官方revision `3776439e7037b2128ac3b5d2d1ce63e424cca166`，原版与skip_vlm适配冻结在runtime/。
主表六项VBench、角色/物体/场景presence、cs_face、cs_object、真实接续边界。
cs_face是GroundingDINO角色区域＋CLIP代表crop＋DINO跨shot质心，不是脸框/原身份参考相似度。
VLM调用须为0；gate缺失，cs_face/cs_object暂定，scope=`partial_without_vlm`。
公开代码episode宏平均与论文gate/实例加权有差异；不能直接宣称论文同协议全项评价。

预计Qwen官方结果：
`evaluation/official_parallel_v2/report_qwen21_native_reference_a14b_4step_continuation_20ep_no_vlm.json/.md`。
评分完成状态为`official_non_vlm_complete_pending_visual_review`，不可把它说成人工视觉审查完成。
生成和评价成功后，`official_comparison.md`与`official_comparison_sources.json`
保存Qwen/FLUX all/easy/medium/hard同范围比较和原始来源SHA。

## 结果存档与给用户查看

总目录：`outputs/entitybench_paper_flux_qwen20_comparison/`。
`README.md`为初步表，`comparison.json`为总JSON，`paper_reference.json`为论文四方法12个共同指标。
论文来源https://arxiv.org/html/2605.15199v1，Table4与附录F.1 Tables12/13；不是完整51项抄录。
`flux20_official_report.json`已保存，Qwen完整覆盖20episode/363shot后自动保存
`qwen20_official_report.json`和`qwen20_parallel_status.json`，填入表；等待中Qwen值保持null。
论文全140/2,491shot与项目20/363shot、素材/模型条件/gate/聚合差异写清，只做初步观察。

归档入口：`pretest/archive_entitybench_preliminary_results.py --wait`。
交接检查发现磁盘脚本曾缺失、后台进程仍在运行，已恢复脚本文件；不要再同时启动第二个归档写者。
若原归档已退出且未完成，可用恢复后的脚本单次补齐；它不改生成/评价输入。
任务完成后先核验覆盖、失败、VLM调用和SHA，再给用户看最终对比表及代表视频。
人工检查单人/多人/回退/接续及明显漂移，复用已有审计图，不用精选正例替代全量结论。

## 旧实验、Git和工作树

已有Qwen三episode/34shot插件诊断也测完了，使用SDXL素材＋Wan5B＋IF/参考DINO；
严格官方Control分叉版本平均IF首帧/视频差为-0.0112/-0.0136，DINO为-0.0056/+0.0016，
不能称作稳定改善。用户已明确不用重做消融；本轮是新素材/首帧整路线的最终效果测试。

最新已推送commit：`0621550c72738a98cb364241becee318ec4683ae`，本地main与origin/main一致。
28个代码/文档/skill副本文件，52项检查通过；权重、媒体、原始报告未随Git推送，位于本机outputs。
本次交接文档与归档恢复等为后续本地更新，尚未再次提交/推送；不要假设它们已在GitHub。
用户其他未提交改动必须保留：`EPISODE_TEST_RESULTS.md`、`docs/frontend_agent_architecture.md`、
`pretest/summarize_current_episode_results.py`及未跟踪`experiment_output/`。不要git reset/clean或全量stage。
主交接文档和Qwen运行说明的归档增补也尚未提交。用户若要求push，再按movie-github-ssh-push skill
选择性提交/推送，SSH443/本地代理方式见该skill；不要先自作主张重生成或扩展scope。

## 2026-10-13 16:48后：Qwen20完成及结果核验

北京时间14:09已完成当前冻结20episode/363shot生成及官方非VLM评分；363段全部generated、失败0。主队列与归档watcher已正常结束，FLUX全140 pane仍停止、状态paused_by_user；未扩全量、未加消融、未执行VLM。

评分按v2两路完整episode分片182/181 shot、各4线程，墙钟2438.96秒（约40.6分钟）。六项VBench每项363镜头覆盖、失败0；VLM调用/密钥均0，strict=true、无config override、scope=partial_without_vlm。39项VLM指标缺失保留null；cs_face/cs_object因无fidelity gate仍为暂定。官方cs_object有20有效episode，FLUX为19；边界两路各19有效episode、172接续对，另一episode无接续。

独立核验通过：完整/不重复episode集合、363 canonical video映射及SHA、191最终首帧来源、172子镜头输入与独立解码父视频尾帧逐像素一致、363官方逐shot entity schedule记录、两个worker manifest和原FLUX评分配置/库/模型一致、204个官方聚合字段一致。固定revision上游原版代码SHA与冻结副本一致。冻结生成源码、原FLUX报告、全140暂停状态、用户其他未提交文件SHA均保持。

| 指标 | FLUX20 | Qwen20 | Qwen−FLUX |
|---|---:|---:|---:|
| 角色出现率 | 0.8649 | 0.9335 | +0.0686 |
| 场景出现率 | 0.7032 | 0.6279 | -0.0753 |
| cs_face（暂定，角色区域质心） | 0.6853 | 0.6958 | +0.0105 |
| cs_object（暂定） | 0.7964 | 0.8186 | +0.0222 |
| dynamic_degree | 0.5930 | 0.4866 | -0.1064 |
| imaging_quality | 63.0349 | 60.2959 | -2.7390 |

完整论文四方法/FLUX/Qwen12共同指标表：outputs/entitybench_paper_flux_qwen20_comparison/README.md；原始两路线报告、总JSON、20episode结果及冻结来源已保存到同目录和provenance/。核验见review/final_integrity_audit.json/md；12组已有视频连续帧图与原视频链接见review/video_samples.md，抽样观察见review/visual_review.md。review/viewer.html可选择20episode同步查看两路成片。

结果表现混合：角色出现率及跨shot相似度较高，动态程度、场景出现率和画质较低；不能宣称整体提升或插件因果收益。论文全140与项目20episode、VLM gate和聚合不同，仅供初步观察。已查看单人、多人、回退、接续、动画及低分代表镜头，不宣称全363镜头人工审查完成；保留官方完成状态official_non_vlm_complete_pending_visual_review。没有提交或推送Git。
