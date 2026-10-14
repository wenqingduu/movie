# 历史实验与交接记录（2026-10-14归档）

本文件完整保留精简前主交接的内容，包括早期三集、20集首帧／视频扩展、SDXL／IP-Adapter／FLUX组合、
新FLUX素材及Portrait Agent尝试、旧40步与四步加速、阶段进度、指标审计和磁盘清理记录。
文中的“当前”“最新”“下一步”按当时上下文保留，**不作为现在的执行指令**。
当前只接续Qwen路线，请读[当前主交接](../../LLM_HANDOFF.md)和[Qwen140执行交接](../LLM_HANDOFF_QWEN140_20261014.md)。

原主交接在归档时为832行；原文SHA256：`abab85a8875ffeb4ad362350eb6bb4e867bc7a845544a4349f864819fd4c0e2c`。
相对Markdown链接已按新目录位置调整，其余原文保留。

---

# 3D 人脸一致性注入项目交接文档

**系统盘清理完成：用户批准pip缓存和pip解包残留，已清空pip缓存并删除11个pip-unpack目录，系统盘实际释放7.11GiB，现空闲约9.30GiB。**
已安装环境保留；其他/tmp只核查未删，当前VS Code socket／代理锁及实验调试产物保留。详见[系统盘核查](../system_disk_usage_20261014.md)。
下文系统盘“只剩2.19GiB、尚未清缓存”属于清理前记录，不代表最新状态；数据盘仍约59.75GiB可用。

**磁盘清理最新状态：用户仅确认A1旧Wan5B，已删除该模型53个文件，释放31.85GiB；其他候选未删。**
数据盘空闲约59.76GiB，已接近Qwen全量60GiB建议预算；重新核验实际产物预算与至少10GiB守护，不再固定要求额外扩40–60GiB。
前端默认视频仍指向已删Wan5B，暂不可直接运行；本Qwen实验使用A14B，不受影响。未修改前端默认配置。
系统盘另为30GiB overlay，已按用户批准清理pip缓存／解包残留，目前空闲约9.30GiB；其他缓存／环境／远程编辑器保留，见[系统盘核查](../system_disk_usage_20261014.md)。
删除记录和其他待确认清单见[清理状态](../disk_cleanup_review_20261014.md)，完整交接／提示词已更新。未启动全量队列。

**最新方向（2026-10-14）：用户决定后续人物素材／首帧统一Qwen，交接新会话扩到完整140集。**
先读[Qwen140详细交接](../LLM_HANDOFF_QWEN140_20261014.md)和[可复制提示词](../qwen140_new_window_prompt.md)。
本会话只测算存储、按批准清理文件并写交接，未启动全量。工作树已出现全量专用入口及测试，可能由另一会话实施，先核对完成状态／进程，不重复创建或启动。核验复用已完成20集，仅补120集／2,128镜头。
新增产物预计30–40GiB，启动目标约60GiB可用；清理旧Wan5B后当前约59.76GiB，仍需按实际新计划核验余量。此前28GiB及扩40–60GiB是清理前预算快照。
视频仍用现有Wan A14B四步蒸馏，不增加消融／API／VLM；FLUX全140继续paused_by_user。
下文旧“不扩140”是先前20集阶段限制，本次仅更新Qwen方向；其他保留边界有效。此次交接更新尚未提交／推送。

**当前交接入口（2026-10-14，北京时间）：三路线20集生成／非VLM评分与只裁脸评分均已完成。**
先读可随Git查看的[三路线结果汇总](../entitybench_results_20ep_20261014.md)、
[脸部评分协议](../entitybench_face_only_dino_20ep_run.md)及[Qwen全140耗时估算](../entitybench_qwen21_full140_estimate.md)。
Qwen正式视频已使用四步蒸馏，363段平均95.904秒／段，视频阶段9.672小时；扩至140集预计另需4–5天，仅估算，未启动。
本Qwen路线全部本地、无需API；新FLUX的Portrait Agent文本调用属于另一条路线。全140FLUX继续停止，当前只提交相关代码和文档，不启动实验。
outputs中的视频、完整报告、查看页和权重留在原机；本结果汇总保留分数、范围和源报告SHA供GitHub查看。
本文保留历次交接记录，下面的历史“正在运行”、剩余空间和未提交描述不是当前实时状态；接手仍先查tmux／进程／JSON。

**2026-10-14按用户最新要求完成“只裁脸＋DINO质心”三路线当前20集评分。**
见[脸部版本结果与查看页](../../outputs/entitybench_face_only_dino_20ep_20261014/README.md)、[执行说明](../entitybench_face_only_dino_20ep_run.md)。
SDXL→FLUX／Qwen／新FLUX各自有效集合宏平均0.8072／0.8225／0.7907，有效集数20／20／19；
空间匹配覆盖率39.26%／54.55%／43.94%，共同16集126个appearance为0.8196／0.8323／0.8119。
原人物区域分数不覆盖。此前1,940人物crop定位修正8张、7缺失结果已完成并保存在
`outputs/entitybench_cs_face_corrected_character_crops_20ep_20261014/`；此目录不能当作脸部版本。
新脸部入口`pretest/evaluate_entitybench_face_only_dino.py`，tmux`entitybench_faceonly_dino20`已退出。
RetinaFace仅检测，不用InsightFace／ArcFace身份特征；五原采样点、脸框10%pad、DINO CLS与质心公式冻结。
2,178调度角色-镜头／1,089原视频全部处理，1,002候选脸图全部查看，排除2相机严重遮挡；60episode缓存和全部crop来源核验通过。
多人只作空间角色关联，同脸轨迹冲突记缺失；未进行全角色身份人工真值校验。脸部输入／有效集合改变，不能直接替代论文原人物区域表格。
全140保持paused_by_user，不新增视频／消融／VLM，所有原素材、冻结源文件、原报告与其他未提交改动保留。

**2026-10-14指标口径审计完成：人物定位污染已确认，原分数未修改。**
详见[审计／裁剪证据](../entitybench_metric_provenance_audit_20261014.md)。
所有516对实际尾帧输入核验通过；9组151张官方crop重算最大误差1.2e-7，无抽查数值错误。
新FLUX/Qwen有油桶、汽车等误入人物crop；两种FLUX的Daniel含手套／门把手crop，角色外观漂移也存在。
`cs_face`为人物区域DINO相似度，不等于脸部身份准确度；`cs_transition_boundary`近1主要是I2V边界保留。
论文matched-content boundary gate／coverage／实例加权与固定公开代码存在差异，DINO gate文字也有未解析矛盾；不能直接按论文表格排名。
审计单独保存，不修改冻结模型、评测源码、生成参数或原报告，不扩140／不增生成消融／不调用VLM。

**完成状态：2026-10-14北京时间11:07，新FLUX摄影版当前20集／363视频及官方非VLM评测完成；独立核验通过。**
112素材、40新3D/标定、191首帧、363视频，视频失败0；硬切55实际注入，136未注入，含1个Kaito参考对齐失败基础图回退。
两路完整episode评测182/181镜头，各4CPU线程，评分墙钟2456.38秒（40.9分钟），VLM调用0，39缺失指标null。
独立核验204聚合字段、全部媒体/来源SHA、363官方逐镜头调度记录、172实际解码尾帧逐像素一致。
原72首帧、旧报告、素材、冻结源码和其他未提交文件保持；全140仍paused_by_user，所有队列已退出。
结果见`outputs/entitybench_paper_flux_qwen_portraitflux_photo20_comparison/README.md`，
三路线查看页`review/viewer.html`，独立核验`review/final_integrity_audit.md`，抽样失败案例`review/visual_review.md`。
新FLUX画质/动态较高，但角色出现率0.8391及cs_face0.6836低于Qwen0.9335/0.6958；不宣称整体提升或插件因果收益。
抽查8/12组已保存对照图，存在Casey人物外观偏离、幽灵外观、多人遗漏等问题；全363镜头人工审片未完成。
非VLM/gate缺失，跨shot分数暂定，论文全140/2491口径不同，只作初步比较。不自动扩全量，不重跑完整结果。

## 历史进度与接续记录（最新状态见本文开头）

**参考对齐修复已实际通过：tmux续跑越过Kaito原失败镜头，首帧已达77/191（2026-10-13深夜，北京时间）。**
112素材和40新3D/标定完成；新兼容入口仅对PuLID参考对齐失败记录基础图回退。
原72首帧及相关488文件核验不变，24冻结源码未改，视频/官方评分尚未开始。
仍由`pretest/run_entitybench_flux_portraitagent_photo_20ep_resume.py`后台接续全部剩余阶段；
当前进度以status.json/firstframes_status.json和实际tmux为准，全140仍停止。

**2026-10-13晚间修复接续：原队列21:40在动画Kaito参考PuLID对齐失败处停止，72/191首帧完成；112素材、40新3D/标定全部完成。**
新入口`pretest/run_entitybench_flux_portraitagent_photo_20ep_resume.py`，沿用tmux和同一输出。
不修改冻结生成源文件、权重、prompt、seed、数值参数或官方算法；新增单独冻结的兼容worker，
仅将精确`facexlib align face fail`视为参考不可用并使用原纯文FLUX基础图策略，无参考绑定/3D注入，
完整镜头保留且逐镜头记录原因/兼容代码SHA。其他错误仍中断，未降低检测阈值或替换素材。
72已完成首帧、112参考及40三维源文件SHA已保存并核验；后续正常首帧仍走原PuLID/v7。
参考回退协议位于新输出`runtime/reference_gate_recovery.json`，原失败状态快照在
`runtime/reference_gate_recovery_before/`。5项回归检查通过。先查tmux和实时status，不重复启动旧入口。

**最新运行状态：摄影版已通过4张小样与实际头部预处理检查，完整20集后台已接续。**
输出`outputs/entitybench_flux_portraitagent_photo_i2v_a14b_4step_continuation_20ep/`；
tmux `entitybench_fluxportrait_a14b4step20`；入口`pretest/run_entitybench_flux_portraitagent_photo_20ep.py`。
先读status.json和实际tmux状态，不重复启动。191硬切/363视频/官方评分尚未完成。
FaceLift自动裁剪两张Viktor成功，无几何回退；半身参考可用，但原生脸193/248像素，需后续3D/标定核验。
112角色参考中40角色有3D资格；不借旧Gaussian。后台只用冻结的Qwen3.7-plus文本prompt，API调用0、VLM不测。
原native插画和初版Agent样本及代码全部保留，全140仍paused_by_user。详情见
[当前执行说明](../entitybench_flux_portraitagent_20ep_run.md)。

**最新准入计划：Portrait Agent摄影前缀版，2026-10-13北京时间。**
有效输出`outputs/entitybench_flux_portraitagent_photo_i2v_a14b_4step_continuation_20ep/`，
入口`pretest/run_entitybench_flux_portraitagent_photo_20ep.py`，tmux沿用`entitybench_fluxportrait_a14b4step20`。
20次Qwen3.7-plus文本调用、112角色输出已保存；19写实/1动画，实际FLUX写实prompt增加DSLR摄影/自然皮肤与布料前缀。
初版Agent四图因Silas仍带渲染感保留停止，不覆盖冻结计划。当前摄影版先做4候选风格验收；
看status.json和pane_dead确认是否已接续完整队列，禁止误恢复旧native/初版Agent输出。
本地后台不加载API key，不做VLM；仍仅当前20集，全140保持暂停。

**最新接续任务（2026-10-13，北京时间）：用户授权 Portrait Agent 文本提示词＋本地 FLUX 当前20集。**
新输出 `outputs/entitybench_flux_portraitagent_i2v_a14b_4step_continuation_20ep/`，说明见
[Portrait Agent FLUX20](../entitybench_flux_portraitagent_20ep_run.md)。
当前交互中由 Qwen3.7-plus 编写每集角色提示词，冻结后本地后台运行；不把个人 Token Plan 密钥用于后台批量调用。
原插画输出保持暂停并保留4候选；不覆盖其冻结脚本或计划。新入口
`pretest/run_entitybench_flux_portraitagent_20ep.py`，预定tmux `entitybench_fluxportrait_a14b4step20`。
先查看新输出 status.json、tmux pane_dead和实际进程再续跑。全140继续停止，仅当前20集，VLM仍不评测。

**最新状态（2026-10-13 17:37北京时间）：新FLUX原生素材20episode队列已停止，phase=paused_for_reference_style_review。**
用户明确反对准入样图的插画风；4个原候选全部保留，仅有2个已选参考，未进入3D/首帧/视频。先前“准入通过”只表示模型推理和单脸检查通过，风格验收不足；不应直接恢复此冻结计划。
原Qwen20、旧SDXL→FLUX20和全140暂停资产保持。当前官方仓库main仍为`3776439e7037b2128ac3b5d2d1ce63e424cca166`，150文件中只有5个代码/说明/启动文件，均无FLUX型号或素材生成配置。源码树与逐文件搜索证据在新输出review/upstream_flux_model_audit.json。
论文E.2仅写FLUX、由Portrait Agent推断风格，未明确dev/schnell/pro；不能声称本机FLUX.1-dev FP8是论文同checkpoint。需明确参考素材风格并小样验证后才能继续；改变prompt必须使用新的冻结计划/输出，不覆盖现有4候选。详见[新FLUX20说明](../entitybench_flux_native_20ep_run.md)。以下启动记录为历史快照。

**当前新任务（2026-10-13 17:29北京时间）：FLUX原生素材→PuLID-FLUX/v7→同4步A14B，当前20episode/363shot已启动后台。**
用户已选择现有FLUX.1-dev/PuLID单人参考路线，多人继续纯文生图；新素材对应全新3D，不借旧SDXL身份资产。
入口`pretest/run_entitybench_flux_native_20ep.py`；tmux`entitybench_fluxnative_a14b4step20`；输出`outputs/entitybench_flux_native_i2v_a14b_4step_continuation_20ep/`。
4个正式候选准入通过，平均29.3秒/张、峰值约16GiB，全部按SHA复用；出现插画风格，保留模型实际输出。新7项及既有17项协议检查通过。
空间约13GiB，预计新增5–7GiB，逐阶段5GiB守护；当前无需下载/扩容，不自动删产物。全140仍保持停止，VLM仍不测，不新增消融视频组。
已完成Qwen20和旧SDXL→FLUX20结果保持；本轮完成后自动保存论文/旧FLUX/Qwen/新FLUX比较到`outputs/entitybench_paper_flux_qwen_nativeflux20_comparison/`。
详见[新FLUX20执行说明](../entitybench_flux_native_20ep_run.md)。后续先检查tmux和status，不重复启动。

**最新完成状态（2026-10-13）：北京时间14:09已完成Qwen20/363官方非VLM评分并归档；16:48之后独立核验通过。下面11:30进度为历史快照。**
结果见[论文/FLUX/Qwen完整对比](../../outputs/entitybench_paper_flux_qwen20_comparison/README.md)，[完成核验](../../outputs/entitybench_paper_flux_qwen20_comparison/review/final_integrity_audit.md)。全量人工审片仍待完成，FLUX全140保持暂停。

最后更新：2026-10-14（北京时间）。仓库：`/root/autodl-tmp/movie`。

**新窗口先读：[Qwen20当前交接快照](../LLM_HANDOFF_QWEN20_20261013.md)。**
2026-10-13 11:30快照：112素材、39有效3D/标定、191首帧全部完成，A14B视频约292/363，
官方评价尚未启动。队列和归档watcher正常；实时以generation_report.json/评分状态为准。
完整首帧66实际注入、125未注入；最终图与是否注入看frame_record.json，不按文件名判断。
归档脚本磁盘文件缺失已补回，原watcher未重启；论文/FLUX已归档，Qwen评分完整后自动补齐。
[新窗口提示词](../qwen20_new_window_prompt.md)可直接复制。


**当前任务已切换（2026-10-12 21:03）：用户要求停止FLUX全140，先跑Qwen同一20episode。**
FLUX全140已停止，`status.json`为`paused_by_user`，新增可注入标定144/217完成并保留。
原FLUX20episode/363视频及官方非VLM分数已完成，保持只读，作为同范围系统路线比较基准。
Qwen新队列21:12启动，tmux `entitybench_qwen21_a14b4step20`，只覆盖冻结20episode/363shot，
不自动扩展全140、不新增消融视频组。当前入口`pretest/run_entitybench_qwen21_20ep_accelerated.py`（原始核心路线不变），
输出`outputs/entitybench_qwen21_i2v_a14b_4step_continuation_20ep/`，整体进度看status.json，
总日志`outputs/entitybench_qwen21_20ep_pipeline.log`，各阶段日志在新输出logs/。
112个硬切角色生成新Qwen素材，40个单角色硬切角色有3D准备资格；其他19个仅接续角色
不额外造参考，原全部131角色调度保留。新素材要重新FaceLift/标定，不能借用旧SDXL Gaussian。
已核验本机7份Qwen权重SHA并进入实际素材生成，首个候选107.3秒已保存。
然后两路新资产准备、191硬切首帧、363新A14B视频（172自己的尾帧接续）、来源审计，
再按evaluate-face-injection-v2两路官方非VLM评分及FLUX/Qwen同范围比较。
已有FLUX正式评分虽为串行，v2重新聚合和22视频并发一致性已验证，不因新调度重跑旧分数。
VLM仍跳过，保真度gate缺失，cs_face/cs_object暂定；旧三episode身份指标不混入官方表。
22:59完成独立设备放置/两路吞吐验证，素材阶段改为单进程CUDA驻留：两候选各约21秒，
原各约105秒；两图PNG SHA完全一致。显存约37–38GiB，29份完整素材保留校验复用。
两进程model offload两图约89秒，慢于CUDA驻留两图42秒；不使用素材双进程。
源码/角色/seed/40步原计划冻结，变更证据在runtime/reference_acceleration.json。
首帧仍sequential offload（多参考驻GPU未测），标定两路、视频与官方v2不变。
当前磁盘约18.5GiB空闲，阶段保留5GiB守护，不自动删历史；各GPU阶段分开。
详见[Qwen20episode运行说明](../entitybench_qwen21_20ep_run.md)、
[下一步计划与已有Qwen证据](../entitybench_qwen21_next_experiment_plan.md)。
全140暂停产物见[原完整140说明](../entitybench_a14b_full140_run.md)。
2026-10-13用户要求结果归档：论文四方法的12共同指标及FLUX原始报告已存到
outputs/entitybench_paper_flux_qwen20_comparison/，README.md为初步对比表。
独立tmux entitybench_results_archive20等待Qwen完整20episode/363shot官方评分后补齐原始报告
和总表，未完成字段null；论文全140与项目20/gate/聚合差异写清，不按严格排名解释。
首帧看新Qwen输出firstframes/，最终文件按frame_record.json的final_image，不把control.png当作最终首帧。

**本轮实验（2026-10-11 23:53启动）：20episode / 363shot 全部改用LightX2V 4步蒸馏
Wan2.2-I2V-A14B，硬切复用191个已核验首帧，172个接续仅使用本路线上一视频尾帧，不再次注入。**
用户确认仍跳过VLM，本轮只跑官方非VLM指标，不运行项目IF/参考DINO/脸框质心辅助。
官方实际6项VBench、GroundingDINO/CLIP实体出现率、DINO跨镜头与接续边界评测已通过
两镜头实际接入检查（40秒，零VLM调用）；这些检查结果不进入正式评分。
旧40步队列已按要求停止，7个完整镜头保留，不混入新4步视频。
**当前状态：2026-10-12 09:41已生成363/363镜头，失败0；10:42官方非VLM
20/20 episode评分完成、VLM调用0，363来源/172接续通过审计，人工效果审查仍待完成。**

后续官方评测使用新skill
[evaluate-face-injection-v2](/root/.codex/skills/evaluate-face-injection-v2/SKILL.md)：
默认单卡2进程、按完整episode分片，每路4个CPU线程、独立缓存、最后官方统一聚合。
先测小样本吞吐和分数一致性再调3/4路；本轮已完成的分数无需因新skill重跑。
旧skill保留用于配对InsightFace/参考DINO身份诊断。

- 当前入口：`pretest/run_entitybench_a14b_4step_20ep.py`，tmux `entitybench_a14b4step20`。
- 新输出：`outputs/entitybench_pulid_i2v_a14b_4step_continuation_20ep/`；状态`status.json`，
  逐镜头进度`generation_report.json`，日志`pipeline.log`及`logs/generate_formal.log`。
- 完成生成后自动审计363个来源/尾帧依赖、拼接20条episode，再运行官方非VLM评测。
- 最新[完整方案](../entitybench_collage_vs_3d_evaluation_plan.md)、
  [4步执行/续跑说明](../entitybench_a14b_4step_run.md)。旧40步计划和下面的配对结果是历史记录。
- 首轮全量生成与官方非VLM评分均已完成；历史小样本89秒/shot估算不再是实时进度。
- VLM外观保真度gate缺失，因此官方`cs_face/cs_object`暂定，不能直接冒充完整论文表格分数。

历史已完成结果：PuLID-FLUX、SDXL/IP-Adapter 的 20 episode 单人首帧与视频评测已完成，
详见第 11、12 节。528/528 个视频任务完成，失败 0；2026-10-10 核对时 tmux 无运行会话。
本会话后续补齐了四项指标解释、动漫风格来源核查和 20 张首帧正例对比图（其中 14 张为
两批写实正例）。新窗口先读第 13 节和 [20 episode 评估说明](../entitybench_20ep_evaluation.md)。

本轮随后补齐了 20 episode 的 DINOv2 跨镜头人脸外观质心相似度；见第 14 节。
该指标采用论文质心公式，但复用现有脸部裁剪，不是官方 `cs_face`，不替代原有四项身份指标。

## 1. 接手顺序

1. 保留未提交工作树，不要 reset 或 checkout；前后端目录还有用户的并行改动。
2. 阅读本文和 [20 episode 评估说明](../entitybench_20ep_evaluation.md)；
   `EPISODE_TEST_RESULTS.md` 仍是三 episode/Qwen 结果，不能当作 20 episode 总表。执行或审计评测时使用
   `/root/.codex/skills/evaluate-face-injection/SKILL.md`（`$evaluate-face-injection`）。
3. 核对 GPU、模型路径和 Python 环境。
4. `c3c9fcccd558befb7302a046794e789c9f71f615` 是本轮 Qwen 全 episode 批处理前的基线；
   以仓库最新提交和本文为准，不要恢复已删除的旧调色、自注意力或旧多人路径。

## 2. 当前任务定义

本项目当前评测一个首帧/关键帧身份增强插件：

```text
文本 + 固定身份/3D 资产
→ 官方基础图像管线正常生成 Control，并截取进入注入点的 latent
→ 从该 latent 预测 pred_x0，完成人脸检测、角色映射、姿态标定和 v7 reference
→ Treatment 从截取的同一 latent 继续去噪
→ 仅 Treatment 接收逐角色 v7 局部 3D residual
→ 相同 Wan2.2 生成成对视频
→ 原始身份参考锚定的逐帧评测
```

EntityBench 提供完整有序 episode、实体调度和逐镜头提示词。Wan2.2 当前逐镜头独立运行，
没有跨镜头历史记忆。不能把本轮结果表述为完整的一句话多镜头智能体评测。

## 3. 当前唯一有效算法

只保留 v7 color-safe：

1. 从注入起点 `pred_x0` 检测目标人脸和 pitch/yaw/roll；
2. 读取该角色自己的 `gaussians.ply.pose_calibration.json`；
3. 从 FaceLift Gaussian 连续渲染目标姿态并做 InsightFace 姿态回检；
4. 在目标/3D 皮肤交集估计低频 log-RGB gain，并应用到纯 3D inner-face；
5. BiSeNet v7 mask 排除发际线、太阳穴、外脸颊、下巴边缘、耳朵和耳饰；
6. 将调色后的纯 3D reference 编码为 latent，在相同 timestep 做局部 trajectory residual；
7. 正常脸基准强度 `0.4`；小脸继续使用 v7 的自适应降强度和提前结束。
8. Qwen 路线在 step 12 用角色参考图与 `pred_x0` 目标脸的 InsightFace cosine
   选择窗口：`<0.40` 注入 6 步（step 12～17），`>=0.40` 只注入 step 12。

公式：

```text
target_next = scheduler_step(target_state)
reference_next = 同 timestep 的带噪 3D reference latent
target_next += mask * strength * (reference_next - target_next)
```

### 3.1 统一执行协议（不得自行改动）

PuLID-FLUX、SDXL/IP-Adapter 和 Qwen-Image-2.1 统一使用“中期 `pred_x0` +
同一中间 latent 分叉”。Qwen 的严格实现是：Control 必须完整走未修改的官方
`QwenImage21Pipeline.__call__`；callback 只读取并保存进入 step 12 的 latent，不能用手写
采样器代替 Control。Control 完成后 Treatment 才从该 latent 继续：

```text
官方 Control trajectory
  → callback 截取 shared latent
  → 注入点 pred_x0
  → 检测 / 角色映射 / pitch-yaw-roll / 3D 渲染 / 调色 / v7 mask
  → Treatment 在同 timestep 加局部 residual 后继续去噪
```

Qwen 20 步的 step 12 对应 PuLID/IP 50 步的 step 30，均为约 60% 去噪进度。
不得再用“完整生成 Control 后，依据最终 Control 准备 3D，然后从噪声重跑
Treatment”作为当前方案。该两遍式仅作为历史 pilot，不能与统一协议结果混合。

上述 `0.40` 阈值和 1/6 步二选一规则已经用户确认。除此之外仍只允许单变量消融；
不得自行加入新 mask、新权重规则、调色下限或其他 gate。

## 4. 安全回退

PuLID-FLUX/IP-Adapter 单人路线遇到以下情况时 Treatment 复用 Control。Qwen 多参考路线
改为逐角色容错：缺脸、映射失败、3D 渲染/调色/mask/latent reference 失败时只跳过该角色，
其余成功角色继续注入；只有没有任何可用角色或命中整镜头闭眼冲突 gate 时才整张复用 Control。

相关 gate 包括：

- 无可靠脸或脸高 `<24 px`；
- Gaussian 渲染失败或姿态回检不通过；
- v7 color-safe core 为空；
- 提示词明确要求闭眼，记录 `skip_reason=prompt_eye_closure_conflict`；
- 当前 PuLID-FLUX/IP-Adapter 路线的多人镜头。

“复用”意味着 Control 正常生成，Treatment 文件与 Control 完全相同；不是删除难例。

run1517 / 4:2 已验证闭眼 gate。IP-Adapter 当前结果状态为
`completed_control_reuse`，Control/Treatment SHA256 相同，身份 cosine 均为 `0.339437`；PuLID-FLUX
对应身份 cosine 均为 `0.822210`。两条路线的 Wan 视频也已重跑并验证 Control/Treatment 文件哈希
完全相同，视频身份均值差为 `0`。

## 5. 当前代码入口

- PuLID-FLUX 单人 v7：`multishot/pulid_flux_inner_face_experiment.py`
- IP-Adapter 单人 v7：`multishot/ip_adapter_pulid_style_injection_experiment.py`
- 闭眼提示词 gate：`multishot/prompt_injection_safety.py`
- FaceLift 渲染/姿态回检：`multishot/mcp_asset_server.py`
- Gaussian 标定：`multishot/facelift_pose_calibration.py`
- 当前多人 v7 reference 工具：`multishot/multi_face_reference.py`
- IP episode 首帧：`pretest/prepare_entitybench_ip_adapter_pairs.py`
- PuLID episode 首帧：`pretest/prepare_entitybench_pulid_pairs.py`
- Wan：`pretest/run_wan22_i2v_manifest.py`
- 视频 InsightFace：`pretest/evaluate_video_identity.py`
- 参考图 DINOv2：`pretest/evaluate_reference_dino_identity.py`
- 当前 episode 总结：`pretest/summarize_current_episode_results.py`
- Qwen 多参考 Control：`pretest/run_qwen_image21_multiref.py`
- Qwen 多脸 reference 准备：`pretest/prepare_qwen_multiface_residual.py`
- Qwen 同步多人 residual：`pretest/run_qwen_image21_multiface_residual.py`
- Qwen 多人首帧逐角色评测：`pretest/evaluate_qwen_multiface_pair.py`
- Qwen 官方 Control latent 分叉、逐角色容错、自适应 1/6 步入口：
  `pretest/run_qwen_image21_shared_prefix_adaptive.py`
- Qwen 自适应三 episode 全链路：
  `pretest/run_qwen_image21_shared_prefix_adaptive_full3.sh`
- Qwen 单/多人视频逐角色评测：`pretest/evaluate_qwen_multiface_videos.py`
- Qwen 三 episode 可恢复批处理：`pretest/run_qwen_image21_entitybench_batch.py`
- Qwen 三 episode InsightFace 汇总：`pretest/evaluate_qwen_image21_entitybench_batch.py`
- Qwen 三 episode DINOv2：`pretest/evaluate_qwen_reference_dino.py`
- Qwen 完整逐 shot 可视化：`pretest/render_qwen_evaluation_comparisons.py`
- Qwen 历史/当前汇总：`pretest/compare_qwen_historical_adaptive.py`

`--shot` 单镜头重跑已改成增量更新：只替换指定镜头，保留 episode 报告和 manifest 中其余镜头。

## 6. 环境和模型

环境：

- 主图像/注入/评测：`/root/autodl-tmp/movie/.venv`
- Wan2.2：`/root/autodl-tmp/wan22-venv`
- EntityBench evaluator：`/root/autodl-tmp/envs/entitybench-eval`
- Qwen-Image-2.1：`/root/autodl-tmp/qwen-image21-venv`

模型：

- FLUX/PuLID：`third_party/PuLID`
- SDXL：`models/diffusion/sdxl-base-1.0`
- IP-Adapter：`models/ip_adapter/h94-IP-Adapter`
- Wan2.2-TI2V-5B：`models/video/Wan2.2-TI2V-5B`（约 32 GB）
- Qwen-Image-2.1：`models/diffusion/qwen-image-2.1`（约 31 GB，全部权重哈希已校验）
- DINOv2：`/root/autodl-tmp/models/dinov2-base`

Qwen 固定 revision：`790c92633540aa0cb11d9abf19eb46d861714758`。48 GB 单卡使用
`sequential_cpu_offload`；不要在未知显存占用下直接 full-CUDA。

## 7. 数据、资产和正式范围

EntityBench：`benchmarks/entitybench/`。

原始三 episode 范围（20 episode 扩展另见第 11、12 节）：

- run719：12 镜头，8 个单角色；资产在
  `outputs/entitybench_wan22_smoke/episode_00053051/assets/`；
- run893：10 镜头，5 个单角色；
- run1517：12 镜头，10 个单角色；
- run893/run1517 资产在 `outputs/entitybench_pilot3/episode_*/assets/`。

三个 episode 共 12 个角色，均有固定参考图、FaceLift Gaussian 和独立姿态标定。资产必须
episode-local，不能跨 episode 按同名角色复用。

## 8. 当前权威结果

- PuLID run719：`outputs/entitybench_wan22_colornested_v7_s04_12/episode_00053051/`
- IP run719：`outputs/entitybench_wan22_ip_adapter_v7_s04_12/episode_00053051/`
- PuLID run893/run1517：`outputs/entitybench_wan22_pulid_v7_pilot3/`
- IP run893/run1517：`outputs/entitybench_wan22_ip_adapter_v7_pilot3/`
- 参考图 DINOv2：`outputs/entitybench_reference_dino_pilot3/reference_dino_summary.json`
- Qwen 历史两遍式三 episode：`outputs/entitybench_qwen_image21_v7_full3/`，其中
  `evaluation_report.json` 为 InsightFace，`dino_evaluation_report.json` 为参考图锚定 DINOv2；
- Qwen 当前官方 Control 分叉、逐角色容错三 episode：
  `outputs/entitybench_qwen_image21_official_control_partial_adaptive_040_full3/`；
  其中 `HISTORICAL_VS_ADAPTIVE.md` 是历史/当前完整路线对照；
- 当前机器可读总表：`outputs/entitybench_current_summary.json`
- 当前 Markdown 总表：`EPISODE_TEST_RESULTS.md`

以上项目根目录总表仍对应三 episode 范围。最新 PuLID-FLUX/IP-Adapter 的 20 episode
首帧、视频结果分别以第 11、12 节的独立报告为准，不能把三 episode 总表当作最新扩展汇总。

重新汇总：

```bash
cd /root/autodl-tmp/movie
.venv/bin/python pretest/summarize_current_episode_results.py
```

完整评测流程、可视化契约和验收规则已迁移到自动发现的 Codex skill：
`/root/.codex/skills/evaluate-face-injection/SKILL.md`。后续要求大模型使用
`$evaluate-face-injection` 执行或审计评测；不再维护独立的评测方案 Markdown。

该报告包含每个 episode 和每个镜头的触发状态、跳过原因、首帧 InsightFace、视频
InsightFace 与参考图 DINOv2。EntityBench 非 API 指标需要按当前闭眼 gate 重算；VLM/LLM API
指标未运行，缺失项不能当作 0。

闭眼冲突 gate 加入前曾报告过一版历史快照：PuLID-FLUX 17 个注入镜头，InsightFace
首帧/视频 Δ `+0.0331/+0.0344`，DINOv2 `+0.0071/+0.0072`；IP-Adapter 18 个注入镜头，
InsightFace `+0.1029/+0.0831`，DINOv2 `+0.0273/+0.0092`。这组数仅用于追溯，当前主表必须
以自动重算的 gate 后 16/17 镜头口径为准。

## 9. 当前多人边界

PuLID-FLUX 与当前 SDXL/IP-Adapter 不再执行多人注入，因为它们没有可靠的多参考身份绑定。
多人研究保留在 Qwen-Image-2.1 路线：

```text
多张角色参考图 + 文本
→ Qwen-Image-2.1 原生 multi-reference Control
→ 每张检测脸独立姿态渲染、调色和 v7 mask
→ 同一步同步叠加多脸 residual
```

通用多人能力没有取消，只是从不合适的基础模型迁移到原生多参考模型。

最初已完成 run719 / shot 1:1 的 Qwen 多人 pilot：

- 首帧 Julian：`0.674653 → 0.735510`，Δ `+0.060858`；
- 首帧 Viktor：`0.445645 → 0.454448`，Δ `+0.008803`；
- 视频 mean Julian：Δ `+0.074233`；
- 视频 mean Viktor：Δ `-0.014289`；
- 视频两角色宏平均：Δ `+0.029972`，两角色检测覆盖率均为 `1.0`。

结果目录：`outputs/qwen_image21_multiface_v7/run719_shot_1_1/`。该 pilot 证明同步多人链路可运行，
但存在角色间增益不均匀。

2026-09-27 已完成三个 episode 的 Qwen 两遍式历史路线：

- 范围：34 个 shot，23 个单人、11 个多人；
- 首帧：34/34 Control/Treatment 已完成；28 个实际注入、6 个安全复用 Control；
- 回退构成：3 个背脸/缺脸、1 个 Qwen 多人漏角色、1 个闭眼冲突、1 个 v7 core 为空；
- 所有 28 个可准备 shot 的 Treatment 均成功生成，没有运行期降级；
- Wan manifest：68 个 job，其中 62 个实际推理、6 个 Treatment 复用对应 Control；
- 运行目录：`outputs/entitybench_qwen_image21_v7_full3/`；
- Wan 视频已全部完成：68 个 job 中 62 个实际推理、6 个 Treatment 复用 Control，失败 0；
- 逐角色评测已完成：40 个角色-shot 对。InsightFace 首帧/视频 mean Δ 分别为
  `-0.0627/-0.0510`；DINOv2 首帧/视频 mean Δ 分别为 `+0.0043/+0.0024`；
- 34 个 shot 的 `comparison.jpg` 已统一为身份参考、连续 3D 渲染、调色 3D reference、
  最终 mask、Control、Treatment，并标注逐角色指标。

这批数值保留用于追溯，但其 Treatment 使用了最终 Control 准备姿态、mask
和调色 reference，不再作为统一插件协议的算法结论。

### 9.1 Qwen 官方 Control 分叉、逐角色容错、自适应 1/6 步

用户已确认只保留两种窗口：step 12 单步和历史 6 步。调度依据是 step 12 `pred_x0`
中角色参考图与已分配目标脸的 InsightFace cosine：低于 `0.40` 用 6 步，否则用 1 步。
多人镜头逐角色独立调度，同一步仍使用归一化后的多 mask residual 同步叠加。

阈值依据之一是 run719 / 4:1：step-12 cosine 为 `0.2656`；单步只得到
`0.3483→0.3527 (+0.0044)`，而 canonical shared-prefix 6 步为
`0.3483→0.4502 (+0.1018)`，且接近历史两遍式 6 步的 `+0.1009`。

2026-09-28 已完成修正后的 34/34 个首帧、68/68 个 Wan 视频及全部非 API 评测：

- Control 完整走官方 `QwenImage21Pipeline.__call__`，callback 只截取进入 step 12 的 latent；
  34/34 个 Control PNG 与历史官方 Control 逐字节一致；
- 30 个 shot 实际注入、4 个整镜头安全复用 Control，其中 3 个 shot 是部分角色注入；
- 48 个计划角色中 41 个注入、7 个跳过；28 个角色走 1 步、13 个走 6 步；
- 首帧 InsightFace 为 `0.5376→0.5264 (-0.0112)`，`18/41` 个角色为正；
- 视频 InsightFace mean 为 `0.4078→0.3942 (-0.0136)`，`17/41` 个角色为正；
- DINOv2 首帧为 `0.6093→0.6037 (-0.0056)`，视频 mean 为
  `0.5222→0.5238 (+0.0016)`；两者分别有 `21/41`、`19/41` 个角色为正；
- Wan 68 个任务包括 34 个哈希验证后的历史 Control 视频复用、30 个新 Treatment 推理、
  4 个整镜头 Treatment 复用对应 Control；失败 0；
- 历史两遍式 6 步为 IF 首帧/视频 `-0.0627/-0.0510`。修正版明显降低破坏，但 IF 均值仍为负，
  不能表述为稳定身份提升。

当前权威输出根目录：
`outputs/entitybench_qwen_image21_official_control_partial_adaptive_040_full3/`。最终报告为
`evaluation_report.json`、`dino_evaluation_report.json`，34 个 shot 均有 `comparison.jpg`；
历史/当前完整路线对照见 `HISTORICAL_VS_ADAPTIVE.md`。

相同的 34 个 Control 图像已经哈希冻结。历史/当前共有的 39 个可评角色对中，Control identity
cosine 最大绝对差为 `0.001216`；这来自两条路线使用不同注入点 bbox/角色映射元数据评测同一张
Control，而不是 Control 像素发生变化。两条路线的 Treatment 执行协议也不同，因此仍属于完整
路线对比，不是仅注入步数不同的严格消融。

肉眼检查单人正例、负例、部分角色注入和闭眼回退：角色映射与面板完整，未观察到明显 mask
越界或头发泄漏；高相似度正脸仍会出现轻微脸型/质感重绘，与 run1517 的 IF 负增益一致。

旧目录 `outputs/entitybench_qwen_image21_shared_prefix_adaptive_040_full3/` 使用手写 Control
采样路径，仅保留为排错记录，不得再作为当前算法结论或与官方 Control 结果混算。

此前 1/2/5 步和临时 `gain floor=0.55` 的实验目录已移到系统回收站；当前入口不再暴露
该临时调色参数。这里删除的是试验开关，不是 v7 调色：低频 log-RGB harmonization、姿态标定、
纯 3D reference 和 color-safe mask 均保持不变。

run893 / 6:1 的 Qwen Control 计划 4 人但生成了额外背景人物；当前只选择面积最大的 4 张主脸。
该镜头仍需在最终完整 comparison 中单独检查，不得只报告宏平均。

## 10. 下一步

1. 20 episode 单人首帧和视频扩展已完成；优先按脸部尺寸、难度和 episode 分析视频增益、
   负例与检测覆盖率，尤其审计 IP-Adapter 较低的绝对身份分数；
2. Qwen 34-shot 修正版全链路已完成，继续分析高身份单步组，尤其 run1517 的高置信正脸重绘；
3. 决定 `0.40` 阈值是否保留时，继续同时报告 IF、DINOv2、绝对值、差值和肉眼结果；
4. 不重新引入已删除的 2/5 步、临时调色下限、self-attention、旧调色路径或两遍式
   “先完整 Control 再从噪声重跑 Treatment”。

## 11. 2026-09-29 至 2026-10-01：20 episode 首帧扩展评测

用户要求先只扩展 PuLID-FLUX 和 SDXL/IP-Adapter 的首帧评测，不跑 Wan 视频。正式清单在：

`outputs/entitybench_firstframe_pulid_ip_20ep/manifest.json`

清单在看结果前冻结，包含已有 4 个 episode 和按难度确定性抽取的新增 16 个 episode：

- 20 个 episode：Easy 11、Medium 6、Hard 3；
- 363 个官方 shot，其中 132 个单人 shot 进入这两条插件路线；
- 新增部分为 105 个单人 shot、46 份 episode-local 角色资产；
- 多人和无人镜头不在本轮范围内；没有把它们伪装成负例或零分；
- 固定 v7 color-safe mask、注入强度 `0.4`、最多 12 个 active steps、最小脸高 24 px；
- Control/Treatment 保持同 prompt、seed/noise、资产、分辨率和模型，仅 v7 residual 不同；
- 明确闭眼、未检测到可靠脸、脸过小或 v7 core 为空时安全复用 Control。

可恢复运行入口：

- `pretest/run_entitybench_firstframe_pulid_ip_batch.py`
- `pretest/run_entitybench_firstframe_pulid_ip_20ep.sh`
- IP 单人范围通过 `pretest/prepare_entitybench_ip_adapter_pairs.py --single-character-only`
- 汇总器：`pretest/evaluate_entitybench_firstframe_pulid_ip_batch.py`

本轮 tmux `entitybench_ff20` 已正常结束；不存在仍在后台运行的任务。生成完整性：资产 20/20、
PuLID 报告 20/20、IP 报告 20/20。逐 shot 的 Control、Treatment、原参考、3D render、调色 reference
和 comparison 均已落盘。此阶段只生成首帧指标；后续第 12 节已补齐视频和逐帧评测。
VLM/LLM API 指标仍未运行。

角色原始正脸素材由 `pretest/prepare_entitybench_character_assets.py` 显式指定
SDXL Base 1.0（`stabilityai/stable-diffusion-xl-base-1.0`）生成，再用 FaceLift 构建 Gaussian。
通用 diffusion 后端的默认 `juggernaut-xl-v9` 不代表本轮评测素材使用的模型。

实际触发注入且 Control/Treatment 都可测量的 micro mean：

| 路线 | 单人 shot | 实际注入 | 安全复用 Control | InsightFace C→T (Δ) | DINOv2 C→T (Δ) |
|---|---:|---:|---:|---:|---:|
| PuLID-FLUX | 132 | 104 | 28 | `0.6825→0.7160 (+0.0335)` | `0.7513→0.7685 (+0.0172)` |
| IP-Adapter | 132 | 84 | 48 | `0.0831→0.1997 (+0.1166)` | `0.4432→0.4793 (+0.0360)` |

PuLID 的 104 个有效注入对中 86 正、18 负。IP 有 84 个实际注入 shot，但其中 3 个最终条件缺少
可靠 InsightFace/DINO face box，因此聚合使用 81 个有效对：69 正、12 负。IP 的绝对身份分数很低，
后续必须结合逐 shot 参考资产、远景/动画风格和检测质量审计，不能只用较大的正增益下结论。

权威输出：

- Markdown 总表：`outputs/entitybench_firstframe_pulid_ip_20ep/evaluation/firstframe_evaluation.md`
- 逐难度、逐 episode、逐 shot JSON：
  `outputs/entitybench_firstframe_pulid_ip_20ep/evaluation/firstframe_evaluation.json`
- 运行状态：`outputs/entitybench_firstframe_pulid_ip_20ep/status.json`
- 总日志：`outputs/entitybench_firstframe_pulid_ip_20ep/logs/tmux.log`

已肉眼抽查最大正负例。PuLID 最大正例为 run53/3:3，最大负例为 run323/2:3；IP 最大正例为
run146/28:1，最大负例为 run354/4:1。负例必须保留，不得依据分数事后剔除。

### 11.1 Git 状态与交接注意事项

2026-10-01 交接时首帧扩展代码尚未提交；后续已通过 `b27393f`
（`Add 20-episode first-frame evaluation pipeline`）提交。2026-10-04 核对时分支为 `main`，
`HEAD` 和本地 `origin/main` 均为 `b6c06e9`。本次视频协调器、视频评测/续跑修改和交接文档
当时仍在工作树中，尚未提交。后续已通过 `65a07c2`
（`Add 20-episode paired video evaluation and update handoff`）提交并推送到 GitHub `main`。
2026-10-10 核对时本地 `main` / `origin/main` 为 `a62e5b4`（后续产品文档/同源访问提交）。
`docs/frontend_agent_architecture.md` 有用户未提交改动，`experiment_output/` 为未跟踪输出，
均须保留；本次交接更新另产生评测文档改动。`outputs/` 下报告、视频与精选图为本机忽略文件，
没有随着 `65a07c2` 推送到 GitHub。不要 stage 实验输出。

后续提交评测代码时仍应保留用户工作树，检查是否有并行 app/frontend/product-pipeline 改动：

- 保留整个脏工作树，禁止 reset/checkout 覆盖用户改动；
- 只 stage 评测文档、评测/运行脚本、Qwen 当前评测链路、
  `multishot/face_analysis_backend.py` 和 `multishot/facelift_pose_calibration.py`；
- 不 stage `app/`、`frontend/`、产品 pipeline 改动、`experiment_output/` 或任何 `outputs/`；
- 提交前重新执行 Python compile、shell `bash -n`、`git diff --check`，检查 staged diff 后再 push。

## 12. 2026-10-04：20 episode 单人镜头视频扩展已完成

用户要求将第 11 节的 PuLID-FLUX 和 SDXL/IP-Adapter 首帧队列补齐 Wan 视频。
tmux `entitybench_video20` 已结束；`progress.json` 为 `complete`，当前没有后台评测任务。

- 执行入口：`pretest/run_entitybench_firstframe_pulid_ip_20ep_videos.sh`。
- 可恢复协调器：`pretest/run_entitybench_firstframe_pulid_ip_20ep_videos.py`。
- 输出：`outputs/entitybench_firstframe_pulid_ip_20ep/video_evaluation/`，其中 `wan_manifest.json`、`first_frame_hashes.json`、`wan_report.json`、`progress.json` 和 `logs/tmux.log` 可核对进度与复跑。
- 范围：20 episode × 两条路线，各 132 个单人 shot，共 264 组 Control/Treatment、528 个视频任务。多人与无人镜头不在这次扩展范围。
- 最终完整性：528/528 个视频任务完成，408 个新生成、44 个历史视频经过输入/输出 SHA256 及 Wan 参数验证后复用、76 个安全回退 Treatment 复用 Control；失败 0。
- 参数沿用 Wan2.2-TI2V-5B、1280×704、49 帧、24 fps、50 steps、shift 5.0、guidance 5.0、各镜头原 seed。Control/Treatment 仅首帧的 v7 注入不同。
- 20 episode × 两路线的 40 份视频 InsightFace 报告、40 个 DINOv2 case 和 264 组配对明细已完成。InsightFace 报告中全部 528 条视频均解码为 49 帧。
- 身份视频跟踪使用首帧 bbox 初始化的空间连续跟踪，缺少初始 bbox 时从面积最大的检测脸启动；不按最高身份相似度逐帧挑脸。DINOv2 视频复用相同的跟踪 bbox，face crop expansion 为 0.15。

实际注入且两侧可测量的配对均值如下。首帧与视频均报告 Control、Treatment 实际值及差值：

| 路线 | 单人 shot | 实际注入 / 安全复用 | 首帧 InsightFace C→T (Δ) | 视频 InsightFace mean C→T (Δ) | 首帧 DINOv2 C→T (Δ) | 视频 DINOv2 mean C→T (Δ) |
|---|---:|---:|---:|---:|---:|---:|
| PuLID-FLUX | 132 | 104 / 28 | `0.6825→0.7160 (+0.0335)` | `0.5697→0.5852 (+0.0155)` | `0.7513→0.7685 (+0.0172)` | `0.6522→0.6624 (+0.0102)` |
| IP-Adapter | 132 | 84 / 48 | `0.0831→0.1997 (+0.1166)` | `0.0782→0.1653 (+0.0870)` | `0.4432→0.4793 (+0.0360)` | `0.4165→0.4425 (+0.0261)` |

统计口径与限制：

- 视频先对每条视频可检测/可评分帧求均值，再对有效注入镜头等权求均值；上表视频 DINOv2 使用 `video_frame_mean`，不是 shot median。
- PuLID 首帧/视频有效对均为 104；视频 InsightFace 74 正、30 负，视频 DINOv2 63 正、41 负。
- IP 首帧有效对为 81，视频有效对为 84；视频 InsightFace 71 正、13 负，视频 DINOv2 58 正、26 负。
  首帧与视频样本集合不同，不能直接用两组整体增量相除宣称严格的增益保留率。
- 缺脸帧分数保留为 null，并计入检测覆盖率分母，不作为 0 分；安全复用镜头保留在完整清单，未计入实际注入均值。
- 两路线首帧、视频均值均为正增益，但存在负例；IP 视频 Treatment 身份分数仅 `0.1653`，不能仅凭较大增量宣称高质量身份保持。
- EntityBench validated 清单共 140 episode、2,491 shot、847 个单人 shot；本轮覆盖 20 episode 的 132 个单人 shot。
  这是 benchmark 子集评测，每镜头为独立的 49 帧短视频（约 2.04 秒），没有跨镜头记忆，多人和无人镜头未纳入本轮扩展。
- VLM/LLM API 指标未运行，缺失项不能当作 0。

权威输出（均在 `outputs/entitybench_firstframe_pulid_ip_20ep/video_evaluation/`）：

- `video_evaluation.md`：路线首帧/视频总表；当前落盘版本没有逐 episode Markdown 汇总。
- `video_evaluation.json`：路线指标和 264 组逐 shot、逐角色视频配对明细；当前落盘版本顶层为 `routes`、`shots`。
- `cases.json`：40 个 case 的 manifest 与 InsightFace 报告路径索引，**不是逐帧评分文件**。
- `cases/<episode_id>/<route>/video_identity.json`：每条视频的逐帧身份分数、跟踪 bbox、检测覆盖率、first/mean/p10/last 和配对差值；route 为 `pulid_flux` 或 `ip_adapter`。
- `dinov2_video.json`、`dinov2_video.md`：40 个 case 的参考图锚定 DINOv2 和逐 episode/路线聚合。
- `videos/<episode_id>/<route>/shot_<scene>_<shot>/control.mp4` 与 `treatment.mp4`：正式视频文件。
- `wan_report.json`、`wan_manifest.json`、`first_frame_hashes.json`、`progress.json`、`logs/tmux.log`：任务完整性、输入/输出哈希与运行记录。

用户要求实测双进程并行：隔离目录 `parallel_probe/` 的测试视频完成，耗时约 365 秒；
单进程近期每条约 184 秒，两进程同时去噪时每步约 6.7 秒（单进程约 3.1 秒），
合计显存一度约 43 GB。没有观察到明显吞吐提升，正式队列保持单进程。
`parallel_probe/` 不属于正式 528 条视频/264 组配对，不得混入评测。

后续只审计现有结果时读取上述报告，不要重新生成视频。若需要恢复中断任务，可重新运行同一入口，
协调器会校验已有输入图及视频哈希。当前代码的汇总器比后台启动时的版本多了字段和逐 episode 表，
再次执行 `summarize` 会扩展报告结构；不可假定这些新字段已存在于本次落盘报告。

## 13. 2026-10-10：新窗口接续、对比图与风格核查

本会话最新任务是展示有首帧增益的对比图，随后用户要求改找非动漫例子，并再次追加一批。
全部使用已有实验文件排版，没有重新生成首帧、视频，也没有修改正式配对分数、清单或 gate。
当前只是交接文档更新，未启动新实验。

### 13.1 首帧 DINOv2 已完成，旧视频表只是漏列

首帧 InsightFace、首帧 DINOv2、视频 InsightFace mean、视频 DINOv2 mean 四项都已完成。
旧 `video_evaluation/video_evaluation.md` 只有三项，旧 JSON 路线汇总也缺少
`firstframe_dinov2` 字段；首帧 DINOv2 应从 `evaluation/firstframe_evaluation.json` 的
`routes.<route>.reference_dinov2` 和 `shots` 读取，不能解释成“尚未评测”。四项总表见第 12 节。

### 13.2 对比图格式与本机索引

用户认可的顺序是 **2 行 × 3 列**：

| 左列 | 中列 | 右列 |
|---|---|---|
| 原始身份参考 | 连续 FaceLift 3D 渲染 | 调色并对齐后的纯 3D reference |
| 最终 v7 注入 mask | Control 最终首帧 | Treatment 最终首帧 |

标题写路线、run/episode、shot、角色；Control/Treatment 下分别写实际首帧 InsightFace 和
DINOv2，底部写两项 `Control→Treatment (Δ)`。mask 必须来自实际实验，不能用 `pred_x0`、
3D 参考 mask 或手绘区域代替。首帧展示图明确标注 `FIRST FRAME`；完整首帧/视频评估图再补
视频 InsightFace/DINOv2 mean，首帧分数不能冒充视频分数。详细文件映射、复现方式、多人和回退
格式见 [评估说明的对比图契约](../entitybench_20ep_evaluation.md#对比图格式)。

输出根目录：`outputs/entitybench_firstframe_pulid_ip_20ep/`。

| 子目录 | 已展示数量 | 内容 |
|---|---:|---|
| `selected_positive_firstframes/` | 6 | 首批大增益正例，混合写实/动画风格；4 张明显偏动画 |
| `selected_realistic_positive_firstframes/` | 6 | 第一批写实正例，PuLID/IP 各 3 张 |
| `selected_realistic_positive_firstframes_batch2/` | 8 | 第二批写实正例，PuLID/IP 各 4 张，与前两批不重复 |

每批均有 `README.md`、`selection.json`、`render_examples.py` 和完整六面板 PNG（1800×1080）。
`selection.json` 保留正式结果行、生成图路径、六个原始面板文件路径；继续找例子前读取三批索引，
按 `(route, run, shot_key)` 去重。同一 run/shot 的不同路线是不同配对，允许并列。
20 张的分数和可点击图索引已写入 [20 episode 评估说明](../entitybench_20ep_evaluation.md)。

注意原实验目录里的旧 `comparison.jpg`：本轮查看的 PuLID/IP 旧版第四格是中期 `pred_x0`，
不是最终 mask，且缺少指标文字。新精选 PNG 补齐了这些内容，旧图未覆盖。
不能因文件名叫 `comparison.jpg` 就声称满足六面板评估契约，也不能声称全部 264 组旧图已修好。

### 13.3 动漫风格来源与正例偏差

已经核对原始 EntityBench 镜头提示词及实际生成配置：

- PuLID run53/3:1 明确写 `In an animated medium shot ...`，官方要求动画风格；
- PuLID run53/3:3 只描述 Kaito 在广场张望、产生灵感，没有明确动画要求；
- IP run146/28:1（Nia 起床）和 run44/11:1（Daniel 在办公室走廊）都没有动画要求；
  这些图的风格来自基础模型输出，具体成因尚未做受控实验，不可声称由上一镜头“记忆”导致；
- PuLID 使用官方镜头提示词；IP 的 `_sdxl_prompt` 为 CLIP-77 调整动作/实体描述顺序，
  保留原文，没有统一加动漫词，也没有强制所有镜头写实；
- 身份参考与 3D 素材是写实资产。注入把动画脸拉向写实参考可能提高身份指标，同时破坏画风；
  写实正例也有皮肤平滑、年龄感、表情和肤色变化，不能把正增量直接等同于全面视觉改善；
- 首批按大增益选例，6 张中 4 张偏动画，不代表整个 132-shot 队列的风格比例。
  两批 14 张写实正例是肉眼筛选的展示样本，尚未做全量风格标注/写实子集总体均值。

第二批查看后因风格化而未纳入的 IP run1775/7:1、run444/4:1 图保留在
`selected_positive_firstframes/realistic_review/excluded/`，原因见其 `README.md`；
正式 benchmark 行仍保留，未据展示筛选删除或重算。

### 13.4 接下来如何接着做

1. 用户再要例子：从正式首帧 JSON 筛 `plugin_applied=true`，同时读两项 C/T/Δ，
   先查看实际 Control/Treatment 风格、检测与可见变化，再按上述格式排版；不重跑生成。
2. 需要写实子集结论时：先按可复核规则标注全部镜头风格，再报告人数、有效对、正负例、
   四项绝对值和差值；不能将这 14 张挑选正例的均值当作正式写实子集结果。
3. 用户若要求提交本次更新，使用 `$movie-github-ssh-push`；仅提交本次评测文档/相关说明，
   保留 `docs/frontend_agent_architecture.md` 等用户改动，不提交 `outputs/`。

## 14. 新增 20 episode 跨镜头人脸外观质心评测

用户要求补充“外观质心相似度”。已在现有 20 episode PuLID-FLUX / IP-Adapter 的
264 组首帧和视频配对上完成；没有重新生成图像/视频，没有改动原有评分、清单或 gate。
Qwen 三 episode 不在这次新增范围内。

- 入口：`pretest/evaluate_cross_shot_face_centroid.py`。
- 回归测试：`tests/test_cross_shot_face_centroid.py`。
- 报告根目录：`outputs/entitybench_firstframe_pulid_ip_20ep/cross_shot_evaluation/`。
- 主报告：`face_centroid_evaluation.json` / `.md`；源文件完整性与独立复算见 `validation.json`。
- `features/` 是源文件内容、模型与 bbox 参数共同确定的可恢复特征缓存；`crops/` 为实际评分图，
  `audit/` 有 88 张跨镜头 C/T 联系表。原有六面板 `comparison.jpg` 未改写，旧格式缺口仍存在。

计算口径：按路线、episode-local 角色分组，Control/Treatment 在相同的配对有效镜头集合上
各自计算归一化 DINOv2 CLS 向量均值质心；每个镜头与该侧质心求 cosine，至少两个不同镜头。
采用论文公式（质心包含该镜头自身），但沿用现有人脸 bbox、0.15 扩展裁剪和空间视频跟踪，
不包含官方 GroundingDINO 全角色定位、CLIP 筛选、VLM fidelity gate；必须标为项目指标。
视频从 5 个等距采样帧按清晰度×人脸面积选代表帧，不按身份 cosine 选帧，也不是逐帧均值。

| 口径 | 路线 | 首帧有效角色-shot / C→T (Δ) | 视频代表帧有效角色-shot / C→T (Δ) |
|---|---|---|---|
| 全部单人，含安全复用 | PuLID-FLUX | 85 / 0.8925→0.8958 (+0.0033) | 91 / 0.8286→0.8338 (+0.0052) |
| 全部单人，含安全复用 | IP-Adapter | 60 / 0.7783→0.8067 (+0.0284) | 78 / 0.7565→0.7647 (+0.0083) |
| 仅实际注入，重算质心 | PuLID-FLUX | 78 / 0.9120→0.9159 (+0.0039) | 78 / 0.8581→0.8636 (+0.0055) |
| 仅实际注入，重算质心 | IP-Adapter | 58 / 0.7836→0.8135 (+0.0299) | 61 / 0.7856→0.7946 (+0.0091) |

每路线 132 个 shot 包含 54 个 episode 内角色，其中 29 个只出现一次；剩余 25 个角色的
103 个 shot 是全量口径的潜在有效集合。缺失和单次出现有明确原因，不填零。
路线/episode 均按可评分角色-shot 等权；仅注入子集独立重算质心。
回退图像 C/T 相同，但因为其他镜头导致两侧质心不同，其质心分数 Δ 可以非零，不能强制置零。

独立复算了 151 个有效角色组（两种口径×两种模态）、核对了 1,122 个输入/模型/原报告 SHA256，
均通过；安全回退媒体哈希一致。已查看首帧/视频正负例及回退裁剪；正分数也可能伴随画风变化、
鼻部异常、皮肤平滑或年龄感改变，不得表述为统一视觉改善。

本轮新增质心评测使用 **CPU**：首帧/视频代表帧的 DINOv2 特征提取及质心汇总均在 CPU 上完成，
参数为 `--device cpu --cpu-threads 1 --batch-size 1`。本轮执行环境无可用 GPU，CPU 配额约
0.5 核、内存 2 GB；这不代表此前图像/视频生成或原四项评测也使用 CPU。
7 项回归测试通过；续跑完整复用 1,056 条 appearance 特征记录，分数与首次完成的报告一致。
汇总命中缓存时不加载模型；复现使用评估说明中的单线程命令。全部产物仍在忽略的 `outputs/` 下。
本轮 Git 提交范围为质心评测脚本、回归测试、本文和 20 episode 评估说明；
用户前端文档及其他已有未提交改动保留在本机，不纳入本轮提交。

## 2026-10-11：20 episode 完整接续评测方案（未启动）

最新方案见 [完整视频评测方案](../entitybench_collage_vs_3d_evaluation_plan.md)，文件沿用原路径。
用户已确定首轮只做一组：硬切复用符合中间步协议的PuLID-FLUX首帧，接续直接使用本路线
上一shot尾帧，不编辑或再次注入；视频使用普通Wan2.2-I2V-A14B，不加载StoryMem LoRA。
范围是冻结20episode完整363shot：191硬切（74单人/117多人），172接续（58单人/113多人/1无人）。
多人硬切沿用现有PuLID-FLUX不注入/基础首帧边界；逐个核验硬切库存，缺失首帧列出待补，
不将已有132个单人评测当作完整首帧覆盖。接续不使用其旧独立首帧或历史5B尾帧。
官方EntityBench三pillar为新视频主表，项目IF/参考DINO/人脸质心为辅助；新评测配置须冻结。
单组只报告系统表现，不计算插件视频增量；独立镜头消融和拼贴/EntityMem对比后置。
旧5episode/41shot/三组方案已被替换。新输出独立保存，尾帧变更须使后续接续依赖链失效。
论文发布/模型核查保留在方案末节；作者全量汇总不能代入本20episode子集。
本轮仅更新文档，未改代码、下载模型或运行生成/评测；已有20episode分数不变。

### 2026-10-11：用户扩容后已启动 A14B 模型下载

用户随后明确授权下载。已确认 `/root/autodl-tmp` 总容量300G、启动前可用152G。
下载官方 `Wan-AI/Wan2.2-I2V-A14B`，固定revision
`206a9ee1b7bfaaf8f7e4d81335650533490646a3`；33个文件共126,205,359,703字节（117.54GiB）。
目标：`models/video/Wan2.2-I2V-A14B/`，直接下载、不另存一份Hub权重缓存。
后台tmux会话：`wan22_i2v_a14b_download`；断点续传、失败重试，LFS权重使用官方SHA256校验。
下载状态、固定版本清单及日志位于 `outputs/model_downloads/wan22_i2v_a14b/`
（`status.json`、`manifest.json`、`download.log`）；当前已开始传输，**尚未完成**。
Hub客户端HTTPS连接失败后已改用aria2；后续以实时 `status.json` 为准，
只有 `state=complete` 才表示全部文件大小及LFS校验通过。没有运行模型、生成视频或重新评测。

### 2026-10-11：旧40步接续实验启动记录（已由第16节替代）

A14B已于北京时间16:06完成33文件下载与LFS校验。用户明确要求开始实验、VLM先不测。
新后台入口 `pretest/run_entitybench_a14b_continuation_20ep.py`，tmux `entitybench_a14b20`；
输出独立保存在 `outputs/entitybench_pulid_i2v_a14b_continuation_20ep/`，以其 `status.json` 为实时状态。
先核验并复用80个有效硬切首帧（56单人注入、18安全回退、6多人基础）；
缺失111个多人硬切使用现有零身份权重FLUX基础生成器补齐，不做多人注入、不改旧产物。
run719的4:1→4:2两镜头接入测试已完成：81帧/16fps/6步，峰值CUDA allocated约41.1GiB；
第二镜头输入与上一视频真实解码尾帧SHA256一致。测试隔离在smoke，不能进入正式报告。
正式固定480P面积（max_area=832×480，按输入比例，样例848×464）、81帧、16fps、40步、
UniPC、shift3、guidance=(3.5,3.5)、BF16与CPU offload。随后单组生成全部363shot/20episode。
8项依赖续跑测试通过，模型加载与GPU推理已实测；同一GPU上各阶段顺序运行。
非VLM评测沿用本地审计过的skip_vlm适配，官方原始入口/适配/patch都冻结在runtime；
VLM与保真度gate未运行，跨镜头分数暂定，不能直接当成论文完整官方分数；API缺失项不填零。
项目身份辅助仍限132单人shot，全部多人/无人shot保留在官方非VLM范围并明确辅助缺失。
本轮尚在运行，不能报告363shot生成或评测已完成；旧结果不变。未自动提交或推送新代码。

### 2026-10-11：加速小样本历史记录（当前已切4步全量）

用户授权尝试LightX2V；见 [A14B加速试验](../a14b_acceleration_trial.md)。
独立环境和两份4step_1022蒸馏LoRA已准备，官方SHA256通过；
固定run719的2:1及4:1→4:2，与已有正式视频比较生成耗时及身份诊断。
小样本已完成：同一近景原版40步1238.2秒，LightX2V普通40步993.1秒（1.25×），
蒸馏4步79.9秒（15.49×）；4:1、4:2各约93.6秒（约13×），均81帧/16fps/848×464。
近景IF均值0.5817→0.6522、参考DINO0.5703→0.6637，CPU诊断已完成；
远景IF0.0642→0.0575、DINO0.2973→0.3558，可检测帧范围不同；多人辅助缺失不填零。
已查看完整对照表：4步近景转头较少、接续抽样出现第三个人，不能宣称质量等价。
正式40步队列已恢复并校验6个已保存镜头，续跑run719 4:4；原manifest、视频/尾帧哈希不变。
候选仅在`outputs/entitybench_a14b_acceleration_trial/`，未替换正式363shot配置或旧指标。
详细耗时、加载适配、失败尝试与质量限制见加速试验文档、`report.md`及`quality/report.json`。


## 16. 2026-10-11：用户选定4步全量与官方非VLM评测

当前任务覆盖原20episode全部363shot（单人132、多人230、无人1），单组系统路线。
191个硬切中56个实际中间步注入、18个安全回退、117个多人基础首帧；多人不新增注入。
旧40步准备阶段补齐的111个多人基础首帧作为图像资产继续复用，旧视频不复用。
4步条件是原A14B加high/low rank64 `4step_1022` LoRA，DMD2、shift5、CFG关闭，
81帧/16fps、480p面积、BF16/torch_sdpa；完整版本、SHA和环境见执行说明。

使用evaluate-face-injection skill审计来源、缺失、覆盖率和续跑，但用户明确要求本轮官方指标，
故不执行skill旧配对IF/参考DINO流程。官方原始入口、skip_vlm适配和补丁同时冻结，
检测/crop/代表帧/特征/质心/VBench评分逻辑保持官方版本；missing API项不填零。
新生成和评测都使用GPU，GPU阶段顺序独占；日志、全镜头审计和逐episode覆盖须实际检查，
不能因tmux启动或脚本退出就称整批完成。当前后台生成中，正式评分在363shot完整后自动开始。
续跑必须核验模型/LoRA/实现/脚本/seed/输入/父视频/输出SHA；旧40步不得自动恢复。
官方缓存另绑定363视频SHA，视频改变时拒绝复用旧分数，应使用新评测目录。

正式首段run719 1:1已完成，81帧/16fps/848×464，耗时84.5秒；后续队列继续。
9项新协议测试及8项旧尾帧续跑测试通过，编译与diff检查通过。不同窗口的重复run
以完整episode_id命名浏览成片，避免覆盖；正式评测一直使用完整episode_id。

## 17. 2026-10-12：官方评测skill v2与后续并发偏好

用户要求后续评测并发。新skill已安装在`/root/.codex/skills/evaluate-face-injection-v2/`，
含官方口径说明、单GPU多进程入口、分片续跑/汇总及验证脚本。原始评分函数不改；
当前默认官方非VLM，缺失gate时cs_face/cs_object仍标为暂定。两路分片按182/181个shot
分配完整episode；缓存汇总与原官方all/easy/medium/hard报告完全一致，9项回归检查通过。
实际两路神经网络推理提速尚未实测；下一次获授权的评测先做代表性小样本测吞吐/显存/
主存与逐episode分数一致性，然后并发全量。新入口不生成视频、不自动启用收费VLM或推送GitHub。

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
