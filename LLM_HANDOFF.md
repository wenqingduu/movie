# 主交接：原始完整benchmark与当前多人3D集成

更新：2026-11-05 23:03（北京时间）。仓库：`/root/autodl-tmp/movie`。

本主交接只保留两条主线：已完成的原始完整benchmark，以及当前用于后续重跑的多人3D集成方案。历史探索过程不纳入本文；原始媒体、指标、素材、Gaussian和历史记录均保留。

## 当前状态

- **2026-11-05 23:03北京时间运行核验：33个P0/P1/P2候选已于20:19全部生成完成，errors=[]，未执行P3/P4其余77个，也未插入全benchmark评分。** 已自动交接`tmux qwen140_v2_remaining_20261105`；新增run4/i130_j148 50/50、run2067/i5_j40 10/10并完成官方12项非VLM，当前run238/i0_j34 8/50、8:1/main，GPU正常运行。新方案完整集由51增至53；89集后续队列已有2集新完成，其余状态以逐集complete/progress为准。run53/i65_j78 4:3仍为既有API过渡拒绝，错误已留档并继续其他集，没有改词或重提POST。后续范围仍只用v2，无守护Agent。评测skill已补充verified_cache_acceleration：兼容整集复用、视频SHA核验的6项VBench复用、新视频原算法评分、完整受影响集中心/边界重算、测量通过再用双路；原通用并行工具尚未自动注入VBench缓存，不可直接运行旧PID绑定切换脚本。仓库及本机skill副本一致并验证通过。

- **最新用户纠正范围，覆盖下方“继续全部动态候选”的安排：只完成33个P0/P1/P2，然后直接新episode。** 33＝1个已有帧＋13个明确位移/运镜＋19个具体操作，接手时8完成、25待完成；P3/P4另77个取消后续执行。当前`tmux qwen140_v2_scope33_20261105`、入口`pretest/qwen140_prompt_v2_scope33_20261105.py`，原批根status改为candidate_completed/expected=8/33。旧215-case父491387已在子worker退出/僵尸边界终止，未杀有效GPU worker；原priority入口的hold guard重新禁止广范围启动，**不要重启旧priority入口**。新33专用入口有单独用户授权，v2 prompt/模型/门控不变。完成33后不插入全benchmark评分，直接交共享锁给已恢复等待的episode父510205/tmux`qwen140_v2_remaining_20261105`；每shot原动态评分和每episode官方12项照常。历史115/218是旧累计口径，不能再作为当前剩余目标。冻结范围`outputs/entitybench_prompt_v2_badshots_20261104/scope33_20261105.json`，说明`README_scope33_20261105.md`；已成媒体全部保留。

- **2026-11-05最新确认：保持已验证v2，不加新prompt修改；动态候选shot和未完成episode都按v2继续。** 当前shot队列`tmux qwen140_v2_resume_20261105`继续；后续普通调度`tmux qwen140_v2_remaining_20261105`已启动，等前者共享锁释放（包括它的评分）后接手。新入口`pretest/qwen140_v2_remaining_episodes_20261105.py run`，冻结计划与状态根`outputs/entitybench_v2_remaining_episodes_20261105/`。本次逐项核验126集清单：37已完成＋原先14＝51新方案完整集；剩89集、1445shot，其中34集/720shot属于原55集低DINO队列的未完成部分，其余55集/725shot随后。**此前hold、只shot不新整集的历史限制已被用户最新明确授权覆盖；不恢复旧整集调度源码。** 保留4集58个完成shot、5张在途关键帧（它们按原缓存输入完成，后续新规划才用v2）；新计划统一v2 installer，无v3。24源/输入SHA、199既有结果及在途输入文件SHA冻结；已验证run53和未开始run44正确窗口/v2编译器，API调用0。每集后官方12项strict/skip_vlm，默认reviewedDINO仍需定位复核。无守护Agent、quota=None/reserve0、不自动删除。已知run53/i65_j78/4:3旧API过渡FAILED/IPInfringementSuspect保留，不改词规避或自动重提POST，未解决则记未完成继续其他集。详见[后续队列](outputs/entitybench_v2_remaining_episodes_20261105/README.md)。

- **2026-11-05最新用户已授权后续动态候选shot与未完成episode沿用已验证v2。** 旧hold已存档并撤销，优先队列在tmux`qwen140_v2_resume_20261105`续跑，当前run1/i50/16:1 main，保留已成帧；`priority_resume_20261105.log`与批根status为准。未改成v3。用户最新问题是原因与最小思路：抽查8个仍0＋2通过，原RAFT计分全部复现，run323起身8/10、run474走路7/10、run1手部上摇9/10均因二值门限未过；run1537首帧已在车外、run617滑行首帧直立，Control注入前就错。建议只有具体物理起点＋去掉无依据静止约束，未实际改prompt。详见`outputs/entitybench_v2_completed_snapshot_20261105/motion_failure_audit_20261105/README.md`。未完成episode须按旧55集cohort和实际有效前缀核验后使用v2，不能把89个保留原始的episode直接等同失败或误算只余4集；旧progress计数可能过期。

- **2026-11-05 15:21北京时间：这次108替换shot的新完整140／2491评估全部可用项完成。** 官方12项已完成，默认可见单脸＋定位纠错DINO完成80个分数隐藏样本复核：77保留、3排除（N001 Jae-hyun误框seated Min-jun、N022 So-hee误框lying Yuna、N039 Marcus误框Daniel）。N020变装后Ethan关联不清、N047重复Leo、N064重复Anya均保留，不因身份漂移/重复生成删低分。默认DINO .7943→.7955（原始.7863），前后均131有效集；完整表`outputs/entitybench_v2_completed_snapshot_20261105/complete_report.md`，`full140_all_metrics_comparison.json`、`target_shot_comparison_with_reviewed_dino.csv`、`final_validation_20261105.json`。当前后台评估已结束；新110shot生成仍用户hold，不得自行恢复。VLM调用0，25份保留评分SHA未改。

- **2026-11-05 14:37北京时间：本次108替换shot的全140官方12项指标已完成。** 两路剩16集556shot耗时1889.33秒，41受影响集834shot全部评分与输入核验通过、108目标逐shot全指标表已保存；99未变集SHA核验复用，完整140/2491已聚合，缺失替换0，VLM调用0。文件`outputs/entitybench_v2_completed_snapshot_20261105/full140_official_comparison.json`、`report.md`及`target_shot_comparison.csv`。替换前51集快照→新：动态.4634→.4750、美学.5854→.5851、成像60.8480→60.7765、角色出现.9458→.9449、场景.6246→.6263、raw cs_face.7230→.7231、边界.8679→.8621。默认reviewedDINO仍在可见单脸CPU检测，之后需冻结新定位人工审阅；不能把raw值当默认DINO。25份保留评分SHA均未改。新生成110shot继续用户hold，不能因完成评分自动重启。

- **2026-11-05 14:03北京时间提速验证通过并已执行：** 保留25集旧完整分数，剩16集556shot两路GPU评分（worker206854/206855，主192668，tmux`qwen140_snapshot_accelerated`），分片280/276shot。剩余504个未变视频复用6项VBench，52新视频原方法推理；定位/中心/边界仍全算。20shot验证缓存版串行181.79s、双路120.84s（1.504×），12指标最大误差0、119canonical SHA全同、计数一致、GPU峰值6903MiB；详见`outputs/entitybench_v2_completed_snapshot_20261105/acceleration/README.md`。新增结果尚在worker目录，主affected/results在全部worker结束后统一合并，**不可只据主目录仍25集判断停跑**。新增镜头生成hold保持，旧串行113871/113886/113895已在持久化集边界终止；原源代码/媒体无改。一次性提速入口绑定旧PID，不可重复启动，应按冻结remaining分片与缓存版本化恢复。

- **2026-11-05用户授权评估提速，新增生成仍保持hold。** 新调度`pretest/qwen140_snapshot_accelerated_20261105.py`正在tmux `qwen140_snapshot_accelerated`运行，`/root/autodl-tmp/movie/outputs/entitybench_v2_completed_snapshot_20261105/acceleration/status.json`为实时状态。它等待当前集持久化后替换旧评估进程；已完成评分及历史源码不改。新VBench缓存适配器只复用视频SHA与配置/模型/库一致的726个未变镜头，108新视频仍走原6维推理；定位、跨镜头中心/边界仍完整计算。先对run560/i85/2:1所在整集与run654整集做串行缓存vs原结果、两路vs串行验证（12指标容差1e-5，计数与canonical crop SHA一致），再按实测吞吐选择剩余任务并行数。验证尚未完成前不能宣称已提速或一致；结果保存在`acceleration/validation.json`。

- 9集阶段表角色出现下降已逐shot排查：仅run599/3:2 Marcus（CLIP .217957→.181701）与run560/i85_j105/2:1 Anya（.226805→.199801）两项present→weak，阈值.20。前者人物仍可见；后者新Anya canonical实际错框红头巾男Javier，女性在光效/头盔变化中存在。合计9集均值降1.41975pp，分别贡献1.11111/.30864pp，其他角色状态无变。官方分数不改，默认DINO定位审核分开。见`outputs/entitybench_v2_completed_snapshot_20261105/presence_drop_diagnosis_20261105.md`与evidence图片。

- 新版早期配对子集已保存：`outputs/entitybench_v2_completed_snapshot_20261105/interim_ready_episode_comparison_20261105.{json,md}`，冻结已完成9／41集、95shot，同一子集集间等权；不是全140最终结果。动态0.4874→0.5189、美学0.6054→0.6041、成像66.8232→66.2915、角色出现0.9901→0.9759、场景0.5074→0.5465、raw cs_face0.7658→0.7640、边界0.7187→0.6849；默认reviewedDINO尚待新定位复核。不得由这9集宣布全benchmark其他指标无下降。

- **2026-11-05 13:24北京时间最新用户指令覆盖此前自动续跑：先完成108替换镜头的新完整benchmark评估，用户看其他指标是否下降后再决定新生成。** 已终止仅等待的priority waiter125280（无子worker），`user_hold_after_snapshot_20261105.json`设generation_authorized_to_resume=false；优先队列入口加hold guard并更新独立contract。评估113871/113886/113895继续，不影响已生成视频。**不得自动恢复剩余110个镜头**；排序计划保留待用户决定。

- **2026-11-05 13:18北京时间：用户要求先给全benchmark，再分析22个动态0→1并优先重跑可能成功的剩余shot。** 已冻结105新增＋3复用＝108shot、41受影响集的新完整140／2491快照，正在tmux `qwen140_v2_snapshot_eval`评分；根`outputs/entitybench_v2_completed_snapshot_20261105/`。已有逐shot分数＋输入SHA可精确更新动态0.46335244→0.47504346（24提升含2复用），其他新指标不能套用旧值；默认DINO待新定位审阅。已核验旧完整表仍是51整集替换版，默认DINO0.7943。详见[本次快照、分析与队列](docs/entitybench_v2_snapshot_and_priority_20261105.md)。
- **优先顺序已接入独立调度包装，不改冻结生成实现。** `pretest/qwen140_prompt_v2_priority_20261105.py run`，原215case完全相同、冻结`cases.json`不动。剩余110按P0既有首帧1、P1明确位移/运镜13、P2具体操作19、P3其他18、P4小表情59；全部保留。22成功的采样图全部查看，不能等同动作正确：部分来自移镜、额外人物或B过渡。18个原文整体移动案例7通过，其他87个15通过；原文明确运镜仅3个2通过，样本很小。
- **运行交接：** 旧新增shot调度825932已在全部子worker退出的边界终止（保留run1／16:1已生成帧）；tmux `qwen140_v2_priority`中的普通等待脚本等当前评估supervisor113871退出/GPU空闲后启动优先队列。看`priority_wait_status_20261105.json`及`status.json`；不要恢复825932或旧整集队列。10个历史代码指纹异常已有active policy补跑配置，参数和素材核验不变，仍属未完成项；错误记录保留。无守护Agent、无新增整集生成、无磁盘人为配额或自动删除。

- **2026-11-04 23:10北京时间：用户明确收敛为“只重生成异常shot，不追加生成完整episode或其下游shot；用局部替换后的完整序列评测全benchmark”。** 215新增＋3复用的生成范围不变，完整46/140episode仅做评分，真实邻接/cut照常计边界，不以本批声称完整新链。原整集队列父446545/756528已在子GPU worker结束后终止，媒体/前缀保留，**不得自动恢复旧整集生成**（legacy_episode_queue_stopped.json、selective_replacement_design.json为准，覆盖此前自动恢复说明）。当前新批PID825932仍正常运行extended入口，2:5/main；完成后检查到旧父进程已退出会跳过恢复。将来恢复当前批次使用 `pretest/qwen140_prompt_v2_selective_only_20261104.py run`（新入口已source/status预检；selective_only_resume.json绑定SHA），可跳过旧父进程必须存在的启动约束；切勿直接用旧extended入口重启。磁盘quota=None/reserve0和自动后续全部可用评分安排不变。

- **2026-11-04 23:04北京时间：追加用户授权已落实：生成后比较目标shot全部可用指标，再版本化替换并完整评测全140；取消磁盘人工上限/8GiB预留。** 当前入口已换`pretest/qwen140_prompt_v2_badshot_batch_extended_20261104.py run`，同tmux `qwen140_prompt_v2_badshots`，PID825932。原生成-only源码/plan均保留，当前授权/新源码SHA在`outputs/entitybench_prompt_v2_badshots_20261104/full_evaluation/extension_plan.json`；`disk_policy.json`明确quota=None/reserve=0、不自动删历史结果。旧plan.json中8GiB为历史条件，已被扩展覆盖；用户明天扩容。已等子worker完成再退出旧调度器822292，首个run406/2:3关键帧/视频/动态评分全部保留（coordinator_handoff.json），新调度正常跑run406/2:5/frame，新完成1/215＋复用3。
  自动后续：`pretest/qwen140_badshot_full_evaluation_20261104.py run`，生成后评测受影响46集全shot；导出218目标shot全部可用指标CSV/JSON及关联边界对；发布新replacement_manifest（成功新视频含降分均替换，缺失留旧并单列），然后全140/2491官方strict/skip_vlm评测，46集本轮结果经输入核验复用、其余94集新推理。比较原始完整benchmark／替换前51集快照／v2替换后。当前queued_after_generation，不可误报评分完成。跨镜头centroid和边界用完整episode及替换后真实邻接；独立生成父链与新剪辑邻接差异单列，不宣称重建完整接续链。
  默认DINO后续`pretest/qwen140_badshot_reviewed_dino_20261104.py auto`：全140可见单脸FP32 RetinaFace＋固定canonical region-DINO，历史定位排除仅按crop SHA沿用，选新变化目标crop最低60＋随机20隐藏分数复查，不足80看全部。**自动提取/复查图已入队，但后台不会完成模型式假复查**；必须在真实完成定位判断并保存manifest绑定的review/decisions.json后运行`...py final`，才生成默认DINO及13项完整表。复查前官方12项表可用，默认行明确pending，VLM依赖项null，不能填raw或auto。新post纯预检复现run406九项shot指标的episode均值、既有默认DINO；4991基线gate行核验通过，无新评测神经推理。自动后续阶段及日志：full_evaluation/status.json、followup.log；最终complete_report.md、full140_all_metrics_comparison.json、target_shot_comparison_with_reviewed_dino.csv。原24–32小时仅为生成估计，后续全量评分和人工定位复查另计。

- **2026-11-04 22:39北京时间：用户已授权把此前问题shot切换v2，后台新批次已启动。** `tmux qwen140_prompt_v2_badshots`，PID822292；入口`pretest/qwen140_prompt_v2_badshot_batch_20261104.py run`，实时根`outputs/entitybench_prompt_v2_badshots_20261104/`。218候选＝194动态1→0＋24其他动态0且实际视频prompt带图号；复用刚跑好的3例（包括未改善的run1），新增215，原路由C192/B17/A6。首20为C14/B4/A2，优先run406/i5_j41 2:3、2:5，当前2:3/frame已实际启动。每shot自动原VBench动态前后评分和视频页index.html；独立原父尾帧对照，不是完整新episode链，不替换正式benchmark，不新增整集DINO/出现率结论。2D/3D资产/风格/模型/种子/采样/v7门控不变，原19源SHA及215份window/policy核验通过。原整集queue PID446545、run4父756528在worker结束边界暂停；此批结束finally恢复（旧源仍v1），canonical active_queue指向本批并链接暂停的原队列。预计首20约2–3小时（11月5日00:40–01:40），全批24–32小时（11月5日22:40–11月6日06:40），API排队/额度/异常可延后；ETA JSON为估算，不是deadline。初次NVML释放延迟导致调度预检中止、无新shot生成；只修正CUDA释放等待并版本归档后重启，无旧结果覆盖。异常记录继续其他shot，鉴权/额度/SIGKILL停批次；无守护Agent、无语义重试，磁盘保留8GiB。README/plan/cases/status/errors/batch.log为证据。

- **2026-11-04 22:22北京时间：用户授权的最小提示词v2独立试跑已完成3shot。** 新文档`docs/entitybench_qwen140_route_agent_prompt_v2_20261104.md`，新适配器`pretest/qwen140_prompt_v2_20261104.py`，入口`pretest/qwen140_prompt_trial_20261104.py`；限定图号仅用于frame_prompt、B主段不再拼过渡前tail_identity、短约束保留动作起点/幅度。v1及19份冻结生成源不变，原队列仍用v1；未替换正式benchmark。`outputs/entitybench_prompt_v2_trial_20261104/review/index.html`为前后视频/抽帧，README/report.json为结果。对比的是此前集成重跑，非原full140：run719/i35_j44 3:1(C)动态0→1，run1/i50_j84 2:1(C)0→0，run227/i10_j20 10:2(B，含过渡合片)0→1；原VBench/RAFT阈值未改，三张新关键帧均实际3D注入，种子/模型/采样/风格/视频输入链核验通过。单次规划/生成，无语义重试；首帧动作起点仍有未改善例，B目标图仍继承惊讶表情，Silas动作落点也不完全准确，不声称整体benchmark提升。生成及评测在GPU任务边界串行维护，最终`final_metric_queue_resumed.json`确认22:20恢复原queue PID446545及run4父进程756528，无守护Agent。

- **2026-11-04 21:07北京时间：按请求冻结51完整重跑集／988shot的全140快照已完成。** `outputs/entitybench_full140_completed51_snapshot_20261104/report.md`与140行CSV/JSON、输入/后验验证齐全。51重跑＋89原版＝140集2491shot，冻结20:54，后续新完成集留下一快照；含新版run594。默认可见单脸＋定位纠错DINO0.7863→0.7943（131有效集、1380→1469计分，上次41集快照0.7920）；raw cs_face0.7155→0.7230，角色出现0.9212→0.9458，成像59.6517→60.8480，美学0.5817→0.5854；动态0.4853→0.4634、场景0.6303→0.6246、边界0.9822→0.8679下降。新10集318shot CPU RetinaFace589次约6分钟；80项隐藏分数复查排除6项(N013/N038/N051/N053/N060/N079)，N079为Silas背部区域混入Nadia可见脸，明确记录；真实漂移/重复和未决保留。此前41集3轮复查完全沿用。原all/easy/medium/hard完全复现，4607媒体来源/54评分manifest，988shot裁剪及_eval/所有评分/review SHA、89未替换集全部13项delta0及独立均值验证通过。19冻结生成源码不变，队列未打断；无视频/付费API/VLM/新DINO特征推理。新入口`pretest/evaluate_full140_completed51_snapshot_20261104.py`、`pretest/evaluate_full140_new10_dino_20261104.py`；详见`docs/entitybench_full140_completed51_snapshot_20261104.md`。实时队列21/55集完整、修复12/16，当前00000865-b77b-32f3-b606-c2451a77a12a__run4__i130_j148__T240 27/50shot；run53/4:3 API拒绝的旧异常保留，其余无新增。本轮评测代码/文档尚未提交，前次push 0d72a00已远端核验。

- **2026-11-04 17:35北京时间推送前状态：** 后台`tmux qwen140_episode_repairs`仍在运行，入口`pretest/qwen140_repair_priority_queue_v2_20261103.py run`；实时以`outputs/entitybench_episode_repairs_20261103/status.json`、`progress.json`及各集`progress.json`为准。本轮55集完整19集、原16中断集补完10集，加此前30集累计49集完成生成和官方非VLM；当前`00002522-ca24-3c7f-acd6-982a08b3e038__run34__i0_j23__T300`，13/15shot完成、6:6/frame。run53/i65_j78在4:3过渡API返回`IPInfringementSuspect`，保留前8shot并跳过继续；未自动改提示词或更换模型，不把该集计入完成。技术修复后未再出现T5加载SIGKILL。已按用户最终指令仅删本地`models/llm/Qwen2.5-14B-Instruct-AWQ`，释放9.31GiB；SDXL/IP-Adapter仍在。当前空闲约20.6GiB，保留8GiB，容量不足等待。所有19份冻结生成源码SHA核验通过；此次不改运行实验或推送outputs/模型/密钥。

- **11月3日20:10北京时间：按请求时冻结41个完整重跑episode／670shot，完整140集混合快照已完成。** 输出`outputs/entitybench_full140_completed41_snapshot_20261103/report.md`、全140逐集CSV、JSON、all/easy/medium/hard官方聚合与SHA核验；41重跑＋99原版＝140集2491shot，冻结17:20后新完成集留下一快照。默认可见单脸＋定位纠错DINO0.7863→0.7920（131有效集，1380→1417次）；raw cs_face0.7155→0.7214，角色出现0.9212→0.9437，成像59.6517→60.4016，美学0.5817→0.5856；动态0.4853→0.4724、场景0.6303→0.6219、边界0.9822→0.8935下降。此前35复查保留，新6集150shot CPU RetinaFace177.8秒、80项隐藏分数复查确认6错角色排除（N012/N025/N026/N049/N057/N072），歧义/重复/身份漂移保留。原131集基线all/easy/medium/hard完全复现；3971份媒体SHA、44评分manifest，所有670shot裁剪/_eval/评分及review SHA后验核验；未替换99集全部13项delta0。无GPU生成干扰、付费API/VLM/DINO特征新推理，strict/skip_vlm partial_without_vlm。入口`pretest/evaluate_full140_completed_snapshot_20261103.py`与`pretest/evaluate_full140_new6_dino_20261103.py`，详见`docs/entitybench_full140_completed41_snapshot_20261103.md`。技术修复队列继续；20:12实时为原55队列12完整/修复3完整，run406进行中，不将其状态误写成41快照范围。

- **11月3日13:00北京时间起，当前后台已切换为“技术修复后优先补跑16个中断集”。** `tmux qwen140_episode_repairs`，入口`pretest/qwen140_repair_priority_queue_v2_20261103.py run`，实时状态/日志/冻结计划在`outputs/entitybench_episode_repairs_20261103/`（最终计划`queue_v2.json`，文档README）。canonical pointer为remaining126/active_queue.json；旧csface队列status已标记superseded，原父进程在worker结束后退出，不再运行。先run1172→其余15个失败集→保留run237当前前缀→同一原55集未完成部分；不是另开重叠GPU队列，也没有守护修复Agent。启动时原队列9集300shot完整、319shot生成，run237维护时又保留1:1视频/result和1:2关键帧；累计完整39集643shot。16个中断为13次SIGKILL（最后加载日志全在T5）、2次人物查询ID错误、1次TLS任务GET错误。独立适配器`qwen140_technical_resume_v2_20261103.py`：仅T5 torch.load mmap=True（242张量/11.36GB逐张量bit-exact，proof JSON）；真实run1172原失败1:1已生成81帧并记为完成，13:04正在2:1主视频。工具原schema保持供应商兼容，用工具说明/输入限定有效人物ID；run323/4:2真实重规划查[]且画Julian，run4/i80_j98/2:7查[]且匿名人物frame_characters=[]仍描述完整行动。run53/3:2已从原task_id取回过渡、create SHA不变，无新POST；GET仅有限网络重试，POST不重试。v1扩展JSON schema返回400，已归档失败缓存及v1源码/计划，最终v2实际接口通过。5项回归检查通过，旧19个冻结源码、模型/BF16/LoRA/种子/采样器/v7门控/身份资产均不变，960份既有结果/视频/尾帧SHA每集核验。每集完成后官方12项非VLM；新默认DINO另需定位复核，不能自动当已完成。空间约14GiB，仍保留8GiB，缺容量等待，不删历史。原cgroup OOM计数为0，SIGKILL内核根因不能断言；降低加载峰值已通过真实视频验证。新补跑如再次SIGKILL或全局额度/认证错误会停止留档，不对系统性错误盲目遍历。下方11月1日原队列“当前运行”记录已是历史，实时以此条和active_queue为准。

- **11月2日21:02北京时间：累计已完成35集／520shot的同集前后统计已完成。** 输出`outputs/entitybench_completed35_comparison_20261102/report.md`、35行CSV及新增5行CSV；固定此前30集343shot（含新版run594）＋本轮run146、run719、run4/i35_j83、run184、run221/i20_j38五集177shot，排除run264和中断前缀，不是全140集。官方12项重核输入SHA、父链、配置后用官方函数集间等权聚合，旧30集汇总精确复现。新增5集CPU RetinaFace177shot／292次前向，约193秒；原人物区域/已存DINO特征不变，80项分数隐藏定位复查，排除N015(Seo-joon→Dong-hyun)、N017(Nia→Casey)、N046(Daichi→Ryota)、N049(Min-jun→Dong-hyun)，6项未决/重复生成保留。默认DINO全35有效：0.7823→0.7879；新5为0.6627→0.7210；覆盖1030调度role-shot，保留384→422、计分出现318→358。全35官方raw cs_face0.7157→0.7341，角色出现0.9097→0.9890，成像62.5434→64.8233，美学0.5833→0.5955；动态0.4911→0.4510、场景0.6206→0.6066、物体一致性0.8245→0.8027、边界0.9846→0.6800下降。13项全部可用、VLM0；无新视频/官方评分/DINO特征推理或付费API；仅新增CPU检测，未动生成队列/冻结源码。入口`pretest/aggregate_completed35_snapshot_20261102.py`与`pretest/evaluate_completed35_new5_dino_20261102.py`；默认DINO不再pending，不能拿全140原0.7863作35集基线。

- **当前运行：11月1日23:41北京时间已启动原人物DINO低分优先后台队列。** `tmux qwen140_csface_priority`，入口`pretest/qwen140_csface_priority_queue.py run`，队列状态/日志在`outputs/entitybench_csface_priority_20261101/`。原可见单脸＋定位纠错DINO均值0.7863（131有效集），排除30集已完成结果后，55集1413shot按原分数升序；从run323/i0_j14→run146/i0_j46→run221/i0_j18开始。首集references已在GPU运行，Agent qwen3.8-max连通预检通过。只加调度adapter，不改原19个冻结源码或cohort，模型/提示词/3D门控保持原方案。输出复用remaining126/experiments合法根，原.queue.lock互斥；原status为历史，新active_queue指向本队列。每集自动官方12项非VLM，新默认DINO定位复查明确pending；不跑旧full140滚动聚合。守护修复Agent关闭，门控正常跳过照常继续；技术集异常留档继续，下游全局额度/认证停止。历史估算整批约5.8天、新增51GiB，现可用约36GiB，后续须扩容；保留8GiB及单集估算余量，不足等待，不删历史。详见`docs/entitybench_csface_priority_queue_20261101.md`。

- **最新绘图交付：Leo、Marcus各四个角度3D渲染已导出。** `outputs/run594_leo_marcus_3d_four_views_20261101/README.md`及ZIP（2.14 MiB）；每人4张512×512白底无字PNG，面向画面左上/右上/左下/右下，另有无字2×2 PNG和带标签JPG。来自相同episode人物Gaussian既有pose_calibration/samples真实渲染，非front/left/right建模输入；模型/参考/图片SHA均核验，实际camera与detected姿态记录于manifest。无新推理/渲染/API，入口`pretest/export_run594_leo_marcus_3d_views.py`。

- **最新绘图补充已完成：参考latent编码与逐步加噪图已补齐。** 完整交付`outputs/run594_new_injected_denoising_with_noised_refs_20261101/README.md`及ZIP（106.90 MiB），191张PNG/JPG（原148＋新增43）；3:4 Leo6份、3:5 Leo6份与Marcus1份实际带噪参考，含原R、VAE编码解码E(R)、共享初始噪声、每步全幅与bbox裁剪、概览。新图在每shot的`reference_latents/角色名/`。入口`pretest/export_run594_noised_references.py`，源`outputs/run594_noised_reference_reconstruction_20261101/`保留E(R)/alpha/噪声/sigmas/13份r与精确残差张量。仅VAE编码解码与保存状态重建，12次注入更新下一latent逐元素bit-exact、所有残差norm相同；无新噪声采样、DiT、检测、3D渲染、视频/API或评测。强噪声下绿/紫VAE解码是实际值表现，不美化。manifest图像数191，含4份TXT提示词的记录总数195，勿混为195张图。原图片包、模型生成与评测分数未覆盖。

- **最新评估已完成：仅30个重跑episode／343shot，纳入run594/i75_j98新版3:4、3:5。** 输出`outputs/entitybench_rerun30_latest_evaluation_20261101/report.md`，30行逐集CSV/JSON齐全，`status.json`为complete_non_vlm_and_reviewed_dino。29集官方12项精确输入核验后复用，run594完整10shot重新评分；343shot RetinaFace补完（18缓存，GPU阶段约90秒），80项无分数定位复查，确认5项错角色排除，75项保留（含重复人物、生成身份/风格失败及未决歧义）。默认可见单脸＋定位纠错DINO：原版同30集0.8022→当前0.7991，30/30有效，计分出现202→237；不要拿全140原0.7863当此子集基线。官方raw cs_face0.7286→0.7459，角色出现0.9060→0.9885，成像63.3384→65.2810，美学0.5836→0.5939，动态0.5011→0.4639，边界0.9854→0.6979。交集补充21集77次为0.8064→0.8168，不替代主指标。视频/eval/crop/代码/协议/决策SHA与接续链全部核验，官方失败0、VLM/API调用0；未更新全140或启动首帧IF/参考DINO消融。主文档`docs/entitybench_rerun30_latest_evaluation_20261101.md`。

- **最新：11月1日完成用户要求的新版3:4、3:5注入分支去噪重放。** 图片交付`outputs/run594_new_injected_denoising_images_20261101/README.md`与同名ZIP（92.68 MiB），148张PNG/JPG；每shot21张真实z、20张pred_x0、12张step12–17注入前后图，含Leo/Marcus原参考与3D/mask，无Control图片/轨迹/视频。3:4 Leo6步、3:5 Leo6步+Marcus1步；两张最终图及step12预测均精确复现原图（SHA/像素MAE0），所有实际after与z_(k+1)核验通过。原始张量在`outputs/run594_new_injected_denoising_replay_20261101/runtime/shot_*/trajectory_states.pt`。原v7内部Control仅用于共享分叉，不作为用户交付。3:5必须保留其原目标生成条件尾帧，不能换为新3:4尾帧。实际入口`pretest/export_run594_injected_trajectory_v2.py`；原入口和冻结源码保留，首次仅后处理RGB/RGBA比较异常，3:4直接从现存图片恢复metadata未重跑推理，代码修订已入plan。当前complete，无新API或评测。

- **最新交付：用户选择“先导出现有结果”，不补跑Control视频。** 只含新版的完整导出在`outputs/run594_i75_j98_new_only_control_vs_3d_20261101/README.md`及同名ZIP（239.42 MiB）；3:4 Leo6步，3:5 Leo6步、Marcus1步，关键帧Control／注入配对、全部现存准备图片、新版视频及589张逐帧PNG已导出，原件与派生图SHA、视频端点、ZIP完整性通过。没有旧版媒体，视频仅注入路径；完整20步去噪轨迹未记录。导出脚本`pretest/export_run594_new_only.py`，不新增GPU/API或指标计算。

- **最新：11月1日GPU已恢复，run594/i75_j98的3:4–3:5定向修复已生成并导出。** 实查为48GB RTX4090、20核/72GiB；输出`outputs/run594_i75_j98_shot34_35_prompt_fix_20261101/`，`status.json`为complete，当前无生成worker。本条覆盖下方旧的无GPU环境描述。用户本次明确只继续该任务，没有启动其它episode或评测。3:4保留原图像输入/种子/3D设置、强化单人构图，新图/视频只有一个Leo，实际6步注入。3:5原目标图SHA逐字节复用（Marcus1步、Leo6步），新3:4实际尾帧→一次3秒Wan3过渡→实际尾帧→新81帧主段；旧tail_identity矛盾未拼入新主段。13秒左右新旧并排视频在`review/original_vs_rerun.mp4`；普通素材包`delivery.zip`。用户已明确接受后半段触碰Marcus身体的动作，不再重跑。过渡中段短暂叠影已披露，输入链/原文件哈希验证通过；未更新完整episode或benchmark指标。入口`pretest/qwen140_run594_fix_shot34_35.py`，初始化输入与提示词冻结，勿原地改后覆写续跑。
- **11月1日用户指定后续默认评测口径：可见单脸筛选＋确认错角色定位剔除后的原人物区域DINO。** 已更新并校验仓库及本机安装的`evaluate-face-injection-v2` skill，细则见`skills/evaluate-face-injection-v2/references/visible_single_face_region_dino.md`。后续默认项目人物一致性行用`cs_face_visible_single_face_reviewed`；官方原始`cs_face`保留并独立标注，不混作同一口径。原版三项排除只对原episode/角色/shot和原裁剪SHA生效，新生成图重新复查；先剔除后重算质心与集均值，不按低分删除真实身份/画风失败。此轮只更新规范，未完成或捏造替换14/29/30集后的新版诊断分数。
- **11月1日新增任务：按原0.7863诊断口径重算，并找身份提升绘图素材，正在执行。** 已固定30集343shot完整重跑为输入，原0.7863（131有效集、1380计分实例）已完整数值复现；新版检测/人工定位复查尚未全部完成，不能给出新版诊断最终分数。当前实例实际无GPU、0.5核CPU/2GiB，用户已选择恢复GPU实例；实例切换需用户控制台操作，先实查资源再续跑。完整任务与命令见[11月1日诊断及素材任务](docs/entitybench_visible_region_and_figure_task_20261101.md)。IF、DINO、RetinaFace分阶段串行，缓存保留；不要把10月31日status当作实际进程仍活着的证据，不重复启动重型worker或修复Agent。
- **11月1日素材筛选已结束，全量诊断等待GPU恢复。** 72项优先候选中IF成功71项、23项正向；前48项参考DINO完成、19项双正向，前16组已逐组目视核查。普通图片目录`outputs/rerun_identity_figure_materials_20261101/`含5组有限正向、2组副作用及两组系统级跨镜头对比。最大涨分Silas5:2也削弱了提示词所需微笑，不能当干净正例；未发现可无保留宣称明显身份恢复的新案例。新版全量检测18/343缓存，GPU尚未出现，当前无模型worker在跑，不承诺断开会话后自动恢复。
- **10月31日23:07历史交接，非当前进程状态。** 当时本批126集2316shot中，15集158shot已生成并完成官方非VLM；共生成175shot，含run155的10shot未完成前缀及run654/i20_j41的7/10shot。当时加此前14集175shot，累计29集333shot完整重跑并评分；11月1日固定输入已为30集343shot。旧tmux名`qwen140_remaining126`，入口`pretest/qwen140_multiface_remaining_episode_batch.py`，输出`outputs/entitybench_qwen140_multiface_remaining126_20261030/`。先读[历史完整交接](docs/entitybench_session_handoff_20261031.md)，实时核对后继续，勿把旧status当活进程。
- 磁盘：10月31日约15:09空间不足等待，20:05扩容320→350GiB后恢复；旧进程退出原因未证实。当前约36.80GiB空余，每集前保留8GiB并估算所需空间，不自动删除；全队列可能仍需扩容。守护修复Agent保持关闭，普通队列及镜头路由Agent继续。
- **未解决：** run155/i5_j39在8:1查询`the yellow electric pokémon`不在人物catalog而KeyError，前10shot保留；全140滚动快照helper约195行软链接断言失败，`full140_snapshot/`可能部分写入，不能当最新一致结果；本批15集配对汇总有效。run617/i75_j127的Jordan FaceLift因4通道输入失败，保留2D、该角色未构建可用3D，其他角色及整集继续。详见本次交接，不把资产技术失败说成正常门控。
- 本批15集相同子集：角色出现0.9649→0.9919、人物区域cs_face 0.7618→0.7807、成像65.5121→67.1443、美学0.5968→0.6036、场景0.5914→0.6135；动态0.5165→0.4852、cs_object 0.8606→0.8283、边界0.9868→0.7564下降。完整人工审阅待完成，非全140或插件单独消融。19个冻结源码SHA已核验；修复源文件须明确版本，不能原地改后无记录续跑。
- **最新全量汇总：已按14集175shot重跑＋126集2316shot原版，重新聚合完整140集2491shot。** 独立输出`outputs/entitybench_qwen140_partial_rerun_full140_20261030/`；12项前后表、140行逐集CSV和14集视频入口齐全。全量角色出现率0.9212→0.9364、人物区域cs_face 0.7155→0.7172、成像59.6517→59.9107；动态0.4853→0.4828、场景0.6303→0.6259、边界0.9822→0.9469。详见[全量重新聚合说明](docs/entitybench_qwen140_partial_rerun_full140_20261030.md)。这是部分重跑后的混合全量快照，**不是140集统一新方法重跑**；原始完整benchmark数值和报告仍保持原样。
- 全量核验：原all/easy/medium/hard官方报告精确复现，140集唯一覆盖、2491规范视频映射，评分代码/配置/模型/库一致，17份评分manifest VLM调用0；复用核验通过的逐集评分，本次无神经推理。126集未替换指标差值均0。边界有效137集，缺失不填0。
- **此前已结束批次（非当前GPU状态）：10月30日21:58（北京时间），9集117shot全部生成与12项非VLM评分完成，旧队列退出。** 第二批focus4（run317、run15、run354、run348）4集55shot；extra5（run974、run227、run470、run1537、run893）5集62shot。充值后续跑86shot，保留31shot。状态根`outputs/entitybench_resume_supervisor_20261030/`，详见[完成说明](docs/entitybench_recharge_resume_20261030.md)。
- 九集同一配对子集由冻结官方实现重新解序列化并集等权聚合，输出`outputs/entitybench_resume_supervisor_20261030/final9/`：人物出现率0.8254→0.9899、人物区域cs_face 0.6704→0.7068、成像60.6169→65.0132、美学0.5778→0.5892；动态0.5054→0.4704、场景0.6454→0.5955、边界0.9833→0.6510下降。不是全140或插件单独消融，完整人工视频审阅待完成。
- 完成核验：9集各12项有有效值且n_failed=0、VLM总调用0；117个视频/尾帧SHA及接续父链、官方路径均通过，93项既有结果/视频/尾帧哈希未变。A/B/C为3/16/98，114张新关键帧中112张实际注入、2张正常Control沿用、204角色次实际注入；两次门控正常跳过不算技术失败。
- **用户明确关闭守护修复Agent，保持关闭。** 本轮只用普通队列调度，max_repair_rounds=0、工具轮数0、修复白名单为空；修复调用0、技术异常0。镜头路由Agent与修复Agent分开，前者仍为实验组成部分。
- 夜间曾因月度API额度耗尽于03:30停止，同接口修复调用两次也失败；充值后14:08续跑，全部完成。历史状态/额度异常保留在原目录及before_resume/，不能把最终0异常解释为从未中断。
- Clara原配置4:1、5:1于02:23完成，Control/Treatment最终图均与原图逐像素一致，各82张主轨迹图及注入前后图、原始张量；输出`outputs/model_figure_clara_denoising_replay_20261029/`，独立PNG/JPG目录`outputs/clara_figure_images_20261030/`。

- 原始完整benchmark：140集／2491镜头，生成与官方非VLM评测完成；不可覆盖或重新解释为新方案结果。
- 当前多人3D方案：run700完整10镜头及12项非VLM评测完成；首批低指标4集／48shot于北京时间2026-10-29 21:59完成生成及逐集官方非VLM评测，四集配对聚合也已完成，缺失0；完整人工视频审阅待完成。后台队列已正常结束，不重复启动。
- run484 / 5:2继承Nora白名单冲突已修复并跑完。保留既有Agent计划、人物资产和41个已完成shot；最后7shot使用`inherited_cast_20261029`兼容修复。历史异常已归档，不把最终状态的0异常解读为从未中断。
- 用户最新授权：整张没有角色通过3D门控属于正常跳过，使用同次Control继续，不需确认；技术异常记录并继续处理其他episode，尽量完成整批。不得放宽阈值、隐瞒缺失或自动更换模型。
- 10月29日硬切人物规则修订`hardcut_cast_20261029`：既有22shot保持原结果，剩余26shot使用新提示词；未执行的run1404/5:1旧计划已归档，恢复后重新规划，详见[续跑记录](docs/entitybench_weak4_resume_20261029.md)。
- VLM评测继续关闭：strict、skip_vlm，调用0。Agent规划与B路线视频生成API已获授权，不属于VLM评分。

## 首帧绘图素材skill

已安装`movie-firstframe-figure-assets`，仓库副本`skills/movie-firstframe-figure-assets/`。原Clara精确重放和183张图片包保留；新做320个大脸角色配对的CPU身份筛选，推荐仅小幅正向候选，不能宣称大幅身份恢复。原Clara4:1最终IF 0.6047→0.6044，仅适合流程展示。详见[skill及选例说明](docs/entitybench_firstframe_figure_skill_selection_20261030.md)；未启动额外GPU生成或修改官方指标。

## 原始完整benchmark设计

| 项目 | 原始固定设置 |
|---|---|
| 范围 | EntityBench完整140-ID split；140集／2491镜头，easy/medium/hard为80/40/20 |
| 镜头关系 | 1136硬切、1355接续；保留原prompt、cut标记、镜头次序 |
| 首帧 | 339个单人v7路线硬切、797个基础/原生参考硬切；实际注入/跳过以生成记录为准 |
| 人物与3D | episode-local人物参考、对应FaceLift Gaussian、273点／512姿态标定，参考/Gaussian SHA绑定 |
| 图像 | Qwen-Image-2.1，revision `790c92633540aa0cb11d9abf19eb46d861714758`；素材1024²/40步，首帧1024×576/20步，KV cache、sequential offload |
| v7 | color-safe局部残差；共享Control在step12的latent分叉，强度0.4；身份cosine<0.4注入6步，否则1步 |
| 多人硬切 | 原生多参考生成，无多人3D注入 |
| 接续 | 自身前一视频实际解码尾帧＋当前文字；不另生成首帧，不再次注入，不向Wan额外输入人物图 |
| 视频 | Wan2.2-I2V-A14B，revision `206a9ee1b7bfaaf8f7e4d81335650533490646a3`；LightX2V high/low 4step_1022 LoRA；BF16、四步、81帧/16fps、480p面积上限399360像素 |
| 官方评测 | revision `3776439e7037b2128ac3b5d2d1ce63e424cca166`；strict/skip_vlm，无配置覆盖；完整episode评分，集间等权聚合 |

原始输出：`outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140/`。1355个接续输入链均已核验；状态为非VLM完成、全片人工视频审阅未完成。不要重复启动原始队列。

### 原始140集正式指标

| 指标↑ | 原始完整benchmark |
|---|---:|
| 主体一致性 | 0.9362 |
| 时间稳定性 | 0.9858 |
| 运动平滑度 | 0.9917 |
| 动态程度 | 0.4853 |
| 美学质量 | 0.5817 |
| 成像质量（0–100） | 59.6517 |
| 角色出现率 | 0.9212 |
| 物体出现率 | 0.8942 |
| 场景出现率 | 0.6303 |
| 跨镜头人物区域DINO cs_face | 0.7155 |
| 跨镜头物体DINO cs_object | 0.8143 |
| 接续边界一致性 | 0.9822 |

这些是集等权全140集结果。官方cs_face/cs_object缺少VLM保真度门控，scope为`partial_without_vlm`，不能与论文完整门控口径严格排名。独立脸部裁剪DINO、可见单脸子集诊断都不替代上述cs_face或出现率；详细解释按需见[指标解释](docs/entitybench_qwen140_metric_interpretation_20261021.md)。

## 当前多人3D集成方案

本方案保留Qwen＋FaceLift/v7＋Wan四步核心，同时接入Agent路由、人物提示词强化和episode画风约束。单人/多人是在一张关键帧上逐角色使用同一v7，并非对同一张脸重复两次注入。

### Agent与路由

Agent使用配置中的`SHOT_AGENT_MODEL`（当前qwen3.8-max），读取上一实际尾帧、前后shot文本、当前调度及数据集人物原句；一次人物素材查询后给出短JSON计划，不加多轮语义验收或重试。提示词见[路由提示词](docs/entitybench_qwen140_route_agent_prompt_v1_20261026.md)。

| 路由 | 条件与输入 | 执行 |
|---|---|---|
| A直接接续 | 上一实际尾帧中的人物、位置、机位能直接承接当前动作 | 该尾帧＋强化行动prompt生成视频，不新增关键帧或注入 |
| B过渡＋行动 | 要在同一可连续视觉空间显示人物进入，且之后还有当前shot行动 | 生成进入完成、行动尚未完成的目标尾帧；Wan3.0-video API用实际前尾帧＋目标图生成3秒过渡；再用过渡视频实际解码尾帧生成81帧行动段，固定1＋1 |
| C新镜头 | 主体、拍摄区域、机位或构图需重建，故事不要求连续移镜 | 结合人物参考、当前文字生成动作起始帧；接续C只用上一实际尾帧作已有环境/画风线索，硬切C不传前尾帧；从新帧生成视频 |

原始cut=true强制C；cut=false并不等于必须A。硬切首帧默认包含全部调度角色，仅原文明示稍后入场、画外或仅声音时排除，不由Agent自行推迟人物出现；首帧保留动作起点。原cut标记仍保留给官方边界评测。B增加过渡时长，新结果与原始不保证等时长。没有独立背景素材库；接续帧唯一已有环境图是自身上一实际尾帧。

### 人物素材、提示词与画风

- 人物参考只含数据集角色原句、统一episode画风、单人单视角纯背景约束，不带剧情、整集摘要或多动作。每张参考绑定自己的Gaussian；禁止把新人物图配旧身份的3D。
- Agent查询当前关键帧所需的新角色或无法确认角色；上一尾帧中已可确认的人物不重复加入Qwen条件图。所有可见角色的身份参考仍供3D匹配使用，图像生成条件与3D身份资产分开。
- 帧生成提示词明确图像编号与人物对应；主视频拼接调度角色的一句话描述及Agent支持的位置身份描述，指定哪个人执行哪个动作。C当前没有额外模型复查新生成首帧，不能声称已自动验证所有位置绑定。
- 同一episode的素材、首帧、B目标尾帧、过渡视频和主视频使用同一画风。当前这批4集统一写实电影风格；不把人物素材背景作为场景参考。全部图像约束单幅连续画面、每人一次、禁止分屏/上下拼图。

### 多人匹配与v7门控

1. 在step12 pred_x0中检测全部可靠人脸（置信度≥0.5），用episode-local身份参考做全局一对一匹配，再逐角色准备3D与mask。不得先按面积取N张再强制分配全部角色，避免排掉真正人物后把身份注入额外人脸。
2. 不新增面积禁注入阈值。当前按脸高/128把强度缩放至基础强度的30%–100%。身份cosine 0.4仅决定6步/1步，不是角色匹配可靠性的通过线。
3. 保留原Gaussian渲染、姿态回测（pitch/yaw误差≤12°、roll≤15°）、color-safe内脸mask及明确闭眼提示词门控。逐角色失败不妨碍其他合格角色注入。
4. 用户已授权整张没有可用角色时也使用同次Control继续，记录原因与跳过角色；不把它计作实际注入，不重生以筛选高分。3D素材准备失败也保留2D参考并记录该角色缺失3D，不移用其他身份。
5. 技术异常保存日志和有效前缀；无法继续的episode列为未完成，队列继续下一集。全局输入/冻结代码完整性错误仍记录为批次异常，不能用替换输入掩盖。

关键帧和主视频模型、尺寸、步数延续原设置；B专用首尾帧模型是配置API中的Wan3.0-video。输入SHA、parent video/tail链和实际残差均留档；只对合法缓存续跑。

### 已完成验证：run700（10镜头）

| 指标↑ | 原始run700 | 当前多人3D集成 |
|---|---:|---:|
| 主体一致性 | 0.9365 | 0.9610 |
| 时间稳定性 | 0.9893 | 0.9926 |
| 运动平滑度 | 0.9936 | 0.9949 |
| 动态程度 | 0.5000 | 0.5000 |
| 美学质量 | 0.5861 | 0.6084 |
| 成像质量（0–100） | 53.9346 | 54.6676 |
| 角色出现率 | 0.9667 | 1.0000 |
| 物体出现率 | 0.9630 | 0.9630 |
| 场景出现率 | 0.9000 | 0.7000 |
| 跨镜头人物区域DINO cs_face | 0.7851 | 0.7698 |
| 跨镜头物体DINO cs_object | 0.9036 | 0.9008 |
| 接续边界一致性 | 0.9821 | 0.7407 |

这张表只比较同一run700，不能当作全140集改进。当前多人3D整集857帧、约53.56秒；9张新关键帧中8张实际注入、15角色次，5张同时注入至少两人；3:2是A无新关键帧，1:2按授权正常跳过。前6shot保留原匹配版本，3:4起修复全脸匹配，错误尝试未进入评测。

实际仍有3:4多余Stefan、3:5残留Nora后消失、4:1楼梯起点和酒杯归属偏差。已查每shot首/中/尾帧，未见上下分层；并未完成逐帧身份/画风评测。角色出现100%不代表身份、人数和动作100%正确。没有另生成全套Control视频，不能把整集差异当作3D插件独立因果增益。

产物根目录：`outputs/entitybench_qwen140_run700_multiface_episode_20261028/`；原始输入/生成记录、`injection_audit.json`、`audit/completed_episode_validation.json`、`evaluation/run700_multiface_episode_v1/comparison.json`为事实来源。

## 已完成的低指标episode首批

从全140集逐集结果筛选：低于完整benchmark动态、美学、成像、角色出现、场景出现、cs_face任一均值的共有138集。本轮明确只做首批4集、48shot，不宣称全量重跑。run771和run455六项均低于均值；run484、run1404补充之前讨论过的角色进入/画风问题。筛选依据、精确阈值、原结果SHA、代码快照冻结在`cohort.json`。

| episode | shot数 | 原结果低于全benchmark均值的指标 |
|---|---:|---|
| run771／i15_j24 | 11 | 动态、美学、成像、角色出现、场景出现、cs_face |
| run455／i0_j24 | 15 | 动态、美学、成像、角色出现、场景出现、cs_face |
| run484／i15_j47 | 12 | 动态、美学、角色出现、cs_face |
| run1404／i170_j220 | 10 | cs_face |

这批22个角色使用干净单人参考和对应新Gaussian，逐episode冻结；因此相对原始结果同时改变人物锚点、路由、提示词和注入范围，属于完整系统改进试验。素材和视频等GPU重任务串行；每集完成后按原冻结官方口径评测12项非VLM，不与生成抢占GPU。

- 后台tmux已随任务完成退出；保留入口`pretest/qwen140_multiface_weak_episode_batch.py run`，当前不需要重启。
- 根目录：`outputs/entitybench_qwen140_multiface_weak4_20261028/`。
- 实时总进度：`status.json`；各集：`experiments/<run__tag>/progress.json`；日志：`queue.log`和各集`logs/`。
- 技术异常：`technical_exceptions.json`及各集`technical_exception.json`；素材异常：各集`asset_failures.json`。门控跳过详见每shot的`frame/injection/result.json`与`preparation_attempt.json`，不混入技术异常计数。
- 视频/指标入口：根目录`index.html`及`report.md`；四集`review/index.html`含原始/新整集及逐shot链接，各自`report.md`含前后12项指标。
- 2026-10-29最终状态：4/4完整episode、48/48shot、每集12项官方非VLM覆盖、评分失败0、VLM调用0。视频/尾帧SHA与完整四集配对聚合已核验。历史兼容修复及22/41shot保留记录见[续跑记录](docs/entitybench_weak4_resume_20261029.md)，不能声称48shot全由最后一次修订生成。

### 四集最终配对结果

以下按同四集等权聚合，不与全140集均值直接作前后对照。官方12项全表见运行根`report.md`。

| 指标↑ | 原始同四集 | 当前多人3D集成 |
|---|---:|---:|
| 动态程度 | 0.4595 | 0.4515 |
| 美学质量 | 0.5442 | 0.5603 |
| 成像质量（0–100） | 62.6650 | 61.6548 |
| 角色出现率 | 0.8278 | 0.9792 |
| 场景出现率 | 0.6176 | 0.6281 |
| 人物区域cs_face | 0.6921 | 0.6736 |
| 接续边界一致性 | 0.9852 | 0.5836 |

出现率和美学改善，但动态、成像、cs_face及接续边界退步，不能称为全面优化成功。完整人工视频审阅与退步原因分析待完成，保留所有负例；cs_face缺VLM保真度门控，不是人脸身份准确率。

cs_face裁剪已于10月29日审计：发现原/新版语义定位错人、手部/衣服裁剪以及DINO中心裁剪去头等问题；170个入分裁剪复算与报告误差0，8个原帧像素核验无坐标/取帧错位。不能把cs_face下降直接解读为身份变差。详见[裁剪审计](docs/entitybench_weak4_cs_face_crop_audit_20261029.md)，官方指标未覆盖修改。

## 接手与结果解释

先检查`git status --short`、上述实时状态、tmux、实际进程和GPU/cgroup额度，不根据本快照重复启动。保留所有历史媒体、素材、Gaussian、冻结源码、评测报告及无关未提交改动。outputs不随Git上传；本轮未授权提交或推送。

官方评测使用[evaluate-face-injection-v2技能](/root/.codex/skills/evaluate-face-injection-v2/SKILL.md)：完整episode评分，验证视频/SHA/自身尾帧链；固定阈值、采样、canonical裁剪和官方聚合。非VLM范围完成与人工视频质量验收分开；不得将缺失值填0、筛掉负样本或仅报提高的指标。

cs_face的输入是可能包含衣服/背景的人物区域，既不是InsightFace，也不是对人物参考的身份准确率。presence是定位/CLIP通过率，场景未过线不等于没有背景。动态程度是运动判定通过率；美学、成像、人物身份和动作正确性也需结合视频审阅。

按需入口：[原始benchmark执行/核验](docs/LLM_HANDOFF_QWEN140_20261014.md)、[指标解释](docs/entitybench_qwen140_metric_interpretation_20261021.md)、[重跑设计](docs/entitybench_qwen140_shot_rerun_plan_20261026.md)、[新会话提示词](docs/qwen140_new_window_prompt.md)。
