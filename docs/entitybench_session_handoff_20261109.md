# 会话交接：累计35条动态替换版与独立整集队列（2026-11-09）

<!-- push-state-20261110:start -->
## 代码归档前状态（2026-11-10 15:55 北京时间）

用户已授权按movie-github-ssh-push skill提交并推送实验代码和交接文档。此记录的运行快照：78集4步v2整集队列正常运行，生成154/1218shot、完整完成11/78集官方12项，当前`00087828-20a3-3c31-82c5-88bf08bd0d61__run988__i55_j74__T240`／`silas_facelift`。此前7个Agent工具调用结构错误仍留档，无新增中断；未改运行中的源码/计划/模型/阈值，未重启队列。

最新完整benchmark仍为`outputs/entitybench_full140_current35_newepisodes_20261110/`的35条版叠加7集77shot冻结版本；其后完成集尚未合并，不将当前完整11集误报为均已进入该表。所有outputs、模型、密钥、PPT、模型图草稿及无关前端修改只留本地；代码推送不包含这些产物。推送结果以会话最终提交hash和远端验证为准。
<!-- push-state-20261110:end -->


<!-- current35-new-episodes-20261110:start -->
## 最新benchmark更新：35条版叠加7个完整新集（2026-11-10 12:09 北京时间）

用户要求把新结果替换到当前35条版。最新权威版本为`outputs/entitybench_full140_current35_newepisodes_20261110/`，13项全部完成；冻结11:58时已完成7集77shot，其余133集2414shot沿用35条版，完整140集2491shot。新整集覆盖其中4个旧独立动态替换，余31保留；不覆盖原文件，不以分数筛选新集。默认DINO .7990→.7976；主体 .9414→.9431；动态 .4873→.4731；角色出现 .9479→.9501；场景出现 .6249→.6206；边界 .8441→.8243。所有指标见`complete_report.md`与`complete_comparison.json`。完整说明`docs/entitybench_full140_current35_newepisodes_20261110.md`。

官方12项同协议完整episode缓存重聚合，默认DINO对新裁剪重做可见单脸门控、43项全部视觉复核（41keep/2明确错人exclude），保留真实漂移/重复/不确定，非外部人工签核；有效episode130，角色计分次1486。已精确复现35版，2491媒体SHA/软链接与178项冻结绑定全部验证。strict/skip_vlm、VLM0、partial_without_vlm，非51项全评。原35版保留为历史基线，下文称其“最新”的历史段落由本段覆盖。

后台仍在跑78集4步队列：12:09实查119/1218shot生成，8/78集完整（89shot）；第8集run909在本表冻结后完成，尚未纳入。当前run910/i20_j43构建3D素材。7个Agent lookup_characters调用结构异常的episode留档中断，已继续其他集，未自动语义修复；细节见完整说明。主supervisor已恢复运行，无新守护Agent，无旧40步/4步动态队列恢复。保持当前授权与现有源码；下次先实查进程，勿重复启动。没有Git推送。
<!-- current35-new-episodes-20261110:end -->


<!-- paused8-benchmark-current:start -->
## 最新benchmark：40步动态实验暂停，累计35替换版已完成（2026-11-09 22:11 北京时间）

本会话用户要求先停40步动态实验、更新当前benchmark；本会话最新“继续”承接benchmark收尾。**40步动态队列保持暂停，automatic_resume=false；本轮没有恢复任何生成。** 实查工作区另有独立78集4步整集队列正在运行（见其最新恢复记录），本次不停止、不修改该无关队列。下方19:40动态队列运行及本8条收尾中状态已被本段覆盖。

原1277新生成队列只完成8条，补齐全部六项VBench后，M0010仍0，M0012/M0069/M0074/M0189/M0022/M0045/M0079为0→1。新增7，累计64条原版40步尝试／35条替换／2456条冻结基线媒体不变；剩1269条尚未生成。全部负例/下降/历史媒体保留，仅动态增长选择，无其他质量门控，不覆盖原媒体。原生成计划/源码保持冻结；重启时先读实况、验证原plan/SHA，不能重跑已有8条。恢复本40步动态队列仍须用户后续明确要求；独立整集队列状态不由本benchmark记录覆盖。

**最新权威输出**：`outputs/entitybench_original40_priority_remaining_20261109/paused8_benchmark_20261109/`。`benchmark_summary.md`／`benchmark_summary.json`为冻结替换前→累计28→当前35的13行汇总；`complete_report.md`／`complete_comparison.json`是28→35；`replacement_manifest.json`及`candidate_media/`是完整2491媒体版本；`completion_integrity_audit.json`为审计。`table4_alignment.md`明确Table4的9项非VLM已有值、15项VLM未评；`official51_metric_status.json`保留39项null。覆盖完整140集不等于论文51项全评完。

本次7个完整episode／161shot上下文重算，161×6=966个VBench缓存值逐条精确核验，全部评分失败0；其余133集沿用精确结果，官方函数全140聚合。所有2491媒体SHA、209冻结绑定、161份提取及原canonical crop SHA均核验。默认DINO助手查看3份隐藏分数裁剪＋3个精确完整帧＋角色描述和shot调度，3保留／0新增排除，57个历史排除仅相同crop SHA沿用；Riley(run44/9:1)疑似前景错角色定位，但背景正确目标无法无歧义确认，保留未决，不凭分数剔除，不宣称外部人工签核。默认最终为`visible_region/report.json`，automatic为中间产物。覆盖130有效episode不变，计分角色次1467→1465、复核后保留1782→1781，覆盖下降照报。

当前28→35（官方四位舍入）：默认DINO .7990→.7990；主体.9415→.9414；时间稳定.9864→.9863；平滑.9920→.9920；动态.4840→.4873；美学.5859→.5859；成像60.8665→60.8634；角色出现.9479→.9479；物体出现.8981→.8981；场景出现.6238→.6249；raw人物DINO .7240→.7238；raw物体DINO .8077→.8077；接缝.8448→.8441。动态精确.4840030568846357→.4872541391357179，本批+.3251082251082251个百分点；相对最初冻结.47335298473456355累计+1.3901154401154359个百分点。相同舍入值不表示精确不变。

本次benchmark收尾无新增媒体生成、无API/VLM、无删除/推送；此句仅描述本次收尾，不描述另一个整集生成任务。strict/skip_vlm、缺fidelity gate/partial_without_vlm保留；raw cs_face是人物区域DINO而非人脸识别，项目默认可见单脸定位复核DINO另列。此次仅使用评分入口`pretest/qwen140_original40_paused8_benchmark_20261109.py`（无生成action）。原根status.json已更新为暂停＋benchmark收尾完成，暂停与8条评分证据均保留。详细记录`docs/entitybench_original40_paused_benchmark_20261109.md`。
<!-- paused8-benchmark-current:end -->


## 最新恢复：2026-11-09 22:09（北京时间），扩容后继续78集4步整集队列

用户明确“继续吧，我刚刚扩容了一下”。实查数据盘450GiB、可用约113GiB，覆盖此前约60～65GiB的新增估计；不设reserve、不删除旧产物。旧整集tmux已退出，状态停留run714/i20_j34的1/10shot完成、2:1首帧第8/20步，日志无退出异常尾部，不能断言是磁盘不足。已完成的人物3D资产、1:1视频及2:1 Agent/提示词/帧task全部保留。

恢复tmux仍为`qwen140_never_rerun_v2_20261109`，**实际新入口为 `pretest/qwen140_never_rerun_resume_20261109.py run`**，日志及范围仍`outputs/entitybench_never_rerun_episodes_v2_20261109/`。原78集/1218shot冻结plan、原生成与评测源代码不改。新恢复入口只绕开旧v1迁移器对“无成品帧”的自动archive/replan，直接安装同一v2并继续`run_remaining`，从2:1相同输入重算未完成的采样，不新查角色/重写Agent计划。17项已存结果与计划SHA绑定在`resume_after_expansion_20261109.json`，编译与v2/run/tag探测通过，预检API0。正常后续新shot仍按原流程调Agent与B过渡API；主视频仍4步、原多人v7门控，不恢复40步动态队列。

恢复前另一已生成8条的完整上下文评测已释放GPU，状态为official_full140_complete_default_dino_review_pending（新增7、累计35替换），不覆盖其结果，不把默认DINO未复核当完成。该评测的“暂停所有生成”交接是此前状态，**本轮用户明确继续78集整集生成为当前授权**，无新增动态40步授权。第一集实际恢复阶段以新status/progress/log为准，完成进度不重复清零。预计仍约5～6天，首集完成后校准。无守护Agent、无自动改模型/提示词/阈值、无推送。



更新：2026-11-09 22:11（北京时间）；仓库 `/root/autodl-tmp/movie`。此文和新窗口提示词覆盖旧会话交接中的运行状态；旧文只作历史，不恢复其队列。

## 最新：按恢复证据排序的剩余1277条40步队列已启动

用户先要求“其他动态0都重跑”，随后明确“优先分析之前0→1，先跑容易恢复的”。范围保留全部，改变优先级：当前累计28版剩1305条0，其中28条已在相同40步条件下失败，保留精确记录不重复GPU；1277条新生成。**新全范围授权覆盖下方旧的“未授权全1333”文字，但不恢复旧4步1333调度或整集。**

分析已核对56条历史结果/首帧、12组4步/40步的首中尾代表帧、队首30条实际首帧。粗类别：运镜3/4、持续表演4/6、普通travel14/32、prop/arm1/8，加6先导6/6，合计28/56（相对同词4步0则25/53，另3个4步1保留）。不是随机成功概率，标签有误归类。更强优先依据：**30条当前benchmark0、已有严格同PNG/词/种子的4步结果1、尚未试40步**；视频/生成/receipt/评测SHA全部核对，优先这30；其余1247按原文运镜、大幅持续/横向位移、远小/细微/静止约束等可审计线索排序，不删除难例。VBench的0不等于完全不动，1也不代表身份/动作正确，不能据此恢复质量筛选。报告`docs/entitybench_original40_recovery_priority_20261109.md`，排队唯一来源`outputs/entitybench_original40_recovery_priority_analysis_20261109/priority_manifest_v2.json`；早期无v2的JSON为未执行草稿，不读取恢复。

tmux `qwen140_original40_priority_remaining_20261109`于19:40启动，入口`pretest/qwen140_original40_priority_remaining_20261109.py run`，输出`outputs/entitybench_original40_priority_remaining_20261109/`；首批M0010/M0012/M0069/M0074/M0189/M0022/M0045/M0079。实时`status.json`／`queue.log`／`priority_batch_*_worker.log`／`priority_batch_*_evaluate.log`。启动时0/1277不是出片完成，以实时日志为准，不重复启动。全部1333条纯动作强化词已编写，只有部分实际生成/验证，不能说都跑过；本轮沿用纯动作冻结词，**不追加旧469条背景句**，不改首帧/3D/种子/81帧主段16fps/尺寸和原版40/WanScheduler/CFG3.5/noLoRA/shift5/.9/TorchComplexRoPE。

每8条生成后原6VBench，按0→1追加原28，未增益保留旧媒体，其他下降不筛。`progressive_replacement_manifest.json`和`candidate_media/`为最新动态选择版本，`selection_checkpoints/`冻结稳定source/SHA（不用会变化的live链接冒充旧快照）。完整presence/DINO/接缝数值在新视频全完成后统一重算完整受影响episode并聚合140，进度表中的这些指标保持pending，不能填旧数；上下文入口`pretest/qwen140_original40_priority_context_20261109.py`，最终默认DINO仍需隐藏分数定位复核后final。15条B保留既有过渡，只重生81帧主段，再按原方式拼接并评完整视频；代表B拼接回放逐像素一致，完整140官方/默认DINO当前28基线回放通过。验证在新输出`validation/`。约14天，首批约2小时；无语义重试、改词/降配、删除、付费API/VLM、推送或守护修复。磁盘约13GB可用，不设reserve。其余历史状态见下方。

## 最新：19:19新增24条完成，累计28条替换

`outputs/entitybench_original40_next24_20261109/`：24/24生成与六项VBench完成，8条0→1（N36、N39、N40、N47、N49、N50、N52、N53），16条未增长保留旧媒体。累计56条原版40步尝试／28条动态增长替换；独立完整140/2491映射在`replacement/replacement_manifest.json`（本批replaced=8，cumulative_replaced=28），实际媒体在`replacement/candidate_media/`。原20条及2463个原基线视频SHA保持不变；旧媒体与负例全部保留，不按其他指标过滤。

新增8个受影响episode／171完整shot上下文评分、可见单脸提取与默认DINO重算已完成。默认DINO助手逐项查看6份隐藏分数裁剪、全部4个精确完整帧及角色原文，6保留／0新增剔除，57项历史排除仅精确SHA沿用；不称外部人工签核。所有评分失败0、171×6缓存值精确一致、模型配置／strict/skip_vlm／VLM0核验通过，原质心算术及全140官方聚合不变。tmux／worker已退出，GPU空闲，旧1333／整集不恢复，未自动新增下一批。

最终报告`replacement/complete_report.md`／`complete_comparison.json`，三版本完整表`replacement/full140_three_versions_report.md`／`full140_three_versions_comparison.json`（冻结替换前→selected20→累计28）。媒体与报告SHA、171shot定位协议和裁剪绑定审计见`generation_completion_validation.json`／`replacement/completion_integrity_audit.json`等；默认DINO的最终结果为`replacement/visible_region/report.json`，automatic仍是中间产物。

selected20→累计28：默认DINO **.7989→.7990**；主体.9417→.9415；时间稳定.9864→.9864；平滑.9920→.9920；美学.5859→.5859；成像60.8571→60.8665；角色出现.9479→.9479；物体出现.8981→.8981；场景出现.6238→.6238；raw人物DINO .7239→.7240；raw物体DINO .8077→.8077；接续边界 **.8452→.8448**。动态精确 **.480048511430→.484003056885（本批+.395455个百分点）**，相对最初冻结替换前.473352984735累计 **+1.065007个百分点**。其他值为官方四位舍入，不把显示相同说成精确不变。DINO130有效episode／1467计分角色次不变，全角色调度4991，present4735→4735、自动合格1838→1839、复核保留1781→1782。VLM相关项仍缺失，raw cs_face/cs_object无fidelity gate，partial_without_vlm。完整新批次文档`docs/entitybench_original40_next24_20261109.md`。

## 12:43新增24条动态0候选启动（历史，已完成）

用户在确认selected20完整结果后要求继续跑此前动态0镜头并替换。本轮已独立冻结24条新候选N33–N56／19episode，排除原32，优先原文明确位移；当前selected20基线和已有同词4步参照均为动态0。沿用原PNG／3D／逐字提示词／种子／81帧16fps／尺寸与原版40步CFG3.5/noLoRA/TorchRoPE配方，无新增动作或运镜句。

tmux `qwen140_original40_next24_20261109`，入口 `pretest/qwen140_original40_next24_20261109.py run`；输出`outputs/entitybench_original40_next24_20261109/`。363项冻结绑定、24条实际4步生成与评分SHA通过；纯缓存完整140官方与DINO基线复现通过。12:45模型加载完成（85.9秒），resolved_runtime确认原版40／CFG3.5／noLoRA／WanScheduler／TorchRoPE配方，N33已编码并进入真实40步采样，初始0/24生成，不当作GPU出片成功。每8条生成后6VBench，全24结束后仅将新动态0→1叠加此前20条，完整媒体映射与后续上下文评分在`replacement/`，保留原文件与全部失败／降分。成功数未知，不说“新增替换24”。

基线固定`baseline_snapshot.json`（此前完成selected20的140集版本）；新增成功所在完整episode自动定位／presence／DINO中心／真实接缝重算，其他集复用并聚合全140。可见单脸材料仍须逐项视觉复核后运行`pretest/qwen140_original40_next24_context_20261109.py final`；auto不充最终。旧1333和整集保持暂停，无守护／自动改词／自动降低配方／恢复旧队列。预计净生成约6小时20分，再加上下文与复核。新文档含完整名单：`docs/entitybench_original40_next24_20261109.md`。下方11:45“下一批尚未冻结”均为此次授权前历史。

## 11:45最终状态：20条替换版全部可用指标已完成

- 自动评分及539shot可见单脸提取于11:39前完成；18/18个受影响episode、539/539shot完整，2worker正常退出，122集原评分复用，官方函数聚合140集／2491shot。strict／skip_vlm，VLM0，所有12项非VLM评分失败0；缓存收据18份／539shot复用验证通过。原tmux已退出，GPU利用率0、显存约1MiB；数据盘约13GB可用。未恢复旧队列或启动新生成。
- `complete_report.md`／`complete_comparison.json`为最终13行表（官方12项＋默认条件DINO）；`official_report.md`仍保留官方12项；`official_completion_validation.json`／`completion_validation.json`记录完整性与输入SHA。输出根目录仍为`outputs/entitybench_original40_dynamic_gain_selected20_20261109/`。
- 本次定位复核由Codex逐项查看4页隐藏分数材料、原文／角色描述，4个歧义case另看原尺寸代表帧；20项全部保留，新增剔除0。**这是助手视觉复核，不宣称用户或外部人工签核。** Akira关联不清、Vittorio重复、Ethan宽区域／前景与阳台分裂、Caleb／Silas服饰与反射关联未决均保留，未为提分删除。20个决定与证据已冻结在`visible_region/review/decisions.json`，绑定manifest SHA；11-07基线的57项历史排除仅按精确输入沿用。
- `visible_region/report.json`为重算结果，`automatic_results.json`保留作中间产物。人物一致性DINO（可见单脸＋定位纠错）**.7986→.7989**，使用11-07扩展复核的同口径当前基线；有效episode130→130，计分角色次1467→1467，canonical present4734→4735、自动合格1837→1838、复核后保留1780→1781，完整调度4991均保留覆盖统计。原区域像素／原DINO、官方质心算术／集间等权未改。

| 指标 | 冻结140基线 | 20条替换版 |
|---|---:|---:|
| 默认人物DINO（可见单脸＋定位纠错） | .7986 | .7989 |
| 主体一致性 | .9418 | .9417 |
| 时间稳定性 | .9865 | .9864 |
| 平滑度 | .9921 | .9920 |
| 动态程度（官方四位舍入） | .4734 | .4800 |
| 美学 | .5861 | .5859 |
| 成像0–100 | 60.8533 | 60.8571 |
| 角色出现率 | .9479 | .9479 |
| 物体出现率 | .8989 | .8981 |
| 场景出现率 | .6238 | .6238 |
| 原始人物区域cs_face | .7239 | .7239 |
| 原始物体区域cs_object | .8076 | .8077 |
| 接续边界一致性 | .8470 | .8452 |

动态按未舍入逐集均值及媒体映射再核验：**.473352984735→.480048511430，+.669553个百分点**。不能用官方表已舍入值作差代替精确增益。下降照报，尤其边界-.0018、物体出现率-.0008；没有恢复质量门控或撤销20条选择。VLM相关项仍缺失，raw cs_face/cs_object的fidelity gate不可用，范围仍partial_without_vlm，不称完整论文全指标。下一批小实验名单仍未冻结，勿自动恢复1333或整集队列。

## 11:17新会话实际接手核验（以下为完成前历史）

- 已读本页、主交接顶部及`evaluate-face-injection-v2`技能／可见单脸协议，并检查Git、tmux、进程、GPU、磁盘。唯一相关tmux仍为`qwen140_selected20_context_20261109`，原父调度与2个评分worker正常，无新生成或恢复的旧队列。工作区既有无关改动原样保留，未提交／推送。
- `status.json`实时为`parallel_official_evaluation`，两worker各5/9集，共10/18集完成、139/539完整上下文shot完成；剩余8集各50shot，按镜头进度仍有大量工作，不能仅按10/18估计剩余成本。已有日志无ERROR／Traceback，已完成集接缝失败0。单张48GB 4090，两worker合计约7GB显存、GPU利用率约75%；数据盘约14GB可用，未删除文件或设reserve。
- 本会话重新核对2491个实际视频链接及内容SHA、300项冻结输入绑定，20替换／2471不变／32尝试／12未增益保留，错误0。凭据未读取。审计落盘`outputs/entitybench_original40_dynamic_gain_selected20_20261109/takeover_input_validation_20261109.json`。
- 新增独立收尾适配器`pretest/qwen140_selected20_review_finalize_20261109.py`，不改正在运行的scorer、plan或媒体，不启动GPU评分／生成。`baseline`动作已实际执行：复现11-07扩展复核当前基线140个逐集结果与官方宏聚合，4991角色次／57个精确crop排除／1467计分appearance／130有效episode。结果`reviewed_dino_baseline_validation.json`。这不是20替换版DINO完成声明。
- 收尾顺序：先让原任务完成评分与539shot可见单脸提取，生成`visible_region/review/manifest.json`和隐藏分数图；逐项核对完整帧／原文／角色，保存绑定manifest SHA的`review/decisions.json`；之后用同一评测环境运行新适配器`final`动作。它会验证18集评分、缓存收据和动态精确值，复验新裁剪输入，按精确SHA沿用旧决定，复算完整140人物中心和官方聚合，输出`complete_report.md`／`complete_comparison.json`／`visible_region/report.json`。**final动作尚未运行，须等真正完成定位复核；未新增等待守护进程或自动作出审阅决定。** 其他指标和默认DINO仍以实时产物为准，不把auto或旧分数填成最终结果。

## 用户最新决定

1. 保留现有“为增强运动而强化过的提示词”，不恢复11-06原词，不继续修改动作句或额外加背景句。32条中4步与40步逐条正提示词完全相同；首帧/3D结果/种子/81帧16fps/尺寸保留。
2. 使用原版Wan2.2-I2V-A14B：40步、WanScheduler、CFG3.5、shift5、boundary=.9按时间步切换高低噪声、不加载蒸馏LoRA；BF16/SDPA/TorchComplexRoPE。与4步蒸馏是适配配方对照，不能把差异孤立归因于步数或去LoRA。
3. 32＋24共56条已完成、累计28替换。用户最新已授权其他0全部40步重跑并要求按恢复证据优先；新队列1277、旧40步失败28精确复用，见本页顶部。旧4步1333队列与整集仍不恢复，没有守护修复Agent。
4. 之前撤销多指标质量门控仍有效：不按美学/成像/主体/.01等过滤。**最新要求仅替换动态有增长的结果，按当前完整benchmark的0→1判定。** 非动态指标的下降也要报告，不设新硬门槛。定位复核是保证DINO指标正确，不是偷偷恢复媒体质量筛选。
5. 保留旧媒体、负例、原报告/计划；独立版本映射替换，不覆盖。正常技术完整性/SHA检查保留；不更改评测阈值/裁剪来提分，无VLM/付费API。

## 32条最终实验（已完成）

输出：`outputs/entitybench_original40_scope32_20261108/`，`report.md`、`comparison.json`、`completion_validation.json`。11月9日07:03结束，全32生成和6VBench已完成，无质量过滤。

| 指标 | 同词4步蒸馏 | 原版40步 |
|---|---:|---:|
| 动态程度 | .093750 | .625000 |
| 主体一致性 | .968172 | .959792 |
| 时间稳定性 | .987968 | .984634 |
| 运动平滑度 | .992946 | .991425 |
| 美学 | .617906 | .612731 |
| 成像0–100 | 62.956418 | 63.047187 |

同词4步的29条动态0中17条变1；原3条1均保留。新增24条0→13/24，原8条3/8→7/8。源类别描述性归组：运镜4条0→3通过、位移10条1→8、持续表演8条1→6、具体操作10条1→3。不能将VBench主体一致性当人脸身份DINO，也不能把这些目的性样本均值叫完整benchmark。

32条里并非所有人都视觉正确，尚未完整语义验收；动态1只表示原RAFT光流判定通过。此前前3条抽帧在原8输出，N02拉远明显，N03人物/背景移动增加；其他case需要继续看视频。全32输入/视频/评分SHA、40步实际配置、6均值独立复算已核验。

原8：`outputs/entitybench_original40_vs_distill4_torch_rope_20261108/`，入口`pretest/qwen140_original40_torch_rope_20261108.py`。首次customJSON漏rope_type，默认走未安装FlashInfer导致首步报错、0输出，日志保留在`outputs/entitybench_original40_vs_distill4_20261108/`。修正显式torch_complex_rope＋double_precision_rope=True，与4步API默认一致；实际首条已通过，不要再走失败配置。

## 已执行：只替换动态增长的20条

新版本：`outputs/entitybench_original40_dynamic_gain_selected20_20261109/`。

- `replacement_manifest.json`：完整140集2491shot映射，20条替换、2471条视频SHA不变；32尝试及12条未增益均保留记录。
- `candidate_media/<完整episode_id>/<scene:03d>_<shot:03d>.mp4`：完整版本的实际视频符号链接；旧源文件没有覆盖。
- `selected_cases/N##/result.json`：20条选中记录；不是重新生成视频。
- `selection_validation.json`：2491媒体、20/2471范围与539缓存输入核验。
- `dynamic_exact.json`：按140个episode等权的动态精确算术更新，0.473352984735→0.480048511430，增加0.669553个百分点。不是全指标重算完成的声明。

**为什么20而不是17：** 17是相对同词强化4步的新增成功；原3个强化4步就能成功的参照，在冻结benchmark中仍是0，故其40步视频对当前benchmark也有增长，合计20。不能把29当已经替换数，也不能说替换后动态变.625（那是32条均值）。

此处“当前benchmark”固定在`outputs/entitybench_full140_latest57_v2_141_20261106/snapshot.json`，140集2491shot＝57完整重跑＋83原版＋141独立v2覆盖。不是140集统一新版，也不是11-08尚未替换的230条候选。原始完整benchmark路径`outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140/`；冻结当前原动态.4733529847，最初完整benchmark动态约.4853，口径不同不混。

## 已完成：20条替换后的完整上下文评测（下列启动信息留档）

- 原tmux：`qwen140_selected20_context_20261109`，现已退出，不恢复。
- 实际启动入口：`pretest/qwen140_original40_selected_gain_run_20261109.py`（run wrapper，仅合并status字段防止重复关键字，不改变评分）。
- 准备/worker主逻辑：`pretest/qwen140_original40_selected_gain_20261109.py`。**不要直接重新执行旧的1333 context wait-run。**
- 已完成18个受影响episode、539个完整上下文shot，2个GPU worker各4CPU线程。所有worker已退出，未新增生成任务。
- 六项VBench全部按视频SHA验证缓存复用，人物/物体/场景定位、完整人物/物体中心与真实边界仍按完整episode重新计算；其余122集复用原评分后官方函数聚合全140。
- `status.json`，`queue.log`，`workers/0/worker.log`、`workers/1/worker.log`及各results目录可查进度。方法名参数沿用历史helper字符串`v2_independent_shots_only_141`，只是名称；真实split/plan/media完全是新18集/20替换，不能按那个字符串推断范围。
- 完成后：`official_report.md` / `official_comparison.json` 为官方非VLM12项完整表；原始raw cs_face仍注明人物区域DINO、无VLM保真门控。
- 随后GPU提取可见单脸与默认DINO复核材料，`visible_region/review/manifest.json`与分数隐藏图；`visible_region/automatic_results.json`只是自动中间结果，不报为默认最终DINO。人工决定并重算受影响质心、再全140聚合后才补默认DINO；未决/真实身份漂移不能为提分排除。
- 评分/定位审阅不会改变已明确的20条动态选择，也不会恢复多指标质量门控；下降照报。无新视频、无API/VLM、无自动恢复旧队列。

`status`由复用helper阶段写入时可能只带stage/jobs，缺少外层quality_gate字段不表示恢复门控，选择规则固定在plan/replacement_manifest。

## 新会话优先工作（11:45更新）

1. 先读本页顶部完成状态、LLM_HANDOFF最新记录和真实tmux/进程/GPU/磁盘；selected20已完成，不因下方旧启动信息重复运行。保留缓存、媒体与所有决定，不换模型、降CFG/步数或更改阈值。
2. 官方12项、20份隐藏分数材料的助手视觉复核、默认可见单脸DINO重算和完整13行汇总已完成，读取`complete_report.md`／`visible_region/report.json`及验证文件。不要把automatic当最终，不把本次助手复核称为外部人工签核；不额外套用旧角色名排除。
3. 向用户给20条映射及完整140前后指标，明确动态只增加约.67个百分点、不是32条.625直接推广。报告身份/出现/美学代价，不掩盖下降。
4. 用户说可以继续实验，但没有要求全1333继续40步（约两周）。先完成这次替换评测，再以小批量延续明确位移/运镜/持续动作；不要自行恢复低收益精细操作大队列。下一批名单尚未冻结或启动。用户保留运动强化词，不能再误说已恢复原词。
5. 可以查看32条中动态仍0的12条，区分首帧动作已结束、机位/可见面积限制、原文动作本就细微等原因。不要为刷动态凭空加动作/运镜。

## 保留的历史与安全操作

- 原1333动作为0队列输出`outputs/entitybench_all_zero_motion_background_20261108/`，已生成230、评分198、旧多指标初筛11；用户暂停，原调度/worker/context waiter都已退出。对应纯动作前版`outputs/entitybench_all_zero_fixed_frame_motion_20261108/`也不要启动。未替换正式benchmark。
- 背景26条配对结果在原1333输出`scene_presence_completed_cohort_20261108/`；7场景过阈值，但有代表裁剪变化/疑似颈部错框，不作成功真值；不是当前32条配方（当前32无背景追加句）。
- 原人物匹配缓存审计与默认扩展DINO在`outputs/entitybench_cached_matching_audit_20261107/`；后续纠正新框的GPU补充`outputs/entitybench_localization_repair_gpu_20261108/`是另口径诊断，不能顶替默认原像素协议。
- 3D是核心方法，不取消。当前首帧固定，沿用既有单/多人v7注入；视频实验没有重新做3D或改变门控。
- 不读/打印`.env.portrait`密钥，不复制用户曾贴出的密钥。不要自行commit/push，用户另有推送skill才用。保留无关`docs/frontend_agent_architecture.md`、`experiment_output/`、PPT和中文模型图文档，不reset/clean。
- 数据盘不设人为额度、不自动删模型/结果。之前明确删的Qwen2.5-14B-AWQ已删；不推断更多删除授权。磁盘当前需实时检查，评测会增加裁剪缓存。
- 旧PID只作历史，不拿本页PID当恢复命令。普通tmux任务可在SSH断开后继续，服务器需开机，无守护修复Agent。

## 选中与保留名单

| case | 镜头 | 冻结benchmark动态 | 同词4步动态 | 40步动态 | 是否替换 |
|---|---|---:|---:|---:|---|
| N01 | run323/i60_j94/2:1 | 0 | 0 | 0 | 保留原版 |
| N02 | run466/i15_j24/2:1 | 0 | 0 | 1 | 是 |
| N03 | run1/i50_j84/18:1 | 0 | 0 | 1 | 是 |
| N04 | run4/i80_j98/7:6 | 0 | 0 | 1 | 是 |
| N05 | run264/i50_j98/1:6 | 0 | 0 | 1 | 是 |
| N06 | run103/i30_j54/24:2 | 0 | 1 | 1 | 是 |
| N07 | run146/i0_j46/7:1 | 0 | 1 | 1 | 是 |
| N08 | run221/i0_j18/16:5 | 0 | 1 | 1 | 是 |
| N09 | run617/i15_j40/3:1 | 0 | 0 | 1 | 是 |
| N10 | run1285/i30_j45/3:1 | 0 | 0 | 1 | 是 |
| N11 | run4/i35_j83/25:1 | 0 | 0 | 1 | 是 |
| N12 | run4/i80_j98/8:1 | 0 | 0 | 0 | 保留原版 |
| N13 | run34/i0_j23/2:1 | 0 | 0 | 1 | 是 |
| N14 | run34/i20_j44/1:1 | 0 | 0 | 1 | 是 |
| N15 | run152/i5_j56/1:1 | 0 | 0 | 1 | 是 |
| N16 | run155/i5_j39/1:1 | 0 | 0 | 1 | 是 |
| N17 | run318/i10_j61/2:1 | 0 | 0 | 1 | 是 |
| N18 | run328/i0_j47/5:1 | 0 | 0 | 0 | 保留原版 |
| N19 | run103/i30_j54/27:1 | 0 | 0 | 1 | 是 |
| N20 | run298/i40_j64/23:1 | 0 | 0 | 1 | 是 |
| N21 | run447/i20_j34/3:1 | 0 | 0 | 0 | 保留原版 |
| N22 | run542/i35_j49/5:2 | 0 | 0 | 0 | 保留原版 |
| N23 | run740/i5_j53/3:1 | 0 | 0 | 1 | 是 |
| N24 | run298/i40_j64/23:5 | 0 | 0 | 1 | 是 |
| N25 | run4/i35_j83/10:1 | 0 | 0 | 0 | 保留原版 |
| N26 | run44/i70_j90/11:1 | 0 | 0 | 0 | 保留原版 |
| N27 | run69/i25_j44/5:7 | 0 | 0 | 1 | 是 |
| N28 | run103/i30_j54/28:1 | 0 | 0 | 0 | 保留原版 |
| N29 | run184/i25_j59/6:1 | 0 | 0 | 0 | 保留原版 |
| N30 | run267/i80_j90/3:1 | 0 | 0 | 0 | 保留原版 |
| N31 | run298/i40_j64/6:1 | 0 | 0 | 0 | 保留原版 |
| N32 | run323/i0_j14/6:1 | 0 | 0 | 0 | 保留原版 |
