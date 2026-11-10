# 主交接：原始完整benchmark与当前多人3D集成

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


## 最新：2026-11-09 21:53 北京时间，暂停40步动态实验，先跑从未开始整集生成的78集

用户明确切换优先级，并在异步回答中确认“沿用原整集4步配置”。这覆盖下方19:42的1277条40步自动执行授权：**动态队列已停，不自动恢复；此前28条完整替换版及所有历史媒体保留。** 停止时第一批8/1277条已全部生成、0条完成新评分，保存于`outputs/entitybench_original40_priority_remaining_20261109/cases/`，未作为新增增长替换；status已标记用户暂停。

新普通后台tmux `qwen140_never_rerun_v2_20261109`，入口 `pretest/qwen140_never_rerun_episodes_20261109.py run`；调度输出 `outputs/entitybench_never_rerun_episodes_v2_20261109/`，`plan.json`为唯一新范围，`inventory_at_launch.json`保留启动前证据。78集／1218shot均在冻结140快照中为original，原整集目录无shot目录、无rerun_manifest、未complete。其间允许此前独立shot实验；“未整集重跑”不等于从未生成过任何独立候选。run714只存在人物素材构建前缀，复用精确素材，尚无整集shot。另4个整集部分完成（run155、run53、run44、run237）本次不纳入；run445已有complete但未纳入11-06冻结快照，亦不重复。首次预检验证所有原冻结代码SHA、78/1218范围、v2编译器与run/tag绑定通过，API0。

沿用已经验证的v2_20261104 Agent提示词与身份绑定（非本轮运动强化词）、episode统一画风、全部B/C关键帧多人v7 3D、A自身真实上一尾帧。主视频 **Wan2.2-I2V-A14B LightX2V 4步／81帧16fps**；B首尾帧过渡仍为原有wan3.0-video API，本次未换本地首尾帧模型。3D正常门控跳过可沿用Control并记录，不改阈值，不取消3D，不做语义检查重试／自动降配／修复Agent。不修改旧冻结生成代码。新wrapper只选范围、串行调用原素材/episode/eval入口，原媒体仍位于`outputs/entitybench_qwen140_multiface_remaining126_20261030/experiments/<完整episode_id>/`。

最新启动阶段：run714/i20_j34（10shot）references模型已加载，正在准备/核验素材；尚无新shot出片，不把预检当完成。每集生成完整序列后跑官方strict/skip_vlm12项，VLM0；默认可见单脸＋定位纠错DINO仍需单独定位复核，不用raw代替。主体一致性与人物出现可能受路由改善，但不预设必涨。日志为新OUT/status.json、queue.log、stage_timings.jsonl以及各episode/logs/never_*。单卡串行，SSH可断开，机器须保持运行。初估78集约5～6天，首集完成后校准；约13GiB空余，全队列预计新增约60～65GiB，需扩容才能全跑完。不设磁盘reserve/人为额度、不自动删除；真实磁盘/认证/OOM等系统异常停队列记录，独立技术异常留档后继续其他集。旧整集大队列、动态实验都不另行启动。未推送。



更新：2026-11-09 22:11（北京时间）。仓库：`/root/autodl-tmp/movie`。

本主交接只保留两条主线：已完成的原始完整benchmark，以及当前用于后续重跑的多人3D集成方案。历史探索过程不纳入本文；原始媒体、指标、素材、Gaussian和历史记录均保留。

## 当前状态

- **2026-11-09 19:40启动：用户已明确要求其他动态0全部重跑，并补充先分析0→1、优先容易恢复，现为1277条按证据排序的原版40步后台队列。** 当前累计28版仍1305条0，其中28条此前相同40步条件已失败，精确结果复用，不重复同种子GPU；新1277条，优先30条当前0但已有同PNG/词/种子的4步动态1（视频、实际生成、评分receipt及源码SHA核验），后1247按原文可见运动线索排队，难例不删除。分析`docs/entitybench_original40_recovery_priority_20261109.md`及`outputs/entitybench_original40_recovery_priority_analysis_20261109/report.md`，实际排队只用`priority_manifest_v2.json`（无v2的同名文件为未执行草稿）。tmux`qwen140_original40_priority_remaining_20261109`，入口`pretest/qwen140_original40_priority_remaining_20261109.py run`，输出`outputs/entitybench_original40_priority_remaining_20261109/`；首批M0010/M0012/M0069/M0074/M0189/M0022/M0045/M0079，当前模型加载/首条实际进度看status和priority_batch_001_worker.log，尚无新完成视频。全部1333纯运动词已编写但未全部实际生成；本轮逐字使用冻结纯运动词、不加旧469背景句、不新增动作/运镜、首帧/3D/种子/81帧主段/16fps及原版40CFG3.5noLoRA/WanScheduler/shift5/.9/TorchRoPE固定。B15条保留原过渡且完整拼接后评分，代表B拼接回放128帧逐像素一致（含30fps→16fps正常采样）；完整140官方和默认DINO基线复现通过。每8条6VBench后保存progressive_replacement_manifest.json／candidate_media及稳定source/SHA的selection_checkpoints，动态0→1追加原28，其他质量不筛；其余完整上下文指标明确pending，全新视频完成后统一完整episode重算、官方140聚合和默认DINO复核，auto不能冒充最终。整体约14天、首批约2小时，技术错误保留输出停止，不自动改词/降配/删文件。**本条新授权覆盖旧“不跑全1333”范围限制；旧4步1333调度仍不恢复，整集也不恢复，无守护/付费API/VLM/推送。** 下方19:19“GPU空闲／未启动下一批”均为新授权前历史。

- **2026-11-09 19:19北京时间新增24条全部可用评测已完成，累计28条动态增长替换，后台退出／GPU空闲。** 输出`outputs/entitybench_original40_next24_20261109/`，24/24生成和6VBench，8/24动态0→1，16条保留旧视频；叠加旧20，独立140/2491版本28替换／2463不变，原媒体和全部负例不覆盖。新增8集／171完整shot上下文评分、可见单脸提取、6份隐藏分数裁剪的助手视觉复核（6保留／0新增剔除，57项旧排除按精确SHA沿用，不称外部人工签核）和140默认DINO重算完成，评分失败0，171×6缓存值／输入协议与SHA核验通过。最终`replacement/complete_report.md`和三版本`replacement/full140_three_versions_report.md`／comparison JSON；默认结果`replacement/visible_region/report.json`，完整审计`replacement/completion_integrity_audit.json`。selected20→累计28：DINO .7989→.7990，主体.9417→.9415，时间稳定.9864→.9864，平滑.9920→.9920，美学.5859→.5859，成像60.8571→60.8665，角色出现.9479→.9479，物体出现.8981→.8981，场景出现.6238→.6238，raw人物DINO .7239→.7240，raw物体DINO .8077→.8077，接缝.8452→.8448。动态精确.480048511430→.484003056885，本批+.395455个百分点；相对最初冻结.473352984735累计+1.065007个百分点。DINO130有效集／1467计分角色次不变，官方表其他项四位舍入。VLM0／partial_without_vlm／缺fidelity门控保留，下降不筛。旧1333与整集仍暂停，未启动下一批／推送／删除。文档`docs/entitybench_original40_next24_20261109.md`。下方12:43“正在跑”为历史，勿重复启动。

- **2026-11-09 12:43北京时间：用户看过selected20完整结果后要求继续跑此前动态0镜头并替换；已冻结并启动新增24条（N33–N56），不是恢复1333。** tmux`qwen140_original40_next24_20261109`，入口`pretest/qwen140_original40_next24_20261109.py run`，输出`outputs/entitybench_original40_next24_20261109/`，首worker正在加载原版40步模型，初始0/24生成不是已完成。24条／19episode，当前selected20基线动态0＋同词强化4步动态0，原32排除，优先原文明确位移；原PNG／3D／词／种子／81帧16fps／尺寸与40步CFG3.5noLoRA/WanScheduler/shift5/.9/TorchRoPE固定。24个4步视频与评分receipt精确SHA核验，363输入绑定；纯缓存完整140官方＋默认DINO基线复现通过，正式新视频未评。每8条生成后6VBench，全24后仅新增0→1叠加原20，独立`replacement/`完整2491映射，其他指标下降不筛、旧文件不覆盖。再自动重算新增成功的完整episode上下文、全140官方12项和可见单脸材料，`automatic_results`待逐项定位复核后`context`脚本final，不冒充最终DINO。按此前每条934秒约6小时20分生成，再加上下文与复核；12:43启动预计19点左右生成结束，以真实status为准。旧1333与整集仍暂停，无守护／自动恢复，无API/VLM／推送／删除。详见`docs/entitybench_original40_next24_20261109.md`；下方“下一批尚未冻结”已被本条覆盖。

- **2026-11-09 11:45北京时间selected20全部可用指标及默认DINO重算完成，后台退出，GPU空闲。** 输出`outputs/entitybench_original40_dynamic_gain_selected20_20261109/complete_report.md`／`complete_comparison.json`，官方12项另存official_report.md。18/18集、539/539完整上下文、18份6VBench缓存收据、全140/2491聚合与精确动态核验通过，评分失败0，VLM0。Codex逐项视觉复核20份隐藏分数裁剪（4页＋歧义原帧），20保留／0新增剔除，57项历史排除仅精确SHA沿用；这是助手视觉复核，未宣称外部人工签核。默认同协议DINO .7986→.7989，130有效集／1467计分角色次均不变；动态精确.473352984735→.480048511430（+.669553个百分点）；主体.9418→.9417，时间稳定.9865→.9864，平滑.9921→.9920，美学.5861→.5859，成像60.8533→60.8571，角色出现.9479→.9479，物体出现.8989→.8981，场景出现.6238→.6238，raw人物DINO .7239→.7239，raw物体DINO .8076→.8077，接缝.8470→.8452。官方表为原聚合函数4位舍入，不用舍入差冒充精确动态。decisions／visible_region/report.json／official_completion_validation.json／completion_validation.json已落盘，20条选择未按其他下降过滤，旧媒体不覆盖；partial_without_vlm和缺保真门控说明保留。原tmux已退出、GPU0%／约1MiB，盘约13GB可用，无删除／推送／新生成。下一批尚未冻结，1333与整集旧队列不自动恢复；下方11:17及11:11为运行期历史。

- **2026-11-09 11:17北京时间新会话已按实际状态接手，没有重复启动GPU任务。** 唯一相关tmux仍为`qwen140_selected20_context_20261109`，2worker同卡正常评分，实时10/18集完成（139/539shot；剩余8集各50shot），尚非完整指标完成。重新验证2491媒体SHA／链接及300冻结绑定错误0，20替换／2471不变；单张48GB4090使用约7GB，数据盘约14GB可用，无删除、旧队列恢复或推送。独立新增`pretest/qwen140_selected20_review_finalize_20261109.py`，`baseline`已实际复现11-07同协议扩展复核基线140逐集与全聚合（57精确crop排除、1467计分appearance、130有效集），`reviewed_dino_baseline_validation.json`留证。原scorer／计划未改；待原任务产出隐藏分数review材料，完成逐项定位复核并冻结decisions后，适配器`final`才重算默认DINO和完整13行表。final尚未运行，未创建守护／等待队列，auto仍不报最终。详细实际状态见`docs/entitybench_session_handoff_20261109.md`顶部。

- **2026-11-09 11:11北京时间最新交接与选择规则：用户只替换动态有增长的结果，已建selected20完整版本并启动上下文评分。** 相对冻结140当前benchmark，32尝试中20条0→1（不是17，17是相对同词强化4步），其余12保留原版；完整2491媒体映射20替换/2471同SHA不变，旧文件不覆盖。输出 `outputs/entitybench_original40_dynamic_gain_selected20_20261109/`，replacement_manifest.json/selection_validation.json/dynamic_exact.json齐全。动态按140集等权精确算术为.473352984735→.480048511430，+.669553个百分点，其他指标未报旧数。**不恢复多指标质量门控**，只按动态提升选择，所有下降披露。后台tmux `qwen140_selected20_context_20261109`、入口 `pretest/qwen140_original40_selected_gain_run_20261109.py`；18集/539完整shot、2worker/4threads，6VBench全按SHA复用，人物/物体/场景定位、中心/边界重算，结束官方全14012项＋默认DINO复核材料（automatic不是最终），无新视频/API。新交接 `docs/entitybench_session_handoff_20261109.md`，`docs/qwen140_new_window_prompt.md`已全面更新（过期前版归档docs/archive），10-31文档加历史警示。旧1333及整集保持暂停，不自行启动；用户允许后续小批量试验但新名单未冻结，先完成selected20评测。提示词继续运动强化版，两边逐字相同，未恢复原词、未添加背景；3D固定首帧不变。

- **2026-11-09 10:57北京时间核验：原版40步32条队列已于07:03全部生成＋6VBench完成，后台已退出，GPU空闲，旧1333/整集队列均保持暂停，不自动扩大。** 输出 `outputs/entitybench_original40_scope32_20261108/report.md`、comparison.json、completion_validation.json；32全部保留，无质量门控/负例过滤。原8按SHA复用＋24新增均完成；32视频/SHA、同PNG/词/种子/81帧16fps、4步参照与评分SHA、冻结源码和实际WanScheduler40/CFG3.5/noLoRA/TorchRoPE均通过，6均值逐条独立复算误差<1e-12。同词4步→40：主体.96817195→.95979163（-.00838032），时间稳定.98796775→.98463426（-.00333349），平滑.99294567→.99142533（-.00152034），动态.09375→.625（3/32→20/32；29旧0中17→1，3旧1全部保留，无1→0），美学.61790623→.61273145（-.00517478），成像62.95641799→63.04718744（+.09076946）。新增24动态0→13/24(.541667)；原8动态3/8→7/8。描述性分组：运镜4条0→3通过，位移10条1→8，持续表演8条1→6，操作10条1→3；强运动收益更明显，操作有限，不能把原粗类别当因果或身份指标。32均为目的性样本且共享运动强化词，非全benchmark/孤立步数蒸馏消融。人物/场景出现、默认身份DINO/真实接缝及全部32动作语义未评，不自动替换。用户此前撤销质量门控仍有效；要评这些需完整上下文，而不是用主体一致性顶替。主文档docs/entitybench_original40_scope32_20261108.md已更新最终状态。

- **2026-11-08 23:59北京时间用户确认先32条，已补排24条，32总量包含现有8条，不是额外32。** 此前异步范围问题现已由本轮确认解决，不跑全部1333。新普通依赖队列tmux `qwen140_original40_scope32_20261108`，入口 `pretest/qwen140_original40_scope32_20261108.py run`，输出 `outputs/entitybench_original40_scope32_20261108/`，文档 `docs/entitybench_original40_scope32_20261108.md`。等待当前8条生成＋评分结束，8条媒体/分数按SHA复用；新增24分3批每8条生成后6VBench，全部正负结果纳入，无质量门控。新增来源为2运镜、8位移、6持续表演、8具体操作，原描述已核对，4步同词动态0且无背景句、非B。全32均保留逐字运动强化词/PNG/3D/种子/帧数尺寸，纯换原版40CFG3.5noLoRA适配配方，不恢复原词不添加背景。29个同词4步动态0＋3个正参照。32输入/来源SHA/config/worker分批合同验证通过(validation.json)，模型仍由当前8条进程独占，无重复GPU作业。等待阶段新status.generated=0只表示尚未导入8条；实际当前完成数看original8_generated（最新5，N06生成）。预计11月9日07–08点完成32与6项评分，技术异常另计。后续报告三列含冻结benchmark/同词4步/同词40步，首列提示词可能不同，不混因果。原1333大队列/整集仍暂停、不自动恢复、不替换benchmark，无守护Agent或付费API。

- **2026-11-08 23:51北京时间用户明确本轮先不加指标质量门控：40步实验全部生成/全部评测，指标下降的shot也保留并计入汇总，不按旧动态0→1/.01等阈值剔除。** 当前8条执行器本来就没有这种质量过滤，已代码核对，无需改冻结生成源码或计划。记录 `outputs/entitybench_original40_vs_distill4_torch_rope_20261108/unfiltered_evaluation_policy.json`。旧1333提示词候选政策为历史，队列保持暂停，不用于本轮8条筛选；如后续扩展本实验，默认沿用无质量筛选、完整披露。只取消按评测分数筛选，不取消文件/SHA/帧数/评分有效性的技术检查，也不改3D注入门控（此处首帧固定、无新注入）。本轮未授权自动覆盖正式benchmark，保留独立无筛选实验表；下降不能隐藏。提示词仍保留运动强化版，与4步逐条完全相同。

- **2026-11-08 23:46北京时间用户澄清并确认：继续保留当前“增强运动”的提示词，不恢复11-06原词。** 这覆盖紧前一轮“放弃改词/恢复原词”的计划；新的原词批次从未启动，也未改变当前8条任务。核对8条均与其4步参照使用完全相同的已运动强化词（6条手写先导＋2条运镜规则追加），均没有背景追加句；当前比较只在原版40步/CFG3.5/noLoRA适配配方与4步蒸馏配方之间。不要宣称这些是原词实验，也不要重新改首帧/人物/3D/种子。用户暂停的1333候选大队列仍保持暂停，不因这次确认自动恢复；后续扩展32还是全部1333的范围尚未确认（此前异步范围问题仍待答）。

- **2026-11-08 23:33北京时间用户要求先评测：前3条40步对照已完成原6项VBench配对，2条动态0→1，成像均值+1.7923，其余均值小降。** 输出 `outputs/entitybench_original40_vs_distill4_torch_rope_20261108/completed3_evaluation/report.md`、comparison.json；新入口 `pretest/qwen140_original40_partial_eval_20261108.py`，3条全部纳入未选择，N01同SHA首条缓存复用、N02/N03新GPU评分22.14秒，峰值1.16GiB。均值4步→40步：主体.972196→.963738，时间稳定.976572→.970013，平滑.989246→.984539，动态0→.666667，美学.626676→.619778，成像69.123556→70.915810。逐条N01/run323/2:1=0→0，N02/run466/2:1=0→1，N03/run1/18:1=0→1。N02多项跌幅>.01不符此前质量筛选；N03可过原6项初筛但未算presence/DINO/最终画面资格，不替换benchmark。N02肉眼拉远明显，N03人物/背景相对移动更明显，抽帧N02_comparison.jpg/N03_comparison.jpg。短暂停N04进程101416评分，finally于23:33:07已恢复，原230大队列仍暂停、不得自动恢复。剩余5条40步继续生成，全部8条后自动统一6VBench；此3条表不是完整benchmark或独立步数/蒸馏消融。

- **2026-11-08 22:51北京时间：原版40步首条N01/run323/2:1已真实生成并完成6项即时评分，动态仍0，未显示运动优势；其余7条继续，大队列仍暂停。** 原版22:49完成，934.45秒、81帧/16fps/832×464，Torch RoPE兼容问题已解决；不是仅CPU预检。首条4步→40步：主体.972420→.973479，时间稳定.982426→.982804，平滑.992054→.992251，动态0→0，美学.628701→.627574，成像76.043637→76.353144。新输出 `outputs/entitybench_original40_vs_distill4_torch_rope_20261108/cases/N01/video.mp4`；即时评测 `first_case_evaluation/report.md`，原版/4步采样对比 `first_case_comparison.jpg`。评测入口 `pretest/qwen140_original40_first_case_eval_20261108.py`，独立原6VBench/相同模型阈值/无VLM，17.94秒、额外GPU峰值1.16GiB；为立即回应用户，短暂停N02推理进程101416，finally已恢复（resume receipt），没有重新采样或改变任务，未恢复用户暂停的大队列。N02/run466/2:1正在生成。首条抽帧主要转头抬头，横移幅度不明显；只据这一例不能断言蒸馏总体无影响。所有8条同词同首帧同种子，最终表仍等余7条，不覆盖benchmark。

- **22:31:56北京时间更新：大队列已正式暂停，230个生成结果全部保留、198个已初评、11个初筛通过，最后M0230已保存，GPU worker退出、父调度终止、context waiter关闭，automatic_resume=false。** 新Torch RoPE原版40步试验已启动并进入loading_original40_pipeline；不再等待旧批次。实际新结果看新目录status，尚未完成首条，不将CPU预检当GPU成功。

- **2026-11-08 22:30北京时间最新用户指令：大规模4步动态候选队列先停，只重试原版非蒸馏40步；不得自动恢复旧队列。** 父45449已STOP、空闲context waiter48699已TERM；worker93303仅完成M0199–M0230当前批次（最新228生成/198初评/11初筛），结束后由新执行器核对32结果、终止空闲父调度并标旧status=paused_by_user，全部媒体/评分保留。此前“试验结束恢复大队列”已撤销。首次40步于20:41因customJSON遗漏rope_type而默认用未安装FlashInfer，在第一条首步报ImportError，0输出；不冒称原版已经成功或OOM。新 `pretest/qwen140_original40_torch_rope_20261108.py schedule`、tmux `qwen140_original40_torch_rope_20261108`、输出 `outputs/entitybench_original40_vs_distill4_torch_rope_20261108/`，只显式设torch_complex_rope+double_precision_rope=True，与现4步API默认一致，其余40步/CFG3.5/noLoRA/WanScheduler/.9切换/8条PNG词种子参数不变；不修改冻结旧源/计划。真实WanSelfAttention Rope实例及q/k旋转CPU验证与直接复数旋转bit-exact，预检通过(preflight.json)，还需实际GPU首条。先N01/run323/2:1通过81帧再其余7条，随后自动6VBench；成功/失败都不恢复大队列，无守护Agent、无语义重试或降级。新文档 `docs/entitybench_original40_torch_rope_20261108.md`，原自动恢复文档已加撤销提示。预计8条2–4小时，以首条实测校准，旧20:17 ETA失效。

- **2026-11-08 20:19北京时间用户授权原版多步对照：已准备8条并启动后台等待当前批次边界。** tmux `qwen140_original40_test_20261108`，入口 `pretest/qwen140_original40_vs_distill4_20261108.py schedule`，输出 `outputs/entitybench_original40_vs_distill4_20261108/`，文档 `docs/entitybench_original40_vs_distill4_20261108.md`。现父调度45449 STOP是有意排程，GPU worker69726仍正常完成M0135–M0166；确认32结果齐全并退出释放模型内存后原版测试独占GPU，结束/失败finally恢复父调度，不能误当旧队列卡住。冻结M0007/M0008/M0001/M0002/M0006/M0003/M0004/M0005八条，5动态0＋3正参照；全部同PNG/提示词/种子/81帧16fps/尺寸/BF16/SDPA。原版无LoRA、WanScheduler40步、CFG3.5、shift5、boundary.9按timestep切换，空negative；4步已有视频与分数精确SHA复用。CPU配置及8条件预检通过(preflight.json)，但当前还没有原版GPU输出，不冒称试跑已通过。先N01真实81帧通过再其余7条；总约2–4小时，第一条实测后更新，原1333队列相应延后。8条原6项VBench自动汇总，5个旧动态0子集单列；不涉及新首帧/3D/Agent/付费API/后续shot/benchmark替换。原版/蒸馏是适配配方对照，不孤立归因步数或LoRA。无守护修复Agent、无自动改设置语义重试；错误保留并恢复原队列。

- **2026-11-08 20:05北京时间：用户要求立即计算背景组场景出现率，已完成冻结26shot/18episode的独立配对场景评分，生成已恢复。** 输出 `outputs/entitybench_all_zero_motion_background_20261108/scene_presence_completed_cohort_20261108/report.md`；入口 `pretest/qwen140_background_scene_presence_20261108.py`。不按VBench/动态结果选择，含请求时所有已完成26个背景增强shot，共27个调度场景。原规则shot均值0.01923077→0.28846154，调度实例1/27→8/27；7提升、19不变、0下降。子集episode等权0.02777778→0.375，但只包含目标shot，不是完整18集或全benchmark分数。该组是刻意选择的基线weak，不能与全140的0.6238直接比较；动作词与背景词同时变，不是背景句的独立消融。7改善中M0076/run845/9:1已过VBench，M0080/M0088/M0091/M0112/M0127未过，M0140尚未VBench评分；没有最终替换资格。调用原5帧采样+unmodified ground_shot_unified(places-only)+原CLIP/.20门槛，旧M0076回放代表采样/裁剪SHA/状态一致，CLIP误差0；无VLM/API/新媒体/跨镜头指标或分数覆盖。为不等整批，在当前shot保存后暂停worker69726，单独评分45.3秒，额外GPU峰值2.75GiB，finally于20:05:41 SIGCONT恢复，resume receipt已核验；前后裁剪对比在improved_canonical_comparison.jpg，原生成plan及源码SHA未改。后台整批仍继续，完整出现率/DINO/画面/联合序列复核依旧未完成。

- **2026-11-08 17:31（北京时间）用户要求这批动态重跑同时补强背景，已启动版本化批次边界接续。** 原父调度29497已STOP但原GPU worker40556继续完成M0039–M0070，未中断视频；旧空闲context waiter已停。tmux `qwen140_allzero_background_20261108`、入口 `pretest/qwen140_all_zero_background_20261108.py handoff-run`在确认70项完成后切换，不重做已生成/在途shot。新输出 `outputs/entitybench_all_zero_motion_background_20261108/`、文档 `docs/entitybench_all_zero_motion_background_20261108.md`。同1333目标，前70复用；剩余1263中469个基线场景weak且有具体数据集描述者追加短背景上下文，794仅原动作词；原首帧/3D/种子/模型/参数/过渡/其他shot不变、无API。1333任务contract、469仅prompt追加、冻结生成源码SHA及新context路由CPU验证通过；8例首帧+词抽查在 `prompt_input_preview.jpg`，不能承诺首帧缺背景者靠视频词修复。原plan/视频不覆盖，草拟计划在draft_plan_preexecution单独留档，最终plan已冻结。用户既定动态0→1及所有质量/出现率门槛不变，人工复核/联合序列重算后才发布筛选版；无守护Agent、不重启整集、不放宽阈值。新普通后续评分tmux在边界交接后自动启动；读新目录status，不将旧目录最后生成记录误当当前。预计仍11月10日前后生成＋初评，完整语境/人工另计。

- **2026-11-08用户扩大授权：全部动态0的1333个shot各生成一个固定首帧候选，质量筛选后版本化替换；已启动后台。** 覆盖下方旧6例/旧暂停范围，但不授权重新启动整集队列。用户明确确认门槛：动态0→1，人物/物体/场景出现率不降；默认人物DINO、主体、时间稳定、平滑、美学各最多下降.01，成像最多降1分；无新增明显漏人/身份漂移/分屏/动作错误。**最终符合门槛的替换已获授权，无需再次索要批准；先完成完整语境评分及复核，不得把VBench初筛当最终通过。** 主文档`docs/entitybench_all_zero_motion_rerun_20261108.md`，输出`outputs/entitybench_all_zero_fixed_frame_motion_20261108/`。基线仍固定11-06的140集2491shot、57完整重跑＋83原版＋141独立v2；动态0为702原版/602C/15B/14A，共140集。6先导候选直接复用，新增1327；前6按新门槛均不通过（3个动态1样本主体降.0293/.0191/.0393），保留不替换。新`pretest/qwen140_all_zero_motion_20261108.py`已冻结1333名单/每条原新词/实际I2V首帧/种子/模型/源SHA和`selection_policy.json`，规则根据原动作段补简短过程句（非1333条人工手写，也不调用Agent/API），不为静态死者或细微动作编造走动/运镜；`prompts.jsonl`逐项公开。702原版从原generation.json取input_image；15B原过渡保留、只改主段再原样拼接，无新首尾帧API。3D结果通过完全相同输入PNG继承，不重做人物/首帧/素材、不重跑其他shot或episode。持久Wan worker两例162帧逐像素/视频SHA与先导原worker相同，105.69秒模型加载、两例总286.88秒；验证重放2次是相同条件一致性检查，不是新采样候选。生成tmux`qwen140_allzero_motion_20261108`，每32例共用模型后释放GPU，原6项VBench评新视频、验证缓存复用旧分数；16:06已7/1333完成（6复用＋1新），初评6/1333，首批正在M0008 run466/2:1。后续普通依赖任务tmux`qwen140_allzero_context_20261108`、入口`pretest/qwen140_all_zero_context_20261108.py wait-run`，等候选/初评全完成后按既有双路缓存评测执行完整受影响episode的presence/人物物体中心/真实边界，再提可见单脸与定位/动作复核清单，自动阶段明确停在pending_human_review。该context adapter目前仅导入/契约检查通过，完整候选尚未生成因此未假称已全量运行验证；若异常保留日志，不借旧PID恢复脚本。复核完成后再按同门槛筛选、重算实际选中组合的中心/边界，必要时单调剔除不符合项重算，最后单独发布筛选版全140表，原媒体/官方报告不覆盖。须披露1333总尝试和全部负例/拒绝原因，不能拿筛选涨分当无筛选方法改进。其余VLM仍关闭、API0、无守护修复Agent，无自动清理/额度上限；约13GiB空余。生成＋初评估36–48小时，约11-10 04:00–16:00北京；完整语境评分/人工复核另计，实时读status/log，非完成承诺。未推送。

- **2026-11-08 15:39固定首帧提示词强化6例已全部生成并完成6项VBench，耗时20分31秒；后台已退出，GPU空闲。** 本条覆盖下一条的运行中状态。输出`outputs/entitybench_fixed_frame_motion_prompt_20261108/report.md`、`comparison.json`、`vbench_receipt.json`；6个新视频位于`P01`～`P06/main/video.mp4`。动态0→1为3/6：P02 run103/i30_j54/24:2走路、P03 run221/i0_j18/16:5双节棍、P05 run146/i0_j46/7:1抬手；P01 run1/18:1、P04 run4/i80_j98/7:6、P06 run264/1:6仍0。目标6shot等权均值：主体.967394→.942741、时间稳定.994037→.991283、运动平滑.995484→.994273、动态0→.5、美学.615403→.604721、成像63.658651→62.967766。只说明这6个预选目标的prompt-only试验，其他质量项下降，不能直接认定新方案全面更好；逐视频动作人工验收尚未完成。人物出现率、跨镜头人物DINO/边界未评，本次不生成整个episode、不替换benchmark、不扩队列。复查6个输入首帧、6旧视频和6新视频SHA通过；保留原3D首帧/种子/模型/步数，API0。用户当前只询问状态，未授权新扩跑或替换。

- **2026-11-08 15:19用户明确授权固定首帧的独立shot动作提示词强化，已后台启动6例；原整集队列仍停止。** 用户澄清：“只利用动态程度为0的shot的首帧重跑一下这个shot就行了（强化一下提示词），不需要重跑整个episode”。新入口`pretest/qwen140_fixed_frame_motion_20261108.py`，输出`outputs/entitybench_fixed_frame_motion_prompt_20261108/`，tmux`qwen140_motion_prompt_20261108`，运行状态以该目录`status.json`及`queue.log`/`P##/generation.log`为准。预先固定6个当前v2动态0：P01 run1/i50_j84/18:1走路，P02 run103/i30_j54/24:2走廊走路，P03 run221/i0_j18/16:5双节棍，P04 run4/i80_j98/7:6开消防柜门，P05 run146/i0_j46/7:1抬手准备击打（不擅加击打），P06 run264/i50_j98/1:6扫弦演唱。已目视核对6张原实际输入图；源任务所有非prompt字段逐项相同，仅替换简短动作句，保留角色描述、画风、原3D首帧PNG/SHA、种子、模型、4步81帧和原机位要求。身份素材/Agent/首帧/过渡不重做，API调用0；只在新目录写候选，不修改后续shot、不启动整个episode、不自动替换benchmark。完成6视频后单GPU按原官方6项VBench评分12个新旧视频，输出`comparison.json`、`report.md`；是目标shot等权，不是全benchmark，也不含角色出现率/跨镜头人物中心或拼接边界，不能宣称其他所有指标不受影响。新尾帧未来若替换可能影响真实拼接质量，即便后续视频文件没变。预计15–25分钟（约15:34–15:44北京时间），以实跑日志为准；每例旧视频生成约78–90秒另有加载/评估时间。后台为普通tmux任务，不是守护修复Agent；技术异常留档停止，不自动换方案、扩范围或重试付费接口。所有新旧提示词见`prompts.md`，输入及代码SHA冻结于`plan.json`，不要原地改冻结脚本后续跑。

- **2026-11-08 GPU恢复后已完成本轮人物错定位的补充评估；没有恢复生成或守护Agent。** 输出`outputs/entitybench_localization_repair_gpu_20261108/report.md`、`comparison.json`、`coverage_and_validation.json`、`role_changes.csv`、`episode_changes.csv`；入口`pretest/qwen140_localization_repair_gpu_20261108.py`（prepare/extract/sheets/freeze/score/validate/finish）。继续冻结11-06的140集2491shot、57完整重跑＋83原版＋141独立shot，不纳入第58集。此前63确认错框去重61键＋5下降案例，补原/当前对应输入，66键118去重输入、33集887完整上下文。全部候选隐藏分数/版本复核并先冻结：74份可重新定位（映射82条版本角色记录）、28份对应原框正确、16份缺失/混淆/无可用框保持未决。使用原描述＋补充person候选，在人工批准的目标区域集合内按原selection取代表；保持原采样时间、FP32 CLIP/DINO及.20阈值，不凭肉眼改present。重新编码2970原canonical＋847候选embedding；82个变框版本shot全部角色重做单脸/跨时空间track冲突门控。118旧CLIP复现误差0，66份逐集raw/默认DINO误差最大4.87e-9/6.64e-8；203条旧门控GPU重放差异0。**新框是独立补充诊断，不能覆盖原始官方指标或原像素默认DINO。** 当前补充出现率.947937820→.947933852（四舍五入仍.9479）；未加单脸区域DINO .723891101→.724136197；可见单脸条件DINO .798598377→.798077019。原版同步补充为出现.921238315→.920330417、区域DINO .715525911→.715550433、单脸DINO .788729967→.788467252。当前Meilin(run654/5:3)错Arthur框及Min-jun(run1/5:1)错Sung-min框修复后weak→present；Julian共享旧框修正后present→weak，其他原版也有负变化，全部保留。当前单脸复核保留1780→1806、计分1467→1492，原版1663→1681、1345→1364；两侧仍130有效集、有效集ID无增减，人物集合/中心仍变。官方其余指标各难度聚合完全不变，原官方.7239及默认扩展.7986继续保留，不能只拿涨分值替换。固定样本缺失/背身/身份漂移/重复与未决项未强修；988track冲突大多仍原拒绝，物体/场景未冒称全语义修复。GPU约48GiB4090；候选约106秒、门控约28秒、原区域embedding约40秒，另有人审/加载时间；API/VLM/生成调用0，GPU作业已结束，未推送。

- **2026-11-07用户选择“先用缓存完成排查”，已完成扩展人物定位审计与同口径重算，生成仍停止。** 报告`outputs/entitybench_cached_matching_audit_20261107/report.md`、`comparison.json`、`confirmed_localization_errors.csv`、`decisions.json`、`final_validation.json`；入口`pretest/qwen140_cached_matching_audit_20261107.py`。仍绑定11-06 12:37北京时间的57完整重跑＋83原版＋141独立shot快照，不混入第58集。原/新全140集2491shot、22,874逻辑人物/物体/场景记录做缓存/门槛检查，16,650去重项无文件/状态错误；全部2,673去重可见单脸裁剪隐藏分数初筛，206项原文＋精确完整帧确认，63份确认错定位（含已有决定及两版共享输入）。冻结后与历史精确SHA排除取并集，对原/新对称应用：原版排除3→37，当前35→57。**扩展复核DINO原版.7887、当前.7986；历史抽样复核.7863／.7970原样保留，不能混用。** 两侧有效集131→130（run893/i5_j15的Nadia错记者框去除后无足够跨镜头样本），固定130有效集前后原版.786370→.788730、当前.797236→.798598；条件人群/中心仍变化，不是视频身份改善。原/新280逐集历史分数及原始Gram中心均复现；官方12项完全不改，角色出现仍.9479，raw cs_face仍.7239。406份weak人物裁剪初筛；988项face-track冲突保持原拒绝，仅192可视预览；物体/场景仅全量技术检查，未冒充全部语义人工复核。身份/服装/性别漂移、目标缺失、重复及无法判明角色者保留；不能按肉眼把weak写成present。无GPU、0.5CPU/2GiB，只用缓存算术及抽帧，无新模型/embedding/API/VLM/媒体生成。另存227组重复像素多实体引用候选，未自动改分。后续可用正确新框补CLIP/DINO，但本轮未启动，也未恢复生成或守护Agent；未推送。

- **2026-11-07已定位局部v2角色出现率下降的具体来源，不能笼统解释为漏人。** 审计`outputs/entitybench_v2_presence_change_audit_20261107/report.md`、changed_roles.json、five_declines_review.jpg、anya_all5.jpg、full43_context_validation.json。全43集899shot仅7个目标presence计数变化，758未替换shot完全不变；5降2升，5降全部present→weak，无新增absent，均C路线且frame_characters/视频文字保留目标角色。141目标276角色次present273→270、weak3→6；目标shot均值.9917→.9811、43集宏平均.9899→.9868，分母不同。逐图：Anya变装前在场、代表框误到红头巾Javier（.2268→.1998）；Marcus画面在但匹配.2180→.1817；Meilin红袍在右、旧新代表框都主要Arthur（.2100→.1757）；run1/1:1 Min-jun黑衣在场、框正确但.2273→.1994略不过.20；run1/5:1 Min-jun桌边在场、旧新代表框都到蓝牛仔Sung-min（.2419→.1963）。因此旧高分也含错误定位碰巧过线，分数不等于真正出现/消失。另有两项Jing、Seo-joon改善；新增33出现率1→1。保留官方原presence，不调阈值/手工改分；原DINO定位复核不等于presence纠错。v2图片仍有重复/分屏/构图等问题，不宣称所有画面正确。此次只读旧评分/CPU抽帧，无新生成/神经评分/VLM，生成继续停止。

- **2026-11-07用户纠正后已补齐正确的整集对比口径：当时明确只重跑v1中低动态目标shot，后续shot保留，不要求纯v2整集。** 之前“没有整集对照”的说法过于绝对；实际已有43个相同episode／899shot的完整前后评分，可比较v1完整序列与替换141个v2目标、保留758个其他视频的序列。现已仅用已有评分做官方episode宏平均，输出`outputs/entitybench_v2_43_episode_splice_comparison_20261107/report.md`、comparison/aggregates/reviewed_dino_aggregates/bindings.json。动态.4299→.4733、默认DINO .7759→.7788（两侧43有效集，条件角色人群仍不同）、主体.9494→.9485、时间稳定.9866→.9864、平滑.9921→.9920、美学.5931→.5920、成像63.9349→63.6924、角色出现.9899→.9868、物体出现.9066→.9042、场景.5963→.6037、raw cs_face .7218→.7217、cs_object .7933→.7909、边界.6619→.6420。原/新完整上下文人物/物体中心和真实接缝均已由既有评测覆盖，此次未生成媒体、无新神经评分/VLM。141目标shot等权表和该43集episode宏平均表回答不同问题，不混用。无需为了已有实验对比重跑后续shot，也不能把此表称为每shot均v2或固定首帧prompt-only消融。生成继续停止。

- **2026-11-07版本口径核对：不能把新增6集整体当成统一v2整集实验。** `outputs/entitybench_full140_latest57_v2_141_20261106/planner_version_audit_20261107.json`按实际`agent/planning_inputs.json`系统文本逐镜头比对：143shot中101精确v2、42其他保留计划；run722 10/10、run801 11/11、run1517 12/12为v2，run2067 9/10、run4/i130_j148 10/50、run238 49/50为v2。原队列明确保留完成前缀/已有关键帧，不能仅凭完成时间贴版本标签。现有同镜头版本对照仍是`outputs/entitybench_v2_shots_only_comparison_20261105/report.md`的141对v1/v2，低动态目标预筛选，允许重新规划/生成关键帧，不是固定首帧的prompt-only消融；动态0→.2340，默认DINO独立有效群体.7396→.7462，但共同74目标.7502→.7383，不能宣称身份改善。尚无同一批完整episode的纯v1对纯v2配对。用户把未来强化视频动作提示词作为展望，未授权新实验；生成保持停止。

- **已按用户要求抽查“原接续／当前C”407个镜头的所有可用指标，生成继续停止。** 输出`outputs/entitybench_continuation_route_c_407_20261106/report.md`、`comparison.json`、`shots.csv`、`inspection_notes.md`；入口`pretest/qwen140_continuation_route_c_metrics_20261106.py`。与上文动态统计完全相同407shot／56完整上下文，前=原始完整benchmark、后=12:37冻结57集＋141独立替换快照；目标shot等权，不是全benchmark宏平均或单独路由消融。12项全407缓存汇总：默认DINO .7343→.7506（193/216有效；共同121 .7372→.7521），主体.8933→.9628、稳定.9829→.9894、平滑.9907→.9935、动态.7322→.3833、美学.5654→.5913、成像53.3135→61.8377、角色出现.8744→.9864、物体出现.8570→.9000、场景.6454→.5543、raw cs_face .6714→.6903、cs_object .7793→.7526。边界缺逐边缓存且本实例无GPU、仅0.5核/2GiB，遵循“稍微抽查”预先均匀随机16入边（seed20261106），CPU原FP32算法补算，**.9770→.4716，只代表16样本，不冒充407全量**。两个联调样本与既有GPU数值差0、5.96e-8，不入随机均值；保存18条实际CPU结果和16样本manifest。56集原/新人物中心逐集复现已发表分数，1486评分缓存文件SHA留档，VLM调用0。抽帧run238/i0_j34/13:3当前更清楚且人物接近镜头，但仍有重复角色；run560/i100_j120/4:2更接近过肩构图却更暗，成像70.91→35.73，不能只凭主体稳定认定更好。run264/11:7当前仍拍到大幅面部，手指/琴弦极特写执行不足。当前旧pause PID已不存在，未恢复或启动生成；不得据旧running状态重启队列。本任务新增代码/文档未推送。

- **用户最新要求先不要跑，已于2026-11-06 12:48（北京时间）暂停整个生成队列，不能自动恢复。** 父510205、当前run929/i15_j65素材worker4777均已SIGSTOP核验T状态，保留内存/已完成输出；恢复必须用户指示并核对`outputs/entitybench_v2_remaining_episodes_20261105/user_pause_20261106.json`中的PID/start_ticks。普通status.json保留暂停前stage，不能据此误认为仍在运行；run445暂停前已经完成，整集重跑实际增至58，但下方全量表冻结于57集，不混入后来结果。动态诊断见`outputs/entitybench_dynamic_degree_analysis_20261106/report.md`及summary/shots/episodes.json：全2491原1225动态1→现1158，119个0→1、186个1→0；141独立v2覆盖使同一57集快照宏平均从.4600拉回.4734。当前C/原硬切547shot .2523→.3583，C/原接续407shot .7322→.3833；175个C降0中153个原接续。抽帧表明部分原接续运动来自换构图/画面变换（run264/i50_j98/11:7从其他人物转吉他），C从目标构图开始会消除这部分光流；另有真实走动不足/首帧已就位（run227/i25_j77/4:1、run1/i50_j84/18:1）。最后33 P1位移/运镜6/13通过、P2具体操作2/19通过。仅分析与CPU抽帧，未改代码/提示/门控、未新推理、未请求收费API，守护Agent仍关闭。

- **最新完整benchmark已汇总并完成默认DINO定位复核，覆盖140集／2491shot。** 快照冻结于2026-11-06 12:37（北京时间），输出`outputs/entitybench_full140_latest57_v2_141_20261106/report.md`、`comparison.json`、`coverage_and_final_validation.json`。纳入57集完整重跑、83集保留原版，以及此前已完成的141个独立v2镜头替换（包含新增33个）；未完成episode前缀不纳入。这取代旧51集＋108镜头的全量快照，不是140集均已统一v2重跑。原版→当前：默认可见单脸＋定位纠错人物DINO **.7863→.7970**，主体.9362→.9418、时间稳定.9858→.9865、运动平滑.9917→.9921、动态.4853→.4734、美学.5817→.5861、成像59.6517→60.8533、角色出现.9212→.9479、物体出现.8942→.8989、场景出现.6303→.6238、raw cs_face .7155→.7239、cs_object .8143→.8076、边界.9822→.8470。官方12项复用兼容逐集结果并核验全部2491视频SHA，无新官方神经评分；新增6集143shot用CPU补RetinaFace门控，不暂停生成。80项隐藏分数定位样本逐项复核70保留、10排除，决策绑定manifest和crop SHA；服装/身份/画风漂移、重复生成及无法分辨的关联保留。默认DINO原/新均131有效集；4991计划角色次、4734 present、1837自动合格、1802复核后保留、1490形成跨镜头计分；条件计分集合变化，不能凭均值上升断言全部身份均提升。VLM调用0，依赖VLM的其余指标仍未评。后台`qwen140_v2_remaining_20261105`正常，6/89新集complete，run445/i25_j35已经15shot生成完成、正在逐集评分；守护Agent关闭。此次新增汇总代码尚未推送。

- **141个已完成独立v2镜头对各自v1的全部可用指标评估已完成，生成已恢复。** 输出`outputs/entitybench_v2_shots_only_comparison_20261105/`，主`report.md`、`comparison.json`、`shots.csv`、`final_validation.json`。这是**目标shot等权均值，不是完整benchmark分数**；另列新增33与此前108。43集899上下文，32整集复用，11集515shot重算；482视频复用6项VBench、33新推理；482未变视频的2074代表裁剪SHA及942角色次的可见单脸门控与旧输入完全相同。官方43上下文零失败、VLM0；39个隐藏分数新定位样本真实逐项复核，37保留、2排除（N009 run98/33:1 Alex误框前景短发眼镜灰衫，真正剃头灰制服Alex在背景；N033 run1/16:1 Min-jun误框蓝牛仔背包Sung-min，黑衣Min-jun在右）。Chul泛化描述歧义、Simon动作/条纹衣身份交换、Walter皮草/开衫冲突保留。默认条件DINO .7396→.7462，88/88有效但集合不同；**共同74镜头 .7502→.7383下降，不能因条件均值上升认定身份更好**。动态0→.2340（33/141过阈值）、成像63.2500→62.6089、角色出现.9917→.9811、场景.5257→.5515、raw cs_face .7170→.7112、边界.5713→.5490。新增33默认DINO .6923→.6675（20/18），共同16 .7032→.6607；动态0→.2727、角色出现保持1。完整对照见报告。原report auto在复核页导出时缺scored_crops索引，**仅导出失败，推理均保留**；通过`pretest/qwen140_v2_shots_report_resume_20261105.py auto/final`补齐原调用契约完成，`report_resume_contract.json`留来源，未改评分或重做推理。评估finally已核对start_ticks恢复607999/510205，run238继续，守护Agent仍关闭。未再次推送此任务新增代码。

- **2026-11-05晚用户最新请求：先评全部已完成v2独立shot对比对应v1，范围确认141个，不做完整benchmark汇总。** 当前独立评估`tmux qwen140_v2_shot_eval_20261105`，新入口`pretest/qwen140_v2_shots_only_eval_20261105.py run`，输出`outputs/entitybench_v2_shots_only_comparison_20261105/`。43集899上下文shot只供中心/边界；其中32整集兼容结果复用，仅11集515shot重算、482视频复用6项VBench、33新视频原算法推理，两路GPU各4CPU线程。141目标含此前108及新33，分组表均只算目标shot等权均值；不以整集均值冒充。43集旧reviewedDINO基线已按完整中心逐集复现。已在run238视频worker边界暂停生成父510205/607999（记录PID/start_ticks），未杀worker；评估入口finally自动恢复，后续普通report waiter`qwen140_v2_shot_report_20261105`运行`pretest/qwen140_v2_shots_only_report_20261105.py auto`，会生成隐藏分数的新增定位复核图。**必须真实完成复核并保存绑定manifest的decisions后执行report脚本final，才能报默认DINO完整表。** 用户没有要求此任务再次推送代码。评估异常先看status/evaluation.log；恢复生成前核对`generation_resumed.json`，不得重复启动GPU作业。

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
