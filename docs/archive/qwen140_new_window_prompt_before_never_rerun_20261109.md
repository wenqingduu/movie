# 新会话接手提示词（2026-11-09）

复制下面内容。先查实际状态，不按旧交接恢复队列。

```text
最新覆盖（2026-11-09 21:53北京时间）：用户已要求暂停实验，先更新已完成benchmark，后续收到明确要求才恢复。禁止自动恢复动态40步、旧4步或整集生成。原1277队列只完成8条，已补齐8条六项VBench，7条0→1（除M0010），累计35条替换/2456不变，剩1269未生成；GPU当前仅做用户授权的benchmark收尾，不生成媒体。独立入口pretest/qwen140_original40_paused8_benchmark_20261109.py（无生成action），输出outputs/entitybench_original40_priority_remaining_20261109/paused8_benchmark_20261109/。7完整episode/161shot上下文重算及默认可见单脸DINO定位复核进行中；动态精确.484003056885→.487254139136，其他新指标未完成前pending。查实际status与complete_report.md/visible_region/report.json，不拿28版旧值冒充35版。原根status.json为paused_by_user_benchmark_update_only，automatic_resume=false；记录及所有媒体保留，无VLM/付费API/删除/推送。下方19:42运行授权与状态已被此次暂停覆盖，旧暂停记录的其他任务reason不作为新生成授权。

请接手 /root/autodl-tmp/movie，使用中文。先读 docs/entitybench_session_handoff_20261109.md 和 LLM_HANDOFF.md 顶部最新记录，再查git status、tmux、进程、GPU、磁盘。保留无关改动和历史媒体，别重复启动GPU任务或自行推送。

最新19:42：用户已要求其他动态0全部40步重跑，随后要求先分析0→1、优先容易恢复。新tmux qwen140_original40_priority_remaining_20261109 于19:40启动，入口pretest/qwen140_original40_priority_remaining_20261109.py run，输出outputs/entitybench_original40_priority_remaining_20261109/。当前1305条动态0，其中28条已同40步条件失败，精确复用不重复；新1277条。最先30条有严格核验的同PNG/强化词/种子的历史4步动态1但当前benchmark0，后1247按可见运动原文线索排序，所有难例保留。历史56粗类运镜3/4、持续表演4/6、travel14/32、prop1/8、先导6/6；不是随机成功率，动态0不等于没动，1不等于语义/身份正确。分析docs/entitybench_original40_recovery_priority_20261109.md，排序唯一来源outputs/entitybench_original40_recovery_priority_analysis_20261109/priority_manifest_v2.json。首批M0010/M0012/M0069/M0074/M0189/M0022/M0045/M0079。全部1333纯运动词已写好但未全部实际生成，沿用冻结纯运动词不加旧背景句，不改PNG/3D/种子/81帧主段16fps/原40CFG3.5noLoRA/TorchRoPE配方。每8条6VBench后只0→1追加原28，最新progressive_replacement_manifest.json/candidate_media与稳定source/SHA历史selection_checkpoints分开，其他指标下降不筛。完整上下文/140指标在全部新视频完成后统一重算；进度表这些项pending、不拿旧数冒充。B15保留过渡、只重生主段、完整拼接评分，拼接逐像素回放与完整基线官方/DINO回放验证通过（validation/）。首批约2小时、整体约两周，无自动改词/降配/删文件/付费API/VLM/推送/守护。新全范围授权覆盖下方“不跑全1333”的旧范围限制，但旧4步1333调度、整集队列仍保持暂停。先看真实status，不重复启动；下方GPU空闲/无下一批为历史。

最新19:19状态：新增24条已全部完成生成、6VBench、8集/171完整shot上下文评分、可见单脸提取及默认DINO重算，8/24动态0→1，累计56次尝试/28条替换，独立140/2491版本在outputs/entitybench_original40_next24_20261109/replacement/，28替换/2463原基线不变，旧20及全部媒体/负例保留。tmux/worker全部退出、GPU空闲，勿按下方12:43历史重复启动。完整13行表complete_report.md；冻结替换前/selected20/累计28三版本表full140_three_versions_report.md及comparison JSON；默认最终DINO为visible_region/report.json。助手逐项视觉复核6份新裁剪（1页＋4个精确完整帧），6保留/0新增剔除，57历史排除仅精确SHA沿用，不宣称外部人工签核。171×6缓存分数、协议/输入SHA、全部评分失败0已验证，审计completion_integrity_audit.json。selected20→累计28动态精确.480048511430→.484003056885（本批+.395455个百分点），相对最初冻结.473352984735累计+1.065007个百分点；默认DINO .7989→.7990（130有效集/1467角色次不变），主体.9417→.9415、接缝.8452→.8448，其他值详完整报告。VLM0、保真门控缺失/partial_without_vlm仍保留，所有下降不筛；旧1333及整集仍暂停，未自行启动下一批。最新完整交接见docs/entitybench_session_handoff_20261109.md顶部。下方12:43和11:45均为历史。

最新12:43状态：用户确认selected20结果后要求继续跑此前动态0镜头并替换，已独立冻结新增24条N33–N56（排除原32，24/24当前selected20基线动态0且同词4步动态0，19episode，优先原文明确位移）。tmux qwen140_original40_next24_20261109，入口pretest/qwen140_original40_next24_20261109.py run，输出outputs/entitybench_original40_next24_20261109/。363输入绑定、4步真实生成与评分SHA及纯缓存140官方/DINO基线复现通过；首worker已启动加载原版模型，尚未首条出片，不重复启动。每8条生成后原6VBench，全24后只新增0→1叠加原20，完整映射与全上下文评分在replacement/，其余质量下降不筛，负例和原媒体保留。baseline_snapshot.json为已完成selected20完整140版本；按新增成功所在完整episode重算定位/presence/中心/真实边界，其余集复用。可见单脸材料仍需逐项视觉复核后运行pretest/qwen140_original40_next24_context_20261109.py final，auto不是最终。预计生成约6小时20分（12:43启动），再加上下文与定位复核。详见docs/entitybench_original40_next24_20261109.md。旧1333和整集不恢复，无守护/自动改词/降配/删除/推送。下方“下一批未冻结”已由本段覆盖。

最新决策：保留为增强运动而强化过的视频提示词，不恢复11-06原词；同一shot的4步和40步逐字同词、同实际首帧/3D/种子/81帧16fps/尺寸。原版40步WanScheduler、CFG3.5、noLoRA、shift5、boundary.9、TorchComplexRoPE，已修正首次遗漏RoPE导致FlashInfer缺失的错误。不通过减步数/关CFG/改词绕过异常。

32条配对实验已在11月9日07:03全部生成和6VBench完成，无质量过滤；输出 outputs/entitybench_original40_scope32_20261108/。同词4步→40步：动态.09375→.625（17/29个0变1，原3个1均保留），主体.968172→.959792、时间稳定.987968→.984634、平滑.992946→.991425、美学.617906→.612731、成像62.956418→63.047187。不是全benchmark，也不是独立步数/LoRA消融。

用户最新要求只替换“动态有增长”的结果，不恢复原来的多指标质量门控。以当前完整140冻结benchmark为基准共有20条0→1（同词4步比较才是17条）；独立版本 outputs/entitybench_original40_dynamic_gain_selected20_20261109/replacement_manifest.json，完整2491媒体链接在candidate_media，20替换/2471不变，12个无增长保留原视频，所有尝试/下降保留。原始媒体未覆盖。按140episode宏平均，动态精确.473352984735→.480048511430，+.669553个百分点，已与完整上下文重算结果核验。

11月9日11:45已完成selected20全部可用指标，原tmux qwen140_selected20_context_20261109及worker已退出，GPU空闲；不要因旧启动信息重复运行。18受影响episode/539shot已评分，六项VBench精确SHA缓存复用，其余定位/presence/中心/真实边界完整重算，另122集复用并官方聚合140。complete_report.md／complete_comparison.json为13行最终表，official_report.md为官方12项。Codex逐项视觉复核20份新裁剪（4页，歧义另看原代表帧），20保留/0新增剔除，旧57项排除仅精确SHA沿用；这是助手视觉复核，不称外部人工签核。decisions绑定manifest SHA后由pretest/qwen140_selected20_review_finalize_20261109.py重算默认DINO，结果visible_region/report.json；automatic_results仍仅为中间结果。官方完整性、媒体和动态精度验证均已落盘。

完整140前后：默认同协议人物DINO .7986→.7989（130有效集/1467计分角色次），主体.9418→.9417、时间稳定.9865→.9864、平滑.9921→.9920、动态官方舍入.4734→.4800、美学.5861→.5859、成像60.8533→60.8571、角色出现.9479→.9479、物体出现.8989→.8981、场景出现.6238→.6238、raw人物DINO .7239→.7239、raw物体DINO .8076→.8077、接续边界.8470→.8452。官方函数均值4位舍入，动态精确增加+.669553个百分点，不从舍入表作差冒充精确值。下降照报，20条选择不撤销。

评测用 /root/.codex/skills/evaluate-face-injection-v2/SKILL.md 及visible_single_face_region_dino协议。默认可见单脸＋确认定位错误剔除的原人物区域DINO，保留raw cs_face（非人脸识别）；新裁剪排除只按角色/shot/原crop SHA验证，不能删除真实漂移/重复/未决以提分。当前strict/skip_vlm、VLM0，fidelity gate缺失说明保留。用户取消质量筛选不等于取消评测正确性；presence/DINO下降照报，不再据此删掉20个已选媒体。

旧1333提示词候选队列保持暂停：230生成/198评分，旧11初筛只是历史；整集队列也不恢复，无守护修复Agent。20条替换版评测及交接已完成，最新授权的独立新增24条队列已启动（见顶部），不是恢复旧大队列。12:45已加载模型并进入N33真实40步采样，resolved_runtime已核对原配方，尚无首条出片。不要自行扩大至全1333（约两周）；本批后继续扩展需另定名单。优先原文明确位移、运镜、持续动作，不擅自增加动作、运镜或改词。

所有主要路径、选择名单和反复确认过的偏好见最新handoff。docs/entitybench_session_handoff_20261031.md及新窗口提示词历史已归档，不按其中过期PID/auto-resume/磁盘reserve策略操作。不读/打印.env.portrait秘密，不自动删文件，无用户请求不推送GitHub。
```
