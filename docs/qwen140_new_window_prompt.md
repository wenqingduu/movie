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

# 新会话接手提示词（2026-11-09，整集优先）

```text
最新benchmark更新（2026-11-09 22:11北京时间）：本会话仅完成暂停时8条40步视频的评测收尾；8条6VBench均完成，7条0→1（除M0010），累计64尝试/35替换/2456不变，剩1269条40步未生成。40步动态队列仍暂停，不自动恢复；下述独立78集4步整集恢复队列保持其自身授权与状态，本次未停止或修改。最新完整140非VLM结果与默认人物DINO复核均已完成，输出outputs/entitybench_original40_priority_remaining_20261109/paused8_benchmark_20261109/benchmark_summary.md及JSON。7完整episode/161shot重算、966项缓存值与2491媒体SHA验证通过，评分失败0；默认DINO3份助手隐藏分数视觉复核（含3精确原帧），3保留/0新增排除，Riley(run44/9:1)身份关联未决保留、57旧排除按精确SHA沿用，不称外部人工签核。默认人物DINO .7990（130有效集/1465角色次），主体.9414、时间稳定.9863、平滑.9920、动态.4873、美学.5859、成像60.8634、角色.9479、物体.8981、场景.6249、raw人物.7238、raw物体.8077、接缝.8441；动态精确.487254139136，比28版+.325108个百分点，累计+1.390115个百分点。39项VLM仍null，Table4缺15项，partial_without_vlm；所有下降保留。visible_region/report.json为最终，automatic为中间。下方“8条0新评分/默认DINO待审”是历史，勿覆盖这份已完成35版报告；新整集生成结果尚未纳入本快照。

最新恢复：2026-11-09 22:09（北京时间），扩容后继续78集4步整集队列

用户明确“继续吧，我刚刚扩容了一下”。实查数据盘450GiB、可用约113GiB，覆盖此前约60～65GiB的新增估计；不设reserve、不删除旧产物。旧整集tmux已退出，状态停留run714/i20_j34的1/10shot完成、2:1首帧第8/20步，日志无退出异常尾部，不能断言是磁盘不足。已完成的人物3D资产、1:1视频及2:1 Agent/提示词/帧task全部保留。

恢复tmux仍为`qwen140_never_rerun_v2_20261109`，**实际新入口为 `pretest/qwen140_never_rerun_resume_20261109.py run`**，日志及范围仍`outputs/entitybench_never_rerun_episodes_v2_20261109/`。原78集/1218shot冻结plan、原生成与评测源代码不改。新恢复入口只绕开旧v1迁移器对“无成品帧”的自动archive/replan，直接安装同一v2并继续`run_remaining`，从2:1相同输入重算未完成的采样，不新查角色/重写Agent计划。17项已存结果与计划SHA绑定在`resume_after_expansion_20261109.json`，编译与v2/run/tag探测通过，预检API0。正常后续新shot仍按原流程调Agent与B过渡API；主视频仍4步、原多人v7门控，不恢复40步动态队列。

恢复前另一已生成8条的完整上下文评测已释放GPU，状态为official_full140_complete_default_dino_review_pending（新增7、累计35替换），不覆盖其结果，不把默认DINO未复核当完成。该评测的“暂停所有生成”交接是此前状态，**本轮用户明确继续78集整集生成为当前授权**，无新增动态40步授权。第一集实际恢复阶段以新status/progress/log为准，完成进度不重复清零。预计仍约5～6天，首集完成后校准。无守护Agent、无自动改模型/提示词/阈值、无推送。

请接手 /root/autodl-tmp/movie，用中文。先读 docs/entitybench_session_handoff_20261109.md 顶部与 LLM_HANDOFF.md，再核对实际tmux/进程/GPU/磁盘；不要按历史队列自动恢复或重复启动。

最新用户要求停掉动态增强，先跑从未整集重跑的episode，并明确选择原整集4步配置。40步1277队列已停止，保留第一批8生成/0新评分，不新增媒体替换，不自动恢复。此前累计28替换版全指标已完成，结果保留。

新tmux qwen140_never_rerun_v2_20261109，入口 pretest/qwen140_never_rerun_episodes_20261109.py run，范围唯一来源 outputs/entitybench_never_rerun_episodes_v2_20261109/plan.json：78episode/1218shot，启动前无整集shot、无rerun_manifest、未complete，启动证据inventory_at_launch.json。部分有独立shot实验，“未整集”不等于从未做过任何候选。另4个部分整集不跑，已complete的run445不重复。首集run714/i20_j34/10shot，复用其已生成的人物素材构建前缀；启动时处于references，尚无新shot。

沿用v2_20261104路由Agent+人物描述绑定+episode画风+多人v7 3D；主视频LightX2V4步81帧16fps，不使用这轮动作/背景强化词，不换40步。B仍用原wan3.0-video首尾帧API，A自身真实上一尾帧，B/C正常单/多人注入。保留正常小脸/姿态等门控和Control跳过，不取消3D，不自动改词、语义重试、降模型/阈值。没有修复Agent。

新调度status.json/queue.log/stage_timings.jsonl在新输出；素材、shot、整集视频与12项官方评测仍写原outputs/entitybench_qwen140_multiface_remaining126_20261030/experiments/<完整eid>/。每集完整生成后strict/skip_vlm评12项，VLM0；默认可见单脸+定位纠错DINO需单独复核，raw cs_face不是人脸识别，auto不充final。评测依evaluate-face-injection-v2 skill。初估5～6天，按新实测校准；13GiB余量不足预计新增60～65GiB的全队列，用户需扩容，不设reserve/不删旧产物。系统性API/磁盘/OOM错误停队列记录，单集技术错误留档继续其他，不创建守护。SSH可断开但服务器保持开机。

不读/打印.env.portrait密钥，不自行push；保留所有无关改动/PPT/中文模型图、docs/frontend_agent_architecture.md、experiment_output。不恢复旧动态1333、1277或原整集调度。新wrapper已通过编译、冻结SHA与run/tag/v2编译器probe，真实出片看日志不看预检。所有细节以最新handoff和实际status为准。
```
