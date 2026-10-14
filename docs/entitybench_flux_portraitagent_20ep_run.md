# Portrait Agent 提示词＋FLUX 原生人物素材：当前 20 集

**2026-10-14最新补充：摄影版已完成，另按用户要求完成只裁脸评分0.7907（19有效集），共同样本0.8119。**
见[三路线结果汇总](entitybench_results_20ep_20261014.md)和[脸部协议](entitybench_face_only_dino_20ep_run.md)。
本文保留初版／摄影版启动与失败接续历史；下面“尚未开始视频”等快照不代表当前状态，不重启已完成队列。

**2026-10-14新增指标审计：[完整证据](entitybench_metric_provenance_audit_20261014.md)。**
`cs_face`已确认人物定位污染，同时存在真实外观漂移；不能当作脸部身份准确度。
boundary近1主要对应末帧I2V保留，论文matched-content gate／coverage／实例聚合与公开代码不同，不直接排名。
原报告和冻结协议保持，只读核验全部516边界及9组151张官方crop，无新视频或VLM调用。

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

**摄影版已通过小样验收并接续完整20集后台队列（2026-10-13，北京时间）。**
4候选均已查看；两张Viktor实际FaceLift预处理成功，生成512×512头部输入，未走几何裁剪失败回退。
原生脸高约193/248像素；头部裁剪放大不增加原始细节，仍须后续三维多视角和标定核验。
Julian/Silas为本轮不进入3D的多人硬切角色；40个单人硬切角色有3D准备资格，不是112张素材都重建。
预处理检查不生成新3D或新增候选、不修改源图。摄影版阶段输入、模型、参数和旧产物SHA核验通过。
完整队列启动后只处理本地FLUX→新3D/标定→191首帧→363视频→两路官方非VLM→自动归档。
查看样图及实际头部输入：新输出`review/admission_and_head_inputs.jpg`；
准入`review/style_admission.json`，预处理`review/facelift_preprocessing_check.json`。

**最新有效计划：`outputs/entitybench_flux_portraitagent_photo_i2v_a14b_4step_continuation_20ep/`。**
初版Agent样本4张全部保留：Julian/Viktor已有摄影质感，Silas全身仍明显带渲染感，
初版标记reference_style_revision_required，不进入正式视频队列。
新计划复用全部20次已保存的文本Agent输出，仅对19集写实prompt增加实拍DSLR、
自然皮肤和布料质感的正向风格前缀；动画集不加这个前缀。
Agent原始JSON不改；实际送入FLUX的最终prompt以新preparation_plan.json为准。
新前缀后T5最大仍为113/128，不改128长度、任何模型或采样参数；74个需保留文件SHA通过。
当前入口`pretest/run_entitybench_flux_portraitagent_photo_20ep.py`；
日志`outputs/entitybench_flux_portraitagent_photo20ep_pipeline.log`；
新归档`outputs/entitybench_paper_flux_qwen_portraitflux_photo20_comparison/`。
以下初版路径为保留记录，不应直接恢复。

用户于 2026-10-13 授权接续：在当前交互中用 Token Plan 的 Qwen3.7-plus
编写人物素材提示词，冻结后由本地 FLUX 后台生成，仍只覆盖当前 20 episode／363 shot。

## 边界与保留

新输出为 `outputs/entitybench_flux_portraitagent_i2v_a14b_4step_continuation_20ep/`。
先前插画样本所在 `outputs/entitybench_flux_native_i2v_a14b_4step_continuation_20ep/`
保持暂停，其计划、代码、4 候选和已选参考均不覆盖。Qwen20、旧 SDXL→FLUX20、
全140已暂停的资产、其他未提交改动和 `experiment_output/` 保留。

沿用现有 FLUX.1-dev FP8／PuLID 路线，未下载模型或改变视频、注入、评测参数。
新素材意味着新 FaceLift／标定；不借旧身份、Gaussian 或视频。
多人仍为 FLUX 文生图，无原生多人参考绑定。

## Portrait Agent 的实现范围

入口 `pretest/portrait_agent_interactive.py`：每次只接受一个 episode ID，
读取故事概述、各角色原始描述、首次出场上下文及镜头文本，调用一次 Qwen3.7-plus。
temperature=0、enable_thinking=false、JSON 输出；保存完整输入、Agent 指令、
输出、requested/returned model、Token 用量和输入指纹。套餐不提供我们原拟使用的
qwen-plus-2025-12-01，故使用实际模型列表中的 qwen3.7-plus；不声称固定了服务端内部快照。

明确动画的故事保持动画风格；无明确风格标记时采用写实摄影。
“animatedly”动作措辞不视为动画风格。描述必须逐字包含于输出 prompt，禁止添加
未提供的年龄、族裔、发型等；儿童和模糊性别描述保持。模型只输入文本，不调用图像判断。

这是论文 Portrait Agent 的文本任务改编，保留本项目人脸／FaceLift 构图要求，
并非论文完整 Agent／SAM2 分割／看图选图和验证流程。候选仍按已有单脸、姿态、面积规则选取，
不按正式指标选择。论文未公开具体 FLUX checkpoint，不能声称与作者同型号复现。

本地 `.env.portrait` 的密钥不写入产物或日志、不加入 Git。
当前对话中交互式调用完成后，后台入口只读取冻结提示词，移除 API 相关环境变量；
不会加载密钥文件、启动提示词 API 队列或测 VLM。

## 本地队列和核验

112 角色、40 单人硬切角色各3候选、其余72各1，共192候选。
先生成正式队列前4候选做风格检查，通过后才接续完整队列；不增加消融视频组。
首帧191、视频363；172接续只用本路线上一视频实际解码尾帧。
模型、采样、step30／v7、video_config 与前一 FLUX 原生计划一致，仅改变参考 prompt 和输出来源。

入口 `pretest/run_entitybench_flux_portraitagent_20ep.py`，
预计 tmux `entitybench_fluxportrait_a14b4step20`。
状态在新输出 `status.json`，各阶段日志在 `logs/`。
正式完整启动要求 `review/style_admission.json` 已验收且绑定当前冻结计划 SHA。
空间保留5GiB，启动需12GiB；不足时停止并保留，不删除产物或自动扩容。

后续官方非VLM评分继续按 evaluate-face-injection-v2：两个完整episode worker、
相同冻结评分代码与采样参数、strict、skip_vlm、VLM调用0、39项VLM指标null。
无fidelity gate，cs_face/cs_object暂定，scope=partial_without_vlm。
论文／旧FLUX／Qwen／新FLUX结果保存到
`outputs/entitybench_paper_flux_qwen_portraitflux20_comparison/`。
素材prompt和系统路线均不同，分数差异不解释为插件因果收益或论文严格排名。
