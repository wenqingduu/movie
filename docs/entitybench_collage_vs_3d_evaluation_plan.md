# EntityBench：20 episode 首帧插件与尾帧接续的完整视频评测方案

日期：2026-10-11（北京时间）。状态：用户已授权全部切到4步；191个硬切首帧齐备，官方非VLM接入检查通过，4步正式队列运行中。
文件沿用原路径以保持交接链接有效；以下内容以用户最新确定的范围为准。

**首轮只做一组我们的完整路线：硬切复用合规首帧，接续直接用本路线上一镜头尾帧，
视频使用普通 Wan2.2-I2V-A14B 加 LightX2V 4步蒸馏LoRA，最终采用官方 EntityBench 非VLM指标。**
独立镜头 Control/Treatment 消融与拼贴/EntityMem 对比后置，不阻塞本轮。
用户已授权视频生成、补齐必要首帧及非VLM评测；明确要求VLM暂不测。
新实验独立保存，旧图像/视频/分数不变；不调用收费VLM API。

后台入口：`pretest/run_entitybench_a14b_4step_20ep.py`，tmux会话 `entitybench_a14b4step20`。
实时状态见 `outputs/entitybench_pulid_i2v_a14b_4step_continuation_20ep/status.json`。
运行顺序为官方非VLM接入检查→363shot全部重新生成→来源审计/成片→官方非VLM评测。
本轮不运行项目InsightFace、参考锚定DINO或脸框质心指标；完整配置见
[4步执行记录](entitybench_a14b_4step_run.md)。旧40步队列已停止，7个完整视频保留但不混入本轮。

## 1. 首轮任务与数据范围

评价问题：已有首帧身份插件与普通 I2V、原剧本接续策略组合后，在完整故事中的跨镜头一致性、边界连续性、实体出现率和视频质量达到什么水平？
动作遵循与文字外观保真度依赖VLM，本轮不提供这些结论。
首轮是单组系统评测，不计算新 Control/Treatment 视频增量，也不据此声称插件的因果收益。
已有首帧配对信息保留为来源审计；不自动生成第二路视频。

使用冻结清单 `outputs/entitybench_firstframe_pulid_ip_20ep/manifest.json` 的全部20个episode，
完整保留 scene、shot_key、顺序、prompt、entity_schedule 和 cut，不改写画风，不挑选正例。
2026-10-11 静态读取清单和原剧本核对如下：

| 范围 | 硬切 cut=true | 接续 cut=false | 合计 |
|---|---:|---:|---:|
| 单人镜头 | 74 | 58 | 132 |
| 多人镜头 | 117 | 113 | 230 |
| 无人镜头 | 0 | 1 | 1 |
| 全部镜头 | 191 | 172 | 363 |

难度构成为 Easy 11、Medium 6、Hard 3。首次正式运行固定一个 seed replicate，
计划363个shot视频任务、20条完整episode；不沿用旧方案5episode/41shot/三组/123任务的范围。
20episode是固定benchmark子集，不是全量140episode/2,491shot。

## 2. 当前唯一生成路线

```text
原有序剧本
  ├─ 硬切：已有合规 PuLID-FLUX 插件首帧，或其已记录的安全回退首帧
  └─ 接续：本路线紧邻上一 shot 实际输出视频的最后一帧
→ 普通 Wan2.2-I2V-A14B + high/low 4步蒸馏LoRA + 当前 shot 文本
→ 当前 shot 视频与真实尾帧
→ 按顺序继续下一 shot
→ 官方 EntityBench 非VLM视频评价
```

| 镜头类型 | I2V输入图 | 插件行为 |
|---|---|---|
| 硬切，包括episode起始镜头 | 优先复用该shot既有Treatment首帧；来源须符合中间步协议或已声明回退 | 复用原首帧生成时的注入，不再次处理已完成图像 |
| 接续 | 本路线上一shot真实尾帧 | 不编辑、不重绘、不再次注入；不用该shot旧的独立生成首帧 |

不恢复“根据最终Control准备3D、再从噪声重跑Treatment”的历史两遍式。
现有有效首帧算法是从中间 pred_x0 检测/估姿态/准备v7 reference，再从同一中间latent继续去噪。
PuLID-FLUX以既有实验实际配置为准，不把Qwen的step12调度复制到Flux。
依据：[交接文档统一协议](../LLM_HANDOFF.md#31-统一执行协议不得自行改动)。

接续必须来自同一episode、同一路线、紧邻上一shot；不借用历史TI2V-5B视频、Control视频、
其他episode或不相邻镜头的尾帧。上一shot生成失败，则后续依赖它的接续链标为dependency blocked，
不能静默换成独立首帧；后面的硬切可作为独立起点继续。episode首shot若错误标为接续，
记录协议错误，不自行修改cut或跨episode接续。

## 3. 首帧复用与当前适用范围

首帧来源优先指向冻结清单各episode的 `pulid_output`。逐个硬切保存最终图路径/SHA256、
原prompt与seed、角色参考、生成配置、注入协议、实际注入状态、gate/skip_reason、原报告位置。
仅有文件存在或名称为Treatment不足以证明来源正确；最终Control指导的历史两遍式图不得混入。

首轮仍采用现有PuLID-FLUX路线，不能为了多人镜头悄悄切到Qwen或增加多人注入：

- 单人硬切：复用已有中间步注入结果；缺脸、小脸、姿态/渲染失败、空core、闭眼等已记录的
  安全回退也保留，回退图不算实际注入。
- 多人硬切：沿用当前PuLID-FLUX不注入的边界，使用该shot已验证的基础/Control复用首帧，
  明确记录 `multi_character_no_injection`，不把117个多人硬切写成插件成功覆盖。
- 接续中的多人/无人镜头：按尾帧接续生成，保留原角色调度与文本；不能删除。

191个硬切首帧现已全部具备并逐个核验SHA256与原配置：56个实际注入单人硬切、
18个安全回退、117个多人基础首帧。其中111个多人首帧已在旧40步实验准备阶段补齐，
本轮直接复用这些图像，保留旧来源目录；不重新生成首帧、不新增多人注入。
172个接续镜头全部取新4步路线上一视频尾帧，不用旧独立首帧、40步或小样本视频。

角色资产仍是episode-local，沿用原参考/Gaussian/标定来源，保存在asset_manifest中。
官方评价依据原剧本角色描述，不把项目身份参考图作为官方DINO的锚点。

## 4. 视频模型、文本和随机性

使用普通 **Wan2.2-I2V-A14B**，不加载StoryMem M2V/MI2V LoRA，不提供多图memory，
也不把已完成TI2V-5B视频当成新模型输出。它接收一张首帧图与当前shot文本。
官方实现：[Wan2.2 I2V](https://github.com/Wan-Video/Wan2.2#run-image-to-video-generation)。

2026-10-11用户扩容后已启动官方模型下载，固定revision
`206a9ee1b7bfaaf8f7e4d81335650533490646a3`，目标目录
`models/video/Wan2.2-I2V-A14B/`。官方33文件共126,205,359,703字节（117.54GiB）；
启动前磁盘总容量300G、可用152G。后台tmux会话 `wan22_i2v_a14b_download` 使用aria2
断点续传并校验LFS权重SHA256；清单、实时状态和日志保存在
`outputs/model_downloads/wan22_i2v_a14b/`。已于北京时间2026-10-11 16:06下载完成；
33文件大小与LFS权重SHA256校验通过。

当前正式配置为LightX2V `7bd01b7d5d04bf99cdd9ca9082f20c184c66d991`：
原A14B加high/low两份rank64 `4step_1022` 蒸馏LoRA（strength=1），DMD2时间步
1000/750/500/250、专家切换index2、shift5、guidance1、CFG关闭，不传负面提示词。
LoRA来源revision `570044187a5219776ef30a5c60c6f76428a3a10a`，两份官方SHA256通过。
81帧、16fps、max_area=832×480，保持输入比例（试验848×464）；BF16、torch_sdpa、
模型级CPU offload、T5 offload、原VAE在GPU，无量化/缓存/StoryMem LoRA或视频人脸注入。
为适应96GiB主存，仅候选进程关闭torch.empty的固定页分配；不改权重值。

模型、LoRA、实现和输入依赖的指纹均进入manifest与每shot续跑校验。
本轮是用户主动选定的蒸馏生成配置，不能写成原40步的等价工程加速，亦不声称与论文
未公开生成配置相同。旧40步7段视频和加速试验保留在原目录，正式视频全部重新生成。
普通40步/4步小样本的耗时与质量差异见[A14B加速试验](a14b_acceleration_trial.md)。

保留原shot文本与实体调度；如沿用既有首次出现描述补齐逻辑，明确补齐来源、固定版本、
保存最终prompt与哈希。不得用LLM临时改写分镜、画风、cut或根据输出重写prompt。
普通I2V接续时不能额外传入新进入角色的参考图；新人物是否出现及其身份可能受影响，
作为该路线能力限制，本轮用官方presence及非VLM一致性如实评价，不换镜头规避；
fidelity/action暂缺。

按原manifest的episode base_seed与稳定原始全剧shot_index固定一shot一seed；正式执行前
核验索引基数和最终映射，不使用删减后的单人队列索引。固定单候选规则，错误恢复与重试
记录次数和成本，不按人脸分数挑选输出。顺序依赖在episode内部严格执行，不跨episode传状态。

## 5. 尾帧、边界和成片

上一shot落盘成功后，按视频实际解码帧序读取最后一帧，保存PNG、来源视频SHA256、帧索引、
尺寸与PNG哈希；下一接续shot的输入必须可追溯到它。保留I2V内部必要的统一尺寸预处理，
记录变换，不做任何面部修正或隐藏边界差异的图像编辑。

同时保存实际生成视频第一帧，验证“所传尾帧”与“实际生成首帧”的关系；不能只凭接口名称
假定两者像素一致。若成片拼接需去除重复条件帧，单列成片规则与源帧映射，不改写官方评价
使用的逐shot原始视频，不删除难看、漂移或重复的镜头。

边界评分使用原始上一shot末帧与当前shot实际首帧，沿用官方适用集合和算法；不能用两份
输入PNG自比较得到人工满分。另按hard cut/continuation分组作诊断，不能替换官方汇总。
普通首帧I2V不等于StoryMem的motion-frame接续，动作速度/镜头运动的连续性需实际评分。

## 6. 评价：只运行官方非VLM指标

### 6.1 官方EntityBench主表

在这20个完整episode的363个shot上，运行冻结版本的官方三pillar evaluator：
[官方代码](https://github.com/Catherine-R-He/EntityBench/blob/main/eval/evaluate_benchmark.py)。
保留原episode_id、scene/shot编号、实体描述和调度，按官方要求映射
`<results_dir>/<episode_id>/<scene>_<shot>.mp4`，例如 `001_001.mp4`，保存映射清单。
显式选择20episode split，不使用默认全量split再把缺失120个episode误计为方法失败。

本地 `benchmarks/entitybench/eval/` 已有三份评测入口，但 `evaluate_benchmark.py`
含项目新增的 `--skip_vlm` 分支，并非原样官方文件；另外两份入口与已核对官方版本一致。
正式主表应使用官方revision `3776439e7037b2128ac3b5d2d1ce63e424cca166` 的原始入口，
独立保留，不覆盖本地改动；原始主入口SHA256为
`2334aad4b61cae2a0ea38cc2ee9ae85125b6df4e128610903654fd2b49607bb4`。
完整官方评测应核验文件指纹与依赖、不启用 `--skip_vlm`；已有项目部分评测不能当作官方全项结果。
用户随后明确授权本轮VLM暂不测，因此当前调用已审计的本地 `--skip_vlm` 适配版本，
将官方原始入口、适配版本与差异补丁一并冻结到新输出的 `runtime/`。

| 主表维度 | 重点指标与报告要求 |
|---|---|
| 视频质量 | 官方Pillar1全部质量指标，包括动态程度、运动平滑、闪烁、美学与成像质量 |
| 跨镜头一致性 | 官方Pillar2 `cs_face/cs_object`；LLM人脸/物体/场景一致性本轮缺失 |
| 角色/实体与动作 | 官方Pillar3 presence；文字外观保真度与动作遵循本轮缺失 |
| 接续边界 | 官方 `cs_transition_boundary`，保留官方适用集合及有效/跳过/失败数 |

本轮先运行非VLM部分：VBench六项（subject_consistency、temporal_flickering、
motion_smoothness、dynamic_degree、aesthetic_quality、imaging_quality）、GroundingDINO/CLIP presence，以及DINOv2
`cs_face`/`cs_object`/`cs_transition_boundary`；保留官方检测、裁剪、采样与聚合实现。
VLM外观保真度、动作遵循、跨镜头LLM项均记缺失，不填零。**VLM保真度gate也暂未应用，
因此本轮跨镜头分数须标为provisional/partial_without_vlm，不能当作完整官方或论文可比分数。**
以后补VLM后须重新筛选有效外观并计算跨镜头分数，而非仅在当前汇总追加几个字段。
每项给出value、n_eval、n_failed、n_skipped及coverage。
官方规定的gate零贡献与API/模型计算失败不同；后者及无有效项保留缺失，不补零。
方法自身首帧注入gate与官方评分gate分别记录。

官方角色crop可能包含头发、服装和身体；`cs_face`使用官方跨镜头质心集合。
我们之前的脸部bbox质心虽然采用相同数学公式，仍不是官方`cs_face`，不能复制分数代入主表。
评测器、检测/特征模型、VLM版本、阈值及完整配置需冻结并保存，不猜论文的未公开配置。
没有VLM服务或相应模型时报告“官方评测部分完成”，缺失指标不能算实验已完整评完。

### 6.2 历史项目指标不进入本轮

用户本轮要求按官方口径，因此不运行项目InsightFace、固定参考DINO或脸框外观质心。
旧20episode配对评测和加速小样本诊断仍保留，不能复制到新4步官方主表。
官方GroundingDINO角色crop、CLIP代表帧选择、DINO质心与原有脸框/身份锚点不同。
当前官方版本实际启用6项VBench；文件头仍有“7项”旧说明，以代码常量为准。

### 6.3 适用性、成本与质量审计

报告191个硬切中的实际插件/安全回退/基础多人/缺失首帧数，172个接续的尾帧来源核验，
各类失败与完整episode覆盖。分别统计首帧历史准备成本、必要补生成、新14B视频及评分成本。
复用成本和首次端到端成本分列，不将历史3D建模/标定成本隐去。
保留逐镜头预览、连续边界联系表、跨镜头角色联系表及正负例/漂移/回退审计。
单组不要求新造插件C/T六面板；原首帧配对图只作为来源证据，不能标成新14B视频对比。

## 7. 产物与依赖续跑

独立输出根：`outputs/entitybench_pulid_i2v_a14b_4step_continuation_20ep/`。
不覆盖旧PuLID/IP首帧、TI2V-5B视频、质心报告或根目录三episode/Qwen汇总。

```text
manifest.json                 # 单组、20episode/363shot、剧本/模型/config/seed/实现哈希
firstframe_inventory.json     # 191个硬切的复用来源、合规协议、回退/缺失清单
asset_manifest.json           # 身份参考与历史3D资产、角色覆盖与来源
inputs/<episode>/<shot>/      # 实际I2V输入；硬切来源或上一shot尾帧的依赖链
videos/<episode>/<shot>/      # 正式逐shot视频、首/尾帧与生成报告
official_results/<episode>/   # 官方命名视频的可核验映射与链接/副本
evaluation/official/          # 三pillar官方分数、crop、gate/错误与评测manifest
preflight/                    # 两段旧4步试验视频上的官方接入验证，不是正式分数
audit/                        # 全部预览、真实边界/跨镜头联系表
episodes/                     # 20条按原剧本顺序拼接的浏览成片及帧映射
report.md                     # 单组官方非VLM表、成本、限制与完成状态
```

每shot provenance包括input_source_type、输入图/源视频哈希、previous_shot_key、cut、
完整prompt、seed、模型与生成配置指纹、输出哈希和失败状态。只在全部依赖与配置相同且
原输出完整有效时续跑复用。上一视频改变，必须使依赖其尾帧的后续接续链失效，直到下一硬切；
配置/模型/实现改变则重新核验受影响任务，不能只看mp4存在。

## 8. 后续执行次序与完成标准

以下执行顺序已获用户授权；VLM环节暂不执行。

1. 核验冻结20episode原剧本、363shot/cut清单、191个硬切首帧库存及角色参考覆盖。
2. 接入普通I2V-A14B的LightX2V 4步蒸馏配置，冻结LoRA/模型/规格/seed及尾帧依赖。
3. 按预先选定、包含硬切→接续关系的短链作接入测试，验证输入尾帧与真实输出首帧；
   测试输出与正式配置分开，不从测试分数挑参数或删难例。
4. 复用齐备的191个硬切首帧，重新完成4步单组363shot、20条episode。
5. 运行官方三pillar的非VLM部分；核验ID映射、依赖链、有效数、零VLM调用及报告一致。
6. 查看全部边界/角色联系表与代表性正负例；汇总失败、回退、coverage、成本和模型限制。

生成完成须满足363shot都有正式输出且20条故事完整；失败或依赖阻断时明确为未完成。
评价完成须核验全部适用官方非VLM指标及缺失原因，API指标未运行不得报告官方全项完成。
不能仅凭退出码、文件数量或成片可播放判定成功。4步小样本平均89秒/shot，
363镜头纯生成粗略约9小时，完整耗时还需加评测、加载及重试；不是整批实测完成时间。

## 9. 后置实验与允许结论

用户最新决定先完成本单组完整路线，以下另行开展，不作为本轮完成条件：

- 基础参考Control vs 插件Treatment的独立镜头14B消融，复用同一配对首帧，隔离视频模型传递效应。
- 同接续协议下的基础/插件完整rollout对比，各路使用自己的尾帧，评价累计系统差异。
- 拼贴/EntityMem系统对比：沿用共享身份资产，核验其原生memory条件与完整配置；不做拼贴加插件。

本轮可报告固定20episode子集上的绝对系统表现及覆盖率，不证明插件优于基础或拼贴，
不宣称完整EntityBench优越性，不把论文140episode均值作为同子集的配对基线。
后续对比需相同episode/资产/输出规格与评分版本，记录视频LoRA/输入机制及完整预算差异；
参数规模相近不能单独证明公平。

## 10. 论文方法与开源核查记录（后续对比参考）

2026-10-11核查EntityBench main revision `3776439e7037b2128ac3b5d2d1ce63e424cca166`：
完整文件树150个文件，公开140份剧本、split、评测代码与示例；未发现EntityMem生成入口、
角色/拼贴资产、完整生成配置或批量生成视频，[Releases](https://github.com/Catherine-R-He/EntityBench/releases)为空。
这是当前官方发布范围，不能据此断言作者没有其他未公开产物。

| 方法 | 方法原论文/官方实现 | EntityBench配置确认范围 |
|---|---|---|
| StoryMem | Wan2.2-I2V-A14B + M2V LoRA，支持memory与可选首帧/运动帧 | 精确生成revision/模式/参数未明确 |
| EntityMem | 扩展StoryMem，无额外训练；实体参考与拼贴memory，接续用上一shot尾帧 | 输入机制确定；A14B具体组合据上游推断，完整配置待确认 |
| HoloCine | Wan2.2 14B基础的专用多镜头训练模型，公开full/sparse DiT权重 | 具体所选checkpoint未明确 |
| CineTrans | 公开DiT版基于Wan2.1-T2V-1.3B，另有LaVie UNet版 | EntityBench未注明所选变体，不能认定为14B |

依据：[EntityBench §4/§5.1](https://arxiv.org/html/2605.15199v1#S4)、
[StoryMem官方实现](https://github.com/Kevin-thu/StoryMem#model-download)、
[HoloCine §4.1/§4.2](https://arxiv.org/html/2510.20822#S4.SS1)、
[CineTrans §5.1](https://arxiv.org/html/2508.11484#S5.SS1)。
核查StoryMem revision为 `052c68d2627a22a95f79dbb4d3376cc30c96f1a3`；其当前示例832×480/
MI2V参数不能当作EntityMem论文实际参数。HoloCine原论文也使用关键帧生成方法+普通Wan2.2
14B I2V作系统基线；该先例不是EntityBench使用配置的证据。

拿到作者逐shot视频后，需同时核验episode/shot映射、原身份资产、生成及评价配置，在共同
子集重评；不同参考身份不能用我们的参考图评分后宣称身份保持更差。只有论文汇总数字时
单列外部结果，不替代本子集分数。作者现成视频不能换模型；换模型需要重新生成。
一句话自动规划研究见 [多镜头调研](multishot_benchmark_research.md)，不改变本轮固定剧本。

### 2026-10-11：加速小样本历史记录（现已切换4步全量）

用户授权尝试LightX2V；见 [A14B加速试验](a14b_acceleration_trial.md)。
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

2026-10-11 23:42，用户要求全部改用4步，旧40步队列停止并保留7个已完整落盘镜头。
以上小样本恢复40步的记录仅为历史；当前入口、范围与指标以上文4步方案为准。
