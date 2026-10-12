# 下一步实验：Qwen-Image-2.1统一人物素材与首帧生成

编写日期：2026-10-12，北京时间。21:03用户要求停止FLUX全140，先跑相同20episode；
21:12已启动Qwen20episode队列，当前仅363shot，不自动继续全140。
实际冻结参数、入口及进度见[20episode运行说明](entitybench_qwen21_20ep_run.md)。
下文140规模预算保留作后续规划，当前执行范围以20episode运行说明为准。

## 目标与比较对象

用Qwen-Image-2.1同时生成人物素材和参考条件首帧，再通过现有3D人脸插件与普通
Wan2.2-I2V-A14B生成完整多镜头视频，检验整条路线在EntityBench上的表现。

对照对象为已完成的PuLID-FLUX20episode路线：相同benchmark、硬切/接续标记、
视频模型配置和官方非VLM评价。当前只比较冻结20episode；全140暂不执行。
原20episode的插件配对实验保留为已有诊断证据；本轮不增加新的无插件视频消融组。
新旧路线同时更换素材生成与首帧模型，比较结论是系统路线差异，不归因成插件增量。

## 已完成的Qwen实验与下一轮的区别

2026-10-12再次核对Git记录和本机原始JSON：Qwen-Image-2.1已经接入并测试过人脸插件，
不需要把它当作全新模型重做接入。三个episode为run719、run893、run1517，共34shot
（23单角色、11多角色）；修正后官方Control分叉版本30shot实际注入、4shot安全复用，
41个可评注入角色-shot对，34份首帧结果、68个配对视频job、失败0。
30份各类身份评测JSON和34份完整comparison.jpg均存在。

| 已有诊断指标 | Control | Treatment | 配对均值差 |
|---|---:|---:|---:|
| 首帧InsightFace | 0.5376 | 0.5264 | -0.0112 |
| 视频InsightFace逐帧均值 | 0.4078 | 0.3942 | -0.0136 |
| 首帧参考锚定DINOv2 | 0.6093 | 0.6037 | -0.0056 |
| 视频参考锚定DINOv2 | 0.5222 | 0.5238 | +0.0016 |

统计单位为41个注入角色-shot对。来源是
`outputs/entitybench_qwen_image21_official_control_partial_adaptive_040_full3/evaluation_report.json`
及`dino_evaluation_report.json`，不是官方EntityBench的cs_face。
已有结果说明链路可运行，且插件在这批Qwen输出上没有稳定改善身份，不能预设换Qwen必然提高。
高相似度脸也存在注入后身份下降；正负例及回退必须全部保留。

Git证据：`0a1b44d`（2026-09-27）加入Qwen下载及完整episode评测链路；
`b27393f`（2026-10-01）纳入严格官方Control分叉代码和记录为2026-09-28完成的修正版结果；
`b6c06e9`（2026-10-04）将Qwen 3D注入接入产品工作流。

旧实验使用既有SDXL身份素材、Wan2.2-TI2V-5B、49帧/24fps/50步的逐镜头视频，
评价是InsightFace与参考锚定DINO；下一轮新增的是Qwen生成人物素材、同模型参考首帧、
普通A14B 4步真实尾帧接续与官方v2整剧本评价。已有图像/注入代码可复用，
但换身份素材后仍需新3D资产，旧视频不能当成新路线结果。
历史最终Control准备姿态的两遍式目录及手写Control的shared-prefix目录只用于追溯，
不与当前严格官方Control分叉结果混算。主表及已有说明见`EPISODE_TEST_RESULTS.md`和
`LLM_HANDOFF.md`第9.1节。

当前全140基准入口为`pretest/run_entitybench_a14b_full140.py`，
输出`outputs/entitybench_pulid_i2v_a14b_4step_full140/`；已按要求停止，144/217新增标定资产保留。
完整数据为140episode、2,491shot、1,136硬切、1,355接续；20episode子集为363shot。

## 路线

```text
原EntityBench剧本、实体描述、镜头调度及画风
  → Qwen-Image-2.1文生图：每个episode内角色生成并冻结素材
  → 需要人脸注入的角色：FaceLift Gaussian → 角色自己的姿态标定
  → 硬切：Qwen-Image-2.1以固定素材为参考生成首帧
       → 单角色且满足人脸gate：在中间去噪步应用v7 3D residual
       → 多角色：原生多参考条件；本轮不新增多人3D注入
       → 非人脸角色或缺脸：保持基础输出，明确记录跳过原因
  → 接续：直接使用本路线前一视频的解码尾帧，不编辑、不再次注入
  → 同一个LightX2V 4步Wan2.2-I2V-A14B
  → 来源/尾帧审计、完整episode成片
  → evaluate-face-injection-v2官方非VLM评测
```

参考人物在episode内只生成一次并跨镜头使用。使用同一个模型不会自动固定身份，
每个硬切必须明确提供已冻结参考；同名角色不能跨episode共享身份。
人物素材按原描述保留年龄、服饰和风格，不统一强制成年、写实皮肤或人形。
动物、玩偶等也可有Qwen参考素材，但不作为人脸3D资产。

多人镜头先采用角色素材逐图输入并建立名字到图片的对应；角色数超过官方10张参考图
上限时，在接入阶段制定并冻结完整覆盖的条件组织方式，不能静默丢角色。
可研究带标签素材板作为输入组织方式，但这不等于复现EntityMem的拼贴memory路线。
新模型生成的首帧必须传给I2V；不能借用旧FLUX视频或旧Qwen素材下的视频充数。

## 是否仍需要标定

**继续使用现有3D插件就仍需标定；Qwen本身生成/编辑图像不要求这一步。**

标定解决的是FaceLift相机姿态到目标脸检测姿态的映射，与图像生成器品牌无关。
沿用273个相机网格采样（7pitch×13yaw×3roll）、512标定分辨率及现有姿态回检，
完成后才生成需要注入的首帧。标定和Gaussian、素材SHA绑定。

本轮人物素材改为Qwen生成，身份源改变，因此必须重建相应FaceLift与标定；
不能按角色名字复用旧SDXL素材产生的Gaussian。只有同一素材SHA、同一Gaussian SHA、
同一标定配置全部核验一致，才允许复用已有资产。

3D只为单角色硬切中具备适用人脸的角色准备；不为仅接续、仅多人出现的角色额外建3D。
当前全140原始清单有266个单角色硬切角色键；已核对9个非人脸角色后，
单角色注入资产计划约257份，最终以新路线冻结的资格清单为准。
当前FLUX队列的217份是新增120episode范围，不能当作新Qwen全140资产数。

沿用全140修正后的episode-local非人脸清单：Sunny、Tingo、Goro、Blaze、Gizmo、
Crag、Dizzy、Poppy、The Shadow，依据见当前输出`face_injection_routing.json`。
这些角色和所有镜头都参加官方评测；跳过人脸插件不意味着排除benchmark样本。

## 模型已经在本机

官方开放权重模型：[Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)。
官方统一支持文生图、编辑、最多10张参考及人物身份保持；这些是能力说明，
不是已证明的本项目benchmark收益。

| 项目 | 本地核对结果 |
|---|---|
| 模型目录 | `/root/autodl-tmp/movie/models/diffusion/qwen-image-2.1/` |
| 已有revision | `790c92633540aa0cb11d9abf19eb46d861714758` |
| 模型环境 | `/root/autodl-tmp/qwen-image21-venv` |
| Safetensors文件 | 7份；文本编码器4分片、DiT 2分片、VAE 1份 |
| 权重字节 | 33,115,613,408bytes，即33.12GB / 30.84GiB |
| 完整性 | 两个权重索引列出的分片全部存在；下载metadata的revision一致 |
| 历史校验 | `LLM_HANDOFF.md`记载已有全部权重SHA校验通过；本次只核对目录、大小及索引，不宣称重新全量哈希 |

本机已有模型，可复用，不必重复下载33GB。正式运行前重核验历史SHA并冻结revision；
不把远端main更新自动同步进正在运行的环境。

官方另已发布8步Turbo（见[官方仓库](https://github.com/QwenLM/Qwen-Image-2.1)），
本计划采用已有普通Qwen-Image-2.1权重及已验证注入协议；Turbo的时间表与注入适配
另行研究，不把它混入本轮。文生素材建议1024×1024、40步；参考首帧从已有
1024×576、20步协议接续，正式参数在接入阶段确认并一次冻结。

## 显存与执行安排

当前硬件为单张48GiB RTX4090（`nvidia-smi`为49,140MiB），主存96GiB、CPU配额20核。
Qwen的“7B”仅指视觉生成DiT，不包括文本/图像编码器和VAE，不能仅按7B估整套显存。
[官方架构与offload说明](https://github.com/QwenLM/Qwen-Image-2.1#architecture)。

已有严格官方Control/shared-latent注入的三个episode、34个首帧结果中，
`peak_cuda_memory_mib`为5,954.34–12,160.71MiB，约5.8–11.9GiB；入口为
`pretest/run_qwen_image21_shared_prefix_adaptive.py`，使用BF16与
`enable_sequential_cpu_offload()`，默认1024×576、20步。
证据目录为`outputs/entitybench_qwen_image21_official_control_partial_adaptive_040_full3/`。
这是PyTorch已分配峰值，不含所有CUDA上下文/外部库/缓存，不能称为整卡最低需求。

据此，新路线从单任务CPU offload开始，工程预留16–24GiB显存，当前48GiB卡能沿用
已验证配置。多参考数量增加、2K输出或不同offload模式的峰值需另测；本次没有启动新GPU测试。
不保证所有组件常驻GPU或任意多图配置都能装入48GiB。

图像模型、FaceLift和视频模型分阶段加载；新Qwen正式实验在用户要求停止FLUX队列后开始，
避免争抢同一GPU。角色资产可从两路开始；Qwen首帧单进程，评测按skill v2两进程。
CPU offload降低显存但可能降低速度，先量单素材、单/多人首帧、3D标定和视频的实测墙钟，
再估全量耗时。当前4步视频平均96.47秒/shot：20episode纯视频约9.7小时，
全140纯视频约66.8小时；这些数不含新素材、标定、首帧与评价，不是端到端工期。

## 磁盘容量

2026-10-12约20:52，本机数据盘300GiB，可用约19.5GiB；Qwen权重已经计入已用容量。
当前FLUX实验仍在生成资产并将产生首帧、视频和评分文件，这些空间不能全分给下一轮。

**已有模型加载不需要再腾出33GB下载空间，但目前余量不足以安排第二套全140产物。**
下一轮不重复存视频基础模型、不复制旧权重和旧输出；所有新增素材/3D/视频在独立目录。

| 新增存储项 | 初步预算，需小样本修正 |
|---|---:|
| 约257份FaceLift/标定 | 约21–25GiB，按当前每角色约84MiB估计 |
| 角色素材及候选，完整调度约992个episode-local角色键 | 约5–15GiB |
| 硬切首帧、必要共享latent及注入审计 | 约10–25GiB |
| shot视频及完整episode成片 | 约5–15GiB |
| 官方评分裁剪、缓存、报告 | 约4–10GiB |
| 临时文件及运行余量 | 约10–20GiB |

上述是预算估计，不是实测最低容量。建议额外扩容约100GiB作为起点，给当前队列
剩余产物及下一轮共同留余量；根据接入样本实测更新。正式启动前分别按每类产物数量
乘实测大小核对空间，继续保留至少5GiB运行守护。不自动删除历史实验或缓存腾空间。

## 接入与正式执行

1. 使用已完成的同一FLUX20episode官方报告和来源/尾帧SHA；全140目前停止，不等待其完成。
2. 核验Qwen本地权重、已有环境与采样实现；复用已验证的官方shared-latent注入入口，
   适配完整episode_id与硬切子集选择，避免同run不同窗口碰撞。
3. 先对固定的单人硬切、多人硬切、非人脸硬切和真实接续做接入检查，
   检查模型素材生成、角色绑定、姿态标定、画风、RGBA/RGB处理、峰值显存和实际耗时。
   预检用于发现接线问题，不按评分挑选正式样本或删失败/负例。
4. 冻结20episode子集，生成新Qwen素材、必要3D/标定、硬切首帧与363段本路线视频；
   做完整官方非VLM评价，与同一20episode的FLUX路线比较。
5. 全140为后续规划，需用户另行要求才扩展；按同一冻结协议只复用本Qwen路线已核验的完整20episode，
   每个episode内按原镜头顺序滚动，完成所有2,491shot后统一评测/聚合。

虽然不新增视频消融组，现有Qwen算法内部仍需走官方Control trajectory并通过callback
截取进入step12的latent，再从同一latent继续Treatment；该步骤是已验证实现的一部分。
它不等于依据最终图准备姿态/mask后重生成Treatment。正式视频只使用最终路线首帧。
沿用v7强度0.4、step12按参考cosine 0.40阈值选择1/6步，保持人脸尺寸、闭眼、缺脸、
姿态回检、mask及失败门槛。对不适用角色记录原因，运行故障不伪装成安全回退。

新建输出建议：`outputs/entitybench_qwen21_i2v_a14b_4step_full140/`；
20episode先在独立对应子集目录执行。冻结split、剧本SHA、参考SHA、Gaussian/标定SHA、
首帧配置、代码SHA、角色绑定和视频runtime。整个新路线禁止跨身份源复用旧首帧/视频。

## 视频与官方评测口径

视频配置保持与当前FLUX路线一致：普通Wan2.2-I2V-A14B，LightX2V high/low
`4step_1022`蒸馏LoRA，4步DMD2、shift5、CFG关闭、BF16/torch_sdpa，
81帧、16fps、480p面积。普通I2V接收首帧；不加入StoryMem/EntityMem memory LoRA。
接续来源必须是同episode本路线上一视频实际解码尾帧，同时核验parent video SHA。

使用[evaluate-face-injection-v2](/root/.codex/skills/evaluate-face-injection-v2/SKILL.md)：
整episode两进程评分，独立缓存、输入核验、最后调用官方聚合器一次。
保持官方revision `3776439e7037b2128ac3b5d2d1ce63e424cca166`及现有skip_vlm适配。

主表为六项VBench、角色/物体/场景presence、`cs_face`、`cs_object`及真实接续边界。
官方角色区域crop与CLIP代表图选择/DINO质心不变；旧InsightFace/参考DINO/脸框质心
不填入官方表，也不作为本轮新增必跑项。
VLM继续不测，缺失项保持null，scope为`partial_without_vlm`；fidelity gate不可用，
`cs_face/cs_object`为暂定值。沿用公开代码episode宏平均，与论文gate/实例加权差异明确披露。

比较时分别给20episode和140episode的同范围结果、难度分层、有效/失败/跳过数量，
新旧首帧模型与参考素材来源同时披露。官方`cs_face`是各路线内部跨shot的一致性，
不跨路线把不同身份互相算相似度。论文结果仅作外部参考，素材条件与视频路线差异保留说明。
人工复核单人/多人/动物/接续及漂移例子，与指标完成状态分别记录。

## 本次记录边界

初次编写只核对已有模型/结果及官方资料；随后按用户要求，已在21:03停止FLUX全140，
21:12启动同一20episode的Qwen推理与后续自动流水线。没有下载新权重、升级环境或推送GitHub。
全140规划暂不执行，实际范围见20episode运行说明。

### 素材阶段设备放置验证，2026-10-12 22:59

同种子两个素材候选在CUDA驻留时PNG SHA与sequential offload完全相同，
约21秒/张，allocated约36.9GiB、reserved约37.8GiB；两路model offload吞吐更低。
当前20episode素材阶段已采用单进程CUDA驻留，29份已完成素材保留。
原来的低显存预算仍适用于后续多参考首帧的sequential offload，不代表当前素材GPU驻留占用。
详见20episode运行说明；权重/采样不变，未升级到Turbo、未引入量化或新消融。
