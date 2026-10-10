# EntityBench 20 episode 评估说明与首帧对比图索引

更新：2026-10-10（北京时间）。这是已有实验结果与展示图的交接说明；评测执行规则仍以
`/root/.codex/skills/evaluate-face-injection/SKILL.md` 为准。项目根目录为 `/root/autodl-tmp/movie`。

## 当前完成范围

冻结清单覆盖 20 episode（Easy 11、Medium 6、Hard 3）的 363 个官方镜头，其中 132 个单人
镜头进入 PuLID-FLUX 和 SDXL/IP-Adapter 两路线，共 264 组 Control/Treatment 配对。
本轮首帧和 Wan 视频评测均已完成：528/528 个视频任务，失败 0；408 个新生成、44 个经输入/
输出哈希及参数验证后历史复用、76 个安全回退 Treatment 复用对应 Control。
2026-10-10 读取 `progress.json` 为 `complete`，tmux 无运行会话。

视频为 Wan2.2-TI2V-5B、1280×704、49 帧、24 fps、50 steps、shift 5.0、guidance 5.0，
每镜头独立短视频约 2.04 秒。Control/Treatment 同 prompt、seed、模型、资产与生成参数，
仅首帧局部 v7 residual 注入不同。身份素材是 SDXL Base 1.0 生成的写实参考图，再构建 FaceLift
Gaussian；素材不是通用后端默认的 Juggernaut。

EntityBench validated 完整清单为 140 episode、2,491 镜头、847 个单人镜头。
本轮是子集评测，不包含该 20 episode 的全部多人/无人镜头，不包含跨镜头历史记忆或拼接完整
episode 视频；VLM/LLM API 指标未运行。Qwen 当前官方 Control 分叉的三 episode 实验另见
[`../EPISODE_TEST_RESULTS.md`](../EPISODE_TEST_RESULTS.md)和交接文档第 9 节，不混入下面的表。

## 四项指标与统计口径

| 路线 | 单人镜头 | 注入 / 安全复用 | 首帧 InsightFace C→T (Δ) | 首帧 DINOv2 C→T (Δ) | 视频 InsightFace mean C→T (Δ) | 视频 DINOv2 mean C→T (Δ) |
|---|---:|---:|---|---|---|---|
| PuLID-FLUX | 132 | 104 / 28 | 0.6825→0.7160 (+0.0335) | 0.7513→0.7685 (+0.0172) | 0.5697→0.5852 (+0.0155) | 0.6522→0.6624 (+0.0102) |
| IP-Adapter | 132 | 84 / 48 | 0.0831→0.1997 (+0.1166) | 0.4432→0.4793 (+0.0360) | 0.0782→0.1653 (+0.0870) | 0.4165→0.4425 (+0.0261) |

以上仅统计实际注入、两侧均可评分的配对。PuLID 四项有效对均为 104；IP 首帧两项为 81、
视频两项为 84（3 个注入镜头最终首帧缺少可靠人脸框）。视频先对可评分帧取 mean，
再按镜头等权取 mean；DINOv2 使用 `video_frame_mean`，不是 shot median。缺失帧保留 null，
进入检测覆盖率分母，不填 0。回退镜头保留在正式清单，未进入“实际注入均值”。
首帧和视频有效集合不完全相同，不能直接用总均值增量相除宣称严格增益保留率。

InsightFace 以原身份参考为锚点。视频从冻结首帧 bbox 做空间跟踪，缺 bbox 时从最大脸开始，
后续选最近中心，不按最高身份 cosine 挑脸；DINOv2 使用相同跟踪 bbox 和 0.15 扩展裁剪。
负例仍保留：PuLID 首帧 IF 86 正/18 负、DINO 81 正/23 负；IP 首帧 IF 69 正/12 负、
DINO 63 正/18 负。IP 的正增量较大，但 Treatment 绝对身份分数仍较低。

**首帧 DINOv2 已评测。** 旧视频 Markdown 漏了此列，旧视频 JSON 路线字段也缺
`firstframe_dinov2`，应从首帧 JSON 的 `routes.<route>.reference_dinov2` 读取。
不要把列缺失解释为实验未完成，也不要用三 episode 指标补进 20 episode 表。

## 权威结果文件

以下路径相对于 `outputs/entitybench_firstframe_pulid_ip_20ep/`：

| 文件 | 用途 |
|---|---|
| `manifest.json` | 20 episode 的冻结清单、资产与实验目录索引 |
| `evaluation/firstframe_evaluation.json` | 首帧四个 C/T 分数与两项 Δ、bbox、原图路径；路线/难度/episode/逐 shot 聚合 |
| `evaluation/firstframe_evaluation.md` | 首帧 InsightFace 与 DINOv2 总表 |
| `video_evaluation/video_evaluation.json` | 视频路线指标与 264 组明细；当前含 kind/scope/tracking_policy/routes/shots，无 episodes 字段 |
| `video_evaluation/video_evaluation.md` | 旧视频总表，仅三项，首帧 DINOv2 要另读首帧报告 |
| `video_evaluation/cases.json` | 40 个 case 的 manifest/报告路径索引，不是逐帧评分文件 |
| `video_evaluation/cases/<episode_id>/<route>/video_identity.json` | 逐帧 IF、bbox、覆盖率及 first/mean/p10/last |
| `video_evaluation/dinov2_video.json` | 参考图锚定视频 DINOv2、40 个 case 和 episode/路线聚合 |
| `video_evaluation/videos/<episode_id>/<route>/shot_<scene>_<shot>/{control,treatment}.mp4` | 528 个正式视频文件 |
| `video_evaluation/{progress,wan_report,wan_manifest,first_frame_hashes}.json` | 完成状态、任务报告、生成参数和哈希 |

route 目录名为 `pulid_flux` 或 `ip_adapter`。隔离并行测速目录 `video_evaluation/parallel_probe/`
不属于正式 528 个任务。双进程测试约 365 秒/条，近期单进程约 184 秒/条，未见吞吐收益，
因此正式生成保持单进程。当前协调器新增了汇总字段；若以后只执行 `summarize`，会扩展旧报告
结构，不应先假设新字段已在旧文件里存在。

## 对比图格式

### 面板顺序和标签

必须是 **2 行 × 3 列**，从左到右、从上到下：

| 上排左 | 上排中 | 上排右 |
|---|---|---|
| 原始身份参考图 | 连续 FaceLift 3D 渲染 | 调色并对齐后的纯 3D reference |
| **下排左：最终 v7 注入 mask** | **下排中：Control 最终首帧** | **下排右：Treatment 最终首帧** |

- 标题必须有路线、run/episode、shot、角色；多人按角色标注注入/跳过状态。
- Control/Treatment 标签分别给出实际 IF、DINOv2 值，底部显示两项 `Control→Treatment (Δ)`，
  至少 4 位小数。仅显示增量不足以判断身份质量。
- 本会话精选 PNG 为 1800×1080，明确标注 `FIRST FRAME`，只展示首帧指标。
  完整首帧/视频评估图需另加视频 IF 与 DINOv2 mean；不能把首帧数字标成视频数字。
- 面板必须来自同一个配对实验的原始文件，只做等比例缩放排版，不用图像生成模型重绘、
  修脸或改变像素内容。全图展示要保留场景；需要脸部放大时另加 C/T 等倍率裁剪，说明 bbox 来源。
- mask 必须是最终实际注入 mask，保留软边，不归一化成更大的区域；不能用中期 `pred_x0`、
  原始 3D reference mask、皮肤估计 mask 或手绘形状替换。
- 多人使用对应角色的 reference/render，显示合并最终 mask 并列出逐角色指标；部分注入须注明
  每个跳过角色及原因。整镜头回退仍显示相同 C/T、明确原因，缺失 render/mask 用占位说明，
  缺失分数写 N/A，不能填 0。

**旧图的格式缺口：** 本会话检查的 PuLID/IP 原实验 `comparison.jpg` 第四格是 step-30
`pred_x0`，缺少最终 mask 和分数。此次只修复精选展示图，未覆盖原图，也未修复全部
264 组 canonical 图。因此生成/指标完成与全部可视化契约完成应分别陈述。

### 原始面板文件映射

`pair_dir` 从正式首帧 JSON 的 `result_path` 父目录取得，避免凭 run 名猜路径（部分 episode
复用历史目录）。以下为本会话精选的 step-30 实验映射；其他实验以其实际 fork/detection step
及元数据为准。

| 面板 | PuLID：相对于 pair_dir | IP：result.json 的 paths 字段 |
|---|---|---|
| 身份参考 | `input/identity_reference.png` | `reference_portrait` |
| 连续 3D | `step_30/rendered_3d_face.png` | `continuous_3d_render` |
| 调色对齐 3D | `step_30/harmonized_3d_face.png` | `scale_matched_3d_layout` |
| 最终 mask | `step_30/final_inner_face_mask.png` | `target_face_mask`（指向 `input/harmonization/final_injection_mask.png`） |
| Control | `control/final.png` | 正式行的 `control_image`，或 `paths.branches.ip_adapter_baseline` |
| Treatment | `treatment/final.png` | 正式行的 `treatment_image`，或 `paths.branches.ip_adapter_plus_pulid_style_residual` |

首帧分数直接读取 `evaluation/firstframe_evaluation.json` 的对应行，字段为
`insightface_{control,treatment,delta}`、`dinov2_{control,treatment,delta}`，不要混用
`rendered_3d_*` cosine 或不同实验的历史数字。

## 已筛选批次与继续筛选

三批共 20 张，都是实际注入且两项首帧增量为正的展示样本；其中后两批共 14 张经过肉眼检查为
非动漫写实风格。它们不是随机样本，不用于重算正式总体均值，也不代表全量写实子集。

| 根目录下的子目录 | 数量 | 用途 |
|---|---:|---|
| `selected_positive_firstframes/` | 6 | 首批大增益，4 张明显偏动画 |
| `selected_realistic_positive_firstframes/` | 6 | 第一批写实正例，PuLID/IP 各 3 张 |
| `selected_realistic_positive_firstframes_batch2/` | 8 | 第二批写实正例，PuLID/IP 各 4 张 |

每批 `selection.json` 包含原始正式行、`figure` 和 `panel_sources` 六个路径；`README.md` 包含
表格和图链接；`render_examples.py` 可从已有实验素材重新排版。继续筛例子要先读取三批索引，
按 `(route, run, shot_key)` 去重；相同镜头的不同路线属于不同配对。

只刷新已有展示图（无需 GPU，不执行扩散/视频生成）：

```bash
cd /root/autodl-tmp/movie
.venv/bin/python outputs/entitybench_firstframe_pulid_ip_20ep/selected_positive_firstframes/render_examples.py
.venv/bin/python outputs/entitybench_firstframe_pulid_ip_20ep/selected_realistic_positive_firstframes/render_examples.py
.venv/bin/python outputs/entitybench_firstframe_pulid_ip_20ep/selected_realistic_positive_firstframes_batch2/render_examples.py
```

接着选新例子：从正式首帧 JSON 筛实际注入且两项可评分的配对，读取 C/T/Δ；正例展示可要求
两项 Δ>0，再逐张检查写实/动画风格、bbox、可见身份改善、肤色、年龄感、表情和 mask 边界。
风格与可见质量不能单凭提示词或分数判断。查看后不适合展示的样本只从展示名单移出，
正式 benchmark 中必须保留。

第二批视觉检查排除的 IP run1775/7:1、run444/4:1 保存在
`selected_positive_firstframes/realistic_review/excluded/`，该目录 README 写明因风格化未展示。
`realistic_review/contact_*.jpg` 是候选 C/T 联系表，不是标准六面板评估图。

## 动漫风格与指标解释

已经核对官方提示词及实际 config/result：PuLID run53/3:1 明确写
`In an animated medium shot ...`；同路线 run53/3:3 没有明确动画要求。
IP run146/28:1 只写 Nia 起床，run44/11:1 只写 Daniel 在办公室走廊，也没有动画要求。
PuLID 使用官方 prompt；IP 的 `_sdxl_prompt` 调整动作/实体定义顺序以应对 CLIP-77，
没有统一加动漫词或强制写实。未明确风格的镜头仍可能生成动画，具体成因尚未做受控实验，
本轮逐镜头独立生成，不能解释成上一镜头记忆传递。

写实身份/3D 参考注入动画脸时可能提高 IF、DINOv2，同时改变画风；写实正例也可能出现
皮肤平滑、年龄感变化、表情和肤色变化。两项正增量不是整体视觉质量保证。
目前尚未给全量 132 个单人镜头做风格标注，也没有正式“只统计写实”的总体均值。
用户若要该结论，需先完整标注并报告有效对、回退、正负例和全部四项绝对值/差值，
不得以精选正例代替正式子集。

## 跨镜头外观质心相似度

新增入口 `pretest/evaluate_cross_shot_face_centroid.py`，只读取冻结的首帧、视频和现有
角色映射/空间跟踪人脸框；不重新生成图像或视频，不改动上面的四项身份指标。
输出独立保存在 `outputs/entitybench_firstframe_pulid_ip_20ep/cross_shot_evaluation/`。

已完成 264 组首帧/视频配对的特征提取、两种统计口径及 88 张跨镜头审计图。
以下数值均来自该目录的 `face_centroid_evaluation.json`，有效数的单位为角色-shot：

| 口径 | 路线 | 首帧有效数 | 首帧 Control→Treatment (Δ) | 视频代表帧有效数 | 视频代表帧 Control→Treatment (Δ) |
|---|---|---:|---|---:|---|
| 全部单人，含安全复用 | PuLID-FLUX | 85 | 0.8925→0.8958 (+0.0033) | 91 | 0.8286→0.8338 (+0.0052) |
| 全部单人，含安全复用 | IP-Adapter | 60 | 0.7783→0.8067 (+0.0284) | 78 | 0.7565→0.7647 (+0.0083) |
| 仅实际注入，重算质心 | PuLID-FLUX | 78 | 0.9120→0.9159 (+0.0039) | 78 | 0.8581→0.8636 (+0.0055) |
| 仅实际注入，重算质心 | IP-Adapter | 58 | 0.7836→0.8135 (+0.0299) | 61 | 0.7856→0.7946 (+0.0091) |

每路线 132 个单人 shot 涉及 54 个 episode-local 角色，其中 29 个只出现一次；
25 个重复出现角色覆盖 103 个 shot，成为全量口径的潜在有效集合。
配对缺脸或不足两个镜头后，首帧 PuLID/IP 分别评分 20/16 个角色，视频分别为 22/20 个。
首帧角色均值正/负数为 PuLID 10/10、IP 13/3；视频为 PuLID 10/12、IP 10/10。
全部负例保留。首帧/视频及全量/仅注入集合不同，不能据这四组增量直接计算严格保留率。

抽查了首帧和视频正负例，包括 run1517 Jin-woo/Hyeon-sik、run53 Rico、run44 Julian、
run354 Clara、run146 Casey/Nia、run893 Chloe。部分正例仍伴随动画向写实变化、鼻部异常、
平滑或年龄感变化；质心正增量不等于整体视觉质量提升，也不保证指定身份正确。
独立复算了 151 个可评分组（包含两种口径和两种模态），校验了 1,122 个源文件 SHA256，
安全回退 C/T 媒体哈希相同；7 项回归测试通过。续跑完整复用 1,056 条 appearance 特征记录，
分数与首次完成的报告一致；结果见 `validation.json`。

该指标采用 EntityBench 的 DINOv2 质心公式：每个镜头提取一个单位 CLS 向量，按
`路线 + episode + 角色` 分组；向量均值再归一化为质心，各镜头与质心计算 cosine。
Control/Treatment 分别计算自己的质心，且先取两侧都可测量的相同镜头集合；
每个角色至少需要两个不同镜头，同名角色不能跨 episode 混算。
质心包含被评分镜头自身，分数受镜头数量影响，不应直接与参考图锚定分数或官方表格比较。

这是**项目的人脸裁剪质心指标，不是官方 `cs_face`**：复用已有首帧 bbox 和空间跟踪 bbox，
采用 0.15 扩展裁剪，没有重跑 GroundingDINO 全角色定位、CLIP 筛选或 VLM fidelity gate。
报告首帧与视频代表帧两项；视频从 5 个等距采样帧按清晰度 × 人脸面积选择一个代表帧，
不按身份相似度选帧，且不是此前视频 `frame_mean` 的另一种名称。

主口径包含全部单人镜头，包括安全复用；另列仅注入镜头子集，该子集独立重算质心。
路线和 episode 按可评分角色-shot 等权汇总；同时保存逐角色、逐镜头、最偏离/最代表镜头、
角色内 pairwise median、缺失原因和覆盖率。缺脸、仅出现一次或配对后不足两镜头均写 null，
不填 0。回退镜头像素相同也可能因为两侧质心变化而产生非零质心分数 Δ，不能强制置零。

运行及续跑（当前低内存 CPU 配置；有 GPU 时可改 device 和 batch size）：

```bash
cd /root/autodl-tmp/movie
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python pretest/evaluate_cross_shot_face_centroid.py \
  --device cpu --cpu-threads 1 --batch-size 1
```

`face_centroid_evaluation.json` 保存全量/仅注入两种口径、路线/难度/episode/角色/镜头明细、
两侧质心向量、裁剪图路径、视频选帧记录及源文件/模型 SHA256；Markdown 为独立总表和审计图索引。
`features/` 的缓存由模型、裁剪参数、源文件内容和 bbox/跟踪记录共同确定，允许安全续跑；
`crops/` 与 `audit/` 保存实际评分的脸部裁剪和跨镜头 C/T 联系表，原有 canonical comparison 不覆盖。
本轮新增质心评测的 DINOv2 特征提取和质心汇总均使用 **CPU**，实际参数为
`--device cpu --cpu-threads 1 --batch-size 1`。本轮执行环境无可用 GPU，内存为 2 GB、
CPU 配额约 0.5 核；该说明仅针对此次新增评测，不代表此前图像/视频生成或原四项评测的设备。
特征提取与缓存汇总分开，缓存命中时不加载 Torch/Transformers 模型；上述线程设置可避免
在该环境中汇总时过度分配资源。

完整报告：[Markdown](../outputs/entitybench_firstframe_pulid_ip_20ep/cross_shot_evaluation/face_centroid_evaluation.md)、
[JSON](../outputs/entitybench_firstframe_pulid_ip_20ep/cross_shot_evaluation/face_centroid_evaluation.json)。

## Git 与本机产物

20 episode 首帧入口已提交为 `b27393f`，视频入口/空间跟踪/哈希续跑与原交接文档已提交并推送为
`65a07c2`。2026-10-10 本地 `main` / `origin/main` 为 `a62e5b4`。
本轮 Git 提交范围为质心评测脚本、回归测试、评估说明和交接文档。
`outputs/` 下全部原始素材、视频、报告、精选图与复排版脚本
均被 Git 忽略，只有本机有这些文件；GitHub checkout 不会自动带回它们。
用户的 `docs/frontend_agent_architecture.md` 改动及未跟踪 `experiment_output/` 必须保留。
本轮新增质心评测脚本、测试及独立报告，并更新说明；没有重新运行生成或修改原四项指标分数。

## 已展示 20 张图的完整索引

| 批次 | 路线 / 镜头 / 角色 | 首帧 InsightFace C→T (Δ) | 首帧 DINOv2 C→T (Δ) | 六面板图 |
|---|---|---|---|---|
| 首批混合风格 | PuLID-FLUX / run53 3:3 / Kaito | 0.2638→0.7601 (+0.4963) | 0.3504→0.6439 (+0.2935) | [打开](../outputs/entitybench_firstframe_pulid_ip_20ep/selected_positive_firstframes/pulid_run53_shot_3_3.png) |
| 首批混合风格 | PuLID-FLUX / run53 3:1 / Kaito | 0.4812→0.7360 (+0.2547) | 0.4442→0.6323 (+0.1881) | [打开](../outputs/entitybench_firstframe_pulid_ip_20ep/selected_positive_firstframes/pulid_run53_shot_3_1.png) |
| 首批混合风格 | PuLID-FLUX / run146 24:1 / Nia | 0.4773→0.6035 (+0.1261) | 0.6504→0.6942 (+0.0438) | [打开](../outputs/entitybench_firstframe_pulid_ip_20ep/selected_positive_firstframes/pulid_run146_shot_24_1.png) |
| 首批混合风格 | IP-Adapter / run146 28:1 / Nia | -0.0710→0.3166 (+0.3876) | 0.4464→0.6140 (+0.1676) | [打开](../outputs/entitybench_firstframe_pulid_ip_20ep/selected_positive_firstframes/ip_adapter_run146_shot_28_1.png) |
| 首批混合风格 | IP-Adapter / run1775 2:1 / Nora | 0.0638→0.4192 (+0.3554) | 0.6118→0.6558 (+0.0440) | [打开](../outputs/entitybench_firstframe_pulid_ip_20ep/selected_positive_firstframes/ip_adapter_run1775_shot_2_1.png) |
| 首批混合风格 | IP-Adapter / run44 11:1 / Daniel | -0.0595→0.2795 (+0.3390) | 0.5723→0.6815 (+0.1093) | [打开](../outputs/entitybench_firstframe_pulid_ip_20ep/selected_positive_firstframes/ip_adapter_run44_shot_11_1.png) |
| 写实第一批 | PuLID-FLUX / run146 4:1 / Casey | 0.5513→0.6625 (+0.1112) | 0.7892→0.7961 (+0.0068) | [打开](../outputs/entitybench_firstframe_pulid_ip_20ep/selected_realistic_positive_firstframes/pulid_run146_shot_4_1.png) |
| 写实第一批 | PuLID-FLUX / run755 3:4 / Dante | 0.6733→0.7511 (+0.0778) | 0.6938→0.7022 (+0.0084) | [打开](../outputs/entitybench_firstframe_pulid_ip_20ep/selected_realistic_positive_firstframes/pulid_run755_shot_3_4.png) |
| 写实第一批 | PuLID-FLUX / run728 2:2 / Marcus | 0.6831→0.7503 (+0.0672) | 0.7057→0.7312 (+0.0254) | [打开](../outputs/entitybench_firstframe_pulid_ip_20ep/selected_realistic_positive_firstframes/pulid_run728_shot_2_2.png) |
| 写实第一批 | IP-Adapter / run146 24:1 / Nia | 0.1136→0.4037 (+0.2901) | 0.5469→0.5991 (+0.0522) | [打开](../outputs/entitybench_firstframe_pulid_ip_20ep/selected_realistic_positive_firstframes/ip_adapter_run146_shot_24_1.png) |
| 写实第一批 | IP-Adapter / run728 2:2 / Marcus | 0.1034→0.3787 (+0.2753) | 0.4528→0.4627 (+0.0099) | [打开](../outputs/entitybench_firstframe_pulid_ip_20ep/selected_realistic_positive_firstframes/ip_adapter_run728_shot_2_2.png) |
| 写实第一批 | IP-Adapter / run1517 6:1 / Hyeon-sik | 0.4216→0.6517 (+0.2301) | 0.7647→0.7850 (+0.0203) | [打开](../outputs/entitybench_firstframe_pulid_ip_20ep/selected_realistic_positive_firstframes/ip_adapter_run1517_shot_6_1.png) |
| 写实第二批 | PuLID-FLUX / run719 4:5 / Viktor | 0.4464→0.5291 (+0.0827) | 0.6584→0.6928 (+0.0344) | [打开](../outputs/entitybench_firstframe_pulid_ip_20ep/selected_realistic_positive_firstframes_batch2/pulid_run719_shot_4_5.png) |
| 写实第二批 | PuLID-FLUX / run44 3:10 / Julian | 0.6908→0.7583 (+0.0675) | 0.8546→0.8766 (+0.0220) | [打开](../outputs/entitybench_firstframe_pulid_ip_20ep/selected_realistic_positive_firstframes_batch2/pulid_run44_shot_3_10.png) |
| 写实第二批 | PuLID-FLUX / run1775 7:1 / Nora | 0.7545→0.8189 (+0.0644) | 0.8668→0.8826 (+0.0158) | [打开](../outputs/entitybench_firstframe_pulid_ip_20ep/selected_realistic_positive_firstframes_batch2/pulid_run1775_shot_7_1.png) |
| 写实第二批 | PuLID-FLUX / run53 4:1 / Cassidy | 0.7184→0.7793 (+0.0609) | 0.8445→0.8734 (+0.0289) | [打开](../outputs/entitybench_firstframe_pulid_ip_20ep/selected_realistic_positive_firstframes_batch2/pulid_run53_shot_4_1.png) |
| 写实第二批 | IP-Adapter / run44 3:10 / Julian | 0.1499→0.2926 (+0.1426) | 0.4407→0.4588 (+0.0180) | [打开](../outputs/entitybench_firstframe_pulid_ip_20ep/selected_realistic_positive_firstframes_batch2/ip_adapter_run44_shot_3_10.png) |
| 写实第二批 | IP-Adapter / run719 2:1 / Viktor | 0.3158→0.4327 (+0.1170) | 0.5512→0.6452 (+0.0939) | [打开](../outputs/entitybench_firstframe_pulid_ip_20ep/selected_realistic_positive_firstframes_batch2/ip_adapter_run719_shot_2_1.png) |
| 写实第二批 | IP-Adapter / run1517 1:4 / Hyeon-sik | 0.3742→0.5330 (+0.1588) | 0.6992→0.7267 (+0.0275) | [打开](../outputs/entitybench_firstframe_pulid_ip_20ep/selected_realistic_positive_firstframes_batch2/ip_adapter_run1517_shot_1_4.png) |
| 写实第二批 | IP-Adapter / run501 2:2 / Clara | 0.0221→0.1150 (+0.0929) | 0.5523→0.6285 (+0.0761) | [打开](../outputs/entitybench_firstframe_pulid_ip_20ep/selected_realistic_positive_firstframes_batch2/ip_adapter_run501_shot_2_2.png) |
