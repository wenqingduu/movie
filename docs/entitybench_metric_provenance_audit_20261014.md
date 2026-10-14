# EntityBench 人物一致性／接续边界口径审计

审计日期：2026-10-14。范围：已完成的 SDXL→FLUX、Qwen、Portrait Agent摄影素材→FLUX 三路线，各20集／363镜头。只读核验既有媒体、官方裁剪和评分；不生成新实验组，不扩全量，不调用VLM。原评分、素材、视频、冻结参数和源文件未修改。

**结论：发现了人物定位污染和论文／开源实现的口径差异，没有发现抽查中的DINO数值计算错误。当前 `cs_face` 不能直接解释为脸部身份准确度，`cs_transition_boundary` 不能与论文表格直接排名。人物生成的真实漂移也存在，不能把差距全部归因于评估。**

证据目录：[审计JSON](../outputs/entitybench_paper_flux_qwen_portraitflux_photo20_comparison/review/metric_provenance/audit.json)；[可复算脚本](../outputs/entitybench_paper_flux_qwen_portraitflux_photo20_comparison/review/audit_metric_provenance.py)；[官方源码逐字节验证](../outputs/entitybench_paper_flux_qwen_portraitflux_photo20_comparison/review/metric_provenance/upstream_verification.json)。

## 1. 版本和数值实现

锁定的作者源码revision为 `3776439e7037b2128ac3b5d2d1ce63e424cca166`。本次重新下载该revision的原文件，与本地冻结文件逐字节相同，SHA256为 `2334aad4b61cae2a0ea38cc2ee9ae85125b6df4e128610903654fd2b49607bb4`。

`--skip_vlm`适配器只跳过API阶段；保留人物检测、CLIP阈值、代表裁剪、DINOv2特征、质心计算、boundary公式和原聚合逻辑。没有通过改阈值或裁剪方式抬高分数。

所有60个episode的 `cs_face` 均从保存的每角色均值和appearance数量重新核验，与原episode分数一致。进一步对每路线的Casey、Daniel、Viktor共9组／151张官方canonical crop，使用原官方 `compute_dino_embedding`、原评测Python环境和本地dinov2-base重算：每crop质心相似度的最大绝对误差为 `1.1920928955078125e-7`。

这排除了这些样本中的归一化、质心公式、特征模型或缓存值错误；未声称151张覆盖所有人物裁剪。

源码：[作者固定版本](https://github.com/Catherine-R-He/EntityBench/blob/3776439e7037b2128ac3b5d2d1ce63e424cca166/eval/evaluate_benchmark.py)。相关位置：人物查询1047–1120行、DINO1283行、质心1318行、continuation选择2967行、fidelity gate3380行、boundary3459行、聚合3785行。

## 2. `cs_transition_boundary` 为什么接近1

### 输入、筛选与实际帧

作者开源代码逐scene枚举镜头，仅选scene内后镜头 `cut=False` 且有前镜头的相邻对。虽然配置写 `n_boundary_frames=3`，实际headline调用固定 `n_boundary=1`，比较**前视频末帧与后视频首帧的全画面DINOv2 CLS余弦**。不比较三帧平均，不裁人物，不把hardcut算入。

三路线分别从原benchmark script的cut重新构建选择集合，均与生成依赖及评测集合相同：每路线172对，19个episode有可评分边界。第20个episode没有continuation，保留null而不填零。

独立解码所有516对：后镜头实际I2V输入逐像素等于自己的前镜头第80帧，同时等于保存的尾帧。评测读取的后镜头第0帧也与此前末帧很接近；它们不是完全相同的文件或像素，516对中精确边界像素相同的为0。

| 路线 | 原boundary episode宏平均 | boundary RGB平均绝对误差（0–255） | boundary平均PSNR dB |
|---|---:|---:|---:|
| SDXL→FLUX | 0.9884 | 2.9822 | 36.4505 |
| Qwen | 0.9820 | 2.9161 | 36.7888 |
| 新FLUX | 0.9882 | 2.9611 | 36.4609 |

RGB／PSNR是本次核验输入关系的诊断证据，不加入正式指标。近乎重建同一输入帧的I2V边界自然容易得到高DINO余弦，这是根据实际像素和生成依赖作出的解释。该指标不能说明接续几秒后的身份保持、hardcut人物变化或全故事一致性；原始作者视频未在本机获得，不能断言作者输入帧保留方式与我们完全相同。

### 论文与公开代码没有完全一致

[论文v1附录B.4.2](https://arxiv.org/html/2605.15199v1)的boundary公式也使用上述末／首帧，cut筛选原则一致。但F.2额外说明只评可检测匹配内容的边界，并给出raw与coverage分解；固定开源代码boundary段没有匹配内容检测或fidelity拒绝，不读取人物grounding结果或VLM保真度，不把不可匹配边界计为零。

论文B.4.1还规定跨episode采用实例加权、将gate跳过／失败计零的均值；公开 `aggregate_results` 实际为有效episode值的宏平均，并忽略null。这是另一处明确口径差异。

| 方法（论文Table15） | boundary raw | coverage | Table14主表 |
|---|---:|---:|---:|
| EntityMem | 0.738 | 1.00 | 0.738 |
| StoryMem | 0.795 | 0.83 | 0.663 |
| HoloCine | 0.509 | 0.98 | 0.498 |
| CineTrans | 0.508 | 1.00 | 0.508 |

Table15为舍入后的数值，乘积不必精确等于主表。即便比较raw，作者最高约0.795仍低于本轮约0.99，**不能用缺少gate单独解释全部差距**；I2V边界保留机制、不同视频生成路线和20／140集样本范围也需要区分。

## 3. `cs_face` 的低分：定位污染与真实外观漂移

### 这个指标实际测什么

GroundingDINO根据角色文字描述找区域，CLIP匹配描述，在5个采样帧的候选中按CLIP×清晰度×面积选一个canonical crop。只将status=present的角色裁剪纳入，角色在episode内出现至少2次后，用DINOv2人物区域特征到自身外观质心的余弦计算分数。它没有人脸必需条件，也没有使用本项目InsightFace身份参考。

因而衣服、人物全身／上半身／特写裁剪、背影、背景和画风变化都可能降低分数。角色区域没有脸，也不一定就是评估程序出错；但把油桶或汽车当该人物是明确的定位污染。

### 本次已直接看见的污染

新FLUX的Casey：`4:1`的canonical crop是红色油桶，`9:2`是红色布料，`21:1`是汽车区域；它们被标记present，CLIP分别约0.237／0.234／0.222，均超过官方0.2阈值，质心相似度约0.141／0.169／0.169，进入了正式人物分数。

Qwen的Casey `4:1`也选到主要是油桶／衣物的区域，CLIP约0.295、质心相似度约0.180。旧FLUX和新FLUX的Daniel `16:1`选的是手套／手与门把手，人物脸和身体不在该crop内；两者质心相似度都约0.240。FLUX的Daniel `4:1`选到女性角色，而登记的Daniel是男性。

源码解释了这一风险：GroundingDINO仅用描述前12个词，返回所有通过阈值的框；它并未检验检测短语是不是人，随后把每个框直接赋上请求的角色名称／类型。描述中的红帽、衣服等词可以把物体框带入，再由CLIP及面积／清晰度选择。新FLUX中Casey的查询为“Person wearing a backwards red-and-white baseball cap and a red and brown”；这不是人脸或实例身份约束。缺少VLM fidelity gate使这些已通过present阈值的错误裁剪继续进入人物质心。

| 全20集人物定位记录 | SDXL→FLUX | Qwen | 新FLUX |
|---|---:|---:|---:|
| scheduled shot-character | 726 | 726 | 726 |
| present | 635 | 685 | 620 |
| present但CLIP在0.20–0.25 | 298 | 204 | 290 |
| 同shot内多角色canonical crop逐像素相同的组数 | 11 | 6 | 10 |
| 最终进入cs_face的appearance | 612 | 664 | 591 |

低CLIP及同crop多角色是风险提示，不等于上述所有样本错误：同框可能含多个人。这里只人工审查9组／151张crop，尚未给出全cohort定位误检率。present与cs_face样本数不同，是因为只有episode内至少2次appearance的角色进入质心指标。

### 真实生成问题仍然存在

9组裁剪也显示跨hardcut的演员外观、服装和画风变化，以及FLUX多人未绑定参考导致的角色错配。即使正确裁剪到人，人物外观也不总是稳定。Viktor等样本还会在近景／远景、背影、不同照明和不同角色区域之间变化。官方整个人物DINO指标对这些差异敏感，无法把它们与脸部身份变化完全分离。

| 抽查角色的官方原均值 | SDXL→FLUX | Qwen | 新FLUX |
|---|---:|---:|---:|
| Casey（run146） | 0.4219 | 0.4968 | 0.4335 |
| Daniel（run44 i70） | 0.5137 | 0.5532 | 0.4803 |
| Viktor（run719） | 0.6476 | 0.6326 | 0.6926 |

这些是原本的角色分数，不是删除可疑框后的修正分数。抽查发现足以否定“低cs_face完全证明人脸差”，但不足以否定真实漂移，或量化总分差距中定位错误所占比例。

## 4. 聚合、gate和论文对比的限制

| `cs_face` | SDXL→FLUX | Qwen | 新FLUX |
|---|---:|---:|---:|
| 原官方开源episode宏平均 | 0.6853 | 0.6958 | 0.6836 |
| 同一缓存按appearance加权的raw诊断值 | 0.6486 | 0.6700 | 0.6484 |

长episode的角色分数更低，改成实例加权反而降低本轮分数。**episode宏平均差异并不能解释为何本轮比论文低**；也不能把上表第二行当作完整论文复现，因为没有fidelity gate／gate失败分母。

论文B.4.1描述基于0.5 fidelity阈值的硬过滤，但F.2又描述DINO指标在embedding similarity层处理gate、`cs_face` coverage为1，且没有给出该相似度修正的可复现代码；公开代码使用硬过滤、只平均已纳入crop，最终宏平均。论文文字内部及公开实现之间存在未解析的不一致。因此仅补VLM也不能保证精确复现论文表格，更不能手工删除几个差框后称为官方修正结果。

当前保留原分数，将跨论文比较明确标为**不可严格横比**。后续若要复现论文，应先取得作者表格对应的评测revision、matched-content boundary gate、DINO fidelity处理与实例分母；若要判断3D插件的脸部身份收益，应另列参考锚定的人脸身份诊断和检测覆盖率，不混进官方 `cs_face`。本次不自动启动这两类新评测。

## 5. 查看原始裁剪

九张图均已实际查看；每张包含该角色进入原质心的全部canonical crop，标注shot、原质心相似度、CLIP。原PNG和SHA在审计JSON中。

| 角色 | SDXL→FLUX | Qwen | 新FLUX |
|---|---|---|---|
| Casey | [27张](../outputs/entitybench_paper_flux_qwen_portraitflux_photo20_comparison/review/metric_provenance/flux_sdxl20_Casey_official_crops.jpg) | [37张](../outputs/entitybench_paper_flux_qwen_portraitflux_photo20_comparison/review/metric_provenance/qwen20_Casey_official_crops.jpg) | [26张](../outputs/entitybench_paper_flux_qwen_portraitflux_photo20_comparison/review/metric_provenance/flux_native20_Casey_official_crops.jpg) |
| Daniel | [13张](../outputs/entitybench_paper_flux_qwen_portraitflux_photo20_comparison/review/metric_provenance/flux_sdxl20_Daniel_official_crops.jpg) | [12张](../outputs/entitybench_paper_flux_qwen_portraitflux_photo20_comparison/review/metric_provenance/qwen20_Daniel_official_crops.jpg) | [13张](../outputs/entitybench_paper_flux_qwen_portraitflux_photo20_comparison/review/metric_provenance/flux_native20_Daniel_official_crops.jpg) |
| Viktor | [8张](../outputs/entitybench_paper_flux_qwen_portraitflux_photo20_comparison/review/metric_provenance/flux_sdxl20_Viktor_official_crops.jpg) | [9张](../outputs/entitybench_paper_flux_qwen_portraitflux_photo20_comparison/review/metric_provenance/qwen20_Viktor_official_crops.jpg) | [6张](../outputs/entitybench_paper_flux_qwen_portraitflux_photo20_comparison/review/metric_provenance/flux_native20_Viktor_official_crops.jpg) |

执行本次审计耗时121.35秒，单个DINO进程／GPU0、4CPU线程、本地离线权重，VLM调用0；没有完整重跑评分。
