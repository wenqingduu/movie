# 当前20集：只裁脸的DINO跨镜头质心评分

**已完成并独立核验：脸部cs_face为SDXL→FLUX0.8072、Qwen0.8225、新FLUX0.7907。**
各自有效评分20／20／19集，匹配覆盖率39.26%／54.55%／43.94%；共同16集126个appearance为0.8196／0.8323／0.8119。
1,002脸crop全部查看，2张严重相机遮挡排除；1,000可匹配脸部角色-镜头中，只有至少2镜头角色进入质心。
完整表格／裁剪查看页见[结果](../outputs/entitybench_face_only_dino_20ep_20261014/README.md)。全部1,089视频／1,002脸crop来源、60episode评分核验通过。
原人物区域报告和上一轮定位修正结果保持；全140仍停止，无VLM／InsightFace身份特征调用。

2026-10-14用户明确要求只裁脸，不使用InsightFace身份特征。范围为已有SDXL→FLUX、Qwen、新FLUX三路线，各20集／363视频；不生成新视频、不扩全140、不调用VLM。

独立输出 `outputs/entitybench_face_only_dino_20ep_20261014/`，入口 `pretest/evaluate_entitybench_face_only_dino.py`，tmux `entitybench_faceonly_dino20`。
本次已完成，tmux队列已退出；核对status.json与实际进程后仅检查或复用完成结果。旧人物区域修正输出保持在 `outputs/entitybench_cs_face_corrected_character_crops_20ep_20261014/`。

协议冻结在plan.json：

- 使用本地facexlib RetinaFace ResNet50，只做人脸检测，不加载InsightFace／ArcFace身份编码器。FP32、置信度0.9、NMS0.4、原图尺寸，脸框宽／高至少24原生像素。
- 与原评分同五个视频采样点：81帧视频的1／20／40／59／79帧。只裁人脸bbox，边缘增加10%padding；不取人物衣服／全身，不作人脸对齐。
- 角色来自原benchmark schedule。使用原canonical人物区域的实际位置关联脸框，可使用weak区域作空间锚点；单角色可用唯一脸回退。跨五采样点只按空间运动关联，不按DINO／身份相似度选人。多人同脸轨迹冲突、空间多解、漏检／小脸保留缺失原因。
- 一角色／一镜头最多一个脸部canonical crop；按检测置信度×原清晰度项×原面积项选择，与DINO相似度无关。
- 原dinov2-base权重、原评测Python环境、作者固定源码的CLS单位特征、归一化角色质心、每episode appearance池化公式不变。角色至少有2个有效镜头才计跨镜头分数。
- 保存各路线检测／空间匹配覆盖率及共同角色／镜头交集，单列episode宏平均和实例加权raw；缺失不悄悄填0。

先提取、保存与目视检查脸部crop，再冻结检查决策并计算DINO。评测不以当前DINO分数筛选脸，不因为角色变样而删除脸。
本版本仅改变为脸部输入，不复现作者原人物区域／VLM gate的完整表格口径。原官方报告不覆盖，脸部版本单独命名和查看。

提取／评分各使用单GPU进程、4CPU线程，保留逐完整episode缓存及输入SHA。前后核验原视频、原人物裁剪和原官方报告；全140继续paused_by_user。
