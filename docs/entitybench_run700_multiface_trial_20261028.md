# run700：多人首帧3D注入试验

先测试4:1酒吧三人镜头Stefan、Elias、Walter。已有`run_qwen_image21_shared_prefix_adaptive.py:run_shot`支持多人：在同一个step12预览中检测脸，人物参考与脸做全局一对一身份匹配；每角色独立渲染对应姿态的3D、准备v7内脸mask、决定1或6步局部注入。角色准备失败有单独记录，不把同一3D用于不同人物。

原full140的分流只对单角色硬切启用了这条路径，多人走原生参考；“代码有多人实现”和“上轮实际用了多人注入”是两回事。此次保留v7算法，只扩展首帧worker的参数接入。

Stefan使用原full140的2D＋3D配对；Elias、Walter从已冻结的单人参考补建FaceLift Gaussian和273点/512姿态标定。禁止重生参考身份、跨episode借素材、换演员后继续复用旧Gaussian。

沿用刚完成单人注入整集里4:1的图像提示词、三张图顺序、seed、20步及视频提示词/seed/81帧/16fps。Control由官方Qwen生成，Treatment从同次Control的step12 latent分支，强度0.4。视频两支分别输入各自首帧；其它设置相同。未修改原整集或后继镜头，不作为整集提升结论。

执行：`pretest/qwen140_run700_multiface_trial.py`，依次`assets`、`frame`、`videos`、`evaluate`。GPU阶段串行。没有语义检查重试；技术问题或所有角色无法注入时停止。少于两角色实际注入则停止视频阶段，不能把单脸注入称为多人注入成功。

评测：配对首帧InsightFace、81帧视频InsightFace、参考脸DINO，空间匹配/跟踪评价人物，保留负增益和缺脸覆盖率；它们不同于官方人物区域`cs_face`。完整对照图含每人的原参考、连续3D渲染、对齐3D、最终mask、Control、Treatment。输出`outputs/entitybench_qwen140_run700_multiface_v7_20261028/`。

实际已注入3/3：左Walter、中Stefan、右Elias；无角色跳过。Stefan在step12的cosine为0.2248，注入6步；Walter 0.4891、Elias 0.5931，各1步。原有按脸大小缩放强度继续生效，有效强度分别约0.3063、0.2625、0.2750（按左中右）。

Control原文件为RGBA，上轮原生worker写RGB，原PNG SHA因此不同；规范化RGB后的PNG SHA与上轮4:1完全一致，解码RGB逐像素一致。原始文件保留，核验记录在`control_validation.json`。

两支视频均已生成81帧/16fps，并已抽查0/27/54/80帧，未见上下拼接或角色位置交换。原有Stefan也举杯的动作/道具偏差仍在两版中保留，不能把身份注入的试验解读为调度问题已解决。

生成、配对评分和完整对照图已完成，三角色在两段视频各81/81帧均可评分。三角色平均视频InsightFace为0.3654→0.3562，参考脸DINO为0.3922→0.3969；指标意见不一致，尚无稳定身份收益。Stefan首帧IF 0.3185→0.3516，但视频IF 0.2695→0.2464，首帧收益未延续。全部角色及负增益均保留。

查看[视频及指标](../outputs/entitybench_qwen140_run700_multiface_v7_20261028/index.html)、[完整图证](../outputs/entitybench_qwen140_run700_multiface_v7_20261028/shot_4_1/frame/injection/comparison.jpg)、[逐角色报告](../outputs/entitybench_qwen140_run700_multiface_v7_20261028/report.md)。这是多人首帧能力试验，未替换整集默认策略，不能据此推断整集官方`cs_face`变化。

DINO裁剪阶段省去了不参与该指标的人脸识别/年龄/姿态头，使用同一SCRFD检测权重、640检测尺寸、0.5置信度及15%扩展裁剪；两张采样图的脸框和置信度与完整分析器逐值一致，见`dino_detector_validation.json`。已保存的IF评分复用；未修改特征模型、身份匹配或评分公式。
