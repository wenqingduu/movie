# 当前实验：Qwen＋3D注入＋Wan A14B四步

更新：2026-10-21（北京时间）。仓库：`/root/autodl-tmp/movie`。

本次交接只包含 **Qwen＋3D注入＋Wan A14B四步，140集／2491镜头** 的最新设置、最终结果及指标解释。旧实验仅留[历史引用](docs/history/README.md)，不纳入本次交接，不要求接手会话通读。

## 当前状态

北京时间2026-10-19 21:57完成视频生成、官方非VLM评测、独立脸部DINO评分及归档；生成失败0。140集拼接视频已保存，1355个接续依赖均通过自身前一视频实际解码尾帧核验。

主状态为 `complete_non_vlm_pending_full_video_visual_review`。全部人脸裁剪已审阅，**全部成片的人工质量审阅尚未完成**。原队列正常退出0，不要重复启动生成或评分。

当前先不处理VLM：调用0、39项指标保持null；不接入API、不启动付费评分。

## 实验设置

| 项目 | 本次固定设置 |
|---|---|
| 数据范围 | EntityBench完整140-ID split，140集／2491镜头；easy／medium／hard为80／40／20集 |
| 镜头关系 | 1136个硬切、1355个接续；保持benchmark prompt、cut标记和镜头次序 |
| 首帧来源 | 339个单人v7路线硬切、797个基础／参考硬切；逐镜头实际注入／回退以生成记录为准；接续直接使用自身前一视频的尾帧 |
| 图像模型 | Qwen-Image-2.1，revision `790c92633540aa0cb11d9abf19eb46d861714758` |
| 人物素材 | 1024×1024、40步；单角色硬切所需参考3候选，其余1候选，沿用冻结prompt模板 |
| 3D与标定 | 参考对应的FaceLift Gaussian，273点／512标定；episode-local角色键，参考／Gaussian SHA绑定 |
| 首帧生成 | 1024×576、20步、sequential CPU offload、KV cache；共享latent的Control分叉，step12 pred_x0准备3D／mask |
| 单人3D注入 | v7 color-safe，强度0.4；身份cosine阈值0.40，低于阈值注入6步，否则1步；最小人脸置信度0.5 |
| 多人硬切 | Qwen原生多参考，无多人3D注入；非人脸角色按冻结分流使用基础首帧 |
| 视频模型 | Wan2.2-I2V-A14B，revision `206a9ee1b7bfaaf8f7e4d81335650533490646a3` |
| 视频加速 | LightX2V＋high／low `4step_1022`蒸馏LoRA；DMD2四步，时间步 `[1000, 750, 500, 250]`，shift5、guidance1.0（CFG关） |
| 视频规格 | 81帧／16fps，480p面积上限399360像素；BF16、torch_sdpa，不量化、不使用feature caching |
| 接续规则 | 只用本路线自身前一视频实际解码尾帧；不再次编辑或注入，不跨路线借用媒体 |
| 官方评测 | revision `3776439e7037b2128ac3b5d2d1ce63e424cca166`，strict、skip_vlm、无config覆盖 |
| 评测聚合 | 完整episode分片，GPU0两进程、每路4CPU线程；完整split经官方反序列化器／聚合器重算，集间等权 |
| 独立脸部DINO | RetinaFace、五采样点、脸框10% padding、不对齐、空间匹配、DINOv2单位CLS与角色质心；每角色至少两个有效镜头 |

生成及评测参数以输出根目录的 `manifest.json`、`generation_manifest.json` 和冻结runtime为准。没有另生成Control视频消融组；本表是单路线绝对结果，不代表插件因果增益。

## 最终指标与论文对照

本次结果来自[最终验收报告](outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140/archive/completion_20261019.md)；论文数值来自[EntityBench附录F.1，表12–14](https://arxiv.org/html/2605.15199v1#A6.SS1)。箭头表示指标的优化方向，跨论文数值仅作参照。

| 指标 ↑ | 本次最终结果 | EntityMem（论文方法） | StoryMem | HoloCine | CineTrans |
|---|---:|---:|---:|---:|---:|
| 主体一致性 | **0.9362** | 0.8810 | 0.7590 | 0.8600 | 0.9680 |
| 时间稳定性 | **0.9858** | 0.9760 | 0.8380 | 0.9570 | 0.9790 |
| 运动平滑度 | **0.9917** | 0.9880 | 0.8490 | 0.9640 | 0.9900 |
| 动态程度 | **0.4853** | 0.6570 | 0.5620 | 0.7210 | 0.6880 |
| 美学质量 | **0.5817** | 0.5930 | 0.4750 | 0.5180 | 0.5960 |
| 成像质量（0–100） | **59.6517** | 66.0000 | 56.4100 | 49.9700 | 68.5700 |
| 角色出现率 | **0.9212** | 0.9670 | 0.8490 | 0.8820 | 0.7960 |
| 物体出现率 | **0.8942** | 0.8880 | 0.8930 | 0.7230 | 0.7760 |
| 场景出现率 | **0.6303** | 0.6870 | 0.6810 | 0.6240 | 0.6510 |
| 跨镜头人物区域DINO `cs_face` † | **0.7155** | 0.7370 | 0.7920 | 0.7510 | 0.7720 |
| 跨镜头物体DINO `cs_object` † | **0.8143** | 0.7980 | 0.8390 | 0.8030 | 0.7940 |
| 接续边界一致性 † | **0.9822** | 0.7380 | 0.6630 | 0.4980 | 0.5080 |
| 自定义脸部裁剪DINO ‡ | **0.8237** | — | — | — | — |

† **仅作数值参照，不能严格排名**：本次未运行VLM及保真度门控，采用公开代码的集等权聚合；论文采用门控修正口径，边界筛选也存在差异。官方 `cs_face/cs_object` 为暂定非VLM结果，scope为 `partial_without_vlm`。论文EntityMem同样使用尾帧接续；尾帧输入保留可解释我们的边界绝对高分，但不足以解释与论文0.7380的差距，也不代表整体故事一致性。

‡ 脸部裁剪DINO与论文的人物区域 `cs_face` 输入和可评分集合不同，不能直接横比；它不是InsightFace或对参考身份的相似度。39项VLM指标本次未测。

从数值看，本次时间稳定、运动平滑，但动态程度偏低；人物区域一致性低于论文四种方法，成像和美学质量仍低于EntityMem、CineTrans。

## 指标讨论：接手时保留的结论

完整定义、全量拆分、五组抽查证据、论文依据与不确定性见[当前实验指标解释](docs/entitybench_qwen140_metric_interpretation_20261021.md)。以下结论不修改上面的正式分数。

- **评测执行正确与指标局限可以同时成立。** 本次按冻结官方开源非VLM算法执行并通过独立核验，目前未发现执行／数值错误。出现率是检测与CLIP匹配通过率，不直接等同人工可见率或身份正确率；不能因个别人工可见样本未通过就否定官方结果或单独调低阈值。
- **角色失分更多集中在接续。** 实例统计的硬切通过率96.95%、接续88.87%；接续中上一镜头已调度角色92.41%、未调度角色86.45%。这是实例比率，不是正式的集等权0.9212；“新进入”按相邻调度名单定义，不证明实际尾帧中的可见性。尾帧约束可能影响新角色／构图变化，但当前只是观察关联，未证明四步蒸馏或3D注入的因果作用。
- **场景分数不能解释为对应比例的镜头没有背景。** 921条场景未通过记录均为weak，有候选区域但CLIP低于0.20。五组帧／裁剪抽查可见停机坪存在却未过线、卧室定位为照片、角色调度包含车轮特写中不可见的人等情况；没有全量人工误判率，不能把低分全部归于评测。未跑VLM不是presence偏低的直接原因。
- **硬切／接续安排继承benchmark，EntityMem也使用上一镜头末帧。** 本次1136个硬切按镜头要求生成新首帧，1355个接续用自身前视频实际尾帧。边界headline比较接续的末／首各一帧，不评硬切或几秒后的漂移。0.9822的绝对高分合理，但它相对EntityMem 0.7380的差距尚未解释清楚；已知门控／聚合差异不足以量化全部差距，缺少作者原始帧和对应执行配置，不能声称同等幅度的质量优势。
- **两套DINO分别解释。** 官方0.7155是人物区域指标，独立0.8237是脸部裁剪质心指标，两者输入和有效集合不同，均不是对参考身份的准确率；VLM仍暂不处理，正式数值不改。

## 覆盖与审阅

六项VBench、三个presence指标及人物／物体区域DINO均覆盖140／140集；接续边界覆盖137／140集，其余3集没有可评分接续边界，保持null。

脸部版覆盖140／140集：4991个计划角色-镜头中，2428个候选裁剪已逐图审阅（51页），排除2个错误定位，有效2426项（48.6075%）；其中2124个appearance满足跨镜头评分条件。缺脸、歧义、角色冲突及不足两个镜头等原因完整保留，不按分数挑选样本。角色身份归属尚未逐项人工确认。

## 结果入口与接手要求

输出根目录：`outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140/`。

- [最终验收报告](outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140/archive/completion_20261019.md)、[视频查看页](outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140/archive/viewer.html)。
- [官方逐指标报告](outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140/evaluation/official_parallel_v2/report_qwen21_native_reference_a14b_4step_continuation_full140_no_vlm.md)、[脸部版报告](outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140/evaluation/face_only_dino/report.json)。
- [媒体核验](outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140/audit/final_media.json)、[官方评分核验](outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140/audit/final_official.json)、[脸部版独立核验](outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140/evaluation/face_only_dino/independent_validation.json)。
- [当前指标解释与诊断证据](docs/entitybench_qwen140_metric_interpretation_20261021.md)、[执行与核验入口](docs/LLM_HANDOFF_QWEN140_20261014.md)、[新会话提示词](docs/qwen140_new_window_prompt.md)。

接手先检查 `git status`、`status.json`、实际进程和容器资源，不根据历史快照启动任务；容器内存以cgroup额度为准。评测审计使用[evaluate-face-injection-v2技能](/root/.codex/skills/evaluate-face-injection-v2/SKILL.md)。

保留模型、素材、3D、视频、冻结代码、原始报告、历史档案和其他会话未提交改动；不执行reset／clean或全量stage。未收到后续指令前，不自动新增生成、重评、VLM、全片审阅或提交推送。outputs中的媒体与报告仅保存在本机，不随Git上传。
