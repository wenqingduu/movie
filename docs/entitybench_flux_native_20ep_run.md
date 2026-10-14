# FLUX原生人物素材＋PuLID-FLUX首帧：相同20episode

**2026-10-14说明：本文是被拒绝插画风的历史尝试，仍保持停止。**
后续独立摄影版已完成20集／363视频，见[Portrait Agent路线](entitybench_flux_portraitagent_20ep_run.md)及[三路线结果汇总](entitybench_results_20ep_20261014.md)。
不得把摄影版结果归入本文旧输出，也不恢复下述插画队列；全140FLUX继续停止。

**当前状态（2026-10-13 17:37北京时间）：因用户拒绝插画风素材，队列已停止；4候选保留，尚未进入3D/首帧/视频。**
先前技术准入不能代表风格符合要求。不直接恢复此冻结prompt计划；修正风格时保留原计划，用新输出做小样确认。

2026-10-13（北京时间）。用户要求在已有Qwen20、SDXL→FLUX20结果之外，新增一组FLUX生成素材和首帧的最终路线；仍只跑当前冻结20episode/363shot。
用户明确选择本机现有FLUX.1-dev/PuLID路线：单人可绑定参考；多人沿用FLUX文生图，不宣称具有Qwen原生多参考能力。

## 固定范围与输出

- 复用Qwen和旧FLUX的相同split、完整episode_id、顺序、prompt、seed及cut；20episode=11 easy/6 medium/3 hard，363shot=191硬切+172接续。
- 输出：`outputs/entitybench_flux_native_i2v_a14b_4step_continuation_20ep/`。
- 正式入口：`pretest/run_entitybench_flux_native_20ep.py`；tmux `entitybench_fluxnative_a14b4step20`。
- 新素材112个硬切所需episode-local角色。40个单人硬切角色各3候选，其余72各1候选，共192；19个仅接续角色不额外生参考。
- 人物素材prompt与角色seed沿用Qwen20冻结计划，保持年龄、物种、描述和故事风格；不强制所有角色成年/写实。
- 新参考对应全新FaceLift/273点、512标定；三候选都缺可靠单脸时保留基础参考及资格缺失，不能借用旧SDXL/Qwen Gaussian或删除镜头。
- 191硬切全部新生成、363视频全部新生成；内部Control只为共享latent分叉，不生成新Control消融视频组。

## 模型与执行协议

本机FLUX.1-dev FP8：`third_party/PuLID/models/flux-dev-fp8.safetensors`，原quantization_map、VAE、T5、CLIP、PuLID0.9.1、EVA与Antelope权重均已存在，运行前冻结完整SHA。
不用新模型下载或环境升级。论文只明确FLUX家族，不能把这组说成作者同checkpoint复现。

- 参考素材：1024×1024、40步、FLUX guidance4、true CFG1、T5长度128；纯文生图，id=None、id_weight=0，不借用旧身份作carrier。
- 参考选择沿用Qwen20唯一单脸、姿态和面积规则，不通过官方指标挑素材。
- 单人首帧沿用既有FLUX协议：640×352、50步、step30共享latent分叉、v7 color-safe强度0.4、最多12 active steps、小脸自适应、最小脸高24、检测重试4、PuLID身份权重1。
- 不将Qwen step12或cosine0.40的1/6步规则复制到FLUX；这是现有FLUX原协议。
- 多人、无人或缺可靠参考脸：同FLUX模型纯文生图，无身份参考绑定、无3D注入；缺目标脸/闭眼等继续沿用原FLUX安全回退并保留原因。
- 现有单人生成器内部仍保存Control、pose与诊断文件；本轮不启动额外IF/参考DINO/脸框质心评测，这些内部字段不进入官方对比表。
- 视频同已有两路线：普通Wan2.2-I2V-A14B＋LightX2V high/low `4step_1022` LoRA，DMD2四步、shift5、CFG关闭、BF16/torch_sdpa、81帧/16fps/480p面积。
- 接续仅用本路线上一视频真实解码尾帧，无二次编辑/注入，不借用旧视频或尾帧。

模型对象跨镜头复用，与原shared-model FLUX执行方式一致；参考/基础图的文本条件先计算，文本编码器移回CPU后执行图像去噪。只缓存已冻结权重的模型记录SHA，不改采样、mask、检测或官方评测逻辑。

## 阶段与续跑

固定计划和已有权重SHA → 新FLUX参考 → 有资格角色的新FaceLift/标定（两路）→ 191新硬切 → 冻结manifest → 363新A14B → 全来源/尾帧审计及20条成片 → v2两路官方非VLM → 三路线/论文归档。

实时阶段看`status.json`；素材看`references_status.json`及candidate JSON；首帧看`firstframes_status.json`；视频看`generation_report.json`；评分看`evaluation/official_parallel_v2/parallel_status.json`。
`runtime/flux_implementation.json`冻结权重与原生成代码/新入口SHA，`preparation_plan.json`冻结prompt/seed/参数，首帧绑定新参考/3D/标定与最终媒体SHA。不得修改冻结原代码后直接复用缓存。
主入口有进程锁，先检查tmux pane_dead与实际Python进程，不重复启动。

准入检查使用正式队列最前4个候选（Julian、Silas、Viktor），生成文件按相同seed/SHA校验续用，未增加候选组或视频消融。

## 官方非VLM与归档

继续使用evaluate-face-injection-v2、revision `3776439e7037b2128ac3b5d2d1ce63e424cca166`、原版与skip_vlm适配的相同SHA。
两进程各4CPU线程，按完整episode分片182/181shot，GPU阶段分开，最后一次原split官方统一聚合。
六项VBench、GroundingDINO/CLIP出现率、DINO角色/物体区域跨shot质心与真实接续边界；VLM调用0，strict且无config覆盖。
39项VLM缺失保持null，fidelity gate不可用，cs_face/cs_object暂定，scope=partial_without_vlm。指标完成与全量人工审片分开记录。

官方报告：`evaluation/official_parallel_v2/report_flux_native_reference_pulid_a14b_4step_continuation_20ep_no_vlm.json/.md`。
`official_comparison.md`含all/easy/medium/hard旧SDXL→FLUX、Qwen、新FLUX→FLUX均值、差值及有效episode数。
归档：`outputs/entitybench_paper_flux_qwen_nativeflux20_comparison/README.md`、`comparison.json`及三份原始官方报告。
自动归档由主入口在准备/完成时调用，无第二个watcher竞争写入。新路线未完成字段null，不借旧分数填入。
素材、参考绑定、模型与首帧协议存在差异，三路线差值不解释为插件因果收益，也不是公平论文排名。

## 空间与保留边界

启动前数据盘可用约13.6GiB。本次新增预计5–7GiB（参照Qwen20素材/3D约4.1GiB、首帧/视频/输入/成片/审计/评分产物），启动需至少12GiB，逐素材/角色/首帧及阶段保留5GiB守护。
按当前预算无需扩容；若实际产物增长超预期，守护中止并保留全部缓存，不自动删除历史产物。
FLUX全140保持停止、已有144/217资产保留；Qwen20、旧FLUX20分数和媒体保持，只读用于比较。
用户其他未提交改动及experiment_output保留，未提交或推送Git。

## 2026-10-13 17:29北京时间启动快照

前4个正式素材候选准入通过，平均29.26秒/张、峰值CUDA allocated约15.96GiB。样本Viktor两候选都有可靠单脸；Julian/Silas为不需要3D的多人硬切角色，不按人脸挑选。四张样图实际呈插画风格，已查看并记录，不为预期分数改写prompt或重生成写实参考。

主tmux已启动完整20episode队列，原候选和Julian/Silas已选参考按同seed/输入SHA续用；目前素材阶段，不能宣称191首帧/363视频或官方评分完成。

准入证据：输出runtime/reference_admission.json及admission_reference_samples.jpg。7项新路径/范围/输入变更/错误Gaussian检查、17项既有A14B协议检查通过，Python编译和git diff --check通过。

## 官方FLUX型号核验与风格纠正

官方EntityBench main revision3776439e7037b2128ac3b5d2d1ce63e424cca166共有150文件；代码/说明/启动文件为README.md、eval/compare_methods.py、eval/evaluate_benchmark.py、eval/run_eval_distributed.py、examples/run_eval_example.sh。逐文件无FLUX型号、生成portrait实现或FLUX权重配置。论文仅写FLUX，E.2由Portrait Agent推断anime/photorealistic/3D等风格，不能推出具体checkpoint。

本轮移植的Qwen20素材prompt使用“适合故事的风格”，未明确摄影写实；4图插画倾向的成因尚未做受控测试，不能直接归因FP8/FLUX型号。之前技术准入缺少风格验收，已记录拒绝。核验证据与拒绝理由位于review/upstream_flux_model_audit.json、review/style_admission_review.json。
