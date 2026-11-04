# 2026-11-01：按0.7863口径重算与身份提升素材

用户本轮要求：把已重跑episode按原“可见单脸门控＋定位纠错”口径重新算人物区域DINO，并寻找更明显的身份一致性提升素材。用户已选择恢复带GPU的实例，切换需在AutoDL控制台完成。不能把用户的选项答复当作GPU已恢复；应实查CUDA、CPU及内存。当前读到的旧benchmark status不能证明生成进程仍在。

## 输入与已完成工作

- 固定输入30个完整重跑episode、343shot，不随新完成集漂移。输入清单在`outputs/entitybench_rerun_visible_region_identity_20261101/input_manifest.json`。其余110集仍用原版；共140集2491shot、4991个计划角色—镜头实例。
- 已用冻结官方函数及原始Gram矩阵，**完整复现0.7863**（1697个保留裁剪、1380个计分实例、131个有效集）。`baseline_validation.json`为核验记录，没有改官方0.7155。
- 新版检测脚本`pretest/qwen140_rerun_region_diagnostic.py`，原始3项人工排除只按裁剪SHA转移，不能按同名角色套到新图。新人工复查决策需在最终评分前冻结，不剔除真实身份/画风漂移、低DINO或服装变化。
- 新80项人工复查仅从被替换的30集新版计分裁剪中选最低60＋固定种子随机20；未替换110集保持原复查结果，不能只在新版一侧追加纠正共同旧图。新版最终诊断尚待全部检测和复查。
- 原4991实例已验证“仅检测present原区域的最小至最大采样位置，保留之间所有跟踪步骤”与全部五帧原缓存的门控决策完全一致，差异0。记录`required_span_validation.json`。是加速采样检测，不改取帧、0.9置信度、24像素、80%区域包含等规则。B视频按实际解码帧数使用官方五帧采样公式，不固定假设81帧。
- CPU检测缓存已保存18个shot；记录在`extraction/<episode_id>/<scene_shot>.json`，不同源码/设备协议在`protocol*.json`，老版源码存`runtime/extraction_v1/`、`extraction_v2/`。源文件未触及原benchmark冻结的19个源码。

## 当前环境与顺序

本轮实际环境为0.5核、2GiB内存、无GPU。并行模型和InsightFace重复加载完整ONNX图导致多次exit137；没有改变模型权重或静默填分。不要同时启动CPU RetinaFace、InsightFace及DINO三个模型。

`pretest/qwen140_rerun_figure_screen_cpu.py`为分阶段省内存适配器，继续使用素材skill中相同的评测函数/CLS特征及像素。只加载SCRFD和glintr100，避免先加载再丢掉其它网络；避免ArcFace在ORT已加载权重后又解析一遍大权重。输入归一化元信息已从原ONNX解析并按模型SHA冻结在`arcface_graph_metadata.json`；不修改模型文件。ORT仅CPU、关闭图优化/arena、缓存相同输入检测结果。与3D生成门控无关。

**素材候选优先规则**：30集产生194个已实际注入、预览脸高至少128像素的角色配对；资源受限时先筛选其中预览脸高至少224像素或原6步注入的72项，规则不读取最终分数。其余标为deferred，不冒充已评估。原128px部分试跑结果保留在`runtime/partial_128px/`。194候选及全部排除原因在`outputs/rerun_firstframe_identity_candidates_20261101/input_candidates.json`。

执行前先检查实际进程，防止重复。当前任务的普通worker不属于用户已关闭的“守护修复Agent”。主视频/图片不重新生成，不调用收费API。适配器可使用工具的长任务session或tmux；恢复GPU后先确认没有别的重型生成进程占卡。

```bash
# 阶段1，CPU：InsightFace筛选，同一时刻只启动一个模型worker。
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -u pretest/qwen140_rerun_figure_screen_cpu.py if
# 阶段2，CPU：IF完成后再算前48项参考DINO及打包候选，保留负例/错误。
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -u pretest/qwen140_rerun_figure_screen_cpu.py dino
# 阶段3，GPU恢复后：RetinaFace FP32，复用所有合法CPU缓存。
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -u pretest/qwen140_rerun_region_diagnostic.py extract --device cuda
# 无GPU时可改为--device cpu，但需串行且会慢。
# 阶段4：完整343shot检测结束后，自动规则及原始纠错SHA转移版；生成80项无分数复查图。
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u pretest/qwen140_rerun_region_diagnostic.py score
# 先读review/manifest.json，检查所有review/page_*.jpg与原采样帧/人物原句，
# 在review/decisions.json中保存80项含review_id、episode_id、character、shot、
# original_crop_sha256、decision(keep/exclude)、证据及理由，记录manifest_sha256。
# 然后才执行最终重算：
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u pretest/qwen140_rerun_region_diagnostic.py final
```

自动版与最终人工版要分开标注；补充全140表为原诊断0.7863→新版诊断，不把它替换官方cs_face。另报告原/新版有效集数、保留区域/计分实例和同角色—镜头交集的对照，原版已知R051错配仍在历史0.7863中，不能声称全量纠错。

## 素材选择与当前观察

使用`movie-firstframe-figure-assets` skill。`outputs/rerun_firstframe_identity_candidates_20261101/`包含IF进度、诊断文件及最终候选原图。最终交付必须为普通PNG/JPG文件夹，保留reference、Control、Treatment、相同显示ROI的脸部裁剪、对齐3D、mask、提示词/种子和输入SHA；不能只给HTML。

两种证据不能混淆：

1. 同一次生成的Control-v7 Treatment可说明3D注入效果，prompt/seed/参考等一致。IF和参考DINO一起看，实际打开图核对身份、表情、姿态、接缝。
2. 原benchmark-vs整套流程重跑可展示系统效果，人物素材、路由、提示词和注入都改变，不是3D单独消融。`outputs/entitybench_rerun_visible_region_identity_20261101/system_examples/`有12组待审的跨镜头原区域表；已经目视检查，很多高涨幅来自衣服/错角色定位，不能据其原始DINO涨幅推荐脸部身份案例。run317/Alex及run470/Morgan更适合进一步核对“人物出现＋外观统一”系统案例。

阶段1、2现已结束：72项中IF成功71项、1项原检测人数不足；23项IF正向。IF前48项参考DINO已完成，19项双正向，自动打包前16项后已逐组目视检查。IF适配器每8项退出释放模型内存；DINO独立进程结束。当前无此任务的模型worker在跑。

已交付`outputs/rerun_identity_figure_materials_20261101/README.md`：`paired_reviewed/`有5组有限正向，`tradeoffs/`保留2组副作用例子，`system_comparison/`保留Alex/Morgan整套流程对比。7组配对各10份来源文件复制SHA一致，另有原提示词和核查说明。`manual_review.json`记录全部16组的结论，71项完整排名、负例和4项IF/DINO分歧仍在筛选根目录。选择性筛选不能估计全体增益。

目视结论：run470/Silas5:2 IF0.2246→0.4106、参考DINO0.5384→0.5871，虽最大涨分，但把提示词要求的微笑明显削弱，不作干净正例。run594/Leo3:4 IF0.2184→0.3367，但视线/眼睛开合变化、全帧还有相似男孩重复问题，亦不作主图。Erik2:2 IF0.1793→0.2607、DINO0.6061→0.6469，Leo3:5 IF0.2448→0.3038、DINO0.2011→0.2243，属于有限正向，最终绝对IF仍低。Chloe2:4和2:5及Kenji4:3保持较好但涨幅小。**没有找到可无保留宣称明显身份恢复且表情/场景保持的新增例子**；不得迎合绘图需求夸大。

GPU恢复前停止重型检测，保留18/343缓存。用户选“恢复带GPU实例”不代表控制台切换已完成；最新检查仍无GPU、0.5核、2GiB。恢复后先核对实际资源和进程，再执行阶段3；不能声称本轮全量重算完成或后台仍在跑。

本轮新增脚本/文档未提交Git。既有30集原媒体、3D、官方分数、用户无关文件均保留；不启动修复Agent，不改原生成阈值，不凭分数选择删除真实失败案例。

## 后续用户选定素材与脚本面板

用户选定run594/i75_j98的Leo3:4、Leo3:5、Erik2:2，认可其定性展示价值。三组同一episode素材已单独导出至`outputs/run594_i75_j98_identity_examples_20261101/`，原图和来源SHA保留。后按示例生成`episode_script/`内的MD/TXT/JSON、可编辑SVG、PNG和矢量PDF，附原人物参考生成提示词、原episode与来源哈希。脚本`pretest/export_run594_episode_script_panel.py`可复现导出。

用户曾明确要求按全部30集做无3D/有3D首帧消融（InsightFace＋参考脸DINO），随后说“算了，先导出脚本面板”，当前优先交付后者。全部30集消融的输入已核查：343shot，331个新关键帧配对，311个实际注入图像/595角色配对，20个整图Control沿用；10个A继承和2个早期未配对路径。`pretest/qwen140_firstframe_ablation30.py`及独立输出manifest/protocol已初始化，但模型评测队列尚未启动，不得说已在后台评测或已出最终指标。脚本后续启动前仍需检查资源和输入冻结；B评测的是过渡目标关键帧，不是实际解码后主shot首帧。不可把先前筛出的正向三组均值作为全30集消融。

## GPU恢复后的当前优先任务已完成

实查环境已恢复48GB RTX4090、20核CPU、72GiB内存；上文CPU限制为历史阶段。用户明确先只继续run594/i75_j98的3:4–3:5修复，该任务已在`outputs/run594_i75_j98_shot34_35_prompt_fix_20261101/`完成；入口`pretest/qwen140_run594_fix_shot34_35.py`。3:4仅增强提示词、保留原条件图/参考/种子/3D门控，单人生成成功且6步注入；3:5原目标图逐字节复用，新3:4实际尾帧生成3秒过渡，新的实际尾帧再生成主动作段，旧人物状态未混入提示词。用户接受触碰Marcus的手势，未继续重试；中段短暂叠影保留披露。新旧并排、原图/视频素材包、全部SHA/输入链验证已导出。未启动全30集消融、诊断或其它episode，也未更新其原始分数/清单。当前无生成worker。

## 后续用户缩小为30集的当前评估已完成

用户确认只评估已重跑30集，并纳入run594新版3:4/3:5。结果见[最新30集评估](entitybench_rerun30_latest_evaluation_20261101.md)与`outputs/entitybench_rerun30_latest_evaluation_20261101/`。原343shot输入18条检测缓存按哈希复用，其余检测全部补齐；未在旧全140混合输出根继续运行final，亦未生成全140新诊断总分。30集主诊断0.8022→0.7991；这不是历史全140 0.7863的重新聚合。用户此前首帧IF/参考DINO消融仍未启动。
