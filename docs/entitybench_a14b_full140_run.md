# EntityBench完整140 episode扩展

2026-10-12，用户提出先跑完整benchmark。沿用现有单组PuLID首帧/普通4步I2V/自己的尾帧接续，
不增加Control或拼贴路线，不开启VLM。完整validated split为140 episode、2,491 shot。
本轮既有20 episode的363段视频及其首帧/资产保持只读，通过整episode兼容性审计后复用。

## 范围与耗时

| 范围 | 完整140 episode | 新增120 episode |
|---|---:|---:|
| 视频镜头 | 2,491 | 2,128 |
| 硬切 | 1,136 | 945 |
| 接续 | 1,355 | 1,183 |
| 单人硬切 | 426 | 352 |
| 多人/无人硬切 | 710 | 593 |

最初单角色硬切清单涉及226个episode内角色键。2026-10-12续跑时按原始角色描述核对，
其中9个动物、玩偶、非人脸实体改走基础首帧，共11个硬切；实际新增3D角色217个，
插件候选硬切341个、基础硬切604个。原始352/593统计和expansion_plan.json保留为初始清单，
新路由单独冻结在face_injection_routing.json，不修改剧本、镜头顺序或seed。
需要注入的人脸角色准备参考图→FaceLift Gaussian→角色角度标定。
不为未注入的多人/无人或仅接续角色额外建立3D，不跨episode共享同名身份。
角色标定在生成插件首帧之前完成；官方DINO评测本身不要求3D标定。
无需重标定或重新生成已核验的20 episode资产/视频。

当前单张48GB4090实测4步视频均值96.47秒/shot，新增视频纯推理约57.0小时，
全2,491视频从零约66.8小时。角色准备、首帧与评分另计：过去两路资产准备的单角色墙钟均值
约20.6分钟（含竞争），PuLID配对首帧均值约269秒、基本FLUX首帧约11.4秒，均有冷启动差异。
当前按整个新增路线约4–6天预留；根据新阶段实测更新，不把57小时说成端到端完成时间。
本轮限制ONNX会话线程，旧准备耗时不能作为精确结束日期。

数据盘约32GiB剩余；新增3D资产按历史平均约18.5GiB，再加参考、首帧、视频和评分产物。
容量接近预算，准备/生成阶段设置5GiB剩余空间检查；不足时明确停队列并可续跑，不清理历史产物。

## 固定协议

- 新角色参考沿用SDXL Base：1024、30步、guidance5，三候选，固定seed及近正面单脸筛选。
- FaceLift沿用50步及原相机网格/512标定。新资产两路并发，每路CPU/BLAS4线程、ONNX会话2线程。
  不改变标定相机采样或降低检测门槛；source reference和Gaussian/标定SHA绑定。
- 单人硬切仅运行PuLID-FLUX：640×352、50步、step30 pred_x0准备3D、同latent续去噪，
  v7 strength0.4、最多12步、24px小脸gate、最多4次检测重试，保留安全回退。
  `--allow-selected-only`只允许显式硬切子集，原全剧shot_index不重排，不补接续独立首帧。
- 多人/无人硬切沿用零身份权重FLUX基础首帧、50步、guidance4，无3D注入。
  通用neutral图仅为权重0的接口载体，不声明它是新episode任何角色的身份资产。
- 所有接续用本路线上一视频解码尾帧，不编辑/再注入。视频沿用LightX2V原A14B加high/low
  4step_1022 LoRA、DMD2、shift5、CFG关闭、81帧/16fps、480p面积、BF16/torch_sdpa。
- 新episode seed按完整split顺序冻结；旧20 episode的原seed保持不变。
  只复用完整episode：核验旧源实现SHA、模型/LoRA/运行配置、剧本/seed、输入和父视频/媒体SHA。
  原dependency_fingerprint保留，新本地视频/首尾PNG通过链接复用，另记录原generation.json来源。

## 流水线

入口：`pretest/run_entitybench_a14b_full140.py`，tmux `entitybench_a14b4step_full140`。
输出：`outputs/entitybench_pulid_i2v_a14b_4step_full140/`。

1. 冻结完整split、2,491镜头/cut/seed与226个待准备角色，保存在expansion_plan.json。
2. 使用既有两episode、22视频，在独立parallel_probe/实测官方单路和两路评分。
   相同模型/阈值/采样，每项逐episode值允许≤1e-5数值误差，有效/失败/跳过计数须一致。
   记录吞吐比与显存/主存采样，检查结果不作为正式质量指标，不重生成视频。
3. 按face_injection_routing.json两路补齐217个可注入角色的参考/FaceLift/标定，
   9个非人脸角色使用基础首帧，异常停止新增任务并保留已完成资产。
4. 生成341个插件候选硬切与604个基础硬切；逐个校验来源，冻结正式manifest。
5. 一次加载4步模型，新生成2,128视频并审计复用旧363视频；严格episode内顺序与输入依赖。
6. 全2,491镜头审计、1,355真实接续核验及140条完整episode成片；使用完整episode_id命名。
7. 按evaluate-face-injection-v2两路完整episode评分、独立缓存、一次官方聚合，保留完整原始split。

```bash
cd /root/autodl-tmp/movie
python -u pretest/run_entitybench_a14b_full140.py
```

仅在现有tmux/子进程停止且GPU可用时执行续跑，避免重复队列。
实时phase见status.json、pipeline.log；角色状态见asset_status/，日志见logs/assets/。
正式生成进度见generation_report.json；评分见evaluation/official_parallel_v2/parallel_status.json。
split_20ep.json为沿用接口文件名，内容实际是完整140-ID split，不能按文件名猜范围。

官方原始/skip_vlm/patch均冻结到runtime/，沿用本次6项VBench、presence、跨镜头DINO与边界。
VLM/API调用仍0；外观保真度gate缺失，cs_face/cs_object暂定，官方JSON按公开代码episode宏平均。
论文实例加权/gate口径与公开代码存在差异，完整数据集也不自动消除这些差异或共享角色素材。
本轮单组绝对系统评价，不证明插件因果收益；人工视觉效果审查独立于生成/评分完成。

23项A14B协议检查通过，包括6项新增覆盖：完整清单/仅硬切角色选择、原媒体及fingerprint保持、
配置/seed/父视频变化拒绝复用，以及硬切筛选保留原始全剧索引。新入口/准备脚本编译通过。

## 启动核验

2026-10-12 11:44已启动tmux。实际22视频性能检查：单进程228.28秒、两进程157.35秒，
吞吐比1.451×；逐episode全部指标差异≤1e-5，所有有效/失败/跳过计数一致，VLM调用0。
11:51已自动进入两路角色资产准备，首批Chloe/Leo在各自新episode-local目录中创建。
这个样本速度不能直接当作完整140episode的精确加速倍数。

首批角色已实际完成：Leo150.63秒、Chloe154.38秒，两路并发，参考图、Gaussian和
原273点角度标定均齐备。限制实际ONNX线程池后远快于历史过度并发时的角色准备；
端到端预算由历史5–7天调整为当前4–6天，仍待新增首帧阶段计时，不将两角色速度视作全量保证。

## 2026-10-12 20:28续跑

14:03资产阶段停止：119个角色完成，Blaze的FaceLift预处理产生RGBA四通道而VAE要求RGB。
原角色描述和参考图确认Blaze是斑点狗，单角色并不等于单人；未绕过报错强行做人脸3D。
逐个核对226项描述后，9个基础路由角色为Sunny、Tingo、Goro、Blaze、Gizmo、Crag、Dizzy、
Poppy、The Shadow（按episode-local键限定，完整依据见face_injection_routing.json）。
它们的11个硬切保留原官方prompt/seed，用零身份权重FLUX基础首帧；所有接续仍用自己尾帧。

原119份已完成资产保留，其中Sunny/Tingo/Goro三份不用；116份可注入资产校验后复用，
尚需准备101个可注入角色。已恢复同名tmux两路队列，后续自动进入硬切、视频与官方非VLM评分。
旧20episode保持只读，新全量仍140episode/2,491shot，无新增VLM调用。
28项检查通过；续跑原因、源文件SHA与原失败状态见runtime/non_face_resume_audit.json。
额外基础单角色CLI仅接受显式镜头列表；默认基础入口仍排除单角色。
队列设置remain-on-exit，退出后保留tmux供查错；是否运行须检查pane_dead及status.json。

## 2026-10-12 21:03按用户要求暂停

用户要求FLUX全140先停，改跑已有官方评测的同一20episode Qwen路线。
已终止全140生成/资产子进程，状态paused_by_user；可注入角色资产144/217完成，
另2个正在标定的角色未计完成。全140新增视频尚未启动，旧20episode完整结果保留。
不自动恢复全140；新的Qwen队列和范围见[20episode运行说明](entitybench_qwen21_20ep_run.md)。
