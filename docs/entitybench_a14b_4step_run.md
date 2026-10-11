# EntityBench 20 episode：4步完整接续路线执行记录

2026-10-11（北京时间），用户要求全部改用4步，生成后按官方指标评价，并确认仍跳过VLM。
这是单组完整路线：20 episode、363 shot，191硬切与172接续；单人132、多人230、无人1。
后台已于23:53启动，正式生成前的两镜头官方非VLM接入检查全部通过、零VLM调用。
接入检查只验证实际评分链路，不选择质量或删除负例，其结果不进入正式主表。

191个硬切首帧齐备：56实际中间步v7注入、18安全回退、117多人基础首帧。
全部复用经SHA/配置核验的图像，111个补齐多人首帧仍取旧准备目录；不新增多人注入。
所有接续只用本4步路线紧邻上一视频的真实解码尾帧，不编辑、不再注入。
旧40步队列停止，7个完整视频及加速小样本保留；正式363视频从头重新生成。
完整研究范围见[评测方案](entitybench_collage_vs_3d_evaluation_plan.md)。

## 固定配置与模型

| 项目 | 本轮配置 |
|---|---|
| 基础模型 | 普通Wan2.2-I2V-A14B，无StoryMem LoRA/多图memory |
| 基础revision | `206a9ee1b7bfaaf8f7e4d81335650533490646a3`，33文件/LFS权重已校验 |
| 推理代码 | LightX2V `7bd01b7d5d04bf99cdd9ca9082f20c184c66d991` |
| 蒸馏权重 | high/low rank64 `4step_1022`，各strength=1，CPU在线BF16合并 |
| LoRA来源 | `lightx2v/Wan2.2-Distill-Loras`，revision `570044187a5219776ef30a5c60c6f76428a3a10a` |
| 采样 | DMD2，4步1000/750/500/250，专家切换index2，shift5，guidance1，CFG关闭 |
| 视频 | 81帧，16fps，max_area=832×480，保持输入比例，试验实际848×464 |
| 计算 | BF16、torch_sdpa，模型/T5 CPU offload，原VAE在GPU，无量化或特征缓存 |
| 文本与seed | 原剧本prompt不改写，base_seed+原全剧shot_index；不传负面提示词 |
| 机器 | 单张48GB RTX4090；96GiB主存、300GB数据盘；GPU各阶段顺序独占 |

high LoRA SHA256：`887c3bdeb74e83859c920438e16ca31f39ab18ce189abc5f0e36f8348c5bbb19`。
low LoRA SHA256：`8833bd4fd7c8eabebf0bc8ee5cfaf47f4f310ce116928a02c1adf8941dd4b0f1`。
高噪声调整1095/1095匹配；低噪声1054/1262，290个未匹配张量键均属基础A14B未使用的
图像编码器分支，不能写成全键匹配。见加速试验的lora_header_audit.json。

独立环境`/root/autodl-tmp/envs/lightx2v-trial`，Torch2.5.1+cu118。
为避免固定页CPU allocator累计约81GiB导致初始化被终止，仅该进程将
`torch.empty(pin_memory=True)`改成普通CPU存储；权重值不变，传输性能可能受影响。
4步改变蒸馏权重、时间步与CFG，不能声称是原40步数值等价的推理优化。
原Wan环境/源码和旧生成记录不改动。

## 本轮只使用官方非VLM指标

官方revision `3776439e7037b2128ac3b5d2d1ce63e424cca166`；原始文件已在仓库保留为
`benchmarks/entitybench/eval/evaluate_benchmark_official.py`，SHA256
`2334aad4b61cae2a0ea38cc2ee9ae85125b6df4e128610903654fd2b49607bb4`。
调用已有skip_vlm适配（SHA256
`79caf1327e7f1a63527c0e18bc9c936b8618b59080197d2f50677ee2ba5cb5a0`），
原文件、适配文件、差异补丁与来源记录均冻结在输出runtime/；不改官方非VLM评分逻辑。

- Pillar1：当前代码实际启用subject_consistency、temporal_flickering、motion_smoothness、
  dynamic_degree、aesthetic_quality、imaging_quality这6项VBench。background_consistency已由上游删除。
- Pillar2：官方DINOv2跨镜头`cs_face/cs_object`与真实视频尾/首帧`cs_transition_boundary`。
- Pillar3：官方GroundingDINO/CLIP角色、物体、场景出现率，保留覆盖和失败计数。

官方以角色区域crop（可含头发、服装、身体）与CLIP×清晰度×面积选择代表外观，
按官方采样/质心/聚合运行。此前InsightFace、参考图DINO、人脸框质心不进入本轮。
未运行VLM保真度、动作遵循或LLM跨镜头一致性；未应用外观保真度gate，
故cs_face/cs_object标为provisional/partial_without_vlm，不直接冒充完整论文表格数值。
缺失项保留null，指标沿用官方value/n_evaluated/n_failed/n_skipped；严格模式，不改评分参数。

评测独立环境`/root/autodl-tmp/envs/entitybench-eval`，使用本地GroundingDINO、CLIP、
DINOv2和VBench缓存，device=cuda。所有VLM/API凭据从子进程环境移除，须核验最终total_calls=0。
正式评分目录绑定全部363视频SHA及脚本、split和评测器SHA；输入改变时拒绝复用旧分数。

## 运行与产物

入口`pretest/run_entitybench_a14b_4step_20ep.py`；tmux `entitybench_a14b4step20`。
新输出根：`outputs/entitybench_pulid_i2v_a14b_4step_continuation_20ep/`。

```bash
cd /root/autodl-tmp/movie
python -u pretest/run_entitybench_a14b_4step_20ep.py
```

只有现有会话停止且GPU空闲时才运行续跑命令，避免两个队列同时占用GPU。
生成子进程一次加载模型，20episode内严格按shot顺序执行；每个视频完整解码核验81帧/16fps，
保存首/尾PNG与SHA。配置、代码、LoRA、源脚本、seed、输入、父视频和输出都匹配才复用。
父视频改变，即使尾帧像素相同，也使接续缓存失效；生成运行时不得修改相关脚本。
资源/模型异常停队列并写failed，不静默换首帧、删镜头、补零或改参数。

| 路径 | 用途 |
|---|---|
| status.json / pipeline.log | 总阶段与调度日志 |
| generation_report.json / generation_manifest.json | 逐shot完成数、耗时、显存与运行配置 |
| logs/generate_formal.log | 实时生成日志 |
| videos/ / inputs/ | 正式逐shot视频、首尾帧、实际输入、来源/依赖报告 |
| official_results/ | 原episode和scene/shot编号的363条视频链接 |
| preflight/ | 隔离的两镜头官方集成验证，不计正式进度或分数 |
| audit/ | 全363镜头四面板来源/输入/视频首/尾联系表与机器审计 |
| episodes/ | 20条浏览成片，保留全部原帧；官方仍评价逐shot源视频 |
| evaluation/official/ | 官方逐episode和汇总报告、run_manifest与输入指纹 |
| report.md | 生成/官方非VLM完成后自动报告，人工视觉审查另记 |

完整流程为预检→363shot正式生成→哈希/接续审计与20条成片→官方非VLM评分。
9项新协议测试覆盖父视频/蒸馏权重/seed/媒体损坏引起的缓存失效、评分缓存输入变更、
官方评分函数保持、VLM环境隔离及重复run窗口的成片命名；实际官方预检验证全部6项VBench，零VLM调用。
已测4步小样本平均89秒/shot，363镜头纯生成粗略约9小时，评测/加载/重试另计。
首次正式视频已落盘：81帧/16fps、848×464、84.5秒；模型加载98.4秒。
当前已启动但尚未整批完成，不把小样本预检分数、进程存活或零退出码当作完整效果结论。
