# Qwen140准备复核（2026-10-14，北京时间）

当前为 `prepared_waiting_resources`，全量生成和GPU评分均未启动。没有重复下载、调用API／VLM或恢复旧路线。
本机已有六份全量适配器及测试继续使用；原冻结核心和用户其他未提交改动保持。

## 已核验

- 从原validated split重算：140集／2,491镜头，1,136硬切／1,355接续。整集复用20集／363镜头，新增120集／2,128镜头，其中945硬切／1,183接续。
- 新增711个episode-local参考键、226个单角色3D候选键、1,163张候选；保持旧20集seed，新增seed按完整split序号冻结。参考／首帧参数和prompt模板与原Qwen20一致。
- 已有完整复用审计通过：112参考、39份有效3D／标定、191硬切、363视频、172个独立解码尾帧接续。当前旧Qwen目录29,687个文件SHA与保存清单一致；100份保留源码／其他用户文件未变。
- 当前29份适配器、测试及模型配置来源与冻结SHA一致。此前两个准备版本的冻结清单保留在 `runtime/full140_adapters.preflight_revision_1.json`、`preflight_revision_2.json`；未产生新素材、首帧或视频，原preparation_plan未改。
- 6项全量检查、9项skill官方并发／聚合只读检查通过，Python编译及diff空白检查通过。整集链接复用、父视频篡改拒绝、原算法函数不变、逐图决策覆盖和manifest绑定已检查。
- Qwen、A14B、LoRA和四个Python环境保留；本次核对已安装包版本及Qwen／A14B文件大小，LoRA重新核对SHA。此前完整权重SHA审计已通过，本次完整重算被交互中断，不能称为新的完整权重复核；开跑入口仍会重新hash全部模型并检查GPU运行环境。

准备证据：[JSON](../outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140/audit/preparation_check_20261014.json)。
原20集复用证据：[reuse20.json](../outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140/audit/reuse20.json)。

## 当前准入

实时数据盘约59.747GiB、系统盘约9.259GiB空闲；GPU设备未挂载，CPU配额0.5核、容器内存上限2GiB。
协调器的原启动条件保持：数据盘≥60GiB、GPU空闲显存≥45,000MiB、利用率≤10%、CPU≥8线程配额、容器内存≥64GiB；逐阶段／任务保留至少10GiB数据盘余量。
数据盘距当前60GiB硬门槛约0.253GiB，其他资源也未满足。没有降低参数或删除项目来绕过准入。
以上是本次快照；启动前再次读取实际状态，不能沿用旧27.9GiB的admission。

`--prepare-only`只做CPU文件核验并返回，不会因资源后来恢复而自动启动GPU阶段。
中断的CPU复核日志保留在 `logs/prepare_only_20261014_recheck.log`；其中阶段提示不表示整次复核完成。

## 接续和真实裁剪决策

资源恢复后，先查tmux pane_dead、实际进程、status和SHA，再使用现有 `pretest/run_entitybench_qwen21_full140.py` 在tmux中运行。
流水线为参考→新3D／标定→945硬切→新2,128视频及核验复用363视频→来源核验／140成片→官方非VLM→脸部提取→逐图检查→脸部评分→归档。
GPU串行生成阶段与完整episode两路评分阶段分开；并发验证只读已有视频，不新增生成消融。

脸部裁剪提取后，队列等待 `evaluation/face_only_dino/review/decisions.json`。
审阅者须实际查看每张脸crop，再记录 `face_crop_manifest_sha256`、`reviewer`、`reviewed_at`、全部 `reviewed_pages`、全部 `reviewed_ids`、`rejected_ids`及逐项`rejection_reasons`。
criteria固定为 `face-localization only; retain generated identity/style drift; no DINO-score filtering`。
检查不以身份／画风漂移或DINO分数删图，不伪造自动验收；冻结真实决策后才进入DINO特征计算。
官方人物区域和脸部版分别保存，最终另核验140集覆盖、2,491映射／SHA、1,355真实接续、六项VBench及缺失／失败／覆盖统计。

2026-10-14准备核验时，代码／测试修正和准备说明仍在本机未提交；后续提交以Git记录为准。
