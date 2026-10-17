# Qwen140：非人脸角色分流修复与接续

2026-10-16北京时间13:44:58，RGBA接续队列在run55／Blaze的标定阶段退出1。
报错为 `Insufficient detected calibration samples: near_frontal=1, positive=5, negative=3, positive_extreme=7, negative_extreme=0`。
原剧本明确Blaze为Dalmatian pup；Qwen参考的InsightFace检测误把它视为可标定人脸。
旧FLUX全量已有9项episode-local非人脸角色规则，Qwen协调器仅看检测结果，漏用了语义分流。

## 修复与来源

沿用 `prepare_entitybench_a14b_full140.NON_FACE_ROLES` 中已逐剧本核验的9项规则：
Sunny、Tingo、Goro、Blaze、Gizmo、Crag、Dizzy、Poppy、The Shadow，各自绑定完整episode_id及原描述证据。
这些角色涉及11个单人硬切，继续使用既有Qwen参考和原prompt／seed生成基础首帧，不做人体人脸3D注入。
所有140集／2491镜头仍在生成和官方非VLM评测范围；不删角色、镜头或失败记录。

冻结核心、原全量适配器和RGBA修复未改。独立恢复模块 `pretest/qwen140_routing_recovery.py`
仅纠正上述9个参考metadata的 `face_eligible`／`face_skip_reason`，并添加明确的语义覆盖来源。
候选图、选图、参考PNG、选择seed、选择fingerprint全部保持。
原metadata和失败状态先归档，再冻结覆盖内容；支持中断后幂等接续，拒绝未知修改。
更正发生在首帧／manifest生成之前，后续原首帧fingerprint自然绑定更正后的metadata SHA。

来源保存在全量输出：

- `runtime/face_routing_recovery_20261016.json`：9个覆盖、原metadata SHA、脚本证据及独立恢复实现SHA。
- `runtime/face_routing_before_20261016/`：原metadata、失败状态、资产报告、tmux现场及受保护文件清单。
- `audit/face_routing_recovery_validation_20261016.json`：711参考／1163候选SHA、11硬切依赖和资格数核验。

修复前115份3D／标定完成，其中4份属于上述非人脸角色，全部保留但不再用于注入。
有效人脸任务由217改为208：接续前111份可复用、97份待补。
69,740个已有资产文件SHA核验保持；仅上述9个metadata发生已归档的语义资格纠正。
3项新恢复检查、7项全量检查和3项RGBA检查通过，Python／shell语法检查通过。

## 磁盘预算与启动

沿用用户已批准的40GiB产物总预算和10GiB余量，未修改原50GiB批准或10GiB运行守护。
续跑将本轮 `assets/` 内已生成文件的实际分配空间计入已用预算，排除符号链接并按inode去重；抵扣最多40GiB。
本次约13.656GiB已生成，故剩余产品预算约26.344GiB，要求当前空闲至少36.344GiB。
实际空闲约46.005GiB，通过；GPU显存／利用率、20核CPU及96GiB内存仍按原准入检查。
计算保存在 `runtime/resume_storage_admission_20261016.json`；不要求为每次重启重新空出整批50GiB。

用户回复“继续吧”后，在原tmux `entitybench_qwen21_full140` 接续：

```bash
bash outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140/runtime/resume_routing_20261016.sh
```

入口为 `pretest/run_entitybench_qwen21_full140_resume_routing.py`，日志 `logs/resume_routing_20261016.log`，
启动来源为 `runtime/face_routing_resume_launch_record_20261016.json`。
入口校验恢复清单、原资产、原20集及全套模型SHA，沿用RGBA兼容后接续原流水线。
后续先看此日志、tmux pane_dead、实际进程和实时JSON，不重复启动旧入口。
北京时间14:59:33已观察到run152／Leo和Nathan两份新资产完成原规格3D／标定，Gaussian与标定绑定核验通过。
实时状态推进至113／208，失败列表为空，两路已继续后续角色；不是仅启动后停留在预检。
该推进证据保存在 `audit/face_routing_resume_progress_20261016.json`。
官方非VLM两进程、零API／VLM调用、逐图真实脸部审阅等待条件及其他冻结参数保持。
2026-10-16恢复时尚未提交或推送GitHub；后续提交以Git记录为准，用户其他工作树修改保留。
