# Qwen140：FaceLift RGBA格式故障与接续

2026-10-16北京时间06:05，主队列在Annelise的FaceLift阶段退出1。此前711角色参考／1,163候选已全部完成，217个实际合格新3D键中67个完成、1个失败、149个尚未开始；首帧和新视频尚未生成。

失败角色为 `000a4cab-514f-33ce-af41-2b0b6f31cfea__run1185__i0_j10__T120 / Annelise`。Qwen原参考为RGB，SHA `11e001341be9c8e744f683ab9b952132caf0a2334aafc19f669eae9ea72c9686`。
FaceLift自动预处理进入原有回退分支，背景移除结果为512×512 RGBA；未合成为RGB即送入VAE，触发 `expected input[6,4,512,512] to have 3 channels`。
ONNX CUDA12库警告也存在，但本次实际退出栈为VAE通道不匹配，容器无OOM记录。

## 独立兼容修复

冻结核心、FaceLift上游源码、已选参考、prompt、seed、步数和检测阈值均未修改。
新增 `pretest/facelift_rgba_compat.py` 仅在FaceLift管线入口遇到RGBA时按原预处理常量的白色背景alpha合成RGB；普通RGB对象原样交给原函数。
前后PNG、来源SHA及原参考SHA保存在角色 `faces_3d/annelise/rgba_compat/`，修复后实际3D构建已成功，非复制参考的假3D回退。

独立资产适配器为 `pretest/prepare_entitybench_character_assets_rgba_compat.py`；只在原资产助手启动FaceLift子进程时安装兼容，不启动参考阶段。
接续协调器为 `pretest/run_entitybench_qwen21_full140_resume_rgba.py`，委托已冻结原全量入口，只重定向资产助手并在全量权重SHA核验后释放干净权重页缓存，避免文件缓存占满容器内存造成评分准入误判。
三项新格式／来源检查和七项既有全量检查通过，Python及shell语法检查通过。

原失败状态及角色文件完整复制到全量输出 `runtime/rgba_recovery_before_20261016/`；67份已完成资产和全部素材的41,871个文件SHA核验保持。
独立新增代码、测试和原核心SHA冻结在 `runtime/rgba_recovery_implementation.json`，不覆盖原 `runtime/full140_adapters.json`。

## 运行及来源

失败角色单独构建验证tmux `entitybench_qwen140_rgba_check` 已退出0；成功产物留作正式任务复用，不另加消融。
Annelise的273点／512规格标定也已完成，Gaussian／标定SHA核验通过，目前完整资产为68／217份。
主tmux仍为 `entitybench_qwen21_full140`，已使用 `runtime/resume_rgba_20261016.sh` 完成该角色标定并接续完整流水线；最近检查正在重核来源／权重，之后复用素材和资产继续剩余阶段。
新日志为 `logs/resume_rgba_20261016.log`；后续逐资产仍见 `logs/assets/`，状态以实时JSON及实际进程为准。
恢复后完整核验复用已有素材和资产，只补剩余任务；人物素材／首帧／视频／评测参数及脸部真实审阅条件保持。
启动容量参数继续使用已批准的50GiB预算准入，运行中保留10GiB，不删除产物、模型或安装环境。

来源记录：`audit/rgba_repair_validation_20261016.json`、`runtime/rgba_resume_launch_record.json`。
2026-10-16恢复时本修复尚未提交或推送GitHub；后续提交以Git记录为准，用户其他未提交改动保留。

2026-10-17提交检查：`facelift_rgba_compat.py`、`prepare_entitybench_character_assets_rgba_compat.py`、
`run_entitybench_qwen21_full140_resume_rgba.py`各保留冻结版本已有的末尾空行，避免改变运行中源码SHA。
仅这三文件的空白检查排除 `blank-at-eof`，仍检查行尾空格及缩进；其余暂存文件使用默认完整空白检查。
