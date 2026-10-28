# Qwen140 集成实验迁往 A800：数据与续跑清单

核对日期：2026-10-28。本机 `/` 是 30G overlay 系统盘，`/root/autodl-tmp` 是独立 320G XFS 数据盘；因此仅保存系统盘镜像**不足以**恢复本实验。当前脚本和 JSON 记录使用 `/root/autodl-tmp/movie` 等绝对路径，目标机应保持同一路径。Git 已包含代码和画风配置，但不包含模型、`.env.portrait`、冻结基线和新生成视频。

## 需要从旧数据盘复制什么

| 路径（均位于 `/root/autodl-tmp/` 下） | 当前大小 | 用途 |
|---|---:|---|
| `movie/models/video/Wan2.2-I2V-A14B` | 118G | Wan2.2 I2V/FLF2V 原权重、T5、VAE |
| `movie/models/diffusion/qwen-image-2.1` | 31G | 人物素材和起始/目标帧生成 |
| `movie/models/video/Wan2.2-LightX2V-Distill-Loras` | 1.3G | 高低噪四步 LoRA；脚本检查 SHA |
| `movie/outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140` | 31G | 冻结 140 集基线、二维/3D 素材、旧视频、官方评测与父尾帧 |
| `movie/outputs/entitybench_qwen21_i2v_a14b_4step_continuation_20ep` | 5.4G | 某些旧角色参考的回退来源 |
| `movie/outputs/entitybench_qwen140_rerun_pilot_20261026` | 68M | 早期正确人物锚点与 run1404 复用视频 |
| `movie/outputs/entitybench_qwen140_integrated_episodes_v2_20261026` | 37M | run700 完成视频/评测、run714 未完成缓存、实际尾帧与审计记录 |
| `models` | 2.5G | EntityBench 的 DINOv2、CLIP、GroundingDINO、VBench 离线模型与缓存 |
| `LightX2V` | 48M | 已验证的源码 revision `7bd01b7d5d04bf99cdd9ca9082f20c184c66d991` |
| `qwen-image21-venv`, `envs` | 4.8G、873M | Qwen、Wan、EntityBench 的虚拟环境；依赖原系统镜像的 `/root/miniconda3` |
| `onnx_cuda12_libs` | 1.1G | 旧评测/人脸链路的 CUDA 动态库，建议保留 |

另复制 `movie` 中的代码、`.git`、`.env.portrait`（含 API 密钥，**只走 SSH 私有链路，不提交 Git**）和未跟踪的本地文档。以上选择性复制约 195G，不含新生成结果的增长空间；目标数据盘建议至少有 250G 空闲，300G 以上更宽裕。若还要保留本机其他实验和将来重新运行 FaceLift/PuLID，直接迁移整个约 292G 已用数据盘，目标盘建议 400G 以上；`movie/third_party` 另占约 33G，当前集成路线未调用新 3D 注入。

当前集成结果：`run700` 10/10 完成并已做官方非 VLM 评测；`run714` 4/10；`run484`、`run1404` 尚未开始集成重跑。当前没有实验进程正在写这些目录。

## 建议复制顺序

先在 A800 实例确认 `df -hT /root/autodl-tmp` 显示有足够容量的数据盘，`nvidia-smi -L` 和 `nvidia-smi --query-gpu=name,memory.total --format=csv` 显示实际可用 GPU；A800 物理卡可能为 40G/80G，也可能被 MIG 切片。当前 81 帧视频实测 CUDA 保留显存峰值约 34.5GiB，推荐完整 80G 实例，避免按“A800”名称推断可用显存。

如果旧机能通过 SSH 连到新机，在**旧机**运行下列命令。填写目标 SSH 地址与端口；复制期间不要同时生成视频。命令没有 `--delete`，可在中断后重复执行。不要先在旧机打一个大压缩包：旧数据盘只剩约 29G。

```bash
export A800_SSH='root@YOUR_A800_HOST'
export A800_PORT='YOUR_SSH_PORT'

ssh -p "$A800_PORT" "$A800_SSH" 'mkdir -p /root/autodl-tmp/movie'
rsync -aH --partial --info=progress2 -e "ssh -p $A800_PORT" \
  --exclude '/models/' --exclude '/outputs/' --exclude '/third_party/' --exclude '/.venv/' \
  /root/autodl-tmp/movie/ "$A800_SSH:/root/autodl-tmp/movie/"

for rel in \
  movie/models/video/Wan2.2-I2V-A14B \
  movie/models/video/Wan2.2-LightX2V-Distill-Loras \
  movie/models/diffusion/qwen-image-2.1 \
  movie/outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140 \
  movie/outputs/entitybench_qwen21_i2v_a14b_4step_continuation_20ep \
  movie/outputs/entitybench_qwen140_rerun_pilot_20261026 \
  movie/outputs/entitybench_qwen140_integrated_episodes_v2_20261026 \
  models LightX2V qwen-image21-venv envs onnx_cuda12_libs; do
  ssh -p "$A800_PORT" "$A800_SSH" "mkdir -p /root/autodl-tmp/$(dirname "$rel")"
  rsync -aH --partial --info=progress2 -e "ssh -p $A800_PORT" \
    "/root/autodl-tmp/$rel" "$A800_SSH:/root/autodl-tmp/$(dirname "$rel")/"
done
```

若两台实例不能互通 SSH，按同样的路径清单通过平台的数据盘迁移/对象存储中转。必须保留目录结构和符号链接；不要只上传 `report.md`，因为集成结果的 `official_results` 链接指向基线或本轮视频。新机应限制 `.env.portrait` 权限：`chmod 600 /root/autodl-tmp/movie/.env.portrait`。

## 目标机验收与续跑

在**A800 新机**执行；先验证路径和完整性，再启动 GPU 作业。`prepare` 只读已有视频并检查它们的 SHA、父尾帧关系、基线及评测器版本，不启动评测 GPU。

```bash
cd /root/autodl-tmp/movie
df -hT /root/autodl-tmp
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total --format=csv
git rev-parse --short HEAD
git -C /root/autodl-tmp/LightX2V rev-parse HEAD
test -s models/video/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth
test -s models/diffusion/qwen-image-2.1/model_index.json
test -s outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140/manifest.json
test -s .env.portrait
export QWEN140_INTEGRATED_OUTPUT_ROOT=/root/autodl-tmp/movie/outputs/entitybench_qwen140_integrated_episodes_v2_20261026
python pretest/qwen140_integrated_episodes.py status
python pretest/qwen140_integrated_episode_eval.py prepare --name run700_v1 --runs run700
```

预期状态为 run700 `10/10 complete`、run714 `4/10 in_progress`；`prepare` 需输出 `1 episodes 10 videos`。若报 SHA 不符或路径不存在，先补传相应目录，**不要清空旧缓存重来**。确认 A800 能见到完整显存并支持 BF16 后，可用 `python pretest/qwen140_integrated_episodes.py run --run run714` 从已有结果继续；新镜头会在 `QWEN140_INTEGRATED_OUTPUT_ROOT` 下生成。当前 run714 之后的三集结果尚未在 A800 验证，首次开跑先观察一个镜头的模型加载、显存与尾帧审计。

系统盘镜像可复用原 `/root/miniconda3`、驱动用户空间及常用工具，但 GPU 驱动和 MIG 配置由新实例提供；虚拟环境路径若改变需重建。当前代码以 BF16、`torch_sdpa` 和单卡顺序运行 Qwen/Wan，未要求 H100 专有算子。新机首先运行上述无 GPU 校验，再做一个 GPU 镜头冒烟测试。
