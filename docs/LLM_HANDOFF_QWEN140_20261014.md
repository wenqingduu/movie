# Qwen140：当前执行与核验入口

更新：2026-10-21（北京时间）。本次只交接当前全140集实验；实验设置与最终指标统一见[主交接](../LLM_HANDOFF.md)，解释与证据见[当前指标说明](entitybench_qwen140_metric_interpretation_20261021.md)。旧实验仅留[历史引用](history/README.md)，不要求通读。

## 完成状态

Qwen＋3D注入＋Wan A14B四步，完整140集／2491镜头的生成、官方非VLM、独立脸部DINO及归档均已完成，生成失败0，1355个自身尾帧接续核验通过。原队列正常退出0，不要重复启动。

状态为 `complete_non_vlm_pending_full_video_visual_review`。全部脸部裁剪已审阅；全部成片人工审阅尚未完成。当前先不处理VLM，不调用API，不启动付费评分。

## 文件与入口

输出根目录：`outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140/`。以下产物路径均相对此目录。

| 内容 | 文件 |
|---|---|
| 状态与冻结设置 | `status.json`、`manifest.json`、`generation_manifest.json`、`runtime/` |
| 生成记录 | `generation_report.json` |
| 最终验收与视频查看 | `archive/completion_20261019.md`、`archive/viewer.html` |
| 媒体及接续核验 | `audit/final_media.json` |
| 官方评分核验 | `audit/final_official.json` |
| 官方原始结果 | `evaluation/official_parallel_v2/` |
| 脸部版与独立核验 | `evaluation/face_only_dino/report.json`、`evaluation/face_only_dino/independent_validation.json` |
| 裁剪审阅决策 | `evaluation/face_only_dino/review/decisions.json`、`evaluation/face_only_dino/frozen_review.json` |

以下代码路径相对仓库根目录，列出仅供查证，不构成重新执行指令。

| 用途 | 文件 |
|---|---|
| 全量计划与素材／首帧 | `pretest/prepare_entitybench_qwen21_full140.py`、`pretest/run_qwen21_entitybench_full140_images.py` |
| 视频生成 | `pretest/run_lightx2v_a14b_4step_qwen140.py` |
| 媒体与官方评分最终核验 | `pretest/audit_entitybench_qwen140_final.py` |
| 独立脸部DINO | `pretest/evaluate_entitybench_face_only_dino_qwen140.py` |
| 脸部最终核验与归档 | `pretest/audit_entitybench_qwen140_face_final.py`、`pretest/finalize_entitybench_qwen140.py` |

## 接手顺序

1. 阅读主交接、当前指标说明及最终验收报告，检查git状态、实时状态与实际进程，保留未提交改动。解释时明确区分官方结果、实例诊断、抽查观察和待验证因素。
2. 按需核查既有JSON、SHA、覆盖／失败／缺失计数，不修改冻结核心、门控规则、裁剪选择或原分数。
3. 需要运行重任务时重新检查GPU、cgroup内存额度、CPU与磁盘；不得沿用旧资源快照。
4. 等待用户下一条任务，不自动启动生成、重评、VLM、全片审阅或提交推送。

评测审计使用[evaluate-face-injection-v2技能](/root/.codex/skills/evaluate-face-injection-v2/SKILL.md)。只有需要排查实现时才按需阅读[本次恢复与完成记录](entitybench_qwen140_posteval_recovery_20261019.md)，本次接手无需回顾早期过程。
