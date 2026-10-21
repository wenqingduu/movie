# 历史实验说明与索引

更新：2026-10-21。这里保存旧方法、旧范围的结果，以及本次实验的早期过程记录，均不作为当前结果或执行指令。

当前主实验只有 **Qwen＋3D注入＋Wan A14B四步，140集／2491镜头**。其设置、最终指标与论文对照统一见[主交接](../../LLM_HANDOFF.md)；原始结果见[最终验收报告](../../outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140/archive/completion_20261019.md)。

## 旧方法与结果

| 历史范围 | 说明与结果入口 |
|---|---|
| 旧三集、34镜头及Qwen多人pilot | [三集结果归档](EPISODE_TEST_RESULTS_legacy_20261021.md)，保留原Control／Treatment、InsightFace、参考DINO和消融记录 |
| PuLID-FLUX／IP-Adapter旧20集 | [首帧／视频评估与身份指标](../entitybench_20ep_evaluation.md) |
| 旧SDXL→FLUX、Qwen20、新FLUX三路线 | [三路线／论文结果汇总](../entitybench_results_20ep_20261014.md)；旧20集分数不再混入当前全140集表 |
| 旧A14B 40步与四步加速试验 | [加速试验](../a14b_acceleration_trial.md) |
| 原生FLUX与Portrait Agent素材路线 | [原生FLUX记录](../entitybench_flux_native_20ep_run.md)、[Portrait Agent记录](../entitybench_flux_portraitagent_20ep_run.md) |
| 已停止的旧全140路线 | [旧全140运行记录](../entitybench_a14b_full140_run.md) |
| 人物裁剪、门控、聚合与接续边界口径 | [指标审计](../entitybench_metric_provenance_audit_20261014.md)，作为论文横比限制的历史证据保留 |
| 旧Qwen20运行过程 | [Qwen20交接](../LLM_HANDOFF_QWEN20_20261013.md) |

根目录 `EPISODE_TEST_RESULTS.md` 现在仅指向当前全140集结果。旧三集汇总脚本 `pretest/summarize_current_episode_results.py` 默认写入本目录的 `EPISODE_TEST_RESULTS_legacy_generated.md`，避免重新生成时覆盖当前入口；原有机器可读输出及评分逻辑保持。

## 本次整理前的完整快照

以下2026-10-21快照保存了本次整理前的工作树内容，包括当时尚未提交的文字；归档时修正了相对链接，并记录原文件SHA256。旧启动许可、资源数字和进度均为当时记录，不能据此自动启动任务。

- [原主交接](LLM_HANDOFF_before_current_20261021.md)：旧Qwen20分数、2026-10-14至19日准备／生成／恢复快照和历史资源记录。
- [原三集结果页](EPISODE_TEST_RESULTS_legacy_20261021.md)：旧方法与消融的完整结果。
- [原Qwen140执行交接](Qwen140_execution_before_current_20261021.md)：早期准入、启动、RGBA／角色分流恢复及阶段记录。
- [原新会话提示词](Qwen140_prompt_before_current_20261021.md)：旧的未启动时期提示词，已由当前提示词替代。

## 更早的交接档案

- [2026-10-14原主交接完整归档](LLM_HANDOFF_experiments_20261014.md)：早期三集、20集首帧／视频、其他组合、快照和清理记录。
- [2026-10-14 Qwen140整理前交接](Qwen140_handoff_before_refactor_20261014.md)：更早的容量／状态变更记录。

本次仅整理文档入口并调整旧汇总脚本的默认Markdown归档路径；历史媒体、模型、素材、3D、原始报告、冻结生成／评测核心和分数均保留。
