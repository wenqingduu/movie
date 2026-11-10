# 固定首帧：动作强化＋背景线索（2026-11-08）

用户本轮授权：在当前动态0重跑shot中补强背景描述，同时处理动态与场景出现问题。只调整尚未派发的目标shot视频提示词，不改Agent、首帧、3D、人物素材、种子、模型参数、其他shot、过渡或整集。

## 切换边界

旧调度父进程已SIGSTOP，当前M0039–M0070的GPU worker正常继续，不中断或浪费正在生成的shot。新后台tmux `qwen140_allzero_background_20261108`等当前worker退出、确认前70项全部生成后，结束旧的空闲调度父进程并接续。旧context waiter仍在等待、未评分，已停止，由新版本普通依赖评分任务接续。没有守护/修复Agent。

旧目录 `outputs/entitybench_all_zero_fixed_frame_motion_20261108/` 的plan、task、视频与评分均不覆盖。新目录 `outputs/entitybench_all_zero_motion_background_20261108/` 保留1333个相同目标：前70项（6先导＋当前两批64项）复用已有/当前在途结果，不因新提示词再生成；剩余1263项中469项追加背景上下文、794项继续原动作强化词。当前批次结束后补评尚未打分的32项，再继续GPU生成。

## 最小提示词变化

只为当前基线有weak场景、且数据集有具体简短环境描述的469项补一句背景上下文。取同一调度场景原描述的一个不超过36词片段，随后要求已有可见背景线索保持可辨、遵从原运镜和景深、不显露画外元素。无具体信息的黑房间/未指明空间等不编造细节。默认只补一个场景，避免堆砌长描述；全部新旧词和来源在 `prompt_changes.json` / `prompts.jsonl`。

例M0076（run845/9:1）：

> Setting context: The Glass Corridor — An office corridor with a doorway and a large exterior glass window. Keep its already-visible background features recognizable within the specified camera movement and focus; do not reveal off-screen features.

这主要约束已有环境在运动中保留。首帧本来没有所需环境、极近特写看不到背景、原场景描述与当前视角不匹配的情况，不能保证靠视频词修好；不为追分突然拉远、加道具、改亮度或改首帧。6个原运镜类别均在保留的前70项；不得把新背景句解释成冻结原有动作。

## 验证、评分与替换

1333项逐条验证：任务除prompt外完全一致；原首帧路径/SHA、种子、模型、帧数/分辨率/步数、基线视频/评分及筛选政策不变。469项有且仅有背景句追加，前70保持原任务。旧冻结生成代码SHA全部通过。CPU检查证明在 `validation/prompt_contract.json`。持久模型原两例162帧bit-exact验证原样引用并保留来源SHA，不重复GPU试跑。

评分口径与门槛沿用用户确认版本：动态0→1；人物/物体/场景出现率不降；默认人物DINO、主体/时间稳定/平滑/美学各最多降.01，成像最多降1分；无新增漏人、身份漂移、分屏、动作错误。先6项VBench，再仅对初筛通过项重算完整受影响episode的presence、跨镜头中心与真实边界；做定位及画面复核后再发布独立筛选版。初筛通过不算最终通过；无分数或歧义不伪造提升，不改变CLIP阈值或裁剪口径。

保留前70个motion-only与后1263个motion/background任务标签，不能将混合策略称为同一纯版本实验，也不能宣称背景描述独立因果收益。本轮每shot仍只有一个候选，无多种子、重试或付费API。媒体/负例与尝试总数全保留。

## 运行位置

- 入口：`pretest/qwen140_all_zero_background_20261108.py handoff-run`。
- 新状态：`outputs/entitybench_all_zero_motion_background_20261108/status.json`。
- 新日志：同目录 `queue.log`、`chunk_###_generation.log`、`chunk_###_evaluation.log`。
- 后续评分：新目录 `context_evaluation/`，tmux `qwen140_allzero_background_context_20261108`在交接完成后自动启动。
- 原目录会写 `handoff.json` 记录新版本指针；旧status可能停留在最后一条旧生成状态，不再以其当新版本进度。
- 仍预计11月10日前后完成生成＋初评，完整语境评分与人工复核另计；增加背景句可能略增推理时间，依据新批次实测更新。

本轮不会恢复旧整集队列，不设磁盘配额或自动清理，不推送GitHub。
