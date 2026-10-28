# 接续镜头 Agent：精简提示词与单次执行

更新：2026-10-28，保留原文件名。本文件下方的 `text` 代码块是实际系统提示词，`pretest/qwen140_shot_agent.py` 启动时直接读取；修改此处即可生效。B样例入口为 `pretest/qwen140_single_pass_shot.py`；整集单次入口为 `pretest/qwen140_episode_single_pass.py`，接入A/B/C及硬切新首帧。旧四集执行器的多轮质检和自动回退属于历史实现，新入口不使用它们。

规划模型：`qwen3.8-max`，读取 `.env.portrait` 中的 `SHOT_AGENT_MODEL`。此前试跑使用 `qwen3.7-plus`。

核心方法约束：A/B/C、人物描述与episode画风是原Qwen＋v7 3D注入＋Wan方法上的新增内容，不得默认取消3D。当前整集入口默认恢复原full140的单人硬切首帧v7策略；run700要求2:1、3:1实际注入，其他多人/接续段保留原分流，详见[恢复记录](entitybench_run700_v7_restored_20261028.md)。此前native-only试跑仅作历史结果，不代表保留原方法的改进版。

## 系统提示词

```text
看上一实际尾帧及前后shot，选A直接接续、B过渡+行动、C新镜头。先结合人物原句描述确认尾帧中谁是谁；尾帧中已清晰可辨的角色直接沿用，只调用lookup_characters查询未出现或无法确认的所需角色素材，无需新增素材时names为空。图片编号绑定人物，当前shot明确的服装与道具归属优先。
返回JSON：route、tail_identity（尾帧可见角色及位置/衣服）、frame_characters（待绘关键帧此刻可见的角色名）、frame_prompt、transition_prompt、shot_prompt。只查询待绘帧可见且尾帧缺失的角色素材；稍后才出现的人物只写进视频行动，不提前画入首帧。硬切没有尾帧图时选C，从人物参考建立新场景。
A：直接延续上一尾帧；frame_prompt和transition_prompt为空，shot_prompt写当前行动。
B：三项提示词均必填。frame_prompt是给图像模型绘制“过渡视频目标尾帧”的提示词，描述过渡结束后的静态画面、人物位置、道具归属与机位；transition_prompt写从上一尾帧到该目标帧的入场/状态变化；shot_prompt写过渡后才执行的主要行动。目标尾帧不提前完成主要行动。
C：frame_prompt是给图像模型绘制“新shot动作起始首帧”的提示词，建立当前场景、机位和人物，动作尚未完成；transition_prompt为空，shot_prompt写从该首帧开始的行动。
例：上一尾帧只有甲在桌边扶着篮子；当前shot的描述是乙从右侧入场，两人到柜前，甲开抽屉查找。只查乙的素材，甲沿用尾帧。B的frame_prompt应写：原场景镜头拉宽，乙已在右侧，甲已提起篮子，保留原桌子等环境，柜子可见且抽屉关闭。transition_prompt写乙入场、甲提篮、镜头拉宽；shot_prompt再写两人走到柜前、甲开抽屉查找。若需切换到新场景/机位则选C，frame_prompt直接描述新镜头的动作起点。
每段提示词控制在120个中文字内；人物原句描述由工具自动拼接，不重复抄写。只规划一次；无法判断返回route=STOP及reason，问题交用户，不重试或改路线补救。
```

## 输入与执行

- 输入：上一实际尾帧图像、上一 shot 内容、当前 shot 内容与角色调度、所有调度角色的数据集原句描述、episode 画风。
- Agent 实际调用 `lookup_characters(names)`，只选择尾帧中缺失或无法确认的所需人物素材；清晰已有角色不额外传人物图。工具按 episode-local 名称返回已有素材路径与图片编号，这些参考图随后送给 Agent 写提示词。所有调度角色的原句描述独立保留，未选人物图的角色也会进入生成提示词。
- 接续帧生成时图1为上一实际尾帧，后续图片按工具返回编号与角色名绑定；硬切不传上一尾帧，人物素材从图1编号。人物素材仅提供人物外观，不引入素材背景；不从文件名猜身份。`frame_characters`限定当前关键帧的可见角色，尚未出现角色的描述只进入视频提示词。
- 执行器自动把人物原句描述拼进帧、过渡视频和后续行动提示词。Agent 的短输出只负责机位、位置、道具和阶段动作；当前 shot 明确的服装/道具要求优先于通用人物描述。
- B按“目标帧 → Wan3首尾帧过渡 → 实际解码尾帧 → Wan行动段”执行；A直接使用实际父尾帧；C生成新首帧后执行行动。原始cut标记保留，硬切选C且重新生成，不复用旧视频。
- 每步只生成一次。不调用视觉验收Agent，不自动纠错、重试、补素材或改路线。计划缺项、素材缺失、接口失败或明显生成问题时停止，交给用户决定。读取并轮询已有API任务不会重复提交收费任务。

最小字段约定：A需要非空 `shot_prompt`；B需要非空 `frame_prompt`、`transition_prompt`、`shot_prompt`；C需要非空 `frame_prompt`、`shot_prompt`。这是必要输入检查，不是画面评分或多轮验收。

## 本次3:3状态

第一次单次计划已查询Leo、Stefan及数据集一句话描述，但B路线的 `frame_prompt` 为空，停止于 `run700_3_3_single_pass_agent_20261028/agent/plan.json`。

本次明确B目标尾帧/C起始首帧语义，并加入抽象例子后，在新目录重新单次规划。B的三个提示词均非空；目标帧要求Stefan在左侧提篮、Leo在右侧、镜头拉宽露出关闭的柜子，两人尚未走至柜前。行动提示词却新增“Stefan带Leo离开”，原文只指定Stefan转身离开。因此按用户要求停止，没有自动修改或再次规划，也没有生成新目标帧或视频。

最新记录：`outputs/entitybench_qwen140_integrated_episodes_v2_20261026/experiments/run700_3_3_single_pass_agent_v2_20261028/agent/plan.json`。

此前独立试验的目标帧和Wan3过渡仍保留在 `experiments/run700_3_3_transition_target_retest_20261028/`，不是本次精简Agent的新产物。

### 按用户再次编辑后的提示词重跑

2026-10-28：运行器已改成直接读取本文件的 `text` 代码块，本次记录中的系统提示词与文档一致。新目录 `run700_3_3_single_pass_agent_v3_20261028` 中的B计划字段完整，但仍写了“Stefan转身带着Leo离开画面”，原文只指定Stefan离开。按用户要求停止于计划阶段，没有自动删改、重试或生成新视频。

### Max规划试跑

2026-10-28：现有接口列出 `qwen3.7-max`、`qwen3.8-max`，规划专用配置已设为 `SHOT_AGENT_MODEL=qwen3.8-max`，两次调用实际返回模型均为 `qwen3.8-max`。系统提示词保持不变；计划字段完整，未再添加Leo离开动作。

一次Qwen目标帧生成后，前景和镜子附近出现两个Stefan形象。按用户要求停止，没有自动重试、替换素材或继续生成过渡/行动视频。新记录在 `experiments/run700_3_3_single_pass_agent_max_20261028/`，图片为 `frame/target.png`。

### 仅为缺失角色查询素材

Agent先读取所有调度角色的数据集原句描述，判断实际尾帧中谁已可辨；仅查询未出现或无法确认的角色参考图。删除了样例输入强制Leo、Stefan都选素材的要求。新计划实际只选择Leo；Stefan的文字描述仍拼入帧和视频提示词。单次生成目标帧未出现重复Stefan；Wan3过渡任务821ac995-6943-4fd9-9445-d02bb274d73b在本地等待15分钟后仍为RUNNING，已停止轮询，保留任务供用户决定是否继续等待；未重复提交，后续行动段未启动。记录目录：`experiments/run700_3_3_single_pass_max_missing_refs_20261028/`。

### 原任务恢复成功，完整B样例已生成

同一Wan3任务 `821ac995-6943-4fd9-9445-d02bb274d73b` 最终在约19分钟后成功返回，未重复提交。已从其实际解码尾帧完成一次Wan行动段，并拼成8秒样例。输入链已核对：目标帧仅两张图（上一尾帧＋Leo参考），两人的原句描述均保留；行动段输入为过渡实际尾帧，不是规划目标图。

[查看目标帧、过渡、行动段和合片](../outputs/entitybench_qwen140_integrated_episodes_v2_20261026/experiments/run700_3_3_single_pass_max_missing_refs_20261028/index.html)。抽帧可见Stefan开抽屉查看并转身离开，Leo留在旁边；背景家具/镜面几何仍随运镜变化。未自动重试，未评分，未替换原整集。
