# run700：清理人物素材后整集单次重跑

2026-10-28。范围：`00050e2d-6c9a-38e6-b702-87f8df189087__run700__i25_j50__T300`，10个shot、6个角色。

原人物素材生成模板虽写了单人、纯背景，仍附加多角色、多场景的完整故事梗概。Leo素材为六格拼图，Walter素材为上下两层的多人画面；原4:1首帧也存在上下拼接。旧整集试验保留原硬切视频，因此没有消除4:1的分屏。故事上下文与单人要求冲突是可疑诱因，尚无消融证明其为唯一原因。

本轮使用新的输出目录，不覆盖冻结full140或之前试验：

`outputs/entitybench_qwen140_run700_clean_refs_single_pass_20261028/`

实际入口：`pretest/qwen140_episode_single_pass.py`。人物素材调用`qwen140_integrated_episodes.clean_reference_prompt`：仅角色原句描述＋写实画风＋单人中近景纯灰背景，明确禁止分屏、多格、其他人物和文字，不传故事上下文。全部6个角色重新生成；这会重新确定描述约束下的人脸外观，不保证与旧参考逐脸相同。采用Qwen原生人物参考条件，本轮不构建新3D素材或声称应用了3D注入。

所有10个shot重新生成。原硬切使用新人物素材建立新首帧；接续由qwen3.8-max读取本轮上一实际尾帧、前后shot、原句人物描述及物体/场景描述，调用一次人物素材查询，再输出A/B/C与短提示词。尾帧已可辨的人物不重复传参考图。当前关键帧可见角色通过`frame_characters`指定，隐藏人物不提前绘入首帧，仍保留其视频描述。

B用Qwen生成过渡目标尾帧，Wan3.0-video API接收首尾帧生成3秒过渡，再从解码实际尾帧生成81帧/16fps行动段。B合片约8秒，其他路线约5.0625秒，属于延长时长探索，不是固定5秒预算的对照。Wan API同一任务最多轮询30分钟，不重复提交收费任务。

每阶段单次生成，操作者查看一次素材/关键帧；明显生成错误或接口失败即停止报告用户，不自动修改计划、切路线或重试。阶段入口为prepare、refs、plan、frame、transition、main、finish、assemble，停止位置记录到输出目录`stopped.json`。

完成后输出新旧整集和逐shot视频，使用冻结官方非VLM口径比较六项VBench、实体出现率、跨镜头人物/物体DINO及接续边界；cs_face保持partial_without_vlm暂定口径，不采用额外人脸筛选抬分。

当前状态：6/6人物素材单次生成完成并逐张查看，均为单人单幅，无分屏或背景人物。整集生成与官方非VLM复评已完成，状态见输出根目录`progress.json`，指标与已知缺陷见本文末节和最终报告。素材浏览页：`outputs/entitybench_qwen140_run700_clean_refs_single_pass_20261028/references/run700/index.html`。

## 本轮停止记录

前5/10镜头已完成：1:1 C、1:2 C、2:1 C、3:1 C、3:2 A。3:3单次规划选B、只查Leo素材，但shot_prompt又增加“带着Leo离开”，且目标尾帧已站到柜前，提前处理主要行动中的移动。已按用户要求停止，无目标帧、过渡或重试，未开始整集复评。

[查看5个镜头前后视频及错误计划](../outputs/entitybench_qwen140_run700_clean_refs_single_pass_20261028/review/index.html)；[查看6张人物素材](../outputs/entitybench_qwen140_run700_clean_refs_single_pass_20261028/references/run700/index.html)。待用户决定如何处理计划，当前没有后台任务继续生成。

## 用户指示继续

用户明确表示“没事继续把这个episode生成来看效果吧”。沿用现有3:3计划，不修改提示词，不因已知语义偏差拒绝该计划，继续生成剩余镜头。原停止记录保留为`stopped_before_user_resume.json`，恢复指示见`resume_authorization.json`，当前状态以`progress.json`为准。技术错误仍停止，不自动重试或切换路线。

## 最终完成状态

10/10镜头及官方非VLM复评完成；A×1、B×1、C×8，新视频857帧约53.56秒。原分屏案例未在本轮抽查画面中再现，但3:5有旧人物残留、4:1仍有多余人物及酒杯归属错误。角色出现率96.67%→86.67%，cs_face 0.7851→0.7512，成像质量53.93→49.36，动态0.5持平，美学0.5861→0.5983。整体未改善，没有自动再优化。

[最终视频与完整报告](../outputs/entitybench_qwen140_run700_clean_refs_single_pass_20261028/report.md)。状态：`complete_generation_and_non_vlm_pending_full_video_visual_review`；先前停止段落仅保留为历史记录。
