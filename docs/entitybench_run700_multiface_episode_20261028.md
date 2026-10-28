# run700：多人3D接入后的完整episode

用户要求完整episode验证。输出：`outputs/entitybench_qwen140_run700_multiface_episode_20261028/`；入口：`pretest/qwen140_run700_multiface_episode.py`。不覆盖单人集成版和4:1配对试验。

本轮范围是所有新生成的B过渡目标尾帧、C动作起始帧；单人/多人都使用同一v7，逐角色匹配和准备3D。A继续复用自身上一视频实际解码尾帧，不生成额外图像。相比上轮只注入两个单人硬切，本轮还扩大到了接续镜头的新关键帧，不是仅硬切多人开关的单因素消融。

人物素材清理已在无3D探索版使用：6人素材都去掉故事上下文，只用角色原句、画风和单人参考约束。恢复单人版沿用其中5人，Stefan恢复原full140参考以绑定原Gaussian；本轮继续冻结这些身份，不重生人物。Stefan沿用原3D，Elias/Walter沿用4:1试验新建的3D，Arthur/Leo/Nora由对应干净参考补建Gaussian和273点/512姿态标定。

Agent继续用qwen3.8-max，输入前后shot、当前要求、上一实际尾帧和原句描述；查询新角色素材，输出A/B/C及短提示词。模型、seed规则、episode写实画风、角色原句拼接保持现有实现。每个阶段单次生成，保留语义偏差供审阅，不自动修复计划、重试、切路线或换模型。

Qwen的生成条件图与3D身份资产分开：Control和Treatment严格使用同一组Agent选择的条件图，包括上一实际尾帧及缺失角色参考；尾帧中已有角色的2D参考仅供身份匹配与3D准备，不重复加入生成条件。这避免把旧角色重画成多个副本。原v7算法文件未修改，worker仅在两个条件构建入口传入指定图像序列。

Qwen关键帧1024×576、20步、CFG1、KV cache、sequential offload；step12分叉原v7，强度0.4、身份阈值0.4、低身份6步/高身份1步，沿用脸大小缩放、置信度及渲染/姿态/mask门控。允许角色级跳过并记录；整张图无可用角色时停止，不能静默用Control继续。

B使用Wan3.0-video API首尾帧生成3秒过渡，再以实际过渡尾帧交给Wan2.2-I2V-A14B＋LightX2V四步LoRA生成81帧/16fps行动段。C从新首帧生成同规格视频；A直接从上一实际尾帧生成。B额外增加时长，保留原benchmark cut标记用于评测。

生成完整10shot后，运行冻结官方strict/skip_vlm复评全部12项非VLM指标，与原始run700及上轮单人集成版同时对比。VLM调用0，cs_face/cs_object缺保真度门控，保持暂定口径。保存实际注入/跳过角色、父子尾帧SHA链、整集和逐shot视频。本轮不另生成全套Control视频，不能把整集变化当作插件独立增益。

运行状态以输出根目录`progress.json`、`logs/`和`stopped.json`为准；若技术错误或全部角色无法注入，立即停止并报告用户决策，不启动替代方案。

1:1已完成：Arthur/Nora实际注入；Leo侧躺3D姿态未过门控，角色级跳过。1:2只有Leo，目标roll−49.55°，回测−23.20°，误差26.35°超过原15°门限，曾按要求停止。用户随后明确授权“仅该帧跳过3D注入，继续整集；其他关键帧仍按计划注入”。已冻结该次Control作为1:2首帧，没有重生或放宽门限；决定记录在`user_decisions/shot_1_2_skip_injection.json`。其它帧仍使用原注入策略，无整集回退授权。

先前中断：完成前6/10时停止在3:4。该帧Control多保留了Stefan；多人准备先按面积取两脸，排除了真正的Nora，随后将右侧Stefan错配成Nora并注入。Walter侧脸3D准备失败。这张图不能算作Nora正确注入。当时已停止整集及子进程，未自动修复或重跑；随后用户决策及修复记录见下文。

只读验证：对全部三张检测脸做参考身份匹配，可选回中间真正Nora，cosine 0.3517；错误右脸对Nora仅0.0713。修正匹配不会自动移除多余Stefan，后者仍是关键帧生成问题。证据和已完成视频见[前6shot及错配对照](../outputs/entitybench_qwen140_run700_multiface_episode_20261028/review_partial/index.html)；机器记录在`audit/shot_3_4_wrong_assignment/diagnostic.json`。当时尚无完整episode指标；现已完成，见末尾结果入口。


## 3:4 用户批准匹配修复后续跑

用户明确选择“修正多人脸匹配，从3:4继续，保留前6个shot”。原错误尝试完整归档在 `failed_attempts/shot_3_4_area_selection_v1`；前6镜头结果和视频SHA冻结。v2策略改为在所有可靠检测脸上执行角色与人脸的全局一对一匹配，再选择注入目标；不改3D、姿态门控、mask和注入参数。3:4保持同一Agent计划、提示词、参考、seed，重生成关键帧。额外Stefan仍保留为生成问题。原v1策略与脚本快照保留；后4镜头使用v2策略，不能称为完全统一的匹配版本。

匹配修复验证已完成：3:4新旧Control转RGB后逐像素一致；Nora实际注入框从错误右脸 `[704,166,796,292]` 改为中间真人物 `[416,198,505,306]`，cosine 0.351664。Walter仍因原姿态门控跳过。200组原有矩形匹配案例与旧全局最优结果一致，另验证缺人和无脸行为。详见 `audit/shot_3_4_wrong_assignment/fix_validation.json`、`matcher_regression.json` 和 `before_after_fix.jpg`。

用户随后确认“还按照你之前的修复继续吧”。因此本轮保持v2全脸身份匹配，不新增面积/尺寸禁注入阈值；已完成7/10，原进程从3:5暂停点恢复。面积筛选讨论单独记录，不改变本轮冻结策略。

### 面积筛选与身份匹配的区别

原 `select_faces` 按面积取前N（N为调度角色数），并无最小面积阈值。3:4中三脸面积分别约Walter10388、Nora9558、额外Stefan11567像素²，所以Nora在身份匹配前被截掉；随后固定角色集合的一对一配对将Nora错误赋给Stefan。v2取消该数量截断，先在全部可靠检测脸中匹配。本轮不新增小脸禁注入规则；仍按脸高/128将强度缩放到基础值的30%至100%。身份cosine 0.4只分配低身份6步/高身份1步，并非身份可靠性通过线。未来若增加尺寸资格，必须对已匹配的脸过滤，被过滤角色跳过且不补位；此项尚未接入本轮。

## 完成状态

10/10生成及12项官方非VLM复评全部完成，输入链、前6shot保持、注入残差与评测输入SHA校验通过。VLM调用0，非VLM失败0。详细指标与剩余语义问题见[本轮报告](../outputs/entitybench_qwen140_run700_multiface_episode_20261028/report.md)，视频见[三版本对照](../outputs/entitybench_qwen140_run700_multiface_episode_20261028/review/index.html)。抽查范围为每镜头首/中/尾帧，不冒称逐帧或VLM身份/画风验收。
