# run700：恢复原3D主线后叠加镜头调度与提示词

用户明确要求：3D注入是核心方法，不能取消；A/B/C调度、角色提示词和统一画风属于新增内容。上一轮native-only整集不能作为保留原核心方法的改进版结论。

四图的角色/版本：第1张是native-only重跑3:2的Stefan，第2张是原full140的Stefan；第3张为native-only重跑1:1的Arthur，第4张为同轮4:2的Arthur。不是四张同一人物。前两张还更换了Stefan参考身份，此次纠正。

原full140实际分流为单角色硬切首帧＋可靠人物参考/可用3D资产时执行v7 color-safe。run700实际注入2:1、3:1，都是Stefan；多人硬切不注入，接续视频不逐帧注入。闭眼、安全脸框、Gaussian渲染、姿态和mask门控继续使用原实现。此次没有扩展多人注入。Arthur的1:1、4:2均为多人镜头，不能宣称恢复单人注入就解决他的身份一致性。

恢复参数：1024×576、20步、step12、强度0.4、cosine阈值0.40，低相似度6步、高相似度1步，最小脸置信度0.5，sequential CPU offload、KV cache。Control由官方Qwen原生流程生成，Treatment从同一次Control的step12 latent分支，逐步施加局部3D残差。

Stefan恢复原full140参考图、FaceLift Gaussian及273点/512标定，SHA成对冻结；其余5人保留上一轮已经清理的素材，不再随机生成人脸。硬切首帧使用原shot seed；新增C/B帧使用既有派生seed规则。所有镜头均保留人物原句描述、episode写实画风及A/B/C新调度；本轮为集成重跑，不是单因素消融。

实际调用原`pretest/run_qwen_image21_shared_prefix_adaptive.py:run_shot`，未改v7算法。两个要求注入的镜头若未真正注入则停止，保留Control、prepared和原因，不静默使用Control继续。多角色和接续段有明确skip记录，直接接续继承上一实际尾帧。

输出：`outputs/entitybench_qwen140_run700_v7_restored_20261028/`。

启动：设置`QWEN140_INTEGRATED_OUTPUT_ROOT`为该目录，再运行`pretest/qwen140_episode_single_pass.py run-remaining`。`face_injection_policy.json`冻结全部人物参考、Stefan 3D、标定与v7实现SHA；`rerun_manifest.json`标明核心方法；`progress.json`记录进度。native-only默认准备入口已禁用，必须提供明确注入策略。

完成后核验2:1、3:1的`injection_applied`、活跃步与非零残差，并验证视频实际采用Treatment，再做整集官方非VLM复评。旧full140、native-only试验全部保留。

当前状态：10/10生成及官方非VLM复评完成；检查了全10镜头的开始/中间/结尾采样，尚非逐帧人工验收。结果与残留问题见[本轮报告](../outputs/entitybench_qwen140_run700_v7_restored_20261028/report.md)，视频见[整集对比](../outputs/entitybench_qwen140_run700_v7_restored_20261028/review/index.html)。角色出现率96.67%→100%，人物区域cs_face 0.7851→0.7720，动态程度仍0.5。恢复核心3D不意味着所有指标都会提升。

用户后续要求尝试多人首帧注入，另设4:1配对试验；不覆盖本轮完整episode。见[多人注入方案](entitybench_run700_multiface_trial_20261028.md)。
