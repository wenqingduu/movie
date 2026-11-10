# 再增24条原版40步动态零候选（2026-11-09）

## 最终状态（北京时间19:19）

本轮已完成24/24生成＋六项VBench，8/24动态0→1，新增8条替换；16条未增长保留原媒体，叠加旧20，完整140集2491shot累计28替换／2463不变。旧20媒体和所有原文件均保留。8个受影响episode／171完整shot的上下文评分、可见单脸检测和默认DINO重算完成，评分失败0，171×6缓存分数一致。tmux／worker已退出，GPU0%／约1MiB，无新队列。

- 最终完整表：`outputs/entitybench_original40_next24_20261109/replacement/complete_report.md`。
- 三版本完整表（冻结替换前／selected20／累计28）：`replacement/full140_three_versions_report.md`及同名comparison JSON。
- 媒体映射：`replacement/replacement_manifest.json`（本批replaced=8，cumulative_replaced=28），实际2491链接在`replacement/candidate_media/`。
- 验证：`generation_completion_validation.json`、`replacement/official_completion_validation.json`、`replacement/completion_validation.json`、`replacement/completion_integrity_audit.json`。
- 默认DINO：`replacement/visible_region/report.json`；6份隐藏分数新裁剪，助手核对1页材料＋全部4个精确完整帧和原文，6保留／0新增剔除，57项旧排除精确SHA沿用。保留Walter外套互换、Isabella重复、Julian动作错误；这是助手视觉复核，不是外部人工签核。decisions已绑定manifest SHA后重算140人物质心与聚合；auto仍为历史中间产物。

| 完整140集指标 | selected20版 | 累计28版 |
|---|---:|---:|
| 默认人物DINO（可见单脸＋定位纠错） | .7989 | .7990 |
| 主体一致性 | .9417 | .9415 |
| 时间稳定性 | .9864 | .9864 |
| 运动平滑度 | .9920 | .9920 |
| 动态（官方四位舍入） | .4800 | .4840 |
| 美学 | .5859 | .5859 |
| 成像0–100 | 60.8571 | 60.8665 |
| 角色出现率 | .9479 | .9479 |
| 物体出现率 | .8981 | .8981 |
| 场景出现率 | .6238 | .6238 |
| raw人物区域DINO | .7239 | .7240 |
| raw物体区域DINO | .8077 | .8077 |
| 接续边界 | .8452 | .8448 |

动态精确.480048511430→.484003056885，**本批+.395455个百分点**；相对最初冻结替换前.473352984735，累计**+1.065007个百分点**。DINO有效集130／计分角色次1467不变；全调度4991，present4735→4735，自动合格1838→1839，复核保留1781→1782。其余数字为官方函数四位舍入，显示相同不代表未舍入完全相同。主体与接缝下降照报，无额外质量过滤。strict/skip_vlm、VLM0，VLM相关项仍缺失，raw cs_face/cs_object缺保真门控，仍为partial_without_vlm。

新增成功：N36/run184/14:1、N39/run227/4:1、N40/run348/5:1、N47/run582/3:1、N49/run755/5:1、N50/run793/5:1、N52/run801/4:1、N53/run851/1:1（完整窗口见plan与下方名单）。全部24尝试均保留在comparison.json；旧1333和整集保持暂停，未自动启动下一批。下方为原启动方案和历史耗时预估。

北京时间12:43启动。用户在selected20完整评测后要求继续跑此前动态0的镜头并替换；本轮按小批量选24条，不恢复1333旧队列。此前20条独立替换版保留，本批只追加新的动态0→1，其他指标下降全部报告。

12:45启动核验：模型已加载（85.9秒），`resolved_runtime.json`确认WanScheduler40／CFG3.5／noLoRA／shift5／boundary.9／TorchComplexRoPE与双精度旋转；N33/run4/i35_j83/4:1已完成首帧VAE与文本编码并进入真实40步采样。尚未首条81帧出片，不把启动核验当生成完成。GPU使用约34GB，仅本轮一个生成worker。

- 输出：`outputs/entitybench_original40_next24_20261109/`。
- tmux：`qwen140_original40_next24_20261109`。
- 入口：`pretest/qwen140_original40_next24_20261109.py run`；上下文后续入口`pretest/qwen140_original40_next24_context_20261109.py`。
- `plan.json`／`validation.json`：24条名单、363项绑定、已有4步实际生成与评分SHA、相同PNG／逐字prompt／seed／帧数尺寸合同；新case编号N33–N56。
- `baseline_snapshot.json`：**已完成selected20的140集／2491shot版本**，不是11-06替换前快照，也不是32条目的样本均值。
- `context_contract_validation.json`：纯缓存复现selected20官方140集聚合、同协议DINO基线130有效集／1467角色次，无新GPU推理。正式新视频上下文评测在生成＋六项VBench结束后运行。

## 固定配方与选择

沿用原版WanScheduler40步、CFG3.5、shift5、boundary.9、noLoRA、BF16／SDPA／TorchComplexRoPE及双精度旋转。首帧／3D／种子／81帧16fps／尺寸／已有运动强化词全部沿用，不添加动作、运镜或背景句。全部24条当前benchmark动态0、已有同词强化4步动态0，排除原32条。原文确有行走、进出、上下台阶、跑出、攀爬等位移；原category=travel里不符合该语义的细小手部动作没有纳入这次小批量。

这是一组有目的选择的后端配方比较，不能把结果当随机全benchmark或独立步数／LoRA消融。只能按动态增长决定媒体替换，没有美学、主体、出现率、DINO等新增过滤门槛。

## 队列顺序与产物

1. 三批，每批8条原版40步生成；单个worker持续加载模型完成该批，退出后评原六项VBench。每条生成结果即时持久保存，所有负例和降分保留。
2. 全24结束后，`comparison.json`记录24条完整尝试和三列分数；构建`replacement/replacement_manifest.json`，保留原20、仅新增本批动态0→1，完整2491媒体符号链接写入`replacement/candidate_media/`，旧文件不覆盖。成功数尚未知，不能说“新增替换24”。
3. 为新增成功镜头所在的完整episode重算定位／presence／人物物体质心／真实接缝，精确SHA复用六项VBench，两个评分进程同一张4090各4CPU线程；其余episode复用selected20结果，官方函数聚合全140。`replacement/official_report.md`和验证JSON记录官方非VLM12项。
4. 原代表人物区域做可见单脸检测，输出`replacement/visible_region/review/manifest.json`、隐藏分数图与`automatic_results.json`。**自动结果不是默认最终DINO。** 必须逐项视觉复核、按精确crop SHA冻结`review/decisions.json`，再运行上下文脚本`final`动作，重算完整人物质心和140聚合，输出`complete_report.md`。不得删除真实漂移、重复、未决以提分；助手复核和外部人工签核如实区分。

`status.json`／`queue.log`／`next24_batch_*_generation.log`／`next24_batch_*_evaluation.log`／`context.log`为实时依据；上下文细节看`replacement/status.json`及`replacement/workers/*/worker.log`。阶段之间释放生成模型，不与上下文评测重复争GPU。脚本只运行本轮依赖任务，没有守护修复、自动改词、降低配方或恢复旧队列。

上一批24条实测每条约934秒，24条净生成约6小时14分，另加三次模型加载、VBench和完整上下文重算。本轮12:43启动，预计北京时间19点左右生成结束，随后完成自动评测；全量指标还需定位复核，技术异常单独计。不要把预计时间当完成证据。

strict／skip_vlm、API／VLM调用0，VLM相关项仍缺失，raw cs_face/cs_object缺fidelity门控，保留partial_without_vlm。磁盘约13GB可用，无人为reserve或自动删除。不读凭据、不提交／推送、不覆盖原媒体。

## 冻结名单（24条／19个episode）

| 新case | 旧候选 | episode窗口 | shot | 原文已有主要运动 |
|---|---|---|---|---|
| N33 | M0014 | run4/i35_j83 | 4:1 | 走向警署正门 |
| N34 | M0015 | run4/i35_j83 | 5:1 | 进入审讯室并走向桌子 |
| N35 | M0026 | run155/i5_j39 | 3:1 | 走向栖架 |
| N36 | M0029 | run184/i25_j59 | 14:1 | 服务员端箱走过 |
| N37 | M0030 | run184/i25_j59 | 14:2 | 走到柜台后放下箱子 |
| N38 | M0032 | run184/i25_j59 | 15:1 | 走入巷子后停下呼喊 |
| N39 | M0040 | run227/i25_j77 | 4:1 | 闯入房间并挥手 |
| N40 | M0051 | run348/i0_j26 | 5:1 | 穿过停车场 |
| N41 | M0052 | run354/i90_j105 | 3:1 | 沿走廊向前走 |
| N42 | M0053 | run354/i90_j105 | 8:1 | 从走廊远处走向镜头 |
| N43 | M0057 | run473/i10_j29 | 2:1 | 走进房间 |
| N44 | M0058 | run474/i25_j57 | 3:1 | 走向咖啡桌 |
| N45 | M0059 | run474/i25_j57 | 3:5 | 赤脚穿过街道 |
| N46 | M0061 | run501/i0_j17 | 1:4 | 提篮穿过田垄 |
| N47 | M0064 | run582/i0_j47 | 3:1 | 从门内跑出并跳舞 |
| N48 | M0067 | run629/i115_j139 | 1:2 | 走到院中再坐台阶 |
| N49 | M0070 | run755/i15_j49 | 5:1 | 进门并起身相拥 |
| N50 | M0072 | run793/i55_j64 | 5:1 | 端餐从背景走近 |
| N51 | M0073 | run799/i5_j15 | 5:1 | 三人走向入口，原文跟拍 |
| N52 | M0075 | run801/i35_j45 | 4:1 | 攀爬消防梯 |
| N53 | M0077 | run851/i20_j44 | 1:1 | 从阴影走向前 |
| N54 | M0082 | run1092/i5_j22 | 2:1 | 从沙地起身走向桌子 |
| N55 | M0094 | run1535/i40_j60 | 4:1 | 轿车驶来、下车并走向入口 |
| N56 | M0095 | run1663/i50_j74 | 6:1 | 多人一起走下楼梯 |
