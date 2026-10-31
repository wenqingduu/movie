# 第二批低分episode：优先cs_face与人物出现率（2026-10-29）

已获用户授权启动第二批完整episode重跑。排除上一批4集及run700；限定每集不超过15shot，以控制本轮规模，非全140重跑，也非全局最低cs_face的所有长集。

| episode | shot数 | 原人物出现率 | 原cs_face | 选择原因 |
|---|---:|---:|---:|---|
| run317 / i25_j60 | 15 | 0.7556 | 0.6103 | 两项均低 |
| run15 / i0_j19 | 10 | 0.7000 | 0.7624 | 未重跑短集中人物出现率最低 |
| run354 / i90_j105 | 15 | 1.0000 | 0.5949 | 未重跑≤15shot集中cs_face最低 |
| run348 / i30_j49 | 15 | 0.7333 | 0.7063 | 两项均低 |

共55shot、24个episode-local角色身份。全140基线均值：人物出现率0.9212383153，cs_face 0.7155259111。所有选择基于原版正式逐集结果，不按新生成分数筛选镜头。

## 固定实验设置

- 首个shot起统一冻结当前硬切全人物规则、接续继承人物描述修复及精简Agent（qwen3.8-max）；A/B/C，B固定1+1。不混用上一批生成的shot。
- 保留Qwen-Image-2.1＋FaceLift/v7＋Wan2.2-I2V-A14B四步；所有新B/C关键帧按可见角色执行单人/多人3D，A使用自身实际尾帧。全检测脸全局匹配，原门控、强度与姿态阈值不变。整张未注入仍使用同次Control并记录。
- 基于数据集角色原句生成干净单人参考及配对Gaussian/标定，统一episode写实电影画风；抽查四集首/中/尾共12张原版图以确认摄影画面，run317黑白照片特写可作为单镜头要求。素材身份与原版不固定，不作3D插件独立因果结论。
- 每集完整生成后按原冻结官方strict/skip_vlm评12项，采样/阈值/DINO处理器不变，VLM调用0。cs_face仍标partial_without_vlm人物区域DINO。生成与官方评测串行，不争抢单GPU。
- 每集评测后自动整理原/新canonical人物裁剪、是否入分、采样帧序号、质心相似度。图像索引不代表人工身份验收，不根据裁剪好坏覆盖官方指标。
- 原始benchmark与前四集结果保留只读，输出新目录；代码/提示词与素材SHA冻结，输入不符拒绝续跑。技术异常保存有效前缀并继续其他episode；缺失集不会填0或混入完整配对。

## 后台与查看入口

- tmux：`qwen140_multiface_focus4`。入口：`pretest/qwen140_multiface_focus_episode_batch.py run`。
- 根目录：`outputs/entitybench_qwen140_multiface_focus4_20261029/`。
- [视频/指标索引](../outputs/entitybench_qwen140_multiface_focus4_20261029/index.html)、`status.json`、`eta.json`、`queue.log`。
- 各集`experiments/<run__tag>/review/index.html`：原始/新整集与逐shot视频。
- 各集`review/cs_face_crops/index.html`：原/新官方真实人物裁剪对照及`inventory.json`。
- 各集`report.md`和根`report.md`：逐集及同一已完成episode集合的正式对比。
- 预计6–8小时，即北京时间10月30日约05:00–07:00完成；GPU/API连续可用为前提，当前正式ETA以运行根`eta.json`为准。SSH可断开，需保持实例及GPU。

## 启动前核验

55个shot全部调度角色均有数据集描述；Agent模型确认qwen3.8-max；冻结源码哈希匹配。当前6项关键帧素材绑定/继承人物回归通过，两个新脚本语法检查通过。裁剪整理器以此前run771和run1404正式产物验证，弱匹配/未入分情形保留。未改变官方评测算法或历史分数。
