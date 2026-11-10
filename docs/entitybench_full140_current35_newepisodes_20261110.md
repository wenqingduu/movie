# 当前35条版叠加新整集：2026-11-10 benchmark更新

统计范围冻结于2026-11-10 11:58（北京时间）：7个新增完整episode、77个shot覆盖当前35条版中的对应整集，其他133集、2414个shot保持原输入。完整范围仍140集、2491个shot。未按分数挑选整集；不替换未完成episode的前缀。历史媒体未覆盖。

整集优先于旧独立动态候选：本次覆盖run740/i5_j53/3:1、run744/i20_j54/4:1、run793/i55_j64/5:1、run851/i20_j44/1:1这4个旧35条中的独立替换，保留另外31个。因此“叠加”不表示35条旧候选全部留用。

| 指标 ↑ | 当前35条替换版 | 叠加7个新整集后 | 变化 |
|---|---:|---:|---:|
| 人物DINO（可见单脸＋定位复核） | 0.7990 | 0.7976 | -0.0014 |
| 主体一致性 | 0.9414 | 0.9431 | +0.0017 |
| 时间稳定性 | 0.9863 | 0.9866 | +0.0003 |
| 运动平滑度 | 0.9920 | 0.9921 | +0.0001 |
| 动态程度 | 0.4873 | 0.4731 | -0.0142 |
| 美学质量 | 0.5859 | 0.5863 | +0.0004 |
| 成像质量（0–100） | 60.8634 | 60.9004 | +0.0370 |
| 角色出现率 | 0.9479 | 0.9501 | +0.0022 |
| 物体出现率 | 0.8981 | 0.8986 | +0.0005 |
| 场景出现率 | 0.6249 | 0.6206 | -0.0043 |
| 人物区域cs_face | 0.7238 | 0.7247 | +0.0009 |
| 物体区域cs_object | 0.8077 | 0.8071 | -0.0006 |
| 接续边界一致性 | 0.8441 | 0.8243 | -0.0198 |

角色出现、主体一致性有所提高；动态、场景出现和边界一致性下降。该比较是最新整集替换版本对当前35条版，不是单因素消融，不把全部变化归因于路由或3D。主视频沿用4步集成方案，不恢复40步动态队列。

默认DINO按evaluate-face-injection-v2 skill：原始人物区域DINO、可见单脸筛选、同帧明确错角色定位排除；与raw cs_face另列。43个新增计分区域全部做助手视觉复核，41保留、2确认错人排除（run893/4:3 Marcus误定位Adrian；run740/i5_j53/4:1 Carol误定位Brenda）。真实身份漂移、重复人物、无法确认的映射均保留；没有按相似度筛除。有效episode数130→130，计分角色次1465→1486，复核保留区域1781→1800。非外部人工签核。

官方12项复用已验证同配置、同模型与同范围的完整episode评分，由官方聚合函数重算140集；评分失败0。已复现替换前35条版的官方及默认DINO结果。strict/skip_vlm、VLM0，raw cs_face/cs_object无fidelity gate，partial_without_vlm；完整140集不等于论文51项全评。

输出：

- `outputs/entitybench_full140_current35_newepisodes_20261110/complete_report.md`：13项最终表。
- `complete_comparison.json`、`official_aggregates.json`、`visible_region/report.json`：机器可读结果。
- `replacement_manifest.json`、`candidate_media/`：当前完整2491视频版本映射及软链接；原始文件保留。
- `final_integrity_audit.json`：逐一核验2491媒体SHA与软链接、178项冻结输入绑定。
- `visible_region/review/decisions.json`：全部43项定位复核。
- 复现入口 `pretest/qwen140_full140_current35_newepisodes_20261110.py`，已有snapshot需保留，不能修改绑定源码或重prepare改变范围。

## 后台状态（2026-11-10 12:09 北京时间）

78集/1218shot队列共生成119shot；8集/89shot已完整生成并完成官方12项。run909在上述7集范围冻结后完成，未纳入本表，留待下次版本。当前run910/i20_j43在人物3D素材校准阶段。生成supervisor已恢复且运行，无守护Agent，未恢复旧动态队列。

另外7个episode因`Expected one lookup_characters call; stop for user review`中断，错误已记录，队列继续其他集；这7集未作为新版整集替换：run714/i20_j34、run728/i50_j84、run728/i80_j100、run755/i15_j49、run799/i5_j15、run893/i95_j118、run900/i0_j32。此次只记录，不自动改Agent提示词或进行语义重试。

队列入口与范围仍为`pretest/qwen140_never_rerun_resume_20261109.py run`、`outputs/entitybench_never_rerun_episodes_v2_20261109/plan.json`。下一次状态必须实查，不按此快照盲目重启。磁盘约102GiB可用，不设reserve、不删除历史产物；本轮无Git推送。
