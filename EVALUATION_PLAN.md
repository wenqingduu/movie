# 人脸一致性注入插件评测方案

更新时间：2026-09-27。

## 1. 当前评测目标

当前只评测“3D 人脸一致性注入插件”，不把剧本拆分、镜头规划或跨镜头记忆包装成已经完成的
智能体能力。

每个镜头采用严格成对设计：

```text
Control：首帧生成模型正常生成 → Wan2.2
Treatment：相同 seed/噪声/条件 + v7 局部 3D residual → Wan2.2
```

除是否执行 v7 外，提示词、身份资产、首帧模型、随机种子、分辨率和 Wan 参数保持一致。

## 2. 当前唯一有效的插件版本

当前只保留 v7 color-safe 路径：

1. 在注入起点的 `pred_x0` 检测人脸框和 pitch/yaw/roll；
2. 从 episode-local FaceLift Gaussian 连续渲染目标姿态；
3. 用 InsightFace 对渲染结果做姿态回检；
4. 用目标脸与 3D 脸的皮肤交集估计低频 log-RGB 光影；
5. 通过 BiSeNet 构造 v7 color-safe 身份核心，排除发际线、太阳穴、外脸颊、下巴边缘和耳朵；
6. 将调色后的纯 3D reference 编码为 latent，在同 timestep 做局部 trajectory residual；
7. 根据脸高自适应缩短注入窗口并降低小脸强度，正常脸基准强度为 `0.4`。

## 3. 预先固定的安全 gate

gate 必须在查看 Treatment 指标之前确定，不能事后删除负例。

| Gate | 处理 |
|---|---|
| 无可靠人脸或脸高 `<24 px` | Treatment 复用 Control |
| 连续 Gaussian 渲染失败 | Treatment 复用 Control |
| 渲染后姿态回检超阈值 | Treatment 复用 Control |
| v7 color-safe core 为空 | Treatment 复用 Control |
| 提示词明确要求闭眼 | Treatment 复用 Control，记录 `prompt_eye_closure_conflict` |
| 当前 PuLID-FLUX/IP-Adapter 多角色镜头 | Treatment 复用 Control |

“复用 Control”表示 Control 正常生成，Treatment 复制同一结果并在 manifest 中引用同一视频；该镜头
仍属于完整 episode，所有成对增益记为 0。

## 4. 当前数据范围

固定三个完整有序 EntityBench easy episode，共 34 个镜头、23 个单角色镜头：

| Episode | 总镜头 | 单角色镜头 |
|---|---:|---:|
| `run719` | 12 | 8 |
| `run893` | 10 | 5 |
| `run1517` | 12 | 10 |

角色参考图、FaceLift Gaussian 和姿态标定均为 episode-local 固定资产。所有镜头保持 EntityBench
原始顺序、`video_prompts`、`action_descriptions`、`cut` 与 `entity_schedule`。

## 5. 当前模型矩阵

| 首帧路线 | 当前单角色策略 | 当前多角色策略 | 下游视频 |
|---|---|---|---|
| PuLID-FLUX | Control vs v7 | 复用 Control | Wan2.2-TI2V-5B |
| SDXL/IP-Adapter | Control vs v7 | 复用 Control | Wan2.2-TI2V-5B |
| Qwen-Image-2.1 | Control vs v7 | 原生多参考 Control vs 同步多人 v7 | Wan2.2-TI2V-5B |

Wan2.2 对每个镜头独立首帧驱动，不读取上一镜头历史。当前两条路线不具备可靠的多参考身份绑定，
因此不在它们上面继续做多人注入。多人验证迁移到 Qwen-Image-2.1 原生多参考 Control，并继续使用
逐角色 v7 mask 和同步 residual。

run719 / shot 1:1 的首个多人 pilot 已完成。2026-09-27 在其基础上启动 Qwen 三 episode 全量实验：
34 个 shot（23 单人、11 多人）均生成 Control/Treatment 首帧，28 个实际注入，6 个按预先固定 gate
复用 Control。Wan 仍采用完全相同参数的成对设计。Qwen 结果单独成表，不与 PuLID/IP-Adapter
单角色主表混合。

Qwen 全量实验的额外规则：

- 输入每个 scheduled character 的 episode-local 参考图；
- Control 中可靠脸少于 scheduled character 时整镜头复用 Control；
- 可靠脸多于计划人数时选择面积最大的主脸，并记录额外脸数量；
- 角色映射采用 InsightFace 全局一对一最大 cosine，保存矩阵、选择分数和 margin；
- 视频角色跟踪只使用初始空间位置，不用身份 embedding 挑选待测脸，避免指标泄漏。

## 6. 指标

### 6.1 原始身份锚定 InsightFace

对原始角色参考图与生成首帧/视频帧的人脸计算 cosine：

- 首帧 Control、Treatment 及成对差值；
- 视频 `first`、`mean`、`median`、`p10`、`minimum`、`last`；
- 首尾漂移、全视频回归斜率和人脸检测覆盖率。

### 6.2 原始参考图锚定 DINOv2

对扩展人脸裁剪计算 DINOv2 cosine，报告首帧和视频逐帧聚合。它衡量更宽泛的脸部视觉特征，
不能由 InsightFace 替代；两者不同向时必须同时报告。

### 6.3 EntityBench 本地指标（待按当前 gate 重算）

下一轮可运行不需要大模型 API 的部分：

- VBench：subject consistency、temporal flickering、motion smoothness、dynamic degree、
  aesthetic quality、imaging quality；
- GroundingDINO/CLIP presence；
- DINOv2 `cs_face`、`cs_object`、`cs_transition_boundary`。

旧非 API 汇总在闭眼 gate 加入前生成，已从当前结果区移除。重算完成前不得把旧值当作当前结果。
VLM 角色属性、动作判断和 LLM judge 尚未运行，缺失值不能当作 0，也不能声称完整官方排名。

## 7. 自动汇总脚本

当前所有 episode 的总表和逐镜头明细由一个脚本生成：

```bash
cd /root/autodl-tmp/movie
.venv/bin/python pretest/summarize_current_episode_results.py
```

输出：

- `EPISODE_TEST_RESULTS.md`：人类可读的 episode 汇总和每镜头明细；
- `outputs/entitybench_current_summary.json`：机器可读完整结果；
- 脚本同时读取各 episode 的首帧结果、视频 InsightFace 报告和参考图 DINOv2 报告，不手工
  复制指标；它输出跨 episode 总表、每个 episode 汇总和逐 shot 明细。

在任何首帧、视频或评测报告更新后重新执行该脚本即可。

Qwen 三 episode 独立入口：

```bash
pretest/run_qwen_image21_entitybench_full3.sh
.venv/bin/python pretest/evaluate_qwen_image21_entitybench_batch.py \
  --output-root outputs/entitybench_qwen_image21_v7_full3
```

输出 `outputs/entitybench_qwen_image21_v7_full3/evaluation_report.json`，包含逐 episode、逐 shot、
逐角色首帧/视频 InsightFace 和检测覆盖率。批处理可恢复；已有 Control、Treatment 和视频不会重算。

## 8. 当前权威结果根目录

```text
PuLID-FLUX / run719
outputs/entitybench_wan22_colornested_v7_s04_12/episode_00053051/

IP-Adapter / run719
outputs/entitybench_wan22_ip_adapter_v7_s04_12/episode_00053051/

PuLID-FLUX / run893、run1517
outputs/entitybench_wan22_pulid_v7_pilot3/

IP-Adapter / run893、run1517
outputs/entitybench_wan22_ip_adapter_v7_pilot3/

参考图 DINOv2
outputs/entitybench_reference_dino_pilot3/reference_dino_summary.json

Qwen-Image-2.1 三 episode（正在完成视频/评测）
outputs/entitybench_qwen_image21_v7_full3/

```

具体数值以自动生成的 `EPISODE_TEST_RESULTS.md` 为准。

## 9. 结果解释规则

1. 主结果同时报告 InsightFace 与 DINOv2，不能只选择有利指标；
2. 分别报告实际注入镜头和完整 episode（安全复用镜头按差值 0）；
3. 保留所有负例和视觉失败，不按 Treatment 最终分数筛样本；
4. 多人镜头不计入当前两条单角色路线的插件增益；
5. 当前只有三个 episode、一个主要视频 seed，结论限定为 pilot；
6. 下一阶段扩展 episode/seed 前先冻结代码提交、模型 revision 和输出 manifest。

## 10. 下一阶段

1. 完成 Qwen 三 episode 的视频与逐角色报告，分别分析单人、多人及安全回退覆盖率；
2. 检查所有负增益角色，尤其是自动映射 margin 偏低的 run893 / 6:1 Leo；
3. 再决定是否验证逐角色自适应强度和停止步，不能按 Treatment 结果事后删例；
4. 增加风格/OOD gate 与局部关键点对齐可靠性；
5. 扩大 episode 和随机种子，加入 bootstrap 置信区间；
6. 在需要完整 EntityBench 时再配置 VLM/LLM API；智能体端到端评测另行设计。
