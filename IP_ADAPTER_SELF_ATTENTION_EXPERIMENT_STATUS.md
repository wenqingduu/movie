# IP-Adapter + 3D 人脸 Self-Attention 注入实验状态

> 给后续接手本仓库的大模型：本实验用于回答“现有 IP-Adapter 策略是否通过 self-attention 注入 3D 采样人脸，以及为什么此前看起来只注入了皮肤纹理”。实验已经真实跑完，不要把配置存在误认为注入成功，也不要重新下载模型。

## 结论

截至 2026-08-16，项目中的人脸路径应拆成两个并行机制理解：

1. IP-Adapter 是全局图像条件。它没有目标脸空间 mask，主要通过 UNet cross-attention 的图像 embedding 影响整张图。
2. 3D 人脸的局部注入由项目自定义的 masked mutual self-attention 完成：reference 和 target 的 `x_t` 合并进同一次 UNet，目标脸 query 在指定 self-attention 层关注 3D 参考脸的 K/V，再仅于目标脸 mask 内融合。
3. `MULTISHOT_DYNAMIC_IP_ADAPTER_REFERENCE=1` 只是把连续 3D render 的 embedding 混进 IP-Adapter 条件，并不等同于 self-attention 注入。

本轮对照验证了上述差异：动态 IP-Adapter 单独使用时只造成较弱且偏全局的外观变化，身份指标反而下降；加入 masked mutual self-attention 后，变化更集中在脸区，眉眼、鼻翼、嘴部和脸部明暗结构更明显地向 3D 参考移动，但身份提升仍然有限，尚不能判定为合格的身份保持方案。

## 对照设置

三组使用相同 SDXL 初始噪声、prompt、seed 和 step-30 latent。step 0～29 只运行一次，step 30～49 从完全相同的状态分叉：

| 分支 | 原始照片 IP-Adapter | 动态 3D IP-Adapter | 3D masked mutual self-attention |
|---|---|---|---|
| `ip_adapter_baseline` | 是 | 否 | 否 |
| `dynamic_ip_adapter_only` | 是 | 是 | 否 |
| `ip_adapter_plus_self_attention` | 是 | 是 | 是 |

- 模型：SDXL Base 1.0 + `ip-adapter_sdxl.safetensors`
- Prompt：`cinematic medium close-up portrait of a man standing beneath warm neon lights on a rainy night street, three-quarter view, natural skin texture, shallow depth of field, photorealistic, subtle rim light`
- Seed：`42`
- 分辨率：`1024×1024`
- 总步数：`50`
- 分叉与注入区间：step `30～49`
- 注入 lambda：`0.6`
- self-attention scale：`0.85`，有效强度 `0.51`
- 连续 3D render 姿态：pitch `-0.4158°`、yaw `-2.1247°`、roll `0.0640°`
- 目标脸 bbox：`[290.59, 108.78, 759.70, 762.26]`
- 参考布局：`match_target_scale`，目标脸 `469.11×653.48`，参考脸 `449×654`

本轮继续使用 `experiment_assets/pulid_reference.jpg`，它是来源 URL 未记录的外部真实照片，不是项目生成的人脸。连续 3D render 来自此前由该照片构建的 FaceLift Gaussian。

## 结果

| 分支 | 对原始照片 cosine | 对连续 3D render cosine | 全图 RGB MAE vs baseline | 目标脸 bbox RGB MAE vs baseline |
|---|---:|---:|---:|---:|
| IP-Adapter baseline | 0.316020 | 0.314810 | 0.000000 | 0.000000 |
| Dynamic IP-Adapter only | 0.257251 | 0.253881 | 3.619008 | 3.240374 |
| IP-Adapter + self-attention | 0.322964 | 0.335000 | 5.707939 | 6.317089 |

相对 baseline：

- 动态 IP-Adapter only 对原始照片和 3D render 的 cosine 分别下降 `0.058769` 和 `0.060929`；脸区 MAE 还低于全图 MAE，符合“弱、偏全局的纹理/风格迁移”。
- 加 self-attention 后，对原始照片和 3D render 的 cosine 分别上升 `0.006944` 和 `0.020190`；脸区 MAE 高于全图 MAE，说明空间 mask 确实让变化集中到目标脸。
- 人工检查未发现 PuLID masked residual 方案中的硬边贴脸伪影，但身份增益幅度很小。当前结果只证明 self-attention 路径真实生效，不能证明身份一致性已经解决。

三个最终 PNG 的 SHA256：

- baseline：`d2de3985100c7b25e57f4088609b61dcd40414787301af4b408b2e92c96db720`
- dynamic IP-Adapter only：`508dbfce7b10592c19ec3257fffe9fdf3131ed8b03a66b8380db51207b513794`
- IP-Adapter + self-attention：`36dd23aa76c9291af86a7b6225566afd058f47869cdbf4208a450c6204c2ad0c`

## 注入执行证据

`logs/ip_adapter_plus_self_attention.json` 记录：

- dynamic IP-Adapter 在 20/20 个步骤应用。
- masked mutual self-attention 在 20/20 个步骤应用。
- 每一步有 16 个启用层的统计，配置层范围为 `[54, 69]`。
- 代表性 attention 层中 target/reference 脸区 token 数为 `534/516`，尺度接近。

同时修复了一个仅影响日志归属的错误：两种机制一起开启时，旧代码会把 self-attention 层统计挂到同一步的 dynamic IP-Adapter 日志项；现在统计会写入真正的 `masked_mutual_self_attention` 项。该修复不改变生成数学或结果图。

## 代码和结果入口

- 可执行脚本：`multishot/ip_adapter_self_attention_experiment.py`
- 对照拼图：`experiment_output/ip_adapter_self_attention_comparison/comparison.jpg`
- 结构化结果：`experiment_output/ip_adapter_self_attention_comparison/result.json`
- 三张最终图：`experiment_output/ip_adapter_self_attention_comparison/branches/`
- 连续 3D render、尺度匹配布局和 mask：`experiment_output/ip_adapter_self_attention_comparison/input/`
- 共享 step-30 x0 与目标 mask：`experiment_output/ip_adapter_self_attention_comparison/shared/`
- 每分支注入日志：`experiment_output/ip_adapter_self_attention_comparison/logs/`

模型权重和虚拟环境不提交 GitHub。保存下来的 `input/rendered_3d_face.png` 是本轮实际使用的连续 Gaussian 渲染结果；复跑时优先复用它，不会静默退回 front/left/right 离散视角。

## 本地复现

```bash
cd /root/autodl-tmp/movie
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD"

.venv/bin/python -m multishot.ip_adapter_self_attention_experiment
```

## 建议下一步

1. 保留本轮三分支作为固定基线，调强度时不要只比较最终视觉图。
2. 分离实验“原始照片 IP-Adapter + self-attention”和“动态 3D IP-Adapter + self-attention”，量化动态 embedding 是否在联合路径中提供正贡献。
3. 尝试 part-guided mask（eyes/nose/mouth）和不同 self-attention 层范围，而不是继续提高全脸固定强度。
4. 改用项目自身生成且来源明确的角色参考图，再判断身份指标上限。
