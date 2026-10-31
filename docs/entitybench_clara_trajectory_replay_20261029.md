# Clara两首帧：原配置重放与去噪素材导出

任务：run354 / i90_j105，Clara的4:1墙边特写、5:1暗房俯拍。原绘图包`outputs/model_figure_handoff_clara_20261022/`的两张首帧SHA已与原20集输出核验一致。

[素材输出目录](../outputs/model_figure_clara_denoising_replay_20261029/index.html)。**已于2026-10-30 02:23北京时间完成**，两帧Control及Treatment最终图都与原图SHA一致、像素MAE=0。各82张主轨迹图，另有注入前后图和原始CPU张量。后台队列已退出，不重复启动。与第二批新方法run354分开保存。

## 重放设置

- 完全沿用原实际Qwen提示词（包含原角色条件前缀）和原官方镜头描述，分别保存在`inputs/shot_4_1/`及`inputs/shot_5_1/`的两个prompt文本中。
- 原seed为20009006、20009007；Qwen-Image-2.1原revision、1024×576、20步、step12、原0.4基础注入强度、自适应1/6步、KV cache、sequential CPU offload。
- 沿用原Clara参考图、原已通过门控的3D姿态对齐图与mask，重新生成Control/共享前缀并回放原3D引导；不重新建立人物身份，也不调用新Agent。两张原例都为1步实际注入，第二张按脸高缩放后的强度为0.325。
- 两个首帧单独输出，不覆盖原绘图包、原始benchmark、当前第二批新方法run354。
- CPU记录回归测试通过：输出不变、注入前后状态区分、KV预填充不误入主轨迹。GPU重放已完成，两帧Control及Treatment均已验证逐像素复现旧图。

## 已导出文件

每个`shot_4_1/`、`shot_5_1/`目录包含：

| 文件 | 用途 |
|---|---|
| `treatment.png` | 本次重放最终首帧 |
| `control/z_00.png`至`z_20.png` | 原始Control每个真实latent状态的VAE解码 |
| `treatment/z_00.png`至`z_20.png` | 实际有注入轨迹每步真实latent的VAE解码，step12前共享原前缀 |
| `control/pred_x0_00.png`至`pred_x0_19.png` | Control每步预测的干净图 |
| `treatment/pred_x0_00.png`至`pred_x0_19.png` | 注入路线每步预测的干净图 |
| `injection/step_12_euler_before.png` | 该步Euler输出、尚未加残差 |
| `injection/step_12_actual_after.png` | 加完残差后实际送入下一步的latent解码 |
| `trajectory_states.pt` | CPU原始latent、速度、pred_x0和sigma张量，供重新排图/解码 |
| `treatment_trajectory.jpg` | z0/1/4/5/8/9/12/13/14/19/20缩略总览 |
| `trajectory_metadata.json` | 所有图片路径/SHA、索引定义、原版与重放图的像素差异 |

`z_0`是初始噪声，`z_n`表示完成n次更新后的真实状态；`pred_x0_i = z_i - sigma_i * velocity_i`是第i步的干净预测，两者必须在图上区分。代码step12为从z12开始的0基索引，注入发生在Euler输出与z13之间。

原运行当时没有保存完整20步latent；本任务保存的是本次同配置重放的真实状态。已核验新旧最终图逐像素一致；这些中间状态仍来源于本次同配置重放，不是原运行当时保存的文件。
