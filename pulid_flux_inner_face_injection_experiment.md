# PuLID-FLUX 3D Inner-Face Token 注入实验方案

> 状态说明：本文是实验设计基线，末尾的“当前仓库状态”保留了设计时的历史快照。实验现已实现并完成 0.6 主实验及 0.4 诊断组；最新结果和后续交接请以 `PULID_FLUX_EXPERIMENT_STATUS.md` 为准。

## 1. 实验目标

在保留官方 PuLID-FLUX ID Cross-Attention 的基础上，增加一个独立的 3D 人脸参考轨迹插件：

1. 在目标首帧去噪到第 30 步时估计干净图像并检测人脸姿态。
2. 从对应角色的 FaceLift 3D Gaussian 人脸资产中渲染相同角度的参考脸。
3. 将参考脸反演成与目标 FLUX 50-step schedule 对齐的 token/latent 轨迹。
4. 在第 30～49 步，将参考轨迹中 inner-face 区域的 token 注入目标生成轨迹。
5. 对比“官方 PuLID”与“官方 PuLID + 3D inner-face 插件”的最终效果。

这个插件不替换官方 PuLID，而是作为额外的局部身份与五官结构约束。

## 2. 固定实验参数

| 参数 | 固定值 |
|---|---:|
| FLUX 总去噪步数 | 50 |
| 插件开始注入 step | 30（0-based） |
| 插件实际注入区间 | step 30～49，共 20 步 |
| 插件注入强度 | 0.6 |
| 插件 mask 类型 | inner_face |
| 官方 PuLID | 开启 |
| 官方 PuLID `id_weight` | 1.0，实验组与对照组保持一致 |
| 官方 PuLID `start_step` | 0，实验组与对照组保持一致 |
| Guidance | 4.0，实验组与对照组保持一致 |
| True CFG | 1.0（fake CFG） |
| 随机种子 | 每一对实验使用相同 seed |

除了“是否开启自定义插件”以外，对照组和实验组的所有参数必须完全一致。

## 3. 总体流程

```text
角色原始参考图
  ├─> 官方 PuLID：InsightFace + EVA-CLIP + IDFormer
  │                  ↓
  │              ID Cross-Attention（全程保留）
  │
  └─> FaceLift：构建角色 3D Gaussian 人脸资产

目标 FLUX 去噪 step 0～29（不启用自定义插件）
                  ↓
         在 step 30 估计目标 pred_x0
                  ↓
     检测人物身份、bbox、yaw、pitch、roll
                  ↓
       从该人物 3D 资产渲染匹配角度参考脸
                  ↓
       空间对齐 + inner-face semantic mask
                  ↓
      FLUX 逆向 ODE 生成参考脸 50-step 轨迹
                  ↓
     step 30～49 执行 masked token residual injection
                  ↓
               解码最终首帧
```

## 4. 第 30 步目标人脸检测

### 4.1 检测时机

完成 step 0～29 的 30 次正常去噪更新后，在开始第 30 步插件注入之前检测目标人脸。

中间状态仍然是带噪 `x_t`，不能直接作为人脸检测图片。应先根据 FLUX Flow Matching 的 velocity 估计当前干净状态：

```python
pred_x0 = x_t - t * velocity
```

然后通过 FLUX AE 解码 `pred_x0`，得到 step-30 预览图。

### 4.2 检测内容

对预览图运行 InsightFace，取得：

- 人脸 bbox
- yaw
- pitch
- roll
- 人脸 embedding
- face confidence
- 对应角色 ID

如果 step 30 没有检测到可靠人脸：

1. 当前 step 不注入。
2. 后续 step 继续尝试检测。
3. 检测成功后再初始化 3D 参考轨迹。
4. 日志中必须记录延迟开始注入的原因和实际 start step。

人脸姿态和 mask 初始化成功后应固定到该 shot 结束，避免逐步重新检测导致参考视角和 mask 抖动。

## 5. 3D 人脸视角提取

### 5.1 角色资产

每个角色先通过 FaceLift 从正脸参考图生成：

- 3D Gaussian 人脸模型
- 多视角渲染图
- FaceLift 状态与相机参数

### 5.2 姿态匹配

根据 step-30 目标脸的 yaw、pitch、roll，从对应角色 3D 资产动态渲染相同角度的参考脸。

优先采用连续角度渲染，不只在 `front/left/right` 三张图片中选择。只有动态渲染不可用时才回退到最近的离散视角。

### 5.3 空间对齐

渲染参考脸需要：

1. 缩放到目标人脸 bbox 对应的尺寸。
2. 对齐到目标人脸中心位置。
3. 保持目标 yaw、pitch、roll。
4. 放置到与目标首帧相同大小的中性灰色画布。
5. 记录源人脸 bbox、目标 bbox、缩放比例、粘贴位置和相机参数。

必须先完成空间对齐，再生成参考 latent 和反演轨迹，否则参考 token 与目标 token 的空间位置无法对应。

## 6. Inner-Face Semantic Mask

本实验只保留最终效果最好的 `inner_face` 模式，不再实现普通 bbox 和 shrink mask 实验分支。

### 6.1 保留区域

- 面部皮肤
- 左右眉毛
- 左右眼睛
- 鼻子
- 嘴
- 上唇和下唇

### 6.2 排除区域

- 头发
- 耳朵和耳饰
- 脖子
- 衣服
- 帽子
- 背景

眼镜是否保留应作为配置项；默认排除，避免眼镜形状被强制复制到不佩戴眼镜的角色或镜头。

### 6.3 最终注入 mask

分别对以下两张图片进行语义分割：

1. step-30 目标 `pred_x0` 预览图。
2. 空间对齐后的 3D 参考脸。

将参考 mask 映射到目标布局后，使用二者的交集：

```python
final_inner_face_mask = target_inner_face_mask * aligned_reference_inner_face_mask
```

对边缘做少量 Gaussian feather，避免硬边界，但不能扩大到头发和背景区域。

最终将 mask 缩放到 FLUX packed token 网格。默认 `896×1152` 图像对应约 `56×72` 个空间 token；mask 应保持 `[1, token_count, 1]` 形状，以便广播到 token channel。

## 7. FLUX 参考脸反演轨迹

现有 SDXL ReNoise 代码不能直接用于 FLUX。FLUX 使用 Flow Matching，需要实现对应的逆向 ODE。

### 7.1 轨迹生成

1. 使用 FLUX AE 编码空间对齐后的 3D 参考脸，得到 `reference_x0`。
2. 使用和目标生成完全相同的 50-step timestep schedule。
3. 从 `t=0` 向 `t=1` 做逆向 ODE 积分。
4. 保存每个 timestep 对应的参考状态。
5. 将轨迹顺序转换为目标生成时的 `t=1 → t=0` 顺序。

```text
reference_trajectory[0]  <-> target generation step 0
reference_trajectory[1]  <-> target generation step 1
...
reference_trajectory[50] <-> final clean state
```

### 7.2 内存约束

只保存每一步的 packed latent/token，不保存 DiT 各层 hidden states，也不能保留 autograd graph。

所有轨迹生成使用：

```python
torch.inference_mode()
```

默认 `896×1152`、BF16、batch size 1 时，单个 packed latent 约 0.5MB，51 个状态约 25MB，轨迹本身不会成为显存瓶颈。

### 7.3 防止参考图泄露

- 参考脸放在中性画布上再反演。
- 参考轨迹进入缓存前，将 inner-face mask 外 token 清零或替换为中性轨迹。
- 不使用当前代码中硬编码的“中年亚洲男性戴眼镜”参考 prompt。
- 参考反演 prompt 应根据当前角色生成，或采用不包含具体身份偏置的中性人脸 prompt。
- 目标与参考必须使用同一 AE、dtype、分辨率和 timestep schedule。

## 8. Token 注入公式

注入发生在每次正常 Flow Euler 更新之后，并使用 `t_next` 对应的参考状态。

```python
velocity = flux_model_with_official_pulid(
    target_t,
    timestep=t,
    id=id_embedding,
    id_weight=1.0,
)

target_next_base = target_t + (t_next - t) * velocity

if step >= 30 and plugin_ready:
    target_next = target_next_base + 0.6 * inner_face_mask * (
        reference_next - target_next_base
    )
else:
    target_next = target_next_base
```

等价形式：

```python
alpha = 0.6 * inner_face_mask
target_next = target_next_base * (1 - alpha) + reference_next * alpha
```

这里的“token 相加”必须实现为差分残差/线性融合，不能使用下面这种裸加：

```python
# 禁止：连续 20 步可能导致 token norm 膨胀
target_next = target_next_base + 0.6 * inner_face_mask * reference_next
```

Mask 外的 token 必须保持完全不变。

## 9. 与官方 PuLID 的组合方式

官方 PuLID 保持原有逻辑：

```text
原始角色参考图
  -> InsightFace embedding
  -> EVA-CLIP visual features
  -> IDFormer
  -> PuLID Cross-Attention residual
  -> FLUX velocity prediction
```

自定义插件在 FLUX velocity 更新完成后工作。因此：

- 官方 PuLID 负责全局身份语义。
- 3D inner-face 插件负责匹配姿态下的局部五官结构。
- 两者可以同时启用。
- 两组实验中的官方 PuLID 参数必须完全一致。

如果叠加后出现脸部僵硬、过拟合或 prompt editability 明显下降，主实验仍保留插件强度 `0.6`，额外增加 `0.4` 作为诊断对照，不能直接修改主实验参数。

## 10. 对照实验

### 10.1 主对照

| 组别 | 官方 PuLID | 3D inner-face 插件 | 插件强度 |
|---|---|---|---:|
| Control | 开启 | 关闭 | 0.0 |
| Treatment | 开启 | 开启 | 0.6 |

这里的“不注入”仅表示关闭自定义插件，不表示关闭官方 PuLID。

### 10.2 可选诊断组

仅在主实验出现明显伪影时增加：

| 组别 | 官方 PuLID | 3D inner-face 插件 | 插件强度 |
|---|---|---|---:|
| Diagnostic | 开启 | 开启 | 0.4 |

### 10.3 公平性要求

每一对 Control/Treatment 必须固定：

- prompt
- negative prompt
- 原始人物参考图
- 3D 人脸资产
- seed
- timestep schedule
- 分辨率
- PuLID 版本
- PuLID `id_weight`
- guidance
- true CFG
- 模型精度

## 11. 输出与日志

每次实验至少保存：

```text
experiment_output/
├── config.json
├── input/
│   ├── identity_reference.png
│   └── prompt.txt
├── step_30/
│   ├── pred_x0.png
│   ├── face_detection.json
│   ├── target_inner_face_mask.png
│   ├── rendered_3d_face.png
│   ├── aligned_3d_face.png
│   ├── reference_inner_face_mask.png
│   └── final_inner_face_mask.png
├── trajectory/
│   └── metadata.json
├── control/
│   └── final.png
├── treatment/
│   └── final.png
└── metrics.json
```

逐 step 日志需要包含：

- step 和 timestep
- 是否执行插件注入
- 实际插件 start step
- mask token 数量和占比
- `target_next_base` norm
- `reference_next` norm
- 注入 residual norm
- `target_next` norm
- 是否出现 NaN/Inf
- 3D 渲染角度
- 参考轨迹索引

## 12. 评价指标

### 12.1 身份一致性

- 最终图与原始角色参考图的 InsightFace cosine similarity。
- 最终图与匹配角度 3D 渲染图的 InsightFace cosine similarity。
- Control 与 Treatment 的 similarity 增量。

### 12.2 自然度与泄露

- 面部边界是否有硬接缝。
- 肤色、光照是否与目标场景一致。
- 参考图背景、发型、服装是否泄露。
- 目标 yaw/pitch/roll 是否被保持。
- 五官是否出现重复、变形或过度锐化。

### 12.3 Prompt 遵循

- 人物动作和表情是否仍遵循 prompt。
- 场景构图是否被参考脸破坏。
- 官方 PuLID 的 editability 是否因插件明显下降。

## 13. 显存与运行建议

官方 PuLID-FLUX 给出的峰值显存参考：

| 模式 | 官方峰值显存 |
|---|---:|
| BF16，无 offload | 小于 45GB |
| BF16 + offload | 小于 30GB |
| BF16 + aggressive offload | 约 23GB，速度很慢 |
| FP8 + offload | 小于 17GB |
| FP8 + offload + ONNX CPU | 小于 15GB |
| FP8 + aggressive offload | 约 11GB，速度非常慢 |

官方说明：<https://github.com/ToTheBeginning/PuLID/blob/main/docs/pulid_for_flux.md>

推荐正式实验配置：

```text
GPU: 48GB 或更高
系统内存: 至少 64GB，建议 96GB
磁盘: 预留 60～80GB
精度: BF16
分辨率: 896×1152
batch size: 1
```

24GB 显卡可采用：

```text
--offload --fp8
```

并先用 `768×1024` 验证流程。16GB/12GB 虽然官方低显存模式可以运行，但余量小、速度慢，并且 FP8 对面部细节有一定损失，不建议作为正式对比环境。

50 步本身主要增加运行时间，不显著增加峰值显存。目标生成 50 次 FLUX forward，再加参考反演约 50 次 forward，相比同样 50 步的无插件对照，插件实验耗时预计接近两倍。

FaceLift 与 PuLID-FLUX 必须串行运行：FaceLift 完成人脸资产或当前角度渲染后释放不再需要的模型和显存，再加载/继续 FLUX。不能同时常驻两套大型模型，也不能为参考轨迹重新加载第二份 FLUX。

## 14. 所需权重

至少需要准备：

1. FLUX.1-dev 或 FLUX.1-Krea-dev 权重。
2. 对应 FLUX AE 权重。
3. PuLID-FLUX v0.9.1 权重。
4. T5 和 CLIP 文本编码器。
5. InsightFace AntelopeV2 权重。
6. FaceXLib RetinaFace 和 BiSeNet parsing 权重。
7. EVA02-CLIP-L-14-336 权重。
8. FaceLift multi-view diffusion 权重。
9. FaceLift GS-LRM 权重。

模型目录、版本和 SHA256 应写入每次实验的 `config.json`，避免后续无法复现。

## 15. 实现原则

1. 不直接修改 `third_party/PuLID` 官方快照，优先在 `multishot` 中增加包装后端和 hook。
2. 自定义插件必须可以通过一个配置开关完全关闭。
3. Control 与 Treatment 应使用同一个推理入口，避免两套代码路径造成额外差异。
4. 参考反演必须复用当前 FLUX 实例。
5. 轨迹只保存 latent/token，不保存中间网络激活。
6. 所有推理使用 `torch.inference_mode()`。
7. 在正式长实验前先执行单人物、单 shot、低分辨率 smoke test。
8. 对失败的人脸检测、3D 渲染、语义分割和反演步骤禁止静默 fallback；必须在日志中明确标记。

## 16. 后续实施清单

- [ ] 创建 Python 3.10/3.11 独立环境。
- [ ] 安装 PuLID-FLUX、FaceLift、InsightFace 和 parsing 依赖。
- [ ] 下载并校验所有模型权重。
- [ ] 先验证官方 PuLID-FLUX 50-step 推理。
- [ ] 验证 FaceLift 角色建模与动态角度渲染。
- [ ] 实现 FLUX 中间 `pred_x0` 估计与解码。
- [ ] 实现 inner-face semantic mask。
- [ ] 实现参考脸空间对齐。
- [ ] 实现 FLUX 逆向 ODE 参考轨迹。
- [ ] 在 FLUX Euler 更新后增加可开关的 masked residual injection。
- [ ] 增加逐 step 数值日志与中间图片。
- [ ] 运行低分辨率 Control/Treatment smoke test。
- [ ] 运行 BF16 正式对比实验。
- [ ] 计算 InsightFace 指标并人工检查自然度与泄露。

## 17. 当前状态

截至本文档创建时：

- 仓库包含官方 PuLID-FLUX 源码快照，但尚未接入 `multishot` 主流程。
- 自定义 3D inner-face token 注入尚未实现。
- 当前机器没有检测到可用 GPU。
- 当前 Python 和核心依赖尚未配置。
- FLUX、PuLID-FLUX、InsightFace 和 FaceLift 权重尚未下载。
