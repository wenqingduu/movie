# 两种图像去噪路线

> 说明：两条路线本质上都属于扩散生成。这里按去噪主干分为传统 **U-Net Diffusion** 和 **DiT（Diffusion Transformer）**。

## 路线 A：U-Net Diffusion（SDXL + IP-Adapter）

```mermaid
flowchart LR
    IN[随机噪声 zT<br/>文本条件：CLIP<br/>身份条件：IP-Adapter] --> PRE[SDXL U-Net 去噪<br/>step 0-29]
    PRE --> S30[step 30<br/>得到 pred_x0]

    S30 --> DET[InsightFace<br/>检测姿态、位置、脸高 h]
    DET --> RENDER[FaceLift 3D 渲染<br/>Gaussian + 姿态标定]
    RENDER --> REF[构建 3D 参考脸<br/>光照匹配 + BiSeNet 人脸 Mask]
    REF --> VAE[VAE 编码并加同时间噪声<br/>得到 z_ref,next]

    S30 --> STEP[后段 U-Net 单步去噪<br/>得到 z_base,next]
    VAE --> MIX
    MASK["α = λ(h) × 人脸 Mask M<br/>小脸自动降低权重、缩短窗口"] --> MIX
    STEP --> MIX["人脸区域加权融合<br/>z_next = (1 - α) × z_base,next<br/>+ α × z_ref,next"]

    MIX --> LOOP{是否还在<br/>注入窗口内？}
    LOOP -- 是 --> STEP
    LOOP -- 否 --> DEC[VAE 解码]
    DEC --> OUT[一致性首帧]

    classDef main fill:#eaf1ff,stroke:#4b78d1,color:#142840,stroke-width:1.5px;
    classDef face fill:#e9f8f5,stroke:#079b92,color:#143b43,stroke-width:1.5px;
    classDef mix fill:#123d49,stroke:#123d49,color:#ffffff,stroke-width:2px;
    class IN,PRE,S30,STEP,DEC,OUT main;
    class DET,RENDER,REF,VAE,MASK face;
    class MIX mix;
```

计算上，人脸注入就是一次带 Mask 的加权融合：

```text
alpha  = lambda(h) * face_mask
z_next = z_base_next + alpha * (z_ref_next - z_base_next)
       = (1 - alpha) * z_base_next + alpha * z_ref_next
```

- Mask 外 `alpha = 0`：完全保留原生成结果。
- 人脸内部 `0 < alpha <= 0.4`：加入相应比例的 3D 参考脸 latent。
- 原始身份照片始终通过 IP-Adapter 提供全局身份条件。

## 路线 B：DiT（FLUX.1-dev + PuLID-FLUX）

```mermaid
flowchart LR
    IN[随机噪声 zT<br/>文本条件：T5 / CLIP<br/>身份条件：PuLID] --> PRE[FLUX Transformer 去噪<br/>step 0-29]
    PRE --> S30[step 30<br/>得到 pred_x0]

    S30 --> DET[InsightFace<br/>检测姿态、位置、脸高 h]
    DET --> RENDER[FaceLift 3D 渲染<br/>Gaussian + 姿态标定]
    RENDER --> REF[构建 3D 参考脸<br/>光照匹配 + BiSeNet 人脸 Mask]
    REF --> AE[AE 编码 3D 参考脸<br/>构建同时间参考轨迹]
    AE --> PACK[2×2 latent packing<br/>Mask 展开到四个子位置]

    S30 --> STEP[后段 FLUX Transformer 单步去噪<br/>得到 z_base,next]
    PACK --> MIX
    MASK["α4ch = λ(h) × packed Mask<br/>四个子位置分别计算权重"] --> MIX
    STEP --> MIX["人脸区域加权融合<br/>z_next = (1 - α4ch) × z_base,next<br/>+ α4ch × z_ref,next"]

    MIX --> LOOP{是否还在<br/>注入窗口内？}
    LOOP -- 是 --> STEP
    LOOP -- 否 --> DEC[AE 解码并还原空间布局]
    DEC --> OUT[一致性首帧]

    classDef main fill:#f0ebff,stroke:#7659e8,color:#282044,stroke-width:1.5px;
    classDef face fill:#e9f8f5,stroke:#079b92,color:#143b43,stroke-width:1.5px;
    classDef mix fill:#123d49,stroke:#123d49,color:#ffffff,stroke-width:2px;
    class IN,PRE,S30,STEP,DEC,OUT main;
    class DET,RENDER,REF,AE,PACK,MASK face;
    class MIX mix;
```

DiT 路线使用同样的加权融合公式，区别是 FLUX 会把 `2×2` 个 VAE latent 位置打包成一个 token。因此 Mask 也拆成四个子位置权重：

```text
alpha_4ch = lambda(h) * packed_face_mask
z_next    = (1 - alpha_4ch) * z_base_next
          + alpha_4ch * z_ref_next
```

这样不是给整个 packed token 使用同一个权重，而是分别控制其中四个空间子位置，小脸边界会更精细。
