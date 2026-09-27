# EntityBench 当前三 episode 评测结果

> 本文件由 `pretest/summarize_current_episode_results.py` 自动生成。不要手工修改数值。

## 范围

共 3 个完整有序 episode、34 个镜头；当前两条路线只对单角色镜头执行 v7。

| 路线 | Episode | 总镜头 | 单角色 | 实际注入 | 安全复用 Control |
|---|---|---:|---:|---:|---:|
| PuLID-FLUX | run719 | 12 | 8 | 4 | 4 |
| IP-Adapter | run719 | 12 | 8 | 5 | 3 |
| PuLID-FLUX | run893 | 10 | 5 | 4 | 1 |
| IP-Adapter | run893 | 10 | 5 | 3 | 2 |
| PuLID-FLUX | run1517 | 12 | 10 | 8 | 2 |
| IP-Adapter | run1517 | 12 | 10 | 9 | 1 |

## 跨 episode 总结果（实际注入镜头）

| 路线 | 注入/单角色 | 首帧 InsightFace Δ | 视频 InsightFace mean Δ | 首帧 DINOv2 Δ | 视频 DINOv2 mean Δ |
|---|---:|---:|---:|---:|---:|
| IP-Adapter | 17/23 | +0.1207 | +0.0940 | +0.0242 | +0.0100 |
| PuLID-FLUX | 16/23 | +0.0330 | +0.0320 | +0.0079 | +0.0072 |

安全复用镜头按 Δ=0 纳入完整单角色口径后：

| 路线 | 单角色对数 | 首帧 InsightFace Δ | 视频 InsightFace mean Δ | 首帧 DINOv2 Δ | 视频 DINOv2 mean Δ |
|---|---:|---:|---:|---:|---:|
| IP-Adapter | 23 | +0.0892 | +0.0695 | +0.0179 | +0.0074 |
| PuLID-FLUX | 23 | +0.0230 | +0.0222 | +0.0055 | +0.0050 |

## Episode 汇总

| 路线 | Episode | 首帧 InsightFace Δ | 视频 InsightFace mean Δ | 参考图 DINOv2 首帧 Δ | 参考图 DINOv2 视频 mean Δ |
|---|---|---:|---:|---:|---:|
| PuLID-FLUX | run719 | +0.0635 | +0.0561 | +0.0093 | -0.0044 |
| IP-Adapter | run719 | +0.1644 | +0.1145 | +0.0324 | +0.0171 |
| PuLID-FLUX | run893 | +0.0311 | +0.0400 | +0.0074 | +0.0076 |
| IP-Adapter | run893 | +0.0921 | +0.0556 | -0.0177 | -0.0530 |
| PuLID-FLUX | run1517 | +0.0188 | +0.0159 | +0.0075 | +0.0127 |
| IP-Adapter | run1517 | +0.1060 | +0.0955 | +0.0337 | +0.0271 |

## Qwen-Image-2.1 多角色 pilot

run719 / shot 1:1 使用 Viktor、Julian 两张身份参考图生成双人 Control，随后在两张独立 v7 mask 内同步注入。下游仍为 Wan2.2-TI2V-5B；Qwen 是首帧图像模型，不是视频模型。

| 角色 | 首帧 Control | 首帧 Treatment | 首帧 Δ | 视频 Control mean | 视频 Treatment mean | 视频 Δ | 检测覆盖率 C/T |
|---|---:|---:|---:|---:|---:|---:|---:|
| Julian | 0.6747 | 0.7355 | +0.0609 | 0.5268 | 0.6011 | +0.0742 | 1.00/1.00 |
| Viktor | 0.4456 | 0.4544 | +0.0088 | 0.4109 | 0.3967 | -0.0143 | 1.00/1.00 |

首帧两角色宏平均 Δ=+0.0348；视频 mean 宏平均 Δ=+0.0300。该单镜头 pilot 验证了多人链路可运行，但视频增益尚未做到逐角色一致。

首帧对比：`/root/autodl-tmp/movie/outputs/qwen_image21_multiface_v7/run719_shot_1_1/paired_v7_s04/comparison.jpg`
视频并排对比：`/root/autodl-tmp/movie/outputs/qwen_image21_multiface_v7/run719_shot_1_1/videos/comparison_side_by_side.mp4`

## PuLID-FLUX / run719 逐镜头

| Shot | 角色 | 插件处理 | 首帧 C→T (Δ) | 视频身份 mean Δ | DINO 视频 mean Δ | 跳过原因 |
|---|---|---|---:|---:|---:|---|
| 2:1 | Viktor | 注入 | 0.6346→0.6892 (+0.0546) | +0.0356 | +0.0066 | — |
| 4:1 | Viktor | 复用 Control | 0.1003→0.1003 (+0.0000) | +0.0000 | +0.0000 | detected face height 13px is below the 24px injection threshold |
| 4:4 | Viktor | 注入 | 0.5816→0.6606 (+0.0790) | +0.1058 | +0.0106 | — |
| 4:5 | Viktor | 注入 | 0.4464→0.5291 (+0.0827) | +0.0483 | -0.0158 | — |
| 4:6 | Roman | 复用 Control | — | — | — | no reliable face detected in 3 attempts through denoising step 32 |
| 5:1 | Viktor | 复用 Control | 0.0279→0.0279 (+0.0000) | +0.0000 | +0.0000 | no reliable face detected in 3 attempts through denoising step 32 |
| 5:2 | Viktor | 复用 Control | -0.0194→-0.0194 (+0.0000) | +0.0000 | +0.0000 | no reliable face detected in 3 attempts through denoising step 32 |
| 6:1 | Viktor | 注入 | 0.6575→0.6953 (+0.0379) | +0.0346 | -0.0190 | — |

## IP-Adapter / run719 逐镜头

| Shot | 角色 | 插件处理 | 首帧 C→T (Δ) | 视频身份 mean Δ | DINO 视频 mean Δ | 跳过原因 |
|---|---|---|---:|---:|---:|---|
| 1:1 | Viktor, Julian | 复用 Control | — | — | — | multi-character shot: a single global IP-Adapter portrait cannot represent all identities; no validated multi-target IP-Adapter path |
| 2:1 | Viktor | 注入 | 0.3158→0.4327 (+0.1170) | +0.1546 | +0.0757 | — |
| 3:1 | Silas, Viktor | 复用 Control | — | — | — | multi-character shot: a single global IP-Adapter portrait cannot represent all identities; no validated multi-target IP-Adapter path |
| 4:1 | Viktor | 复用 Control | — | +0.0000 | +0.0000 | no reliable face detected in 4 attempts through denoising step 33 |
| 4:2 | Viktor, Roman | 复用 Control | — | — | — | multi-character shot: a single global IP-Adapter portrait cannot represent all identities; no validated multi-target IP-Adapter path |
| 4:3 | Silas, Leo | 复用 Control | — | — | — | multi-character shot: a single global IP-Adapter portrait cannot represent all identities; no validated multi-target IP-Adapter path |
| 4:4 | Viktor | 注入 | 0.1712→0.3401 (+0.1689) | +0.0672 | -0.0057 | — |
| 4:5 | Viktor | 注入 | 0.1379→0.2616 (+0.1237) | +0.0050 | -0.0043 | — |
| 4:6 | Roman | 复用 Control | — | — | — | no reliable face detected in 4 attempts through denoising step 33 |
| 5:1 | Viktor | 复用 Control | — | — | — | no reliable face detected in 4 attempts through denoising step 33 |
| 5:2 | Viktor | 注入 | 0.0003→0.0630 (+0.0627) | +0.0730 | +0.0292 | — |
| 6:1 | Viktor | 注入 | -0.0334→0.3166 (+0.3500) | +0.2725 | -0.0095 | — |

## PuLID-FLUX / run893 逐镜头

| Shot | 角色 | 插件处理 | 首帧 C→T (Δ) | 视频身份 mean Δ | DINO 视频 mean Δ | 跳过原因 |
|---|---|---|---:|---:|---:|---|
| 2:1 | Leo | 复用 Control | — | — | — | no reliable face detected in 3 attempts through denoising step 32 |
| 3:1 | Chloe | 注入 | 0.8039→0.8329 (+0.0290) | +0.0517 | -0.0023 | — |
| 4:3 | Chloe | 注入 | 0.7614→0.8049 (+0.0435) | +0.0814 | +0.0047 | — |
| 5:1 | Chloe | 注入 | 0.7063→0.7447 (+0.0384) | +0.0259 | +0.0202 | — |
| 5:2 | Chloe | 注入 | 0.8456→0.8592 (+0.0136) | +0.0012 | +0.0080 | — |

## IP-Adapter / run893 逐镜头

| Shot | 角色 | 插件处理 | 首帧 C→T (Δ) | 视频身份 mean Δ | DINO 视频 mean Δ | 跳过原因 |
|---|---|---|---:|---:|---:|---|
| 1:1 | Chloe, Julian | 复用 Control | — | — | — | multi-character shot: a single global IP-Adapter portrait cannot represent all identities; no validated multi-target IP-Adapter path |
| 2:1 | Leo | 复用 Control | — | — | — | no reliable face detected in 4 attempts through denoising step 33 |
| 2:2 | Chloe, Leo | 复用 Control | — | — | — | multi-character shot: a single global IP-Adapter portrait cannot represent all identities; no validated multi-target IP-Adapter path |
| 3:1 | Chloe | 注入 | 0.3352→0.3569 (+0.0217) | +0.0582 | +0.0212 | — |
| 4:1 | Chloe, Julian | 复用 Control | — | — | — | multi-character shot: a single global IP-Adapter portrait cannot represent all identities; no validated multi-target IP-Adapter path |
| 4:2 | Chloe, Julian | 复用 Control | — | — | — | multi-character shot: a single global IP-Adapter portrait cannot represent all identities; no validated multi-target IP-Adapter path |
| 4:3 | Chloe | 注入 | 0.1053→0.2959 (+0.1906) | +0.0391 | -0.1974 | — |
| 5:1 | Chloe | 复用 Control | — | — | — | no reliable face detected in 4 attempts through denoising step 33 |
| 5:2 | Chloe | 注入 | 0.2637→0.3276 (+0.0639) | +0.0695 | +0.0172 | — |
| 6:1 | Chloe, Leo, Eleanor, Julian | 复用 Control | — | — | — | multi-character shot: a single global IP-Adapter portrait cannot represent all identities; no validated multi-target IP-Adapter path |

## PuLID-FLUX / run1517 逐镜头

| Shot | 角色 | 插件处理 | 首帧 C→T (Δ) | 视频身份 mean Δ | DINO 视频 mean Δ | 跳过原因 |
|---|---|---|---:|---:|---:|---|
| 1:3 | Jin-woo | 注入 | 0.7926→0.8179 (+0.0253) | +0.0047 | -0.0060 | — |
| 1:4 | Hyeon-sik | 注入 | 0.8396→0.8632 (+0.0236) | +0.0363 | +0.0273 | — |
| 1:5 | Jin-woo | 注入 | 0.8361→0.8218 (-0.0143) | -0.0566 | -0.0239 | — |
| 2:1 | Hyeon-sik | 注入 | 0.7561→0.8078 (+0.0517) | +0.0565 | +0.0166 | — |
| 3:1 | Hyeon-sik | 注入 | 0.6254→0.6270 (+0.0016) | -0.0258 | +0.0485 | — |
| 4:1 | Jin-woo | 注入 | 0.8457→0.8534 (+0.0077) | +0.0178 | +0.0259 | — |
| 4:2 | Hyeon-sik | 复用 Control | 0.8222→0.8222 (+0.0000) | +0.0000 | +0.0000 | prompt_eye_closure_conflict |
| 5:1 | Hyeon-sik | 注入 | 0.7731→0.7913 (+0.0183) | +0.0431 | +0.0216 | — |
| 6:1 | Hyeon-sik | 注入 | 0.8368→0.8730 (+0.0362) | +0.0507 | -0.0088 | — |
| 7:1 | Jin-woo | 复用 Control | -0.0350→-0.0350 (+0.0000) | +0.0000 | +0.0000 | detected face height 14px is below the 24px injection threshold |

## IP-Adapter / run1517 逐镜头

| Shot | 角色 | 插件处理 | 首帧 C→T (Δ) | 视频身份 mean Δ | DINO 视频 mean Δ | 跳过原因 |
|---|---|---|---:|---:|---:|---|
| 1:1 | Tae-hwan, Hyeon-sik, Jin-woo | 复用 Control | — | — | — | multi-character shot: a single global IP-Adapter portrait cannot represent all identities; no validated multi-target IP-Adapter path |
| 1:2 | Tae-hwan, Jin-woo | 复用 Control | — | — | — | multi-character shot: a single global IP-Adapter portrait cannot represent all identities; no validated multi-target IP-Adapter path |
| 1:3 | Jin-woo | 注入 | 0.4130→0.5787 (+0.1656) | +0.1483 | +0.0391 | — |
| 1:4 | Hyeon-sik | 注入 | 0.3742→0.5330 (+0.1588) | +0.1503 | +0.0476 | — |
| 1:5 | Jin-woo | 注入 | 0.2967→0.3630 (+0.0663) | +0.0824 | +0.0094 | — |
| 2:1 | Hyeon-sik | 注入 | 0.0895→0.2973 (+0.2078) | +0.1314 | +0.0279 | — |
| 3:1 | Hyeon-sik | 注入 | 0.2628→0.3059 (+0.0431) | +0.0148 | +0.0276 | — |
| 4:1 | Jin-woo | 注入 | 0.1760→0.3397 (+0.1637) | +0.1463 | +0.0278 | — |
| 4:2 | Hyeon-sik | 复用 Control | 0.3394→0.3394 (+0.0000) | +0.0000 | +0.0000 | prompt_eye_closure_conflict |
| 5:1 | Hyeon-sik | 注入 | 0.3403→0.2619 (-0.0784) | -0.0755 | +0.0283 | — |
| 6:1 | Hyeon-sik | 注入 | 0.4216→0.6517 (+0.2301) | +0.2031 | +0.0142 | — |
| 7:1 | Jin-woo | 注入 | 0.0262→0.0234 (-0.0028) | +0.0580 | +0.0221 | — |

## 指标解释

- InsightFace：以角色原始参考脸为锚点，主要回答身份识别是否更接近。
- 参考图 DINOv2：比较扩展人脸裁剪的整体视觉特征；与 InsightFace 不同向时，不能只按身份 cosine 宣称视觉改善。
- 未通过预先固定 gate 的镜头保留在完整 episode，Treatment 复用 Control，成对增益为 0。
- `prompt_eye_closure_conflict` 表示提示词明确要求闭眼，而 3D 身份参考为中性睁眼，因此安全跳过局部 residual。
- VLM/LLM API 指标尚未运行，不得把缺失值当作 0。

## 权威机器可读输出

`/root/autodl-tmp/movie/outputs/entitybench_current_summary.json`
