# 磁盘清理：A1已完成，其余待确认（2026-10-14）

**用户仅批准A1旧Wan5B，已删除该模型53个文件，实际释放31.85GiB；其他候选全部保持。**
当前数据盘320GiB、约59.76GiB可用；下面原候选占用是删除前的盘点，只有A1已执行。其他项目此前保留要求仍有效。
数据盘删除证明：`outputs/disk_cleanup_review_20261014/delete_A1_result.json`。
用户随后另批准系统盘pip缓存和11个pip-unpack残留，已释放7.11GiB，系统盘现约9.30GiB可用；其他/tmp未删，见[占用核查](system_disk_usage_20261014.md)。

## 推荐的最小方案A：不删除实验结果

| ID | 精确目录 | 实占GiB | 删除影响 |
|---|---|---:|---|
| A1 | `/root/autodl-tmp/movie/models/video/Wan2.2-TI2V-5B` | 31.85 | 当前Qwen全量使用A14B，不依赖5B；前端产品默认视频仍指向5B，删除后该默认路线需重新下载或明确迁移。旧5B视频／报告不删。 |
| A2 | `/root/autodl-tmp/movie/models/ip_adapter/h94-IP-Adapter` | 4.09 | 当前Qwen路线不加载；旧IP-Adapter实验重新生成需要重新下载，已有媒体／评测证据保留。 |
| A3 | `/root/autodl-tmp/qwen-wheel-cache` | 0.79 | 保留已安装环境，仅删除两个wheel安装包；以后离线重建环境需重新获取。 |

A1＋A2＋A3原方案共约36.73GiB；用户只批准A1，当前空闲约59.76GiB，A2／A3没有删除授权。
**前端默认视频仍依赖A1的Wan5B。若要继续直接运行该前端路线，保留A1，不能把“不用于benchmark”解释成“整个项目无用”。**
本方案仅删权重／安装包，不删旧视频、素材、3D、Control/Treatment消融证据、正式分数或三路线比较。

## 可选方案B：不默认执行

| ID | 路径／范围 | 实占GiB | 删除影响 |
|---|---|---:|---|
| B1 | `/root/autodl-tmp/movie/models/diffusion/sdxl-base-1.0` | 6.46 | 当前Qwen benchmark及FaceLift本地checkpoint不依赖该模型；前端资产Agent默认仍用SDXL，删除会影响前端素材生成和旧SDXL实验重跑。 |
| B2 | `/root/autodl-tmp/movie/outputs/entitybench_pulid_i2v_a14b_4step_full140/assets` | 13.17 | 当前Qwen全量不能复用这些身份；删除属于放弃FLUX全量续跑，需明确撤销此前保留要求。必须保留旧全量plan／split／status、正式三路线20集、报告和其他历史消融。 |
| B3 | `third_party/PuLID/models内下列指定权重，保留antelopev2` | 23.58 | 当前Qwen只复用PuLID源码的渲染／调色助手而不加载这些生成权重；删除后不能直接恢复FLUX生成。绝不可整删models目录，antelopev2仍为Qwen素材筛选／身份gate／标定必需。 |

B3精确路径（只有明确批准B3才可删除；不删除PuLID源代码或antelopev2）：

- `/root/autodl-tmp/movie/third_party/PuLID/models/flux-dev-fp8.safetensors`。
- `/root/autodl-tmp/movie/third_party/PuLID/models/flux-vae-bf16.safetensors`。
- `/root/autodl-tmp/movie/third_party/PuLID/models/flux_dev_quantization_map.json`。
- `/root/autodl-tmp/movie/third_party/PuLID/models/pulid_flux_v0.9.1.safetensors`。
- `/root/autodl-tmp/movie/third_party/PuLID/models/xflux_text_encoders`。
- `/root/autodl-tmp/movie/third_party/PuLID/models/clip-vit-large-patch14`。
- `/root/autodl-tmp/movie/third_party/PuLID/models/eva_clip`。
- `/root/autodl-tmp/movie/third_party/PuLID/models/ae.safetensors`。

本次不建议删旧消融或正式实验结果：它们是后续论文证据，收益比大模型清理小。
B2的assets目录文件实占约13.17GiB，旧全量根目录约13.22GiB；144份扩展身份资产此前明确要求保留，用户需明确选择放弃FLUX全量续跑才能删。

## 必须保留

Qwen-Image-2.1、A14B基础权重／蒸馏LoRA、FaceLift权重、Antelope、facexlib检测／解析模型、官方DINO／CLIP／GroundingDINO／VBench依赖和全部现有运行环境。
当前Qwen20、新FLUX20、旧FLUX20、论文比较归档、脸部裁剪／评分、人物定位修正、已有消融和其他未提交文件保持。
本地Qwen2.5-14B／vLLM不是本benchmark依赖，但前端规划和Agent仍用，默认保留。
`/root/.cache/pip`约5.63GiB位于系统overlay，删除不能作为数据盘空间增量；本次未清缓存。

逐文件stat／hardlink／设备信息和审批状态见[disk_cleanup_review_20261014.json](disk_cleanup_review_20261014.json)。
只有用户明确确认ID／精确路径后再执行删除，重算实际df并更新交接；本文不是删除授权。
