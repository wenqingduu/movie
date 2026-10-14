# 系统盘占用核查（2026-10-14，北京时间）

**最新：用户已明确批准清理pip缓存和pip解包残留，已清空 `/root/.cache/pip` 并删除11个 `/tmp/pip-unpack-*` 目录。**
系统盘实际释放7.11GiB，空闲从约2.19GiB增至约9.30GiB；保留已安装环境及空的pip缓存根目录。
执行前核验无pip进程、目标无活动FD／映射／工作目录引用，各安装环境没有软链接指向这些目标。
其他/tmp文件只盘点，未删除；用户对“/tmp能否删除”的询问不作为整目录清空授权。
完成证明：`outputs/disk_cleanup_review_20261014/delete_pip_cache_and_unpack_result.json`。
下文2.19GiB及各目录大小为清理前快照；当前系统盘约20.70GiB已用／9.30GiB可用，数据盘仍约59.75GiB可用。

本次用户只批准删除旧Wan5B；仅删除 `/root/autodl-tmp/movie/models/video/Wan2.2-TI2V-5B`，
释放31.85GiB，数据盘空闲约59.76GiB。系统盘与数据盘是不同文件系统，系统盘容量因此不变。
当前系统盘overlay总30GiB、已用约27.81GiB、可用约2.19GiB（93%）；数据盘总320GiB、已用约260GiB、可用约59.76GiB。

## 系统盘中的大项

以下为du可见目录占用；overlay的镜像层与硬链接会使du和df的统计边界不同，不能把目录体积简单相加当作必然可释放空间。

| 目录 | 实测约GiB | 用途及处置 |
|---|---:|---|
| `/root/miniconda3` | 7.56 | 基础Python及已装依赖。所有Qwen／FaceLift／评分环境的python最终链接到这里，不能整删 |
| `/root/.vscode-server` | 5.78 | VS Code远程服务、多个server安装、插件；其中cli约4.34GiB。保留当前使用版本，旧版本需另核对后确认 |
| `/root/.cache/pip` | 5.63 | pip下载HTTP／wheel缓存，属于安装缓存，不是已安装环境；可作为单独待确认清理项 |
| `/usr/local/cuda-11.8` | 5.15 | CUDA工具链，Gaussian扩展和现有环境可能依赖，不能随意删除 |
| `/tmp` | 3.23 | pip解包、npm、构建和实验／浏览器临时目录，不应整删 |
| `/root/.cache/clip` | 1.20 | 模型缓存，不能按普通安装缓存整删 |
| `/root/.local/share/Trash` | 0.85 | 回收站；需要用户确认是否有要恢复的内容 |
| `/root/.cache/puppeteer` | 0.57 | 浏览器权重／程序，用于网页检查，未删除 |
| `/root/.cache/openfacelift` | 0.50 | FaceLift相关缓存，保留，不能因为名称是cache就删 |

系统盘满主要来自Python／CUDA环境、远程编辑器服务、包下载缓存和临时解包；项目Qwen／A14B大权重及outputs在数据盘。
这不意味着生成的视频填满系统盘，也不能通过删除数据盘的模型腾出系统盘容量。

## pip清理完成与其余/tmp候选

`/root/.cache/pip`约5.63GiB和11个`/tmp/pip-unpack-*`约1.48GiB已按用户批准删除，实际df释放约7.11GiB。
保留全部已安装依赖，没有清理其他模型cache或整个/tmp。
剩余/tmp约1.75GiB：`codex-npm-cache`约0.62GiB、四个旧tmp二进制文件合计约0.84GiB、构建／浏览器／实验调试目录等。
四个旧tmp文件分别为`tmpcoywx23d`、`tmp_hhoelin`、`tmp_raxllz7`、`tmpk37ym5w7`，修改时间为9月23／27日；其中三份有ZIP头但不能完整打开，可能为中断下载，来源未确认，未删。
`/tmp/code-*`中已发现当前VS Code通信socket，`clashctl-autostart.lock`被活动进程使用；会话／通信／锁文件保留。
实验调试素材、对比图、构建目录及`cairosvg_pkg`也不能仅因位于/tmp就视为垃圾；需要确认依赖或备份后再按精确路径处理。
剩余目录大小与活动引用快照：`outputs/disk_cleanup_review_20261014/tmp_remaining_review.json`。当前无打开FD不等于未来工具不依赖，其他候选仍待明确确认。
旧VS Code server、回收站也可后续单独列清单，但不删除正在使用的IDE、基础Conda、CUDA或FaceLift／评测模型缓存。

若以后需要安装依赖，可将`PIP_CACHE_DIR`及`TMPDIR`指向数据盘专用缓存／临时目录，先记录配置、设置容量守护；
本次没有修改环境变量、搬迁已安装环境或启动新实验；只执行已批准的pip缓存／解包清理。

## 数据盘清理与前端影响

用户仅批准清理候选A1的旧Wan5B，其他A2／A3／B1／B2／B3全部待确认。
当前Qwen benchmark使用A14B＋蒸馏LoRA，必需模型均保留；正式三路线20集、消融、报告、旧素材／3D与其他未提交改动保留。
前端默认视频配置仍指向已删除的Wan5B，暂不可直接运行；需要另行决定迁移为A14B路线或重新下载，不能自动改前端配置。
删除核验记录见`outputs/disk_cleanup_review_20261014/delete_A1_result.json`，候选状态见[disk_cleanup_review_20261014.json](disk_cleanup_review_20261014.json)。
