# 剧情到多镜头视频：前后端与 Agent 方案

更新：2026-10-06（北京时间）。按 `/root/autodl-tmp/movie` 当前代码核对，仅描述已实现的本地自用链路，不包含 benchmark 评测方案和未来功能。

## 1. 总体架构

用户输入剧情，创建项目后启动一次完整生成任务：

```text
React 页面
  -> FastAPI：创建项目、投递任务、查询状态、提供产物
  -> Redis / Celery：gpu 队列中的完整项目任务
  -> LangGraph：剧情 -> 分镜规划 -> 参考资产 -> 3D 人脸 -> 首帧
  -> Wan2.2：逐镜头图生视频
  -> ffmpeg：按分镜顺序拼接成片
```

| 层 | 当前职责与入口 |
| --- | --- |
| 前端 | React + Vite；`frontend/src/App.jsx`、`api.js` |
| API | FastAPI；`app/api/main.py`，不在请求进程里执行模型推理 |
| 状态存储 | MySQL/MariaDB；`app/services/store.py`，保存项目和 shot 状态 |
| 异步任务 | Redis 为 broker 和 Celery result backend；`app/workers/tasks.py` 执行 `movie.generate_project` |
| 生成编排 | `multishot/product_pipeline.py` 与 `graph.py`；串联核心生成能力 |
| 工具接口 | 本地 stdio MCP；产品进程入口 `multishot/product_mcp_server.py` |
| 真实产物 | `outputs/projects/{project_id}/`；存放规划、资产、图片、视频和日志 |

数据库保存状态索引，文件系统保存生成内容，LangGraph state 只负责本次运行的上下文传递。三者不互相替代。

## 2. 前端与 API

### 前端交互

- 输入剧情并创建项目；创建不等于生成，需点击“启动生成”。
- 选择已有项目，查看整体状态、shot 状态和错误信息。
- 首帧路线固定为 `qwen_image21`，没有模型选择器或背景/人脸上传入口；参考图由资产 Agent 生成。
- 当前选中项目每 2.5 秒轮询一次详情；项目列表在进入页面、手动刷新、创建或启动任务后刷新，不使用 WebSocket。
- 成片独立播放；产物区预览前 12 个图片/视频，全部文件提供访问链接。
- 手动读取规划、资产索引和 Wan manifest，用于检查生成内容。

浏览器默认同源访问 `/api` 与 `/media`，不把服务器的 `127.0.0.1` 写进浏览器请求地址。开发前端通常在 `http://127.0.0.1:3000`，Vite 将这两个路径代理到本机 `8000`；后端地址变化时可设置 `MOVIE_API_TARGET`。`VITE_API_BASE` 只用于明确需要跨源访问的构建配置。

日常访问优先使用构建版：FastAPI 同时提供 `frontend/dist`、API 和 media，一个端口即可访问完整页面。

### AutoDL 6006 访问

MySQL/MariaDB 启动后，构建页面并在 AutoDL 映射端口启动 Web 服务：

```bash
cd /root/autodl-tmp/movie
export PATH=/root/autodl-tmp/node-v22.23.2-linux-x64/bin:$PATH
npm --prefix frontend run build
HOST=0.0.0.0 PORT=6006 bash scripts/start_local_backend.sh
```

启动命令会前台常驻。浏览器打开 AutoDL 控制台中 **6006 对应的 HTTPS 映射地址**，不打开本机 `127.0.0.1`，也不另映射数据库或 LLM 端口。平台外部的 `8443` 是映射地址端口，不是应用内部监听端口。

访问已有项目、素材和视频只需要数据库与 Web 服务；新生成还需要可用 GPU、本地 vLLM、Redis 和 Celery worker。应用没有登录鉴权，不分享映射链接，应使用平台入口访问保护；没有入口保护时改用 SSH 隧道。

### API 契约

| 接口 | 用途 |
| --- | --- |
| `GET /api/health` | 后端健康信息 |
| `POST /api/projects` | 创建项目，写原始剧情和数据库记录 |
| `GET /api/projects` | 项目列表 |
| `GET /api/projects/{id}` | 项目、shots 和产物列表 |
| `POST /api/projects/{id}/run` | 投递完整 Celery 任务；项目已为 `running` 时返回 409 |
| `GET /api/projects/{id}/shots` | shot 状态列表 |
| `GET /api/projects/{id}/project-plan`、`GET /api/projects/{id}/asset-index`、`GET /api/projects/{id}/wan-manifest` | 读取对应 JSON；尚未生成时返回 404 |
| `GET /api/projects/{id}/artifacts` | 文件列表及 media URL |
| `GET /api/projects/{id}/final-video` | 返回成片 MP4 |
| `GET /media/{project_id}/...` | 访问项目内的图片、视频、JSON 和日志 |

创建请求只需 `story`；`backend` 可省略，传入时只允许 `qwen_image21`。API 不对外拆成规划、生图、视频、拼接多个任务接口。

### 状态与进度的实际粒度

`projects` 保存 ID、原始剧情、目录、backend、Celery task ID、状态、提示、错误和时间；`shots` 保存所属项目、shot ID、状态、提示、错误和时间。镜头内容与素材路径保留在项目 JSON 中，不重复存入数据库。

- 项目状态：`created / running / succeeded / failed`。排队阶段也是 `running`，通过 `message=Queued` 区分。
- shot 状态：`pending / running / succeeded / failed`；当前 worker 在全部首帧生成完成后才初始化/更新 shot 行，不是分镜规划一结束就出现。
- 视频开始时所有 shot 一起标为 `running`；待 Wan 批次报告返回，再批量更新结果，并非每段视频完成即实时更新数据库。
- 拼接成功后项目标为 `succeeded`。异常写入项目的 `error`；shot 错误只在返回的 Wan 报告能定位失败镜头时更新，不能保证所有阶段的异常都有逐 shot 标记。

## 3. Agent 编排

LangGraph 当前是固定顺序的线性图，不是多个自主 Agent 互相协商，也没有审核回路或自动重规划：

```text
input_story -> script_planning -> asset_generation
            -> face_3d_modeling -> shot_first_frame -> END
```

| 节点 / 执行者 | 执行方式 | 输入与输出 |
| --- | --- | --- |
| `InputStoryAgent` | 确定性执行，不调用 LLM | 保存剧情到 `input_story.txt` |
| `ScriptPlanningAgent` | 一次 LLM 调用，不使用工具 | 从剧情输出角色、场景子剧本、有序 shots，写 `project_plan.json` |
| `AssetGenerationAgent` | LLM ReAct 工具调用 + 确定性补全 | 根据规划调用场景/人物 MCP 工具，写参考图、`asset_plan.json` 和 `asset_index.json` |
| `Face3DModelingAgent` | 确定性循环，不调用 LLM | 每个角色调用建模 MCP 工具，再执行姿态标定；将角色的 3D 资产写回索引 |
| `ShotFirstFrameAgent` | 确定性循环，不调用 LLM | 按 shots 数组顺序调用首帧 MCP 工具；补充首帧和日志路径，更新规划文件 |

### 规划 LLM

本地启动配置使用 **Qwen2.5-14B-Instruct-AWQ**，vLLM 提供 `http://127.0.0.1:8001/v1`，服务名为 `qwen-local`。规划和资产工具调用共用该聊天模型；它不是负责生图的 Qwen-Image-2.1。

`build_qwen_model()` 使用 OpenAI-compatible 客户端，由 `DASHSCOPE_BASE_URL`、`MULTISHOT_QWEN_MODEL` 等环境变量决定实际服务。`scripts/start_local_worker.sh` 配置本地服务；若绕过脚本且不设置环境变量，代码仍有云端默认值，不能仅凭变量名或类名判定运行在本地。

规划提示词要求模型输出 JSON，目前通过 `json.loads` 读取，没有完整 schema 校验或失败后的自动修复。镜头数量没有固定为三个；此前验证输入明确要求三个镜头，模型在该约束下生成具体规划。

### 资产 Agent 与 MCP 的边界

资产 Agent 通过 `create_react_agent` 绑定两个工具：

- `generate_scene_asset(subscript_id, scene_name, prompt)`：生成场景背景图。
- `generate_character_asset(character_id, character_name, prompt)`：生成单人、清晰正脸身份参考图。

LLM 负责工具选择、提示词和参数；MCP 工具负责模型执行、落盘并更新索引。`asset_plan.json` 是 LLM 的总结，后续节点实际检索的是工具写入的 `asset_index.json`。执行后按规划检查场景与角色 ID，漏生成的资产由代码调用相同工具补齐。

另两个工具由确定性节点调用，不交给资产 LLM 自主选择：

- `build_3d_face_asset(character_id, reference_image_path)`：FaceLift 建模。
- `generate_shot_first_frame(shot_id, subscript_id, character_ids, first_frame_prompt, backend)`：检索素材并生成首帧。

工具实现在 `mcp_asset_server.py`，产品通过 `product_mcp_server.py` 启动；MCP 是本地子进程通信，不是对外 HTTP 服务。项目目录通过子进程环境注入。

## 4. 素材绑定与首帧生成

规划中的 ID 是后续检索依据：

```text
shot.subscript_id -> asset_index.scene_assets[场景 ID]
shot.character_ids -> asset_index.character_assets[角色 ID]
角色资产.face_3d -> Gaussian 模型与角色专属姿态标定
```

- `project_plan.json` 包含 `characters[]`、`subscripts[]`、`shots[]`。每个 shot 记录场景 ID、角色 ID、镜头内容、动作、镜头与首帧提示词；数组顺序就是生成和拼接顺序。
- 背景与人物参考图在规划之后、首帧之前由资产 Agent 生成。产品默认用 **SDXL Base 1.0，30 步、guidance 5.0**，每个场景/角色生成一份参考，不做 benchmark 的多候选身份素材筛选。
- FaceLift 从角色参考图构建 Gaussian，并在 512px 执行角色专属姿态标定。3D 模型在角色层生成一次，镜头所需姿态在首帧去噪过程中再连续渲染，不预先为每个 shot 固定一张 3D 图。
- 首帧 MCP 工具读取索引，按“背景参考第一张、该 shot 的角色参考随后”的顺序输入 Qwen-Image-2.1；每个角色的身份参考和 3D 模型按 ID 绑定，不依赖角色名称猜测。

### Qwen 产品路线

实现入口为 `multishot/qwen_image21_first_frame.py`，默认 1024x576、20 步、sequential CPU offload：

1. 文本、背景参考和角色参考共同参与一条去噪轨迹。
2. 在代码索引 step 12 预测 `pred_x0`，检测目标脸、匹配角色并估计姿态。
3. 使用对应角色的 Gaussian 与姿态标定连续渲染，做姿态回检、调色和 v7 color-safe mask。
4. 在同一轨迹加入逐角色局部 residual：基准强度 0.4；step-12 身份 cosine >= 0.40 时注入 1 步，否则 6 步；小脸降低强度，多脸重叠贡献归一化。
5. 完成剩余去噪，只保存最终首帧，写入注入状态、失败原因和逐步日志。

产品不生成 Control/Treatment 两张图，不在 step 12 分叉实验。缺脸、角色匹配或 reference 准备失败时，跳过不可用角色并继续原轨迹；无人或明确闭眼镜头不注入。CUDA OOM 直接报错，不伪装成成功回退。资产/建模阶段的硬错误仍会使项目失败。

Web 产品只开放 Qwen；PuLID-FLUX/IP-Adapter 属于独立评测路线，尚未接入该多素材产品流程。

## 5. 视频与拼接

这两步由 Celery worker 调度，位于 LangGraph 之外，不经过 LLM，也没有单独封装成 MCP：

1. `run_story_project()` 完成规划到首帧，并构建 `wan_manifest_product.json`。
2. manifest 为每个 shot 指定最终首帧、视频路径、提示词和 seed；视频提示词组合镜头内容、动作、镜头与首帧提示词。
3. `generate_shot_videos()` 在独立 `wan22-venv` 子进程中复用 `pretest.run_wan22_i2v_manifest` 的生成 runner，一次加载 Wan 模型，按清单串行生成视频；不执行 benchmark 资产规划或评分。
4. 默认 Wan2.2-TI2V-5B、49 帧、24 fps、50 步，请求尺寸 `1280*704`；每镜头约 2.04 秒，实际输出尺寸由 runner 决定。
5. `assemble_videos()` 按 manifest 顺序用 ffmpeg 拼接，Web worker 使用 H.264 重编码，输出 `final/final_video.mp4`。

Wan 每个镜头独立生成，没有跨镜头历史记忆。当前没有配音、音乐或字幕生成；规划中的 `dialogue` 不会自动变为音轨。成片是镜头串联，不是另一个长视频生成模型的输出。

## 6. GPU 与进程调度

- worker 默认并发为 1，完整项目任务持有 `product_gpu_lock()` 文件锁。即使提高 Celery 并发，遵守该锁的产品任务仍串行使用 GPU；直接 CLI 或其他评测任务不自动受此锁保护。
- 本地 LLM 开启 vLLM sleep mode。规划/工具选择前唤醒；资产工具运行前休眠，工具子进程退出后再唤醒。
- 资产工具调用被串行拦截，避免并行调用同时切换 LLM 的睡眠状态。
- 3D 建模、标定、Qwen 首帧和 Wan 阶段保持 LLM 休眠，下一次规划时再唤醒。sleep 控制只允许本地地址，不将开发控制端点暴露到公网。
- 资产工具使用短生命周期 MCP 进程；所有 shot 首帧共用一个 Qwen MCP session，复用已加载权重。该进程关闭后才启动 Wan，避免两个图像/视频模型进程同时驻留。
- Qwen 在 `qwen-image21-venv` 运行，Wan 在 `wan22-venv` 运行；产品 MCP 将模型日志送到 stderr，保留独立 JSON-RPC stdout。Celery 关闭 stdout 重定向，保证 MCP 能取得真实文件描述符。

当前默认串行生成首帧，不启用多 Qwen 并发。

## 7. 产物组织

```text
outputs/projects/{project_id}/
  input_story.txt
  project_plan.json                 # 角色、场景、shots；首帧完成后补路径
  asset_plan.json                   # 资产 Agent 总结
  asset_index.json                  # 后续检索依据；包含角色 face_3d
  assets/scenes/                    # 背景参考
  assets/characters/                # 身份参考
  assets/faces_3d/                  # Gaussian、姿态标定与建模日志
  frames/shot_*.png                 # 最终首帧
  frames/shot_*_qwen/               # 注入参考、mask、result.json
  wan_manifest_product.json
  wan_video_report.json
  shot_videos/shot_*.mp4
  final/final_video.mp4
```

`ShotFirstFrameAgent` 接收工具返回值，将 `first_frame_path`、`first_frame_denoise_log_path`、`first_frame_backend` 写回每个 shot。前端通过 API 返回的 media URL 访问文件，不直接读取服务器绝对路径。

## 8. 当前验证与边界

已有完整 API + Celery 验证项目 `proj_30_a45ac642`：本地 LLM 规划，自动生成新参考资产，FaceLift 与标定，3/3 首帧实际注入，3/3 Wan 视频生成并拼接，成片 147 帧、6.125 秒；没有 CUDA OOM。验证记录在该项目的 `verification/chain_verification.json`。

仍需明确的边界：

- 这次产品实跑是单角色、单场景、三个镜头；三个镜头由测试输入指定，不能据此宣称已完成自主镜头数量、多角色或多场景产品验收。
- 第三镜头背景汽车重复，按用户选择保留并记录；链路成功不代表画面质量全部合格。
- 390px 窄屏仍有横向溢出。
- 没有人工审核、自动质量修复、跨镜头漂移反馈、取消或分阶段断点续跑机制；`run` 是完整项目入口，重新启动不等同于可靠续跑。
- 当前是本地工作台，没有账号、鉴权和公网服务防护；不把现有 API/media 服务当作公网多用户产品。

核心实现索引：`app/api/main.py`、`app/workers/tasks.py`、`frontend/src/App.jsx`、`multishot/graph.py`、`agents.py`、`product_pipeline.py`、`product_gpu.py`、`product_mcp_server.py`、`product_face_assets.py`、`qwen_image21_first_frame.py`。模型启动配置见 `scripts/start_local_llm.sh`、`scripts/start_local_worker.sh`。
