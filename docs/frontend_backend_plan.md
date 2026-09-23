# 剧情到长视频本地前后端方案

本文基于当前 `/root/autodl-tmp/movie` 代码，描述我们现在要实现的本地自用版前后端。目标不是一开始就做公网服务，而是先让自己能在本地浏览器里输入剧情、创建项目、启动生成、查看状态和预览产物。

未来如果要变成“大家都可以访问”的服务，统一放在最后一章作为展望。

---

## 1. 当前目标

当前阶段只做本地自用：

```text
浏览器
  |
  | http://127.0.0.1:3000
  v
Vite + React 前端
  |
  | http://127.0.0.1:8000/api
  v
FastAPI 后端
  |
  v
SQLite + 本地项目目录 + 单线程后台生成任务
```

当前不做：

- 公网部署。
- 用户登录。
- 多用户隔离。
- Redis / Celery。
- Nginx。
- PostgreSQL。
- 对象存储。
- 多 GPU worker。

这些都放到最后的未来展望里。

---

## 2. 当前代码基础

已有模型和生成链路：

- `movie/multishot/graph.py`
  - 已有 LangGraph 流程：故事输入 -> 剧本规划 -> 资产生成 -> 3D 人脸建模 -> 首帧生成。

- `movie/multishot/agents.py`
  - 已有 `ScriptPlanningAgent`、`AssetGenerationAgent`、`Face3DModelingAgent`、`ShotFirstFrameAgent`。

- `movie/multishot/mcp_asset_server.py`
  - 已有 MCP 工具：生成场景图、人物参考图、构建 3D 人脸、生成 shot 首帧。

- `movie/pretest/run_wan22_i2v_manifest.py`
  - 已有 Wan2.2 图生视频 manifest runner，可以一次加载模型批量跑多个 shot。

当前已经新增的产品入口：

- `movie/multishot/product_pipeline.py`
  - 产品链路入口。
  - 不放在 `pretest`。
  - 不依赖 EntityBench 评测脚本。
  - 负责从用户剧情生成项目产物、写产品用 Wan manifest、拼接已有 shot 视频。

当前已经新增的本地后端：

- `movie/app/api/main.py`
  - FastAPI 入口。

- `movie/app/services/store.py`
  - SQLite 项目和任务状态存储。

- `movie/app/services/project_runner.py`
  - FastAPI 进程内单线程后台执行器。
  - 当前用于串行执行生成任务，避免同时启动多个重任务占满显存。

当前已经新增的前端：

- `movie/frontend/`
  - Vite + React 项目。
  - 本地开发时跑在 `3000` 端口。

---

## 3. 当前目录结构

当前建议结构如下：

```text
movie/
  multishot/
    product_pipeline.py          # 产品链路入口
    graph.py                     # 现有 LangGraph
    agents.py                    # 现有各阶段 agent

  pretest/                       # 评测、benchmark、EntityBench 脚本

  app/
    api/
      main.py                    # FastAPI API 入口
    services/
      store.py                   # SQLite 状态存储
      project_runner.py          # 本地后台任务执行器

  frontend/
    package.json
    vite.config.js
    index.html
    src/
      App.jsx
      api.js
      main.jsx
      styles.css

  outputs/
    projects/
      {project_id}/              # 每个本地项目的产物目录
```

设计边界：

- `multishot/`：核心生成能力。
- `pretest/`：评测和实验，不作为产品入口。
- `app/`：本地 Web 后端。
- `frontend/`：本地 Web 前端。
- `outputs/projects/`：本地项目结果。

---

## 4. 当前本地链路

用户在前端输入剧情后，链路是：

```text
React 前端
-> POST /api/projects
-> FastAPI 创建项目
-> 写入 SQLite projects 表
-> 写入 outputs/projects/{project_id}/input_story.txt
```

点击“启动生成”后：

```text
React 前端
-> POST /api/projects/{project_id}/run
-> FastAPI 创建 job
-> 单线程后台执行器调用 multishot.product_pipeline.run_story_project
-> 生成 project_plan.json / asset_index.json / 首帧 / wan_manifest_product.json
-> 更新 job 状态
```

点击“拼接视频”后：

```text
React 前端
-> POST /api/projects/{project_id}/assemble
-> 后台执行器调用 multishot.product_pipeline.assemble_videos
-> 读取 wan_manifest_product.json 中的 shot_videos
-> ffmpeg 拼接 final/final_video.mp4
```

注意：当前 `run_story_project` 主要跑到首帧和 manifest。Wan2.2 shot video 生成还需要继续接入产品链路。

---

## 5. 后端 API 设计

当前已实现的主要接口：

```text
GET    /api/health

POST   /api/projects
GET    /api/projects
GET    /api/projects/{project_id}

POST   /api/projects/{project_id}/run
POST   /api/projects/{project_id}/assemble

GET    /api/jobs
GET    /api/jobs/{job_id}

GET    /api/projects/{project_id}/project-plan
GET    /api/projects/{project_id}/asset-index
GET    /api/projects/{project_id}/wan-manifest
GET    /api/projects/{project_id}/artifacts
GET    /api/projects/{project_id}/final-video

GET    /media/{project_id}/...
```

创建项目请求示例：

```json
{
  "title": "雨夜旧书店",
  "story": "一个年轻女孩在雨夜的小巷中奔跑...",
  "generation_model": "juggernaut-xl-v9",
  "base_seed": 1000,
  "autorun": false
}
```

返回结构大致是：

```json
{
  "project": {
    "id": "proj_xxx",
    "title": "雨夜旧书店",
    "status": "created",
    "project_dir": "/root/autodl-tmp/movie/outputs/projects/proj_xxx"
  },
  "job": null
}
```

---

## 6. 任务状态设计

当前本地 MVP 的任务状态存在 SQLite 里。

项目状态：

```text
created
running
first_frames_ready
assembling
completed
failed
```

任务状态：

```text
queued
running
succeeded
failed
```

当前后台执行器是单线程：

```text
ThreadPoolExecutor(max_workers=1)
```

原因：

- 本地自用不需要复杂队列。
- 单 GPU 机器不能同时跑多个重生成任务。
- 先保证行为简单、可调试。

---

## 7. 项目产物目录

每个项目一个目录：

```text
/root/autodl-tmp/movie/outputs/projects/{project_id}/
  input_story.txt
  project_plan.json
  asset_plan.json
  asset_index.json
  wan_manifest_product.json
  first_frames/
  shot_videos/
  final/
    final_video.mp4
```

前端通过后端返回的 URL 访问产物：

```text
/media/{project_id}/...
```

本地自用阶段可以直接这样访问。以后做公网服务时，需要把 `/media` 改成鉴权下载或签名 URL。

---

## 8. 前端设计

当前前端使用 Vite + React，不用 Next.js。

当前文件：

```text
movie/frontend/
  package.json
  vite.config.js
  index.html
  src/
    App.jsx
    api.js
    main.jsx
    styles.css
```

当前页面能力：

- 创建项目。
- 输入剧情。
- 可选创建后立即生成。
- 项目列表。
- 项目详情。
- 查看最新 job 状态。
- 启动生成。
- 拼接视频。
- 预览图片和视频产物。
- 查看全部文件列表。
- 读取 `project_plan.json`、`asset_index.json`、`wan_manifest_product.json`。

前端默认请求后端：

```text
http://127.0.0.1:8000
```

如果后端地址不同，可以设置：

```bash
VITE_API_BASE=http://你的后端地址:8000 npm run dev
```

---

## 9. 本地启动方式

### 9.1 启动后端

```bash
cd /root/autodl-tmp/movie
source .venv/bin/activate
export PYTHONPATH=/root/autodl-tmp/movie:$PYTHONPATH
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

如果只在同一台机器本地访问，也可以：

```bash
uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

后端文档地址：

```text
http://127.0.0.1:8000/docs
```

### 9.2 启动前端

机器默认 `node` 可能比较老，先使用本机已有 Node 22：

```bash
export PATH=/root/autodl-tmp/node-v22.23.2-linux-x64/bin:$PATH
node -v
npm -v
```

启动前端：

```bash
cd /root/autodl-tmp/movie/frontend
npm install
npm run dev
```

打开：

```text
http://127.0.0.1:3000
```

如果前端跑在 AutoDL 上、浏览器在自己电脑上，就打开 AutoDL 映射出来的 `3000` 端口。

### 9.3 当前不需要单独启动 worker

当前本地 MVP 不需要单独启动 worker。任务执行器在 FastAPI 进程里：

```text
movie/app/services/project_runner.py
```

后续如果要拆成独立 worker，再新增：

```text
movie/app/workers/gpu_worker.py
```

---

## 10. 当前已完成检查

后端已做过基础检查：

```text
python -m py_compile app/api/main.py app/services/store.py app/services/project_runner.py multishot/product_pipeline.py
```

FastAPI 轻量接口检查通过：

```text
GET /api/health
GET /api/projects
```

前端代码已写好，但当前环境 `npm install` 下载依赖时卡住，所以还没有完成 `npm run build` 验证。后续等 npm 依赖装好后需要执行：

```bash
cd /root/autodl-tmp/movie/frontend
npm install
npm run build
```

---

## 11. 下一步开发任务

按优先级：

1. 解决前端 `npm install` 下载依赖问题，完成 `npm run build`。
2. 启动前端和后端，走通“创建项目 -> 启动生成 -> 查看状态”。
3. 把 Wan2.2 shot video 生成接入 `multishot.product_pipeline`。
4. 让 `run_story_project` 可以继续跑到 `shot_videos/`。
5. 用 `assemble_videos` 拼出 `final/final_video.mp4`。
6. 前端增加 shot 级别状态。
7. 前端增加单个 shot 重跑入口。
8. 增加失败任务继续跑。

---

## 12. 当前设计结论

当前阶段的设计就是：

```text
本地前后端分离
React 3000 + FastAPI 8000
SQLite 记录项目和任务
FastAPI 内置单线程后台执行器
multishot.product_pipeline 作为产品生成入口
pretest 继续只做评测和实验
```

这套设计适合学习前后端，也适合先跑通产品闭环。它故意没有引入复杂部署组件，避免还没跑通生成链路就先被工程基础设施拖住。

---

## 13. 未来展望：变成大家都可以访问的服务

以后如果要从“本地自用”升级成“多人可访问服务”，再引入下面这些能力。

### 13.1 推荐生产拓扑

```text
用户浏览器
  |
  | HTTPS
  v
公网服务器
  - Nginx
  - React 静态前端
  - FastAPI 后端
  - PostgreSQL
  - Redis
  - 对象存储或静态文件服务
  |
  | 任务队列 / API / 对象存储
  v
AutoDL / GPU 机器
  - movie 代码
  - multishot pipeline
  - Wan2.2
  - GPU worker
```

原则：

- 公网服务器负责访问入口、HTTPS、用户、数据库、任务队列。
- AutoDL 不作为稳定公网 Web 服务端。
- AutoDL 只作为 GPU worker，负责拉任务、跑生成、上传结果。

### 13.2 为什么不直接让 AutoDL 当公网服务端

AutoDL 更适合作为算力节点，不适合长期当公网服务：

- 实例可能停机、重启、换端口。
- 域名和 HTTPS 不方便长期维护。
- 机器上有模型、API key、生成结果，直接暴露公网风险更高。
- GPU 推理会占用 CPU、显存、IO，可能影响 Web API 稳定性。
- 多人访问时需要限流、排队、计费和配额。

### 13.3 需要新增的生产能力

后端：

- 用户登录和鉴权。
- 项目归属和权限。
- PostgreSQL 替代 SQLite。
- Redis + Celery/RQ/Dramatiq 替代进程内线程池。
- 任务取消、重试、优先级。
- shot 级别状态。
- API 限流。

前端：

- 登录页。
- 项目历史。
- 多版本生成。
- shot 级别编辑和重跑。
- 生成成本和队列位置提示。
- 更完整的错误提示。

存储：

- 对象存储。
- 文件下载鉴权。
- 大视频 Range 支持。
- 定期清理临时产物。

部署：

- Nginx。
- HTTPS。
- systemd 或 Docker Compose。
- 日志收集。
- 监控告警。
- 多 GPU worker。

### 13.4 公网部署路线

建议路线：

```text
阶段 1：本地自用版跑通
阶段 2：公网服务器部署前端 + API + DB
阶段 3：AutoDL 改成远程 GPU worker
阶段 4：结果上传对象存储
阶段 5：加入登录、限流、任务配额
阶段 6：多 GPU worker 和监控
```

也就是说，未来不是推翻当前设计，而是把当前的：

```text
FastAPI 内置后台执行器
```

替换成：

```text
Redis 队列 + 独立 GPU worker
```

把当前的：

```text
本地 outputs/projects
```

替换成：

```text
对象存储 + 签名 URL
```

把当前的：

```text
SQLite
```

替换成：

```text
PostgreSQL
```
