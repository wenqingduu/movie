import asyncio
import json
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from .mcp_asset_server import PROJECT_DIR_ENV
from .prompts import ASSET_GENERATION_PROMPT, SCRIPT_PLANNING_PROMPT

# 这里使用阿里云 DashScope 的 OpenAI 兼容接口。
# API key 先留空，后续可以改成从环境变量读取。
QWEN_API_KEY = ""
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen-plus"


def build_qwen_model():
    """构建 LangChain 版本的 Qwen chat model。

    ChatOpenAI 是 LangChain 的 OpenAI-compatible wrapper。
    由于 Qwen 提供 OpenAI 兼容接口，所以这里实际调用的是 qwen-plus。
    """

    return ChatOpenAI(
        model=QWEN_MODEL,
        api_key=QWEN_API_KEY,
        base_url=QWEN_BASE_URL,
    )

class InputStoryAgent:
    """输入 Agent。

    这个 Agent 不调用大模型，只做一件事：
    把用户输入的原始故事保存到项目目录里。
    """

    def run(self, story: str, project_dir: str):
        project_path = Path(project_dir)
        project_path.mkdir(parents=True, exist_ok=True)

        input_path = project_path / "input_story.txt"
        input_path.write_text(story, encoding="utf-8")

        return {
            "story": story,
            "project_dir": str(project_path),
            "input_path": str(input_path),
        }


class ScriptPlanningAgent:
    """剧本规划 Agent。

    输入：用户原始故事。
    输出：project_plan.json。

    project_plan 是后面所有步骤的语义基础，里面包含：
    - characters：人物表
    - subscripts：按场景变化划分的子剧本
    - shots：每个子剧本下细化出的镜头
    """

    def __init__(self, model=None):
        self.model = model or build_qwen_model()

    def run(self, state: dict):
        # 剧本规划目前不需要工具，直接要求 Qwen 输出 JSON。
        response = self.model.invoke([
            ("system", SCRIPT_PLANNING_PROMPT),
            ("user", state["story"]),
        ])
        plan = json.loads(response.content)

        plan_path = Path(state["project_dir"]) / "project_plan.json"
        plan_path.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        state["project_plan"] = plan
        state["project_plan_path"] = str(plan_path)
        return state


class AssetGenerationAgent:
    """资产生成 Agent。

    这里使用更标准的现代 Agent 方式：
    1. MCP server 暴露 generate_scene_asset / generate_character_asset。
    2. MultiServerMCPClient 从 MCP server 拉取 tools。
    3. create_react_agent 把 MCP tools 绑定到 Qwen-compatible LLM。
    4. 模型自主决定调用哪个工具、传什么参数。
    5. MCP 工具执行资产生成，并在工具侧更新 asset_index.json。

    这和手写 OpenAI tools 参数不同：
    Agent 不再自己维护工具 schema，也不再自己写 tool_call loop。
    """

    def __init__(self, model=None):
        self.model = model or build_qwen_model()

    def run(self, state: dict):
        # LangGraph 当前用同步节点，所以这里用 asyncio.run 包一层。
        return asyncio.run(self.arun(state))

    async def arun(self, state: dict):
        project_dir = Path(state["project_dir"])

        # 每次资产生成都启动一个 MCP stdio server。
        # project_dir 通过环境变量注入给 server，避免暴露成模型工具参数。
        mcp_client = MultiServerMCPClient({
            "multishot_assets": {
                "command": "python",
                "args": ["-m", "multishot.mcp_asset_server"],#当前程序会启动一个新的 Python 子进程，运行：multishot/mcp_asset_server.py
                "transport": "stdio",
                "env": {PROJECT_DIR_ENV: str(project_dir)},
            }
        })

        tools = await mcp_client.get_tools()
        tools = [
            tool for tool in tools
            if tool.name in {"generate_scene_asset", "generate_character_asset"}
        ]
        agent = create_react_agent(self.model, tools)

        result = await agent.ainvoke({
            "messages": [
                ("system", ASSET_GENERATION_PROMPT),
                ("user", json.dumps(state["project_plan"], ensure_ascii=False)),
            ]
        })

        # create_react_agent 会负责多轮 tool calling。
        # 最后一条 assistant message 应该是提示词要求的 asset_plan JSON。
        final_message = result["messages"][-1]
        asset_plan = json.loads(final_message.content)

        asset_plan_path = project_dir / "asset_plan.json"
        asset_index_path = project_dir / "asset_index.json"
        asset_index = json.loads(asset_index_path.read_text(encoding="utf-8"))

        asset_plan_path.write_text(
            json.dumps(asset_plan, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        state["asset_plan"] = asset_plan
        state["asset_index"] = asset_index
        state["asset_plan_path"] = str(asset_plan_path)
        state["asset_index_path"] = str(asset_index_path)
        return state


class Face3DModelingAgent:
    """3D 人脸建模 Agent。

    输入：asset_index.json 里的 character_assets。
    输出：为每个人物参考图生成一个 3D 人脸资产，并把结果写回 asset_index.json。

    这个节点目前不需要 LLM 自主规划，因为目标很确定：
    每个 character reference image 都调用一次 build_3d_face_asset。
    后续如果有多个 3D 建模工具、质量检查工具，再考虑改成 tool-bound agent。
    """

    def run(self, state: dict):
        return asyncio.run(self.arun(state))

    async def arun(self, state: dict):
        project_dir = Path(state["project_dir"])
        asset_index_path = project_dir / "asset_index.json"
        asset_index = json.loads(asset_index_path.read_text(encoding="utf-8"))

        mcp_client = MultiServerMCPClient({
            "multishot_assets": {
                "command": "python",
                "args": ["-m", "multishot.mcp_asset_server"],
                "transport": "stdio",
                "env": {PROJECT_DIR_ENV: str(project_dir)},
            }
        })

        tools = await mcp_client.get_tools()
        build_face_tool = next(tool for tool in tools if tool.name == "build_3d_face_asset")

        face_3d_assets = {}
        for character_id, character_asset in asset_index["character_assets"].items():
            result = await build_face_tool.ainvoke({
                "character_id": character_id,
                "reference_image_path": character_asset["path"],
            })
            face_3d_assets[character_id] = result

        asset_index = json.loads(asset_index_path.read_text(encoding="utf-8"))

        state["asset_index"] = asset_index
        state["face_3d_assets"] = face_3d_assets
        return state


class ShotFirstFrameAgent:
    """Shot 首帧生成 Agent。

    输入：project_plan 里的 shots，以及 asset_index 里的场景/人物/3D 人脸资产。
    输出：每个 shot 的首帧图片路径和内部去噪控制日志。

    这个节点的复杂逻辑放在 MCP 工具内部：
    - 根据 shot_id / subscript_id / character_ids 检索相关素材。
    - 调用伪 diffusion 生成流程。
    - 在固定步数检测人脸清晰度。
    - 清晰后做人脸特征、相似度、3D 人脸检索和 rollout 注入。
    """

    def run(self, state: dict):
        return asyncio.run(self.arun(state))

    async def arun(self, state: dict):
        project_dir = Path(state["project_dir"])
        generation_model = state.get("generation_model", "pseudo_diffusion_v1")

        mcp_client = MultiServerMCPClient({
            "multishot_assets": {
                "command": "python",
                "args": ["-m", "multishot.mcp_asset_server"],
                "transport": "stdio",
                "env": {PROJECT_DIR_ENV: str(project_dir)},
            }
        })

        tools = await mcp_client.get_tools()
        first_frame_tool = next(tool for tool in tools if tool.name == "generate_shot_first_frame")

        for shot in state["project_plan"]["shots"]:
            result = await first_frame_tool.ainvoke({
                "shot_id": shot["shot_id"],
                "subscript_id": shot["subscript_id"],
                "character_ids": shot["character_ids"],
                "first_frame_prompt": shot["first_frame_prompt"],
                "generation_model": generation_model,
            })
            shot["first_frame_path"] = result["frame_path"]
            shot["first_frame_denoise_log_path"] = result["denoise_log_path"]
            shot["first_frame_generation_model"] = generation_model

        project_plan_path = project_dir / "project_plan.json"
        project_plan_path.write_text(
            json.dumps(state["project_plan"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return state
