from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .agents import (
    AssetGenerationAgent,
    Face3DModelingAgent,
    InputStoryAgent,
    ShotFirstFrameAgent,
    ScriptPlanningAgent,
    build_qwen_model,
)


class MultiShotState(TypedDict, total=False):
    """LangGraph 中流转的全局状态。

    这个 state 是所有节点共享的“工作台”：
    前面的节点把结果写进来，后面的节点从里面读取上下文。
    total=False 表示字段可以随着流程推进逐步补齐。
    """

    story: str
    project_dir: str
    input_path: str
    project_plan: dict[str, Any]
    project_plan_path: str
    asset_plan: dict[str, Any]
    asset_plan_path: str
    asset_index: dict[str, Any]
    asset_index_path: str
    face_3d_assets: dict[str, Any]
    generation_model: str


def build_multishot_graph(model=None):
    """构建当前最小闭环的 LangGraph。

    现在图里只有三步：
    1. input_story：保存原始故事。
    2. script_planning：把故事规划成 characters / subscripts / shots。
    3. asset_generation：让模型通过 MCP 工具生成场景和人物资产。
    4. face_3d_modeling：读取人物参考图，调用 3D 人脸建模工具。
    5. shot_first_frame：逐 shot 生成首帧，并在工具内部预留去噪注入实验逻辑。

    之后要加视频生成、漂移检测时，只需要继续在这里加节点和边。
    """

    model = model or build_qwen_model()

    # Agent 是节点内部的执行者；LangGraph 负责调度节点顺序和传递 state。
    input_story_agent = InputStoryAgent()
    script_planning_agent = ScriptPlanningAgent(model)
    asset_generation_agent = AssetGenerationAgent(model)
    face_3d_modeling_agent = Face3DModelingAgent()
    shot_first_frame_agent = ShotFirstFrameAgent()

    graph = StateGraph(MultiShotState)

    def input_story_node(state: MultiShotState):
        """图节点：把用户故事保存到项目目录。"""

        return input_story_agent.run(state["story"], state["project_dir"])

    def script_planning_node(state: MultiShotState):
        """图节点：调用 LLM 生成结构化剧本规划。"""

        return script_planning_agent.run(state)

    def asset_generation_node(state: MultiShotState):
        """图节点：调用 MCP tool-bound agent 生成资产并写资产索引。"""

        return asset_generation_agent.run(state)

    def face_3d_modeling_node(state: MultiShotState):
        """图节点：为每个人物参考图生成 3D 人脸资产。"""

        return face_3d_modeling_agent.run(state)

    def shot_first_frame_node(state: MultiShotState):
        """图节点：逐 shot 生成首帧。"""

        return shot_first_frame_agent.run(state)

    graph.add_node("input_story", input_story_node)
    graph.add_node("script_planning", script_planning_node)
    graph.add_node("asset_generation", asset_generation_node)
    graph.add_node("face_3d_modeling", face_3d_modeling_node)
    graph.add_node("shot_first_frame", shot_first_frame_node)

    # 当前是线性图。后面如果某个节点失败、需要人工审核或重试，可以改成条件边。
    graph.add_edge(START, "input_story")
    graph.add_edge("input_story", "script_planning")
    graph.add_edge("script_planning", "asset_generation")
    graph.add_edge("asset_generation", "face_3d_modeling")
    graph.add_edge("face_3d_modeling", "shot_first_frame")
    graph.add_edge("shot_first_frame", END)

    return graph.compile()



def run_demo():
    """本地 demo 入口。

    可以通过下面命令运行：
    python -m multishot.graph
    """

    story = """
一个年轻女孩在雨夜的小巷中奔跑，身后有一名侦探追踪她。
女孩冲进一家旧书店，发现桌上有一张写着自己名字的照片。
"""
    output_dir = "outputs/demo_project"

    graph = build_multishot_graph()
    state = graph.invoke({
        "story": story,
        "project_dir": output_dir,
    })

    print("project_plan:", state["project_plan_path"])
    print("asset_index:", state["asset_index_path"])
    print("face_3d_assets:", list(state.get("face_3d_assets", {}).keys()))
    print("shots updated in:", state["project_plan_path"])


if __name__ == "__main__":
    run_demo()
