"""Isolated prompt experiment; never edits the frozen production planner."""
from copy import deepcopy
from pathlib import Path

import qwen140_shot_agent as agent

PROMPT_DOC = Path(__file__).resolve().parents[1] / 'docs/entitybench_qwen140_route_agent_prompt_v2_20261104.md'
SYSTEM = PROMPT_DOC.read_text().split('```text\n', 1)[1].split('\n```', 1)[0].strip()
_compile = agent.compile_prompts


def compile_prompts(plan, style):
    prompts = _compile(plan, style)
    if plan['decision']['route'] == 'B':
        # The old tail describes the transition's start, not the action's start.
        action_plan = deepcopy(plan)
        action_plan['decision']['tail_identity'] = ''
        prompts['shot'] = _compile(action_plan, style)['shot']
    return prompts


def install():
    import qwen140_technical_resume_v2_20261103 as technical
    agent.SYSTEM = SYSTEM
    episode = technical.install_episode()
    episode.compile_prompts = compile_prompts
    return episode
