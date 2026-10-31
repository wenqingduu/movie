#!/usr/bin/env python3
"""Compact shot planning with one character-lookup tool call; no repair loop."""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path

from qwen140_integrated_episodes import digest, sha, write_once
from qwen140_pilot_route_agent import config, image_data_url

PROMPT_DOC = Path(__file__).resolve().parents[1] / "docs/entitybench_qwen140_route_agent_prompt_v1_20261026.md"
SYSTEM = PROMPT_DOC.read_text().split("```text\n", 1)[1].split("\n```", 1)[0].strip()


LOOKUP_TOOL = {"type": "function", "function": {
    "name": "lookup_characters", "description": "查询待绘关键帧中可见、且尾帧未出现或无法确认的角色素材。硬切无尾帧时默认查询全部调度角色，仅排除原文明示稍后入场、画外或仅声音者。返回描述、参考图及编号；无需素材时names为空。",
    "parameters": {"type": "object", "properties": {
        "names": {"type": "array", "items": {"type": "string"}}},
        "required": ["names"], "additionalProperties": False}}}


def validate_plan(decision: dict) -> None:
    route = decision.get("route")
    if route not in {"A", "B", "C", "STOP"}:
        raise ValueError(f"Invalid route: {route}")
    if route == "STOP":
        raise ValueError(f"Agent stopped: {decision.get('reason', '')}")
    required = {"A": ("shot_prompt",), "B": ("frame_prompt", "transition_prompt", "shot_prompt"),
                "C": ("frame_prompt", "shot_prompt")}[route]
    missing = [key for key in required if not isinstance(decision.get(key), str) or not decision[key].strip()]
    if missing:
        raise ValueError(f"Agent route {route} has empty required fields: {', '.join(missing)}; stopped for user")


def chat_once(messages: list, dest: Path, *, choose: object, json_output: bool = False) -> dict:
    if dest.exists():
        return json.loads(dest.read_text())
    cfg = config()
    payload = {"model": cfg.get("SHOT_AGENT_MODEL", cfg["PORTRAIT_QWEN_MODEL"]), "messages": messages,
               "tools": [LOOKUP_TOOL], "tool_choice": choose, "temperature": 0,
               "enable_thinking": False, "max_tokens": 1100}
    if json_output:
        payload["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(cfg["DASHSCOPE_BASE_URL"].rstrip("/") + "/chat/completions",
                                 data=json.dumps(payload).encode(), headers={
                                     "Authorization": "Bearer " + cfg["DASHSCOPE_API_KEY"],
                                     "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace").replace(cfg["DASHSCOPE_API_KEY"], "[REDACTED]")
        write_once(dest.with_suffix(".error.json"), {"http_status": error.code, "response": detail})
        raise RuntimeError(f"Agent API HTTP {error.code}: {detail}") from error
    write_once(dest, result)
    return result


def lookup_characters(names: list[str], catalog: dict, first_index: int = 2) -> list[dict]:
    if len(set(names)) != len(names):
        raise ValueError("Agent requested a duplicate character")
    rows = []
    for index, name in enumerate(names, first_index):
        item = catalog[name]
        path = Path(item["reference"])
        if not path.is_file():
            raise FileNotFoundError(f"Missing reference for {name}: {path}")
        rows.append({"name": name, "description": item["description"], "image_index": index,
                     "reference": str(path.resolve()), "sha256": sha(path)})
    return rows


def plan_shot(context: dict, tail: Path | None, catalog: dict, folder: Path) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    descriptions = {name: item["description"] for name, item in catalog.items()}
    context = {**context, "character_descriptions": descriptions}
    cfg = config()
    inputs = {"system": SYSTEM, "model": cfg.get("SHOT_AGENT_MODEL", cfg["PORTRAIT_QWEN_MODEL"]),
              "context": context, "tail_sha256": sha(tail) if tail else None, "catalog": catalog}
    write_once(folder / "planning_inputs.json", inputs)
    condition = digest(inputs)
    plan_file = folder / "plan.json"
    if plan_file.exists():
        saved = json.loads(plan_file.read_text())
        if saved["input_fingerprint"] != condition:
            raise ValueError("Existing plan has different inputs; choose a new output directory")
        validate_plan(saved["decision"])
        return saved
    content = [{"type": "text", "text": json.dumps(context, ensure_ascii=False)
                + ("\n图1是上一实际尾帧。" if tail else
                   "\n本镜头为硬切，无上一尾帧图。必须选C；首帧默认包含全部调度角色并查询其素材，仅原文明示稍后入场、画外或仅声音者可排除。人物参考从图1编号，建立动作起点。") }]
    if tail:
        content.append({"type": "image_url", "image_url": {"url": image_data_url(tail)}})
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
    response = chat_once(messages, folder / "lookup_call.json",
                         choose={"type": "function", "function": {"name": "lookup_characters"}})
    message = response["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    if len(calls) != 1 or calls[0]["function"]["name"] != "lookup_characters":
        raise ValueError("Expected one lookup_characters call; stop for user review")
    names = json.loads(calls[0]["function"]["arguments"])["names"]
    assets = lookup_characters(names, catalog, 2 if tail else 1)
    write_once(folder / "selected_characters.json", assets)
    messages += [{"role": "assistant", "content": message.get("content"), "tool_calls": calls},
                 {"role": "tool", "tool_call_id": calls[0]["id"],
                  "content": json.dumps(assets, ensure_ascii=False)}]
    references = []
    for asset in assets:
        references += [{"type": "text", "text": f"图{asset['image_index']}仅参考{asset['name']}的人物外观：{asset['description']}"},
                       {"type": "image_url", "image_url": {"url": image_data_url(Path(asset["reference"]))}}]
    messages.append({"role": "user", "content": references + [
        {"type": "text", "text": "据此输出一次完整JSON计划。素材背景不用于场景。"
         + ("图1里的已有角色保留一份。" if tail else "硬切独立建立场景，tail_identity为空。")}]})
    response = chat_once(messages, folder / "prompt_call.json", choose="none", json_output=True)
    decision = json.loads(response["choices"][0]["message"]["content"])
    saved = {"input_fingerprint": condition, "decision": decision, "assets": assets,
             "character_descriptions": descriptions, "has_tail": tail is not None}
    write_once(plan_file, saved)
    validate_plan(decision)
    return saved



def include_inherited_descriptions(plan: dict, inherited: dict[str, str]) -> dict:
    """Retain tail roles without querying or adding their reference images."""
    descriptions = plan["character_descriptions"]
    visible = plan["decision"]["frame_characters"]
    if not isinstance(visible, list) or any(not isinstance(n, str) for n in visible):
        raise ValueError("frame_characters must be a list of role names")
    missing = set(visible) - set(descriptions)
    allowed = inherited if plan.get("has_tail", False) else {}
    unknown = missing - set(allowed)
    if unknown:
        raise ValueError(f"Frame contains roles outside current and inherited cast: {sorted(unknown)}")
    return {**plan, "character_descriptions": {
        **descriptions, **{name: allowed[name] for name in visible if name in missing}}}


def frame_inputs(plan: dict) -> tuple[list[dict], str]:
    """Keep only visible-role references and align their image numbers."""
    visible = set(plan["decision"].get("frame_characters", plan["character_descriptions"]))
    selected = [dict(a) for a in plan["assets"] if a["name"] in visible]
    indices = {}
    for index, asset in enumerate(selected, 2 if plan.get("has_tail", True) else 1):
        indices[asset["image_index"]] = index
        asset["image_index"] = index
    prompt = plan["decision"].get("frame_prompt", "")
    prompt = re.sub(r"图(\d+)", lambda match: "图" + str(indices.get(int(match.group(1)), int(match.group(1)))), prompt)
    return selected, prompt


def compile_prompts(plan: dict, style: str) -> dict:
    decision, assets = plan["decision"], plan["assets"]
    validate_plan(decision)
    descriptions = plan["character_descriptions"]
    people = " ".join(f"{name}: {description}" for name, description in descriptions.items())
    frame_names = decision.get("frame_characters", list(descriptions))
    frame_people = " ".join(f"{name}: {descriptions[name]}" for name in frame_names)
    frame_assets, frame_prompt = frame_inputs(plan)
    labels = " ".join(f"图{a['image_index']}是{a['name']}，只作人物参考。" for a in frame_assets)
    identity = decision["tail_identity"]
    frame_context = ("按当前shot建立全新的场景、机位与动作起始画面。" if not plan.get("has_tail", True) else
                     f"图1是上一实际尾帧，延续原场景与已有角色。{identity}"
                     if decision["route"] == "B"
                     else "图1提供上一画面的环境与画风线索，按当前shot建立场景、机位与所需角色。")
    return {
        "frame": (f"{style} {frame_context} {labels} {frame_people} {frame_prompt} "
                  "单幅连续电影画面，每人仅出现一次；无分屏、上下拼接、多格画面、插图或字幕。"
                  if decision["route"] in {"B", "C"} else ""),
        "transition": (f"{style} {people} {identity} {decision['transition_prompt']}"
                       if decision["route"] == "B" else ""),
        "shot": f"{style} {people} {identity if decision['route'] in {'A', 'B'} else ''} "
                f"{decision['shot_prompt']} 单幅连续电影画面，无分屏、拼图或画风变化。",
    }
