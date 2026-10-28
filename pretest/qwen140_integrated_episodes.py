#!/usr/bin/env python3
"""Input-verified, episode-ordered route/identity/style pilot for four Qwen140 stories.

This is deliberately an isolated experiment. It never mutates the frozen full140 run.
Heavy Qwen and Wan jobs run in separate processes to fit the single-GPU host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image

from qwen140_pilot_qwen import BASE, OLD20, OUT as PILOT, MODEL, sha
from qwen140_pilot_route_agent import SYSTEM as ROUTE_SYSTEM, config as api_config, image_data_url, validate as validate_route

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("QWEN140_INTEGRATED_OUTPUT_ROOT",
                          str(ROOT / "outputs/entitybench_qwen140_integrated_episodes_20261026"))).resolve()
STYLE_FILE = OUT / "style_specs.json"
TRACKED_STYLE_FILE = ROOT / "docs/entitybench_qwen140_integrated_style_specs_20261026.json"
QWEN_PYTHON = "/root/autodl-tmp/qwen-image21-venv/bin/python"
WAN_PYTHON = "/root/autodl-tmp/envs/lightx2v-trial/bin/python"
TAGS = {"run714": "i20_j34", "run484": "i15_j47", "run1404": "i170_j220", "run700": "i25_j50"}
REBUILD_HARDCUT = {"run1404": {"1:1", "3:1", "5:1"}}
PILOT_REUSE = {("run1404", "6:1"): PILOT / "videos/run1404/main/video.mp4"}
ROUTE_MODEL_VERSION = "route_v1_dynamic_tail"
BIND_SYSTEM = """You inspect ONE actual starting image for a video shot, plus labeled character reference images and the official role descriptions. Return only JSON: {"bindings":[{"name":...,"status":"visible|not_visible|uncertain","clue":...}]}. One item per scheduled role, no others. For visible roles, clue is a concise description of THAT PERSON ONLY: left/right position, apparent age, hair, face and clothing. Exclude bedding, furniture, screens, background and unrelated props from identity clues. Use reference images and description to distinguish people. Do not assign a role when evidence is weak; use uncertain instead. Not-visible is correct for an entering role still outside the frame. Do not infer visible roles solely from the shot text. Do not invent identity, movements, or people."""
ENV_SYSTEM = """Inspect the actual previous video tail and the current shot's required new starting state. Return only JSON {"use_previous_tail":true|false,"reason":"..."}. Use the previous tail as an environment reference only when its visible place, props and people are compatible with the new state. Prefer false if old bedding, signs, vehicles, crowd, clothing or people would contradict or contaminate the new composition. Same location name alone does not imply compatibility. The previous tail is the only existing environmental image; false means build the new view from official text plus character references."""
FRAME_AUDIT_SYSTEM = """Check ONE generated keyframe against the official shot requirements, intended stage and episode medium. Return only JSON {"accept":true|false,"issues":["...",...]}. Reject clear wrong person, extra duplicate person, wrong location, wrong required prop, action already completed when this must be its beginning, or a clearly mixed/incorrect rendering style. Do not reject harmless differences in camera color, lighting, or minor uncertain detail. Base judgment on visible pixels and the supplied requirement, not on wishes. Keep issues concise and concrete."""
FRAME_AUDIT_SYSTEM_V2 = """Check ONE generated keyframe against the official shot requirements, intended stage and episode medium. Return only JSON {"accept":true|false,"issues":["...",...]}. For any start, clean_start, fallback_clean_start or hardcut stage, a still moment BEFORE the later shot motion is correct: NEVER reject merely because the later action has not begun. Reject a clear wrong person, extra duplicate person, wrong location, wrong required prop, later action already completed, or clearly mixed/incorrect rendering style. A role explicitly hidden or outside the frame at this stage is allowed to be absent. Do not reject harmless lighting or uncertain detail. Base judgment on visible pixels and the supplied stage, not the final action state. Keep issues concise and concrete."""
POSITIVE_STATE_SYSTEM = """Convert an intended shot starting state and quality-check findings into ONE short affirmative visual instruction. Return only JSON {"state":"..."}. Name each scheduled person exactly once with the correct visible costume, held prop, and spatial relation. Describe only what should be present and happening at this instant, before the later shot action. Do not quote the failed image, do not mention incorrect assignments, do not use negation, and do not complete the later action. Preserve requirements from the official shot text. When a person or prop assignment is explicit in the official shot, state it directly."""
START_STATE_SYSTEM = """Verify and correct one proposed starting state for a new video shot. Return only JSON {"state":"..."}. The actual action-description sentence and scheduled place determine the location, people, camera and timing. Entity-definition text may describe a prop or character but must not move the shot to an unrelated place. The state is the visible instant BEFORE later motion or interaction finishes. State correct visible character costumes and assigned props. Use only supported facts, keep the result concise, and do not include an explanation."""
VISIBLE_REF_SYSTEM = """Given one keyframe's intended instant and the scheduled character names, return only JSON {"visible_names":[...],"hidden_names":[...]}. Put every scheduled name in exactly one list. Visible means that this character's body or face should already appear in this still image. A person hidden behind a curtain, outside the frame, or entering only during the later video action belongs in hidden_names. Use the intended instant, not the full action's final state. Do not invent any names."""
VISIBLE_FRAME_SYSTEM = """Rewrite one intended keyframe state as a short affirmative image prompt containing only the people and objects VISIBLE at this instant. Return only JSON {"state":"..."}. Include every supplied visible person by name, costume and position. Do not mention any hidden or offscreen name even to say that person is absent; describing them can make an image model draw them. Describe a closed curtain or doorway by its visible state alone. Keep later motion unperformed and do not invent anything."""


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_once(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        assert path.read_text() == blob, f"Immutable record changed: {path}"
    else:
        path.write_text(blob)


def run_worker(python: str, action: str, *args: str) -> None:
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    if action == "worker-video":
        env.update(DTYPE="BF16", SENSITIVE_LAYER_DTYPE="None")
    subprocess.run([python, __file__, action, *args], env=env, check=True)


def source() -> tuple[dict, dict, dict, dict]:
    manifest = json.loads((BASE / "manifest.json").read_text())
    prep = json.loads((BASE / "preparation_plan.json").read_text())
    old_prep = json.loads((OLD20 / "preparation_plan.json").read_text())
    jobs = json.loads((BASE / "generation_report.json").read_text())
    return manifest, prep, old_prep, {(r["episode_id"], r["shot_key"]): r for r in jobs["jobs"]}


def episode_for(run: str, manifest: dict) -> dict:
    tag = TAGS[run]
    hits = [e for e in manifest["episodes"] if e["run"] == run and tag in e["episode_id"]]
    assert len(hits) == 1
    return hits[0]


def style_for(run: str) -> dict:
    if not STYLE_FILE.exists():
        STYLE_FILE.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(TRACKED_STYLE_FILE, STYLE_FILE)
    data = json.loads(STYLE_FILE.read_text())
    return data["episodes"][f"{run}__{TAGS[run]}"]


def descriptions(ep: dict) -> dict[str, str]:
    story = json.loads(Path(ep["episode_path"]).read_text())
    return {v["name"]: v["description"] for v in story["entity_descriptions"].values()}


def role_ref(run: str, ep: dict, name: str, prep: dict, old_prep: dict) -> Path | None:
    if (run, name) in {("run700", "Nora"), ("run1404", "Chloe")}:
        return OUT / "references" / run / f"{name.lower()}_photoreal.png"
    special = {("run714", "Leo"): PILOT / "keyframes/run714/frame_identity_anchor_v2.png",  # replaced below by actual earlier crop
               ("run484", "Chloe"): PILOT / "references/run484/chloe.png",
               ("run484", "Leo"): PILOT / "references/run484/leo.png",
               ("run1404", "Kenji"): PILOT / "references/run1404/kenji.png"}
    if (run, name) == ("run714", "Leo"):
        from qwen140_pilot_qwen import earlier_role_crop
        path = earlier_role_crop("run714", "Leo")
        assert path.is_file()
        return path
    if (run, name) in special:
        path = special[run, name]
        assert path.is_file(), path
        return path
    for plan in (prep, old_prep):
        rows = [r for r in plan["roles"] if r["episode_id"] == ep["episode_id"] and r["character"] == name]
        if rows:
            path = Path(rows[0]["reference"])
            if path.is_file():
                return path
    if (run, name) == ("run714", "Alex"):
        return None  # indistinct silhouette: text constraint, no invented face asset
    return OUT / "references" / run / f"{name.lower()}.png"


def audited_single_person_ref(run: str, ep: dict, shot: dict, stage: str,
                              attempt: int, name: str, default: Path | None) -> Path | None:
    clean_crop = {
        ("run700", "Arthur"): "arthur_v2.png",
        ("run700", "Walter"): "walter_v2.png",
        ("run714", "Nadia"): "nadia.png",
        ("run714", "Owen"): "owen.png",
        ("run484", "Nora"): "nora.png",
    }
    if (run, name) in clean_crop and not (run == "run700" and shot["shot_key"] == "1:1"):
        clean = OUT / "references" / "clean" / run / clean_crop[run, name]
        assert clean.is_file(), clean
        return clean
    # The supplied run700 Leo asset is a six-panel contact sheet with other people.
    # Use the already accepted solo Leo keyframe only for new work, keeping old
    # generation tasks immutable and traceable to their original inputs.
    if ((run, shot["shot_key"], stage, name) == ("run700", "3:3", "fallback_start", "Leo") and attempt >= 4
            or (run, shot["shot_key"], stage, name) == ("run700", "3:3", "fallback_clean_start", "Leo")):
        clean = OUT / "episodes" / ep["episode_id"] / "shot_1_2" / "start_attempt0.png"
        assert clean.is_file()
        return clean
    return default


def api_json(system: str, text: str, images: list[tuple[str, Path]], cache: Path, max_tokens: int = 900) -> dict:
    cfg = api_config()
    condition = {"model": cfg["PORTRAIT_QWEN_MODEL"], "system": system, "text": text,
                 "images": [{"label": label, "sha256": sha(path)} for label, path in images], "max_tokens": max_tokens}
    fp = digest(condition)
    if cache.exists():
        row = json.loads(cache.read_text())
        assert row["fingerprint"] == fp
        return row["answer"]
    content: list[dict] = [{"type": "text", "text": text}]
    for label, path in images:
        content += [{"type": "text", "text": label}, {"type": "image_url", "image_url": {"url": image_data_url(path)}}]
    payload = {"model": cfg["PORTRAIT_QWEN_MODEL"], "messages": [
        {"role": "system", "content": system}, {"role": "user", "content": content}],
        "temperature": 0, "enable_thinking": False, "max_tokens": max_tokens,
        "response_format": {"type": "json_object"}}
    req = urllib.request.Request(cfg["DASHSCOPE_BASE_URL"].rstrip("/") + "/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + cfg["DASHSCOPE_API_KEY"]})
    with urllib.request.urlopen(req, timeout=180) as response:
        result = json.load(response)
    answer = json.loads(result["choices"][0]["message"]["content"])
    write_once(cache, {"fingerprint": fp, "condition": condition, "answer": answer,
                       "returned_model": result.get("model"), "usage": result.get("usage")})
    return answer


def route(run: str, ep: dict, previous: dict, shot: dict, parent_tail: Path, desc: dict, shot_dir: Path) -> dict:
    item = {"episode_id": ep["episode_id"], "previous_shot": {"shot_key": previous["shot_key"],
            "prompt": previous["prompt"], "scheduled_characters": previous["entity_schedule"].get("characters", [])},
            "current_shot": {"shot_key": shot["shot_key"], "original_cut": False,
            "prompt": shot["prompt"], "action_description": shot["action_description"],
            "entity_schedule": shot["entity_schedule"]},
            "character_descriptions": {n: desc[n] for n in shot["entity_schedule"].get("characters", [])},
            "episode_style": style_for(run), "actual_parent_tail_sha256": sha(parent_tail),
            "version": ROUTE_MODEL_VERSION}
    answer = api_json(ROUTE_SYSTEM, "Route this current shot from the actual preceding video tail.\n" + json.dumps(item, ensure_ascii=False),
                      [("ACTUAL previous video tail", parent_tail)], shot_dir / "route.json", 900)
    validate_route(answer)
    return answer


def bind(run: str, ep: dict, shot: dict, image: Path, desc: dict,
         prep: dict, old_prep: dict, cache: Path) -> dict:
    names = shot["entity_schedule"].get("characters", [])
    refs = [(n, audited_single_person_ref(run, ep, shot,
                                          "fallback_clean_start" if cache.name == "binding_fallback_clean.json"
                                          else "binding", 4 if cache.name == "binding_fallback.json" else 0,
                                          n, role_ref(run, ep, n, prep, old_prep)))
            for n in names]
    info = {"episode_id": ep["episode_id"], "shot_key": shot["shot_key"],
            "characters": {n: desc[n] for n in names}, "episode_style": style_for(run)}
    images: list[tuple[str, Path]] = [("ACTUAL video start image", image)]
    images.extend((f"Reference for {n}", p) for n, p in refs if p is not None and p.is_file())
    answer = api_json(BIND_SYSTEM, json.dumps(info, ensure_ascii=False), images, cache, 1000)
    rows = answer.get("bindings")
    assert isinstance(rows, list) and {r.get("name") for r in rows} == set(names), answer
    assert len(rows) == len(names)
    for r in rows:
        assert r.get("status") in {"visible", "not_visible", "uncertain"}, r
        assert isinstance(r.get("clue"), str), r
    return answer


def use_environment_tail(run: str, ep: dict, previous: dict, shot: dict, state: str,
                         tail: Path, cache: Path) -> bool:
    old_places = set(previous["entity_schedule"].get("places", []))
    new_places = set(shot["entity_schedule"].get("places", []))
    if not old_places.intersection(new_places):
        write_once(cache, {"decision": {"use_previous_tail": False,
                                         "reason": "No common scheduled place"}, "tail_sha256": sha(tail)})
        return False
    text = json.dumps({"episode_id": ep["episode_id"], "previous_shot": previous["prompt"],
                       "current_shot": shot["prompt"], "new_start_state": state,
                       "episode_style": style_for(run)}, ensure_ascii=False)
    answer = api_json(ENV_SYSTEM, text, [("ACTUAL previous video tail", tail)], cache, 500)
    assert type(answer.get("use_previous_tail")) is bool and isinstance(answer.get("reason"), str)
    return answer["use_previous_tail"]


def audit_frame(run: str, ep: dict, shot: dict, stage: str, state: str,
                image: Path, cache: Path) -> dict:
    text = json.dumps({"episode_id": ep["episode_id"], "shot_key": shot["shot_key"],
                       "stage": stage, "required_state": state,
                       "official_action": shot["action_description"],
                       "scheduled_characters": shot["entity_schedule"].get("characters", []),
                       "scheduled_objects": shot["entity_schedule"].get("objects", []),
                       "episode_style": style_for(run)}, ensure_ascii=False)
    system = FRAME_AUDIT_SYSTEM if cache.exists() else FRAME_AUDIT_SYSTEM_V2
    answer = api_json(system, text, [("Generated keyframe to inspect", image)], cache, 600)
    assert type(answer.get("accept")) is bool and isinstance(answer.get("issues"), list), answer
    return answer


def positive_state(run: str, ep: dict, shot: dict, intended: str,
                   issues: list[str], cache: Path) -> str:
    data = {"episode_id": ep["episode_id"], "shot_key": shot["shot_key"],
            "intended_start": intended, "official_prompt": shot["prompt"],
            "scheduled_characters": shot["entity_schedule"].get("characters", []),
            "issues_to_correct": issues, "episode_style": style_for(run)}
    answer = api_json(POSITIVE_STATE_SYSTEM, json.dumps(data, ensure_ascii=False), [], cache, 700)
    state = answer.get("state")
    assert isinstance(state, str) and len(state.strip()) > 25, answer
    return state.strip()


def verified_start_state(run: str, ep: dict, shot: dict, proposed: str,
                         shot_dir: Path) -> str:
    policy = shot_dir / "start_state_policy.json"
    if not policy.exists():
        write_once(policy, {"mode": "legacy" if (shot_dir / "start_attempt0.task.json").exists()
                    else "verified"})
    if json.loads(policy.read_text())["mode"] == "legacy":
        return proposed
    info = {"episode_id": ep["episode_id"], "shot_key": shot["shot_key"],
            "proposed_state": proposed, "actual_action_description": shot["action_description"],
            "scheduled_places": shot["entity_schedule"].get("places", []),
            "scheduled_characters": shot["entity_schedule"].get("characters", []),
            "full_prompt_for_entity_descriptions": shot["prompt"],
            "episode_style": style_for(run)}
    answer = api_json(START_STATE_SYSTEM, json.dumps(info, ensure_ascii=False), [],
                      shot_dir / "start_state_verified.json", 600)
    state = answer.get("state")
    assert isinstance(state, str) and len(state.strip()) > 25, answer
    return state.strip()


def visible_keyframe_names(ep: dict, shot: dict, state: str, cache: Path) -> set[str]:
    names = shot["entity_schedule"].get("characters", [])
    info = {"episode_id": ep["episode_id"], "shot_key": shot["shot_key"],
            "intended_instant": state, "scheduled_characters": names,
            "later_action": shot["action_description"]}
    answer = api_json(VISIBLE_REF_SYSTEM, json.dumps(info, ensure_ascii=False), [], cache, 500)
    visible, hidden = answer.get("visible_names"), answer.get("hidden_names")
    assert isinstance(visible, list) and isinstance(hidden, list)
    assert len(visible) + len(hidden) == len(names) and set(visible) | set(hidden) == set(names)
    assert not set(visible) & set(hidden)
    return set(visible)


def visible_only_state(ep: dict, shot: dict, state: str, visible: set[str], cache: Path) -> str:
    info = {"episode_id": ep["episode_id"], "shot_key": shot["shot_key"],
            "intended_state": state, "visible_names": sorted(visible),
            "hidden_names": sorted(set(shot["entity_schedule"].get("characters", [])) - visible)}
    answer = api_json(VISIBLE_FRAME_SYSTEM, json.dumps(info, ensure_ascii=False), [], cache, 500)
    visual = answer.get("state")
    assert isinstance(visual, str) and len(visual.strip()) > 25, answer
    for hidden in info["hidden_names"]:
        assert hidden.lower() not in visual.lower(), (hidden, visual)
    return visual.strip()


def accepted_clean_frame(run: str, ep: dict, shot: dict, intended: str, issues: list[str],
                         desc: dict, prep: dict, old_prep: dict, shot_dir: Path,
                         stage: str = "fallback_clean_start") -> Path:
    state = positive_state(run, ep, shot, intended, issues, shot_dir / "fallback_clean_state0.json")
    visibility_policy = shot_dir / f"{stage}_visibility_policy.json"
    if not visibility_policy.exists():
        write_once(visibility_policy, {"apply_from_attempt": 1 if (shot_dir / f"{stage}_attempt0.task.json").exists()
                    else 0})
    apply_from = json.loads(visibility_policy.read_text())["apply_from_attempt"]
    visual_policy = shot_dir / f"{stage}_visible_prompt_policy.json"
    if not visual_policy.exists():
        existing = [i for i in range(4) if (shot_dir / f"{stage}_attempt{i}.task.json").exists()]
        write_once(visual_policy, {"apply_from_attempt": max(existing) + 1 if existing else 0})
    visual_from = json.loads(visual_policy.read_text())["apply_from_attempt"]
    for attempt in range(4):
        dest = shot_dir / f"{stage}_attempt{attempt}.png"
        skip_refs: set[str] = set()
        frame_state = state
        if attempt >= apply_from:
            visible = visible_keyframe_names(ep, shot, state,
                                             shot_dir / f"{stage}_visible{attempt}.json")
            skip_refs = set(shot["entity_schedule"].get("characters", [])) - visible
            if skip_refs and attempt >= visual_from:
                frame_state = visible_only_state(ep, shot, state, visible,
                                                 shot_dir / f"{stage}_visual_state{attempt}.json")
        frame = frame_spec(run, ep, shot, stage, frame_state, None, desc, prep, old_prep,
                           dest, skip_refs=skip_refs, strict_pre_action=True)
        verdict = audit_frame(run, ep, shot, stage, state, frame,
                              shot_dir / f"{stage}_audit{attempt}.json")
        if verdict["accept"]:
            return frame
        print("rejected clean keyframe", run, shot["shot_key"], attempt,
              verdict["issues"], flush=True)
        if attempt < 3:
            state = positive_state(run, ep, shot, state, verdict["issues"],
                                   shot_dir / f"fallback_clean_state{attempt+1}.json")
    raise RuntimeError(f"No acceptable clean fallback frame for {run} {shot['shot_key']}")


def prompt(run: str, shot: dict, desc: dict, binding: dict, action: str) -> str:
    b = {x["name"]: x for x in binding["bindings"]}
    clauses = []
    for n in shot["entity_schedule"].get("characters", []):
        x = b[n]
        clue = x["clue"].strip() if x["status"] == "visible" else ""
        clauses.append(f"{n} is {clue if clue else desc[n]}")
    style = style_for(run)
    return (f"{style['medium']}. {style['character_rendering']}. "
            + ("Character identity: " + "; ".join(clauses) + ". " if clauses else "")
            + f"Required shot action: {action} "
            + f"Setting: {style['environment_rendering']}. Preserve the actual starting image where the shot is continuous. "
            + "Keep names, ages, clothes and positions unambiguous; coherent natural motion; no duplicate people or style changes.")


def frame_spec(run: str, ep: dict, shot: dict, stage: str, state: str, tail: Path | None,
               desc: dict, prep: dict, old_prep: dict, dest: Path,
               extra_constraints: str = "", skip_refs: set[str] | None = None,
               strict_pre_action: bool = False) -> Path:
    names = shot["entity_schedule"].get("characters", [])
    attempt = int(dest.stem.rsplit("attempt", 1)[1])
    refs = [(n, audited_single_person_ref(run, ep, shot, stage, attempt, n,
                                          role_ref(run, ep, n, prep, old_prep)))
            for n in names if n not in (skip_refs or set())]
    paths = [(f"Reference for {n}", p) for n, p in refs if p is not None]
    assert all(p.is_file() for _, p in paths)
    if tail is not None:
        paths.append(("Previous actual video tail: environment and continuing people only", tail))
    assert len(paths) <= 6, (run, shot["shot_key"], len(paths))
    label_text = " ".join(f"Image {i} is {label}." for i, (label, _) in enumerate(paths, 1))
    role_text = "; ".join(f"{n}: {desc[n]}" for n in names
                          if not strict_pre_action or n not in (skip_refs or set()))
    sty = style_for(run)
    frame_prompt = (f"{label_text} Create ONE new {sty['medium']} frame for shot {shot['shot_key']}. "
                    f"Characters: {role_text}. Desired frame state: {state}. "
                    +
                    ("Keep later shot action entirely unperformed in this target frame. " if strict_pre_action
                     else f"Official shot action for context: {shot['action_description']}. ")
                    +
                    f"{sty['character_rendering']}. {sty['environment_rendering']}. "
                    "Preserve correct age, costume and identity from each labeled reference. "
                    "The previous tail, when present, supplies only compatible environment and prior people; "
                    "recompose for the current shot. No duplicates, collage, caption or mixed rendering style. "
                    "Do not complete an action scheduled to happen after this frame. "
                    + extra_constraints)
    condition = {"episode_id": ep["episode_id"], "shot_key": shot["shot_key"], "stage": stage,
                 "style_sha256": sha(STYLE_FILE), "prompt": frame_prompt,
                 "images": [{"label": label, "path": str(path.resolve()), "sha256": sha(path)} for label, path in paths],
                 "seed": int(shot["seed"]) + (9_000_000 if stage == "hardcut" else 10_000_000 if stage == "start" else 11_000_000),
                 "steps": 20, "width": 1024, "height": 576, "model_revision": "790c92633540aa0cb11d9abf19eb46d861714758"}
    task = dest.with_suffix(".task.json")
    write_once(task, condition)
    rec = dest.with_suffix(".json")
    if rec.exists():
        row = json.loads(rec.read_text())
        assert row["condition_sha256"] == digest(condition) and sha(dest) == row["sha256"]
    else:
        run_worker(QWEN_PYTHON, "worker-frame", "--task", str(task), "--output", str(dest))
    return dest


def accepted_frame(run: str, ep: dict, shot: dict, stage: str, state: str, tail: Path | None,
                   desc: dict, prep: dict, old_prep: dict, shot_dir: Path) -> Path:
    issues: list[str] = []
    strict_from_start = False
    if stage == "transition_target":
        policy_path = shot_dir / "transition_target_prompt_policy.json"
        if not policy_path.exists():
            # Existing attempts retain their exact input contract on resume.
            legacy = (shot_dir / "transition_target_attempt0.task.json").exists()
            write_once(policy_path, {"strict_from_start": not legacy})
        strict_from_start = json.loads(policy_path.read_text())["strict_from_start"]
    max_attempts = 6 if stage in {"transition_target", "fallback_start"} else 2
    for attempt in range(max_attempts):
        dest = shot_dir / f"{stage}_attempt{attempt}.png"
        extra = "" if not issues else "CRITICAL corrections from a rejected first frame: " + "; ".join(issues) + ". "
        skip_refs: set[str] = set()
        if stage == "transition_target" and tail is not None and (attempt >= 2 or strict_from_start):
            prior = bind(run, ep, shot, tail, desc, prep, old_prep,
                         shot_dir / "binding_transition_prior_tail.json")
            skip_refs = {r["name"] for r in prior["bindings"] if r["status"] == "visible"}
            extra += ("Exactly one copy of each scheduled person. People already visible in the previous "
                      "tail stay the SAME people and must NOT be redrawn from separate reference images. "
                      "Only introduce the absent characters; keep the later action visibly unstarted. ")
        if stage == "transition_target" and attempt >= 4:
            if attempt == 5:
                tail = None  # Last bounded retry: old image has repeatedly duplicated a person.
                skip_refs = set()
                extra += "Create the two-person target from character references and shot text only. "
            else:
                extra += ("The prior tail already contains the existing adult. Add only the absent boy. "
                          "Do not draw the adult again. The drawer must remain closed. ")
        if stage == "fallback_start" and attempt >= 2:
            extra += ("Render exactly the scheduled people once each. Keep their named clothes and held "
                      "props assigned to the correct person. The main action has not started; leave "
                      "drawers, doors, and other action targets in their starting state. ")
        image = frame_spec(run, ep, shot, stage, state, tail, desc, prep, old_prep,
                           dest, extra, skip_refs,
                           strict_pre_action=strict_from_start or attempt >= 4
                           or (stage == "fallback_start" and attempt >= 2))
        verdict = audit_frame(run, ep, shot, stage, state, image, shot_dir / f"{stage}_audit{attempt}.json")
        if verdict["accept"]:
            return image
        issues = [str(x) for x in verdict["issues"]]
        print("rejected keyframe", run, shot["shot_key"], stage, attempt, issues, flush=True)
    raise RuntimeError(f"No acceptable keyframe for {run} {shot['shot_key']} {stage}: {issues}")


def audited_or_clean_frame(run: str, ep: dict, shot: dict, stage: str, state: str,
                           tail: Path | None, desc: dict, prep: dict, old_prep: dict,
                           shot_dir: Path) -> tuple[Path, bool]:
    try:
        return accepted_frame(run, ep, shot, stage, state, tail, desc, prep, old_prep, shot_dir), False
    except RuntimeError as error:
        failure = json.loads((shot_dir / f"{stage}_audit1.json").read_text())["answer"]
        write_once(shot_dir / f"{stage}_failure.json",
                   {"error": str(error), "issues": failure["issues"],
                    "fallback": "positive_state_single_frame"})
        clean_stage = "clean_hardcut" if stage == "hardcut" else "clean_start"
        clean = accepted_clean_frame(run, ep, shot, state, failure["issues"],
                                     desc, prep, old_prep, shot_dir, clean_stage)
        return clean, True


def video_spec(ep: dict, shot: dict, segment: str, input_image: Path, video_prompt: str,
               frames: int, output_dir: Path, last_frame: Path | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    task = {"episode_id": ep["episode_id"], "shot_key": shot["shot_key"], "segment": segment,
            "input_path": str(input_image.resolve()), "input_sha256": sha(input_image),
            "last_frame_path": str(last_frame.resolve()) if last_frame else None,
            "last_frame_sha256": sha(last_frame) if last_frame else None,
            "prompt": video_prompt, "frames": frames, "fps": 16, "size": [480, 832], "steps": 4,
            "seed": int(shot["seed"]) + {"direct": 4_000_000, "new": 5_000_000,
                                        "transition": 6_000_000, "main": 7_000_000,
                                        "hardcut": 8_000_000}[segment],
            "model_revision": json.loads((BASE / "manifest.json").read_text())["model_revision"],
            "style_sha256": sha(STYLE_FILE)}
    write_once(output_dir / "task.json", task)
    rec_path = output_dir / "generation.json"
    if rec_path.exists():
        row = json.loads(rec_path.read_text())
        assert row["condition_sha256"] == digest(task)
        assert sha(output_dir / "video.mp4") == row["video_sha256"]
        assert sha(output_dir / "last.png") == row["last_sha256"]
    else:
        run_worker(WAN_PYTHON, "worker-video", "--task", str(output_dir / "task.json"), "--output", str(output_dir))
        row = json.loads(rec_path.read_text())
    return {"video": output_dir / "video.mp4", "last": output_dir / "last.png", "sha256": row["video_sha256"]}


def baseline_media(ep: dict, shot: dict, jobs: dict) -> dict:
    j = jobs[ep["episode_id"], shot["shot_key"]]
    assert j["status"] == "generated"
    video, tail = Path(j["output_video"]), Path(j["last_frame"])
    assert sha(video) == j["video_sha256"] and sha(tail) == j["last_frame_sha256"]
    return {"video": video, "last": tail, "sha256": j["video_sha256"]}


def pilot_media(run: str, ep: dict, shot: dict, parent: dict | None) -> dict | None:
    path = PILOT_REUSE.get((run, shot["shot_key"]))
    if path is None:
        return None
    rec = json.loads((path.parent / "generation.json").read_text())
    assert sha(path) == rec["video_sha256"]
    if run == "run484":
        first = json.loads((PILOT / "videos/run484/transition/generation.json").read_text())
        required = first["inputs"]["parent_tail_sha256"]
    elif run == "run1404":
        required = None  # hard-cut kitchen first frame has no parent visual condition
    else:
        required = rec["inputs"]["parent_tail_sha256"]
    if required is not None:
        assert parent is not None and sha(parent["last"]) == required, f"Pilot candidate invalidated at {run} {shot['shot_key']}"
    tail = path.parent / "last.png"
    assert tail.is_file() and sha(tail) == rec["last_sha256"]
    return {"video": path, "last": tail, "sha256": rec["video_sha256"]}


def link_official(ep: dict, shot: dict, media: dict) -> Path:
    scene, local = map(int, shot["shot_key"].split(":"))
    link = OUT / "official_results" / ep["episode_id"] / f"{scene:03d}_{local:03d}.mp4"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        assert link.resolve() == media["video"].resolve()
    else:
        assert not link.exists()
        link.symlink_to(media["video"].resolve())
    return link


def stitch(ep: dict, shot: dict, first: dict, second: dict, dest: Path) -> dict:
    from run_lightx2v_a14b_4step_qwen140 import decode_video
    dest.mkdir(parents=True, exist_ok=True)
    inputs = {"episode_id": ep["episode_id"], "shot_key": shot["shot_key"], "first_video_sha256": first["sha256"],
              "second_video_sha256": second["sha256"], "join": "41 first frames + second frames 1..40", "fps": 16}
    rec = dest / "generation.json"
    video = dest / "video.mp4"
    if rec.exists():
        old = json.loads(rec.read_text())
        assert old["condition_sha256"] == digest(inputs) and sha(video) == old["video_sha256"]
        assert sha(dest / "last.png") == old["last_sha256"]
    else:
        temp = dest / "video.tmp.mp4"
        assert not video.exists() and not temp.exists()
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(first["video"]),
                        "-i", str(second["video"]), "-filter_complex",
                        "[0:v]setpts=PTS-STARTPTS[v0];[1:v]trim=start_frame=1,setpts=PTS-STARTPTS[v1];[v0][v1]concat=n=2:v=1:a=0[out]",
                        "-map", "[out]", "-an", "-r", "16", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p",
                        str(temp)], check=True)
        decoded = decode_video(temp, dest / "first.png", dest / "last.png", 81, 16)
        temp.replace(video)
        write_once(rec, {"condition_sha256": digest(inputs), "inputs": inputs, "decoded": decoded,
                         "video_sha256": sha(video), "last_sha256": sha(dest / "last.png")})
    row = json.loads(rec.read_text())
    return {"video": video, "last": dest / "last.png", "sha256": row["video_sha256"]}


def run_episode(run: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "pretest/qwen140_clean_reference_assets.py"),
                    "--run", run], check=True)
    manifest, prep, old_prep, jobs = source()
    ep = episode_for(run, manifest)
    desc = descriptions(ep)
    style = style_for(run)
    assert style["medium"] == "photorealistic live-action film"
    root = OUT / "episodes" / ep["episode_id"]
    root.mkdir(parents=True, exist_ok=True)
    missing = []
    for name in sorted({n for s in ep["shots"] for n in s["entity_schedule"].get("characters", [])}):
        ref = role_ref(run, ep, name, prep, old_prep)
        if ref is not None and not ref.exists():
            missing.append({"name": name, "description": desc[name], "path": str(ref.resolve()),
                            "seed": ep["base_seed"] + 12_000_000 + len(missing), "style": style})
    if missing:
        task = root / "missing_refs.task.json"
        write_once(task, {"episode_id": ep["episode_id"], "items": missing, "style_sha256": sha(STYLE_FILE)})
        run_worker(QWEN_PYTHON, "worker-refs", "--task", str(task))
    parent = None
    previous = None
    completed = []
    for shot in ep["shots"]:
        sk = shot["shot_key"]
        shot_dir = root / f"shot_{sk.replace(':', '_')}"
        shot_dir.mkdir(parents=True, exist_ok=True)
        route_answer = None
        source_kind = None
        if shot["cut"]:
            if sk in REBUILD_HARDCUT.get(run, set()):
                state = "At the beginning of the official shot, before its described action is completed."
                start, cleaned = audited_or_clean_frame(run, ep, shot, "hardcut", state, None,
                                                        desc, prep, old_prep, shot_dir)
                binding = bind(run, ep, shot, start, desc, prep, old_prep, shot_dir / "binding_hardcut.json")
                media = video_spec(ep, shot, "hardcut", start,
                                   prompt(run, shot, desc, binding, shot["action_description"]),
                                   81, shot_dir / "video")
                source_kind = "regenerated_hardcut_clean_fallback" if cleaned else "regenerated_hardcut"
            else:
                media = pilot_media(run, ep, shot, parent) or baseline_media(ep, shot, jobs)
                source_kind = "reused_pilot_hardcut" if (run, sk) in PILOT_REUSE else "verified_baseline_hardcut"
        else:
            assert parent is not None and previous is not None
            route_answer = route(run, ep, previous, shot, parent["last"], desc, shot_dir)
            chosen = route_answer["route"]
            if chosen == "needs_review":
                raise RuntimeError(f"Agent needs review: {run} {sk}: {route_answer['reason']}")
            reused = pilot_media(run, ep, shot, parent)
            if reused is not None:
                assert chosen == ("transition_then_continue" if run == "run484" else "new_shot"), (run, sk, chosen)
                media = reused
                source_kind = "reused_verified_pilot"
            elif chosen == "direct_continue":
                binding = bind(run, ep, shot, parent["last"], desc, prep, old_prep, shot_dir / "binding_direct.json")
                media = video_spec(ep, shot, "direct", parent["last"],
                                   prompt(run, shot, desc, binding, shot["action_description"]),
                                   81, shot_dir / "video")
                source_kind = chosen
            else:
                use_tail = chosen == "transition_then_continue"
                if chosen == "new_shot":
                    use_tail = use_environment_tail(run, ep, previous, shot, route_answer["new_start_state"],
                                                    parent["last"], shot_dir / "environment_decision.json")
                    start_state = verified_start_state(run, ep, shot, route_answer["new_start_state"],
                                                       shot_dir)
                    start, cleaned = audited_or_clean_frame(run, ep, shot, "start", start_state,
                                                            parent["last"] if use_tail else None,
                                                            desc, prep, old_prep, shot_dir)
                    binding = bind(run, ep, shot, start, desc, prep, old_prep, shot_dir / "binding_new.json")
                    media = video_spec(ep, shot, "new", start,
                                       prompt(run, shot, desc, binding, shot["action_description"]),
                                       81, shot_dir / "video")
                    source_kind = "new_shot_clean_fallback" if cleaned else chosen
                else:
                    assert chosen == "transition_then_continue"
                    transition = route_answer["transition"]
                    target = accepted_frame(run, ep, shot, "transition_target", transition["target_state"],
                                            parent["last"], desc, prep, old_prep, shot_dir)
                    target_binding = bind(run, ep, shot, target, desc, prep, old_prep, shot_dir / "binding_transition_target.json")
                    entrance = video_spec(ep, shot, "transition", parent["last"],
                                          prompt(run, shot, desc, target_binding, transition["action"]),
                                          41, shot_dir / "transition", target)
                    actual_tail_requirement = (transition["target_state"]
                                               + " Preserve the named person's required clothes and held props "
                                                 "from this official shot: " + shot["action_description"])
                    actual_tail_check = audit_frame(run, ep, shot, "transition_actual_tail",
                                                    actual_tail_requirement, entrance["last"],
                                                    shot_dir / "transition_actual_tail_audit.json")
                    if not actual_tail_check["accept"]:
                        write_once(shot_dir / "transition_failure.json",
                                   {"route": chosen, "actual_tail_sha256": sha(entrance["last"]),
                                    "issues": actual_tail_check["issues"],
                                    "fallback": "new_shot_from_role_references"})
                        try:
                            fallback_start = accepted_clean_frame(run, ep, shot,
                                                                  transition["target_state"],
                                                                  actual_tail_check["issues"],
                                                                  desc, prep, old_prep, shot_dir)
                        except RuntimeError as error:
                            write_once(shot_dir / "fallback_keyframe_failure.json",
                                       {"error": str(error), "fallback": "direct_from_actual_parent_tail"})
                            fallback_binding = bind(run, ep, shot, parent["last"], desc, prep, old_prep,
                                                    shot_dir / "binding_fallback_direct.json")
                            fallback_action = ("First introduce any required person not visible in the starting "
                                               "image, then execute the official action: "
                                               + shot["action_description"])
                            media = video_spec(ep, shot, "direct", parent["last"],
                                               prompt(run, shot, desc, fallback_binding, fallback_action),
                                               81, shot_dir / "fallback_direct_video")
                            source_kind = "transition_rejected_fallback_direct"
                        else:
                            fallback_binding = bind(run, ep, shot, fallback_start, desc, prep, old_prep,
                                                    shot_dir / "binding_fallback_clean.json")
                            media = video_spec(ep, shot, "new", fallback_start,
                                               prompt(run, shot, desc, fallback_binding,
                                                      shot["action_description"]),
                                               81, shot_dir / "fallback_clean_video")
                            source_kind = "transition_rejected_fallback_clean_new_shot"
                    else:
                        main_binding = bind(run, ep, shot, entrance["last"], desc, prep, old_prep,
                                            shot_dir / "binding_transition_actual_tail.json")
                        main = video_spec(ep, shot, "main", entrance["last"],
                                          prompt(run, shot, desc, main_binding, transition["main_action"]),
                                          41, shot_dir / "main")
                        media = stitch(ep, shot, entrance, main, shot_dir / "combined")
                        source_kind = chosen
        link = link_official(ep, shot, media)
        original = baseline_media(ep, shot, jobs)
        row = {"episode_id": ep["episode_id"], "shot_key": sk, "cut": shot["cut"],
               "source_kind": source_kind, "route": route_answer, "style_sha256": sha(STYLE_FILE),
               "video": str(media["video"].resolve()), "video_sha256": media["sha256"],
               "last_frame": str(media["last"].resolve()), "last_sha256": sha(media["last"]),
               "parent_video_sha256": parent["sha256"] if parent else None,
               "parent_tail_sha256": sha(parent["last"]) if parent else None,
               "baseline_video_sha256": original["sha256"], "official_link": str(link.resolve())}
        write_once(shot_dir / "result.json", row)
        completed.append(row)
        parent, previous = media, shot
        print("completed", run, sk, source_kind, flush=True)
    write_once(root / "episode_result.json", {"episode_id": ep["episode_id"], "run": run,
                "style_sha256": sha(STYLE_FILE), "shots": completed})
    print("episode complete", ep["episode_id"], len(completed), flush=True)


def worker_refs(task_path: Path) -> None:
    import torch
    from qwen140_pilot_qwen import make_pipe
    task = json.loads(task_path.read_text())
    pipe = make_pipe("refs")
    for item in task["items"]:
        dest = Path(item["path"])
        rec = dest.with_suffix(".json")
        prompt_text = (f"{item['style']['medium']}. One person only, one continuous waist-up portrait "
                       f"against a neutral background. Character {item['name']}: {item['description']}. "
                       f"{item['style']['character_rendering']}. No collage, inset, text or illustration.")
        condition = {"task": task["episode_id"], "name": item["name"], "description": item["description"],
                     "prompt": prompt_text, "seed": item["seed"], "steps": 40, "style_sha256": task["style_sha256"]}
        if rec.exists():
            old = json.loads(rec.read_text())
            assert old["condition_sha256"] == digest(condition) and sha(dest) == old["sha256"]
            continue
        assert not dest.exists()
        dest.parent.mkdir(parents=True, exist_ok=True)
        image = pipe(prompt=prompt_text, width=1024, height=1024, output_resolution=1024,
                     num_inference_steps=40, true_cfg_scale=1.0, use_kv_cache=True,
                     generator=torch.Generator("cuda").manual_seed(item["seed"])).images[0]
        image.convert("RGB").save(dest)
        write_once(rec, {"condition_sha256": digest(condition), "inputs": condition, "sha256": sha(dest)})
        print("reference", item["name"], dest, flush=True)


def worker_frame(task_path: Path, output: Path) -> None:
    import torch
    from qwen140_pilot_qwen import make_pipe
    task = json.loads(task_path.read_text())
    assert not output.exists()
    pipe = make_pipe("frames")
    images = [Image.open(x["path"]).convert("RGB") for x in task["images"]]
    start = time.monotonic()
    result = pipe(prompt=task["prompt"], image=images if images else None,
                  width=task["width"], height=task["height"], output_resolution=1024,
                  num_inference_steps=task["steps"], true_cfg_scale=1.0, use_kv_cache=True,
                  generator=torch.Generator("cuda").manual_seed(task["seed"])).images[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    result.convert("RGB").save(output)
    write_once(output.with_suffix(".json"), {"condition_sha256": digest(task), "inputs": task,
                    "sha256": sha(output), "elapsed_seconds": time.monotonic() - start})
    print("keyframe", output, flush=True)


def worker_video(task_path: Path, output_dir: Path) -> None:
    import torch
    from qwen140_pilot_video import load_pipeline
    from run_lightx2v_a14b_4step_qwen140 import decode_video
    task = json.loads(task_path.read_text())
    assert not (output_dir / "video.mp4").exists()
    pipeline, _ = load_pipeline("flf2v" if task["last_frame_path"] else "i2v", task["frames"])
    temp = output_dir / "video.tmp.mp4"
    assert not temp.exists()
    kwargs = {"seed": task["seed"], "prompt": task["prompt"], "image_path": task["input_path"],
              "save_result_path": str(temp)}
    if task["last_frame_path"]:
        kwargs["last_frame_path"] = task["last_frame_path"]
    start = time.monotonic()
    pipeline.generate(**kwargs)
    torch.cuda.synchronize()
    decoded = decode_video(temp, output_dir / "first.png", output_dir / "last.png", task["frames"], 16)
    temp.replace(output_dir / "video.mp4")
    write_once(output_dir / "generation.json", {"condition_sha256": digest(task), "inputs": task,
               "decoded": decoded, "video_sha256": sha(output_dir / "video.mp4"),
               "last_sha256": sha(output_dir / "last.png"), "elapsed_seconds": time.monotonic() - start})
    print("video", output_dir / "video.mp4", flush=True)


def status() -> None:
    manifest, _, _, _ = source()
    for run in TAGS:
        ep = episode_for(run, manifest)
        root = OUT / "episodes" / ep["episode_id"]
        count = sum((root / f"shot_{s['shot_key'].replace(':', '_')}" / "result.json").exists() for s in ep["shots"])
        print(run, count, '/', len(ep["shots"]), "complete" if (root / "episode_result.json").exists() else "in_progress")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run", "status", "worker-refs", "worker-frame", "worker-video"])
    parser.add_argument("--run", choices=list(TAGS))
    parser.add_argument("--task", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "run":
        assert args.run
        run_episode(args.run)
    elif args.action == "status":
        status()
    elif args.action == "worker-refs":
        assert args.task
        worker_refs(args.task)
    elif args.action == "worker-frame":
        assert args.task and args.output
        worker_frame(args.task, args.output)
    elif args.action == "worker-video":
        assert args.task and args.output
        worker_video(args.task, args.output)
