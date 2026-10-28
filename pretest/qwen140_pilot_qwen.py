#!/usr/bin/env python3
"""Isolated Qwen-Image-2.1 references and keyframes for three audited Qwen140 shots."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140"
OLD20 = ROOT / "outputs/entitybench_qwen21_i2v_a14b_4step_continuation_20ep"
OUT = ROOT / "outputs/entitybench_qwen140_rerun_pilot_20261026"
MODEL = ROOT / "models/diffusion/qwen-image-2.1"

TARGETS = {
    "run714": ("i20_j34", "6:2", ["Leo"]),
    "run298": ("i40_j64", "23:2", ["Julian", "Simon"]),
    "run484": ("i15_j47", "5:2", ["Chloe", "Leo"]),
    "run755": ("i15_j49", "3:5", ["Dante", "Chloe"]),
    "run1404": ("i170_j220", "6:1", ["Kenji", "Ryan", "Mei"]),
}
NEW_REFS = [("run714", "Leo"), ("run298", "Simon"), ("run484", "Chloe"), ("run484", "Leo"), ("run1404", "Kenji")]
STYLE = "Photorealistic live-action cinematic photography; consistent natural skin, realistic clothing and anatomy."


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def data(run: str) -> tuple[dict, dict, dict, dict]:
    tag, shot_key, _ = TARGETS[run]
    plan = json.loads((BASE / "preparation_plan.json").read_text())
    ep = next(e for e in plan["episodes"] if e["run"] == run and tag in e["episode_id"])
    i = next(i for i, s in enumerate(ep["shots"]) if s["shot_key"] == shot_key)
    shot, previous = ep["shots"][i], ep["shots"][i - 1]
    script = json.loads(Path(ep["episode_path"]).read_text())
    report = json.loads((BASE / "generation_report.json").read_text())
    parent = next(j for j in report["jobs"] if j["episode_id"] == ep["episode_id"] and j["shot_key"] == previous["shot_key"])
    assert sha(Path(parent["last_frame"])) == parent["last_frame_sha256"]
    return ep, shot, script, parent


def ref_path(run: str, name: str) -> Path:
    if (run, name) in NEW_REFS:
        return OUT / "references" / run / f"{name.lower()}.png"
    ep, _, _, _ = data(run)
    plans = [json.loads((BASE / "preparation_plan.json").read_text()), json.loads((OLD20 / "preparation_plan.json").read_text())]
    role = next(r for plan in plans for r in plan["roles"] if r["episode_id"] == ep["episode_id"] and r["character"] == name)
    path = Path(role["reference"])
    assert path.is_file()
    return path


def earlier_role_crop(run: str, name: str) -> Path:
    ep, shot, _, _ = data(run)
    files = list((BASE / "evaluation/official_parallel_v2/.shards").glob(
        f"worker_*/results/{ep['episode_id']}.json"))
    assert len(files) == 1
    role = json.loads(files[0].read_text())["per_entity_scores"][f"face_{name}"]
    earlier = [s["shot_key"] for s in ep["shots"]].index(role["representative_shot"])
    current = [s["shot_key"] for s in ep["shots"]].index(shot["shot_key"])
    assert earlier < current
    path = Path(role["representative_crop_path"])
    assert path.is_file()
    return path


def ref_prompt(run: str, name: str, description: str) -> str:
    qualifiers = {
        ("run714", "Leo"): "adult blond man, distinct from the dark-haired woman Clara; studio portrait",
        ("run298", "Simon"): "adult blond man with slicked-back hair, white button-down shirt and dark trousers",
        ("run484", "Chloe"): "adult woman with shoulder-length black hair, white blazer, earrings, necklace",
        ("run484", "Leo"): "young school-age boy, yellow jacket over a red polo shirt, visibly a child",
        ("run1404", "Kenji"): "adult man with short black hair, dark charcoal business suit, photorealistic face and proportions",
    }
    return (f"{STYLE} ONE PERSON ONLY, one continuous photograph, centered waist-up full-color character reference "
            f"on a plain neutral gray background. Character {name}: {description}. "
            f"{qualifiers[(run, name)]}. Clear face, natural eyes, clean skin, distinct clothing. "
            "No second person, no collage, no grid, no panels, no inset, no text, no illustration, no anime.")


def frame_prompt(run: str, shot: dict, script: dict, variant: str = "v1") -> str:
    descriptions = {r["name"]: r["description"] for r in script["entity_descriptions"].values()}
    if run == "run714":
        if variant == "identity_anchor_v2":
            return ("Image 1 is the ACTUAL Leo seen earlier in this same episode, cropped from verified shot "
                    "5:1: preserve his exact blond side-parted hairstyle, face shape, apparent age and realistic "
                    "skin. He is the same man, not a newly designed model. Keep his black jacket with white "
                    "sleeve stripes and a light gray hood over a dark top, because no clothing change is specified. "
                    "Image 2 is the ACTUAL PREVIOUS VIDEO TAIL showing Clara inside the plain white studio; "
                    "use it ONLY for the studio environment and lighting. Create a NEW medium close-up of Leo "
                    "alone in that white studio, holding a DSLR camera with both hands at chest height just "
                    "before raising it to his eye. The camera does not cover his face yet. No Clara, no second "
                    "person, no collage. " + STYLE)
        return (f"Image 1 is the fixed character reference for Leo: {descriptions['Leo']}. "
                "Image 2 is the ACTUAL PREVIOUS VIDEO TAIL and is ONLY a visual reference for the stark white studio "
                "and its lighting. Create a NEW medium close-up shot focused on Leo alone, the blond adult man. "
                "Clara, the brunette woman in Image 2, is NOT in this new shot. "
                "Leo stands against the plain white studio background holding a DSLR camera with both hands at chest height, "
                "just before raising it to his eye. The camera does not yet cover his face. " + STYLE)
    if run == "run298":
        if variant == "stage_lit_v2":
            return (f"Image 1 is Julian, {descriptions['Julian']}. Image 2 is Simon, {descriptions['Simon']}. "
                    "Image 3 is the previous nightclub stage shot: use its dramatic blue and red concert lighting "
                    "and atmosphere ONLY, never copy its singer. NEW chest-up shot at the auditorium sound booth. "
                    "Julian is a dark-haired man on the LEFT wearing a white suit. Simon is a slicked-back blond man "
                    "on the RIGHT wearing a white button-down shirt. Both stand behind a clearly visible mixing desk "
                    "with faders, looking toward visibly EMPTY dark-red auditorium chairs. Bright balanced cinematic "
                    "exposure, blue and red rim lights on the men's faces and clothing, soft warm front fill, rich "
                    "saturated but natural color, sharp facial details, detailed chairs and sound desk, no crushed "
                    "blacks. Exactly two men, no singer, no other person. They have not started talking yet. " + STYLE)
        return (f"Image 1 is Julian, {descriptions['Julian']}. Image 2 is Simon, {descriptions['Simon']}. "
                "Image 3 is the ACTUAL PREVIOUS VIDEO TAIL: use it ONLY for the nightclub's blue/red stage lighting "
                "and visual atmosphere, NOT for the old shot composition or its female singers. "
                "Create a NEW chest-up two-man shot behind the sound booth inside The Gilded Stage. "
                "Julian is the dark-haired adult man in a white suit, Simon is the slicked-back blond adult man in a white "
                "button-down shirt. Both stand side by side behind the sound mixing desk, looking toward empty rows of dark red "
                "auditorium chairs, poised to start talking to one another. Clear two distinct men, no singer visible. " + STYLE)
    if run == "run484":
        if variant == "doorway_v2":
            return (f"Use Image 1 for Chloe only: {descriptions['Chloe']}. Use Image 2 for young boy Leo only: "
                    f"{descriptions['Leo']}. Image 3 shows the prior living room and Nora on a patterned couch. "
                    "Create a WIDE SHOT with EXACTLY THREE people. Place Chloe and young Leo TOGETHER in the "
                    "BACK-LEFT DOORWAY, both just stepping through the door; they are small and far from the couch. "
                    "Nora is the only other person; she remains SEATED on the patterned couch on the RIGHT. "
                    "The couch takes up the right foreground; an open stretch of floor separates doorway from couch. "
                    "Chloe wears a white blazer, Leo is clearly a CHILD in a yellow jacket and red polo, Nora wears "
                    "green. No fourth person, no extra woman in the hall, no duplicates. Do not move Chloe or Leo "
                    "next to the sofa yet. Camera at eye level, balanced photographic composition. " + STYLE)
        return (f"Image 1 is Chloe: {descriptions['Chloe']}. Image 2 is Leo: {descriptions['Leo']}. "
                "Image 3 is the ACTUAL PREVIOUS VIDEO TAIL and is the only visual reference for the townhouse living room, "
                "its patterned couch, staircase, lighting, and Nora in a green blouse on the couch. "
                "Target END of an entrance transition: Chloe, an adult black-haired woman in a white blazer, and Leo, a young "
                "boy in a yellow jacket over a red polo shirt, have JUST entered the living room through its doorway. "
                "They are near the entrance and have NOT yet walked up to the patterned couch. Nora may remain seated on the couch. "
                "Do not duplicate Chloe or turn Leo into an adult woman. " + STYLE)
    if run == "run755":
        return (f"Image 1 is Dante: {descriptions['Dante']}. Image 2 is Chloe: {descriptions['Chloe']}. "
                "Image 3 is the ACTUAL PREVIOUS VIDEO TAIL showing the CAB NEWS screen inside the black London taxi. "
                "Use it only to establish the dark cab interior, blue screen glow, and photorealistic film style. "
                "Create a NEW medium two-shot inside the taxi's rear passenger seat, facing Dante and Chloe. "
                "Dante is the bald bearded man on the LEFT in a hooded white forensic garment. Chloe is the woman "
                "on the RIGHT with long curly dark hair and gold highlights in a patterned robe. They look toward "
                "the small off-camera news screen with tense faces, before their expressions change. Exactly two "
                "people in the rear seat; no screen filling the frame, no men from the news broadcast, no inset. " + STYLE)
    if run == "run1404":
        if variant == "style_clean_v2":
            return (f"Image 1 is Kenji: {descriptions['Kenji']}. Image 2 is Ryan: {descriptions['Ryan']}. "
                    f"Image 3 is Mei: {descriptions['Mei']}. Create a NEW photorealistic film frame inside "
                    "The Red Lotus commercial kitchen. EXACTLY FOUR PEOPLE IN TOTAL: (1) Mei in a pink apron "
                    "at the flaming wok on the LEFT; (2) one anonymous chef in realistic white cook clothes "
                    "at a rear prep station; (3) ONE Kenji, a short black-haired man in a dark charcoal suit, "
                    "standing at the RIGHT doorway; (4) Ryan, the blond man in a green lettered bomber jacket, "
                    "standing BESIDE Kenji at the SAME RIGHT doorway. Kenji and Ryan are only beginning to enter "
                    "the kitchen. NO person in the center foreground. Exactly one dark-suited man, exactly one "
                    "green-jacketed man, exactly one pink-apron woman, exactly one white-clad chef. All four people "
                    "are live-action photographic humans, including the chef: natural skin, realistic anatomy. "
                    "No duplicates, no anime, no illustration, no cel shading. Stainless steel prep stations, "
                    "steel hood, red-toned restaurant walls, one flaming wok. " + STYLE)
        return (f"Image 1 is a NEW photorealistic reference for Kenji: {descriptions['Kenji']}. "
                f"Image 2 is Ryan: {descriptions['Ryan']}. Image 3 is Mei: {descriptions['Mei']}. "
                "Image 4 is the ACTUAL PREVIOUS VIDEO TAIL from outside the restaurant: use ONLY its live-action "
                "photographic rendering as a style cue, not its car or street. NEW medium-wide shot inside the Red "
                "Lotus commercial kitchen, with stainless steel counters, a large steel hood, and red-toned decor. "
                "Mei, the dark-haired woman in a pink apron, is at the flaming wok on the left. A background "
                "chef in realistic white cook clothes works at a prep station. Kenji, a dark-haired man in a "
                "charcoal suit, and Ryan, the blond man in a green bomber jacket, are visible at the RIGHT kitchen "
                "doorway, just beginning to rush inside. All four people are PHOTOREALISTIC LIVE-ACTION HUMANS with "
                "consistent natural skin and anatomy. No anime face, no illustrated chef, no cel shading, no inset. " + STYLE)
    raise ValueError(run)


def make_pipe(stage: str):
    import torch
    from diffusers import QwenImage21Pipeline
    torch.set_num_threads(4)
    pipe = QwenImage21Pipeline.from_pretrained(str(MODEL), dtype=torch.bfloat16, local_files_only=True)
    if stage == "refs":
        pipe.to("cuda")
    else:
        pipe.enable_sequential_cpu_offload()
    return pipe


def generate_refs(pipe, runs: list[str]) -> None:
    import torch
    for run, name in NEW_REFS:
        if run not in runs:
            continue
        if shutil.disk_usage(OUT).free < 10 * 1024**3:
            raise RuntimeError("Less than 10 GiB free")
        dest = ref_path(run, name)
        ep, _, script, _ = data(run)
        description = next(x["description"] for x in script["entity_descriptions"].values() if x["name"] == name)
        prompt = ref_prompt(run, name, description)
        seed = {("run714", "Leo"): 71462, ("run298", "Simon"): 298232, ("run484", "Chloe"): 484521, ("run484", "Leo"): 484522,
                ("run1404", "Kenji"): 140461}[(run, name)]
        rec = dest.with_suffix(".json")
        inputs = {"episode_id": ep["episode_id"], "name": name, "description": description, "prompt": prompt, "seed": seed, "model_revision": "790c92633540aa0cb11d9abf19eb46d861714758", "steps": 40}
        if rec.exists() and dest.exists():
            row = json.loads(rec.read_text())
            assert row["inputs"] == inputs and sha(dest) == row["sha256"]
            print("verified", run, name, flush=True)
            continue
        image = pipe(prompt=prompt, width=1024, height=1024, output_resolution=1024, num_inference_steps=40,
                     true_cfg_scale=1.0, use_kv_cache=True, generator=torch.Generator("cuda").manual_seed(seed)).images[0]
        dest.parent.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(dest)
        rec.write_text(json.dumps({"inputs": inputs, "path": str(dest.resolve()), "sha256": sha(dest)}, ensure_ascii=False, indent=2) + "\n")
        print("generated", run, name, dest, flush=True)


def generate_frames(pipe, runs: list[str], variant: str = "v1") -> None:
    import torch
    from PIL import Image
    for run in runs:
        if shutil.disk_usage(OUT).free < 10 * 1024**3:
            raise RuntimeError("Less than 10 GiB free")
        ep, shot, script, parent = data(run)
        names = TARGETS[run][2]
        refs = ([earlier_role_crop(run, "Leo")] if run == "run714" and variant == "identity_anchor_v2"
                else [ref_path(run, name) for name in names])
        assert all(p.exists() for p in refs)
        tail = Path(parent["last_frame"])
        condition_paths = refs if run == "run1404" and variant == "style_clean_v2" else refs + [tail]
        images = [Image.open(p).convert("RGB") for p in condition_paths]
        prompt = frame_prompt(run, shot, script, variant)
        seed = int(shot["seed"]) + (2_000_000 if variant != "v1" else 1_000_000)
        dest = OUT / "keyframes" / run / ("frame.png" if variant == "v1" else f"frame_{variant}.png")
        rec = dest.with_suffix(".json")
        inputs = {"episode_id": ep["episode_id"], "shot_key": shot["shot_key"], "kind": "transition_target" if run == "run484" else "new_shot_start",
                  "prompt": prompt, "condition_images": [{"path": str(p.resolve()), "sha256": sha(p)} for p in condition_paths],
                  "seed": seed, "model_revision": "790c92633540aa0cb11d9abf19eb46d861714758", "steps": 20}
        if variant != "v1":
            inputs["variant"] = variant
        if run == "run1404" and variant == "style_clean_v2":
            inputs["excluded_parent_tail"] = {"path": str(tail.resolve()), "sha256": sha(tail),
                                               "reason": "Hard cut to a different room; old street people caused duplicates in v1"}
        if rec.exists() and dest.exists():
            row = json.loads(rec.read_text())
            assert row["inputs"] == inputs and sha(dest) == row["sha256"]
            print("verified", run, dest, flush=True)
            continue
        image = pipe(prompt=prompt, image=images, width=1024, height=576, output_resolution=1024,
                     num_inference_steps=20, true_cfg_scale=1.0, use_kv_cache=True,
                     generator=torch.Generator("cuda").manual_seed(seed)).images[0]
        dest.parent.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(dest)
        rec.write_text(json.dumps({"inputs": inputs, "path": str(dest.resolve()), "sha256": sha(dest)}, ensure_ascii=False, indent=2) + "\n")
        print("generated", run, dest, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["refs", "frames"], required=True)
    parser.add_argument("--runs", nargs="+", choices=list(TARGETS), required=True)
    parser.add_argument("--variant", choices=["v1", "doorway_v2", "stage_lit_v2", "style_clean_v2", "identity_anchor_v2"], default="v1")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    pipe = make_pipe(args.stage)
    if args.stage == "refs":
        generate_refs(pipe, args.runs)
    else:
        assert args.variant == "v1" or (args.variant == "doorway_v2" and args.runs == ["run484"]) or (args.variant == "stage_lit_v2" and args.runs == ["run298"]) or (args.variant == "style_clean_v2" and args.runs == ["run1404"]) or (args.variant == "identity_anchor_v2" and args.runs == ["run714"])
        generate_frames(pipe, args.runs, args.variant)
