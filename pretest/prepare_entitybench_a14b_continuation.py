"""Freeze the full 20-episode, single-route A14B rollout and hard-cut inputs.

Reuses audited PuLID-FLUX single-character frames. Missing multi-character
hard cuts use the existing zero-identity-weight FLUX fallback generator.
Historical paired reports and media are read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
COHORT = PROJECT / "outputs/entitybench_firstframe_pulid_ip_20ep/manifest.json"
OUTPUT = PROJECT / "outputs/entitybench_pulid_i2v_a14b_continuation_20ep"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def ordered_shots(episode):
    shots = []
    for scene in episode["scenes"]:
        prompts = scene["video_prompts"]
        assert len(prompts) == len(scene["cut"]), "Incomplete cut annotations"
        for index, prompt in enumerate(prompts, 1):
            key = f"{scene['scene_num']}:{index}"
            assert isinstance(scene["cut"][index - 1], bool)
            shots.append({
                "shot_key": key, "shot_index": len(shots), "prompt": prompt,
                "cut": scene["cut"][index - 1],
                "entity_schedule": episode["entity_schedule"].get(key, {}),
                "action_description": (scene.get("action_descriptions") or prompts)[index - 1],
            })
    assert shots and shots[0]["cut"], "Episode must start at a hard cut"
    return shots


def single_source(record, shot):
    pair = Path(record["output_dir"])
    config_path = pair / "config.json"
    config = json.loads(config_path.read_text())
    image = pair / "treatment/final.png"
    expected = record.get("treatment_sha256")
    assert expected and sha256(image) == expected, f"Changed Treatment: {image}"
    assert config["prompt"] == shot["prompt"], f"Wrong prompt: {image}"
    if record.get("plugin_skipped"):
        control = pair / "control/final.png"
        assert sha256(control) == expected, f"Fallback differs from Control: {image}"
        kind = "single_safety_fallback"
    else:
        assert config["steps"] == 50 and config["plugin_start_step"] == 30
        assert (pair / "step_30").is_dir(), f"Missing intermediate preparation: {pair}"
        kind = "single_midstep_v7"
    return {
        "input_image": str(image), "input_sha256": expected, "source_kind": kind,
        "source_config": str(config_path), "source_config_sha256": sha256(config_path),
        "firstframe_seed": config["seed"], "skip_reason": record.get("skip_reason"),
        "reference_image": record["reference_image"],
        "reference_sha256": sha256(record["reference_image"]),
        "facelift_result": record.get("facelift_result"),
    }


def multi_source(ep_record, shot, output):
    name = "shot_" + shot["shot_key"].replace(":", "_")
    candidates = [
        Path(ep_record["pulid_output"]) / "first_frames" / name / "flux_control",
        output / "supplemental_firstframes" / ep_record["episode_id"] / "first_frames" / name / "flux_control",
    ]
    for pair in candidates:
        image, config_path = pair / "control/final.png", pair / "config.json"
        if not image.is_file() or not config_path.is_file():
            continue
        config = json.loads(config_path.read_text())
        assert config["pulid_id_weight"] == 0.0 and config["plugin_applied"] is False
        assert config["prompt"] == shot["prompt"] and config["steps"] == 50
        assert config["seed"] == ep_record["base_seed"] + shot["shot_index"]
        return {
            "input_image": str(image), "input_sha256": sha256(image),
            "source_kind": "multi_character_no_injection",
            "source_config": str(config_path), "source_config_sha256": sha256(config_path),
            "firstframe_seed": config["seed"],
            "supplemental": str(pair).startswith(str(output / "supplemental_firstframes")),
        }
    return None


def prepare(output=OUTPUT, generate_missing=False):
    output = Path(output).resolve()
    cohort = json.loads(COHORT.read_text())
    episodes, inventory, assets = [], [], []
    for ep in cohort["episodes"]:
        data = json.loads(Path(ep["episode_path"]).read_text())
        shots = ordered_shots(data)
        report_path = Path(ep["pulid_output"]) / "pulid_first_frame_report.json"
        report = json.loads(report_path.read_text())
        singles = {row["shot_key"]: row for row in report["shots"]}
        missing = [s for s in shots if s["cut"] and len(s["entity_schedule"].get("characters", [])) > 1
                   and multi_source(ep, s, output) is None]
        if missing and generate_missing:
            neutral = sorted((Path(ep["asset_root"]) / "characters").glob("*_reference.png"))[0]
            command = [str(PROJECT / ".venv/bin/python"),
                       str(PROJECT / "pretest/prepare_entitybench_flux_controls.py"),
                       "--episode", ep["episode_path"], "--output-dir",
                       str(output / "supplemental_firstframes" / ep["episode_id"]),
                       "--base-seed", str(ep["base_seed"]), "--neutral-reference", str(neutral),
                       "--width", "640", "--height", "352", "--steps", "50", "--guidance", "4.0"]
            for shot in missing:
                command.extend(["--shot", shot["shot_key"]])
            print(f"Preparing {len(missing)} missing multi-character hard cuts: {ep['run']}", flush=True)
            env = os.environ.copy()
            env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="8",
                       OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8")
            log = output / "logs" / f"firstframes_{ep['run']}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a") as stream:
                subprocess.run(command, cwd=PROJECT, env=env, stdout=stream,
                               stderr=subprocess.STDOUT, check=True)
        previous = None
        for shot in shots:
            shot.update(episode_id=ep["episode_id"], seed=ep["base_seed"] + shot["shot_index"],
                        previous_shot_key=previous)
            if shot["cut"]:
                characters = shot["entity_schedule"].get("characters", [])
                if len(characters) == 1:
                    source = single_source(singles[shot["shot_key"]], shot)
                else:
                    source = multi_source(ep, shot, output)
                if source is None:
                    source = {"source_kind": "missing_multi_hardcut"}
                shot["firstframe_source"] = source
                inventory.append({"episode_id": ep["episode_id"], "shot_key": shot["shot_key"], **source})
            previous = shot["shot_key"]
        descriptions = data.get("entity_descriptions", {})
        scheduled = sorted({name for shot in shots for name in shot["entity_schedule"].get("characters", [])})
        for name in scheduled:
            slug = "_".join(name.lower().replace("-", " ").split())
            reference = Path(ep["asset_root"]) / "characters" / f"{slug}_reference.png"
            assets.append({"episode_id": ep["episode_id"], "character": name,
                           "reference_image": str(reference) if reference.is_file() else None,
                           "reference_sha256": sha256(reference) if reference.is_file() else None,
                           "status": "available" if reference.is_file() else "missing_original_reference"})
        episodes.append({"episode_id": ep["episode_id"], "run": ep["run"], "difficulty": ep["difficulty"],
                         "episode_path": ep["episode_path"], "episode_sha256": sha256(ep["episode_path"]),
                         "source_report": str(report_path), "source_report_sha256": sha256(report_path),
                         "entity_descriptions": descriptions, "shots": shots})
    counts = Counter(row["source_kind"] for row in inventory)
    assert len(episodes) == 20 and sum(len(e["shots"]) for e in episodes) == 363
    assert len(inventory) == 191
    manifest = {
        "kind": "entitybench_pulid_i2v_a14b_continuation_20ep", "schema_version": 1,
        "cohort_manifest": str(COHORT), "cohort_sha256": sha256(COHORT),
        "output_root": str(output), "route": "single_pulid_hardcut_own_tail_continuation",
        "model_revision": "206a9ee1b7bfaaf8f7e4d81335650533490646a3",
        "model_path": str(PROJECT / "models/video/Wan2.2-I2V-A14B"),
        "video_config": {"max_area": 832 * 480, "frame_num": 81, "fps": 16,
                         "sampling_steps": 40, "shift": 3.0, "guide_scale": [3.5, 3.5],
                         "solver": "unipc", "param_dtype": "bfloat16", "t5_cpu": True,
                         "offload_model": True, "init_on_cpu": True, "convert_model_dtype": True},
        "evaluation": {"vlm_enabled": False, "scope": "partial_without_vlm",
                       "cross_shot_fidelity_gate": "not_applied_without_vlm; provisional scores"},
        "totals": {"episode_count": 20, "shot_count": 363, "hardcut": 191, "continuation": 172,
                   "firstframe_source_counts": dict(counts)}, "episodes": episodes,
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "firstframe_inventory.json", {"counts": dict(counts), "items": inventory})
    write_json(output / "asset_manifest.json", {"items": assets, "counts": dict(Counter(a["status"] for a in assets))})
    write_json(output / "split_20ep.json", {tier: [ep["episode_id"] for ep in episodes if ep["difficulty"] == tier]
                                          for tier in ["easy", "medium", "hard"]})
    print(json.dumps(manifest["totals"], indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--generate-missing", action="store_true")
    args = parser.parse_args()
    prepare(args.output_root, args.generate_missing)
