#!/usr/bin/env python3
"""Create reproducible SDXL identity references and FaceLift assets.

All characters scheduled to appear in the episode are selected by default.
The former single-character-only scope remains available explicitly for
reproducing early pilot runs.
The reference stage renders several deterministic SDXL Base candidates and
selects the clearest near-frontal single face. FaceLift construction and pose
calibration are explicit stages so the large models never need to coexist.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from multishot.diffusion_backend import get_diffusion_backend
from multishot.face_analysis_backend import get_face_backend
from multishot.facelift_backend import build_facelift_asset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "sdxl-base-1.0"
MODEL_PATH = PROJECT_ROOT / "models" / "diffusion" / "sdxl-base-1.0"


def _slug(name: str) -> str:
    return "_".join(name.lower().replace("-", " ").split())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _characters(episode: dict, scope: str) -> list[dict]:
    scheduled_names: set[str] = set()
    single_names: set[str] = set()
    for schedule in episode.get("entity_schedule", {}).values():
        names = schedule.get("characters", []) or []
        scheduled_names.update(names)
        if len(names) == 1:
            single_names.add(names[0])
    selected_names = scheduled_names if scope == "scheduled" else single_names
    result = []
    for item in episode.get("entity_descriptions", {}).values():
        name = item.get("name")
        if name in selected_names:
            result.append({"name": name, "description": item.get("description", "")})
    missing = selected_names - {item["name"] for item in result}
    if missing:
        raise ValueError(f"Missing character descriptions: {sorted(missing)}")
    return sorted(result, key=lambda item: item["name"])


def _prompt(name: str, description: str) -> str:
    # Put the non-negotiable FaceLift constraints before the benchmark's often
    # long description so CLIP's 77-token truncation cannot remove them.
    return (
        "Single adult, centered frontal head-and-shoulders identity portrait, "
        "looking directly at camera, neutral expression, both eyes visible, "
        "symmetrical face, no eyeglasses, plain gray background, soft even studio "
        f"light, photorealistic natural skin, sharp eyes, no text. {name}: {description}"
    )


def _candidate_score(faces: list[dict], width: int, height: int) -> tuple[float, dict | None]:
    if len(faces) != 1:
        return -1_000.0 - abs(len(faces) - 1), None
    face = faces[0]
    x1, y1, x2, y2 = face["face_bbox"]
    area_ratio = max(0.0, (x2 - x1) * (y2 - y1) / float(width * height))
    height_ratio = max(0.0, (y2 - y1) / float(height))
    pose = face["pose"]
    score = (
        float(face["face_confidence"])
        + 1.5 * min(math.sqrt(area_ratio), math.sqrt(0.35))
        - 2.0 * max(0.0, 0.25 - height_ratio)
        - 0.012 * abs(float(pose["yaw"]))
        - 0.006 * abs(float(pose["pitch"]))
        - 0.004 * abs(float(pose["roll"]))
    )
    return score, face


def _reference_stage(args: argparse.Namespace, episode: dict, characters: list[dict]) -> None:
    from PIL import Image

    os.environ["MULTISHOT_IMAGE_HEIGHT"] = str(args.size)
    os.environ["MULTISHOT_IMAGE_WIDTH"] = str(args.size)
    os.environ["MULTISHOT_DIFFUSION_STEPS"] = str(args.steps)
    os.environ["MULTISHOT_GUIDANCE_SCALE"] = str(args.guidance)
    backend = get_diffusion_backend(MODEL_NAME)
    face_backend = get_face_backend()

    for char_index, character in enumerate(characters):
        slug = _slug(character["name"])
        char_dir = args.asset_root / "characters"
        final_path = char_dir / f"{slug}_reference.png"
        metadata_path = char_dir / f"{slug}_reference.metadata.json"
        prompt = _prompt(character["name"], character["description"])
        if final_path.is_file() and not args.overwrite:
            if not metadata_path.is_file():
                with Image.open(final_path) as image:
                    width, height = image.size
                faces = face_backend.analyze(str(final_path))
                score, selected_face = _candidate_score(faces, width, height)
                _write_json(metadata_path, {
                    "episode_id": args.episode.stem,
                    "character": character,
                    "model": MODEL_NAME,
                    "model_repo": "stabilityai/stable-diffusion-xl-base-1.0",
                    "model_path": str(MODEL_PATH.resolve()),
                    "prompt": (
                        final_path.with_suffix(".prompt.txt").read_text(encoding="utf-8").strip()
                        if final_path.with_suffix(".prompt.txt").is_file()
                        else prompt
                    ),
                    "width": width,
                    "height": height,
                    "selected_sha256": _sha256(final_path),
                    "selected_face": selected_face,
                    "selection_score": score,
                    "selection_mode": "adopted_existing_episode_reference",
                    "candidates": [],
                })
            print(f"[reference] reuse {character['name']}: {final_path}")
            continue

        records = []
        for candidate_index in range(args.candidates):
            seed = args.base_seed + char_index * 100 + candidate_index
            candidate = char_dir / "candidates" / slug / f"seed_{seed}.png"
            backend.generate_image(prompt, str(candidate), steps=args.steps, seed=seed)
            with Image.open(candidate) as image:
                width, height = image.size
            faces = face_backend.analyze(str(candidate))
            score, selected_face = _candidate_score(faces, width, height)
            records.append({
                "path": str(candidate.resolve()),
                "seed": seed,
                "sha256": _sha256(candidate),
                "face_count": len(faces),
                "selected_face": selected_face,
                "selection_score": score,
            })
            print(f"[reference] {character['name']} seed={seed} score={score:.4f}")

        best = max(records, key=lambda item: item["selection_score"])
        if best["selected_face"] is None:
            raise RuntimeError(f"No single-face candidate for {character['name']}")
        source = Path(best["path"])
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_bytes(source.read_bytes())
        final_path.with_suffix(".prompt.txt").write_text(prompt + "\n", encoding="utf-8")
        _write_json(metadata_path, {
            "episode_id": args.episode.stem,
            "character": character,
            "model": MODEL_NAME,
            "model_repo": "stabilityai/stable-diffusion-xl-base-1.0",
            "model_path": str(MODEL_PATH.resolve()),
            "prompt": prompt,
            "negative_prompt": backend.negative_prompt,
            "width": args.size,
            "height": args.size,
            "steps": args.steps,
            "guidance_scale": args.guidance,
            "candidate_count": args.candidates,
            "selected_seed": best["seed"],
            "selected_sha256": _sha256(final_path),
            "selected_face": best["selected_face"],
            "candidates": records,
        })
        print(f"[reference] selected {character['name']}: seed={best['seed']} -> {final_path}")

    # Do not keep SDXL resident while FaceLift runs in a child process.
    backend._pipe = None
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def _facelift_stage(args: argparse.Namespace, characters: list[dict]) -> None:
    for character in characters:
        slug = _slug(character["name"])
        reference = args.asset_root / "characters" / f"{slug}_reference.png"
        face_dir = args.asset_root / "faces_3d" / slug
        result_path = face_dir / "facelift_result.json"
        if result_path.is_file() and not args.overwrite:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if not result.get("source_reference_sha256"):
                legacy_input = face_dir / "facelift_input" / "input.png"
                if not legacy_input.is_file() or _sha256(legacy_input) != _sha256(reference):
                    raise RuntimeError(
                        f"Cannot adopt legacy FaceLift asset for {character['name']}: "
                        "the recorded input does not match the current identity reference"
                    )
                result["source_reference"] = str(reference.resolve())
                result["source_reference_sha256"] = _sha256(reference)
                result["source_metadata_mode"] = "verified_legacy_facelift_input_match"
                _write_json(result_path, result)
            print(f"[facelift] reuse {character['name']}: {result_path}")
            continue
        if not reference.is_file():
            raise FileNotFoundError(reference)
        result = build_facelift_asset(str(reference), str(face_dir))
        result["source_reference"] = str(reference.resolve())
        result["source_reference_sha256"] = _sha256(reference)
        _write_json(result_path, result)
        print(f"[facelift] {character['name']}: {result['facelift_status']}")


def _selection_stage(args: argparse.Namespace, characters: list[dict]) -> None:
    overrides = {}
    for value in args.select:
        if "=" not in value:
            raise ValueError(f"--select must be CHARACTER=SEED, got {value!r}")
        name, seed = value.rsplit("=", 1)
        overrides[name] = int(seed)
    known = {item["name"] for item in characters}
    if set(overrides) - known:
        raise ValueError(f"Unknown selected characters: {sorted(set(overrides) - known)}")

    for character in characters:
        name = character["name"]
        if name not in overrides:
            continue
        slug = _slug(name)
        char_dir = args.asset_root / "characters"
        final_path = char_dir / f"{slug}_reference.png"
        metadata_path = char_dir / f"{slug}_reference.metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        matches = [item for item in metadata["candidates"] if item["seed"] == overrides[name]]
        if len(matches) != 1:
            raise ValueError(f"Candidate seed {overrides[name]} not found for {name}")
        selected = matches[0]
        if selected["selected_face"] is None:
            raise ValueError(f"Candidate seed {overrides[name]} has no unique face for {name}")
        source = Path(selected["path"])
        final_path.write_bytes(source.read_bytes())
        metadata.update({
            "selected_seed": selected["seed"],
            "selected_sha256": _sha256(final_path),
            "selected_face": selected["selected_face"],
            "selection_mode": "manual_visual_review",
        })
        _write_json(metadata_path, metadata)
        print(f"[select] {name}: seed={selected['seed']} -> {final_path}")


def _calibration_stage(args: argparse.Namespace, characters: list[dict]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    for character in characters:
        slug = _slug(character["name"])
        face_dir = args.asset_root / "faces_3d" / slug
        result = json.loads((face_dir / "facelift_result.json").read_text(encoding="utf-8"))
        model = Path(result["model_path"])
        calibration = model.with_suffix(".pose_calibration.json")
        output = face_dir / "pose_calibration"
        if calibration.is_file() and (output / "calibration_result.json").is_file() and not args.overwrite:
            print(f"[calibrate] reuse {character['name']}: {calibration}")
            continue
        command = [
            args.python,
            "-m",
            "multishot.facelift_pose_calibration",
            "--model", str(model),
            "--output", str(output),
            "--calibration-output", str(calibration),
            "--image-size", str(args.calibration_size),
        ]
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)
        print(f"[calibrate] {character['name']}: {calibration}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("references", "select", "facelift", "calibrate", "all"),
        default="all",
    )
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument(
        "--character-scope",
        choices=("scheduled", "single-shot"),
        default="scheduled",
        help=(
            "Build assets for every character in the episode schedule (default), "
            "or only characters that appear alone in at least one shot."
        ),
    )
    parser.add_argument(
        "--only-character",
        action="append",
        default=[],
        help="Restrict the selected scope to these exact character names; repeat as needed.",
    )
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--calibration-size", type=int, default=512)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--select", action="append", default=[], metavar="CHARACTER=SEED",
        help="With --stage select, override automatic choice after visual review.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.episode = args.episode.resolve()
    args.asset_root = args.asset_root.resolve()

    episode = json.loads(args.episode.read_text(encoding="utf-8"))
    characters = _characters(episode, args.character_scope)
    if args.only_character:
        requested = set(args.only_character)
        known = {item["name"] for item in characters}
        if requested - known:
            raise ValueError(f"Unknown requested characters: {sorted(requested - known)}")
        characters = [item for item in characters if item["name"] in requested]
    print(
        f"Selected characters ({args.character_scope}): "
        f"{[item['name'] for item in characters]}"
    )
    if args.stage in {"references", "all"}:
        _reference_stage(args, episode, characters)
    if args.stage == "select":
        _selection_stage(args, characters)
    if args.stage in {"facelift", "all"}:
        _facelift_stage(args, characters)
    if args.stage in {"calibrate", "all"}:
        _calibration_stage(args, characters)


if __name__ == "__main__":
    main()
