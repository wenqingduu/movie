#!/usr/bin/env python3
"""Prepare per-character v7 3D references and masks for a Qwen Control frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from facexlib.parsing import init_parsing_model
from PIL import Image

from multishot.ip_adapter_experiment_utils import _face_app
from multishot.multi_face_reference import (
    assign_characters_to_faces,
    build_color_safe_reference,
    detected_faces,
    select_faces,
)


def _slug(value: str) -> str:
    return "_".join(value.lower().replace("-", " ").split())


def _parse_character(value: str) -> dict:
    parts = value.split("=", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError(
            "--character must be NAME=REFERENCE_IMAGE=FACELIFT_RESULT"
        )
    return {
        "name": parts[0],
        "reference_image": str(Path(parts[1]).resolve()),
        "facelift_result": str(Path(parts[2]).resolve()),
    }


def run(args: argparse.Namespace, *, app=None, parser=None) -> dict:
    if len(args.character) < 1:
        raise ValueError("At least one --character entry is required")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    control = Image.open(args.control).convert("RGB")
    control.save(output / "control.png")

    app = app or _face_app()
    detected = detected_faces(app, control, args.min_face_confidence)
    faces = select_faces(detected, len(args.character))
    if len(faces) != len(args.character):
        raise RuntimeError(
            f"Qwen Control has {len(faces)} reliable faces, expected {len(args.character)}"
        )

    characters = []
    for item in args.character:
        copied = dict(item)
        copied["reference"] = Image.open(item["reference_image"]).convert("RGB")
        characters.append(copied)
    ordered, assignment = assign_characters_to_faces(app, characters, faces)

    device = torch.device("cuda")
    parser = parser or init_parsing_model(model_name="bisenet", device=device)
    helper = SimpleNamespace(face_parse=parser)
    pulid_like = SimpleNamespace(app=app, face_helper=helper)
    targets = []
    for item, face in zip(ordered, faces):
        character_dir = output / "characters" / _slug(item["name"])
        target = build_color_safe_reference(
            item=item,
            face=face,
            preview=control,
            pulid=pulid_like,
            device=device,
            output_dir=character_dir,
        )
        targets.append({
            "name": item["name"],
            "reference_image": item["reference_image"],
            "facelift_result": item["facelift_result"],
            "bbox": target["bbox"],
            "pose": target["pose"],
            "harmonized_reference": str((character_dir / "harmonized_3d_face.png").resolve()),
            "mask": str((character_dir / "final_injection_mask.png").resolve()),
        })

    result = {
        "kind": "qwen_image21_multiface_v7_reference_preparation",
        "control": str(args.control.resolve()),
        "control_size": list(control.size),
        "reliable_face_count": len(detected),
        "selected_face_count": len(faces),
        "extra_reliable_face_count": max(0, len(detected) - len(faces)),
        "assignment": assignment,
        "targets": targets,
    }
    (output / "prepared.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--character", action="append", type=_parse_character, required=True)
    parser.add_argument("--min-face-confidence", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
