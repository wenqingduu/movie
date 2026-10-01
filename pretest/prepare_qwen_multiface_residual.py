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


def prepare_from_image(
    *,
    preview: Image.Image,
    characters_spec: list[dict],
    output_dir: Path,
    minimum_confidence: float,
    app=None,
    parser=None,
    preview_source: str | None = None,
) -> dict:
    """Prepare v7 references from the actual injection-point x0 preview."""

    if len(characters_spec) < 1:
        raise ValueError("At least one character entry is required")
    output = output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    preview = preview.convert("RGB")
    preview.save(output / "pred_x0.png")

    app = app or _face_app()
    detected = detected_faces(app, preview, minimum_confidence)
    # Partial appearances are valid.  A multi-reference prompt may render only
    # a subset of its scheduled cast; prepare every face that can be mapped
    # instead of making one missing role invalidate the entire shot.
    faces = select_faces(detected, len(characters_spec))

    characters = []
    for item in characters_spec:
        copied = dict(item)
        copied["reference"] = Image.open(item["reference_image"]).convert("RGB")
        characters.append(copied)
    ordered, assignment = assign_characters_to_faces(app, characters, faces)

    device = torch.device("cuda")
    parser = parser or init_parsing_model(model_name="bisenet", device=device)
    helper = SimpleNamespace(face_parse=parser)
    pulid_like = SimpleNamespace(app=app, face_helper=helper)
    targets = []
    target_failures = []
    for item, face in zip(ordered, faces):
        character_dir = output / "characters" / _slug(item["name"])
        try:
            target = build_color_safe_reference(
                item=item,
                face=face,
                preview=preview,
                pulid=pulid_like,
                device=device,
                output_dir=character_dir,
            )
        except Exception as exc:
            target_failures.append({
                "name": item["name"],
                "bbox": [float(value) for value in face.bbox],
                "det_score": float(face.det_score),
                "reason": "reference_preparation_failed",
                "error": repr(exc),
            })
            continue
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
        "preview_source": preview_source or "in_memory_pred_x0",
        "preview": str((output / "pred_x0.png").resolve()),
        "preview_size": list(preview.size),
        # Compatibility key used by the paired video evaluators.  For the
        # shared-prefix protocol this is the pred_x0 canvas size, which is also
        # the final Control/Treatment canvas size.
        "control_size": list(preview.size),
        "reliable_face_count": len(detected),
        "selected_face_count": len(faces),
        "extra_reliable_face_count": max(0, len(detected) - len(faces)),
        "assignment": assignment,
        "requested_characters": [item["name"] for item in characters_spec],
        "unassigned_scheduled_characters": assignment[
            "unassigned_scheduled_characters"
        ],
        "target_failures": target_failures,
        "successful_target_count": len(targets),
        "targets": targets,
    }
    # Always preserve the audit trail, including attempts where no role could
    # be prepared.  Only a non-empty target set is eligible for injection.
    (output / "preparation_attempt.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    prepared_path = output / "prepared.json"
    if targets:
        prepared_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        prepared_path.unlink(missing_ok=True)
    return result


def run(args: argparse.Namespace, *, app=None, parser=None) -> dict:
    control = Image.open(args.control).convert("RGB")
    result = prepare_from_image(
        preview=control,
        characters_spec=args.character,
        output_dir=args.output_dir,
        minimum_confidence=args.min_face_confidence,
        app=app,
        parser=parser,
        preview_source=str(args.control.resolve()),
    )
    if not result["targets"]:
        raise RuntimeError("No detected role produced a usable v7 reference")
    # Preserve the legacy keys while the old two-pass batch remains readable.
    result["control"] = str(args.control.resolve())
    result["control_size"] = list(control.size)
    (args.output_dir.resolve() / "prepared.json").write_text(
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
