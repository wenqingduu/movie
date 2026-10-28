#!/usr/bin/env python3
"""Prepare per-character v7 3D references and masks for a Qwen Control frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from facexlib.parsing import init_parsing_model
from PIL import Image

from multishot.ip_adapter_experiment_utils import _face_app
from multishot.multi_face_reference import (
    _face_embedding,
    single,
    build_color_safe_reference,
    detected_faces,
)


def _slug(value: str) -> str:
    return "_".join(value.lower().replace("-", " ").split())


def match_all_detected_faces(app, characters: list[dict], faces: list):
    """Match roles globally before discarding any extra detected faces."""
    if not faces:
        raise RuntimeError("detected 0 reliable faces")
    references = [_face_embedding(single._largest_face(app, item["reference"]))
                  for item in characters]
    candidates = [_face_embedding(face) for face in faces]
    matrix = np.asarray([[float(np.dot(ref, face)) for face in candidates]
                         for ref in references], dtype=np.float32)
    role_indices, face_indices = linear_sum_assignment(-matrix)
    assignments = sorted(zip(role_indices.tolist(), face_indices.tolist()), key=lambda pair: pair[1])
    ordered = [characters[role] for role, _ in assignments]
    selected = [faces[face] for _, face in assignments]
    pairs = []
    for role, face in assignments:
        alternatives = [float(matrix[row, face]) for row in range(len(characters)) if row != role]
        second = max(alternatives) if alternatives else None
        pairs.append({"character": characters[role]["name"], "face_index_left_to_right": face,
                      "cosine": float(matrix[role, face]), "next_best_character_cosine": second,
                      "chosen_margin_over_next_best": float(matrix[role, face]) - second if second is not None else None})
    return ordered, selected, {
        "method": "insightface_all_detected_faces_global_one_to_one_maximum_cosine",
        "total_score": sum(float(matrix[role, face]) for role, face in assignments),
        "cosine_matrix_character_by_face": matrix.tolist(),
        "input_character_order": [item["name"] for item in characters],
        "assigned_characters_left_to_right": [item["name"] for item in ordered],
        "unassigned_scheduled_characters": [item["name"] for i, item in enumerate(characters) if i not in role_indices],
        "unassigned_detected_face_indices": [i for i in range(len(faces)) if i not in face_indices],
        "pairs": pairs}


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

    characters = []
    for item in characters_spec:
        copied = dict(item)
        copied["reference"] = Image.open(item["reference_image"]).convert("RGB")
        characters.append(copied)
    ordered, faces, assignment = match_all_detected_faces(app, characters, detected)

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
        "kind": "qwen_image21_multiface_v7_all_detected_reference_preparation",
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

