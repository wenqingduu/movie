"""Current color-safe per-face reference preparation for multi-reference models."""

from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter

from multishot import pulid_flux_inner_face_experiment as single


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def detected_faces(app, image: Image.Image, minimum_confidence: float) -> list:
    rgb = np.asarray(image.convert("RGB"))
    faces = app.get(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)) or []
    faces = [face for face in faces if float(face.det_score) >= minimum_confidence]
    return sorted(faces, key=lambda face: float((face.bbox[0] + face.bbox[2]) / 2.0))


def select_faces(faces: list, maximum_count: int) -> list:
    """Keep up to the largest scheduled-face count in deterministic spatial order."""

    if not faces:
        raise RuntimeError("detected 0 reliable faces")
    if len(faces) > maximum_count:
        faces = sorted(
            faces,
            key=lambda face: float(
                (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
            ),
            reverse=True,
        )[:maximum_count]
    return sorted(faces, key=lambda face: float((face.bbox[0] + face.bbox[2]) / 2.0))


def _face_embedding(face) -> np.ndarray:
    value = np.asarray(face.embedding, dtype=np.float32)
    return value / (np.linalg.norm(value) + 1e-8)


def assign_characters_to_faces(
    app, characters: list[dict], faces: list
) -> tuple[list[dict], dict]:
    """Assign detected faces to references with a global one-to-one optimum."""

    references = [
        _face_embedding(single._largest_face(app, item["reference"]))
        for item in characters
    ]
    candidates = [_face_embedding(face) for face in faces]
    matrix = np.asarray(
        [
            [float(np.dot(reference, candidate)) for candidate in candidates]
            for reference in references
        ],
        dtype=np.float32,
    )
    if len(faces) > len(characters):
        raise ValueError("Detected face count cannot exceed scheduled character count")
    best_score = -float("inf")
    best_characters = None
    for character_indices in permutations(range(len(characters)), len(faces)):
        score = float(
            sum(
                matrix[character_index, face_index]
                for face_index, character_index in enumerate(character_indices)
            )
        )
        if score > best_score:
            best_score = score
            best_characters = character_indices
    if best_characters is None:
        raise RuntimeError("No one-to-one character assignment was found")
    character_for_face = [characters[index] for index in best_characters]
    pairs = []
    for face_index, character_index in enumerate(best_characters):
        alternatives = [
            float(matrix[row, face_index])
            for row in range(len(characters))
            if row != character_index
        ]
        second_best = max(alternatives) if alternatives else None
        pairs.append(
            {
                "character": characters[character_index]["name"],
                "face_index_left_to_right": int(face_index),
                "cosine": float(matrix[character_index, face_index]),
                "next_best_character_cosine": second_best,
                "chosen_margin_over_next_best": (
                    float(matrix[character_index, face_index]) - second_best
                    if second_best is not None
                    else None
                ),
            }
        )
    return character_for_face, {
        "method": "insightface_partial_global_one_to_one_maximum_cosine",
        "total_score": best_score,
        "cosine_matrix_character_by_face": matrix.tolist(),
        "input_character_order": [item["name"] for item in characters],
        "assigned_characters_left_to_right": [
            item["name"] for item in character_for_face
        ],
        "unassigned_scheduled_characters": [
            item["name"]
            for index, item in enumerate(characters)
            if index not in set(best_characters)
        ],
        "pairs": pairs,
    }


def build_color_safe_reference(
    *,
    item: dict,
    face,
    preview: Image.Image,
    pulid,
    device: torch.device,
    output_dir: Path,
) -> dict:
    """Build the current v7 harmonized reference and mask for one detected face."""

    target_bbox = single._bbox(face, preview.width, preview.height)
    pose = [float(value) for value in getattr(face, "pose", [0.0, 0.0, 0.0])]
    facelift_asset = single._load_facelift_asset_record(Path(item["facelift_result"]))

    from multishot.mcp_asset_server import _render_3d_face_reference

    gaussian_model_path = Path(facelift_asset["model_path"])
    gaussian_asset_dir = Path(
        facelift_asset.get("facelift_output_dir") or gaussian_model_path.parent
    )
    rendered_path = _render_3d_face_reference(
        {"model_path": str(gaussian_model_path), "path": str(gaussian_asset_dir)},
        {"pitch": pose[0], "yaw": pose[1], "roll": pose[2]},
        target_bbox,
    )
    if not rendered_path:
        raise RuntimeError(f"3D render failed or pose validation rejected for {item['name']}")
    rendered = Image.open(rendered_path).convert("RGB")
    source_face = single._largest_face(pulid.app, rendered)
    source_bbox = single._bbox(source_face, rendered.width, rendered.height)
    aligned, alignment = single._align_reference(
        rendered,
        source_bbox,
        target_bbox,
        (preview.width, preview.height),
        "facelift_continuous_gaussian_pose_render",
        None,
    )
    aligned_face = single._largest_face(pulid.app, aligned)
    aligned_bbox = single._bbox(aligned_face, aligned.width, aligned.height)

    parser = pulid.face_helper.face_parse
    target_inner = single._semantic_inner_face_mask(preview, target_bbox, parser, device)
    target_features = single._semantic_inner_face_mask(
        preview,
        target_bbox,
        parser,
        device,
        included_labels=single.IDENTITY_FEATURE_LABELS,
    )
    reference_inner = single._semantic_inner_face_mask(
        aligned, aligned_bbox, parser, device
    )
    reference_features = single._semantic_inner_face_mask(
        aligned,
        aligned_bbox,
        parser,
        device,
        included_labels=single.IDENTITY_FEATURE_LABELS,
    )
    target_context = single._semantic_probability_mask(
        preview,
        target_bbox,
        parser,
        device,
        included_labels=single.COLOR_CONTEXT_LABELS,
    )
    reference_context = single._semantic_probability_mask(
        aligned,
        aligned_bbox,
        parser,
        device,
        included_labels=single.COLOR_CONTEXT_LABELS,
    )
    target_skin = single._semantic_inner_face_mask(
        preview,
        target_bbox,
        parser,
        device,
        included_labels=(single.HARMONIZATION_POLICY["skin_label"],),
    )
    reference_skin = single._semantic_inner_face_mask(
        aligned,
        aligned_bbox,
        parser,
        device,
        included_labels=(single.HARMONIZATION_POLICY["skin_label"],),
    )

    target_core = single._connected_identity_feature_core_mask(
        target_inner, target_features, target_bbox
    )
    reference_core = single._connected_identity_feature_core_mask(
        reference_inner, reference_features, aligned_bbox
    )
    core_intersection = Image.fromarray(
        np.minimum(np.asarray(target_core), np.asarray(reference_core)).astype(np.uint8),
        mode="L",
    )
    skin_intersection = Image.fromarray(
        np.minimum(np.asarray(target_skin), np.asarray(reference_skin)).astype(np.uint8),
        mode="L",
    )
    color_application, color_metadata = single._color_context_application_mask(
        core_intersection, target_context, reference_context, target_bbox
    )
    hard_injection, safety_metadata = (
        single._safe_injection_mask_from_color_application(
            core_intersection, color_application, target_bbox
        )
    )
    final_mask = hard_injection.filter(
        ImageFilter.GaussianBlur(
            radius=single.CONSERVATIVE_MASK_POLICY["feather_radius_px"]
        )
    )
    harmonized, harmonization_images, harmonization_metadata = (
        single._harmonize_reference(
            aligned,
            preview,
            skin_intersection,
            color_application,
            final_mask,
            target_bbox,
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    rendered.save(output_dir / "rendered_3d_face.png")
    aligned.save(output_dir / "aligned_3d_face.png")
    target_core.save(output_dir / "target_identity_core.png")
    reference_core.save(output_dir / "reference_identity_core.png")
    core_intersection.save(output_dir / "identity_core_intersection.png")
    color_application.save(output_dir / "harmonization_application_mask.png")
    final_mask.save(output_dir / "final_injection_mask.png")
    for filename, image in harmonization_images.items():
        image.save(output_dir / filename)
    _write_json(
        output_dir / "reference_metadata.json",
        {
            "character": item["name"],
            "strategy": "color_safe_core",
            "target_bbox": target_bbox,
            "pitch_yaw_roll": pose,
            "det_score": float(face.det_score),
            "alignment": alignment,
            "color_context": color_metadata,
            "injection_safety": safety_metadata,
            "harmonization": harmonization_metadata,
        },
    )
    return {
        **item,
        "bbox": target_bbox,
        "pose": pose,
        "trajectory_reference": harmonized,
        "mask": final_mask,
    }
