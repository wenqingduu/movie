"""Absolute identity scores for the validated single-character shot scope.

First-frame role mapping may use the fixed identity reference. Video tracking
is spatial thereafter. Multi-character auxiliary scores are explicitly missing;
they remain included in the full official partial evaluator.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from prepare_entitybench_a14b_continuation import OUTPUT, sha256, write_json
from run_wan22_i2v_a14b_continuation import fingerprint
from evaluate_video_identity import _evaluate_video, _largest_face, _cosine
from evaluate_reference_dino_identity import DinoEncoder, _crop_face, _decode_video_crops
from evaluate_cross_shot_face_centroid import representative_video_crop, centroid_scores
from multishot.face_analysis_backend import get_face_backend


def mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def summarize(rows):
    return {"scheduled_role_shots": len(rows), "evaluated_shots": sum(r["status"] == "evaluated" for r in rows),
            "first_insightface": mean(r.get("insightface", {}).get("first") for r in rows),
            "video_insightface": mean(r.get("insightface", {}).get("mean") for r in rows),
            "first_reference_dino": mean(r.get("dino", {}).get("first") for r in rows),
            "video_reference_dino": mean(r.get("dino", {}).get("mean") for r in rows),
            "video_face_coverage": mean(r.get("insightface", {}).get("detection_coverage") for r in rows),
            "valid_video_insightface_role_shots": sum(r.get("insightface", {}).get("mean") is not None for r in rows),
            "valid_video_dino_role_shots": sum(r.get("dino", {}).get("mean") is not None for r in rows)}


@torch.inference_mode()
def cls_feature(encoder, image):
    inputs = encoder.processor(images=[image], return_tensors="pt")
    output = encoder.model(**{key: value.to(encoder.device) for key, value in inputs.items()})
    return F.normalize(output.last_hidden_state[:, 0].float(), dim=-1)[0].cpu().numpy().tolist()


def run(root):
    torch.set_num_threads(8)
    manifest = json.loads((root / "manifest.json").read_text())
    generation = json.loads((root / "generation_report.json").read_text())
    assert generation["status"] == "complete" and len(generation["jobs"]) == 363
    output = root / "evaluation/project"
    backend = get_face_backend()
    encoder = DinoEncoder(Path("/root/autodl-tmp/models/dinov2-base"), "cuda", 16)
    model_path = Path("/root/autodl-tmp/models/dinov2-base")
    weight_hashes = {p.name: sha256(p) for p in sorted(model_path.glob("*")) if p.suffix in {".bin", ".safetensors"}}
    config = {"implementation_sha256": sha256(__file__), "dino_weights": weight_hashes,
              "mapping": "identity_reference_on_actual_first_frame_only", "tracking": "spatial",
              "crop_expansion": 0.15, "centroid": "CLS; quality-selected representative from five equal samples; self included",
              "scope": "single_character_shots; multi-character auxiliary mapping unavailable"}
    write_json(output / "evaluation_manifest.json", config)
    references = {(r["episode_id"], r["character"]): r for r in json.loads((root / "asset_manifest.json").read_text())["items"]}
    reference_cache, rows, exclusions = {}, [], []
    jobs = {(r["episode_id"], r["shot_key"]): r for r in generation["jobs"]}
    for episode in manifest["episodes"]:
        for shot in episode["shots"]:
            characters = shot["entity_schedule"].get("characters", [])
            if len(characters) != 1:
                exclusions.append({"episode_id": episode["episode_id"], "shot_key": shot["shot_key"],
                                   "characters": characters, "reason": "multi_character_auxiliary_mapping_unavailable" if characters else "no_scheduled_character"})
                continue
            character = characters[0]
            generation_row = jobs[(episode["episode_id"], shot["shot_key"])]
            directory = Path(generation_row["output_video"]).parent
            reference = references[(episode["episode_id"], character)]
            row = {"episode_id": episode["episode_id"], "run": episode["run"], "shot_key": shot["shot_key"],
                   "character": character, "cut": shot["cut"], "source_type": generation_row["input_source_type"],
                   "video_sha256": generation_row["video_sha256"], "reference_sha256": reference.get("reference_sha256")}
            item_path = output / "shots" / episode["episode_id"] / ("shot_" + shot["shot_key"].replace(":", "_")) / "metrics.json"
            expected = fingerprint({"generation": generation_row["dependency_fingerprint"],
                                    "video": row["video_sha256"], "reference": row["reference_sha256"], "evaluation": config})
            assert sha256(directory / "video.mp4") == row["video_sha256"]
            if reference["reference_image"]:
                assert sha256(reference["reference_image"]) == reference["reference_sha256"]
            if item_path.is_file():
                cached = json.loads(item_path.read_text())
                if cached.get("fingerprint") == expected:
                    rows.append(cached)
                    continue
            row["fingerprint"] = expected
            if not reference["reference_image"]:
                row.update(status="missing_reference", missing_reason="missing_original_reference")
                rows.append(row)
                write_json(item_path, row)
                continue
            refkey = (episode["episode_id"], character)
            if refkey not in reference_cache:
                reference_path = Path(reference["reference_image"])
                assert sha256(reference_path) == reference["reference_sha256"]
                face = _largest_face(backend.analyze(str(reference_path)))
                if face is None:
                    reference_cache[refkey] = None
                else:
                    crop = _crop_face(Image.open(reference_path).convert("RGB"), face["face_bbox"], 0.15)
                    reference_cache[refkey] = {"identity": face["face_embedding"], "dino": encoder.encode([crop])[0]}
            ref = reference_cache[refkey]
            if ref is None:
                row.update(status="unscorable_reference", missing_reason="no_face_in_original_reference")
                rows.append(row)
                write_json(item_path, row)
                continue
            first_image = Image.open(directory / "first.png").convert("RGB")
            faces = backend.analyze(str(directory / "first.png"))
            selected = max(faces, key=lambda face: _cosine(ref["identity"], face["face_embedding"]), default=None)
            item_path.parent.mkdir(parents=True, exist_ok=True)
            tracking = _evaluate_video(directory / "video.mp4", ref["identity"], item_path.parent / "tracking.jpg",
                tracking_policy="spatial", initial_bbox=selected["face_bbox"] if selected else None,
                initial_image_size=list(first_image.size))
            write_json(item_path.parent / "tracking.json", tracking)
            crops, indexes, decoded = _decode_video_crops(directory / "video.mp4", tracking["frames"], 0.15)
            similarities = [None] * decoded
            if crops:
                embeddings = encoder.encode(crops)
                for index, embedding in zip(indexes, embeddings):
                    similarities[index] = float(np.dot(ref["dino"], embedding))
            first_bbox = tracking["frames"][0].get("selected_face_bbox")
            if first_bbox:
                crop = _crop_face(first_image, first_bbox, 0.15)
                crop.save(item_path.parent / "first_crop.png")
                row["first_cls_feature"] = cls_feature(encoder, crop)
            representative, selection = representative_video_crop(directory / "video.mp4", tracking["frames"], 0.15, 5)
            row["representative_selection"] = selection
            if representative is not None:
                representative.save(item_path.parent / "representative_crop.png")
                row["video_cls_feature"] = cls_feature(encoder, representative)
            row.update(status="evaluated", insightface=tracking["summary"],
                       dino={"first": similarities[0], "mean": mean(similarities),
                             "last": similarities[-1], "frame_scores": similarities,
                             "coverage": sum(v is not None for v in similarities) / len(similarities)})
            rows.append(row)
            write_json(item_path, row)
            write_json(output / "progress.json", {"completed_single_shots": len(rows), "target": 132})
            print(f"Identity {len(rows)}/132: {episode['run']} {shot['shot_key']}", flush=True)
    assert len(rows) == 132 and len(exclusions) == 231
    groups = defaultdict(list)
    for row in rows:
        groups[(row["episode_id"], row["character"])].append(row)
    centroids = {}
    for modality, feature in [("first_frame", "first_cls_feature"), ("video_representative", "video_cls_feature")]:
        entity_rows, pooled = [], []
        for (episode_id, character), group in sorted(groups.items()):
            valid = [r for r in group if feature in r]
            entry = {"episode_id": episode_id, "character": character, "scheduled": len(group), "valid": len(valid), "shots": []}
            if len(valid) < 2:
                entry["missing_reason"] = "fewer_than_two_valid_single_character_shots"
            else:
                centroid, scores, pairwise = centroid_scores([r[feature] for r in valid])
                entry.update(mean=float(np.mean(scores)), pairwise_median=pairwise, centroid=centroid.tolist())
                entry["shots"] = [{"shot_key": r["shot_key"], "score": float(score)} for r, score in zip(valid, scores)]
                pooled.extend(float(score) for score in scores)
            entity_rows.append(entry)
        centroids[modality] = {"mean": mean(pooled), "scored_role_shots": len(pooled), "entities": entity_rows}
    summary = dict(summarize(rows), single_character_shots=132, unscored_multi_or_empty_shots=231)
    per_episode = [{"episode_id": episode["episode_id"], **summarize([r for r in rows if r["episode_id"] == episode["episode_id"]])}
                   for episode in manifest["episodes"]]
    per_character = [{"episode_id": episode_id, "character": character, **summarize(group)}
                     for (episode_id, character), group in sorted(groups.items())]
    cut_diagnostics = {"hardcut": summarize([r for r in rows if r["cut"]]),
                       "continuation": summarize([r for r in rows if not r["cut"]])}
    episode_macro = {key: mean(r[key] for r in per_episode)
                     for key in ["first_insightface", "video_insightface", "first_reference_dino",
                                 "video_reference_dino", "video_face_coverage"]}
    write_json(output / "identity_report.json", {"scope": config["scope"], "aggregation": "valid single role-shot micro mean",
               "summary": summary, "episode_macro": episode_macro, "per_episode": per_episode,
               "per_character": per_character, "cut_diagnostics": cut_diagnostics,
               "jobs": rows, "excluded_shots": exclusions, "centroids": centroids})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    run(parser.parse_args().output_root)
