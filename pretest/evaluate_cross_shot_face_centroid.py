#!/usr/bin/env python3
"""Add paired DINOv2 face-centroid scores to the frozen 20-episode cohort.

This uses EntityBench's normalized-mean centroid formula, not its official
GroundingDINO character crops, CLIP selection, or VLM fidelity gate. Existing
first-frame mappings and spatial video tracks supply face boxes. Generation and
the reference-anchored identity reports are read-only inputs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONDITIONS = ("control", "treatment")
MODALITIES = ("first_frame", "video_representative")
FEATURE_VERSION = "tracked-face-cls-v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(values):
    values = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.all(np.isfinite(values)) or np.any(norm < 1e-9):
        raise ValueError("nonfinite or zero feature vector")
    return values / norm


def centroid_scores(embeddings):
    """One normalized CLS feature per distinct shot; include self as in paper."""
    if len(embeddings) < 2:
        raise ValueError("centroid consistency requires at least two shots")
    embeddings = normalized(embeddings)
    centroid = normalized(embeddings.mean(axis=0))
    scores = embeddings @ centroid
    pairs = (embeddings @ embeddings.T)[np.triu_indices(len(embeddings), 1)]
    return centroid, scores, float(np.median(pairs))


def crop_face(image, bbox, expansion):
    x1, y1, x2, y2 = map(float, bbox)
    if not np.all(np.isfinite(bbox)) or x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid face bbox: {bbox}")
    fw, fh = x2 - x1, y2 - y1
    box = (max(0, int(np.floor(x1 - fw * expansion))),
           max(0, int(np.floor(y1 - fh * expansion))),
           min(image.width, int(np.ceil(x2 + fw * expansion))),
           min(image.height, int(np.ceil(y2 + fh * expansion))))
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"bbox outside image: {bbox}")
    return image.crop(box), list(box)


def representative_video_crop(path, frames, expansion, sample_count):
    """Select by sharpness * relative face area, never by identity similarity."""
    sampled = sorted(set(np.linspace(0, len(frames) - 1, sample_count).astype(int).tolist())) if frames else []
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode video: {path}")
    candidates, best_crop, best = [], None, None
    decoded = 0
    try:
        while capture.grab():
            if decoded >= len(frames):
                raise ValueError(f"video has more frames than tracking report: {path}")
            if decoded in sampled:
                ok, bgr = capture.retrieve()
                if not ok:
                    raise RuntimeError(f"cannot retrieve sampled frame {decoded}: {path}")
                frame_record = frames[decoded]
                if frame_record.get("frame_index") != decoded:
                    raise ValueError(f"tracking frame index mismatch: {path}")
                bbox = frame_record.get("selected_face_bbox")
                record = {"frame_index": decoded, "bbox": bbox}
                if bbox is None:
                    record["skip_reason"] = "missing_tracked_face"
                else:
                    image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    crop, expanded = crop_face(image, bbox, expansion)
                    gray = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2GRAY)
                    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (image.width * image.height) * 100
                    score = float(1 / (1 + np.exp(np.clip(-(sharpness - 100) / 200, -700, 700)))
                                  * 1 / (1 + np.exp(np.clip(-(area - 2) / 5, -700, 700))))
                    record.update(expanded_bbox=expanded, laplacian_variance=sharpness,
                                  face_area_percent=area, selection_score=score)
                    if best is None or score > best["selection_score"]:
                        best_crop, best = crop, record
                candidates.append(record)
            decoded += 1
    finally:
        capture.release()
    if decoded != len(frames):
        raise ValueError(f"video/report frame-count mismatch: {path}, {decoded}/{len(frames)}")
    return best_crop, {"decoded_frame_count": decoded, "sampled_frame_indices": sampled,
                       "sampled_detected_count": sum(c.get("bbox") is not None for c in candidates),
                       "candidates": candidates, "selected": best,
                       "missing_reason": "no_face_in_sampled_frames" if best is None else None}


def paired_groups(rows, modality, injected_only=False):
    """Match shots before calculating separate condition-specific centroids."""
    grouped = defaultdict(list)
    for row in rows:
        if not injected_only or row["plugin_applied"]:
            grouped[(row["route"], row["episode_id"], row["character"])].append(row)
    entities = []
    for (route, episode, character), scheduled in sorted(grouped.items()):
        scheduled = sorted(scheduled, key=lambda r: tuple(map(int, r["shot_key"].split(":"))))
        if len({r["shot_key"] for r in scheduled}) != len(scheduled):
            raise ValueError(f"duplicate character-shot: {route} {episode} {character}")
        matched = [r for r in scheduled if all(r[modality][c].get("embedding") is not None for c in CONDITIONS)]
        exclusions = []
        for r in scheduled:
            reasons = {c: r[modality][c].get("missing_reason") for c in CONDITIONS
                       if r[modality][c].get("embedding") is None}
            if reasons:
                exclusions.append({"shot_key": r["shot_key"], "reasons": reasons})
        entity = {"route": route, "episode_id": episode, "run": scheduled[0]["run"],
                  "difficulty": scheduled[0]["difficulty"], "character": character,
                  "scheduled_shot_count": len(scheduled), "matched_shot_count": len(matched),
                  "scheduled_injected_count": sum(r["plugin_applied"] for r in scheduled),
                  "excluded_shots": exclusions, "control": None, "treatment": None,
                  "delta": None, "shots": [], "skip_reason": None}
        if len(matched) < 2:
            entity["skip_reason"] = ("single_scheduled_appearance" if len(scheduled) < 2
                                     else "fewer_than_two_paired_appearances")
        else:
            try:
                computed = {c: centroid_scores([r[modality][c]["embedding"] for r in matched]) for c in CONDITIONS}
            except ValueError as exc:
                entity["skip_reason"] = f"degenerate_features: {exc}"
                entities.append(entity)
                continue
            for c, (centroid, scores, pair_median) in computed.items():
                entity[c] = {"mean": float(np.mean(scores)), "minimum": float(np.min(scores)),
                             "maximum": float(np.max(scores)), "pairwise_median": pair_median,
                             "representative_shot": matched[int(np.argmax(scores))]["shot_key"],
                             "worst_shot": matched[int(np.argmin(scores))]["shot_key"],
                             "centroid": centroid.tolist()}
            entity["delta"] = entity["treatment"]["mean"] - entity["control"]["mean"]
            for i, r in enumerate(matched):
                control, treatment = (float(computed[c][1][i]) for c in CONDITIONS)
                entity["shots"].append({"shot_key": r["shot_key"], "plugin_applied": r["plugin_applied"],
                                        "control": control, "treatment": treatment, "delta": treatment - control,
                                        "control_crop": r[modality]["control"]["crop_path"],
                                        "treatment_crop": r[modality]["treatment"]["crop_path"]})
        entities.append(entity)
    return entities


def aggregate(entities):
    shots = [s for e in entities for s in e["shots"]]
    eligible = sum(e["scheduled_shot_count"] for e in entities if e["scheduled_shot_count"] >= 2)
    return {"scheduled_entity_count": len(entities), "scored_entity_count": sum(bool(e["shots"]) for e in entities),
            "scheduled_appearance_count": sum(e["scheduled_shot_count"] for e in entities),
            "eligible_recurring_appearance_count": eligible, "scored_paired_appearance_count": len(shots),
            "paired_coverage": len(shots) / eligible if eligible else None,
            "control_mean": float(np.mean([s["control"] for s in shots])) if shots else None,
            "treatment_mean": float(np.mean([s["treatment"] for s in shots])) if shots else None,
            "mean_paired_delta": float(np.mean([s["delta"] for s in shots])) if shots else None,
            "positive_entity_count": sum(e["delta"] is not None and e["delta"] > 1e-12 for e in entities),
            "negative_entity_count": sum(e["delta"] is not None and e["delta"] < -1e-12 for e in entities)}


def summarize(rows):
    scopes = {}
    for scope, injected_only in (("all_single_character", False), ("injected_only", True)):
        scopes[scope] = {}
        for modality in MODALITIES:
            entities = paired_groups(rows, modality, injected_only)
            scopes[scope][modality] = {
                "routes": {r: aggregate([e for e in entities if e["route"] == r]) for r in sorted({e["route"] for e in entities})},
                "episodes": {ep: {r: aggregate([e for e in entities if e["route"] == r and e["episode_id"] == ep])
                                   for r in sorted({e["route"] for e in entities if e["episode_id"] == ep})}
                             for ep in sorted({e["episode_id"] for e in entities})},
                "difficulty": {tier: {r: aggregate([e for e in entities if e["route"] == r and e["difficulty"] == tier])
                                      for r in sorted({e["route"] for e in entities if e["difficulty"] == tier})}
                               for tier in sorted({e["difficulty"] for e in entities})},
                "entities": entities}
    return scopes


def audit_sheets(result, destination):
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    for modality, report in result["scopes"]["all_single_character"].items():
        for entity in report["entities"]:
            for start in range(0, len(entity["shots"]), 6):
                selected = entity["shots"][start:start + 6]
                sheet = Image.new("RGB", (960, 100 + 220 * len(selected)), "#f0f0f0")
                draw = ImageDraw.Draw(sheet)
                draw.text((15, 10), f"{entity['route']} {entity['run']} | {entity['character']} | {modality}", fill="black", font=font)
                draw.text((15, 40), f"Centroid: {entity['control']['mean']:.4f} -> {entity['treatment']['mean']:.4f} ({entity['delta']:+.4f})", fill="black", font=font)
                for i, s in enumerate(selected):
                    y = 90 + i * 220
                    draw.text((10, y), f"Shot {s['shot_key']}\n{'injected' if s['plugin_applied'] else 'Control reuse'}", fill="black", font=font)
                    for c, x in (("control", 220), ("treatment", 550)):
                        crop = Image.open(s[f"{c}_crop"]).convert("RGB")
                        crop.thumbnail((280, 175))
                        sheet.paste(crop, (x, y + 28))
                        draw.text((x, y), f"{c}: {s[c]:.4f}", fill="black", font=font)
                token = hashlib.sha256((entity["episode_id"] + entity["character"]).encode()).hexdigest()[:12]
                path = destination / "audit" / modality / f"{entity['route'].lower().replace('-', '_')}_{entity['run']}_{token}_{start // 6 + 1}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                sheet.save(path, quality=92)
                entity.setdefault("audit_sheets", []).append(str(path))


def number(value, signed=False):
    return (f"{value:+.4f}" if signed else f"{value:.4f}") if value is not None else "N/A"


def markdown(result):
    lines = ["# 20 episode 跨镜头人脸外观质心相似度", "",
             "项目 DINOv2 人脸裁剪指标：采用 EntityBench 质心公式，不是官方 cs_face。", "",
             "按路线、episode、角色分组；Control/Treatment 使用两侧均有效的相同镜头集合，各自计算质心。",
             "每角色至少两个不同镜头。对单位 CLS 向量求均值并归一化，再计算各镜头到自身组质心的 cosine。",
             "路线/episode 总分按可评分角色-shot 等权汇总，不按帧或角色等权。缺失值为 null，不填 0。", "",
             "首帧使用现有最终人脸框；视频从 5 个等距帧中按清晰度×人脸面积选一个代表帧，复用空间跟踪框。",
             "视频代表帧分数不是逐帧均值。没有 GroundingDINO 全角色裁剪、CLIP 筛选或 VLM fidelity gate。",
             "质心包含被评分镜头自身（与论文公式相同）；不同镜头数量下绝对值不宜直接比较。", "",
             "全部单人镜头口径包含安全回退。即使某回退镜头像素不变，两侧质心也可能不同，所以其质心分数 Δ 不保证为 0。",
             "仅注入子集另外重新计算质心，不从全量质心分数中筛选；它与全量口径的镜头集合不同。", ""]
    for scope, reports in result["scopes"].items():
        lines += [f"## {scope}", "", "| 路线 | 指标 | 角色组 | 有效角色-shot / 可重复出现角色-shot | Control | Treatment | Δ |", "|---|---|---:|---:|---:|---:|---:|"]
        for modality, report in reports.items():
            for route, v in report["routes"].items():
                lines.append(f"| {route} | {modality} | {v['scored_entity_count']} | {v['scored_paired_appearance_count']}/{v['eligible_recurring_appearance_count']} | {number(v['control_mean'])} | {number(v['treatment_mean'])} | {number(v['mean_paired_delta'], True)} |")
        lines.append("")
    lines += ["## 全量单人镜头：逐 episode", "", "| Run | 路线 | 指标 | 有效角色-shot | Control | Treatment | Δ |", "|---|---|---|---:|---:|---:|---:|"]
    for modality, report in result["scopes"]["all_single_character"].items():
        runs = {e["episode_id"]: e["run"] for e in report["entities"]}
        for ep, routes in report["episodes"].items():
            for route, v in routes.items():
                lines.append(f"| {runs[ep]} | {route} | {modality} | {v['scored_paired_appearance_count']} | {number(v['control_mean'])} | {number(v['treatment_mean'])} | {number(v['mean_paired_delta'], True)} |")
    lines += ["", "## 全量单人镜头：逐角色", "", "| Run | 路线 | 角色 | 指标 | 有效/计划 shot | Control | Treatment | Δ / 原因 | 审计图 |", "|---|---|---|---|---:|---:|---:|---|---|"]
    for modality, report in result["scopes"]["all_single_character"].items():
        for e in report["entities"]:
            links = " ".join(f"[图{i+1}]({p})" for i, p in enumerate(e.get("audit_sheets", [])))
            lines.append(f"| {e['run']} | {e['route']} | {e['character']} | {modality} | {e['matched_shot_count']}/{e['scheduled_shot_count']} | {number(e['control']['mean'] if e['control'] else None)} | {number(e['treatment']['mean'] if e['treatment'] else None)} | {number(e['delta'], True) if e['delta'] is not None else e['skip_reason']} | {links} |")
    lines += ["", "完整逐镜头分数、缺失原因、两侧质心向量、裁剪/代表帧和源文件 SHA256 见同目录 JSON。", ""]
    return "\n".join(lines)


def evaluate(args):
    # Cached aggregation must not load the inference framework or model.
    torch = None
    cv2.setNumThreads(1)
    evaluator_hash = sha256(Path(__file__))
    source = args.output_root.resolve()
    destination = source / "cross_shot_evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    hash_cache = {}

    def fingerprint(path):
        path = Path(path).resolve()
        if str(path) not in hash_cache:
            hash_cache[str(path)] = sha256(path)
        return hash_cache[str(path)]

    config = {"feature_version": FEATURE_VERSION, "embedding": "normalized last_hidden_state[:,0] CLS",
              "model": str(args.dino_model.resolve()),
              "model_files": {p.name: fingerprint(p) for p in sorted(args.dino_model.glob('*')) if p.is_file()},
              "crop_expansion": args.crop_expansion, "sample_count": args.sample_count,
              "video_selection": "sigmoid((LapVar-100)/200) * sigmoid((face_area_percent-2)/5); earliest tie"}
    model_token = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    first_path = source / "evaluation/firstframe_evaluation.json"
    cases_path = source / "video_evaluation/cases.json"
    rows = read(first_path)["shots"]
    cohort = read(source / "manifest.json")
    expected = set()
    for episode in cohort["episodes"]:
        script = read(episode["episode_path"])
        fingerprint(episode["episode_path"])
        for route in ("PuLID-FLUX", "IP-Adapter"):
            for shot, schedule in script["entity_schedule"].items():
                if len(schedule.get("characters", [])) == 1:
                    expected.add((route, episode["episode_id"], shot, schedule["characters"][0]))
    actual = [(r["route"], r["episode_id"], r["shot_key"], r["character"]) for r in rows]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("first-frame report does not match frozen single-character schedule")
    cases = read(cases_path)
    videos = {}
    for case in cases:
        if read(case["identity_report"]).get("tracking_policy") != "spatial":
            raise ValueError(f"non-spatial video tracking: {case['identity_report']}")
        fingerprint(case["identity_report"])
        for job in read(case["identity_report"])["jobs"]:
            key = (case["route"], case["episode_id"], job["shot_key"], job["condition"])
            if key in videos:
                raise ValueError(f"duplicate video report key: {key}")
            videos[key] = job
    fingerprint(first_path)
    fingerprint(cases_path)
    fingerprint(source / "manifest.json")
    processor = model = None
    pending = []
    cache_hits = 0

    def flush():
        nonlocal processor, model, torch
        if not pending:
            return
        if model is None:
            import torch
            from transformers import AutoImageProcessor, AutoModel
            torch.set_num_threads(args.cpu_threads)
            processor = AutoImageProcessor.from_pretrained(str(args.dino_model), local_files_only=True)
            model = AutoModel.from_pretrained(str(args.dino_model), local_files_only=True).to(args.device).eval()
        with torch.inference_mode():
            inputs = processor(images=[p[0] for p in pending], return_tensors="pt")
            features = model(**{k: v.to(args.device) for k, v in inputs.items()}).last_hidden_state[:, 0]
            features = torch.nn.functional.normalize(features.float(), dim=-1).cpu().numpy()
        for (_, record, cache_path), feature in zip(pending, features):
            record["embedding"] = feature.tolist()
            write(cache_path, record)
        pending.clear()

    for modality in MODALITIES:
        for index, row in enumerate(rows):
            row[modality] = {}
            for c in CONDITIONS:
                if modality == "first_frame":
                    path = Path(row[f"{c}_image"])
                    selection_input = {"bbox": row.get(f"{c}_bbox")}
                else:
                    job = videos[(row["route"], row["episode_id"], row["shot_key"], c)]
                    if job["target_character"] != row["character"]:
                        raise ValueError(f"video role mismatch for {job['job_id']}")
                    path = Path(job["video"]["video_path"])
                    frames = job["video"]["frames"]
                    selection_input = {"frames": [{"frame_index": f["frame_index"], "selected_face_bbox": f.get("selected_face_bbox")} for f in frames]}
                source_hash = fingerprint(path)
                cache_key = hashlib.sha256(json.dumps({"encoder": model_token, "modality": modality,
                                                       "source_sha256": source_hash, "selection": selection_input}, sort_keys=True).encode()).hexdigest()
                cache_path = destination / "features" / f"{cache_key}.json"
                if cache_path.is_file():
                    record = read(cache_path)
                    if record["source_sha256"] != source_hash or record["cache_key"] != cache_key:
                        raise ValueError(f"invalid feature cache: {cache_path}")
                    if record.get("embedding") is not None:
                        normalized(record["embedding"])
                        if not Path(record["crop_path"]).is_file():
                            raise FileNotFoundError(record["crop_path"])
                    if record["source_path"] != str(path.resolve()):
                        record["embedding_reused_from_source_path"] = record["source_path"]
                        record["source_path"] = str(path.resolve())
                    row[modality][c] = record
                    cache_hits += 1
                    continue
                record = {"source_path": str(path.resolve()), "source_sha256": source_hash,
                          "cache_key": cache_key, "feature_cache": str(cache_path),
                          "embedding": None, "crop_path": None, "missing_reason": None}
                if modality == "first_frame":
                    bbox = selection_input["bbox"]
                    if bbox is None:
                        crop = None
                        record["missing_reason"] = "missing_frozen_first_frame_bbox"
                    else:
                        with Image.open(path) as image:
                            crop, expanded = crop_face(image.convert("RGB"), bbox, args.crop_expansion)
                        record.update(bbox=bbox, expanded_bbox=expanded)
                else:
                    crop, audit = representative_video_crop(path, frames, args.crop_expansion, args.sample_count)
                    record.update(audit)
                row[modality][c] = record
                if crop is None:
                    write(cache_path, record)
                else:
                    crop_path = destination / "crops" / f"{cache_key}.png"
                    crop_path.parent.mkdir(parents=True, exist_ok=True)
                    crop.save(crop_path)
                    record["crop_path"] = str(crop_path)
                    pending.append((crop, record, cache_path))
                    if len(pending) >= args.batch_size:
                        flush()
            if (index + 1) % 20 == 0:
                print(f"{modality}: {index + 1}/{len(rows)} paired shots, {cache_hits} cached appearances", flush=True)
        flush()
        print(f"{modality}: complete ({len(rows)} paired shots)", flush=True)

    model = processor = None
    scopes = summarize(rows)
    report = {"kind": "paired_cross_shot_dinov2_face_centroid", "official_entitybench_cs_face": False,
              "source_root": str(source), "methodology": {**config,
                  "grouping": "route + episode-local character; condition-specific centroids on identical paired shot sets",
                  "formula": "mu=normalize(mean(unit_CLS)); score_i=unit_CLS_i dot mu; N>=2 distinct shots",
                  "centroid_includes_scored_appearance": True,
                  "aggregation": "micro mean across scored character-shot appearances",
                  "scopes": "all single-character scheduled shots including fallback; injected-only recomputes centroids",
                  "missing_policy": "null + reason; paired intersection; singletons unscored",
                  "fidelity_gate": "not computed; VLM metrics remain missing",
                  "crop_policy": "existing mapped final-frame bbox or spatially tracked video bbox; no identity-based frame selection",
                  "fallback_note": "unchanged fallback images can have nonzero score delta because centroids change"},
              "run": {"device": args.device, "torch_version": version("torch"), "batch_size": args.batch_size,
                      "cpu_threads": args.cpu_threads, "cached_appearance_count": cache_hits,
                      "evaluator_sha256": evaluator_hash},
              "source_hashes": hash_cache, "pair_count": len(rows), "scopes": scopes, "shots": rows}
    audit_sheets(report, destination)
    write(destination / "face_centroid_evaluation.json", report)
    (destination / "face_centroid_evaluation.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({s: {m: r["routes"] for m, r in v.items()} for s, v in scopes.items()}, ensure_ascii=False, indent=2))
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/entitybench_firstframe_pulid_ip_20ep")
    parser.add_argument("--dino-model", type=Path, default=Path("/root/autodl-tmp/models/dinov2-base"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--crop-expansion", type=float, default=0.15)
    parser.add_argument("--sample-count", type=int, default=5)
    args = parser.parse_args()
    if args.batch_size < 1 or args.cpu_threads < 1 or args.sample_count < 1 or args.crop_expansion < 0:
        parser.error("batch size, threads and sample count must be positive; crop expansion must be nonnegative")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
