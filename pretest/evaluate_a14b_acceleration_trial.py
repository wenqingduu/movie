"""CPU identity diagnostics and frame sheets for the isolated speed trial."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from run_a14b_acceleration_trial import ROOT, sha, write
from evaluate_video_identity import _cosine, _evaluate_video, _largest_face
from evaluate_reference_dino_identity import DinoEncoder, _crop_face, _decode_video_crops
from multishot.face_analysis_backend import get_face_backend


def main():
    torch.set_num_threads(8)
    cv2.setNumThreads(2)
    # InsightFace's router drops sess_options; bound its session constructor
    # locally so five default physical-core pools do not oversubscribe this host.
    import onnxruntime as ort
    from insightface.model_zoo.model_zoo import PickableInferenceSession
    session_init = PickableInferenceSession.__init__
    def bounded_session(self, *args, **kwargs):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        kwargs["sess_options"] = options
        session_init(self, *args, **kwargs)
    PickableInferenceSession.__init__ = bounded_session
    plan = json.loads((ROOT / "plan.json").read_text())
    sources = [("original_bf16_40step", [dict(row, video=row["output_video"], generate_seconds=row["elapsed_seconds"])
                for row in plan["baseline_records"].values()])]
    for route in plan["routes"]:
        path = ROOT / route / "report.json"
        if path.exists():
            sources.append((route, [row for row in json.loads(path.read_text())["jobs"] if row["status"] == "generated"]))
    backend = get_face_backend()
    encoder = DinoEncoder(Path("/root/autodl-tmp/models/dinov2-base"), "cpu", 8)
    shots = {shot["shot_key"]: shot for shot in plan["shots"]}
    rows, refs = [], {}
    dino_hash = {p.name: sha(p) for p in Path("/root/autodl-tmp/models/dinov2-base").iterdir()
                 if p.suffix in {".bin", ".safetensors"}}
    assert dino_hash, "DINOv2 weight inventory is empty"
    for route, jobs in sources:
        for job in jobs:
            shot = shots[job["shot_key"]]
            row = dict(route=route, shot_key=shot["shot_key"], generate_seconds=job["generate_seconds"],
                       video=job["video"], video_sha256=sha(job["video"]), cut=shot["cut"])
            if len(shot["entity_schedule"]["characters"]) != 1:
                row.update(status="auxiliary_not_scored", reason="multi_character_mapping_not_available")
                rows.append(row)
                continue
            reference = Path(shot["firstframe_source"]["reference_image"])
            fingerprint = {"implementation_sha256": sha(__file__), "video": row["video_sha256"],
                           "reference": sha(reference), "dino_weights": dino_hash, "tracking": "spatial",
                           "mapping": "identity_reference_on_actual_first_frame_only", "crop_expansion": .15,
                           "onnx_intra_threads": 2, "onnx_inter_threads": 1, "onnx_spinning": False}
            out = ROOT / "quality" / route / ("shot_" + shot["shot_key"].replace(":", "_"))
            if (out / "metrics.json").exists():
                existing = json.loads((out / "metrics.json").read_text())
                if existing.get("fingerprint") == fingerprint:
                    rows.append(existing)
                    continue
            out.mkdir(parents=True, exist_ok=True)
            if reference not in refs:
                face = _largest_face(backend.analyze(str(reference)))
                assert face is not None, f"No face in fixed reference: {reference}"
                crop = _crop_face(Image.open(reference).convert("RGB"), face["face_bbox"], .15)
                refs[reference] = dict(identity=face["face_embedding"], dino=encoder.encode([crop])[0])
            ref = refs[reference]
            capture = cv2.VideoCapture(row["video"])
            ok, frame = capture.read()
            capture.release()
            assert ok
            first = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            faces = backend.analyze_bgr(frame)
            face = max(faces, key=lambda f: _cosine(ref["identity"], f["face_embedding"]), default=None)
            tracking = _evaluate_video(Path(row["video"]), ref["identity"], out / "tracking.jpg", tracking_policy="spatial",
                initial_bbox=face["face_bbox"] if face else None, initial_image_size=list(first.size))
            write(out / "tracking.json", tracking)
            crops, indexes, count = _decode_video_crops(Path(row["video"]), tracking["frames"], .15)
            assert count == 81
            values = [None] * count
            if crops:
                for index, feature in zip(indexes, encoder.encode(crops)):
                    values[index] = float(np.dot(ref["dino"], feature))
            valid = [value for value in values if value is not None]
            row.update(status="evaluated", reference=str(reference), fingerprint=fingerprint, insightface=tracking["summary"],
                       reference_dino={"first": values[0], "mean": float(np.mean(valid)) if valid else None,
                                       "scored_frames": len(valid), "frames": values})
            write(out / "metrics.json", row)
            rows.append(row)
            print(f"QUALITY {route} {shot['shot_key']} IF={tracking['summary']['mean']} DINO={row['reference_dino']['mean']}", flush=True)
    sheets = []
    for shot in plan["shots"]:
        key = shot["shot_key"]
        available = [(route, next((job for job in jobs if job["shot_key"] == key), None)) for route, jobs in sources]
        available = [(route, job) for route, job in available if job]
        canvas = Image.new("RGB", (1800, 230 * len(available)), "#171717")
        draw = ImageDraw.Draw(canvas)
        for index, (route, job) in enumerate(available):
            draw.text((8, index * 230 + 6), f"{route} | shot {key} | {job['generate_seconds']:.1f}s", fill="white")
            capture = cv2.VideoCapture(job["video"])
            input_path = Path(job["video"]).parent / "input.png"
            if not input_path.exists():
                input_path = Path(job["input_image"])
            imgs = [Image.open(input_path).convert("RGB")]
            for frame_index in [0, 20, 40, 60, 80]:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                assert ok
                imgs.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            capture.release()
            for panel, image in enumerate(imgs):
                image.thumbnail((295, 185))
                canvas.paste(image, (panel * 300, index * 230 + 38))
                draw.text((panel * 300 + 8, index * 230 + 23), "I2V input" if panel == 0 else f"frame {[0,20,40,60,80][panel-1]}", fill="white")
        path = ROOT / "quality" / f"comparison_shot_{key.replace(':','_')}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path, quality=94)
        sheets.append(str(path))
    write(ROOT / "quality/report.json", {"scope": "pilot diagnostics; no VLM or official benchmark scores",
        "device": "cpu", "rows": rows, "comparison_sheets": sheets,
        "limitations": plan["comparison_limitations"], "tracking": "spatial after fixed-reference first-frame mapping"})
    def fmt(value):
        return "missing" if value is None else f"{value:.4f}"
    lines = ["# A14B 加速小样本实测", "", "独立试验，不替代20episode正式实验。时间为模型加载后单镜头生成、解码与写视频的耗时；各实现写盘/解码范围可能略有差异。", "",
             "| 路线 | 镜头 | 生成秒数 | 视频IF均值 | 参考DINO均值 | 人脸覆盖率 |", "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['route']} | {row['shot_key']} | {row['generate_seconds']:.1f} | {fmt(row.get('insightface',{}).get('mean'))} | {fmt(row.get('reference_dino',{}).get('mean'))} | {fmt(row.get('insightface',{}).get('detection_coverage'))} |")
    lines += ["", "4步版本同时改变蒸馏LoRA、时间步、CFG和shift，不能归因于单一参数。多人接续镜头仅检查视频和自身尾帧来源，身份辅助缺失不填零。", ""]
    for path in sheets:
        lines.append(f"- [{Path(path).name}]({path})")
    (ROOT / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
