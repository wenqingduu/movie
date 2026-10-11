"""Freeze the full 4-step cohort, reusing first-frame assets, never old videos."""
from __future__ import annotations

import argparse
import copy
import difflib
import json
import shutil
from pathlib import Path

from prepare_entitybench_a14b_continuation import PROJECT, OUTPUT as SOURCE, sha256, write_json

OUTPUT = PROJECT / "outputs/entitybench_pulid_i2v_a14b_4step_continuation_20ep"
LIGHT = Path("/root/autodl-tmp/LightX2V")
LIGHT_REVISION = "7bd01b7d5d04bf99cdd9ca9082f20c184c66d991"
OFFICIAL_REVISION = "3776439e7037b2128ac3b5d2d1ce63e424cca166"
OFFICIAL_SHA = "2334aad4b61cae2a0ea38cc2ee9ae85125b6df4e128610903654fd2b49607bb4"
PARTIAL_SHA = "79caf1327e7f1a63527c0e18bc9c936b8618b59080197d2f50677ee2ba5cb5a0"


def prepare(output=OUTPUT, source=SOURCE):
    output, source = Path(output).resolve(), Path(source).resolve()
    assert output != source, "4-step outputs must have a separate root"
    source_path = source / "manifest.json"
    manifest = copy.deepcopy(json.loads(source_path.read_text()))
    assert manifest["totals"]["shot_count"] == 363 and len(manifest["episodes"]) == 20
    assert not manifest["totals"]["firstframe_source_counts"].get("missing_multi_hardcut")
    for episode in manifest["episodes"]:
        assert sha256(episode["episode_path"]) == episode["episode_sha256"]
        for shot in episode["shots"]:
            if shot["cut"]:
                frame = shot["firstframe_source"]
                assert sha256(frame["input_image"]) == frame["input_sha256"]
                assert sha256(frame["source_config"]) == frame["source_config_sha256"]
    lora_path = PROJECT / "outputs/model_downloads/wan22_lightx2v_distill_loras/manifest.json"
    loras = json.loads(lora_path.read_text())
    assert json.loads((lora_path.parent / "status.json").read_text())["state"] == "complete"
    for row in loras["files"]:
        path = Path(loras["destination"]) / row["path"]
        assert path.stat().st_size == row["size"] and sha256(path) == row["sha256"]
    manifest.update(kind="entitybench_pulid_i2v_a14b_4step_continuation_20ep", output_root=str(output),
                    route="pulid_hardcut_own_tail_lightx2v_distill_bf16_4step", schema_version=2,
                    source_firstframe_manifest=str(source_path), source_firstframe_manifest_sha256=sha256(source_path))
    manifest["generation_backend"] = {"name": "LightX2V", "source": str(LIGHT), "revision": LIGHT_REVISION,
        "model_class": "wan2.2_moe", "distillation": "dmd2", "lora_inventory": loras,
        "lora_inventory_sha256": sha256(lora_path), "lora_merge_force_fp32": False,
        "low_noise_unused_keys": "290 image-encoder-branch tensors; ordinary A14B has no CLIP branch"}
    manifest["video_config"] = dict(max_area=832 * 480, frame_num=81, fps=16, sampling_steps=4,
        shift=5.0, guide_scale=1.0, solver="dmd2", denoising_step_list=[1000, 750, 500, 250],
        boundary_step_index=2, param_dtype="bfloat16", attention="torch_sdpa", quantization=False,
        feature_caching="NoCaching", offload_granularity="model", t5_cpu_offload=True,
        vae_cpu_offload=False, cpu_pinned_memory=False, negative_prompt_active=False)
    manifest["evaluation"] = dict(primary="official_entitybench_non_vlm", vlm_enabled=False,
        scope="partial_without_vlm", project_identity_enabled=False,
        cross_shot_fidelity_gate="unavailable_without_vlm; cs_face and cs_object provisional",
        official_revision=OFFICIAL_REVISION, strict=True, config_overrides=False)
    if (output / "manifest.json").exists():
        assert json.loads((output / "manifest.json").read_text()) == manifest, "Frozen 4-step manifest changed; use a new root"
    write_json(output / "manifest.json", manifest)
    for name in ["firstframe_inventory.json", "asset_manifest.json", "split_20ep.json"]:
        shutil.copy2(source / name, output / name)
    shutil.copy2(source_path, output / "source_firstframe_manifest.json")
    runtime = output / "runtime"
    runtime.mkdir(exist_ok=True)
    official = PROJECT / "benchmarks/entitybench/eval/evaluate_benchmark_official.py"
    partial = PROJECT / "benchmarks/entitybench/eval/evaluate_benchmark.py"
    assert sha256(official) == OFFICIAL_SHA and sha256(partial) == PARTIAL_SHA
    shutil.copy2(official, runtime / "evaluate_benchmark_official.py")
    shutil.copy2(partial, runtime / "evaluate_benchmark_partial.py")
    patch = "".join(difflib.unified_diff(official.read_text().splitlines(True), partial.read_text().splitlines(True),
                     fromfile="official/evaluate_benchmark.py", tofile="skip_vlm/evaluate_benchmark.py"))
    (runtime / "skip_vlm.patch").write_text(patch)
    write_json(runtime / "evaluation_provenance.json", dict(official_revision=OFFICIAL_REVISION,
        official_sha256=OFFICIAL_SHA, partial_sha256=PARTIAL_SHA, scope="partial_without_vlm",
        vlm_api_calls_authorized=False, fidelity_gate_applied=False,
        not_comparable_to_full_official_cs_face=True, project_face_crop_metrics_used=False))
    print(json.dumps(manifest["totals"], indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    args = parser.parse_args()
    prepare(args.output_root, args.source_root)
