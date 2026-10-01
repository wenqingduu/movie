#!/usr/bin/env python3
"""Compare historical two-pass Qwen results with the adaptive shared-prefix run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OLD = PROJECT_ROOT / "outputs/entitybench_qwen_image21_v7_full3"
DEFAULT_NEW = (
    PROJECT_ROOT
    / "outputs/entitybench_qwen_image21_official_control_partial_adaptive_040_full3"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _control_hash_audit(old_root: Path, new_root: Path) -> dict[str, Any]:
    old_manifest = _load(old_root / "manifest.json")
    new_manifest = _load(new_root / "manifest.json")
    old = {
        (item["run"], item["shot_key"]): Path(item["shot_dir"]) / "control.png"
        for item in old_manifest["shots"]
    }
    new = {
        (item["run"], item["shot_key"]): Path(item["shot_dir"]) / "control.png"
        for item in new_manifest["shots"]
    }
    common = sorted(set(old) & set(new))
    matched, mismatched, missing = [], [], []
    for key in common:
        if not old[key].is_file() or not new[key].is_file():
            missing.append("/".join(key))
        elif _sha256(old[key]) == _sha256(new[key]):
            matched.append("/".join(key))
        else:
            mismatched.append("/".join(key))
    return {
        "common_shot_count": len(common),
        "byte_identical_count": len(matched),
        "mismatch_count": len(mismatched),
        "missing_count": len(missing),
        "mismatched_shots": mismatched,
        "missing_shots": missing,
    }


def _control_metric_audit(old_root: Path, new_root: Path) -> dict[str, Any]:
    def lookup(root: Path) -> dict[tuple[str, str, str], float]:
        report = _load(root / "evaluation_report.json")
        values = {}
        for shot in report["shots"]:
            for character in shot.get("image_metrics", {}).get("characters", []):
                values[(shot["run"], shot["shot_key"], character["name"])] = float(
                    character["control_cosine"]
                )
        return values

    old = lookup(old_root)
    new = lookup(new_root)
    common = sorted(set(old) & set(new))
    differences = [abs(old[key] - new[key]) for key in common]
    return {
        "historical_role_pair_count": len(old),
        "new_role_pair_count": len(new),
        "common_role_pair_count": len(common),
        "max_absolute_control_cosine_difference": max(differences, default=None),
        "mismatch_over_1e_6_count": sum(value > 1e-6 for value in differences),
        "historical_only": ["/".join(key) for key in sorted(set(old) - set(new))],
        "new_only": ["/".join(key) for key in sorted(set(new) - set(old))],
    }


def _identity_row(root: Path) -> dict[str, Any]:
    report = _load(root / "evaluation_report.json")
    overall = report["overall"]
    image = overall["image_identity"]
    video = overall["video_identity"]
    return {
        "shot_count": overall["shot_count"],
        "injected_shot_count": overall["injected_shot_count"],
        "fallback_shot_count": overall["fallback_shot_count"],
        "image_character_pairs": image["valid_pair_count"],
        "image_control": image["control_mean"],
        "image_treatment": image["treatment_mean"],
        "image_delta": image["mean_paired_delta"],
        "image_positive": image["positive_delta_count"],
        "video_character_pairs": video["valid_pair_count"],
        "video_control": video["control_mean"],
        "video_treatment": video["treatment_mean"],
        "video_delta": video["mean_paired_delta"],
        "video_positive": video["positive_delta_count"],
    }


def _dino_row(root: Path) -> dict[str, Any]:
    overall = _load(root / "dino_evaluation_report.json")["overall"]
    return {
        "character_pairs": overall["evaluated_character_count"],
        "image_control": overall["first_frame_control_mean"],
        "image_treatment": overall["first_frame_treatment_mean"],
        "image_delta": overall["first_frame_mean_paired_delta"],
        "image_positive": overall["first_frame_positive_character_count"],
        "video_control": overall["video_frame_control_mean"],
        "video_treatment": overall["video_frame_treatment_mean"],
        "video_delta": overall["video_frame_mean_paired_delta"],
        "video_positive": overall["video_positive_character_count"],
    }


def _fmt(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "--"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def _pair(row: dict[str, Any], prefix: str) -> str:
    return (
        f"{_fmt(row[prefix + '_control'])}→{_fmt(row[prefix + '_treatment'])} "
        f"({_fmt(row[prefix + '_delta'], True)})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-root", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--new-root", type=Path, default=DEFAULT_NEW)
    args = parser.parse_args()

    rows = {
        "historical_two_pass_six_steps": {
            "root": str(args.historical_root.resolve()),
            "identity": _identity_row(args.historical_root),
            "dino": _dino_row(args.historical_root),
        },
        "official_control_partial_adaptive_040": {
            "root": str(args.new_root.resolve()),
            "identity": _identity_row(args.new_root),
            "dino": _dino_row(args.new_root),
        },
    }
    payload = {
        "kind": "qwen_historical_vs_official_control_partial_adaptive_040",
        "warning": (
            "Both routes use the reproducible official Qwen Control. Historical Treatment "
            "uses the old two-pass sampling path; the new Treatment forks the captured "
            "official latent and uses per-role tolerance, so this is a route comparison, "
            "not an injection-step-only ablation. Aggregate Control means may differ when "
            "the evaluated role populations differ."
        ),
        "rows": rows,
        "control_hash_audit": _control_hash_audit(
            args.historical_root, args.new_root
        ),
        "control_metric_audit": _control_metric_audit(
            args.historical_root, args.new_root
        ),
    }
    output_json = args.new_root / "historical_vs_adaptive.json"
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Qwen 历史两遍式与官方 Control 分叉自适应方案汇总",
        "",
        "> Control 均来自可复现的官方 Qwen 默认流程；Treatment 执行协议和可评角色集合不同，因此这是路线对比，不是仅注入步数不同的严格消融。",
        "",
        "| 方案 | shot（注入/回退） | IF首帧 Control→Treatment (Δ) | IF视频 Control→Treatment (Δ) | DINO首帧 Control→Treatment (Δ) | DINO视频 Control→Treatment (Δ) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "historical_two_pass_six_steps": "历史两遍式6步",
        "official_control_partial_adaptive_040": "官方Control分叉+逐角色容错（1/6步）",
    }
    for key, row in rows.items():
        identity = row["identity"]
        dino = row["dino"]
        lines.append(
            f"| {labels[key]} | {identity['shot_count']} "
            f"({identity['injected_shot_count']}/{identity['fallback_shot_count']}) | "
            f"{_pair(identity, 'image')} | {_pair(identity, 'video')} | "
            f"{_pair(dino, 'image')} | {_pair(dino, 'video')} |"
        )
    lines.extend([
        "",
        "聚合单位为可评测的角色-shot 对；多人镜头按角色分别计数。",
        "",
        "Control 文件哈希审计："
        f"{payload['control_hash_audit']['byte_identical_count']}/"
        f"{payload['control_hash_audit']['common_shot_count']} 个共有镜头逐字节一致；"
        f"不一致 {payload['control_hash_audit']['mismatch_count']}，"
        f"缺失 {payload['control_hash_audit']['missing_count']}。",
        "共有角色对的首帧 Control identity cosine 审计："
        f"{payload['control_metric_audit']['common_role_pair_count']} 对，"
        f"差异超过 1e-6 的有 "
        f"{payload['control_metric_audit']['mismatch_over_1e_6_count']} 对，"
        f"最大绝对差为 "
        f"{payload['control_metric_audit']['max_absolute_control_cosine_difference']}。",
        "",
    ])
    output_md = args.new_root / "HISTORICAL_VS_ADAPTIVE.md"
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(output_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
